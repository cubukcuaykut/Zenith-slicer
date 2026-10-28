"""
SLA-violation prediction model (paper Eq. P(SLA viol.)):

    P(SLA viol. | k_s(t)) = sigma( w^T phi(k_s(t)) + b )

A binary neural classifier over the per-slice KPI vector
    k_s(t) = [ L_e2e^s, Gamma_s, eta_RAN^s, eta_transport^s ]
predicts whether the slice will violate its SLA within the lookahead.
Training labels come from the physics model: we perturb allocations and
offered load to generate (KPI, violation) pairs, then learn the mapping.

When the predicted probability exceeds epsilon, ZENITH triggers preemptive
re-optimization (used by the ZENITH orchestrator).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import network_model as nm
from .network_model import Allocation


FEATURES = 4  # [L_e2e, Gamma, util_ran, util_tx]


class _Classifier(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _feature_vector(kpi: nm.SliceKPI) -> np.ndarray:
    return np.array([kpi.l_e2e, kpi.served, kpi.util_ran, kpi.util_tx],
                    dtype=np.float32)


def _build_dataset(rng: np.random.Generator, n: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random allocations/loads -> (features, violation label)."""
    X, Y = [], []
    per_slice = n // len(C.SLICES)
    for s in C.SLICES:
        spec = C.SLICE_SPECS[s]
        base_load = C.TRAFFIC[s].base
        for _ in range(per_slice):
            offered = max(0.5, base_load * rng.uniform(0.4, 2.2))
            # Random plausible allocation for this slice.
            prb = rng.uniform(5, C.BUDGETS.n_prb_total * 0.6)
            tx = rng.uniform(offered * 0.4, offered * 2.5)
            core = rng.uniform(offered * 0.4, offered * 2.5)
            alloc = Allocation(
                n_prb={s: prb}, b_transport={s: tx}, c_upf={s: core})
            kpi = nm.evaluate_slice(s, offered, alloc)
            X.append(_feature_vector(kpi))
            Y.append(1.0 if kpi.sla_violation else 0.0)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


class SLAViolationClassifier:
    def __init__(self, cfg: C.SLAClassifierConfig = C.SLA_CFG, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.net = _Classifier(cfg.hidden).to(device)
        self.feat_mean = np.zeros(FEATURES, dtype=np.float32)
        self.feat_std = np.ones(FEATURES, dtype=np.float32)
        self.metrics: Dict[str, float] = {}

    def train(self, seed: int = C.SEED) -> Dict[str, float]:
        rng = np.random.default_rng(seed + 99)
        X, Y = _build_dataset(rng, self.cfg.n_samples)

        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0) + 1e-6
        Xn = (X - self.feat_mean) / self.feat_std

        idx = rng.permutation(len(Xn))
        Xn, Y = Xn[idx], Y[idx]
        split = int(0.8 * len(Xn))
        xtr, ytr = Xn[:split], Y[:split]
        xte, yte = Xn[split:], Y[split:]

        xtr_t = torch.tensor(xtr).to(self.device)
        ytr_t = torch.tensor(ytr).to(self.device)

        # Balance classes with pos_weight.
        pos = max(ytr.sum(), 1.0)
        neg = max(len(ytr) - pos, 1.0)
        pos_weight = torch.tensor(neg / pos, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        self.net.train()
        n = len(xtr_t)
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n)
            for b in range(0, n, self.cfg.batch):
                j = perm[b:b + self.cfg.batch]
                opt.zero_grad()
                logits = self.net(xtr_t[j])
                loss = loss_fn(logits, ytr_t[j])
                loss.backward()
                opt.step()

        # Evaluation metrics on held-out set.
        self.net.eval()
        with torch.no_grad():
            xte_t = torch.tensor(xte).to(self.device)
            prob = torch.sigmoid(self.net(xte_t)).cpu().numpy()
        pred = (prob >= self.cfg.lookahead_thresh).astype(np.float32)
        tp = float(((pred == 1) & (yte == 1)).sum())
        tn = float(((pred == 0) & (yte == 0)).sum())
        fp = float(((pred == 1) & (yte == 0)).sum())
        fn = float(((pred == 0) & (yte == 1)).sum())
        acc = (tp + tn) / max(len(yte), 1)
        prec = tp / max(tp + fp, 1e-6)
        rec = tp / max(tp + fn, 1e-6)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        self.metrics = {"accuracy": acc, "precision": prec,
                        "recall": rec, "f1": f1}
        return self.metrics

    def predict_proba(self, kpi: nm.SliceKPI) -> float:
        x = (_feature_vector(kpi) - self.feat_mean) / self.feat_std
        self.net.eval()
        with torch.no_grad():
            logit = self.net(torch.tensor(x).view(1, -1).to(self.device))
            return float(torch.sigmoid(logit).item())
