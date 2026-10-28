"""
LSTM-based per-slice traffic demand forecaster (paper Eq. forecast):

    lambda_hat_s(t+delta) = f_LSTM( lambda_s(t), ..., lambda_s(t-W+1) ; theta )

One compact LSTM per slice. Trained on a generated traffic trace; the
held-out tail of the trace is used to report a *real* MAPE figure.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config as C


class _LSTMNet(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, W, H)
        last = out[:, -1, :]           # (B, H)
        return self.head(last).squeeze(-1)


def _make_windows(series: np.ndarray, window: int, horizon: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(series) - window - horizon + 1):
        xs.append(series[i:i + window])
        ys.append(series[i + window + horizon - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class TrafficForecaster:
    """Holds one LSTM per slice plus per-slice normalization stats."""

    def __init__(self, cfg: C.LSTMConfig = C.LSTM_CFG, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.nets: Dict[str, _LSTMNet] = {}
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.window = cfg.window

    def _train_one(self, name: str, series: np.ndarray) -> float:
        cfg = self.cfg
        mu, sd = float(series.mean()), float(series.std() + 1e-6)
        self.mean[name], self.std[name] = mu, sd
        norm = (series - mu) / sd

        x, y = _make_windows(norm, cfg.window, cfg.horizon)
        split = int(0.8 * len(x))
        xtr, ytr = x[:split], y[:split]
        xte, yte = x[split:], y[split:]

        net = _LSTMNet(cfg.hidden, cfg.layers).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        xtr_t = torch.tensor(xtr).unsqueeze(-1).to(self.device)
        ytr_t = torch.tensor(ytr).to(self.device)

        net.train()
        n = len(xtr_t)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for b in range(0, n, cfg.batch):
                idx = perm[b:b + cfg.batch]
                opt.zero_grad()
                pred = net(xtr_t[idx])
                loss = loss_fn(pred, ytr_t[idx])
                loss.backward()
                opt.step()

        # Real MAPE on held-out tail (de-normalized).
        net.eval()
        with torch.no_grad():
            xte_t = torch.tensor(xte).unsqueeze(-1).to(self.device)
            pred = net(xte_t).cpu().numpy() * sd + mu
        true = yte * sd + mu
        mape = float(np.mean(np.abs((true - pred) / np.maximum(true, 1e-6))) * 100.0)

        self.nets[name] = net
        return mape

    def train(self, series: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train all slice forecasters; return per-slice test MAPE (%)."""
        return {s: self._train_one(s, series[s]) for s in C.SLICES}

    def predict(self, name: str, recent_window: np.ndarray) -> float:
        """One-step-ahead forecast for a slice given its recent W samples."""
        net = self.nets[name]
        mu, sd = self.mean[name], self.std[name]
        w = (np.asarray(recent_window[-self.window:], dtype=np.float32) - mu) / sd
        net.eval()
        with torch.no_grad():
            x = torch.tensor(w).view(1, -1, 1).to(self.device)
            yhat = net(x).item()
        return float(yhat * sd + mu)
