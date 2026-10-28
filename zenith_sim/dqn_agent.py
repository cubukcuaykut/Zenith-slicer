"""
Deep Q-Network (DQN) cross-domain resource optimizer (paper Sec. IV-D).

MDP <S, A, P, R, gamma_RL>:
  * State (Eq. state):  s = [ N_PRB, B_transport, C_UPF, L_e2e, lambda_hat ]
    (+ a small slice-context tail so a single shared policy generalises
     across URLLC / eMBB / mMTC).
  * Action (Eq. action): a = [ dN_PRB, dB_transport, dC_UPF ], each in
    {-1, 0, +1}  ->  3^3 = 27 discrete joint cross-domain actions.
  * Reward (Eq. reward): alpha*throughput - beta*latency
                         - gamma*1[SLA viol] - delta*resource_cost.
  * Q-update (Eq. dqn): target network with soft (tau) updates.

The agent is trained against a single-slice rollout of the physics model
driven by the synthetic traffic and the LSTM forecast.
"""
from __future__ import annotations

import random
from collections import deque
from itertools import product
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import network_model as nm
from .network_model import Allocation

# 27 discrete actions: 3 levels per domain.
ACTIONS: List[Tuple[int, int, int]] = list(product((-1, 0, 1), repeat=3))
N_ACTIONS = len(ACTIONS)
STATE_DIM = 7  # 5 (paper state) + 2 (slice context)


# ----------------------------------------------------------------------
# Per-slice resource envelope (search space bounds + step sizes)
# ----------------------------------------------------------------------
def slice_bounds(slice_name: str) -> Dict[str, float]:
    base = C.TRAFFIC[slice_name].base
    return {
        "prb_min": 2.0,
        "prb_max": 0.60 * C.BUDGETS.n_prb_total,
        "tx_min": 1.0,
        "tx_max": 6.0 * base,
        "core_min": 1.0,
        "core_max": 6.0 * base,
        "prb_step": 3.0,
        "tx_step": 0.10 * base,
        "core_step": 0.10 * base,
        "base": base,
    }


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SliceTrainEnv:
    """Single-slice training environment for the DQN agent."""

    def __init__(self, slice_name: str, traffic_series: np.ndarray,
                 forecast_series: np.ndarray, seed: int = C.SEED):
        self.s = slice_name
        self.spec = C.SLICE_SPECS[slice_name]
        self.b = slice_bounds(slice_name)
        self.traffic = traffic_series
        self.forecast = forecast_series
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.alloc = self._initial_alloc()

    def _initial_alloc(self) -> Allocation:
        base = self.b["base"]
        return Allocation(
            n_prb={self.s: clip(base / self.spec.mbps_per_prb,
                                self.b["prb_min"], self.b["prb_max"])},
            b_transport={self.s: clip(base, self.b["tx_min"], self.b["tx_max"])},
            c_upf={self.s: clip(base, self.b["core_min"], self.b["core_max"])},
        )

    def reset(self) -> np.ndarray:
        self.t = int(self.rng.integers(0, max(1, len(self.traffic) - 200)))
        self.alloc = self._initial_alloc()
        # Randomize start to diversify experience.
        self.alloc.n_prb[self.s] = clip(
            self.alloc.n_prb[self.s] * self.rng.uniform(0.5, 1.5),
            self.b["prb_min"], self.b["prb_max"])
        return self._state()

    def _state(self) -> np.ndarray:
        kpi = nm.evaluate_slice(self.s, self.traffic[self.t], self.alloc)
        f = self.forecast[self.t] if self.t < len(self.forecast) else self.traffic[self.t]
        return np.array([
            self.alloc.n_prb[self.s] / self.b["prb_max"],
            self.alloc.b_transport[self.s] / self.b["tx_max"],
            self.alloc.c_upf[self.s] / self.b["core_max"],
            min(kpi.l_e2e / self.spec.l_max_ms, 3.0),
            f / (self.b["base"] * 3.0),
            self.spec.l_max_ms / 50.0,
            self.spec.mbps_per_prb / 1.5,
        ], dtype=np.float32)

    def _apply(self, action_idx: int) -> None:
        dprb, dtx, dcore = ACTIONS[action_idx]
        self.alloc.n_prb[self.s] = clip(
            self.alloc.n_prb[self.s] + dprb * self.b["prb_step"],
            self.b["prb_min"], self.b["prb_max"])
        self.alloc.b_transport[self.s] = clip(
            self.alloc.b_transport[self.s] + dtx * self.b["tx_step"],
            self.b["tx_min"], self.b["tx_max"])
        self.alloc.c_upf[self.s] = clip(
            self.alloc.c_upf[self.s] + dcore * self.b["core_step"],
            self.b["core_min"], self.b["core_max"])

    def _reward(self, kpi: nm.SliceKPI) -> float:
        served_ratio = kpi.served / max(kpi.offered, 1e-6)
        lat_ratio = kpi.l_e2e / self.spec.l_max_ms
        cost = (self.alloc.n_prb[self.s] / self.b["prb_max"]
                + self.alloc.b_transport[self.s] / self.b["tx_max"]
                + self.alloc.c_upf[self.s] / self.b["core_max"]) / 3.0
        r = (self.spec.alpha * served_ratio
             - self.spec.beta * lat_ratio
             - self.spec.gamma * (1.0 if kpi.sla_violation else 0.0)
             - C.DELTA_REG * 3.0 * cost)
        return float(r)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        self._apply(action_idx)
        # Forecast-driven look-ahead load for the optimization target.
        load = self.traffic[self.t]
        kpi = nm.evaluate_slice(self.s, load, self.alloc)
        reward = self._reward(kpi)
        self.t += 1
        done = self.t >= len(self.traffic) - 1
        return self._state(), reward, done


class _QNet(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.buf.append(transition)

    def sample(self, batch: int):
        items = random.sample(self.buf, batch)
        s, a, r, s2, d = zip(*items)
        return (np.asarray(s, dtype=np.float32),
                np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32),
                np.asarray(s2, dtype=np.float32),
                np.asarray(d, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    """Shared-policy DQN controlling all three domains of a slice."""

    def __init__(self, cfg: C.DQNConfig = C.DQN_CFG, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.q = _QNet(cfg.hidden).to(device)
        self.tgt = _QNet(cfg.hidden).to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.replay_size)
        self.eps = cfg.epsilon_start
        self.loss_fn = nn.SmoothL1Loss()

    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.eps:
            return random.randrange(N_ACTIONS)
        s = torch.tensor(state, dtype=torch.float32).view(1, -1).to(self.device)
        return int(self.q(s).argmax(dim=1).item())

    def _learn(self) -> None:
        if len(self.buf) < max(self.cfg.warmup, self.cfg.batch):
            return
        s, a, r, s2, d = self.buf.sample(self.cfg.batch)
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r).to(self.device)
        s2 = torch.tensor(s2).to(self.device)
        d = torch.tensor(d).to(self.device)

        q = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            # Double-DQN target.
            next_a = self.q(s2).argmax(dim=1, keepdim=True)
            q_next = self.tgt(s2).gather(1, next_a).squeeze(1)
            target = r + self.cfg.gamma_rl * q_next * (1.0 - d)
        loss = self.loss_fn(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        # Soft target update.
        with torch.no_grad():
            for tp, p in zip(self.tgt.parameters(), self.q.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    def train(self, envs: List[SliceTrainEnv]) -> List[float]:
        """Train across all slice environments; return per-episode returns."""
        cfg = self.cfg
        returns: List[float] = []
        eps_lin = (cfg.epsilon_start - cfg.epsilon_end) / cfg.epsilon_decay_episodes
        for ep in range(cfg.episodes):
            env = envs[ep % len(envs)]
            state = env.reset()
            ep_ret = 0.0
            for _ in range(cfg.steps_per_episode):
                a = self.act(state)
                s2, r, done = env.step(a)
                self.buf.push(state, a, r, s2, float(done))
                state = s2
                ep_ret += r
                self._learn()
                if done:
                    break
            self.eps = max(cfg.epsilon_end, cfg.epsilon_start - eps_lin * (ep + 1))
            returns.append(ep_ret)
        return returns
