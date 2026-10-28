"""
Per-slice traffic generator.

Each slice's offered load lambda_s(t) (Mbps) is synthesised as:
    diurnal sinusoid (slow trend)
  + AR(1) stochastic process (short-term correlation)
  + sporadic multiplicative bursts (flash crowds / mMTC sync events)

Deterministic given a seed -> reproducible traces for fair scheme comparison.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import config as C


class TrafficGenerator:
    def __init__(self, seed: int = C.SEED):
        self.rng = np.random.default_rng(seed)
        # AR(1) running state per slice.
        self._ar_state: Dict[str, float] = {s: 0.0 for s in C.SLICES}
        self._t = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._ar_state = {s: 0.0 for s in C.SLICES}
        self._t = 0

    def step(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in C.SLICES:
            spec = C.TRAFFIC[s]
            diurnal = spec.diurnal_amp * np.sin(2 * np.pi * self._t / spec.period)
            innov = self.rng.normal(0.0, spec.noise_std)
            self._ar_state[s] = spec.ar * self._ar_state[s] + innov
            load = spec.base + diurnal + self._ar_state[s]
            if self.rng.random() < spec.burst_prob:
                load *= spec.burst_gain
            out[s] = float(max(load, 0.5))
        self._t += 1
        return out

    def generate_series(self, n: int) -> Dict[str, np.ndarray]:
        """Generate a full trace of length n for each slice."""
        series: Dict[str, List[float]] = {s: [] for s in C.SLICES}
        for _ in range(n):
            sample = self.step()
            for s in C.SLICES:
                series[s].append(sample[s])
        return {s: np.asarray(v, dtype=np.float32) for s, v in series.items()}
