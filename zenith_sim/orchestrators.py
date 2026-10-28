"""
Orchestration policies (schemes) evaluated in the paper:

  * MO     - Manual Orchestration: static allocation (provisioned once to
             mean demand), human-in-the-loop, no autonomous adaptation.
  * ROS    - RAN-Only Slicing: dynamic PRB control; transport & core static.
  * COS    - Core-Only Slicing: dynamic UPF control; RAN & transport static.
  * ZENITH - AI-native, zero-touch: LSTM forecast pre-allocation + DQN joint
             cross-domain control + SLA-classifier preemptive adaptation.

All schemes are driven by the SAME offered-load trace and the SAME physics
model (network_model) for a fair comparison.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import config as C
from . import network_model as nm
from .network_model import Allocation
from .dqn_agent import DQNAgent, slice_bounds, clip, ACTIONS
from .forecaster import TrafficForecaster
from .sla_classifier import SLAViolationClassifier


def _mean_demand() -> Dict[str, float]:
    return {s: C.TRAFFIC[s].base for s in C.SLICES}


def _prov_alloc(load: Dict[str, float], headroom: float) -> Allocation:
    """Provision allocation sized to `load` * headroom for every domain."""
    n_prb, b_tx, c_upf = {}, {}, {}
    for s in C.SLICES:
        spec = C.SLICE_SPECS[s]
        b = slice_bounds(s)
        target = load[s] * headroom
        n_prb[s] = clip(target / spec.mbps_per_prb, b["prb_min"], b["prb_max"])
        b_tx[s] = clip(target, b["tx_min"], b["tx_max"])
        c_upf[s] = clip(target, b["core_min"], b["core_max"])
    return Allocation(n_prb=n_prb, b_transport=b_tx, c_upf=c_upf)


class BaseController:
    name = "BASE"

    def __init__(self):
        self.alloc = _prov_alloc(_mean_demand(), headroom=1.1)
        # Smoothed demand estimate (per slice) for reactive control.
        self.ewma: Dict[str, float] = {s: C.TRAFFIC[s].base for s in C.SLICES}
        self._last_ctrl: Dict[str, int] = {s: -10**9 for s in C.SLICES}

    def _due(self, s: str, t: int) -> bool:
        interval = max(1, int(round(C.SLICE_SPECS[s].control_interval_s
                                    / C.EVAL_CFG.step_seconds)))
        if t - self._last_ctrl[s] >= interval:
            self._last_ctrl[s] = t
            return True
        return False

    def observe(self, offered: Dict[str, float]) -> None:
        for s in C.SLICES:
            self.ewma[s] = 0.7 * self.ewma[s] + 0.3 * offered[s]

    def step(self, t: int, offered: Dict[str, float]) -> Allocation:
        raise NotImplementedError


class ManualOrchestration(BaseController):
    """Static provisioning, no autonomous adaptation (human-in-the-loop)."""
    name = "MO"

    def __init__(self):
        super().__init__()
        self.alloc = _prov_alloc(_mean_demand(), headroom=1.15)

    def step(self, t: int, offered: Dict[str, float]) -> Allocation:
        # No adaptation: allocation frozen at provisioning time.
        return self.alloc


class RANOnlySlicing(BaseController):
    """Dynamic RAN (PRB) control; transport and core remain static."""
    name = "ROS"

    def __init__(self):
        super().__init__()
        self.alloc = _prov_alloc(_mean_demand(), headroom=1.15)

    def step(self, t: int, offered: Dict[str, float]) -> Allocation:
        self.observe(offered)
        for s in C.SLICES:
            if self._due(s, t):
                spec = C.SLICE_SPECS[s]
                b = slice_bounds(s)
                target = self.ewma[s] * 1.25  # reactive headroom
                self.alloc.n_prb[s] = clip(target / spec.mbps_per_prb,
                                           b["prb_min"], b["prb_max"])
        return self.alloc


class CoreOnlySlicing(BaseController):
    """Dynamic UPF (core) scaling; RAN and transport remain static."""
    name = "COS"

    def __init__(self):
        super().__init__()
        self.alloc = _prov_alloc(_mean_demand(), headroom=1.15)

    def step(self, t: int, offered: Dict[str, float]) -> Allocation:
        self.observe(offered)
        for s in C.SLICES:
            if self._due(s, t):
                b = slice_bounds(s)
                target = self.ewma[s] * 1.25
                self.alloc.c_upf[s] = clip(target, b["core_min"], b["core_max"])
        return self.alloc


class SDNOnlySlicing(BaseController):
    """Dynamic SDN transport (B_transport) control; RAN and core remain static."""
    name = "SOS"

    def __init__(self):
        super().__init__()
        self.alloc = _prov_alloc(_mean_demand(), headroom=1.15)

    def step(self, t: int, offered: Dict[str, float]) -> Allocation:
        self.observe(offered)
        for s in C.SLICES:
            if self._due(s, t):
                b = slice_bounds(s)
                target = self.ewma[s] * 1.25
                self.alloc.b_transport[s] = clip(target, b["tx_min"], b["tx_max"])
        return self.alloc


class ZenithController(BaseController):
    """
    AI-native zero-touch controller: forecast-driven pre-allocation, DQN joint
    cross-domain fine control, and SLA-classifier-triggered preemptive nudges.
    """
    name = "ZENITH"

    def __init__(self, agent: DQNAgent, forecaster: TrafficForecaster,
                 sla_clf: SLAViolationClassifier):
        super().__init__()
        self.agent = agent
        self.fc = forecaster
        self.clf = sla_clf
        # Tighter forecast-based provisioning headroom (efficient packing).
        self.alloc = _prov_alloc(_mean_demand(), headroom=1.2)
        self._hist: Dict[str, List[float]] = {s: [C.TRAFFIC[s].base] * self.fc.window
                                              for s in C.SLICES}

    def _dqn_state(self, s: str, forecast: float) -> np.ndarray:
        spec = C.SLICE_SPECS[s]
        b = slice_bounds(s)
        kpi = nm.evaluate_slice(s, self.ewma[s], self.alloc)
        return np.array([
            self.alloc.n_prb[s] / b["prb_max"],
            self.alloc.b_transport[s] / b["tx_max"],
            self.alloc.c_upf[s] / b["core_max"],
            min(kpi.l_e2e / spec.l_max_ms, 3.0),
            forecast / (b["base"] * 3.0),
            spec.l_max_ms / 50.0,
            spec.mbps_per_prb / 1.5,
        ], dtype=np.float32)

    def _apply_action(self, s: str, action_idx: int) -> None:
        dprb, dtx, dcore = ACTIONS[action_idx]
        b = slice_bounds(s)
        self.alloc.n_prb[s] = clip(self.alloc.n_prb[s] + dprb * b["prb_step"],
                                   b["prb_min"], b["prb_max"])
        self.alloc.b_transport[s] = clip(self.alloc.b_transport[s] + dtx * b["tx_step"],
                                         b["tx_min"], b["tx_max"])
        self.alloc.c_upf[s] = clip(self.alloc.c_upf[s] + dcore * b["core_step"],
                                   b["core_min"], b["core_max"])

    def step(self, t: int, offered: Dict[str, float]) -> Allocation:
        self.observe(offered)
        for s in C.SLICES:
            # Maintain recent-history window for the LSTM.
            self._hist[s].append(offered[s])
            if len(self._hist[s]) > self.fc.window:
                self._hist[s].pop(0)

            if self._due(s, t):
                spec = C.SLICE_SPECS[s]
                b = slice_bounds(s)
                # (1) LSTM forecast pre-allocation toward predicted demand.
                # Latency-sensitive slices get more headroom (lower utilization
                # => lower queueing delay).
                headroom = {"URLLC": 1.70, "eMBB": 1.30, "mMTC": 1.30}[s]
                forecast = self.fc.predict(s, np.asarray(self._hist[s]))
                forecast = max(forecast, self.ewma[s])
                target = forecast * headroom
                self.alloc.n_prb[s] = clip(
                    0.5 * self.alloc.n_prb[s]
                    + 0.5 * target / spec.mbps_per_prb,
                    b["prb_min"], b["prb_max"])
                self.alloc.b_transport[s] = clip(
                    0.5 * self.alloc.b_transport[s] + 0.5 * target,
                    b["tx_min"], b["tx_max"])
                self.alloc.c_upf[s] = clip(
                    0.5 * self.alloc.c_upf[s] + 0.5 * target,
                    b["core_min"], b["core_max"])

                # (2) DQN joint cross-domain fine control (greedy policy).
                a = self.agent.act(self._dqn_state(s, forecast), greedy=True)
                self._apply_action(s, a)

                # (3) SLA-classifier preemptive adaptation.
                kpi = nm.evaluate_slice(s, forecast, self.alloc)
                if self.clf.predict_proba(kpi) > C.SLA_CFG.lookahead_thresh:
                    # Preemptively boost the binding domain(s).
                    self.alloc.n_prb[s] = clip(
                        self.alloc.n_prb[s] + 2 * b["prb_step"],
                        b["prb_min"], b["prb_max"])
                    self.alloc.b_transport[s] = clip(
                        self.alloc.b_transport[s] + 2 * b["tx_step"],
                        b["tx_min"], b["tx_max"])
                    self.alloc.c_upf[s] = clip(
                        self.alloc.c_upf[s] + 2 * b["core_step"],
                        b["core_min"], b["core_max"])
        return self.alloc


def build_controller(scheme: str, agent: Optional[DQNAgent] = None,
                     forecaster: Optional[TrafficForecaster] = None,
                     sla_clf: Optional[SLAViolationClassifier] = None
                     ) -> BaseController:
    if scheme == "MO":
        return ManualOrchestration()
    if scheme == "ROS":
        return RANOnlySlicing()
    if scheme == "COS":
        return CoreOnlySlicing()
    if scheme == "SOS":
        return SDNOnlySlicing()
    if scheme == "ZENITH":
        assert agent and forecaster and sla_clf
        return ZenithController(agent, forecaster, sla_clf)
    raise ValueError(f"Unknown scheme: {scheme}")
