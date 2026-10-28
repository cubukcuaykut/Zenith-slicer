"""
Evaluation loop and KPI computation for the ZENITH simulation.

Runs a given orchestration scheme over a fixed offered-load trace through the
physics model and records per-slice, per-step KPIs:

  * E2E latency        L_e2e^s   (ms)
  * throughput         Gamma_s   (Mbps)
  * resource efficiency eta      (network-wide, %)
  * SLA violation rate rho_s     (%)
  * zero-touch adaptation response time T_adapt (ms) -- mean time the system
    dwells in an SLA-violation episode before control restores compliance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import config as C
from . import network_model as nm
from .orchestrators import BaseController


@dataclass
class SchemeResult:
    scheme: str
    latency_ms: Dict[str, float]           # mean E2E latency per slice
    p99_latency_ms: Dict[str, float]       # 99th-percentile latency per slice
    throughput_mbps: Dict[str, float]      # mean delivered throughput per slice
    sla_violation_pct: Dict[str, float]    # SLA violation rate per slice
    efficiency_pct: float                  # mean network resource efficiency
    t_adapt_ms: float                      # mean adaptation response time
    # Mean E2E latency decomposition per slice: {slice: {seg: ms}}.
    latency_components: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-step traces (for plotting), keyed by slice.
    latency_trace: Dict[str, np.ndarray] = field(default_factory=dict)
    throughput_trace: Dict[str, np.ndarray] = field(default_factory=dict)
    offered_trace: Dict[str, np.ndarray] = field(default_factory=dict)
    efficiency_trace: np.ndarray = field(default_factory=lambda: np.array([]))


def _adaptation_time_ms(violation_flags: np.ndarray) -> float:
    """
    Mean duration (ms) of contiguous SLA-violation episodes. A larger value
    means the scheme takes longer to recover compliance (slower adaptation).
    Schemes that rarely violate yield small values.
    """
    step_ms = C.EVAL_CFG.step_seconds * 1000.0
    durations: List[int] = []
    run = 0
    for v in violation_flags:
        if v:
            run += 1
        elif run > 0:
            durations.append(run)
            run = 0
    if run > 0:
        durations.append(run)
    if not durations:
        return 0.0
    return float(np.mean(durations) * step_ms)


def run_scheme(controller: BaseController, offered_series: Dict[str, np.ndarray]
               ) -> SchemeResult:
    n = len(next(iter(offered_series.values())))
    lat = {s: np.zeros(n) for s in C.SLICES}
    thr = {s: np.zeros(n) for s in C.SLICES}
    viol = {s: np.zeros(n, dtype=bool) for s in C.SLICES}
    eff = np.zeros(n)
    segs = ["fronthaul", "ran", "transport", "core"]
    comp = {s: {seg: np.zeros(n) for seg in segs} for s in C.SLICES}

    for t in range(n):
        offered = {s: float(offered_series[s][t]) for s in C.SLICES}
        alloc = controller.step(t, offered)
        kpis = nm.evaluate_network(offered, alloc)
        for s in C.SLICES:
            lat[s][t] = kpis[s].l_e2e
            thr[s][t] = kpis[s].served
            viol[s][t] = kpis[s].sla_violation
            comp[s]["fronthaul"][t] = kpis[s].lat_fronthaul
            comp[s]["ran"][t] = kpis[s].lat_ran
            comp[s]["transport"][t] = kpis[s].lat_transport
            comp[s]["core"][t] = kpis[s].lat_core
        eff[t] = nm.resource_efficiency(kpis, alloc) * 100.0

    latency_ms = {s: float(lat[s].mean()) for s in C.SLICES}
    p99 = {s: float(np.percentile(lat[s], 99)) for s in C.SLICES}
    throughput = {s: float(thr[s].mean()) for s in C.SLICES}
    sla_pct = {s: float(viol[s].mean() * 100.0) for s in C.SLICES}
    latency_components = {
        s: {seg: float(comp[s][seg].mean()) for seg in segs} for s in C.SLICES
    }
    # Adaptation time aggregated across slices (mean of per-slice episode means).
    t_adapt = float(np.mean([_adaptation_time_ms(viol[s]) for s in C.SLICES]))

    return SchemeResult(
        scheme=controller.name,
        latency_ms=latency_ms,
        p99_latency_ms=p99,
        throughput_mbps=throughput,
        sla_violation_pct=sla_pct,
        efficiency_pct=float(eff.mean()),
        t_adapt_ms=t_adapt,
        latency_components=latency_components,
        latency_trace={s: lat[s] for s in C.SLICES},
        throughput_trace={s: thr[s] for s in C.SLICES},
        offered_trace={s: offered_series[s] for s in C.SLICES},
        efficiency_trace=eff,
    )
