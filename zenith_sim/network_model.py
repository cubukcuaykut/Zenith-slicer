"""
Network physics model for the ZENITH X-haul simulation.

Computes per-slice End-to-End KPIs from the current resource allocation
(N_PRB, B_transport, C_UPF) and the offered traffic load lambda_s.

E2E latency is decomposed (paper Sec. III) as:

    L_e2e = fronthaul_prop                     (eCPRI)
          + RAN processing + RAN queueing       (numerology slot + load)
          + transport propagation + queueing    (SDN midhaul)
          + core processing + queueing          (per-slice UPF backhaul)

Queueing follows an M/M/1-style term  d * rho/(1-rho)  bounded for stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from . import config as C


@dataclass
class Allocation:
    """Per-slice resource allocation across the three X-haul domains."""
    n_prb: Dict[str, float]
    b_transport: Dict[str, float]
    c_upf: Dict[str, float]

    def copy(self) -> "Allocation":
        return Allocation(
            n_prb=dict(self.n_prb),
            b_transport=dict(self.b_transport),
            c_upf=dict(self.c_upf),
        )


@dataclass
class SliceKPI:
    served: float          # delivered throughput Gamma_s (Mbps)
    offered: float         # offered load lambda_s (Mbps)
    l_e2e: float           # E2E latency (ms)
    l_max: float           # SLA latency bound (ms)
    sla_violation: bool    # latency-bound or throughput-floor breach
    util_ran: float
    util_tx: float
    util_core: float
    # E2E latency decomposition (ms) across the X-haul segments.
    lat_fronthaul: float = 0.0
    lat_ran: float = 0.0
    lat_transport: float = 0.0
    lat_core: float = 0.0


def _queue_term(slot_or_serial_ms: float, rho: float, cap_factor: float = 0.985) -> float:
    """Bounded M/M/1 waiting-time term: d * rho/(1-rho)."""
    rho = min(rho, cap_factor)
    if rho <= 0.0:
        return 0.0
    return slot_or_serial_ms * (rho / (1.0 - rho))


def slice_capacity(slice_name: str, alloc: Allocation) -> Dict[str, float]:
    """Per-domain serving capacity (Mbps) for a slice."""
    spec = C.SLICE_SPECS[slice_name]
    cap_ran = max(alloc.n_prb[slice_name], 1e-6) * spec.mbps_per_prb
    cap_tx = max(alloc.b_transport[slice_name], 1e-6)
    cap_core = max(alloc.c_upf[slice_name], 1e-6)
    return {"ran": cap_ran, "tx": cap_tx, "core": cap_core}


def evaluate_slice(slice_name: str, offered: float, alloc: Allocation) -> SliceKPI:
    """Compute KPIs for one slice given offered load and allocation."""
    spec = C.SLICE_SPECS[slice_name]
    num = C.NUMEROLOGY[slice_name]
    cap = slice_capacity(slice_name, alloc)

    # Bottleneck throughput across the three domains.
    served = min(offered, cap["ran"], cap["tx"], cap["core"])

    # Domain utilizations driven by offered load.
    rho_ran = offered / cap["ran"]
    rho_tx = offered / cap["tx"]
    rho_core = offered / cap["core"]

    # --- Latency decomposition (ms) ---
    # RAN: one slot for frame alignment + numerology-scaled processing + queue.
    ran_proc = num.slot_ms                      # slot-aligned processing
    ran_queue = _queue_term(num.slot_ms, rho_ran)

    # Transport: propagation (URLLC via New vSwitch shortest path) + queueing.
    tx_serial = 0.05 + 0.5 * num.slot_ms        # serialization scale
    tx_queue = _queue_term(tx_serial, rho_tx)

    # Core: per-slice UPF processing + queueing.
    core_queue = _queue_term(spec.core_proc_ms, rho_core)

    l_e2e = (
        C.FRONTHAUL_PROP_MS
        + ran_proc + ran_queue
        + spec.tx_prop_ms + tx_queue
        + spec.core_proc_ms + core_queue
    )

    # SLA violation: latency bound exceeded OR throughput floor not met.
    served_ratio = served / max(offered, 1e-6)
    sla_violation = (l_e2e > spec.l_max_ms) or (served_ratio < spec.min_served_ratio)

    return SliceKPI(
        served=served,
        offered=offered,
        l_e2e=l_e2e,
        l_max=spec.l_max_ms,
        sla_violation=sla_violation,
        util_ran=min(rho_ran, 1.0),
        util_tx=min(rho_tx, 1.0),
        util_core=min(rho_core, 1.0),
        lat_fronthaul=C.FRONTHAUL_PROP_MS,
        lat_ran=ran_proc + ran_queue,
        lat_transport=spec.tx_prop_ms + tx_queue,
        lat_core=spec.core_proc_ms + core_queue,
    )


def evaluate_network(offered: Dict[str, float], alloc: Allocation) -> Dict[str, SliceKPI]:
    return {s: evaluate_slice(s, offered[s], alloc) for s in C.SLICES}


def resource_efficiency(kpis: Dict[str, SliceKPI], alloc: Allocation) -> float:
    """
    Resource utilization efficiency eta in [0,1]: SLA-compliant delivered
    throughput (goodput) per unit of allocated serving capacity.

        eta = sum_s goodput_s / sum_s effective_capacity_s

    where goodput counts throughput only when the slice meets its SLA, and the
    effective serving capacity is the binding (min) domain capacity. This
    rewards allocating *just enough* to satisfy SLAs and penalises both
    over-provisioning (idle capacity) and congestion (SLA-violating goodput=0).
    """
    goodput = 0.0
    capacity = 0.0
    for s, k in kpis.items():
        cap = slice_capacity(s, alloc)
        eff_cap = min(cap["ran"], cap["tx"], cap["core"])
        capacity += eff_cap
        if not k.sla_violation:
            goodput += k.served
    if capacity <= 0.0:
        return 0.0
    return min(goodput / capacity, 1.0)


def total_allocated(alloc: Allocation) -> float:
    """Sum of normalized allocated resources (for the reward regularizer)."""
    prb = sum(alloc.n_prb.values()) / C.BUDGETS.n_prb_total
    tx = sum(alloc.b_transport.values()) / C.BUDGETS.b_transport_total
    core = sum(alloc.c_upf.values()) / C.BUDGETS.c_upf_total
    return prb + tx + core
