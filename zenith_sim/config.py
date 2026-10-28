"""
ZENITH simulation configuration.

All parameters mirror the notation and tables of the paper
"ZENITH: A Zero-Touch AI-Native Slice Orchestration Framework for
End-to-End 6G X-Haul Networks".

Slices: URLLC, eMBB, mMTC.
Domains per slice: RAN (N_PRB), Transport (B_transport), Core (C_UPF).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

SLICES: List[str] = ["URLLC", "eMBB", "mMTC"]

# Master random seed for full reproducibility.
SEED: int = 7


# ----------------------------------------------------------------------
# 5G NR numerology-to-service mapping  (paper Table II)
# mu | SCS (kHz) | slot duration (ms) | service
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Numerology:
    mu: int
    scs_khz: float
    slot_ms: float


NUMEROLOGY: Dict[str, Numerology] = {
    "mMTC": Numerology(mu=1, scs_khz=30.0, slot_ms=0.500),
    "eMBB": Numerology(mu=2, scs_khz=60.0, slot_ms=0.250),
    "URLLC": Numerology(mu=3, scs_khz=120.0, slot_ms=0.125),
}


# ----------------------------------------------------------------------
# Per-slice radio / transport / core physics + SLA targets
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SliceSpec:
    name: str
    # RAN: achievable rate per PRB (Mbps/PRB), reflects MCS robustness.
    mbps_per_prb: float
    # SLA latency bound L_max^s (ms):  URLLC << eMBB <= mMTC
    l_max_ms: float
    # Minimum throughput share that must be served to avoid SLA violation.
    min_served_ratio: float
    # Reward weights (paper Eq. reward): alpha (throughput),
    # beta (latency), gamma (SLA penalty).
    alpha: float
    beta: float
    gamma: float
    # Transport propagation base (ms). URLLC uses the dynamically
    # instantiated "New vSwitch" shortest path => lowest propagation.
    tx_prop_ms: float
    # Core processing base (ms).
    core_proc_ms: float
    # Control interval T_c (seconds): URLLC < eMBB < mMTC (paper Eq. tc).
    control_interval_s: float


SLICE_SPECS: Dict[str, SliceSpec] = {
    "URLLC": SliceSpec(
        name="URLLC",
        mbps_per_prb=0.80,
        l_max_ms=2.0,
        min_served_ratio=0.97,
        alpha=1.0,
        beta=1.2,
        gamma=12.0,
        tx_prop_ms=0.10,   # steered via New vSwitch shortest path
        core_proc_ms=0.15,
        control_interval_s=0.5,
    ),
    "eMBB": SliceSpec(
        name="eMBB",
        mbps_per_prb=1.50,
        l_max_ms=10.0,
        min_served_ratio=0.92,
        alpha=1.4,
        beta=0.3,
        gamma=4.0,
        tx_prop_ms=0.40,
        core_proc_ms=0.25,
        control_interval_s=2.0,
    ),
    "mMTC": SliceSpec(
        name="mMTC",
        mbps_per_prb=0.30,
        l_max_ms=50.0,
        min_served_ratio=0.88,
        alpha=0.6,
        beta=0.1,
        gamma=2.0,
        tx_prop_ms=0.80,
        core_proc_ms=0.40,
        control_interval_s=5.0,
    ),
}


# ----------------------------------------------------------------------
# Global resource budgets  (sum_s N_PRB^s <= N_PRB^total, paper Eq. prb)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Budgets:
    n_prb_total: int = 273          # 100 MHz @ 30 kHz SCS (3GPP)
    b_transport_total: float = 12000.0   # Mbps  (12 Gbps X-haul)
    c_upf_total: float = 12000.0          # Mbps-equiv core processing


BUDGETS = Budgets()

# Resource-efficiency regularization coefficient delta (reward Eq.).
DELTA_REG: float = 0.02


# ----------------------------------------------------------------------
# Fixed latency contributions (ms)
# ----------------------------------------------------------------------
FRONTHAUL_PROP_MS: float = 0.05   # eCPRI fronthaul propagation


# ----------------------------------------------------------------------
# Traffic generator parameters (per-slice offered load, Mbps)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TrafficSpec:
    base: float        # mean offered load (Mbps)
    diurnal_amp: float # amplitude of slow (diurnal) variation
    period: float      # diurnal period (steps)
    noise_std: float   # AR(1) innovation std
    ar: float          # AR(1) coefficient
    burst_prob: float  # per-step burst probability
    burst_gain: float  # multiplicative burst magnitude


TRAFFIC: Dict[str, TrafficSpec] = {
    "URLLC": TrafficSpec(base=18.0, diurnal_amp=6.0, period=240,
                         noise_std=1.6, ar=0.80, burst_prob=0.02, burst_gain=1.8),
    "eMBB": TrafficSpec(base=120.0, diurnal_amp=55.0, period=240,
                        noise_std=9.0, ar=0.85, burst_prob=0.03, burst_gain=2.0),
    "mMTC": TrafficSpec(base=45.0, diurnal_amp=15.0, period=480,
                        noise_std=4.0, ar=0.75, burst_prob=0.015, burst_gain=1.6),
}


# ----------------------------------------------------------------------
# AI model + training hyperparameters
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LSTMConfig:
    window: int = 24          # observation window W (paper Eq. forecast)
    horizon: int = 1          # delta look-ahead
    hidden: int = 32
    layers: int = 1
    epochs: int = 25
    lr: float = 1e-2
    batch: int = 64
    train_steps: int = 4000   # length of traffic series for training


LSTM_CFG = LSTMConfig()


@dataclass(frozen=True)
class SLAClassifierConfig:
    hidden: int = 32
    epochs: int = 30
    lr: float = 5e-3
    batch: int = 128
    n_samples: int = 12000
    lookahead_thresh: float = 0.5  # epsilon (paper) decision threshold


SLA_CFG = SLAClassifierConfig()


@dataclass(frozen=True)
class DQNConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma_rl: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 160
    episodes: int = 220
    steps_per_episode: int = 160
    replay_size: int = 20000
    batch: int = 64
    target_sync: int = 200     # steps between target-net soft sync
    tau: float = 0.05          # soft update coefficient
    warmup: int = 500


DQN_CFG = DQNConfig()


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    eval_steps: int = 1500     # evaluation horizon (steps) per scheme
    step_seconds: float = 0.5  # wall-clock seconds represented by one step


EVAL_CFG = EvalConfig()

SCHEMES: List[str] = ["MO", "ROS", "COS", "SOS", "ZENITH"]
