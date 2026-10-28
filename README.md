# ZENITH Slicer

Simulation of ZENITH, an AI-native, zero-touch network slicing orchestrator for cross-domain (RAN, transport, core) resource control. It is compared against baseline orchestration schemes on the same offered-load trace and the same network model.

## Schemes compared

| Scheme | Description |
|--------|-------------|
| MO | Manual Orchestration: static allocation, no autonomous adaptation |
| ROS | RAN-Only Slicing: dynamic PRB control; transport and core static |
| COS | Core-Only Slicing: dynamic UPF control; RAN and transport static |
| SOS | SDN-Only Slicing: dynamic transport control |
| ZENITH | LSTM forecast pre-allocation + DQN joint cross-domain control + SLA-violation classifier for preemptive adaptation |

Slices: URLLC, eMBB, mMTC.

## Pipeline

`python -m zenith_sim.run_simulation` runs the following steps:

1. Generate synthetic per-slice X-haul traffic.
2. Train the LSTM traffic forecaster (reports MAPE).
3. Train the SLA-violation classifier (accuracy, precision, recall).
4. Train the DQN cross-domain optimizer.
5. Evaluate all schemes on the same held-out trace.
6. Write KPI tables (console and LaTeX), a CSV summary, and figures.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy torch matplotlib
```

## Run

```powershell
python -m zenith_sim.run_simulation
```

Outputs are written to `results/` (`kpi_summary.csv`, `scalability.csv`, `tables.tex`, and `fig_*.png`).

## Layout

| Path | Purpose |
|------|---------|
| `zenith_sim/config.py` | Slice specs, traffic and evaluation parameters |
| `zenith_sim/traffic.py` | Synthetic traffic generator |
| `zenith_sim/network_model.py` | RAN / transport / core latency and throughput model |
| `zenith_sim/forecaster.py` | LSTM traffic forecaster |
| `zenith_sim/sla_classifier.py` | SLA-violation classifier |
| `zenith_sim/dqn_agent.py` | DQN agent and training environment |
| `zenith_sim/orchestrators.py` | MO / ROS / COS / SOS / ZENITH controllers |
| `zenith_sim/evaluation.py` | Scheme evaluation and KPI computation |
| `zenith_sim/run_simulation.py` | End-to-end runner |
| `results/` | KPI CSVs, LaTeX tables, figures |
| `figures/` | Selected figures |
