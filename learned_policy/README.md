# COMPFEST AIC 2026 — Haulio Backhaul Graph Policy

This is the isolated development workspace for the learned core-inference
component. The model is trained from random initialization on one A100 and then
frozen. Public or synthetic data are development curriculum only; the resulting
checkpoint is not presented as validated on proprietary Haulio outcomes.

## Model

`BackhaulGraphPolicy` is a constraint-conditioned heterogeneous graph network.
It receives one corridor snapshot containing:

- IoT truck state: GNSS, speed, heading, accuracy, fuel, cargo/load reading,
  optional CAN/IMU/device health, freshness, and missingness;
- fleet state: body type, capacity, current commitment, and home location;
- order state: origin, destination, weight, cargo class, required vehicle,
  pickup/delivery windows, price, priority, and manifest binding;
- pairwise road state: deadhead/loaded distance, travel time, weather,
  congestion, toll, road quality, freshness, capacity ratio, and downstream
  demand density.

Four relation-message-passing blocks jointly reason over all trucks and orders.
The output heads estimate policy score, contribution value, ETA, and uncertainty
for every feasible edge, plus a learned WAIT score. A conflict-free decoder
turns those scores into recommendations. Hard constraints are masks and are
rechecked outside the model.

This learned core solves a bounded, one-step bipartite assignment problem (up
to 16 trucks by 32 orders). It is not a full VRP tour constructor: route-graph
search, multi-hop sequencing, anomaly detection, and dispatcher workflow stay
in the existing Haulio stack or the explicit operational fallback. The
training run is offline supervised/multi-task policy learning with a
differentiable reward surrogate, not reinforcement learning.

## Offline training

The A100 environment already provides PyTorch 1.13.1, NumPy, and SciPy; no
package installation is required.

```bash
cd learned_policy
./run_a100.sh
```

The training process has two independent limits:

1. `training/train.py` rejects a budget above 9.5 hours and observes its own
   monotonic deadline.
2. `run_a100.sh` enforces a 9 hour 20 minute process timeout.

The trainer uses BF16, solver-style conflict-free imitation targets, direct
expected-reward optimization, sensor-dropout stress curriculum, validation,
early stopping, best-checkpoint retention, and TorchScript parity verification.
Its data are exclusively synthetic snapshots from the included generator; no
Deliveree, OpenStreetMap, DT-CARGO, or proprietary Haulio records entered this
training run.

Generated evidence appears in `artifacts/`:

- `backhaul_policy_frozen.pt`
- `manifest.json`
- `metrics.json`
- `training_complete.json`
- `run_config.json`
- `training_history.jsonl`
- `best_checkpoint.pt` (development only)

## Submission separation

`submission/` is the only directory intended for the preliminary runtime. It
contains no training package and exposes only `/health` and `/infer` through a
loopback Docker Compose service. Model weights and all parameters remain static
during the demonstration. Recommendations always require a human decision.

The notebook under `notebooks/` is an explanatory development artifact and is
not required by the runtime.
