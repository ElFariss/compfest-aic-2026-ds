# Model Card

## Identity

- Model: Constraint-Conditioned Backhaul Graph Policy
- Version: `2026.08.25-a100`
- Frozen SHA-256: `c33c4a83ac3a6925cc9634ecf6d29fb929427918ed70440cf8560beb3527b733`
- Parameters: 4,393,161
- Initialization: random; no downloaded pretrained checkpoint
- Runtime behavior: immutable inference only

## Intended use

Rank compatible heavy-truck backhaul assignments from current IoT, fleet,
order, and road context. The model produces recommendations and uncertainty;
deterministic code enforces hard feasibility, and a dispatcher remains the
authority for acceptance.

## Training evidence

- Hardware: NVIDIA A100-SXM4 80 GB
- Framework: PyTorch 1.13.1 + CUDA 11.6, BF16
- Steps: 6,000; best checkpoint selected at step 4,000
- Wall time: 360.4 seconds
- Data: labelled synthetic Indonesia-style digital twin only
- Frozen/eager maximum absolute difference: 0.0

The recorded `0.0` parity result is the eight-snapshot export check. A separate
30-snapshot audit observed a maximum absolute difference of `7.629e-06`, below
the `1e-5` export gate.

Final fixed-seed synthetic validation:

| Slice | Learned reward | Static baseline | Difference |
| --- | ---: | ---: | ---: |
| Standard | 1.7768 | 1.6661 | +0.1107 |
| Sensor-dropout stress | 0.7381 | 0.6764 | +0.0617 |

Across the final 3,520 synthetic snapshots, decoding reported zero hard
constraint violations and zero duplicate assignments. These are simulator
metrics, not evidence of real-world business uplift.

## Limits

- The model has not been trained or validated on joined Haulio order,
  decision, and realized-outcome histories.
- No public dataset entered this training run; evaluation uses held-out seeds
  from the same synthetic generator family rather than chronological or
  external-domain validation.
- Maximum input is 16 trucks and 32 orders per overlapping corridor snapshot.
- Unknown vehicle/cargo categories, stale telemetry, unbound manifests, broken
  time windows, unavailable roads, and over-capacity pairs are rejected before
  recommendation.
- The model must not automatically accept assignments or update itself.
- The model performs one-step truck-order assignment, not full VRP routing,
  multi-hop sequencing, learned anomaly detection, or calibrated pricing.
- ETA, uncertainty, and contribution heads are not field-calibrated. The
  reported A100 latency is batched model-forward timing per snapshot, not HTTP
  or CPU end-to-end latency.
- Production claims require chronological field data, shadow deployment, and a
  controlled dispatcher trial.
