# Haulio Frozen Backhaul Policy — Preliminary Submission

This directory is the competition-compliant runtime bundle. It contains one
immutable, randomly initialized-and-trained graph-policy artifact and only its
core inference path.

## Compliance boundary

- Model weights, preprocessing dimensions, thresholds, and version are static
  for the entire demonstration.
- Changing IoT, fleet, order, facility, and road state are inference inputs;
  they never update the model.
- No optimizer, trainer, auto-tuning, bulk test runner, feedback loop, model
  registry promotion, or network data collection exists in this directory.
- Deterministic masks enforce capacity, body type, manifest binding, pickup and
  delivery windows, fuel, telemetry freshness, and road availability.
- Loaded trucks must declare whether their current cargo is released before
  the candidate pickup; otherwise only measured residual capacity is usable.
- IDs, booleans, coordinates, sensor ages, weights, distances, times, and
  optional sensor objects are strictly validated before model execution.
- The model recommends assignments; it never commits them. A dispatcher must
  accept a recommendation through the existing Haulio application.
- Training used a clearly labelled synthetic digital twin because joined real
  order/decision/outcome history is not yet available. Do not claim real-world
  uplift from the included validation metrics.

## Run locally

```bash
docker compose up --build
```

Health:

```bash
curl http://127.0.0.1:8088/health
```

Inference:

```bash
curl -sS http://127.0.0.1:8088/infer \
  -H 'Content-Type: application/json' \
  --data-binary @demo_input.json
```

For a direct CLI demonstration inside an environment with PyTorch 1.13:

```bash
python service.py --artifacts artifacts --input demo_input.json
```

The input schema supports at most 16 trucks and 32 open orders per corridor
snapshot. Larger markets must be partitioned upstream into overlapping feasible
corridors; that orchestration is not part of this preliminary inference bundle.

This artifact replaces candidate edge ranking only. It does not construct a
multi-stop VRP tour or learn empty-return detection, anomaly detection, road
routing, or price calibration. Those capabilities remain deterministic or
upstream in the hybrid MVP and must not be attributed to this checkpoint.
