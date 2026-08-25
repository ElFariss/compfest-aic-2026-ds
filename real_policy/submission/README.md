# Frozen competition inference

This directory is the preliminary-round runtime boundary. It exposes only
static inference and health endpoints. Model parameters are loaded once from a
checksum-bound TorchScript file and cannot be updated by any API.

```bash
docker compose up --build
curl http://127.0.0.1:8088/health
```

Run the committed held-out public-source examples:

```bash
curl -sS -H 'Content-Type: application/json' \
  --data @examples/singapore_telemetry_heldout.json \
  http://127.0.0.1:8088/infer/telemetry
curl -sS -H 'Content-Type: application/json' \
  --data @examples/dtcargo_track_heldout.json \
  http://127.0.0.1:8088/infer/truck-track
curl -sS -H 'Content-Type: application/json' \
  --data @examples/scania_health_heldout.json \
  http://127.0.0.1:8088/infer/health
curl -sS -H 'Content-Type: application/json' \
  --data @examples/vius_deadhead_heldout.json \
  http://127.0.0.1:8088/infer/deadhead
curl -sS -H 'Content-Type: application/json' \
  --data @examples/tlc_price_heldout.json \
  http://127.0.0.1:8088/infer/price
```

Endpoints:

- `POST /infer/route` — graph-pointer rollout wrapped in hard capacity,
  pickup-before-delivery, vehicle/cargo compatibility, time-window, route-time,
  and road-availability checks.
- `POST /infer/telemetry` — Singapore commercial-vehicle prefix forecast.
- `POST /infer/truck-track` — DT-CARGO N3 truck track/GNSS head.
- `POST /infer/health` — Scania APS source-domain failure probability.
- `POST /infer/deadhead` — VIUS annual deadhead/reposition prior, never a
  per-trip empty-state claim.
- `POST /infer/price` — NYC TLC source-domain cost proxy, never an operational
  Indonesian freight quote.

Missing or stale required data returns `ABSTAIN` with a reason. There is no
optimizer, training dependency, automatic update, auto-tuning, feedback loop,
automatic dispatch, or externally bound port. Every recommendation requires a
dispatcher decision.

## Input contracts

- Telemetry accepts 1--64 strictly increasing samples. The first sample uses
  `delta_t_s=0`; later intervals must match timestamp differences. Required
  fields are speed km/h, engine state, road grade, engine-load percent, mass
  airflow g/s, longitudinal acceleration m/s2, interval seconds, and observed
  fraction. It never accepts DT-CARGO fields or predicts cargo load.
- Truck track accepts the eight DT-CARGO-native semantic fields under `track`:
  distance m, track gap m, average/maximum speed m/s, HDOP, GVWR kg, GCWR kg,
  and axle class. Missing public fields use JSON `null` and an internal missing
  bit; the endpoint remains separate from telemetry.
- Health accepts exactly 170 train-normalized Scania values plus 170 Boolean
  missing bits. A missing value must use the documented zero sentinel.
- Deadhead accepts the twelve named raw VIUS fields; frozen train-split median
  and IQR values are stored in the artifact manifest.
- Price accepts twelve already canonicalized TLC source features. The response
  always sets `operational_quote=false`.
- Route requires actual kg and cm3 capacity/load, paired shipment deltas,
  freshness, fuel/GNSS/operator thresholds, and a complete directed road
  matrix. No Amazon route payload is emitted because that source lacks measured
  kg; the release does not fabricate those required truck facts.

Every example contains a `_provenance` object identifying the public held-out
row and raw artifact hash. Unknown metadata is ignored by inference. Examples
from different sources are never merged into one payload.
