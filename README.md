# Autonomous Backhaul Optimizer

A local, dependency-free MVP for the AI Innovation Challenge problem: identify impending empty returns, match feasible cargo, propose multi-hop backhauls, estimate ETA/margin, verify IoT telemetry, and re-plan for meaningful route deviations.

The demo is intentionally honest about its evidence boundary:

- The Java map points and marketplace orders are a **synthetic digital-twin scenario**.
- The risk, ETA, price, and anomaly values are explainable **baseline heuristics**, not models trained on a claimed proprietary fleet dataset.
- Google traffic is an on-demand dispatcher confirmation. Its result is not persisted or used to train, test, or tune a model.

## Run

Python 3.11+ is sufficient; no `pip install` is required.

```bash
cd compfest-aic-2026-ds
cp .env.example .env # only if you do not already have .env
python3 run.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). The server deliberately binds to loopback by default, so the unauthenticated MVP cannot be reached from the network.

For the current split, map-first interface, launch the sibling backend instead:

```bash
cd ../compfest-aic-2026-be
python3 run.py
```

It serves the dedicated `compfest-aic-2026-fe` client while retaining this directory as the model/data-science source of truth.

Run the checks with:

```bash
python3 -m unittest discover -s tests -v
```

## Environment

The existing server-side Maps key is read from `GOOGLE_MAP_API`.

```dotenv
GOOGLE_MAP_API=your-ip-restricted-server-key
IOT_SHARED_SECRET=replace-with-a-long-random-secret
HOST=127.0.0.1
PORT=8080
OSRM_BASE_URL=https://router.project-osrm.org
REGION_GEOJSON_URL=https://raw.githubusercontent.com/AlfianAliM/Indonesia-GeoJSON/master/provinsi.geojson
```

In Google Cloud, enable **Routes API** and restrict the server key to the public outbound IP that Google sees. The application invokes it only when a dispatcher selects **Check live traffic**. The key never leaves the server, is ignored by Git, and is never returned by an endpoint.

For a future browser-rendered Google map, create a separate `Maps JavaScript API` key restricted by website referrer; do not reuse this server key in the browser.

## What the MVP does

| Capability | Implementation |
| --- | --- |
| Empty-return risk | Explainable probability from expected empty location, nearby compatible open cargo, next-job approval, fuel flexibility, and vehicle type. |
| ETA | P50/P90 operating baseline from remaining job time, road distance, road class assumptions, and GPS accuracy. Google can give a current traffic confirmation on demand. |
| Cargo–vehicle matching | Hard capacity/vehicle-type/pickup-window checks before profit-and-confidence ranking. |
| Multi-hop backhaul | Enumerates compatible one- and two-load plans; keeps only plans that meet all time windows and margin floors. |
| Data/journey anomaly | HMAC device signature, monotonic sequence, geographic bounds, GPS quality, unusual speed, and route-corridor checks. |
| Price and margin | Transparent fuel, driver, maintenance, and stop-cost calculation; exposes minimum viable quote and expected margin. |

### Route deviation policy

The product does not punish a driver just for taking another road.

1. A dispatcher accepts a recommendation before it becomes an active plan.
2. Verified GPS is measured against a route **corridor**, not an exact polyline.
3. A modest deviation with sufficient pickup-window slack is labelled `valid_reroute`.
4. A larger deviation becomes `replan_required`, which should re-rank cargo and alert a dispatcher.
5. Invalid signatures, impossible speeds, or low GPS accuracy are independent data-quality signals.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Readiness and traffic-adapter status. |
| `GET` | `/api/v1/metrics` | Dispatcher metrics and audit totals. |
| `GET` | `/api/v1/fleet` | Fleet state, risk, ETA, and anomaly status. |
| `GET` | `/api/v1/orders` | Open/assigned cargo orders. |
| `GET` | `/api/v1/regions` | All 38 province geometries, coloured by current truck and accepted telemetry-log activity. |
| `GET` | `/api/v1/recommendations` | Ranked direct and multi-hop plans. |
| `GET` | `/api/v1/recommendations/:id/route-options` | Up to three road-following OSRM alternatives, gray alternative geometry, ETA bands, and local GPS traffic bands. |
| `POST` | `/api/v1/recommendations/:id/decision` | Dispatcher body: `{ "action": "accept" | "reject", "note": "optional" }`. |
| `GET` | `/api/v1/recommendations/:id/live-traffic` | One-time Google Routes API confirmation; `Cache-Control: no-store`. |
| `POST` | `/api/v1/telemetry` | HMAC-signed device event. |
| `POST` | `/api/v1/simulation/tick` | Produce signed digital-twin telemetry for every vehicle. |

### Telemetry contract

```json
{
  "truck_id": "TRK-01",
  "timestamp": "2026-08-23T12:00:00Z",
  "lat": -6.231,
  "lon": 106.967,
  "speed_kph": 54,
  "heading": 89,
  "gps_accuracy_m": 9,
  "cargo_status": "loaded",
  "fuel_pct": 68,
  "sequence": 1,
  "signature": "hex-hmac-sha256"
}
```

The signature is `HMAC-SHA256` of canonical JSON—sorted compact JSON of all fields except `signature`—using `IOT_SHARED_SECRET`. The simulator follows exactly this contract.

## Production next steps

1. Replace the synthetic marketplace with immutable manifest/order events, scans, and dispatcher decisions.
2. Download an OpenStreetMap Indonesia/Java extract and run OSRM/GraphHopper for production road-network matrices; the public OSRM address in `.env.example` is a hackathon fallback, not a production dependency.
3. Ingest real tracker data via an authenticated MQTT adapter, rotate device credentials, and replace the development HMAC secret.
4. Add dispatcher SSO/RBAC, rate limits, PostgreSQL/PostGIS, encrypted secret management, and production observability before exposing the service.
5. Train/calibrate models only from consented historical fleet data under time-based validation; report P50/P90 ETA calibration, precision/recall for empty-return risk, and achieved-versus-predicted margin.
