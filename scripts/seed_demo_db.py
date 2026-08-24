"""Seed the local SQLite audit store with a synthetic 300-truck Indonesia scenario.

This script is deliberately idempotent. It creates labelled historical telemetry
only; all data is generated for the hackathon demo and is not real fleet data.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.engine import FLEET_SEED, NATIONAL_FLEET_TOTAL, national_fleet_seed  # noqa: E402
from app.store import Repository  # noqa: E402


SEED_ID = "synthetic-indonesia-operations-v3"


def timestamp(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_events(now: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create two accepted GPS records for every live synthetic truck."""
    fleet = [*FLEET_SEED, *national_fleet_seed()]
    if len(fleet) != NATIONAL_FLEET_TOTAL:
        raise RuntimeError("Synthetic telemetry must match the 300-truck live fleet")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for truck_index, truck in enumerate(fleet, start=1):
        for sample_index in range(2):
            sequence = 100000 + truck_index * 10 + sample_index
            accepted.append(
                {
                    "truck_id": truck["id"],
                    "device_id": f"gps-demo-{truck['id'].lower()}",
                    "timestamp": timestamp(now - timedelta(minutes=7 * (truck_index * 2 + sample_index))),
                    "lat": round(truck["position"]["lat"] + math.sin(truck_index + sample_index) * 0.004, 6),
                    "lon": round(truck["position"]["lon"] + math.cos(truck_index + sample_index) * 0.004, 6),
                    "speed_kph": max(0, truck["speed_kph"] - sample_index * 3),
                    "heading": (truck["heading"] + sample_index * 8) % 360,
                    "gps_accuracy_m": truck["gps_accuracy_m"],
                    "cargo_status": truck["cargo_status"],
                    "fuel_pct": truck["fuel_pct"],
                    "sequence": sequence,
                    "source": "synthetic_hackathon_seed",
                    "seed_id": SEED_ID,
                    "scenario_note": "Synthetic 300-truck Indonesia digital-twin history; not real fleet telemetry",
                }
            )

    for rejected_index in range(18):
        rejected.append(
            {
                "truck_id": f"SYNTH-REJECTED-{rejected_index + 1:02d}",
                "timestamp": timestamp(now - timedelta(minutes=35 * (rejected_index + 1))),
                "lat": -6.2,
                "lon": 106.8,
                "source": "synthetic_hackathon_seed",
                "seed_id": SEED_ID,
                "scenario_note": "Synthetic rejected device event for audit-volume realism",
            }
        )
    return accepted, rejected


def main() -> int:
    settings = get_settings()
    repository = Repository(settings.data_dir / "optimizer.sqlite3")
    accepted, rejected = build_events(datetime.now(UTC))
    inserted = repository.seed_telemetry(SEED_ID, accepted, rejected)
    metrics = repository.metrics()
    if inserted:
        print(f"Seeded {inserted} synthetic telemetry events for {NATIONAL_FLEET_TOTAL} live demo trucks ({len(accepted)} accepted, {len(rejected)} rejected).")
    else:
        print("Synthetic 300-truck scenario already exists; database left unchanged.")
    print(f"Database totals: {metrics['telemetry_total']} telemetry events, {metrics['telemetry_accepted']} accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
