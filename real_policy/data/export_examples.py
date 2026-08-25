#!/usr/bin/env python3
"""Export inference examples from held-out real-source rows only."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from prepare_real import (
    _singapore_location,
    _singapore_read,
    parse_datetime,
    source_identifier,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> Mapping[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def clean_number(value: Any, *, positive: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0.0) or (not positive and number < 0.0):
        return None
    return number


def write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def provenance(manifest: Mapping[str, Any], source: str, split: str, sample_id: str, source_id: str, transform: str) -> dict[str, Any]:
    return {
        "public_source": source,
        "split": split,
        "sample_id": sample_id,
        "source_id": source_id,
        "transform": transform,
        "raw_artifacts": manifest["sources"][source]["raw_files"],
    }


def telemetry_example(raw: Path, processed: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    ds = load_npz(processed / "singapore_test.npz")
    source_id = str(ds["source_id"][0])
    sample_id = str(ds["sample_id"][0])
    tokens = dict(part.split("=", 1) for part in source_id.split(";"))
    vehicle, trip_id = tokens["vehicle"], tokens["trip"]
    directory, archive = _singapore_location(raw)
    frame = _singapore_read(f"{vehicle}_GPS_OBD.csv", directory, archive)
    frame["trip_source_id"] = frame["TripID"].map(source_identifier)
    frame["timestamp"] = pd.to_datetime(frame["timestamp_GMTplus8"], errors="coerce")
    trip = frame.loc[frame["trip_source_id"] == trip_id].sort_values("timestamp", kind="mergesort").dropna(subset=["timestamp"])
    prefix = trip.iloc[: max(4, len(trip) // 2)]
    indices = np.unique(np.linspace(0, len(prefix) - 1, min(64, len(prefix)), dtype=int))
    observed = prefix.iloc[indices]
    fields = {
        "speed_kmh": pd.to_numeric(observed["speed_kmh"], errors="coerce"),
        "engine_status": pd.to_numeric(observed["engine_status"], errors="coerce"),
        "road_grade": pd.to_numeric(observed["road_grade_proportion"], errors="coerce"),
        "engine_load_pct": pd.to_numeric(observed["engine_load_obd_percent"], errors="coerce"),
        "mass_air_flow_g_s": pd.to_numeric(observed["mass_air_flow_g_per_s"], errors="coerce"),
    }
    samples: list[dict[str, Any]] = []
    previous_time: pd.Timestamp | None = None
    previous_speed: float | None = None
    for index, row in observed.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Singapore")
        timestamp = timestamp.tz_convert("UTC")
        values = {name: float(series.loc[index]) for name, series in fields.items()}
        if not all(math.isfinite(value) for value in values.values()):
            continue
        if not (0 <= values["speed_kmh"] <= 180 and 0 <= values["engine_status"] <= 2 and -0.4 <= values["road_grade"] <= 0.4 and 0 <= values["engine_load_pct"] <= 100 and values["mass_air_flow_g_s"] >= 0):
            continue
        delta = 0.0 if previous_time is None else (timestamp - previous_time).total_seconds()
        if previous_time is not None and delta <= 0:
            continue
        acceleration = 0.0 if previous_speed is None else (values["speed_kmh"] - previous_speed) / 3.6 / delta
        if not -15 <= acceleration <= 15:
            continue
        samples.append({
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            **values,
            "longitudinal_acceleration_m_s2": acceleration,
            "delta_t_s": delta,
            "observed_fraction": 1.0,
        })
        previous_time, previous_speed = timestamp, values["speed_kmh"]
    if not samples:
        raise RuntimeError("no fully observed Singapore held-out prefix sample")
    return {
        "schema_version": "haulio.real-policy.telemetry.v1",
        "request_id": sample_id,
        "samples": samples,
        "_provenance": provenance(manifest, "singapore", "test", sample_id, source_id, "fully observed samples from the same held-out first-half trip prefix"),
    }


def track_example(raw: Path, processed: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    ds = load_npz(processed / "dtcargo_test.npz")
    wanted = str(ds["source_id"][0])
    sample_id = str(ds["sample_id"][0])
    tracks_path = next(path for path in (raw / "dt-cargo" / "tracks.csv", raw / "dt-cargo" / "input" / "public" / "tracks.csv") if path.exists())
    fleet_path = next(path for path in (raw / "dt-cargo" / "fleet.csv", raw / "dt-cargo" / "input" / "public" / "fleet.csv") if path.exists())
    data = pd.read_csv(tracks_path, low_memory=False).merge(pd.read_csv(fleet_path, low_memory=False), on="vehicle_id", how="left", validate="many_to_one")
    keys = data.apply(lambda row: f"track={row.get('track_id')};vehicle={str(row['vehicle_id'])}", axis=1)
    row = data.loc[keys == wanted].iloc[0]
    start = parse_datetime(row.get("start_time"))
    if not math.isfinite(start):
        raise RuntimeError("DT-CARGO held-out start timestamp is unavailable")
    return {
        "schema_version": "haulio.real-policy.truck-track.v1",
        "request_id": sample_id,
        "track": {
            # Python 3.8's runtime parser accepts microseconds but rejects the
            # nanosecond string Pandas can produce from floating-point epochs.
            "start_time_utc": datetime.fromtimestamp(start, timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "distance_m": clean_number(row.get("distance")),
            "track_gap_m": clean_number(row.get("track_gap")),
            "avg_speed_m_s": clean_number(row.get("avg_speed")),
            "max_speed_m_s": clean_number(row.get("max_speed")),
            "avg_hdop": clean_number(row.get("avg_hdop")),
            "gvwr_kg": clean_number(row.get("gross_vehicle_weight"), positive=True),
            "gcwr_kg": clean_number(row.get("total_mass_with_trailer"), positive=True),
            "axle_class": clean_number(row.get("axle_class")),
        },
        "_provenance": provenance(manifest, "dtcargo", "test", sample_id, wanted, "same-source semantic fields from one held-out track and fleet row; UTC timestamp serialized at microsecond precision"),
    }


def vius_example(raw: Path, processed: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    ds = load_npz(processed / "vius_test.npz")
    source_id = str(ds["source_id"][0])
    sample_id = str(ds["sample_id"][0])
    wanted = source_id.split("=", 1)[1]
    archive = next(path for path in (raw / "vius" / "vius_2021_puf_csv.zip", raw / "vius-2021" / "vius_2021_puf_csv.zip") if path.exists())
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.lower().endswith(".csv"))
        with bundle.open(member) as handle:
            frame = pd.read_csv(handle, low_memory=False)
    row = frame.loc[frame["ID"].map(source_identifier) == wanted].iloc[0]
    columns = manifest["sources"]["vius"]["normalization"]["columns"]
    raw_values = {}
    for column in columns:
        value = row[column]
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        raw_values[column] = number if math.isfinite(number) else None
    return {
        "schema_version": "haulio.real-policy.deadhead.v1",
        "feature_schema": "vius-2021-12-values-plus-mask.v1",
        "request_id": sample_id,
        "raw_values": raw_values,
        "_provenance": provenance(manifest, "vius", "test", sample_id, source_id, "unmodified fields from one state-disjoint public-use survey row"),
    }


def transformed_examples(processed: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scania = load_npz(processed / "scania_final.npz")
    scania_features = scania["features"][0].astype(float)
    scania_id, scania_source = str(scania["sample_id"][0]), str(scania["source_id"][0])
    health = {
        "schema_version": "haulio.real-policy.health.v1",
        "feature_schema": "scania-aps-170-plus-missing-mask.v1",
        "request_id": scania_id,
        "normalized_sensor_values": scania_features[:170].tolist(),
        "missing_mask": [bool(value) for value in scania_features[170:]],
        "_provenance": provenance(manifest, "scania", "final", scania_id, scania_source, "exact train-normalized values and missing bits from one official test row"),
    }
    tlc = load_npz(processed / "tlc_final.npz")
    tlc_id, tlc_source = str(tlc["sample_id"][0]), str(tlc["source_id"][0])
    price = {
        "schema_version": "haulio.real-policy.price.v1",
        "feature_schema": "nyc-tlc-logfare-12.v1",
        "request_id": tlc_id,
        "normalized_source_features": tlc["features"][0].astype(float).tolist(),
        "_provenance": provenance(manifest, "tlc", "final", tlc_id, tlc_source, "exact canonical features from one chronological held-out trip row"),
    }
    return health, price


def main() -> int:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.processed / "manifest.json").read_text(encoding="utf-8"))
    health, price = transformed_examples(args.processed, manifest)
    payloads = {
        "singapore_telemetry_heldout.json": telemetry_example(args.raw, args.processed, manifest),
        "dtcargo_track_heldout.json": track_example(args.raw, args.processed, manifest),
        "scania_health_heldout.json": health,
        "vius_deadhead_heldout.json": vius_example(args.raw, args.processed, manifest),
        "tlc_price_heldout.json": price,
    }
    for name, payload in payloads.items():
        write(args.output / name, payload)
    print(json.dumps({"status": "complete", "examples": sorted(payloads)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
