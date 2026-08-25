#!/usr/bin/env python3
"""Prepare leakage-safe arrays from public, observed logistics records.

This program deliberately does not synthesize rows and does not join records
from unrelated sources.  Every output sample keeps its public source id.  All
derived labels are deterministic functions of fields on that same source row,
trip, route, or courier-day.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import math
import os
import re
import sqlite3
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


SEED_NAMESPACE = "compfest-aic-2026-real-v1"
MAX_CANDIDATES = 32
TELEMETRY_STEPS = 64
SOURCES = ("amazon", "lade", "singapore", "dtcargo", "vius", "scania", "tlc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", "--raw-root", dest="raw", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help="comma-separated subset: " + ",".join(SOURCES),
    )
    parser.add_argument("--amazon-train-cap", type=int, default=60_000)
    parser.add_argument("--amazon-val-cap", type=int, default=10_000)
    parser.add_argument("--amazon-final-cap", type=int, default=10_000)
    parser.add_argument("--lade-train-cap", type=int, default=60_000)
    parser.add_argument("--lade-val-cap", type=int, default=10_000)
    parser.add_argument("--lade-test-cap", type=int, default=10_000)
    parser.add_argument("--tlc-train-cap", type=int, default=100_000)
    parser.add_argument("--tlc-val-cap", type=int, default=20_000)
    parser.add_argument("--tlc-final-cap", type=int, default=20_000)
    parser.add_argument("--skip-raw-sha256", action="store_true")
    return parser.parse_args()


def stable_digest(*parts: object) -> bytes:
    value = "\x1f".join((SEED_NAMESPACE, *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).digest()


def stable_id(*parts: object, length: int = 24) -> str:
    return stable_digest(*parts).hex()[:length]


def stable_unit(*parts: object) -> float:
    return int.from_bytes(stable_digest(*parts)[:8], "big") / float(2**64)


def categorical_unit(value: object) -> float:
    if value is None or str(value).strip() in {"", "nan", "NaN", "NA", "N/A", "X"}:
        return float("nan")
    return stable_unit("category", str(value).strip())


def split_threshold(group: object, names: Sequence[str], cutoffs: Sequence[float]) -> str:
    score = stable_unit("split", group)
    for name, cutoff in zip(names, cutoffs):
        if score < cutoff:
            return name
    return names[-1]


def ranked_group_assignment(
    groups: Iterable[str], names: Sequence[str], fractions: Sequence[float]
) -> dict[str, str]:
    unique = sorted(set(groups), key=lambda value: (stable_digest("group-rank", value), value))
    if not unique:
        return {}
    counts: list[int] = []
    used = 0
    for index, fraction in enumerate(fractions[:-1]):
        count = int(round(len(unique) * fraction))
        if len(unique) >= len(names):
            count = max(1, count)
        count = min(count, len(unique) - used - max(0, len(names) - index - 1))
        counts.append(max(0, count))
        used += counts[-1]
    counts.append(len(unique) - used)
    result: dict[str, str] = {}
    cursor = 0
    for name, count in zip(names, counts):
        for group in unique[cursor : cursor + count]:
            result[group] = name
        cursor += count
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text in {"", "na", "NA", "N/A", "nan", "NaN", "null", "None", "X"}:
        return float("nan")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def bool_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return 1.0
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return 0.0
    return float("nan")


def source_identifier(value: object) -> str:
    """Canonicalize provider IDs without turning integer IDs into ``1.0``."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def first_present(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            value = lowered[name.lower()]
            if value is not None and str(value).strip() != "":
                return value
    return default


def parse_datetime(value: object, date_hint: object | None = None) -> float:
    """Parse public timestamps to UTC-like epoch seconds for intervals only."""
    if value is None or str(value).strip() in {"", "nan", "NaN", "NA", "N/A"}:
        return float("nan")
    text = str(value).strip().replace("/", "-")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        numeric = float(text)
        return numeric if numeric > 100_000_000 else float("nan")
    hint = str(date_hint).strip().replace("/", "-") if date_hint is not None else ""
    candidates = [text]
    if re.fullmatch(r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?", text) and hint:
        candidates.insert(0, f"{hint} {text}")
    if re.fullmatch(r"\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?", text):
        candidates.insert(0, "2022-" + text)
    formats = (
        "%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S",
    )
    for candidate in candidates:
        normalized = candidate.replace("Z", "+00:00")
        for fmt in formats:
            try:
                parsed = datetime.strptime(normalized, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                continue
    return float("nan")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(math.isfinite(value) for value in (lat1, lon1, lat2, lon2)):
        return float("nan")
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value) if math.isfinite(value) else float("nan")


@dataclass
class BoundedSamples:
    cap: int

    def __post_init__(self) -> None:
        self.heap: list[tuple[int, str, dict[str, Any]]] = []
        self.eligible = 0

    def add(self, sample_id: str, sample: dict[str, Any]) -> None:
        self.eligible += 1
        score = int.from_bytes(stable_digest("reservoir", sample_id)[:8], "big")
        item = (-score, sample_id, sample)
        if self.cap <= 0:
            return
        if len(self.heap) < self.cap:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def rows(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heap, key=lambda item: item[1])]


def stack_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not samples:
        raise RuntimeError("refusing to write an empty split")
    result: dict[str, np.ndarray] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, str):
            width = max(1, min(128, max(len(str(value)) for value in values)))
            result[key] = np.asarray(values, dtype=f"U{width}")
        else:
            result[key] = np.asarray(values)
    return result


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    first = next(iter(arrays.values()))
    return {
        "path": path.name,
        "rows": int(first.shape[0]),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "sample_id_order_sha256": hashlib.sha256(
            "\n".join(map(str, arrays.get("sample_id", []))).encode("utf-8")
        ).hexdigest(),
        "group_set_sha256": hashlib.sha256(
            "\n".join(sorted(set(map(str, arrays.get("route_group", arrays.get("split_group", [])))))).encode("utf-8")
        ).hexdigest(),
    }


def raw_ledger(paths: Iterable[Path], with_sha: bool, raw_root: Path) -> list[dict[str, Any]]:
    records = []
    root = raw_root.resolve()
    for path in sorted(set(path.resolve() for path in paths)):
        try:
            display_path = path.relative_to(root).as_posix()
        except ValueError:
            display_path = path.name
        record = {"path": display_path, "bytes": path.stat().st_size}
        if with_sha:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def ensure_group_disjoint(arrays: Mapping[str, Mapping[str, np.ndarray]], group_key: str) -> None:
    seen: set[str] = set()
    for split, values in arrays.items():
        groups = set(map(str, values[group_key]))
        overlap = seen.intersection(groups)
        if overlap:
            raise RuntimeError(f"group leakage into {split}: {sorted(overlap)[:3]}")
        seen.update(groups)


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-8), scale, 1.0).astype(np.float32)
    return median, scale


def robust_apply(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(values)
    transformed = (values.astype(np.float32) - median) / scale
    transformed = np.clip(transformed, -20.0, 20.0)
    transformed[~observed] = 0.0
    return transformed.astype(np.float32), observed


# ---------------------------------------------------------------------------
# Amazon Last Mile Routing Challenge: actual route transitions


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def amazon_package_aggregate(packages: Mapping[str, Any]) -> dict[str, float]:
    count = 0
    volume_cm3 = 0.0
    service_seconds = 0.0
    starts: list[float] = []
    ends: list[float] = []
    for package in packages.values():
        if not isinstance(package, Mapping):
            continue
        count += 1
        dimensions = package.get("dimensions", {})
        depth = float_or_nan(dimensions.get("depth_cm")) if isinstance(dimensions, Mapping) else float("nan")
        height = float_or_nan(dimensions.get("height_cm")) if isinstance(dimensions, Mapping) else float("nan")
        width = float_or_nan(dimensions.get("width_cm")) if isinstance(dimensions, Mapping) else float("nan")
        if all(math.isfinite(value) and value >= 0 for value in (depth, height, width)):
            volume_cm3 += depth * height * width
        service = float_or_nan(package.get("planned_service_time_seconds"))
        if math.isfinite(service) and service >= 0:
            service_seconds += service
        window = package.get("time_window", {})
        if isinstance(window, Mapping):
            start = parse_datetime(window.get("start_time_utc"))
            end = parse_datetime(window.get("end_time_utc"))
            if math.isfinite(start) and math.isfinite(end):
                starts.append(start)
                ends.append(end)
    return {
        "package_count": float(count),
        "volume_cm3": volume_cm3,
        "service_seconds": service_seconds,
        "tw_start": max(starts) if starts else float("nan"),
        "tw_end": min(ends) if ends else float("nan"),
        "tw_present": float(bool(starts)),
    }


def candidate_window(
    remaining_ids: Sequence[str], stops: Mapping[str, Any], current_lat: float,
    current_lng: float, route_id: str, step: int
) -> list[str]:
    """Choose the operational candidate neighbourhood without reading the label.

    The previous implementation guaranteed inclusion of the actual next stop,
    which would leak the label whenever more than 32 stops remained.  This
    version selects the nearest published coordinates and only then checks
    whether the observed next stop happened to be covered.
    """
    scored: list[tuple[float, bytes, str]] = []
    for stop_id in remaining_ids:
        stop = stops.get(stop_id, {})
        lat = float_or_nan(stop.get("lat")) if isinstance(stop, Mapping) else float("nan")
        lng = float_or_nan(stop.get("lng")) if isinstance(stop, Mapping) else float("nan")
        distance = haversine_km(current_lat, current_lng, lat, lng)
        if math.isfinite(distance):
            scored.append((distance, stable_digest("candidate-order", route_id, step, stop_id), stop_id))
    selected = [item[2] for item in sorted(scored)[:MAX_CANDIDATES]]
    selected.sort(key=lambda stop: (stable_digest("candidate-order", route_id, step, stop), stop))
    return selected


def amazon_sample(
    route_id: str,
    route: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, float]],
    sequence: Sequence[str],
    step: int,
) -> dict[str, Any] | None:
    stops = route.get("stops", {})
    if not isinstance(stops, Mapping):
        return None
    current_id, target_id = sequence[step], sequence[step + 1]
    current = stops.get(current_id)
    if not isinstance(current, Mapping) or target_id not in stops:
        return None
    remaining_ids = [stop for stop in sequence[step + 1 :] if stop in stops]
    if target_id not in remaining_ids:
        return None
    capacity = float_or_nan(route.get("executor_capacity_cm3"))
    if not math.isfinite(capacity) or capacity <= 0:
        return None
    departure = parse_datetime(
        f"{route.get('date_YYYY_MM_DD', '')} {route.get('departure_time_utc', '')}"
    )
    if not math.isfinite(departure):
        return None
    current_lat = float_or_nan(current.get("lat"))
    current_lng = float_or_nan(current.get("lng"))
    if not all(math.isfinite(value) for value in (current_lat, current_lng)):
        return None
    candidates = candidate_window(
        remaining_ids, stops, current_lat, current_lng, route_id, step
    )
    if target_id not in candidates:
        return None
    station_ids = [
        stop for stop, value in stops.items()
        if isinstance(value, Mapping) and str(value.get("type", "")).lower() == "station"
    ]
    depot_id = station_ids[0] if station_ids else sequence[0]
    depot = stops.get(depot_id, current)
    depot_lat = float_or_nan(depot.get("lat"))
    depot_lng = float_or_nan(depot.get("lng"))
    if not all(math.isfinite(value) for value in (depot_lat, depot_lng)):
        depot_lat, depot_lng = current_lat, current_lng

    total_volume = sum(value["volume_cm3"] for value in aggregates.values())
    current_remaining = sum(aggregates[stop]["volume_cm3"] for stop in remaining_ids)
    nodes = np.zeros((MAX_CANDIDATES, 16), dtype=np.float32)
    mask = np.zeros((MAX_CANDIDATES,), dtype=bool)
    candidate_ids = np.full((MAX_CANDIDATES,), "", dtype="U24")
    target_index = -1
    for index, stop_id in enumerate(candidates):
        stop = stops[stop_id]
        lat = float_or_nan(stop.get("lat"))
        lng = float_or_nan(stop.get("lng"))
        if not all(math.isfinite(value) for value in (lat, lng)):
            return None
        agg = aggregates[stop_id]
        has_window = bool(agg["tw_present"])
        tw_start = (agg["tw_start"] - departure) / 86400.0 if has_window else 0.0
        tw_end = (agg["tw_end"] - departure) / 86400.0 if has_window else 0.0
        after = current_remaining - agg["volume_cm3"]
        stop_type = str(stop.get("type", "")).lower()
        nodes[index] = np.asarray(
            [
                float(stop_type == "station"),
                0.0,
                min(2.0, math.log1p(haversine_km(current_lat, current_lng, lat, lng)) / math.log1p(2000.0)),
                0.0,
                min(2.0, (agg["service_seconds"] / 3600.0) / 24.0),
                max(-2.0, min(2.0, tw_start)),
                max(-2.0, min(2.0, tw_end)),
                0.0,
                min(2.0, agg["volume_cm3"] / capacity),
                min(2.0, math.log1p(haversine_km(depot_lat, depot_lng, lat, lng)) / math.log1p(2000.0)),
                step / max(1, len(sequence) - 1),
                float(str(stop.get("zone_id")) == str(current.get("zone_id"))),
                0.0,
                0.0,
                0.0,
                0.375,
            ],
            dtype=np.float32,
        )
        mask[index] = True
        candidate_ids[index] = stable_id("amazon-stop", route_id, stop_id)
        if stop_id == target_id:
            target_index = index
    if target_index < 0 or not np.all(np.isfinite(nodes[mask])):
        return None

    departure_dt = datetime.fromtimestamp(departure, tz=timezone.utc)
    dep_angle = 2 * math.pi * (departure_dt.hour * 3600 + departure_dt.minute * 60 + departure_dt.second) / 86400
    dow_angle = 2 * math.pi * departure_dt.weekday() / 7
    current_aggregate = aggregates[current_id]
    context = np.asarray(
        [
            math.sin(dep_angle), math.cos(dep_angle),
            math.sin(dow_angle), math.cos(dow_angle),
            0.0,
            max(0.0, min(2.0, (capacity - current_remaining) / capacity)),
            0.0,
            0.0,
            0.0,
            max(0.0, min(2.0, current_remaining / capacity)),
            len(candidates) / 64.0,
            step / max(1, len(sequence) - 1),
            0.0,
            0.0,
            0.0,
            0.5,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(context)):
        return None
    sample_id = stable_id("amazon-sample", route_id, step)
    return {
        "nodes": nodes,
        "context": context,
        "mask": mask,
        "target": np.int64(target_index),
        "route_group": stable_id("amazon-route", route_id),
        "sample_id": sample_id,
        "source_id": f"{route_id}:{current_id}>{target_id}",
        "candidate_id": candidate_ids,
    }


def prepare_amazon(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    reservoirs = {
        "train": BoundedSamples(args.amazon_train_cap),
        "val": BoundedSamples(args.amazon_val_cap),
        "final": BoundedSamples(args.amazon_final_cap),
    }
    raw_paths: list[Path] = []
    route_counts: dict[str, int] = defaultdict(int)
    skipped = 0
    for official in ("train", "eval"):
        folder = raw / "amazon" / official
        paths = {
            key: folder / f"{key}.json"
            for key in ("route_data", "package_data", "actual_sequences")
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Amazon raw layout missing: " + ", ".join(missing))
        raw_paths.extend(paths.values())
        routes = load_json(paths["route_data"])
        packages = load_json(paths["package_data"])
        sequences = load_json(paths["actual_sequences"])
        for route_id, sequence_record in sequences.items():
            route = routes.get(route_id)
            package_route = packages.get(route_id, {})
            if not isinstance(route, Mapping) or not isinstance(sequence_record, Mapping):
                skipped += 1
                continue
            actual = sequence_record.get("actual", sequence_record)
            if not isinstance(actual, Mapping):
                skipped += 1
                continue
            sequence = [key for key, _ in sorted(actual.items(), key=lambda item: float_or_nan(item[1]))]
            if len(sequence) < 2:
                skipped += 1
                continue
            split = "final" if official == "eval" else split_threshold(
                route_id, ("train", "val"), (0.85, 1.0)
            )
            route_counts[split] += 1
            stops = route.get("stops", {})
            aggregates = {
                stop: amazon_package_aggregate(
                    package_route.get(stop, {})
                    if isinstance(package_route.get(stop, {}), Mapping)
                    else {}
                )
                for stop in stops
            }
            # Select real transitions deterministically before reading their
            # targets. This keeps preparation bounded without inventing rows.
            step_indices = sorted(
                sorted(
                    range(len(sequence) - 1),
                    key=lambda step: stable_digest("amazon-step-cap", route_id, step),
                )[:24]
            )
            for step in step_indices:
                sample = amazon_sample(route_id, route, aggregates, sequence, step)
                if sample is None:
                    skipped += 1
                else:
                    reservoirs[split].add(sample["sample_id"], sample)
        del routes, packages, sequences

    split_arrays = {split: stack_samples(reservoir.rows()) for split, reservoir in reservoirs.items()}
    ensure_group_disjoint(split_arrays, "route_group")
    files = {}
    for split, arrays in split_arrays.items():
        files[split] = write_npz(output / f"amazon_{split}.npz", arrays)
        files[split]["eligible_rows"] = reservoirs[split].eligible
    return {
        "kind": "real operational last-mile routes",
        "observed_target": "actual next stop in driver sequence",
        "route_counts": dict(route_counts),
        "skipped_rows_or_routes": skipped,
        "raw_files": raw_ledger(raw_paths, not args.skip_raw_sha256, raw),
        "files": files,
    }


# ---------------------------------------------------------------------------
# LaDe: real courier-day task transitions and observed completion intervals


def _lade_time(value: object, ds: object) -> float:
    parsed = parse_datetime(value)
    if math.isfinite(parsed):
        return parsed
    text = str(value).strip()
    if re.fullmatch(r"\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", text):
        return parse_datetime("2022-" + text)
    day = str(ds).strip()
    if re.fullmatch(r"\d{3,4}", day) and re.fullmatch(r"\d{2}:\d{2}:\d{2}", text):
        month, date = int(day) // 100, int(day) % 100
        return parse_datetime(f"2022-{month:02d}-{date:02d} {text}")
    return float("nan")


def _lade_event_rows(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "delivery":
        finish_col, start_col = "sign_time", "receipt_time"
        gps_lng_col, gps_lat_col = "receipt_lng", "receipt_lat"
        window_start_col = window_end_col = None
    else:
        finish_col, start_col = "got_time", "accept_time"
        gps_lng_col, gps_lat_col = "accept_gps_lng", "accept_gps_lat"
        window_start_col, window_end_col = "book_start_time", "expect_got_time"
    rows = pd.DataFrame(
        {
            "order_id": frame["order_id"].astype(str),
            "courier": frame["delivery_user_id"].astype(str),
            "city": frame["from_city_name"].astype(str),
            "ds": frame["ds"].astype(str),
            "x": pd.to_numeric(frame["poi_lng"], errors="coerce"),
            "y": pd.to_numeric(frame["poi_lat"], errors="coerce"),
            "aoi": frame.get("aoi_id", "").astype(str),
            "typecode": frame.get("typecode", "").astype(str),
            "start": [
                _lade_time(value, ds) for value, ds in zip(frame[start_col], frame["ds"])
            ],
            "finish": [
                _lade_time(value, ds) for value, ds in zip(frame[finish_col], frame["ds"])
            ],
            "accept_x": pd.to_numeric(frame.get(gps_lng_col), errors="coerce"),
            "accept_y": pd.to_numeric(frame.get(gps_lat_col), errors="coerce"),
            "kind": kind,
        }
    )
    if window_start_col:
        rows["window_start"] = [
            _lade_time(value, ds) for value, ds in zip(frame[window_start_col], frame["ds"])
        ]
        rows["window_end"] = [
            _lade_time(value, ds) for value, ds in zip(frame[window_end_col], frame["ds"])
        ]
    else:
        rows["window_start"] = np.nan
        rows["window_end"] = np.nan
    rows = rows[
        np.isfinite(rows["x"]) & np.isfinite(rows["y"]) & np.isfinite(rows["finish"])
    ].copy()
    return rows


def _lade_route_samples(group: pd.DataFrame, group_id: str) -> Iterator[dict[str, Any]]:
    ordered = group.sort_values(["finish", "order_id"], kind="mergesort").reset_index(drop=True)
    if len(ordered) < 2:
        return
    xs = ordered["x"].to_numpy(float)
    ys = ordered["y"].to_numpy(float)
    centre_x, centre_y = float(np.nanmedian(xs)), float(np.nanmedian(ys))
    step_indices = sorted(
        sorted(
            range(len(ordered) - 1),
            key=lambda step: stable_digest("lade-step-cap", group_id, step),
        )[:12]
    )
    for step in step_indices:
        current = ordered.iloc[step]
        now = float(current["finish"])
        known_at_decision = np.isfinite(ordered["start"].astype(float)) & (
            ordered["start"].astype(float) <= now
        )
        progress = min(1.0, (step + 1) / max(1, int(known_at_decision.sum())))
        remaining = ordered.iloc[step + 1 :].copy()
        # A courier cannot select an order that has not yet been accepted or
        # received at this decision timestamp. Rows with an unknown start time
        # are excluded rather than treated as if they were already available.
        remaining = remaining[
            np.isfinite(remaining["start"].astype(float))
            & (remaining["start"].astype(float) <= now)
        ].copy()
        if remaining.empty:
            continue
        remaining["distance_km"] = [
            haversine_km(
                float(current["y"]),
                float(current["x"]),
                float(candidate_y),
                float(candidate_x),
            )
            for candidate_x, candidate_y in zip(remaining["x"], remaining["y"])
        ]
        candidates = remaining.sort_values(["distance_km", "order_id"], kind="mergesort").head(MAX_CANDIDATES)
        target_order = str(ordered.iloc[step + 1]["order_id"])
        if target_order not in set(candidates["order_id"].astype(str)):
            continue
        candidates = candidates.assign(
            candidate_sort=candidates["order_id"].map(
                lambda value: stable_id("lade-candidate", group_id, step, value)
            )
        ).sort_values("candidate_sort", kind="mergesort")
        nodes = np.zeros((MAX_CANDIDATES, 16), np.float32)
        mask = np.zeros((MAX_CANDIDATES,), bool)
        target = -1
        for index, (_, candidate) in enumerate(candidates.iterrows()):
            window_present = math.isfinite(float(candidate["window_start"])) and math.isfinite(float(candidate["window_end"]))
            window_start = (float(candidate["window_start"]) - now) / 3600.0 if window_present else 0.0
            window_end = (float(candidate["window_end"]) - now) / 3600.0 if window_present else 0.0
            nodes[index] = np.asarray(
                [
                    0.0,
                    float(candidate["kind"] == "pickup"),
                    min(2.0, math.log1p(max(0.0, float(candidate["distance_km"]))) / math.log1p(2000.0)),
                    0.0,
                    0.0,
                    max(-48.0, min(48.0, window_start)) / 24.0,
                    max(-48.0, min(48.0, window_end)) / 24.0,
                    0.0,
                    0.0,
                    min(
                        2.0,
                        math.log1p(
                            max(
                                0.0,
                                haversine_km(
                                    centre_y,
                                    centre_x,
                                    float(candidate["y"]),
                                    float(candidate["x"]),
                                ),
                            )
                        )
                        / math.log1p(2000.0),
                    ),
                    progress,
                    float(str(candidate["aoi"]) == str(current["aoi"])),
                    0.0,
                    0.0,
                    0.0,
                    0.625,
                ], np.float32,
            )
            mask[index] = True
            if str(candidate["order_id"]) == target_order:
                target = index
        if target < 0 or not np.all(np.isfinite(nodes[mask])):
            continue
        timestamp = datetime.fromtimestamp(now, tz=timezone.utc)
        day_angle = 2 * math.pi * (timestamp.hour * 60 + timestamp.minute) / 1440.0
        context = np.asarray(
            [
                math.sin(day_angle), math.cos(day_angle),
                math.sin(2 * math.pi * timestamp.weekday() / 7.0),
                math.cos(2 * math.pi * timestamp.weekday() / 7.0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                len(candidates) / 64.0,
                progress,
                0.0,
                0.0,
                1.0,
                0.75,
            ], np.float32,
        )
        eta_hours = max(0.0, (float(ordered.iloc[step + 1]["finish"]) - now) / 3600.0)
        sample_id = stable_id("lade-sample", group_id, step)
        yield {
            "nodes": nodes,
            "context": context,
            "mask": mask,
            "target": np.int64(target),
            "eta_hours": np.float32(min(eta_hours, 72.0)),
            "stop_target": np.int64(step == len(ordered) - 2),
            "route_group": stable_id("lade-group", group_id),
            "sample_id": sample_id,
            "source_id": f"{group_id}:{current['order_id']}>{target_order}",
        }


def prepare_lade(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    folder = raw / "lade"
    paths = [folder / "delivery_five_cities.csv", folder / "pickup_five_cities.csv"]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("LaDe merged delivery/pickup CSV files are required under raw/lade")
    delivery = pd.read_csv(paths[0], low_memory=False)
    pickup = pd.read_csv(paths[1], low_memory=False)
    events = pd.concat(
        (_lade_event_rows(delivery, "delivery"), _lade_event_rows(pickup, "pickup")),
        ignore_index=True,
    )
    del delivery, pickup
    events["group_id"] = (
        events["kind"].astype(str) + ":" + events["courier"].astype(str) + ":" + events["ds"].astype(str)
    )
    courier_split = ranked_group_assignment(
        events["courier"].astype(str), ("train", "val", "test"), (0.8, 0.1, 0.1)
    )
    caps = {"train": args.lade_train_cap, "val": args.lade_val_cap, "test": args.lade_test_cap}
    reservoirs = {key: BoundedSamples(value) for key, value in caps.items()}
    group_counts: dict[str, int] = defaultdict(int)
    for group_id, group in events.groupby("group_id", sort=False):
        split = courier_split[str(group.iloc[0]["courier"])]
        group_counts[split] += 1
        for sample in _lade_route_samples(group, str(group_id)):
            reservoirs[split].add(sample["sample_id"], sample)
    arrays = {split: stack_samples(reservoir.rows()) for split, reservoir in reservoirs.items()}
    ensure_group_disjoint(arrays, "route_group")
    files: dict[str, Any] = {}
    for split, values in arrays.items():
        files[split] = write_npz(output / f"lade_{split}.npz", values)
        files[split]["eligible_rows"] = reservoirs[split].eligible
    return {
        "kind": "real courier task events",
        "observed_target": "actual next task by observed completion time",
        "eta_target": "same-source interval between observed completion timestamps",
        "courier_disjoint": True,
        "group_counts": dict(group_counts),
        "raw_files": raw_ledger(paths, not args.skip_raw_sha256, raw),
        "files": files,
    }


# ---------------------------------------------------------------------------
# Singapore commercial vehicle OBD/fuel: trip-level temporal windows


def _singapore_location(raw: Path) -> tuple[Path | None, Path | None]:
    directories = (
        raw / "singapore-commercial-truck" / "figshare",
        raw / "singapore-commercial-vehicle" / "figshare",
    )
    for directory in directories:
        if directory.exists():
            return directory, None
    archives = (
        raw / "singapore-commercial-truck.zip",
        raw / "singapore-commercial-vehicle" / "figshare_TRR.zip",
    )
    for archive in archives:
        if archive.exists():
            return None, archive
    raise FileNotFoundError("Singapore Figshare archive or extracted figshare directory is missing")


def _singapore_names(directory: Path | None, archive: Path | None) -> list[str]:
    if directory is not None:
        return sorted(path.name for path in directory.glob("*_GPS_OBD.csv"))
    assert archive is not None
    with zipfile.ZipFile(archive) as bundle:
        return sorted(Path(name).name for name in bundle.namelist() if name.endswith("_GPS_OBD.csv"))


def _singapore_read(name: str, directory: Path | None, archive: Path | None) -> pd.DataFrame:
    if directory is not None:
        return pd.read_csv(directory / name, low_memory=False)
    assert archive is not None
    with zipfile.ZipFile(archive) as bundle:
        member = next(item for item in bundle.namelist() if Path(item).name == name)
        with bundle.open(member) as handle:
            return pd.read_csv(handle, low_memory=False)


def prepare_singapore(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    directory, archive = _singapore_location(raw)
    names = _singapore_names(directory, archive)
    split_vehicles = {
        "train": set("ABCDEF"),
        "val": set("GH"),
        "test": set("IJ"),
    }
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in split_vehicles}
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        vehicle = name.split("_", 1)[0]
        split = next(key for key, values in split_vehicles.items() if vehicle in values)
        gps = _singapore_read(name, directory, archive)
        fuel_name = f"{vehicle}_tripFuelUsed.csv"
        fuel_frame = _singapore_read(fuel_name, directory, archive)
        fuel_map = {
            source_identifier(trip_id): float_or_nan(fuel_value)
            for trip_id, fuel_value in zip(
                fuel_frame["TripID"], fuel_frame["OBD - Fuel used (L)"]
            )
        }
        gps["timestamp"] = pd.to_datetime(gps["timestamp_GMTplus8"], errors="coerce")
        gps["trip_source_id"] = gps["TripID"].map(source_identifier)
        for trip_id, trip in gps.groupby("trip_source_id", sort=False):
            trip = trip.sort_values("timestamp", kind="mergesort").dropna(subset=["timestamp"])
            if len(trip) < 8:
                continue
            prefix_end = max(4, len(trip) // 2)
            prefix = trip.iloc[:prefix_end]
            future = trip.iloc[prefix_end:]
            indices = np.linspace(0, len(prefix) - 1, min(TELEMETRY_STEPS, len(prefix)), dtype=int)
            observed = prefix.iloc[np.unique(indices)]
            sequence = np.zeros((TELEMETRY_STEPS, 8), np.float32)
            mask = np.zeros((TELEMETRY_STEPS,), bool)
            timestamps = observed["timestamp"].astype("int64").to_numpy() / 1.0e9
            deltas = np.diff(timestamps, prepend=timestamps[0])
            speed_kmh = pd.to_numeric(observed["speed_kmh"], errors="coerce").to_numpy(float)
            acceleration = np.full((len(observed),), np.nan, dtype=float)
            if len(observed):
                acceleration[0] = 0.0 if math.isfinite(speed_kmh[0]) else np.nan
            valid_acceleration = (
                (deltas[1:] > 0.0)
                & np.isfinite(speed_kmh[1:])
                & np.isfinite(speed_kmh[:-1])
            )
            acceleration[1:][valid_acceleration] = (
                (speed_kmh[1:][valid_acceleration] - speed_kmh[:-1][valid_acceleration])
                / 3.6
                / deltas[1:][valid_acceleration]
            )
            columns = {
                "speed": speed_kmh,
                "engine": pd.to_numeric(observed["engine_status"], errors="coerce").to_numpy(float),
                "grade": pd.to_numeric(observed["road_grade_proportion"], errors="coerce").to_numpy(float),
                "load": pd.to_numeric(observed["engine_load_obd_percent"], errors="coerce").to_numpy(float),
                "maf": pd.to_numeric(observed["mass_air_flow_g_per_s"], errors="coerce").to_numpy(float),
                "acceleration": acceleration,
            }
            n = len(observed)
            valid_count = np.sum(np.stack([np.isfinite(value) for value in columns.values()]), axis=0) / 6.0
            sequence[:n] = np.stack(
                (
                    np.nan_to_num(columns["speed"] / 100.0),
                    np.nan_to_num(columns["engine"] / 2.0),
                    np.nan_to_num(columns["grade"] * 20.0),
                    np.nan_to_num(columns["load"] / 100.0),
                    np.nan_to_num(np.log1p(np.clip(columns["maf"], 0, None)) / 5.0),
                    np.nan_to_num(np.clip(columns["acceleration"] / 10.0, -1.5, 1.5)),
                    np.clip(deltas, 0, 300) / 300.0,
                    valid_count,
                ), axis=1,
            ).astype(np.float32)
            mask[:n] = True
            full_timestamps = trip["timestamp"].astype("int64").to_numpy() / 1.0e9
            duration_seconds = max(0.0, full_timestamps[-1] - full_timestamps[0])
            fuel_l = fuel_map.get(str(trip_id), float("nan"))
            future_engine = pd.to_numeric(future["engine_status"], errors="coerce").to_numpy(float)
            observed_future_engine = np.isfinite(future_engine)
            idle_label = (
                float(np.mean(future_engine[observed_future_engine] == 2.0) >= 0.2)
                if observed_future_engine.any()
                else float("nan")
            )
            targets = np.asarray(
                [
                    math.log1p(fuel_l) if math.isfinite(fuel_l) and fuel_l >= 0 else np.nan,
                    math.log1p(duration_seconds),
                    np.nan,
                    idle_label,
                ], np.float32,
            )
            sample_id = stable_id("singapore-trip", vehicle, trip_id)
            samples[split].append(
                {
                    "sequence": sequence,
                    "mask": mask,
                    "targets": targets,
                    "split_group": vehicle,
                    "sample_id": sample_id,
                    "source_id": f"vehicle={vehicle};trip={trip_id}",
                }
            )
            counts[split] += 1
    arrays = {split: stack_samples(rows) for split, rows in samples.items()}
    ensure_group_disjoint(arrays, "split_group")
    files = {split: write_npz(output / f"singapore_{split}.npz", values) for split, values in arrays.items()}
    raw_paths = [archive] if archive else sorted(directory.glob("*.csv"))
    return {
        "kind": "real commercial-vehicle OBD observations",
        "features": ["first-half trip speed", "engine state", "road grade", "OBD engine load", "OBD mass air flow", "speed-derived acceleration", "sample interval", "observed fraction"],
        "leakage_control": "only the first half of each trip is encoded and the publisher instantaneous fuel stream is excluded",
        "targets": ["full-trip published OBD fuel", "full-trip timestamp-derived duration", "load unavailable", "future-half idle-heavy label"],
        "vehicle_disjoint": True,
        "trip_counts": dict(counts),
        "raw_files": raw_ledger(raw_paths, not args.skip_raw_sha256, raw),
        "files": files,
    }


# ---------------------------------------------------------------------------
# DT-CARGO heavy-truck track summaries


def _find_one(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"{label} not found; tried: {', '.join(map(str, paths))}")


def prepare_dtcargo(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    tracks_path = _find_one(
        (raw / "dt-cargo" / "tracks.csv", raw / "dt-cargo" / "input" / "public" / "tracks.csv"),
        "DT-CARGO tracks.csv",
    )
    fleet_path = _find_one(
        (raw / "dt-cargo" / "fleet.csv", raw / "dt-cargo" / "input" / "public" / "fleet.csv"),
        "DT-CARGO fleet.csv",
    )
    tracks = pd.read_csv(tracks_path, low_memory=False)
    fleet = pd.read_csv(fleet_path, low_memory=False)
    data = tracks.merge(fleet, on="vehicle_id", how="left", validate="many_to_one")
    vehicle_split = ranked_group_assignment(
        data["vehicle_id"].astype(str), ("train", "val", "test"), (0.7, 0.15, 0.15)
    )
    output_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "val", "test")}
    for _, row in data.iterrows():
        start = parse_datetime(row.get("start_time"))
        stop = parse_datetime(row.get("stop_time"))
        duration = stop - start if math.isfinite(start) and math.isfinite(stop) else float("nan")
        if not math.isfinite(duration) or duration < 0:
            continue
        hour = datetime.fromtimestamp(start, tz=timezone.utc).hour if math.isfinite(start) else 0
        raw_values = np.asarray(
            [
                signed_log1p(float_or_nan(row.get("distance"))),
                signed_log1p(float_or_nan(row.get("track_gap"))),
                float_or_nan(row.get("avg_speed")) / 40.0,
                float_or_nan(row.get("max_speed")) / 50.0,
                float_or_nan(row.get("avg_hdop")) / 5.0,
                signed_log1p(float_or_nan(row.get("gross_vehicle_weight"))) / 12.0,
                signed_log1p(float_or_nan(row.get("total_mass_with_trailer"))) / 12.0,
                float_or_nan(row.get("axle_class")) / 100.0,
            ], np.float32,
        )
        missing = (~np.isfinite(raw_values)).astype(np.float32)
        features = np.concatenate((np.nan_to_num(raw_values), missing)).astype(np.float32)
        # Hour is deliberately omitted to retain the exact 8 values + 8 masks schema.
        targets = np.asarray(
            [
                math.log1p(duration),
                float_or_nan(row.get("r_signal_loss")),
                bool_or_nan(row.get("home_base")),
                bool_or_nan(row.get("long_haul")),
            ], np.float32,
        )
        vehicle = str(row["vehicle_id"])
        split = vehicle_split[vehicle]
        sample_id = stable_id("dtcargo-track", row.get("track_id"))
        output_rows[split].append(
            {
                "features": features,
                "targets": targets,
                "split_group": vehicle,
                "sample_id": sample_id,
                "source_id": f"track={row.get('track_id')};vehicle={vehicle}",
            }
        )
    arrays = {split: stack_samples(rows) for split, rows in output_rows.items()}
    ensure_group_disjoint(arrays, "split_group")
    files = {split: write_npz(output / f"dtcargo_{split}.npz", values) for split, values in arrays.items()}
    return {
        "kind": "real class-N3 heavy-truck track summaries",
        "vehicle_disjoint": True,
        "targets": ["timestamp-derived duration", "observed signal-loss ratio", "published home-base flag", "published long-haul flag"],
        "raw_files": raw_ledger((tracks_path, fleet_path), not args.skip_raw_sha256, raw),
        "files": files,
    }


# ---------------------------------------------------------------------------
# VIUS static real-truck deadhead, repositioning and loaded-mile survey labels


def prepare_vius(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = _find_one(
        (raw / "vius" / "vius_2021_puf_csv.zip", raw / "vius-2021" / "vius_2021_puf_csv.zip"),
        "VIUS PUF ZIP",
    )
    with zipfile.ZipFile(path) as bundle:
        member = next(name for name in bundle.namelist() if name.lower().endswith(".csv"))
        with bundle.open(member) as handle:
            frame = pd.read_csv(handle, low_memory=False)
    feature_columns = (
        "AVGWEIGHT", "GVWR_CLASS", "MPG", "MILESANNL", "MILESLIFE", "MONTHOPERATE",
        "TOWCAPACITY", "WEIGHOUTPCT", "RO_0_50", "RO_51_100", "RO_201_500", "RO_GT500",
    )
    raw_features = np.stack(
        [pd.to_numeric(frame[column].replace("X", np.nan), errors="coerce").to_numpy(float) for column in feature_columns],
        axis=1,
    )
    targets = np.stack(
        [
            pd.to_numeric(frame[column].replace("X", np.nan), errors="coerce").to_numpy(float) / 100.0
            for column in ("DEADHEADPCT", "REPOSITIONPCT", "LOADEDPCT")
        ], axis=1,
    ).astype(np.float32)
    keep = np.isfinite(targets).any(axis=1)
    frame = frame.loc[keep].reset_index(drop=True)
    raw_features, targets = raw_features[keep], targets[keep]
    state_split = ranked_group_assignment(
        frame["REGSTATE"].astype(str), ("train", "val", "test"), (0.7, 0.15, 0.15)
    )
    split_values = np.asarray([state_split[str(value)] for value in frame["REGSTATE"]])
    train_median, train_scale = robust_fit(raw_features[split_values == "train"])
    scaled, observed = robust_apply(raw_features, train_median, train_scale)
    features = np.concatenate((scaled, (~observed).astype(np.float32)), axis=1).astype(np.float32)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        indices = np.flatnonzero(split_values == split)
        arrays[split] = {
            "features": features[indices],
            "targets": targets[indices],
            "split_group": frame.loc[indices, "REGSTATE"].astype(str).to_numpy(dtype="U4"),
            "sample_id": np.asarray([stable_id("vius", value) for value in frame.loc[indices, "ID"]], dtype="U24"),
            "source_id": np.asarray([f"VIUS-ID={value}" for value in frame.loc[indices, "ID"]], dtype="U32"),
        }
    ensure_group_disjoint(arrays, "split_group")
    files = {split: write_npz(output / f"vius_{split}.npz", values) for split, values in arrays.items()}
    return {
        "kind": "real public-use truck survey records",
        "state_disjoint": True,
        "targets": ["annual DEADHEADPCT", "annual REPOSITIONPCT", "annual LOADEDPCT"],
        "normalization": {"columns": list(feature_columns), "median": train_median.tolist(), "iqr": train_scale.tolist(), "missing_mask_appended": True},
        "raw_files": raw_ledger((path,), not args.skip_raw_sha256, raw),
        "files": files,
    }


# ---------------------------------------------------------------------------
# Scania APS real operational sensor/counter failure classification


def _read_scania(bundle: zipfile.ZipFile, name_fragment: str) -> pd.DataFrame:
    member = next(name for name in bundle.namelist() if name_fragment in name)
    with bundle.open(member) as handle:
        header_row = None
        for index, line in enumerate(handle):
            if line.startswith(b"class,"):
                header_row = index
                break
        if header_row is None:
            raise RuntimeError(f"Scania CSV header not found in {member}")
        handle.seek(0)
        return pd.read_csv(
            handle,
            skiprows=header_row,
            na_values=["na"],
            low_memory=False,
        )


def prepare_scania(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = _find_one(
        (raw / "scania" / "aps_failure_at_scania_trucks.zip", raw / "scania-aps" / "aps+failure+at+scania+trucks.zip"),
        "Scania APS ZIP",
    )
    with zipfile.ZipFile(path) as bundle:
        train_frame = _read_scania(bundle, "training_set.csv")
        final_frame = _read_scania(bundle, "test_set.csv")
    feature_columns = [column for column in train_frame.columns if column != "class"]
    if len(feature_columns) != 170:
        raise RuntimeError(f"Scania expected 170 fields, found {len(feature_columns)}")
    internal_split = np.asarray(
        ["val" if stable_unit("scania-val", index) < 0.15 else "train" for index in range(len(train_frame))]
    )
    train_raw = train_frame[feature_columns].to_numpy(float)
    final_raw = final_frame[feature_columns].to_numpy(float)
    median, scale = robust_fit(train_raw[internal_split == "train"])
    transformed_train, observed_train = robust_apply(train_raw, median, scale)
    transformed_final, observed_final = robust_apply(final_raw, median, scale)
    features_train = np.concatenate((transformed_train, (~observed_train).astype(np.float32)), axis=1)
    features_final = np.concatenate((transformed_final, (~observed_final).astype(np.float32)), axis=1)
    labels_train = (train_frame["class"].astype(str).str.lower() == "pos").to_numpy(np.float32)
    labels_final = (final_frame["class"].astype(str).str.lower() == "pos").to_numpy(np.float32)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val"):
        indices = np.flatnonzero(internal_split == split)
        arrays[split] = {
            "features": features_train[indices].astype(np.float32),
            "label": labels_train[indices],
            "split_group": np.asarray([f"train-row-{index}" for index in indices], dtype="U24"),
            "sample_id": np.asarray([stable_id("scania-train", index) for index in indices], dtype="U24"),
            "source_id": np.asarray([f"training-row={index}" for index in indices], dtype="U32"),
        }
    final_indices = np.arange(len(final_frame))
    arrays["final"] = {
        "features": features_final.astype(np.float32),
        "label": labels_final,
        "split_group": np.asarray([f"test-row-{index}" for index in final_indices], dtype="U24"),
        "sample_id": np.asarray([stable_id("scania-final", index) for index in final_indices], dtype="U24"),
        "source_id": np.asarray([f"official-test-row={index}" for index in final_indices], dtype="U40"),
    }
    files = {split: write_npz(output / f"scania_{split}.npz", values) for split, values in arrays.items()}
    return {
        "kind": "real Scania heavy-truck operational sensors and counters",
        "official_test_held_out": True,
        "target": "APS component failure versus other failure",
        "normalization": {"feature_count": 170, "missing_mask_appended": True, "median_sha256": hashlib.sha256(median.tobytes()).hexdigest(), "iqr_sha256": hashlib.sha256(scale.tobytes()).hexdigest()},
        "raw_files": raw_ledger((path,), not args.skip_raw_sha256, raw),
        "files": files,
    }


# ---------------------------------------------------------------------------
# NYC TLC: real metered trip fare proxy, chronological split


def prepare_tlc(raw: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = _find_one(
        (raw / "tlc" / "yellow_tripdata_2024-01.parquet", raw / "nyc-tlc" / "yellow_tripdata_2024-01.parquet"),
        "NYC TLC parquet",
    )
    frame = pd.read_parquet(path)
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"], errors="coerce")
    duration = (dropoff - pickup).dt.total_seconds()
    distance = pd.to_numeric(frame["trip_distance"], errors="coerce")
    fare = pd.to_numeric(frame["fare_amount"], errors="coerce")
    valid = (
        pickup.notna() & dropoff.notna() & (pickup.dt.year == 2024) & (pickup.dt.month == 1)
        & duration.between(60, 6 * 3600) & distance.between(0.05, 200)
        & fare.between(2.5, 1000)
    )
    frame, pickup, duration, distance, fare = (
        value.loc[valid].reset_index(drop=True) for value in (frame, pickup, duration, distance, fare)
    )
    hour_angle = 2 * np.pi * (pickup.dt.hour.to_numpy() * 60 + pickup.dt.minute.to_numpy()) / 1440.0
    dow_angle = 2 * np.pi * pickup.dt.dayofweek.to_numpy() / 7.0
    passenger = pd.to_numeric(frame["passenger_count"], errors="coerce").to_numpy(float)
    ratecode = pd.to_numeric(frame["RatecodeID"], errors="coerce").to_numpy(float)
    pickup_zone = pd.to_numeric(frame["PULocationID"], errors="coerce").to_numpy(float)
    dropoff_zone = pd.to_numeric(frame["DOLocationID"], errors="coerce").to_numpy(float)
    missing_fraction = (~np.isfinite(np.stack((passenger, ratecode, pickup_zone, dropoff_zone), axis=1))).mean(axis=1)
    features = np.stack(
        (
            np.log1p(distance.to_numpy(float)),
            np.log1p(duration.to_numpy(float) / 60.0),
            np.clip(distance.to_numpy(float) / (duration.to_numpy(float) / 3600.0), 0, 120) / 120.0,
            np.nan_to_num(passenger) / 6.0,
            missing_fraction,
            np.nan_to_num(ratecode) / 10.0,
            np.nan_to_num(pickup_zone) / 300.0,
            np.nan_to_num(dropoff_zone) / 300.0,
            np.sin(hour_angle), np.cos(hour_angle), np.sin(dow_angle), np.cos(dow_angle),
        ), axis=1,
    ).astype(np.float32)
    target = np.log1p(fare.to_numpy(float)).astype(np.float32)
    day = pickup.dt.day.to_numpy()
    split_masks = {"train": day <= 20, "val": (day >= 21) & (day <= 25), "final": day >= 26}
    caps = {"train": args.tlc_train_cap, "val": args.tlc_val_cap, "final": args.tlc_final_cap}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split, split_mask in split_masks.items():
        available = np.flatnonzero(split_mask)
        if len(available) > caps[split]:
            order = sorted(available, key=lambda index: stable_digest("tlc-cap", int(index)))[: caps[split]]
            indices = np.asarray(sorted(order), dtype=int)
        else:
            indices = available
        arrays[split] = {
            "features": features[indices],
            "target": target[indices],
            "split_group": pickup.iloc[indices].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
            "sample_id": np.asarray([stable_id("tlc", int(index)) for index in indices], dtype="U24"),
            "source_id": np.asarray([f"row={int(index)}" for index in indices], dtype="U24"),
        }
    ensure_group_disjoint(arrays, "split_group")
    files = {split: write_npz(output / f"tlc_{split}.npz", values) for split, values in arrays.items()}
    return {
        "kind": "real vendor-submitted metered passenger trips; cost-domain proxy only",
        "chronological_split": "days 1-20 train, 21-25 validation, 26-31 final",
        "features": ["log distance", "log duration", "average speed", "passenger count", "source-field missing fraction", "rate code", "pickup zone", "dropoff zone", "time-of-day sine/cosine", "day-of-week sine/cosine"],
        "target": "log1p observed fare_amount",
        "raw_files": raw_ledger((path,), not args.skip_raw_sha256, raw),
        "files": files,
    }


def main() -> int:
    args = parse_args()
    selected = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    unknown = sorted(set(selected) - set(SOURCES))
    if unknown:
        raise ValueError(f"unknown sources: {unknown}")
    raw = args.raw.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    builders: dict[str, Callable[[Path, Path, argparse.Namespace], dict[str, Any]]] = {
        "amazon": prepare_amazon,
        "lade": prepare_lade,
        "singapore": prepare_singapore,
        "dtcargo": prepare_dtcargo,
        "vius": prepare_vius,
        "scania": prepare_scania,
        "tlc": prepare_tlc,
    }
    evidence: dict[str, Any] = {}
    started = datetime.now(timezone.utc)
    for source in selected:
        print(json.dumps({"event": "prepare_start", "source": source}), flush=True)
        evidence[source] = builders[source](raw, output, args)
        print(json.dumps({"event": "prepare_complete", "source": source}), flush=True)
    manifest = {
        "schema_version": "1.0",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "contains_synthetic": False,
        "cross_source_rows_joined": False,
        "random_feature_generation": False,
        "source_specific_batches": True,
        "candidate_selection_reads_target": False,
        "selected_sources": list(selected),
        "preprocessing_sha256": sha256_file(Path(__file__).resolve()),
        "sources": evidence,
        "claim_boundary": "per-source component validation only; no joined Haulio operational outcome",
    }
    json_write(output / "manifest.json", manifest)
    print(json.dumps({"event": "all_complete", "manifest": str(output / 'manifest.json')}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
