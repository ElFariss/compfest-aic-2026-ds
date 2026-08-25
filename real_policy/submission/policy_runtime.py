"""Immutable inference runtime for the real-public-data backhaul policy.

Only frozen TorchScript inference is implemented here.  The module contains no
optimizer, gradient update, feedback store, model reload, or automatic dispatch
action.  Missing runtime modalities produce an explicit ABSTAIN response.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROUTE_SCHEMA = "haulio.real-policy.route.v1"
TELEMETRY_SCHEMA = "haulio.real-policy.telemetry.v1"
TRACK_SCHEMA = "haulio.real-policy.truck-track.v1"
HEALTH_SCHEMA = "haulio.real-policy.health.v1"
DEADHEAD_SCHEMA = "haulio.real-policy.deadhead.v1"
PRICE_SCHEMA = "haulio.real-policy.price.v1"

ROUTE_NODE_FEATURES = (
    "is_depot",
    "is_pickup",
    "current_to_candidate_distance_km",
    "current_to_candidate_travel_time_h",
    "service_time_h",
    "window_open_delta_h",
    "window_close_delta_h",
    "demand_weight_ratio",
    "demand_volume_ratio",
    "candidate_to_depot_distance_km",
    "route_progress_fraction",
    "same_zone",
    "priority_norm",
    "hazmat_flag",
    "temperature_control_flag",
    "missing_fraction",
)
ROUTE_CONTEXT_FEATURES = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "remaining_weight_ratio",
    "remaining_volume_ratio",
    "fuel_fraction",
    "telemetry_age_s",
    "gps_accuracy_m",
    "active_load_fraction",
    "candidate_count",
    "completed_fraction",
    "route_elapsed_h",
    "remaining_budget_h",
    "domain_shift_score",
    "missing_fraction",
)
TELEMETRY_FEATURES = (
    "speed_kmh",
    "engine_status",
    "road_grade",
    "engine_load_pct",
    "mass_air_flow_g_s",
    "longitudinal_acceleration_m_s2",
    "delta_t_s",
    "observed_fraction",
)
DTCARGO_FEATURES = (
    "log_distance_m",
    "log_track_gap_m",
    "avg_speed_m_s",
    "max_speed_m_s",
    "avg_hdop",
    "gvwr_kg",
    "gcwr_kg",
    "axle_class",
    "distance_missing",
    "track_gap_missing",
    "avg_speed_missing",
    "max_speed_missing",
    "hdop_missing",
    "gvwr_missing",
    "gcwr_missing",
    "axle_class_missing",
)
VIUS_FEATURES = (
    "AVGWEIGHT",
    "GVWR_CLASS",
    "MPG",
    "MILESANNL",
    "MILESLIFE",
    "MONTHOPERATE",
    "TOWCAPACITY",
    "WEIGHOUTPCT",
    "RO_0_50",
    "RO_51_100",
    "RO_201_500",
    "RO_GT500",
    "AVGWEIGHT_missing",
    "GVWR_CLASS_missing",
    "MPG_missing",
    "MILESANNL_missing",
    "MILESLIFE_missing",
    "MONTHOPERATE_missing",
    "TOWCAPACITY_missing",
    "WEIGHOUTPCT_missing",
    "RO_0_50_missing",
    "RO_51_100_missing",
    "RO_201_500_missing",
    "RO_GT500_missing",
)
PRICE_FEATURES = (
    "source_feature_0",
    "source_feature_1",
    "source_feature_2",
    "source_feature_3",
    "source_feature_4",
    "source_feature_5",
    "source_feature_6",
    "source_feature_7",
    "source_feature_8",
    "source_feature_9",
    "source_feature_10",
    "source_feature_11",
)

MAX_CANDIDATES = 32
APS_VALUE_COUNT = 170
APS_INPUT_COUNT = 340


class InputIssue(ValueError):
    """A request cannot be evaluated without inventing or trusting bad data."""

    def __init__(self, code: str, field: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message


def abstain(request_id: str | None, issue: InputIssue) -> Dict[str, Any]:
    return {
        "status": "ABSTAIN",
        "request_id": request_id,
        "reason_code": issue.code,
        "field": issue.field,
        "message": issue.message,
        "dispatcher_approval_required": True,
        "model_updated": False,
    }


def _request_id(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        value = payload.get("request_id")
        if isinstance(value, str) and value.strip():
            return value.strip()[:128]
    return None


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputIssue("MISSING_REQUIRED_MODALITY", field, f"{field} must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in mapping or mapping[key] is None:
        name = f"{field}.{key}" if field else key
        raise InputIssue("MISSING_REQUIRED_MODALITY", name, f"{name} is required")
    return mapping[key]


def _array(value: Any, field: str, *, minimum: int = 0, maximum: int | None = None) -> List[Any]:
    if not isinstance(value, list):
        raise InputIssue("MISSING_REQUIRED_MODALITY", field, f"{field} must be an array")
    if len(value) < minimum or (maximum is not None and len(value) > maximum):
        limit = f"{minimum}..{maximum}" if maximum is not None else f"at least {minimum}"
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} must contain {limit} entries")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputIssue("MISSING_REQUIRED_MODALITY", field, f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > 128:
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} exceeds 128 characters")
    return result


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputIssue("MISSING_REQUIRED_MODALITY", field, f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} must be finite")
    return result


def _bounded(value: Any, field: str, lower: float, upper: float) -> float:
    result = _number(value, field)
    if not lower <= result <= upper:
        raise InputIssue(
            "INVALID_RUNTIME_VALUE", field, f"{field} must be between {lower} and {upper}"
        )
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _number(value, field)
    if result < 0.0:
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} must be non-negative")
    return result


def _positive(value: Any, field: str) -> float:
    result = _number(value, field)
    if result <= 0.0:
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} must be positive")
    return result


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise InputIssue("MISSING_REQUIRED_MODALITY", field, f"{field} must be boolean")
    return value


def _choice(value: Any, field: str, choices: Sequence[str]) -> str:
    result = _identifier(value, field)
    if result not in choices:
        raise InputIssue(
            "INVALID_RUNTIME_VALUE", field, f"{field} must be one of {', '.join(choices)}"
        )
    return result


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise InputIssue("MISSING_REQUIRED_MODALITY", field, f"{field} must be an ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} is not valid ISO-8601") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise InputIssue("INVALID_RUNTIME_VALUE", field, f"{field} must include a UTC offset")
    return result.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _scale_distance(km: float) -> float:
    return _clip(math.log1p(km) / math.log1p(2000.0), 0.0, 2.0)


def _scale_hours(hours: float) -> float:
    return _clip(hours / 24.0, -2.0, 2.0)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(value, 80.0))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(value, -80.0))
    return exponent / (1.0 + exponent)


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactContract:
    manifest: Mapping[str, Any]
    manifest_path: Path
    artifact_path: Path
    artifact_sha256: str
    model_config: Mapping[str, Any]


def validate_artifacts(artifact_dir: Path) -> ArtifactContract:
    root = artifact_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("frozen manifest.json is required")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen manifest.json is unreadable or invalid") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("frozen manifest must be an object")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("manifest.artifact is required")
    filename = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError("manifest artifact path must be one local filename")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError("manifest artifact SHA-256 is invalid")
    artifact_path = root / filename
    if not artifact_path.is_file():
        raise RuntimeError(f"frozen artifact is missing: {filename}")
    actual_hash = _sha256(artifact_path)
    if actual_hash != expected_hash.lower():
        raise RuntimeError("frozen artifact checksum does not match manifest")

    config = manifest.get("model_config")
    if not isinstance(config, Mapping):
        raise RuntimeError("manifest.model_config is required")
    expected_dimensions = {
        "node_dim": 16,
        "context_dim": 16,
        "telemetry_dim": 8,
        "dtcargo_dim": 16,
        "vius_dim": 24,
        "health_dim": 340,
        "price_dim": 12,
        "max_candidates": 32,
        "telemetry_steps": 64,
    }
    for name, expected in expected_dimensions.items():
        if config.get(name) != expected:
            raise RuntimeError(f"manifest model_config.{name} must equal {expected}")
    required_flags = {
        "contains_synthetic": False,
        "cross_dataset_rows_joined": False,
        "dispatcher_approval_required": True,
        "runtime_auto_update": False,
    }
    for name, expected in required_flags.items():
        if manifest.get(name) is not expected:
            raise RuntimeError(f"manifest must declare {name}={str(expected).lower()}")
    preprocessing = manifest.get("preprocessing")
    output_schema = manifest.get("output_schema")
    if not isinstance(preprocessing, Mapping) or not isinstance(output_schema, Mapping):
        raise RuntimeError("manifest preprocessing and output_schema are required")
    vius = preprocessing.get("vius")
    if not isinstance(vius, Mapping):
        raise RuntimeError("manifest preprocessing.vius is required")
    columns = vius.get("columns")
    medians = vius.get("median")
    scales = vius.get("iqr")
    numeric = lambda value: (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    if not (
        isinstance(columns, list)
        and isinstance(medians, list)
        and isinstance(scales, list)
        and len(columns) == len(medians) == len(scales) == 12
        and all(isinstance(column, str) and bool(column.strip()) for column in columns)
        and len(set(columns)) == 12
        and all(numeric(value) for value in medians)
        and all(numeric(value) and float(value) > 0.0 for value in scales)
    ):
        raise RuntimeError(
            "manifest VIUS normalization requires twelve unique column names, "
            "finite medians, and finite positive IQRs"
        )
    price = preprocessing.get("price")
    if not isinstance(price, Mapping) or not isinstance(price.get("feature_schema"), str):
        raise RuntimeError("manifest price feature schema is required")
    for name in ("route_eta", "telemetry", "dtcargo", "vius", "health", "price"):
        if not isinstance(output_schema.get(name), str):
            raise RuntimeError(f"manifest output_schema.{name} is required")
    return ArtifactContract(manifest, manifest_path, artifact_path, actual_hash, config)


@dataclass
class Candidate:
    stop_id: str
    stop_type: str
    shipment_id: str | None
    cargo_type: str
    required_vehicle_type: str
    zone_id: str
    load_delta_kg: float
    load_delta_cm3: float
    service_time_s: float
    window_start: datetime
    window_end: datetime
    distance_to_depot_km: float
    priority: float
    hazmat: bool
    temperature_controlled: bool
    missing_fraction: float


@dataclass
class VehicleState:
    vehicle_id: str
    origin_id: str
    vehicle_type: str
    compatible_cargo_types: Tuple[str, ...]
    zone_id: str
    capacity_kg: float
    capacity_cm3: float
    load_kg: float
    load_cm3: float
    fuel_fraction: float
    minimum_fuel_fraction: float
    telemetry_age_s: float
    max_telemetry_age_s: float
    gps_accuracy_m: float
    max_gps_accuracy_m: float
    current_time: datetime
    route_started: datetime
    route_time_budget_s: float
    domain_shift_score: float
    max_domain_shift_score: float
    max_plausible_speed_kmh: float
    missing_fraction: float
    onboard_shipments: set[str]


class RouteRequest:
    """Validated semantic route input and canonical feature construction."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != ROUTE_SCHEMA:
            raise InputIssue("INVALID_SCHEMA", "schema_version", f"schema_version must be {ROUTE_SCHEMA}")
        self.request_id = _identifier(_required(payload, "request_id", ""), "request_id")
        vehicle = _object(_required(payload, "vehicle", ""), "vehicle")
        vehicle_id = _identifier(_required(vehicle, "vehicle_id", "vehicle"), "vehicle.vehicle_id")
        origin_id = _identifier(_required(vehicle, "origin_id", "vehicle"), "vehicle.origin_id")
        vehicle_type = _identifier(
            _required(vehicle, "vehicle_type", "vehicle"), "vehicle.vehicle_type"
        )
        cargo_types = tuple(
            _identifier(item, f"vehicle.compatible_cargo_types[{index}]")
            for index, item in enumerate(
                _array(
                    _required(vehicle, "compatible_cargo_types", "vehicle"),
                    "vehicle.compatible_cargo_types",
                    minimum=1,
                )
            )
        )
        if len(set(cargo_types)) != len(cargo_types):
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                "vehicle.compatible_cargo_types",
                "compatible cargo types must be unique",
            )
        capacity_kg = _positive(_required(vehicle, "capacity_kg", "vehicle"), "vehicle.capacity_kg")
        capacity_cm3 = _positive(
            _required(vehicle, "capacity_cm3", "vehicle"), "vehicle.capacity_cm3"
        )
        load_kg = _bounded(
            _required(vehicle, "current_load_kg", "vehicle"),
            "vehicle.current_load_kg",
            0.0,
            capacity_kg,
        )
        load_cm3 = _bounded(
            _required(vehicle, "current_load_cm3", "vehicle"),
            "vehicle.current_load_cm3",
            0.0,
            capacity_cm3,
        )
        current_time = _timestamp(
            _required(vehicle, "current_time_utc", "vehicle"), "vehicle.current_time_utc"
        )
        route_started = _timestamp(
            _required(vehicle, "route_started_time_utc", "vehicle"),
            "vehicle.route_started_time_utc",
        )
        if route_started > current_time:
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                "vehicle.route_started_time_utc",
                "route start cannot be after current time",
            )
        telemetry_age = _nonnegative(
            _required(vehicle, "telemetry_age_s", "vehicle"), "vehicle.telemetry_age_s"
        )
        max_telemetry_age = _positive(
            _required(vehicle, "max_telemetry_age_s", "vehicle"),
            "vehicle.max_telemetry_age_s",
        )
        if telemetry_age > max_telemetry_age:
            raise InputIssue(
                "STALE_RUNTIME_MODALITY",
                "vehicle.telemetry_age_s",
                "truck telemetry is older than the declared safety limit",
            )
        onboard = {
            _identifier(item, f"vehicle.onboard_shipment_ids[{index}]")
            for index, item in enumerate(
                _array(
                    _required(vehicle, "onboard_shipment_ids", "vehicle"),
                    "vehicle.onboard_shipment_ids",
                )
            )
        }
        fuel_fraction = _bounded(
            _required(vehicle, "fuel_fraction", "vehicle"),
            "vehicle.fuel_fraction",
            0.0,
            1.0,
        )
        minimum_fuel_fraction = _bounded(
            _required(vehicle, "minimum_fuel_fraction", "vehicle"),
            "vehicle.minimum_fuel_fraction",
            0.0,
            1.0,
        )
        if fuel_fraction < minimum_fuel_fraction:
            raise InputIssue(
                "INSUFFICIENT_FUEL_RESERVE",
                "vehicle.fuel_fraction",
                "observed fuel is below the operator-declared routing reserve",
            )
        gps_accuracy_m = _nonnegative(
            _required(vehicle, "gps_accuracy_m", "vehicle"), "vehicle.gps_accuracy_m"
        )
        max_gps_accuracy_m = _positive(
            _required(vehicle, "max_gps_accuracy_m", "vehicle"),
            "vehicle.max_gps_accuracy_m",
        )
        if gps_accuracy_m > max_gps_accuracy_m:
            raise InputIssue(
                "GNSS_QUALITY_UNSAFE",
                "vehicle.gps_accuracy_m",
                "observed GNSS accuracy is worse than the operator-declared safety limit",
            )

        domain_shift_score = _bounded(
            _required(vehicle, "domain_shift_score", "vehicle"),
            "vehicle.domain_shift_score",
            0.0,
            1.0,
        )
        max_domain_shift_score = _bounded(
            _required(vehicle, "max_domain_shift_score", "vehicle"),
            "vehicle.max_domain_shift_score",
            0.0,
            1.0,
        )
        if domain_shift_score > max_domain_shift_score:
            raise InputIssue(
                "DOMAIN_SHIFT_UNSAFE",
                "vehicle.domain_shift_score",
                "domain-shift risk exceeds the operator-declared inference limit",
            )
        max_plausible_speed_kmh = _positive(
            _required(vehicle, "max_plausible_speed_kmh", "vehicle"),
            "vehicle.max_plausible_speed_kmh",
        )

        self.vehicle = VehicleState(
            vehicle_id=vehicle_id,
            origin_id=origin_id,
            vehicle_type=vehicle_type,
            compatible_cargo_types=cargo_types,
            zone_id=_identifier(_required(vehicle, "zone_id", "vehicle"), "vehicle.zone_id"),
            capacity_kg=capacity_kg,
            capacity_cm3=capacity_cm3,
            load_kg=load_kg,
            load_cm3=load_cm3,
            fuel_fraction=fuel_fraction,
            minimum_fuel_fraction=minimum_fuel_fraction,
            telemetry_age_s=telemetry_age,
            max_telemetry_age_s=max_telemetry_age,
            gps_accuracy_m=gps_accuracy_m,
            max_gps_accuracy_m=max_gps_accuracy_m,
            current_time=current_time,
            route_started=route_started,
            route_time_budget_s=_positive(
                _required(vehicle, "route_time_budget_s", "vehicle"),
                "vehicle.route_time_budget_s",
            ),
            domain_shift_score=domain_shift_score,
            max_domain_shift_score=max_domain_shift_score,
            max_plausible_speed_kmh=max_plausible_speed_kmh,
            missing_fraction=_bounded(
                _required(vehicle, "source_missing_fraction", "vehicle"),
                "vehicle.source_missing_fraction",
                0.0,
                1.0,
            ),
            onboard_shipments=onboard,
        )

        raw_candidates = _array(
            _required(payload, "candidates", ""),
            "candidates",
            minimum=1,
            maximum=MAX_CANDIDATES,
        )
        self.candidates: List[Candidate] = []
        stop_ids: set[str] = set()
        shipment_stops: Dict[Tuple[str, str], str] = {}
        for index, raw in enumerate(raw_candidates):
            field = f"candidates[{index}]"
            item = _object(raw, field)
            stop_id = _identifier(_required(item, "stop_id", field), f"{field}.stop_id")
            if stop_id == origin_id or stop_id in stop_ids:
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE", f"{field}.stop_id", "stop IDs must be unique and differ from origin_id"
                )
            stop_ids.add(stop_id)
            stop_type = _choice(
                _required(item, "stop_type", field),
                f"{field}.stop_type",
                ("pickup", "delivery", "service", "depot"),
            )
            shipment_value = item.get("shipment_id")
            shipment_id = None
            if stop_type in ("pickup", "delivery"):
                shipment_id = _identifier(shipment_value, f"{field}.shipment_id")
                key = (shipment_id, stop_type)
                if key in shipment_stops:
                    raise InputIssue(
                        "INVALID_RUNTIME_VALUE",
                        f"{field}.shipment_id",
                        f"shipment {shipment_id} has duplicate {stop_type} stops",
                    )
                shipment_stops[key] = stop_id
            elif shipment_value is not None:
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE",
                    f"{field}.shipment_id",
                    "service and depot stops cannot declare shipment_id",
                )
            delta_kg = _number(_required(item, "load_delta_kg", field), f"{field}.load_delta_kg")
            delta_cm3 = _number(
                _required(item, "load_delta_cm3", field), f"{field}.load_delta_cm3"
            )
            if stop_type == "pickup" and (delta_kg <= 0.0 or delta_cm3 <= 0.0):
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE", field, "pickup load deltas must both be positive"
                )
            if stop_type == "delivery" and (delta_kg >= 0.0 or delta_cm3 >= 0.0):
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE", field, "delivery load deltas must both be negative"
                )
            if stop_type in ("service", "depot") and (delta_kg != 0.0 or delta_cm3 != 0.0):
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE", field, "service and depot load deltas must be zero"
                )
            window_start = _timestamp(
                _required(item, "time_window_start_utc", field), f"{field}.time_window_start_utc"
            )
            window_end = _timestamp(
                _required(item, "time_window_end_utc", field), f"{field}.time_window_end_utc"
            )
            if window_end <= window_start:
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE", field, "time-window end must be after its start"
                )
            self.candidates.append(
                Candidate(
                    stop_id=stop_id,
                    stop_type=stop_type,
                    shipment_id=shipment_id,
                    cargo_type=_identifier(
                        _required(item, "cargo_type", field), f"{field}.cargo_type"
                    ),
                    required_vehicle_type=_identifier(
                        _required(item, "required_vehicle_type", field),
                        f"{field}.required_vehicle_type",
                    ),
                    zone_id=_identifier(_required(item, "zone_id", field), f"{field}.zone_id"),
                    load_delta_kg=delta_kg,
                    load_delta_cm3=delta_cm3,
                    service_time_s=_nonnegative(
                        _required(item, "service_time_s", field), f"{field}.service_time_s"
                    ),
                    window_start=window_start,
                    window_end=window_end,
                    distance_to_depot_km=_nonnegative(
                        _required(item, "distance_to_depot_km", field),
                        f"{field}.distance_to_depot_km",
                    ),
                    priority=_bounded(
                        _required(item, "priority", field), f"{field}.priority", 0.0, 1.0
                    ),
                    hazmat=_boolean(_required(item, "hazmat", field), f"{field}.hazmat"),
                    temperature_controlled=_boolean(
                        _required(item, "temperature_controlled", field),
                        f"{field}.temperature_controlled",
                    ),
                    missing_fraction=_bounded(
                        _required(item, "source_missing_fraction", field),
                        f"{field}.source_missing_fraction",
                        0.0,
                        1.0,
                    ),
                )
            )

        pickup_shipments = {
            shipment for shipment, stop_type in shipment_stops if stop_type == "pickup"
        }
        delivery_shipments = {
            shipment for shipment, stop_type in shipment_stops if stop_type == "delivery"
        }
        unpaired_pickups = pickup_shipments - delivery_shipments
        if unpaired_pickups:
            raise InputIssue(
                "MISSING_REQUIRED_MODALITY",
                "candidates",
                "every pickup must include its delivery in the same immutable route snapshot",
            )
        unknown_deliveries = delivery_shipments - pickup_shipments - onboard
        if unknown_deliveries:
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                "candidates",
                "a delivery without an in-request pickup must be declared onboard",
            )
        paired_candidates: Dict[str, Dict[str, Candidate]] = {}
        for candidate in self.candidates:
            if candidate.shipment_id is not None:
                paired_candidates.setdefault(candidate.shipment_id, {})[candidate.stop_type] = candidate
        for shipment_id in pickup_shipments:
            pickup = paired_candidates[shipment_id]["pickup"]
            delivery = paired_candidates[shipment_id]["delivery"]
            if not (
                math.isclose(pickup.load_delta_kg, -delivery.load_delta_kg, rel_tol=1.0e-6, abs_tol=1.0e-6)
                and math.isclose(pickup.load_delta_cm3, -delivery.load_delta_cm3, rel_tol=1.0e-6, abs_tol=1.0e-3)
            ):
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE",
                    "candidates",
                    f"shipment {shipment_id} pickup and delivery load deltas must balance",
                )
            if (
                pickup.cargo_type != delivery.cargo_type
                or pickup.required_vehicle_type != delivery.required_vehicle_type
            ):
                raise InputIssue(
                    "INVALID_RUNTIME_VALUE",
                    "candidates",
                    f"shipment {shipment_id} has inconsistent pickup/delivery compatibility metadata",
                )

        self.travel: Dict[Tuple[str, str], Tuple[float, float, bool]] = {}
        matrix = _object(_required(payload, "travel_matrix", ""), "travel_matrix")
        origins = [origin_id] + [candidate.stop_id for candidate in self.candidates]
        destinations = [candidate.stop_id for candidate in self.candidates]
        for origin in origins:
            row = _object(_required(matrix, origin, "travel_matrix"), f"travel_matrix.{origin}")
            for destination in destinations:
                if origin == destination:
                    continue
                edge_field = f"travel_matrix.{origin}.{destination}"
                edge = _object(_required(row, destination, f"travel_matrix.{origin}"), edge_field)
                travel_time_s = _nonnegative(
                    _required(edge, "travel_time_s", edge_field), f"{edge_field}.travel_time_s"
                )
                distance_km = _nonnegative(
                    _required(edge, "distance_km", edge_field), f"{edge_field}.distance_km"
                )
                road_available = _boolean(
                    _required(edge, "road_available", edge_field), f"{edge_field}.road_available"
                )
                if road_available and distance_km > 0.0 and travel_time_s <= 0.0:
                    raise InputIssue(
                        "INVALID_RUNTIME_VALUE",
                        edge_field,
                        "an available non-zero-distance edge must have positive travel time",
                    )
                if road_available and travel_time_s > 0.0:
                    implied_speed_kmh = distance_km / (travel_time_s / 3600.0)
                    if implied_speed_kmh > self.vehicle.max_plausible_speed_kmh:
                        raise InputIssue(
                            "IMPLAUSIBLE_TRAVEL_EDGE",
                            edge_field,
                            "edge speed exceeds the operator-declared plausibility limit",
                        )
                self.travel[(origin, destination)] = (
                    travel_time_s,
                    distance_km,
                    road_available,
                )

    def edge(self, origin: str, destination: str) -> Tuple[float, float, bool]:
        try:
            return self.travel[(origin, destination)]
        except KeyError as error:
            raise InputIssue(
                "MISSING_REQUIRED_MODALITY",
                f"travel_matrix.{origin}.{destination}",
                "a complete observed/road-derived travel matrix is required",
            ) from error

    def feasibility(
        self,
        candidate: Candidate,
        origin: str,
        current_time: datetime,
        load_kg: float,
        load_cm3: float,
        onboard: set[str],
        picked_up: set[str],
    ) -> Tuple[List[str], datetime, datetime, datetime]:
        reasons: List[str] = []
        travel_s, _, road_available = self.edge(origin, candidate.stop_id)
        try:
            arrival = datetime.fromtimestamp(current_time.timestamp() + travel_s, tz=timezone.utc)
            service_start = max(arrival, candidate.window_start)
            service_end = datetime.fromtimestamp(
                service_start.timestamp() + candidate.service_time_s, tz=timezone.utc
            )
            route_deadline = datetime.fromtimestamp(
                self.vehicle.route_started.timestamp() + self.vehicle.route_time_budget_s,
                tz=timezone.utc,
            )
        except (OverflowError, OSError, ValueError):
            return ["TIME_VALUE_OUT_OF_RANGE"], current_time, current_time, current_time
        if not road_available:
            reasons.append("ROAD_UNAVAILABLE")
        if service_end > candidate.window_end:
            reasons.append("TIME_WINDOW_MISSED")
        if service_end > route_deadline:
            reasons.append("ROUTE_TIME_BUDGET_EXCEEDED")
        next_kg = load_kg + candidate.load_delta_kg
        next_cm3 = load_cm3 + candidate.load_delta_cm3
        if next_kg < -1.0e-6 or next_kg > self.vehicle.capacity_kg + 1.0e-6:
            reasons.append("WEIGHT_CAPACITY_VIOLATION")
        if next_cm3 < -1.0e-6 or next_cm3 > self.vehicle.capacity_cm3 + 1.0e-6:
            reasons.append("VOLUME_CAPACITY_VIOLATION")
        if candidate.required_vehicle_type not in ("any", self.vehicle.vehicle_type):
            reasons.append("VEHICLE_TYPE_INCOMPATIBLE")
        if candidate.hazmat:
            reasons.append("HAZMAT_COMPLIANCE_UNVERIFIED")
        if candidate.temperature_controlled:
            reasons.append("TEMPERATURE_CONTROL_UNVERIFIED")
        if candidate.stop_type in ("pickup", "delivery"):
            if candidate.cargo_type not in self.vehicle.compatible_cargo_types:
                reasons.append("CARGO_TYPE_INCOMPATIBLE")
            assert candidate.shipment_id is not None
            if candidate.stop_type == "pickup" and candidate.shipment_id in onboard:
                reasons.append("SHIPMENT_ALREADY_ONBOARD")
            if candidate.stop_type == "delivery" and not (
                candidate.shipment_id in onboard or candidate.shipment_id in picked_up
            ):
                reasons.append("PICKUP_NOT_COMPLETED")
        return reasons, arrival, service_start, service_end

    def canonical_features(
        self,
        active: Sequence[bool],
        origin: str,
        current_zone: str,
        current_time: datetime,
        load_kg: float,
        load_cm3: float,
        completed: int,
    ) -> Tuple[List[List[float]], List[float]]:
        total = len(self.candidates)
        remaining = sum(active)
        progress = completed / max(1, total)
        nodes: List[List[float]] = []
        for candidate in self.candidates:
            if candidate.stop_id == origin:
                travel_s, distance_km = 0.0, 0.0
            else:
                travel_s, distance_km, _ = self.edge(origin, candidate.stop_id)
            # Public route supervision observes windows relative to departure,
            # not the driver's actual per-stop clock. Runtime timing remains a
            # hard verifier input; it is not injected into an unseen feature.
            open_delta = (candidate.window_start - self.vehicle.route_started).total_seconds() / 3600.0
            close_delta = (candidate.window_end - self.vehicle.route_started).total_seconds() / 3600.0
            nodes.append(
                [
                    float(candidate.stop_type == "depot"),
                    float(candidate.stop_type == "pickup"),
                    _scale_distance(distance_km),
                    0.0,
                    _scale_hours(candidate.service_time_s / 3600.0),
                    _scale_hours(open_delta),
                    _scale_hours(close_delta),
                    0.0,
                    _clip(abs(candidate.load_delta_cm3) / self.vehicle.capacity_cm3, 0.0, 2.0),
                    _scale_distance(candidate.distance_to_depot_km),
                    progress,
                    float(candidate.zone_id == current_zone),
                    0.0,
                    0.0,
                    0.0,
                    candidate.missing_fraction,
                ]
            )
        hour = current_time.hour + current_time.minute / 60.0 + current_time.second / 3600.0
        weekday = float(current_time.weekday())
        elapsed_h = (current_time - self.vehicle.route_started).total_seconds() / 3600.0
        budget_h = self.vehicle.route_time_budget_s / 3600.0 - elapsed_h
        context = [
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            math.sin(2.0 * math.pi * weekday / 7.0),
            math.cos(2.0 * math.pi * weekday / 7.0),
            0.0,
            _clip((self.vehicle.capacity_cm3 - load_cm3) / self.vehicle.capacity_cm3, 0.0, 2.0),
            0.0,
            0.0,
            0.0,
            # The public route branch is supervised on active volumetric load.
            # Weight remains a hard feasibility constraint, but is not injected
            # into this learned slot without paired route-plus-weight labels.
            _clip(load_cm3 / self.vehicle.capacity_cm3, 0.0, 2.0),
            remaining / 64.0,
            progress,
            0.0,
            0.0,
            self.vehicle.domain_shift_score,
            self.vehicle.missing_fraction,
        ]
        return nodes, context


def telemetry_features(
    payload: Mapping[str, Any], telemetry_steps: int
) -> Tuple[List[List[float]], List[bool]]:
    if payload.get("schema_version") != TELEMETRY_SCHEMA:
        raise InputIssue("INVALID_SCHEMA", "schema_version", f"schema_version must be {TELEMETRY_SCHEMA}")
    samples = _array(
        _required(payload, "samples", ""), "samples", minimum=1, maximum=telemetry_steps
    )
    result: List[List[float]] = []
    previous_time: datetime | None = None
    for index, raw in enumerate(samples):
        field = f"samples[{index}]"
        sample = _object(raw, field)
        timestamp = _timestamp(_required(sample, "timestamp_utc", field), f"{field}.timestamp_utc")
        expected_delta_s = None
        if previous_time is not None and timestamp <= previous_time:
            raise InputIssue(
                "INVALID_RUNTIME_VALUE", f"{field}.timestamp_utc", "telemetry timestamps must increase"
            )
        if previous_time is not None:
            expected_delta_s = (timestamp - previous_time).total_seconds()
        previous_time = timestamp
        speed = _bounded(_required(sample, "speed_kmh", field), f"{field}.speed_kmh", 0.0, 180.0)
        engine = _bounded(_required(sample, "engine_status", field), f"{field}.engine_status", 0.0, 2.0)
        grade = _bounded(_required(sample, "road_grade", field), f"{field}.road_grade", -0.4, 0.4)
        load = _bounded(
            _required(sample, "engine_load_pct", field), f"{field}.engine_load_pct", 0.0, 100.0
        )
        maf = _nonnegative(
            _required(sample, "mass_air_flow_g_s", field), f"{field}.mass_air_flow_g_s"
        )
        acceleration = _bounded(
            _required(sample, "longitudinal_acceleration_m_s2", field),
            f"{field}.longitudinal_acceleration_m_s2",
            -15.0,
            15.0,
        )
        delta = _nonnegative(_required(sample, "delta_t_s", field), f"{field}.delta_t_s")
        if index == 0 and delta != 0.0:
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                f"{field}.delta_t_s",
                "the first canonical telemetry sample must use delta_t_s=0",
            )
        if expected_delta_s is not None and not math.isclose(
            delta, expected_delta_s, rel_tol=0.01, abs_tol=0.5
        ):
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                f"{field}.delta_t_s",
                "delta_t_s must match the timestamp interval",
            )
        observed = _bounded(
            _required(sample, "observed_fraction", field),
            f"{field}.observed_fraction",
            0.0,
            1.0,
        )
        if not math.isclose(observed, 1.0, abs_tol=1.0e-9):
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                f"{field}.observed_fraction",
                "all required endpoint fields are present, so observed_fraction must equal 1",
            )
        result.append(
            [
                _clip(speed / 100.0, 0.0, 2.0),
                engine / 2.0,
                _clip(grade * 20.0, -8.0, 8.0),
                load / 100.0,
                _clip(math.log1p(maf) / 5.0, 0.0, 4.0),
                _clip(acceleration / 10.0, -1.5, 1.5),
                _clip(delta / 300.0, 0.0, 1.0),
                observed,
            ]
        )
    mask = [True] * len(result) + [False] * (telemetry_steps - len(result))
    result.extend([[0.0] * len(TELEMETRY_FEATURES) for _ in range(telemetry_steps - len(result))])

    return result, mask


def track_features(payload: Mapping[str, Any]) -> List[float]:
    if payload.get("schema_version") != TRACK_SCHEMA:
        raise InputIssue("INVALID_SCHEMA", "schema_version", f"schema_version must be {TRACK_SCHEMA}")
    raw_track = _object(_required(payload, "track", ""), "track")
    _timestamp(_required(raw_track, "start_time_utc", "track"), "track.start_time_utc")
    specifications = (
        ("distance_m", "nonnegative", lambda value: math.log1p(value)),
        ("track_gap_m", "nonnegative", lambda value: math.log1p(value)),
        ("avg_speed_m_s", "nonnegative", lambda value: _clip(value / 40.0, 0.0, 4.0)),
        ("max_speed_m_s", "nonnegative", lambda value: _clip(value / 50.0, 0.0, 2.0)),
        ("avg_hdop", "nonnegative", lambda value: _clip(value / 5.0, 0.0, 8.0)),
        ("gvwr_kg", "positive", lambda value: math.log1p(value) / 12.0),
        ("gcwr_kg", "positive", lambda value: math.log1p(value) / 12.0),
        ("axle_class", "nonnegative", lambda value: _clip(value / 100.0, 0.0, 4.0)),
    )
    values: List[float] = []
    missing: List[float] = []
    for name, constraint, transform in specifications:
        raw_value = raw_track.get(name)
        if raw_value is None:
            values.append(0.0)
            missing.append(1.0)
            continue
        field = f"track.{name}"
        value = _positive(raw_value, field) if constraint == "positive" else _nonnegative(raw_value, field)
        values.append(transform(value))
        missing.append(0.0)
    return values + missing


def health_features(payload: Mapping[str, Any]) -> List[float]:
    if payload.get("schema_version") != HEALTH_SCHEMA:
        raise InputIssue("INVALID_SCHEMA", "schema_version", f"schema_version must be {HEALTH_SCHEMA}")
    if payload.get("feature_schema") != "scania-aps-170-plus-missing-mask.v1":
        raise InputIssue(
            "INVALID_SCHEMA",
            "feature_schema",
            "feature_schema must be scania-aps-170-plus-missing-mask.v1",
        )
    values = _array(
        _required(payload, "normalized_sensor_values", ""),
        "normalized_sensor_values",
        minimum=APS_VALUE_COUNT,
        maximum=APS_VALUE_COUNT,
    )
    masks = _array(
        _required(payload, "missing_mask", ""),
        "missing_mask",
        minimum=APS_VALUE_COUNT,
        maximum=APS_VALUE_COUNT,
    )
    normalized: List[float] = []
    missing: List[float] = []
    for index, (raw_value, raw_mask) in enumerate(zip(values, masks)):
        value = _bounded(raw_value, f"normalized_sensor_values[{index}]", -20.0, 20.0)
        mask = _boolean(raw_mask, f"missing_mask[{index}]")
        if mask and value != 0.0:
            raise InputIssue(
                "INVALID_RUNTIME_VALUE",
                f"normalized_sensor_values[{index}]",
                "a masked APS value must use the documented zero sentinel",
            )
        normalized.append(value)
        missing.append(float(mask))
    return normalized + missing


def deadhead_features(payload: Mapping[str, Any], normalization: Mapping[str, Any]) -> List[float]:
    if payload.get("schema_version") != DEADHEAD_SCHEMA:
        raise InputIssue("INVALID_SCHEMA", "schema_version", f"schema_version must be {DEADHEAD_SCHEMA}")
    if payload.get("feature_schema") != "vius-2021-12-values-plus-mask.v1":
        raise InputIssue(
            "INVALID_SCHEMA", "feature_schema",
            "feature_schema must be vius-2021-12-values-plus-mask.v1",
        )
    columns = normalization.get("columns")
    medians = normalization.get("median")
    scales = normalization.get("iqr")
    if not (
        isinstance(columns, list) and isinstance(medians, list) and isinstance(scales, list)
        and len(columns) == len(medians) == len(scales) == 12
    ):
        raise RuntimeError("frozen manifest VIUS normalization is invalid")
    values = _object(_required(payload, "raw_values", ""), "raw_values")
    scaled: List[float] = []
    missing: List[float] = []
    for index, column in enumerate(columns):
        raw = values.get(column)
        if raw is None:
            scaled.append(0.0)
            missing.append(1.0)
            continue
        value = _number(raw, f"raw_values.{column}")
        scale = float(scales[index])
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("frozen manifest VIUS scale is invalid")
        scaled.append(_clip((value - float(medians[index])) / scale, -20.0, 20.0))
        missing.append(0.0)
    return scaled + missing


def price_features(payload: Mapping[str, Any], expected_schema: str) -> List[float]:
    if payload.get("schema_version") != PRICE_SCHEMA:
        raise InputIssue("INVALID_SCHEMA", "schema_version", f"schema_version must be {PRICE_SCHEMA}")
    if payload.get("feature_schema") != expected_schema:
        raise InputIssue(
            "INVALID_SCHEMA", "feature_schema", f"feature_schema must be {expected_schema}"
        )
    values = _array(
        _required(payload, "normalized_source_features", ""),
        "normalized_source_features",
        minimum=len(PRICE_FEATURES),
        maximum=len(PRICE_FEATURES),
    )
    return [
        _bounded(value, f"normalized_source_features[{index}]", -20.0, 20.0)
        for index, value in enumerate(values)
    ]


class FrozenPolicy:
    """Checksum-bound TorchScript policy with no mutable runtime state."""

    def __init__(self, artifact_dir: Path) -> None:
        self.contract = validate_artifacts(artifact_dir)
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to load the frozen policy") from error
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.model = torch.jit.load(str(self.contract.artifact_path), map_location="cpu").eval()
        for method in ("telemetry", "dtcargo", "vius", "health", "price"):
            if not hasattr(self.model, method):
                raise RuntimeError(f"frozen policy is missing exported method: {method}")
        self.lock = threading.Lock()
        preprocessing = self.contract.manifest.get("preprocessing", {})
        price_config = preprocessing.get("price", {}) if isinstance(preprocessing, Mapping) else {}
        self.vius_normalization = (
            preprocessing.get("vius", {}) if isinstance(preprocessing, Mapping) else {}
        )
        self.price_feature_schema = (
            price_config.get("feature_schema")
            if isinstance(price_config, Mapping) and isinstance(price_config.get("feature_schema"), str)
            else "model-native-price-12.v1"
        )
        output_schema = self.contract.manifest.get("output_schema", {})
        self.price_output_space = (
            output_schema.get("price")
            if isinstance(output_schema, Mapping) and isinstance(output_schema.get("price"), str)
            else "model_native_source_domain"
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "mode": "frozen_inference_only",
            "weights_sha256": self.contract.artifact_sha256,
            "weights_static": True,
            "online_learning": False,
            "auto_tuning": False,
            "feedback_loop": False,
            "automatic_dispatch": False,
            "dispatcher_approval_required": True,
            "contains_synthetic": False,
            "cross_dataset_rows_joined": False,
            "schemas": {
                "route": ROUTE_SCHEMA,
                "telemetry": TELEMETRY_SCHEMA,
                "truck_track": TRACK_SCHEMA,
                "health": HEALTH_SCHEMA,
                "deadhead": DEADHEAD_SCHEMA,
                "price": PRICE_SCHEMA,
                "route_node_features": list(ROUTE_NODE_FEATURES),
                "route_context_features": list(ROUTE_CONTEXT_FEATURES),
                "telemetry_features": list(TELEMETRY_FEATURES),
                "dtcargo_features": list(DTCARGO_FEATURES),
                "vius_features": list(VIUS_FEATURES),
                "health_feature_count": APS_INPUT_COUNT,
                "price_feature_schema": self.price_feature_schema,
            },
        }

    def infer_route(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = _request_id(payload)
        try:
            request = RouteRequest(payload)
        except InputIssue as issue:
            return abstain(request_id, issue)
        torch = self.torch
        count = len(request.candidates)
        active = [True] * count
        state = request.vehicle
        origin = state.origin_id
        zone = state.zone_id
        current_time = state.current_time
        load_kg = state.load_kg
        load_cm3 = state.load_cm3
        onboard = set(state.onboard_shipments)
        picked_up: set[str] = set()
        route: List[Dict[str, Any]] = []
        final_rejections: Dict[str, List[str]] = {}
        termination = "ALL_CANDIDATES_VISITED"

        for _ in range(count):
            feasible = [False] * count
            schedule: Dict[int, Tuple[datetime, datetime, datetime]] = {}
            rejections: Dict[str, List[str]] = {}
            for index, candidate in enumerate(request.candidates):
                if not active[index]:
                    continue
                reasons, arrival, service_start, service_end = request.feasibility(
                    candidate,
                    origin,
                    current_time,
                    load_kg,
                    load_cm3,
                    onboard,
                    picked_up,
                )
                if reasons:
                    rejections[candidate.stop_id] = reasons
                else:
                    feasible[index] = True
                    schedule[index] = (arrival, service_start, service_end)
            final_rejections = rejections
            if not any(feasible):
                termination = "NO_FEASIBLE_CANDIDATE"
                break
            nodes, context = request.canonical_features(
                active, origin, zone, current_time, load_kg, load_cm3, len(route)
            )
            nodes_tensor = torch.tensor([nodes], dtype=torch.float32)
            context_tensor = torch.tensor([context], dtype=torch.float32)
            active_tensor = torch.tensor([active], dtype=torch.bool)
            feasible_tensor = torch.tensor([feasible], dtype=torch.bool)
            with self.lock, torch.inference_mode():
                logits, eta, stop_logits = self.model(
                    nodes_tensor, context_tensor, active_tensor, feasible_tensor
                )
            logits_row = [float(value) for value in logits[0].tolist()]
            stop_row = [float(value) for value in stop_logits[0].tolist()]
            if len(stop_row) != 2 or not all(math.isfinite(value) for value in stop_row):
                return abstain(
                    request.request_id,
                    InputIssue("NONFINITE_MODEL_OUTPUT", "route.stop", "model STOP output is invalid"),
                )
            # The training label means "stop after the selected next action",
            # not "stop before serving it".  More importantly, an auxiliary
            # STOP prediction is never allowed to strand a submitted job: the
            # hard verifier keeps rolling out while any candidate remains.
            stop_after_selected = stop_row[1] > stop_row[0]
            ranked = [index for index in range(count) if feasible[index]]
            if any(not math.isfinite(logits_row[index]) for index in ranked):
                return abstain(
                    request.request_id,
                    InputIssue("NONFINITE_MODEL_OUTPUT", "route.logits", "model route output is invalid"),
                )
            selected = max(ranked, key=lambda index: logits_row[index])
            if not feasible[selected]:
                return abstain(
                    request.request_id,
                    InputIssue("SAFETY_VERIFIER_REJECTED", "route", "model selected an infeasible stop"),
                )
            candidate = request.candidates[selected]
            arrival, service_start, service_end = schedule[selected]
            eta_values = [float(value) for value in eta[0, selected].tolist()]
            if len(eta_values) != 2 or not all(math.isfinite(value) for value in eta_values):
                return abstain(
                    request.request_id,
                    InputIssue("NONFINITE_MODEL_OUTPUT", "route.eta", "model ETA output is invalid"),
                )
            load_kg += candidate.load_delta_kg
            load_cm3 += candidate.load_delta_cm3
            if candidate.stop_type == "pickup" and candidate.shipment_id is not None:
                onboard.add(candidate.shipment_id)
                picked_up.add(candidate.shipment_id)
            elif candidate.stop_type == "delivery" and candidate.shipment_id is not None:
                onboard.discard(candidate.shipment_id)
                picked_up.discard(candidate.shipment_id)
            travel_s, distance_km, _ = request.edge(origin, candidate.stop_id)
            route.append(
                {
                    "sequence": len(route) + 1,
                    "stop_id": candidate.stop_id,
                    "stop_type": candidate.stop_type,
                    "shipment_id": candidate.shipment_id,
                    "from_origin_id": origin,
                    "travel_time_s": travel_s,
                    "distance_km": distance_km,
                    "arrival_time_utc": _iso(arrival),
                    "service_start_time_utc": _iso(service_start),
                    "service_end_time_utc": _iso(service_end),
                    "load_after_kg": load_kg,
                    "load_after_cm3": load_cm3,
                    "model_logit": logits_row[selected],
                    "model_eta_q50_native": eta_values[0],
                    "model_eta_q90_native": eta_values[1],
                }
            )
            active[selected] = False
            origin = candidate.stop_id
            zone = candidate.zone_id
            current_time = service_end
            if stop_after_selected and not any(active):
                termination = "MODEL_STOP_AFTER_FINAL_CANDIDATE"
                break

        if picked_up:
            response = abstain(
                request.request_id,
                InputIssue(
                    "INCOMPLETE_PICKUP_DELIVERY",
                    "candidates",
                    "the learned rollout stopped before completing a selected pickup-delivery pair",
                ),
            )
            response["candidate_rejections"] = final_rejections
            return response
        if not route:
            reason = "no candidate satisfies all hard constraints"
            response = abstain(
                request.request_id,
                InputIssue("NO_SAFE_ROUTE", "candidates", reason),
            )
            response["candidate_rejections"] = final_rejections
            return response
        return {
            "status": "RECOMMENDATION",
            "request_id": request.request_id,
            "route": route,
            "termination": termination,
            "unserved_candidate_rejections": final_rejections,
            "hard_constraints_verified": True,
            "dispatcher_approval_required": True,
            "automatic_dispatch": False,
            "model_updated": False,
        }

    def infer_telemetry(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = _request_id(payload)
        try:
            sequence, mask = telemetry_features(
                payload, int(self.contract.model_config["telemetry_steps"])
            )
        except InputIssue as issue:
            return abstain(request_id, issue)
        torch = self.torch
        with self.lock, torch.inference_mode():
            temporal = self.model.telemetry(
                torch.tensor([sequence], dtype=torch.float32),
                torch.tensor([mask], dtype=torch.bool),
            )[0]
        temporal_values = [float(value) for value in temporal.tolist()]
        if not all(math.isfinite(value) for value in temporal_values):
            return abstain(
                request_id,
                InputIssue("NONFINITE_MODEL_OUTPUT", "telemetry", "model telemetry output is invalid"),
            )
        return {
            "status": "INFERENCE",
            "request_id": request_id,
            "source_domain_outputs": {
                "fuel_log_native": temporal_values[0],
                "predicted_trip_fuel_l": max(0.0, math.expm1(min(temporal_values[0], 20.0))),
                "duration_log_native": temporal_values[1],
                "predicted_trip_duration_s": max(0.0, math.expm1(min(temporal_values[1], 20.0))),
                "load_state_prediction_available": False,
                "future_idle_heavy_probability": _sigmoid(temporal_values[3]),
            },
            "claim_boundary": "Singapore commercial-vehicle source-domain prefix forecast; cargo load remains an actual required sensor/manifest field",
            "dispatcher_approval_required": True,
            "model_updated": False,
        }

    def infer_track(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = _request_id(payload)
        try:
            features = track_features(payload)
        except InputIssue as issue:
            return abstain(request_id, issue)
        torch = self.torch
        with self.lock, torch.inference_mode():
            output = self.model.dtcargo(torch.tensor([features], dtype=torch.float32))[0]
        values = [float(value) for value in output.tolist()]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            return abstain(
                request_id,
                InputIssue("NONFINITE_MODEL_OUTPUT", "track", "model truck-track output is invalid"),
            )
        return {
            "status": "INFERENCE",
            "request_id": request_id,
            "source_domain_outputs": {
                "track_duration_log_native": values[0],
                "predicted_track_duration_s": max(0.0, math.expm1(min(values[0], 20.0))),
                "signal_loss_ratio": _clip(values[1], 0.0, 1.0),
                "home_base_probability": _sigmoid(values[2]),
                "long_haul_probability": _sigmoid(values[3]),
            },
            "claim_boundary": "DT-CARGO class-N3 source-domain inference",
            "dispatcher_approval_required": True,
            "model_updated": False,
        }

    def infer_health(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = _request_id(payload)
        try:
            features = health_features(payload)
        except InputIssue as issue:
            return abstain(request_id, issue)
        torch = self.torch
        with self.lock, torch.inference_mode():
            raw = float(self.model.health(torch.tensor([features], dtype=torch.float32))[0])
        if not math.isfinite(raw):
            return abstain(
                request_id,
                InputIssue("NONFINITE_MODEL_OUTPUT", "health", "model health output is invalid"),
            )
        return {
            "status": "INFERENCE",
            "request_id": request_id,
            "aps_failure_probability": _sigmoid(raw),
            "claim_boundary": "Scania APS source-domain health classification",
            "dispatcher_approval_required": True,
            "model_updated": False,
        }

    def infer_deadhead(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = _request_id(payload)
        try:
            features = deadhead_features(payload, self.vius_normalization)
        except InputIssue as issue:
            return abstain(request_id, issue)
        torch = self.torch
        with self.lock, torch.inference_mode():
            output = self.model.vius(torch.tensor([features], dtype=torch.float32))[0]
        values = [float(value) for value in output.tolist()]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            return abstain(
                request_id,
                InputIssue("NONFINITE_MODEL_OUTPUT", "deadhead", "model deadhead output is invalid"),
            )
        return {
            "status": "INFERENCE",
            "request_id": request_id,
            "annual_source_domain_fractions": {
                "deadhead": _clip(values[0], 0.0, 1.0),
                "repositioning": _clip(values[1], 0.0, 1.0),
                "loaded_miles": _clip(values[2], 0.0, 1.0),
            },
            "claim_boundary": "VIUS annual survey prior; not a per-trip empty-state label",
            "dispatcher_approval_required": True,
            "model_updated": False,
        }

    def infer_price(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = _request_id(payload)
        try:
            features = price_features(payload, self.price_feature_schema)
        except InputIssue as issue:
            return abstain(request_id, issue)
        torch = self.torch
        with self.lock, torch.inference_mode():
            raw = self.model.price(torch.tensor([features], dtype=torch.float32))[0]
        values = [float(value) for value in raw.tolist()]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            return abstain(
                request_id,
                InputIssue("NONFINITE_MODEL_OUTPUT", "price", "model price output is invalid"),
            )
        quantiles = [
            values[0],
            values[0] + _softplus(values[1]),
            values[0] + _softplus(values[1]) + _softplus(values[2]),
        ]
        return {
            "status": "INFERENCE",
            "request_id": request_id,
            "model_native_quantiles": {"p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]},
            "nyc_tlc_fare_amount_proxy": {
                "p10": max(0.0, math.expm1(min(quantiles[0], 20.0))),
                "p50": max(0.0, math.expm1(min(quantiles[1], 20.0))),
                "p90": max(0.0, math.expm1(min(quantiles[2], 20.0))),
            },
            "output_space": self.price_output_space,
            "operational_quote": False,
            "claim_boundary": "real source-domain proxy; not a calibrated Indonesian truckload price",
            "dispatcher_approval_required": True,
            "model_updated": False,
        }
