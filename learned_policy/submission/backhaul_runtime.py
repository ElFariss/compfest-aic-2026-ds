"""Frozen inference runtime for the Haulio backhaul graph policy.

There is deliberately no optimizer, backward pass, model update, feedback loop,
or automatic acceptance in this module.  IoT and order state may change between
requests; model weights and preprocessing constants remain immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor


VEHICLE_TYPES = ("box", "reefer", "flatbed", "wingbox", "tanker")
CARGO_CLASSES = ("general", "chilled", "industrial", "hazmat", "liquid")
TRUCK_DIM = 32
ORDER_DIM = 24
PAIR_DIM = 16
MAX_TRUCKS = 16
MAX_ORDERS = 32


class InputError(ValueError):
    pass


def _number(value: Any, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{field} must be finite")
    return result


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > 128:
        raise InputError(f"{field} must contain at most 128 characters")
    return result


def _boolean(value: Any, field: str, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if type(value) is not bool:
        raise InputError(f"{field} must be boolean")
    return value


def _nonnegative(value: Any, field: str, default: float | None = None) -> float:
    result = _number(value, field, default)
    if result < 0.0:
        raise InputError(f"{field} must be non-negative")
    return result


def _unit_interval(value: Any, field: str, default: float | None = None) -> float:
    result = _number(value, field, default)
    if not 0.0 <= result <= 1.0:
        raise InputError(f"{field} must be between 0 and 1")
    return result


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _coord(latitude: float, longitude: float, field: str) -> Tuple[float, float]:
    if not (-11.5 <= latitude <= 6.5 and 94.0 <= longitude <= 142.0):
        raise InputError(f"{field} is outside the supported Indonesia operating bounds")
    return (latitude + 11.0) / 17.0, (longitude - 95.0) / 46.0


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, (a_lat, a_lon, b_lat, b_lon))
    value = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))


def _one_hot(index: int, size: int) -> List[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_context_lookup(payload: Mapping[str, Any]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    result: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    items = payload.get("pair_context", [])
    if not isinstance(items, list):
        raise InputError("pair_context must be an array")
    for item in items:
        if not isinstance(item, Mapping):
            raise InputError("pair_context entries must be objects")
        key = (
            _identifier(item.get("truck_id"), "pair_context.truck_id"),
            _identifier(item.get("order_id"), "pair_context.order_id"),
        )
        if key in result:
            raise InputError(f"duplicate pair_context for {key[0]} and {key[1]}")
        result[key] = item
    return result


class FrozenPolicy:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir.resolve()
        manifest_path = self.artifact_dir / "manifest.json"
        model_path = self.artifact_dir / "backhaul_policy_frozen.pt"
        if not manifest_path.is_file() or not model_path.is_file():
            raise RuntimeError("Frozen model and manifest are required")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = self.manifest["artifact"]["sha256"]
        actual_hash = _sha256(model_path)
        if actual_hash != expected_hash:
            raise RuntimeError("Frozen model checksum does not match manifest")
        if not self.manifest["runtime"].get("weights_static"):
            raise RuntimeError("Manifest does not declare immutable weights")
        torch.set_grad_enabled(False)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.model_hash = actual_hash

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "mode": "frozen_inference_only",
            "model_name": self.manifest["model_name"],
            "model_version": self.manifest["model_version"],
            "weights_sha256": self.model_hash,
            "weights_static": True,
            "online_learning": False,
        }

    def _build_features(
        self, payload: Mapping[str, Any]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Mapping[str, Any]], List[Mapping[str, Any]], List[List[str]], float]:
        if payload.get("schema_version") != "haulio.backhaul-policy-input.v1":
            raise InputError("schema_version must be haulio.backhaul-policy-input.v1")
        trucks = payload.get("trucks")
        orders = payload.get("orders")
        if not isinstance(trucks, list) or not isinstance(orders, list):
            raise InputError("trucks and orders must be arrays")
        if not 1 <= len(trucks) <= MAX_TRUCKS:
            raise InputError(f"trucks must contain 1..{MAX_TRUCKS} entries")
        if not 1 <= len(orders) <= MAX_ORDERS:
            raise InputError(f"orders must contain 1..{MAX_ORDERS} entries")
        if not all(isinstance(item, Mapping) for item in trucks):
            raise InputError("truck entries must be objects")
        if not all(isinstance(item, Mapping) for item in orders):
            raise InputError("order entries must be objects")
        truck_ids = [_identifier(item.get("truck_id"), "truck_id") for item in trucks]
        order_ids = [_identifier(item.get("order_id"), "order_id") for item in orders]
        if len(set(truck_ids)) != len(trucks):
            raise InputError("truck_id values must be unique")
        if len(set(order_ids)) != len(orders):
            raise InputError("order_id values must be unique")

        road_lookup = _pair_context_lookup(payload)
        unknown_pairs = sorted(set(road_lookup) - {(truck_id, order_id) for truck_id in truck_ids for order_id in order_ids})
        if unknown_pairs:
            raise InputError(f"pair_context references unknown truck/order pair {unknown_pairs[0]}")
        truck_features = torch.zeros(1, MAX_TRUCKS, TRUCK_DIM, dtype=torch.float32)
        order_features = torch.zeros(1, MAX_ORDERS, ORDER_DIM, dtype=torch.float32)
        pair_features = torch.zeros(1, MAX_TRUCKS, MAX_ORDERS, PAIR_DIM, dtype=torch.float32)
        truck_mask = torch.zeros(1, MAX_TRUCKS, dtype=torch.bool)
        order_mask = torch.zeros(1, MAX_ORDERS, dtype=torch.bool)
        feasible = torch.zeros(1, MAX_TRUCKS, MAX_ORDERS, dtype=torch.bool)
        rejection_reasons: List[List[str]] = [[] for _ in range(len(trucks) * len(orders))]
        raw_trucks: List[Dict[str, Any]] = []
        raw_orders: List[Dict[str, Any]] = []
        ood_values: List[float] = []

        for index, truck in enumerate(trucks):
            truck_id = truck_ids[index]
            latitude = _number(truck.get("lat"), f"{truck_id}.lat")
            longitude = _number(truck.get("lon"), f"{truck_id}.lon")
            lat_norm, lon_norm = _coord(latitude, longitude, f"{truck_id}.position")
            home_lat = _number(truck.get("home_lat"), f"{truck_id}.home_lat", latitude)
            home_lon = _number(truck.get("home_lon"), f"{truck_id}.home_lon", longitude)
            home_lat_norm, home_lon_norm = _coord(home_lat, home_lon, f"{truck_id}.home")
            vehicle_type = str(truck.get("vehicle_type", ""))
            vehicle_index = VEHICLE_TYPES.index(vehicle_type) if vehicle_type in VEHICLE_TYPES else -1
            capacity = _number(truck.get("capacity_kg"), f"{truck_id}.capacity_kg")
            if capacity <= 0:
                raise InputError(f"{truck_id}.capacity_kg must be positive")
            fuel_pct = _number(truck.get("fuel_pct"), f"{truck_id}.fuel_pct")
            if not 0.0 <= fuel_pct <= 100.0:
                raise InputError(f"{truck_id}.fuel_pct must be between 0 and 100")
            fuel = fuel_pct / 100.0
            speed_kph = _nonnegative(truck.get("speed_kph"), f"{truck_id}.speed_kph", 0.0)
            if speed_kph > 130.0:
                raise InputError(f"{truck_id}.speed_kph must not exceed 130")
            speed = speed_kph / 100.0
            heading_degrees = _number(truck.get("heading"), f"{truck_id}.heading", 0.0)
            if not 0.0 <= heading_degrees <= 360.0:
                raise InputError(f"{truck_id}.heading must be between 0 and 360")
            heading = math.radians(heading_degrees % 360.0)
            gps_accuracy_m = _nonnegative(truck.get("gps_accuracy_m"), f"{truck_id}.gps_accuracy_m")
            if gps_accuracy_m > 1000.0:
                raise InputError(f"{truck_id}.gps_accuracy_m must not exceed 1000")
            gps_accuracy = _bounded(gps_accuracy_m / 100.0, 0.0, 4.0)
            available_after_min = _nonnegative(
                truck.get("available_after_min"), f"{truck_id}.available_after_min", 0.0
            )
            available_after = _bounded(available_after_min / 720.0, 0.0, 4.0)
            telemetry_age = _nonnegative(truck.get("telemetry_age_s"), f"{truck_id}.telemetry_age_s")
            replayed = 1.0 if _boolean(truck.get("replayed"), f"{truck_id}.replayed", False) else 0.0
            cargo_status = truck.get("cargo_status", "empty")
            if cargo_status not in {"empty", "loaded"}:
                raise InputError(f"{truck_id}.cargo_status must be empty or loaded")
            cargo_empty = 1.0 if cargo_status == "empty" else 0.0
            cargo_weight_value = truck.get("cargo_weight_kg")
            cargo_present = 0.0 if cargo_weight_value is None else 1.0
            current_weight = None if cargo_weight_value is None else _nonnegative(
                cargo_weight_value, f"{truck_id}.cargo_weight_kg"
            )
            if current_weight is not None and current_weight > capacity:
                raise InputError(f"{truck_id}.cargo_weight_kg must not exceed capacity_kg")
            current_ratio = 0.0 if current_weight is None else current_weight / capacity
            unloads_before_pickup = _boolean(
                truck.get("cargo_released_before_pickup"),
                f"{truck_id}.cargo_released_before_pickup",
                cargo_status == "empty",
            )
            if unloads_before_pickup:
                available_capacity_at_pickup = capacity
            elif cargo_status == "loaded" and current_weight is None:
                available_capacity_at_pickup = 0.0
            else:
                available_capacity_at_pickup = max(0.0, capacity - (current_weight or 0.0))
            can_value = truck.get("can")
            imu_value = truck.get("imu")
            health_value = truck.get("health")
            if can_value is not None and not isinstance(can_value, Mapping):
                raise InputError(f"{truck_id}.can must be an object")
            if imu_value is not None and not isinstance(imu_value, Mapping):
                raise InputError(f"{truck_id}.imu must be an object")
            if health_value is not None and not isinstance(health_value, Mapping):
                raise InputError(f"{truck_id}.health must be an object")
            can = can_value
            imu = imu_value
            health = health_value
            can_present = 1.0 if can else 0.0
            imu_present = 1.0 if imu else 0.0
            health_present = 1.0 if health else 0.0
            rpm = 0.0 if not can else _bounded(_number(can.get("engine_rpm"), "can.engine_rpm", 0.0) / 3000.0, 0.0, 2.0)
            coolant = 0.0 if not can else _bounded(_number(can.get("coolant_temp_c"), "can.coolant_temp_c", 0.0) / 130.0, 0.0, 2.0)
            accel = 0.0 if not imu else _bounded(abs(_number(imu.get("accel_x_g"), "imu.accel_x_g", 0.0)) / 2.0, 0.0, 2.0)
            gyro = 0.0 if not imu else _bounded(abs(_number(imu.get("gyro_z_dps"), "imu.gyro_z_dps", 0.0)) / 30.0, 0.0, 2.0)
            power = 0.0 if not health else _bounded(_number(health.get("power_v"), "health.power_v", 0.0) / 15.0, 0.0, 2.0)
            signal = 0.0 if not health else _bounded((_number(health.get("signal_dbm"), "health.signal_dbm", -120.0) + 120.0) / 80.0, 0.0, 1.5)
            uptime = 0.0 if not health else _bounded(math.log1p(max(0.0, _number(health.get("uptime_s"), "health.uptime_s", 0.0))) / math.log1p(604800.0), 0.0, 2.0)
            quality = (1.0 - _bounded(telemetry_age / 600.0, 0.0, 1.0)) * (1.0 - _bounded(gps_accuracy, 0.0, 1.0))
            values = [
                lat_norm, lon_norm, home_lat_norm, home_lon_norm, capacity / 20000.0,
                current_ratio * cargo_present, fuel, speed, math.sin(heading), math.cos(heading),
                gps_accuracy, available_after, _bounded(telemetry_age / 600.0, 0.0, 1.0), replayed,
                cargo_empty, cargo_present, can_present, imu_present, health_present, rpm,
                coolant, accel, gyro, power, signal, uptime,
                *_one_hot(vehicle_index, 5), quality,
            ]
            truck_features[0, index] = torch.tensor(values)
            truck_mask[0, index] = True
            ood_values.extend(max(0.0, abs(value) - 1.5) for value in values)
            raw_trucks.append({
                **dict(truck), "lat": latitude, "lon": longitude, "home_lat": home_lat,
                "home_lon": home_lon, "vehicle_index": vehicle_index, "capacity": capacity,
                "fuel": fuel, "telemetry_age": telemetry_age,
                "available_after_min": available_after_min, "quality": quality,
                "available_capacity_at_pickup": available_capacity_at_pickup,
                "health_risk": 0.34 * max(0.0, coolant - 0.82) + 0.26 * (1.0 - signal)
                + 0.20 * (1.0 - fuel) + 0.20 * (1.0 - quality),
            })

        for index, order in enumerate(orders):
            order_id = order_ids[index]
            pickup_lat = _number(order.get("pickup_lat"), f"{order_id}.pickup_lat")
            pickup_lon = _number(order.get("pickup_lon"), f"{order_id}.pickup_lon")
            dropoff_lat = _number(order.get("dropoff_lat"), f"{order_id}.dropoff_lat")
            dropoff_lon = _number(order.get("dropoff_lon"), f"{order_id}.dropoff_lon")
            pickup_norm = _coord(pickup_lat, pickup_lon, f"{order_id}.pickup")
            dropoff_norm = _coord(dropoff_lat, dropoff_lon, f"{order_id}.dropoff")
            weight = _number(order.get("weight_kg"), f"{order_id}.weight_kg")
            if weight <= 0.0:
                raise InputError(f"{order_id}.weight_kg must be positive")
            pickup_by = _nonnegative(order.get("pickup_by_min"), f"{order_id}.pickup_by_min")
            delivery_sla = _number(order.get("delivery_sla_min"), f"{order_id}.delivery_sla_min")
            if delivery_sla <= 0.0:
                raise InputError(f"{order_id}.delivery_sla_min must be positive")
            offer = _nonnegative(order.get("offer_idr"), f"{order_id}.offer_idr")
            required_vehicle = str(order.get("required_vehicle", ""))
            vehicle_index = VEHICLE_TYPES.index(required_vehicle) if required_vehicle in VEHICLE_TYPES else -1
            cargo_class = str(order.get("cargo_class", ""))
            cargo_index = CARGO_CLASSES.index(cargo_class) if cargo_class in CARGO_CLASSES else -1
            manifest_bound = 1.0 if _boolean(
                order.get("manifest_bound"), f"{order_id}.manifest_bound", False
            ) else 0.0
            order_age = _bounded(_nonnegative(
                order.get("order_age_min"), f"{order_id}.order_age_min", 0.0
            ) / 1440.0, 0.0, 3.0)
            cancellation = _unit_interval(
                order.get("cancellation_probability"), f"{order_id}.cancellation_probability", 0.08
            )
            priority = _unit_interval(order.get("priority"), f"{order_id}.priority", 0.5)
            service = _nonnegative(order.get("service_minutes"), f"{order_id}.service_minutes", 45.0)
            fragile = 1.0 if cargo_class in {"chilled", "hazmat"} else 0.0
            values = [
                *pickup_norm, *dropoff_norm, weight / 20000.0, pickup_by / 1440.0,
                delivery_sla / 2880.0, offer / 50_000_000.0, order_age, cancellation,
                manifest_bound, *_one_hot(vehicle_index, 5), *_one_hot(cargo_index, 5),
                priority, service / 180.0, fragile,
            ]
            order_features[0, index] = torch.tensor(values)
            order_mask[0, index] = True
            ood_values.extend(max(0.0, abs(value) - 1.5) for value in values)
            raw_orders.append({
                **dict(order), "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
                "dropoff_lat": dropoff_lat, "dropoff_lon": dropoff_lon,
                "weight": weight, "pickup_by": pickup_by, "delivery_sla": delivery_sla,
                "offer": offer, "vehicle_index": vehicle_index, "cargo_index": cargo_index,
                "manifest_bound_value": manifest_bound, "service": service,
            })

        densities = []
        for order in raw_orders:
            distances = [
                _haversine_km(order["dropoff_lat"], order["dropoff_lon"], other["pickup_lat"], other["pickup_lon"])
                for other in raw_orders
            ]
            densities.append(sum(math.exp(-distance / 220.0) for distance in distances) / len(distances))

        for truck_index, truck in enumerate(raw_trucks):
            for order_index, order in enumerate(raw_orders):
                reason_index = truck_index * len(raw_orders) + order_index
                reasons = rejection_reasons[reason_index]
                context = road_lookup.get((str(truck["truck_id"]), str(order["order_id"])), {})
                deadhead = _nonnegative(context.get("deadhead_km"), "pair.deadhead_km", _haversine_km(
                    truck["lat"], truck["lon"], order["pickup_lat"], order["pickup_lon"]
                ) * 1.22)
                loaded = _nonnegative(context.get("loaded_km"), "pair.loaded_km", _haversine_km(
                    order["pickup_lat"], order["pickup_lon"], order["dropoff_lat"], order["dropoff_lon"]
                ) * 1.22)
                weather = _unit_interval(context.get("weather_severity"), "pair.weather_severity", 0.15)
                congestion = _unit_interval(context.get("congestion"), "pair.congestion", 0.20)
                road_quality = _unit_interval(context.get("road_quality"), "pair.road_quality", 0.72)
                road_factor = 1.0 + 0.42 * weather + 0.58 * congestion + 0.24 * (1.0 - road_quality)
                route_km = (deadhead + loaded) * (1.05 + 0.18 * (1.0 - road_quality))
                road_minutes = _nonnegative(context.get("road_minutes"), "pair.road_minutes", route_km / 52.0 * 60.0 * road_factor)
                pickup_arrival = truck["available_after_min"] + deadhead / 48.0 * 60.0 * road_factor
                delivery_arrival = pickup_arrival + order["service"] + loaded / 52.0 * 60.0 * road_factor
                pickup_slack = order["pickup_by"] - pickup_arrival
                delivery_slack = order["delivery_sla"] - delivery_arrival
                capacity_ratio = order["weight"] / truck["capacity"]
                current_home = _haversine_km(truck["lat"], truck["lon"], truck["home_lat"], truck["home_lon"])
                drop_home = _haversine_km(order["dropoff_lat"], order["dropoff_lon"], truck["home_lat"], truck["home_lon"])
                alignment = _bounded((current_home - drop_home) / 1280.0, -1.0, 1.0)
                toll = _nonnegative(context.get("toll_idr"), "pair.toll_idr", route_km * 900.0)
                operating_cost = route_km * (5200.0 + 900.0 * capacity_ratio) + toll + 120000.0
                margin = order["offer"] - operating_cost
                road_stale = 1.0 if _boolean(context.get("road_stale"), "pair.road_stale", False) else 0.0
                road_available = _boolean(context.get("road_available"), "pair.road_available", True)
                values = [
                    deadhead / 1000.0, loaded / 1500.0, road_minutes / 1440.0,
                    _bounded(pickup_slack, -1440.0, 1440.0) / 1440.0,
                    _bounded(delivery_slack, -2880.0, 2880.0) / 2880.0,
                    _bounded(capacity_ratio, 0.0, 2.0),
                    _bounded(margin, -50_000_000.0, 50_000_000.0) / 50_000_000.0,
                    alignment, densities[order_index], weather, congestion, toll / 5_000_000.0,
                    road_quality, truck["quality"], truck["health_risk"], road_stale,
                ]
                pair_features[0, truck_index, order_index] = torch.tensor(values)
                ood_values.extend(max(0.0, abs(value) - 1.5) for value in values)
                if truck["vehicle_index"] < 0 or order["vehicle_index"] < 0:
                    reasons.append("UNKNOWN_VEHICLE_TYPE")
                elif truck["vehicle_index"] != order["vehicle_index"]:
                    reasons.append("VEHICLE_INCOMPATIBLE")
                if order["cargo_index"] < 0:
                    reasons.append("UNKNOWN_CARGO_CLASS")
                if order["weight"] > truck["available_capacity_at_pickup"]:
                    reasons.append("CAPACITY_EXCEEDED")
                if truck["fuel"] < 0.10:
                    reasons.append("LOW_FUEL")
                if truck["telemetry_age"] > 300.0:
                    reasons.append("STALE_TELEMETRY")
                if order["manifest_bound_value"] < 0.5:
                    reasons.append("MANIFEST_NOT_BOUND")
                if pickup_slack < 0.0:
                    reasons.append("PICKUP_WINDOW_MISSED")
                if delivery_slack < 0.0:
                    reasons.append("DELIVERY_SLA_MISSED")
                if not road_available:
                    reasons.append("ROAD_UNAVAILABLE")
                feasible[0, truck_index, order_index] = not reasons

        ood_score = sum(ood_values) / max(1, len(ood_values))
        return (
            truck_features, order_features, pair_features, truck_mask, order_mask,
            feasible, raw_trucks, raw_orders, rejection_reasons, ood_score,
        )

    @staticmethod
    def _decode(
        pair_score: Tensor,
        wait_score: Tensor,
        feasible: Tensor,
        trucks: Sequence[Mapping[str, Any]],
        orders: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[int, int, float]]:
        edges = []
        for truck_index in range(len(trucks)):
            for order_index in range(len(orders)):
                if bool(feasible[0, truck_index, order_index]):
                    score = float(pair_score[0, truck_index, order_index])
                    advantage = score - float(wait_score[0, truck_index])
                    edges.append((advantage, truck_index, order_index, score))
        edges.sort(reverse=True)
        used_trucks = set()
        used_orders = set()
        assignments: List[Tuple[int, int, float]] = []
        for advantage, truck_index, order_index, score in edges:
            if advantage <= 0.0:
                continue
            if truck_index in used_trucks or order_index in used_orders:
                continue
            used_trucks.add(truck_index)
            used_orders.add(order_index)
            assignments.append((truck_index, order_index, score))
        return assignments

    def infer(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(payload, Mapping):
            raise InputError("request body must be an object")
        request_id = _identifier(payload.get("request_id"), "request_id")
        (
            truck_features, order_features, pair_features, truck_mask, order_mask,
            feasible, trucks, orders, rejection_reasons, ood_score,
        ) = self._build_features(payload)
        with torch.inference_mode():
            policy, wait, utility, log_variance, eta_hours = self.model(
                truck_features, order_features, pair_features, truck_mask, order_mask, feasible
            )
        risk_adjusted = policy.float() - 0.10 * torch.exp(0.5 * log_variance.float())
        learned_eta_ok = torch.zeros_like(feasible)
        for order_index, order in enumerate(orders):
            learned_eta_ok[0, : len(trucks), order_index] = (
                eta_hours[0, : len(trucks), order_index].float() * 60.0
                <= float(order["delivery_sla"])
            )
        conservative_feasible = feasible & learned_eta_ok
        assignments = self._decode(
            risk_adjusted, wait.float(), conservative_feasible, trucks, orders
        )
        recommendations = []
        for truck_index, order_index, score in assignments:
            std = float(torch.exp(0.5 * log_variance[0, truck_index, order_index]).item())
            recommendations.append({
                "truck_id": str(trucks[truck_index]["truck_id"]),
                "order_id": str(orders[order_index]["order_id"]),
                "policy_score": round(score, 6),
                "predicted_contribution_idr": round(float(utility[0, truck_index, order_index]) * 10_000_000.0),
                "eta_minutes": round(float(eta_hours[0, truck_index, order_index]) * 60.0),
                "uncertainty": round(std, 6),
                "requires_human_acceptance": True,
            })
        rejected_pairs = []
        for truck_index, truck in enumerate(trucks):
            for order_index, order in enumerate(orders):
                reasons = rejection_reasons[truck_index * len(orders) + order_index]
                if (
                    not reasons
                    and bool(feasible[0, truck_index, order_index])
                    and not bool(learned_eta_ok[0, truck_index, order_index])
                ):
                    reasons.append("ETA_RISK_ABOVE_SLA")
                if reasons:
                    rejected_pairs.append({
                        "truck_id": str(truck["truck_id"]),
                        "order_id": str(order["order_id"]),
                        "reasons": reasons,
                    })
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "schema_version": "haulio.backhaul-policy-output.v1",
            "request_id": request_id,
            "model": {
                "name": self.manifest["model_name"],
                "version": self.manifest["model_version"],
                "weights_sha256": self.model_hash,
                "weights_static": True,
            },
            "recommendations": recommendations,
            "rejected_pairs": rejected_pairs,
            "abstain": len(recommendations) == 0,
            "ood_score": round(ood_score, 6),
            "latency_ms": round(elapsed_ms, 3),
            "automatic_commit": False,
        }
