from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .store import Repository


HUBS = {
    "Jakarta DC": {"lat": -6.1745, "lon": 106.8227, "name": "Jakarta DC"},
    "Cikarang Hub": {"lat": -6.2839, "lon": 107.1735, "name": "Cikarang Hub"},
    "Cirebon Hub": {"lat": -6.7320, "lon": 108.5523, "name": "Cirebon Hub"},
    "Semarang Hub": {"lat": -6.9900, "lon": 110.4200, "name": "Semarang Hub"},
    "Surabaya Hub": {"lat": -7.2575, "lon": 112.7521, "name": "Surabaya Hub"},
    "Bandung Hub": {"lat": -6.9175, "lon": 107.6191, "name": "Bandung Hub"},
}


FLEET_SEED = [
    {
        "id": "TRK-01",
        "name": "Nusantara 12T",
        "vehicle_type": "box",
        "capacity_kg": 12000,
        "position": {"lat": -6.231, "lon": 106.967, "name": "Jakarta–Cikarang corridor"},
        "status": "en_route",
        "fuel_pct": 68,
        "speed_kph": 31,
        "heading": 89,
        "gps_accuracy_m": 9,
        "cargo_status": "loaded",
        "current_job": {"cargo": "FMCG cartons", "destination": "Cikarang Hub", "remaining_min": 58, "load_kg": 9200},
        "sequence": 0,
        "active_plan": None,
    },
    {
        "id": "TRK-02",
        "name": "Jawa Reefer 8T",
        "vehicle_type": "reefer",
        "capacity_kg": 8000,
        "position": {"lat": -6.944, "lon": 110.299, "name": "Semarang approach"},
        "status": "en_route",
        "fuel_pct": 44,
        "speed_kph": 18,
        "heading": 91,
        "gps_accuracy_m": 11,
        "cargo_status": "loaded",
        "current_job": {"cargo": "Chilled dairy", "destination": "Semarang Hub", "remaining_min": 32, "load_kg": 5300},
        "sequence": 0,
        "active_plan": None,
    },
    {
        "id": "TRK-03",
        "name": "Archipelago Flatbed 16T",
        "vehicle_type": "flatbed",
        "capacity_kg": 16000,
        "position": {"lat": -7.352, "lon": 112.682, "name": "Surabaya industrial zone"},
        "status": "available",
        "fuel_pct": 82,
        "speed_kph": 0,
        "heading": 0,
        "gps_accuracy_m": 7,
        "cargo_status": "empty",
        "current_job": None,
        "sequence": 0,
        "active_plan": None,
    },
    {
        "id": "TRK-04",
        "name": "Cirebon Box 6T",
        "vehicle_type": "box",
        "capacity_kg": 6000,
        "position": {"lat": -6.739, "lon": 108.539, "name": "Cirebon Hub"},
        "status": "available",
        "fuel_pct": 37,
        "speed_kph": 0,
        "heading": 0,
        "gps_accuracy_m": 8,
        "cargo_status": "empty",
        "current_job": None,
        "sequence": 0,
        "active_plan": None,
    },
]


ORDER_SEED = [
    {
        "id": "ORD-101",
        "cargo": "Packaged consumer goods",
        "cargo_class": "general",
        "weight_kg": 7800,
        "required_vehicle": "box",
        "pickup": "Cikarang Hub",
        "dropoff": "Semarang Hub",
        "pickup_by_min": 180,
        "delivery_sla_min": 760,
        "offer_idr": 19200000,
        "status": "open",
    },
    {
        "id": "ORD-102",
        "cargo": "Ambient food ingredients",
        "cargo_class": "general",
        "weight_kg": 4200,
        "required_vehicle": "box",
        "pickup": "Semarang Hub",
        "dropoff": "Surabaya Hub",
        "pickup_by_min": 700,
        "delivery_sla_min": 1080,
        "offer_idr": 12600000,
        "status": "open",
    },
    {
        "id": "ORD-103",
        "cargo": "Temperature-controlled produce",
        "cargo_class": "chilled",
        "weight_kg": 5100,
        "required_vehicle": "reefer",
        "pickup": "Semarang Hub",
        "dropoff": "Surabaya Hub",
        "pickup_by_min": 210,
        "delivery_sla_min": 640,
        "offer_idr": 15400000,
        "status": "open",
    },
    {
        "id": "ORD-104",
        "cargo": "Steel coils",
        "cargo_class": "industrial",
        "weight_kg": 13800,
        "required_vehicle": "flatbed",
        "pickup": "Surabaya Hub",
        "dropoff": "Cikarang Hub",
        "pickup_by_min": 220,
        "delivery_sla_min": 940,
        "offer_idr": 28600000,
        "status": "open",
    },
    {
        "id": "ORD-105",
        "cargo": "E-commerce parcels",
        "cargo_class": "general",
        "weight_kg": 3600,
        "required_vehicle": "box",
        "pickup": "Cirebon Hub",
        "dropoff": "Bandung Hub",
        "pickup_by_min": 155,
        "delivery_sla_min": 490,
        "offer_idr": 8800000,
        "status": "open",
    },
    {
        "id": "ORD-106",
        "cargo": "Returnable packaging",
        "cargo_class": "general",
        "weight_kg": 2900,
        "required_vehicle": "box",
        "pickup": "Bandung Hub",
        "dropoff": "Cikarang Hub",
        "pickup_by_min": 590,
        "delivery_sla_min": 980,
        "offer_idr": 9600000,
        "status": "open",
    },
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hub(name: str) -> dict[str, Any]:
    return copy.deepcopy(HUBS[name])


def haversine_km(a: dict[str, Any], b: dict[str, Any]) -> float:
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def road_km(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Offline baseline distance; a production deployment replaces this with OSRM matrix output."""
    direct = haversine_km(a, b)
    return round(direct * (1.08 if direct < 25 else 1.22), 1)


def drive_minutes(distance_km: float, urban: bool = False) -> int:
    speed = 34 if urban else 51
    return max(5, math.ceil(distance_km / speed * 60))


def rupiah(amount: float) -> int:
    return int(round(amount / 50000) * 50000)


def canonical_telemetry(payload: dict[str, Any]) -> bytes:
    material = {key: value for key, value in payload.items() if key != "signature"}
    return json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sign_telemetry(payload: dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_telemetry(payload), hashlib.sha256).hexdigest()


def point_to_segment_km(point: dict[str, Any], start: dict[str, Any], end: dict[str, Any]) -> float:
    """Equirectangular local approximation, sufficient for a route-corridor alert threshold."""
    km_per_lat = 110.574
    km_per_lon = 111.320 * math.cos(math.radians(point["lat"]))
    ax, ay = (start["lon"] - point["lon"]) * km_per_lon, (start["lat"] - point["lat"]) * km_per_lat
    bx, by = (end["lon"] - point["lon"]) * km_per_lon, (end["lat"] - point["lat"]) * km_per_lat
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / (dx * dx + dy * dy)))
    return math.hypot(ax + t * dx, ay + t * dy)


def distance_to_corridor_km(point: dict[str, Any], geometry: list[dict[str, Any]]) -> float:
    if len(geometry) < 2:
        return float("inf")
    return min(point_to_segment_km(point, start, end) for start, end in zip(geometry, geometry[1:]))


class Optimizer:
    """Explainable baseline for the six challenge capabilities.

    The formulas are intentionally labelled as baselines: the demo has no proprietary
    historical orders, so it must not claim that its risk or price scores were trained.
    """

    def __init__(self, settings: Settings, repository: Repository) -> None:
        self.settings = settings
        self.repository = repository
        self.fleet = copy.deepcopy(FLEET_SEED)
        self.orders = copy.deepcopy(ORDER_SEED)
        self.rejected_recommendations: set[str] = set()
        self.last_anomalies: dict[str, dict[str, Any]] = {}

    def _truck(self, truck_id: str) -> dict[str, Any] | None:
        return next((truck for truck in self.fleet if truck["id"] == truck_id), None)

    def _order(self, order_id: str) -> dict[str, Any] | None:
        return next((order for order in self.orders if order["id"] == order_id), None)

    def _empty_location(self, truck: dict[str, Any]) -> dict[str, Any]:
        job = truck.get("current_job")
        return hub(job["destination"]) if job else copy.deepcopy(truck["position"])

    def _available_after_min(self, truck: dict[str, Any]) -> int:
        return int(truck.get("current_job", {}).get("remaining_min", 0)) if truck.get("current_job") else 0

    def empty_return_risk(self, truck: dict[str, Any]) -> dict[str, Any]:
        endpoint = self._empty_location(truck)
        compatible_nearby = [
            order
            for order in self.orders
            if order["status"] == "open"
            and order["required_vehicle"] == truck["vehicle_type"]
            and order["weight_kg"] <= truck["capacity_kg"]
            and road_km(endpoint, hub(order["pickup"])) <= 70
        ]
        if truck.get("active_plan"):
            probability, reasons = 0.05, ["Backhaul already accepted by dispatcher"]
        else:
            base = 0.78 if truck.get("current_job") else 0.46
            probability = base - min(0.42, len(compatible_nearby) * 0.13)
            if truck["fuel_pct"] < 30:
                probability += 0.06
            if truck["vehicle_type"] == "reefer":
                probability += 0.05
            probability = max(0.03, min(0.95, probability))
            reasons = [
                f"{len(compatible_nearby)} compatible open load(s) within 70 km of the expected empty location",
                "Higher risk before delivery because the vehicle has no dispatcher-approved next job",
            ]
            if truck["fuel_pct"] < 30:
                reasons.append("Fuel below 30% reduces usable matching flexibility")
        return {
            "probability": round(probability, 2),
            "level": "high" if probability >= 0.65 else "medium" if probability >= 0.35 else "low",
            "expected_empty_location": endpoint["name"],
            "reasons": reasons,
        }

    def eta(self, truck: dict[str, Any]) -> dict[str, Any]:
        job = truck.get("current_job")
        if not job:
            return {"p50_min": 0, "p90_min": 0, "destination": truck["position"]["name"], "source": "at hub"}
        p50 = int(job["remaining_min"])
        reliability_penalty = 1.28 + max(0, truck["gps_accuracy_m"] - 15) / 100
        return {
            "p50_min": p50,
            "p90_min": math.ceil(p50 * reliability_penalty),
            "destination": job["destination"],
            "source": "offline operating baseline",
        }

    def traffic_state(self, truck: dict[str, Any]) -> dict[str, Any]:
        """Traffic band inferred from verified fleet speed, not from a route provider's data."""
        if truck["status"] in {"available", "awaiting_backhaul"}:
            return {"level": "free", "label": "No congestion observed", "color": "blue", "source": "vehicle available at hub"}
        speed = float(truck.get("speed_kph", 0))
        if speed < 25:
            return {"level": "jammed", "label": "Heavy congestion", "color": "red", "source": f"verified speed {speed:.0f} km/h"}
        if speed < 42:
            return {"level": "slow", "label": "Moderate congestion", "color": "yellow", "source": f"verified speed {speed:.0f} km/h"}
        return {"level": "free", "label": "Free flow", "color": "blue", "source": f"verified speed {speed:.0f} km/h"}

    def _is_compatible(self, truck: dict[str, Any], order: dict[str, Any]) -> bool:
        return (
            order["status"] == "open"
            and truck["vehicle_type"] == order["required_vehicle"]
            and truck["capacity_kg"] >= order["weight_kg"]
        )

    def _plan_for_orders(self, truck: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, Any] | None:
        start = self._empty_location(truck)
        available_after = self._available_after_min(truck)
        points: list[dict[str, Any]] = [copy.deepcopy(start)]
        stops: list[dict[str, Any]] = [{"kind": "empty_available", "name": start["name"], "lat": start["lat"], "lon": start["lon"]}]
        cursor = start
        elapsed = available_after
        total_distance = 0.0
        service_minutes = 0
        for order in orders:
            pickup = hub(order["pickup"])
            dropoff = hub(order["dropoff"])
            deadhead = road_km(cursor, pickup)
            elapsed += drive_minutes(deadhead, deadhead < 35)
            total_distance += deadhead
            if elapsed > order["pickup_by_min"]:
                return None
            points.append(copy.deepcopy(pickup))
            stops.append({"kind": "pickup", "order_id": order["id"], "cargo": order["cargo"], **pickup})
            service_minutes += 25
            elapsed += 25
            haul = road_km(pickup, dropoff)
            total_distance += haul
            elapsed += drive_minutes(haul)
            points.append(copy.deepcopy(dropoff))
            stops.append({"kind": "dropoff", "order_id": order["id"], "cargo": order["cargo"], **dropoff})
            elapsed += 20
            service_minutes += 20
            cursor = dropoff

        fuel_cost = total_distance * 0.27 * 11500
        driver_cost = total_distance * 1650
        maintenance_cost = total_distance * 700
        stop_cost = 80000 * len(orders)
        operating_cost = rupiah(fuel_cost + driver_cost + maintenance_cost + stop_cost)
        revenue = sum(order["offer_idr"] for order in orders)
        margin = revenue - operating_cost
        if margin <= 0:
            return None
        min_quote = rupiah(operating_cost * 1.12)
        score = round((margin / max(revenue, 1)) * 60 + min(25, margin / 1_000_000 * 4) - total_distance / 180, 1)
        primary = orders[0]
        return {
            "id": f"REC-{truck['id']}-{'-'.join(order['id'].replace('ORD-', '') for order in orders)}",
            "truck_id": truck["id"],
            "truck_name": truck["name"],
            "order_ids": [order["id"] for order in orders],
            "cargo_summary": " → ".join(order["cargo"] for order in orders),
            "is_multi_hop": len(orders) > 1,
            "capacity_used_kg": max(order["weight_kg"] for order in orders),
            "capacity_pct": round(max(order["weight_kg"] for order in orders) / truck["capacity_kg"] * 100),
            "expected_empty_location": start["name"],
            "eta_to_first_pickup_min": self._available_after_min(truck) + drive_minutes(road_km(start, hub(primary["pickup"])), road_km(start, hub(primary["pickup"])) < 35),
            "eta_final_delivery_min": elapsed,
            "distance_km": round(total_distance, 1),
            "service_minutes": service_minutes,
            "revenue_idr": revenue,
            "operating_cost_idr": operating_cost,
            "expected_margin_idr": margin,
            "margin_pct": round(margin / revenue * 100, 1),
            "minimum_viable_quote_idr": min_quote,
            "suggested_quote_idr": max(min_quote, revenue),
            "score": score,
            "confidence": round(min(0.93, 0.63 + (0.11 if len(orders) == 2 else 0) + (0.08 if truck["gps_accuracy_m"] <= 15 else 0)), 2),
            "stops": stops,
            "geometry": points,
            "explanation": [
                f"{truck['vehicle_type'].title()} capacity and {max(order['weight_kg'] for order in orders):,} kg payload are compatible",
                f"Pickup window remains feasible with a {self._available_after_min(truck)} minute current-job ETA",
                f"Estimated margin is Rp{margin:,.0f} after fuel, driver, maintenance, and stop costs",
            ],
            "status": "proposed",
        }

    def recommendations(self) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        for truck in self.fleet:
            if truck.get("active_plan"):
                active = copy.deepcopy(truck["active_plan"])
                active["status"] = "accepted"
                plans.append(active)
                continue
            compatible = [order for order in self.orders if self._is_compatible(truck, order)]
            candidates: list[dict[str, Any]] = []
            for order in compatible:
                plan = self._plan_for_orders(truck, [order])
                if plan:
                    candidates.append(plan)
            for first in compatible:
                for second in compatible:
                    if first["id"] == second["id"]:
                        continue
                    if road_km(hub(first["dropoff"]), hub(second["pickup"])) > 75:
                        continue
                    plan = self._plan_for_orders(truck, [first, second])
                    if plan:
                        candidates.append(plan)
            candidates.sort(key=lambda plan: (plan["score"], plan["expected_margin_idr"]), reverse=True)
            plans.extend(plan for plan in candidates[:3] if plan["id"] not in self.rejected_recommendations)
        return sorted(plans, key=lambda plan: (plan["status"] == "accepted", plan["score"]), reverse=True)

    def recommendation(self, recommendation_id: str) -> dict[str, Any] | None:
        return next((plan for plan in self.recommendations() if plan["id"] == recommendation_id), None)

    def fleet_view(self) -> list[dict[str, Any]]:
        result = []
        for truck in self.fleet:
            item = copy.deepcopy(truck)
            item["empty_return_risk"] = self.empty_return_risk(truck)
            item["eta"] = self.eta(truck)
            item["traffic"] = self.traffic_state(truck)
            item["anomaly"] = self.last_anomalies.get(truck["id"], {"status": "normal", "score": 0.02, "signals": []})
            item.pop("active_plan", None)
            result.append(item)
        return result

    def orders_view(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.orders)

    def dashboard_metrics(self) -> dict[str, Any]:
        fleet = self.fleet_view()
        recommendations = self.recommendations()
        audit = self.repository.metrics()
        return {
            "fleet_total": len(fleet),
            "fleet_at_empty_risk": sum(1 for truck in fleet if truck["empty_return_risk"]["probability"] >= 0.5),
            "open_orders": sum(1 for order in self.orders if order["status"] == "open"),
            "recommendation_count": len(recommendations),
            "recoverable_margin_idr": sum(plan["expected_margin_idr"] for plan in recommendations if plan["status"] == "proposed"),
            "google_routes_configured": self.settings.google_enabled,
            "iot_demo_secret_warning": self.settings.using_demo_iot_secret,
            **audit,
        }

    def decide(self, recommendation_id: str, action: str, note: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
        if action not in {"accept", "reject"}:
            return None, "action must be accept or reject"
        plan = self.recommendation(recommendation_id)
        if not plan:
            return None, "recommendation not found or no longer feasible"
        if action == "reject":
            self.rejected_recommendations.add(recommendation_id)
            self.repository.log_decision(recommendation_id, action, note)
            plan["status"] = "rejected"
            return plan, None
        truck = self._truck(plan["truck_id"])
        if not truck or truck.get("active_plan"):
            return None, "truck already has an active backhaul plan"
        for order_id in plan["order_ids"]:
            order = self._order(order_id)
            if not order or order["status"] != "open":
                return None, f"{order_id} is no longer available"
        plan["status"] = "accepted"
        truck["active_plan"] = copy.deepcopy(plan)
        truck["status"] = "assigned_backhaul"
        truck["route_stop_index"] = 0
        for order_id in plan["order_ids"]:
            self._order(order_id)["status"] = "assigned"
        self.repository.log_decision(recommendation_id, action, note)
        return copy.deepcopy(plan), None

    def _anomaly(self, truck: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        signals: list[str] = []
        score = 0.02
        if float(payload.get("gps_accuracy_m", 999)) > 80:
            signals.append("low GPS accuracy")
            score += 0.22
        if float(payload.get("speed_kph", 0)) > 110:
            signals.append("truck speed exceeds 110 km/h")
            score += 0.43
        active = truck.get("active_plan")
        if active:
            deviation = distance_to_corridor_km(payload, active["geometry"])
            if deviation > 6:
                signals.append(f"{deviation:.1f} km outside approved route corridor")
                score += 0.56
            elif deviation > 2:
                signals.append(f"{deviation:.1f} km route deviation; evaluate a valid re-route")
                score += 0.18
            slack = min(self._order(order_id)["pickup_by_min"] for order_id in active["order_ids"]) - active["eta_to_first_pickup_min"]
            if 2 < deviation <= 6 and slack >= 90:
                status = "valid_reroute"
            elif deviation > 6:
                status = "replan_required"
            else:
                status = "normal"
        else:
            deviation = 0.0
            status = "normal"
        if score >= 0.65:
            status = "replan_required" if active else "review"
        return {"status": status, "score": round(min(score, 0.99), 2), "signals": signals, "deviation_km": round(deviation, 2)}

    def process_telemetry(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        required = {"truck_id", "timestamp", "lat", "lon", "speed_kph", "heading", "gps_accuracy_m", "cargo_status", "fuel_pct", "sequence", "signature"}
        missing = sorted(required - set(payload))
        truck_id = payload.get("truck_id")
        if missing:
            self.repository.log_telemetry(truck_id, False, f"missing fields: {', '.join(missing)}", payload)
            return {"accepted": False, "reason": f"missing fields: {', '.join(missing)}"}, 400
        truck = self._truck(str(truck_id))
        expected_signature = sign_telemetry(payload, self.settings.iot_shared_secret)
        if not truck or not hmac.compare_digest(str(payload["signature"]), expected_signature):
            self.repository.log_telemetry(truck_id, False, "invalid device signature or truck", payload)
            return {"accepted": False, "reason": "invalid device signature or truck"}, 401
        try:
            sequence = int(payload["sequence"])
            lat, lon = float(payload["lat"]), float(payload["lon"])
            if not (-11.5 <= lat <= 6.5 and 94 <= lon <= 142):
                raise ValueError("outside Indonesian operating area")
        except (TypeError, ValueError) as error:
            self.repository.log_telemetry(truck_id, False, str(error), payload)
            return {"accepted": False, "reason": str(error)}, 400
        if sequence <= int(truck.get("sequence", 0)):
            self.repository.log_telemetry(truck_id, False, "non-monotonic telemetry sequence", payload)
            return {"accepted": False, "reason": "non-monotonic telemetry sequence"}, 409

        truck.update(
            {
                "position": {"lat": lat, "lon": lon, "name": truck["position"].get("name", "live GPS")},
                "speed_kph": round(float(payload["speed_kph"]), 1),
                "heading": round(float(payload["heading"]), 1),
                "gps_accuracy_m": round(float(payload["gps_accuracy_m"]), 1),
                "cargo_status": str(payload["cargo_status"]),
                "fuel_pct": round(float(payload["fuel_pct"]), 1),
                "sequence": sequence,
                "last_seen": str(payload["timestamp"]),
            }
        )
        anomaly = self._anomaly(truck, payload)
        self.last_anomalies[truck_id] = anomaly
        self.repository.log_telemetry(truck_id, True, anomaly["status"], payload)
        return {"accepted": True, "truck_id": truck_id, "anomaly": anomaly}, 202

    def simulator_tick(self) -> list[dict[str, Any]]:
        """Move demo vehicles a small amount and submit genuine HMAC-signed telemetry."""
        results = []
        for truck in self.fleet:
            target = None
            if truck.get("active_plan"):
                stops = truck["active_plan"]["stops"]
                index = min(int(truck.get("route_stop_index", 0)) + 1, len(stops) - 1)
                target = stops[index]
            elif truck.get("current_job"):
                target = hub(truck["current_job"]["destination"])
            if target:
                distance = haversine_km(truck["position"], target)
                fraction = min(0.12, 3.5 / max(distance, 0.001))
                truck["position"]["lat"] += (target["lat"] - truck["position"]["lat"]) * fraction
                truck["position"]["lon"] += (target["lon"] - truck["position"]["lon"]) * fraction
                truck["speed_kph"] = 48 if distance > 4 else 18
                if distance < 4:
                    truck["position"]["name"] = target["name"]
                    if truck.get("active_plan") and index < len(stops) - 1:
                        truck["route_stop_index"] = index
                    elif truck.get("current_job"):
                        truck["current_job"] = None
                        truck["status"] = "awaiting_backhaul"
                        truck["cargo_status"] = "empty"
            truck["fuel_pct"] = max(8, truck["fuel_pct"] - 0.15)
            payload = {
                "truck_id": truck["id"],
                "timestamp": utc_now(),
                "lat": round(truck["position"]["lat"], 6),
                "lon": round(truck["position"]["lon"], 6),
                "speed_kph": truck["speed_kph"],
                "heading": truck["heading"],
                "gps_accuracy_m": truck["gps_accuracy_m"],
                "cargo_status": truck["cargo_status"],
                "fuel_pct": round(truck["fuel_pct"], 1),
                "sequence": int(truck.get("sequence", 0)) + 1,
            }
            payload["signature"] = sign_telemetry(payload, self.settings.iot_shared_secret)
            result, _ = self.process_telemetry(payload)
            results.append(result)
        return results
