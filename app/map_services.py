from __future__ import annotations

import copy
import json
import math
import time
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings
from .engine import Optimizer, haversine_km
from .store import Repository


CORRIDOR_VIAS = [
    {"name": "Northern Java corridor", "lat": -6.885, "lon": 109.126},  # Tegal
    {"name": "Central Java corridor", "lat": -7.797, "lon": 110.370},  # Yogyakarta
    {"name": "West Java interior corridor", "lat": -6.596, "lon": 106.806},  # Bogor
]


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "CompfestAICBackhaulMVP/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=18) as response:
        return json.loads(response.read().decode("utf-8"))


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    previous = len(ring) - 1
    for current, pair in enumerate(ring):
        x1, y1 = pair[0], pair[1]
        x2, y2 = ring[previous][0], ring[previous][1]
        crosses = (y1 > lat) != (y2 > lat)
        if crosses and lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        previous = current
    return inside


def _point_in_polygon(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    polygons = [geometry["coordinates"]] if geometry.get("type") == "Polygon" else geometry.get("coordinates", [])
    for polygon in polygons:
        if polygon and _point_in_ring(lon, lat, polygon[0]) and not any(_point_in_ring(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False


def _dedupe_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for point in points:
        if not result or haversine_km(point, result[-1]) > 0.05:
            result.append(copy.deepcopy(point))
    return result


def _simplify_coordinates(coordinates: list[list[float]], maximum_points: int = 1600) -> list[list[float]]:
    """Cap payload/render cost while retaining sampled road-following geometry."""
    if len(coordinates) <= maximum_points:
        return coordinates
    step = math.ceil((len(coordinates) - 1) / (maximum_points - 1))
    simplified = coordinates[::step]
    if simplified[-1] != coordinates[-1]:
        simplified.append(coordinates[-1])
    return simplified


class RegionService:
    """Loads open ADM1 polygons and enriches them with current local fleet activity."""

    def __init__(self, repository: Repository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self._geojson: dict[str, Any] | None = None
        self._expires_at = 0.0
        self._lock = Lock()

    def _base_geojson(self) -> dict[str, Any]:
        with self._lock:
            if self._geojson and time.monotonic() < self._expires_at:
                return self._geojson
            self._geojson = _request_json(self.settings.region_geojson_url)
            if len(self._geojson.get("features", [])) != 38:
                raise RuntimeError("Indonesia province boundary source did not contain all 38 provinces")
            self._expires_at = time.monotonic() + 3600
            return self._geojson

    def regions(self, optimizer: Optimizer) -> dict[str, Any]:
        geojson = copy.deepcopy(self._base_geojson())
        fleet = optimizer.fleet_view()
        logs = self.repository.accepted_telemetry()
        max_trucks = 1
        max_logs = 1
        enriched: list[dict[str, Any]] = []
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry") or {}
            trucks = [truck for truck in fleet if _point_in_polygon(truck["position"]["lon"], truck["position"]["lat"], geometry)]
            regional_logs = [event for event in logs if _point_in_polygon(float(event["lon"]), float(event["lat"]), geometry)]
            max_trucks = max(max_trucks, len(trucks))
            max_logs = max(max_logs, len(regional_logs))
            enriched.append({"feature": feature, "trucks": trucks, "logs": regional_logs})
        for entry in enriched:
            feature, trucks, logs_for_region = entry["feature"], entry["trucks"], entry["logs"]
            name = feature.get("properties", {}).get("shapeName") or feature.get("properties", {}).get("name") or "Unknown region"
            truck_count, log_count = len(trucks), len(logs_for_region)
            activity = round(min(1.0, truck_count / max_trucks * 0.74 + log_count / max_logs * 0.26), 2) if (truck_count or log_count) else 0.0
            traffic = {"jammed": 0, "slow": 0, "free": 0}
            for truck in trucks:
                traffic[truck["traffic"]["level"]] += 1
            feature["properties"] = {
                "name": name,
                "truck_count": truck_count,
                "log_count": log_count,
                "activity": activity,
                "activity_level": "high" if activity >= 0.66 else "medium" if activity >= 0.3 else "low" if activity else "none",
                "traffic": traffic,
            }
        return {
            "type": "FeatureCollection",
            "features": [entry["feature"] for entry in enriched],
            "meta": {
                "boundary_source": "Indonesia 38-province GeoJSON",
                "boundary_license": "MIT; configured by REGION_GEOJSON_URL",
                "color_metric": "current trucks weighted 74% and accepted telemetry logs weighted 26%",
            },
        }


class RouteService:
    """Returns map-ready road geometry from OSRM plus local fleet traffic bands."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = Lock()

    def _osrm(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points = _dedupe_points(points)
        if len(points) < 2:
            return []
        coordinates = ";".join(f"{point['lon']:.6f},{point['lat']:.6f}" for point in points)
        query = urlencode({"alternatives": "true", "overview": "full", "geometries": "geojson", "steps": "false"})
        try:
            response = _request_json(f"{self.settings.osrm_base_url}/route/v1/driving/{coordinates}?{query}")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return []
        if response.get("code") != "Ok":
            return []
        return response.get("routes", [])

    @staticmethod
    def _route_signature(route: dict[str, Any]) -> tuple[int, int]:
        return (round(float(route.get("distance", 0)) / 100), round(float(route.get("duration", 0)) / 10))

    @staticmethod
    def _traffic_level(primary: str, index: int) -> str:
        levels = ["jammed", "slow", "free"]
        return levels[(levels.index(primary) + index) % len(levels)]

    @staticmethod
    def _traffic_metrics(static_minutes: int, traffic_level: str) -> tuple[int, int]:
        multiplier = {"jammed": 1.42, "slow": 1.16, "free": 1.0}[traffic_level]
        p50 = math.ceil(static_minutes * multiplier)
        return p50, math.ceil(p50 * 1.22)

    def route_options(self, plan: dict[str, Any], truck: dict[str, Any], traffic: dict[str, Any]) -> dict[str, Any]:
        cache_key = f"{plan['id']}:{truck['position']['lat']:.4f}:{truck['position']['lon']:.4f}:{truck['sequence']}"
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < 90:
                return copy.deepcopy(cached[1])

        required = _dedupe_points([truck["position"], *plan["geometry"]])
        raw_routes = self._osrm(required)
        candidates: list[tuple[str, str, dict[str, Any]]] = []
        if raw_routes:
            candidates.extend(("recommended", "Recommended road route", route) for route in raw_routes)
        # OSRM cannot guarantee three alternatives. Add distinct real road routes through
        # operational corridors only when the native alternatives do not fill the set.
        segment_index = max(range(1, len(required)), key=lambda index: haversine_km(required[index - 1], required[index]))
        for via in CORRIDOR_VIAS:
            if len(candidates) >= 3:
                break
            if min(haversine_km(via, point) for point in required) < 18:
                continue
            route = next(iter(self._osrm([*required[:segment_index], via, *required[segment_index:]])), None)
            if route:
                candidates.append(("alternative", via["name"], route))

        seen: set[tuple[int, int]] = set()
        options = []
        for kind, label, route in candidates:
            signature = self._route_signature(route)
            if signature in seen:
                continue
            seen.add(signature)
            coordinates = _simplify_coordinates(route.get("geometry", {}).get("coordinates", []))
            if not coordinates:
                continue
            index = len(options)
            traffic_level = self._traffic_level(traffic["level"], index)
            static_min = max(1, math.ceil(float(route.get("duration", 0)) / 60))
            eta_p50, eta_p90 = self._traffic_metrics(static_min, traffic_level)
            options.append(
                {
                    "id": f"{plan['id']}-route-{index + 1}",
                    "rank": index + 1,
                    "kind": kind if index == 0 else "alternative",
                    "label": "Primary recommendation" if index == 0 else f"Alternative {index}: via {label}",
                    "coordinates": coordinates,
                    "distance_km": round(float(route.get("distance", 0)) / 1000, 1),
                    "static_eta_min": static_min,
                    "eta_p50_min": eta_p50,
                    "eta_p90_min": eta_p90,
                    "traffic": {
                        "level": traffic_level,
                        "label": {"jammed": "Heavy congestion", "slow": "Moderate congestion", "free": "Free flow"}[traffic_level],
                        "source": "verified vehicle speed with corridor operating estimate",
                    },
                }
            )
            if len(options) == 3:
                break

        if not options:
            # Degrades visibly but safely if the public demo router is unreachable.
            coordinates = [[point["lon"], point["lat"]] for point in required]
            eta = plan["eta_final_delivery_min"]
            options = [{"id": f"{plan['id']}-route-fallback", "rank": 1, "kind": "fallback", "label": "Planning fallback", "coordinates": coordinates, "distance_km": plan["distance_km"], "static_eta_min": eta, "eta_p50_min": eta, "eta_p90_min": math.ceil(eta * 1.25), "traffic": {"level": traffic["level"], "label": traffic["label"], "source": "fallback geometry; router unavailable"}}]

        result = {
            "plan_id": plan["id"],
            "truck_id": truck["id"],
            "route_source": "OSRM road routing on OpenStreetMap data" if options[0]["kind"] != "fallback" else "fallback",
            "routes": options,
            "stops": plan["stops"],
            "traffic_disclaimer": "Traffic colors are local fleet-GPS operating estimates. Google live traffic remains an on-demand dispatcher confirmation.",
        }
        with self._lock:
            self._cache[cache_key] = (time.monotonic(), copy.deepcopy(result))
        return result
