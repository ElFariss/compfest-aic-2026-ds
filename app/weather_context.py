from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class WeatherContext:
    """Read an optional, cached public-weather snapshot without calling an API per ETA.

    The cache is deliberately external context: it is not truck telemetry and it is
    safe for the model API to operate without it when an operator has not pulled a
    recent snapshot.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime_ns: int | None = None
        self._locations: list[dict[str, Any]] = []
        self._retrieved_at: str | None = None

    def for_position(self, position: dict[str, Any]) -> dict[str, Any] | None:
        self._reload_if_changed()
        if not self._locations:
            return None
        latitude, longitude = float(position["lat"]), float(position["lon"])
        nearest = min(
            self._locations,
            key=lambda location: math.hypot(
                latitude - float(location["latitude"]),
                longitude - float(location["longitude"]),
            ),
        )
        current = nearest.get("current")
        if not isinstance(current, dict):
            return None
        # Open-Meteo precipitation already includes rain; use the larger reading
        # rather than double-counting the same precipitation in ETA adjustment.
        rain = max(_number(current.get("rain")), _number(current.get("precipitation")))
        gust = _number(current.get("wind_gusts_10m"))
        factor = 1.0
        if rain >= 10:
            factor += 0.20
        elif rain >= 4:
            factor += 0.12
        elif rain >= 1:
            factor += 0.05
        if gust >= 55:
            factor += 0.10
        elif gust >= 35:
            factor += 0.04
        return {
            "source": "cached public weather context",
            "retrieved_at": self._retrieved_at,
            "nearest_operating_centre": nearest.get("name"),
            "temperature_c": _number(current.get("temperature_2m")),
            "rain_mm": round(rain, 1),
            "wind_gust_kph": round(gust, 1),
            "eta_factor": round(factor, 2),
        }

    def _reload_if_changed(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            self._locations, self._retrieved_at, self._mtime_ns = [], None, None
            return
        if self._mtime_ns == stat.st_mtime_ns:
            return
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
            locations = parsed.get("locations", []) if isinstance(parsed, dict) else []
            self._locations = [
                location
                for location in locations
                if isinstance(location, dict)
                and isinstance(location.get("name"), str)
                and isinstance(location.get("current"), dict)
                and _finite(location.get("latitude"))
                and _finite(location.get("longitude"))
            ]
            self._retrieved_at = parsed.get("retrieved_at") if isinstance(parsed, dict) else None
        except (OSError, ValueError, TypeError):
            self._locations, self._retrieved_at = [], None
        self._mtime_ns = stat.st_mtime_ns


def _number(value: Any) -> float:
    return float(value) if _finite(value) else 0.0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
