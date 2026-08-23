from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings


ROUTES_ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"


def _waypoint(point: dict[str, Any]) -> dict[str, Any]:
    return {"location": {"latLng": {"latitude": point["lat"], "longitude": point["lon"]}}}


def _duration_minutes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)s", value)
    return round(float(match.group(1)) / 60) if match else None


def live_traffic_confirmation(plan: dict[str, Any], settings: Settings) -> tuple[dict[str, Any], int]:
    """One dispatcher-triggered request; Google route content is not retained or used for training."""
    if not settings.google_maps_key:
        return {
            "available": False,
            "reason": "GOOGLE_MAP_API is not configured on the server",
        }, 503
    geometry = plan["geometry"]
    body: dict[str, Any] = {
        "origin": _waypoint(geometry[0]),
        "destination": _waypoint(geometry[-1]),
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        # Routes API requires a future timestamp for traffic-aware requests; one minute
        # avoids a race between client-side creation and Google's validation.
        "departureTime": (datetime.now(UTC) + timedelta(minutes=1)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if len(geometry) > 2:
        body["intermediates"] = [_waypoint(point) for point in geometry[1:-1]]
    request = Request(
        ROUTES_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": settings.google_maps_key,
            "X-Goog-FieldMask": "routes.duration,routes.staticDuration,routes.distanceMeters",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=12) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # Google error bodies are summarized without relaying headers or credentials.
        try:
            message = json.loads(error.read().decode("utf-8")).get("error", {}).get("message")
        except (UnicodeDecodeError, json.JSONDecodeError):
            message = None
        reason = message or "Check that Routes API is enabled and this server IP is allowed."
        return {"available": False, "reason": f"Google Routes API returned HTTP {error.code}: {reason}"}, 502
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"available": False, "reason": f"Google traffic request failed: {type(error).__name__}"}, 502
    route = (raw.get("routes") or [{}])[0]
    live_minutes = _duration_minutes(route.get("duration"))
    static_minutes = _duration_minutes(route.get("staticDuration"))
    if live_minutes is None:
        return {"available": False, "reason": "Google returned no route duration for this plan."}, 502
    return {
        "available": True,
        "provider": "Google Routes API",
        "live_eta_min": live_minutes,
        "static_eta_min": static_minutes,
        "traffic_delay_min": max(0, live_minutes - static_minutes) if static_minutes is not None else None,
        "distance_km": round((route.get("distanceMeters") or 0) / 1000, 1),
        "notice": "Live traffic is displayed for dispatcher confirmation only; it is not stored or used to train the model.",
    }, 200
