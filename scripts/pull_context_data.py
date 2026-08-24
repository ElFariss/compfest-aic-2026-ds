#!/usr/bin/env python3
"""Fetch public Indonesian weather context used to adjust ETA confidence.

This intentionally fetches no vehicle, driver, customer, or cargo data.  It creates
an ignored cache which the model can read locally, avoiding an API call for each ETA.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import NATIONAL_FLEET_CENTRES  # noqa: E402


OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
CURRENT_FIELDS = "temperature_2m,precipitation,rain,weather_code,wind_speed_10m,wind_gusts_10m"


def request_weather() -> list[dict]:
    latitudes = ",".join(str(latitude) for _, latitude, _ in NATIONAL_FLEET_CENTRES)
    longitudes = ",".join(str(longitude) for _, _, longitude in NATIONAL_FLEET_CENTRES)
    query = urlencode(
        {
            "latitude": latitudes,
            "longitude": longitudes,
            "current": CURRENT_FIELDS,
            "timezone": "Asia/Jakarta",
            "forecast_days": 1,
        },
        safe=",",
    )
    request = Request(
        f"{OPEN_METEO_ENDPOINT}?{query}",
        headers={"User-Agent": "haulio-ai-challenge-context/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, list) else [payload]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache current Indonesian public-weather context")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "context" / "weather-indonesia.json",
        help="output cache path (default: data/context/weather-indonesia.json)",
    )
    args = parser.parse_args()
    responses = request_weather()
    if len(responses) != len(NATIONAL_FLEET_CENTRES):
        raise RuntimeError(f"Expected {len(NATIONAL_FLEET_CENTRES)} weather locations, received {len(responses)}")
    locations = []
    for (name, latitude, longitude), response in zip(NATIONAL_FLEET_CENTRES, responses):
        current = response.get("current") if isinstance(response, dict) else None
        if not isinstance(current, dict):
            raise RuntimeError(f"Weather response for {name} has no current conditions")
        locations.append(
            {
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "current": current,
                "current_units": response.get("current_units", {}),
            }
        )
    output = {
        "source": "Open-Meteo Forecast API",
        "source_url": "https://open-meteo.com/en/docs",
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "locations": locations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(locations)} public-weather contexts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
