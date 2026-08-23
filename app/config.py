from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Minimal .env loader so the demo has no third-party runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    google_maps_key: str | None
    iot_shared_secret: str
    host: str
    port: int
    data_dir: Path
    osrm_base_url: str
    region_geojson_url: str

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_maps_key)

    @property
    def using_demo_iot_secret(self) -> bool:
        return self.iot_shared_secret == "demo-change-me"


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    return Settings(
        google_maps_key=os.getenv("GOOGLE_MAP_API") or None,
        iot_shared_secret=os.getenv("IOT_SHARED_SECRET", "demo-change-me"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        data_dir=data_dir,
        osrm_base_url=os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/"),
        region_geojson_url=os.getenv("REGION_GEOJSON_URL", "https://raw.githubusercontent.com/AlfianAliM/Indonesia-GeoJSON/master/provinsi.geojson"),
    )
