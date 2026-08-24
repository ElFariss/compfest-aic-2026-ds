from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse

from .config import ROOT, Settings
from .engine import Optimizer
from .google_routes import live_traffic_confirmation
from .map_services import RegionService, RouteService
from .store import Repository


STATIC_DIR = Path(os.getenv("STATIC_DIR", str(ROOT / "static"))).resolve()


class AppServer(ThreadingHTTPServer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = Repository(settings.data_dir / "optimizer.sqlite3")
        self.optimizer = Optimizer(settings, self.repository)
        self.region_service = RegionService(self.repository, settings)
        self.route_service = RouteService(settings)
        super().__init__((settings.host, settings.port), RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer
    protocol_version = "HTTP/1.1"

    def handle(self) -> None:
        """Ignore a client disconnect before a complete HTTP request is received."""
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        # Retain useful local request logging without ever printing request bodies.
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def _send_json(self, value: object, status: int = 200, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, content in (headers or {}).items():
            self.send_header(key, content)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (content_type or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid Content-Length")
        if length <= 0 or length > 1_000_000:
            raise ValueError("request body must be between 1 byte and 1 MB")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/":
            self._send_file(STATIC_DIR / "index.html")
            return
        if path == "/static/app.js":
            self._send_file(STATIC_DIR / "app.js")
            return
        if path == "/static/style.css":
            self._send_file(STATIC_DIR / "style.css")
            return
        if path == "/api/v1/health":
            self._send_json({"status": "ok", "service": "autonomous-backhaul-optimizer", "google_routes_configured": self.server.settings.google_enabled})
            return
        if path == "/api/v1/metrics":
            self._send_json(self.server.optimizer.dashboard_metrics())
            return
        if path == "/api/v1/fleet":
            self._send_json({"fleet": self.server.optimizer.fleet_view()})
            return
        if path == "/api/v1/orders":
            self._send_json({"orders": self.server.optimizer.orders_view()})
            return
        if path == "/api/v1/regions":
            try:
                self._send_json(self.server.region_service.regions(self.server.optimizer))
            except (RuntimeError, OSError, ValueError, HTTPError, URLError) as error:
                self._send_json({"error": f"Indonesia boundary data is temporarily unavailable: {type(error).__name__}"}, HTTPStatus.BAD_GATEWAY)
            return
        if path == "/api/v1/recommendations":
            self._send_json({"recommendations": self.server.optimizer.recommendations()})
            return
        parts = path.strip("/").split("/")
        if len(parts) >= 4 and parts[:3] == ["api", "v1", "recommendations"]:
            recommendation_id = parts[3]
            plan = self.server.optimizer.recommendation(recommendation_id)
            if not plan:
                self._send_json({"error": "recommendation not found"}, HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 5 and parts[4] == "live-traffic":
                content, status = live_traffic_confirmation(plan, self.server.settings)
                self._send_json(content, status)
                return
            if len(parts) == 5 and parts[4] == "route-options":
                truck = self.server.optimizer._truck(plan["truck_id"])
                if not truck:
                    self._send_json({"error": "truck not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json(self.server.route_service.route_options(plan, truck, self.server.optimizer.traffic_state(truck)))
                return
            if len(parts) == 4:
                self._send_json(plan)
                return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            payload = self._read_json()
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/v1/telemetry":
            content, status = self.server.optimizer.process_telemetry(payload)
            self._send_json(content, status)
            return
        if path == "/api/v1/simulation/tick":
            self._send_json({"events": self.server.optimizer.simulator_tick()})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "v1", "recommendations"] and parts[4] == "decision":
            result, error = self.server.optimizer.decide(parts[3], str(payload.get("action", "")), payload.get("note"))
            if error:
                self._send_json({"error": error}, HTTPStatus.CONFLICT)
            else:
                self._send_json({"recommendation": result})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def serve(settings: Settings) -> None:
    server = AppServer(settings)
    print(f"Autonomous Backhaul Optimizer on http://{settings.host}:{settings.port}")
    print("Google Routes traffic confirmation: " + ("configured" if settings.google_enabled else "not configured"))
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
