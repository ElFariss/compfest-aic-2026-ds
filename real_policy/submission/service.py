#!/usr/bin/env python3
"""Loopback-oriented HTTP wrapper around immutable core inference."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

from policy_runtime import FrozenPolicy


MAX_BODY_BYTES = 2 * 1024 * 1024
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).with_name("artifacts")))
POLICY = FrozenPolicy(ARTIFACT_DIR)


class Handler(BaseHTTPRequestHandler):
    server_version = "HaulioRealPolicy/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the default audit-friendly request log but omit request bodies.
        super().log_message(format, *args)

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._json(HTTPStatus.OK, POLICY.health())

    def do_POST(self) -> None:
        handlers: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
            "/infer/route": POLICY.infer_route,
            "/infer/telemetry": POLICY.infer_telemetry,
            "/infer/truck-track": POLICY.infer_track,
            "/infer/health": POLICY.infer_health,
            "/infer/deadhead": POLICY.infer_deadhead,
            "/infer/price": POLICY.infer_price,
        }
        operation = handlers.get(self.path)
        if operation is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "application_json_required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_size_invalid"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        if not isinstance(payload, Mapping):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "json_object_required"})
            return
        try:
            result = operation(payload)
        except Exception:
            # Do not expose local paths, model internals, or stack traces.
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference_failure"})
            return
        status = HTTPStatus.OK if result.get("status") != "ABSTAIN" else HTTPStatus.UNPROCESSABLE_ENTITY
        self._json(status, result)


def main() -> int:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8088"))
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
