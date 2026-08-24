#!/usr/bin/env python3
"""Minimal local HTTP API for the immutable backhaul policy."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backhaul_runtime import FrozenPolicy, InputError


class Handler(BaseHTTPRequestHandler):
    policy: FrozenPolicy
    protocol_version = "HTTP/1.1"

    def _send(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(HTTPStatus.OK, self.policy.health())
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise InputError("request body must be between 1 byte and 2 MB")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise InputError("request body must be a JSON object")
            self._send(HTTPStatus.OK, self.policy.infer(payload))
        except (InputError, json.JSONDecodeError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference failed", "detail": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    policy = FrozenPolicy(args.artifacts)
    if args.input:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        print(json.dumps(policy.infer(payload), indent=2))
        return 0
    Handler.policy = policy
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "ready", "host": args.host, "port": args.port, **policy.health()}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
