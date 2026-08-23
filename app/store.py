from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class Repository:
    """Small persistent audit log; operational scenario state remains explicit in memory."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    truck_id TEXT,
                    accepted INTEGER NOT NULL,
                    reason TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dispatcher_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    recommendation_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    note TEXT
                );
                """
            )

    def log_telemetry(self, truck_id: str | None, accepted: bool, reason: str, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO telemetry_events (truck_id, accepted, reason, payload_json) VALUES (?, ?, ?, ?)",
                (truck_id, int(accepted), reason, json.dumps(payload, separators=(",", ":"))),
            )

    def log_decision(self, recommendation_id: str, action: str, note: str | None) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO dispatcher_decisions (recommendation_id, action, note) VALUES (?, ?, ?)",
                (recommendation_id, action, note),
            )

    def metrics(self) -> dict[str, int]:
        with self._lock:
            events = self._connection.execute("SELECT COUNT(*) AS total, SUM(accepted) AS accepted FROM telemetry_events").fetchone()
            decisions = self._connection.execute("SELECT COUNT(*) AS total FROM dispatcher_decisions").fetchone()
        return {
            "telemetry_total": int(events["total"] or 0),
            "telemetry_accepted": int(events["accepted"] or 0),
            "dispatcher_decisions": int(decisions["total"] or 0),
        }

    def accepted_telemetry(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return recent accepted GPS payloads for the regional activity overlay."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM telemetry_events WHERE accepted = 1 ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]
