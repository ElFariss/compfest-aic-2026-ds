from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.engine import Optimizer, sign_telemetry, utc_now
from app.store import Repository


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        data_dir = Path(self.temporary.name)
        settings = Settings(
            google_maps_key=None,
            iot_shared_secret="test-iot-secret",
            host="127.0.0.1",
            port=0,
            data_dir=data_dir,
            osrm_base_url="https://router.project-osrm.org",
            region_geojson_url="https://example.invalid/regions.geojson",
        )
        self.optimizer = Optimizer(settings, Repository(data_dir / "test.sqlite3"))

    def tearDown(self) -> None:
        self.optimizer.repository.close()
        self.temporary.cleanup()

    def test_matching_includes_feasible_multi_hop(self) -> None:
        plans = self.optimizer.recommendations()
        plan = next(plan for plan in plans if plan["id"] == "REC-TRK-01-101-102")
        self.assertTrue(plan["is_multi_hop"])
        self.assertEqual(plan["order_ids"], ["ORD-101", "ORD-102"])
        self.assertGreater(plan["expected_margin_idr"], 0)
        self.assertLessEqual(plan["capacity_used_kg"], 12000)

    def test_national_demo_fleet_has_300_visible_trucks_without_extra_matches(self) -> None:
        fleet = self.optimizer.fleet_view()
        plans = self.optimizer.recommendations()
        self.assertEqual(len(fleet), 300)
        self.assertEqual(sum(truck["id"].startswith("TRK-NAT-") for truck in fleet), 296)
        self.assertFalse(any(plan["truck_id"].startswith("TRK-NAT-") for plan in plans))

    def test_dispatcher_acceptance_locks_truck_and_orders(self) -> None:
        plan = next(plan for plan in self.optimizer.recommendations() if plan["id"] == "REC-TRK-01-101-102")
        result, error = self.optimizer.decide(plan["id"], "accept", "dispatcher accepted")
        self.assertIsNone(error)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(self.optimizer._truck("TRK-01")["status"], "assigned_backhaul")
        self.assertEqual(self.optimizer._order("ORD-101")["status"], "assigned")
        duplicate, duplicate_error = self.optimizer.decide(plan["id"], "accept")
        self.assertIsNone(duplicate)
        self.assertIn("active backhaul", duplicate_error)

    def test_telemetry_requires_a_valid_signature_and_sequence(self) -> None:
        payload = {
            "truck_id": "TRK-04",
            "timestamp": utc_now(),
            "lat": -6.739,
            "lon": 108.539,
            "speed_kph": 0,
            "heading": 0,
            "gps_accuracy_m": 8,
            "cargo_status": "empty",
            "fuel_pct": 37,
            "sequence": 1,
            "signature": "not-valid",
        }
        response, status = self.optimizer.process_telemetry(payload)
        self.assertEqual(status, 401)
        self.assertFalse(response["accepted"])
        payload["signature"] = sign_telemetry(payload, "test-iot-secret")
        response, status = self.optimizer.process_telemetry(payload)
        self.assertEqual(status, 202)
        self.assertTrue(response["accepted"])
        response, status = self.optimizer.process_telemetry(payload)
        self.assertEqual(status, 409)
        self.assertIn("non-monotonic", response["reason"])


if __name__ == "__main__":
    unittest.main()
