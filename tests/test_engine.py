from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from app.config import Settings
from app.engine import Optimizer, sign_telemetry, synthetic_gateway_snapshot, utc_now
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

    def test_telemetry_preserves_gateway_sensors_and_flags_unsafe_readings(self) -> None:
        payload = {
            "device_id": "gw-cirebon-04",
            "truck_id": "TRK-04",
            "timestamp": utc_now(),
            "lat": -6.739,
            "lon": 108.539,
            "speed_kph": 28,
            "heading": 91,
            "gps_accuracy_m": 7,
            "cargo_status": "loaded",
            "fuel_pct": 37,
            "cargo_weight_kg": 6800,
            "can": {"coolant_temp_c": 114},
            "imu": {"accel_x_g": 2.9},
            "health": {"signal_dbm": -116, "power_v": 10.2, "uptime_s": 999},
            "sequence": 1,
        }
        payload["signature"] = sign_telemetry(payload, "test-iot-secret")
        response, status = self.optimizer.process_telemetry(payload)
        truck = self.optimizer._truck("TRK-04")
        self.assertEqual(status, 202)
        self.assertTrue(response["accepted"])
        self.assertEqual(truck["device_id"], "gw-cirebon-04")
        self.assertEqual(truck["cargo_weight_kg"], 6800.0)
        self.assertEqual(truck["can"]["coolant_temp_c"], 114.0)
        self.assertEqual(response["anomaly"]["status"], "review")
        self.assertIn("cargo sensor reading exceeds vehicle capacity", response["anomaly"]["signals"])

    def test_cached_weather_context_adjusts_eta_without_becoming_telemetry(self) -> None:
        context_dir = self.optimizer.settings.data_dir / "context"
        context_dir.mkdir()
        (context_dir / "weather-indonesia.json").write_text(
            json.dumps(
                {
                    "retrieved_at": "2026-08-24T00:00:00Z",
                    "locations": [
                        {
                            "name": "Jakarta",
                            "latitude": -6.2,
                            "longitude": 106.82,
                            "current": {"rain": 5, "precipitation": 0, "wind_gusts_10m": 20, "temperature_2m": 29},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        eta = self.optimizer.eta(self.optimizer._truck("TRK-01"))
        self.assertEqual(eta["p50_min"], 65)
        self.assertEqual(eta["weather_context"]["source"], "cached public weather context")

    def test_synthetic_gateway_snapshot_is_labelled_and_within_sensor_ranges(self) -> None:
        truck = self.optimizer._truck("TRK-01")
        snapshot = synthetic_gateway_snapshot(truck, 3)
        self.assertEqual(snapshot["device_id"], "sim-gw-trk-01")
        self.assertLessEqual(snapshot["cargo_weight_kg"], truck["capacity_kg"])
        self.assertGreater(snapshot["health"]["power_v"], 10)
        self.assertGreater(snapshot["can"]["coolant_temp_c"], 70)
        self.assertEqual(synthetic_gateway_snapshot(self.optimizer._truck("TRK-04"), 4)["cargo_weight_kg"], 0)

    def test_simulation_tick_submits_sensor_payloads_for_every_truck(self) -> None:
        results = self.optimizer.simulator_tick()
        truck = self.optimizer._truck("TRK-01")
        self.assertEqual(len(results), 300)
        self.assertTrue(all(result["accepted"] for result in results))
        self.assertEqual(truck["telemetry_source"], "synthetic_digital_twin_simulator")
        self.assertIn("coolant_temp_c", truck["can"])
        self.assertIn("accel_x_g", truck["imu"])
        self.assertIn("signal_dbm", truck["device_health"])


if __name__ == "__main__":
    unittest.main()
