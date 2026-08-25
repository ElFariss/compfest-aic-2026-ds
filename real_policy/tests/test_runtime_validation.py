from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path


SUBMISSION = Path(__file__).resolve().parents[1] / "submission"
sys.path.insert(0, str(SUBMISSION))

import policy_runtime as runtime  # noqa: E402


class FakeTensor:
    def __init__(self, value) -> None:
        self.value = value

    def __getitem__(self, key):
        value = self.value
        if isinstance(key, tuple):
            for part in key:
                value = value[part]
        else:
            value = value[key]
        return FakeTensor(value) if isinstance(value, (list, tuple)) else value

    def tolist(self):
        return self.value


class FakeTorch:
    float32 = "float32"
    bool = "bool"

    @staticmethod
    def tensor(value, dtype=None):
        del dtype
        return FakeTensor(value)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class AlwaysStopModel:
    """Minimal tensor-compatible model that requests STOP on every call."""

    def __call__(self, nodes, context, active, feasible):
        del nodes, context
        active_row = active.value[0]
        feasible_row = feasible.value[0]
        logits = [
            10.0 - index if is_active and is_feasible else -10000.0
            for index, (is_active, is_feasible) in enumerate(zip(active_row, feasible_row))
        ]
        eta = [[[0.25, 0.5] for _ in active_row]]
        return FakeTensor([logits]), FakeTensor(eta), FakeTensor([[0.0, 1.0]])


def valid_artifact_manifest(artifact: Path) -> dict:
    return {
        "artifact": {
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "model_config": {
            "node_dim": 16,
            "context_dim": 16,
            "telemetry_dim": 8,
            "dtcargo_dim": 16,
            "vius_dim": 24,
            "health_dim": 340,
            "price_dim": 12,
            "max_candidates": 32,
            "telemetry_steps": 64,
        },
        "contains_synthetic": False,
        "cross_dataset_rows_joined": False,
        "dispatcher_approval_required": True,
        "runtime_auto_update": False,
        "preprocessing": {
            "vius": {
                "columns": [f"f{index}" for index in range(12)],
                "median": [0.0] * 12,
                "iqr": [1.0] * 12,
            },
            "price": {"feature_schema": "nyc-tlc-logfare-12.v1"},
        },
        "output_schema": {
            name: "source-domain" for name in
            ("route_eta", "telemetry", "dtcargo", "vius", "health", "price")
        },
    }


def paired_route_payload() -> dict:
    candidate_common = {
        "cargo_type": "general",
        "required_vehicle_type": "any",
        "zone_id": "z1",
        "service_time_s": 0.0,
        "time_window_start_utc": "2026-08-25T00:00:00Z",
        "time_window_end_utc": "2026-08-25T02:00:00Z",
        "distance_to_depot_km": 1.0,
        "priority": 0.5,
        "hazmat": False,
        "temperature_controlled": False,
        "source_missing_fraction": 0.0,
    }
    return {
        "schema_version": runtime.ROUTE_SCHEMA,
        "request_id": "route-contract-1",
        "vehicle": {
            "vehicle_id": "truck-1",
            "origin_id": "origin",
            "vehicle_type": "box",
            "compatible_cargo_types": ["general"],
            "zone_id": "z1",
            "capacity_kg": 1000.0,
            "capacity_cm3": 10000.0,
            "current_load_kg": 0.0,
            "current_load_cm3": 0.0,
            "current_time_utc": "2026-08-25T00:00:00Z",
            "route_started_time_utc": "2026-08-25T00:00:00Z",
            "route_time_budget_s": 7200.0,
            "telemetry_age_s": 0.0,
            "max_telemetry_age_s": 60.0,
            "fuel_fraction": 0.8,
            "minimum_fuel_fraction": 0.1,
            "gps_accuracy_m": 2.0,
            "max_gps_accuracy_m": 10.0,
            "domain_shift_score": 0.1,
            "max_domain_shift_score": 0.5,
            "max_plausible_speed_kmh": 120.0,
            "source_missing_fraction": 0.0,
            "onboard_shipment_ids": [],
        },
        "candidates": [
            {
                **candidate_common,
                "stop_id": "pickup",
                "stop_type": "pickup",
                "shipment_id": "shipment-1",
                "load_delta_kg": 100.0,
                "load_delta_cm3": 1000.0,
            },
            {
                **candidate_common,
                "stop_id": "delivery",
                "stop_type": "delivery",
                "shipment_id": "shipment-1",
                "load_delta_kg": -100.0,
                "load_delta_cm3": -1000.0,
            },
        ],
        "travel_matrix": {
            "origin": {
                "pickup": {"travel_time_s": 60.0, "distance_km": 1.0, "road_available": True},
                "delivery": {"travel_time_s": 120.0, "distance_km": 2.0, "road_available": True},
            },
            "pickup": {
                "delivery": {"travel_time_s": 60.0, "distance_km": 1.0, "road_available": True},
            },
            "delivery": {
                "pickup": {"travel_time_s": 60.0, "distance_km": 1.0, "road_available": True},
            },
        },
    }


class RuntimeValidationTests(unittest.TestCase):
    def test_compose_health_contract_uses_ready_status(self) -> None:
        policy = object.__new__(runtime.FrozenPolicy)
        policy.contract = type(
            "Contract",
            (),
            {
                "artifact_sha256": "0" * 64,
                "model_config": {"telemetry_steps": 64},
            },
        )()
        policy.price_feature_schema = "nyc-tlc-logfare-12.v1"
        health = policy.health()
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["schemas"]["vius_features"][1], "GVWR_CLASS")

    def test_stale_or_missing_inputs_abstain_instead_of_imputation(self) -> None:
        issue = runtime.InputIssue("MISSING_REQUIRED_MODALITY", "cargo_weight_kg", "required")
        response = runtime.abstain("req-real-1", issue)
        self.assertEqual(response["status"], "ABSTAIN")
        self.assertEqual(response["field"], "cargo_weight_kg")
        self.assertTrue(response["dispatcher_approval_required"])
        self.assertFalse(response["model_updated"])

    def test_scania_missing_bit_requires_zero_token(self) -> None:
        payload = {
            "schema_version": runtime.HEALTH_SCHEMA,
            "feature_schema": "scania-aps-170-plus-missing-mask.v1",
            "normalized_sensor_values": [0.0] * 169 + [1.0],
            "missing_mask": [False] * 169 + [True],
        }
        with self.assertRaises(runtime.InputIssue):
            runtime.health_features(payload)

    def test_vius_transform_uses_frozen_training_statistics_and_masks(self) -> None:
        columns = [f"f{index}" for index in range(12)]
        normalization = {"columns": columns, "median": [10.0] * 12, "iqr": [2.0] * 12}
        payload = {
            "schema_version": runtime.DEADHEAD_SCHEMA,
            "feature_schema": "vius-2021-12-values-plus-mask.v1",
            "raw_values": {**{name: 12.0 for name in columns}, "f5": None},
        }
        features = runtime.deadhead_features(payload, normalization)
        self.assertEqual(len(features), 24)
        self.assertEqual(features[0], 1.0)
        self.assertEqual(features[5], 0.0)
        self.assertEqual(features[12 + 5], 1.0)

    def test_truck_track_is_a_separate_source_schema_with_missing_bits(self) -> None:
        payload = {
            "schema_version": runtime.TRACK_SCHEMA,
            "track": {
                "start_time_utc": "2026-08-25T00:00:00Z",
                "distance_m": 12500.0,
                "track_gap_m": 20.0,
                "avg_speed_m_s": 18.0,
                "max_speed_m_s": 29.0,
                "avg_hdop": None,
                "gvwr_kg": 18000.0,
                "gcwr_kg": None,
                "axle_class": 42.0,
            },
        }
        features = runtime.track_features(payload)
        self.assertEqual(len(features), 16)
        self.assertEqual(features[4], 0.0)
        self.assertEqual(features[8 + 4], 1.0)
        self.assertEqual(features[8 + 6], 1.0)

    def test_singapore_telemetry_does_not_accept_cross_source_track_fields(self) -> None:
        payload = {
            "schema_version": runtime.TELEMETRY_SCHEMA,
            "samples": [{
                "timestamp_utc": "2026-08-25T00:00:00Z",
                "speed_kmh": 30.0,
                "engine_status": 1.0,
                "road_grade": 0.0,
                "engine_load_pct": 40.0,
                "mass_air_flow_g_s": 12.0,
                "longitudinal_acceleration_m_s2": 0.2,
                "delta_t_s": 0.0,
                "observed_fraction": 1.0,
            }],
            "truck_track": {"distance_m": 999.0},
        }
        sequence, mask = runtime.telemetry_features(payload, telemetry_steps=4)
        self.assertEqual(len(sequence[0]), 8)
        self.assertEqual(mask, [True, False, False, False])
        second = {
            **payload["samples"][0],
            "timestamp_utc": "2026-08-25T00:00:10Z",
            "delta_t_s": 2.0,
        }
        with self.assertRaisesRegex(runtime.InputIssue, "must match"):
            runtime.telemetry_features(
                {**payload, "samples": [payload["samples"][0], second]},
                telemetry_steps=4,
            )
        incomplete = {**payload, "samples": [{**payload["samples"][0], "observed_fraction": 0.5}]}
        with self.assertRaisesRegex(runtime.InputIssue, "must equal 1"):
            runtime.telemetry_features(incomplete, telemetry_steps=4)

    def test_price_rejects_extreme_out_of_domain_features(self) -> None:
        payload = {
            "schema_version": runtime.PRICE_SCHEMA,
            "feature_schema": "nyc-tlc-logfare-12.v1",
            "normalized_source_features": [0.0] * 11 + [21.0],
        }
        with self.assertRaises(runtime.InputIssue):
            runtime.price_features(payload, "nyc-tlc-logfare-12.v1")

    def test_route_rejects_unbalanced_pair_and_unsafe_domain_shift(self) -> None:
        unbalanced = paired_route_payload()
        unbalanced["candidates"][1]["load_delta_kg"] = -90.0
        with self.assertRaisesRegex(runtime.InputIssue, "must balance"):
            runtime.RouteRequest(unbalanced)

        shifted = paired_route_payload()
        shifted["vehicle"]["domain_shift_score"] = 0.8
        with self.assertRaises(runtime.InputIssue) as raised:
            runtime.RouteRequest(shifted)
        self.assertEqual(raised.exception.code, "DOMAIN_SHIFT_UNSAFE")

    def test_route_masks_unverified_hazmat_and_implausible_edges(self) -> None:
        hazmat = paired_route_payload()
        hazmat["candidates"][0]["hazmat"] = True
        request = runtime.RouteRequest(hazmat)
        reasons, *_ = request.feasibility(
            request.candidates[0],
            request.vehicle.origin_id,
            request.vehicle.current_time,
            request.vehicle.load_kg,
            request.vehicle.load_cm3,
            set(),
            set(),
        )
        self.assertIn("HAZMAT_COMPLIANCE_UNVERIFIED", reasons)

        impossible = paired_route_payload()
        impossible["travel_matrix"]["origin"]["pickup"]["travel_time_s"] = 1.0
        with self.assertRaises(runtime.InputIssue) as raised:
            runtime.RouteRequest(impossible)
        self.assertEqual(raised.exception.code, "IMPLAUSIBLE_TRAVEL_EDGE")

    def test_route_time_overflow_becomes_an_infeasible_reason(self) -> None:
        payload = paired_route_payload()
        payload["travel_matrix"]["origin"]["pickup"]["travel_time_s"] = 1.0e300
        request = runtime.RouteRequest(payload)
        reasons, *_ = request.feasibility(
            request.candidates[0],
            request.vehicle.origin_id,
            request.vehicle.current_time,
            request.vehicle.load_kg,
            request.vehicle.load_cm3,
            set(),
            set(),
        )
        self.assertEqual(reasons, ["TIME_VALUE_OUT_OF_RANGE"])

        policy = object.__new__(runtime.FrozenPolicy)
        policy.torch = None
        response = policy.infer_route(payload)
        self.assertEqual(response["status"], "ABSTAIN")
        self.assertIn(
            "TIME_VALUE_OUT_OF_RANGE",
            response["candidate_rejections"]["pickup"],
        )

    def test_stop_prediction_is_applied_only_after_the_selected_action(self) -> None:
        policy = object.__new__(runtime.FrozenPolicy)
        policy.torch = FakeTorch()
        policy.model = AlwaysStopModel()
        policy.lock = threading.Lock()

        response = policy.infer_route(paired_route_payload())

        self.assertEqual(response["status"], "RECOMMENDATION")
        self.assertEqual(
            [stop["stop_id"] for stop in response["route"]],
            ["pickup", "delivery"],
        )
        self.assertEqual(response["termination"], "MODEL_STOP_AFTER_FINAL_CANDIDATE")

    def test_artifact_contract_rejects_invalid_vius_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.pt"
            artifact.write_bytes(b"checksum-bound-placeholder")
            valid = valid_artifact_manifest(artifact)
            (root / "manifest.json").write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(runtime.validate_artifacts(root).artifact_path, artifact)

            invalid_cases = {
                "duplicate columns": {
                    "columns": ["duplicate"] * 12,
                    "median": [0.0] * 12,
                    "iqr": [1.0] * 12,
                },
                "non-finite median": {
                    "columns": [f"f{index}" for index in range(12)],
                    "median": [0.0] * 11 + [float("nan")],
                    "iqr": [1.0] * 12,
                },
                "zero IQR": {
                    "columns": [f"f{index}" for index in range(12)],
                    "median": [0.0] * 12,
                    "iqr": [1.0] * 11 + [0.0],
                },
            }
            for label, normalization in invalid_cases.items():
                with self.subTest(label=label):
                    manifest = copy.deepcopy(valid)
                    manifest["preprocessing"]["vius"] = normalization
                    (root / "manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(RuntimeError, "VIUS normalization"):
                        runtime.validate_artifacts(root)

    def test_artifact_contract_rejects_a_synthetic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.pt"
            artifact.write_bytes(b"not-a-real-model-but-checksum-valid-for-contract-test")
            config = {
                "node_dim": 16, "context_dim": 16, "telemetry_dim": 8,
                "dtcargo_dim": 16, "vius_dim": 24, "health_dim": 340,
                "price_dim": 12, "max_candidates": 32, "telemetry_steps": 64,
            }
            manifest = {
                "artifact": {
                    "path": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
                "model_config": config,
                "contains_synthetic": True,
                "cross_dataset_rows_joined": False,
                "dispatcher_approval_required": True,
                "runtime_auto_update": False,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "contains_synthetic=false"):
                runtime.validate_artifacts(root)

    def test_submission_has_no_training_or_update_primitive(self) -> None:
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in SUBMISSION.glob("*.py")
        )
        self.assertNotIn("torch.optim", text)
        self.assertNotIn(".backward(", text)
        self.assertNotIn("training.train_real", text)
        self.assertNotIn("/update", text)


if __name__ == "__main__":
    unittest.main()
