"""Regression checks for the frozen runtime's fail-closed input boundary."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))

from backhaul_runtime import FrozenPolicy, InputError  # noqa: E402


class RuntimeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads((ROOT / "submission" / "demo_input.json").read_text())
        cls.policy = FrozenPolicy(ROOT / "submission" / "artifacts")

    def assert_rejected(self, mutate) -> None:
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        with self.assertRaises(InputError):
            self.policy.infer(payload)

    def test_rejects_non_object_truck(self) -> None:
        self.assert_rejected(lambda payload: payload["trucks"].__setitem__(0, None))

    def test_rejects_null_identifier(self) -> None:
        self.assert_rejected(lambda payload: payload["trucks"][0].__setitem__("truck_id", None))

    def test_rejects_negative_weight_and_telemetry_age(self) -> None:
        self.assert_rejected(lambda payload: payload["orders"][0].__setitem__("weight_kg", -1))
        self.assert_rejected(lambda payload: payload["trucks"][0].__setitem__("telemetry_age_s", -1))

    def test_rejects_string_boolean(self) -> None:
        self.assert_rejected(
            lambda payload: payload["orders"][0].__setitem__("manifest_bound", "false")
        )

    def test_rejects_negative_road_distance(self) -> None:
        self.assert_rejected(
            lambda payload: payload.__setitem__(
                "pair_context",
                [{"truck_id": "TRK-01", "order_id": "ORD-101", "deadhead_km": -1}],
            )
        )

    def test_residual_capacity_is_fail_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["trucks"][0]["cargo_released_before_pickup"] = False
        payload["orders"] = [copy.deepcopy(self.payload["orders"][5])]
        payload["orders"][0]["weight_kg"] = 2900
        result = self.policy.infer(payload)
        reasons = [
            reason
            for pair in result["rejected_pairs"]
            if pair["truck_id"] == "TRK-01"
            for reason in pair["reasons"]
        ]
        self.assertIn("CAPACITY_EXCEEDED", reasons)


if __name__ == "__main__":
    unittest.main()
