from __future__ import annotations

import unittest

from onnx_splitpoint_tool.ranking_methods import (
    METHOD_ORDER,
    compute_ranking_predictions,
    ranking_method_policy,
)


class RankingMethodsV60Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "case_id": "b010",
                "boundary": 10,
                "cut_bytes": 1 * 1024 * 1024,
                "imbalance_val": 0.70,
                "n_cut_tensors": 1,
                "predicted_stage1_ms": 3.0,
                "predicted_stage2_ms": 9.0,
                "strict_ok": True,
            },
            {
                "case_id": "b020",
                "boundary": 20,
                "cut_bytes": 2 * 1024 * 1024,
                "imbalance_val": 0.10,
                "n_cut_tensors": 1,
                "predicted_stage1_ms": 6.0,
                "predicted_stage2_ms": 6.5,
                "strict_ok": True,
            },
            {
                "case_id": "b030",
                "boundary": 30,
                "cut_bytes": 4 * 1024 * 1024,
                "imbalance_val": 0.20,
                "n_cut_tensors": 2,
                "predicted_stage1_ms": 7.5,
                "predicted_stage2_ms": 5.0,
                "strict_ok": True,
            },
            {
                "case_id": "b040",
                "boundary": 40,
                "cut_bytes": 8 * 1024 * 1024,
                "imbalance_val": 0.50,
                "n_cut_tensors": 3,
                "predicted_stage1_ms": 10.0,
                "predicted_stage2_ms": 3.0,
                "strict_ok": True,
            },
        ]
        self.generic_context = [{
            "direction": "hailo8_to_tensorrt",
            "stage1": "hailo8",
            "stage2": "tensorrt",
            "runner_regime": "generic",
        }]
        self.native_context = [{
            "direction": "hailo8_to_tensorrt",
            "stage1": "hailo8",
            "stage2": "tensorrt",
            "runner_regime": "native_fifo",
        }]

    def test_all_four_methods_are_generated_for_generic_context(self) -> None:
        rows = compute_ranking_predictions(self.candidates, self.generic_context, {})
        self.assertEqual(len(rows), len(self.candidates) * len(METHOD_ORDER))
        self.assertEqual({row["method_id"] for row in rows}, set(METHOD_ORDER))

        no_handover = {
            row["case_id"]: float(row["predicted_value"])
            for row in rows
            if row["method_id"] == "cycle_time_no_handover"
        }
        with_handover = {
            row["case_id"]: float(row["predicted_value"])
            for row in rows
            if row["method_id"] == "cycle_time_with_handover"
        }
        self.assertEqual(set(no_handover), set(with_handover))
        self.assertTrue(all(with_handover[key] >= no_handover[key] for key in no_handover))

        cut_rows = [row for row in rows if row["method_id"] == "cut_bytes_only"]
        self.assertEqual(min(cut_rows, key=lambda row: row["predicted_rank"])["case_id"], "b010")

    def test_native_handover_is_not_silently_borrowed_from_generic_model(self) -> None:
        rows = compute_ranking_predictions(self.candidates, self.native_context, {})
        native_with = [row for row in rows if row["method_id"] == "cycle_time_with_handover"]
        self.assertEqual(len(native_with), len(self.candidates))
        self.assertTrue(all(row["prediction_available"] is False for row in native_with))
        self.assertTrue(all(row["prediction_source"].endswith("native_handover_model_unconfigured") for row in native_with))

    def test_configured_native_constant_model_is_applied(self) -> None:
        policy = {
            "cycle_time_with_handover": {
                "handover_models": {
                    "native_fifo": {
                        "hailo8_to_tensorrt": {
                            "mode": "constant",
                            "value_ms": 0.25,
                        }
                    }
                }
            }
        }
        rows = compute_ranking_predictions(self.candidates, self.native_context, policy)
        no_handover = {
            row["case_id"]: float(row["predicted_value"])
            for row in rows
            if row["method_id"] == "cycle_time_no_handover"
        }
        with_handover = {
            row["case_id"]: float(row["predicted_value"])
            for row in rows
            if row["method_id"] == "cycle_time_with_handover"
        }
        for case_id in no_handover:
            self.assertAlmostEqual(with_handover[case_id], no_handover[case_id] + 0.25, places=9)

    def test_explicit_zero_weight_is_preserved(self) -> None:
        policy = ranking_method_policy({
            "weighted_score": {"w_comm": 0.0, "w_imb": 0.0, "w_tensors": 0.0}
        })
        self.assertEqual(policy["weighted_score"]["w_comm"], 0.0)
        self.assertEqual(policy["weighted_score"]["w_imb"], 0.0)
        self.assertEqual(policy["weighted_score"]["w_tensors"], 0.0)


if __name__ == "__main__":
    unittest.main()
