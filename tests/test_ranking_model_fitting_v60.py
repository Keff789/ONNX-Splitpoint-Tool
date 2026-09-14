from __future__ import annotations

import unittest

from onnx_splitpoint_tool.ranking_methods import compute_ranking_predictions
from onnx_splitpoint_tool.ranking_model_fitting import fit_ranking_models


class RankingModelFittingV60Tests(unittest.TestCase):
    def _rows(self):
        rows = []
        for index in range(1, 5):
            rows.append({
                "model_id": "dev_classifier",
                "evaluation_role": "development",
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "generic",
                "flops_left": index * 1e9,
                "flops_right": (5 - index) * 1e9,
                "stage1_ms": 0.5 + index * 0.7,
                "stage2_ms": 0.3 + (5 - index) * 0.4,
                "handover_ms": 0.2 + index * 0.03,
                "cut_bytes": index * 1024 * 1024,
                "n_cut_tensors": 1 + (index % 2),
                "imbalance_val": abs(index - 2.5) / 2.5,
            })
            rows.append({
                "model_id": "dev_detector",
                "evaluation_role": "development",
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "native_fifo",
                "flops_left": (index + 1) * 1e9,
                "flops_right": (6 - index) * 1e9,
                "stage1_ms": 0.6 + (index + 1) * 0.65,
                "stage2_ms": 0.4 + (6 - index) * 0.38,
                "native_fifo_handoff_ms": 0.08 + index * 0.01,
                "cut_bytes": index * 512 * 1024,
                "n_cut_tensors": 1,
                "imbalance_val": 0.1 * index,
            })
        rows.append({
            "model_id": "unseen_holdout",
            "evaluation_role": "holdout",
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "native_fifo",
            "flops_left": 100e9,
            "flops_right": 100e9,
            "stage1_ms": 999.0,
            "stage2_ms": 999.0,
            "native_fifo_handoff_ms": 999.0,
            "cut_bytes": 999 * 1024 * 1024,
        })
        return rows

    def test_fitter_excludes_holdout_and_fits_native_and_generic_regimes(self) -> None:
        bundle = fit_ranking_models(self._rows(), minimum_rows=3)
        self.assertEqual(bundle["excluded_holdout_row_count"], 1)
        self.assertNotIn("unseen_holdout", bundle["development_model_ids"])
        self.assertGreaterEqual(bundle["fitted_stage_model_count"], 2)
        self.assertEqual(bundle["handover_models"]["native_fifo"]["hailo8_to_tensorrt"]["status"], "fitted")
        self.assertEqual(bundle["handover_models"]["generic"]["hailo8_to_tensorrt"]["status"], "fitted")

    def test_unlabeled_and_screening_rows_are_not_silently_used_for_fitting(self) -> None:
        rows = self._rows()
        rows.extend([
            {
                "model_id": "legacy_unlabeled",
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "native_fifo",
                "flops_left": 999e9,
                "flops_right": 999e9,
                "stage1_ms": 999.0,
                "stage2_ms": 999.0,
                "handover_ms": 999.0,
            },
            {
                "model_id": "screening_only",
                "evaluation_role": "screening",
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "generic",
                "flops_left": 999e9,
                "flops_right": 999e9,
                "stage1_ms": 999.0,
                "stage2_ms": 999.0,
                "handover_ms": 999.0,
            },
        ])
        bundle = fit_ranking_models(rows, minimum_rows=3)
        self.assertEqual(bundle["excluded_non_development_row_count"], 2)
        self.assertEqual(bundle["fit_policy"]["explicit_development_role_required"], True)
        self.assertEqual(bundle["fit_policy"]["unlabeled_rows_treated_as_development"], False)
        self.assertNotIn("legacy_unlabeled", bundle["development_model_ids"])
        self.assertNotIn("screening_only", bundle["development_model_ids"])

        legacy = fit_ranking_models(rows, minimum_rows=3, allow_unlabeled_as_development=True)
        self.assertIn("legacy_unlabeled", legacy["development_model_ids"])
        self.assertTrue(legacy["fit_policy"]["unlabeled_rows_treated_as_development"])

    def test_fitted_profile_patch_drives_cycle_time_predictions(self) -> None:
        bundle = fit_ranking_models(self._rows(), minimum_rows=3)
        policy = bundle["profile_patch"]["ranking_validation"]
        policy.update({
            "methods": ["cycle_time_no_handover", "cycle_time_with_handover"],
            "weighted_score": {"w_comm": 1.0, "w_imb": 3.0, "w_tensors": 0.2},
        })
        candidates = [
            {"case_id": "b010", "boundary": 10, "flops_left": 2e9, "flops_right": 3e9, "cut_bytes": 1024 * 1024, "n_cut_tensors": 1, "imbalance_val": 0.2, "strict_ok": True},
            {"case_id": "b020", "boundary": 20, "flops_left": 3e9, "flops_right": 2e9, "cut_bytes": 2 * 1024 * 1024, "n_cut_tensors": 2, "imbalance_val": 0.2, "strict_ok": True},
        ]
        rows = compute_ranking_predictions(candidates, [{"direction": "hailo8_to_tensorrt", "stage1": "hailo8", "stage2": "tensorrt", "runner_regime": "native_fifo"}], policy)
        no_handover = [row for row in rows if row["method_id"] == "cycle_time_no_handover"]
        with_handover = [row for row in rows if row["method_id"] == "cycle_time_with_handover"]
        self.assertEqual(len(no_handover), 2)
        self.assertEqual(len(with_handover), 2)
        self.assertTrue(all(row["prediction_available"] is True for row in no_handover + with_handover))
        for base, total in zip(sorted(no_handover, key=lambda row: row["case_id"]), sorted(with_handover, key=lambda row: row["case_id"])):
            self.assertGreaterEqual(float(total["predicted_value"]), float(base["predicted_value"]))
            self.assertIn("profile_fitted_stage_time_models", str(total.get("prediction_source")))


if __name__ == "__main__":
    unittest.main()
