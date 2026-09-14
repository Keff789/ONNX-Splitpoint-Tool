from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onnx_splitpoint_tool.workflow.generator_binding import (
    _copy_scientific_freeze_artifacts,
    materialize_suite_from_candidate_plan,
)


class GeneratorReportingBindingV60Tests(unittest.TestCase):
    def test_prediction_and_four_method_freeze_are_copied_into_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            analysis = model_dir / "analysis"
            suite = root / "suite"
            analysis.mkdir(parents=True)
            names = (
                "prediction.json",
                "predictions_frozen.csv",
                "holdout_predictions_frozen.csv",
                "ranking_predictions_frozen.csv",
                "holdout_ranking_predictions_frozen.csv",
                "prediction_freeze_manifest.json",
            )
            for name in names:
                (analysis / name).write_text("{}" if name.endswith(".json") else "x\n", encoding="utf-8")

            copied = _copy_scientific_freeze_artifacts(model_dir, suite)
            self.assertEqual(set(copied), set(names))
            for name in names:
                self.assertTrue((suite / name).is_file(), name)

    def test_generated_plan_preserves_quality_ranking_and_holdout_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            model_dir = run_dir / "models" / "holdout_model"
            analysis = model_dir / "analysis"
            analysis.mkdir(parents=True)
            for name, content in {
                "prediction.json": '{"model_id":"holdout_model","candidates":[]}',
                "predictions_frozen.csv": "case_id\n",
                "ranking_predictions_frozen.csv": "case_id,method_id\n",
                "prediction_freeze_manifest.json": '{"prospective":true,"freeze_status":"prospective_frozen"}',
            }.items():
                (analysis / name).write_text(content, encoding="utf-8")

            quality_gate = {
                "profile_id": "task-quality-v1",
                "dataset_tier": "final",
                "frozen_before_final_campaign": True,
            }
            ranking_validation = {
                "enabled": True,
                "methods": [
                    "cut_bytes_only",
                    "weighted_score",
                    "cycle_time_no_handover",
                    "cycle_time_with_handover",
                ],
                "require_frozen_predictions": True,
                "require_complete_candidate_universe": True,
            }
            result = materialize_suite_from_candidate_plan(
                model_id="holdout_model",
                model_path=str(run_dir / "missing.onnx"),
                model_dir=model_dir,
                run_dir=run_dir,
                profile_id="profile-v60",
                run_id="run-v60",
                prediction={"artifact_id": "prediction-v60"},
                candidate_plan={
                    "artifact_id": "candidate-plan-v60",
                    "selected_candidates": [{"case_id": "b001", "boundary": 1}],
                },
                targets=["hailo8_to_tensorrt"],
                full_baseline_plan={},
                output_contracts={},
                profile_payload={
                    "quality_gate": quality_gate,
                    "ranking_validation": ranking_validation,
                },
                model_entry={
                    "id": "holdout_model",
                    "evaluation_role": "holdout",
                    "validation_tier": "final",
                    "candidate_universe_complete": True,
                },
                dry_run=True,
            )
            plan_path = result.suite_dir / "benchmark_plan.json"  # type: ignore[operator]
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["quality_gate"], quality_gate)
            self.assertEqual(plan["ranking_validation"], ranking_validation)
            self.assertEqual(plan["model_suite"]["primary"][0]["evaluation_role"], "holdout")
            self.assertTrue(plan["model_suite"]["primary"][0]["candidate_universe_complete"])
            self.assertIn("ranking_predictions_frozen.csv", plan["prediction_freeze_artifacts"])
            self.assertTrue((result.suite_dir / "scientific_reporter_v60.py").is_file())  # type: ignore[operator]


if __name__ == "__main__":
    unittest.main()
