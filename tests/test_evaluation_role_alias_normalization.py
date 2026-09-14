from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.ranking_model_fitting import fit_ranking_models
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


ROLE_ALIASES = ("holdout", "confirmatory_holdout")


def _execution_profile(role: str) -> dict[str, Any]:
    return {
        "model_suite": {
            "primary": [
                {
                    "id": "alias_model",
                    "enabled": True,
                    "evaluation_role": role,
                    "candidate_universe": {
                        "mode": "deterministic_audit",
                        "audit_size": 7,
                    },
                }
            ]
        },
        "selection_policy": {"max_accepted_cases_per_model": 2},
        "run_profiles": [{"id": "hailo8", "enabled": True}],
        "execution_preset": {
            "snapshot": {
                "defaults": {"native_enabled": False, "energy_enabled": False},
                "runtime": {"native": {}, "benchmark": {}},
                "ranking": {
                    "enabled": True,
                    "minimum_candidates_for_correlation": 3,
                },
                "holdout": {"audit_size": 20},
            }
        },
    }


def _runner(root: Path, *, approval_required: bool = False) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.profile_payload = {
        "campaign": {
            "prediction_freeze_enabled": True,
            "require_prediction_freeze_approval": approval_required,
        },
        "ranking_validation": {
            "enabled": True,
            "candidate_universe": "deterministic_audit",
            "audit_size": 7,
        },
    }
    runner.profile_path = ""
    runner.run_dir = root
    runner.options = SimpleNamespace(no_model_hash=False, dry_run=True)
    runner._stop_requested = False
    return runner


def _standalone_reporter_namespace() -> dict[str, Any]:
    template = (
        Path(__file__).parents[1]
        / "onnx_splitpoint_tool"
        / "resources"
        / "templates"
        / "scientific_reporter_v60.py.txt"
    )
    namespace: dict[str, Any] = {"__name__": "role_alias_reporter_test"}
    exec(compile(template.read_text(encoding="utf-8"), str(template), "exec"), namespace)
    return namespace


def _standalone_ranking_result(namespace: dict[str, Any], role: str) -> Any:
    predictions = [
        {
            "case_id": "b001",
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "method_id": "cut_bytes_only",
            "prediction_available": True,
            "predicted_value": 1.0,
            "predicted_rank": 1,
            "prediction_unit": "ms",
        }
    ]
    namespace["_freeze"] = lambda _root, _plan: ({}, predictions, True, {})
    rows = [
        {
            "model_id": "alias_model",
            "evaluation_role": role,
            "case_id": "b001",
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "variant": "split",
            "cycle_ms": 1.0,
            "ranking_eligible": True,
        }
    ]
    plan = {
        "ranking_validation": {
            "methods": ["cut_bytes_only"],
            "k_values": [1],
            "elite_q_values": [1],
            "primary_k": 1,
            "minimum_candidates_for_correlation": 1,
            "require_complete_candidate_universe": False,
            "require_frozen_predictions": False,
        }
    }
    return namespace["_ranking_comparison"](Path(), rows, plan)


class EvaluationRoleAliasNormalizationTests(unittest.TestCase):
    def test_execution_plan_treats_both_holdout_spellings_identically(self) -> None:
        legacy = build_effective_execution_plan(_execution_profile("holdout"))
        canonical = build_effective_execution_plan(
            _execution_profile("confirmatory_holdout")
        )

        self.assertEqual(legacy, canonical)
        self.assertEqual(canonical["holdout_audit_counts"], {"alias_model": 7})
        self.assertEqual(canonical["candidate_counts_by_model"], {"alias_model": 7})
        self.assertFalse(
            any(
                warning.get("id") == "ranking_candidate_shortfall"
                for warning in canonical["warnings"]
            )
        )

    def test_ranking_fitter_excludes_both_holdout_spellings_from_training(self) -> None:
        rows = [
            {"model_id": "legacy", "evaluation_role": "holdout"},
            {
                "model_id": "canonical",
                "evaluation_role": "confirmatory_holdout",
            },
            {"model_id": "development", "evaluation_role": "development"},
        ]

        bundle = fit_ranking_models(rows, minimum_rows=3)

        self.assertEqual(bundle["development_row_count"], 1)
        self.assertEqual(bundle["excluded_holdout_row_count"], 2)
        self.assertEqual(bundle["excluded_non_development_row_count"], 0)
        self.assertEqual(bundle["excluded_non_development_roles"], [])
        self.assertEqual(bundle["development_model_ids"], ["development"])

    def test_runner_applies_holdout_freeze_rejection_to_both_aliases(self) -> None:
        for role in ROLE_ALIASES:
            with self.subTest(role=role), TemporaryDirectory() as temp:
                root = Path(temp)
                model_id = role
                analysis_dir = root / "models" / model_id / "analysis"
                analysis_dir.mkdir(parents=True)
                prediction_path = analysis_dir / "prediction.json"
                prediction_path.write_text("{}\n", encoding="utf-8")
                runner = _runner(root)

                artifacts, metrics = runner._freeze_prediction_artifact(
                    model_id=model_id,
                    row={"evaluation_role": role},
                    prediction={},
                    prediction_path=prediction_path,
                )

                self.assertEqual(
                    metrics["prediction_freeze_status"],
                    "holdout_real_analysis_required",
                )
                self.assertEqual(
                    metrics["evaluation_role"], "confirmatory_holdout"
                )
                self.assertIs(
                    metrics["predictions_frozen_valid_for_holdout"], False
                )
                self.assertTrue(
                    artifacts["prediction_freeze_rejection_json"].is_file()
                )

    def test_runner_requires_prediction_approval_for_both_aliases(self) -> None:
        for role in ROLE_ALIASES:
            with self.subTest(role=role), TemporaryDirectory() as temp:
                root = Path(temp)
                model_id = role
                analysis_dir = root / "models" / model_id / "analysis"
                analysis_dir.mkdir(parents=True)
                (analysis_dir / "prediction.json").write_text(
                    '{"candidates": []}\n', encoding="utf-8"
                )
                runner = _runner(root, approval_required=True)

                artifacts, metrics, _message, status = (
                    runner._stage_select_split_candidates(
                        model_id,
                        {"evaluation_role": role},
                    )
                )

                self.assertEqual(status, "partial")
                self.assertIs(metrics["approval_required"], True)
                self.assertIs(metrics["approval_valid"], False)
                self.assertTrue(
                    artifacts["prediction_freeze_approval_request_md"].is_file()
                )
                self.assertIs(runner._stop_requested, True)

    def test_standalone_reporter_treats_both_holdout_spellings_identically(self) -> None:
        namespace = _standalone_reporter_namespace()

        legacy = _standalone_ranking_result(namespace, "holdout")
        canonical = _standalone_ranking_result(
            namespace, "confirmatory_holdout"
        )

        self.assertEqual(legacy, canonical)
        details, macros, summary = canonical
        self.assertEqual(
            details[0]["evaluation_role"], "confirmatory_holdout"
        )
        self.assertEqual(details[0]["status"], "holdout_validated")
        self.assertEqual(macros[0]["holdout_group_count"], 1)
        self.assertEqual(macros[0]["validated_holdout_group_count"], 1)
        self.assertEqual(summary["status"], "holdout_available")


if __name__ == "__main__":
    unittest.main()
