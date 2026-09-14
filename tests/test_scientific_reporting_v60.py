from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from onnx_splitpoint_tool.ranking_methods import METHOD_ORDER, compute_ranking_predictions
from onnx_splitpoint_tool.campaign import create_candidate_universe_manifest
from onnx_splitpoint_tool.workflow.artifacts import sha256_file, write_csv, write_json
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _load_suite_prediction,
    _ranking_method_comparison,
    build_benchmarkset_scientific_report,
)


class ScientificReportingV60Tests(unittest.TestCase):
    def _candidate_rows(self):
        return [
            {"case_id": "b001", "boundary": 1, "cut_bytes": 1_000_000, "imbalance_val": 0.8, "n_cut_tensors": 1, "predicted_stage1_ms": 2.0, "predicted_stage2_ms": 8.0, "strict_ok": True},
            {"case_id": "b002", "boundary": 2, "cut_bytes": 2_000_000, "imbalance_val": 0.1, "n_cut_tensors": 1, "predicted_stage1_ms": 5.0, "predicted_stage2_ms": 5.5, "strict_ok": True},
            {"case_id": "b003", "boundary": 3, "cut_bytes": 3_000_000, "imbalance_val": 0.2, "n_cut_tensors": 2, "predicted_stage1_ms": 6.0, "predicted_stage2_ms": 5.0, "strict_ok": True},
            {"case_id": "b004", "boundary": 4, "cut_bytes": 6_000_000, "imbalance_val": 0.6, "n_cut_tensors": 3, "predicted_stage1_ms": 8.0, "predicted_stage2_ms": 3.0, "strict_ok": True},
        ]

    def _plan(self):
        return {
            "model_id": "holdout_model",
            "evaluation_profile": {"profile_id": "synthetic-v60-holdout"},
            "quality_gate": {
                "profile_id": "quality-v1",
                "frozen_before_final_campaign": True,
                "dataset_tier": "final",
                "classification": {
                    "primary_metric": "top1_accuracy",
                    "non_inferiority_margin": 0.01,
                    "guardrails": {"top5_accuracy_margin": 0.01},
                },
                "statistics": {
                    "confidence_level": 0.95,
                    "bootstrap_repetitions": 2000,
                    "seed": 7,
                    "decision": "lower_one_sided_bound",
                },
            },
            "ranking_validation": {
                "enabled": True,
                "require_frozen_predictions": True,
                "require_complete_candidate_universe": True,
                "candidate_universe_complete": True,
                "methods": [
                    "cut_bytes_only",
                    "weighted_score",
                    "cycle_time_no_handover",
                    "cycle_time_with_handover",
                    "onnx_real_boundary_hardware_aware",
                ],
                "k_values": [1, 3, 5],
                "elite_q_values": [1, 3],
                "primary_k": 5,
                "minimum_candidates_for_correlation": 3,
                "near_optimal_relative_epsilon": 0.01,
                "cycle_time_with_handover": {
                    "handover_models": {
                        "generic": {
                            "hailo8_to_tensorrt": {"mode": "constant", "value_ms": 0.2}
                        }
                    }
                },
            },
            "model_suite": {
                "primary": [{
                    "id": "holdout_model",
                    "task": "classification",
                    "evaluation_role": "holdout",
                    "validation_tier": "final",
                    "candidate_universe_complete": True,
                }],
                "reserve": [],
            },
        }

    def _runtime_rows(self):
        measured = {
            "b001": 8.4,
            "b002": 5.8,
            "b003": 6.4,
            "b004": 8.7,
        }
        rows = []
        for case_id, cycle_ms in measured.items():
            rows.append({
                "model_id": "holdout_model",
                "task": "classification",
                "evaluation_role": "holdout",
                "case_id": case_id,
                "run_id": "hailo8_to_trt",
                "backend": "hailo8_to_tensorrt",
                "variant": "composed",
                "buildable": True,
                "runtime_executable": True,
                "contract_consistent": True,
                "interface_contract_pass": True,
                "pipeline_cycle_selected_ms": cycle_ms,
                "throughput_primary_fps": 1000.0 / cycle_ms,
                "task_quality_gate": {
                    "status": "pass",
                    "decision": "pass",
                    "tier": "final",
                    "primary": {
                        "metric": "top1_accuracy",
                        "candidate": 0.799,
                        "reference": 0.800,
                        "delta": -0.001,
                        "ci_low": -0.004,
                        "ci_high": 0.002,
                        "margin": 0.01,
                        "n": 1000,
                    },
                },
            })
        return rows

    def _write_freeze(self, root: Path, plan) -> None:
        candidates = self._candidate_rows()
        prediction = {
            "schema": "onnx-splitpoint/split-prediction",
            "schema_version": 1,
            "artifact_id": "synthetic-prediction",
            "model_id": "holdout_model",
            "candidates": candidates,
        }
        prediction_path = write_json(root / "prediction.json", prediction)
        base_rows = [
            {
                "model_id": "holdout_model",
                "case_id": item["case_id"],
                "boundary": item["boundary"],
            }
            for item in candidates
        ]
        base_csv = write_csv(root / "predictions_frozen.csv", base_rows)
        ranking_rows = compute_ranking_predictions(
            candidates,
            [{
                "direction": "hailo8_to_tensorrt",
                "stage1": "hailo8",
                "stage2": "tensorrt",
                "runner_regime": "generic",
            }],
            plan["ranking_validation"],
        )
        for row in ranking_rows:
            row.update({
                "model_id": "holdout_model",
                "evaluation_role": "holdout",
                "candidate_universe_complete": True,
            })
        ranking_csv = write_csv(root / "ranking_predictions_frozen.csv", ranking_rows)
        universe_path, universe_csv, universe = create_candidate_universe_manifest(
            model_id="holdout_model",
            candidates=candidates,
            mode="all_feasible",
            output_dir=root,
            minimum_valid_candidates=int(plan["ranking_validation"]["minimum_candidates_for_correlation"]),
            source_prediction_sha256=sha256_file(prediction_path) or "",
        )
        write_json(root / "prediction_freeze_manifest.json", {
            "schema": "onnx-splitpoint/prediction-freeze-manifest",
            "schema_version": 1,
            "model_id": "holdout_model",
            "evaluation_role": "holdout",
            "freeze_status": "prospective_frozen",
            "prospective": True,
            "valid_for_holdout": True,
            "prediction_json": prediction_path.name,
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_csv": base_csv.name,
            "prediction_csv_sha256": sha256_file(base_csv),
            "ranking_prediction_csv": ranking_csv.name,
            "ranking_prediction_csv_sha256": sha256_file(ranking_csv),
            "candidate_universe_manifest": universe_path.name,
            "candidate_universe_sha256": universe["universe_sha256"],
            "candidate_universe_csv": universe_csv.name,
            "candidate_universe_csv_sha256": sha256_file(universe_csv),
            "candidate_count": len(candidates),
            "candidate_universe_complete": True,
        })

    def test_complete_holdout_generates_five_method_thesis_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self._plan()
            self._write_freeze(root, plan)
            # Prove regeneration removes the superseded report family.
            old_report = root / "scientific_report"
            old_report.mkdir(parents=True)
            (old_report / "ranking_validation.csv").write_text("obsolete\n", encoding="utf-8")
            (old_report / "thesis_tables").mkdir(parents=True)
            (old_report / "thesis_tables" / "ranking_validation.tex").write_text("obsolete\n", encoding="utf-8")

            result = build_benchmarkset_scientific_report(
                root,
                self._runtime_rows(),
                plan=plan,
                tool_version="0.14.0+v60.rankingmethods",
            )
            self.assertEqual(result["status"], "ok")
            report = root / "scientific_report"
            self.assertTrue((report / "ranking_method_comparison.csv").is_file())
            self.assertTrue((report / "ranking_method_macro.csv").is_file())
            self.assertTrue(
                (report / "ranking_method_cohort_sensitivity.csv").is_file()
            )
            self.assertTrue(
                (report / "ranking_method_cohort_sensitivity.json").is_file()
            )
            self.assertTrue((
                report / "thesis_tables"
                / "ranking_method_cohort_sensitivity.tex"
            ).is_file())
            self.assertTrue((report / "thesis_tables" / "ranking_method_comparison.tex").is_file())
            self.assertFalse((report / "ranking_validation.csv").exists())
            self.assertFalse((report / "thesis_tables" / "ranking_validation.tex").exists())

            with (report / "ranking_method_macro.csv").open(newline="", encoding="utf-8") as handle:
                macros = list(csv.DictReader(handle))
            self.assertEqual({row["method_id"] for row in macros}, {
                "cut_bytes_only",
                "weighted_score",
                "cycle_time_no_handover",
                "cycle_time_with_handover",
                "onnx_real_boundary_hardware_aware",
            })
            self.assertEqual(len(macros), len(METHOD_ORDER))
            self.assertTrue(all(row["status"] == "holdout_available" for row in macros))

            with (report / "ranking_method_comparison.csv").open(newline="", encoding="utf-8") as handle:
                details = list(csv.DictReader(handle))
            self.assertEqual(len(details), len(METHOD_ORDER))
            self.assertTrue(all(row["status"] == "holdout_validated" for row in details))
            by_method = {row["method_id"]: row for row in details}
            self.assertEqual(by_method["cut_bytes_only"]["mae_ms"], "")
            self.assertEqual(by_method["weighted_score"]["mape_percent"], "")
            self.assertNotEqual(by_method["cycle_time_no_handover"]["mae_ms"], "")
            self.assertNotEqual(by_method["cycle_time_with_handover"]["mape_percent"], "")

            cohort_rows = json.loads(
                (report / "ranking_method_cohort_sensitivity.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(cohort_rows), len(METHOD_ORDER) * 3)
            self.assertEqual({row["cohort"] for row in cohort_rows}, {
                "technical",
                "quality_pass",
                "quality_pass_or_inconclusive",
            })
            self.assertEqual(
                {row["method_id"] for row in cohort_rows}, set(METHOD_ORDER)
            )
            self.assertTrue(all(
                row["correlation_claim_eligible"] is False
                for row in cohort_rows
            ))

            payload = json.loads((report / "scientific_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["ranking"]["status"], "holdout_available")
            self.assertEqual(payload["summary"]["ranking"]["method_count"], len(METHOD_ORDER))

    def test_holdout_audit_metrics_exclude_deployment_only_measurements(self) -> None:
        candidates = [
            {
                "case_id": f"b{index:03d}",
                "boundary": index,
                "cut_bytes": index * 1000,
                "imbalance_val": (index % 7) / 7.0,
                "n_cut_tensors": 1 + index % 3,
                "predicted_stage1_ms": 1.0 + index / 10.0,
                "predicted_stage2_ms": 5.0 - index / 100.0,
                "strict_ok": True,
            }
            for index in range(1, 31)
        ]
        method_rows = compute_ranking_predictions(
            candidates,
            [{
                "direction": "hailo8_to_tensorrt",
                "stage1": "hailo8",
                "stage2": "tensorrt",
                "runner_regime": "generic",
            }],
            {},
        )
        for method_row in method_rows:
            method_row.update({"model_id": "holdout_model", "evaluation_role": "holdout"})
        audit_ids = [f"b{index:03d}" for index in range(1, 21)]
        predictions = {
            "holdout_model": {
                "candidates": candidates,
                "_ranking_method_predictions": method_rows,
                "_prediction_freeze": {
                    "valid": True,
                    "ranking_predictions_valid": True,
                    "status": "prospective_frozen",
                    "candidate_universe_valid": True,
                    "candidate_universe_scope": "predeclared_audit_universe",
                    "candidate_universe_selected_case_ids": audit_ids,
                    "candidate_universe_minimum_valid_candidates": 10,
                },
            }
        }
        profile = {
            "ranking_validation": {"require_complete_candidate_universe": True},
            "model_suite": {
                "primary": [{
                    "id": "holdout_model",
                    "evaluation_role": "holdout",
                    "candidate_universe_complete": True,
                }],
                "reserve": [],
            },
        }
        policy = {
            "methods": ["cut_bytes_only"],
            "k_values": [1, 3, 5],
            "elite_q_values": [1, 3],
            "primary_k": 5,
            "minimum_candidates_for_correlation": 3,
            "require_complete_candidate_universe": True,
            "require_frozen_predictions": True,
            "near_optimal_relative_epsilon": 0.01,
            "method_policy": {},
        }

        def measured(deployment_cycle: float, *, omit: str = "") -> list[dict]:
            rows = []
            for index in range(1, 21):
                case_id = f"b{index:03d}"
                if case_id == omit:
                    continue
                rows.append({
                    "model_id": "holdout_model",
                    "evaluation_role": "holdout",
                    "case_id": case_id,
                    "backend": "hailo8_to_tensorrt",
                    "run_id": "hailo8_to_trt",
                    "runner_regime": "generic",
                    "variant": "composed",
                    "pipeline_cycle_selected_ms": 5.0 + index / 10.0,
                    "ranking_eligible": True,
                    "candidate_execution_roles": ["audit"],
                })
            rows.append({
                "model_id": "holdout_model",
                "evaluation_role": "holdout",
                "case_id": "b030",
                "backend": "hailo8_to_tensorrt",
                "run_id": "hailo8_to_trt",
                "runner_regime": "generic",
                "variant": "composed",
                "pipeline_cycle_selected_ms": deployment_cycle,
                "ranking_eligible": True,
                "candidate_execution_roles": ["deployment_shortlist"],
            })
            return rows

        first, _macro, _summary = _ranking_method_comparison(measured(0.001), predictions, profile, policy)
        second, _macro, _summary = _ranking_method_comparison(measured(999.0), predictions, profile, policy)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["expected_candidate_count"], 20)
        self.assertEqual(first[0]["paired_candidate_count"], 20)
        self.assertEqual(first[0]["deployment_only_measurement_count_excluded"], 1)
        self.assertEqual(first[0]["status"], "holdout_validated")
        for field in ("best_measured_case", "spearman_rho", "kendall_tau_b", "hit_at_1", "regret_at_5"):
            self.assertEqual(first[0][field], second[0][field])

        undeclared_profile = json.loads(json.dumps(profile))
        undeclared_profile["model_suite"]["primary"][0][
            "candidate_universe_complete"
        ] = False
        _details, _macro, undeclared_summary = _ranking_method_comparison(
            measured(0.001), predictions, undeclared_profile, policy
        )
        undeclared_technical = next(
            row
            for row in undeclared_summary["cohort_sensitivity_rows"]
            if row["cohort"] == "technical"
        )
        self.assertFalse(
            undeclared_technical["declared_universe_complete_for_cohort"]
        )
        self.assertIsNone(undeclared_technical["global_hit_at_1"])
        self.assertEqual(
            undeclared_technical["global_top1_status"],
            "unavailable_incomplete_declared_universe",
        )

        incomplete, _macro, _summary = _ranking_method_comparison(
            measured(0.001, omit="b020"), predictions, profile, policy
        )
        self.assertFalse(incomplete[0]["candidate_universe_complete"])
        self.assertEqual(incomplete[0]["expected_candidate_count"], 20)
        self.assertEqual(incomplete[0]["status"], "candidate_universe_not_auditable")

        crashed_rows = measured(0.001)
        crashed = next(
            row for row in crashed_rows if row["case_id"] == "b013"
        )
        crashed.update({
            "measurement_valid": False,
            "terminal_failure": True,
            "runtime_executable": False,
            "ranking_eligible": False,
            "row_status": "terminal_process_failure",
        })
        crashed_details, _macro, crashed_summary = (
            _ranking_method_comparison(
                crashed_rows, predictions, profile, policy
            )
        )
        self.assertFalse(
            crashed_details[0]["candidate_measurement_coverage_complete"]
        )
        self.assertEqual(
            crashed_details[0]["measured_candidate_count"], 20
        )
        self.assertEqual(
            crashed_details[0]["diagnostic_paired_candidate_count"], 19
        )
        sensitivity = crashed_summary["cohort_sensitivity_rows"]
        technical = next(
            row for row in sensitivity if row["cohort"] == "technical"
        )
        self.assertEqual(technical["paired_candidate_count"], 19)
        self.assertFalse(technical["declared_universe_complete_for_cohort"])
        self.assertIsNone(technical["global_hit_at_1"])
        self.assertEqual(
            technical["global_top1_status"],
            "unavailable_incomplete_declared_universe",
        )

    def test_benchmarkset_loader_verifies_universe_self_hash_and_audit_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = [
                {"case_id": f"b{index:03d}", "boundary": index, "strict_ok": True}
                for index in range(1, 13)
            ]
            prediction = write_json(root / "prediction.json", {
                "model_id": "holdout_model",
                "candidates": candidates,
            })
            base = write_csv(root / "predictions_frozen.csv", candidates)
            ranking = write_csv(root / "ranking_predictions_frozen.csv", [{
                "case_id": "b001",
                "method_id": "cut_bytes_only",
                "predicted_value": 1.0,
            }])
            universe_path, universe_csv, universe = create_candidate_universe_manifest(
                model_id="holdout_model",
                candidates=candidates,
                mode="deterministic_audit",
                output_dir=root,
                audit_size=12,
                minimum_valid_candidates=10,
                source_prediction_sha256=str(sha256_file(prediction) or ""),
            )
            write_json(root / "prediction_freeze_manifest.json", {
                "prospective": True,
                "valid_for_holdout": True,
                "prediction_json": prediction.name,
                "prediction_sha256": sha256_file(prediction),
                "prediction_csv": base.name,
                "prediction_csv_sha256": sha256_file(base),
                "ranking_prediction_csv": ranking.name,
                "ranking_prediction_csv_sha256": sha256_file(ranking),
                "candidate_universe_manifest": universe_path.name,
                "candidate_universe_sha256": universe["universe_sha256"],
                "candidate_universe_csv": universe_csv.name,
                "candidate_universe_csv_sha256": sha256_file(universe_csv),
            })
            plan = {"model_id": "holdout_model", "campaign": {}}
            loaded = _load_suite_prediction(root, plan, [])
            freeze = loaded["holdout_model"]["_prediction_freeze"]
            self.assertTrue(freeze["valid"])
            self.assertTrue(freeze["candidate_universe_valid"])
            self.assertEqual(len(freeze["candidate_universe_selected_case_ids"]), 12)
            self.assertEqual(freeze["candidate_universe_minimum_valid_candidates"], 10)

            universe["selected_case_ids"] = ["attacker"]
            write_json(universe_path, universe)
            loaded = _load_suite_prediction(root, plan, [])
            freeze = loaded["holdout_model"]["_prediction_freeze"]
            self.assertFalse(freeze["valid"])
            self.assertFalse(freeze["candidate_universe_valid"])
            self.assertEqual(freeze["status"], "candidate_universe_hash_mismatch")


if __name__ == "__main__":
    unittest.main()
