from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from onnx_splitpoint_tool.campaign import create_candidate_universe_manifest
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.ranking_methods import METHOD_ORDER, compute_ranking_predictions
from onnx_splitpoint_tool.workflow.artifacts import sha256_file, sha256_json, write_csv, write_json


class GeneratedSuiteReporterV60Tests(unittest.TestCase):
    def _load_generated_reporter(self, root: Path, module_name: str):
        write_benchmark_suite_script(root)
        reporter = root / "scientific_reporter_v60.py"
        spec = importlib.util.spec_from_file_location(module_name, reporter)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec else None)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def test_generated_payload_hash_matches_campaign_producer_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            module = self._load_generated_reporter(Path(temp), "scientific_reporter_v61e_hash_contract")
            payload = {
                "schema": "onnx-splitpoint/candidate-universe-manifest",
                "model_id": "mödel-a",
                "selected_case_ids": ["b001", "b017"],
                "audit": {"seed": 17, "selection_uses_predictions": False},
            }
            self.assertEqual(module._sha256_payload(payload), sha256_json(payload))

    def test_controller_vendors_companion_and_companion_reports_nested_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = Path(write_benchmark_suite_script(root))
            reporter = root / "scientific_reporter_v60.py"
            self.assertTrue(script.is_file())
            self.assertTrue(reporter.is_file())

            plan = {
                "model_id": "holdout_model",
                "evaluation_profile": {"profile_id": "generated-suite-v60"},
                "quality_gate": {
                    "frozen_before_final_campaign": True,
                    "dataset_tier": "final",
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
                    "cycle_time_with_handover": {
                        "handover_models": {
                            "generic": {
                                "hailo8_to_tensorrt": {"mode": "constant", "value_ms": 0.1}
                            }
                        }
                    },
                },
                "model_suite": {
                    "primary": [{
                        "id": "holdout_model",
                        "evaluation_role": "holdout",
                        "candidate_universe_complete": True,
                    }],
                    "reserve": [],
                },
            }
            candidates = [
                {"case_id": "b001", "boundary": 1, "cut_bytes": 1_000_000, "imbalance_val": 0.7, "n_cut_tensors": 1, "predicted_stage1_ms": 2.0, "predicted_stage2_ms": 8.0, "strict_ok": True},
                {"case_id": "b002", "boundary": 2, "cut_bytes": 2_000_000, "imbalance_val": 0.1, "n_cut_tensors": 1, "predicted_stage1_ms": 5.0, "predicted_stage2_ms": 5.5, "strict_ok": True},
                {"case_id": "b003", "boundary": 3, "cut_bytes": 3_000_000, "imbalance_val": 0.2, "n_cut_tensors": 2, "predicted_stage1_ms": 6.0, "predicted_stage2_ms": 5.0, "strict_ok": True},
            ]
            prediction_path = write_json(root / "prediction.json", {"model_id": "holdout_model", "candidates": candidates})
            base_csv = write_csv(root / "predictions_frozen.csv", [{"case_id": row["case_id"]} for row in candidates])
            method_rows = compute_ranking_predictions(
                candidates,
                [{"direction": "hailo8_to_tensorrt", "stage1": "hailo8", "stage2": "tensorrt", "runner_regime": "generic"}],
                plan["ranking_validation"],
            )
            ranking_csv = write_csv(root / "ranking_predictions_frozen.csv", method_rows)
            universe_path, universe_csv, universe = create_candidate_universe_manifest(
                model_id="holdout_model",
                candidates=candidates,
                mode="all_feasible",
                output_dir=root,
                source_prediction_sha256=sha256_file(prediction_path) or "",
            )
            write_json(root / "prediction_freeze_manifest.json", {
                "prospective": True,
                "evaluation_role": "holdout",
                "valid_for_holdout": True,
                "freeze_status": "prospective_frozen",
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
            })

            measured = {"b001": 8.3, "b002": 5.7, "b003": 6.2}
            rows = []
            for case_id, cycle in measured.items():
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
                    "pipeline_cycle_selected_ms": cycle,
                    "task_quality_gate": {
                        "tier": "final",
                        "status": "pass",
                        "decision": "pass",
                        "primary": {
                            "metric": "top1_accuracy",
                            "candidate": 0.799,
                            "reference": 0.800,
                            "delta": -0.001,
                            "ci_low": -0.004,
                            "ci_high": 0.002,
                            "margin": 0.01,
                        },
                    },
                })

            spec = importlib.util.spec_from_file_location("scientific_reporter_v60_test", reporter)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec else None)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            payload = module.write_scientific_report(root, {"synthetic": rows}, plan)

            report = root / "scientific_report"
            self.assertEqual(payload["summary"]["ranking"]["status"], "holdout_available")
            self.assertTrue((report / "ranking_method_comparison.csv").is_file())
            self.assertFalse((report / "ranking_validation.csv").exists())
            with (report / "task_quality.csv").open(newline="", encoding="utf-8") as handle:
                quality_rows = list(csv.DictReader(handle))
            self.assertEqual(quality_rows[0]["task_quality_metric"], "top1_accuracy")
            self.assertEqual(float(quality_rows[0]["task_quality_candidate"]), 0.799)
            with (report / "ranking_method_macro.csv").open(newline="", encoding="utf-8") as handle:
                macros = list(csv.DictReader(handle))
            self.assertEqual(len(macros), len(METHOD_ORDER))

    def test_generated_suite_cleanup_removes_superseded_presentation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = Path(write_benchmark_suite_script(root))
            (root / "benchmark_summary_example.md").write_text("old\n", encoding="utf-8")
            (root / "benchmark_table_example.tex").write_text("old\n", encoding="utf-8")
            (root / "v42_pipeline_summary.json").write_text("{}", encoding="utf-8")
            figures = root / "paper_figures_example"
            figures.mkdir()
            (figures / "plot.png").write_bytes(b"old")

            spec = importlib.util.spec_from_file_location("benchmark_suite_v60_cleanup_test", script)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec else None)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            removed = module._v60_cleanup_legacy(root)

            self.assertGreaterEqual(len(removed), 4)
            self.assertFalse((root / "benchmark_summary_example.md").exists())
            self.assertFalse((root / "benchmark_table_example.tex").exists())
            self.assertFalse((root / "v42_pipeline_summary.json").exists())
            self.assertFalse(figures.exists())

    def test_generated_reporter_does_not_claim_top_k_below_minimum_n(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = self._load_generated_reporter(root, "scientific_reporter_v61e_small_n")
            plan = {
                "ranking_validation": {
                    "methods": ["cut_bytes_only"],
                    "k_values": [1, 3, 5],
                    "elite_q_values": [1, 3],
                    "minimum_candidates_for_correlation": 3,
                    "require_complete_candidate_universe": True,
                    "require_frozen_predictions": True,
                    "candidate_universe_complete": True,
                },
                "model_suite": {
                    "primary": [{
                        "id": "holdout_model",
                        "evaluation_role": "holdout",
                        "candidate_universe_complete": True,
                    }],
                    "reserve": [],
                },
            }
            candidates = [{"case_id": "b001", "boundary": 1, "cut_bytes": 1000, "strict_ok": True}]
            prediction = write_json(root / "prediction.json", {"model_id": "holdout_model", "candidates": candidates})
            base = write_csv(root / "predictions_frozen.csv", [{"case_id": "b001"}])
            ranking = write_csv(root / "ranking_predictions_frozen.csv", [{
                "model_id": "holdout_model",
                "case_id": "b001",
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "generic",
                "method_id": "cut_bytes_only",
                "prediction_available": True,
                "predicted_value": 1.0,
                "predicted_rank": 1,
                "prediction_unit": "score",
            }])
            universe_path, universe_csv, universe = create_candidate_universe_manifest(
                model_id="holdout_model",
                candidates=candidates,
                mode="all_feasible",
                output_dir=root,
                source_prediction_sha256=sha256_file(prediction) or "",
            )
            write_json(root / "prediction_freeze_manifest.json", {
                "prospective": True,
                "evaluation_role": "holdout",
                "valid_for_holdout": True,
                "freeze_status": "prospective_frozen",
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
            rows = [{
                "model_id": "holdout_model",
                "evaluation_role": "holdout",
                "case_id": "b001",
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "generic",
                "variant": "split",
                "cycle_ms": 1.0,
                "ranking_eligible": True,
            }]

            details, _macros, _summary = module._ranking_comparison(root, rows, plan)
            self.assertEqual(len(details), 1)
            detail = details[0]
            self.assertEqual(detail["status"], "insufficient_candidates_for_correlation")
            self.assertIsNone(detail["spearman_rho"])
            self.assertIsNone(detail["kendall_tau_b"])
            for k in (1, 3, 5):
                self.assertIsNone(detail[f"hit_at_{k}"])
                self.assertIsNone(detail[f"near_optimal_hit_at_{k}"])
                self.assertIsNone(detail[f"regret_at_{k}"])
                self.assertIsNone(detail[f"elite_recall_at_{k}_q1"])

    def test_generated_reporter_rejects_tampered_candidate_universe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = self._load_generated_reporter(root, "scientific_reporter_v61e_tamper")
            prediction = write_json(root / "prediction.json", {"model_id": "holdout_model"})
            base = write_csv(root / "predictions_frozen.csv", [{"case_id": "b001"}])
            ranking = write_csv(root / "ranking_predictions_frozen.csv", [{"case_id": "b001"}])
            universe_csv = write_csv(root / "candidate_universe.csv", [{"case_id": "b001"}])
            universe = {
                "schema": "onnx-splitpoint/candidate-universe-manifest",
                "created_at": "2026-07-18T00:00:00Z",
                "claim_scope": "predeclared_audit_universe",
                "declared_complete": True,
                "selected_case_ids": ["b001"],
                "audit": {"minimum_valid_candidates": 10},
            }
            # Hash this through the actual campaign producer contract.  The
            # vendored standalone reporter must accept exactly that encoding.
            universe["universe_sha256"] = sha256_json({
                key: value for key, value in universe.items() if key not in {"universe_sha256", "created_at"}
            })
            universe_path = write_json(root / "candidate_universe_manifest.json", universe)
            manifest = {
                "prospective": True,
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
            }
            write_json(root / "prediction_freeze_manifest.json", manifest)
            _freeze, _rows, valid, details = module._freeze(root, {})
            self.assertTrue(valid)
            self.assertTrue(details["candidate_universe_valid"])

            universe["selected_case_ids"] = ["attacker"]
            write_json(universe_path, universe)
            _freeze, _rows, valid, details = module._freeze(root, {})
            self.assertFalse(valid)
            self.assertFalse(details["candidate_universe_valid"])

    def test_generated_reporter_requires_declared_universe_for_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = self._load_generated_reporter(root, "scientific_reporter_v61e_universe_required")
            candidates = [{"case_id": "b001", "boundary": 1, "cut_bytes": 1000, "strict_ok": True}]
            prediction = write_json(root / "prediction.json", {"model_id": "holdout_model", "candidates": candidates})
            base = write_csv(root / "predictions_frozen.csv", candidates)
            ranking = write_csv(root / "ranking_predictions_frozen.csv", [{"case_id": "b001"}])
            universe_path, universe_csv, universe = create_candidate_universe_manifest(
                model_id="holdout_model",
                candidates=candidates,
                mode="all_feasible",
                output_dir=root,
                source_prediction_sha256=sha256_file(prediction) or "",
            )
            manifest = {
                "prospective": True,
                "model_id": "holdout_model",
                "evaluation_role": "holdout",
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
            }
            manifest_path = write_json(root / "prediction_freeze_manifest.json", manifest)
            plan = {
                "model_id": "holdout_model",
                "model_suite": {"primary": [{"id": "holdout_model", "evaluation_role": "holdout"}]},
            }

            _freeze, _rows, valid, details = module._freeze(root, plan)
            self.assertTrue(valid)
            self.assertTrue(details["candidate_universe_required"])
            self.assertTrue(details["candidate_universe_valid"])

            stripped = {
                key: value
                for key, value in manifest.items()
                if key not in {
                    "candidate_universe_manifest",
                    "candidate_universe_sha256",
                    "candidate_universe_csv",
                    "candidate_universe_csv_sha256",
                }
            }
            write_json(manifest_path, stripped)
            _freeze, _rows, valid, details = module._freeze(root, plan)
            self.assertFalse(valid)
            self.assertTrue(details["candidate_universe_required"])
            self.assertFalse(details["candidate_universe_valid"])


if __name__ == "__main__":
    unittest.main()
