from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.remote_run import _infer_suite_benchmark_task
from onnx_splitpoint_tool.campaign import build_campaign_readiness
from onnx_splitpoint_tool.energy.collector import _extract_work_units_from_text_v60i
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy, apply_accuracy_gate_to_row
from onnx_splitpoint_tool.workflow.results import (
    _validation_report_to_row,
    discover_result_files,
    normalize_benchmark_row,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import build_scientific_reports
from onnx_splitpoint_tool.workflow.execution_binding import _prepare_suite_for_runtime

ROOT = Path(__file__).resolve().parents[1]


class FinalEvidencePipelineV60iTests(unittest.TestCase):
    def test_explicit_model_task_overrides_stale_coco_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / "resnet50_coco_stale"
            suite.mkdir()
            (suite / "benchmark_plan.json").write_text(
                json.dumps(
                    {
                        "task": "classification",
                        "benchmark_task": "classification",
                        "runs": [{"benchmark_task": "classification", "validation_images": "coco_50_data"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_infer_suite_benchmark_task(suite), "classification")


    def test_suite_preflight_stamps_effective_policy_hash_and_model_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / "suite"
            case = suite / "b001"
            case.mkdir(parents=True)
            full = case / "full.onnx"
            full.write_bytes(b"synthetic")
            (case / "split_manifest.json").write_text(json.dumps({"full_model": "full.onnx"}), encoding="utf-8")
            plan = {"runs": [{"id": "hailo8_to_tensorrt", "benchmark_task": "auto"}]}
            (suite / "benchmark_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (suite / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_dir": "b001"}]}), encoding="utf-8")
            policy_map = {
                "dataset_tier": "screening",
                "statistics": {"confidence_level": 0.95, "bootstrap_repetitions": 5000, "seed": 20260710},
                "classification": {"non_inferiority_margin": 0.01},
            }
            expected = AccuracyGatePolicy.from_mapping(policy_map).sha256()
            result = _prepare_suite_for_runtime(
                suite_dir=suite, run_root=root, model_id="resnet50",
                suite_payload={"cases": [{"case_dir": "b001"}]},
                benchmark_plan=plan, model_task="classification",
                quality_gate_policy=policy_map,
            )
            self.assertEqual(result["model_task"], "classification")
            self.assertEqual(result["quality_gate_policy_sha256"], expected)
            written = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(written["task"], "classification")
            self.assertEqual(written["quality_gate_policy_sha256"], expected)
            self.assertEqual(written["runs"][0]["benchmark_task"], "classification")
            self.assertEqual(written["runs"][0]["quality_gate_policy_sha256"], expected)

    def test_nested_runtime_gate_is_ingested_and_policy_mismatch_blocks_claim(self) -> None:
        runtime_policy = {
            "dataset_tier": "screening",
            "classification": {"non_inferiority_margin": 0.01},
            "detection": {"non_inferiority_margin": 0.01},
            "statistics": {"confidence_level": 0.95, "bootstrap_repetitions": 500, "seed": 7},
        }
        report = {
            "case_id": "b001",
            "primary_variant": "composed",
            "run_cfg": {"provider": "hailo8_to_trt", "benchmark_task": "detection"},
            "timings": {"composed": {"mean_ms": 10.0}},
            "benchmark_task_requested": "detection",
            "benchmark_task_used": "detection",
            "semantic_validation_passed": True,
            "final_pass": True,
            "task_quality_policy": runtime_policy,
            "task_quality_gates_by_variant": {
                "composed": {
                    "task": "detection",
                    "tier": "screening",
                    "decision": "fail",
                    "status": "fail",
                    "policy": runtime_policy,
                    "primary": {
                        "metric": "coco_ap_50_95",
                        "candidate": 0.40,
                        "reference": 0.45,
                        "delta": -0.05,
                        "ci_low": -0.06,
                        "ci_high": -0.04,
                        "margin": 0.01,
                        "n": 50,
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "b001" / "results_hailo8_to_trt" / "validation_report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report), encoding="utf-8")
            raw = _validation_report_to_row(report, path)
            row = normalize_benchmark_row(raw, model_id="yolo", source_path=path, tag="hailo8_to_trt")
            self.assertEqual(row["task"], "detection")
            self.assertEqual(row["task_quality_gate"]["decision"], "fail")
            final_policy = AccuracyGatePolicy.from_mapping(
                {
                    **runtime_policy,
                    "statistics": {"confidence_level": 0.95, "bootstrap_repetitions": 5000, "seed": 7},
                }
            )
            apply_accuracy_gate_to_row(row, final_policy)
            self.assertEqual(row["accuracy_gate_decision"], "fail")
            self.assertFalse(row["accuracy_gate_policy_match"])
            self.assertFalse(row["performance_eligible"])

    def test_legacy_pipeline_summaries_are_not_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "v47_pipeline_summary.json").write_text("{}", encoding="utf-8")
            (root / "v42_pipeline_summary.csv").write_text("a,b\n", encoding="utf-8")
            (root / "benchmark_results_real.json").write_text("[]", encoding="utf-8")
            names = {path.name for path in discover_result_files([root])}
            self.assertIn("benchmark_results_real.json", names)
            self.assertNotIn("v47_pipeline_summary.json", names)
            self.assertNotIn("v42_pipeline_summary.csv", names)

    def test_classification_top1_mismatch_cannot_be_claim_ok(self) -> None:
        row = {
            "model": "resnet50",
            "task": "classification",
            "ok": True,
            "claim_ok": True,
            "self_reference_available": True,
            "self_reference_ok": True,
            "semantic_ok": True,
            "top1_match": False,
            "top5_overlap": 5,
            "buildable": True,
            "runtime_executable": True,
            "structural_contract_pass": True,
        }
        apply_accuracy_gate_to_row(row, AccuracyGatePolicy())
        self.assertTrue(row["contract_consistent"])
        self.assertFalse(row["numerical_similarity_pass"])
        self.assertFalse(row["performance_eligible"])
        self.assertEqual(
            row["numerical_similarity_reason"],
            "classification_top1_self_reference_mismatch",
        )

    def test_development_preflight_is_not_reported_as_final_ready(self) -> None:
        report = build_campaign_readiness(
            {
                "campaign": {"mode": "development"},
                "model_suite": {"primary": [], "reserve": []},
                "quality_gate": {"dataset_tier": "screening", "frozen_before_final_campaign": False},
            }
        )
        self.assertEqual(report["status"], "development_ready")
        self.assertTrue(report["development_ready"])
        self.assertFalse(report["final_ready"])
        self.assertFalse(report["ready"])

    def test_scientific_report_writes_separate_screening_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            model_dir = run / "models" / "yolo" / "benchmark_results"
            model_dir.mkdir(parents=True)
            (run / "reports").mkdir(parents=True)
            profile = {
                "name": "development",
                "campaign": {"mode": "development"},
                "model_suite": {"primary": [{"id": "yolo", "task": "detection", "evaluation_role": "development"}], "reserve": []},
                "quality_gate": {
                    "dataset_tier": "screening",
                    "frozen_before_final_campaign": False,
                    "statistics": {"confidence_level": 0.95, "bootstrap_repetitions": 500, "seed": 7},
                    "detection": {"non_inferiority_margin": 0.01},
                },
            }
            (run / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
            policy = AccuracyGatePolicy.from_mapping(profile["quality_gate"])
            gate = {
                "task": "detection",
                "tier": "screening",
                "decision": "pass",
                "status": "pass",
                "policy": profile["quality_gate"],
                "primary": {"metric": "coco_ap_50_95", "candidate": 0.45, "reference": 0.45, "delta": 0.0, "ci_low": -0.005, "ci_high": 0.005, "margin": 0.01, "n": 50},
            }
            row = {
                "model_id": "yolo",
                "case_id": "b001",
                "backend": "hailo8_to_trt",
                "variant": "split",
                "task": "detection",
                "build_ok": True,
                "runtime_ok": True,
                "semantic_validation_passed": True,
                "contract_consistent": True,
                "throughput_primary_fps": 42.0,
                "task_quality_gate": gate,
                "task_quality_policy": profile["quality_gate"],
            }
            apply_accuracy_gate_to_row(row, policy)
            (model_dir / "normalized_results.json").write_text(json.dumps({"results": [row]}), encoding="utf-8")
            build_scientific_reports(run, tool_version="test", workflow_version="v60i")
            out = run / "reports" / "scientific"
            self.assertTrue((out / "screening_performance_observations.csv").is_file())
            self.assertTrue((out / "claim_eligible_performance.csv").is_file())
            self.assertIn("development", (out / "scientific_report.md").read_text(encoding="utf-8").lower())

    def test_native_energy_plan_rejects_unbound_technical_rows_before_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "native_producer_summary.json"
            validation = root / "native_validation" / "native_producer_validation_summary.json"
            validation.parent.mkdir()
            summary.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"backend": "hailo8_to_trt", "model": "yolo", "case": "b001", "precision": "uint8_cast_fp16", "ok": True, "fps_makespan": 100.0, "report": ""},
                            {"backend": "hailo8_to_trt", "model": "yolo", "case": "b001", "precision": "uint8_dequant_fp16", "ok": True, "fps_makespan": 100.0, "report": "/tmp/result.json", "frames": 1000},
                            {"backend": "deepx_to_trt", "model": "yolo", "case": "b001", "precision": "fp16", "ok": True, "fps_makespan": 30.0},
                            {"backend": "native_full_hailo8", "model": "yolo", "case": "full", "precision": "uint8_dequant_fp16", "ok": True, "fps_makespan": 80.0, "setup_id": "orin_nx_hailo8_01", "comparison_backend": "hailo8"},
                            {"backend": "native_full_tensorrt", "model": "yolo", "case": "full", "precision": "uint8_dequant_fp16", "ok": True, "fps_makespan": 150.0, "setup_id": "orin_nx_hailo8_01", "comparison_backend": "hailo8"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"backend": "hailo8_to_trt", "model": "yolo", "case": "b001", "precision": "uint8_cast_fp16", "task": "detection", "claim_ok": True, "semantic_ok": True, "contract_consistent": True, "status": "claim_ok"},
                            {"backend": "hailo8_to_trt", "model": "yolo", "case": "b001", "precision": "uint8_dequant_fp16", "task": "detection", "claim_ok": True, "semantic_ok": True, "contract_consistent": True, "status": "claim_ok"},
                            {"backend": "deepx_to_trt", "model": "yolo", "case": "b001", "precision": "fp16", "task": "detection", "claim_ok": False, "semantic_ok": False, "contract_consistent": False, "status": "semantic_fail"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = root / "plan"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "native_producer_energy_plan.py"),
                    "--summary", str(summary),
                    "--validation-summary", str(validation),
                    "--out-dir", str(out),
                    "--hailo8-ssh", "nx@example",
                    "--duration-s", "1",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=ROOT,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads((out / "native_producer_energy_plan.json").read_text(encoding="utf-8"))
            runnable = [row for row in payload["rows"] if not row.get("skipped")]
            self.assertEqual(len(runnable), 0)
            self.assertEqual(payload["deduplicated_count"], 0)
            self.assertTrue(all(
                row.get("reason") == "successful_command_contract_missing_or_invalid"
                for row in payload["excluded_rows"]
            ))
            self.assertFalse(any(
                row.get("reason") == "semantic_fail"
                for row in payload["excluded_rows"]
            ))

    def test_runtime_work_unit_wrapper_and_collector_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "result.json"
            child = (
                "import json,pathlib;"
                f"p=pathlib.Path({str(report)!r});"
                "p.write_text(json.dumps({'frames': 123}),encoding='utf-8');"
                "print(json.dumps({'ok': True, 'report': str(p)}))"
            )
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_and_report_work_units.py"), "--", sys.executable, "-c", child],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=ROOT,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("__SPLITPOINT_WORK_UNITS__=123", proc.stdout)
            count, source = _extract_work_units_from_text_v60i(proc.stdout)
            self.assertEqual(count, 123)
            self.assertTrue(source)

    def test_runner_orders_native_validation_before_energy_and_syncs_work_counter(self) -> None:
        source = (ROOT / "onnx_splitpoint_tool" / "workflow" / "runner.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index(
                "Native output validation/visualization runs before native Energy"
            ),
            source.index(
                "Native-producer u.RECS energy uses technical P0.3 admission"
            ),
        )
        self.assertIn('script_name="run_and_report_work_units.py"', source)
        self.assertIn(
            '("--setup-id", "--comparison-backend", "--comparison-precision", "--duration-s", "--repetitions", "--quality-request-binding-set")',
            source,
        )
        self.assertIn('native_full_remote_import_preflight', source)


if __name__ == "__main__":
    unittest.main()


def test_v60i_console_smoke_command() -> None:
    from onnx_splitpoint_tool.v60i_smoke import run_smoke
    report = run_smoke()
    assert report["status"] == "ok", report
    assert report["failed"] == 0
    assert report["passed"] >= 6
    from onnx_splitpoint_tool import __version__
    from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
    assert report["tool_version"] == __version__
    assert report["workflow_version"] == WORKFLOW_VERSION


def test_v60i_packaged_remote_script_assets_match_source() -> None:
    for name in (
        "native_fifo_eval_runner.py",
        "native_hailo_trt_fifo_from_benchmarkset.py",
        "native_hailo10_trt_e2e_from_benchmarkset.py",
        "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_deepx_full_energy_hotloop.py",
        "native_producer_e2e_eval_runner.py",
        "native_full_baseline_eval_runner.py",
        "run_and_report_work_units.py",
        "native_producer_validate_visualize.py",
        "validate_output_dumps.py",
        "validate_classification_output_dump.py",
        "native_producer_energy_plan.py",
        "native_split_energy_preflight.py",
        "native_producer_final_report.py",
        "run_native_producer_energy_from_summary.py",
    ):
        source = (ROOT / "scripts" / name).read_bytes()
        packaged = (ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name).read_bytes()
        assert packaged == source, name
