from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.remote.bundle import REMOTE_MINIMAL_INCLUDES
from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack
from onnx_splitpoint_tool.workflow.run_discovery import (
    build_measurement_set_contract,
    discover_evaluation_run,
    inspect_evaluation_run,
    is_evaluation_run_dir,
    run_dirs_from_latest_log,
)


ROOT = Path(__file__).resolve().parents[1]


class ResultsBundleFixesV60dTests(unittest.TestCase):
    def _run_dir(self, root: Path, name: str, created_at: str) -> Path:
        path = root / name
        (path / "reports").mkdir(parents=True)
        (path / "run_manifest.json").write_text(
            json.dumps({
                "schema": "onnx-splitpoint/evaluation-run-manifest",
                "schema_version": 1,
                "run_id": name,
                "profile_id": "test",
                "status": "partial",
                "created_at": created_at,
            }),
            encoding="utf-8",
        )
        (path / "profile.yaml").write_text("profile_id: test\n", encoding="utf-8")
        (path / "evaluation_workflow.log").write_text(
            "interrupted evaluation evidence\n", encoding="utf-8"
        )
        return path

    def test_stale_gui_path_discovers_latest_valid_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "EvaluationRuns"
            root.mkdir()
            old = self._run_dir(root, "run_20260709", "2026-07-09T12:00:00+02:00")
            new = self._run_dir(root, "run_20260710", "2026-07-10T12:00:00+02:00")
            stale = root / "missing_run"
            latest_log = root / "_latest_evaluation_workflow.log"
            latest_log.write_text(f"Run directory: {new}\n", encoding="utf-8")

            result = discover_evaluation_run(
                preferred=[stale],
                output_roots=[root],
                latest_logs=[latest_log],
            )
            self.assertEqual(result.status, "discovered")
            self.assertEqual(result.selected.resolve(), new.resolve())
            self.assertTrue(is_evaluation_run_dir(result.selected))
            self.assertEqual(run_dirs_from_latest_log(latest_log)[0].resolve(), new.resolve())
            self.assertNotEqual(result.selected.resolve(), old.resolve())

    def test_valid_explicit_run_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            explicit = self._run_dir(root, "explicit", "2026-07-01T12:00:00+02:00")
            self._run_dir(root, "newer", "2026-07-10T12:00:00+02:00")
            result = discover_evaluation_run(preferred=[explicit], output_roots=[root])
            self.assertEqual(result.status, "explicit")
            self.assertEqual(result.selected.resolve(), explicit.resolve())

    def test_analysis_pack_uses_canonical_report_without_legacy_claim_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            model = run / "models" / "m"
            (model / "benchmark_results").mkdir(parents=True)
            (run / "reports").mkdir(parents=True)
            (run / "run_manifest.json").write_text(
                json.dumps({
                    "schema": "onnx-splitpoint/evaluation-run-manifest",
                    "schema_version": 1,
                    "run_id": "run",
                    "profile_id": "test",
                    "status": "ok",
                    "model_count": 1,
                    "models": {"m": {"model_id": "m"}},
                    "created_at": "2026-07-10T00:00:00+00:00",
                }),
                encoding="utf-8",
            )
            (run / "profile.yaml").write_text(
                yaml.safe_dump(
                    {
                        "profile_id": "test",
                        "quality_gate": {
                            "dataset_tier": "screening",
                            "frozen_before_final_campaign": False,
                        },
                        "model_suite": {
                            "primary": [
                                {
                                    "id": "m",
                                    "task": "classification",
                                    "evaluation_role": "development",
                                }
                            ],
                            "reserve": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (model / "model_manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": "m",
                        "task": "classification",
                        "profile_entry": {
                            "id": "m",
                            "task": "classification",
                            "evaluation_role": "development",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (model / "benchmark_results" / "normalized_results.json").write_text(
                json.dumps(
                    {
                        "schema": "onnx-splitpoint/normalized-benchmark-results",
                        "schema_version": 2,
                        "evaluation_run_id": "run",
                        "model_id": "m",
                        "status": "measured",
                        "matrix_complete": True,
                        "result_count": 1,
                        "missing_measurement_count": 0,
                        "missing_required_profile_result_count": 0,
                        "duplicate_required_profile_result_count": 0,
                        "validation_cardinality_mismatch_count": 0,
                        "results": [
                            {
                                "model_id": "m",
                                "run_id": "ort_cpu",
                                "backend": "ort_cpu",
                                "case_id": "full",
                                "variant": "full",
                                "runtime_ok": True,
                                "build_ok": True,
                                "semantic_validation_passed": True,
                                "throughput_primary_fps": 10.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            measurement_set = build_measurement_set_contract(run)
            self.assertTrue(measurement_set["valid"])
            (run / "reports" / "run_status_summary.json").write_text(
                json.dumps({
                    "schema": "onnx-splitpoint/run-status-summary",
                    "schema_version": 1,
                    "run_id": "run",
                    "status": "ok",
                }),
                encoding="utf-8",
            )
            (run / "reports" / "results_bundle_manifest.json").write_text(
                json.dumps({
                    "schema": "onnx-splitpoint/results-bundle-manifest",
                    "schema_version": 2,
                    "run_id": "run",
                    "contains_measured_benchmarks": True,
                    "measurement_set": measurement_set,
                    "contains_scientific_report": True,
                    "outputs": {
                        "scientific_report_json":
                            "reports/scientific/scientific_report.json"
                    },
                }),
                encoding="utf-8",
            )
            out = run.parent / "analysis.zip"
            result = create_analysis_pack(
                run, out, tool_version="0.14.3+v60d.resultsfix",
                materialize_missing_report=True,
            )
            self.assertTrue(out.is_file())
            self.assertEqual(result["model_count"], 1)
            report_path = (
                run / "reports" / "scientific" / "scientific_report.json"
            )
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["run_id"], "run")
            self.assertEqual(
                report["measurement_set_sha256"],
                measurement_set["measurement_set_sha256"],
            )
            self.assertTrue(inspect_evaluation_run(run).analysis_ready)
            self.assertFalse((run / "reports" / "claim_table_comprehensive.csv").exists())

    def test_generated_suite_vendors_and_refreshes_scientific_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            write_benchmark_suite_script(suite)
            reporter = suite / "scientific_reporter_v60.py"
            self.assertTrue(reporter.is_file())
            reporter.unlink()
            # A suite without cases is still refreshable at root level.
            result = refresh_suite_harness(suite)
            self.assertTrue(reporter.is_file())
            self.assertIn("scientific_reporter_v60.py", REMOTE_MINIMAL_INCLUDES)
            self.assertTrue(
                result.get("scientific_reporter_updated")
                or reporter.is_file()
            )

    def test_runtime_template_maps_native_trt_outputs_and_uses_backend_neutral_full_helper(self) -> None:
        template = (
            ROOT
            / "onnx_splitpoint_tool"
            / "resources"
            / "templates"
            / "run_split_onnxruntime.py.txt"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            template.count(
                "{o.name: a for o, a in zip(native_p2_sess.get_outputs(), native_out)}"
            ),
            2,
        )
        full_block = template[
            template.index("# full\n") :
            template.index("# part2 (isolated)", template.index("# full\n"))
        ]
        self.assertIn("_run_full_variant_outputs_map", full_block)
        self.assertNotIn("ORT full session not available", full_block)

    def test_legacy_native_full_tensorrt_energy_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "backend": "native_full_tensorrt",
                                "model": "resnet50",
                                "case": "full",
                                "precision": "fp16",
                                "setup_id": "orin_nx_hailo8_01",
                                "comparison_backend": "hailo8",
                                "ok": True,
                                "fps_makespan": 100.0,
                            }
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
                    "--summary",
                    str(summary),
                    "--out-dir",
                    str(out),
                    "--hailo8-ssh",
                    "nx@example",
                    "--allow-unpaired",
                    "--duration-s",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads((out / "native_producer_energy_plan.json").read_text())
            self.assertEqual(payload["rows"], [])
            self.assertEqual(payload["semantically_admitted_rows"], 0)
            self.assertEqual(
                payload["excluded_rows"][0]["reason"],
                "successful_command_contract_missing_or_invalid",
            )
            self.assertEqual(
                payload["excluded_rows"][0]["command_contract_status"],
                "full_command_contract_missing",
            )


if __name__ == "__main__":
    unittest.main()
