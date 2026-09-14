from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onnx_splitpoint_tool.benchmark.suite_refresh import _normalize_suite_validation_payloads
from onnx_splitpoint_tool.deepx.artifacts import (
    cache_dxnn_artifact,
    deepx_cache_key,
    deepx_cached_artifact_compatible,
)
from onnx_splitpoint_tool.gui.benchmark_workflow import _manual_deepx_effective_calib_dir
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.execution_binding import _remote_args_from_options
from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
from onnx_splitpoint_tool.workflow.runner import (
    _benchmark_stage_status_v60p,
    _benchmark_stage_status_v60r,
    _selected_run_completeness_v60r,
    expected_profile_measurements_v60r,
    missing_profile_measurements_v60r,
    validation_cardinality_mismatches_v60r,
)


class V60RThreeFixesTests(unittest.TestCase):
    def test_run_mode_validation_budget_overrides_legacy_50_in_remote_args(self) -> None:
        options = WorkflowOptions(profile="profile.yaml", out="out")
        options.remote_validation_max_images = 50
        profile = {
            "execution_preset": {"id": "smoke", "snapshot": {}},
            "validation_execution": {"max_items": {"classification": 16, "detection": 12}},
            "remote_benchmark": {},
        }
        cls_args = _remote_args_from_options(options, profile, model_task="classification")
        det_args = _remote_args_from_options(options, profile, model_task="detection")
        self.assertEqual(cls_args.validation_max_images, 16)
        self.assertEqual(det_args.validation_max_images, 12)
        self.assertTrue(cls_args.validation_budget_authoritative)
        self.assertTrue(det_args.validation_budget_authoritative)

    def test_suite_refresh_keeps_per_run_authoritative_budget_even_with_cli_50(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "suite"
            source = Path(td) / "images"
            source.mkdir(parents=True)
            for idx in range(30):
                (source / f"img_{idx:03d}.jpg").write_bytes(f"image-{idx}".encode())
            suite.mkdir()
            run = {
                "id": "ort_cpu",
                "benchmark_task": "classification",
                "validation_images": str(source),
                "validation_items_requested": 16,
                "validation_budget_authoritative": True,
                "validation_max_images": 50,
                "mini_classification_eval": True,
            }
            plan = {"runs": [run]}
            (suite / "benchmark_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (suite / "benchmark_set.json").write_text(json.dumps({"model_name": "resnet50", "plan": plan}), encoding="utf-8")
            result = _normalize_suite_validation_payloads(
                suite,
                benchmark_set_json=suite / "benchmark_set.json",
                validation_images=str(source),
                validation_max_images=50,
                validation_reference_mode="auto",
                mini_coco_ap50=False,
                benchmark_task="classification",
                mini_classification_eval=True,
                log=None,
            )
            updated = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))["runs"][0]
            self.assertEqual(updated["validation_max_images"], 16)
            self.assertTrue(updated["validation_budget_authoritative"])
            self.assertIn("_n16_", updated["validation_images"])
            self.assertEqual(result["validation_max_images"], 16)

    def test_detection_task_cannot_be_flipped_to_imagenette_by_stale_validation_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            coco = root / "coco_calibration"
            imagenette = root / "imagenette_validation"
            coco.mkdir(); imagenette.mkdir()
            (coco / "a.jpg").write_bytes(b"detection")
            (imagenette / "a.jpg").write_bytes(b"classification")
            selected = _manual_deepx_effective_calib_dir(
                root,
                str(imagenette),
                str(coco),
                task_hint="detection",
                model_path="yolo26s_part1.onnx",
                input_shape=[1, 3, 640, 640],
            )
            self.assertEqual(Path(selected).resolve(), coco.resolve())

    def test_deepx_cache_key_and_compatibility_bind_task_manifest_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            onnx = root / "part1.onnx"
            cfg = root / "config.json"
            dxnn = root / "model.dxnn"
            onnx.write_bytes(b"same-onnx")
            cfg.write_text("{}", encoding="utf-8")
            dxnn.write_bytes(b"dxnn")
            det_contract = {
                "task": "detection",
                "calibration_manifest_identity": "sha256:det",
                "calibration_count": 500,
                "preprocessing_contract": "det-v1",
            }
            cls_contract = {
                "task": "classification",
                "calibration_manifest_identity": "sha256:cls",
                "calibration_count": 500,
                "preprocessing_contract": "cls-v1",
            }
            det_key = deepx_cache_key(onnx_path=onnx, config_path=cfg, target="deepx_m1", variant="part1_b038", cache_contract=det_contract)
            cls_key = deepx_cache_key(onnx_path=onnx, config_path=cfg, target="deepx_m1", variant="part1_b038", cache_contract=cls_contract)
            self.assertNotEqual(det_key, cls_key)
            cache_root = root / "cache"
            cache_dxnn_artifact(
                dxnn_path=dxnn,
                cache_root=cache_root,
                cache_key=det_key,
                manifest={"cache_contract": det_contract},
            )
            ok, reason, _ = deepx_cached_artifact_compatible(cache_dir=cache_root / det_key, expected_contract=det_contract)
            self.assertTrue(ok, reason)
            ok, reason, _ = deepx_cached_artifact_compatible(cache_dir=cache_root / det_key, expected_contract=cls_contract)
            self.assertFalse(ok)
            self.assertEqual(reason, "cache_contract_mismatch")

    def test_missing_selected_hailo8_full_marks_matrix_partial(self) -> None:
        plan = {"runs": [
            {"id": "hailo8", "type": "hailo", "stage1": {"hw_arch": "hailo8"}, "stage2": {"hw_arch": "hailo8"}},
            {"id": "hailo8_to_trt", "type": "matrix", "stage1": {"hw_arch": "hailo8"}, "stage2": {"provider": "tensorrt"}},
            {"id": "ort_tensorrt", "type": "onnxruntime", "stage1": {"provider": "tensorrt"}, "stage2": {"provider": "tensorrt"}},
        ]}
        sources = [
            {"path": "/tmp/benchmark_results_hailo8_auto.json", "tag": "hailo8_auto", "row_count": 0},
            {"path": "/tmp/benchmark_results_hailo8_to_trt_auto.json", "tag": "hailo8_to_trt_auto", "row_count": 1},
            {"path": "/tmp/benchmark_results_ort_tensorrt_auto.json", "tag": "ort_tensorrt_auto", "row_count": 1},
        ]
        result = _selected_run_completeness_v60r(benchmark_plan=plan, source_records=sources)
        self.assertFalse(result["matrix_complete"])
        self.assertEqual(result["missing_selected_run_ids"], ["hailo8"])
        self.assertEqual(result["missing_full_baseline_run_ids"], ["hailo8"])
        metrics = {"selected_run_missing_count": 1, "selected_run_matrix_complete": False}
        self.assertEqual(
            _benchmark_stage_status_v60p(normalized_row_count=4, executor_status="ok", executor_metrics=metrics),
            "partial",
        )


    def test_required_profile_matrix_detects_missing_selected_full_row(self) -> None:
        plan = {"runs": [
            {"id": "ort_cpu"},
            {"id": "ort_tensorrt"},
            {"id": "hailo8"},
            {"id": "hailo8_to_trt"},
        ]}
        benchmark_set = {"cases": [{"case_dir": "b038", "boundary": 38}]}
        expected = expected_profile_measurements_v60r(
            model_id="yolo26s", benchmark_plan=plan, benchmark_set_contract=benchmark_set
        )
        measured = [
            {"backend": "cpu_ort", "variant": "full", "case_id": "b038"},
            {"backend": "cpu_ort", "variant": "split", "case_id": "b038"},
            {"backend": "tensorrt", "variant": "full", "case_id": "b038"},
            {"backend": "tensorrt", "variant": "split", "case_id": "b038"},
            {"backend": "hailo8_to_tensorrt", "variant": "split", "case_id": "b038"},
        ]
        missing = missing_profile_measurements_v60r(expected, measured)
        self.assertEqual([(r["backend"], r["variant"]) for r in missing], [("hailo8", "full")])
        self.assertEqual(
            _benchmark_stage_status_v60r(
                normalized_row_count=len(measured), executor_status="ok", executor_metrics={},
                required_missing_count=len(missing), cardinality_mismatch_count=0,
            ),
            "partial",
        )

    def test_runtime_cardinality_mismatch_is_detected_against_run_mode_budget(self) -> None:
        plan = {"runs": [{
            "id": "ort_cpu",
            "validation_items_requested": 12,
            "validation_max_images": 12,
            "validation_budget_authoritative": True,
        }]}
        rows = [{
            "model_id": "yolo26s", "case_id": "b038", "backend": "cpu_ort", "variant": "split",
            "task_quality_gate": {"n": 50},
        }]
        mismatches = validation_cardinality_mismatches_v60r(benchmark_plan=plan, normalized_rows=rows)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["requested_count"], 12)
        self.assertEqual(mismatches[0]["evaluated_count"], 50)

    def test_cardinality_mismatch_blocks_row_eligibility(self) -> None:
        row = {
            "model_id": "yolo26s", "case_id": "b038", "backend": "tensorrt", "variant": "split",
            "runtime_ok": True, "compile_ok": True, "contract_consistent": True,
            "task": "detection",
            "task_quality_gate": {
                "task": "detection", "tier": "final", "decision": "pass",
                "primary": {"metric": "coco_ap_50_95", "candidate": 0.5, "reference": 0.5, "delta": 0.0, "margin": 0.01},
            },
            "validation_cardinality_contract": {"enforced": True, "pass": False, "requested_count": 12, "evaluated_count": 50},
        }
        apply_accuracy_gate_to_row(row, {"dataset_tier": "final", "frozen_before_final_campaign": True})
        self.assertFalse(row["ranking_eligible"])
        self.assertEqual(row["contract_gate_reason"], "validation_cardinality_mismatch")

    def test_generated_runtime_template_records_and_enforces_validation_cardinality(self) -> None:
        template = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
        self.assertIn("onnx-splitpoint/validation-cardinality-contract", template)
        self.assertIn("validation_cardinality_contract=validation_cardinality_contract", template)
        self.assertIn("validation_cardinality_mismatch", template)

    def test_nonempty_failure_row_counts_as_complete_evidence(self) -> None:
        plan = {"runs": [{"id": "hailo8", "type": "hailo"}]}
        sources = [{"path": "/tmp/benchmark_results_hailo8_auto.json", "tag": "hailo8_auto", "row_count": 1}]
        result = _selected_run_completeness_v60r(benchmark_plan=plan, source_records=sources)
        self.assertTrue(result["matrix_complete"])
        self.assertEqual(result["missing_selected_run_ids"], [])


if __name__ == "__main__":
    unittest.main()
