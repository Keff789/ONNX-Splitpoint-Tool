from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# The calibration helpers tested below do not need ONNX itself, but the
# optional Hailo backend imports the module at import time. Keep this unit test
# runnable in the lightweight reporting/test environment where onnx may be
# absent.
try:
    import onnx  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types

    _fake_onnx = types.ModuleType("onnx")
    _fake_onnx.AttributeProto = type("AttributeProto", (), {})
    _fake_onnx.ModelProto = type("ModelProto", (), {})
    _fake_onnx.NodeProto = type("NodeProto", (), {})
    _fake_onnx.ValueInfoProto = type("ValueInfoProto", (), {})
    _fake_onnx.helper = types.SimpleNamespace()
    sys.modules["onnx"] = _fake_onnx

from onnx_splitpoint_tool.benchmark.remote_run import (
    _extract_run_ids_from_add_args,
    _filter_benchmark_plan_for_run_ids,
    _stable_suite_cache_key,
)
from onnx_splitpoint_tool.campaign import create_dataset_manifest, verify_dataset_manifest
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.hailo_backend import _clamp_calib_count, _try_build_calib_from_dir, _hailo_cache_key
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _validation_defaults, _infer_run_switches
from onnx_splitpoint_tool.workflow.results import _best_full_latency


class V60OPerformanceNativeEnergyTests(unittest.TestCase):
    def _profile(self, mode: str = "standard", *, native: bool = True, native_energy: bool = False) -> dict:
        cfg = default_run_modes_config()
        return {
            "name": "v60o_test",
            "selection_policy": {
                "max_accepted_cases_per_model": 1,
                "preferred_shortlist": 10,
                "selection_strategy": "stratified_windows",
            },
            "model_suite": {"primary": [
                {"id": "resnet50", "task": "classification", "evaluation_role": "development", "enabled": True},
                {"id": "yolo26s", "task": "detection", "evaluation_role": "development", "enabled": True},
            ]},
            "run_profiles": [
                {"id": "hailo8", "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8"},
                {"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt"},
                {"id": "deepx_m1", "full": "deepx_m1", "stage1": "deepx_m1", "stage2": "deepx_m1"},
            ],
            "execution_preset": {
                "id": mode,
                "follow_tool_config": False,
                "snapshot": cfg["modes"][mode],
                "overrides": {"native_enabled": native, "energy_enabled": native_energy},
            },
        }

    def test_evalrun_energy_is_native_only(self) -> None:
        profile, audit = apply_run_mode(self._profile("final", native=True, native_energy=True))
        self.assertFalse(profile["energy"]["enabled"])
        self.assertFalse(profile["energy"]["generic_enabled"])
        self.assertEqual(profile["energy"]["measurement_path"], "native_only")
        self.assertTrue(profile["native_producers"]["energy"]["enabled"])
        self.assertEqual(profile["native_producers"]["energy"]["mode"], "measure")
        self.assertTrue(audit["energy_enabled"])

    def test_relaxed_manifest_verification_normalizes_hash_prefix_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "images"; root.mkdir()
            for i in range(30):
                (root / f"n{i:04d}.jpg").write_bytes((f"image-{i}" * 5).encode())
            out = Path(td) / "manifest.json"
            create_dataset_manifest(task="classification", role="validation", dataset_id="x", split="val", root=root, output=out)
            data = json.loads(out.read_text())
            # Mix serializations intentionally.
            data["items"][0]["sha256"] = "sha256:" + str(data["items"][0]["sha256"]).removeprefix("sha256:")
            # Recompute manifest hashes by using the creator's representation only;
            # the verifier must tolerate sha256: prefixes in content hashes.
            from onnx_splitpoint_tool.workflow.artifacts import sha256_json
            data["manifest_payload_sha256"] = sha256_json({k: v for k, v in data.items() if k != "manifest_payload_sha256"})
            result = verify_dataset_manifest(data, verification_mode="sampled", sample_size=7)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["checked_item_count"], 7)
            self.assertEqual(result["verification_mode"], "sampled")

    def test_standard_uses_registered_manifest_and_500_item_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "val"; root.mkdir()
            manifest = Path(td) / "val_manifest.json"
            manifest.write_text(json.dumps({"root": str(root), "item_count": 5000}), encoding="utf-8")
            p, _ = apply_run_mode(self._profile("standard"))
            p["campaign"]["dataset_manifests"]["classification"]["validation"] = str(manifest)
            row = {"id": "resnet50", "task": "classification", "development_subset": "imagenette_val_mini_200"}
            out = _validation_defaults(p, row, "resnet50")
            self.assertEqual(out["validation_images"], str(root.resolve()))
            self.assertEqual(out["validation_max_images"], 500)
            self.assertEqual(out["validation_source_kind"], "content_addressed_manifest")

    def test_memmap_calibration_honors_requested_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "calib"; root.mkdir()
            for i in range(20):
                np.save(root / f"{i:03d}.npy", np.full((4, 4, 3), i, dtype=np.uint8))
            mmap_path = Path(td) / "calib.mmap"
            arr = _try_build_calib_from_dir(
                calib_dir=root, expected_shape=[4, 4, 3], limit=20,
                preprocess="norm", storage_mode="memmap", memmap_path=mmap_path,
            )
            self.assertIsNotNone(arr)
            self.assertEqual(arr.shape[0], 20)
            self.assertTrue(mmap_path.is_file())
            self.assertEqual(_clamp_calib_count([640, 640, 3], 500, cap_bytes=1, storage="memmap"), 500)
            if isinstance(arr, np.memmap) and getattr(arr, "_mmap", None) is not None:
                arr._mmap.close()

    def test_multi_run_id_filter_and_stable_cache_key(self) -> None:
        ids = _extract_run_ids_from_add_args("--run-ids hailo8,hailo8_to_trt --provider auto")
        self.assertEqual(ids, ["hailo8", "hailo8_to_trt"])
        plan = {"runs": [{"id": "hailo8"}, {"id": "hailo8_to_trt"}, {"id": "deepx_m1"}]}
        self.assertEqual([r["id"] for r in _filter_benchmark_plan_for_run_ids(plan, ids)["runs"]], ids)
        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "model" / "benchmark_set"; suite.mkdir(parents=True)
            (suite / "benchmark_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            key1 = _stable_suite_cache_key(suite)
            key2 = _stable_suite_cache_key(suite)
            self.assertEqual(key1, key2)

    def test_tensorrt_full_latency_uses_full_timing_block(self) -> None:
        row = {"latency_ms": 100.0, "full_mean_ms": 99.0, "timings": {"composed": {"mean_ms": 100.0}, "full": {"mean_ms": 10.0}}}
        total, raw, _e2e, status = _best_full_latency(row)
        self.assertEqual(total, 10.0)
        self.assertEqual(raw, 10.0)
        self.assertEqual(status, "full")

    def test_execution_plan_exposes_hidden_expansion_and_native_energy(self) -> None:
        profile, _ = apply_run_mode(self._profile("standard", native=True, native_energy=True))
        plan = build_effective_execution_plan(profile)
        self.assertEqual(plan["model_count"], 2)
        self.assertFalse(plan["generic_energy_enabled"])
        self.assertTrue(plan["native_energy_enabled"])
        self.assertIn("ort_cpu", plan["automatic_reference_profiles"])
        self.assertIn("ort_tensorrt", plan["automatic_reference_profiles"])
        self.assertLess(plan["batched_remote_dispatches_total"], plan["generic_rows_total"])

    def test_native_analysis_tag_preserves_multiple_hailo10_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "native"; analysis = root / "analysis_tables"; analysis.mkdir(parents=True)
            for tag, model in (("a", "resnet50"), ("b", "yolo26s")):
                (analysis / f"native_hailo10h_producer_e2e_eval__{tag}.json").write_text(json.dumps({"rows": [{"model": model, "case": "b001", "precision": "fp16", "ok": True, "fps_makespan": 10.0}]}), encoding="utf-8")
            out = Path(td) / "out"
            cmd = [sys.executable, "scripts/native_producer_final_report.py", "--root", str(root), "--recursive", "--out-dir", str(out)]
            cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            data = json.loads((out / "native_producer_combined_summary.json").read_text())
            rows = [r for r in data["rows"] if r.get("backend") == "hailo10h_to_trt"]
            self.assertEqual({r["model"] for r in rows}, {"resnet50", "yolo26s"})

    def test_runner_source_reports_partial_native_evidence(self) -> None:
        source = Path("onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
        self.assertIn('stage["orchestration_status"]', source)
        self.assertIn('stage["evidence_status"]', source)
        self.assertIn('native-full baselines requested but no native-full rows were collected', source)


    def test_native_only_energy_survives_v60m_profile_normalization(self) -> None:
        from onnx_splitpoint_tool.v60m_policy import normalize_profile
        profile, _ = apply_run_mode(self._profile("final", native=True, native_energy=True))
        normalized, audit = normalize_profile(profile)
        self.assertTrue(normalized["native_producers"]["energy"]["enabled"])
        self.assertTrue(audit["energy"]["master_enabled"])
        self.assertEqual(audit["energy"]["master_source"], "energy.requested_native_energy")

    def test_selected_hailo_direction_avoids_unused_part2_build(self) -> None:
        profile = self._profile("standard")
        switches = _infer_run_switches(profile, [])
        self.assertTrue(switches["hailo_full_requested"])
        self.assertTrue(switches["hailo_part1_requested"])
        self.assertFalse(switches["hailo_part2_requested"])

    def test_hailo_cache_key_is_stable_and_configuration_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model.onnx"
            model.write_bytes(b"model-v1")
            calib = root / "calib"
            calib.mkdir()
            (calib / "sample.npy").write_bytes(b"sample")
            env = {
                "ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY": "relaxed",
                "ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST": "",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                key1, _ = _hailo_cache_key(
                    model_path=model, activation_part1=None, hw_arch="hailo8", opt_level=1,
                    calib_dir=calib, calib_count=500, calib_batch_size=8,
                    extra_model_script="", start_nodes=None, end_nodes=None,
                )
                key2, _ = _hailo_cache_key(
                    model_path=model, activation_part1=None, hw_arch="hailo8", opt_level=1,
                    calib_dir=calib, calib_count=500, calib_batch_size=8,
                    extra_model_script="", start_nodes=None, end_nodes=None,
                )
                key3, _ = _hailo_cache_key(
                    model_path=model, activation_part1=None, hw_arch="hailo8", opt_level=2,
                    calib_dir=calib, calib_count=500, calib_batch_size=8,
                    extra_model_script="", start_nodes=None, end_nodes=None,
                )
            self.assertEqual(key1, key2)
            self.assertNotEqual(key1, key3)


    def test_cached_detection_bootstrap_matches_legacy_resampling(self) -> None:
        import ast
        from typing import Any, Dict, List, Optional, Sequence, Tuple

        template = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
        tree = ast.parse(template)
        wanted = {
            "_boxes_iou_xyxy",
            "_mini_coco_ap50_101",
            "_mini_coco_ap50_for_variant",
            "_mini_coco_ap_50_95",
            "_prepare_detection_bootstrap_cache",
            "_detection_metric_from_bootstrap_cache",
            "_resampled_detection_maps",
        }
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted]
        self.assertEqual({node.name for node in nodes}, wanted)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {
            "np": np,
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Sequence": Sequence,
            "Tuple": Tuple,
        }
        exec(compile(module, "<runtime-bootstrap-test>", "exec"), ns)

        image_ids = ["a", "b", "c", "d"]
        gt = {
            "a": [{"class_id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}],
            "b": [{"class_id": 0, "x1": 5, "y1": 5, "x2": 15, "y2": 15}],
            "c": [{"class_id": 1, "x1": 0, "y1": 0, "x2": 8, "y2": 8}],
            "d": [{"class_id": 1, "x1": 4, "y1": 4, "x2": 12, "y2": 12}],
        }
        pred = {
            "a": [
                {"class_id": 0, "score": 0.99, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"class_id": 0, "score": 0.31, "x1": 20, "y1": 20, "x2": 30, "y2": 30},
            ],
            "b": [{"class_id": 0, "score": 0.88, "x1": 5, "y1": 5, "x2": 15, "y2": 15}],
            "c": [{"class_id": 1, "score": 0.77, "x1": 0, "y1": 0, "x2": 8, "y2": 8}],
            "d": [{"class_id": 1, "score": 0.66, "x1": 6, "y1": 6, "x2": 14, "y2": 14}],
        }
        cache = ns["_prepare_detection_bootstrap_cache"](gt_by_image=gt, pred_by_image=pred, image_ids=image_ids)
        rng = np.random.default_rng(20260714)
        for _ in range(30):
            indices = rng.integers(0, len(image_ids), size=len(image_ids))
            multiplicities = np.bincount(indices, minlength=len(image_ids)).astype(np.float32)
            cached = ns["_detection_metric_from_bootstrap_cache"](cache, multiplicities)
            gt_b, pred_b = ns["_resampled_detection_maps"](image_ids, indices, gt, pred)
            legacy = ns["_mini_coco_ap_50_95"](gt_by_image=gt_b, pred_by_image=pred_b)
            self.assertAlmostEqual(float(cached["ap_50_95"]), float(legacy["ap_50_95"]), places=7)
            self.assertAlmostEqual(float(cached["ap50"]), float(legacy["ap50"]), places=7)

    def test_scientific_report_exposes_bootstrap_runtime_metadata(self) -> None:
        from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
        from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row

        gate = {
            "task": "detection",
            "tier": "screening",
            "decision": "pass",
            "primary": {
                "metric": "coco_ap_50_95",
                "candidate": 0.4,
                "reference": 0.4,
                "delta": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "margin": 0.01,
                "n": 50,
                "decision": "pass",
                "bootstrap_repetitions_requested": 500,
                "bootstrap_repetitions": 0,
                "bootstrap_engine": "paired_detection_cached_matching_v1",
                "bootstrap_skipped_reason": "candidate_reference_identical",
                "bootstrap_elapsed_s": 0.0,
                "bootstrap_candidate_event_count": 123,
                "bootstrap_reference_event_count": 123,
            },
        }
        row = {
            "model_id": "yolo",
            "task": "detection",
            "case_id": "b001",
            "backend": "tensorrt",
            "variant": "composed",
            "buildable": True,
            "runtime_executable": True,
            "contract_consistent": True,
            "task_quality_gate": gate,
        }
        apply_accuracy_gate_to_row(row, {"dataset_tier": "screening"})
        out = _scientific_row(row)
        self.assertEqual(out["task_quality_bootstrap_engine"], "paired_detection_cached_matching_v1")
        self.assertEqual(out["task_quality_bootstrap_repetitions_requested"], 500)
        self.assertEqual(out["task_quality_bootstrap_repetitions"], 0)
        self.assertEqual(out["task_quality_bootstrap_skipped_reason"], "candidate_reference_identical")

    def test_generic_energy_guard_is_hard_disabled(self) -> None:
        from onnx_splitpoint_tool.workflow.execution_binding import _energy_enabled_for_profile
        profile, _ = apply_run_mode(self._profile("final", native=True, native_energy=True))
        enabled = _energy_enabled_for_profile(mock.Mock(), profile)
        self.assertFalse(enabled)
        self.assertEqual(profile["energy"]["measurement_path"], "native_only")


if __name__ == "__main__":
    unittest.main()
