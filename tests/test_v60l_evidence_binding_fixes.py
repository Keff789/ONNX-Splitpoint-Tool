from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from onnx_splitpoint_tool.campaign import create_dataset_manifest
from onnx_splitpoint_tool.dataset_provisioning import load_registry, save_registry
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.dataset_binding import (
    bind_profile_dataset_registry,
    calibration_manifest_for_task,
    manifest_dataset_root,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _bare_sha256_v60l,
    _sync_remote_script_v60i,
)

ROOT = Path(__file__).resolve().parents[1]


class V60LEvidenceBindingTests(unittest.TestCase):
    def test_hash_prefix_and_bare_digest_compare_identically(self) -> None:
        self.assertEqual(_bare_sha256_v60l("sha256:ABCDEF"), "abcdef")
        self.assertEqual(_bare_sha256_v60l("abcdef"), "abcdef")

    def test_remote_sync_accepts_project_prefixed_local_hash(self) -> None:
        responses = [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"sha256": "abc123", "missing_tokens": []}) + "\n",
                stderr="",
            ),
        ]
        with mock.patch(
            "onnx_splitpoint_tool.workflow.runner.subprocess.run",
            side_effect=responses,
        ), mock.patch(
            "onnx_splitpoint_tool.workflow.runner.sha256_file",
            return_value="sha256:abc123",
        ):
            steps = _sync_remote_script_v60i(
                ssh="tester@example",
                remote_tool_dir="/tmp/tool",
                script_name="native_fifo_eval_runner.py",
                required_tokens=(),
            )
        self.assertTrue(steps[-1]["sha256_ok"])
        self.assertEqual(steps[-1]["expected_sha256"], "abc123")
        self.assertEqual(steps[-1]["remote_sha256"], "abc123")

    def _runtime_policy_loader(self):
        template = (
            ROOT
            / "onnx_splitpoint_tool"
            / "resources"
            / "templates"
            / "run_split_onnxruntime.py.txt"
        ).read_text(encoding="utf-8")
        tree = ast.parse(template)
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_load_task_quality_policy"
        )
        module = ast.Module(body=[func], type_ignores=[])
        namespace = {
            "Any": object,
            "Dict": dict,
            "Path": Path,
            "json": json,
            "hashlib": hashlib,
        }
        exec(compile(module, "<runtime-policy-loader>", "exec"), namespace)
        return namespace["_load_task_quality_policy"]

    def test_long_inline_quality_json_keeps_exact_profile_policy(self) -> None:
        policy = {
            "schema": "onnx-splitpoint/task-quality-policy",
            "schema_version": 2,
            "name": "task_quality_v1",
            "profile_id": "task_quality_v1",
            "frozen_before_final_campaign": False,
            "dataset_tier": "screening",
            "canonical_reference": "canonical_full_onnx",
            "classification": {
                "primary_metric": "top1_accuracy",
                "non_inferiority_margin": 0.01,
                "guardrails": {"top5_accuracy_margin": 0.01},
            },
            "detection": {
                "primary_metric": "coco_ap_50_95",
                "non_inferiority_margin": 0.01,
                "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
            },
            "statistics": {
                "method": "paired_bootstrap",
                "confidence_level": 0.95,
                "bootstrap_repetitions": 5000,
                "seed": 20260710,
                "decision": "lower_one_sided_bound",
            },
            "native_contract": {
                "detection_self_reference_min_match_ratio": 0.9,
                "self_reference_counts_as_task_accuracy": False,
            },
            "screening_eligible_for_ranking": False,
            "legacy_point_estimate_eligible_for_ranking": False,
        }
        inline = json.dumps(policy, separators=(",", ":"))
        self.assertGreater(len(inline), 500)
        parsed = self._runtime_policy_loader()(inline)
        expected = AccuracyGatePolicy.from_mapping(policy)
        self.assertEqual(parsed["statistics"]["bootstrap_repetitions"], 5000)
        self.assertEqual(parsed["policy_sha256"], expected.sha256())
        self.assertEqual(AccuracyGatePolicy.from_mapping(inline).sha256(), expected.sha256())

    def _manifest(self, root: Path, task: str, role: str, name: str, content: bytes) -> Path:
        data_root = root / name
        if task == "classification":
            data_root = data_root / "n00000001"
        data_root.mkdir(parents=True, exist_ok=True)
        (data_root / f"{name}.jpg").write_bytes(content)
        manifest_root = data_root.parent if task == "classification" else data_root
        return create_dataset_manifest(
            task=task,
            role=role,
            dataset_id=name,
            split=role,
            root=manifest_root,
            output=root / "manifests" / f"{name}.json",
            max_items=0,
            selection_seed=7,
        )

    def test_missing_profile_dataset_fields_are_bound_from_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = root / "dataset_registry.json"
            manifests = {
                "classification_calibration": self._manifest(root, "classification", "calibration", "cls_cal", b"a"),
                "classification_validation": self._manifest(root, "classification", "validation", "cls_val", b"b"),
                "detection_calibration": self._manifest(root, "detection", "calibration", "det_cal", b"c"),
                "detection_validation": self._manifest(root, "detection", "validation", "det_val", b"d"),
            }
            ann = root / "instances_val2017.json"
            ann.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
            registry = load_registry(registry_path)
            registry["manifests"] = {k: str(v) for k, v in manifests.items()}
            registry["datasets"] = {
                "coco2017_validation": {
                    "root": str(root / "det_val"),
                    "annotations": str(ann),
                }
            }
            save_registry(registry, registry_path)
            profile = {
                "campaign": {
                    "mode": "development",
                    "dataset_registry": str(registry_path),
                    "auto_bind_dataset_registry": True,
                    "dataset_manifests": {
                        "classification": {"calibration": "", "validation": ""},
                        "detection": {"calibration": "", "validation": ""},
                    },
                },
                "official_coco_evaluation": {"enabled": True, "annotations": ""},
            }
            bound, report = bind_profile_dataset_registry(profile, verify_manifests=True)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(len(report["bound"]), 5)
            self.assertEqual(
                calibration_manifest_for_task(bound, "classification"),
                str(manifests["classification_calibration"].resolve()),
            )
            self.assertEqual(
                manifest_dataset_root(calibration_manifest_for_task(bound, "classification")),
                str((root / "cls_cal").resolve()),
            )
            self.assertEqual(bound["official_coco_evaluation"]["annotations"], str(ann))

    def test_development_preflight_warning_is_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner = object.__new__(EvaluationWorkflowRunner)
            runner.profile_payload = {"campaign": {"mode": "development"}}
            runner.run_dir = Path(temp)
            (runner.run_dir / "campaign").mkdir(parents=True)
            (runner.run_dir / "campaign" / "campaign_readiness.json").write_text(
                json.dumps({"status": "development_ready"}), encoding="utf-8"
            )
            self.assertTrue(
                runner._is_non_blocking_stage_status(
                    {"stage": "campaign_preflight", "status": "warn", "model_id": ""}
                )
            )

    def test_phase_run_probe_is_not_limited_to_first_50k(self) -> None:
        template = (
            ROOT
            / "onnx_splitpoint_tool"
            / "resources"
            / "templates"
            / "benchmark_suite.py.txt"
        ).read_text(encoding="utf-8")
        self.assertIn('runner_supports_phase_runs = "--phase-runs" in _runner_probe', template)
        self.assertNotIn('read_text(encoding="utf-8", errors="ignore")[:50000]', template)


if __name__ == "__main__":
    unittest.main()
