from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
    apply_accuracy_gate_to_row,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def _functions(path: Path, names: Sequence[str]) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    ns: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
        "Sequence": Sequence, "Tuple": Tuple, "Path": Path,
        "np": np, "json": json, "hashlib": hashlib,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _runtime_policy(*, reps: int = 500) -> Dict[str, Any]:
    return {
        "name": "task_quality_development_500",
        "profile_id": "task_quality_development_500",
        "dataset_tier": "final",
        "statistics": {
            "bootstrap_repetitions": reps,
            "confidence_level": 0.95,
            "seed": 7,
            "decision": "lower_one_sided_bound",
        },
        "classification": {"primary_metric": "top1_accuracy", "non_inferiority_margin": 0.01},
        "detection": {"primary_metric": "coco_ap_50_95", "non_inferiority_margin": 0.01},
    }


def _passing_gate(policy: Dict[str, Any]) -> Dict[str, Any]:
    effective = AccuracyGatePolicy.from_mapping(policy).as_dict()
    return {
        "decision": "pass", "status": "pass", "tier": "final", "policy": effective,
        "primary": {
            "metric": "top1_accuracy", "candidate": 0.80, "reference": 0.80,
            "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "margin": 0.01,
            "bootstrap_repetitions_requested": 500, "bootstrap_repetitions": 500,
        },
    }


def test_runtime_embedded_policy_is_authoritative_without_hidden_2000_fallback() -> None:
    policy = _runtime_policy(reps=500)
    row = {
        "task": "classification", "backend": "tensorrt", "variant": "full",
        "runtime_ok": True, "build_ok": True, "interface_contract_pass": True,
        "task_quality_policy": policy, "task_quality_gate": _passing_gate(policy),
    }
    apply_accuracy_gate_to_row(row)
    assert row["accuracy_gate_policy_source"] == "runtime_embedded"
    assert row["accuracy_gate_policy"]["bootstrap_repetitions"] == 500
    assert row["accuracy_gate_policy_match"] is True
    assert row["execution_ok"] is True
    assert row["interface_valid"] is True
    assert row["quality_valid"] is True


def test_cpu_ort_is_semantic_reference_only_not_a_performance_candidate() -> None:
    policy = _runtime_policy()
    row = {
        "task": "classification", "backend": "cpu_ort", "variant": "full",
        "runtime_ok": True, "build_ok": True, "interface_contract_pass": True,
        "task_quality_policy": policy, "task_quality_gate": _passing_gate(policy),
    }
    apply_accuracy_gate_to_row(row)
    assert row["semantic_reference_only"] is True
    assert row["quality_valid"] is True
    assert row["ranking_exclusion_reason"] == "semantic_reference_only"
    for field in ("ranking_eligible", "performance_eligible", "energy_eligible", "pareto_eligible", "thesis_valid"):
        assert row[field] is False


def test_raw_head_cannot_be_claimed_as_bn6_without_frozen_decoder_and_nms() -> None:
    policy = _runtime_policy()
    row = {
        "task": "detection", "backend": "native_full_hailo8", "variant": "full",
        "runtime_ok": True, "build_ok": True, "contract_consistent": True,
        "contract_family": "raw_head", "output_format": "bn6_detections",
        "raw_head_contract_status": "raw_head_only", "host_tail_required": True,
        "host_tail_available": False,
        "task_quality_policy": policy,
        "task_quality_gate": {
            **_passing_gate(policy),
            "primary": {**_passing_gate(policy)["primary"], "metric": "coco_ap_50_95"},
        },
    }
    apply_accuracy_gate_to_row(row)
    assert row["execution_ok"] is True
    assert row["interface_valid"] is False
    assert row["quality_valid"] is True  # independent E2E result remains visible
    assert row["contract_gate_reason"] in {
        "raw_head_misdeclared_as_bn6_detections", "raw_head_host_tail_missing",
    }
    assert row["thesis_valid"] is False


def test_hailo_to_trt_interface_failure_is_separate_from_e2e_quality() -> None:
    policy = _runtime_policy()
    row = {
        "task": "classification", "backend": "hailo8_to_trt", "variant": "split",
        "stage1_provider": "hailo8", "stage2_provider": "tensorrt",
        "runtime_ok": True, "build_ok": True, "semantic_ok": True,
        "interface_contract_pass": False,
        "task_quality_policy": policy, "task_quality_gate": _passing_gate(policy),
    }
    apply_accuracy_gate_to_row(row)
    assert row["execution_ok"] is True
    assert row["interface_valid"] is False
    assert row["quality_valid"] is True
    assert row["contract_gate_reason"] == "explicit_hailo_trt_interface_contract"
    assert row["thesis_valid"] is False


def test_central_management_policy_skips_local_bootstrap_and_exports_portable_request(tmp_path: Path) -> None:
    ns = _functions(RUNNER, [
        "_load_task_quality_policy", "_task_quality_execution_location",
        "_quality_gate_decision", "_one_sided_interval", "_paired_binary_metric_gate",
        "_classification_task_quality_gate", "_quality_json_safe",
        "_write_stable_quality_json", "_export_central_quality_inputs",
    ])
    policy = ns["_load_task_quality_policy"]({
        **_runtime_policy(),
        "statistics": {**_runtime_policy()["statistics"], "execution_location": "central_management"},
    })
    rows = [
        {
            "image": "0001.jpg", "label_id": 3,
            "gt": {"top1_hit": True, "top5_hit": True, "top1": 3, "top5": [3, 2, 1]},
            "gt_reference": {"top1_hit": True, "top5_hit": True, "top1": 3, "top5": [3, 1, 2]},
        },
        {
            "image": "0002.jpg", "label_id": 4,
            "gt": {"top1_hit": False, "top5_hit": True, "top1": 1, "top5": [1, 4]},
            "gt_reference": {"top1_hit": True, "top5_hit": True, "top1": 4, "top5": [4, 1]},
        },
    ]
    gate = ns["_classification_task_quality_gate"](rows, "composed", policy)
    assert gate["decision"] == "pending_central_evaluation"
    assert gate["primary"]["bootstrap_repetitions"] == 0
    assert gate["primary"]["bootstrap_skipped_reason"] == "delegated_to_central_management"
    assert gate["primary"]["delta"] is not None

    request = ns["_export_central_quality_inputs"](
        out_dir=tmp_path, task="classification", variant="composed",
        policy=policy, classification_rows=rows,
        endpoint_contract={
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "a" * 64,
            "stage": "classification_logits",
            "contract_family": "classification_logits",
        },
        runtime_precision_identity="float32_layout_fp16",
    )
    request_path = Path(request["request"]["path"])
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload["status"] == "pending_central_evaluation"
    assert payload["pairing_key"] == "image_id"
    assert payload["reference"]["source"] == "management_cpu_reference"
    assert "path" not in payload["reference"]
    assert payload["candidate"]["path"] == "composed_candidate.json"
    artifact = request_path.parent / payload["candidate"]["path"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == payload["candidate"]["sha256"]
    assert not (request_path.parent / "canonical_classification_reference.json").exists()
    candidate = json.loads((request_path.parent / payload["candidate"]["path"]).read_text(encoding="utf-8"))
    assert [r["image_id"] for r in candidate["records"]] == ["0001.jpg", "0002.jpg"]
    assert [r["label_id"] for r in candidate["records"]] == [3, 4]
    assert payload["reference"]["expected_image_ids"] == ["0001.jpg", "0002.jpg"]
    assert len(payload["reference"]["expected_image_ids_sha256"]) == 64


def test_cpu_reference_only_mode_is_the_only_export_that_writes_canonical_reference(
    tmp_path: Path, monkeypatch,
) -> None:
    ns = _functions(RUNNER, [
        "_task_quality_execution_location", "_quality_json_safe",
        "_write_stable_quality_json", "_export_central_quality_inputs",
    ])
    policy = {
        "statistics": {"execution_location": "central_management"},
        "classification": {"non_inferiority_margin": 0.01},
    }
    rows = [{
        "image": "i.jpg", "label_id": 2, "label_name": "two",
        "gt": {"top1_hit": True, "top5_hit": True},
        "gt_reference": {"top1_hit": True, "top5_hit": True, "top1": 2, "top5": [2]},
    }]
    monkeypatch.setenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY", "1")
    result = ns["_export_central_quality_inputs"](
        out_dir=tmp_path, task="classification", variant="full",
        policy=policy, classification_rows=rows,
    )
    assert result == {}
    reference_path = tmp_path / "task_quality_inputs/canonical_classification_reference.json"
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["reference"]["top1"] == 2
    assert not (tmp_path / "task_quality_inputs/full_request.json").exists()


def test_part_variants_do_not_export_end_to_end_central_quality(tmp_path: Path) -> None:
    ns = _functions(RUNNER, [
        "_task_quality_execution_location", "_quality_json_safe",
        "_write_stable_quality_json", "_export_central_quality_inputs",
    ])
    policy = {"statistics": {"execution_location": "central_management"}}
    for variant in ("part1", "part2"):
        assert ns["_export_central_quality_inputs"](
            out_dir=tmp_path, task="classification", variant=variant,
            policy=policy, classification_rows=[{"image": "i.jpg", "gt": {}}],
        ) == {}
    assert not (tmp_path / "task_quality_inputs").exists()


def test_every_dataset_cpu_reference_inference_is_guarded_from_central_accelerator_runs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER))
    parents: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    dataset_cpu_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "run" or not isinstance(node.func.value, ast.Name) or node.func.value.id != "sess_full":
            continue
        if not any(isinstance(arg, ast.Name) and arg.id == "local_feeds_full" for arg in node.args):
            continue
        dataset_cpu_calls.append(node)
        ancestor = parents.get(node)
        guarded = False
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                condition = ast.get_source_segment(source, ancestor.test) or ""
                if "central_quality_remote_mode" in condition:
                    guarded = True
                    break
            ancestor = parents.get(ancestor)
        assert guarded, f"unguarded dataset CPU reference inference at line {node.lineno}"
    assert len(dataset_cpu_calls) == 5


def test_final_policy_without_repetition_value_uses_documented_500_compatibility_default() -> None:
    ns = _functions(RUNNER, ["_load_task_quality_policy"])
    policy = ns["_load_task_quality_policy"]({"dataset_tier": "final", "statistics": {}})
    assert policy["statistics"]["bootstrap_repetitions"] == 500
    assert policy["statistics"]["bootstrap_repetitions_source"] == "compatibility_default_500"


def test_detection_central_mode_never_builds_local_matching_cache() -> None:
    ns = _functions(RUNNER, [
        "_task_quality_execution_location", "_quality_gate_decision",
        "_detection_task_quality_gate",
    ])
    ns["_mini_coco_ap_50_95"] = lambda **kw: {
        "ap_50_95": 0.40 if (kw["pred_by_image"]["i1"][0].get("candidate")) else 0.41,
        "ap50": 0.50 if (kw["pred_by_image"]["i1"][0].get("candidate")) else 0.51,
        "iou_thresholds": [0.5],
    }
    ns["_prepare_detection_bootstrap_cache"] = lambda **_kw: (_ for _ in ()).throw(
        AssertionError("remote matching cache must not be built in central mode")
    )
    policy = {
        "dataset_tier": "screening",
        "statistics": {"bootstrap_repetitions": 500, "execution_location": "central_management"},
        "detection": {"non_inferiority_margin": 0.01, "guardrails": {"ap50_margin": 0.01}},
    }
    gate = ns["_detection_task_quality_gate"](
        gt_by_image={"i1": [{"class_id": 0}]},
        candidate_by_image={"i1": [{"class_id": 0, "candidate": True}]},
        reference_by_image={"i1": [{"class_id": 0, "candidate": False}]},
        variant="composed", policy=policy,
    )
    assert gate["decision"] == "pending_central_evaluation"
    assert gate["primary"]["bootstrap_repetitions"] == 0
    assert gate["primary"]["bootstrap_skipped_reason"] == "delegated_to_central_management"


def test_deepx_decoder_routes_yolo26_and_yolov7_and_rejects_raw_bn6(tmp_path: Path) -> None:
    ns = _functions(SUITE, [
        "_deepx_yolo_nms", "_deepx_yolo_decode", "_deepx_yolov7_decode",
        "_deepx_bn6_decode", "_deepx_model_family", "_deepx_detection_decode",
    ])
    y26 = np.zeros((1, 84, 4), dtype=np.float32)
    y26[0, :4, 0] = [100, 100, 20, 20]
    y26[0, 4, 0] = 0.9
    detections, contract = ns["_deepx_detection_decode"](
        root=tmp_path, run={"model_id": "yolo26s"}, contract={"postprocessing": {"type": "yolo_host_decode_or_model_postprocess"}},
        outputs=[y26], orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert contract["pass"] is True
    assert contract["decoder_id"] == "yolo26_one2one_decoded_v1"
    assert detections

    y7 = np.zeros((1, 2, 85), dtype=np.float32)
    y7[0, 0, :6] = [100, 100, 20, 20, 0.9, 0.8]
    detections, contract = ns["_deepx_detection_decode"](
        root=tmp_path, run={"model_id": "yolov7_paper"}, contract={"postprocessing": {"type": "yolo_host_decode_or_model_postprocess"}},
        outputs=[y7], orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert contract["decoder_id"] == "yolov7_xywh_objectness_v1"
    assert abs(detections[0]["confidence"] - 0.72) < 1e-6

    raw_bn6 = np.zeros((1, 10, 6), dtype=np.float32)
    detections, contract = ns["_deepx_detection_decode"](
        root=tmp_path, run={"model_id": "yolo26s"},
        contract={"contract_family": "raw_head", "requires_external_postprocess": True},
        outputs=[raw_bn6], orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert detections == []
    assert contract["pass"] is False
    assert contract["status"] == "raw_head_decoder_contract_missing"


def test_suite_status_keeps_interface_and_quality_outcomes_independent() -> None:
    ns = _functions(SUITE, ["_v263_status_fields"])
    row = ns["_v263_status_fields"]({
        "task": "classification", "backend": "hailo8_to_trt",
        "stage1_provider": "hailo8", "stage2_provider": "tensorrt",
        "buildable": True, "runtime_ok": True,
        "interface_contract_pass": False,
        "task_quality_gate": {"decision": "pass"},
    })
    assert row["execution_ok"] is True
    assert row["interface_valid"] is False
    assert row["quality_valid"] is True
    assert row["evidence_complete"] is True
    assert row["thesis_valid"] is False
