from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    _validate_candidate_execution_contract,
    quality_request_from_manifest,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
    runtime_numeric_input_identity,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
PREFLIGHT = ROOT / "scripts/preflight_v27541_deepx_calibration_1000.py"


def _prepared_input_audit_fields(
    *, task: str, shape: list[int], layout: str,
    normalization: str,
) -> dict[str, Any]:
    target = [shape[0], shape[1]] if layout == "HWC" else [shape[1], shape[2]]
    preprocessing = canonical_image_preprocessing_contract(task, target)
    preprocessing_sha = preprocessing_contract_sha256(preprocessing)
    numeric, numeric_sha = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task=task,
        preprocessing_contract_sha256_value=preprocessing_sha,
        runtime_input_name="images",
        runtime_input_shape=shape,
        runtime_input_dtype="uint8",
        runtime_input_layout=layout,
        runtime_color_space="RGB",
        runtime_normalization=normalization,
    )
    binding = {
        "binding_verified": True,
        "prepared_input_sha256": hashlib.sha256(
            f"{task}:{shape}".encode("utf-8")
        ).hexdigest(),
        "prepared_input_bytes": int(np.prod(shape)),
        "prepared_input_name": "images",
        "prepared_input_shape": shape,
        "prepared_input_dtype": "uint8",
        "prepared_input_layout": layout,
        "prepared_rgb_uint8_sha256": hashlib.sha256(
            f"rgb:{task}:{shape}".encode("utf-8")
        ).hexdigest(),
        "prepared_rgb_uint8_bytes": int(np.prod(shape)),
        "runtime_preprocessing_identity": preprocessing,
        "runtime_preprocessing_sha256": preprocessing_sha,
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": numeric_sha,
    }
    return {
        "runtime_preprocessing_identity": preprocessing,
        "runtime_preprocessing_sha256": preprocessing_sha,
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": numeric_sha,
        "prepared_tensor_binding": binding,
    }


def _bind_prepared_source(
    audit: dict[str, Any], image_path: Path,
) -> dict[str, Any]:
    binding = audit["prepared_tensor_binding"]
    binding["source_image_id"] = image_path.name
    binding["source_image_sha256"] = hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()
    return audit


def _performance_input_evidence(
    image_path: Path, audit: Mapping[str, Any],
) -> dict[str, Any]:
    binding = dict(audit["prepared_tensor_binding"])
    return {
        "status": "ok",
        "image": str(image_path),
        "prepared_input_source_image_id": image_path.name,
        "prepared_input_source_image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
        "prepared_input_binding_verified": True,
        **{
            key: binding[key]
            for key in (
                "prepared_input_sha256", "prepared_input_bytes",
                "prepared_input_name", "prepared_input_shape",
                "prepared_input_dtype", "prepared_input_layout",
            )
        },
        "runtime_preprocessing_identity": dict(
            audit["runtime_preprocessing_identity"]
        ),
        "runtime_preprocessing_sha256": str(
            audit["runtime_preprocessing_sha256"]
        ),
        "runtime_numeric_input_identity": dict(
            audit["runtime_numeric_input_identity"]
        ),
        "runtime_numeric_input_sha256": str(
            audit["runtime_numeric_input_sha256"]
        ),
    }


def _authoritative_endpoint_contract(root: Path, *, task: str) -> dict[str, Any]:
    model_id = "resnet50" if task == "classification" else "yolo26s"
    raw = task == "detection"
    (root / "output_contracts.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts", "schema_version": 1,
        "model_id": model_id, "task": task,
        "contracts": [{
            "model_id": model_id, "backend": "deepx_m1", "variant": "full",
            "task": task,
            "endpoint_mode": "raw_detection_head" if raw else "decoded",
            "contract_status": "recorded",
            "host_tail_required": raw, "postprocessing_required": raw,
        }],
    }), encoding="utf-8")
    return load_authoritative_output_contract(
        root, backend="deepx_m1", model_id=model_id,
        variant="full", task=task,
    )


def _suite_module() -> ModuleType:
    name = "_osp_v269a_benchmark_suite_test"
    module = ModuleType(name)
    module.__file__ = str(SUITE)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(SUITE.read_text(encoding="utf-8"), str(SUITE), "exec"), module.__dict__)
    return module


def _preflight_module() -> ModuleType:
    name = "_osp_v27541_preflight_cross_helper_test"
    spec = importlib.util.spec_from_file_location(name, PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_functions(names: Sequence[str]) -> Dict[str, Any]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    wanted = set(names)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    namespace: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping,
        "Optional": Optional, "Sequence": Sequence, "Tuple": Tuple,
        "Path": Path, "np": np, "json": json, "hashlib": hashlib,
        "re": __import__("re"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


def _policy() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/task-quality-policy", "schema_version": 2,
        "name": "test", "profile_id": "test", "dataset_tier": "screening",
        "canonical_reference": "canonical_full_onnx",
        "statistics": {
            "method": "paired_bootstrap", "execution_location": "central_management",
            "bootstrap_repetitions": 25, "confidence_level": 0.95,
            "seed": 7, "decision": "lower_one_sided_bound", "workers": 4,
        },
        "classification": {
            "primary_metric": "top1_accuracy", "non_inferiority_margin": 0.01,
            "guardrails": {"top5_accuracy_margin": 0.01},
        },
        "detection": {
            "primary_metric": "coco_ap_50_95", "non_inferiority_margin": 0.01,
            "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
        },
    }


def _base_tree(tmp_path: Path, *, task: str, samples: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    root = tmp_path / "suite"
    (root / "models").mkdir(parents=True)
    source = root / "models/source.onnx"
    source.write_bytes(b"same-frozen-source-onnx")
    dxnn = root / "deepx/deepx_m1/full/model.dxnn"
    dxnn.parent.mkdir(parents=True)
    dxnn.write_bytes(b"exact-vendor-compiled-dxnn")
    validation = root / "validation"
    validation.mkdir()
    for sample in samples:
        (validation / str(sample["image"])).write_bytes(b"image")
    (validation / "manifest.json").write_text(
        json.dumps({"task": task, "samples": samples}, sort_keys=True), encoding="utf-8",
    )
    output_contract = root / "deepx/deepx_m1/full/output_contract.json"
    model_id = "yolo26s" if task == "detection" else "resnet50"
    authoritative = {
        "schema": "onnx-splitpoint/output-contract",
        "schema_version": 1,
        "model_id": model_id,
        "backend": "deepx_m1",
        "variant": "full",
        "task": task,
        "contract_status": "recorded",
        "endpoint_mode": (
            "raw_detection_head" if task == "detection" else "decoded"
        ),
        "host_tail_required": task == "detection",
        "postprocessing_required": task == "detection",
    }
    output_contract.write_text(json.dumps({
        "model_id": model_id,
        "backend": "deepx_m1",
        "variant": "full",
        "contract_complete_enough_for_reports": True,
        "input": {
            "shape": [640, 640, 3] if task == "detection" else [224, 224, 3],
            "layout": "HWC", "dtype": "uint8", "color_space": "RGB",
            "preprocess_mode": "letterbox" if task == "detection" else "resize",
            "normalization": "embedded_dxcom_preprocessing",
        },
        # Real legacy contract: shape/compute precision are placeholders.  The
        # candidate endpoint must be proven by runtime observations instead.
        "output": {"status": "declared_no_parse_model_pending_placeholder"},
        "outputs": [{"dtype": "float32", "shape": None}],
        "endpoint_semantic_attestation": {
            "schema": (
                "onnx-splitpoint/deepx-endpoint-semantic-attestation"
            ),
            "schema_version": 1,
            "status": "attested",
            "pass": True,
            "model_id": model_id,
            "backend": "deepx_m1",
            "variant": "full",
            "endpoint_mode": authoritative["endpoint_mode"],
            "host_tail_required": authoritative["host_tail_required"],
            "postprocessing_required": authoritative[
                "postprocessing_required"
            ],
            "source_endpoint_has_integrated_nms": (
                False if task == "detection" else None
            ),
            "authoritative_contract": authoritative,
            "authoritative_contract_sha256": hashlib.sha256(
                json.dumps(
                    authoritative, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }, sort_keys=True), encoding="utf-8")
    return root, source, dxnn


def _detection_semantic(root: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    audits = [
        {"shape": [640, 640, 3], "dtype": "uint8", "layout": "HWC", "color_space": "RGB",
         "preprocess_mode": "letterbox", "normalization": "embedded_dxcom_preprocessing",
         # 640 - 361 is odd: preprocessing uses the integer top pad 139.
         # This catches the former half-pixel/fractional re-projection bug.
         "scale": 1.0, "pad_x": 0, "pad_y": 139, "letterbox_pad_value": 114,
         "source_shape_hw": [361, 640], "contract_source": "test",
         **_prepared_input_audit_fields(
             task="detection", shape=[640, 640, 3], layout="HWC",
             normalization="embedded_dxcom_preprocessing",
         )},
        {"shape": [640, 640, 3], "dtype": "uint8", "layout": "HWC", "color_space": "RGB",
         "preprocess_mode": "letterbox", "normalization": "embedded_dxcom_preprocessing",
         "scale": 1.0, "pad_x": 160, "pad_y": 0, "letterbox_pad_value": 114,
         "source_shape_hw": [640, 320], "contract_source": "test",
         **_prepared_input_audit_fields(
             task="detection", shape=[640, 640, 3], layout="HWC",
             normalization="embedded_dxcom_preprocessing",
         )},
    ]
    decoder = {
        "status": "ok", "pass": True, "family": "yolo26", "decoder_id": "yolo26_one2one_decoded_v1",
        "nms_included": False, "source_endpoint_semantics": "pre_nms_yolo26_xywh_class_scores",
        "source_endpoint_has_integrated_nms": False, "host_decoder_applied": True,
        "host_nms_applied": True, "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.45, "nms_max_detections": 300,
        "output_shape": [1, 84, 8400], "output_dtype": "float32",
    }
    for sample, audit in zip(samples, audits):
        image_path = root / "validation" / sample["image"]
        _bind_prepared_source(audit, image_path)
        records.append({
            "image": str(image_path),
            "detections": [{"class_id": 0, "confidence": 0.75, "box_xyxy": [10.0, 20.0, 40.0, 60.0]}],
            "num_detections": 1, "decoder_contract": decoder,
            "preprocessing_audit": audit,
        })
    path = root / "results/deepx_m1_full/detections.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"images": records}), encoding="utf-8")
    invariant = {key: value for key, value in audits[0].items() if key not in {"source_shape_hw", "scale", "pad_x", "pad_y", "prepared_tensor_binding"}}
    return {
        "enabled": True, "status": "ok", "task": "detection",
        "image_count": 2, "validated_image_count": 2, "error_count": 0,
        "input_size": 640, "input_contract": {"input": {"shape": [640, 640, 3], "layout": "HWC"}},
        "preprocessing_contract": {"pass": True, "all_samples_same_contract": True, "invariant_sample": invariant},
        "runtime_output_contract": {
            "pass": True, "all_samples_same_contract": True, "validated_observation_count": 2,
            "sample": {"outputs": [{"index": 0, "shape": [1, 84, 8400], "dtype": "float32"}]},
        },
        "decoder_postprocess_contract": {
            "pass": True, "all_samples_same_contract": True,
            "validated_contract_count": 2, "contracts": [decoder],
        },
        "detections_json": str(path.relative_to(root)), "mini_coco_ap50_full": 0.5,
    }


def _run(task: str) -> dict[str, Any]:
    model_id = "resnet50" if task == "classification" else "yolo26s"
    return {
        "id": "deepx_m1_full", "benchmark_task": task,
        "model_id": model_id,
        "model_suite": {"primary": model_id},
        "backend": "deepx_m1", "provider": "deepx_m1",
        "variant": "full", "variants": ["full"],
        "dxnn_path": "deepx/deepx_m1/full/model.dxnn",
        "contract_path": "deepx/deepx_m1/full/output_contract.json",
        "validation_images": "validation", "task_quality_gate": _policy(),
    }


def _set_decoded_detection_endpoint_attestation(
    root: Path, *, model_id: str = "yolo26s",
    include_coordinates: bool = True,
) -> None:
    contract_path = root / "deepx/deepx_m1/full/output_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    authoritative: dict[str, Any] = {
        "schema": "onnx-splitpoint/output-contract",
        "schema_version": 1,
        "model_id": model_id,
        "backend": "deepx_m1",
        "variant": "full",
        "task": "detection",
        "contract_status": "recorded",
        "endpoint_mode": "decoded",
        "host_tail_required": False,
        "postprocessing_required": False,
    }
    if include_coordinates:
        authoritative.update({
            "coordinate_format": "xyxy_score_class",
            "coordinate_space": "model_input_letterbox_xyxy_pixels",
        })
    payload.update({
        "model_id": model_id,
        "backend": "deepx_m1",
        "variant": "full",
        "endpoint_semantic_attestation": {
            "schema": (
                "onnx-splitpoint/deepx-endpoint-semantic-attestation"
            ),
            "schema_version": 1,
            "status": "attested",
            "pass": True,
            "model_id": model_id,
            "backend": "deepx_m1",
            "variant": "full",
            "endpoint_mode": "decoded",
            "host_tail_required": False,
            "postprocessing_required": False,
            "source_endpoint_has_integrated_nms": True,
            "authoritative_contract": authoritative,
            "authoritative_contract_sha256": hashlib.sha256(
                json.dumps(
                    authoritative, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    })
    contract_path.write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8",
    )


def _decoded_detection_semantic(
    root: Path, samples: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic = _detection_semantic(root, samples)
    semantic["runtime_output_contract"]["sample"] = {
        "outputs": [{
            "index": 0, "shape": [1, 300, 6], "dtype": "float32",
        }],
    }
    decoder = semantic["decoder_postprocess_contract"]["contracts"][0]
    decoder.update({
        "source_endpoint_semantics": "decoded_final_output",
        "source_endpoint_has_integrated_nms": True,
        "host_decoder_applied": False,
        "host_nms_applied": True,
    })
    return semantic


def _cpu_contract(
    *, source: Path, validation: Path, image_ids: list[str], ground_truth: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    namespace = _runner_functions([
        "_quality_json_safe", "_quality_file_sha256", "_quality_contract_sha256",
        "_portable_dataset_manifest_sha256",
        "_build_detection_quality_contract",
    ])
    return namespace["_build_detection_quality_contract"](
        model_path=source, validation_source=validation, image_ids=image_ids,
        ground_truth_by_image=ground_truth, image_scale="norm", letterbox=True,
        input_hw=(640, 640), input_dtype=np.float32,
        output_format="multiscale_head", output_names=["p3", "p4", "p5"],
        yolo_conf=0.25, yolo_iou=0.45, yolo_max_det=300,
        det_conf=0.25, det_iou=0.45, det_max_det=300,
        labels=["person"], runner_path=RUNNER,
        endpoint_attestor_sha256="a" * 64,
    )


def _completed_yolo26_hotloop_evidence(image_path: Path) -> dict[str, Any]:
    prepared = _prepared_input_audit_fields(
        task="detection", shape=[640, 640, 3], layout="HWC",
        normalization="embedded_dxcom_preprocessing",
    )
    outputs: dict[str, np.ndarray] = {}
    conv = 61
    for size in (80, 40, 20):
        outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
            (size, size, 4), dtype=np.float32,
        )
        outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
            (size, size, 80), -20.0, dtype=np.float32,
        )
        conv += 16
    frozen = build_frozen_postprocess_contract(
        model_id="yolo26s", outputs=outputs, input_hw=[640, 640],
        original_wh=[640, 640],
    )
    processor = FrozenDetectionPostprocessor(frozen)
    result = processor.process(outputs, original_wh=[640, 640])
    attestation = build_completed_detection_endpoint_attestation(
        frozen, result, completed_frames=2,
        postprocess_completed_frames=2,
    )
    return {
        **_performance_input_evidence(image_path, prepared),
        "status": "ok",
        "completed_frames": 2,
        "postprocess_included": True,
        "postprocess_completed_frames": 2,
        "postprocess_completion_verified": True,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "frozen_host_postprocess_contract": frozen,
        "completed_task_comparison_endpoint_contract": attestation[
            "completed_task_comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract_hash": attestation[
            "completed_task_comparison_endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": attestation[
            "completed_task_comparison_output_endpoint_id"
        ],
        "completed_task_endpoint_attestation": attestation,
    }


def test_portable_dataset_identity_ignores_cross_host_absolute_paths(
    tmp_path: Path,
) -> None:
    content_sha = hashlib.sha256(b"same-image-content").hexdigest()
    identity_rows = [{
        "sample_id": "5600",
        "relative_path": "val2017/000000005600.jpg",
        "sha256": content_sha,
        "class_name": "",
    }]
    items_sha = json_fingerprint(identity_rows)

    def write_manifest(path: Path, *, root: str, created_at: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": "onnx-splitpoint/dataset-manifest",
            "schema_version": 2,
            "created_at": created_at,
            "dataset_id": "coco2017-val",
            "task": "detection",
            "role": "validation",
            "split": "val2017",
            "root": root,
            "hash_mode": "content",
            "item_count": 1,
            "items": [{
                "sample_id": "5600",
                "image_id": 5600,
                "relative_path": "val2017/000000005600.jpg",
                "path": root + "/val2017/000000005600.jpg",
                "sha256": "sha256:" + content_sha,
            }],
            "items_identity_sha256": items_sha,
            "annotations": {
                "path": root + "/annotations/instances_val2017.json",
                "sha256": "a" * 64,
            },
            "labels": {"path": "", "sha256": ""},
        }, sort_keys=True), encoding="utf-8")
        return path

    management = write_manifest(
        tmp_path / "management/manifest.json",
        root="/home/kmika/datasets/coco",
        created_at="2026-07-30T10:00:00Z",
    )
    accelerator = write_manifest(
        tmp_path / "accelerator/manifest.json",
        root="/home/nx/datasets/coco",
        created_at="2026-07-31T10:00:00Z",
    )
    funcs = _runner_functions([
        "_quality_json_safe", "_quality_file_sha256",
        "_quality_contract_sha256",
        "_portable_dataset_manifest_sha256",
    ])
    generic_identity = funcs["_portable_dataset_manifest_sha256"]
    suite_identity = _suite_module()._deepx_portable_dataset_manifest_sha256
    assert generic_identity(management) == generic_identity(accelerator)
    assert suite_identity(management) == suite_identity(accelerator)
    assert generic_identity(management) == suite_identity(accelerator)

    changed = json.loads(accelerator.read_text(encoding="utf-8"))
    changed["items"][0]["sha256"] = "b" * 64
    changed["items_identity_sha256"] = json_fingerprint([{
        **identity_rows[0], "sha256": "b" * 64,
    }])
    accelerator.write_text(json.dumps(changed), encoding="utf-8")
    assert generic_identity(management) != generic_identity(accelerator)


def test_portable_dataset_identity_helpers_accept_only_normalized_or_legacy(
    tmp_path: Path,
) -> None:
    items = []
    for index in range(500):
        digest = hashlib.sha256(f"validation-image-{index}".encode()).hexdigest()
        relative_path = (
            f"n{index % 1000:08d}/ILSVRC2012_val_{index + 1:08d}.JPEG"
        )
        items.append({
            "class_name": f"n{index % 1000:08d}",
            "relative_path": relative_path,
            "sample_id": relative_path,
            "sha256": f"sha256:{digest}",
            "size_bytes": 1000 + index,
        })
    legacy_rows = [{
        "sample_id": item["sample_id"],
        "relative_path": item["relative_path"],
        "sha256": item["sha256"],
        "class_name": item["class_name"],
    } for item in items]
    normalized_rows = [{
        "sample_id": item["sample_id"],
        "relative_path": item["relative_path"],
        "sha256": str(item["sha256"]).removeprefix("sha256:"),
        "class_name": item["class_name"],
    } for item in items]
    legacy_items_sha = json_fingerprint(legacy_rows)
    normalized_items_sha = json_fingerprint(normalized_rows)
    assert legacy_items_sha != normalized_items_sha

    base_payload = {
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "dataset_id": "ilsvrc2012-val",
        "task": "classification",
        "role": "validation",
        "split": "val",
        "hash_mode": "content",
        "item_count": len(items),
        "items": items,
        "annotations": {"path": "", "sha256": ""},
        "labels": {
            "path": "/dataset/LOC_synset_mapping.txt",
            "sha256": f"sha256:{'a' * 64}",
        },
    }
    expected_portable = json_fingerprint({
        "schema": "onnx-splitpoint/portable-dataset-identity",
        "schema_version": 1,
        "dataset_id": "ilsvrc2012-val",
        "task": "classification",
        "role": "validation",
        "split": "val",
        "hash_mode": "content",
        "item_count": len(items),
        "items_identity_sha256": normalized_items_sha,
        "annotations_sha256": "",
        "labels_sha256": "a" * 64,
    })
    runner = _runner_functions([
        "_quality_json_safe", "_quality_file_sha256",
        "_quality_contract_sha256", "_portable_dataset_manifest_sha256",
    ])["_portable_dataset_manifest_sha256"]
    suite = _suite_module()._deepx_portable_dataset_manifest_sha256
    preflight = _preflight_module()

    def clone(value: Any) -> Any:
        return json.loads(json.dumps(value))

    def evaluate(payload: dict[str, Any], label: str) -> dict[str, str]:
        manifest = tmp_path / f"{label}.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "runner": runner(manifest),
            "suite": suite(manifest),
            "preflight": preflight._portable_dataset_manifest_sha256(payload),
        }

    def assert_blocked(payload: dict[str, Any], label: str) -> None:
        manifest = tmp_path / f"blocked-{label}.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError):
            runner(manifest)
        with pytest.raises(RuntimeError):
            suite(manifest)
        with pytest.raises(preflight.PreflightError):
            preflight._portable_dataset_manifest_sha256(payload)

    def legacy_hash(payload: dict[str, Any]) -> str:
        return json_fingerprint([{
            "sample_id": item.get("sample_id"),
            "relative_path": item.get("relative_path"),
            "sha256": item.get("sha256", ""),
            "class_name": item.get("class_name", ""),
        } for item in payload["items"]])

    observed: dict[str, list[str]] = {
        "runner": [], "suite": [], "preflight": [],
    }
    for label, declared in (
        ("legacy", legacy_items_sha),
        ("normalized", normalized_items_sha),
    ):
        payload = clone(base_payload)
        payload["items_identity_sha256"] = f"sha256:{declared}"
        result = evaluate(payload, label)
        for helper, identity in result.items():
            observed[helper].append(identity)
    assert observed == {
        "runner": [expected_portable, expected_portable],
        "suite": [expected_portable, expected_portable],
        "preflight": [expected_portable, expected_portable],
    }

    current = clone(base_payload)
    for item in current["items"]:
        item["sha256"] = str(item["sha256"]).removeprefix("sha256:")
    current["items_identity_sha256"] = f"sha256:{normalized_items_sha}"
    assert evaluate(current, "current-bare") == {
        helper: expected_portable
        for helper in ("runner", "suite", "preflight")
    }

    host_variant = clone(base_payload)
    host_variant.update({
        "root": "/another/host/imagenet",
        "created_at": "2030-01-02T03:04:05Z",
        "items_identity_sha256": f"sha256:{legacy_items_sha}",
    })
    assert evaluate(host_variant, "host-variant") == {
        helper: expected_portable
        for helper in ("runner", "suite", "preflight")
    }

    for label, declared in (
        ("missing", None),
        ("empty", ""),
        ("malformed", "sha256:not-a-digest"),
        ("unknown", f"sha256:{'f' * 64}"),
    ):
        payload = clone(base_payload)
        if declared is None:
            payload.pop("items_identity_sha256", None)
        else:
            payload["items_identity_sha256"] = declared
        assert_blocked(payload, f"declared-{label}")

    stale_mutations = {
        "digest": lambda payload: payload["items"][0].update({
            "sha256": f"sha256:{'b' * 64}",
        }),
        "path": lambda payload: payload["items"][0].update({
            "relative_path": "n00000000/renamed.JPEG",
        }),
        "class": lambda payload: payload["items"][0].update({
            "class_name": "n99999999",
        }),
        "order": lambda payload: payload.update({
            "items": list(reversed(payload["items"])),
        }),
    }
    for label, mutate in stale_mutations.items():
        payload = clone(base_payload)
        payload["items_identity_sha256"] = f"sha256:{legacy_items_sha}"
        mutate(payload)
        assert_blocked(payload, f"stale-{label}")

    arbitrary_legacy = clone(current)
    arbitrary_legacy["items_identity_sha256"] = f"sha256:{legacy_items_sha}"
    assert_blocked(arbitrary_legacy, "legacy-over-bare-digests")

    invalid_writer_shapes: dict[str, Any] = {
        "backslash": lambda item: item.update({
            "relative_path": str(item["relative_path"]).replace("/", "\\", 1),
        }),
        "null": lambda item: item.update({"class_name": None}),
        "uppercase-prefix": lambda item: item.update({
            "sha256": str(item["sha256"]).replace("sha256:", "SHA256:", 1),
        }),
        "spaced-digest": lambda item: item.update({
            "sha256": f" {item['sha256']}",
        }),
    }
    for label, mutate in invalid_writer_shapes.items():
        payload = clone(base_payload)
        mutate(payload["items"][0])
        payload["items_identity_sha256"] = f"sha256:{legacy_hash(payload)}"
        assert_blocked(payload, f"writer-shape-{label}")


def test_deepx_full_request_binds_completed_hotloop_endpoint(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": [
            {"bbox": [10, 20, 30, 40], "category_id": 1},
        ]},
        {"image": "tall.jpg", "annotations": [
            {"bbox": [10, 20, 30, 40], "category_id": 1},
        ]},
    ]
    root, _source, dxnn = _base_tree(
        tmp_path, task="detection", samples=samples,
    )
    completed = _completed_yolo26_hotloop_evidence(
        root / "validation/wide.jpg"
    )
    exported = module._deepx_export_central_quality_request(
        root, dxnn, _run("detection"),
        _detection_semantic(root, samples),
        root / "results/deepx_m1_full",
        completed_task_evidence=completed,
    )
    completed_hash = completed[
        "completed_task_comparison_endpoint_contract_hash"
    ]
    assert exported["quality_join_endpoint"] == (
        "completed_task_decoded_nms"
    )
    assert exported["completed_task_endpoint_contract_hash"] == (
        completed_hash
    )
    assert exported["producer_identity"][
        "completed_task_endpoint_contract_hash"
    ] == completed_hash
    assert exported["endpoint_contract_hash"] != completed_hash
    join_binding = exported["prepared_input_join_binding"]
    assert join_binding["binding_verified"] is True
    assert join_binding["source_image_id"] == "wide.jpg"
    assert join_binding["source_image_sha256"] == hashlib.sha256(
        (root / "validation/wide.jpg").read_bytes()
    ).hexdigest()
    assert join_binding["prepared_input_shape"] == [640, 640, 3]
    assert join_binding["prepared_input_dtype"] == "uint8"
    assert join_binding["prepared_input_layout"] == "HWC"
    assert exported["producer_identity"][
        "prepared_input_join_binding"
    ] == join_binding

    identity = EvaluationWorkflowRunner._quality_request_identity(
        Path(exported["request"]["path"]), model_id="yolo26s",
    )
    assert identity["identity_valid"] is True
    assert identity["completed_task_endpoint_contract_hash"] == (
        completed_hash
    )
    validated, _producer_sha = _validate_candidate_execution_contract(
        exported["producer_identity"], role="completed DeepX candidate",
        task="detection",
    )
    assert validated["completed_task_endpoint_contract_hash"] == (
        completed_hash
    )

    tampered = json.loads(json.dumps(exported["producer_identity"]))
    tampered["completed_task_endpoint_contract_hash"] = "f" * 64
    tampered.pop("producer_identity_sha256")
    tampered["producer_identity_sha256"] = json_fingerprint(tampered)
    with pytest.raises(
        QualityArtifactIntegrityError,
        match="completed-task endpoint binding is inconsistent",
    ):
        _validate_candidate_execution_contract(
            tampered, role="tampered completed DeepX candidate",
            task="detection",
        )


def test_v27516_deepx_decoded_endpoint_projects_verified_nested_semantics(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": []},
        {"image": "tall.jpg", "annotations": []},
    ]
    root, _source, dxnn = _base_tree(
        tmp_path, task="detection", samples=samples,
    )
    _set_decoded_detection_endpoint_attestation(root)

    exported = module._deepx_export_central_quality_request(
        root, dxnn, _run("detection"),
        _decoded_detection_semantic(root, samples),
        root / "results/deepx_m1_full",
        completed_task_evidence=_completed_yolo26_hotloop_evidence(
            root / "validation/wide.jpg"
        ),
    )

    assert exported["endpoint_contract_hash"] == (
        "d0972f0eb8de8e451288e18e2d2cd3497cf48cd3a02b854522aca5f2ae417e73"
    )
    identity = exported["producer_identity"]["endpoint"]["identity"]
    assert identity["semantic"] == {
        "coordinate_format": "xyxy_score_class",
        "coordinate_space": "model_input_letterbox_xyxy_pixels",
    }


def test_v27516_deepx_export_rejects_internally_valid_swapped_model_attestation(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": []},
        {"image": "tall.jpg", "annotations": []},
    ]
    root, _source, dxnn = _base_tree(
        tmp_path, task="detection", samples=samples,
    )
    _set_decoded_detection_endpoint_attestation(
        root, model_id="other_detector",
    )

    with pytest.raises(
        RuntimeError,
        match="deepx_quality_endpoint_semantic_model_id_conflict",
    ):
        module._deepx_export_central_quality_request(
            root, dxnn, _run("detection"),
            _decoded_detection_semantic(root, samples),
            root / "results/deepx_m1_full",
            completed_task_evidence=_completed_yolo26_hotloop_evidence(
                root / "validation/wide.jpg"
            ),
        )


def test_v27516_deepx_decoded_endpoint_requires_nested_coordinate_semantics(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": []},
        {"image": "tall.jpg", "annotations": []},
    ]
    root, _source, dxnn = _base_tree(
        tmp_path, task="detection", samples=samples,
    )
    _set_decoded_detection_endpoint_attestation(
        root, include_coordinates=False,
    )

    with pytest.raises(
        RuntimeError,
        match="deepx_quality_endpoint_semantic_decoded_coordinates_missing",
    ):
        module._deepx_export_central_quality_request(
            root, dxnn, _run("detection"),
            _decoded_detection_semantic(root, samples),
            root / "results/deepx_m1_full",
            completed_task_evidence=_completed_yolo26_hotloop_evidence(
                root / "validation/wide.jpg"
            ),
        )


def test_v27516_deepx_suite_run_identity_is_bound_before_export() -> None:
    module = _suite_module()
    benchmark_set = {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 2,
        "model_name": "yolo26s",
        "model": "models/yolo26s.onnx",
        "artifact_manifest": {
            "schema": "onnx-splitpoint/benchmark-set",
            "schema_version": 2,
            "files": {"models": ["models/yolo26s.onnx"]},
            "counts": {"models": 1},
        },
    }
    run = {
        "backend": "deepx_m1",
        "provider": "deepx_m1",
        "stage1": {"backend": "deepx_m1"},
        "stage2": {"backend": "deepx_m1"},
        "variants": ["full"],
    }

    assert module._deepx_verified_suite_run_identity(
        benchmark_set, run,
    ) == {
        "model_id": "yolo26s",
        "backend": "deepx_m1",
        "variant": "full",
    }

    swapped = json.loads(json.dumps(benchmark_set))
    swapped["model_name"] = "yolov7_paper"
    with pytest.raises(RuntimeError, match="deepx_suite_model_identity_invalid"):
        module._deepx_verified_suite_run_identity(swapped, run)

    conflicting_run = json.loads(json.dumps(run))
    conflicting_run["provider"] = "another_backend"
    with pytest.raises(
        RuntimeError,
        match="deepx_suite_run_identity_missing_or_conflicting",
    ):
        module._deepx_verified_suite_run_identity(
            benchmark_set, conflicting_run,
        )


def test_deepx_quality_rejects_one_byte_performance_tensor_drift(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": []},
        {"image": "tall.jpg", "annotations": []},
    ]
    root, _source, dxnn = _base_tree(
        tmp_path, task="detection", samples=samples,
    )
    semantic = _detection_semantic(root, samples)
    records_path = root / str(semantic["detections_json"])
    records_payload = json.loads(records_path.read_text(encoding="utf-8"))
    byte_count = 640 * 640 * 3
    quality_tensor = bytes(byte_count)
    quality_sha = hashlib.sha256(quality_tensor).hexdigest()
    for record in records_payload["images"]:
        record["preprocessing_audit"]["prepared_tensor_binding"][
            "prepared_input_sha256"
        ] = quality_sha
    records_path.write_text(json.dumps(records_payload), encoding="utf-8")

    performance = _completed_yolo26_hotloop_evidence(
        root / "validation/wide.jpg"
    )
    one_byte_drift = bytearray(quality_tensor)
    one_byte_drift[-1] ^= 0x01
    performance["prepared_input_sha256"] = hashlib.sha256(
        bytes(one_byte_drift)
    ).hexdigest()

    with pytest.raises(
        RuntimeError,
        match="deepx_quality_performance_prepared_input_byte_mismatch",
    ):
        module._deepx_export_central_quality_request(
            root, dxnn, _run("detection"), semantic,
            root / "results/deepx_m1_full",
            completed_task_evidence=performance,
        )


def test_deepx_full_completed_request_merges_exactly_once_offline(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": [
            {"bbox": [10, 20, 30, 40], "category_id": 1},
        ]},
        {"image": "tall.jpg", "annotations": [
            {"bbox": [10, 20, 30, 40], "category_id": 1},
        ]},
    ]
    root, _source, dxnn = _base_tree(
        tmp_path / "producer", task="detection", samples=samples,
    )
    exported = module._deepx_export_central_quality_request(
        root, dxnn, _run("detection"),
        _detection_semantic(root, samples),
        root / "results/deepx_m1_full",
        completed_task_evidence=_completed_yolo26_hotloop_evidence(
            root / "validation/wide.jpg"
        ),
    )

    run_root = tmp_path / "run"
    setup_id = "orin_nx_deepx_m1_01"
    request_path = (
        run_root / "quality_inputs" / setup_id / "results" / "full"
        / "results_deepx_m1_full" / "task_quality_inputs"
        / "full_request.json"
    )
    request_path.parent.mkdir(parents=True)
    source_request = Path(exported["request"]["path"])
    request_path.write_bytes(source_request.read_bytes())
    request_identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path, model_id="yolo26s",
    )
    assert request_identity["identity_valid"] is True
    assert request_identity["setup_id"] == setup_id

    row_request = json.loads(request_path.read_text(encoding="utf-8"))
    row_request["request"] = {
        "path": str(request_path),
        "sha256": request_identity["source_request_sha256"],
    }
    pending_gate = {
        "schema": "onnx-splitpoint/task-quality-gate",
        "schema_version": 2,
        "task": "detection",
        "variant": "full",
        "status": "pending_central_evaluation",
        "decision": "pending_central_evaluation",
        "quality_input_request": row_request,
    }
    row = {
        "model_id": "yolo26s",
        "case_id": "full",
        "run_id": "deepx_m1_full",
        "source_tag": "deepx_m1_full_auto",
        "quality_source_run_id": "deepx_m1_full",
        "quality_source_setup_ids": [setup_id],
        "variant": "full",
        "primary_variant": "full",
        "quality_source_variant": "full",
        "task": "detection",
        "task_quality_gates_by_variant": {"full": pending_gate},
        "task_quality_gate": pending_gate,
        "task_quality_policy": _policy(),
        "quality_evaluation_pending": True,
        "technical_status": "completed",
        "endpoint_contract_hash": request_identity[
            "endpoint_contract_hash"
        ],
        "runtime_precision_identity": request_identity[
            "runtime_precision_identity"
        ],
    }
    normalized = (
        run_root / "models/yolo26s/benchmark_results"
        / "normalized_results.json"
    )
    normalized.parent.mkdir(parents=True)
    normalized.write_text(
        json.dumps({"results": [row]}), encoding="utf-8",
    )

    central_result = {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "schema_version": 1,
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "model_id": "yolo26s",
        "case_id": "full",
        "run_id": "deepx_m1_full",
        "source_run_id": "deepx_m1_full",
        "source_setup_id": setup_id,
        "variant": "full",
        "task": "detection",
        "source_request": str(request_path.relative_to(run_root)),
        "source_request_sha256": request_identity[
            "source_request_sha256"
        ],
        "endpoint_contract_hash": request_identity[
            "endpoint_contract_hash"
        ],
        "completed_task_endpoint_contract_hash": request_identity[
            "completed_task_endpoint_contract_hash"
        ],
        "completed_task_output_endpoint_id": request_identity[
            "completed_task_output_endpoint_id"
        ],
        "quality_join_endpoint": "completed_task_decoded_nms",
        "runtime_precision_identity": request_identity[
            "runtime_precision_identity"
        ],
        "request_identity": request_identity,
        "n": 2,
        "primary": {
            "metric": "coco_ap_50_95",
            "candidate": 0.5,
            "reference": 0.5,
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "margin": 0.01,
        },
        "guardrails": {},
    }
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_root
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {
                "execution_location": "central_management",
                "workers": 1,
            },
        },
    }
    merge = runner._merge_central_quality_results([central_result])
    assert merge["matched_primary_result_count"] == 1
    assert merge["unmatched_result_count"] == 0
    merged = json.loads(normalized.read_text(encoding="utf-8"))[
        "results"
    ][0]
    assert merged["quality_evaluation_pending"] is False
    assert merged["central_quality_request_identity"][
        "completed_task_endpoint_contract_hash"
    ] == request_identity["completed_task_endpoint_contract_hash"]
    assert merged["central_quality_request_identity"][
        "quality_join_endpoint"
    ] == "completed_task_decoded_nms"

    mismatch_root = tmp_path / "mismatch"
    mismatch_path = (
        mismatch_root / "models/yolo26s/benchmark_results"
        / "normalized_results.json"
    )
    mismatch_path.parent.mkdir(parents=True)
    mismatch_path.write_text(
        json.dumps({"results": [row]}), encoding="utf-8",
    )
    mismatched = json.loads(json.dumps(central_result))
    mismatched["completed_task_endpoint_contract_hash"] = "f" * 64
    mismatched["request_identity"][
        "completed_task_endpoint_contract_hash"
    ] = "f" * 64
    mismatch_runner = object.__new__(EvaluationWorkflowRunner)
    mismatch_runner.run_dir = mismatch_root
    mismatch_runner.profile_payload = runner.profile_payload
    mismatch = mismatch_runner._merge_central_quality_results(
        [mismatched]
    )
    assert mismatch["matched_primary_result_count"] == 0
    assert mismatch["unmatched_result_count"] == 1
    assert mismatch["unmatched_results"][0]["join_status"] == (
        "no_exact_row"
    )


def test_native_full_validator_joins_quality_on_completed_not_raw_endpoint(
) -> None:
    validator_path = ROOT / "scripts/native_producer_validate_visualize.py"
    spec = importlib.util.spec_from_file_location(
        "_v273_yolo26_completed_quality_validator", validator_path,
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    policy = validator.AccuracyGatePolicy()
    completed = _completed_yolo26_hotloop_evidence(SUITE)
    physical_hash = "a" * 64
    request_sha = "b" * 64
    model_sha = "c" * 64
    dataset_sha = "d" * 64
    image_ids_sha = "e" * 64
    ground_truth_sha = "f" * 64
    quality_sha = "1" * 64
    preprocessing_identity = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    preprocessing_sha = preprocessing_contract_sha256(
        preprocessing_identity
    )
    decoder_sha = "3" * 64
    nms_sha = "4" * 64
    quality_record_sha = "6" * 64
    completed_hash = completed[
        "completed_task_comparison_endpoint_contract_hash"
    ]
    completed_id = completed[
        "completed_task_comparison_output_endpoint_id"
    ]
    row = {
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "case": "full",
        "setup_id": "orin_nx_deepx_m1_01",
        "task": "detection",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": physical_hash,
        "runtime_precision_identity": "deepx_dxnn_sha256:" + "5" * 64,
        **completed,
    }
    identity = {
        "schema_version": 3,
        "identity_valid": True,
        "model_id": "yolo26s",
        "task": "detection",
        "case_id": "full",
        "source_run_id": "deepx_m1_full",
        "setup_id": "orin_nx_deepx_m1_01",
        "variant": "full",
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "validation_image_ids_sha256": image_ids_sha,
        "validation_ground_truth_sha256": ground_truth_sha,
        "endpoint_contract_hash": physical_hash,
        "completed_task_endpoint_contract": completed[
            "completed_task_comparison_endpoint_contract"
        ],
        "completed_task_endpoint_contract_hash": completed_hash,
        "completed_task_output_endpoint_id": completed_id,
        "quality_join_endpoint": "completed_task_decoded_nms",
        "runtime_precision_identity": row[
            "runtime_precision_identity"
        ],
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "task_quality_policy_sha256": policy.sha256(),
    }
    result = {
        "model_id": "yolo26s",
        "task": "detection",
        "case_id": "full",
        "source_run_id": "deepx_m1_full",
        "source_setup_id": "orin_nx_deepx_m1_01",
        "variant": "full",
        "status": "completed",
        "technical_status": "completed",
        "decision": "pass",
        "policy_sha256": policy.sha256(),
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "validation_image_ids_sha256": image_ids_sha,
        "validation_ground_truth_sha256": ground_truth_sha,
        "endpoint_contract_hash": physical_hash,
        "completed_task_endpoint_contract": identity[
            "completed_task_endpoint_contract"
        ],
        "completed_task_endpoint_contract_hash": completed_hash,
        "completed_task_output_endpoint_id": completed_id,
        "quality_join_endpoint": "completed_task_decoded_nms",
        "runtime_precision_identity": row[
            "runtime_precision_identity"
        ],
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "request_identity": identity,
        "primary": {
            "metric": "coco_ap_50_95",
            "delta": 0.0,
            "ci_low": 0.0,
            "margin": 0.01,
        },
    }
    binding = {
        "schema": (
            "onnx-splitpoint/native-full-quality-request-binding"
        ),
        "schema_version": 1,
        "eval_run_id": "test-run",
        "setup_id": "orin_nx_deepx_m1_01",
        "comparison_backend": "deepx",
        "backend": "native_full_deepx",
        "model_id": "yolo26s",
        "source_run_id": "deepx_m1_full",
        "source_case_id": "full",
        "task": "detection",
        "variant": "full",
        "runtime_precision_identity": row[
            "runtime_precision_identity"
        ],
        "endpoint_contract_hash": physical_hash,
        "source_request": "request.json",
        "source_request_file": "/test/request.json",
        "source_request_file_sha256": request_sha,
        "central_quality_result_sha256": (
            validator._canonical_json_sha256(result)
        ),
        "preprocessing_contract": {
            "identity": preprocessing_identity,
            "sha256": preprocessing_sha,
        },
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "validation_dataset_image_ids_sha256": image_ids_sha,
        "validation_dataset_ground_truth_sha256": ground_truth_sha,
        "task_quality_policy_sha256": policy.sha256(),
        "runtime_quality_gate_policy_sha256": policy.sha256(),
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "quality_record_endpoint_contract_sha256": (
            quality_record_sha
        ),
    }
    binding["binding_sha256"] = validator._canonical_json_sha256(
        binding
    )
    binding_set_sha = "7" * 64
    row.update({
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "validation_dataset_image_ids_sha256": image_ids_sha,
        "validation_dataset_ground_truth_sha256": ground_truth_sha,
        "task_quality_policy_sha256": policy.sha256(),
        "runtime_quality_gate_policy_sha256": policy.sha256(),
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "quality_record_endpoint_contract_sha256": quality_record_sha,
        "runtime_preprocessing_identity": preprocessing_identity,
        "quality_request_binding_status": "verified_exact",
        "quality_request_binding": binding,
        "quality_request_binding_sha256": binding[
            "binding_sha256"
        ],
        "quality_request_binding_set_sha256": binding_set_sha,
        "full_command_contract": {
            "quality_request_binding": binding,
            "quality_request_binding_sha256": binding[
                "binding_sha256"
            ],
            "quality_request_binding_set_sha256": binding_set_sha,
            "runtime_preprocessing_binding": {
                "identity": preprocessing_identity,
                "sha256": preprocessing_sha,
            },
        },
    })
    pristine_row = json.loads(json.dumps(row))
    validator._bind_central_quality_evidence(row, [result], policy)
    assert row["central_quality_evidence_verified"] is True
    assert row["precision_quality_binding_verified"] is True
    assert row["central_quality_binding_candidate_count"] == 1
    assert row["quality_join_endpoint"] == "completed_task_decoded_nms"
    assert row["quality_join_endpoint_contract_hash"] == completed_hash
    assert row["physical_endpoint_contract_hash"] == physical_hash

    rejected_row = pristine_row
    mismatched = json.loads(json.dumps(result))
    mismatched["completed_task_endpoint_contract_hash"] = "6" * 64
    mismatched["request_identity"][
        "completed_task_endpoint_contract_hash"
    ] = "6" * 64
    validator._bind_central_quality_evidence(
        rejected_row, [mismatched], policy,
    )
    assert rejected_row["central_quality_evidence_verified"] is False
    assert rejected_row["central_quality_binding_status"] == (
        "no_exact_identity_match"
    )


def test_deepx_detection_export_keeps_original_image_coordinates_without_double_transform(tmp_path: Path) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": [{"bbox": [10, 20, 30, 40], "category_id": 1}]},
        {"image": "tall.jpg", "annotations": [{"bbox": [10, 20, 30, 40], "category_id": 1}]},
    ]
    root, source, dxnn = _base_tree(tmp_path, task="detection", samples=samples)
    run = _run("detection")
    run.pop("model_id")
    exported = module._deepx_export_central_quality_request(
        root, dxnn, run, _detection_semantic(root, samples),
        root / "results/deepx_m1_full",
        completed_task_evidence=_completed_yolo26_hotloop_evidence(
            root / "validation/wide.jpg"
        ),
    )
    candidate_path = Path(exported["request"]["path"]).parent / exported["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    by_id = {row["image_id"]: row for row in candidate["records"]}

    # DeepX decoding has already inverted its letterbox transform.  The
    # central exporter must therefore preserve original-image pixels instead
    # of projecting both GT and predictions into the 640x640 input again.
    expected_box = {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0}
    for image_id in ("wide.jpg", "tall.jpg"):
        ground_truth_box = by_id[image_id]["ground_truth"][0]
        candidate_box = by_id[image_id]["candidate"][0]
        for coordinate, expected in expected_box.items():
            assert ground_truth_box[coordinate] == pytest.approx(expected)
            assert candidate_box[coordinate] == pytest.approx(expected)
    assert by_id["wide.jpg"]["ground_truth"][0]["score"] == 1.0
    assert by_id["wide.jpg"]["ground_truth"][0]["class_name"] == "person"
    transforms = {row["image_id"]: row for row in candidate["per_image_transforms"]}
    assert transforms["wide.jpg"]["canonical_record_coordinate_space"] == "original_image_xyxy_pixels"
    assert transforms["tall.jpg"]["canonical_record_coordinate_space"] == "original_image_xyxy_pixels"

    producer = exported["producer_identity"]
    endpoint = producer["quality_record_endpoint"]["identity"]
    assert endpoint["canonical_coordinate_space"] == "original_image_xyxy_pixels"
    assert endpoint["source_endpoint"]["has_integrated_nms"] is False
    assert endpoint["host_decoder_nms"]["decoder_applied"] is True
    assert endpoint["host_decoder_nms"]["nms_applied"] is True
    assert endpoint["host_decoder_nms"]["iou_threshold"] == 0.45
    assert producer["implementation_runner_sha256"] == hashlib.sha256(SUITE.read_bytes()).hexdigest()
    assert producer["precision"]["identity"]["precision_semantics"] == "opaque_vendor_compiled_artifact_identity"
    assert producer["precision"]["identity"]["declared_precision"] is None
    assert producer["runtime_precision_identity"].startswith("deepx_dxnn_sha256:")
    assert producer["model"]["source_onnx_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    native_endpoint = runtime_output_contract(
        "detection", {"output_0": np.zeros((1, 84, 8400), dtype=np.float32)},
        raw_fallback=True,
        declared_contract=_authoritative_endpoint_contract(root, task="detection"),
    )
    assert producer["endpoint"]["identity"]["stage"] == "raw_head"
    assert producer["endpoint_contract_hash"] == native_endpoint["endpoint_contract_hash"]
    for field in (
        "quality_contract_sha256", "preprocessing_contract_sha256",
        "decoder_contract_sha256", "nms_contract_sha256",
    ):
        assert len(producer[field]) == 64
        assert exported[field] == producer[field]
    request_identity = EvaluationWorkflowRunner._quality_request_identity(
        Path(exported["request"]["path"]), model_id="yolo26s",
    )
    assert request_identity["identity_valid"] is True
    assert request_identity["quality_contract_sha256"] == producer["quality_contract_sha256"]
    assert request_identity["decoder_contract_sha256"] == producer["decoder_contract_sha256"]
    assert request_identity["nms_contract_sha256"] == producer["nms_contract_sha256"]
    conflicting_request = json.loads(json.dumps(exported))
    conflicting_request["preprocessing_contract_sha256"] = "f" * 64
    conflicting_identity = EvaluationWorkflowRunner._quality_request_identity(
        Path(exported["request"]["path"]), model_id="yolo26s",
        manifest=conflicting_request,
    )
    assert conflicting_identity["identity_valid"] is False
    assert "preprocessing_contract_sha256_conflict" in conflicting_identity["identity_errors"]
    malformed_request = json.loads(json.dumps(exported))
    malformed_request["quality_contract_sha256"] = "not-a-sha256"
    malformed_identity = EvaluationWorkflowRunner._quality_request_identity(
        Path(exported["request"]["path"]), model_id="yolo26s",
        manifest=malformed_request,
    )
    assert malformed_identity["identity_valid"] is False
    assert "quality_contract_sha256_malformed" in malformed_identity["identity_errors"]

    ground_truth = {row["image_id"]: row["ground_truth"] for row in candidate["records"]}
    cpu_contract = _cpu_contract(
        source=source, validation=root / "validation",
        image_ids=sorted(ground_truth), ground_truth=ground_truth,
    )
    reference = {
        "schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "detection", "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True,
        "quality_contract": cpu_contract,
        "quality_contract_sha256": cpu_contract["quality_contract_sha256"],
        "records": [
            {"image_id": row["image_id"], "ground_truth": row["ground_truth"], "reference": row["candidate"]}
            for row in candidate["records"]
        ],
    }
    reference_path = tmp_path / "cpu_reference.json"
    reference_path.write_text(json.dumps(reference, sort_keys=True), encoding="utf-8")
    loaded = quality_request_from_manifest(exported["request"]["path"], reference_artifact=reference_path)
    assert loaded.metric_gate_config["producer_identity_sha256"] == producer["producer_identity_sha256"]
    assert cpu_contract["decoder"]["identity"]["implementation_runner_sha256"] != producer["implementation_runner_sha256"]

    request_path = Path(exported["request"]["path"])
    tampered_request = json.loads(request_path.read_text(encoding="utf-8"))
    tampered_request["quality_contract_sha256"] = "f" * 64
    request_path.write_text(json.dumps(tampered_request, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request top-level quality_contract_sha256 differs from producer identity",
    ):
        quality_request_from_manifest(request_path, reference_artifact=reference_path)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("source", "exact unique member of per-image transforms"),
        ("bytes", "prepared-input evidence is invalid"),
        ("shape", "prepared-input evidence is invalid"),
        ("layout", "prepared-input evidence is invalid"),
    ),
)
def test_deepx_resealed_arbitrary_join_is_not_a_quality_member(
    tmp_path: Path, mutation: str, expected_error: str,
) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": []},
        {"image": "tall.jpg", "annotations": []},
    ]
    root, source, dxnn = _base_tree(
        tmp_path, task="detection", samples=samples,
    )
    exported = module._deepx_export_central_quality_request(
        root, dxnn, _run("detection"),
        _detection_semantic(root, samples),
        root / "results/deepx_m1_full",
        completed_task_evidence=_completed_yolo26_hotloop_evidence(
            root / "validation/wide.jpg"
        ),
    )
    request_path = Path(exported["request"]["path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_path = request_path.parent / request["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    ground_truth = {
        row["image_id"]: row["ground_truth"]
        for row in candidate["records"]
    }
    cpu_contract = _cpu_contract(
        source=source,
        validation=root / "validation",
        image_ids=sorted(ground_truth),
        ground_truth=ground_truth,
    )
    reference = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "provenance_required": True,
        "quality_contract": cpu_contract,
        "quality_contract_sha256": cpu_contract[
            "quality_contract_sha256"
        ],
        "records": [
            {
                "image_id": row["image_id"],
                "ground_truth": row["ground_truth"],
                "reference": row["candidate"],
            }
            for row in candidate["records"]
        ],
    }
    reference_path = tmp_path / "cpu_reference.json"
    reference_path.write_text(
        json.dumps(reference, sort_keys=True), encoding="utf-8",
    )
    quality_request_from_manifest(
        request_path, reference_artifact=reference_path,
    )

    arbitrary_join = json.loads(json.dumps(
        request["prepared_input_join_binding"]
    ))
    if mutation == "source":
        arbitrary_join["source_image_id"] = "valid-looking-ghost.jpg"
        arbitrary_join["source_image_sha256"] = "f" * 64
    elif mutation == "bytes":
        arbitrary_join["prepared_input_bytes"] += 1
    elif mutation == "shape":
        arbitrary_join["prepared_input_shape"] = [320, 1280, 3]
    elif mutation == "layout":
        arbitrary_join["prepared_input_layout"] = "CHW"
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(mutation)
    join_sha = json_fingerprint(arbitrary_join)

    quality_contract = json.loads(json.dumps(request["quality_contract"]))
    quality_contract["prepared_input_evidence"][
        "performance_quality_input_binding"
    ] = arbitrary_join
    quality_contract["prepared_input_evidence"][
        "performance_quality_input_binding_sha256"
    ] = join_sha
    quality_contract.pop("quality_contract_sha256")
    quality_sha = json_fingerprint(quality_contract)
    quality_contract["quality_contract_sha256"] = quality_sha

    for payload in (request, candidate):
        payload["quality_contract"] = quality_contract
        payload["quality_contract_sha256"] = quality_sha
        payload["prepared_input_join_binding"] = arbitrary_join
        payload["prepared_input_join_binding_sha256"] = join_sha
        producer = json.loads(json.dumps(payload["producer_identity"]))
        evidence = json.loads(json.dumps(
            producer["prepared_input_evidence"]
        ))
        evidence["performance_quality_input_binding"] = arbitrary_join
        evidence[
            "performance_quality_input_binding_sha256"
        ] = join_sha
        producer["prepared_input_evidence"] = evidence
        producer["prepared_input_join_binding"] = arbitrary_join
        producer["prepared_input_join_binding_sha256"] = join_sha
        producer["quality_contract"] = quality_contract
        producer["quality_contract_sha256"] = quality_sha
        producer.pop("producer_identity_sha256")
        producer_sha = json_fingerprint(producer)
        producer["producer_identity_sha256"] = producer_sha
        payload["producer_identity"] = producer
        payload["producer_identity_sha256"] = producer_sha

    candidate_bytes = json.dumps(
        candidate, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    candidate_path.write_bytes(candidate_bytes)
    request["candidate"]["sha256"] = hashlib.sha256(
        candidate_bytes
    ).hexdigest()
    request["candidate"]["size_bytes"] = len(candidate_bytes)
    request_path.write_text(
        json.dumps(request, sort_keys=True), encoding="utf-8",
    )

    with pytest.raises(
        QualityArtifactIntegrityError,
        match=expected_error,
    ):
        quality_request_from_manifest(
            request_path, reference_artifact=reference_path,
        )


def test_cpu_detection_canonicalization_inverts_odd_letterbox_padding_exactly_once(
    tmp_path: Path,
) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    namespace = _runner_functions([
        "_original_image_geometry", "_detections_to_original_image_space",
    ])
    namespace["Image"] = pil_image

    image_path = tmp_path / "odd_letterbox.png"
    pil_image.new("RGB", (640, 361), color=(0, 0, 0)).save(image_path)
    geometry = namespace["_original_image_geometry"](
        image_path, fallback_hw=(640, 640), letterbox=True,
    )
    assert geometry == (640, 361, 1.0, 1.0, 0, 139)

    original = {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0,
                "score": 0.75, "class_id": 0, "class_name": "person"}
    model_input_record = {
        **original,
        "y1": original["y1"] + 139.0,
        "y2": original["y2"] + 139.0,
    }
    converted = namespace["_detections_to_original_image_space"](
        [model_input_record], img_file=image_path,
        img_hw=(640, 640), letterbox=True,
    )[0]
    for coordinate in ("x1", "y1", "x2", "y2"):
        assert converted[coordinate] == pytest.approx(original[coordinate])
    assert converted["score"] == original["score"]
    assert converted["class_id"] == original["class_id"]

    # Applying the inverse a second time would move the already-original box.
    # The DeepX export test above asserts that its decoded records bypass this
    # CPU-only conversion and therefore remain in original-image space.
    double_converted = namespace["_detections_to_original_image_space"](
        [converted], img_file=image_path,
        img_hw=(640, 640), letterbox=True,
    )[0]
    assert double_converted["y1"] != pytest.approx(original["y1"])


def test_deepx_detection_decoder_attests_pre_nms_and_rejects_shape_only_bn6(tmp_path: Path) -> None:
    module = _suite_module()
    raw = np.zeros((1, 84, 8), dtype=np.float32)
    _, contract = module._deepx_detection_decode(
        root=tmp_path / "yolo26s", run={"id": "deepx_m1_full"},
        contract={"postprocessing": {"type": "yolo_host_decode", "decoder_id": "d"}},
        outputs=[raw], orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert contract["source_endpoint_has_integrated_nms"] is False
    assert contract["host_nms_applied"] is True
    assert contract["nms_included"] is False

    _, bn6_contract = module._deepx_detection_decode(
        root=tmp_path / "unknown", run={"id": "deepx_m1_full"}, contract={},
        outputs=[np.zeros((1, 3, 6), dtype=np.float32)],
        orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert bn6_contract["pass"] is False
    assert bn6_contract["status"] == "bn6_endpoint_not_explicitly_attested"


def test_deepx_candidate_tampering_and_missing_runtime_endpoint_fail_closed(tmp_path: Path) -> None:
    module = _suite_module()
    samples = [
        {"image": "wide.jpg", "annotations": [{"bbox": [10, 20, 30, 40], "category_id": 1}]},
        {"image": "tall.jpg", "annotations": [{"bbox": [10, 20, 30, 40], "category_id": 1}]},
    ]
    root, _source, dxnn = _base_tree(tmp_path, task="detection", samples=samples)
    semantic = _detection_semantic(root, samples)
    missing = dict(semantic)
    missing["runtime_output_contract"] = {"pass": False}
    with pytest.raises(RuntimeError, match="runtime_output_contract_not_attested"):
        module._deepx_export_central_quality_request(
            root, dxnn, _run("detection"), missing, root / "results/deepx_m1_full",
            completed_task_evidence=_completed_yolo26_hotloop_evidence(
                root / "validation/wide.jpg"
            ),
        )
    assert not (root / "results/deepx_m1_full/task_quality_inputs/full_request.json").exists()

    exported = module._deepx_export_central_quality_request(
        root, dxnn, _run("detection"), semantic, root / "results/deepx_m1_full",
        completed_task_evidence=_completed_yolo26_hotloop_evidence(
            root / "validation/wide.jpg"
        ),
    )
    request_path = Path(exported["request"]["path"])
    candidate_path = request_path.parent / exported["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["producer_identity"]["quality_record_endpoint"]["identity"]["host_decoder_nms"]["iou_threshold"] = 0.99
    encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    candidate_path.write_bytes(encoded)
    with pytest.raises(QualityArtifactIntegrityError, match="candidate execution contract SHA-256 mismatch"):
        _validate_candidate_execution_contract(
            candidate["producer_identity"], role="candidate", task="detection",
        )


def test_deepx_classification_exports_real_topk_with_opaque_dxnn_identity(tmp_path: Path) -> None:
    module = _suite_module()
    samples = [{"image": "a.jpg", "label_id": 3, "label_name": "three"}]
    root, _source, dxnn = _base_tree(tmp_path, task="classification", samples=samples)
    path = root / "results/deepx_m1_full/classification_topk.json"
    path.parent.mkdir(parents=True)
    audit = {
        "shape": [224, 224, 3], "dtype": "uint8", "layout": "HWC", "color_space": "RGB",
        "preprocess_mode": "resize", "normalization": "embedded_dxcom_preprocessing",
        "scale": 0.56, "pad_x": 0, "pad_y": 0, "letterbox_pad_value": 0,
        "source_shape_hw": [300, 400], "contract_source": "test",
        **_prepared_input_audit_fields(
            task="classification", shape=[224, 224, 3], layout="HWC",
            normalization="embedded_dxcom_preprocessing",
        ),
    }
    _bind_prepared_source(audit, root / "validation/a.jpg")
    performance_evidence = _performance_input_evidence(
        root / "validation/a.jpg", audit,
    )
    path.write_text(json.dumps({"images": [{
        "image": str(root / "validation/a.jpg"), "label_id": 3,
        # This name comes from an unbound vendor label map and must not enter
        # the candidate identity.  The manifest-bound name is "three".
        "label_name": "external-vendor-three",
        "top1": 3, "top5": [3, 1, 2, 4, 5], "scores": [1.0, 0.5, 0.4, 0.3, 0.2],
        "top1_correct": True, "top5_correct": True, "preprocessing_audit": audit,
    }]}), encoding="utf-8")
    invariant = {key: value for key, value in audit.items() if key not in {"source_shape_hw", "scale", "pad_x", "pad_y", "prepared_tensor_binding"}}
    semantic = {
        "status": "ok", "task": "classification", "image_count": 1,
        "validated_image_count": 1, "error_count": 0, "input_size": 224,
        "input_contract": {"input": {"shape": [224, 224, 3], "layout": "HWC"}},
        "preprocessing_contract": {"pass": True, "all_samples_same_contract": True, "invariant_sample": invariant},
        "runtime_output_contract": {
            "pass": True, "all_samples_same_contract": True, "validated_observation_count": 1,
            "sample": {"outputs": [{"index": 0, "shape": [1, 1000], "dtype": "float32"}]},
        },
        "classification_source_endpoint_contract": {
            "status": "ok", "pass": True,
            "stage": "classification_logits",
            "validated_observation_count": 1,
        },
        "classification_topk_json": str(path.relative_to(root)),
        "classification_top1_accuracy": 1.0, "classification_top5_accuracy": 1.0,
    }
    exported = module._deepx_export_central_quality_request(
        root, dxnn, _run("classification"), semantic, root / "results/deepx_m1_full",
        completed_task_evidence=performance_evidence,
    )
    candidate_path = Path(exported["request"]["path"]).parent / exported["candidate"]["path"]
    record = json.loads(candidate_path.read_text(encoding="utf-8"))["records"][0]
    assert record["label_id"] == 3
    assert record["label_name"] == "three"
    assert record["candidate"] == {
        "top1": 3, "top1_hit": True, "top5": [3, 1, 2, 4, 5], "top5_hit": True,
    }
    assert exported["producer_identity"]["quality_record_endpoint"]["identity"]["canonical_record_endpoint"] == "classification_topk_hits"
    producer = exported["producer_identity"]
    native_endpoint = runtime_output_contract(
        "classification", {"logits": np.zeros((1, 1000), dtype=np.float32)},
        raw_fallback=False,
        declared_contract=_authoritative_endpoint_contract(
            root, task="classification",
        ),
    )
    assert producer["endpoint_contract_hash"] == native_endpoint["endpoint_contract_hash"]
    assert len(producer["quality_contract_sha256"]) == 64
    assert len(producer["preprocessing_contract_sha256"]) == 64
    assert producer["decoder_contract_sha256"] == ""
    assert producer["nms_contract_sha256"] == ""
    request_identity = EvaluationWorkflowRunner._quality_request_identity(
        Path(exported["request"]["path"]), model_id="resnet50",
    )
    assert request_identity["identity_valid"] is True
    assert request_identity["quality_contract_sha256"] == producer["quality_contract_sha256"]
    assert request_identity["preprocessing_contract_sha256"] == producer["preprocessing_contract_sha256"]

    namespace = _runner_functions([
        "_quality_json_safe", "_quality_file_sha256", "_quality_contract_sha256",
        "_portable_dataset_manifest_sha256",
        "_build_classification_quality_contract",
    ])
    cpu_contract = namespace["_build_classification_quality_contract"](
        model_path=root / "models/source.onnx",
        validation_source=root / "validation",
        # Human-readable names are presentation metadata and may legitimately
        # differ between the management reference and a vendor label map.
        rows=[{"image_id": "a.jpg", "label_id": 3, "label_name": "3"}],
        image_scale="imagenet", letterbox=False,
        input_hw=(224, 224), input_dtype=np.float32, runner_path=RUNNER,
        endpoint_attestor_sha256="a" * 64,
    )
    reference = {
        "schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "classification", "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort", "semantic_reference_only": True,
        "provenance_required": True, "quality_contract": cpu_contract,
        "quality_contract_sha256": cpu_contract["quality_contract_sha256"],
        "records": [{
            "image_id": "a.jpg", "label_id": 3, "label_name": "3",
            "reference": record["candidate"],
        }],
    }
    reference_path = tmp_path / "classification_cpu_reference.json"
    reference_path.write_text(json.dumps(reference, sort_keys=True), encoding="utf-8")
    loaded = quality_request_from_manifest(
        exported["request"]["path"], reference_artifact=reference_path,
    )
    assert loaded.metric_gate_config["producer_identity_sha256"] == producer["producer_identity_sha256"]
    assert loaded.annotations == [{"image_id": "a.jpg", "label_id": 3, "label_name": "three"}]

    canonical_label_identity = [{
        "image_id": "a.jpg", "label_id": 3,
    }]
    assert producer["dataset"]["ground_truth_sha256"] == json_fingerprint(
        canonical_label_identity
    )

    wrong_contract_reference = json.loads(json.dumps(reference))
    wrong_contract = wrong_contract_reference["quality_contract"]
    wrong_contract["dataset"]["ground_truth_sha256"] = "f" * 64
    wrong_contract.pop("quality_contract_sha256")
    wrong_contract["quality_contract_sha256"] = json_fingerprint(wrong_contract)
    wrong_contract_reference["quality_contract_sha256"] = wrong_contract[
        "quality_contract_sha256"
    ]
    wrong_reference_path = tmp_path / "wrong_ground_truth_reference.json"
    wrong_reference_path.write_text(
        json.dumps(wrong_contract_reference, sort_keys=True), encoding="utf-8",
    )
    with pytest.raises(
        QualityArtifactIntegrityError,
        match="frozen label contract",
    ):
        quality_request_from_manifest(
            exported["request"]["path"],
            reference_artifact=wrong_reference_path,
        )

    wrong_id_contract = namespace["_build_classification_quality_contract"](
        model_path=root / "models/source.onnx",
        validation_source=root / "validation",
        rows=[{"image_id": "a.jpg", "label_id": 4, "label_name": "same-name-is-irrelevant"}],
        image_scale="imagenet", letterbox=False,
        input_hw=(224, 224), input_dtype=np.float32, runner_path=RUNNER,
        endpoint_attestor_sha256="a" * 64,
    )
    wrong_id_reference = {
        **reference,
        "quality_contract": wrong_id_contract,
        "quality_contract_sha256": wrong_id_contract["quality_contract_sha256"],
        "records": [{
            "image_id": "a.jpg", "label_id": 4, "label_name": "three",
            "reference": record["candidate"],
        }],
    }
    wrong_id_path = tmp_path / "wrong_label_id_reference.json"
    wrong_id_path.write_text(
        json.dumps(wrong_id_reference, sort_keys=True), encoding="utf-8",
    )
    with pytest.raises(
        QualityArtifactIntegrityError,
        match="classification candidate/reference label_id differs",
    ):
        quality_request_from_manifest(
            exported["request"]["path"], reference_artifact=wrong_id_path,
        )

    tampered_runtime = json.loads(path.read_text(encoding="utf-8"))
    tampered_runtime["images"][0]["label_id"] = 4
    path.write_text(json.dumps(tampered_runtime), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="deepx_quality_classification_label_missing_or_mismatch",
    ):
        module._deepx_export_central_quality_request(
            root, dxnn, _run("classification"), semantic,
            root / "results/deepx_m1_full_tampered_label",
            completed_task_evidence=performance_evidence,
        )


def test_deepx_endpoint_runtime_signature_is_precision_neutral() -> None:
    module = _suite_module()
    fp32 = {"outputs": [{"index": 0, "shape": [1, 1000], "dtype": "float32"}]}
    int8 = {"outputs": [{"index": 0, "shape": [1, 1000], "dtype": "int8"}]}
    assert module._deepx_endpoint_runtime_signature(fp32) == module._deepx_endpoint_runtime_signature(int8)
    assert module._deepx_quality_json_sha256(
        module._deepx_endpoint_runtime_signature(fp32)
    ) == module._deepx_quality_json_sha256(
        module._deepx_endpoint_runtime_signature(int8)
    )


def test_deepx_composed_precision_identity_matches_native_bridge_tags() -> None:
    precision_identity = _runner_functions([
        "_central_quality_runtime_precision_identity",
    ])["_central_quality_runtime_precision_identity"]
    common = {
        "variant": "composed", "full_provider": "",
        "stage1_provider": "deepx_m1", "stage2_provider": "tensorrt",
        "trt_precision": "fp16",
    }
    assert precision_identity(
        **common, boundary_mode="uint8_cast_bridge",
    ) == "uint8_cast_fp16"
    assert precision_identity(
        **common, boundary_mode="raw_uint8_to_float32",
    ) == "uint8_dequant_fp16"
    assert precision_identity(
        **common, boundary_mode="canonical_float32",
    ) == "float32_layout_fp16"


def _isolated_suite_function(name: str) -> tuple[Any, dict[str, Any]]:
    tree = ast.parse(SUITE.read_text(encoding="utf-8"), filename=str(SUITE))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace: dict[str, Any] = {
        "Path": Path, "Dict": Dict, "Any": Any, "List": List,
        "os": __import__("os"), "subprocess": subprocess,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SUITE), "exec"), namespace)
    return namespace[name], namespace


def test_deepx_ap50_zero_is_a_real_semantic_failure(tmp_path: Path) -> None:
    function, namespace = _isolated_suite_function("_run_deepx_full_run")
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"dxnn")
    namespace["subprocess"] = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
        TimeoutExpired=subprocess.TimeoutExpired, PIPE=subprocess.PIPE,
    )
    namespace["_parse_deepx_run_model_output"] = lambda _text: {}
    namespace["_run_deepx_prepared_feed_benchmark"] = lambda *_args, **_kwargs: {
        "status": "ok", "mean_ms": 5.0, "preprocessing_contract_audit": {"pass": True},
    }
    namespace["_run_deepx_semantic_validation"] = lambda *_args, **_kwargs: {
        "enabled": True, "status": "ok", "task": "detection",
        "preprocessing_contract": {"pass": True},
        "decoder_postprocess_contract": {"pass": True, "contracts": [{"decoder_id": "d"}]},
        "mini_coco_ap50_full": 0.0,
    }
    namespace["_deepx_export_central_quality_request"] = lambda *_args, **_kwargs: {}
    row = function(
        tmp_path, {"id": "deepx_m1_full", "dxnn_path": str(dxnn), "benchmark_task": "detection"},
        SimpleNamespace(runs=1, timeout=30, energy_measurement_only=False, validation_images=""),
        expected_endpoint_identity={
            "model_id": "yolo26s", "backend": "deepx_m1",
            "variant": "full",
        },
    )[0]
    assert row["semantic_validation_metric_gate"]["ap50"] == 0.0
    assert row["semantic_validation_metric_gate"]["pass"] is False
    assert row["semantic_validation_ok"] is False


def test_deepx_full_request_identity_uses_declared_full_scope_without_path_case(tmp_path: Path) -> None:
    request = tmp_path / "quality_inputs/deepx_setup/results/deepx/task_quality_inputs/full_request.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({
        "task": "classification", "variant": "full",
        "producer_identity": {
            "source_run_id": "deepx_m1_full", "case_id": "full", "variant": "full",
            "endpoint_contract_hash": "a" * 64,
            "runtime_precision_identity": "deepx_dxnn_sha256:" + "b" * 64,
        },
    }), encoding="utf-8")
    identity = EvaluationWorkflowRunner._quality_request_identity(request, model_id="resnet50")
    assert identity["identity_valid"] is True
    assert identity["case_id"] == "full"
    assert identity["source_run_id"] == "deepx_m1_full"
    assert identity["setup_id"] == "deepx_setup"


def test_deepx_classification_source_model_is_bound_to_model_id_or_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    model_dir = run_dir / "models/resnet50"
    model_dir.mkdir(parents=True)
    canonical_model = tmp_path / "resnet50.onnx"
    canonical_model.write_bytes(b"canonical-resnet50")
    (model_dir / "model_manifest.json").write_text(json.dumps({
        "model_id": "resnet50", "model_sha256": "",
        "file": {"sha256": "", "path": str(canonical_model)},
        "resolved_path": str(canonical_model),
        "profile_entry": {"resolved_path": str(canonical_model)},
    }), encoding="utf-8")
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    good_sha = hashlib.sha256(canonical_model.read_bytes()).hexdigest()
    good = {
        "producer_provenance_required": True,
        "producer_identity": {"model": {"source_onnx_sha256": good_sha}},
    }
    runner._validate_quality_request_model_binding("resnet50", good)

    wrong = json.loads(json.dumps(good))
    wrong["producer_identity"]["model"]["source_onnx_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match model_id"):
        runner._validate_quality_request_model_binding("resnet50", wrong)

    canonical_model.unlink()
    with pytest.raises(ValueError, match="no unique readable canonical ONNX"):
        runner._validate_quality_request_model_binding("resnet50", good)
