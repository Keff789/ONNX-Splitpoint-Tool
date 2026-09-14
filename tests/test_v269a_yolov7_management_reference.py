from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from onnx_splitpoint_tool.management_reference import (
    _source_contract,
    _validate_generated_reference,
)
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    quality_request_from_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def _functions(names: Sequence[str]) -> Dict[str, Any]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "Path": Path,
        "np": np,
        "json": json,
        "hashlib": hashlib,
        "re": re,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


def _contract(tmp_path: Path) -> dict[str, Any]:
    namespace = _functions(
        [
            "_quality_json_safe",
            "_quality_file_sha256",
            "_quality_contract_sha256",
            "_portable_dataset_manifest_sha256",
            "_build_detection_quality_contract",
        ]
    )
    model = tmp_path / "yolov7.onnx"
    model.write_bytes(b"frozen-yolov7-model")
    dataset = tmp_path / "validation"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/detection-validation-manifest",
                "schema_version": 2,
                "dataset": "coco-test",
                "samples": [{"image": "0001.jpg", "source_sha256": "sha256:" + "1" * 64}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return namespace["_build_detection_quality_contract"](
        model_path=model,
        validation_source=dataset,
        image_ids=["0001.jpg"],
        ground_truth_by_image={"0001.jpg": [_detection()]},
        image_scale="norm",
        letterbox=True,
        input_hw=(640, 640),
        input_dtype=np.float32,
        output_format="multiscale_head",
        output_names=["output_0", "output_1", "output_2"],
        yolo_conf=0.25,
        yolo_iou=0.45,
        yolo_max_det=200,
        det_conf=0.25,
        det_iou=0.45,
        det_max_det=200,
        labels=["person", "car"],
        runner_path=RUNNER,
        endpoint_attestor_sha256=hashlib.sha256(
            RUNNER.read_bytes()
        ).hexdigest(),
    )


def _detection() -> dict[str, Any]:
    return {
        "class_id": 0,
        "class_name": "person",
        "score": 0.9,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 11.0,
        "y2": 12.0,
    }


def _policy() -> dict[str, Any]:
    return {
        "policy_sha256": "a" * 64,
        "statistics": {
            "execution_location": "central_management",
            "bootstrap_repetitions": 25,
            "confidence_level": 0.95,
            "seed": 7,
            "decision": "lower_one_sided_bound",
        },
        "detection": {
            "primary_metric": "coco_ap_50_95",
            "non_inferiority_margin": 0.01,
            "guardrails": {"ap50_margin": 0.01},
        },
    }


def _export_namespace() -> Dict[str, Any]:
    namespace = _functions(
        [
            "_task_quality_execution_location",
            "_quality_json_safe",
            "_quality_file_sha256",
            "_write_stable_quality_json",
            "_verified_suite_vendored_endpoint_attestor_sha256",
            "_export_central_quality_inputs",
        ]
    )
    attestor_sha256 = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    namespace.update({
        "_EXPECTED_VENDORED_ENDPOINT_ATTESTOR_SHA256": attestor_sha256,
        "_endpoint_attestor_sha256": attestor_sha256,
        "_endpoint_attestor_path": str(RUNNER),
        "_endpoint_attestor_source": "suite_vendored",
    })
    return namespace


def test_strict_elementwise_raw_multiscale_cpu_reference_selects_separate_adapter() -> None:
    namespace = _functions(["_detection_quality_adapter_mode"])
    select = namespace["_detection_quality_adapter_mode"]

    assert select(
        "strict_elementwise", cpu_reference_only=True, output_format="multiscale_head"
    ) == "canonical_cpu_raw_multiscale_adapter"
    assert select(
        "strict_elementwise", cpu_reference_only=False, output_format="multiscale_head"
    ) == ""
    assert select(
        "proxy_detections", cpu_reference_only=False, output_format="multiscale_head"
    ) == "validation_proxy_detections"


def test_raw_endpoint_contract_keeps_decoder_and_nms_separate_and_hashed(tmp_path: Path) -> None:
    contract = _contract(tmp_path)

    assert contract["source_endpoint_is_raw"] is True
    assert contract["source_endpoint_role"] == "canonical_reference_model_output"
    assert contract["contract_scope"] == "canonical_quality_record_semantics"
    assert contract["decoder"]["identity"]["source_endpoint_semantics"] == "raw_multiscale_head"
    assert contract["decoder"]["identity"]["source_endpoint_has_integrated_nms"] is False
    assert contract["canonical_record_endpoint"] == "decoded_xyxy_score_class_detections"
    assert contract["decoder"]["sha256"] == json_fingerprint(contract["decoder"]["identity"])
    assert contract["nms"]["sha256"] == json_fingerprint(contract["nms"]["identity"])
    assert len(contract["dataset"]["ground_truth_sha256"]) == 64
    identity = dict(contract)
    identity.pop("quality_contract_sha256")
    assert contract["quality_contract_sha256"] == json_fingerprint(identity)


def test_cpu_reference_export_writes_canonical_detection_records_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path)
    namespace = _export_namespace()
    monkeypatch.setenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY", "1")

    result = namespace["_export_central_quality_inputs"](
        out_dir=tmp_path / "reference",
        task="detection",
        variant="full",
        policy=_policy(),
        gt_by_image={"0001.jpg": [_detection()]},
        candidate_by_image={"0001.jpg": [_detection()]},
        reference_by_image={"0001.jpg": [_detection()]},
        quality_contract=contract,
    )

    assert result == {}
    reference_path = (
        tmp_path
        / "reference/task_quality_inputs/canonical_detection_reference.json"
    )
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["reference"][0]["class_id"] == 0
    assert payload["quality_contract_sha256"] == contract["quality_contract_sha256"]
    assert payload["quality_contract"]["source_endpoint_is_raw"] is True
    metadata = _validate_generated_reference(reference_path)
    assert metadata["record_count"] == 1
    assert metadata["decoder_sha256"] == contract["decoder"]["sha256"]
    assert metadata["nms_sha256"] == contract["nms"]["sha256"]


def test_central_loader_requires_exact_candidate_reference_decoder_nms_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path)
    namespace = _export_namespace()
    maps = {
        "gt_by_image": {"0001.jpg": [_detection()]},
        "candidate_by_image": {"0001.jpg": [_detection()]},
        "reference_by_image": {"0001.jpg": [_detection()]},
    }

    monkeypatch.setenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY", "1")
    namespace["_export_central_quality_inputs"](
        out_dir=tmp_path / "management",
        task="detection",
        variant="full",
        policy=_policy(),
        quality_contract=contract,
        **maps,
    )
    monkeypatch.delenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY")
    request = namespace["_export_central_quality_inputs"](
        out_dir=tmp_path / "remote",
        task="detection",
        variant="full",
        policy=_policy(),
        quality_contract=contract,
        endpoint_contract={
            "task": "detection",
            "stage": "raw_head",
            "contract_family": "raw_head",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "e" * 64,
        },
        runtime_precision_identity="fp16",
        **maps,
    )
    request_path = Path(request["request"]["path"])
    reference_path = (
        tmp_path
        / "management/task_quality_inputs/canonical_detection_reference.json"
    )

    loaded = quality_request_from_manifest(
        request_path, reference_artifact=reference_path
    )
    assert loaded.metric_gate_config["quality_contract_sha256"] == contract["quality_contract_sha256"]

    tampered = json.loads(reference_path.read_text(encoding="utf-8"))
    tampered["quality_contract"]["nms"]["identity"]["iou_threshold"] = 0.99
    reference_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(QualityArtifactIntegrityError, match="contract SHA-256 mismatch"):
        quality_request_from_manifest(request_path, reference_artifact=reference_path)


def test_raw_multiscale_contract_cannot_be_relabelled_as_integrated_nms(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    decoder_identity = contract["decoder"]["identity"]
    decoder_identity["source_endpoint_has_integrated_nms"] = True
    contract["decoder"]["sha256"] = json_fingerprint(decoder_identity)
    quality_endpoint_identity = contract[
        "quality_record_endpoint"
    ]["identity"]
    quality_endpoint_identity["decoder_contract_sha256"] = contract[
        "decoder"
    ]["sha256"]
    quality_endpoint_sha = json_fingerprint(quality_endpoint_identity)
    contract["quality_record_endpoint"]["sha256"] = quality_endpoint_sha
    contract[
        "quality_record_endpoint_contract_sha256"
    ] = quality_endpoint_sha
    identity = dict(contract)
    identity.pop("quality_contract_sha256")
    contract["quality_contract_sha256"] = json_fingerprint(identity)

    payload = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "provenance_required": True,
        "quality_contract": contract,
        "quality_contract_sha256": contract["quality_contract_sha256"],
        "records": [
            {
                "image_id": "0001.jpg",
                "ground_truth": [_detection()],
                "reference": [_detection()],
            }
        ],
    }
    path = tmp_path / "invalid_reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QualityArtifactIntegrityError, match="must not be declared as integrated NMS"):
        _validate_generated_reference(path)


def test_management_reference_cache_identity_binds_runner_model_and_dataset_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_HASH_CACHE", str(tmp_path / "sha256-cache.json")
    )
    source = tmp_path / "suite"
    case = source / "b044"
    model_dir = source / "models"
    validation = source / "validation"
    case.mkdir(parents=True)
    model_dir.mkdir()
    validation.mkdir()
    (source / "benchmark_suite.py").write_text("print('suite')\n", encoding="utf-8")
    runner = case / "run_split_onnxruntime.py"
    runner.write_text("print('runner-v1')\n", encoding="utf-8")
    model = model_dir / "yolov7.onnx"
    model.write_bytes(b"model-v1")
    (case / "split_manifest.json").write_text(
        json.dumps({"full_model": "../models/yolov7.onnx"}), encoding="utf-8"
    )
    dataset_manifest = validation / "manifest.json"
    dataset_manifest.write_text(json.dumps({"samples": ["i1"]}), encoding="utf-8")
    plan = {"runs": [{"validation_images": "validation"}]}
    contract = {"cases": [{"case_id": "b044"}]}

    first = _source_contract(source, plan, contract)
    runner.write_text("print('runner-v2')\n", encoding="utf-8")
    second = _source_contract(source, plan, contract)
    model.write_bytes(b"model-v2")
    third = _source_contract(source, plan, contract)
    dataset_manifest.write_text(json.dumps({"samples": ["i1", "i2"]}), encoding="utf-8")
    fourth = _source_contract(source, plan, contract)

    assert len({first, second, third, fourth}) == 4
