from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from onnx_splitpoint_tool.quality_cache import (
    image_ids_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    _validate_candidate_execution_contract,
    quality_request_from_manifest,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _functions(names: Sequence[str]) -> Dict[str, Any]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping,
        "Optional": Optional, "Sequence": Sequence, "Tuple": Tuple,
        "Path": Path, "hashlib": hashlib, "json": json, "re": re,
        "np": np,
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


def _artifact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _classification_contract(model_sha: str) -> dict[str, Any]:
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    preprocessing_identity = {
        "schema": "onnx-splitpoint/image-preprocessing-contract",
        "schema_version": 2,
        "contract_scope": "prepared_rgb_uint8_semantics",
        "task": "classification",
    }
    preprocessing_sha = json_fingerprint(preprocessing_identity)
    postprocessor_identity = {
        "schema": "onnx-splitpoint/classification-topk-postprocessor-contract",
        "schema_version": 1,
        "implementation_runner_sha256": runner_sha,
        "canonical_record_endpoint": "classification_topk_hits",
    }
    postprocessor_sha = json_fingerprint(postprocessor_identity)
    endpoint_identity = {
        "schema": "onnx-splitpoint/classification-quality-record-endpoint-contract",
        "schema_version": 1,
        "canonical_record_endpoint": "classification_topk_hits",
        "postprocessor_contract_sha256": postprocessor_sha,
        "implementation_runner_sha256": runner_sha,
        "vendored_endpoint_attestor_sha256": runner_sha,
    }
    endpoint_sha = json_fingerprint(endpoint_identity)
    identity = {
        "schema": "onnx-splitpoint/central-classification-quality-contract",
        "schema_version": 1,
        "task": "classification",
        "model": {"artifact_name": "model.onnx", "sha256": model_sha},
        "dataset": {
            "manifest_name": "manifest.json",
            "manifest_sha256": "1" * 64,
            "image_ids_sha256": image_ids_fingerprint(["image-a"]),
            "ground_truth_sha256": json_fingerprint([
                {"image_id": "image-a", "label_id": 1},
            ]),
            "image_count": 1,
            "class_identity": "label_id",
        },
        "preprocessing": {
            "identity": preprocessing_identity,
            "sha256": preprocessing_sha,
        },
        "postprocessor": {
            "identity": postprocessor_identity,
            "sha256": postprocessor_sha,
        },
        "quality_record_endpoint": {
            "identity": endpoint_identity,
            "sha256": endpoint_sha,
        },
        "quality_record_endpoint_contract_sha256": endpoint_sha,
        "contract_scope": "canonical_quality_record_semantics",
        "canonical_record_endpoint": "classification_topk_hits",
    }
    return {**identity, "quality_contract_sha256": json_fingerprint(identity)}


def _binding(tmp_path: Path, *, setup_id: str, source_run_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ns = _functions([
        "_quality_json_safe", "_quality_file_sha256",
        "_quality_contract_sha256", "_quality_artifact_binding",
        "_build_generic_composed_quality_producer_binding",
    ])
    full = _artifact(tmp_path / "model.onnx", b"full-model")
    p1 = _artifact(tmp_path / "part1.onnx", b"part1-model")
    p2 = _artifact(tmp_path / "part2.onnx", b"part2-model")
    stage1 = _artifact(tmp_path / "compiled.hef", b"same-stage1-bytes")
    stage2 = _artifact(tmp_path / "part2.engine", b"same-stage2-bytes")
    contract = _classification_contract(hashlib.sha256(full.read_bytes()).hexdigest())
    endpoint = {
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "2" * 64,
        "stage": "classification_logits",
        "contract_family": "classification_logits",
    }
    producer, completion = ns[
        "_build_generic_composed_quality_producer_binding"
    ](
        context={
            "eval_run_id": "campaign-1", "model_id": "resnet50",
            "setup_id": setup_id, "source_run_id": source_run_id,
            "case_id": "b001", "stage1_backend": source_run_id.split("_to_")[0],
            "stage2_backend": "tensorrt", "source_model_path": str(full),
            "part1_model_path": str(p1), "part2_model_path": str(p2),
            "implementation_runner_path": str(RUNNER),
            "runtime_artifacts": [
                {"role": "stage1_runtime", "backend": source_run_id.split("_to_")[0], "path": str(stage1)},
                {"role": "stage2_runtime", "backend": "native_tensorrt", "path": str(stage2)},
            ],
        },
        task="classification", quality_contract=contract,
        endpoint_contract=endpoint,
        runtime_precision_identity="float32_layout_fp16",
    )
    return producer, completion, contract


def _portable_request(
    root: Path, producer: Mapping[str, Any], completion: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    endpoint = dict(producer["endpoint_contract"])
    common = {
        "task": "classification", "variant": "composed",
        "pairing_key": "image_id", "provenance_required": True,
        "producer_provenance_required": True,
        "artifact_provenance_contract_version": 1,
        "quality_contract": dict(contract),
        "quality_contract_sha256": contract["quality_contract_sha256"],
        "preprocessing_contract_sha256": producer["preprocessing_contract_sha256"],
        "decoder_contract_sha256": "", "nms_contract_sha256": "",
        "quality_record_endpoint_contract_sha256": producer["quality_record_endpoint_contract_sha256"],
        "endpoint_contract": endpoint,
        "endpoint_contract_hash": endpoint["endpoint_contract_hash"],
        "runtime_precision_identity": producer["runtime_precision_identity"],
        "candidate_execution_completion_contract": dict(completion),
        "candidate_execution_completion_contract_sha256": completion["contract_sha256"],
        "producer_identity": dict(producer),
        "producer_identity_sha256": producer["producer_identity_sha256"],
        "eval_run_id": producer["eval_run_id"], "model_id": producer["model_id"],
        "setup_id": producer["setup_id"], "source_run_id": producer["source_run_id"],
        "case_id": producer["case_id"], "backend": producer["backend"],
        "execution_role": producer["execution_role"],
        "performance_claims_emitted": False,
        "source_model_sha256": producer["model"]["source_onnx_sha256"],
        "part1_model_sha256": producer["model"]["part1_onnx_sha256"],
        "part2_model_sha256": producer["model"]["part2_onnx_sha256"],
        "runtime_artifact_set_sha256": producer["runtime_artifact_set_sha256"],
    }
    reference = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1, "task": "classification",
        "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True,
        "quality_contract": dict(contract),
        "quality_contract_sha256": contract["quality_contract_sha256"],
        "records": [{
            "image_id": "image-a", "label_id": 1,
            "reference": {"top1_hit": True, "top5_hit": True},
        }],
    }
    candidate = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1, **common, "record_count": 1,
        "records": [{
            "image_id": "image-a", "label_id": 1,
            "candidate": {"top1_hit": True, "top5_hit": True},
        }],
    }

    def write(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        path = root / name
        path.write_bytes(encoded)
        return {"path": name, "size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}

    reference_descriptor = write("canonical_classification_reference.json", reference)
    candidate_descriptor = write("composed_candidate.json", candidate)
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1, "status": "pending_central_evaluation",
        **common, "execution_location": "management_node",
        "requested_by": "central_management", "record_count": 1,
        "reference_record_count": 1,
        "expected_image_ids": ["image-a"],
        "expected_image_ids_sha256": image_ids_fingerprint(["image-a"]),
        "reference": {
            **reference_descriptor, "required": True,
            "expected_image_ids": ["image-a"],
            "expected_image_ids_sha256": image_ids_fingerprint(["image-a"]),
            "quality_contract_sha256": contract["quality_contract_sha256"],
        },
        "candidate": candidate_descriptor, "policy_sha256": "3" * 64,
        "metric_gate_config": {"primary_metric": "top1_accuracy"},
        "statistics": {"bootstrap_repetitions": 10, "confidence_level": 0.95, "seed": 7},
    }
    request_path = root / "composed_request.json"
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    return request_path


def test_cross_setup_byte_identical_runtime_artifacts_get_distinct_signed_identity(tmp_path: Path) -> None:
    h8, h8_completion, _ = _binding(
        tmp_path / "h8", setup_id="orin_nx_hailo8_01",
        source_run_id="hailo8_to_tensorrt",
    )
    h10, h10_completion, _ = _binding(
        tmp_path / "h10", setup_id="orin_nx_hailo10_01",
        source_run_id="hailo10_to_tensorrt",
    )
    assert h8["runtime_artifacts"][0]["sha256"] == h10["runtime_artifacts"][0]["sha256"]
    assert h8["producer_identity_sha256"] != h10["producer_identity_sha256"]
    assert h8_completion["contract_sha256"] != h10_completion["contract_sha256"]
    validated, digest = _validate_candidate_execution_contract(
        h8, role="test", task="classification",
    )
    assert validated == h8
    assert digest == h8["producer_identity_sha256"]


def test_live_generic_composed_export_repeats_all_quality_contract_bindings(
    tmp_path: Path,
) -> None:
    """Regression for the Complete_Set ORT/TensorRT quality failures."""

    namespace = _functions([
        "_task_quality_execution_location",
        "_quality_json_safe",
        "_write_stable_quality_json",
        "_quality_file_sha256",
        "_quality_contract_sha256",
        "_quality_artifact_binding",
        "_build_generic_composed_quality_producer_binding",
        "_export_central_quality_inputs",
    ])
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    namespace["_verified_suite_vendored_endpoint_attestor_sha256"] = (
        lambda: runner_sha
    )

    full = _artifact(tmp_path / "model.onnx", b"full-model")
    part1 = _artifact(tmp_path / "part1.onnx", b"part1-model")
    part2 = _artifact(tmp_path / "part2.onnx", b"part2-model")
    stage1 = _artifact(tmp_path / "part1.engine", b"stage1-runtime")
    stage2 = _artifact(tmp_path / "part2.engine", b"stage2-runtime")
    contract = _classification_contract(
        hashlib.sha256(full.read_bytes()).hexdigest()
    )
    endpoint = {
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "2" * 64,
        "stage": "classification_logits",
        "contract_family": "classification_logits",
    }
    rows = [{
        "image": "image-a",
        "label_id": 1,
        "gt": {"top1_hit": True, "top5_hit": True},
        "gt_reference": {"top1_hit": True, "top5_hit": True},
    }]
    exported = namespace["_export_central_quality_inputs"](
        out_dir=tmp_path / "results_ort_tensorrt",
        task="classification",
        variant="composed",
        policy={
            "statistics": {
                "execution_location": "central_management",
                "bootstrap_repetitions": 10,
            },
        },
        classification_rows=rows,
        quality_contract=contract,
        endpoint_contract=endpoint,
        runtime_precision_identity="fp16",
        composed_producer_context={
            "eval_run_id": "complete-set-regression",
            "model_id": "mobilenet_v3_large",
            "setup_id": "orin_nx_deepx_m1_01",
            "source_run_id": "ort_tensorrt",
            "case_id": "b027",
            "stage1_backend": "tensorrt",
            "stage2_backend": "tensorrt",
            "source_model_path": str(full),
            "part1_model_path": str(part1),
            "part2_model_path": str(part2),
            "implementation_runner_path": str(RUNNER),
            "runtime_artifacts": [
                {
                    "role": "stage1_runtime",
                    "backend": "tensorrt",
                    "path": str(stage1),
                },
                {
                    "role": "stage2_runtime",
                    "backend": "native_tensorrt",
                    "path": str(stage2),
                },
            ],
        },
    )

    request_path = Path(exported["request"]["path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_path = request_path.parent / request["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    producer = request["producer_identity"]
    for payload in (request, candidate):
        assert payload["decoder_contract_sha256"] == producer[
            "decoder_contract_sha256"
        ]
        assert payload["nms_contract_sha256"] == producer[
            "nms_contract_sha256"
        ]
        assert payload["quality_record_endpoint_contract_sha256"] == producer[
            "quality_record_endpoint_contract_sha256"
        ]
    assert request["decoder_contract_sha256"] == ""
    assert request["nms_contract_sha256"] == ""
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == request[
        "candidate"
    ]["sha256"]

    management_reference = tmp_path / "management_reference.json"
    management_reference.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/task-quality-reference-input",
            "schema_version": 1,
            "task": "classification",
            "pairing_key": "image_id",
            "reference_role": "canonical_cpu_ort",
            "semantic_reference_only": True,
            "provenance_required": True,
            "quality_contract": contract,
            "quality_contract_sha256": contract["quality_contract_sha256"],
            "records": [{
                "image_id": "image-a",
                "label_id": 1,
                "reference": {"top1_hit": True, "top5_hit": True},
            }],
        }, sort_keys=True),
        encoding="utf-8",
    )
    loaded = quality_request_from_manifest(
        request_path,
        reference_artifact=management_reference,
    )
    assert loaded.artifact_provenance_binding_status == "verified"
    assert loaded.artifact_provenance_claim_eligible is True

    identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path,
        model_id="mobilenet_v3_large",
        variant="composed",
        manifest=request,
    )
    assert not any(
        error.startswith("missing_generic_composed_request_duplicate_")
        for error in identity["identity_errors"]
    )

    # Classification intentionally binds decoder/NMS as explicit empty
    # strings.  Absence is still a malformed duplicate and must be detected
    # before the central Quality loader sees the request.
    missing_decoder = dict(request)
    missing_decoder.pop("decoder_contract_sha256")
    rejected = EvaluationWorkflowRunner._quality_request_identity(
        request_path,
        model_id="mobilenet_v3_large",
        variant="composed",
        manifest=missing_decoder,
    )
    assert rejected["identity_valid"] is False
    assert (
        "missing_generic_composed_request_duplicate_decoder_contract_sha256"
        in rejected["identity_errors"]
    )


def test_generic_dispatch_never_inherits_native_full_source_run_default() -> None:
    runner_source = RUNNER.read_text(encoding="utf-8")
    suite_source = SUITE.read_text(encoding="utf-8")
    assert (
        'ap.add_argument("--quality-evidence-source-run-id", type=str, default=""'
        in runner_source
    )
    assert "generic_composed_identity_required" in suite_source
    assert "generic composed central-quality dispatch identity is" in suite_source
    assert '"quality-evidence-source-run-id": str(' in suite_source


def test_loader_and_workflow_reject_setup_or_source_substitution(tmp_path: Path) -> None:
    producer, completion, contract = _binding(
        tmp_path / "artifacts", setup_id="orin_nx_hailo8_01",
        source_run_id="hailo8_to_tensorrt",
    )
    good_root = tmp_path / "quality_inputs/orin_nx_hailo8_01/b001/results_hailo8_to_tensorrt/task_quality_inputs"
    request_path = _portable_request(good_root, producer, completion, contract)
    loaded = quality_request_from_manifest(request_path)
    assert loaded.artifact_provenance_binding_status == "verified"
    assert loaded.artifact_provenance_claim_eligible is True
    identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path, model_id="resnet50",
    )
    assert identity["identity_valid"] is True
    assert identity["schema_version"] == 6
    assert identity["setup_id"] == "orin_nx_hailo8_01"
    assert identity["source_run_id"] == "hailo8_to_trt"

    wrong_path = tmp_path / "quality_inputs/orin_nx_hailo10_01/b001/results_hailo8_to_tensorrt/task_quality_inputs/composed_request.json"
    wrong_path.parent.mkdir(parents=True)
    wrong_path.write_bytes(request_path.read_bytes())
    wrong_identity = EvaluationWorkflowRunner._quality_request_identity(
        wrong_path, model_id="resnet50",
    )
    assert wrong_identity["identity_valid"] is False
    assert any("setup_id" in error for error in wrong_identity["identity_errors"])

    candidate_path = good_root / "composed_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["setup_id"] = "orin_nx_hailo10_01"
    encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    candidate_path.write_bytes(encoded)
    request = json.loads(request_path.read_text())
    request["candidate"].update({
        "size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
    })
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    try:
        quality_request_from_manifest(request_path)
    except QualityArtifactIntegrityError as exc:
        assert "setup_id" in str(exc) or "producer identity" in str(exc)
    else:
        raise AssertionError("cross-setup candidate substitution was accepted")


def test_historical_unbound_composed_request_is_readable_but_not_claimable(tmp_path: Path) -> None:
    producer, completion, contract = _binding(
        tmp_path / "artifacts", setup_id="orin_nx_hailo8_01",
        source_run_id="hailo8_to_tensorrt",
    )
    root = tmp_path / "quality_inputs/orin_nx_hailo8_01/b001/results_hailo8_to_tensorrt/task_quality_inputs"
    request_path = _portable_request(root, producer, completion, contract)
    request = json.loads(request_path.read_text())
    candidate_path = root / "composed_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    remove = {
        "producer_provenance_required", "artifact_provenance_contract_version",
        "producer_identity", "producer_identity_sha256", "eval_run_id",
        "model_id", "setup_id", "source_run_id", "case_id", "backend",
        "execution_role", "performance_claims_emitted", "source_model_sha256",
        "part1_model_sha256", "part2_model_sha256",
        "runtime_artifact_set_sha256",
    }
    for payload in (request, candidate):
        for field in remove:
            payload.pop(field, None)
        payload["candidate_execution_completion_contract"] = {}
        payload["candidate_execution_completion_contract_sha256"] = ""
    encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    candidate_path.write_bytes(encoded)
    request["candidate"].update({
        "size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
    })
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")

    loaded = quality_request_from_manifest(request_path)
    assert loaded.artifact_provenance_binding_status == "legacy_unbound"
    assert loaded.artifact_provenance_claim_eligible is False
    identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path, model_id="resnet50",
    )
    assert identity["identity_valid"] is False
    assert identity["artifact_provenance_binding_status"] == "legacy_unbound"
    assert "generic_composed_producer_binding_missing" in identity["identity_errors"]
