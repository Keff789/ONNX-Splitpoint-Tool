from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


SHA_SOURCE = hashlib.sha256(b"source").hexdigest()
SHA_ENGINE = hashlib.sha256(b"engine").hexdigest()
SHA_TRTEXEC = hashlib.sha256(b"trtexec").hexdigest()
SHA_ENDPOINT = hashlib.sha256(b"endpoint").hexdigest()
SHA_QUALITY = hashlib.sha256(b"quality").hexdigest()
SHA_PREPROCESSING = hashlib.sha256(b"preprocessing").hexdigest()
SHA_DATASET = hashlib.sha256(b"dataset").hexdigest()
SHA_IMAGE_IDS = hashlib.sha256(b"image-ids").hexdigest()
SHA_GROUND_TRUTH = hashlib.sha256(b"ground-truth").hexdigest()
SHA_POLICY = hashlib.sha256(b"policy").hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _producer() -> dict[str, Any]:
    source = {"path": "/remote/source.onnx", "sha256": SHA_SOURCE, "size_bytes": 6}
    build = {
        "path": "/remote/cache/source.onnx", "sha256": SHA_SOURCE,
        "size_bytes": 6, "source_onnx_sha256": SHA_SOURCE,
    }
    engine = {
        "path": "/remote/cache/full_fp16.engine", "sha256": SHA_ENGINE,
        "size_bytes": 6, "source_onnx_sha256": SHA_SOURCE,
        "build_onnx_sha256": SHA_SOURCE,
    }
    trtexec = {"path": "/usr/bin/trtexec", "sha256": SHA_TRTEXEC, "size_bytes": 7}
    receipt_payload = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "source_onnx": build["path"],
        "source_onnx_sha256": SHA_SOURCE,
        "engine": engine["path"],
        "engine_sha256": SHA_ENGINE,
        "trtexec": trtexec["path"],
        "trtexec_sha256": SHA_TRTEXEC,
        "command": [
            trtexec["path"], f"--onnx={build['path']}",
            f"--saveEngine={engine['path']}", "--fp16",
        ],
    }
    receipt = {**receipt_payload, "receipt_sha256": _digest(receipt_payload)}
    receipt_binding = {
        "path": "/remote/cache/engine_build_receipt.json",
        "sha256": _digest(receipt),
        "size_bytes": len(_canonical_bytes(receipt)),
        "receipt": receipt,
    }
    receipt_file_sha = hashlib.sha256(b"persisted-receipt-file").hexdigest()
    quality_contract = {
        "quality_contract_sha256": SHA_QUALITY,
        "model": {"sha256": SHA_SOURCE},
        "dataset": {
            "manifest_sha256": SHA_DATASET,
            "image_ids_sha256": SHA_IMAGE_IDS,
            "ground_truth_sha256": SHA_GROUND_TRUTH,
        },
        "preprocessing": {"sha256": SHA_PREPROCESSING},
    }
    producer = {
        "schema": "onnx-splitpoint/tensorrt-central-quality-producer-identity",
        "schema_version": 1,
        "eval_run_id": "eval-20260721",
        "model_id": "resnet50",
        "setup_id": "orin_nx_hailo8_01",
        "source_run_id": "native_full_tensorrt",
        "originating_plan_run_id": "hailo8_to_tensorrt",
        "case_id": "full",
        "execution_role": "full_quality_only",
        "backend": "native_tensorrt",
        "variant": "full",
        "task": "classification",
        "source_onnx": source,
        "build_onnx": build,
        "engine": engine,
        "trtexec": trtexec,
        "engine_build_receipt": receipt_binding,
        "engine_build_receipt_file_sha256": receipt_file_sha,
        "model": {
            "source_onnx_sha256": SHA_SOURCE,
            "source_onnx_size_bytes": 6,
            "build_onnx_sha256": SHA_SOURCE,
            "build_onnx_size_bytes": 6,
            "runtime_artifact_sha256": SHA_ENGINE,
            "runtime_artifact_size_bytes": 6,
        },
        "dataset": copy.deepcopy(quality_contract["dataset"]),
        "preprocessing": {"sha256": SHA_PREPROCESSING},
        "endpoint": {"identity": {"stage": "classification_logits"}, "sha256": SHA_ENDPOINT},
        "endpoint_authority": {
            "identity": {"terminal_model_sha256": SHA_SOURCE},
            "sha256": hashlib.sha256(b"authority").hexdigest(),
        },
        "quality_record_endpoint": {"identity": {"kind": "topk"}, "sha256": hashlib.sha256(b"topk").hexdigest()},
        "precision": {"identity": {"runtime_precision_identity": "fp16"}, "sha256": hashlib.sha256(b"fp16").hexdigest()},
        "quality_contract": quality_contract,
        "quality_contract_sha256": SHA_QUALITY,
        "preprocessing_contract_sha256": SHA_PREPROCESSING,
        "decoder_contract_sha256": "",
        "nms_contract_sha256": "",
        "quality_record_endpoint_contract_sha256": hashlib.sha256(b"topk").hexdigest(),
        "endpoint_contract_hash": SHA_ENDPOINT,
        "endpoint_contract_complete": True,
        "runtime_precision_identity": "fp16",
        "policy_sha256": SHA_POLICY,
        "performance_claims_emitted": False,
    }
    producer["producer_identity_sha256"] = _digest(producer)
    return producer


def _manifest(producer: dict[str, Any]) -> dict[str, Any]:
    receipt = producer["engine_build_receipt"]
    return {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "producer_provenance_required": True,
        "producer_identity": copy.deepcopy(producer),
        "producer_identity_sha256": producer["producer_identity_sha256"],
        "eval_run_id": producer["eval_run_id"],
        "model_id": producer["model_id"],
        "setup_id": producer["setup_id"],
        "source_run_id": producer["source_run_id"],
        "case_id": "full",
        "execution_role": "full_quality_only",
        "backend": "native_tensorrt",
        "variant": "full",
        "task": "classification",
        "performance_claims_emitted": False,
        "source_model_sha256": SHA_SOURCE,
        "build_onnx_sha256": SHA_SOURCE,
        "runtime_artifact_sha256": SHA_ENGINE,
        "trtexec_sha256": SHA_TRTEXEC,
        "engine_build_receipt_sha256": receipt["sha256"],
        "engine_build_receipt_file_sha256": producer[
            "engine_build_receipt_file_sha256"
        ],
        "trt_engine_build_receipt_sha256": receipt["receipt"]["receipt_sha256"],
        "quality_contract": copy.deepcopy(producer["quality_contract"]),
        "quality_contract_sha256": SHA_QUALITY,
        "preprocessing_contract_sha256": SHA_PREPROCESSING,
        "decoder_contract_sha256": "",
        "nms_contract_sha256": "",
        "quality_record_endpoint_contract_sha256": producer["quality_record_endpoint_contract_sha256"],
        "endpoint_contract": {
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": SHA_ENDPOINT,
            "stage": "classification_logits",
        },
        "endpoint_contract_hash": SHA_ENDPOINT,
        "runtime_precision_identity": "fp16",
        "policy_sha256": SHA_POLICY,
    }


def _request_path(tmp_path: Path) -> Path:
    return (
        tmp_path / "quality_inputs/orin_nx_hailo8_01/"
        "results_native_full_tensorrt/task_quality_inputs/full_request.json"
    )


@pytest.fixture(autouse=True)
def _strict_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx_splitpoint_tool.quality_service as quality_service

    def validate(value: Any, *, role: str, task: str) -> tuple[dict[str, Any], str]:
        assert isinstance(value, dict)
        declared = str(value.get("producer_identity_sha256") or "")
        unsigned = copy.deepcopy(value)
        unsigned.pop("producer_identity_sha256", None)
        if declared != _digest(unsigned):
            raise ValueError("producer SHA-256 mismatch")
        if str(value.get("task") or "") != task:
            raise ValueError("producer task mismatch")
        return copy.deepcopy(value), declared

    monkeypatch.setattr(
        quality_service, "_validate_candidate_execution_contract", validate,
    )


def _write_request(tmp_path: Path, manifest: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    payload = copy.deepcopy(manifest or _manifest(_producer()))
    path = _request_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload


def _result_from_identity(identity: dict[str, Any]) -> dict[str, Any]:
    nested = copy.deepcopy(identity)
    nested["producer_binding_eligible"] = True
    result = {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "schema_version": 1,
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "source_request_sha256": identity["source_request_sha256"],
        "request_identity": nested,
        "producer_identity": copy.deepcopy(identity["producer_identity"]),
        "producer_binding_eligible": True,
        "n": 1,
        "primary": {"metric": "top1_accuracy", "delta": 0.0},
    }
    for field_name in EvaluationWorkflowRunner._trt_quality_flat_field_names():
        result[field_name] = copy.deepcopy(identity[field_name])
    return result


def test_request_scope_v4_preserves_complete_verified_trt_producer(tmp_path: Path) -> None:
    path, _ = _write_request(tmp_path)

    identity = EvaluationWorkflowRunner._quality_request_identity(
        path, model_id="resnet50",
    )

    producer = identity["producer_identity"]
    receipt = producer["engine_build_receipt"]
    assert identity["identity_valid"] is True
    assert identity["schema_version"] == 4
    assert identity["source_run_id"] == "native_full_tensorrt"
    assert identity["case_id"] == identity["variant"] == "full"
    assert identity["producer_identity_sha256"] == producer["producer_identity_sha256"]
    assert identity["engine_build_receipt_sha256"] == receipt["sha256"]
    assert identity["engine_build_receipt_size_bytes"] == receipt["size_bytes"]
    assert identity["trt_engine_build_receipt_sha256"] == receipt["receipt"]["receipt_sha256"]
    assert identity["trt_engine_build_receipt_size_bytes"] == len(
        _canonical_bytes({
            key: value for key, value in receipt["receipt"].items()
            if key != "receipt_sha256"
        })
    )
    assert identity["engine_path"] == producer["engine"]["path"]
    assert identity["producer_identity"]["endpoint_authority"] == producer["endpoint_authority"]


def test_summary_only_request_requires_false_top_level_performance_claim(tmp_path: Path) -> None:
    manifest = _manifest(_producer())
    manifest.pop("performance_claims_emitted")
    path, _ = _write_request(tmp_path, manifest)

    identity = EvaluationWorkflowRunner._quality_request_identity(path, model_id="resnet50")

    assert identity["identity_valid"] is False
    assert identity["schema_version"] == 3
    assert "producer_identity" not in identity
    assert "missing_trt_request_duplicate_performance_claims_emitted" in identity["identity_errors"]


def test_legacy_request_identity_remains_schema_v3(tmp_path: Path) -> None:
    path = tmp_path / "quality_inputs/deepx/results/deepx/task_quality_inputs/full_request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "task": "classification", "variant": "full",
        "producer_identity": {
            "schema": "onnx-splitpoint/central-quality-producer-identity",
            "source_run_id": "deepx_m1_full", "case_id": "full",
            "endpoint_contract_hash": "a" * 64,
            "runtime_precision_identity": "deepx_dxnn_sha256:" + "b" * 64,
        },
    }), encoding="utf-8")

    identity = EvaluationWorkflowRunner._quality_request_identity(path, model_id="resnet50")

    assert identity["schema_version"] == 3
    assert identity["identity_valid"] is True
    assert "producer_identity" not in identity


def test_result_identity_checks_nested_flat_and_canonical_producer(tmp_path: Path) -> None:
    path, _ = _write_request(tmp_path)
    request_identity = EvaluationWorkflowRunner._quality_request_identity(path, model_id="resnet50")
    result = _result_from_identity(request_identity)

    identity, errors = EvaluationWorkflowRunner._central_quality_result_identity(result)
    assert errors == []
    assert identity["producer_identity_validated"] is True
    assert identity["producer_binding_eligible"] is True

    result["engine_sha256"] = "f" * 64
    _, errors = EvaluationWorkflowRunner._central_quality_result_identity(result)
    assert "conflicting_result_engine_sha256" in errors

    tampered = _result_from_identity(request_identity)
    tampered["producer_identity"]["endpoint_authority"]["identity"][
        "terminal_model_sha256"
    ] = "f" * 64
    _, errors = EvaluationWorkflowRunner._central_quality_result_identity(tampered)
    assert any("producer" in error for error in errors)


def _runner_with_sentinel_row(tmp_path: Path) -> tuple[EvaluationWorkflowRunner, Path, dict[str, Any]]:
    path = tmp_path / "models/resnet50/benchmark_results/normalized_results.json"
    path.parent.mkdir(parents=True)
    row = {"model_id": "resnet50", "case_id": "b001", "sentinel": [1, 2, 3]}
    path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "quality_gate": {"statistics": {"execution_location": "central_management"}},
    }
    return runner, path, row


def test_summary_only_merge_never_mutates_performance_row_and_deduplicates(tmp_path: Path) -> None:
    request_path, _ = _write_request(tmp_path / "request")
    identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path, model_id="resnet50",
    )
    result = _result_from_identity(identity)
    runner, normalized_path, original_row = _runner_with_sentinel_row(tmp_path / "run")

    merge = runner._merge_central_quality_results([result, copy.deepcopy(result)])

    assert merge["unmatched_result_count"] == 0
    assert merge["summary_only_native_full_quality_completed_count"] == 1
    assert merge["summary_only_native_full_quality_duplicate_count"] == 1
    assert merge["updated_models"] == []
    payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert payload["results"] == [original_row]


def test_summary_only_conflict_and_failed_result_are_never_bound(tmp_path: Path) -> None:
    request_path, _ = _write_request(tmp_path / "request")
    identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path, model_id="resnet50",
    )
    first = _result_from_identity(identity)
    conflicting = copy.deepcopy(first)
    conflicting["source_request_sha256"] = "f" * 64
    conflicting["request_identity"]["source_request_sha256"] = "f" * 64
    runner, _, _ = _runner_with_sentinel_row(tmp_path / "conflict")

    merge = runner._merge_central_quality_results([first, conflicting])
    assert merge["summary_only_native_full_quality_conflict_count"] == 1
    assert merge["unmatched_result_count"] == 2

    failed = _result_from_identity(identity)
    failed.update({"status": "failed", "technical_status": "failed", "producer_binding_eligible": False})
    failed["request_identity"]["producer_binding_eligible"] = False
    runner, _, _ = _runner_with_sentinel_row(tmp_path / "failed")
    merge = runner._merge_central_quality_results([failed])
    assert merge["summary_only_native_full_quality_failed_count"] == 1
    assert merge["summary_only_native_full_quality_completed_count"] == 0
    assert merge["unmatched_result_count"] == 0
    assert merge["updated_models"] == []


def _vendor_full_only_result(source_run_id: str, setup_id: str) -> dict[str, Any]:
    request_sha = hashlib.sha256(
        f"{source_run_id}@{setup_id}/full".encode("utf-8")
    ).hexdigest()
    identity = {
        "schema": "onnx-splitpoint/central-quality-request-identity",
        "schema_version": 3,
        "identity_valid": True,
        "identity_errors": [],
        "eval_run_id": "eval-20260721",
        "model_id": "resnet50",
        "case_id": "full",
        "source_run_id": source_run_id,
        "setup_id": setup_id,
        "variant": "full",
        "task": "classification",
        "backend": "hailo10h" if source_run_id == "hailo10" else source_run_id,
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "source_request_sha256": request_sha,
        "endpoint_contract_hash": SHA_ENDPOINT,
        "runtime_precision_identity": f"{source_run_id}:full",
    }
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "schema_version": 1,
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "eval_run_id": "eval-20260721",
        "model_id": "resnet50",
        "case_id": "full",
        "source_run_id": source_run_id,
        "run_id": source_run_id,
        "source_setup_id": setup_id,
        "variant": "full",
        "task": "classification",
        "backend": "hailo10h" if source_run_id == "hailo10" else source_run_id,
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "source_request_sha256": request_sha,
        "endpoint_contract_hash": SHA_ENDPOINT,
        "runtime_precision_identity": f"{source_run_id}:full",
        "request_identity": identity,
        "n": 1,
        "primary": {"metric": "top1_accuracy", "delta": 0.0},
    }


def _trt_full_only_result_for_setup(tmp_path: Path, setup_id: str) -> dict[str, Any]:
    producer = _producer()
    producer["setup_id"] = setup_id
    producer.pop("producer_identity_sha256", None)
    producer["producer_identity_sha256"] = _digest(producer)
    manifest = _manifest(producer)
    path = (
        tmp_path / "quality_inputs" / setup_id
        / "results_native_full_tensorrt/task_quality_inputs/full_request.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    identity = EvaluationWorkflowRunner._quality_request_identity(
        path, model_id="resnet50",
    )
    assert identity["identity_valid"] is True, identity["identity_errors"]
    return _result_from_identity(identity)


def _standard_quality_three_model_profile() -> dict[str, Any]:
    targets = [
        {
            "id": "orin_nx_hailo8_01", "accelerator": "hailo8",
            "enabled": True,
            "runtime": {"enabled": True, "host": "h8", "user": "nx"},
        },
        {
            "id": "orin_nx_hailo10_01", "accelerator": "hailo10h",
            "enabled": True,
            "runtime": {"enabled": True, "host": "h10", "user": "nx"},
        },
        {
            "id": "orin_nx_deepx_m1_01", "accelerator": "deepx_m1",
            "enabled": True,
            "runtime": {"enabled": True, "host": "dx", "user": "nx"},
        },
    ]
    targets_sha = hashlib.sha256(json.dumps(
        targets,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()
    return {
        "model_suite": {"primary": [
            {"id": "resnet50", "task": "classification", "enabled": True},
            {"id": "yolo26s", "task": "detection", "enabled": True},
            {"id": "yolov7_paper", "task": "detection", "enabled": True},
        ]},
        "run_profiles": [
            {
                "id": "ort_tensorrt", "type": "same_backend_reference",
                "full": "tensorrt", "stage1": "tensorrt",
                "stage2": "tensorrt",
            },
            {
                "id": "hailo8", "type": "same_backend_reference",
                "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8",
            },
            {
                "id": "hailo8_to_trt", "type": "mixed_backend",
                "stage1": "hailo8", "stage2": "tensorrt",
            },
            {
                "id": "hailo10", "type": "same_backend_reference",
                "full": "hailo10", "stage1": "hailo10",
                "stage2": "hailo10",
            },
            {
                "id": "hailo10_to_tensorrt", "type": "mixed_backend",
                "stage1": "hailo10", "stage2": "tensorrt",
            },
            {
                "id": "deepx_m1_full", "type": "same_backend_reference",
                "full": "deepx_m1", "stage1": "deepx_m1",
                "stage2": "deepx_m1",
            },
            {
                "id": "deepx_m1_to_tensorrt", "type": "mixed_backend",
                "stage1": "deepx_m1", "stage2": "tensorrt",
            },
        ],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
        "selection_policy": {"max_accepted_cases_per_model": 1},
        "hardware": {
            "resolution_frozen_at_start": True,
            "resolved_targets": targets,
            "resolved_targets_sha256": f"sha256:{targets_sha}",
        },
        "native_producers": {
            "enabled": True,
            "remotes": {
                "hailo8": {"setup_id": "orin_nx_hailo8_01"},
                "hailo10h": {"setup_id": "orin_nx_hailo10_01"},
                "deepx": {"setup_id": "orin_nx_deepx_m1_01"},
            },
        },
        "execution_preset": {
            "id": "final",
            "snapshot": {"defaults": {
                "native_enabled": True, "energy_enabled": False,
            }},
        },
    }


def _standard_writer_result(
    root: Path,
    *,
    runner: EvaluationWorkflowRunner,
    model_id: str,
    task: str,
    setup_id: str,
    endpoint_id: str,
) -> dict[str, Any]:
    """Create the duplicated identity shape emitted by the real TRT writer."""

    producer = _producer()
    producer.update({
        "eval_run_id": runner.run_id,
        "model_id": model_id,
        "setup_id": setup_id,
        "task": task,
    })
    if task == "detection":
        producer["decoder_contract_sha256"] = hashlib.sha256(
            f"{model_id}:decoder".encode("utf-8")
        ).hexdigest()
        producer["nms_contract_sha256"] = hashlib.sha256(
            f"{model_id}:nms".encode("utf-8")
        ).hexdigest()
    producer.pop("producer_identity_sha256", None)
    producer["producer_identity_sha256"] = _digest(producer)
    manifest = _manifest(producer)
    manifest.update({
        "task": task,
        "decoder_contract_sha256": producer.get(
            "decoder_contract_sha256", ""
        ),
        "nms_contract_sha256": producer.get("nms_contract_sha256", ""),
    })
    plan_identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": endpoint_id,
        "eval_run_id": runner.run_id,
        "model_id": model_id,
        "setup_id": setup_id,
        "source_run_id": "native_full_tensorrt",
        "backend": "tensorrt",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    manifest.update({
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": copy.deepcopy(plan_identity),
        "full_only_plan_identity_sha256": _digest(plan_identity),
        "quality_canary_id": endpoint_id,
    })
    path = (
        root / model_id / "quality_inputs" / setup_id
        / "results_native_full_tensorrt/task_quality_inputs/full_request.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    request_identity = EvaluationWorkflowRunner._quality_request_identity(
        path, model_id=model_id,
    )
    assert request_identity["identity_valid"] is True, request_identity[
        "identity_errors"
    ]
    bound, errors = runner._bind_full_only_quality_request_to_effective_plan(
        model_id=model_id, request_identity=request_identity,
    )
    assert errors == []
    assert bound["quality_canary_id"] == endpoint_id
    request_identity.update({
        "full_only_plan_identity_validated": True,
        "quality_canary_id": bound["quality_canary_id"],
        "eval_run_id": bound["eval_run_id"],
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": copy.deepcopy(
            bound["full_only_plan_identity"]
        ),
        "full_only_plan_identity_sha256": bound[
            "full_only_plan_identity_sha256"
        ],
        "dispatch_run_id": bound["dispatch_run_id"],
        "backend": bound["backend"],
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    })
    result = _result_from_identity(request_identity)
    result.update({
        "eval_run_id": runner.run_id,
        "model_id": model_id,
        "case_id": "full",
        "source_run_id": "native_full_tensorrt",
        "run_id": "native_full_tensorrt",
        "source_setup_id": setup_id,
        "variant": "full",
        "task": task,
        "backend": "tensorrt",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "full_only_plan_identity_validated": True,
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": copy.deepcopy(plan_identity),
        "full_only_plan_identity_sha256": _digest(plan_identity),
        "quality_canary_id": endpoint_id,
    })
    return result


def test_standard_quality_three_models_three_setups_writer_bind_and_merge(
    tmp_path: Path,
) -> None:
    """Regress the exact overnight Standard/Quality 3-model x 3-setup shape."""

    profile = _standard_quality_three_model_profile()
    plan = build_effective_execution_plan(profile)
    assert plan["quality_canary_enabled"] is False
    contract = plan[
        "setup_local_tensorrt_quality_companion_contract"
    ]
    assert contract["status"] == "ready"
    assert contract["identity_count"] == 3

    run_dir = tmp_path / "eval-20260721"
    run_dir.mkdir()
    (run_dir / "effective_execution_plan.json").write_text(
        json.dumps(plan), encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = "eval-20260721"
    runner.profile_payload = profile
    runner.options = SimpleNamespace(
        resume_missing_full_quality_only=False,
    )

    # Unsealed Standard vendor quality remains a normal row-join request; only
    # the sealed Native TRT companion enters this dedicated plan binder.
    vendor_bound, vendor_errors = (
        runner._bind_full_only_quality_request_to_effective_plan(
            model_id="resnet50",
            request_identity={
                "source_run_id": "hailo8",
                "setup_id": "orin_nx_hailo8_01",
                "variant": "full",
                "backend": "hailo8",
                "execution_role": "performance_owner",
                "full_only_plan_identity": {},
                "full_only_plan_identity_required": None,
                "quality_canary_id": "",
            },
        )
    )
    assert vendor_bound == {}
    assert vendor_errors == []

    models = (
        ("resnet50", "classification"),
        ("yolo26s", "detection"),
        ("yolov7_paper", "detection"),
    )
    results = [
        _standard_writer_result(
            tmp_path / "writer",
            runner=runner,
            model_id=model_id,
            task=task,
            setup_id=str(identity["setup_id"]),
            endpoint_id=str(identity["id"]),
        )
        for model_id, task in models
        for identity in contract["identities"]
    ]
    assert len(results) == 9

    merged = runner._merge_central_quality_results(results)
    assert merged[
        "summary_only_standard_quality_companion_contract_enabled"
    ] is True
    assert merged["summary_only_full_quality_contract_mode"] == (
        "standard_quality_setup_local_tensorrt"
    )
    assert merged["summary_only_full_quality_expected_count"] == 9
    assert merged["summary_only_full_quality_seen_count"] == 9
    assert merged["summary_only_full_quality_completed_count"] == 9
    assert merged["summary_only_full_quality_missing_count"] == 0
    assert merged["unmatched_result_count"] == 0

    # Forge a self-consistent writer/result seal. Admission must still reject
    # it because the endpoint id is not the effective-plan identity.
    forged = copy.deepcopy(results)
    for container in (forged[0], forged[0]["request_identity"]):
        container["quality_canary_id"] = "forged_setup_local_endpoint"
        container["full_only_plan_identity"]["quality_canary_id"] = (
            "forged_setup_local_endpoint"
        )
        container["full_only_plan_identity_sha256"] = _digest(
            container["full_only_plan_identity"]
        )
    rejected = runner._merge_central_quality_results(forged)
    assert rejected["summary_only_full_quality_completed_count"] == 8
    assert rejected["unmatched_result_count"] == 1
    assert "summary_only_quality_canary_id_mismatch" in rejected[
        "unmatched_results"
    ][0]["join_identity_errors"]


def test_standard_final_missing_companion_postcondition_hard_fails_stage(
    tmp_path: Path,
) -> None:
    """Missing expected evidence is blocking even without a failed future."""

    profile = _standard_quality_three_model_profile()
    plan = build_effective_execution_plan(profile)
    run_dir = tmp_path / "eval-20260721"
    run_dir.mkdir()
    (run_dir / "effective_execution_plan.json").write_text(
        json.dumps(plan), encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = "eval-20260721"
    runner.profile_payload = profile
    runner.options = SimpleNamespace(
        resume_missing_full_quality_only=False,
    )
    runner._central_quality_service = None
    runner._central_quality_futures = {}
    runner._preserved_central_quality_results = []
    runner._emit_log = lambda _message: None
    runner._refresh_validation_after_central_quality = lambda _models: []
    runner._merge_central_quality_results = lambda _results: {
        "updated_models": [],
        "updated_row_count": 0,
        "matched_completed_count": 0,
        "matched_failed_count": 0,
        "matched_primary_result_count": 0,
        "matched_supplemental_result_count": 0,
        "summary_only_native_full_quality_candidate_count": 8,
        "summary_only_native_full_quality_unique_count": 8,
        "summary_only_native_full_quality_completed_count": 8,
        "summary_only_native_full_quality_failed_count": 0,
        "summary_only_native_full_quality_duplicate_count": 0,
        "summary_only_native_full_quality_conflict_count": 0,
        "summary_only_full_quality_contract_enabled": True,
        "summary_only_full_quality_contract_mode": (
            "standard_quality_setup_local_tensorrt"
        ),
        "summary_only_full_quality_expected_count": 9,
        "summary_only_full_quality_seen_count": 8,
        "summary_only_full_quality_completed_count": 8,
        "summary_only_full_quality_failed_count": 0,
        "summary_only_full_quality_missing_count": 1,
        "summary_only_full_quality_duplicate_count": 0,
        "summary_only_full_quality_contract_error_count": 0,
        "summary_only_full_quality_identity_key_fields": [
            "model_id", "source_run_id", "setup_id", "backend",
            "variant", "execution_role", "performance_claims_emitted",
        ],
        # This is the critical regression: the postcondition itself, not an
        # incidental failed future/unmatched row, must make Final fail.
        "unmatched_results": [],
        "unmatched_result_count": 0,
    }

    artifacts, metrics, _message, status = runner._stage_evaluate_quality()

    assert status == "failed"
    assert metrics["summary_only_full_quality_missing_count"] == 1
    assert metrics["summary_only_full_quality_contract_verified"] is False
    summary = json.loads(
        Path(artifacts["central_quality_summary_json"])
        .read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["quality_acceptance_identity_contract"][
        "execution_scope"
    ] == "standard_quality_setup_local_tensorrt"
    assert summary["quality_acceptance_identity_contract"][
        "postcondition"
    ]["status"] == "blocked"


def test_full_only_canary_merge_accepts_exact_four_rowless_identities_and_blocks_gaps(
    tmp_path: Path,
) -> None:
    h8 = "orin_nx_hailo8_01"
    h10 = "orin_nx_hailo10_01"
    plan_identities = [
        {"id": "hailo8_full", "source_run_id": "hailo8", "run_id": "hailo8", "setup_id": h8, "backend": "hailo8", "variant": "full", "execution_role": "full_quality_only", "performance_claims_emitted": False},
        {"id": "tensorrt_at_hailo8_full", "source_run_id": "native_full_tensorrt", "run_id": "native_full_tensorrt", "dispatch_run_id": "ort_tensorrt", "setup_id": h8, "backend": "tensorrt", "variant": "full", "execution_role": "full_quality_only", "performance_claims_emitted": False},
        {"id": "hailo10_full", "source_run_id": "hailo10", "run_id": "hailo10", "setup_id": h10, "backend": "hailo10h", "variant": "full", "execution_role": "full_quality_only", "performance_claims_emitted": False},
        {"id": "tensorrt_at_hailo10_full", "source_run_id": "native_full_tensorrt", "run_id": "native_full_tensorrt", "dispatch_run_id": "ort_tensorrt", "setup_id": h10, "backend": "tensorrt", "variant": "full", "execution_role": "full_quality_only", "performance_claims_emitted": False},
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "effective_execution_plan.json").write_text(json.dumps({
        "quality_canary_enabled": True,
        "quality_canary_execution_scope": "full_only",
        "models": ["resnet50"],
        "expected_full_quality_identities": plan_identities,
        "generic_rows_total": 0,
    }), encoding="utf-8")
    results = [
        _vendor_full_only_result("hailo8", h8),
        _trt_full_only_result_for_setup(tmp_path / "trt_h8", h8),
        _vendor_full_only_result("hailo10", h10),
        _trt_full_only_result_for_setup(tmp_path / "trt_h10", h10),
    ]
    for result, expected in zip(results, plan_identities):
        plan_identity = {
            "schema": "onnx-splitpoint/full-only-quality-request-identity",
            "schema_version": 1,
            "quality_canary_id": expected["id"],
            "eval_run_id": "eval-20260721",
            "model_id": "resnet50",
            "setup_id": expected["setup_id"],
            "source_run_id": expected["source_run_id"],
            "backend": expected["backend"],
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }
        seal = {
            "full_only_plan_identity_required": True,
            "full_only_plan_identity_validated": True,
            "full_only_plan_identity": plan_identity,
            "full_only_plan_identity_sha256": _digest(plan_identity),
            "quality_canary_id": expected["id"],
        }
        result.update(copy.deepcopy(seal))
        result["request_identity"].update(copy.deepcopy(seal))
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = "eval-20260721"
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
    }

    merge = runner._merge_central_quality_results(results)
    assert merge["summary_only_full_quality_contract_enabled"] is True
    assert merge["summary_only_full_quality_expected_count"] == 4
    assert merge["summary_only_full_quality_seen_count"] == 4
    assert merge["summary_only_full_quality_completed_count"] == 4
    assert merge["summary_only_full_quality_missing_count"] == 0
    assert merge["summary_only_full_quality_duplicate_count"] == 0
    assert merge["unmatched_result_count"] == 0
    assert merge["updated_models"] == []

    identity_drift_cases = {}
    wrong_eval = copy.deepcopy(results)
    wrong_eval[0]["eval_run_id"] = "replayed-evaluation"
    wrong_eval[0]["request_identity"]["eval_run_id"] = (
        "replayed-evaluation"
    )
    identity_drift_cases["wrong_eval_run_id"] = wrong_eval

    wrong_canary = copy.deepcopy(results)
    wrong_canary[0]["quality_canary_id"] = "different_canary"
    wrong_canary[0]["request_identity"]["quality_canary_id"] = (
        "different_canary"
    )
    identity_drift_cases["wrong_quality_canary_id"] = wrong_canary

    wrong_identity = copy.deepcopy(results)
    for container in (
        wrong_identity[0], wrong_identity[0]["request_identity"],
    ):
        container["full_only_plan_identity"]["quality_canary_id"] = (
            "different_canary"
        )
        container["full_only_plan_identity_sha256"] = _digest(
            container["full_only_plan_identity"]
        )
    identity_drift_cases["wrong_plan_identity"] = wrong_identity

    wrong_identity_sha = copy.deepcopy(results)
    wrong_identity_sha[0]["full_only_plan_identity_sha256"] = "b" * 64
    wrong_identity_sha[0]["request_identity"][
        "full_only_plan_identity_sha256"
    ] = "b" * 64
    identity_drift_cases["wrong_plan_identity_sha256"] = wrong_identity_sha

    for label, drifted_results in identity_drift_cases.items():
        drifted = runner._merge_central_quality_results(drifted_results)
        assert drifted["unmatched_result_count"] == 1, label
        assert drifted["summary_only_full_quality_completed_count"] == 3, label

    wrong_role = copy.deepcopy(results)
    wrong_role[0]["execution_role"] = "performance_owner"
    wrong_role[0]["request_identity"]["execution_role"] = (
        "performance_owner"
    )
    invalid_role = runner._merge_central_quality_results(wrong_role)
    assert invalid_role["unmatched_result_count"] == 1
    assert "invalid_summary_only_execution_role" in invalid_role[
        "unmatched_results"
    ][0]["join_identity_errors"]

    performance_shaped = copy.deepcopy(results)
    performance_shaped[0]["performance_claims_emitted"] = True
    performance_shaped[0]["request_identity"][
        "performance_claims_emitted"
    ] = True
    invalid_performance = runner._merge_central_quality_results(
        performance_shaped
    )
    assert invalid_performance["unmatched_result_count"] == 1
    assert "invalid_summary_only_performance_claims_emitted" in (
        invalid_performance["unmatched_results"][0][
            "join_identity_errors"
        ]
    )

    wrong_backend = copy.deepcopy(results)
    wrong_backend[0]["backend"] = "deepx_m1"
    wrong_backend[0]["request_identity"]["backend"] = "deepx_m1"
    invalid_backend = runner._merge_central_quality_results(wrong_backend)
    assert invalid_backend["unmatched_result_count"] == 1
    assert "invalid_summary_only_backend" in invalid_backend[
        "unmatched_results"
    ][0]["join_identity_errors"]

    missing = runner._merge_central_quality_results(results[:-1])
    assert missing["summary_only_full_quality_missing_count"] == 1
    assert missing["unmatched_result_count"] == 1
    assert missing["unmatched_results"][0]["join_status"] == (
        "missing_summary_only_full_quality"
    )

    duplicate = runner._merge_central_quality_results([
        *results, copy.deepcopy(results[0]),
    ])
    assert duplicate["summary_only_full_quality_duplicate_count"] == 1
    assert duplicate["unmatched_result_count"] == 2
    assert {
        row["join_status"] for row in duplicate["unmatched_results"]
    } == {"duplicate_summary_only_full_quality"}


@dataclass(frozen=True)
class _LoadedRequest:
    reference_identity: str | None = None
    request_id: str = "request"


class _ReferenceIdentity:
    def fingerprint(self) -> str:
        return "reference-fingerprint"


class _Service:
    def evaluate(self, request: _LoadedRequest) -> dict[str, Any]:
        return {"status": "completed", "decision": "pass", "n": 1}


class _FailingService:
    def evaluate(self, request: _LoadedRequest) -> dict[str, Any]:
        raise RuntimeError("paired evaluator failed")


def test_evaluator_preserves_producer_only_after_successful_service_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnx_splitpoint_tool.quality_service as quality_service

    request_path, _ = _write_request(tmp_path)
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.run_id = tmp_path.name
    runner._validate_quality_request_model_binding = lambda *_: None
    runner._ensure_central_quality_service = lambda: _Service()
    runner._management_reference_records = lambda _model: (
        [{"image_id": "a.jpg"}],
        {"reference_path": str(tmp_path / "reference.json")},
        b"sealed-reference-test-bytes",
    )
    runner._central_reference_contract = lambda *_args: (
        _ReferenceIdentity(), {"manifest_path": str(tmp_path / "reference-manifest.json")},
    )
    monkeypatch.setattr(
        quality_service, "quality_request_from_manifest",
        lambda *_args, **_kwargs: _LoadedRequest(),
    )

    result = runner._evaluate_central_quality_request("resnet50", request_path)
    assert result["status"] == "completed"
    assert result["producer_binding_eligible"] is True
    assert result["request_identity"]["schema_version"] == 4
    assert result["producer_identity"] == result["request_identity"]["producer_identity"]

    # A producer may be retained for diagnostics after a successful loader,
    # but a failed evaluator can never turn it into bindable evidence.
    runner._ensure_central_quality_service = lambda: _FailingService()
    evaluated_failure = runner._evaluate_central_quality_request(
        "resnet50", request_path,
    )
    assert evaluated_failure["status"] == "failed"
    assert evaluated_failure["producer_binding_eligible"] is False
    assert evaluated_failure["producer_identity"] == evaluated_failure[
        "request_identity"
    ]["producer_identity"]

    def fail_load(*_args: Any, **_kwargs: Any) -> _LoadedRequest:
        raise ValueError("candidate digest mismatch")

    monkeypatch.setattr(quality_service, "quality_request_from_manifest", fail_load)
    failed = runner._evaluate_central_quality_request("resnet50", request_path)
    assert failed["status"] == "failed"
    assert failed["producer_binding_eligible"] is False
    assert "producer_identity" not in failed
    assert "producer_identity" not in failed["request_identity"]
    assert failed["request_identity"]["identity_valid"] is False
