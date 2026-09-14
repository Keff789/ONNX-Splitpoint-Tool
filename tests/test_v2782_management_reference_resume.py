from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    quality_request_from_manifest,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


MODEL_ID = "yolo11l"
SOURCE_CONTRACT = "a" * 64


def _encoded(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reference_payload(*, image_id: str = "image-a") -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "classification",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "records": [
            {
                "image_id": image_id,
                "label_id": 1,
                "label_name": "one",
                "reference": {"top1_hit": True, "top5_hit": True},
            }
        ],
    }


def _reference_path(run_dir: Path, *, immutable: bool) -> Path:
    base = run_dir / "quality_management" / "references" / MODEL_ID
    if immutable:
        base = base / "by_source_contract" / SOURCE_CONTRACT
    return base / "canonical_cpu_reference.json"


def _write_reference(run_dir: Path, *, immutable: bool) -> tuple[Path, bytes]:
    path = _reference_path(run_dir, immutable=immutable)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encoded(_reference_payload())
    path.write_bytes(encoded)
    return path, encoded


def _status(path: Path, encoded: bytes, *, immutable: bool) -> dict[str, Any]:
    status: dict[str, Any] = {
        "schema": "onnx-splitpoint/management-cpu-reference-job",
        "schema_version": 1,
        "status": "completed",
        "model_id": MODEL_ID,
        "reference_path": str(path),
        "reference_sha256": hashlib.sha256(encoded).hexdigest(),
        "reference_size_bytes": len(encoded),
        "execution_location": "central_management",
        "provider": "onnxruntime_cpu",
        "semantic_reference_only": True,
        "include_in_latency_fps_energy": False,
        "include_in_ranking": False,
        "include_in_pareto": False,
        "source_contract_sha256": SOURCE_CONTRACT,
        "task": "classification",
        "record_count": 1,
        "image_ids_sha256": image_ids_fingerprint(["image-a"]),
    }
    if immutable:
        status.update(
            {
                "reference_storage": "immutable_source_contract",
                "reference_immutable": True,
            }
        )
    return status


def _runner(run_dir: Path, status: dict[str, Any]) -> EvaluationWorkflowRunner:
    future: concurrent.futures.Future[dict[str, Any]] = (
        concurrent.futures.Future()
    )
    future.set_result(status)
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner._management_reference_futures = {MODEL_ID: future}
    runner._management_reference_source_contracts = {
        MODEL_ID: SOURCE_CONTRACT,
    }
    return runner


@pytest.mark.parametrize("immutable", [False, True])
def test_active_runner_admits_only_status_bound_legacy_or_immutable_reference(
    tmp_path: Path,
    immutable: bool,
) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=immutable)
    runner = _runner(run_dir, _status(path, encoded, immutable=immutable))

    records, sealed_status, reference_bytes = (
        runner._management_reference_records(MODEL_ID)
    )

    assert records == _reference_payload()["records"]
    assert sealed_status["reference_path"] == str(path)
    assert sealed_status["reference_size_bytes"] == len(encoded)
    assert sealed_status["reference_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert reference_bytes == encoded
    assert "_active_workflow_reference_bytes" not in sealed_status
    json.dumps(sealed_status)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "status contract"),
        ("failed_status", "reference failed"),
        ("model", "model binding"),
        ("provider", "status semantics"),
        ("size", "size differs|size mismatch"),
        ("sha", "SHA-256 mismatch"),
        ("record_count", "metadata differs"),
        ("image_ids", "Image-ID binding"),
        ("source_contract", "active workflow"),
        ("storage", "immutable marker"),
        ("immutable_marker", "immutable marker"),
        ("outside", "escapes the active run"),
        ("latest", "path layout"),
    ],
)
def test_active_runner_rejects_manipulated_reference_status(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=True)
    status = _status(path, encoded, immutable=True)

    if mutation == "schema":
        status["schema"] = "attacker/status"
    elif mutation == "failed_status":
        status["status"] = "failed"
    elif mutation == "model":
        status["model_id"] = "yolo26m"
    elif mutation == "provider":
        status["provider"] = "onnxruntime_cuda"
    elif mutation == "size":
        status["reference_size_bytes"] = len(encoded) + 1
    elif mutation == "sha":
        status["reference_sha256"] = "f" * 64
    elif mutation == "record_count":
        status["record_count"] = 2
    elif mutation == "image_ids":
        status["image_ids_sha256"] = "f" * 64
    elif mutation == "source_contract":
        status["source_contract_sha256"] = "b" * 64
    elif mutation == "storage":
        status["reference_storage"] = "mutable_latest"
    elif mutation == "immutable_marker":
        status["reference_immutable"] = False
    elif mutation == "outside":
        outside = tmp_path / "outside" / "canonical_cpu_reference.json"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(encoded)
        status["reference_path"] = str(outside)
    elif mutation == "latest":
        latest = (
            run_dir
            / "quality_management/references"
            / MODEL_ID
            / "latest/canonical_cpu_reference.json"
        )
        latest.parent.mkdir(parents=True)
        latest.write_bytes(encoded)
        status["reference_path"] = str(latest)
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=message):
        _runner(run_dir, status)._management_reference_records(MODEL_ID)


def test_active_runner_rejects_reference_bytes_changed_after_status(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=True)
    status = _status(path, encoded, immutable=True)
    changed = encoded.replace(b"image-a", b"image-b")
    assert len(changed) == len(encoded)
    path.write_bytes(changed)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _runner(run_dir, status)._management_reference_records(MODEL_ID)


@pytest.mark.parametrize("link_kind", ["leaf", "directory"])
def test_active_runner_rejects_symlink_reference_components(
    tmp_path: Path,
    link_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=True)
    status = _status(path, encoded, immutable=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    if link_kind == "leaf":
        target = outside / path.name
        path.replace(target)
        path.symlink_to(target)
    else:
        contract_dir = path.parent
        moved = outside / "contract"
        contract_dir.replace(moved)
        contract_dir.symlink_to(moved, target_is_directory=True)

    with pytest.raises(RuntimeError, match="unsafe or unavailable"):
        _runner(run_dir, status)._management_reference_records(MODEL_ID)


def _write_quality_request(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    reference = _reference_payload()
    reference_path = root / "management_reference.json"
    reference_encoded = _encoded(reference)
    reference_path.write_bytes(reference_encoded)

    candidate_payload = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "task": "classification",
        "variant": "full",
        "pairing_key": "image_id",
        "records": [
            {
                "image_id": "image-a",
                "label_id": 1,
                "label_name": "one",
                "candidate": {"top1_hit": True, "top5_hit": True},
            }
        ],
    }
    candidate_path = root / "full_candidate.json"
    candidate_encoded = _encoded(candidate_payload)
    candidate_path.write_bytes(candidate_encoded)
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "status": "pending_central_evaluation",
        "task": "classification",
        "variant": "full",
        "pairing_key": "image_id",
        "execution_location": "management_node",
        "reference": {
            "source": "management_cpu_reference",
            "required": True,
        },
        "candidate": {
            "path": candidate_path.name,
            "size_bytes": len(candidate_encoded),
            "sha256": hashlib.sha256(candidate_encoded).hexdigest(),
        },
        "record_count": 1,
        "reference_record_count": 1,
        "policy_sha256": "b" * 64,
        "metric_gate_config": {
            "primary_metric": "top1_accuracy",
            "non_inferiority_margin": 0.01,
            "guardrails": {"top5_accuracy_margin": 0.01},
        },
        "statistics": {
            "method": "paired_bootstrap",
            "bootstrap_repetitions": 30,
            "seed": 20260710,
            "confidence_level": 0.95,
            "decision": "lower_one_sided_bound",
        },
    }
    request_path = root / "full_request.json"
    request_path.write_bytes(_encoded(request))
    descriptor = {
        "path": str(reference_path),
        "size_bytes": len(reference_encoded),
        "sha256": hashlib.sha256(reference_encoded).hexdigest(),
    }
    return request_path, reference_path, descriptor


def test_management_descriptor_is_reverified_at_actual_quality_load(
    tmp_path: Path,
) -> None:
    request_path, reference_path, descriptor = _write_quality_request(tmp_path)

    loaded = quality_request_from_manifest(
        request_path,
        reference_artifact=descriptor,
    )
    assert loaded.reference_records[0]["image_id"] == "image-a"

    original = reference_path.read_bytes()
    changed = original.replace(b"image-a", b"image-b")
    assert len(changed) == len(original)
    reference_path.write_bytes(changed)
    with pytest.raises(QualityArtifactIntegrityError, match="SHA-256 mismatch"):
        quality_request_from_manifest(
            request_path,
            reference_artifact=descriptor,
        )


def test_active_byte_handoff_survives_path_swap_without_loading_new_payload(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=True)
    records, sealed_status, reference_bytes = _runner(
        run_dir,
        _status(path, encoded, immutable=True),
    )._management_reference_records(MODEL_ID)
    assert records[0]["image_id"] == "image-a"

    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(
        _encoded(_reference_payload(image_id="image-b"))
    )
    path.unlink()
    path.symlink_to(replacement)
    descriptor = {
        "path": sealed_status["reference_path"],
        "size_bytes": sealed_status["reference_size_bytes"],
        "sha256": sealed_status["reference_sha256"],
    }
    request_path, _unused_reference, _unused_descriptor = (
        _write_quality_request(tmp_path / "request")
    )

    loaded = quality_request_from_manifest(
        request_path,
        reference_artifact=descriptor,
        reference_artifact_bytes=reference_bytes,
    )

    assert loaded.reference_records[0]["image_id"] == "image-a"


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_active_byte_handoff_rejects_descriptor_identity_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    request_path, reference_path, descriptor = _write_quality_request(tmp_path)
    reference_bytes = reference_path.read_bytes()
    if field == "size_bytes":
        descriptor[field] = int(descriptor[field]) + 1
    else:
        descriptor[field] = "f" * 64

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="byte-size mismatch|SHA-256 mismatch",
    ):
        quality_request_from_manifest(
            request_path,
            reference_artifact=descriptor,
            reference_artifact_bytes=reference_bytes,
        )


def test_active_runner_rejects_multilink_immutable_reference(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=True)
    status = _status(path, encoded, immutable=True)
    os.link(path, tmp_path / "second-link.json")

    with pytest.raises(RuntimeError, match="not single-link"):
        _runner(run_dir, status)._management_reference_records(MODEL_ID)


def test_active_runner_rejects_parent_model_id_token(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    path, encoded = _write_reference(run_dir, immutable=True)
    status = _status(path, encoded, immutable=True)
    future: concurrent.futures.Future[dict[str, Any]] = (
        concurrent.futures.Future()
    )
    status["model_id"] = ".."
    future.set_result(status)
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner._management_reference_futures = {"..": future}
    runner._management_reference_source_contracts = {"..": SOURCE_CONTRACT}

    with pytest.raises(RuntimeError, match="model binding"):
        runner._management_reference_records("..")
