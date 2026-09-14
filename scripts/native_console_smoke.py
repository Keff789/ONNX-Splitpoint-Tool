#!/usr/bin/env python3
"""Fast, isolated Native integration smokes for development diagnosis.

These commands never update an archived EvaluationRun or create claim evidence.
They either inspect existing artifacts offline or write a tiny hardware replay
into a separate result root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_split_quality_authority import (  # noqa: E402
    resolve_native_split_quality_authority,
)
from onnx_splitpoint_tool.native_command_contract import (  # noqa: E402
    canonical_json_sha256,
    verify_native_command_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (  # noqa: E402
    build_completed_detection_endpoint_attestation,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.native_output_endpoint import (  # noqa: E402
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.hailo_full_contract_promotion import (  # noqa: E402
    promote_verified_hailo_full_contracts as _promote_verified_hailo_full_contracts,
)
from onnx_splitpoint_tool.workflow.runner import (  # noqa: E402
    _native_selection_contract_runs_v270e,
    _native_split_case_support_v270e,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_LAYOUT = "memory_nhwc_to_nchw"
_SPLIT_TRANSFORM_OWNER = "tensorrt_part2_input_bridge"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"native_console_smoke_{path.stem}", path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_token(value: Any, *, label: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    if _SHA256_RE.fullmatch(token) is None:
        raise ValueError(f"{label} is not a SHA-256 digest")
    return token


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                duplicate = True
            payload[key] = value
        return payload

    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_object,
    )
    if duplicate:
        raise ValueError(f"{label} contains duplicate JSON keys")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _positive_shape(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            isinstance(dim, bool)
            or not isinstance(dim, int)
            or dim <= 0
            for dim in value
        )
    ):
        raise ValueError(f"{label} must contain positive integer dimensions")
    return [int(dim) for dim in value]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _local_artifact(
    value: Any,
    *,
    base: Path,
    allowed_root: Path,
    label: str,
) -> Path:
    raw = Path(str(value or "")).expanduser()
    if not str(value or "").strip():
        raise ValueError(f"{label} path is missing")
    path = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    allowed = allowed_root.resolve()
    if not _is_within(path, allowed):
        raise ValueError(f"{label} escapes isolated result root: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    return path


def _verify_file_identity(
    path: Path,
    *,
    sha256: Any,
    size_bytes: Any,
    label: str,
) -> str:
    expected_sha = _sha256_token(sha256, label=f"{label} SHA-256")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise ValueError(f"{label} size is invalid")
    if path.stat().st_size != int(size_bytes):
        raise ValueError(f"{label} size mismatch")
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual_sha


def _verify_external_image(
    payload: Mapping[str, Any], expected_image: Path,
    *, label: str,
) -> str:
    declared = str(
        payload.get("input_image") or payload.get("image") or ""
    ).strip()
    if not declared:
        raise ValueError(f"{label} input image is missing")
    path = Path(declared).expanduser().resolve()
    if path != expected_image.resolve():
        raise ValueError(f"{label} input image path mismatch")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} input image is missing")
    expected = str(
        payload.get("input_image_sha256")
        or payload.get("image_sha256")
        or ""
    )
    actual = _sha256_file(path)
    if _sha256_token(expected, label=f"{label} image SHA-256") != actual:
        raise ValueError(f"{label} input image SHA-256 mismatch")
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("image_sha256"):
        if (
            _sha256_token(
                provenance.get("image_sha256"),
                label=f"{label} provenance image SHA-256",
            )
            != actual
        ):
            raise ValueError(f"{label} provenance image SHA-256 mismatch")
    return actual


def _verify_payload_artifact_set(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    allowed_root: Path,
    required_paths: set[Path],
) -> dict[str, Any]:
    rows = payload.get("payload_artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sealed payload_artifacts are missing")
    declared_set_sha = _sha256_token(
        payload.get("payload_artifacts_sha256"),
        label="payload_artifacts_sha256",
    )
    if canonical_json_sha256(rows) != declared_set_sha:
        raise ValueError("payload_artifacts_sha256 mismatch")
    verified_paths: set[Path] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"payload_artifacts[{index}] is invalid")
        path = _local_artifact(
            raw.get("path"),
            base=manifest_path.parent,
            allowed_root=allowed_root,
            label=f"payload_artifacts[{index}]",
        )
        _verify_file_identity(
            path,
            sha256=raw.get("sha256"),
            size_bytes=raw.get("size_bytes"),
            label=f"payload_artifacts[{index}]",
        )
        verified_paths.add(path)
    missing = sorted(str(path) for path in required_paths - verified_paths)
    if missing:
        raise ValueError(
            "sealed payload set does not cover declared files: "
            + ", ".join(missing)
        )
    return {
        "payload_artifact_count": len(rows),
        "payload_artifacts_sha256": declared_set_sha,
    }


def _strict_output_manifest(
    manifest_path: Path,
    *,
    allowed_root: Path,
    expected_image: Path,
    require_payload_artifacts: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    payload = _strict_json_object(
        manifest_path, label="semantic output manifest",
    )
    if payload.get("schema") != "onnx-splitpoint/runner-output-dump":
        raise ValueError("semantic output manifest schema mismatch")
    _verify_external_image(
        payload, expected_image, label="semantic output manifest",
    )
    rows = payload.get("outputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic output manifest has no outputs")
    names: set[str] = set()
    files: set[Path] = set()
    arrays: dict[str, np.ndarray] = {}
    evidence_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"output[{index}] is invalid")
        name = str(raw.get("name") or "").strip()
        if not name or name in names:
            raise ValueError(f"output[{index}] name is missing or duplicated")
        names.add(name)
        shape = _positive_shape(raw.get("shape"), label=f"output[{index}] shape")
        try:
            dtype = np.dtype(str(raw.get("dtype") or ""))
        except Exception as exc:
            raise ValueError(f"output[{index}] dtype is invalid") from exc
        if dtype.hasobject or dtype.kind not in "fiu":
            raise ValueError(f"output[{index}] dtype is not numeric")
        path = _local_artifact(
            raw.get("file") or raw.get("path"),
            base=manifest_path.parent,
            allowed_root=allowed_root,
            label=f"output[{index}]",
        )
        if path in files:
            raise ValueError(f"output[{index}] file is duplicated")
        files.add(path)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        declared_bytes = (
            raw.get("size_bytes")
            if raw.get("size_bytes") is not None
            else raw.get("bytes")
        )
        if declared_bytes != expected_bytes:
            raise ValueError(f"output[{index}] shape/dtype byte count mismatch")
        digest = _verify_file_identity(
            path,
            sha256=raw.get("sha256"),
            size_bytes=expected_bytes,
            label=f"output[{index}]",
        )
        array = np.fromfile(path, dtype=dtype).reshape(shape)
        if dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"output[{index}] contains non-finite values")
        arrays[name] = array
        evidence_rows.append({
            "name": name,
            "path": str(path),
            "shape": shape,
            "dtype": str(dtype),
            "size_bytes": expected_bytes,
            "sha256": digest,
        })
    sealed: dict[str, Any] = {}
    if require_payload_artifacts:
        sealed = _verify_payload_artifact_set(
            payload,
            manifest_path=manifest_path,
            allowed_root=allowed_root,
            required_paths=files,
        )
    return payload, arrays, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "output_count": len(arrays),
        "outputs": evidence_rows,
        **sealed,
    }


def _semantic_output_validation(
    manifest_path: Path,
    *,
    expected: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    validator = _load_script("validate_output_dumps.py")
    observed, metadata = validator.load_dump(str(manifest_path))
    if set(observed) != set(expected):
        raise ValueError("semantic output validator tensor-name mismatch")
    rows: list[dict[str, Any]] = []
    for name in sorted(observed):
        array = np.asarray(observed[name])
        reference = np.asarray(expected[name])
        if array.shape != reference.shape or array.dtype != reference.dtype:
            raise ValueError(
                f"semantic output validator shape/dtype mismatch: {name}"
            )
        summary = dict(validator.summarize(array))
        if summary.get("finite") is not True:
            raise ValueError(
                f"semantic output validator found non-finite values: {name}"
            )
        rows.append({"name": name, **summary})
    return {
        "schema": "onnx-splitpoint/console-semantic-output-validation",
        "schema_version": 1,
        "ok": bool(rows),
        "manifest": str(manifest_path),
        "manifest_schema": str(metadata.get("schema") or ""),
        "output_count": len(rows),
        "outputs": rows,
        "diagnostic_only": True,
        "claim_eligible": False,
    }


def _strict_boundary_manifest(
    manifest_path: Path,
    *,
    allowed_root: Path,
    expected_image: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _strict_json_object(
        manifest_path, label="semantic boundary manifest",
    )
    if payload.get("schema") != "onnx-splitpoint/native-boundary-dump":
        raise ValueError("semantic boundary manifest schema mismatch")
    _verify_external_image(
        payload, expected_image, label="semantic boundary manifest",
    )
    shape = _positive_shape(payload.get("shape"), label="raw boundary shape")
    runtime_shape = _positive_shape(
        payload.get("runtime_boundary_shape"),
        label="runtime boundary shape",
    )
    if runtime_shape != shape:
        raise ValueError("runtime boundary shape differs from raw shape")
    target_shape = _positive_shape(
        payload.get("trt_input_shape"), label="TensorRT input shape",
    )
    if len(shape) == 3:
        height, width, channels = shape
        expected_target = [1, channels, height, width]
    elif len(shape) == 4 and shape[0] == 1:
        batch, height, width, channels = shape
        expected_target = [batch, channels, height, width]
    else:
        raise ValueError("raw Hailo boundary is not HWC/NHWC")
    if target_shape != expected_target:
        raise ValueError("raw HWC boundary does not map to TensorRT NCHW")
    if payload.get("boundary_layout") != _SPLIT_LAYOUT:
        raise ValueError("boundary_layout is not memory_nhwc_to_nchw")
    if payload.get("layout_transform_owner") != _SPLIT_TRANSFORM_OWNER:
        raise ValueError("layout transform owner is not TensorRT Part2 bridge")
    try:
        dtype = np.dtype(str(payload.get("dtype") or ""))
        trt_dtype = np.dtype(str(payload.get("trt_input_dtype") or ""))
    except Exception as exc:
        raise ValueError("boundary dtype is invalid") from exc
    if (
        dtype.hasobject or dtype.kind not in "fiu"
        or trt_dtype.hasobject or trt_dtype.kind not in "fiu"
    ):
        raise ValueError("boundary/TRT dtype is not numeric")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    boundary_path = _local_artifact(
        payload.get("file"),
        base=manifest_path.parent,
        allowed_root=allowed_root,
        label="boundary payload",
    )
    boundary_sha = _verify_file_identity(
        boundary_path,
        sha256=payload.get("file_sha256"),
        size_bytes=expected_bytes,
        label="boundary payload",
    )
    for field in ("nbytes", "file_size_bytes"):
        if payload.get(field) is not None and payload.get(field) != expected_bytes:
            raise ValueError(f"boundary {field} mismatch")
    array = np.fromfile(boundary_path, dtype=dtype).reshape(shape)
    if dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("boundary payload contains non-finite values")
    expected_trt_bytes = (
        int(np.prod(target_shape, dtype=np.int64)) * trt_dtype.itemsize
    )
    if payload.get("trt_input_bytes") != expected_trt_bytes:
        raise ValueError("TensorRT input byte count mismatch")
    input_shape = _positive_shape(
        payload.get("input_shape_hwc"), label="input RGB shape",
    )
    if len(input_shape) != 3 or input_shape[-1] not in {1, 3, 4}:
        raise ValueError("input dump shape is not HWC")
    input_path = _local_artifact(
        payload.get("input_dump"),
        base=manifest_path.parent,
        allowed_root=allowed_root,
        label="input RGB dump",
    )
    input_bytes = int(np.prod(input_shape, dtype=np.int64))
    input_sha = _verify_file_identity(
        input_path,
        sha256=payload.get("input_dump_sha256"),
        size_bytes=input_bytes,
        label="input RGB dump",
    )
    sealed = _verify_payload_artifact_set(
        payload,
        manifest_path=manifest_path,
        allowed_root=allowed_root,
        required_paths={boundary_path, input_path},
    )
    return payload, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "raw_shape": shape,
        "trt_input_shape": target_shape,
        "raw_dtype": str(dtype),
        "trt_input_dtype": str(trt_dtype),
        "boundary_layout": _SPLIT_LAYOUT,
        "layout_transform_owner": _SPLIT_TRANSFORM_OWNER,
        "boundary_payload_sha256": boundary_sha,
        "input_dump_sha256": input_sha,
        **sealed,
    }


def _isolated_manifest_path(
    value: Any, *, child_root: Path, label: str,
) -> Path:
    return _local_artifact(
        value, base=child_root, allowed_root=child_root, label=label,
    )


def _verify_split_command_and_attestation(
    report: Mapping[str, Any],
    *,
    output_manifest: Path,
    boundary_manifest: Path,
    child_root: Path,
    model_id: str,
    case_id: str,
    setup_id: str,
    eval_run_id: str,
    task: str,
) -> dict[str, Any]:
    command = report.get("native_command_contract")
    expected_identity = {
        "backend": "hailo10h_to_trt",
        "model": model_id,
        "case": case_id,
        "precision": str(report.get("precision") or ""),
        "setup_id": setup_id,
        "comparison_backend": "hailo10h",
    }
    verified, command_status = verify_native_command_contract(
        command, expected_identity=expected_identity,
    )
    if verified is None:
        raise ValueError(f"native command contract rejected: {command_status}")
    command_sha = _sha256_token(
        verified.get("contract_sha256"), label="command contract SHA-256",
    )
    if (
        _sha256_token(
            report.get("native_command_contract_sha256"),
            label="report command contract SHA-256",
        )
        != command_sha
    ):
        raise ValueError("report command contract SHA-256 mismatch")
    boundary_contract = verified.get("boundary_contract")
    if (
        not isinstance(boundary_contract, Mapping)
        or boundary_contract.get("boundary_layout_effective") != _SPLIT_LAYOUT
    ):
        raise ValueError("command contract boundary layout mismatch")
    artifacts = verified.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    actual_hashes = {
        "semantic_output_manifest": _sha256_file(output_manifest),
        "semantic_boundary_manifest": _sha256_file(boundary_manifest),
    }
    for name, path in (
        ("semantic_output_manifest", output_manifest),
        ("semantic_boundary_manifest", boundary_manifest),
    ):
        raw = artifacts.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"command contract artifact missing: {name}")
        declared_path = _isolated_manifest_path(
            raw.get("path"), child_root=child_root,
            label=f"command contract {name}",
        )
        if declared_path != path:
            raise ValueError(f"command contract {name} path mismatch")
        _verify_file_identity(
            path,
            sha256=raw.get("sha256"),
            size_bytes=raw.get("size_bytes"),
            label=f"command contract {name}",
        )
    raw_attestation = report.get(
        "native_split_quality_consumer_attestation"
    )
    if not isinstance(raw_attestation, Mapping):
        raise ValueError("consumer attestation is missing")
    attestation = dict(raw_attestation)
    declared_attestation_sha = _sha256_token(
        attestation.pop("attestation_sha256", ""),
        label="consumer attestation SHA-256",
    )
    recomputed_attestation_sha = canonical_json_sha256(attestation)
    if declared_attestation_sha != recomputed_attestation_sha:
        raise ValueError("consumer attestation SHA-256 mismatch")
    expected_fields = {
        "schema": (
            "onnx-splitpoint/native-split-quality-consumer-attestation"
        ),
        "eval_run_id": eval_run_id,
        "backend": "hailo10h_to_trt",
        "model_id": model_id,
        "case_id": case_id,
        "setup_id": setup_id,
        "task": task,
        "command_contract_sha256": command_sha,
        "semantic_output_manifest_sha256": actual_hashes[
            "semantic_output_manifest"
        ],
        "semantic_boundary_manifest_sha256": actual_hashes[
            "semantic_boundary_manifest"
        ],
    }
    for field, expected in expected_fields.items():
        if str(attestation.get(field) or "") != str(expected):
            raise ValueError(f"consumer attestation {field} mismatch")
    binding_sha = _sha256_token(
        report.get("native_split_quality_binding_sha256"),
        label="quality binding SHA-256",
    )
    if (
        _sha256_token(
            attestation.get("binding_sha256"),
            label="attestation quality binding SHA-256",
        )
        != binding_sha
    ):
        raise ValueError("consumer attestation binding SHA-256 mismatch")
    return {
        "command_contract_status": command_status,
        "command_contract_sha256": command_sha,
        "consumer_attestation_sha256": declared_attestation_sha,
        **actual_hashes,
    }


def _verify_final_split_semantic_join(
    report: Mapping[str, Any],
    *,
    report_path: Path | None,
    output_manifest: Path,
) -> dict[str, Any]:
    """Run the same portable semantic join used by the final collector."""
    final = _load_script("native_producer_final_report.py")
    evidence = final._verify_split_semantic_artifacts(
        result_path=report_path,
        manifest_path=output_manifest,
        manifest_sha256=_sha256_file(output_manifest),
        sources=[dict(report)],
    )
    expected = {
        "native_split_semantic_binding_required": True,
        "native_split_semantic_binding_valid": True,
        "native_split_final_portable_binding_valid": True,
        "native_split_semantic_binding_status": (
            "sealed_manifest_and_payload_bytes_rehashed"
        ),
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            status = str(
                evidence.get("native_split_semantic_binding_status")
                or evidence.get("native_split_final_portable_binding_status")
                or "unknown"
            )
            raise ValueError(
                f"final split semantic join rejected: {field} "
                f"(status={status})"
            )
    return dict(evidence)


def _strict_full_input_manifest(
    manifest_path: Path,
    *,
    child_root: Path,
    expected_image: Path,
) -> dict[str, Any]:
    payload = _strict_json_object(
        manifest_path, label="Native Full input manifest",
    )
    if payload.get("schema") != "onnx-splitpoint/native-full-input-dump":
        raise ValueError("Native Full input manifest schema mismatch")
    image_sha = _verify_external_image(
        payload, expected_image, label="Native Full input manifest",
    )
    input_shape = _positive_shape(
        payload.get("input_shape_hwc"), label="Native Full RGB input shape",
    )
    if len(input_shape) != 3 or input_shape[-1] not in {1, 3, 4}:
        raise ValueError("Native Full RGB input shape is not HWC")
    input_path = _local_artifact(
        payload.get("input_dump"),
        base=manifest_path.parent,
        allowed_root=child_root,
        label="Native Full RGB input dump",
    )
    expected_input_bytes = int(np.prod(input_shape, dtype=np.int64))
    if input_path.stat().st_size != expected_input_bytes:
        raise ValueError("Native Full RGB input byte count mismatch")
    runtime_shape = _positive_shape(
        payload.get("runtime_input_shape"),
        label="Native Full runtime input shape",
    )
    try:
        runtime_dtype = np.dtype(
            str(payload.get("runtime_input_dtype") or "")
        )
    except Exception as exc:
        raise ValueError("Native Full runtime input dtype is invalid") from exc
    if runtime_dtype.hasobject or runtime_dtype.kind not in "fiu":
        raise ValueError("Native Full runtime input dtype is not numeric")
    runtime_path = _local_artifact(
        payload.get("runtime_input_file"),
        base=manifest_path.parent,
        allowed_root=child_root,
        label="Native Full runtime input",
    )
    runtime_bytes = (
        int(np.prod(runtime_shape, dtype=np.int64)) * runtime_dtype.itemsize
    )
    runtime_sha = _verify_file_identity(
        runtime_path,
        sha256=payload.get("runtime_input_sha256"),
        size_bytes=runtime_bytes,
        label="Native Full runtime input",
    )
    if payload.get("runtime_input_bytes") != runtime_bytes:
        raise ValueError("Native Full runtime input declared byte mismatch")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "image_sha256": image_sha,
        "input_dump_sha256": _sha256_file(input_path),
        "runtime_input_sha256": runtime_sha,
        "runtime_input_shape": runtime_shape,
        "runtime_input_dtype": str(runtime_dtype),
    }


def _hailo_contract_backend(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("native_full_"):
        token = token[len("native_full_"):]
    return "hailo10" if token in {"hailo10", "hailo10h"} else token


def _prepare_hailo10_full_contract_overlay(
    benchmark_set: Path,
    *,
    child_root: Path,
    model_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Reconcile an old suite declaration without modifying that suite."""
    benchmark_path = benchmark_set / "benchmark_set.json"
    contracts_path = benchmark_set / "output_contracts.json"
    suite_bench = _strict_json_object(
        benchmark_path, label="BenchmarkSet contract",
    )
    source_payload = _strict_json_object(
        contracts_path, label="suite output contracts",
    )
    if str(source_payload.get("model_id") or "") != str(model_id):
        raise ValueError("suite output-contract model mismatch")
    if str(source_payload.get("task") or "").strip().lower() != "detection":
        raise ValueError("Hailo10 Full console smoke requires detection contracts")
    raw_contracts = source_payload.get("contracts")
    if (
        not isinstance(raw_contracts, list)
        or not raw_contracts
        or any(not isinstance(row, Mapping) for row in raw_contracts)
    ):
        raise ValueError("suite output contracts are missing or invalid")
    contracts = [dict(row) for row in raw_contracts]
    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=benchmark_set,
        model_id=str(model_id),
        task="detection",
        suite_bench=suite_bench,
        contracts=contracts,
        copied_verified={},
    )
    hailo10_promotions = [
        dict(row) for row in promotions
        if isinstance(row, Mapping)
        and _hailo_contract_backend(row.get("backend")) == "hailo10"
    ]
    matching_indexes = [
        index for index, row in enumerate(contracts)
        if _hailo_contract_backend(row.get("backend")) == "hailo10"
        and str(row.get("model_id") or "") == str(model_id)
        and str(row.get("variant") or "full").strip().lower() == "full"
    ]
    if len(hailo10_promotions) != 1 or len(matching_indexes) != 1:
        raise ValueError(
            "exactly one verified Hailo10 Full contract promotion is required"
        )
    index = matching_indexes[0]
    contract = dict(contracts[index])
    if (
        contract.get("contract_reconciliation_status")
        != "verified_suite_artifact_raw_head"
        or str(contract.get("endpoint_mode") or "").strip().lower()
        not in {"raw_head", "raw_detection_head"}
        or contract.get("host_tail_required") is not True
        or contract.get("postprocessing_required") is not True
        or len(list(contract.get("full_end_node_names") or [])) != 6
    ):
        raise ValueError("Hailo10 Full raw-head reconciliation is incomplete")
    artifact_root = benchmark_set.resolve()
    for field in ("artifact_path", "recorded_artifact_path"):
        raw_path = Path(str(contract.get(field) or "")).expanduser()
        artifact = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (benchmark_set / raw_path).resolve()
        )
        if not _is_within(artifact, artifact_root) or not artifact.is_file():
            raise ValueError(f"reconciled Hailo10 artifact is unavailable: {field}")
        contract[field] = str(artifact)
    contracts[index] = contract
    overlay_dir = child_root / "contract_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = overlay_dir / "output_contracts.json"
    overlay_payload = {
        **source_payload,
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": int(source_payload.get("schema_version") or 1),
        "model_id": str(model_id),
        "task": "detection",
        "contracts": contracts,
        "diagnostic_overlay": True,
        "diagnostic_only": True,
        "claim_eligible": False,
        "source_benchmark_set": str(benchmark_set),
        "source_benchmark_set_sha256": _sha256_file(benchmark_path),
        "source_output_contracts_sha256": _sha256_file(contracts_path),
    }
    overlay_path.write_text(
        json.dumps(
            overlay_payload, indent=2, sort_keys=True, ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    declaration = load_authoritative_output_contract(
        overlay_path,
        backend="hailo10h",
        model_id=str(model_id),
        variant="full",
        task="detection",
    )
    if (
        declaration.get("contract_resolution_status") != "attested"
        or declaration.get("authoritative_output_contract") is not True
        or declaration.get("stage") != "raw_head"
        or len(list(declaration.get("full_end_node_names") or [])) != 6
    ):
        raise ValueError(
            "isolated Hailo10 Full contract overlay is not authoritative raw_head"
        )
    return overlay_path, {
        "path": str(overlay_path),
        "sha256": _sha256_file(overlay_path),
        "source_benchmark_set": str(benchmark_set),
        "source_benchmark_set_sha256": _sha256_file(benchmark_path),
        "source_output_contracts": str(contracts_path),
        "source_output_contracts_sha256": _sha256_file(contracts_path),
        "promotion": hailo10_promotions[0],
        "resolution_status": declaration["contract_resolution_status"],
        "stage": declaration["stage"],
        "full_end_node_names": list(
            declaration.get("full_end_node_names") or []
        ),
        "diagnostic_only": True,
        "claim_eligible": False,
    }


def _validate_hailo10_full_evidence(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    child_root: Path,
    expected_image: Path,
    expected_frames: int,
    expected_model_id: str,
    expected_setup_id: str,
    expected_declaration: Path,
) -> dict[str, Any]:
    expected_identity = {
        "task": "detection",
        "backend": "native_full_hailo10h",
        "model": expected_model_id,
        "setup_id": expected_setup_id,
        "comparison_backend": "hailo10h",
        "hw_arch": "hailo10h",
        "runtime_api": "infer_model",
    }
    for field, expected in expected_identity.items():
        if str(report.get(field) or "") != str(expected):
            raise ValueError(f"Hailo10 Full report {field} mismatch")
    if (
        report.get("diagnostic_only") is not True
        or report.get("claim_eligible") is not False
        or report.get("claim_eligible_e2e") is not False
    ):
        raise ValueError("Hailo10 Full child report is not claim-free")
    if (
        report.get("throughput_mode") is not True
        or report.get("copy_outputs") is not True
        or report.get("claim_copy_outputs_verified") is not True
    ):
        raise ValueError("Hailo10 Full measured output-copy path is not verified")
    throughput = report.get("throughput")
    if not isinstance(throughput, Mapping):
        raise ValueError("Hailo10 Full throughput evidence is missing")
    try:
        fps = float(throughput.get("fps"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Hailo10 Full throughput FPS is invalid") from exc
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("Hailo10 Full throughput FPS is not positive")
    for field in (
        "frames",
        "requested_frames",
        "minimum_requested_frames",
        "completed_frames",
        "completed_work_units",
    ):
        if int(throughput.get(field) or 0) != expected_frames:
            raise ValueError(f"Hailo10 Full throughput {field} mismatch")
    if (
        int(report.get("completed_frames") or 0) != expected_frames
        or report.get("completed_work_units_status") != "exact_runtime_counter"
        or throughput.get("completed_work_units_status")
        != "exact_runtime_counter"
        or throughput.get("completed_work_units_source")
        != "hailo_infermodel_frozen_postprocess_success_callback_counter"
        or throughput.get("measurement_control") != "exact_frames"
        or int(throughput.get("warmup_frames") or 0) != 0
        or int(throughput.get("warmup_completed_frames") or 0) != 0
        or int(throughput.get("inflight") or 0) != 1
        or throughput.get("postprocess_included") is not True
        or int(throughput.get("postprocess_completed_frames") or 0)
        != expected_frames
        or throughput.get("postprocess_completion_status")
        != "exact_runtime_counter"
    ):
        raise ValueError("Hailo10 Full exact measurement counters are invalid")
    output_manifest = _isolated_manifest_path(
        report.get("output_manifest"),
        child_root=child_root,
        label="Native Full output manifest",
    )
    input_manifest = _isolated_manifest_path(
        report.get("input_manifest"),
        child_root=child_root,
        label="Native Full input manifest",
    )
    output_payload, outputs, output_evidence = _strict_output_manifest(
        output_manifest,
        allowed_root=child_root,
        expected_image=expected_image,
        require_payload_artifacts=False,
    )
    input_evidence = _strict_full_input_manifest(
        input_manifest,
        child_root=child_root,
        expected_image=expected_image,
    )
    manifest_identity = {
        "task": "detection",
        "backend": "native_full_hailo10h",
        "model": expected_model_id,
        "setup_id": expected_setup_id,
        "comparison_backend": "hailo10h",
    }
    for field, expected in manifest_identity.items():
        actual = (
            output_payload.get("producer")
            if field == "backend"
            else output_payload.get(field)
        )
        if str(actual or "") != str(expected):
            raise ValueError(f"Hailo10 Full output manifest {field} mismatch")
    if (
        output_payload.get("diagnostic_only") is not True
        or output_payload.get("claim_eligible") is not False
        or output_payload.get("claim_eligible_e2e") is not False
    ):
        raise ValueError("Hailo10 Full output manifest is not claim-free")
    declaration = output_payload.get(
        "authoritative_output_contract_resolution"
    )
    if (
        not isinstance(declaration, Mapping)
        or declaration.get("contract_resolution_status") != "attested"
        or declaration.get("authoritative_output_contract") is not True
        or declaration.get("stage") != "raw_head"
        or declaration.get("artifact_binding_status") != "verified"
        or declaration.get("contract_reconciliation_status")
        != "verified_suite_artifact_raw_head"
        or len(list(declaration.get("full_end_node_names") or [])) != 6
        or Path(str(declaration.get("declaration_source") or "")).resolve()
        != expected_declaration.resolve()
    ):
        raise ValueError(
            "Hailo10 Full isolated raw-head declaration was not consumed"
        )
    declared_artifact = Path(str(
        declaration.get("recorded_artifact_path")
        or declaration.get("artifact_path")
        or ""
    )).expanduser().resolve()
    report_artifact = Path(str(report.get("hef") or "")).expanduser().resolve()
    if report_artifact != declared_artifact:
        raise ValueError("Hailo10 Full report HEF differs from declaration")
    _verify_file_identity(
        declared_artifact,
        sha256=(
            declaration.get("recorded_artifact_sha256")
            or declaration.get("artifact_sha256")
        ),
        size_bytes=(
            declaration.get("recorded_artifact_size_bytes")
            or declaration.get("artifact_size_bytes")
        ),
        label="Hailo10 Full declared HEF",
    )
    if len(outputs) != 6:
        raise ValueError("Hailo10 Full must expose exactly six raw heads")
    if (
        report.get("runtime_endpoint_contract_family") != "raw_head"
        or output_payload.get("contract_family") != "raw_head"
    ):
        raise ValueError("Hailo10 Full endpoint is not raw_head")
    if (
        report.get("host_postprocess_frozen") is not True
        or report.get("postprocess_included") is not True
        or output_payload.get("host_postprocess_frozen") is not True
        or output_payload.get("postprocess_included") is not True
    ):
        raise ValueError("Hailo10 Full frozen host postprocess is not sealed")
    completed = int(report.get("completed_frames") or 0)
    postprocess_completed = int(
        report.get("postprocess_completed_frames") or 0
    )
    if completed != expected_frames or postprocess_completed != completed:
        raise ValueError("Hailo10 Full/postprocess frame counts mismatch")
    report_contract = report.get("frozen_host_postprocess_contract")
    manifest_contract = output_payload.get(
        "frozen_host_postprocess_contract"
    )
    if (
        not isinstance(report_contract, Mapping)
        or not isinstance(manifest_contract, Mapping)
        or dict(report_contract) != dict(manifest_contract)
    ):
        raise ValueError("Hailo10 Full frozen contract copies differ")
    verified_contract = verify_frozen_postprocess_contract(
        report_contract, outputs=outputs,
    )
    contract_sha = _sha256_token(
        verified_contract.get("contract_sha256"),
        label="frozen postprocess contract SHA-256",
    )
    for label, value in (
        (
            "report frozen contract SHA-256",
            report.get("frozen_host_postprocess_contract_sha256"),
        ),
        (
            "manifest frozen contract SHA-256",
            output_payload.get("frozen_host_postprocess_contract_sha256"),
        ),
    ):
        if _sha256_token(value, label=label) != contract_sha:
            raise ValueError(f"{label} mismatch")
    signature = verified_contract.get("raw_output_tensor_signature")
    if (
        not isinstance(signature, Mapping)
        or int(signature.get("tensor_count") or 0) != 6
    ):
        raise ValueError("frozen contract does not bind six raw heads")
    report_result = report.get("frozen_host_postprocess_result")
    manifest_result = output_payload.get("frozen_host_postprocess_result")
    if (
        not isinstance(report_result, Mapping)
        or not isinstance(manifest_result, Mapping)
    ):
        raise ValueError("Hailo10 Full frozen result evidence is missing")
    result_hashes: dict[str, str] = {}
    for role, result in (
        ("measured", report_result),
        ("dump_sample", manifest_result),
    ):
        if (
            result.get("contract_family") != "decoded_nms"
            or _sha256_token(
                result.get("postprocess_contract_sha256"),
                label=f"{role} postprocess result contract SHA-256",
            )
            != contract_sha
        ):
            raise ValueError(
                f"Hailo10 Full {role} postprocess result contract mismatch"
            )
        result_hashes[role] = _sha256_token(
            result.get("detections_sha256"),
            label=f"{role} postprocess detections SHA-256",
        )
    semantic_validation = _semantic_output_validation(
        output_manifest, expected=outputs,
    )
    postprocess = _load_script("offline_native_detection_postprocess.py")
    offline_report = postprocess.audit_pack(
        child_root, derive_unbound_raw=False,
    )
    if (
        offline_report.get("ok") is not True
        or int(offline_report.get("detection_manifest_count") or 0) != 1
        or int(offline_report.get("technical_pass_count") or 0) != 1
    ):
        raise ValueError("offline Hailo10 Full postprocess audit failed")
    offline_rows = [
        row for row in list(offline_report.get("rows") or [])
        if isinstance(row, Mapping)
        and row.get("status") != "skipped_non_detection"
    ]
    if len(offline_rows) != 1:
        raise ValueError("offline Hailo10 Full postprocess row is ambiguous")
    offline_row = offline_rows[0]
    if (
        offline_row.get("technical_ok") is not True
        or offline_row.get("contract_source") != "archived_frozen_contract"
        or _sha256_token(
            offline_row.get("postprocess_contract_sha256"),
            label="offline postprocess contract SHA-256",
        )
        != contract_sha
        or _sha256_token(
            offline_row.get("detections_sha256"),
            label="offline detections SHA-256",
        )
        != result_hashes["dump_sample"]
        or "portable_result_hash_mismatch"
        in list(offline_row.get("warnings") or [])
    ):
        raise ValueError("offline Hailo10 Full contract/result hash mismatch")
    endpoint_sha = _sha256_token(
        output_payload.get("endpoint_contract_hash"),
        label="raw-head endpoint contract SHA-256",
    )
    raw_attestation = output_payload.get("output_endpoint_attestation")
    if (
        output_payload.get("endpoint_contract_complete") is not True
        or not isinstance(raw_attestation, Mapping)
        or raw_attestation.get("attested") is not True
        or raw_attestation.get("status") != "passed"
        or raw_attestation.get("stage") != "raw_head"
        or _sha256_token(
            raw_attestation.get("endpoint_contract_hash"),
            label="raw-head endpoint attestation SHA-256",
        )
        != endpoint_sha
    ):
        raise ValueError("Hailo10 Full raw-head endpoint is not attested")
    if (
        output_payload.get("e2e_scope") != "full_task_pipeline"
        or output_payload.get("postprocess_location")
        != "serialized_host_tail_inside_measured_interval"
        or _sha256_token(
            output_payload.get("decoder_contract_sha256"),
            label="decoder contract SHA-256",
        )
        != contract_sha
        or _sha256_token(
            output_payload.get("nms_contract_sha256"),
            label="NMS contract SHA-256",
        )
        != contract_sha
    ):
        raise ValueError("Hailo10 Full measured host-tail contract is invalid")
    completed_attestation = build_completed_detection_endpoint_attestation(
        verified_contract,
        report_result,
        completed_frames=completed,
        postprocess_completed_frames=postprocess_completed,
        source_endpoint_contract_hash=endpoint_sha,
    )
    if (
        completed_attestation.get("attested") is not True
        or completed_attestation.get("status") != "passed"
        or completed_attestation.get("endpoint") != "decoded_nms"
        or completed_attestation.get("postprocess_completion_verified")
        is not True
    ):
        raise ValueError("Hailo10 Full completed endpoint is not attested")
    return {
        "report": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "output_manifest": output_evidence,
        "input_manifest": input_evidence,
        "semantic_output_validation": semantic_validation,
        "offline_postprocess": offline_report,
        "raw_head_count": len(outputs),
        "raw_head_endpoint_contract_sha256": endpoint_sha,
        "completed_endpoint_attestation": completed_attestation,
        "completed_endpoint_attestation_sha256": canonical_json_sha256(
            completed_attestation
        ),
        "frozen_postprocess_contract_sha256": contract_sha,
        "measured_postprocess_result_sha256": canonical_json_sha256(
            dict(report_result)
        ),
        "dump_sample_postprocess_result_sha256": canonical_json_sha256(
            dict(manifest_result)
        ),
        "measured_detections_sha256": result_hashes["measured"],
        "dump_sample_detections_sha256": result_hashes["dump_sample"],
        "completed_frames": completed,
        "postprocess_completed_frames": postprocess_completed,
        "throughput_fps": fps,
        "completed_work_units_source": throughput[
            "completed_work_units_source"
        ],
        "child_claim_fields_suppressed": True,
    }


def _run_child(command: list[str], *, timeout_s: float) -> dict[str, Any]:
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout must be a positive finite number")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
    return {
        "returncode": 124 if timed_out else int(process.returncode or 0),
        "stdout": str(stdout or ""),
        "stderr": str(stderr or ""),
        "timed_out": timed_out,
        "timeout_s": float(timeout_s),
    }


def _canonical_protected_roots(
    benchmark_set: Path,
    *,
    quality_binding: Path | None = None,
) -> list[Path]:
    benchmark = benchmark_set.resolve()
    protected = [benchmark]
    if (
        benchmark.name.lower() in {"benchmark_set", "benchmarkset"}
        and len(benchmark.parents) >= 2
    ):
        protected.append(benchmark.parent.parent.resolve())
    for ancestor in benchmark.parents:
        marker_count = sum(
            int((ancestor / marker).exists())
            for marker in ("models", "native_producers", "reports", "stages")
        )
        if marker_count >= 2:
            protected.append(ancestor.resolve())
            break
    if quality_binding is not None:
        binding = quality_binding.resolve()
        protected.append(binding.parent)
        for ancestor in binding.parents:
            if ancestor.name.lower() in {
                "quality_first",
                "quality_management",
            }:
                protected.append(ancestor.resolve())
                break
    unique: list[Path] = []
    for path in protected:
        if path not in unique:
            unique.append(path)
    return unique


def _assert_isolated_result_root(
    result_root: Path,
    benchmark_set: Path,
    *,
    quality_binding: Path | None = None,
) -> list[Path]:
    protected_roots = _canonical_protected_roots(
        benchmark_set, quality_binding=quality_binding,
    )
    _assert_outside_protected_roots(result_root, protected_roots)
    return protected_roots


def _assert_outside_protected_roots(
    result_root: Path,
    protected_roots: list[Path],
) -> None:
    result = result_root.resolve()
    for protected in protected_roots:
        protected = protected.resolve()
        if (
            result == protected
            or _is_within(result, protected)
            or _is_within(protected, result)
        ):
            raise ValueError(
                "console result root must not overlap the BenchmarkSet, "
                "EvaluationRun, or Quality-Binding artifacts"
            )


def _resolved_result_root(value: Path | None, label: str) -> Path:
    return (
        value.expanduser().resolve()
        if value is not None
        else (
            Path(tempfile.gettempdir())
            / "onnx_splitpoint_console_smokes"
            / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}"
        ).resolve()
    )


def _result_root(value: Path | None, label: str) -> Path:
    root = _resolved_result_root(value, label)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _isolated_result_root(
    value: Path | None,
    label: str,
    *,
    benchmark_set: Path,
    quality_binding: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _resolved_result_root(value, label)
    protected = _assert_isolated_result_root(
        root,
        benchmark_set,
        quality_binding=quality_binding,
    )
    root.mkdir(parents=True, exist_ok=False)
    return root, {
        "status": "write_root_outside_protected_archives",
        "result_root": str(root),
        "protected_roots": [str(path) for path in protected],
        "checked_before_result_root_creation": True,
    }


def _protected_result_root(
    value: Path | None,
    label: str,
    *,
    protected_roots: list[Path],
) -> tuple[Path, dict[str, Any]]:
    root = _resolved_result_root(value, label)
    unique: list[Path] = []
    for path in protected_roots:
        resolved = path.resolve()
        if resolved not in unique:
            unique.append(resolved)
    _assert_outside_protected_roots(root, unique)
    root.mkdir(parents=True, exist_ok=False)
    return root, {
        "status": "write_root_outside_protected_archives",
        "result_root": str(root),
        "protected_roots": [str(path) for path in unique],
        "checked_before_result_root_creation": True,
    }


def _write_summary(root: Path, payload: dict[str, Any]) -> int:
    payload.update({"diagnostic_only": True, "claim_eligible": False})
    payload.setdefault("archived_evaluation_run_mutated", None)
    payload.setdefault(
        "archived_evaluation_run_mutation_status", "not_applicable",
    )
    path = root / "console_smoke_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ok": payload.get("ok"),
        "kind": payload.get("kind"),
        "summary": str(path),
        "result_root": str(root),
        "diagnostic_only": True,
    }, indent=2))
    return 0 if payload.get("ok") is True else 3


def _row_inventory(
    rows: Any, field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            value = str(row.get(field) or "<missing>").strip() or "<missing>"
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _warning_inventory(rows: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for warning in list(row.get("warnings") or []):
                value = str(warning or "").strip()
                if value:
                    counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _offline_pack(args: argparse.Namespace) -> int:
    result_root = _result_root(args.result_root, "offline_pack")
    postprocess = _load_script("offline_native_detection_postprocess.py")
    boundary = _load_script("offline_native_boundary_contract.py")
    post_report = postprocess.audit_pack(
        args.debug_pack.expanduser().resolve(),
        derive_unbound_raw=bool(args.derive_unbound_raw),
    )
    boundary_report = boundary.audit_pack(
        args.debug_pack.expanduser().resolve(),
    )
    (result_root / "postprocess_audit.json").write_text(
        json.dumps(post_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (result_root / "boundary_audit.json").write_text(
        json.dumps(boundary_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ok = bool(post_report.get("ok") and boundary_report.get("ok"))
    payload = {
        "schema": "onnx-splitpoint/native-console-smoke",
        "schema_version": 1,
        "kind": "offline_pack",
        "source": str(args.debug_pack.expanduser().resolve()),
        "ok": ok,
        "postprocess": {
            "manifest_count": post_report.get("manifest_count"),
            "detection_manifest_count": post_report.get(
                "detection_manifest_count"
            ),
            "technical_pass_count": post_report.get("technical_pass_count"),
            "technical_failure_count": post_report.get(
                "technical_failure_count"
            ),
            "backend_inventory": _row_inventory(
                post_report.get("rows"), "backend",
            ),
            "status_inventory": _row_inventory(
                post_report.get("rows"), "status",
            ),
            "warning_inventory": _warning_inventory(
                post_report.get("rows"),
            ),
        },
        "boundary": {
            "manifest_count": boundary_report.get("manifest_count"),
            "technical_pass_count": boundary_report.get(
                "technical_pass_count"
            ),
            "technical_failure_count": boundary_report.get(
                "technical_failure_count"
            ),
            "eligible_peer_comparison_count": boundary_report.get(
                "eligible_peer_comparison_count"
            ),
            "backend_inventory": _row_inventory(
                boundary_report.get("rows"), "backend",
            ),
            "status_inventory": _row_inventory(
                boundary_report.get("rows"), "status",
            ),
        },
    }
    status = _write_summary(result_root, payload)
    return status if args.strict else 0


def _authority(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    stage = (
        args.stage.expanduser().resolve()
        if args.stage is not None
        else run_dir / "reports" / "native_producer_stage.json"
    )
    protected_roots = [run_dir]
    if not _is_within(stage, run_dir):
        protected_roots.append(stage.parent)
    result_root, isolation = _protected_result_root(
        args.result_root,
        "authority",
        protected_roots=protected_roots,
    )
    authority = resolve_native_split_quality_authority(
        run_manifest_path=run_dir / "run_manifest.json",
        stage_path=stage,
    )
    (result_root / "native_split_quality_authority.json").write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _write_summary(result_root, {
        "schema": "onnx-splitpoint/native-console-smoke",
        "schema_version": 1,
        "kind": "authority",
        "run_dir": str(run_dir),
        "stage": str(stage),
        "ok": authority.get("valid") is True,
        "mode": authority.get("mode"),
        "workflow_version": authority.get("workflow_version"),
        "selection_identity_mode": authority.get("selection_identity_mode"),
        "profile_selection_fingerprint": authority.get(
            "profile_selection_fingerprint"
        ),
        "errors": list(authority.get("errors") or []),
        "archive_isolation": isolation,
        "archived_evaluation_run_mutated": False,
        "archived_evaluation_run_mutation_status": (
            "write_root_verified_outside_protected_archives"
        ),
    })


def _source(args: argparse.Namespace) -> int:
    result_root = _result_root(args.result_root, "source")
    tests = [
        "tests/test_v27548_official_coco_sha.py",
        "tests/test_v27547_yolov7_decoder_contract.py",
        "tests/test_v272_detection_completion_runtime.py",
        "tests/test_v2711_direct_bn6_producer_binding.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    process = subprocess.run(
        command, cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    (result_root / "pytest_stdout.log").write_text(
        process.stdout, encoding="utf-8",
    )
    (result_root / "pytest_stderr.log").write_text(
        process.stderr, encoding="utf-8",
    )
    return _write_summary(result_root, {
        "schema": "onnx-splitpoint/native-console-smoke",
        "schema_version": 1,
        "kind": "source",
        "ok": process.returncode == 0,
        "command": command,
        "returncode": process.returncode,
        "tests": tests,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
    })


def _native_plan(args: argparse.Namespace) -> int:
    benchmark_sets: dict[str, Path] = {}
    for raw in list(args.benchmark_set or []):
        model, separator, value = str(raw or "").partition("=")
        model = model.strip()
        path = Path(value).expanduser().resolve() if separator else Path()
        if (
            not separator or not model or model in benchmark_sets
            or not (path / "benchmark_set.json").is_file()
        ):
            raise ValueError(
                "--benchmark-set must be unique MODEL=/path/to/benchmark_set"
            )
        benchmark_sets[model] = path
    if not benchmark_sets:
        raise ValueError("at least one --benchmark-set is required")

    selected: dict[str, list[str]] = {}
    for model, benchmark_set in benchmark_sets.items():
        selected[model] = sorted(
            child.name
            for child in benchmark_set.iterdir()
            if child.is_dir() and re.fullmatch(r"b[0-9]+", child.name)
        )
        if not selected[model]:
            raise ValueError(f"no split cases found for {model}")

    backends = list(args.backend or ["hailo8", "hailo10h", "deepx"])
    result_root, isolation = _protected_result_root(
        args.result_root,
        "native_plan",
        protected_roots=list(benchmark_sets.values()),
    )
    support = _native_split_case_support_v270e(
        benchmark_sets, selected,
    )
    supported = {
        model: [
            str(row["case"])
            for row in support
            if row.get("model") == model
            and row.get("native_supported") is True
        ]
        for model in benchmark_sets
    }
    supported = {
        model: cases for model, cases in supported.items() if cases
    }
    contracts_by_backend = {
        backend: _native_selection_contract_runs_v270e(
            backend, supported,
        )
        for backend in backends
    }
    planned_rows = [
        {
            "backend": backend,
            "model": model,
            "case": case,
            "precision": str(contract.get("precision") or ""),
            "contract_id": str(contract.get("id") or ""),
        }
        for backend, contracts in contracts_by_backend.items()
        for contract in contracts
        for model, cases in dict(contract.get("case_map") or {}).items()
        for case in list(cases or [])
    ]
    exclusions = [
        {
            **dict(row),
            "backend": backend,
        }
        for backend in backends
        for row in support
        if row.get("native_supported") is not True
    ]
    expected_rows = len(backends) * sum(
        len(cases) for cases in supported.values()
    )
    payload = {
        "schema": "onnx-splitpoint/native-console-plan-audit",
        "schema_version": 1,
        "kind": "native_plan",
        "ok": bool(planned_rows) and len(planned_rows) == expected_rows,
        "selected_case_map": selected,
        "supported_case_map": supported,
        "case_support": support,
        "backends": backends,
        "planned_split_row_count": len(planned_rows),
        "planned_split_rows": planned_rows,
        "capability_exclusion_count": len(exclusions),
        "capability_exclusions": exclusions,
        "generic_multi_io_unchanged": True,
        "archive_isolation": isolation,
        "archived_evaluation_run_mutated": False,
        "archived_evaluation_run_mutation_status": (
            "read_only_plan_audit_with_isolated_output"
        ),
    }
    (result_root / "native_split_plan_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return _write_summary(result_root, payload)


def _hailo10_split(args: argparse.Namespace) -> int:
    benchmark_set = args.benchmark_set.expanduser().resolve()
    quality_binding = args.quality_binding.expanduser().resolve()
    image = args.image.expanduser().resolve()
    result_root, isolation = _isolated_result_root(
        args.result_root,
        "hailo10_split",
        benchmark_set=benchmark_set,
        quality_binding=quality_binding,
    )
    child_root = result_root / "child"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "native_hailo10_trt_e2e_from_benchmarkset.py"),
        "--benchmark-set", str(benchmark_set),
        "--case", str(args.case),
        "--hw-arch", "hailo10h",
        "--frames", str(args.frames),
        "--warmup", "0",
        "--queue-depth", "1",
        "--inflight", "1",
        "--repetitions", "1",
        "--producer-impl", "async_fifo",
        "--copy-outputs",
        "--dump-outputs",
        "--dump-boundary",
        "--image", str(image),
        "--task", str(args.task),
        "--out-dir", str(child_root),
        "--setup-id", str(args.setup_id),
        "--eval-run-id", str(args.eval_run_id),
        "--source-run-id", str(args.source_run_id),
        "--model-id", str(args.model_id),
        "--native-split-quality-binding", str(quality_binding),
    ]
    process = _run_child(command, timeout_s=float(args.timeout_s))
    (result_root / "child_stdout.log").write_text(
        process["stdout"], encoding="utf-8",
    )
    (result_root / "child_stderr.log").write_text(
        process["stderr"], encoding="utf-8",
    )
    report_path = child_root / "hailo10_native_fifo_e2e_results.json"
    report: dict[str, Any] = {}
    gates = {
        "process_not_timed_out": process["timed_out"] is False,
        "process_returncode_zero": process["returncode"] == 0,
        "child_ok": False,
        "completed_requested_frames": False,
        "no_output_dump_error": False,
        "output_manifest_strict": False,
        "boundary_manifest_strict": False,
        "raw_hwc_to_trt_nchw": False,
        "boundary_layout_memory_nhwc_to_nchw": False,
        "transform_owner_tensorrt_part2_bridge": False,
        "native_command_contract_verified": False,
        "consumer_attestation_hash_recomputed": False,
        "final_portable_semantic_join_verified": False,
        "semantic_output_validator_passed": False,
        "offline_boundary_validator_passed": False,
    }
    validation: dict[str, Any] = {}
    validation_error = ""
    try:
        report = _strict_json_object(
            report_path, label="Hailo10 Split child report",
        )
        gates["child_ok"] = report.get("ok") is True
        gates["completed_requested_frames"] = (
            int(report.get("completed_frames") or 0) == int(args.frames)
        )
        gates["no_output_dump_error"] = not bool(
            report.get("output_dump_error")
        )
        output_manifest = _isolated_manifest_path(
            report.get("native_fifo_output_manifest"),
            child_root=child_root,
            label="Hailo10 Split output manifest",
        )
        boundary_manifest = _isolated_manifest_path(
            report.get("native_fifo_boundary_manifest"),
            child_root=child_root,
            label="Hailo10 Split boundary manifest",
        )
        output_payload, outputs, output_evidence = _strict_output_manifest(
            output_manifest,
            allowed_root=child_root,
            expected_image=image,
            require_payload_artifacts=True,
        )
        del output_payload
        gates["output_manifest_strict"] = True
        semantic = _semantic_output_validation(
            output_manifest, expected=outputs,
        )
        (result_root / "semantic_output_validation.json").write_text(
            json.dumps(
                semantic, indent=2, sort_keys=True, ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        gates["semantic_output_validator_passed"] = (
            semantic.get("ok") is True
        )
        _boundary_payload, boundary_evidence = _strict_boundary_manifest(
            boundary_manifest,
            allowed_root=child_root,
            expected_image=image,
        )
        gates["boundary_manifest_strict"] = True
        gates["raw_hwc_to_trt_nchw"] = (
            boundary_evidence["trt_input_shape"]
            == [
                1,
                boundary_evidence["raw_shape"][-1],
                boundary_evidence["raw_shape"][-3],
                boundary_evidence["raw_shape"][-2],
            ]
        )
        gates["boundary_layout_memory_nhwc_to_nchw"] = (
            boundary_evidence["boundary_layout"] == _SPLIT_LAYOUT
        )
        gates["transform_owner_tensorrt_part2_bridge"] = (
            boundary_evidence["layout_transform_owner"]
            == _SPLIT_TRANSFORM_OWNER
        )
        command_evidence = _verify_split_command_and_attestation(
            report,
            output_manifest=output_manifest,
            boundary_manifest=boundary_manifest,
            child_root=child_root,
            model_id=str(args.model_id),
            case_id=str(report.get("case") or args.case),
            setup_id=str(args.setup_id),
            eval_run_id=str(args.eval_run_id),
            task=str(args.task),
        )
        gates["native_command_contract_verified"] = True
        gates["consumer_attestation_hash_recomputed"] = True
        final_join = _verify_final_split_semantic_join(
            report,
            report_path=report_path,
            output_manifest=output_manifest,
        )
        gates["final_portable_semantic_join_verified"] = True
        boundary_validator = _load_script(
            "offline_native_boundary_contract.py"
        )
        boundary_audit = boundary_validator.audit_pack(child_root)
        (result_root / "boundary_offline_audit.json").write_text(
            json.dumps(
                boundary_audit,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        gates["offline_boundary_validator_passed"] = bool(
            boundary_audit.get("ok") is True
            and int(boundary_audit.get("manifest_count") or 0) == 1
            and int(boundary_audit.get("technical_pass_count") or 0) == 1
        )
        validation = {
            "report_sha256": _sha256_file(report_path),
            "output_manifest": output_evidence,
            "boundary_manifest": boundary_evidence,
            "semantic_output_validation": semantic,
            "offline_boundary_audit": boundary_audit,
            "command_and_consumer_attestation": command_evidence,
            "final_portable_semantic_join": final_join,
        }
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
    return _write_summary(result_root, {
        "schema": "onnx-splitpoint/native-console-smoke",
        "schema_version": 1,
        "kind": "hailo10_split",
        "ok": all(gates.values()),
        "command": command,
        "returncode": process["returncode"],
        "timed_out": process["timed_out"],
        "timeout_s": process["timeout_s"],
        "gates": gates,
        "child_report": str(report_path),
        "fps_makespan": report.get("fps_makespan"),
        "runtime_boundary_evidence": (
            (report.get("native_command_contract") or {}).get(
                "runtime_boundary_evidence"
            )
            if isinstance(report.get("native_command_contract"), dict)
            else {}
        ),
        "validation": validation,
        "validation_error": validation_error,
        "archive_isolation": isolation,
        "archived_evaluation_run_mutated": False,
        "archived_evaluation_run_mutation_status": (
            "write_root_verified_outside_protected_archives"
        ),
        "stdout_tail": process["stdout"][-2000:],
        "stderr_tail": process["stderr"][-2000:],
    })


def _hailo10_full(args: argparse.Namespace) -> int:
    benchmark_set = args.benchmark_set.expanduser().resolve()
    image = args.image.expanduser().resolve()
    result_root, isolation = _isolated_result_root(
        args.result_root,
        "hailo10_full",
        benchmark_set=benchmark_set,
    )
    child_root = result_root / "child"
    report_path = child_root / "hailo10_native_full_results.json"
    dump_dir = child_root / "native_full_outputs"
    artifacts_dir = child_root / "artifacts"
    contract_overlay, contract_overlay_evidence = (
        _prepare_hailo10_full_contract_overlay(
            benchmark_set,
            child_root=child_root,
            model_id=str(args.model_id),
        )
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "smoke_hailo10_full_from_benchmarkset.py"),
        "--benchmark-set", str(benchmark_set),
        "--hw-arch", "hailo10h",
        "--model", str(args.model_id),
        "--frames", str(args.frames),
        "--warmup", "0",
        "--inflight", "1",
        "--runtime-api", "infer_model",
        "--image", str(image),
        "--task", "detection",
        "--preprocess-mode", "auto",
        "--dump-outputs",
        "--dump-dir", str(dump_dir),
        "--artifacts-dir", str(artifacts_dir),
        "--declared-output-contract-json", str(contract_overlay),
        "--diagnostic-only",
        "--backend-label", "native_full_hailo10h",
        "--setup-id", str(args.setup_id),
        "--comparison-backend", "hailo10h",
        "--json-out", str(report_path),
    ]
    process = _run_child(command, timeout_s=float(args.timeout_s))
    (result_root / "child_stdout.log").write_text(
        process["stdout"], encoding="utf-8",
    )
    (result_root / "child_stderr.log").write_text(
        process["stderr"], encoding="utf-8",
    )
    report: dict[str, Any] = {}
    gates = {
        "process_not_timed_out": process["timed_out"] is False,
        "process_returncode_zero": process["returncode"] == 0,
        "child_ok": False,
        "completed_five_frames": False,
        "six_raw_heads": False,
        "raw_head_contract_family": False,
        "frozen_host_postprocess": False,
        "postprocess_frames_equal_completed_frames": False,
        "contract_hashes_verified": False,
        "result_hashes_verified": False,
        "semantic_output_validator_passed": False,
        "offline_postprocess_validator_passed": False,
        "completed_endpoint_attestation_verified": False,
        "contract_overlay_verified": False,
        "all_artifacts_isolated": False,
        "child_claim_fields_suppressed": False,
    }
    evidence: dict[str, Any] = {}
    validation_error = ""
    try:
        report = _strict_json_object(
            report_path, label="Hailo10 Full child report",
        )
        gates["child_ok"] = report.get("ok") is True
        evidence = _validate_hailo10_full_evidence(
            report,
            report_path=report_path,
            child_root=child_root,
            expected_image=image,
            expected_frames=int(args.frames),
            expected_model_id=str(args.model_id),
            expected_setup_id=str(args.setup_id),
            expected_declaration=contract_overlay,
        )
        completed = int(evidence["completed_frames"])
        postprocess_completed = int(
            evidence["postprocess_completed_frames"]
        )
        gates.update({
            "completed_five_frames": (
                int(args.frames) == 5 and completed == 5
            ),
            "six_raw_heads": evidence["raw_head_count"] == 6,
            "raw_head_contract_family": True,
            "frozen_host_postprocess": True,
            "postprocess_frames_equal_completed_frames": (
                postprocess_completed == completed
            ),
            "contract_hashes_verified": True,
            "result_hashes_verified": True,
            "semantic_output_validator_passed": (
                evidence["semantic_output_validation"].get("ok") is True
            ),
            "offline_postprocess_validator_passed": (
                evidence["offline_postprocess"].get("ok") is True
            ),
            "completed_endpoint_attestation_verified": (
                evidence["completed_endpoint_attestation"].get("attested")
                is True
            ),
            "contract_overlay_verified": (
                contract_overlay_evidence["resolution_status"] == "attested"
                and contract_overlay_evidence["stage"] == "raw_head"
            ),
            "all_artifacts_isolated": True,
            "child_claim_fields_suppressed": (
                evidence["child_claim_fields_suppressed"] is True
            ),
        })
        (result_root / "postprocess_offline_audit.json").write_text(
            json.dumps(
                evidence["offline_postprocess"],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (result_root / "semantic_output_validation.json").write_text(
            json.dumps(
                evidence["semantic_output_validation"],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
    return _write_summary(result_root, {
        "schema": "onnx-splitpoint/native-console-smoke",
        "schema_version": 1,
        "kind": "hailo10_full",
        "ok": all(gates.values()),
        "command": command,
        "returncode": process["returncode"],
        "timed_out": process["timed_out"],
        "timeout_s": process["timeout_s"],
        "gates": gates,
        "child_report": str(report_path),
        "fps": (report.get("throughput") or {}).get("fps")
        if isinstance(report.get("throughput"), Mapping) else None,
        "validation": evidence,
        "contract_overlay": contract_overlay_evidence,
        "validation_error": validation_error,
        "archive_isolation": isolation,
        "archived_evaluation_run_mutated": False,
        "archived_evaluation_run_mutation_status": (
            "write_root_verified_outside_protected_archives"
        ),
        "stdout_tail": process["stdout"][-2000:],
        "stderr_tail": process["stderr"][-2000:],
    })


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fast, non-claim Native integration checks without a complete "
            "Evaluation Workflow."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser(
        "offline-pack",
        help="Replay frozen detection postprocessing and audit boundaries.",
    )
    offline.add_argument("debug_pack", type=Path)
    offline.add_argument("--result-root", type=Path)
    offline.add_argument(
        "--strict", action=argparse.BooleanOptionalAction, default=True,
    )
    offline.add_argument(
        "--derive-unbound-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    offline.set_defaults(func=_offline_pack)

    authority = sub.add_parser(
        "authority",
        help="Validate start-snapshot/Native-stage Quality-FIRST authority.",
    )
    authority.add_argument("run_dir", type=Path)
    authority.add_argument("--stage", type=Path)
    authority.add_argument("--result-root", type=Path)
    authority.set_defaults(func=_authority)

    source = sub.add_parser(
        "source",
        help="Run the narrow 2.75.48 source regression block.",
    )
    source.add_argument("--result-root", type=Path)
    source.set_defaults(func=_source)

    plan = sub.add_parser(
        "native-plan",
        help=(
            "Audit selected split interfaces and the selection-derived "
            "Native plan without hardware execution."
        ),
    )
    plan.add_argument(
        "--benchmark-set",
        action="append",
        required=True,
        metavar="MODEL=PATH",
    )
    plan.add_argument(
        "--backend",
        action="append",
        choices=["hailo8", "hailo10h", "deepx"],
        help="Repeat to restrict producers; default is all three.",
    )
    plan.add_argument("--result-root", type=Path)
    plan.set_defaults(func=_native_plan)

    split = sub.add_parser(
        "hailo10-split",
        help="Replay one bound Hailo10H->TensorRT case into an isolated root.",
    )
    split.add_argument("--benchmark-set", required=True, type=Path)
    split.add_argument("--case", required=True)
    split.add_argument("--image", required=True, type=Path)
    split.add_argument("--quality-binding", required=True, type=Path)
    split.add_argument("--eval-run-id", required=True)
    split.add_argument("--model-id", required=True)
    split.add_argument(
        "--task", required=True, choices=["classification", "detection"],
    )
    split.add_argument("--setup-id", required=True)
    split.add_argument("--source-run-id", default="hailo10h_to_trt")
    split.add_argument("--frames", type=int, default=5)
    split.add_argument(
        "--timeout-s", type=float, default=300.0,
        help="Kill the isolated process group after this many seconds.",
    )
    split.add_argument("--result-root", type=Path)
    split.set_defaults(func=_hailo10_split)

    full = sub.add_parser(
        "hailo10-full",
        help=(
            "Replay one five-frame Hailo10H Full raw-head + frozen "
            "Decode/NMS contract into an isolated root."
        ),
    )
    full.add_argument("--benchmark-set", required=True, type=Path)
    full.add_argument("--image", required=True, type=Path)
    full.add_argument(
        "--model-id", required=True, choices=["yolo26s"],
        help="2.70d's six-head Full smoke is intentionally YOLO26s-specific.",
    )
    full.add_argument("--setup-id", required=True)
    full.add_argument(
        "--frames", type=int, default=5,
        help="Fixed diagnostic work count; 2.70d requires exactly five.",
    )
    full.add_argument(
        "--timeout-s", type=float, default=300.0,
        help="Kill the isolated process group after this many seconds.",
    )
    full.add_argument("--result-root", type=Path)
    full.set_defaults(func=_hailo10_full)

    args = parser.parse_args()
    if getattr(args, "frames", 1) < 1:
        parser.error("--frames must be >= 1")
    if args.command == "hailo10-full" and int(args.frames) != 5:
        parser.error("hailo10-full is a fixed five-frame diagnostic")
    if (
        hasattr(args, "timeout_s")
        and (
            not np.isfinite(float(args.timeout_s))
            or float(args.timeout_s) <= 0.0
        )
    ):
        parser.error("--timeout-s must be a positive finite number")
    try:
        return int(args.func(args))
    except Exception as exc:
        print(
            json.dumps({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "diagnostic_only": True,
                "claim_eligible": False,
            }, indent=2),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
