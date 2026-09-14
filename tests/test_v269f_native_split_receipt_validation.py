from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
    seal_native_command_contract,
    verify_native_command_contract,
)
from onnx_splitpoint_tool.native_split_quality import (
    bind_quality_to_native_split,
    known_native_split_policy,
    materialize_native_split_preselection,
    native_split_quality_selection_duplicates,
    seal_native_split_quality_binding,
    select_central_native_split_quality_binding,
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy


def _write_bytes(path: Path, value: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    if executable:
        path.chmod(0o755)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8",
    )


def _artifact(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _receipt_with_inner_sha(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = canonical_json_sha256(value)
    return value


def _fixture_payload(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "part1_runtime": tmp_path / "cache" / "part1.hef",
        "boundary_metadata": tmp_path / "cache" / "part1_boundary_metadata.json",
        "source_part2_onnx": tmp_path / "cache" / "source_part2.onnx",
        "build_part2_onnx": tmp_path / "cache" / "part2_uint8_dequant_bridge.onnx",
        "engine": tmp_path / "cache" / "part2_uint8_dequant_fp16.engine",
        "native_trt_meta": tmp_path / "cache" / "native_trt_meta.json",
        "engine_build_receipt": tmp_path / "cache" / "engine_build_receipt.json",
        "trtexec": tmp_path / "bin" / "trtexec",
    }
    _write_bytes(paths["part1_runtime"], b"HEF-v1-boundary-cut")
    _write_bytes(paths["source_part2_onnx"], b"ONNX-source-part2-v1")
    _write_bytes(paths["build_part2_onnx"], b"ONNX-dequant-bridge-v1")
    _write_bytes(paths["engine"], b"TensorRT-engine-bytes-v1")
    _write_bytes(
        paths["trtexec"], b"#!/bin/sh\nexit 0\n", executable=True,
    )

    policy = known_native_split_policy(
        model_id="yolo26s", case_id="b038", setup_id="hailo8_setup",
        backend="hailo8_to_trt",
    )
    assert policy is not None
    part1 = _artifact(paths["part1_runtime"])
    boundary_metadata = {
        "schema": "onnx-splitpoint/native-part1-boundary-metadata",
        "schema_version": 1,
        "model_id": "yolo26s",
        "case_id": "b038",
        "setup_id": "hailo8_setup",
        "backend": "hailo8_to_trt",
        "source_run_id": "hailo8_to_trt",
        "part1_artifact_sha256": part1["sha256"],
        "part1_artifact_size_bytes": part1["size_bytes"],
        "boundary_tensor_count": 1,
        "boundary_tensor": {
            "name": "cut",
            "runtime_name": "cut/hailort",
            "shape": [2, 2, 2],
            "canonical_part2_shape": [1, 2, 2, 2],
            "dtype": "uint8",
            "quantization": {
                "source": "hailort_hef_output_vstream_info",
                "scale": 0.03125,
                "zero_point": 17.0,
            },
        },
        "boundary_layout": "memory_nhwc_to_nchw",
    }
    boundary_metadata["metadata_sha256"] = canonical_json_sha256(boundary_metadata)
    _write_json(paths["boundary_metadata"], boundary_metadata)
    preselection = materialize_native_split_preselection(
        policy=policy,
        part1_artifact=part1,
        boundary_metadata=boundary_metadata,
        boundary_metadata_artifact=_artifact(paths["boundary_metadata"]),
    )

    command = [
        str(paths["trtexec"].resolve()),
        f"--onnx={paths['build_part2_onnx'].resolve()}",
        f"--saveEngine={paths['engine'].resolve()}",
        "--fp16",
    ]
    receipt = _receipt_with_inner_sha({
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": command,
        "source_onnx": str(paths["build_part2_onnx"].resolve()),
        "source_onnx_sha256": _artifact(paths["build_part2_onnx"])["sha256"],
        "engine": str(paths["engine"].resolve()),
        "engine_sha256": _artifact(paths["engine"])["sha256"],
        "trtexec": str(paths["trtexec"].resolve()),
        "trtexec_sha256": _artifact(paths["trtexec"])["sha256"],
    })
    _write_json(paths["engine_build_receipt"], receipt)

    native_meta = {
        "schema": "onnx-splitpoint/native-trt-meta",
        "schema_version": 1,
        "case": "b038",
        "variant": "part2",
        "onnx": str(paths["build_part2_onnx"].resolve()),
        "source_onnx": str(paths["build_part2_onnx"].resolve()),
        "engine": str(paths["engine"].resolve()),
        "inputs": [{
            "name": "cut", "shape": [1, 2, 2, 2],
            "elem_type": "UINT8", "has_dynamic": False,
        }],
        "outputs": [{
            "name": "output0", "shape": [1, 300, 6],
            "elem_type": "FLOAT", "has_dynamic": False,
        }],
        "inputs_static": True,
        "precision": "uint8_dequant_fp16",
        "requested_precision": "uint8_dequant_fp16",
        "uint8_cast_bridge": {
            "schema": "onnx-splitpoint/uint8-dequant-bridge",
            "schema_version": 2,
            "source": str(paths["source_part2_onnx"].resolve()),
            "source_sha256": _artifact(paths["source_part2_onnx"])["sha256"],
            "bridge": str(paths["build_part2_onnx"].resolve()),
            "bridge_sha256": _artifact(paths["build_part2_onnx"])["sha256"],
            "input_name": "cut",
            "input_dtype": "UINT8",
            "input_shape": [1, 2, 2, 2],
            "boundary_layout": {
                "requested": "memory_nhwc_to_nchw",
                "effective": "memory_nhwc_to_nchw",
                "applied": True,
            },
            "scale": 0.03125,
            "zero_point": 17.0,
        },
        "build_ok": True,
        "build": {"returncode": 0, "cmd": command},
        "engine_build_receipt_status": "engine_build_receipt_verified",
        "engine_build_receipt_path": str(paths["engine_build_receipt"].resolve()),
        "engine_build_receipt": receipt,
    }
    _write_json(paths["native_trt_meta"], native_meta)

    artifacts = {name: _artifact(path) for name, path in paths.items()}
    boundary_contract = {
        "precision": preselection["precision"],
        "boundary_layout": preselection["boundary_layout"],
        "boundary_transform": preselection["boundary_transform"],
        "boundary_tensor_name": preselection["boundary_tensor_name"],
        "boundary_tensor_shape": list(preselection["boundary_tensor_shape"]),
        "boundary_tensor_dtype": preselection["boundary_tensor_dtype"],
        "boundary_metadata_sha256": preselection["boundary_metadata_sha256"],
        "boundary_metadata_file_sha256": preselection[
            "boundary_metadata_file_sha256"
        ],
        "dequant_scale": preselection["dequant_scale"],
        "dequant_zero_point": preselection["dequant_zero_point"],
    }
    payload = {
        "eval_run_id": "eval-native-split-001",
        "source_run_id": "hailo8_to_trt",
        "quality_completed": True,
        "performance_claims_emitted": False,
        "preselection": preselection,
        "preselection_sha256": preselection["selection_sha256"],
        "artifacts": artifacts,
        "boundary_contract": boundary_contract,
        "boundary_contract_sha256": canonical_json_sha256(boundary_contract),
        "engine_build_receipt": receipt,
        "engine_build_receipt_sha256": receipt["receipt_sha256"],
        # This is the build-summary row.  The seal separately embeds the exact
        # bytes parsed from native_trt_meta.json.
        "native_trt_meta": native_meta,
        "native_trt_meta_sha256": canonical_json_sha256(native_meta),
        "producer_command": ["python", "native_trt_from_benchmarkset.py"],
    }
    return payload, paths


def _expected_identity() -> dict[str, str]:
    return {
        "model": "yolo26s", "case": "b038", "setup_id": "hailo8_setup",
        "backend": "hailo8_to_trt", "task": "detection",
        "precision": "uint8_dequant_fp16",
    }


def _native_row_for_binding(binding: dict[str, Any]) -> dict[str, Any]:
    artifacts = binding["artifacts"]
    selection_duplicates = (
        native_split_quality_selection_duplicates(binding)
        if "central_quality_selection" in binding else {}
    )
    dummy_sha = "a" * 64
    command_artifacts = {
        "python_executable": {
            "path": "/usr/bin/python3", "sha256": dummy_sha, "size_bytes": 1,
        },
        "hef": copy.deepcopy(artifacts["part1_runtime"]),
        "engine": copy.deepcopy(artifacts["engine"]),
        "native_executable": {
            "path": "/opt/native_fifo", "sha256": "b" * 64, "size_bytes": 1,
        },
        "generated_cpp": {
            "path": "/opt/native_fifo.cpp", "sha256": "c" * 64,
            "size_bytes": 1,
        },
        "cmake": {
            "path": "/usr/bin/cmake", "sha256": "d" * 64, "size_bytes": 1,
        },
        "boundary_metadata": copy.deepcopy(artifacts["boundary_metadata"]),
        "source_part2_onnx": copy.deepcopy(artifacts["source_part2_onnx"]),
        "build_part2_onnx": copy.deepcopy(artifacts["build_part2_onnx"]),
        "native_trt_meta": copy.deepcopy(artifacts["native_trt_meta"]),
        "engine_build_receipt": copy.deepcopy(artifacts["engine_build_receipt"]),
        "trtexec": copy.deepcopy(artifacts["trtexec"]),
        "semantic_output_manifest": {
            "path": "/opt/semantic_output_manifest.json",
            "sha256": "1" * 64, "size_bytes": 1,
        },
        "semantic_boundary_manifest": {
            "path": "/opt/semantic_boundary_manifest.json",
            "sha256": "2" * 64, "size_bytes": 1,
        },
    }
    command = seal_native_command_contract({
        "complete": True,
        "backend": "hailo8_to_trt",
        "model": "yolo26s", "case": "b038",
        "precision": "uint8_dequant_fp16", "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8",
        "runner_sha256": "e" * 64,
        "input_image_sha256": "f" * 64,
        "runner": "/opt/runner.py",
        "python_executable": "/usr/bin/python3",
        "benchmark_set": "/opt/benchmark_set",
        "input_image": "/opt/input.jpg",
        "interpreter_identity": {
            "executable": "/usr/bin/python3", "executable_sha256": dummy_sha,
        },
        "runtime_options": {"warmup": 10, "queue_depth": 4},
        "boundary_contract": {
            "boundary_layout_effective": "memory_nhwc_to_nchw",
        },
        "quality_boundary_contract": copy.deepcopy(binding["boundary_contract"]),
        "quality_boundary_contract_sha256": binding[
            "boundary_contract_sha256"
        ],
        "quality_preselection": copy.deepcopy(binding["preselection"]),
        "quality_preselection_sha256": binding["preselection_sha256"],
        "eval_run_id": binding["eval_run_id"],
        "source_run_id": binding["source_run_id"],
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_eval_run_id": binding["eval_run_id"],
        "native_split_quality_source_run_id": binding["source_run_id"],
        **selection_duplicates,
        "native_split_quality_local_verification": copy.deepcopy(
            binding["local_artifact_verification"]
        ),
        "artifacts": command_artifacts,
    })
    row = {
        "backend": "hailo8_to_trt",
        "model": "yolo26s", "case": "b038",
        "precision": "uint8_dequant_fp16", "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8", "task": "detection",
        "eval_run_id": binding["eval_run_id"],
        "source_run_id": binding["source_run_id"],
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_eval_run_id": binding["eval_run_id"],
        "native_split_quality_source_run_id": binding["source_run_id"],
        **selection_duplicates,
        "native_command_contract": command,
        "native_command_contract_sha256": command["contract_sha256"],
    }
    attestation = {
        "schema": "onnx-splitpoint/native-split-quality-consumer-attestation",
        "schema_version": 1,
        "status": "local_files_rehashed_and_exact_command_join_verified",
        "binding_sha256": binding["binding_sha256"],
        "command_contract_sha256": command["contract_sha256"],
        "eval_run_id": binding["eval_run_id"],
        "source_run_id": binding["source_run_id"],
        "backend": "hailo8_to_trt",
        "model_id": "yolo26s", "case_id": "b038",
        "setup_id": "hailo8_setup", "task": "detection",
        "precision": "uint8_dequant_fp16",
        "local_artifact_verification_sha256": canonical_json_sha256(
            binding["local_artifact_verification"]
        ),
        "semantic_output_manifest_sha256": "1" * 64,
        "semantic_boundary_manifest_sha256": "2" * 64,
        **selection_duplicates,
    }
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    row["native_split_quality_consumer_attestation"] = attestation
    return row


def _reseal_command(command: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(command)
    payload.pop("contract_sha256", None)
    return seal_native_command_contract(payload)


def test_quality_first_command_verifier_is_portable_and_complete(
    tmp_path: Path,
) -> None:
    payload, paths = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    command = _native_row_for_binding(binding)["native_command_contract"]

    # These paths belong to the producer host.  A central/final verifier must
    # validate only the copied evidence; the later Energy preflight performs
    # the second local file read on the actual consumer host.
    for path in paths.values():
        path.unlink()

    verified, status = verify_native_command_contract(
        command, expected_identity=_expected_identity(),
    )

    assert verified == command
    assert status == (
        "hash_schema_identity_artifacts_and_quality_first_crosslinks_verified"
    )


@pytest.mark.parametrize(
    ("binding_name", "command_name"),
    [
        ("part1_runtime", "hef"),
        ("boundary_metadata", "boundary_metadata"),
        ("source_part2_onnx", "source_part2_onnx"),
        ("build_part2_onnx", "build_part2_onnx"),
        ("engine", "engine"),
        ("native_trt_meta", "native_trt_meta"),
        ("engine_build_receipt", "engine_build_receipt"),
        ("trtexec", "trtexec"),
    ],
)
@pytest.mark.parametrize("field", ["path", "sha256", "size_bytes"])
def test_quality_first_command_verifier_rejects_every_artifact_crosslink_drift(
    tmp_path: Path, binding_name: str, command_name: str, field: str,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    command = _native_row_for_binding(binding)["native_command_contract"]
    raw = copy.deepcopy(command)
    raw.pop("contract_sha256")
    if field == "path":
        raw["artifacts"][command_name][field] += ".other"
    elif field == "sha256":
        raw["artifacts"][command_name][field] = "0" * 64
    else:
        raw["artifacts"][command_name][field] += 1

    verified, status = verify_native_command_contract(
        seal_native_command_contract(raw), expected_identity=_expected_identity(),
    )

    assert verified is None
    assert status == (
        f"native_command_contract_quality_artifact_{binding_name}_crosslink_mismatch"
    )


@pytest.mark.parametrize(
    ("artifact_name", "mutation", "status_suffix"),
    [
        ("semantic_output_manifest", "missing", "invalid"),
        ("semantic_boundary_manifest", "missing", "invalid"),
        ("semantic_output_manifest", "sha256", "legacy_sha256_missing"),
        ("semantic_boundary_manifest", "sha256", "legacy_sha256_missing"),
        ("semantic_output_manifest", "size_bytes", "size_missing"),
        ("semantic_boundary_manifest", "size_bytes", "size_missing"),
    ],
)
def test_quality_first_command_verifier_requires_both_semantic_manifests(
    tmp_path: Path, artifact_name: str, mutation: str, status_suffix: str,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    command = _native_row_for_binding(binding)["native_command_contract"]
    raw = copy.deepcopy(command)
    raw.pop("contract_sha256")
    if mutation == "missing":
        raw["artifacts"].pop(artifact_name)
    elif mutation == "sha256":
        raw["artifacts"][artifact_name]["sha256"] = "not-a-sha"
    elif mutation == "size_bytes":
        raw["artifacts"][artifact_name]["size_bytes"] = 0
    else:  # pragma: no cover
        raise AssertionError(mutation)

    verified, status = verify_native_command_contract(
        seal_native_command_contract(raw), expected_identity=_expected_identity(),
    )

    assert verified is None
    if status_suffix == "legacy_sha256_missing":
        assert status == (
            f"native_command_contract_artifact_{artifact_name}_sha256_missing"
        )
    else:
        assert status == (
            f"native_command_contract_quality_artifact_{artifact_name}_{status_suffix}"
        )


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        (
            "missing_marker",
            "native_command_contract_quality_first_fields_incomplete",
        ),
        ("eval_run_id", "command_eval_run_id_mismatch"),
        ("source_run_id", "command_source_run_id_mismatch"),
        (
            "binding_payload",
            "native_command_contract_quality_binding_payload_sha256_mismatch",
        ),
        (
            "local_proof_duplicate",
            "native_command_contract_quality_local_verification_payload_mismatch",
        ),
        (
            "preselection_duplicate",
            "native_command_contract_quality_preselection_payload_mismatch",
        ),
        (
            "boundary_duplicate",
            "native_command_contract_quality_boundary_contract_payload_mismatch",
        ),
    ],
)
def test_quality_first_command_verifier_rejects_resealed_closure_drift(
    tmp_path: Path, mutation: str, expected_status: str,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    command = _native_row_for_binding(binding)["native_command_contract"]
    raw = copy.deepcopy(command)
    raw.pop("contract_sha256")
    if mutation == "missing_marker":
        raw.pop("quality_boundary_contract_sha256")
    elif mutation == "eval_run_id":
        raw["eval_run_id"] = "other-eval"
    elif mutation == "source_run_id":
        raw["source_run_id"] = "deepx_to_trt"
    elif mutation == "binding_payload":
        raw["native_split_quality_binding"]["eval_run_id"] = "other-eval"
    elif mutation == "local_proof_duplicate":
        raw["native_split_quality_local_verification"] = {}
    elif mutation == "preselection_duplicate":
        raw["quality_preselection"] = {}
    elif mutation == "boundary_duplicate":
        raw["quality_boundary_contract"] = {}
    else:  # pragma: no cover
        raise AssertionError(mutation)

    verified, status = verify_native_command_contract(
        seal_native_command_contract(raw), expected_identity=_expected_identity(),
    )

    assert verified is None
    assert status == expected_status


def test_quality_first_command_verifier_rejects_resealed_local_proof_crosslink(
    tmp_path: Path,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    command = _native_row_for_binding(binding)["native_command_contract"]
    raw = copy.deepcopy(command)
    raw.pop("contract_sha256")
    nested_binding = raw["native_split_quality_binding"]
    nested_binding.pop("binding_sha256")
    proof = nested_binding["local_artifact_verification"]
    proof.pop("proof_sha256")
    proof["artifact_set_sha256"] = "0" * 64
    proof["proof_sha256"] = canonical_json_sha256(proof)
    nested_binding["binding_sha256"] = canonical_json_sha256(nested_binding)
    raw["native_split_quality_binding_sha256"] = nested_binding["binding_sha256"]
    raw["native_split_quality_local_verification"] = copy.deepcopy(proof)

    verified, status = verify_native_command_contract(
        seal_native_command_contract(raw), expected_identity=_expected_identity(),
    )

    assert verified is None
    assert status == (
        "native_command_contract_quality_local_verification_artifact_set_mismatch"
    )


def test_quality_first_command_verifier_rejects_fully_resealed_source_backend_drift(
    tmp_path: Path,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    command = _native_row_for_binding(binding)["native_command_contract"]
    raw = copy.deepcopy(command)
    raw.pop("contract_sha256")
    nested_binding = raw["native_split_quality_binding"]
    nested_binding.pop("binding_sha256")
    nested_binding["source_run_id"] = "deepx_to_trt"
    nested_binding["binding_sha256"] = canonical_json_sha256(nested_binding)
    raw["native_split_quality_binding_sha256"] = nested_binding["binding_sha256"]
    raw["source_run_id"] = "deepx_to_trt"
    raw["native_split_quality_source_run_id"] = "deepx_to_trt"

    verified, status = verify_native_command_contract(
        seal_native_command_contract(raw), expected_identity=_expected_identity(),
    )

    assert verified is None
    assert status == "native_command_contract_quality_source_backend_mismatch"


def test_portable_consumer_attestation_binds_both_semantic_manifest_hashes(
    tmp_path: Path,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    row = _native_row_for_binding(binding)
    raw = copy.deepcopy(row["native_command_contract"])
    raw.pop("contract_sha256")
    raw["artifacts"]["semantic_boundary_manifest"]["sha256"] = "8" * 64
    row["native_command_contract"] = seal_native_command_contract(raw)
    row["native_command_contract_sha256"] = row["native_command_contract"][
        "contract_sha256"
    ]
    attestation = copy.deepcopy(row["native_split_quality_consumer_attestation"])
    attestation.pop("attestation_sha256")
    attestation["command_contract_sha256"] = row[
        "native_command_contract_sha256"
    ]
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    row["native_split_quality_consumer_attestation"] = attestation

    verified, status = bind_quality_to_native_split(
        native_row=row, quality_binding=binding, verification_mode="portable",
    )

    assert verified is None
    assert status == (
        "native_split_quality_consumer_attestation_"
        "semantic_boundary_manifest_sha256_mismatch"
    )


def _refresh_json_artifacts(
    payload: dict[str, Any], paths: dict[str, Path],
    *, receipt: dict[str, Any] | None = None,
    native_meta: dict[str, Any] | None = None,
) -> None:
    if receipt is not None:
        _write_json(paths["engine_build_receipt"], receipt)
        payload["engine_build_receipt"] = receipt
        payload["engine_build_receipt_sha256"] = receipt["receipt_sha256"]
        payload["artifacts"]["engine_build_receipt"] = _artifact(
            paths["engine_build_receipt"]
        )
    if native_meta is not None:
        _write_json(paths["native_trt_meta"], native_meta)
        payload["native_trt_meta"] = native_meta
        payload["native_trt_meta_sha256"] = canonical_json_sha256(native_meta)
        payload["artifacts"]["native_trt_meta"] = _artifact(
            paths["native_trt_meta"]
        )


def test_local_seal_and_explicit_portable_and_local_validation(tmp_path: Path) -> None:
    payload, paths = _fixture_payload(tmp_path)

    binding = seal_native_split_quality_binding(payload)
    portable, portable_status = validate_native_split_quality_binding(
        binding, expected_identity=_expected_identity(), verification_mode="portable",
    )
    local, local_status = validate_native_split_quality_binding(
        binding, expected_identity=_expected_identity(), verification_mode="local",
    )

    assert portable == binding
    assert local == binding
    assert portable_status == (
        "portable_embedded_evidence_and_cross_links_verified_without_local_rehash"
    )
    assert local_status == "local_files_rehashed_and_exact_cross_links_verified"
    proof = binding["local_artifact_verification"]
    assert proof["verification_kind"] == "producer_local_file_rehash"
    assert set(proof["embedded_json_files"]) == {
        "boundary_metadata", "native_trt_meta", "engine_build_receipt",
    }
    assert binding["native_trt_meta_payload"] == json.loads(
        paths["native_trt_meta"].read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "part1_runtime", "boundary_metadata", "source_part2_onnx",
        "build_part2_onnx", "engine", "native_trt_meta",
        "engine_build_receipt", "trtexec",
    ],
)
def test_later_native_local_rehash_rejects_every_tampered_file(
    tmp_path: Path, artifact_name: str,
) -> None:
    payload, paths = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    paths[artifact_name].write_bytes(paths[artifact_name].read_bytes() + b"tamper")
    if artifact_name == "trtexec":
        paths[artifact_name].chmod(0o755)

    portable, portable_status = validate_native_split_quality_binding(
        binding, verification_mode="portable",
    )
    local, local_status = validate_native_split_quality_binding(
        binding, verification_mode="local",
    )

    # Central can validate the copied evidence but must not pretend to have
    # inspected the now-remote path.  Native performs the decisive second read.
    assert portable is not None
    assert "without_local_rehash" in portable_status
    assert local is None
    assert local_status == f"native_split_quality_local_{artifact_name}_sha256_mismatch"


def test_local_validation_rejects_symlink_substitution(tmp_path: Path) -> None:
    payload, paths = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    engine = paths["engine"]
    replacement = engine.with_name("replacement.engine")
    os.replace(engine, replacement)
    engine.symlink_to(replacement)

    verified, status = validate_native_split_quality_binding(
        binding, verification_mode="local",
    )

    assert verified is None
    assert status == "native_split_quality_local_engine_path_or_symlink_invalid"


def test_local_validation_fails_closed_when_artifact_is_inaccessible(tmp_path: Path) -> None:
    payload, paths = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    paths["source_part2_onnx"].unlink()

    verified, status = validate_native_split_quality_binding(
        binding, verification_mode="local",
    )

    assert verified is None
    assert status == "native_split_quality_local_source_part2_onnx_inaccessible"


def test_native_bind_requires_exact_command_binding_identity(tmp_path: Path) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    row = _native_row_for_binding(binding)

    verified, status = bind_quality_to_native_split(
        native_row=row, quality_binding=binding,
    )

    assert verified == binding
    assert status == "exact_quality_native_engine_command_and_boundary_match"


def test_central_portable_bind_requires_exact_consumer_attestation(
    tmp_path: Path,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    row = _native_row_for_binding(binding)

    verified, status = bind_quality_to_native_split(
        native_row=row, quality_binding=binding, verification_mode="portable",
    )

    assert verified == binding
    assert status == "portable_binding_command_and_consumer_attestation_exact_match"


def test_management_quality_join_uses_portable_consumer_attestation(
    tmp_path: Path,
) -> None:
    """Management must not try to open setup-local artifact paths."""

    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    endpoint_sha = "9" * 64
    policy = AccuracyGatePolicy()
    provenance = {
        "source_request_sha256": "3" * 64,
        "model_sha256": "4" * 64,
        "validation_dataset_sha256": "5" * 64,
        "validation_dataset_image_ids_sha256": "6" * 64,
        "validation_dataset_ground_truth_sha256": "7" * 64,
        "policy_sha256": policy.sha256(),
        "quality_contract_sha256": "a" * 64,
        "preprocessing_contract_sha256": "b" * 64,
        "decoder_contract_sha256": "c" * 64,
        "nms_contract_sha256": "d" * 64,
    }
    identity = {
        "identity_valid": True,
        "eval_run_id": binding["eval_run_id"],
        "model_id": "yolo26s",
        "task": "detection",
        "case_id": "b038",
        "source_run_id": "hailo8_to_trt",
        "setup_id": "hailo8_setup",
        "variant": "composed",
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": "uint8_dequant_fp16",
        "native_split_quality_binding_required": True,
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        **provenance,
    }
    result = {
        "status": "completed",
        "technical_status": "completed",
        "decision": "pass",
        "eval_run_id": binding["eval_run_id"],
        "model_id": "yolo26s",
        "task": "detection",
        "case_id": "b038",
        "source_run_id": "hailo8_to_trt",
        "source_setup_id": "hailo8_setup",
        "variant": "composed",
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": "uint8_dequant_fp16",
        "native_split_quality_binding_required": True,
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "request_identity": identity,
        **provenance,
    }
    selected_binding = select_central_native_split_quality_binding(
        binding,
        source_request_sha256=provenance["source_request_sha256"],
        central_result_sha256=canonical_json_sha256(result),
        central_identity={
            "eval_run_id": binding["eval_run_id"],
            "model_id": "yolo26s", "case_id": "b038",
            "source_run_id": "hailo8_to_trt", "setup_id": "hailo8_setup",
            "task": "detection", "variant": "composed",
            "runtime_precision_identity": "uint8_dequant_fp16",
        },
    )
    row = _native_row_for_binding(selected_binding)
    row.update({
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": "uint8_dequant_fp16",
    })

    validator_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "native_producer_validate_visualize.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v269f_native_split_management_portable", validator_path,
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    validator._bind_central_quality_evidence(row, [result], policy)
    assert row["central_quality_evidence_verified"] is True
    assert row["quality_first_binding_status"] == (
        "central_native_exact_engine_command_boundary_match"
    )

    rejected = _native_row_for_binding(selected_binding)
    rejected.update({
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": "uint8_dequant_fp16",
    })
    attestation = copy.deepcopy(
        rejected["native_split_quality_consumer_attestation"]
    )
    attestation.pop("attestation_sha256")
    attestation["binding_sha256"] = "e" * 64
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    rejected["native_split_quality_consumer_attestation"] = attestation

    validator._bind_central_quality_evidence(rejected, [result], policy)
    assert rejected["central_quality_evidence_verified"] is False
    assert (
        "native_split_quality_consumer_attestation_binding_sha256_mismatch"
        in rejected["quality_first_binding_errors"]
    )


@pytest.mark.parametrize(
    ("field", "replacement", "expected_status"),
    [
        (
            "binding_sha256", "3" * 64,
            "native_split_quality_consumer_attestation_binding_sha256_mismatch",
        ),
        (
            "command_contract_sha256", "4" * 64,
            "native_split_quality_consumer_attestation_command_contract_sha256_mismatch",
        ),
        (
            "eval_run_id", "other-eval",
            "native_split_quality_consumer_attestation_eval_run_id_mismatch",
        ),
        (
            "backend", "deepx_to_trt",
            "native_split_quality_consumer_attestation_backend_mismatch",
        ),
        (
            "semantic_output_manifest_sha256", "5" * 64,
            "native_split_quality_consumer_attestation_semantic_output_manifest_sha256_mismatch",
        ),
    ],
)
def test_central_portable_bind_rejects_resealed_consumer_attestation_drift(
    tmp_path: Path, field: str, replacement: str, expected_status: str,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    row = _native_row_for_binding(binding)
    attestation = copy.deepcopy(
        row["native_split_quality_consumer_attestation"]
    )
    attestation.pop("attestation_sha256")
    attestation[field] = replacement
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    row["native_split_quality_consumer_attestation"] = attestation

    verified, status = bind_quality_to_native_split(
        native_row=row, quality_binding=binding, verification_mode="portable",
    )

    assert verified is None
    assert status == expected_status


def test_central_portable_bind_rejects_missing_consumer_attestation(
    tmp_path: Path,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    row = _native_row_for_binding(binding)
    row.pop("native_split_quality_consumer_attestation")

    verified, status = bind_quality_to_native_split(
        native_row=row, quality_binding=binding, verification_mode="portable",
    )

    assert verified is None
    assert status == "native_split_quality_consumer_attestation_missing"


@pytest.mark.parametrize(
    ("field", "replacement", "expected_status"),
    [
        (
            "native_split_quality_binding_sha256", "1" * 64,
            "native_split_quality_command_binding_sha256_mismatch",
        ),
        (
            "native_split_quality_eval_run_id", "other-eval",
            "native_split_quality_command_eval_run_id_mismatch",
        ),
        (
            "native_split_quality_source_run_id", "deepx_to_trt",
            "native_split_quality_command_source_run_id_mismatch",
        ),
        (
            "backend", "deepx_to_trt",
            "native_split_quality_native_command_contract_backend_mismatch",
        ),
    ],
)
def test_native_bind_rejects_resealed_command_identity_drift(
    tmp_path: Path, field: str, replacement: str, expected_status: str,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    row = _native_row_for_binding(binding)
    raw = copy.deepcopy(row["native_command_contract"])
    raw.pop("contract_sha256")
    raw[field] = replacement
    row["native_command_contract"] = seal_native_command_contract(raw)
    row["native_command_contract_sha256"] = row["native_command_contract"][
        "contract_sha256"
    ]

    verified, status = bind_quality_to_native_split(
        native_row=row, quality_binding=binding,
    )

    assert verified is None
    assert status == expected_status


def test_vendored_module_prefers_vendored_command_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    vendored = types.ModuleType("splitpoint_runners.native_command_contract")
    vendored.canonical_json_sha256 = lambda value: "vendored"  # type: ignore[attr-defined]
    vendored.verify_native_command_contract = lambda *args, **kwargs: (  # type: ignore[attr-defined]
        None, "vendored"
    )
    installed = types.ModuleType("onnx_splitpoint_tool.native_command_contract")
    installed.canonical_json_sha256 = lambda value: "installed"  # type: ignore[attr-defined]
    installed.verify_native_command_contract = lambda *args, **kwargs: (  # type: ignore[attr-defined]
        None, "installed"
    )
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules, "splitpoint_runners.native_command_contract", vendored,
    )
    monkeypatch.setitem(
        sys.modules, "onnx_splitpoint_tool.native_command_contract", installed,
    )
    module_path = Path(__file__).parents[1] / "onnx_splitpoint_tool" / (
        "native_split_quality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "splitpoint_runners.native_split_quality", module_path,
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)

    assert loaded.canonical_json_sha256({}) == "vendored"


def test_generated_case_runner_requires_canonical_local_validation() -> None:
    template = (
        Path(__file__).parents[1] / "onnx_splitpoint_tool" / "resources"
        / "templates" / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    call_start = template.index(
        "validate_native_split_quality_binding(\n", template.index(
            "--native-split-quality-binding"
        ),
    )
    call_end = template.index("\n                )", call_start)
    validator_call = template[call_start:call_end]

    assert 'verification_mode="local"' in validator_call
    assert "expected_identity={" in validator_call
    assert "object_pairs_hook=_native_split_no_duplicate_json_keys" in template
    assert "if verified_split_binding is None:" in template
    assert 'args.native_trt_no_fallback = True' in template
    assert 'args.native_trt_build = False' in template


def test_benchmark_suite_prepares_and_forwards_sealed_split_binding() -> None:
    template = (
        Path(__file__).parents[1] / "onnx_splitpoint_tool" / "resources"
        / "templates" / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")

    assert "prepare_native_split_quality_binding(" in template
    assert template.count("split_quality = _prepare_native_split_quality_for_case(") == 3
    assert template.count('native_split_quality_binding=str(') == 3
    assert template.count('(split_quality or {}).get("binding_path") or ""') == 3
    assert template.count("False if split_quality") == 3
    assert "native_split_quality_binding_must_be_replayed_from_native_command_for_energy" in template


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        (
            "receipt_source_hash",
            "native_split_quality_engine_build_receipt_source_onnx_sha256_mismatch",
        ),
        (
            "receipt_engine_hash",
            "native_split_quality_engine_build_receipt_engine_sha256_mismatch",
        ),
        (
            "receipt_trtexec_hash",
            "native_split_quality_engine_build_receipt_trtexec_sha256_mismatch",
        ),
        (
            "receipt_build_command",
            "native_split_quality_engine_build_receipt_command_source_mismatch",
        ),
        (
            "meta_precision",
            "native_split_quality_native_trt_meta_precision_mismatch",
        ),
        (
            "meta_missing_returncode",
            "native_split_quality_native_trt_meta_build_status_invalid",
        ),
        (
            "bridge_source_hash",
            "native_split_quality_native_trt_meta_bridge_schema_or_artifact_mismatch",
        ),
        (
            "bridge_build_hash",
            "native_split_quality_native_trt_meta_bridge_schema_or_artifact_mismatch",
        ),
    ],
)
def test_local_seal_rejects_rehashed_semantic_tamper(
    tmp_path: Path, mutation: str, expected_status: str,
) -> None:
    payload, paths = _fixture_payload(tmp_path)
    receipt = copy.deepcopy(payload["engine_build_receipt"])
    meta = copy.deepcopy(payload["native_trt_meta"])
    if mutation == "receipt_source_hash":
        receipt["source_onnx_sha256"] = "1" * 64
    elif mutation == "receipt_engine_hash":
        receipt["engine_sha256"] = "2" * 64
    elif mutation == "receipt_trtexec_hash":
        receipt["trtexec_sha256"] = "3" * 64
    elif mutation == "receipt_build_command":
        receipt["command"][1] = f"--onnx={paths['source_part2_onnx'].resolve()}"
    elif mutation == "meta_precision":
        meta["precision"] = "fp32"
    elif mutation == "meta_missing_returncode":
        meta["build"].pop("returncode")
    elif mutation == "bridge_source_hash":
        meta["uint8_cast_bridge"]["source_sha256"] = "4" * 64
    elif mutation == "bridge_build_hash":
        meta["uint8_cast_bridge"]["bridge_sha256"] = "5" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)
    if mutation.startswith("receipt_"):
        receipt = _receipt_with_inner_sha(receipt)
        meta["engine_build_receipt"] = receipt
        if mutation == "receipt_build_command":
            meta["build"]["cmd"] = list(receipt["command"])
        _refresh_json_artifacts(payload, paths, receipt=receipt, native_meta=meta)
    else:
        _refresh_json_artifacts(payload, paths, native_meta=meta)

    with pytest.raises(ValueError, match=expected_status):
        seal_native_split_quality_binding(payload)


@pytest.mark.parametrize(
    ("artifact_name", "expected_status"),
    [
        (
            "source_part2_onnx",
            "native_split_quality_native_trt_meta_bridge_schema_or_artifact_mismatch",
        ),
        (
            "build_part2_onnx",
            "native_split_quality_engine_build_receipt_source_onnx_sha256_mismatch",
        ),
        (
            "engine",
            "native_split_quality_engine_build_receipt_engine_sha256_mismatch",
        ),
        (
            "trtexec",
            "native_split_quality_engine_build_receipt_trtexec_sha256_mismatch",
        ),
    ],
)
def test_local_seal_rejects_redeclared_artifact_without_cross_link_update(
    tmp_path: Path, artifact_name: str, expected_status: str,
) -> None:
    payload, paths = _fixture_payload(tmp_path)
    paths[artifact_name].write_bytes(paths[artifact_name].read_bytes() + b"new-bytes")
    if artifact_name == "trtexec":
        paths[artifact_name].chmod(0o755)
    payload["artifacts"][artifact_name] = _artifact(paths[artifact_name])

    with pytest.raises(ValueError, match=expected_status):
        seal_native_split_quality_binding(payload)


def test_portable_validation_rejects_tampered_embedded_receipt_bytes(
    tmp_path: Path,
) -> None:
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    tampered = copy.deepcopy(binding)
    row = tampered["local_artifact_verification"]["embedded_json_files"][
        "engine_build_receipt"
    ]
    row["content_base64"] = row["content_base64"][:-4] + "AAAA"
    proof = tampered["local_artifact_verification"]
    proof.pop("proof_sha256")
    proof["proof_sha256"] = canonical_json_sha256(proof)
    tampered.pop("binding_sha256")
    tampered["binding_sha256"] = canonical_json_sha256(tampered)

    verified, status = validate_native_split_quality_binding(
        tampered, verification_mode="portable",
    )

    assert verified is None
    assert status in {
        "native_split_quality_embedded_engine_build_receipt_size_mismatch",
        "native_split_quality_embedded_engine_build_receipt_file_sha256_mismatch",
        "native_split_quality_embedded_engine_build_receipt_json_invalid",
    }
