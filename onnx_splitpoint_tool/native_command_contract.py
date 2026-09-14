"""Hash-bound replay contracts for Native performance and energy workloads.

The successful performance command is the source of truth.  Energy and A/B
stages may change only the requested duration and fresh evidence paths; they
must never reconstruct preprocessing, boundary, queue, or runtime options from
profile-wide defaults.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Mapping


NATIVE_COMMAND_CONTRACT_SCHEMA = "onnx-splitpoint/native-command-contract"
NATIVE_COMMAND_CONTRACT_VERSION = 1

SPLIT_ENERGY_PREFLIGHT_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/energy-preflight-attestation"
)
SPLIT_ENERGY_WORKLOAD_BINDING_SCHEMA = (
    "onnx-splitpoint/split-energy-workload-binding"
)
SPLIT_ENERGY_PREFLIGHT_STDOUT_MARKER = (
    "__SPLITPOINT_PREFLIGHT_ATTESTATION__="
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deepx_preprocess_binding(
    options: Mapping[str, Any],
    prepared_contract: Mapping[str, Any],
) -> tuple[bool, str, str, str, int, int]:
    """Validate the exact task-specific DeepX spatial-preparation contract."""
    task = str(options.get("task") or "").strip().lower()
    requested_mode = str(options.get("preprocess_mode_requested") or "").strip().lower()
    effective_mode = str(options.get("preprocess_mode_effective") or "").strip().lower()
    try:
        requested_pad = int(options.get("letterbox_pad_value_requested"))
        effective_pad = int(options.get("letterbox_pad_value_effective"))
    except (TypeError, ValueError, OverflowError):
        return False, task, requested_mode, effective_mode, -1, -1
    expected_mode = (
        "letterbox" if task == "detection" else "resize"
        if task == "classification" else ""
    ) if requested_mode == "auto" else requested_mode
    expected_pad = requested_pad if expected_mode == "letterbox" else 0
    valid = bool(
        task in {"classification", "detection"}
        and requested_mode in {"auto", "resize", "letterbox"}
        and effective_mode in {"resize", "letterbox"}
        and effective_mode == expected_mode
        and 0 <= requested_pad <= 255
        and effective_pad == expected_pad
        and int(options.get("letterbox_pad_value") if options.get("letterbox_pad_value") is not None else -1) == effective_pad
        and str(prepared_contract.get("task") or "").strip().lower() == task
        and str(prepared_contract.get("preprocess_mode_requested") or "").strip().lower() == requested_mode
        and str(prepared_contract.get("preprocess_mode_effective") or "").strip().lower() == effective_mode
        and int(prepared_contract.get("letterbox_pad_value_requested") if prepared_contract.get("letterbox_pad_value_requested") is not None else -1) == requested_pad
        and int(prepared_contract.get("letterbox_pad_value_effective") if prepared_contract.get("letterbox_pad_value_effective") is not None else -1) == effective_pad
        and int(prepared_contract.get("letterbox_pad_value") if prepared_contract.get("letterbox_pad_value") is not None else -1) == effective_pad
    )
    return valid, task, requested_mode, effective_mode, requested_pad, effective_pad


def hailo8_preprocess_binding(
    options: Mapping[str, Any],
    prepared_contract: Mapping[str, Any],
) -> bool:
    """Validate Hailo-8 task geometry and both requested/effective pad values."""
    task = str(options.get("task") or "").strip().lower()
    requested_mode = str(options.get("preprocess_mode_requested") or "").strip().lower()
    effective_mode = str(options.get("preprocess_mode_effective") or "").strip().lower()
    try:
        requested_pad = int(options.get("letterbox_pad_value_requested"))
        effective_pad = int(options.get("letterbox_pad_value_effective"))
        legacy_effective_pad = int(options.get("letterbox_pad_value"))
        prepared_requested_pad = int(prepared_contract.get("letterbox_pad_value_requested"))
        prepared_effective_pad = int(prepared_contract.get("letterbox_pad_value_effective"))
        prepared_legacy_effective_pad = int(prepared_contract.get("letterbox_pad_value"))
        prepared_pad_effective = int(prepared_contract.get("pad_value_effective"))
    except (TypeError, ValueError, OverflowError):
        return False
    expected_mode = (
        "letterbox" if task == "detection" else "resize"
        if task == "classification" else ""
    ) if requested_mode == "auto" else requested_mode
    expected_pad = requested_pad if expected_mode == "letterbox" else 0
    return bool(
        task in {"classification", "detection"}
        and requested_mode in {"auto", "resize", "letterbox"}
        and effective_mode in {"resize", "letterbox"}
        and effective_mode == expected_mode
        and 0 <= requested_pad <= 255
        and effective_pad == expected_pad
        and legacy_effective_pad == expected_pad
        and str(prepared_contract.get("task") or "").strip().lower() == task
        and str(prepared_contract.get("preprocess_mode_requested") or "").strip().lower() == requested_mode
        and str(prepared_contract.get("preprocess_mode_effective") or "").strip().lower() == effective_mode
        and prepared_requested_pad == requested_pad
        and prepared_effective_pad == expected_pad
        and prepared_legacy_effective_pad == expected_pad
        and prepared_pad_effective == expected_pad
    )


def hailo10_preprocess_binding(
    options: Mapping[str, Any],
    prepared_contract: Mapping[str, Any],
) -> bool:
    """Validate the persisted Hailo-10 feed against its task/preprocess contract."""
    task = str(options.get("task") or "").strip().lower()
    requested_mode = str(options.get("preprocess_mode_requested") or "").strip().lower()
    effective_mode = str(options.get("preprocess_mode_effective") or "").strip().lower()
    try:
        requested_pad = int(options.get("letterbox_pad_value_requested"))
        effective_pad = int(options.get("letterbox_pad_value_effective"))
        legacy_effective_pad = int(options.get("letterbox_pad_value"))
        prepared_requested_pad = int(prepared_contract.get("letterbox_pad_value_requested"))
        prepared_effective_pad = int(prepared_contract.get("letterbox_pad_value_effective"))
        prepared_pad = int(prepared_contract.get("letterbox_pad_value"))
        prepared_pad_effective = int(prepared_contract.get("pad_value_effective"))
    except (TypeError, ValueError, OverflowError):
        return False
    expected_mode = (
        "letterbox" if task == "detection" else "resize"
        if task == "classification" else ""
    ) if requested_mode == "auto" else requested_mode
    expected_pad = requested_pad if expected_mode == "letterbox" else 0
    canonical_input_names = [
        str(value) for value in list(options.get("canonical_input_slot_names") or [])
    ]
    prepared_slot_order = [
        str(value) for value in list(prepared_contract.get("slot_order") or [])
    ]
    raw_prepared_entries = list(prepared_contract.get("entries") or [])
    prepared_entries_valid = bool(
        raw_prepared_entries
        and all(isinstance(entry, Mapping) for entry in raw_prepared_entries)
    )
    prepared_entry_names = [
        str(entry.get("name") or "")
        for entry in raw_prepared_entries
        if isinstance(entry, Mapping)
    ]
    prepared_artifact_names = [
        str(entry.get("artifact_name") or "")
        for entry in raw_prepared_entries
        if isinstance(entry, Mapping)
    ]
    expected_normalization = (
        "hef_quant_info_from_imagenet_float32"
        if task == "classification"
        else "hef_quant_info_from_unit_float32"
    )
    try:
        native_entries_valid = bool(
            prepared_entries_valid
            and all(
                str(entry.get("dtype") or "").strip().lower() == "uint8"
                and entry.get("c_contiguous") is True
                and bool(list(entry.get("shape") or []))
                and all(
                    int(dim) > 0 for dim in list(entry.get("shape") or [])
                )
                for entry in raw_prepared_entries
                if isinstance(entry, Mapping)
            )
        )
    except (TypeError, ValueError, OverflowError):
        native_entries_valid = False
    return bool(
        task in {"classification", "detection"}
        and options.get("quantized_inputs") is True
        and options.get("quantized_outputs") is True
        and requested_mode in {"auto", "resize", "letterbox"}
        and effective_mode in {"resize", "letterbox"}
        and effective_mode == expected_mode
        and 0 <= requested_pad <= 255
        and legacy_effective_pad == expected_pad
        and effective_pad == expected_pad
        and str(prepared_contract.get("task") or "").strip().lower() == task
        and str(prepared_contract.get("preprocess_mode_requested") or "").strip().lower() == requested_mode
        and str(prepared_contract.get("preprocess_mode_effective") or "").strip().lower() == effective_mode
        and str(prepared_contract.get("preprocess_mode") or "").strip().lower() == effective_mode
        and str(prepared_contract.get("format") or "").strip().lower()
        == "numpy_npy_v1"
        and str(prepared_contract.get("preprocess") or "").strip().lower()
        == "exact_performance_prepared_tensor_persisted"
        and str(prepared_contract.get("normalization") or "").strip().lower()
        == expected_normalization
        and prepared_requested_pad == requested_pad
        and prepared_effective_pad == expected_pad
        and prepared_pad == expected_pad
        and prepared_pad_effective == expected_pad
        and canonical_input_names
        and len(set(canonical_input_names)) == len(canonical_input_names)
        and prepared_slot_order == canonical_input_names
        and native_entries_valid
        and len(raw_prepared_entries) == len(canonical_input_names)
        and prepared_entry_names == canonical_input_names
        and all(prepared_artifact_names)
        and len(set(prepared_artifact_names)) == len(prepared_artifact_names)
    )


def seal_native_command_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an immutable-content identity for a complete contract payload."""
    contract = dict(payload)
    contract.pop("contract_sha256", None)
    contract["schema"] = NATIVE_COMMAND_CONTRACT_SCHEMA
    contract["schema_version"] = NATIVE_COMMAND_CONTRACT_VERSION
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


_QUALITY_FIRST_BINDING_SCHEMA = "onnx-splitpoint/native-split-quality-binding"
_QUALITY_FIRST_PRESELECTION_SCHEMA = (
    "onnx-splitpoint/native-split-quality-preselection"
)
_QUALITY_FIRST_LOCAL_PROOF_SCHEMA = (
    "onnx-splitpoint/native-split-local-artifact-verification"
)
_QUALITY_FIRST_ARTIFACTS = (
    "part1_runtime",
    "boundary_metadata",
    "source_part2_onnx",
    "build_part2_onnx",
    "engine",
    "native_trt_meta",
    "engine_build_receipt",
    "trtexec",
)
_QUALITY_FIRST_EMBEDDED_JSON_ARTIFACTS = (
    "boundary_metadata",
    "native_trt_meta",
    "engine_build_receipt",
)
_QUALITY_FIRST_MARKERS = (
    "native_split_quality_binding",
    "native_split_quality_binding_sha256",
    "native_split_quality_eval_run_id",
    "native_split_quality_source_run_id",
    "native_split_quality_local_verification",
    "quality_preselection",
    "quality_preselection_sha256",
    "quality_boundary_contract",
    "quality_boundary_contract_sha256",
)
_QUALITY_FIRST_CENTRAL_SELECTION_SCHEMA = (
    "onnx-splitpoint/native-split-quality-central-selection"
)
_QUALITY_FIRST_CENTRAL_BINDING_FIELDS = (
    "producer_binding_sha256",
    "source_request_sha256",
    "central_result_sha256",
    "central_quality_selection",
    "central_quality_selection_sha256",
)
_QUALITY_FIRST_CENTRAL_COMMAND_FIELDS = (
    "native_split_quality_source_request_sha256",
    "native_split_quality_central_result_sha256",
    "native_split_quality_selection_sha256",
)


def _strict_sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def _quality_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _quality_schema_version(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _quality_case(value: Any) -> str:
    token = _quality_token(value)
    digits = token[1:] if token.startswith("b") else token
    return f"b{int(digits):03d}" if digits.isdigit() else token


def _quality_backend(value: Any, setup_id: Any = "") -> str:
    token = _quality_token(value)
    setup = _quality_token(setup_id)
    if token in {"hailo8", "hailo8_to_trt", "hailo8_to_tensorrt"}:
        return "hailo8_to_trt"
    if token in {
        "hailo10", "hailo10h", "hailo10_to_trt", "hailo10h_to_trt",
        "hailo10_to_tensorrt", "hailo10h_to_tensorrt",
    }:
        return "hailo10h_to_trt"
    if token in {
        "deepx", "deepx_m1", "deepx_to_trt", "deepx_m1_to_trt",
        "deepx_to_tensorrt", "deepx_m1_to_tensorrt",
    }:
        return "deepx_to_trt"
    if not token:
        if "hailo8" in setup:
            return "hailo8_to_trt"
        if "hailo10" in setup:
            return "hailo10h_to_trt"
        if "deepx" in setup:
            return "deepx_to_trt"
    return token


def _quality_artifact_identity(
    value: Any, *, role: str,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(value, Mapping):
        return None, f"native_command_contract_quality_artifact_{role}_invalid"
    path = str(value.get("path") or "").strip()
    sha = _strict_sha256(value.get("sha256"))
    size_raw = value.get("size_bytes")
    try:
        size = int(size_raw)
    except (TypeError, ValueError, OverflowError):
        size = 0
    if not path:
        return None, f"native_command_contract_quality_artifact_{role}_path_missing"
    if not sha:
        return None, f"native_command_contract_quality_artifact_{role}_sha256_missing"
    if isinstance(size_raw, bool) or size <= 0:
        return None, f"native_command_contract_quality_artifact_{role}_size_missing"
    return {"path": path, "sha256": sha, "size_bytes": size}, "verified"


def _strict_embedded_json_object(raw: bytes) -> dict[str, Any] | None:
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_object)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) and not duplicate else None


def _verify_quality_first_command_contract(
    contract: Mapping[str, Any], *, artifacts: Mapping[str, Any], backend: str,
) -> str:
    """Validate the portable Quality-FIRST closure without opening remote paths."""
    if not any(field in contract for field in _QUALITY_FIRST_MARKERS):
        return "legacy_contract_without_quality_first_binding"
    if any(field not in contract for field in _QUALITY_FIRST_MARKERS):
        return "native_command_contract_quality_first_fields_incomplete"

    binding_raw = contract.get("native_split_quality_binding")
    if not isinstance(binding_raw, Mapping):
        return "native_command_contract_quality_binding_missing"
    binding = dict(binding_raw)
    binding_sha = _strict_sha256(binding.pop("binding_sha256", ""))
    if not binding_sha or canonical_json_sha256(binding) != binding_sha:
        return "native_command_contract_quality_binding_payload_sha256_mismatch"
    binding["binding_sha256"] = binding_sha
    if (
        binding.get("schema") != _QUALITY_FIRST_BINDING_SCHEMA
        or _quality_schema_version(binding.get("schema_version")) != 1
        or binding.get("quality_completed") is not True
        or binding.get("performance_claims_emitted") is not False
    ):
        return "native_command_contract_quality_binding_schema_or_role_invalid"
    if _strict_sha256(contract.get("native_split_quality_binding_sha256")) != binding_sha:
        # Keep the bind_quality_to_native_split public status stable: that
        # caller prefixes verifier failures with ``native_split_quality_``.
        return "command_binding_sha256_mismatch"

    eval_run_id = str(binding.get("eval_run_id") or "").strip()
    source_run_id = str(binding.get("source_run_id") or "").strip()
    if not eval_run_id or not source_run_id:
        return "native_command_contract_quality_binding_eval_or_source_missing"
    binding_selection = (
        binding.get("preselection")
        if isinstance(binding.get("preselection"), Mapping)
        else {}
    )
    binding_setup_id = binding_selection.get("setup_id")
    for field in ("eval_run_id", "native_split_quality_eval_run_id"):
        if str(contract.get(field) or "").strip() != eval_run_id:
            return "command_eval_run_id_mismatch"
    for field in ("source_run_id", "native_split_quality_source_run_id"):
        if _quality_backend(
            contract.get(field), binding_setup_id,
        ) != _quality_backend(source_run_id, binding_setup_id):
            return "command_source_run_id_mismatch"
    canonical_backend = _quality_backend(backend)
    if _quality_backend(source_run_id) != canonical_backend:
        return "native_command_contract_quality_source_backend_mismatch"

    central_receipt: dict[str, Any] = {}
    binding_selection_present = [
        field in binding for field in _QUALITY_FIRST_CENTRAL_BINDING_FIELDS
    ]
    command_selection_present = [
        field in contract for field in _QUALITY_FIRST_CENTRAL_COMMAND_FIELDS
    ]
    if any(binding_selection_present) or any(command_selection_present):
        if not all(binding_selection_present):
            return "native_command_contract_quality_central_selection_binding_incomplete"
        if not all(command_selection_present):
            return "native_command_contract_quality_central_selection_fields_incomplete"
        receipt_raw = binding.get("central_quality_selection")
        if not isinstance(receipt_raw, Mapping):
            return "native_command_contract_quality_central_selection_receipt_missing"
        receipt = dict(receipt_raw)
        receipt_sha = _strict_sha256(receipt.pop("receipt_sha256", ""))
        if not receipt_sha or canonical_json_sha256(receipt) != receipt_sha:
            return "native_command_contract_quality_central_selection_receipt_sha256_mismatch"
        if (
            receipt.get("schema") != _QUALITY_FIRST_CENTRAL_SELECTION_SCHEMA
            or _quality_schema_version(receipt.get("schema_version")) != 1
        ):
            return "native_command_contract_quality_central_selection_schema_invalid"
        receipt["receipt_sha256"] = receipt_sha
        central_receipt = receipt
        producer_binding = dict(binding)
        producer_binding.pop("binding_sha256", None)
        for field in _QUALITY_FIRST_CENTRAL_BINDING_FIELDS:
            producer_binding.pop(field, None)
        producer_sha = _strict_sha256(binding.get("producer_binding_sha256"))
        request_sha = _strict_sha256(binding.get("source_request_sha256"))
        result_sha = _strict_sha256(binding.get("central_result_sha256"))
        if not producer_sha or canonical_json_sha256(producer_binding) != producer_sha:
            return "native_command_contract_quality_central_selection_producer_binding_mismatch"
        if not request_sha or not result_sha:
            return "native_command_contract_quality_central_selection_hash_missing"
        if _strict_sha256(binding.get("central_quality_selection_sha256")) != receipt_sha:
            return "native_command_contract_quality_central_selection_duplicate_mismatch"
        for field, expected in (
            ("producer_binding_sha256", producer_sha),
            ("source_request_sha256", request_sha),
            ("central_result_sha256", result_sha),
        ):
            if _strict_sha256(receipt.get(field)) != expected:
                return f"native_command_contract_quality_central_selection_{field}_mismatch"
        command_hashes = {
            "native_split_quality_source_request_sha256": request_sha,
            "native_split_quality_central_result_sha256": result_sha,
            "native_split_quality_selection_sha256": receipt_sha,
        }
        for field, expected in command_hashes.items():
            if _strict_sha256(contract.get(field)) != expected:
                return f"native_command_contract_quality_{field}_mismatch"

    selection_raw = binding.get("preselection")
    if not isinstance(selection_raw, Mapping):
        return "native_command_contract_quality_preselection_missing"
    selection = dict(selection_raw)
    selection_sha = _strict_sha256(selection.pop("selection_sha256", ""))
    if not selection_sha or canonical_json_sha256(selection) != selection_sha:
        return "native_command_contract_quality_preselection_sha256_mismatch"
    selection["selection_sha256"] = selection_sha
    if (
        selection.get("schema") != _QUALITY_FIRST_PRESELECTION_SCHEMA
        or _quality_schema_version(selection.get("schema_version")) != 1
        or selection.get("engine_rebuild_allowed_after_quality") is not False
    ):
        return "native_command_contract_quality_preselection_schema_invalid"
    if _strict_sha256(binding.get("preselection_sha256")) != selection_sha:
        return "native_command_contract_quality_binding_preselection_duplicate_mismatch"
    command_selection = contract.get("quality_preselection")
    if not isinstance(command_selection, Mapping) or dict(command_selection) != selection:
        return "native_command_contract_quality_preselection_payload_mismatch"
    if _strict_sha256(contract.get("quality_preselection_sha256")) != selection_sha:
        return "native_command_contract_quality_preselection_duplicate_mismatch"

    identity_checks = (
        (_quality_token(selection.get("model_id")), _quality_token(contract.get("model")), "model"),
        (_quality_case(selection.get("case_id")), _quality_case(contract.get("case")), "case"),
        (_quality_token(selection.get("setup_id")), _quality_token(contract.get("setup_id")), "setup_id"),
        (_quality_backend(selection.get("backend")), canonical_backend, "backend"),
        (_quality_token(selection.get("precision")), _quality_token(contract.get("precision")), "precision"),
    )
    for observed, expected, field in identity_checks:
        if not observed or observed != expected:
            return f"native_command_contract_quality_preselection_{field}_mismatch"
    if central_receipt:
        central_identity_checks = (
            (str(central_receipt.get("eval_run_id") or "").strip(), eval_run_id, "eval_run_id"),
            (
                _quality_backend(
                    central_receipt.get("source_run_id"),
                    selection.get("setup_id"),
                ),
                _quality_backend(source_run_id, selection.get("setup_id")),
                "source_run_id",
            ),
            (_quality_token(central_receipt.get("model_id")), _quality_token(selection.get("model_id")), "model_id"),
            (_quality_case(central_receipt.get("case_id")), _quality_case(selection.get("case_id")), "case_id"),
            (_quality_token(central_receipt.get("setup_id")), _quality_token(selection.get("setup_id")), "setup_id"),
            (_quality_backend(central_receipt.get("backend")), _quality_backend(selection.get("backend")), "backend"),
            (_quality_token(central_receipt.get("task")), _quality_token(selection.get("task")), "task"),
            (_quality_token(central_receipt.get("precision")), _quality_token(selection.get("precision")), "precision"),
            (_quality_token(central_receipt.get("runtime_precision_identity")), _quality_token(selection.get("precision")), "runtime_precision_identity"),
            (_quality_token(central_receipt.get("variant")), "composed", "variant"),
        )
        for observed, expected, field in central_identity_checks:
            if not expected or observed != expected:
                return f"native_command_contract_quality_central_selection_{field}_mismatch"

    binding_artifacts_raw = binding.get("artifacts")
    if not isinstance(binding_artifacts_raw, Mapping):
        return "native_command_contract_quality_binding_artifacts_missing"
    binding_artifacts: dict[str, dict[str, Any]] = {}
    for name in _QUALITY_FIRST_ARTIFACTS:
        identity, status = _quality_artifact_identity(
            binding_artifacts_raw.get(name), role=f"binding_{name}",
        )
        if identity is None:
            return status
        binding_artifacts[name] = identity

    for artifact_name, sha_field, size_field in (
        ("part1_runtime", "part1_artifact_sha256", "part1_artifact_size_bytes"),
        (
            "boundary_metadata", "boundary_metadata_file_sha256",
            "boundary_metadata_file_size_bytes",
        ),
    ):
        try:
            selected_size = int(selection.get(size_field))
        except (TypeError, ValueError, OverflowError):
            selected_size = 0
        if (
            _strict_sha256(selection.get(sha_field))
            != binding_artifacts[artifact_name]["sha256"]
            or selected_size != binding_artifacts[artifact_name]["size_bytes"]
        ):
            return f"native_command_contract_quality_{artifact_name}_selection_mismatch"

    artifact_role_map = {
        "part1_runtime": "dxnn" if canonical_backend == "deepx_to_trt" else "hef",
        "boundary_metadata": "boundary_metadata",
        "source_part2_onnx": "source_part2_onnx",
        "build_part2_onnx": "build_part2_onnx",
        "engine": "engine",
        "native_trt_meta": "native_trt_meta",
        "engine_build_receipt": "engine_build_receipt",
        "trtexec": "trtexec",
    }
    for binding_name, command_name in artifact_role_map.items():
        identity, status = _quality_artifact_identity(
            artifacts.get(command_name), role=command_name,
        )
        if identity is None:
            return status
        if identity != binding_artifacts[binding_name]:
            return f"native_command_contract_quality_artifact_{binding_name}_crosslink_mismatch"

    boundary_raw = binding.get("boundary_contract")
    if not isinstance(boundary_raw, Mapping):
        return "native_command_contract_quality_boundary_contract_missing"
    quality_boundary = dict(boundary_raw)
    boundary_sha = _strict_sha256(binding.get("boundary_contract_sha256"))
    if not boundary_sha or canonical_json_sha256(quality_boundary) != boundary_sha:
        return "native_command_contract_quality_boundary_contract_sha256_mismatch"
    command_boundary = contract.get("quality_boundary_contract")
    if not isinstance(command_boundary, Mapping) or dict(command_boundary) != quality_boundary:
        return "native_command_contract_quality_boundary_contract_payload_mismatch"
    if _strict_sha256(contract.get("quality_boundary_contract_sha256")) != boundary_sha:
        return "native_command_contract_quality_boundary_contract_duplicate_mismatch"
    boundary_fields = (
        ("precision", "precision"),
        ("boundary_layout", "boundary_layout"),
        ("boundary_transform", "boundary_transform"),
        ("boundary_tensor_name", "boundary_tensor_name"),
        ("boundary_tensor_dtype", "boundary_tensor_dtype"),
        ("boundary_tensor_shape", "boundary_tensor_shape"),
        ("boundary_metadata_sha256", "boundary_metadata_sha256"),
        ("boundary_metadata_file_sha256", "boundary_metadata_file_sha256"),
        ("dequant_scale", "dequant_scale"),
        ("dequant_zero_point", "dequant_zero_point"),
    )
    for boundary_field, selection_field in boundary_fields:
        if quality_boundary.get(boundary_field) != selection.get(selection_field):
            return f"native_command_contract_quality_boundary_{boundary_field}_mismatch"

    proof_raw = binding.get("local_artifact_verification")
    if not isinstance(proof_raw, Mapping):
        return "native_command_contract_quality_local_verification_missing"
    proof = dict(proof_raw)
    proof_sha = _strict_sha256(proof.pop("proof_sha256", ""))
    if not proof_sha or canonical_json_sha256(proof) != proof_sha:
        return "native_command_contract_quality_local_verification_sha256_mismatch"
    proof["proof_sha256"] = proof_sha
    if (
        proof.get("schema") != _QUALITY_FIRST_LOCAL_PROOF_SCHEMA
        or _quality_schema_version(proof.get("schema_version")) != 1
        or proof.get("verification_kind") != "producer_local_file_rehash"
        or proof.get("artifact_names") != list(_QUALITY_FIRST_ARTIFACTS)
    ):
        return "native_command_contract_quality_local_verification_schema_invalid"
    command_proof = contract.get("native_split_quality_local_verification")
    if not isinstance(command_proof, Mapping) or dict(command_proof) != proof:
        return "native_command_contract_quality_local_verification_payload_mismatch"
    proof_artifacts_raw = proof.get("artifacts")
    if not isinstance(proof_artifacts_raw, Mapping):
        return "native_command_contract_quality_local_verification_artifacts_missing"
    proof_artifacts: dict[str, dict[str, Any]] = {}
    for name in _QUALITY_FIRST_ARTIFACTS:
        identity, status = _quality_artifact_identity(
            proof_artifacts_raw.get(name), role=f"local_proof_{name}",
        )
        if identity is None:
            return status
        proof_artifacts[name] = identity
    if proof_artifacts != binding_artifacts:
        return "native_command_contract_quality_local_verification_artifacts_mismatch"
    if _strict_sha256(proof.get("artifact_set_sha256")) != canonical_json_sha256(
        binding_artifacts
    ):
        return "native_command_contract_quality_local_verification_artifact_set_mismatch"

    embedded = proof.get("embedded_json_files")
    if not isinstance(embedded, Mapping) or set(embedded) != set(
        _QUALITY_FIRST_EMBEDDED_JSON_ARTIFACTS
    ):
        return "native_command_contract_quality_local_verification_embedded_json_incomplete"
    for name in _QUALITY_FIRST_EMBEDDED_JSON_ARTIFACTS:
        evidence = embedded.get(name)
        if not isinstance(evidence, Mapping) or evidence.get("encoding") != "base64":
            return f"native_command_contract_quality_local_verification_{name}_encoding_invalid"
        try:
            content = base64.b64decode(
                str(evidence.get("content_base64") or ""), validate=True,
            )
            evidence_size = int(evidence.get("file_size_bytes"))
        except Exception:
            return f"native_command_contract_quality_local_verification_{name}_content_invalid"
        artifact = binding_artifacts[name]
        if len(content) != evidence_size or evidence_size != artifact["size_bytes"]:
            return f"native_command_contract_quality_local_verification_{name}_size_mismatch"
        content_sha = hashlib.sha256(content).hexdigest()
        if (
            content_sha != _strict_sha256(evidence.get("file_sha256"))
            or content_sha != artifact["sha256"]
        ):
            return f"native_command_contract_quality_local_verification_{name}_sha256_mismatch"
        parsed = _strict_embedded_json_object(content)
        if (
            parsed is None
            or canonical_json_sha256(parsed)
            != _strict_sha256(evidence.get("canonical_value_sha256"))
        ):
            return f"native_command_contract_quality_local_verification_{name}_json_mismatch"

    for name in ("semantic_output_manifest", "semantic_boundary_manifest"):
        identity, status = _quality_artifact_identity(artifacts.get(name), role=name)
        if identity is None:
            return status
    return "quality_first_binding_and_portable_crosslinks_verified"


def _verify_hailo8_mixed_runtime_contract(
    contract: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
    options: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> str:
    """Verify the system-TRT/process-local-Hailo execution closure.

    This is a portable contract check only.  File existence and byte hashes
    remain the responsibility of the remote split-energy preflight.
    """
    prepared = contract.get("prepared_input_contract")
    prepared_task = (
        str(prepared.get("task") or "").strip().lower()
        if isinstance(prepared, Mapping) else ""
    )
    task = str(options.get("task") or prepared_task).strip().lower()
    producer_impl = str(
        options.get("producer_impl") or ""
    ).strip().lower()
    mixed_raw = contract.get("mixed_runtime_contract")
    completion_contract = options.get(
        "completion_execution_contract"
    )
    activated = bool(
        producer_impl == "hailo8_python_vstreams_fifo"
        or isinstance(mixed_raw, Mapping)
        or "native_trt_consumer_source" in artifacts
        or isinstance(completion_contract, Mapping)
    )
    if not activated:
        return "not_required"
    if task != "detection":
        return "native_command_contract_hailo8_detection_task_invalid"
    if producer_impl != "hailo8_python_vstreams_fifo":
        return (
            "native_command_contract_hailo8_detection_producer_invalid"
        )
    if not isinstance(mixed_raw, Mapping):
        return (
            "native_command_contract_hailo8_mixed_runtime_contract_missing"
        )
    mixed = dict(mixed_raw)
    expected_mode = (
        "system_tensorrt_with_process_local_hailo_sites"
    )
    if (
        str(identity.get("runtime_mode") or "") != expected_mode
        or str(mixed.get("runtime_mode") or "") != expected_mode
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_mode_invalid"
        )
    expected_policy = "site.addsitedir_after_system_defaults"
    if (
        str(options.get("mixed_runtime_site_policy") or "")
        != expected_policy
        or str(mixed.get("site_policy") or "") != expected_policy
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_site_policy_invalid"
        )

    def sites(value: Any) -> list[str] | None:
        if not isinstance(value, list) or not value:
            return None
        result: list[str] = []
        for raw in value:
            path = str(raw or "").strip()
            if (
                not path
                or not os.path.isabs(path)
                or any(char in path for char in ("\x00", "\n", "\r"))
                or path in result
            ):
                return None
            result.append(os.path.normpath(path))
        return result

    option_sites = sites(options.get("process_local_extra_sites"))
    identity_sites = sites(
        identity.get("process_local_extra_sites")
    )
    mixed_sites = sites(mixed.get("process_local_extra_sites"))
    if (
        option_sites is None
        or identity_sites is None
        or mixed_sites is None
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_sites_invalid"
        )
    if not (option_sites == identity_sites == mixed_sites):
        return (
            "native_command_contract_hailo8_mixed_runtime_sites_mismatch"
        )

    contract_python = str(
        contract.get("python_executable") or ""
    )
    resolved_python = str(
        identity.get("resolved_executable") or ""
    )
    if (
        str(mixed.get("python_executable") or "")
        != contract_python
        or not resolved_python
        or str(mixed.get("resolved_python_executable") or "")
        != resolved_python
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_interpreter_mismatch"
        )
    modules = mixed.get("modules")
    required_modules = ("tensorrt", "hailo_platform", "numpy", "PIL")
    if (
        not isinstance(modules, Mapping)
        or any(
            not str(modules.get(name) or "").strip()
            for name in required_modules
        )
        or not str(mixed.get("cudart") or "").strip()
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_modules_incomplete"
        )

    def under(path: str, root: str) -> bool:
        try:
            return os.path.commonpath(
                [os.path.normpath(path), os.path.normpath(root)]
            ) == os.path.normpath(root)
        except (TypeError, ValueError):
            return False

    hailo_module = str(modules.get("hailo_platform") or "")
    tensorrt_module = str(modules.get("tensorrt") or "")
    if (
        not any(under(hailo_module, site) for site in option_sites)
        or any(under(tensorrt_module, site) for site in option_sites)
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_module_origin_invalid"
        )
    if (
        mixed.get("status") != "ready"
        or mixed.get("source_closure_ok") is not True
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_source_closure_invalid"
        )

    source_artifact = artifacts.get(
        "native_trt_consumer_source"
    )
    if not isinstance(source_artifact, Mapping):
        return (
            "native_command_contract_required_artifact_missing:"
            "native_trt_consumer_source"
        )
    source_path = str(source_artifact.get("path") or "").strip()
    source_sha = _strict_sha256(source_artifact.get("sha256"))
    if (
        not source_path
        or not source_sha
        or source_path
        != str(mixed.get("native_trt_consumer_source") or "")
        or source_sha
        != _strict_sha256(
            mixed.get("native_trt_consumer_source_sha256")
        )
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_consumer_source_mismatch"
        )
    native_executable = artifacts.get("native_executable")
    python_artifact = artifacts.get("python_executable")
    if (
        not isinstance(native_executable, Mapping)
        or not isinstance(python_artifact, Mapping)
        or str(native_executable.get("path") or "")
        != resolved_python
        or _strict_sha256(native_executable.get("sha256"))
        != _strict_sha256(python_artifact.get("sha256"))
    ):
        return (
            "native_command_contract_hailo8_mixed_runtime_executable_mismatch"
        )
    return "verified"


def _verify_native_command_contract_technical(
    raw: Any,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify the command's technical replay closure and row identity."""
    if not isinstance(raw, Mapping):
        return None, "native_command_contract_missing"
    contract = dict(raw)
    declared = str(contract.pop("contract_sha256", "") or "").strip().lower()
    if len(declared) != 64 or canonical_json_sha256(contract) != declared:
        return None, "native_command_contract_sha256_mismatch"
    contract["contract_sha256"] = declared
    if (
        contract.get("schema") != NATIVE_COMMAND_CONTRACT_SCHEMA
        or _quality_schema_version(contract.get("schema_version"))
        != NATIVE_COMMAND_CONTRACT_VERSION
        or contract.get("complete") is not True
    ):
        return None, "native_command_contract_incomplete_or_schema_invalid"
    if expected_identity is not None:
        for field in ("backend", "model", "case", "precision", "setup_id", "comparison_backend"):
            expected = str(expected_identity.get(field) or "").strip().lower()
            actual = str(contract.get(field) or "").strip().lower()
            if expected and actual != expected:
                return None, f"native_command_contract_{field}_mismatch"
    required_hashes = ("runner_sha256", "input_image_sha256")
    if any(len(str(contract.get(field) or "").strip()) != 64 for field in required_hashes):
        return None, "native_command_contract_required_sha256_missing"
    backend = str(contract.get("backend") or "").strip().lower()
    required_artifacts = {
        "hailo8_to_trt": (
            "python_executable", "hef", "engine", "native_executable",
            "generated_cpp", "cmake",
        ),
        "hailo10h_to_trt": ("python_executable", "hef", "engine"),
        "deepx_to_trt": ("python_executable", "dxnn", "engine"),
    }.get(backend)
    if required_artifacts is None:
        return None, "native_command_contract_backend_unsupported"
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return None, "native_command_contract_artifacts_missing"
    missing_artifacts = [name for name in required_artifacts if name not in artifacts]
    if missing_artifacts:
        return None, "native_command_contract_required_artifact_missing:" + ",".join(missing_artifacts)
    for name, raw_artifact in artifacts.items():
        if not isinstance(raw_artifact, Mapping):
            return None, f"native_command_contract_artifact_{name}_invalid"
        if not str(raw_artifact.get("path") or "").strip():
            return None, f"native_command_contract_artifact_{name}_path_missing"
        if len(str(raw_artifact.get("sha256") or "").strip()) != 64:
            return None, f"native_command_contract_artifact_{name}_sha256_missing"
    interpreter = artifacts.get("python_executable")
    identity = contract.get("interpreter_identity")
    if not isinstance(interpreter, Mapping) or not isinstance(identity, Mapping):
        return None, "native_command_contract_interpreter_identity_missing"
    if str(interpreter.get("path") or "") != str(contract.get("python_executable") or ""):
        return None, "native_command_contract_interpreter_path_mismatch"
    if str(identity.get("executable") or "") != str(contract.get("python_executable") or ""):
        return None, "native_command_contract_interpreter_identity_path_mismatch"
    if str(identity.get("executable_sha256") or "") != str(interpreter.get("sha256") or ""):
        return None, "native_command_contract_interpreter_identity_sha256_mismatch"
    options = contract.get("runtime_options")
    boundary = contract.get("boundary_contract")
    if not isinstance(options, Mapping) or not isinstance(boundary, Mapping):
        return None, "native_command_contract_runtime_or_boundary_options_missing"
    for field in ("warmup", "queue_depth"):
        if options.get(field) in (None, ""):
            return None, f"native_command_contract_{field}_missing"
    if not str(boundary.get("boundary_layout_effective") or "").strip():
        return None, "native_command_contract_boundary_layout_missing"
    for field in ("runner", "python_executable", "benchmark_set", "input_image"):
        if not str(contract.get(field) or "").strip():
            return None, f"native_command_contract_{field}_missing"
    if backend == "hailo8_to_trt":
        mixed_runtime_status = _verify_hailo8_mixed_runtime_contract(
            contract,
            artifacts=artifacts,
            options=options,
            identity=identity,
        )
        if mixed_runtime_status not in {"not_required", "verified"}:
            return None, mixed_runtime_status
    return contract, "hash_schema_identity_and_artifacts_verified"


def verify_native_energy_command_contract(
    raw: Any,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify only technical facts needed to replay a Native Energy row.

    Quality-FIRST fields remain hash-bound inside the command contract, but
    their semantic/claim role is deliberately not an Energy-plan membership
    condition.  Consumers that authorize a Quality or scientific claim must
    continue to use :func:`verify_native_command_contract` below.
    """

    return _verify_native_command_contract_technical(
        raw, expected_identity=expected_identity,
    )


def verify_native_command_contract(
    raw: Any,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify technical replay closure plus the full Quality-FIRST binding."""

    contract, technical_status = _verify_native_command_contract_technical(
        raw, expected_identity=expected_identity,
    )
    if contract is None:
        return None, technical_status
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):  # defensive; core already checks
        return None, "native_command_contract_artifacts_missing"
    backend = str(contract.get("backend") or "").strip().lower()
    quality_status = _verify_quality_first_command_contract(
        contract, artifacts=artifacts, backend=backend,
    )
    if quality_status not in {
        "legacy_contract_without_quality_first_binding",
        "quality_first_binding_and_portable_crosslinks_verified",
    }:
        return None, quality_status
    return contract, (
        "hash_schema_identity_artifacts_and_quality_first_crosslinks_verified"
        if quality_status == "quality_first_binding_and_portable_crosslinks_verified"
        else "hash_schema_identity_and_artifacts_verified"
    )


def _technical_artifact_identity(
    raw: Any, *, role: str,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, Mapping):
        return None, f"native_energy_part2_{role}_artifact_missing"
    path = str(raw.get("path") or "").strip()
    sha256 = _strict_sha256(raw.get("sha256"))
    size = raw.get("size_bytes")
    if (
        not path
        or not sha256
        or type(size) is not int
        or size <= 0
    ):
        return None, f"native_energy_part2_{role}_artifact_invalid"
    return {
        "path": path,
        "sha256": sha256,
        "size_bytes": size,
    }, "verified"


def verify_native_split_part2_input_contract(
    raw: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Verify one static Part-2 input without consulting Quality outcomes.

    Current producer contracts carry the TensorRT metadata bytes in a sealed
    local-artifact proof.  This verifier binds those bytes to the command's
    actual Part-2 ONNX, engine, receipt and metadata artifacts, then validates
    the single static input.  Quality completion, Semantics, pairing and claim
    roles are intentionally outside this technical proof.
    """

    contract, command_status = _verify_native_command_contract_technical(raw)
    if contract is None:
        return None, command_status
    binding_raw = contract.get("native_split_quality_binding")
    if not isinstance(binding_raw, Mapping):
        return None, "native_energy_part2_sealed_binding_missing"
    binding = dict(binding_raw)
    declared_binding_sha = _strict_sha256(
        binding.pop("binding_sha256", "")
    )
    if (
        not declared_binding_sha
        or canonical_json_sha256(binding) != declared_binding_sha
    ):
        return None, "native_energy_part2_binding_sha256_mismatch"
    binding["binding_sha256"] = declared_binding_sha
    if (
        binding.get("schema") != _QUALITY_FIRST_BINDING_SCHEMA
        or type(binding.get("schema_version")) is not int
        or binding.get("schema_version") != 1
    ):
        return None, "native_energy_part2_binding_schema_invalid"
    if (
        _strict_sha256(
            contract.get("native_split_quality_binding_sha256")
        )
        != declared_binding_sha
    ):
        return None, "native_energy_part2_binding_duplicate_sha256_mismatch"

    binding_artifacts_raw = binding.get("artifacts")
    if not isinstance(binding_artifacts_raw, Mapping):
        return None, "native_energy_part2_binding_artifacts_missing"
    binding_artifacts: dict[str, dict[str, Any]] = {}
    for name in _QUALITY_FIRST_ARTIFACTS:
        identity, status = _technical_artifact_identity(
            binding_artifacts_raw.get(name), role=f"binding_{name}",
        )
        if identity is None:
            return None, status
        binding_artifacts[name] = identity

    proof_raw = binding.get("local_artifact_verification")
    if not isinstance(proof_raw, Mapping):
        return None, "native_energy_part2_local_proof_missing"
    proof = dict(proof_raw)
    declared_proof_sha = _strict_sha256(proof.pop("proof_sha256", ""))
    if not declared_proof_sha or canonical_json_sha256(proof) != declared_proof_sha:
        return None, "native_energy_part2_local_proof_sha256_mismatch"
    proof["proof_sha256"] = declared_proof_sha
    if (
        proof.get("schema") != _QUALITY_FIRST_LOCAL_PROOF_SCHEMA
        or type(proof.get("schema_version")) is not int
        or proof.get("schema_version") != 1
        or proof.get("verification_kind") != "producer_local_file_rehash"
        or proof.get("artifact_names") != list(_QUALITY_FIRST_ARTIFACTS)
    ):
        return None, "native_energy_part2_local_proof_schema_invalid"
    proof_artifacts_raw = proof.get("artifacts")
    if not isinstance(proof_artifacts_raw, Mapping):
        return None, "native_energy_part2_local_proof_artifacts_missing"
    proof_artifacts: dict[str, dict[str, Any]] = {}
    for name in _QUALITY_FIRST_ARTIFACTS:
        identity, status = _technical_artifact_identity(
            proof_artifacts_raw.get(name), role=f"proof_{name}",
        )
        if identity is None:
            return None, status
        proof_artifacts[name] = identity
    if proof_artifacts != binding_artifacts:
        return None, "native_energy_part2_local_proof_artifacts_mismatch"
    if (
        _strict_sha256(proof.get("artifact_set_sha256"))
        != canonical_json_sha256(proof_artifacts)
    ):
        return None, "native_energy_part2_local_proof_artifact_set_mismatch"

    command_artifacts = contract.get("artifacts")
    if not isinstance(command_artifacts, Mapping):
        return None, "native_energy_part2_command_artifacts_missing"
    for name in (
        "build_part2_onnx", "engine", "native_trt_meta",
        "engine_build_receipt",
    ):
        identity, status = _technical_artifact_identity(
            command_artifacts.get(name), role=f"command_{name}",
        )
        if identity is None:
            return None, status
        if identity != binding_artifacts[name]:
            return None, f"native_energy_part2_{name}_command_crosslink_mismatch"

    evidence = proof.get("embedded_json_files")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != set(_QUALITY_FIRST_EMBEDDED_JSON_ARTIFACTS)
    ):
        return None, "native_energy_part2_embedded_evidence_incomplete"
    metadata_evidence = evidence.get("native_trt_meta")
    if (
        not isinstance(metadata_evidence, Mapping)
        or metadata_evidence.get("encoding") != "base64"
    ):
        return None, "native_energy_part2_metadata_evidence_invalid"
    try:
        metadata_bytes = base64.b64decode(
            str(metadata_evidence.get("content_base64") or ""),
            validate=True,
        )
    except Exception:
        return None, "native_energy_part2_metadata_base64_invalid"
    metadata_artifact = binding_artifacts["native_trt_meta"]
    if (
        type(metadata_evidence.get("file_size_bytes")) is not int
        or len(metadata_bytes) != metadata_evidence.get("file_size_bytes")
        or len(metadata_bytes) != metadata_artifact["size_bytes"]
    ):
        return None, "native_energy_part2_metadata_size_mismatch"
    metadata_file_sha = hashlib.sha256(metadata_bytes).hexdigest()
    if (
        metadata_file_sha
        != _strict_sha256(metadata_evidence.get("file_sha256"))
        or metadata_file_sha != metadata_artifact["sha256"]
    ):
        return None, "native_energy_part2_metadata_file_sha256_mismatch"
    metadata = _strict_embedded_json_object(metadata_bytes)
    if metadata is None:
        return None, "native_energy_part2_metadata_json_invalid"
    metadata_value_sha = canonical_json_sha256(metadata)
    if (
        metadata_value_sha
        != _strict_sha256(
            metadata_evidence.get("canonical_value_sha256")
        )
        or metadata_value_sha
        != _strict_sha256(binding.get("native_trt_meta_payload_sha256"))
        or metadata_value_sha
        != _strict_sha256(proof.get("native_trt_meta_payload_sha256"))
    ):
        return None, "native_energy_part2_metadata_value_sha256_mismatch"
    for field in ("native_trt_meta_payload", "native_trt_meta"):
        duplicate = binding.get(field)
        if not isinstance(duplicate, Mapping) or dict(duplicate) != metadata:
            return None, f"native_energy_part2_{field}_mismatch"
    if (
        _strict_sha256(binding.get("native_trt_meta_file_sha256"))
        != metadata_file_sha
        or type(binding.get("native_trt_meta_file_size_bytes")) is not int
        or binding.get("native_trt_meta_file_size_bytes")
        != len(metadata_bytes)
    ):
        return None, "native_energy_part2_metadata_file_duplicate_mismatch"

    inputs = metadata.get("inputs")
    input_row = inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else None
    shape = input_row.get("shape") if isinstance(input_row, Mapping) else None
    precision = str(contract.get("precision") or "").strip().lower()
    metadata_precision = str(metadata.get("precision") or "").strip().lower()
    requested_precision = str(
        metadata.get("requested_precision") or ""
    ).strip().lower()
    dtype = str(
        input_row.get("elem_type") if isinstance(input_row, Mapping) else ""
    ).strip().upper()
    normalized_dtype = {
        "FLOAT": "float32",
        "FLOAT32": "float32",
        "UINT8": "uint8",
    }.get(dtype, "")
    expected_dtype = {
        "float32_layout_fp16": "float32",
        "uint8_dequant_fp16": "uint8",
        "uint8_cast_fp16": "uint8",
    }.get(precision, "")
    if not (
        metadata.get("schema") == "onnx-splitpoint/native-trt-meta"
        and type(metadata.get("schema_version")) is int
        and metadata.get("schema_version") == 1
        and str(metadata.get("variant") or "").strip().lower() == "part2"
        and metadata.get("build_ok") is True
        and metadata.get("inputs_static") is True
        and isinstance(input_row, Mapping)
        and str(input_row.get("name") or "").strip()
        and input_row.get("has_dynamic") is False
        and isinstance(shape, list)
        and bool(shape)
        and all(type(dimension) is int and dimension > 0 for dimension in shape)
        and expected_dtype
        and normalized_dtype == expected_dtype
        and metadata_precision == precision
        and requested_precision == precision
    ):
        return None, "native_energy_part2_single_static_input_invalid"

    path_crosslinks = {
        "onnx": binding_artifacts["build_part2_onnx"]["path"],
        "source_onnx": binding_artifacts["build_part2_onnx"]["path"],
        "engine": binding_artifacts["engine"]["path"],
        "engine_build_receipt_path": binding_artifacts[
            "engine_build_receipt"
        ]["path"],
    }
    if any(
        str(metadata.get(field) or "") != expected
        for field, expected in path_crosslinks.items()
    ):
        return None, "native_energy_part2_metadata_artifact_path_mismatch"
    boundary = contract.get("boundary_contract")
    if (
        not isinstance(boundary, Mapping)
        or str(boundary.get("metadata_path") or "")
        != metadata_artifact["path"]
        or _strict_sha256(boundary.get("metadata_sha256"))
        != metadata_artifact["sha256"]
        or str(contract.get("engine") or "")
        != binding_artifacts["engine"]["path"]
        or _strict_sha256(contract.get("engine_sha256"))
        != binding_artifacts["engine"]["sha256"]
    ):
        return None, "native_energy_part2_command_artifact_duplicate_mismatch"
    return metadata, "verified_single_static_part2_input_technical_proof"


def seal_split_energy_preflight_attestation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a collector-compatible split-energy preflight attestation.

    This intentionally uses the same canonical JSON convention as the energy
    collector.  The preflight performs expensive file hashing before sampling;
    the measured workload verifies only this small seal, its nonce/expiry and
    lightweight file-stat identities.
    """
    attestation = dict(payload)
    attestation.pop("attestation_sha256", None)
    attestation["schema"] = SPLIT_ENERGY_PREFLIGHT_ATTESTATION_SCHEMA
    attestation["schema_version"] = 1
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    return attestation


def _strict_positive_ns(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except Exception:
        return 0
    return parsed if parsed > 0 else 0


def verify_split_energy_preflight_attestation(
    raw: Any,
    *,
    expected_nonce: str,
    expected_command_contract_sha256: str,
    expected_backend: str = "",
    max_age_s: float = 300.0,
    now_unix_ns: int | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify the small attestation without hashing any runtime artefact."""
    if not isinstance(raw, Mapping):
        return None, "split_energy_preflight_attestation_missing"
    data = dict(raw)
    declared = str(data.pop("attestation_sha256", "") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", declared) is None:
        return None, "split_energy_preflight_attestation_sha256_missing"
    if not secrets.compare_digest(declared, canonical_json_sha256(data)):
        return None, "split_energy_preflight_attestation_sha256_mismatch"
    data["attestation_sha256"] = declared
    if (
        data.get("schema") != SPLIT_ENERGY_PREFLIGHT_ATTESTATION_SCHEMA
        or int(data.get("schema_version") or 0) != 1
    ):
        return None, "split_energy_preflight_attestation_schema_invalid"
    if data.get("ok") is not True:
        return None, str(data.get("failure_reason") or "split_energy_preflight_not_ok")
    if str(data.get("artifact_verification_status") or "").lower() != "pass":
        return None, "split_energy_preflight_artifacts_not_verified"
    if not secrets.compare_digest(
        str(data.get("nonce") or ""), str(expected_nonce or "")
    ):
        return None, "split_energy_preflight_nonce_mismatch"
    wanted_contract = str(expected_command_contract_sha256 or "").strip().lower()
    actual_contract = str(data.get("command_contract_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", wanted_contract) is None:
        return None, "split_energy_expected_command_contract_sha256_invalid"
    if not secrets.compare_digest(actual_contract, wanted_contract):
        return None, "split_energy_preflight_command_contract_sha256_mismatch"

    created = _strict_positive_ns(data.get("created_at_unix_ns"))
    expires = _strict_positive_ns(data.get("expires_at_unix_ns"))
    now_ns = int(now_unix_ns if now_unix_ns is not None else time.time_ns())
    max_age_ns = max(1, int(float(max_age_s) * 1_000_000_000))
    if not created or not expires or expires <= created:
        return None, "split_energy_preflight_timestamps_invalid"
    if created > now_ns + 5_000_000_000:
        return None, "split_energy_preflight_created_in_future"
    if expires < now_ns:
        return None, "split_energy_preflight_expired"
    if now_ns - created > max_age_ns:
        return None, "split_energy_preflight_stale"

    binding = data.get("workload_binding")
    if not isinstance(binding, Mapping):
        return None, "split_energy_workload_binding_missing"
    binding_data = dict(binding)
    if (
        binding_data.get("schema") != SPLIT_ENERGY_WORKLOAD_BINDING_SCHEMA
        or int(binding_data.get("schema_version") or 0) != 1
    ):
        return None, "split_energy_workload_binding_schema_invalid"
    if binding_data.get("workload_supported") is not True:
        return None, str(
            binding_data.get("unsupported_reason")
            or "split_energy_workload_not_supported"
        )
    if not secrets.compare_digest(
        str(binding_data.get("command_contract_sha256") or "").lower(),
        wanted_contract,
    ):
        return None, "split_energy_workload_binding_contract_sha256_mismatch"
    backend = str(binding_data.get("backend") or "").strip().lower()
    wanted_backend = str(expected_backend or "").strip().lower()
    if wanted_backend and backend != wanted_backend:
        return None, "split_energy_workload_binding_backend_mismatch"
    return data, "nonce_seal_freshness_contract_and_binding_verified"


def verify_split_energy_artifact_stats(
    binding: Mapping[str, Any],
) -> tuple[bool, str]:
    """Detect ordinary post-preflight replacement without in-window hashing.

    SHA-256 remains authoritative and is checked by the preflight.  The
    workload performs only small ``stat`` calls immediately before execution
    to close the common accidental-change race without moving heavy I/O back
    into the collector window.
    """
    records = binding.get("verified_files")
    if not isinstance(records, list) or not records:
        return False, "split_energy_verified_file_stats_missing"

    def _required_stat_int(raw: Mapping[str, Any], field: str) -> int:
        if (
            field not in raw
            or raw.get(field) is None
            or isinstance(raw.get(field), bool)
        ):
            raise ValueError(field)
        return int(raw[field])

    for raw in records:
        if not isinstance(raw, Mapping):
            return False, "split_energy_verified_file_stat_invalid"
        path = Path(str(raw.get("path") or "")).expanduser()
        try:
            stat = path.stat()
        except Exception:
            return False, "split_energy_verified_file_missing_after_preflight"
        try:
            expected = (
                _required_stat_int(raw, "size_bytes"),
                _required_stat_int(raw, "mtime_ns"),
                _required_stat_int(raw, "device"),
                _required_stat_int(raw, "inode"),
            )
        except (TypeError, ValueError, OverflowError):
            return False, "split_energy_verified_file_stat_invalid"
        actual = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_dev),
            int(stat.st_ino),
        )
        if actual != expected:
            return False, "split_energy_verified_file_changed_after_preflight"
    return True, "preflight_file_stats_unchanged"


def load_split_energy_workload_binding(
    attestation_path: str | Path,
    *,
    expected_nonce: str,
    expected_command_contract_sha256: str,
    expected_backend: str,
    max_age_s: float = 300.0,
) -> tuple[dict[str, Any] | None, str]:
    """Load a workload binding using only a small JSON read and file stats."""
    try:
        raw = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"split_energy_preflight_attestation_read_failed:{type(exc).__name__}"
    attestation, reason = verify_split_energy_preflight_attestation(
        raw,
        expected_nonce=expected_nonce,
        expected_command_contract_sha256=expected_command_contract_sha256,
        expected_backend=expected_backend,
        max_age_s=max_age_s,
    )
    if attestation is None:
        return None, reason
    binding = dict(attestation.get("workload_binding") or {})
    stats_ok, stats_reason = verify_split_energy_artifact_stats(binding)
    if not stats_ok:
        return None, stats_reason
    return binding, reason + ";" + stats_reason


def successful_runtime_argv(
    contract: Mapping[str, Any],
    *,
    duration_s: float,
    fresh_output_root: str,
    remote_tool_dir: str,
) -> list[str]:
    """Build a Native split replay from the archived successful contract.

    This helper intentionally supports only execution contracts with known
    duration-based runners.  Unknown backends fail instead of inheriting global
    defaults.  Full-baseline contracts are built by their runner-specific
    producer but are verified through :func:`verify_native_command_contract`.
    """
    backend = str(contract.get("backend") or "").strip().lower()
    options = dict(contract.get("runtime_options") or {})
    boundary = dict(contract.get("boundary_contract") or {})
    artifacts = dict(contract.get("artifacts") or {})
    python_executable = str(contract["python_executable"])
    runner = f"{str(remote_tool_dir).rstrip('/')}/{str(contract['runner']).lstrip('/')}"
    common = [
        python_executable, "-u", runner,
        "--benchmark-set", str(contract["benchmark_set"]),
        "--case", str(contract["case"]),
        "--precision", str(contract["precision"]),
        "--image", str(contract["input_image"]),
        "--duration-s", str(float(duration_s)),
        "--warmup", str(int(options["warmup"])),
        "--queue-depth", str(int(options["queue_depth"])),
    ]
    if backend == "hailo8_to_trt":
        hef = dict(artifacts.get("hef") or {})
        engine = dict(artifacts.get("engine") or {})
        prepared_contract = contract.get("prepared_input_contract")
        if not isinstance(prepared_contract, Mapping) or not hailo8_preprocess_binding(
            options, prepared_contract,
        ):
            raise ValueError("hailo8 contract does not bind consistent task preprocessing semantics")
        argv = common + [
            "--hw-arch", str(contract.get("hw_arch") or "hailo8"),
            "--hailo-format", str(options["hailo_format"]),
            "--letterbox-pad-value", str(int(options["letterbox_pad_value_requested"])),
            "--result-json", f"{fresh_output_root.rstrip('/')}/native_fifo_results.json",
            "--config-json", f"{fresh_output_root.rstrip('/')}/native_fifo_config.json",
            "--output-dir", f"{fresh_output_root.rstrip('/')}/native_outputs",
            "--boundary-dir", f"{fresh_output_root.rstrip('/')}/native_fifo_boundary",
            "--expected-runner-sha256", str(contract["runner_sha256"]),
            "--expected-image-sha256", str(contract["input_image_sha256"]),
            "--expected-hef-sha256", str(hef.get("sha256") or ""),
            "--expected-engine-sha256", str(engine.get("sha256") or ""),
            "--expected-executable-sha256", str(
                (artifacts.get("native_executable") or {}).get("sha256") or ""
            ),
            "--expected-boundary-layout", str(boundary["boundary_layout_effective"]),
            "--source-contract-sha256", str(contract["contract_sha256"]),
            "--no-build",
        ]
        if str(options.get("task") or "") in {"classification", "detection"}:
            argv += ["--task", str(options["task"])]
        if str(options.get("preprocess_mode_requested") or "") in {
            "auto", "resize", "letterbox",
        }:
            argv += ["--preprocess-mode", str(options["preprocess_mode_requested"])]
        if options.get("copy_outputs") is False:
            argv.append("--no-copy-outputs")
        if options.get("dump_outputs") is True:
            argv.append("--dump-outputs")
        if options.get("dump_boundary") is True:
            argv.append("--dump-boundary")
        if str(options.get("device_id") or "").strip():
            argv += ["--device-id", str(options["device_id"])]
        return argv
    if backend == "hailo10h_to_trt":
        hef = dict(artifacts.get("hef") or {})
        engine = dict(artifacts.get("engine") or {})
        prepared_contract = contract.get("prepared_input_contract")
        if not isinstance(prepared_contract, Mapping) or not hailo10_preprocess_binding(
            options, prepared_contract,
        ):
            raise ValueError("hailo10 contract does not bind consistent task preprocessing semantics")
        argv = common + [
            "--hw-arch", str(contract.get("hw_arch") or "hailo10h"),
            "--inflight", str(int(options["inflight"])),
            "--producer-impl", str(options["producer_impl"]),
            "--boundary-layout", str(boundary["boundary_layout_effective"]),
            "--out-dir", str(fresh_output_root),
            "--expected-runner-sha256", str(contract["runner_sha256"]),
            "--expected-image-sha256", str(contract["input_image_sha256"]),
            "--expected-hef-sha256", str(hef.get("sha256") or ""),
            "--expected-engine-sha256", str(engine.get("sha256") or ""),
            "--source-contract-sha256", str(contract["contract_sha256"]),
            "--task", str(options["task"]),
            "--preprocess-mode", str(options["preprocess_mode_requested"]),
            "--letterbox-pad-value", str(int(options["letterbox_pad_value_requested"])),
        ]
        if boundary.get("dequant_scale") not in (None, ""):
            argv += ["--dequant-scale", str(boundary["dequant_scale"])]
        if boundary.get("dequant_zero_point") not in (None, ""):
            argv += ["--dequant-zero-point", str(boundary["dequant_zero_point"])]
        argv.append("--quantized-inputs" if options.get("quantized_inputs") else "--no-quantized-inputs")
        argv.append("--quantized-outputs" if options.get("quantized_outputs") else "--no-quantized-outputs")
        argv.append("--copy-outputs" if options.get("copy_outputs") else "--no-copy-outputs")
        if options.get("dump_outputs") is True:
            argv.append("--dump-outputs")
        if options.get("dump_boundary") is True:
            argv.append("--dump-boundary")
        return argv
    if backend == "deepx_to_trt":
        dxnn = dict(artifacts.get("dxnn") or {})
        engine = dict(artifacts.get("engine") or {})
        argv = common + [
            "--boundary-layout", str(boundary["boundary_layout_effective"]),
            "--out-dir", str(fresh_output_root),
            "--expected-runner-sha256", str(contract["runner_sha256"]),
            "--expected-image-sha256", str(contract["input_image_sha256"]),
            "--expected-dxnn-sha256", str(dxnn.get("sha256") or ""),
            "--expected-engine-sha256", str(engine.get("sha256") or ""),
            "--source-contract-sha256", str(contract["contract_sha256"]),
        ]
        prepared_contract = contract.get("prepared_input_contract")
        valid_preprocess, task, requested_mode, _effective_mode, requested_pad, _effective_pad = (
            deepx_preprocess_binding(
                options,
                prepared_contract if isinstance(prepared_contract, Mapping) else {},
            )
        )
        if not valid_preprocess:
            raise ValueError("deepx contract does not bind consistent task preprocessing semantics")
        argv += [
            "--task", task,
            "--preprocess-mode", requested_mode,
            "--letterbox-pad-value", str(requested_pad),
        ]
        if options.get("dump_outputs") is True:
            argv.append("--dump-outputs")
        if options.get("dump_boundary") is True:
            argv.append("--dump-boundary")
        return argv
    raise ValueError(f"unsupported native replay backend: {backend or '<missing>'}")


def split_energy_runtime_argv(
    contract: Mapping[str, Any],
    *,
    duration_s: float,
    fresh_output_root: str,
    remote_tool_dir: str,
    preflight_attestation_path: str,
    preflight_nonce: str = "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__",
) -> list[str]:
    """Create a hash-free measured command for a preflight-attested split.

    Unlike :func:`successful_runtime_argv`, this command is only valid after a
    successful ``native_split_energy_preflight.py`` invocation.  It carries no
    expected-file hashes because those are intentionally evaluated before the
    collector starts.  Warmup, builds and dumps are forced off for every
    backend, while queue/boundary/input/runtime choices remain contract-bound.
    """
    backend = str(contract.get("backend") or "").strip().lower()
    options = dict(contract.get("runtime_options") or {})
    boundary = dict(contract.get("boundary_contract") or {})
    artifacts = dict(contract.get("artifacts") or {})
    python_executable = str(contract.get("python_executable") or "")
    runner = f"{str(remote_tool_dir).rstrip('/')}/{str(contract.get('runner') or '').lstrip('/')}"
    contract_sha = str(contract.get("contract_sha256") or "")
    if float(duration_s) <= 0.0:
        raise ValueError("split energy requires duration_s > 0")
    common = [
        python_executable, "-u", runner,
        "--benchmark-set", str(contract.get("benchmark_set") or ""),
        "--case", str(contract.get("case") or ""),
        "--precision", str(contract.get("precision") or ""),
        "--image", str(contract.get("input_image") or ""),
        "--frames", str(max(1, int(options.get("frames") or 1))),
        "--duration-s", str(float(duration_s)),
        "--warmup", "0",
        "--queue-depth", str(int(options.get("queue_depth") or 1)),
        "--source-contract-sha256", contract_sha,
        "--energy-workload-only",
        "--energy-preflight-attestation", str(preflight_attestation_path),
        "--energy-preflight-nonce", str(preflight_nonce),
    ]
    if backend == "hailo8_to_trt":
        if not isinstance(artifacts.get("native_executable"), Mapping):
            raise ValueError("hailo8 split energy requires a bound native executable")
        if options.get("energy_prepared_feed_capable") is not True:
            raise ValueError("hailo8 legacy executable lacks prepared-feed energy mode")
        prepared_contract = contract.get("prepared_input_contract")
        if (
            options.get("prepared_input_bound") is not True
            or not isinstance(artifacts.get("prepared_input"), Mapping)
            or not isinstance(prepared_contract, Mapping)
            or not list(prepared_contract.get("shape") or [])
        ):
            raise ValueError("hailo8 legacy contract has no bound prepared input")
        task = str(options.get("task") or "")
        requested_mode = str(options.get("preprocess_mode_requested") or "")
        if not hailo8_preprocess_binding(options, prepared_contract):
            raise ValueError("hailo8 contract does not bind consistent task preprocessing semantics")
        argv = common + [
            "--hw-arch", str(contract.get("hw_arch") or "hailo8"),
            "--hailo-format", str(options.get("hailo_format") or "uint8"),
            "--task", task,
            "--preprocess-mode", requested_mode,
            "--letterbox-pad-value", str(int(options.get("letterbox_pad_value_requested"))),
            "--result-json", f"{fresh_output_root.rstrip('/')}/native_fifo_results.json",
            "--no-build",
        ]
        argv.append("--copy-outputs" if options.get("copy_outputs") else "--no-copy-outputs")
        if str(options.get("device_id") or "").strip():
            argv += ["--device-id", str(options.get("device_id"))]
        return argv
    if backend == "hailo10h_to_trt":
        if str(options.get("producer_impl") or "").strip().lower() not in {"auto", "async_fifo"}:
            raise ValueError("hailo10 split energy requires a probe-free async_fifo contract")
        if "part1_onnx_used" not in options:
            raise ValueError("hailo10 legacy contract does not bind part1_onnx usage")
        if options.get("part1_onnx_used") is True and not isinstance(
            artifacts.get("part1_onnx"), Mapping
        ):
            raise ValueError("hailo10 contract used part1_onnx but did not bind the artefact")
        if not list(options.get("canonical_input_slot_names") or []):
            raise ValueError("hailo10 contract does not bind canonical input slots")
        if not list(options.get("canonical_output_slot_names") or []):
            raise ValueError("hailo10 contract does not bind canonical output slots")
        prepared_contract = contract.get("prepared_input_contract")
        prepared_entries = (
            list(prepared_contract.get("entries") or [])
            if isinstance(prepared_contract, Mapping) else []
        )
        if options.get("prepared_input_bound") is not True or not prepared_entries:
            raise ValueError("hailo10 legacy contract has no bound prepared input")
        if not isinstance(prepared_contract, Mapping) or not hailo10_preprocess_binding(
            options, prepared_contract,
        ):
            raise ValueError("hailo10 contract does not bind consistent task preprocessing semantics")
        for entry in prepared_entries:
            artifact_name = str(entry.get("artifact_name") or "") if isinstance(entry, Mapping) else ""
            if not artifact_name or not isinstance(artifacts.get(artifact_name), Mapping):
                raise ValueError("hailo10 prepared input artifact is missing")
        argv = common + [
            "--hw-arch", str(contract.get("hw_arch") or "hailo10h"),
            "--inflight", str(int(options.get("inflight") or 1)),
            "--producer-impl", str(options.get("producer_impl") or "auto"),
            "--boundary-layout", str(boundary.get("boundary_layout_effective") or "as_input"),
            "--out-dir", str(fresh_output_root),
            "--task", str(options.get("task") or ""),
            "--preprocess-mode", str(options.get("preprocess_mode_requested") or ""),
            "--letterbox-pad-value", str(int(options.get("letterbox_pad_value_requested"))),
        ]
        if boundary.get("dequant_scale") not in (None, ""):
            argv += ["--dequant-scale", str(boundary.get("dequant_scale"))]
        if boundary.get("dequant_zero_point") not in (None, ""):
            argv += ["--dequant-zero-point", str(boundary.get("dequant_zero_point"))]
        argv.append("--quantized-inputs" if options.get("quantized_inputs") else "--no-quantized-inputs")
        argv.append("--quantized-outputs" if options.get("quantized_outputs") else "--no-quantized-outputs")
        argv.append("--copy-outputs" if options.get("copy_outputs") else "--no-copy-outputs")
        return argv
    if backend == "deepx_to_trt":
        prepared_contract = contract.get("prepared_input_contract")
        if not isinstance(artifacts.get("prepared_input"), Mapping):
            raise ValueError("deepx legacy contract has no bound prepared input artefact")
        if (
            not isinstance(prepared_contract, Mapping)
            or not list(prepared_contract.get("shape") or [])
            or not str(prepared_contract.get("dtype") or "")
            or options.get("prepared_input_bound") is not True
        ):
            raise ValueError("deepx prepared input contract is incomplete")
        valid_preprocess, task, requested_mode, _effective_mode, requested_pad, _effective_pad = (
            deepx_preprocess_binding(options, prepared_contract)
        )
        if not valid_preprocess:
            raise ValueError("deepx contract does not bind consistent task preprocessing semantics")
        return common + [
            "--boundary-layout", str(boundary.get("boundary_layout_effective") or "as_input"),
            "--out-dir", str(fresh_output_root),
            "--task", task,
            "--preprocess-mode", requested_mode,
            "--letterbox-pad-value", str(requested_pad),
        ]
    raise ValueError(f"unsupported native split-energy backend: {backend or '<missing>'}")
