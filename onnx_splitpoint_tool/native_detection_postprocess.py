"""Frozen Native-Full detection host tail.

Vendor accelerators may expose YOLO raw heads even for a Full-model artifact.
Those tensors are not a completed detection task until decode and class-aware
NMS have run.  This module turns the existing :class:`YoloHarness` decoder into
an explicit, hash-bound runtime contract that can be reused by both performance
and energy hotloops.

The contract deliberately describes code and tensor *structure*, not output
values.  A different validation image may therefore reuse the same frozen
decoder, while a changed tensor layout, threshold, implementation, or model
family fails closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ == "splitpoint_runners":
    # Generated remote suites vendor this module beside ``harness`` and must
    # not depend on an installed copy of the main tool package.
    from .harness.base import postprocess_result_to_dict
    from .harness import base as _base_module
    from .harness.yolo import YoloHarness
    from .harness import yolo as _yolo_module
else:
    from .runners.harness.base import postprocess_result_to_dict
    from .runners.harness import base as _base_module
    from .runners.harness.yolo import YoloHarness
    from .runners.harness import yolo as _yolo_module


SCHEMA = "onnx-splitpoint/frozen-native-detection-postprocess"
SCHEMA_VERSION = 2
SCHEMA_LEGACY_VERSION = 1
EXECUTION_POLICY = "one_serial_host_tail_per_successful_accelerator_output"
COMPLETED_ENDPOINT_SCHEMA = (
    "onnx-splitpoint/frozen-native-detection-completed-endpoint"
)
COMPLETED_ENDPOINT_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/frozen-native-detection-completion-attestation"
)
COMPLETED_ENDPOINT_VERSION = 2
COMPLETED_ENDPOINT_LEGACY_VERSION = 1
COMPLETED_ENDPOINT_SOURCE = "frozen_host_decode_nms_inside_measured_interval:v1"
COMPLETED_COMPARISON_ENDPOINT_SCHEMA = (
    "onnx-splitpoint/completed-task-comparison-endpoint"
)
COMPLETED_COMPARISON_ENDPOINT_VERSION = 2
COMPLETED_COMPARISON_ENDPOINT_V1 = 1
DIRECT_NORMALIZATION_SCHEMA = (
    "onnx-splitpoint/frozen-native-decoded-nms-normalization"
)
DIRECT_NORMALIZATION_VERSION = 1
DIRECT_NORMALIZATION_EXECUTION_POLICY = (
    "one_serial_canonical_normalization_per_successful_accelerator_output"
)
DIRECT_NORMALIZED_ENDPOINT_SCHEMA = (
    "onnx-splitpoint/integrated-native-detection-normalized-endpoint"
)
DIRECT_NORMALIZED_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/integrated-native-detection-normalization-attestation"
)
DIRECT_NORMALIZED_ENDPOINT_VERSION = 1
DIRECT_NORMALIZED_ENDPOINT_SOURCE = (
    "integrated_accelerator_plus_frozen_normalization_inside_measured_interval:v1"
)
CANONICAL_COMPLETION_POLICY_ID = (
    "decoded_nms_xyxy_original_classaware_postfilter_v2"
)
YOLOV7_HEAD_MAPPING_SCHEMA = "onnx-splitpoint/yolov7-stride-head-mapping"
YOLOV7_HEAD_MAPPING_VERSION = 2
YOLOV7_HEAD_MAPPING_LEGACY_VERSION = 1
DECODED_NMS_MATERIALIZATION_SCHEMA = (
    "onnx-splitpoint/attested-decoded-nms-materialization"
)
DECODED_NMS_MATERIALIZATION_VERSION = 1
COMPLETION_EXECUTION_SCHEMA = (
    "onnx-splitpoint/detection-completion-execution-contract"
)
COMPLETION_EXECUTION_VERSION = 2
COMPLETION_EXECUTION_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/detection-completion-execution-attestation"
)
COMPLETION_EXECUTION_ATTESTATION_VERSION = 2
COMPLETION_RESULT_ARTIFACT_SCHEMA = (
    "onnx-splitpoint/detection-completion-result-artifact"
)
COMPLETION_RESULT_ARTIFACT_VERSION = 1
COMPLETION_RESULT_ARTIFACT_ENCODING = (
    "canonical_json_utf8_sort_keys_compact_no_nan_v1"
)
COMPLETION_OBSERVATION_RELATIONS = (
    "same_hotloop_sentinel",
    "independent_replay",
)
COMPLETION_HASH_LAYER_SCHEMA = (
    "onnx-splitpoint/detection-completion-hash-layer"
)
COMPLETION_HASH_LAYER_VERSION = 1
CANONICAL_DETECTION_RECORD_SCHEMA = "xyxy_score_class_id_v1"
CANONICAL_DETECTION_SORT_POLICY = (
    "score_desc_class_id_asc_xyxy_lexicographic_v1"
)
FROZEN_COMPLETED_RESULT_ARTIFACT_SCHEMA = (
    "onnx-splitpoint/frozen-completed-detection-result-artifact"
)
FROZEN_COMPLETED_RESULT_ARTIFACT_VERSION = 1
NO_SECOND_NMS_NORMALIZER_ID = (
    "attested_bn6_filter_padding_inverse_letterbox_materialize_no_nms_v1"
)
YOLOV7_EXPECTED_STRIDES = (8, 16, 32)
YOLOV7_ANCHORS_BY_STRIDE: dict[int, tuple[tuple[int, int], ...]] = {
    8: ((12, 16), (19, 36), (40, 28)),
    16: ((36, 75), (76, 55), (72, 146)),
    32: ((142, 110), (192, 243), (459, 401)),
}
YOLOV7_LEGACY_TINY_ANCHORS_BY_STRIDE: dict[
    int, tuple[tuple[int, int], ...]
] = {
    8: ((10, 13), (16, 30), (33, 23)),
    16: ((30, 61), (62, 45), (59, 119)),
    32: ((116, 90), (156, 198), (373, 326)),
}
YOLOV7_STANDARD_ANCHOR_TABLE_ID = str(
    _yolo_module.YOLOV7_STANDARD_ANCHOR_TABLE_ID
)
YOLOV7_PAPER_DECODER_ID = str(_yolo_module.YOLOV7_PAPER_DECODER_ID)
_EMPTY_INVOCATION_CHAIN_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "schema": "onnx-splitpoint/detection-completion-invocation-chain",
            "schema_version": 1,
            "state": "empty",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class FrozenPostprocessError(RuntimeError):
    """Raised when frozen host-tail evidence is incomplete or inconsistent."""

    def __init__(self, *args: Any, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(*args)
        # Keep the scalar error code unchanged for existing callers and logs.
        # Numeric observations belong in the existing result JSON, not in a
        # new artifact identity or in an unbounded exception string.
        self.diagnostics = dict(diagnostics or {})


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one contract value to the exact canonical transport bytes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute_path(value: str | Path) -> Path:
    """Return an absolute path without following any filesystem symlink."""
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _existing_path_kind(path: Path) -> int | None:
    try:
        return path.lstat().st_mode
    except FileNotFoundError:
        return None


def _require_safe_output_path(path: Path) -> None:
    """Create/check the parent without ever accepting symlink components."""
    absolute = _lexical_absolute_path(path)
    parent_chain = [
        candidate
        for candidate in reversed(absolute.parent.parents)
        if candidate != candidate.parent
    ]
    parent_chain.append(absolute.parent)
    for parent in parent_chain:
        mode = _existing_path_kind(parent)
        if mode is None:
            try:
                parent.mkdir()
            except FileExistsError:
                pass
            mode = _existing_path_kind(parent)
        if mode is None or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise FrozenPostprocessError(
                "completed_result_artifact_output_path_unsafe"
            )
    leaf_mode = _existing_path_kind(absolute)
    if leaf_mode is not None and (
        stat.S_ISLNK(leaf_mode) or not stat.S_ISREG(leaf_mode)
    ):
        raise FrozenPostprocessError(
            "completed_result_artifact_output_path_unsafe"
        )


def persist_completed_result_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Persist and re-read one canonical same-hotloop result artifact."""
    if not isinstance(artifact, Mapping) or not artifact:
        raise FrozenPostprocessError(
            "completed_result_artifact_missing"
        )
    canonical = canonical_json_bytes(dict(artifact))
    actual_sha256 = hashlib.sha256(canonical).hexdigest()
    expected = _strict_sha256(
        expected_sha256,
        reason="completed_result_artifact_sha256_invalid",
    )
    if actual_sha256 != expected:
        raise FrozenPostprocessError(
            "completed_result_artifact_sha256_mismatch"
        )
    path = _lexical_absolute_path(output_path)
    _require_safe_output_path(path)
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_text)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_mode = _existing_path_kind(temporary)
        if temporary_mode is None or not stat.S_ISREG(temporary_mode):
            raise FrozenPostprocessError(
                "completed_result_artifact_persistence_invalid"
            )
        # Recheck immediately before the atomic replacement.  In particular,
        # never resolve or write through a pre-existing destination symlink.
        _require_safe_output_path(path)
        os.replace(temporary, path)
        temporary = None
        _require_safe_output_path(path)
        persisted = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if isinstance(exc, FrozenPostprocessError):
            raise
        raise FrozenPostprocessError(
            "completed_result_artifact_persistence_invalid"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if (
        file_sha256(path) != actual_sha256
        or persisted != dict(artifact)
    ):
        raise FrozenPostprocessError(
            "completed_result_artifact_persistence_invalid"
        )
    return {
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": dict(artifact),
        "completed_task_result_artifact_sha256": actual_sha256,
        "completed_task_result_artifact_path": str(path),
        "completed_task_result_artifact_file_sha256": actual_sha256,
    }


def persist_detection_completion_execution_artifact(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Persist the exact artifact sealed by one execution attestation."""
    if not isinstance(payload, Mapping):
        raise FrozenPostprocessError(
            "completion_execution_payload_invalid"
        )
    raw_attestation = payload.get("completion_execution_attestation")
    if not isinstance(raw_attestation, Mapping):
        raw_attestation = payload.get(
            "completed_task_endpoint_attestation"
        )
    if not isinstance(raw_attestation, Mapping):
        raise FrozenPostprocessError(
            "completion_execution_attestation_missing"
        )
    attestation = dict(raw_attestation)
    endpoint_attestation = payload.get(
        "completed_task_endpoint_attestation"
    )
    if (
        isinstance(endpoint_attestation, Mapping)
        and dict(endpoint_attestation) != attestation
    ):
        raise FrozenPostprocessError(
            "completion_execution_attestation_projection_mismatch"
        )
    artifact = attestation.get("artifact")
    last_result = attestation.get("last_result")
    artifact_sha256 = str(
        attestation.get("artifact_sha256") or ""
    )
    if (
        not isinstance(artifact, Mapping)
        or not isinstance(last_result, Mapping)
        or dict(last_result.get("artifact") or {}) != dict(artifact)
        or str(last_result.get("artifact_sha256") or "").strip().lower()
        != artifact_sha256.strip().lower()
    ):
        raise FrozenPostprocessError(
            "completion_execution_result_artifact_invalid"
        )
    return persist_completed_result_artifact(
        artifact,
        expected_sha256=artifact_sha256,
        output_path=output_path,
    )


def persist_detection_completion_execution_artifacts(
    payload: dict[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Persist aggregate and per-repetition execution artifacts."""
    if not isinstance(payload, dict):
        raise FrozenPostprocessError(
            "completion_execution_payload_invalid"
        )
    path = _lexical_absolute_path(output_path)
    evidence = persist_detection_completion_execution_artifact(
        payload,
        output_path=path,
    )
    payload.update(evidence)
    seen_record_lists: set[int] = set()
    for records_key in ("repetition_records", "repetition_evidence"):
        records = payload.get(records_key)
        if records is None:
            continue
        if not isinstance(records, list):
            raise FrozenPostprocessError(
                "completion_execution_repetitions_invalid"
            )
        records_identity = id(records)
        if records_identity in seen_record_lists:
            continue
        seen_record_lists.add(records_identity)
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise FrozenPostprocessError(
                    "completion_execution_repetition_invalid"
                )
            record.update(
                persist_detection_completion_execution_artifact(
                    record,
                    output_path=path.with_name(
                        f"{path.stem}.{records_key}_{index:03d}"
                        f"{path.suffix}"
                    ),
                )
            )
    return evidence


def _strict_sha256(value: Any, *, reason: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    if (
        len(token) != 64
        or any(char not in "0123456789abcdef" for char in token)
    ):
        raise FrozenPostprocessError(reason)
    return token


def _strict_positive_int_list(
    value: Any, *, length: int, reason: str,
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in value
        )
    ):
        raise FrozenPostprocessError(reason)
    return list(value)


def _verify_tensor_signature_structure(
    raw: Any, *, reason: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(reason)
    signature = dict(raw)
    tensors = signature.get("tensors")
    count = signature.get("tensor_count")
    if (
        set(signature) != {"tensor_count", "tensors"}
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(tensors, list)
        or len(tensors) != count
    ):
        raise FrozenPostprocessError(reason)
    names: set[str] = set()
    for expected_index, raw_row in enumerate(tensors):
        if not isinstance(raw_row, Mapping):
            raise FrozenPostprocessError(reason)
        row = dict(raw_row)
        name = row.get("name")
        rank = row.get("rank")
        shape = row.get("shape")
        dtype = row.get("dtype")
        if (
            set(row) != {"index", "name", "rank", "shape", "dtype"}
            or isinstance(row.get("index"), bool)
            or row.get("index") != expected_index
            or not isinstance(name, str)
            or not name
            or name in names
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or not isinstance(shape, list)
            or len(shape) != rank
            or any(
                isinstance(dim, bool)
                or not isinstance(dim, int)
                or dim <= 0
                for dim in shape
            )
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise FrozenPostprocessError(reason)
        names.add(name)
    return signature


def _model_family(model_id: str) -> str:
    token = str(model_id or "").strip().lower().replace("-", "").replace("_", "")
    if "yolo26" in token:
        return "yolo26"
    if "yolo11" in token:
        return "yolo11"
    if "yolov7" in token or "yolo7" in token:
        return "yolov7"
    raise FrozenPostprocessError(f"unsupported_native_full_raw_detection_model:{model_id}")


def _expected_decoder(
    model_id: str, *, source_contract_family: str = "raw_head",
) -> tuple[str, str]:
    family = _model_family(model_id)
    if source_contract_family == "decoded_pre_nms" and family in {"yolo11", "yolo26"}:
        return "ultralytics_decoded_classaware_nms_v1", "ultralytics_decoded"
    if source_contract_family != "raw_head":
        raise FrozenPostprocessError("frozen_postprocess_source_family_invalid")
    if family == "yolo26":
        return "yolo26_regcls_ltrb_classaware_nms_v1", "ultralytics_regcls"
    if family == "yolo11":
        return "yolo11_regcls_dfl16_classaware_nms_v1", "ultralytics_regcls"
    if family == "yolov7":
        return YOLOV7_PAPER_DECODER_ID, "multiscale_head"
    raise FrozenPostprocessError(
        f"unsupported_native_full_raw_detection_model:{model_id}"
    )


def _verify_decoded_pre_nms_signature(
    signature: Mapping[str, Any], *, input_hw: Sequence[int],
) -> None:
    """Bind the existing decoded COCO adapter to its actual tensor geometry."""
    verified = _verify_tensor_signature_structure(
        signature, reason="decoded_pre_nms_tensor_signature_invalid",
    )
    height, width = [int(value) for value in input_hw]
    if height % 32 or width % 32:
        raise FrozenPostprocessError("decoded_pre_nms_input_geometry_invalid")
    anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
    tensors = verified["tensors"]
    if len(tensors) != 1:
        raise FrozenPostprocessError("decoded_pre_nms_exact_one_tensor_required")
    shape = tuple(tensors[0]["shape"])
    if shape not in {(1, 84, anchors), (1, anchors, 84)}:
        raise FrozenPostprocessError("decoded_pre_nms_tensor_geometry_invalid")
    if not np.issubdtype(np.dtype(tensors[0]["dtype"]), np.floating):
        raise FrozenPostprocessError("decoded_pre_nms_float_tensor_required")


def inspect_decoded_pre_nms_values(
    outputs: Mapping[str, Any], *, sample_limit: int = 8,
) -> dict[str, Any]:
    """Return bounded JSON-safe observations; never transform runtime values.

    This deliberately does not attest geometry or declare model semantics.
    Call it only on a failed check or for an explicit untimed diagnostic probe.
    Examples use the original tensor indices and canonical channel/anchor
    indices so both [1,84,N] and [1,N,84] layouts remain reviewable.
    """
    limit = min(16, max(0, int(sample_limit)))

    def scalar(value: Any) -> float | str:
        number = float(value)
        if np.isnan(number):
            return "NaN"
        if np.isposinf(number):
            return "+Inf"
        if np.isneginf(number):
            return "-Inf"
        return number

    def summary(values: Any) -> dict[str, Any]:
        values = np.asarray(values)
        finite = np.isfinite(values)
        finite_values = values[finite]
        return {
            "element_count": int(values.size),
            "finite_count": int(np.count_nonzero(finite)),
            "all_finite": bool(np.all(finite)),
            "nan_count": int(np.count_nonzero(np.isnan(values))),
            "positive_inf_count": int(np.count_nonzero(np.isposinf(values))),
            "negative_inf_count": int(np.count_nonzero(np.isneginf(values))),
            "finite_min": float(np.min(finite_values)) if finite_values.size else None,
            "finite_max": float(np.max(finite_values)) if finite_values.size else None,
        }

    tensors: list[dict[str, Any]] = []
    for name, raw in outputs.items():
        array = np.asarray(raw)
        row: dict[str, Any] = {
            "name": str(name), "shape": [int(size) for size in array.shape],
            "dtype": str(array.dtype),
        }
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            row["inspection_status"] = "unsupported_dtype"
            tensors.append(row)
            continue
        row.update(summary(array))
        values = array[0] if array.ndim == 3 and array.shape[0] == 1 else array
        has_batch = array.ndim == 3 and array.shape[0] == 1
        if values.ndim != 2 or 84 not in values.shape:
            row["inspection_status"] = "decoded_geometry_unrecognized"
            tensors.append(row)
            continue
        channels_first = values.shape[0] == 84
        values = values if channels_first else values.T
        row["inspection_status"] = "inspected"
        row["canonical_shape_channels_anchors"] = [int(size) for size in values.shape]
        row["channel_axis"] = int(has_batch) + (0 if channels_first else 1)
        row["xywh"] = {
            name: summary(values[index])
            for index, name in enumerate(("x", "y", "width", "height"))
        }
        row["class_scores"] = summary(values[4:])

        def examples(mask: Any, *, channel_offset: int = 0) -> dict[str, Any]:
            mask = np.asarray(mask)
            samples = []
            for flat in np.flatnonzero(mask)[:limit]:
                local_channel, anchor = np.unravel_index(int(flat), mask.shape)
                channel = int(local_channel) + channel_offset
                anchor = int(anchor)
                index = ([0] if has_batch else []) + (
                    [channel, anchor] if channels_first else [anchor, channel]
                )
                sample = {
                    "index": index, "channel": channel, "anchor": anchor,
                    "value": scalar(values[channel, anchor]),
                }
                if channel >= 4:
                    sample["class_id"] = channel - 4
                samples.append(sample)
            return {"count": int(np.count_nonzero(mask)), "examples": samples}

        row["violations"] = {
            "negative_width": examples(values[2:3] < 0, channel_offset=2),
            "negative_height": examples(values[3:4] < 0, channel_offset=3),
            "scores_below_zero": examples(values[4:] < 0, channel_offset=4),
            "scores_above_one": examples(values[4:] > 1, channel_offset=4),
            "nan": examples(np.isnan(values)),
            "positive_inf": examples(np.isposinf(values)),
            "negative_inf": examples(np.isneginf(values)),
        }
        tensors.append(row)
    return {
        "schema": "onnx-splitpoint/decoded-pre-nms-value-diagnostics",
        "schema_version": 1, "sample_limit_per_violation": limit,
        "tensor_count": len(tensors), "tensors": tensors,
        "values_modified": False,
        "range_rule": "finite; width>=0; height>=0; 0<=class_scores<=1",
        "geometry_and_semantics_attested_by_this_diagnostic": False,
    }


DECODED_PRE_NMS_SCORE_POLICY_ID = "float32_probability_edges_abs_2pow_minus23_v1"
DECODED_PRE_NMS_SCORE_EPSILON = 2.0 ** -23


def decoded_pre_nms_score_policy() -> dict[str, Any]:
    """Fixed numerical rule, embedded in the existing frozen contract.

    This is not a configurable accuracy margin. Only declared Float32 COCO
    probabilities may use it; raw logits, BN6 records and other dtypes cannot.
    """
    return {
        "policy_id": DECODED_PRE_NMS_SCORE_POLICY_ID,
        "dtype": "float32",
        "absolute_tolerance": DECODED_PRE_NMS_SCORE_EPSILON,
        "action": "normalize_probability_edges_on_processing_copy",
    }


def _checked_decoded_pre_nms_values(
    outputs: Mapping[str, Any], *, score_policy: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Structural/semantic authority remains the caller's frozen contract.
    # Validate in the original dtype; converting to Float32 first could hide
    # an out-of-range Float64 input. Python-float extrema avoid scalar-cast
    # rounding at the acceptance boundary (notably for Float16).
    if not isinstance(outputs, Mapping) or len(outputs) != 1:
        raise FrozenPostprocessError("decoded_pre_nms_exact_one_tensor_required")
    array = np.asarray(next(iter(outputs.values())))
    if (array.ndim != 3 or array.shape[0] != 1 or 84 not in array.shape[1:]
            or array.size == 0):
        raise FrozenPostprocessError("decoded_pre_nms_tensor_geometry_invalid")
    if not np.issubdtype(array.dtype, np.floating):
        raise FrozenPostprocessError("decoded_pre_nms_float_tensor_required")
    values = array[0] if array.shape[1] == 84 else array[0].T
    tolerance = 0.0
    if score_policy is not None:
        if (not isinstance(score_policy, Mapping)
                or dict(score_policy) != decoded_pre_nms_score_policy()
                or str(array.dtype) != "float32"):
            raise FrozenPostprocessError("decoded_pre_nms_score_policy_invalid")
        tolerance = DECODED_PRE_NMS_SCORE_EPSILON
    if not np.all(np.isfinite(values)):
        raise FrozenPostprocessError(
            "decoded_pre_nms_nonfinite_values",
            diagnostics=inspect_decoded_pre_nms_values(outputs),
        )
    scores = values[4:]
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if (np.any(values[2:4] < 0) or minimum < -tolerance
            or maximum > 1.0 + tolerance):
        raise FrozenPostprocessError(
            "decoded_pre_nms_values_invalid",
            diagnostics=inspect_decoded_pre_nms_values(outputs),
        )
    below = int(np.count_nonzero(scores < 0)) if minimum < 0 else 0
    above = int(np.count_nonzero(scores > 1)) if maximum > 1 else 0
    observation = {
        "policy_id": (DECODED_PRE_NMS_SCORE_POLICY_ID if score_policy is not None
                      else "strict_probability_0_1"),
        "absolute_tolerance": tolerance,
        "scores_below_zero": below,
        "scores_above_one": above,
        "corrected_score_count": below + above,
        "maximum_absolute_correction": max(0.0, -minimum, maximum - 1.0),
        "source_modified": False,
        "processing_copy_created": bool(below + above),
    }
    return array, values, observation


def _verify_decoded_pre_nms_values(
    outputs: Mapping[str, Any], *, score_policy: Mapping[str, Any] | None = None,
) -> None:
    """Legacy/no-policy calls remain strict; verification never changes data."""
    _checked_decoded_pre_nms_values(outputs, score_policy=score_policy)


def prepare_decoded_pre_nms_outputs(
    outputs: Mapping[str, Any], *, score_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate once and normalize only permitted score-edge roundoff.

    The accelerator buffer and raw evidence remain byte-identical. All values
    must pass the bounds before a processing copy is allocated. Width/height
    and nonfinite checks have no tolerance. No sigmoid is applied here.
    """
    array, _values, observation = _checked_decoded_pre_nms_values(
        outputs, score_policy=score_policy,
    )
    prepared = dict(outputs)
    if observation["corrected_score_count"]:
        copied = array.copy(order="K")
        channels = copied[0] if copied.shape[1] == 84 else copied[0].T
        # Only these already-validated probability channels can be saturated.
        scores = channels[4:]
        scores[scores < 0.0] = 0.0
        scores[scores > 1.0] = 1.0
        prepared[next(iter(outputs))] = copied
    return prepared, observation


_YOLO11_RAW_ENDPOINT_GEOMETRY = {
    (80, 64), (80, 80),
    (40, 64), (40, 80),
    (20, 64), (20, 80),
}


def _verify_yolo11_raw_endpoint_signature(
    signature: Mapping[str, Any], *, input_hw: Sequence[int],
) -> None:
    """Require the parity-attested YOLO11 DFL16/C80 three-level endpoint."""

    if [int(value) for value in input_hw] != [640, 640]:
        raise FrozenPostprocessError(
            "yolo11_raw_endpoint_640_input_required"
        )
    verified = _verify_tensor_signature_structure(
        signature,
        reason="yolo11_raw_endpoint_tensor_signature_invalid",
    )
    tensors = verified["tensors"]
    if len(tensors) != 6:
        raise FrozenPostprocessError(
            "yolo11_raw_endpoint_exact_six_tensors_required"
        )

    observed: set[tuple[int, int]] = set()
    semantic_pairs: dict[tuple[str, int], tuple[int, int]] = {}
    for row in tensors:
        shape = tuple(int(value) for value in row["shape"])
        if len(shape) == 4 and shape[0] == 1:
            dimensions = shape[1:]
        elif len(shape) == 3:
            dimensions = shape
        else:
            raise FrozenPostprocessError(
                "yolo11_raw_endpoint_tensor_layout_invalid"
            )

        candidates = {
            (grid, channels)
            for grid in (80, 40, 20)
            for channels in (64, 80)
            if dimensions in {
                (channels, grid, grid),
                (grid, grid, channels),
            }
        }
        if len(candidates) != 1:
            raise FrozenPostprocessError(
                "yolo11_raw_endpoint_tensor_geometry_invalid"
            )
        geometry = next(iter(candidates))
        if geometry in observed:
            raise FrozenPostprocessError(
                "yolo11_raw_endpoint_tensor_geometry_duplicate"
            )
        observed.add(geometry)

        name = str(row.get("name") or "")
        base = name.strip().split("/")[-1].split(":")[-1].lower()
        role: str | None = None
        level: int | None = None
        explicit = re.fullmatch(r"(?P<role>reg|cls)(?P<level>\d+)", base)
        if explicit is not None:
            role = str(explicit.group("role"))
            level = int(explicit.group("level"))
        else:
            branch = re.search(
                r"(?:one2one_)?cv(?P<branch>[23])\."
                r"(?P<level>\d+)(?:\.|/)",
                name.lower(),
            )
            if branch is not None:
                role = "reg" if branch.group("branch") == "2" else "cls"
                level = int(branch.group("level"))
        if role is not None and level is not None:
            key = (role, level)
            if key in semantic_pairs:
                raise FrozenPostprocessError(
                    "yolo11_raw_endpoint_semantic_pair_duplicate"
                )
            semantic_pairs[key] = geometry

    if observed != _YOLO11_RAW_ENDPOINT_GEOMETRY:
        raise FrozenPostprocessError(
            "yolo11_raw_endpoint_dfl16_coco_three_level_contract_required"
        )
    if semantic_pairs:
        reg_levels = {
            level for role, level in semantic_pairs if role == "reg"
        }
        cls_levels = {
            level for role, level in semantic_pairs if role == "cls"
        }
        if (
            len(semantic_pairs) != 6
            or len(reg_levels) != 3
            or reg_levels != cls_levels
            or {
                semantic_pairs[("reg", level)][0]
                for level in reg_levels
            } != {80, 40, 20}
            or any(
                semantic_pairs[("reg", level)][1] != 64
                or semantic_pairs[("cls", level)][1] != 80
                or semantic_pairs[("reg", level)][0]
                != semantic_pairs[("cls", level)][0]
                for level in reg_levels
            )
        ):
            raise FrozenPostprocessError(
                "yolo11_raw_endpoint_semantic_dfl16_coco_pairing_invalid"
            )


def tensor_signature(outputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(outputs, Mapping) or not outputs:
        raise FrozenPostprocessError("raw_detection_outputs_missing")
    tensors: list[dict[str, Any]] = []
    for index, (name, value) in enumerate(outputs.items()):
        array = np.asarray(value)
        if array.size <= 0 or array.dtype.kind not in "fiu":
            raise FrozenPostprocessError("raw_detection_output_tensor_invalid")
        tensors.append({
            "index": int(index),
            "name": str(name),
            "rank": int(array.ndim),
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(array.dtype),
        })
    return {"tensor_count": len(tensors), "tensors": tensors}


def _implementation_artifacts() -> dict[str, dict[str, Any]]:
    module_path = Path(__file__).resolve()
    base_path = Path(str(_base_module.__file__ or "")).resolve()
    harness_path = Path(str(_yolo_module.__file__ or "")).resolve()
    if not module_path.is_file() or not base_path.is_file() or not harness_path.is_file():
        raise FrozenPostprocessError("frozen_postprocess_implementation_missing")
    return {
        "native_detection_postprocess": {
            "relative_path": _IMPLEMENTATION_CANONICAL_PATHS[
                "native_detection_postprocess"
            ],
            "sha256": file_sha256(module_path),
        },
        "yolo_harness": {
            "relative_path": _IMPLEMENTATION_CANONICAL_PATHS["yolo_harness"],
            "sha256": file_sha256(harness_path),
        },
        "harness_base": {
            "relative_path": _IMPLEMENTATION_CANONICAL_PATHS["harness_base"],
            "sha256": file_sha256(base_path),
        },
    }


_IMPLEMENTATION_CANONICAL_PATHS = {
    # These are logical source identities, not import locations.  Generated
    # remote suites import byte-identical copies from ``splitpoint_runners``;
    # making that physical package prefix part of the scientific contract
    # caused performance and semantic-dump contracts to disagree.
    "native_detection_postprocess": (
        "onnx_splitpoint_tool/native_detection_postprocess.py"
    ),
    "yolo_harness": "onnx_splitpoint_tool/runners/harness/yolo.py",
    "harness_base": "onnx_splitpoint_tool/runners/harness/base.py",
}
_IMPLEMENTATION_PATH_ALIASES = {
    "native_detection_postprocess": {
        "onnx_splitpoint_tool/native_detection_postprocess.py",
        "splitpoint_runners/native_detection_postprocess.py",
    },
    "yolo_harness": {
        "onnx_splitpoint_tool/runners/harness/yolo.py",
        "splitpoint_runners/harness/yolo.py",
    },
    "harness_base": {
        "onnx_splitpoint_tool/runners/harness/base.py",
        "splitpoint_runners/harness/base.py",
    },
}
_V2776_NATIVE_DETECTION_POSTPROCESS_SHA256 = (
    "12e7b46d1ecbee019daedc0ebbfda5c81b91528c13c432940e56888329b1d20a"
)
_V27922_NATIVE_DETECTION_POSTPROCESS_SHA256 = "e1d454faa35fa1a8d1299ce88a9ca1729edab3d6d9f5b4aec7f9a4bccd7e07a6"
_V27926_NATIVE_DETECTION_POSTPROCESS_SHA256 = "00f0ad986c8021b864ddff3f7697dfc50e93fda8e9adf4e243fcd2f6b2164eed"
_V27927_NATIVE_DETECTION_POSTPROCESS_SHA256 = "696b2d657e69df1961a768a33d8f39359efd2f9e506d668657396c547c3d9fd5"
_V281_NATIVE_DETECTION_POSTPROCESS_SHA256 = "4e65d8b63316d1241e868063d4cf223ca403dab38d292c223fc9ab27fc1e0f0c"
_IMPLEMENTATION_SHA256_COMPATIBILITY = {
    # Run-39 executed this exact decoder implementation.  The 2.70c change to
    # this same file only teaches the verifier about vendored path aliases; it
    # does not change decode/NMS behavior.  Keep that already-sealed code hash
    # as an explicit, narrow compatibility identity.
    "native_detection_postprocess": {
        "b093bce5da80b3c42e3354358b7c0695300769993b1f4b1b72a8547f5eeb314e",
        # 2.70d differs from 2.70f only in the logical path emitted for
        # byte-identical vendored imports.  Preserve exact verification of
        # already archived 2.70d contracts while rejecting arbitrary hashes.
        "20b98b8de711f356d4f19d943b2356492b7b9019f6d3fd43e6ec27000f62f0ff",
        # 2.70j is the sealed input release for the comparison-endpoint
        # projection added in 2.70k.  Its decoder behavior is unchanged; only
        # the new, separate comparison identity builder was appended here.
        "1f96ca9df68152607482b38cc4dabc6408ace1244c9a2cd84ee7ea19229c504d",
        # 2.70k is the V1 comparison-endpoint implementation.  V2 adds a
        # separate Direct-BN6 normalization path but must continue to verify
        # already sealed V1 raw-host-tail contracts byte-for-byte.
        "159e760e5f5535c061d105d5de3df48432dd7285c4cc570580bd05cfed062977",
        # 2.71.4 is the immutable screening input for the 2.72 completion
        # contract.  The new APIs below are additive; retain verification of
        # contracts sealed by that exact implementation.
        "1572efb91cc949a989b52d9aca2d1b8a9ae48b74ea7181cf681a4f3d64037909",
        # Pre-2.72 frozen contracts did not bind the multiscale activation
        # strategy. They remain verifiable as archived evidence, but new
        # YOLOv7 contracts below require the explicit strategy field.
        "3aea92254ae3972bbf53f9e3960efdc4564e80b5af6e0d327168bc33b3fa7c94",
        # 2.73.0 produced the nine same-hotloop YOLOv7 completion artifacts.
        # Persisting the already materialized Full sentinel below is additive
        # and does not reinterpret those sealed decode/NMS results.
        "2996e934ec88a486f8afcb3dc2b0379bc22a4d1fbfda194dc522cff25e87d2d7",
        # v2.75.46 is the last release whose yolov7_paper decoder contract did
        # not bind an exact model-specific anchor table.  It remains readable
        # as archived evidence but is rejected for new runtime execution.
        "2ab5b42dfe741e084e8575da943c22c26f372202879f62070ad566e09dd65a8e",
        # v2.77.6 is the sealed input release for the additive YOLO11 Native-
        # Full family/endpoint route. Preserve its already bound YOLO26 and
        # YOLOv7 contracts without treating it as a pre-anchor legacy decoder.
        _V2776_NATIVE_DETECTION_POSTPROCESS_SHA256,
        # The 2.79.23 decoded pre-NMS route is additive; preserve old raw and
        # attested post-NMS evidence with unchanged processor semantics.
        _V27922_NATIVE_DETECTION_POSTPROCESS_SHA256,
        # v2.79.27 adds bounded failure diagnostics only. Decode/NMS and the
        # strict decoded-pre-NMS value predicates are unchanged from these
        # exact v2.79.26 bytes. Preserve previously sealed runtime evidence.
        _V27926_NATIVE_DETECTION_POSTPROCESS_SHA256,
        # v28 reads these exact archived contracts with STRICT probabilities.
        # They never acquire the new policy merely by being loaded in v28.
        _V27927_NATIVE_DETECTION_POSTPROCESS_SHA256,
        # v2.81 remains readable with its original decoder-contract version;
        # version 1 never acquires the v2.82 sigmoid arithmetic during replay.
        _V281_NATIVE_DETECTION_POSTPROCESS_SHA256,
    },
    "yolo_harness": {
        "31657e2717a8cb6cfda53e7de8e06b7b2286ecb430070811546fdec65327a17a",
        "4e1e81c1a931945ea53acdd1aee47e952b5828176cb0b98cfcefe2cb972f22aa",
        "53228a2ce59c2f15f3ea025e3d52a2312ea121a88fc59982e7536372c05cbfcd",
    },
}
_LEGACY_UNBOUND_ACTIVATION_NATIVE_SHA256 = {
    "b093bce5da80b3c42e3354358b7c0695300769993b1f4b1b72a8547f5eeb314e",
    "20b98b8de711f356d4f19d943b2356492b7b9019f6d3fd43e6ec27000f62f0ff",
    "1f96ca9df68152607482b38cc4dabc6408ace1244c9a2cd84ee7ea19229c504d",
    "159e760e5f5535c061d105d5de3df48432dd7285c4cc570580bd05cfed062977",
    "1572efb91cc949a989b52d9aca2d1b8a9ae48b74ea7181cf681a4f3d64037909",
}
_LEGACY_UNBOUND_ACTIVATION_YOLO_SHA256 = {
    "4e1e81c1a931945ea53acdd1aee47e952b5828176cb0b98cfcefe2cb972f22aa",
}
_LEGACY_UNBOUND_DECODER_NATIVE_SHA256 = set(
    _IMPLEMENTATION_SHA256_COMPATIBILITY["native_detection_postprocess"]
) - {
    _V2776_NATIVE_DETECTION_POSTPROCESS_SHA256,
    _V27922_NATIVE_DETECTION_POSTPROCESS_SHA256,
    _V27926_NATIVE_DETECTION_POSTPROCESS_SHA256,
    _V27927_NATIVE_DETECTION_POSTPROCESS_SHA256,
}
_LEGACY_UNBOUND_DECODER_YOLO_SHA256 = set(
    _IMPLEMENTATION_SHA256_COMPATIBILITY["yolo_harness"]
)


def _legacy_unbound_activation_contract(
    contract: Mapping[str, Any],
) -> bool:
    """Recognize only archived pre-2.72 YOLOv7 implementation identities."""
    if (
        _model_family(str(contract.get("model_id") or "")) != "yolov7"
        or "multiscale_activation_mode" in contract
    ):
        return False
    artifacts = contract.get("implementation_artifacts")
    if not isinstance(artifacts, Mapping):
        return False
    native = artifacts.get("native_detection_postprocess")
    yolo = artifacts.get("yolo_harness")
    invariant = contract.get("invariant_identity")
    return bool(
        isinstance(native, Mapping)
        and isinstance(yolo, Mapping)
        and str(native.get("sha256") or "").strip().lower()
        in _LEGACY_UNBOUND_ACTIVATION_NATIVE_SHA256
        and str(yolo.get("sha256") or "").strip().lower()
        in _LEGACY_UNBOUND_ACTIVATION_YOLO_SHA256
        and isinstance(invariant, Mapping)
        and "multiscale_activation_mode" not in invariant
    )


def _legacy_unbound_decoder_contract(
    contract: Mapping[str, Any],
) -> bool:
    """Recognize archived YOLOv7 contracts that predate anchor binding."""

    try:
        family = _model_family(str(contract.get("model_id") or ""))
    except FrozenPostprocessError:
        return False
    if (
        family != "yolov7"
        or "model_bound_decoder_contract" in contract
        or "model_bound_decoder_contract_sha256" in contract
        or "anchor_table_id" in contract
        or "anchors_by_stride" in contract
        or str(contract.get("decoder_id") or "")
        != "yolov7_multiscale_anchor_classaware_nms_v1"
    ):
        return False
    artifacts = contract.get("implementation_artifacts")
    if not isinstance(artifacts, Mapping):
        return False
    native = artifacts.get("native_detection_postprocess")
    yolo = artifacts.get("yolo_harness")
    invariant = contract.get("invariant_identity")
    return bool(
        isinstance(native, Mapping)
        and isinstance(yolo, Mapping)
        and str(native.get("sha256") or "").strip().lower()
        in _LEGACY_UNBOUND_DECODER_NATIVE_SHA256
        and str(yolo.get("sha256") or "").strip().lower()
        in _LEGACY_UNBOUND_DECODER_YOLO_SHA256
        and isinstance(invariant, Mapping)
        and "model_bound_decoder_contract_sha256" not in invariant
    )


def _implementation_artifacts_match(
    raw: Any, *, model_family: str,
) -> bool:
    """Accept only the main-package and byte-identical vendored path forms."""

    if not isinstance(raw, Mapping):
        return False
    observed = dict(raw)
    expected = _implementation_artifacts()
    if set(observed) != set(expected) or set(observed) != set(
        _IMPLEMENTATION_PATH_ALIASES
    ):
        return False
    for name, expected_row in expected.items():
        row = observed.get(name)
        if not isinstance(row, Mapping):
            return False
        path = str(row.get("relative_path") or "").strip().replace("\\", "/")
        path = path[2:] if path.startswith("./") else path
        sha256 = str(row.get("sha256") or "").strip().lower()
        # v2.77.6 had no YOLO11 family route. Its exact implementation hash is
        # compatible only with families that release could actually bind;
        # otherwise a newly resealed YOLO11 contract could claim impossible
        # historical provenance.
        if (
            name == "native_detection_postprocess"
            and sha256 == _V2776_NATIVE_DETECTION_POSTPROCESS_SHA256
            and str(model_family) not in {"yolo26", "yolov7"}
        ):
            return False
        allowed_hashes = {
            str(expected_row.get("sha256") or "").strip().lower(),
            *_IMPLEMENTATION_SHA256_COMPATIBILITY.get(name, set()),
        }
        if (
            set(row) != {"relative_path", "sha256"}
            or path not in _IMPLEMENTATION_PATH_ALIASES[name]
            or sha256 not in allowed_hashes
        ):
            return False
    return True


def _result_json(result: Any) -> dict[str, Any]:
    normalized = postprocess_result_to_dict(result)
    payload = normalized.get("json")
    if not isinstance(payload, dict):
        raise FrozenPostprocessError("frozen_postprocess_result_invalid")
    return payload


def frozen_postprocess_invariant_identity(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the decoder identity shared across different source images."""
    fields = [
        "schema", "schema_version", "task", "source_contract_family",
        "output_contract_family", "model_id", "model_family", "decoder_id",
        "decoder_format", "nms_implementation", "class_aware",
        "confidence_threshold", "iou_threshold", "max_detections", "input_hw",
        "raw_output_tensor_signature", "implementation_artifacts",
        "execution_policy", "performance_scope", "energy_scope",
        "host_postprocess_frozen", "postprocess_included",
    ]
    if (
        _model_family(str(contract.get("model_id") or "")) == "yolov7"
        and not _legacy_unbound_activation_contract(contract)
    ):
        fields.insert(
            fields.index("confidence_threshold"),
            "multiscale_activation_mode",
        )
    if (
        _model_family(str(contract.get("model_id") or "")) == "yolov7"
        and not _legacy_unbound_decoder_contract(contract)
    ):
        insertion = fields.index("decoder_format") + 1
        fields[insertion:insertion] = [
            "model_bound_decoder_contract_sha256",
            "anchor_table_id",
            "anchors_by_stride",
            "model_sha256",
        ]
    if "decoded_pre_nms_score_policy" in contract:
        fields.append("decoded_pre_nms_score_policy")
    return {field: contract.get(field) for field in fields}


def build_frozen_postprocess_contract(
    *, model_id: str, outputs: Mapping[str, Any], input_hw: Sequence[int],
    original_wh: Sequence[int],
    confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
    max_detections: int = 300,
    model_sha256: str = "",
    source_contract_family: str = "raw_head",
) -> dict[str, Any]:
    """Build a frozen contract after one untimed structural probe."""
    if len(tuple(input_hw)) != 2:
        raise FrozenPostprocessError("frozen_postprocess_input_hw_invalid")
    height, width = (int(value) for value in tuple(input_hw))
    if height <= 0 or width <= 0:
        raise FrozenPostprocessError("frozen_postprocess_input_hw_invalid")
    original_values = [int(value) for value in original_wh]
    if len(original_values) != 2 or any(value <= 0 for value in original_values):
        raise FrozenPostprocessError("frozen_postprocess_original_wh_invalid")
    decoder_id, expected_format = _expected_decoder(
        model_id, source_contract_family=source_contract_family,
    )
    model_family = _model_family(model_id)
    raw_output_signature = tensor_signature(outputs)
    processing_outputs = dict(outputs)
    score_policy = None
    if source_contract_family == "decoded_pre_nms":
        _verify_decoded_pre_nms_signature(raw_output_signature, input_hw=[height, width])
        if raw_output_signature["tensors"][0]["dtype"] == "float32":
            score_policy = decoded_pre_nms_score_policy()
        processing_outputs, _numeric_observation = prepare_decoded_pre_nms_outputs(
            outputs, score_policy=score_policy,
        )
    elif model_family == "yolo11":
        _verify_yolo11_raw_endpoint_signature(
            raw_output_signature,
            input_hw=[height, width],
        )
    activation_mode: str | None = None
    model_bound_decoder: dict[str, Any] | None = None
    if model_family == "yolov7":
        if expected_format != "multiscale_head":
            raise FrozenPostprocessError(
                "yolov7_frozen_multiscale_decoder_required"
            )
        try:
            _names, normalized, _normalization = (
                _yolo_module._normalize_multiscale_outputs(
                    [np.asarray(value) for value in outputs.values()]
                )
            )
            activation_mode = str(
                _yolo_module._infer_multiscale_head_activation_mode(
                    normalized
                )
            )
        except Exception as exc:
            raise FrozenPostprocessError(
                "yolov7_activation_strategy_probe_failed"
            ) from exc
        if activation_mode not in {
            "logits", "activated", "objcls_activated",
        }:
            raise FrozenPostprocessError(
                "yolov7_activation_strategy_ambiguous"
            )
        try:
            supplied_model_hash = str(model_sha256 or "").strip().lower()
            if supplied_model_hash.startswith("sha256:"):
                supplied_model_hash = supplied_model_hash.split(":", 1)[1]
            if not supplied_model_hash:
                raise FrozenPostprocessError(
                    "yolov7_model_sha256_required"
                )
            if supplied_model_hash != str(
                _yolo_module.YOLOV7_PAPER_ONNX_SHA256
            ):
                raise FrozenPostprocessError(
                    "yolov7_model_sha256_not_approved"
                )
            model_bound_decoder = (
                _yolo_module.registered_yolov7_decoder_contract(
                    model_id=str(model_id),
                    model_sha256=supplied_model_hash,
                    activation_mode=activation_mode,
                    variant="standard",
                    input_hw=[height, width],
                    confidence_threshold=float(confidence_threshold),
                    iou_threshold=float(iou_threshold),
                    max_detections=int(max_detections),
                    policy="production",
                )
            )
        except Exception as exc:
            raise FrozenPostprocessError(
                "yolov7_model_bound_decoder_contract_invalid"
            ) from exc
    harness = YoloHarness(
        conf_thresh=float(confidence_threshold),
        iou_thresh=float(iou_threshold),
        max_det=int(max_detections),
        multiscale_activation_mode=activation_mode,
        model_id=str(model_id),
        multiscale_decoder_contract=model_bound_decoder,
    )
    result = _result_json(harness.postprocess(
        processing_outputs, {
            "input_hw": [height, width], "original_wh": original_values,
            "variant": "native_full",
        },
    ))
    observed_format = str(result.get("format") or "")
    if observed_format != expected_format:
        raise FrozenPostprocessError(
            f"frozen_postprocess_decoder_format_mismatch:{observed_format}!={expected_format}"
        )
    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": (
            SCHEMA_VERSION
            if model_family == "yolov7" else SCHEMA_LEGACY_VERSION
        ),
        "task": "detection",
        "source_contract_family": source_contract_family,
        "output_contract_family": "decoded_nms",
        "model_id": str(model_id),
        "model_family": model_family,
        "decoder_id": decoder_id,
        "decoder_format": expected_format,
        "multiscale_activation_mode": activation_mode,
        "nms_implementation": "numpy_class_aware_nms_xyxy_v1",
        "class_aware": True,
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
        "max_detections": int(max_detections),
        "input_hw": [height, width],
        "original_wh": original_values,
        "raw_output_tensor_signature": raw_output_signature,
        "implementation_artifacts": _implementation_artifacts(),
        "execution_policy": EXECUTION_POLICY,
        "performance_scope": "prepared_input_to_post_nms_detections",
        "energy_scope": "same_frozen_prepared_input_to_post_nms_hotloop",
        "host_postprocess_frozen": True,
        "postprocess_included": True,
    }
    if score_policy is not None:
        contract["decoded_pre_nms_score_policy"] = score_policy
    if model_bound_decoder is not None:
        contract.update({
            "model_bound_decoder_contract": model_bound_decoder,
            "model_bound_decoder_contract_sha256": str(
                model_bound_decoder["decoder_contract_sha256"]
            ),
            "anchor_table_id": str(
                model_bound_decoder["anchor_table_id"]
            ),
            "anchors_by_stride": list(
                model_bound_decoder["anchors_by_stride"]
            ),
            "model_sha256": str(model_bound_decoder["model_sha256"]),
        })
    contract["invariant_identity"] = frozen_postprocess_invariant_identity(contract)
    contract["invariant_contract_sha256"] = canonical_json_sha256(
        contract["invariant_identity"]
    )
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def verify_frozen_postprocess_contract(
    raw: Any, *, outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError("frozen_postprocess_contract_missing")
    contract = dict(raw)
    contract.pop("legacy_activation_strategy_unbound", None)
    contract.pop("legacy_decoder_contract_unbound", None)
    declared = str(contract.pop("contract_sha256", "") or "").strip().lower()
    if len(declared) != 64 or canonical_json_sha256(contract) != declared:
        raise FrozenPostprocessError("frozen_postprocess_contract_sha256_mismatch")
    contract["contract_sha256"] = declared
    source_family = str(contract.get("source_contract_family") or "")
    decoder_id, expected_format = _expected_decoder(
        str(contract.get("model_id") or ""), source_contract_family=source_family,
    )
    legacy_unbound_activation = _legacy_unbound_activation_contract(
        contract
    )
    legacy_unbound_decoder = _legacy_unbound_decoder_contract(contract)
    if legacy_unbound_decoder:
        decoder_id = "yolov7_multiscale_anchor_classaware_nms_v1"
    input_hw = contract.get("input_hw")
    original_wh = contract.get("original_wh")
    if (
        contract.get("schema") != SCHEMA
        or int(contract.get("schema_version") or 0) not in {
            SCHEMA_LEGACY_VERSION, SCHEMA_VERSION,
        }
        or contract.get("task") != "detection"
        or source_family not in {"raw_head", "decoded_pre_nms"}
        or contract.get("output_contract_family") != "decoded_nms"
        or contract.get("host_postprocess_frozen") is not True
        or contract.get("postprocess_included") is not True
        or contract.get("execution_policy") != EXECUTION_POLICY
        or contract.get("model_family")
        != _model_family(str(contract.get("model_id") or ""))
        or str(contract.get("decoder_id") or "") != decoder_id
        or str(contract.get("decoder_format") or "") != expected_format
        or (
            not legacy_unbound_activation
            and (
                str(contract.get("multiscale_activation_mode") or "")
                not in {"logits", "activated", "objcls_activated"}
            )
            if _model_family(str(contract.get("model_id") or ""))
            == "yolov7"
            else contract.get("multiscale_activation_mode") is not None
        )
        or contract.get("nms_implementation") != "numpy_class_aware_nms_xyxy_v1"
        or contract.get("class_aware") is not True
        or not isinstance(input_hw, list) or len(input_hw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in input_hw)
        or not isinstance(original_wh, list) or len(original_wh) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in original_wh)
        or float(contract.get("confidence_threshold") or -1.0) != 0.25
        or float(contract.get("iou_threshold") or -1.0) != 0.45
        or int(contract.get("max_detections") or 0) != 300
    ):
        raise FrozenPostprocessError("frozen_postprocess_contract_fields_invalid")
    if _model_family(str(contract.get("model_id") or "")) == "yolov7":
        if legacy_unbound_decoder:
            if int(contract.get("schema_version") or 0) != SCHEMA_LEGACY_VERSION:
                raise FrozenPostprocessError(
                    "frozen_postprocess_legacy_decoder_schema_invalid"
                )
            if any(
                key in contract
                for key in (
                    "model_bound_decoder_contract",
                    "model_bound_decoder_contract_sha256",
                    "anchor_table_id", "anchors_by_stride", "model_sha256",
                )
            ):
                raise FrozenPostprocessError(
                    "frozen_postprocess_legacy_decoder_binding_invalid"
                )
        else:
            if int(contract.get("schema_version") or 0) != SCHEMA_VERSION:
                raise FrozenPostprocessError(
                    "frozen_postprocess_model_bound_decoder_schema_invalid"
                )
            try:
                model_bound = (
                    _yolo_module.verify_yolov7_decoder_contract(
                        contract.get("model_bound_decoder_contract"),
                        expected_model_id=str(contract.get("model_id") or ""),
                    )
                )
            except Exception as exc:
                raise FrozenPostprocessError(
                    "frozen_postprocess_model_bound_decoder_invalid"
                ) from exc
            if model_bound.get("schema_version") == 2:
                # Old implementations could not execute this arithmetic. Keep
                # their exact historical v1 contracts readable, but never let a
                # newly resealed v2 declaration borrow an old implementation.
                current_artifacts = _implementation_artifacts()
                declared_artifacts = contract.get("implementation_artifacts") or {}
                if any(
                    (declared_artifacts.get(name) or {}).get("sha256") != current_artifacts[name]["sha256"]
                    for name in ("native_detection_postprocess", "yolo_harness")
                ):
                    raise FrozenPostprocessError("yolov7_sigmoid_arithmetic_implementation_mismatch")
            if (
                str(
                    contract.get("model_bound_decoder_contract_sha256")
                    or ""
                ) != str(model_bound["decoder_contract_sha256"])
                or contract.get("anchor_table_id")
                != model_bound["anchor_table_id"]
                or contract.get("anchors_by_stride")
                != model_bound["anchors_by_stride"]
                or contract.get("model_sha256")
                != model_bound["model_sha256"]
                or contract.get("multiscale_activation_mode")
                != model_bound["activation_mode"]
                or float(contract.get("confidence_threshold") or -1.0)
                != float(
                    model_bound["nms_identity"]["confidence_threshold"]
                )
                or float(contract.get("iou_threshold") or -1.0)
                != float(model_bound["nms_identity"]["iou_threshold"])
                or int(contract.get("max_detections") or 0)
                != int(model_bound["nms_identity"]["max_detections"])
            ):
                raise FrozenPostprocessError(
                    "frozen_postprocess_model_bound_decoder_projection_mismatch"
                )
    elif int(contract.get("schema_version") or 0) != SCHEMA_LEGACY_VERSION:
        raise FrozenPostprocessError(
            "frozen_postprocess_non_yolov7_schema_invalid"
        )
    if "decoded_pre_nms_score_policy" in contract and source_family != "decoded_pre_nms":
        raise FrozenPostprocessError("decoded_pre_nms_score_policy_invalid")
    if not _implementation_artifacts_match(
        contract.get("implementation_artifacts"),
        model_family=str(contract.get("model_family") or ""),
    ):
        raise FrozenPostprocessError("frozen_postprocess_implementation_sha256_mismatch")
    invariant_identity = frozen_postprocess_invariant_identity(contract)
    if (
        dict(contract.get("invariant_identity") or {}) != invariant_identity
        or str(contract.get("invariant_contract_sha256") or "")
        != canonical_json_sha256(invariant_identity)
    ):
        raise FrozenPostprocessError("frozen_postprocess_invariant_identity_mismatch")
    signature = contract.get("raw_output_tensor_signature")
    if not isinstance(signature, Mapping) or int(signature.get("tensor_count") or 0) <= 0:
        raise FrozenPostprocessError("frozen_postprocess_tensor_signature_missing")
    if outputs is not None and dict(signature) != tensor_signature(outputs):
        raise FrozenPostprocessError("frozen_postprocess_tensor_signature_mismatch")
    if source_family == "decoded_pre_nms":
        implementation = contract["implementation_artifacts"]["native_detection_postprocess"]["sha256"]
        current_implementation = file_sha256(Path(__file__).resolve())
        if implementation not in {
            current_implementation,
            _V27926_NATIVE_DETECTION_POSTPROCESS_SHA256,
            _V27927_NATIVE_DETECTION_POSTPROCESS_SHA256,
            _V281_NATIVE_DETECTION_POSTPROCESS_SHA256,
        }:
            raise FrozenPostprocessError("decoded_pre_nms_implementation_not_supported")
        _verify_decoded_pre_nms_signature(signature, input_hw=input_hw)
        score_policy = contract.get("decoded_pre_nms_score_policy")
        if "decoded_pre_nms_score_policy" in contract:
            if (implementation not in {current_implementation, _V281_NATIVE_DETECTION_POSTPROCESS_SHA256}
                    or score_policy != decoded_pre_nms_score_policy()
                    or signature["tensors"][0]["dtype"] != "float32"):
                raise FrozenPostprocessError("decoded_pre_nms_score_policy_invalid")
        elif implementation in {current_implementation, _V281_NATIVE_DETECTION_POSTPROCESS_SHA256} and signature["tensors"][0]["dtype"] == "float32":
            raise FrozenPostprocessError("decoded_pre_nms_score_policy_missing")
        if outputs is not None:
            _verify_decoded_pre_nms_values(outputs, score_policy=score_policy)
    elif _model_family(str(contract.get("model_id") or "")) == "yolo11":
        _verify_yolo11_raw_endpoint_signature(
            signature,
            input_hw=input_hw,
        )
    if legacy_unbound_activation:
        contract["legacy_activation_strategy_unbound"] = True
    if legacy_unbound_decoder:
        contract["legacy_decoder_contract_unbound"] = True
    return contract


def build_letterbox_geometry_contract(
    *,
    input_hw: Sequence[int],
    original_wh: Sequence[int],
    preprocess: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the exact centered-letterbox geometry used by a Direct-BN6 row."""
    input_values = _strict_positive_int_list(
        list(input_hw) if isinstance(input_hw, (list, tuple)) else input_hw,
        length=2,
        reason="direct_normalization_geometry_dimensions_invalid",
    )
    original_values = _strict_positive_int_list(
        (
            list(original_wh)
            if isinstance(original_wh, (list, tuple)) else original_wh
        ),
        length=2,
        reason="direct_normalization_geometry_dimensions_invalid",
    )
    if not isinstance(preprocess, Mapping):
        raise FrozenPostprocessError(
            "direct_normalization_geometry_dimensions_invalid"
        )
    mode = str(
        preprocess.get("mode") or preprocess.get("preprocess_mode") or ""
    ).strip().lower()
    if mode not in {"letterbox", "letterbox_rgb_uint8"}:
        raise FrozenPostprocessError(
            "direct_normalization_letterbox_contract_required"
        )
    rgb = preprocess.get("rgb")
    color_space = str(preprocess.get("color_space") or "").strip().upper()
    if rgb is not True and color_space != "RGB":
        raise FrozenPostprocessError(
            "direct_normalization_rgb_contract_required"
        )
    raw_pad_value = (
        preprocess.get("pad_value")
        if preprocess.get("pad_value") is not None
        else preprocess.get("letterbox_pad_value")
    )
    if isinstance(raw_pad_value, bool) or not isinstance(
        raw_pad_value, int
    ):
        raise FrozenPostprocessError(
            "direct_normalization_letterbox_pad_invalid"
        )
    pad_value = raw_pad_value
    if pad_value < 0 or pad_value > 255:
        raise FrozenPostprocessError(
            "direct_normalization_letterbox_pad_invalid"
        )
    input_height, input_width = input_values
    original_width, original_height = original_values
    gain = min(
        input_width / float(original_width),
        input_height / float(original_height),
    )
    resized_width = int(round(original_width * gain))
    resized_height = int(round(original_height * gain))
    pad_left = (input_width - resized_width) // 2
    pad_top = (input_height - resized_height) // 2
    if (
        gain <= 0.0
        or resized_width <= 0
        or resized_height <= 0
        or pad_left < 0
        or pad_top < 0
    ):
        raise FrozenPostprocessError(
            "direct_normalization_letterbox_geometry_invalid"
        )
    identity = {
        "schema": "onnx-splitpoint/centered-letterbox-geometry",
        "schema_version": 1,
        "mode": "centered_letterbox_round_v1",
        "input_hw": input_values,
        "original_wh": original_values,
        "gain": float(gain),
        "resized_wh": [resized_width, resized_height],
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(input_width - resized_width - pad_left),
        "pad_bottom": int(input_height - resized_height - pad_top),
        "pad_value": int(pad_value),
        "color_space": "RGB",
    }
    return {
        **identity,
        "geometry_contract_sha256": canonical_json_sha256(identity),
    }


def verify_letterbox_geometry_contract(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "direct_normalization_geometry_contract_missing"
        )
    contract = dict(raw)
    declared = _strict_sha256(
        contract.pop("geometry_contract_sha256", ""),
        reason="direct_normalization_geometry_contract_sha256_invalid",
    )
    if canonical_json_sha256(contract) != declared:
        raise FrozenPostprocessError(
            "direct_normalization_geometry_contract_sha256_mismatch"
        )
    expected = build_letterbox_geometry_contract(
        input_hw=contract.get("input_hw") or [],
        original_wh=contract.get("original_wh") or [],
        preprocess={
            "mode": "letterbox",
            "letterbox_pad_value": contract.get("pad_value"),
            "color_space": contract.get("color_space"),
        },
    )
    _strict_positive_int_list(
        contract.get("input_hw"),
        length=2,
        reason="direct_normalization_geometry_contract_invalid",
    )
    _strict_positive_int_list(
        contract.get("original_wh"),
        length=2,
        reason="direct_normalization_geometry_contract_invalid",
    )
    _strict_positive_int_list(
        contract.get("resized_wh"),
        length=2,
        reason="direct_normalization_geometry_contract_invalid",
    )
    integer_geometry_fields = (
        "pad_left", "pad_top", "pad_right", "pad_bottom", "pad_value",
    )
    if (
        set(contract) != {
            "schema", "schema_version", "mode", "input_hw",
            "original_wh", "gain", "resized_wh", "pad_left",
            "pad_top", "pad_right", "pad_bottom", "pad_value",
            "color_space",
        }
        or
        contract.get("schema")
        != "onnx-splitpoint/centered-letterbox-geometry"
        or isinstance(contract.get("schema_version"), bool)
        or contract.get("schema_version") != 1
        or not isinstance(contract.get("gain"), float)
        or not math.isfinite(contract.get("gain"))
        or contract.get("gain") <= 0.0
        or any(
            isinstance(contract.get(field), bool)
            or not isinstance(contract.get(field), int)
            or contract.get(field) < 0
            for field in integer_geometry_fields
        )
        or dict(expected) != {**contract, "geometry_contract_sha256": declared}
    ):
        raise FrozenPostprocessError(
            "direct_normalization_geometry_contract_invalid"
        )
    return {**contract, "geometry_contract_sha256": declared}


def _verify_decoded_nms_source_attestation(
    raw: Any,
    *,
    outputs: Mapping[str, Any],
    source_endpoint_contract_hash: str,
    model_id: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "direct_normalization_source_attestation_missing"
        )
    attestation = dict(raw)
    source_hash = _strict_sha256(
        source_endpoint_contract_hash,
        reason="direct_normalization_source_endpoint_hash_invalid",
    )
    signature = tensor_signature(outputs)
    declared_contract = attestation.get("declared_contract")
    declared_model = (
        str(declared_contract.get("model_id") or "").strip()
        if isinstance(declared_contract, Mapping) else ""
    )
    if (
        attestation.get("schema")
        != "onnx-splitpoint/runtime-output-endpoint-attestation"
        or isinstance(attestation.get("schema_version"), bool)
        or attestation.get("schema_version") != 3
        or attestation.get("attested") is not True
        or str(attestation.get("status") or "").strip().lower() != "passed"
        or str(attestation.get("endpoint") or "").strip().lower()
        != "decoded_nms"
        or str(attestation.get("stage") or "").strip().lower()
        != "decoded_nms"
        or attestation.get("values_decoded_xyxy_score_class") is not True
        or attestation.get("declaration_attested") is not True
        or _strict_sha256(
            attestation.get("endpoint_contract_hash"),
            reason="direct_normalization_source_endpoint_hash_invalid",
        ) != source_hash
        or dict(attestation.get("tensor_signature") or {}) != signature
        or not isinstance(declared_contract, Mapping)
        or declared_model != str(model_id)
        or declared_contract.get("source_coordinate_space")
        != "model_input_letterbox_xyxy_pixels"
    ):
        raise FrozenPostprocessError(
            "direct_normalization_source_attestation_invalid"
        )
    return attestation


def build_frozen_decoded_nms_normalization_contract(
    *,
    model_id: str,
    outputs: Mapping[str, Any],
    input_hw: Sequence[int],
    original_wh: Sequence[int],
    preprocess: Mapping[str, Any],
    source_coordinate_space: str,
    source_endpoint_contract_hash: str,
    source_output_endpoint_attestation: Mapping[str, Any],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> dict[str, Any]:
    """Freeze canonical BN6 filtering/NMS and inverse-letterbox projection."""
    if str(source_coordinate_space or "").strip().lower() != (
        "model_input_letterbox_xyxy_pixels"
    ):
        raise FrozenPostprocessError(
            "direct_normalization_source_coordinate_space_unattested"
        )
    geometry = build_letterbox_geometry_contract(
        input_hw=input_hw,
        original_wh=original_wh,
        preprocess=preprocess,
    )
    source_hash = _strict_sha256(
        source_endpoint_contract_hash,
        reason="direct_normalization_source_endpoint_hash_invalid",
    )
    attestation = _verify_decoded_nms_source_attestation(
        source_output_endpoint_attestation,
        outputs=outputs,
        source_endpoint_contract_hash=source_hash,
        model_id=model_id,
    )
    harness = YoloHarness(
        conf_thresh=float(confidence_threshold),
        iou_thresh=float(iou_threshold),
        max_det=int(max_detections),
    )
    probe = _result_json(harness.postprocess(
        dict(outputs),
        {
            "input_hw": list(geometry["input_hw"]),
            "original_wh": list(geometry["original_wh"]),
            "variant": "native_full_direct_bn6",
        },
    ))
    if str(probe.get("format") or "") != "bn6_detections":
        raise FrozenPostprocessError(
            "direct_normalization_bn6_format_required"
        )
    contract: dict[str, Any] = {
        "schema": DIRECT_NORMALIZATION_SCHEMA,
        "schema_version": DIRECT_NORMALIZATION_VERSION,
        "task": "detection",
        "source_contract_family": "decoded_nms",
        "source_output_format": "bn6_detections",
        "source_coordinate_space": (
            "model_input_letterbox_xyxy_pixels"
        ),
        "output_contract_family": "decoded_nms",
        "output_record_format": "xyxy_score_class",
        "output_coordinate_space": "original_image_xyxy_pixels",
        "model_id": str(model_id),
        "model_family": _model_family(model_id),
        "normalizer_id": (
            "bn6_classaware_postfilter_inverse_letterbox_v1"
        ),
        "nms_implementation": "numpy_class_aware_nms_xyxy_v1",
        "class_aware": True,
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
        "max_detections": int(max_detections),
        "input_hw": list(geometry["input_hw"]),
        "original_wh": list(geometry["original_wh"]),
        "letterbox_geometry_contract": geometry,
        "letterbox_geometry_contract_sha256": str(
            geometry["geometry_contract_sha256"]
        ),
        "source_output_tensor_signature": tensor_signature(outputs),
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_id": (
            f"detection:decoded_nms:{source_hash}"
        ),
        "source_output_endpoint_attestation_sha256": (
            canonical_json_sha256(attestation)
        ),
        "implementation_artifacts": _implementation_artifacts(),
        "execution_policy": DIRECT_NORMALIZATION_EXECUTION_POLICY,
        "performance_scope": (
            "prepared_input_to_canonical_original_image_detections"
        ),
        "energy_scope": (
            "same_frozen_prepared_input_to_canonical_detection_hotloop"
        ),
        "normalization_frozen": True,
        "postprocess_included": True,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def verify_frozen_decoded_nms_normalization_contract(
    raw: Any,
    *,
    outputs: Mapping[str, Any] | None = None,
    source_output_endpoint_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "direct_normalization_contract_missing"
        )
    contract = dict(raw)
    declared_hash = _strict_sha256(
        contract.pop("contract_sha256", ""),
        reason="direct_normalization_contract_sha256_invalid",
    )
    if canonical_json_sha256(contract) != declared_hash:
        raise FrozenPostprocessError(
            "direct_normalization_contract_sha256_mismatch"
        )
    geometry = verify_letterbox_geometry_contract(
        contract.get("letterbox_geometry_contract")
    )
    input_hw = contract.get("input_hw")
    original_wh = contract.get("original_wh")
    input_hw_values = _strict_positive_int_list(
        input_hw,
        length=2,
        reason="direct_normalization_contract_fields_invalid",
    )
    original_wh_values = _strict_positive_int_list(
        original_wh,
        length=2,
        reason="direct_normalization_contract_fields_invalid",
    )
    source_hash = _strict_sha256(
        contract.get("source_endpoint_contract_hash"),
        reason="direct_normalization_source_endpoint_hash_invalid",
    )
    source_attestation_hash = _strict_sha256(
        contract.get("source_output_endpoint_attestation_sha256"),
        reason="direct_normalization_source_attestation_sha256_invalid",
    )
    source_signature = _verify_tensor_signature_structure(
        contract.get("source_output_tensor_signature"),
        reason="direct_normalization_tensor_signature_invalid",
    )
    expected_keys = {
        "schema", "schema_version", "task", "source_contract_family",
        "source_output_format", "source_coordinate_space",
        "output_contract_family", "output_record_format",
        "output_coordinate_space", "model_id", "model_family",
        "normalizer_id", "nms_implementation", "class_aware",
        "confidence_threshold", "iou_threshold", "max_detections",
        "input_hw", "original_wh", "letterbox_geometry_contract",
        "letterbox_geometry_contract_sha256",
        "source_output_tensor_signature",
        "source_endpoint_contract_hash", "source_output_endpoint_id",
        "source_output_endpoint_attestation_sha256",
        "implementation_artifacts", "execution_policy",
        "performance_scope", "energy_scope", "normalization_frozen",
        "postprocess_included",
    }
    model_id = contract.get("model_id")
    model_family = contract.get("model_family")
    if (
        set(contract) != expected_keys
        or contract.get("schema") != DIRECT_NORMALIZATION_SCHEMA
        or isinstance(contract.get("schema_version"), bool)
        or contract.get("schema_version") != DIRECT_NORMALIZATION_VERSION
        or contract.get("task") != "detection"
        or contract.get("source_contract_family") != "decoded_nms"
        or contract.get("source_output_format") != "bn6_detections"
        or contract.get("source_coordinate_space")
        != "model_input_letterbox_xyxy_pixels"
        or contract.get("output_contract_family") != "decoded_nms"
        or contract.get("output_record_format") != "xyxy_score_class"
        or contract.get("output_coordinate_space")
        != "original_image_xyxy_pixels"
        or contract.get("normalizer_id")
        != "bn6_classaware_postfilter_inverse_letterbox_v1"
        or contract.get("nms_implementation")
        != "numpy_class_aware_nms_xyxy_v1"
        or contract.get("class_aware") is not True
        or isinstance(contract.get("confidence_threshold"), bool)
        or isinstance(contract.get("iou_threshold"), bool)
        or isinstance(contract.get("max_detections"), bool)
        or not isinstance(
            contract.get("confidence_threshold"), (int, float)
        )
        or not isinstance(contract.get("iou_threshold"), (int, float))
        or not isinstance(contract.get("max_detections"), int)
        or float(contract.get("confidence_threshold") or -1.0) != 0.25
        or float(contract.get("iou_threshold") or -1.0) != 0.45
        or int(contract.get("max_detections") or 0) != 300
        or input_hw_values != list(geometry["input_hw"])
        or original_wh_values != list(geometry["original_wh"])
        or str(contract.get("letterbox_geometry_contract_sha256") or "")
        != str(geometry["geometry_contract_sha256"])
        or str(
            contract.get("source_output_endpoint_attestation_sha256")
            or ""
        ).strip().lower() != source_attestation_hash
        or str(
            contract.get("source_endpoint_contract_hash") or ""
        ).strip().lower() != source_hash
        or contract.get("source_output_endpoint_id")
        != f"detection:decoded_nms:{source_hash}"
        or contract.get("execution_policy")
        != DIRECT_NORMALIZATION_EXECUTION_POLICY
        or contract.get("normalization_frozen") is not True
        or contract.get("postprocess_included") is not True
        or contract.get("performance_scope")
        != "prepared_input_to_canonical_original_image_detections"
        or contract.get("energy_scope")
        != "same_frozen_prepared_input_to_canonical_detection_hotloop"
        or not isinstance(model_id, str)
        or not model_id.strip()
        or model_family != _model_family(model_id)
    ):
        raise FrozenPostprocessError(
            "direct_normalization_contract_fields_invalid"
        )
    if not _implementation_artifacts_match(
        contract.get("implementation_artifacts"),
        model_family=str(contract.get("model_family") or ""),
    ):
        raise FrozenPostprocessError(
            "direct_normalization_implementation_sha256_mismatch"
        )
    if outputs is not None:
        if source_signature != tensor_signature(
            outputs
        ):
            raise FrozenPostprocessError(
                "direct_normalization_tensor_signature_mismatch"
            )
        if source_output_endpoint_attestation is None:
            raise FrozenPostprocessError(
                "direct_normalization_source_attestation_missing"
            )
        verified_attestation = _verify_decoded_nms_source_attestation(
            source_output_endpoint_attestation,
            outputs=outputs,
            source_endpoint_contract_hash=source_hash,
            model_id=str(contract.get("model_id") or ""),
        )
        if canonical_json_sha256(verified_attestation) != str(
            contract.get("source_output_endpoint_attestation_sha256") or ""
        ):
            raise FrozenPostprocessError(
                "direct_normalization_source_attestation_sha256_mismatch"
            )
    return {**contract, "contract_sha256": declared_hash}


def build_completed_detection_endpoint_contract(
    frozen_contract: Mapping[str, Any],
    *,
    source_endpoint_contract_hash: str = "",
) -> dict[str, Any]:
    """Describe the completed task endpoint without relabelling raw tensors.

    ``native_full_outputs_manifest.json`` archives accelerator outputs.  Those
    files remain a ``raw_head`` endpoint even when a frozen host tail was
    included in every measured iteration.  This companion contract identifies
    the *completed task* endpoint separately and binds it to the exact raw
    tensor signature and frozen decoder/NMS implementation.
    """
    verified = verify_frozen_postprocess_contract(frozen_contract)
    source_hash = str(source_endpoint_contract_hash or "").strip().lower()
    if source_hash.startswith("sha256:"):
        source_hash = source_hash.split(":", 1)[1]
    if source_hash and (
        len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise FrozenPostprocessError(
            "completed_endpoint_source_contract_sha256_invalid"
        )
    identity: dict[str, Any] = {
        "schema": COMPLETED_ENDPOINT_SCHEMA,
        "schema_version": (
            COMPLETED_ENDPOINT_VERSION
            if (
                str(verified.get("model_family") or "") == "yolov7"
                and verified.get("legacy_decoder_contract_unbound") is not True
            )
            else COMPLETED_ENDPOINT_LEGACY_VERSION
        ),
        "task": "detection",
        "source_stage": verified["source_contract_family"],
        "completed_stage": "decoded_nms",
        "contract_source": COMPLETED_ENDPOINT_SOURCE,
        "source_endpoint_contract_hash": source_hash,
        "raw_output_tensor_signature": verified["raw_output_tensor_signature"],
        "frozen_postprocess_contract_sha256": verified["contract_sha256"],
        "frozen_postprocess_invariant_contract_sha256": verified[
            "invariant_contract_sha256"
        ],
        "decoder_id": verified["decoder_id"],
        "decoder_format": verified["decoder_format"],
        "nms_implementation": verified["nms_implementation"],
        "class_aware": verified["class_aware"],
        "confidence_threshold": verified["confidence_threshold"],
        "iou_threshold": verified["iou_threshold"],
        "max_detections": verified["max_detections"],
        "input_hw": verified["input_hw"],
        "original_wh": verified["original_wh"],
        "performance_scope": verified["performance_scope"],
        "energy_scope": verified["energy_scope"],
    }
    if verified.get("legacy_decoder_contract_unbound") is not True and (
        str(verified.get("model_family") or "") == "yolov7"
    ):
        identity.update({
            "model_sha256": verified["model_sha256"],
            "model_bound_decoder_contract_sha256": verified[
                "model_bound_decoder_contract_sha256"
            ],
            "anchor_table_id": verified["anchor_table_id"],
            "anchors_by_stride": verified["anchors_by_stride"],
        })
    endpoint_hash = canonical_json_sha256(identity)
    return {
        **identity,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": f"detection:decoded_nms:{endpoint_hash}",
    }


def _completed_comparison_v1_identity(
    frozen_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact 2.70k/V1 identity for archived evidence."""
    verified = verify_frozen_postprocess_contract(frozen_contract)
    if verified["source_contract_family"] != "raw_head":
        raise FrozenPostprocessError("completed_comparison_endpoint_v1_requires_raw_tail")
    return {
        "schema": COMPLETED_COMPARISON_ENDPOINT_SCHEMA,
        "schema_version": COMPLETED_COMPARISON_ENDPOINT_V1,
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "model_id": str(verified["model_id"]),
        "model_family": str(verified["model_family"]),
        "decoder_semantics_id": str(verified["decoder_id"]),
        "decoder_format": str(verified["decoder_format"]),
        "output_record_format": "xyxy_score_class",
        "coordinate_space": "original_image_xyxy_pixels",
        "score_semantics": "probability_0_1",
        "class_id_semantics": "integer_model_label_index",
        "class_aware": bool(verified["class_aware"]),
        "score_threshold": float(verified["confidence_threshold"]),
        "iou_threshold": float(verified["iou_threshold"]),
        "max_detections": int(verified["max_detections"]),
        "input_hw": [int(value) for value in verified["input_hw"]],
        # This is the task semantics, not the implementation identity.  The
        # physical attestation still binds the concrete NumPy/accelerator
        # implementation separately.
        "nms_semantics_id": "class_aware_nms_xyxy_v1",
    }


def _completed_comparison_v2_identity(
    *,
    frozen_contract: Mapping[str, Any] | None = None,
    direct_normalization_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project either physical implementation to one canonical task endpoint."""
    if (frozen_contract is None) == (direct_normalization_contract is None):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_exactly_one_source_required"
        )
    if frozen_contract is not None:
        source = verify_frozen_postprocess_contract(frozen_contract)
        model_id = str(source["model_id"])
        model_family = str(source["model_family"])
        input_hw = [int(value) for value in source["input_hw"]]
        class_aware = bool(source["class_aware"])
        score_threshold = float(source["confidence_threshold"])
        iou_threshold = float(source["iou_threshold"])
        max_detections = int(source["max_detections"])
    else:
        source = verify_frozen_decoded_nms_normalization_contract(
            direct_normalization_contract
        )
        model_id = str(source["model_id"])
        model_family = str(source["model_family"])
        input_hw = [int(value) for value in source["input_hw"]]
        class_aware = bool(source["class_aware"])
        score_threshold = float(source["confidence_threshold"])
        iou_threshold = float(source["iou_threshold"])
        max_detections = int(source["max_detections"])
    return {
        "schema": COMPLETED_COMPARISON_ENDPOINT_SCHEMA,
        "schema_version": COMPLETED_COMPARISON_ENDPOINT_VERSION,
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "model_id": model_id,
        "model_family": model_family,
        "output_record_format": "xyxy_score_class",
        "coordinate_space": "original_image_xyxy_pixels",
        "score_semantics": "probability_0_1",
        "class_id_semantics": "integer_model_label_index",
        "class_aware": class_aware,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
        "input_hw": input_hw,
        "canonical_completion_policy_id": CANONICAL_COMPLETION_POLICY_ID,
        "nms_semantics_id": "class_aware_nms_xyxy_v1",
    }


def _seal_completed_comparison_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity_dict = dict(identity)
    contract_hash = canonical_json_sha256(identity_dict)
    return {
        **identity_dict,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": contract_hash,
        "output_endpoint_id": (
            f"detection:decoded_nms:comparison:{contract_hash}"
        ),
    }


def _completed_comparison_schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_schema_version_unsupported"
        )
    if value not in {
        COMPLETED_COMPARISON_ENDPOINT_V1,
        COMPLETED_COMPARISON_ENDPOINT_VERSION,
    }:
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_schema_version_unsupported"
        )
    return value


def build_completed_detection_comparison_endpoint_contract(
    frozen_contract: Mapping[str, Any] | None = None,
    *,
    direct_normalization_contract: Mapping[str, Any] | None = None,
    schema_version: int = COMPLETED_COMPARISON_ENDPOINT_VERSION,
) -> dict[str, Any]:
    """Build a versioned, backend-independent completed task identity.

    V1 remains byte-for-byte reproducible for archived raw-host-tail evidence.
    V2 projects both a raw host tail and a Direct-BN6 frozen normalizer onto
    the same canonical output semantics while leaving physical provenance in
    their separate completion attestations.
    """
    version = _completed_comparison_schema_version(schema_version)
    if version == COMPLETED_COMPARISON_ENDPOINT_V1:
        if frozen_contract is None or direct_normalization_contract is not None:
            raise FrozenPostprocessError(
                "completed_comparison_endpoint_v1_requires_frozen_tail"
            )
        identity = _completed_comparison_v1_identity(frozen_contract)
    elif version == COMPLETED_COMPARISON_ENDPOINT_VERSION:
        identity = _completed_comparison_v2_identity(
            frozen_contract=frozen_contract,
            direct_normalization_contract=direct_normalization_contract,
        )
    return _seal_completed_comparison_identity(identity)


def verify_completed_detection_comparison_endpoint_contract(
    raw: Any,
    *,
    frozen_contract: Mapping[str, Any] | None = None,
    direct_normalization_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify V1/V2 without ever reinterpreting an archived V1 hash."""
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_missing"
        )
    contract = dict(raw)
    declared_hash = str(
        contract.pop("endpoint_contract_hash", "") or "",
    ).strip().lower()
    declared_id = str(
        contract.pop("output_endpoint_id", "") or "",
    ).strip()
    complete = contract.pop("endpoint_contract_complete", None)
    calculated = canonical_json_sha256(contract)
    expected_id = f"detection:decoded_nms:comparison:{calculated}"
    try:
        schema_version = _completed_comparison_schema_version(
            contract.get("schema_version")
        )
    except FrozenPostprocessError as exc:
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_invalid"
        ) from exc
    if (
        complete is not True
        or declared_hash != calculated
        or declared_id != expected_id
        or contract.get("schema") != COMPLETED_COMPARISON_ENDPOINT_SCHEMA
        or schema_version not in {
            COMPLETED_COMPARISON_ENDPOINT_V1,
            COMPLETED_COMPARISON_ENDPOINT_VERSION,
        }
        or contract.get("task") != "detection"
        or contract.get("stage") != "decoded_nms"
        or contract.get("contract_family") != "decoded_nms"
        or contract.get("output_record_format") != "xyxy_score_class"
        or contract.get("coordinate_space")
        != "original_image_xyxy_pixels"
        or contract.get("score_semantics") != "probability_0_1"
        or contract.get("class_id_semantics")
        != "integer_model_label_index"
        or contract.get("class_aware") is not True
    ):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_invalid"
        )
    common_keys = {
            "schema", "schema_version", "task", "stage", "contract_family",
            "model_id", "model_family", "output_record_format",
            "coordinate_space", "score_semantics", "class_id_semantics",
            "class_aware", "score_threshold", "iou_threshold",
            "max_detections", "input_hw", "nms_semantics_id",
    }
    v1_keys = common_keys | {"decoder_semantics_id", "decoder_format"}
    v2_keys = common_keys | {"canonical_completion_policy_id"}
    if (
        (
            schema_version == COMPLETED_COMPARISON_ENDPOINT_V1
            and set(contract) != v1_keys
        )
        or (
            schema_version == COMPLETED_COMPARISON_ENDPOINT_VERSION
            and set(contract) != v2_keys
        )
    ):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_invalid"
        )
    input_hw = contract.get("input_hw")
    model_id = contract.get("model_id")
    model_family = contract.get("model_family")
    try:
        expected_model_family = _model_family(
            model_id if isinstance(model_id, str) else ""
        )
    except FrozenPostprocessError as exc:
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_invalid"
        ) from exc
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or not isinstance(model_family, str)
        or model_family != expected_model_family
        or isinstance(contract.get("score_threshold"), bool)
        or isinstance(contract.get("iou_threshold"), bool)
        or isinstance(contract.get("max_detections"), bool)
        or not isinstance(contract.get("score_threshold"), (int, float))
        or not isinstance(contract.get("iou_threshold"), (int, float))
        or not isinstance(contract.get("max_detections"), int)
        or float(contract.get("score_threshold") or -1.0) != 0.25
        or float(contract.get("iou_threshold") or -1.0) != 0.45
        or int(contract.get("max_detections") or 0) != 300
        or not isinstance(input_hw, list)
        or len(input_hw) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in input_hw
        )
        or contract.get("nms_semantics_id") != "class_aware_nms_xyxy_v1"
    ):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_invalid"
        )
    if schema_version == COMPLETED_COMPARISON_ENDPOINT_V1:
        expected_decoder, expected_format = _expected_decoder(model_id)
        allowed_decoders = {expected_decoder}
        if expected_model_family == "yolov7":
            allowed_decoders.add(
                "yolov7_multiscale_anchor_classaware_nms_v1"
            )
        if (
            contract.get("decoder_semantics_id") not in allowed_decoders
            or contract.get("decoder_format") != expected_format
        ):
            raise FrozenPostprocessError(
                "completed_comparison_endpoint_contract_invalid"
            )
    if schema_version == COMPLETED_COMPARISON_ENDPOINT_VERSION and (
        contract.get("canonical_completion_policy_id")
        != CANONICAL_COMPLETION_POLICY_ID
    ):
        raise FrozenPostprocessError(
            "completed_comparison_endpoint_contract_invalid"
        )
    verified = {
        **contract,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": declared_hash,
        "output_endpoint_id": declared_id,
    }
    if frozen_contract is not None or direct_normalization_contract is not None:
        expected = build_completed_detection_comparison_endpoint_contract(
            frozen_contract,
            direct_normalization_contract=direct_normalization_contract,
            schema_version=schema_version,
        )
        if verified != expected:
            raise FrozenPostprocessError(
                "completed_comparison_endpoint_source_contract_mismatch"
            )
    return verified


def build_completed_detection_endpoint_attestation(
    frozen_contract: Mapping[str, Any],
    frozen_result: Mapping[str, Any],
    *,
    completed_frames: int,
    postprocess_completed_frames: int,
    source_endpoint_contract_hash: str = "",
) -> dict[str, Any]:
    """Attest that the frozen host tail completed once per measured frame."""
    contract = build_completed_detection_endpoint_contract(
        frozen_contract,
        source_endpoint_contract_hash=source_endpoint_contract_hash,
    )
    comparison_contract = (
        build_completed_detection_comparison_endpoint_contract(
            frozen_contract,
        )
    )
    verified = verify_frozen_postprocess_contract(frozen_contract)
    result = dict(frozen_result or {})
    try:
        completed = int(completed_frames)
        postprocess_completed = int(postprocess_completed_frames)
    except (TypeError, ValueError) as exc:
        raise FrozenPostprocessError(
            "completed_endpoint_frame_count_invalid"
        ) from exc
    result_artifact = result.get("completed_result_artifact")
    if result_artifact is not None:
        if not isinstance(result_artifact, Mapping):
            raise FrozenPostprocessError(
                "completed_endpoint_result_artifact_invalid"
            )
        artifact = dict(result_artifact)
        detections = _canonical_detection_records(
            artifact.get("detections"),
            max_detections=int(verified["max_detections"]),
        )
        expected_artifact = {
            "schema": FROZEN_COMPLETED_RESULT_ARTIFACT_SCHEMA,
            "schema_version": FROZEN_COMPLETED_RESULT_ARTIFACT_VERSION,
            "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
            "coordinate_space": "original_image_xyxy_pixels",
            "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
            "detections": detections,
        }
        artifact_hash = canonical_json_sha256(expected_artifact)
        if (
            artifact != expected_artifact
            or result.get("coordinate_space")
            != "original_image_xyxy_pixels"
            or result.get("record_schema")
            != CANONICAL_DETECTION_RECORD_SCHEMA
            or result.get("canonical_sort_policy")
            != CANONICAL_DETECTION_SORT_POLICY
            or result.get("detections") != detections
            or result.get("detection_count") != len(detections)
            or str(result.get("detections_sha256") or "").strip().lower()
            != canonical_json_sha256(detections)
            or str(
                result.get("completed_result_artifact_sha256") or ""
            ).strip().lower() != artifact_hash
        ):
            raise FrozenPostprocessError(
                "completed_endpoint_result_artifact_invalid"
            )
    if (
        completed <= 0
        or postprocess_completed != completed
        or result.get("task") != "detection"
        or result.get("contract_family") != "decoded_nms"
        or str(result.get("postprocess_contract_sha256") or "").strip().lower()
        != str(verified["contract_sha256"])
        or not isinstance(result.get("detection_count"), int)
        or int(result.get("detection_count") or 0) < 0
    ):
        raise FrozenPostprocessError(
            "completed_endpoint_runtime_attestation_invalid"
        )
    return {
        "schema": COMPLETED_ENDPOINT_ATTESTATION_SCHEMA,
        "schema_version": contract["schema_version"],
        "attested": True,
        "status": "passed",
        "task": "detection",
        "source_stage": verified["source_contract_family"],
        "stage": "decoded_nms",
        "endpoint": "decoded_nms",
        "contract_source": COMPLETED_ENDPOINT_SOURCE,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": contract["endpoint_contract_hash"],
        "output_endpoint_id": contract["output_endpoint_id"],
        "completed_frames": completed,
        "postprocess_completed_frames": postprocess_completed,
        "postprocess_completion_verified": True,
        "frozen_postprocess_contract_sha256": verified["contract_sha256"],
        "frozen_postprocess_result": result,
        "completed_endpoint_contract": contract,
        "completed_task_comparison_endpoint_contract": comparison_contract,
        "completed_task_comparison_endpoint_contract_hash": (
            comparison_contract["endpoint_contract_hash"]
        ),
        "completed_task_comparison_output_endpoint_id": (
            comparison_contract["output_endpoint_id"]
        ),
        "completed_task_completion_mode": "frozen_host_tail",
    }


def build_normalized_detection_endpoint_contract(
    direct_normalization_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the Direct-BN6 physical endpoint to its measured normalizer."""
    verified = verify_frozen_decoded_nms_normalization_contract(
        direct_normalization_contract
    )
    identity: dict[str, Any] = {
        "schema": DIRECT_NORMALIZED_ENDPOINT_SCHEMA,
        "schema_version": DIRECT_NORMALIZED_ENDPOINT_VERSION,
        "task": "detection",
        "source_stage": "decoded_nms",
        "completed_stage": "decoded_nms",
        "contract_source": DIRECT_NORMALIZED_ENDPOINT_SOURCE,
        "source_endpoint_contract_hash": verified[
            "source_endpoint_contract_hash"
        ],
        "source_output_endpoint_id": verified[
            "source_output_endpoint_id"
        ],
        "source_output_tensor_signature": verified[
            "source_output_tensor_signature"
        ],
        "frozen_decoded_nms_normalization_contract_sha256": verified[
            "contract_sha256"
        ],
        "letterbox_geometry_contract_sha256": verified[
            "letterbox_geometry_contract_sha256"
        ],
        "normalizer_id": verified["normalizer_id"],
        "nms_implementation": verified["nms_implementation"],
        "class_aware": verified["class_aware"],
        "confidence_threshold": verified["confidence_threshold"],
        "iou_threshold": verified["iou_threshold"],
        "max_detections": verified["max_detections"],
        "input_hw": verified["input_hw"],
        "original_wh": verified["original_wh"],
        "source_coordinate_space": verified["source_coordinate_space"],
        "output_coordinate_space": verified["output_coordinate_space"],
        "performance_scope": verified["performance_scope"],
        "energy_scope": verified["energy_scope"],
    }
    endpoint_hash = canonical_json_sha256(identity)
    return {
        **identity,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": f"detection:decoded_nms:{endpoint_hash}",
    }


def build_normalized_detection_endpoint_attestation(
    direct_normalization_contract: Mapping[str, Any],
    normalized_result: Mapping[str, Any],
    *,
    completed_frames: int,
    postprocess_completed_frames: int,
    allow_legacy_hash_only_v1: bool = False,
) -> dict[str, Any]:
    """Attest one Direct-BN6 normalization for every measured frame."""
    verified = verify_frozen_decoded_nms_normalization_contract(
        direct_normalization_contract
    )
    result = dict(normalized_result or {})
    if (
        isinstance(completed_frames, bool)
        or not isinstance(completed_frames, int)
        or isinstance(postprocess_completed_frames, bool)
        or not isinstance(postprocess_completed_frames, int)
    ):
        raise FrozenPostprocessError(
            "direct_normalization_completed_frame_count_invalid"
        )
    completed = completed_frames
    postprocess_completed = postprocess_completed_frames
    result_artifact = result.get("completed_result_artifact")
    legacy_hash_only_v1 = bool(
        allow_legacy_hash_only_v1
        and result_artifact is None
    )
    if not legacy_hash_only_v1 and not isinstance(
        result_artifact, Mapping
    ):
        raise FrozenPostprocessError(
            "direct_normalization_result_artifact_invalid"
        )
    artifact: dict[str, Any] = {}
    detections: list[dict[str, Any]] = []
    expected_artifact: dict[str, Any] = {}
    artifact_hash = ""
    if not legacy_hash_only_v1:
        artifact = dict(result_artifact)
        detections = _canonical_detection_records(
            artifact.get("detections"),
            max_detections=int(verified["max_detections"]),
        )
        expected_artifact = {
            "schema": FROZEN_COMPLETED_RESULT_ARTIFACT_SCHEMA,
            "schema_version": FROZEN_COMPLETED_RESULT_ARTIFACT_VERSION,
            "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
            "coordinate_space": "original_image_xyxy_pixels",
            "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
            "detections": detections,
        }
        artifact_hash = canonical_json_sha256(expected_artifact)
    detections_hash = _strict_sha256(
        result.get("detections_sha256"),
        reason="direct_normalization_result_sha256_invalid",
    )
    if (
        completed <= 0
        or postprocess_completed != completed
        or set(result) != (
            {
                "task", "contract_family", "decoder_format",
                "coordinate_space", "detection_count",
                "detections_sha256", "normalization_contract_sha256",
            }
            if legacy_hash_only_v1
            else {
                "task", "contract_family", "decoder_format",
                "coordinate_space", "detection_count",
                "detections_sha256", "detections", "record_schema",
                "canonical_sort_policy", "completed_result_artifact",
                "completed_result_artifact_sha256",
                "normalization_contract_sha256",
            }
        )
        or result.get("task") != "detection"
        or result.get("contract_family") != "decoded_nms"
        or result.get("decoder_format") != "bn6_detections"
        or str(
            result.get("normalization_contract_sha256") or ""
        ).strip().lower() != str(verified["contract_sha256"])
        or str(result.get("detections_sha256") or "").strip().lower()
        != detections_hash
        or (
            not legacy_hash_only_v1
            and (
                artifact != expected_artifact
                or result.get("record_schema")
                != CANONICAL_DETECTION_RECORD_SCHEMA
                or result.get("canonical_sort_policy")
                != CANONICAL_DETECTION_SORT_POLICY
                or result.get("detections") != detections
                or detections_hash != canonical_json_sha256(detections)
                or _strict_sha256(
                    result.get("completed_result_artifact_sha256"),
                    reason=(
                        "direct_normalization_result_artifact_sha256_invalid"
                    ),
                ) != artifact_hash
                or int(result.get("detection_count") or 0)
                != len(detections)
            )
        )
        or result.get("coordinate_space")
        != "original_image_xyxy_pixels"
        or isinstance(result.get("detection_count"), bool)
        or not isinstance(result.get("detection_count"), int)
        or int(result.get("detection_count") or 0) < 0
    ):
        raise FrozenPostprocessError(
            "direct_normalization_runtime_attestation_invalid"
        )
    endpoint_contract = build_normalized_detection_endpoint_contract(
        verified
    )
    comparison_contract = (
        build_completed_detection_comparison_endpoint_contract(
            direct_normalization_contract=verified,
        )
    )
    return {
        "schema": DIRECT_NORMALIZED_ATTESTATION_SCHEMA,
        "schema_version": DIRECT_NORMALIZED_ENDPOINT_VERSION,
        "attested": True,
        "status": "passed",
        "task": "detection",
        "source_stage": "decoded_nms",
        "stage": "decoded_nms",
        "endpoint": "decoded_nms",
        "contract_source": DIRECT_NORMALIZED_ENDPOINT_SOURCE,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_contract[
            "endpoint_contract_hash"
        ],
        "output_endpoint_id": endpoint_contract["output_endpoint_id"],
        "completed_frames": completed,
        "postprocess_completed_frames": postprocess_completed,
        "postprocess_completion_verified": True,
        "normalization_completion_verified": True,
        "frozen_decoded_nms_normalization_contract_sha256": verified[
            "contract_sha256"
        ],
        "frozen_decoded_nms_normalization_result": result,
        "completed_endpoint_contract": endpoint_contract,
        "completed_task_comparison_endpoint_contract": comparison_contract,
        "completed_task_comparison_endpoint_contract_hash": (
            comparison_contract["endpoint_contract_hash"]
        ),
        "completed_task_comparison_output_endpoint_id": (
            comparison_contract["output_endpoint_id"]
        ),
        "completed_task_completion_mode": (
            "integrated_accelerator_plus_frozen_normalization"
        ),
    }


class FrozenDetectionPostprocessor:
    """Thread-safe single-worker decoder used inside measured hotloops."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = verify_frozen_postprocess_contract(contract)
        if self.contract.get(
            "legacy_activation_strategy_unbound"
        ) is True:
            raise FrozenPostprocessError(
                "legacy_unbound_activation_contract_nonclaim_runtime"
            )
        if self.contract.get("legacy_decoder_contract_unbound") is True:
            raise FrozenPostprocessError(
                "legacy_unbound_yolov7_decoder_contract_nonruntime"
            )
        model_bound = self.contract.get("model_bound_decoder_contract")
        self._harness = YoloHarness(
            conf_thresh=float(self.contract["confidence_threshold"]),
            iou_thresh=float(self.contract["iou_threshold"]),
            max_det=int(self.contract["max_detections"]),
            multiscale_activation_mode=self.contract.get(
                "multiscale_activation_mode"
            ),
            model_id=str(self.contract.get("model_id") or ""),
            multiscale_decoder_contract=(
                model_bound if isinstance(model_bound, Mapping) else None
            ),
        )
        self._lock = threading.Lock()
        self.completed_count = 0
        self.last_result: dict[str, Any] = {}
        self.last_detections: list[dict[str, Any]] = []

    def process(
        self, outputs: Mapping[str, Any], *, original_wh: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Decode one output and count it only after decode and NMS succeed."""
        with self._lock:
            # Verify code/contract and tensor identity; the declared decoded
            # values are checked once by preparation below, not twice.
            verify_frozen_postprocess_contract(self.contract)
            if tensor_signature(outputs) != self.contract["raw_output_tensor_signature"]:
                raise FrozenPostprocessError("frozen_postprocess_tensor_signature_mismatch")
            processing_outputs = dict(outputs)
            numeric_observation = None
            if self.contract["source_contract_family"] == "decoded_pre_nms":
                processing_outputs, numeric_observation = prepare_decoded_pre_nms_outputs(
                    outputs, score_policy=self.contract.get("decoded_pre_nms_score_policy"),
                )
            context: dict[str, Any] = {
                "input_hw": list(self.contract["input_hw"]),
                "variant": "native_full",
            }
            values = (
                [int(value) for value in original_wh]
                if original_wh is not None else list(self.contract["original_wh"])
            )
            if (
                len(values) != 2 or any(value <= 0 for value in values)
                or values != list(self.contract["original_wh"])
            ):
                raise FrozenPostprocessError("frozen_postprocess_original_wh_mismatch")
            context["original_wh"] = values
            payload = _result_json(self._harness.postprocess(processing_outputs, context))
            if str(payload.get("format") or "") != str(self.contract["decoder_format"]):
                raise FrozenPostprocessError("frozen_postprocess_runtime_format_mismatch")
            detections = payload.get("detections")
            if not isinstance(detections, list):
                raise FrozenPostprocessError("frozen_postprocess_detections_missing")
            canonical = _canonical_detection_records(
                detections,
                max_detections=int(self.contract["max_detections"]),
            )
            completed_result_artifact = {
                "schema": FROZEN_COMPLETED_RESULT_ARTIFACT_SCHEMA,
                "schema_version": FROZEN_COMPLETED_RESULT_ARTIFACT_VERSION,
                "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
                "coordinate_space": "original_image_xyxy_pixels",
                "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
                "detections": canonical,
            }
            self.completed_count += 1
            self.last_detections = canonical
            self.last_result = {
                "task": "detection",
                "contract_family": "decoded_nms",
                "decoder_format": str(payload.get("format") or ""),
                "coordinate_space": "original_image_xyxy_pixels",
                "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
                "canonical_sort_policy": CANONICAL_DETECTION_SORT_POLICY,
                "detection_count": len(self.last_detections),
                "detections_sha256": canonical_json_sha256(self.last_detections),
                "detections": self.last_detections,
                "completed_result_artifact": completed_result_artifact,
                "completed_result_artifact_sha256": canonical_json_sha256(
                    completed_result_artifact
                ),
                "postprocess_contract_sha256": str(self.contract["contract_sha256"]),
            }
            if numeric_observation is not None:
                self.last_result["decoded_pre_nms_score_normalization"] = numeric_observation
            if isinstance(
                self.contract.get("model_bound_decoder_contract"), Mapping
            ):
                self.last_result.update({
                    "model_sha256": str(self.contract["model_sha256"]),
                    "decoder_id": str(self.contract["decoder_id"]),
                    "model_bound_decoder_contract_sha256": str(
                        self.contract[
                            "model_bound_decoder_contract_sha256"
                        ]
                    ),
                    "anchor_table_id": str(
                        self.contract["anchor_table_id"]
                    ),
                    "anchors_by_stride": list(
                        self.contract["anchors_by_stride"]
                    ),
                })
            return dict(self.last_result)


class FrozenDecodedNmsPostprocessor:
    """Canonical Direct-BN6 filter/NMS/deletterbox inside measured loops."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = verify_frozen_decoded_nms_normalization_contract(
            contract
        )
        self._harness = YoloHarness(
            conf_thresh=float(self.contract["confidence_threshold"]),
            iou_thresh=float(self.contract["iou_threshold"]),
            max_det=int(self.contract["max_detections"]),
        )
        self._lock = threading.Lock()
        self.completed_count = 0
        self.last_result: dict[str, Any] = {}
        self.last_detections: list[dict[str, Any]] = []

    def process(
        self,
        outputs: Mapping[str, Any],
        *,
        original_wh: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if tensor_signature(outputs) != dict(
                self.contract["source_output_tensor_signature"]
            ):
                raise FrozenPostprocessError(
                    "direct_normalization_tensor_signature_mismatch"
                )
            values = (
                [int(value) for value in original_wh]
                if original_wh is not None
                else list(self.contract["original_wh"])
            )
            if values != list(self.contract["original_wh"]):
                raise FrozenPostprocessError(
                    "direct_normalization_original_wh_mismatch"
                )
            payload = _result_json(self._harness.postprocess(
                dict(outputs),
                {
                    "input_hw": list(self.contract["input_hw"]),
                    "original_wh": values,
                    "variant": "native_full_direct_bn6",
                },
            ))
            if str(payload.get("format") or "") != "bn6_detections":
                raise FrozenPostprocessError(
                    "direct_normalization_runtime_format_mismatch"
                )
            detections = payload.get("detections")
            if not isinstance(detections, list):
                raise FrozenPostprocessError(
                    "direct_normalization_detections_missing"
                )
            canonical = _canonical_detection_records(
                detections,
                max_detections=int(self.contract["max_detections"]),
            )
            completed_result_artifact = {
                "schema": FROZEN_COMPLETED_RESULT_ARTIFACT_SCHEMA,
                "schema_version": FROZEN_COMPLETED_RESULT_ARTIFACT_VERSION,
                "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
                "coordinate_space": "original_image_xyxy_pixels",
                "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
                "detections": canonical,
            }
            self.completed_count += 1
            self.last_detections = canonical
            self.last_result = {
                "task": "detection",
                "contract_family": "decoded_nms",
                "decoder_format": "bn6_detections",
                "coordinate_space": "original_image_xyxy_pixels",
                "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
                "canonical_sort_policy": CANONICAL_DETECTION_SORT_POLICY,
                "detection_count": len(self.last_detections),
                "detections_sha256": canonical_json_sha256(
                    self.last_detections
                ),
                "detections": self.last_detections,
                "completed_result_artifact": completed_result_artifact,
                "completed_result_artifact_sha256": canonical_json_sha256(
                    completed_result_artifact
                ),
                "normalization_contract_sha256": str(
                    self.contract["contract_sha256"]
                ),
            }
            return dict(self.last_result)


def _yolov7_head_layout_candidates(
    value: Any,
    *,
    expected_grids: Mapping[int, tuple[int, int]],
) -> list[tuple[int, str, np.ndarray]]:
    """Return every valid interpretation of one YOLOv7 output tensor.

    Multiple matches are deliberately retained.  The caller rejects them as an
    ambiguous physical layout instead of choosing whichever branch happens to
    appear first.
    """

    array = np.asarray(value)
    if array.size <= 0 or array.dtype.kind not in "fiu":
        raise FrozenPostprocessError("yolov7_head_tensor_invalid")
    candidates: list[tuple[str, np.ndarray]] = []
    if array.ndim == 5:
        if (
            int(array.shape[0]) == 1
            and int(array.shape[1]) == 3
            and int(array.shape[-1]) == 85
        ):
            candidates.append(("b_a_h_w_c", array))
        if (
            int(array.shape[0]) == 1
            and int(array.shape[-2]) == 3
            and int(array.shape[-1]) == 85
        ):
            candidates.append(("b_h_w_a_c", array.transpose(0, 3, 1, 2, 4)))
    elif array.ndim == 4:
        if (
            int(array.shape[0]) == 1
            and int(array.shape[1]) == 255
        ):
            batch, channels, grid_h, grid_w = array.shape
            candidates.append((
                "b_ac_h_w",
                array.reshape(batch, 3, channels // 3, grid_h, grid_w)
                .transpose(0, 1, 3, 4, 2),
            ))
        if (
            int(array.shape[0]) == 1
            and int(array.shape[-1]) == 255
        ):
            batch, grid_h, grid_w, channels = array.shape
            candidates.append((
                "b_h_w_ac",
                array.reshape(batch, grid_h, grid_w, 3, channels // 3)
                .transpose(0, 3, 1, 2, 4),
            ))
        if (
            int(array.shape[0]) == 3
            and int(array.shape[-1]) == 85
        ):
            candidates.append(("a_h_w_c", array[None, ...]))
        if (
            int(array.shape[-2]) == 3
            and int(array.shape[-1]) == 85
        ):
            candidates.append((
                "h_w_a_c",
                array.transpose(2, 0, 1, 3)[None, ...],
            ))
    elif array.ndim == 3:
        if int(array.shape[-1]) == 255:
            grid_h, grid_w, channels = array.shape
            candidates.append((
                "h_w_ac",
                array.reshape(grid_h, grid_w, 3, channels // 3)
                .transpose(2, 0, 1, 3)[None, ...],
            ))

    matched: list[tuple[int, str, np.ndarray]] = []
    for layout, canonical in candidates:
        shape = tuple(int(dim) for dim in canonical.shape)
        if len(shape) != 5 or shape[0] != 1 or shape[1] != 3 or shape[-1] != 85:
            continue
        for stride, grid in expected_grids.items():
            if shape[2:4] == tuple(grid):
                matched.append((
                    int(stride),
                    layout,
                    np.ascontiguousarray(canonical, dtype=np.float32),
                ))
    return matched


def _canonicalize_yolov7_heads(
    outputs: Mapping[str, Any],
    *,
    input_hw: Sequence[int],
    activation_mode: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not isinstance(outputs, Mapping) or len(outputs) != 3:
        raise FrozenPostprocessError(
            "yolov7_head_mapping_exactly_three_outputs_required"
        )
    input_values = _strict_positive_int_list(
        list(input_hw) if isinstance(input_hw, (list, tuple)) else input_hw,
        length=2,
        reason="yolov7_head_mapping_input_hw_invalid",
    )
    input_height, input_width = input_values
    if any(
        input_height % stride != 0 or input_width % stride != 0
        for stride in YOLOV7_EXPECTED_STRIDES
    ):
        raise FrozenPostprocessError(
            "yolov7_head_mapping_input_hw_not_divisible_by_strides"
        )
    expected_grids = {
        stride: (input_height // stride, input_width // stride)
        for stride in YOLOV7_EXPECTED_STRIDES
    }

    by_stride: dict[int, np.ndarray] = {}
    for _name, value in outputs.items():
        matches = _yolov7_head_layout_candidates(
            value,
            expected_grids=expected_grids,
        )
        if not matches:
            raise FrozenPostprocessError(
                "yolov7_head_mapping_geometry_or_semantics_invalid"
            )
        if len(matches) != 1:
            raise FrozenPostprocessError(
                "yolov7_head_mapping_layout_ambiguous"
            )
        stride, _layout, canonical = matches[0]
        if stride in by_stride:
            raise FrozenPostprocessError(
                f"yolov7_head_mapping_duplicate_stride:{stride}"
            )
        if not bool(np.isfinite(canonical).all()):
            raise FrozenPostprocessError(
                "yolov7_head_mapping_nonfinite_tensor"
            )
        by_stride[stride] = canonical
    if set(by_stride) != set(YOLOV7_EXPECTED_STRIDES):
        raise FrozenPostprocessError(
            "yolov7_head_mapping_missing_required_stride"
        )

    harness_anchors = getattr(
        _yolo_module, "YOLOV7_STANDARD_ANCHORS_640", {},
    )
    for stride, anchors in YOLOV7_ANCHORS_BY_STRIDE.items():
        observed = harness_anchors.get(stride)
        if observed is None or not np.array_equal(
            np.asarray(observed, dtype=np.float32),
            np.asarray(anchors, dtype=np.float32),
        ):
            raise FrozenPostprocessError(
                f"yolov7_anchor_table_implementation_mismatch:{stride}"
            )

    heads = []
    canonical_outputs: dict[str, np.ndarray] = {}
    for role_index, stride in enumerate(YOLOV7_EXPECTED_STRIDES, start=3):
        grid_h, grid_w = expected_grids[stride]
        canonical = by_stride[stride]
        canonical_outputs[f"yolov7_stride_{stride}"] = canonical
        heads.append({
            "role": f"p{role_index}",
            "stride": int(stride),
            "grid_hw": [int(grid_h), int(grid_w)],
            "canonical_shape": [1, 3, int(grid_h), int(grid_w), 85],
            "anchor_wh": [
                [int(width), int(height)]
                for width, height in YOLOV7_ANCHORS_BY_STRIDE[stride]
            ],
        })
    activation_mode = (
        str(activation_mode).strip().lower()
        if activation_mode is not None
        else str(
            _yolo_module.infer_yolov7_activation_mode(canonical_outputs)
        )
    )
    activation_semantics = {
        "logits": (
            "batch_anchor_grid_y_grid_x_xywh_objectness_class_logits",
            "sigmoid_objectness_times_sigmoid_class",
        ),
        "activated": (
            "batch_anchor_grid_y_grid_x_xywh_objectness_class_probabilities",
            "objectness_probability_times_class_probability",
        ),
        "objcls_activated": (
            "batch_anchor_grid_y_grid_x_xywh_logits_objectness_class_probabilities",
            "objectness_probability_times_class_probability",
        ),
    }
    if activation_mode not in activation_semantics:
        raise FrozenPostprocessError(
            "yolov7_head_mapping_activation_mode_invalid"
        )
    record_semantics, combination_semantics = activation_semantics[
        activation_mode
    ]
    identity: dict[str, Any] = {
        "schema": YOLOV7_HEAD_MAPPING_SCHEMA,
        "schema_version": YOLOV7_HEAD_MAPPING_VERSION,
        "model_family": "yolov7",
        "mapping_policy": (
            "validated_grid_and_stride_without_name_or_index_v1"
        ),
        "input_hw": input_values,
        "expected_strides": list(YOLOV7_EXPECTED_STRIDES),
        "anchor_table_id": YOLOV7_STANDARD_ANCHOR_TABLE_ID,
        "activation_mode": activation_mode,
        "head_record_semantics": record_semantics,
        "objectness_class_combination": combination_semantics,
        "heads": heads,
    }
    mapping = {
        **identity,
        "head_mapping_sha256": canonical_json_sha256(identity),
    }
    return canonical_outputs, mapping


def build_yolov7_head_mapping(
    outputs: Mapping[str, Any],
    *,
    input_hw: Sequence[int],
) -> dict[str, Any]:
    """Build a backend-name- and output-order-neutral YOLOv7 head binding."""

    _canonical, mapping = _canonicalize_yolov7_heads(
        outputs,
        input_hw=input_hw,
    )
    return mapping


def verify_yolov7_head_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError("yolov7_head_mapping_missing")
    mapping = dict(raw)
    declared = _strict_sha256(
        mapping.pop("head_mapping_sha256", ""),
        reason="yolov7_head_mapping_sha256_invalid",
    )
    if canonical_json_sha256(mapping) != declared:
        raise FrozenPostprocessError(
            "yolov7_head_mapping_sha256_mismatch"
        )
    common_expected_keys = {
        "schema", "schema_version", "model_family", "mapping_policy",
        "input_hw", "expected_strides", "anchor_table_id",
        "head_record_semantics", "objectness_class_combination", "heads",
    }
    heads = mapping.get("heads")
    mapping_version = mapping.get("schema_version")
    if mapping_version == YOLOV7_HEAD_MAPPING_VERSION:
        expected_keys = common_expected_keys | {"activation_mode"}
        expected_anchor_table_id = YOLOV7_STANDARD_ANCHOR_TABLE_ID
        expected_anchors = YOLOV7_ANCHORS_BY_STRIDE
        activation_mode = str(mapping.get("activation_mode") or "")
        activation_semantics = {
            "logits": (
                "batch_anchor_grid_y_grid_x_xywh_objectness_class_logits",
                "sigmoid_objectness_times_sigmoid_class",
            ),
            "activated": (
                "batch_anchor_grid_y_grid_x_xywh_objectness_class_probabilities",
                "objectness_probability_times_class_probability",
            ),
            "objcls_activated": (
                "batch_anchor_grid_y_grid_x_xywh_logits_objectness_class_probabilities",
                "objectness_probability_times_class_probability",
            ),
        }
        expected_semantics = activation_semantics.get(activation_mode)
    elif mapping_version == YOLOV7_HEAD_MAPPING_LEGACY_VERSION:
        expected_keys = common_expected_keys
        expected_anchor_table_id = "yolov5_yolov7_640_anchor_table_v1"
        expected_anchors = YOLOV7_LEGACY_TINY_ANCHORS_BY_STRIDE
        expected_semantics = (
            "batch_anchor_grid_y_grid_x_xywh_objectness_class_logits",
            "sigmoid_objectness_times_sigmoid_class",
        )
    else:
        raise FrozenPostprocessError("yolov7_head_mapping_fields_invalid")
    if (
        set(mapping) != expected_keys
        or mapping.get("schema") != YOLOV7_HEAD_MAPPING_SCHEMA
        or mapping.get("schema_version") not in {
            YOLOV7_HEAD_MAPPING_LEGACY_VERSION,
            YOLOV7_HEAD_MAPPING_VERSION,
        }
        or mapping.get("model_family") != "yolov7"
        or mapping.get("mapping_policy")
        != "validated_grid_and_stride_without_name_or_index_v1"
        or mapping.get("expected_strides") != list(YOLOV7_EXPECTED_STRIDES)
        or mapping.get("anchor_table_id") != expected_anchor_table_id
        or expected_semantics is None
        or mapping.get("head_record_semantics") != expected_semantics[0]
        or mapping.get("objectness_class_combination") != expected_semantics[1]
        or not isinstance(heads, list)
        or len(heads) != 3
    ):
        raise FrozenPostprocessError("yolov7_head_mapping_fields_invalid")
    input_hw = _strict_positive_int_list(
        mapping.get("input_hw"),
        length=2,
        reason="yolov7_head_mapping_fields_invalid",
    )
    expected_heads = []
    for role_index, stride in enumerate(YOLOV7_EXPECTED_STRIDES, start=3):
        expected_heads.append({
            "role": f"p{role_index}",
            "stride": stride,
            "grid_hw": [input_hw[0] // stride, input_hw[1] // stride],
            "canonical_shape": [
                1, 3, input_hw[0] // stride, input_hw[1] // stride, 85,
            ],
            "anchor_wh": [
                [width, height]
                for width, height in expected_anchors[stride]
            ],
        })
    if (
        any(value % 32 != 0 for value in input_hw)
        or heads != expected_heads
    ):
        raise FrozenPostprocessError("yolov7_head_mapping_fields_invalid")
    return {**mapping, "head_mapping_sha256": declared}


def _canonical_detection_records(
    raw: Any,
    *,
    max_detections: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise FrozenPostprocessError(
            "canonical_detection_records_missing"
        )
    records: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise FrozenPostprocessError(
                "canonical_detection_record_invalid"
            )
        try:
            class_value = value.get("class_id")
            if isinstance(class_value, bool):
                raise TypeError("bool class id")
            class_float = float(class_value)
            class_id = int(class_float)
            score = float(value.get("score"))
            coordinates = [
                float(value.get(key))
                for key in ("x1", "y1", "x2", "y2")
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrozenPostprocessError(
                "canonical_detection_record_invalid"
            ) from exc
        if (
            not math.isfinite(class_float)
            or class_float != float(class_id)
            or class_id < 0
            or not math.isfinite(score)
            or score < 0.0
            or score > 1.0
            or any(not math.isfinite(item) for item in coordinates)
            or coordinates[2] < coordinates[0]
            or coordinates[3] < coordinates[1]
        ):
            raise FrozenPostprocessError(
                "canonical_detection_record_invalid"
            )
        normalized_coordinates = [
            0.0 if item == 0.0 else item
            for item in coordinates
        ]
        records.append({
            "class_id": class_id,
            "score": 0.0 if score == 0.0 else score,
            "x1": normalized_coordinates[0],
            "y1": normalized_coordinates[1],
            "x2": normalized_coordinates[2],
            "y2": normalized_coordinates[3],
        })
    records.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["class_id"]),
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        )
    )
    if len(records) > int(max_detections):
        raise FrozenPostprocessError(
            "canonical_detection_count_exceeds_contract"
        )
    return records


def _tensor_content_sha256(outputs: Mapping[str, Any]) -> str:
    tensors = []
    for ordinal, value in enumerate(outputs.values()):
        array = np.ascontiguousarray(np.asarray(value))
        if array.size <= 0 or array.dtype.kind not in "fiu":
            raise FrozenPostprocessError(
                "completion_source_tensor_content_invalid"
            )
        tensors.append({
            "ordinal": ordinal,
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(array.dtype),
            "payload_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        })
    return canonical_json_sha256({
        "schema": "onnx-splitpoint/canonical-tensor-content",
        "schema_version": 1,
        "tensors": tensors,
    })


def build_attested_decoded_nms_materialization_contract(
    *,
    model_id: str,
    outputs: Mapping[str, Any],
    input_hw: Sequence[int],
    original_wh: Sequence[int],
    preprocess: Mapping[str, Any],
    source_endpoint_contract_hash: str,
    source_output_endpoint_attestation: Mapping[str, Any],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> dict[str, Any]:
    """Seal normalization of an endpoint that already performed decode and NMS.

    The measured host work may remove declared low-score padding, project the
    already-decoded boxes back to original-image coordinates, sort records, and
    materialize them.  It must never decode boxes or apply NMS a second time.
    """

    if (
        float(confidence_threshold) != 0.25
        or float(iou_threshold) != 0.45
        or int(max_detections) != 300
    ):
        raise FrozenPostprocessError(
            "decoded_nms_materialization_canonical_policy_required"
        )
    signature = tensor_signature(outputs)
    tensors = signature.get("tensors") or []
    if (
        len(tensors) != 1
        or tensors[0].get("rank") != 3
        or (tensors[0].get("shape") or [])[-1:] != [6]
        or (tensors[0].get("shape") or [0])[0] != 1
    ):
        raise FrozenPostprocessError(
            "decoded_nms_materialization_exact_bn6_required"
        )
    source_hash = _strict_sha256(
        source_endpoint_contract_hash,
        reason="decoded_nms_materialization_source_hash_invalid",
    )
    source_attestation = _verify_decoded_nms_source_attestation(
        source_output_endpoint_attestation,
        outputs=outputs,
        source_endpoint_contract_hash=source_hash,
        model_id=model_id,
    )
    geometry = build_letterbox_geometry_contract(
        input_hw=input_hw,
        original_wh=original_wh,
        preprocess=preprocess,
    )
    contract: dict[str, Any] = {
        "schema": DECODED_NMS_MATERIALIZATION_SCHEMA,
        "schema_version": DECODED_NMS_MATERIALIZATION_VERSION,
        "task": "detection",
        "model_id": str(model_id),
        "model_family": _model_family(model_id),
        "source_stage": "decoded_nms",
        "completed_stage": "decoded_nms",
        "source_output_format": "bn6_detections",
        "source_coordinate_space": (
            "model_input_letterbox_xyxy_pixels"
        ),
        "output_record_format": CANONICAL_DETECTION_RECORD_SCHEMA,
        "output_coordinate_space": "original_image_xyxy_pixels",
        "normalizer_id": NO_SECOND_NMS_NORMALIZER_ID,
        "source_nms_attested": True,
        "host_nms_applied": False,
        "padding_filter_policy": "score_below_declared_threshold_v1",
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
        "max_detections": int(max_detections),
        "canonical_sort_policy": CANONICAL_DETECTION_SORT_POLICY,
        "input_hw": list(geometry["input_hw"]),
        "original_wh": list(geometry["original_wh"]),
        "letterbox_geometry_contract": geometry,
        "letterbox_geometry_contract_sha256": str(
            geometry["geometry_contract_sha256"]
        ),
        "source_output_tensor_signature": signature,
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_id": (
            f"detection:decoded_nms:{source_hash}"
        ),
        "source_output_endpoint_attestation_sha256": (
            canonical_json_sha256(source_attestation)
        ),
        "implementation_artifacts": _implementation_artifacts(),
        "execution_policy": (
            "one_serial_no_second_nms_materialization_per_source_completion"
        ),
        "performance_scope": (
            "prepared_input_to_attested_decoded_nms_original_coordinates"
        ),
        "energy_scope": (
            "same_completion_execution_contract_inside_energy_hotloop"
        ),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def verify_attested_decoded_nms_materialization_contract(
    raw: Any,
    *,
    outputs: Mapping[str, Any] | None = None,
    source_output_endpoint_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "decoded_nms_materialization_contract_missing"
        )
    contract = dict(raw)
    declared = _strict_sha256(
        contract.pop("contract_sha256", ""),
        reason="decoded_nms_materialization_contract_sha256_invalid",
    )
    if canonical_json_sha256(contract) != declared:
        raise FrozenPostprocessError(
            "decoded_nms_materialization_contract_sha256_mismatch"
        )
    expected_keys = {
        "schema", "schema_version", "task", "model_id", "model_family",
        "source_stage", "completed_stage", "source_output_format",
        "source_coordinate_space", "output_record_format",
        "output_coordinate_space", "normalizer_id",
        "source_nms_attested", "host_nms_applied",
        "padding_filter_policy", "confidence_threshold", "iou_threshold",
        "max_detections", "canonical_sort_policy", "input_hw",
        "original_wh", "letterbox_geometry_contract",
        "letterbox_geometry_contract_sha256",
        "source_output_tensor_signature", "source_endpoint_contract_hash",
        "source_output_endpoint_id",
        "source_output_endpoint_attestation_sha256",
        "implementation_artifacts", "execution_policy",
        "performance_scope", "energy_scope",
    }
    geometry = verify_letterbox_geometry_contract(
        contract.get("letterbox_geometry_contract")
    )
    signature = _verify_tensor_signature_structure(
        contract.get("source_output_tensor_signature"),
        reason="decoded_nms_materialization_tensor_signature_invalid",
    )
    source_hash = _strict_sha256(
        contract.get("source_endpoint_contract_hash"),
        reason="decoded_nms_materialization_source_hash_invalid",
    )
    attestation_hash = _strict_sha256(
        contract.get("source_output_endpoint_attestation_sha256"),
        reason="decoded_nms_materialization_source_attestation_hash_invalid",
    )
    model_id = contract.get("model_id")
    if (
        set(contract) != expected_keys
        or contract.get("schema") != DECODED_NMS_MATERIALIZATION_SCHEMA
        or contract.get("schema_version")
        != DECODED_NMS_MATERIALIZATION_VERSION
        or contract.get("task") != "detection"
        or not isinstance(model_id, str)
        or not model_id
        or contract.get("model_family") != _model_family(model_id)
        or contract.get("source_stage") != "decoded_nms"
        or contract.get("completed_stage") != "decoded_nms"
        or contract.get("source_output_format") != "bn6_detections"
        or contract.get("source_coordinate_space")
        != "model_input_letterbox_xyxy_pixels"
        or contract.get("output_record_format")
        != CANONICAL_DETECTION_RECORD_SCHEMA
        or contract.get("output_coordinate_space")
        != "original_image_xyxy_pixels"
        or contract.get("normalizer_id") != NO_SECOND_NMS_NORMALIZER_ID
        or contract.get("source_nms_attested") is not True
        or contract.get("host_nms_applied") is not False
        or contract.get("padding_filter_policy")
        != "score_below_declared_threshold_v1"
        or float(contract.get("confidence_threshold") or -1.0) != 0.25
        or float(contract.get("iou_threshold") or -1.0) != 0.45
        or int(contract.get("max_detections") or 0) != 300
        or contract.get("canonical_sort_policy")
        != CANONICAL_DETECTION_SORT_POLICY
        or contract.get("input_hw") != geometry["input_hw"]
        or contract.get("original_wh") != geometry["original_wh"]
        or contract.get("letterbox_geometry_contract_sha256")
        != geometry["geometry_contract_sha256"]
        or contract.get("source_output_endpoint_id")
        != f"detection:decoded_nms:{source_hash}"
        or str(
            contract.get("source_output_endpoint_attestation_sha256") or ""
        ).strip().lower() != attestation_hash
        or contract.get("execution_policy")
        != "one_serial_no_second_nms_materialization_per_source_completion"
        or contract.get("performance_scope")
        != "prepared_input_to_attested_decoded_nms_original_coordinates"
        or contract.get("energy_scope")
        != "same_completion_execution_contract_inside_energy_hotloop"
        or len(signature.get("tensors") or []) != 1
        or signature["tensors"][0].get("rank") != 3
        or (signature["tensors"][0].get("shape") or [])[-1:] != [6]
        or (signature["tensors"][0].get("shape") or [0])[0] != 1
    ):
        raise FrozenPostprocessError(
            "decoded_nms_materialization_contract_fields_invalid"
        )
    if not _implementation_artifacts_match(
        contract.get("implementation_artifacts"),
        model_family=str(contract.get("model_family") or ""),
    ):
        raise FrozenPostprocessError(
            "decoded_nms_materialization_implementation_mismatch"
        )
    if outputs is not None:
        if tensor_signature(outputs) != signature:
            raise FrozenPostprocessError(
                "decoded_nms_materialization_tensor_signature_mismatch"
            )
        if source_output_endpoint_attestation is None:
            raise FrozenPostprocessError(
                "decoded_nms_materialization_source_attestation_missing"
            )
        verified_source = _verify_decoded_nms_source_attestation(
            source_output_endpoint_attestation,
            outputs=outputs,
            source_endpoint_contract_hash=source_hash,
            model_id=model_id,
        )
        if canonical_json_sha256(verified_source) != attestation_hash:
            raise FrozenPostprocessError(
                "decoded_nms_materialization_source_attestation_mismatch"
            )
    return {**contract, "contract_sha256": declared}


class _AttestedDecodedNmsMaterializer:
    """Materialize authoritative BN6 output without decoding or re-running NMS."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = (
            verify_attested_decoded_nms_materialization_contract(contract)
        )
        self.completed_count = 0
        self.last_result: dict[str, Any] = {}
        self.last_detections: list[dict[str, Any]] = []

    def process(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        if tensor_signature(outputs) != self.contract[
            "source_output_tensor_signature"
        ]:
            raise FrozenPostprocessError(
                "decoded_nms_materialization_tensor_signature_mismatch"
            )
        array = np.asarray(next(iter(outputs.values())))
        rows = np.asarray(array, dtype=np.float64).reshape(-1, 6)
        if not bool(np.isfinite(rows).all()):
            raise FrozenPostprocessError(
                "decoded_nms_materialization_nonfinite_values"
            )
        scores = rows[:, 4]
        classes = rows[:, 5]
        if (
            bool(np.any(scores < 0.0))
            or bool(np.any(scores > 1.0))
            or bool(np.any(classes < 0.0))
            or not bool(np.equal(classes, np.rint(classes)).all())
            or bool(np.any(rows[:, 2] < rows[:, 0]))
            or bool(np.any(rows[:, 3] < rows[:, 1]))
        ):
            raise FrozenPostprocessError(
                "decoded_nms_materialization_source_values_invalid"
            )

        kept = rows[
            scores >= float(self.contract["confidence_threshold"])
        ]
        if len(kept) > int(self.contract["max_detections"]):
            raise FrozenPostprocessError(
                "decoded_nms_materialization_count_exceeds_source_contract"
            )
        geometry = self.contract["letterbox_geometry_contract"]
        gain = float(geometry["gain"])
        pad_left = float(geometry["pad_left"])
        pad_top = float(geometry["pad_top"])
        original_width, original_height = (
            int(value) for value in geometry["original_wh"]
        )
        detections: list[dict[str, Any]] = []
        for row in kept:
            x1 = min(
                float(original_width),
                max(0.0, (float(row[0]) - pad_left) / gain),
            )
            y1 = min(
                float(original_height),
                max(0.0, (float(row[1]) - pad_top) / gain),
            )
            x2 = min(
                float(original_width),
                max(0.0, (float(row[2]) - pad_left) / gain),
            )
            y2 = min(
                float(original_height),
                max(0.0, (float(row[3]) - pad_top) / gain),
            )
            if x2 < x1 or y2 < y1:
                raise FrozenPostprocessError(
                    "decoded_nms_materialization_inverse_geometry_invalid"
                )
            detections.append({
                "class_id": int(row[5]),
                "score": float(row[4]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })
        canonical = _canonical_detection_records(
            detections,
            max_detections=int(self.contract["max_detections"]),
        )
        result = {
            "task": "detection",
            "contract_family": "decoded_nms",
            "decoder_format": "bn6_detections",
            "coordinate_space": "original_image_xyxy_pixels",
            "detection_count": len(canonical),
            "detections_sha256": canonical_json_sha256(canonical),
            "materialization_contract_sha256": str(
                self.contract["contract_sha256"]
            ),
            "source_nms_attested": True,
            "host_nms_applied": False,
        }
        self.last_detections = canonical
        self.last_result = result
        self.completed_count += 1
        return dict(result)


def _verified_completion_source_endpoint(
    raw: Any,
    *,
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "completion_source_endpoint_contract_missing"
        )
    source = dict(raw)
    signature = _verify_tensor_signature_structure(
        source.get("tensor_signature"),
        reason="completion_source_endpoint_tensor_signature_invalid",
    )
    observed_signature = tensor_signature(outputs)
    endpoint_hash = _strict_sha256(
        source.get("endpoint_contract_hash"),
        reason="completion_source_endpoint_hash_invalid",
    )
    stage = str(source.get("stage") or "").strip().lower()
    family = str(source.get("contract_family") or "").strip().lower()
    attestation = source.get("output_endpoint_attestation")
    output_endpoint_id = (
        f"detection:{stage}:{endpoint_hash}"
        if stage in {"raw_head", "decoded_pre_nms", "decoded_nms"} else ""
    )
    declared_id = str(source.get("output_endpoint_id") or "").strip()
    model_hash_values = {
        str(source.get(key) or "").strip().lower().removeprefix("sha256:")
        for key in (
            "model_sha256", "full_model_sha256", "terminal_model_sha256",
        )
        if str(source.get(key) or "").strip()
    }
    declared_contract = (
        attestation.get("declared_contract")
        if isinstance(attestation, Mapping)
        and isinstance(attestation.get("declared_contract"), Mapping)
        else {}
    )
    for key in (
        "model_sha256", "full_model_sha256", "terminal_model_sha256",
    ):
        value = str(declared_contract.get(key) or "").strip().lower()
        if value:
            model_hash_values.add(value.removeprefix("sha256:"))
    if len(model_hash_values) > 1:
        raise FrozenPostprocessError(
            "completion_source_endpoint_model_sha256_conflict"
        )
    model_sha256 = next(iter(model_hash_values), "")
    if model_sha256:
        _strict_sha256(
            model_sha256,
            reason="completion_source_endpoint_model_sha256_invalid",
        )
    if (
        source.get("task") != "detection"
        or stage not in {"raw_head", "decoded_pre_nms", "decoded_nms"}
        or family != stage
        or source.get("endpoint_contract_complete") is not True
        or signature != observed_signature
        or not isinstance(attestation, Mapping)
        or attestation.get("attested") is not True
        or str(attestation.get("status") or "").strip().lower() != "passed"
        or str(attestation.get("stage") or "").strip().lower() != stage
        or str(attestation.get("endpoint") or "").strip().lower() != stage
        or str(attestation.get("endpoint_contract_hash") or "").strip().lower()
        != endpoint_hash
        or (declared_id and declared_id != output_endpoint_id)
    ):
        raise FrozenPostprocessError(
            "completion_source_endpoint_contract_invalid"
        )
    projected = {
        "task": "detection",
        "stage": stage,
        "contract_family": family,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": output_endpoint_id,
        "tensor_signature": signature,
        "output_endpoint_attestation_sha256": canonical_json_sha256(
            dict(attestation)
        ),
    }
    if model_sha256:
        projected["model_sha256"] = model_sha256
    return projected


def _build_materialized_detection_endpoint_contract(
    materialization_contract: Mapping[str, Any],
) -> dict[str, Any]:
    verified = verify_attested_decoded_nms_materialization_contract(
        materialization_contract
    )
    identity = {
        "schema": (
            "onnx-splitpoint/attested-decoded-nms-completed-endpoint"
        ),
        "schema_version": 1,
        "task": "detection",
        "source_stage": "decoded_nms",
        "completed_stage": "decoded_nms",
        "contract_source": (
            "attested_source_nms_plus_no_second_nms_materialization:v1"
        ),
        "source_endpoint_contract_hash": verified[
            "source_endpoint_contract_hash"
        ],
        "source_output_tensor_signature": verified[
            "source_output_tensor_signature"
        ],
        "materialization_contract_sha256": verified["contract_sha256"],
        "normalizer_id": verified["normalizer_id"],
        "source_nms_attested": True,
        "host_nms_applied": False,
        "confidence_threshold": verified["confidence_threshold"],
        "iou_threshold": verified["iou_threshold"],
        "max_detections": verified["max_detections"],
        "input_hw": verified["input_hw"],
        "original_wh": verified["original_wh"],
        "output_record_format": CANONICAL_DETECTION_RECORD_SCHEMA,
        "output_coordinate_space": "original_image_xyxy_pixels",
        "canonical_sort_policy": CANONICAL_DETECTION_SORT_POLICY,
    }
    endpoint_hash = canonical_json_sha256(identity)
    return {
        **identity,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": f"detection:decoded_nms:{endpoint_hash}",
    }


def _build_materialized_comparison_endpoint_contract(
    materialization_contract: Mapping[str, Any],
) -> dict[str, Any]:
    verified = verify_attested_decoded_nms_materialization_contract(
        materialization_contract
    )
    identity = {
        "schema": COMPLETED_COMPARISON_ENDPOINT_SCHEMA,
        "schema_version": COMPLETED_COMPARISON_ENDPOINT_VERSION,
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "model_id": str(verified["model_id"]),
        "model_family": str(verified["model_family"]),
        "output_record_format": "xyxy_score_class",
        "coordinate_space": "original_image_xyxy_pixels",
        "score_semantics": "probability_0_1",
        "class_id_semantics": "integer_model_label_index",
        "class_aware": True,
        "score_threshold": float(verified["confidence_threshold"]),
        "iou_threshold": float(verified["iou_threshold"]),
        "max_detections": int(verified["max_detections"]),
        "input_hw": [int(value) for value in verified["input_hw"]],
        "canonical_completion_policy_id": CANONICAL_COMPLETION_POLICY_ID,
        "nms_semantics_id": "class_aware_nms_xyxy_v1",
    }
    return _seal_completed_comparison_identity(identity)


def _completion_hash_layer(
    kind: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": COMPLETION_HASH_LAYER_SCHEMA,
        "schema_version": COMPLETION_HASH_LAYER_VERSION,
        "kind": str(kind),
        "identity": dict(identity),
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


def _completion_implementation_identity(
    processor_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "processor_contract_sha256": str(
            processor_contract.get("contract_sha256") or ""
        ),
        "implementation_artifacts": dict(
            processor_contract.get("implementation_artifacts") or {}
        ),
    }


def _completion_result_artifact(
    *,
    execution_contract: Mapping[str, Any],
    observation_relation: str,
    invocation_index: int,
    source_content_sha256: str,
    content_sha256: str,
    detections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the exact per-invocation Completed Detection transport payload.

    Unlike the static implementation identity in the execution contract, this
    artifact binds the concrete materialized result emitted by the measured
    hotloop invocation.  Its SHA-256 is always computed over
    :func:`canonical_json_bytes`.
    """
    return {
        "schema": COMPLETION_RESULT_ARTIFACT_SCHEMA,
        "schema_version": COMPLETION_RESULT_ARTIFACT_VERSION,
        "encoding": COMPLETION_RESULT_ARTIFACT_ENCODING,
        "task": "detection",
        "contract_family": "decoded_nms",
        "observation_relation": str(observation_relation),
        "invocation_index": int(invocation_index),
        "completion_mode": str(
            execution_contract.get("completion_mode") or ""
        ),
        "execution_contract_sha256": str(
            execution_contract.get("contract_sha256") or ""
        ),
        "source_endpoint_contract_hash": str(
            (
                execution_contract.get("source_endpoint")
                if isinstance(
                    execution_contract.get("source_endpoint"), Mapping
                )
                else {}
            ).get("endpoint_contract_hash") or ""
        ),
        "completed_endpoint_contract_hash": str(
            (
                execution_contract.get("completed_endpoint_contract")
                if isinstance(
                    execution_contract.get(
                        "completed_endpoint_contract"
                    ),
                    Mapping,
                )
                else {}
            ).get("endpoint_contract_hash") or ""
        ),
        "comparison_endpoint_contract_hash": str(
            (
                execution_contract.get("comparison_endpoint_contract")
                if isinstance(
                    execution_contract.get(
                        "comparison_endpoint_contract"
                    ),
                    Mapping,
                )
                else {}
            ).get("endpoint_contract_hash") or ""
        ),
        "implementation_sha256": str(
            execution_contract.get("implementation_sha256") or ""
        ),
        "schema_sha256": str(
            execution_contract.get("schema_sha256") or ""
        ),
        "relation_sha256": str(
            execution_contract.get("relation_sha256") or ""
        ),
        "source_content_sha256": str(source_content_sha256),
        "content_sha256": str(content_sha256),
        "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
        "detection_count": len(detections),
        "detections": [dict(record) for record in detections],
    }


def _completion_schema_identity(
    *,
    source_endpoint: Mapping[str, Any],
    completion_mode: str,
    processor_contract: Mapping[str, Any],
    yolov7_head_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task": "detection",
        "completion_mode": str(completion_mode),
        "source_stage": str(source_endpoint.get("stage") or ""),
        "source_tensor_signature": source_endpoint.get("tensor_signature"),
        "processor_contract_schema": str(
            processor_contract.get("schema") or ""
        ),
        "processor_contract_schema_version": processor_contract.get(
            "schema_version"
        ),
        "processor_contract_sha256": str(
            processor_contract.get("contract_sha256") or ""
        ),
        "yolov7_head_mapping_sha256": str(
            yolov7_head_mapping.get("head_mapping_sha256") or ""
        ),
        "completed_record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
        "completed_coordinate_space": "original_image_xyxy_pixels",
        "canonical_sort_policy": CANONICAL_DETECTION_SORT_POLICY,
    }


def _completion_relation_identity(
    *,
    source_endpoint: Mapping[str, Any],
    completed_endpoint: Mapping[str, Any],
    comparison_endpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "relation": "physical_source_to_completed_task_to_comparison",
        "source_endpoint_contract_hash": str(
            source_endpoint.get("endpoint_contract_hash") or ""
        ),
        "source_output_endpoint_id": str(
            source_endpoint.get("output_endpoint_id") or ""
        ),
        "completed_endpoint_contract_hash": str(
            completed_endpoint.get("endpoint_contract_hash") or ""
        ),
        "completed_output_endpoint_id": str(
            completed_endpoint.get("output_endpoint_id") or ""
        ),
        "comparison_endpoint_contract_hash": str(
            comparison_endpoint.get("endpoint_contract_hash") or ""
        ),
        "comparison_output_endpoint_id": str(
            comparison_endpoint.get("output_endpoint_id") or ""
        ),
    }


def build_detection_completion_execution_contract(
    *,
    model_id: str,
    outputs: Mapping[str, Any],
    input_hw: Sequence[int],
    original_wh: Sequence[int],
    source_endpoint_contract: Mapping[str, Any],
    preprocess: Mapping[str, Any] | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> dict[str, Any]:
    """Build one semantic completion command shared by performance and energy."""

    source = _verified_completion_source_endpoint(
        source_endpoint_contract,
        outputs=outputs,
    )
    model_family = _model_family(model_id)
    head_mapping: dict[str, Any] = {}
    if source["stage"] in {"raw_head", "decoded_pre_nms"}:
        if model_family == "yolov7":
            source_model_sha256 = str(source.get("model_sha256") or "")
            registered_model_sha256 = str(
                _yolo_module.YOLOV7_PAPER_ONNX_SHA256
            )
            if (
                source_model_sha256
                and source_model_sha256 != registered_model_sha256
            ):
                raise FrozenPostprocessError(
                    "completion_source_yolov7_model_sha256_mismatch"
                )
            canonical_outputs, head_mapping = _canonicalize_yolov7_heads(
                outputs,
                input_hw=input_hw,
            )
        else:
            canonical_outputs = {
                str(name): np.asarray(value)
                for name, value in outputs.items()
            }
        processor_contract = build_frozen_postprocess_contract(
            model_id=model_id,
            source_contract_family=source["stage"],
            outputs=canonical_outputs,
            input_hw=input_hw,
            original_wh=original_wh,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            model_sha256=(
                str(_yolo_module.YOLOV7_PAPER_ONNX_SHA256)
                if model_family == "yolov7" else ""
            ),
        )
        completion_mode = (
            "decoded_pre_nms_frozen_nms" if source["stage"] == "decoded_pre_nms"
            else "raw_head_frozen_decode_nms"
        )
        completed_endpoint = build_completed_detection_endpoint_contract(
            processor_contract,
            source_endpoint_contract_hash=source[
                "endpoint_contract_hash"
            ],
        )
        comparison_endpoint = (
            build_completed_detection_comparison_endpoint_contract(
                processor_contract
            )
        )
    else:
        if preprocess is None:
            raise FrozenPostprocessError(
                "decoded_nms_materialization_preprocess_contract_missing"
            )
        source_attestation = source_endpoint_contract.get(
            "output_endpoint_attestation"
        )
        processor_contract = (
            build_attested_decoded_nms_materialization_contract(
                model_id=model_id,
                outputs=outputs,
                input_hw=input_hw,
                original_wh=original_wh,
                preprocess=preprocess,
                source_endpoint_contract_hash=source[
                    "endpoint_contract_hash"
                ],
                source_output_endpoint_attestation=source_attestation,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
        )
        completion_mode = "decoded_nms_attested_materialization_no_second_nms"
        completed_endpoint = (
            _build_materialized_detection_endpoint_contract(
                processor_contract
            )
        )
        comparison_endpoint = (
            _build_materialized_comparison_endpoint_contract(
                processor_contract
            )
        )

    implementation_layer = _completion_hash_layer(
        "implementation",
        _completion_implementation_identity(processor_contract),
    )
    schema_layer = _completion_hash_layer(
        "schema",
        _completion_schema_identity(
            source_endpoint=source,
            completion_mode=completion_mode,
            processor_contract=processor_contract,
            yolov7_head_mapping=head_mapping,
        ),
    )
    relation_layer = _completion_hash_layer(
        "relation",
        _completion_relation_identity(
            source_endpoint=source,
            completed_endpoint=completed_endpoint,
            comparison_endpoint=comparison_endpoint,
        ),
    )
    contract: dict[str, Any] = {
        "schema": COMPLETION_EXECUTION_SCHEMA,
        "schema_version": COMPLETION_EXECUTION_VERSION,
        "task": "detection",
        "model_id": str(model_id),
        "model_family": model_family,
        "completion_mode": completion_mode,
        "execution_policy": (
            "async_drain_then_completion_tail_then_count_and_timestamp_v1"
        ),
        "performance_energy_contract_identity": (
            "same_completion_execution_contract_sha256"
        ),
        "source_endpoint": source,
        "processor_contract": processor_contract,
        "yolov7_head_mapping": head_mapping,
        "completed_endpoint_contract": completed_endpoint,
        "comparison_endpoint_contract": comparison_endpoint,
        "hash_layers": {
            "implementation": implementation_layer,
            "schema": schema_layer,
            "relation": relation_layer,
        },
        "implementation_sha256": implementation_layer["sha256"],
        "schema_sha256": schema_layer["sha256"],
        "relation_sha256": relation_layer["sha256"],
        "artifact_hash_policy": (
            "sha256_over_canonical_same_invocation_result_artifact_v1"
        ),
        "content_hash_policy": (
            "canonical_materialized_detection_records_without_ui_labels_v1"
        ),
        "invocation_hash_policy": (
            "execution_contract_observation_source_result_ordinal_v1"
        ),
        "observation_relation_policy": list(
            COMPLETION_OBSERVATION_RELATIONS
        ),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _expected_yolov7_canonical_signature(
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    verified = verify_yolov7_head_mapping(mapping)
    tensors = []
    for index, head in enumerate(verified["heads"]):
        stride = int(head["stride"])
        tensors.append({
            "index": index,
            "name": f"yolov7_stride_{stride}",
            "rank": 5,
            "shape": list(head["canonical_shape"]),
            "dtype": "float32",
        })
    return {"tensor_count": 3, "tensors": tensors}


def verify_detection_completion_execution_contract(
    raw: Any,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "completion_execution_contract_missing"
        )
    contract = dict(raw)
    declared = _strict_sha256(
        contract.pop("contract_sha256", ""),
        reason="completion_execution_contract_sha256_invalid",
    )
    if canonical_json_sha256(contract) != declared:
        raise FrozenPostprocessError(
            "completion_execution_contract_sha256_mismatch"
        )
    expected_keys = {
        "schema", "schema_version", "task", "model_id", "model_family",
        "completion_mode", "execution_policy",
        "performance_energy_contract_identity", "source_endpoint",
        "processor_contract", "yolov7_head_mapping",
        "completed_endpoint_contract", "comparison_endpoint_contract",
        "hash_layers", "implementation_sha256", "schema_sha256",
        "relation_sha256", "artifact_hash_policy", "content_hash_policy",
        "invocation_hash_policy", "observation_relation_policy",
    }
    model_id = contract.get("model_id")
    source = contract.get("source_endpoint")
    processor_raw = contract.get("processor_contract")
    completed = contract.get("completed_endpoint_contract")
    comparison = contract.get("comparison_endpoint_contract")
    head_mapping_raw = contract.get("yolov7_head_mapping")
    hash_layers = contract.get("hash_layers")
    if (
        set(contract) != expected_keys
        or contract.get("schema") != COMPLETION_EXECUTION_SCHEMA
        or contract.get("schema_version") != COMPLETION_EXECUTION_VERSION
        or contract.get("task") != "detection"
        or not isinstance(model_id, str)
        or not model_id
        or contract.get("model_family") != _model_family(model_id)
        or contract.get("completion_mode") not in {
            "raw_head_frozen_decode_nms",
            "decoded_pre_nms_frozen_nms",
            "decoded_nms_attested_materialization_no_second_nms",
        }
        or contract.get("execution_policy")
        != "async_drain_then_completion_tail_then_count_and_timestamp_v1"
        or contract.get("performance_energy_contract_identity")
        != "same_completion_execution_contract_sha256"
        or not isinstance(source, Mapping)
        or not isinstance(processor_raw, Mapping)
        or not isinstance(completed, Mapping)
        or not isinstance(comparison, Mapping)
        or not isinstance(head_mapping_raw, Mapping)
        or not isinstance(hash_layers, Mapping)
        or contract.get("artifact_hash_policy")
        != "sha256_over_canonical_same_invocation_result_artifact_v1"
        or contract.get("content_hash_policy")
        != "canonical_materialized_detection_records_without_ui_labels_v1"
        or contract.get("invocation_hash_policy")
        != "execution_contract_observation_source_result_ordinal_v1"
        or contract.get("observation_relation_policy")
        != list(COMPLETION_OBSERVATION_RELATIONS)
    ):
        raise FrozenPostprocessError(
            "completion_execution_contract_fields_invalid"
        )
    source = dict(source)
    source_expected_keys = {
        "task", "stage", "contract_family", "endpoint_contract_hash",
        "output_endpoint_id", "tensor_signature",
        "output_endpoint_attestation_sha256",
    }
    if source.get("model_sha256") is not None:
        source_expected_keys.add("model_sha256")
    source_hash = _strict_sha256(
        source.get("endpoint_contract_hash"),
        reason="completion_execution_source_endpoint_hash_invalid",
    )
    source_attestation_hash = _strict_sha256(
        source.get("output_endpoint_attestation_sha256"),
        reason="completion_execution_source_attestation_hash_invalid",
    )
    source_signature = _verify_tensor_signature_structure(
        source.get("tensor_signature"),
        reason="completion_execution_source_tensor_signature_invalid",
    )
    source_stage = str(source.get("stage") or "")
    if (
        set(source) != source_expected_keys
        or source.get("task") != "detection"
        or source_stage not in {"raw_head", "decoded_pre_nms", "decoded_nms"}
        or source.get("contract_family") != source_stage
        or source.get("output_endpoint_id")
        != f"detection:{source_stage}:{source_hash}"
        or str(
            source.get("output_endpoint_attestation_sha256") or ""
        ).strip().lower() != source_attestation_hash
        or (
            source.get("model_sha256") is not None
            and _strict_sha256(
                source.get("model_sha256"),
                reason=(
                    "completion_execution_source_model_sha256_invalid"
                ),
            ) != str(source.get("model_sha256") or "").strip().lower()
        )
    ):
        raise FrozenPostprocessError(
            "completion_execution_source_endpoint_invalid"
        )

    completion_mode = str(contract["completion_mode"])
    model_family = str(contract["model_family"])
    if completion_mode in {"raw_head_frozen_decode_nms", "decoded_pre_nms_frozen_nms"}:
        expected_source_stage = (
            "decoded_pre_nms" if completion_mode == "decoded_pre_nms_frozen_nms"
            else "raw_head"
        )
        if source_stage != expected_source_stage:
            raise FrozenPostprocessError(
                "completion_execution_source_mode_mismatch"
            )
        processor = verify_frozen_postprocess_contract(processor_raw)
        if processor["source_contract_family"] != source_stage:
            raise FrozenPostprocessError("completion_execution_source_mode_mismatch")
        if model_family != "yolov7" and processor["raw_output_tensor_signature"] != source_signature:
            raise FrozenPostprocessError("completion_execution_processor_source_mismatch")
        if (
            processor.get("model_id") != model_id
            or processor.get("model_family") != model_family
        ):
            raise FrozenPostprocessError(
                "completion_execution_processor_model_mismatch"
            )
        if model_family == "yolov7":
            registered_model_sha256 = str(
                _yolo_module.YOLOV7_PAPER_ONNX_SHA256
            )
            source_model_sha256 = str(source.get("model_sha256") or "")
            if (
                processor.get("model_sha256") != registered_model_sha256
                or (
                    source_model_sha256
                    and source_model_sha256 != registered_model_sha256
                )
            ):
                raise FrozenPostprocessError(
                    "completion_execution_yolov7_model_binding_mismatch"
                )
            head_mapping = verify_yolov7_head_mapping(head_mapping_raw)
            model_bound_decoder = processor.get(
                "model_bound_decoder_contract"
            )
            if not isinstance(model_bound_decoder, Mapping):
                raise FrozenPostprocessError(
                    "completion_execution_yolov7_decoder_binding_missing"
                )
            expected_mapping_heads = [
                {
                    "role": str(record["head_role"]),
                    "stride": int(record["stride"]),
                    "grid_hw": [int(value) for value in record["grid_hw"]],
                    "canonical_shape": [
                        1, 3,
                        int(record["grid_hw"][0]),
                        int(record["grid_hw"][1]), 85,
                    ],
                    "anchor_wh": [
                        [int(value) for value in pair]
                        for pair in record["anchors_wh"]
                    ],
                }
                for record in model_bound_decoder["anchors_by_stride"]
            ]
            mapping_activation = str(
                model_bound_decoder.get("activation_mode") or ""
            )
            expected_mapping_combination = (
                "sigmoid_objectness_times_sigmoid_class"
                if mapping_activation == "logits"
                else "objectness_probability_times_class_probability"
            )
            if (
                head_mapping.get("schema_version")
                != YOLOV7_HEAD_MAPPING_VERSION
                or head_mapping.get("anchor_table_id")
                != YOLOV7_STANDARD_ANCHOR_TABLE_ID
                or head_mapping.get("anchor_table_id")
                != model_bound_decoder.get("anchor_table_id")
                or head_mapping.get("heads") != expected_mapping_heads
                or head_mapping.get("activation_mode")
                != model_bound_decoder.get("activation_mode")
                or head_mapping.get("objectness_class_combination")
                != expected_mapping_combination
            ):
                raise FrozenPostprocessError(
                    "completion_execution_yolov7_mapping_decoder_mismatch"
                )
            if (
                processor.get("raw_output_tensor_signature")
                != _expected_yolov7_canonical_signature(head_mapping)
            ):
                raise FrozenPostprocessError(
                    "completion_execution_yolov7_processor_mapping_mismatch"
                )
        else:
            if dict(head_mapping_raw):
                raise FrozenPostprocessError(
                    "completion_execution_unexpected_head_mapping"
                )
            head_mapping = {}
        expected_completed = build_completed_detection_endpoint_contract(
            processor,
            source_endpoint_contract_hash=source_hash,
        )
        expected_comparison = (
            build_completed_detection_comparison_endpoint_contract(processor)
        )
    else:
        if source_stage != "decoded_nms" or dict(head_mapping_raw):
            raise FrozenPostprocessError(
                "completion_execution_source_mode_mismatch"
            )
        head_mapping = {}
        processor = (
            verify_attested_decoded_nms_materialization_contract(
                processor_raw
            )
        )
        if (
            processor.get("model_id") != model_id
            or processor.get("model_family") != model_family
            or processor.get("source_output_tensor_signature")
            != source_signature
            or processor.get("source_endpoint_contract_hash") != source_hash
        ):
            raise FrozenPostprocessError(
                "completion_execution_processor_source_mismatch"
            )
        expected_completed = (
            _build_materialized_detection_endpoint_contract(processor)
        )
        expected_comparison = (
            _build_materialized_comparison_endpoint_contract(processor)
        )
    if dict(completed) != expected_completed:
        raise FrozenPostprocessError(
            "completion_execution_completed_endpoint_mismatch"
        )
    verified_comparison = (
        verify_completed_detection_comparison_endpoint_contract(comparison)
    )
    if verified_comparison != expected_comparison:
        raise FrozenPostprocessError(
            "completion_execution_comparison_endpoint_mismatch"
        )

    expected_layers = {
        "implementation": _completion_hash_layer(
            "implementation",
            _completion_implementation_identity(processor),
        ),
        "schema": _completion_hash_layer(
            "schema",
            _completion_schema_identity(
                source_endpoint=source,
                completion_mode=completion_mode,
                processor_contract=processor,
                yolov7_head_mapping=head_mapping,
            ),
        ),
        "relation": _completion_hash_layer(
            "relation",
            _completion_relation_identity(
                source_endpoint=source,
                completed_endpoint=expected_completed,
                comparison_endpoint=expected_comparison,
            ),
        ),
    }
    if (
        dict(hash_layers) != expected_layers
        or str(contract.get("implementation_sha256") or "")
        != expected_layers["implementation"]["sha256"]
        or str(contract.get("schema_sha256") or "")
        != expected_layers["schema"]["sha256"]
        or str(contract.get("relation_sha256") or "")
        != expected_layers["relation"]["sha256"]
    ):
        raise FrozenPostprocessError(
            "completion_execution_hash_layers_mismatch"
        )
    return {**contract, "contract_sha256": declared}


def build_detection_completion_execution_attestation(
    execution_contract: Mapping[str, Any],
    last_result: Mapping[str, Any],
    *,
    completed_work_units: int,
    completion_count: int,
    invocation_chain_sha256: str,
    observation_relation: str = "same_hotloop_sentinel",
) -> dict[str, Any]:
    contract = verify_detection_completion_execution_contract(
        execution_contract
    )
    if (
        isinstance(completed_work_units, bool)
        or not isinstance(completed_work_units, int)
        or isinstance(completion_count, bool)
        or not isinstance(completion_count, int)
        or completed_work_units <= 0
        or completion_count != completed_work_units
        or not isinstance(last_result, Mapping)
    ):
        raise FrozenPostprocessError(
            "completion_execution_attestation_count_invalid"
        )
    relation = str(observation_relation or "").strip().lower()
    if relation not in COMPLETION_OBSERVATION_RELATIONS:
        raise FrozenPostprocessError(
            "completion_execution_observation_relation_invalid"
        )
    result = dict(last_result)
    source_content_hash = _strict_sha256(
        result.get("source_content_sha256"),
        reason="completion_execution_source_content_hash_invalid",
    )
    content_hash = _strict_sha256(
        result.get("content_sha256"),
        reason="completion_execution_content_hash_invalid",
    )
    invocation_hash = _strict_sha256(
        result.get("invocation_sha256"),
        reason="completion_execution_invocation_hash_invalid",
    )
    chain_hash = _strict_sha256(
        invocation_chain_sha256,
        reason="completion_execution_invocation_chain_hash_invalid",
    )
    processor_contract = contract["processor_contract"]
    detections = _canonical_detection_records(
        result.get("detections"),
        max_detections=int(processor_contract["max_detections"]),
    )
    expected_content_hash = canonical_json_sha256({
        "schema": "onnx-splitpoint/completed-detection-content",
        "schema_version": 1,
        "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
        "detections": detections,
    })
    artifact = result.get("artifact")
    expected_artifact = _completion_result_artifact(
        execution_contract=contract,
        observation_relation=relation,
        invocation_index=completion_count,
        source_content_sha256=source_content_hash,
        content_sha256=content_hash,
        detections=detections,
    )
    artifact_hash = _strict_sha256(
        result.get("artifact_sha256"),
        reason="completion_execution_artifact_hash_invalid",
    )
    implementation_hash = _strict_sha256(
        result.get("implementation_sha256"),
        reason="completion_execution_implementation_hash_invalid",
    )
    expected_invocation_hash = canonical_json_sha256({
        "schema": "onnx-splitpoint/detection-completion-invocation",
        "schema_version": 1,
        "execution_contract_sha256": contract["contract_sha256"],
        "observation_relation": relation,
        "invocation_index": completion_count,
        "source_content_sha256": source_content_hash,
        "content_sha256": content_hash,
        "artifact_sha256": artifact_hash,
        "implementation_sha256": implementation_hash,
        "schema_sha256": contract["schema_sha256"],
        "relation_sha256": contract["relation_sha256"],
    })
    if (
        result.get("task") != "detection"
        or result.get("contract_family") != "decoded_nms"
        or result.get("observation_relation") != relation
        or result.get("completion_mode") != contract["completion_mode"]
        or result.get("coordinate_space")
        != "original_image_xyxy_pixels"
        or result.get("record_schema")
        != CANONICAL_DETECTION_RECORD_SCHEMA
        or result.get("detection_count") != len(detections)
        or int(result.get("invocation_index") or 0) != completion_count
        or result.get("execution_contract_sha256")
        != contract["contract_sha256"]
        or not isinstance(artifact, Mapping)
        or dict(artifact) != expected_artifact
        or artifact_hash != canonical_json_sha256(expected_artifact)
        or implementation_hash != contract["implementation_sha256"]
        or result.get("schema_sha256") != contract["schema_sha256"]
        or result.get("relation_sha256") != contract["relation_sha256"]
        or str(result.get("content_sha256") or "").strip().lower()
        != content_hash
        or content_hash != expected_content_hash
        or str(result.get("invocation_sha256") or "").strip().lower()
        != invocation_hash
        or invocation_hash != expected_invocation_hash
    ):
        raise FrozenPostprocessError(
            "completion_execution_attestation_result_invalid"
        )
    attestation = {
        "schema": COMPLETION_EXECUTION_ATTESTATION_SCHEMA,
        "schema_version": COMPLETION_EXECUTION_ATTESTATION_VERSION,
        "attested": True,
        "status": "passed",
        "task": "detection",
        "source_stage": contract["source_endpoint"]["stage"],
        "stage": "decoded_nms",
        "endpoint": "decoded_nms",
        "contract_source": (
            "detection_completion_execution_inside_measured_hotloop:v1"
            if relation == "same_hotloop_sentinel"
            else "detection_completion_execution_independent_replay:v1"
        ),
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": contract[
            "completed_endpoint_contract"
        ]["endpoint_contract_hash"],
        "output_endpoint_id": contract[
            "completed_endpoint_contract"
        ]["output_endpoint_id"],
        "observation_relation": relation,
        "same_hotloop_sentinel": relation == "same_hotloop_sentinel",
        "exact_result_claim_bound": relation == "same_hotloop_sentinel",
        "comparison_identity_verified": True,
        "completion_mode": contract["completion_mode"],
        "execution_contract_sha256": contract["contract_sha256"],
        "completed_frames": completed_work_units,
        "completed_work_units": completed_work_units,
        "completion_count": completion_count,
        "postprocess_completed_frames": completion_count,
        "postprocess_completion_verified": True,
        "completion_count_verified": True,
        "artifact": expected_artifact,
        "artifact_sha256": artifact_hash,
        "implementation_sha256": implementation_hash,
        "schema_sha256": contract["schema_sha256"],
        "content_sha256": content_hash,
        "invocation_sha256": invocation_hash,
        "relation_sha256": contract["relation_sha256"],
        "invocation_chain_sha256": chain_hash,
        "completed_endpoint_contract": contract[
            "completed_endpoint_contract"
        ],
        "comparison_endpoint_contract": contract[
            "comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract": contract[
            "comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract_hash": contract[
            "comparison_endpoint_contract"
        ]["endpoint_contract_hash"],
        "completed_task_comparison_output_endpoint_id": contract[
            "comparison_endpoint_contract"
        ]["output_endpoint_id"],
        "completed_task_completion_mode": (
            "detection_completion_execution_v1"
        ),
        "last_result": result,
    }
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    return attestation


def verify_detection_completion_execution_attestation(
    raw: Any,
    *,
    execution_contract: Mapping[str, Any],
    expected_observation_relation: str | None = None,
    reference_content_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a hotloop sentinel or an explicitly independent replay.

    A same-hotloop sentinel can bind the exact materialized result to the
    measured work unit.  An independent replay verifies the same implementation,
    schema, relation, and comparison endpoint identities, while carrying its own
    result-artifact hash; it deliberately cannot make an exact-result claim even
    when its content happens to match.
    """

    if not isinstance(raw, Mapping):
        raise FrozenPostprocessError(
            "completion_execution_attestation_missing"
        )
    attestation = dict(raw)
    declared = _strict_sha256(
        attestation.pop("attestation_sha256", ""),
        reason="completion_execution_attestation_sha256_invalid",
    )
    if canonical_json_sha256(attestation) != declared:
        raise FrozenPostprocessError(
            "completion_execution_attestation_sha256_mismatch"
        )
    contract = verify_detection_completion_execution_contract(
        execution_contract
    )
    relation = str(
        attestation.get("observation_relation") or ""
    ).strip().lower()
    if relation not in COMPLETION_OBSERVATION_RELATIONS:
        raise FrozenPostprocessError(
            "completion_execution_observation_relation_invalid"
        )
    wanted_relation = (
        str(expected_observation_relation or "").strip().lower()
    )
    if wanted_relation and relation != wanted_relation:
        raise FrozenPostprocessError(
            "completion_execution_observation_relation_mismatch"
        )
    expected_exact = relation == "same_hotloop_sentinel"
    expected_keys = {
        "schema", "schema_version", "attested", "status", "task",
        "source_stage", "stage", "endpoint", "contract_source",
        "endpoint_contract_complete", "endpoint_contract_hash",
        "output_endpoint_id",
        "observation_relation", "same_hotloop_sentinel",
        "exact_result_claim_bound", "comparison_identity_verified",
        "completion_mode", "execution_contract_sha256",
        "completed_frames", "completed_work_units", "completion_count",
        "postprocess_completed_frames", "postprocess_completion_verified",
        "completion_count_verified",
        "artifact", "artifact_sha256", "implementation_sha256",
        "schema_sha256", "content_sha256", "invocation_sha256",
        "relation_sha256",
        "invocation_chain_sha256", "completed_endpoint_contract",
        "comparison_endpoint_contract",
        "completed_task_comparison_endpoint_contract",
        "completed_task_comparison_endpoint_contract_hash",
        "completed_task_comparison_output_endpoint_id",
        "completed_task_completion_mode", "last_result",
    }
    if (
        set(attestation) != expected_keys
        or attestation.get("schema")
        != COMPLETION_EXECUTION_ATTESTATION_SCHEMA
        or attestation.get("schema_version")
        != COMPLETION_EXECUTION_ATTESTATION_VERSION
        or attestation.get("attested") is not True
        or attestation.get("status") != "passed"
        or attestation.get("task") != "detection"
        or attestation.get("source_stage")
        != contract["source_endpoint"]["stage"]
        or attestation.get("stage") != "decoded_nms"
        or attestation.get("endpoint") != "decoded_nms"
        or attestation.get("contract_source")
        != (
            "detection_completion_execution_inside_measured_hotloop:v1"
            if expected_exact
            else "detection_completion_execution_independent_replay:v1"
        )
        or attestation.get("endpoint_contract_complete") is not True
        or attestation.get("endpoint_contract_hash")
        != contract["completed_endpoint_contract"][
            "endpoint_contract_hash"
        ]
        or attestation.get("output_endpoint_id")
        != contract["completed_endpoint_contract"]["output_endpoint_id"]
        or attestation.get("same_hotloop_sentinel") is not expected_exact
        or attestation.get("exact_result_claim_bound") is not expected_exact
        or attestation.get("comparison_identity_verified") is not True
        or attestation.get("completion_mode")
        != contract["completion_mode"]
        or attestation.get("execution_contract_sha256")
        != contract["contract_sha256"]
        or attestation.get("implementation_sha256")
        != contract["implementation_sha256"]
        or attestation.get("schema_sha256") != contract["schema_sha256"]
        or attestation.get("relation_sha256")
        != contract["relation_sha256"]
        or attestation.get("completed_endpoint_contract")
        != contract["completed_endpoint_contract"]
        or attestation.get("comparison_endpoint_contract")
        != contract["comparison_endpoint_contract"]
        or attestation.get(
            "completed_task_comparison_endpoint_contract"
        ) != contract["comparison_endpoint_contract"]
        or attestation.get(
            "completed_task_comparison_endpoint_contract_hash"
        ) != contract["comparison_endpoint_contract"][
            "endpoint_contract_hash"
        ]
        or attestation.get(
            "completed_task_comparison_output_endpoint_id"
        ) != contract["comparison_endpoint_contract"][
            "output_endpoint_id"
        ]
        or attestation.get("completed_task_completion_mode")
        != "detection_completion_execution_v1"
    ):
        raise FrozenPostprocessError(
            "completion_execution_attestation_fields_invalid"
        )
    try:
        completed_work_units = int(attestation["completed_work_units"])
        completed_frames = int(attestation["completed_frames"])
        completion_count = int(attestation["completion_count"])
        postprocess_count = int(
            attestation["postprocess_completed_frames"]
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise FrozenPostprocessError(
            "completion_execution_attestation_count_invalid"
        ) from exc
    if (
        any(
            isinstance(attestation.get(field), bool)
            for field in (
                "completed_work_units", "completion_count",
                "completed_frames", "postprocess_completed_frames",
            )
        )
        or completed_work_units <= 0
        or completed_frames != completed_work_units
        or completion_count != completed_work_units
        or postprocess_count != completed_work_units
        or attestation.get("postprocess_completion_verified") is not True
        or attestation.get("completion_count_verified") is not True
    ):
        raise FrozenPostprocessError(
            "completion_execution_attestation_count_invalid"
        )
    rebuilt = build_detection_completion_execution_attestation(
        contract,
        attestation.get("last_result") or {},
        completed_work_units=completed_work_units,
        completion_count=completion_count,
        invocation_chain_sha256=str(
            attestation.get("invocation_chain_sha256") or ""
        ),
        observation_relation=relation,
    )
    if rebuilt != {**attestation, "attestation_sha256": declared}:
        raise FrozenPostprocessError(
            "completion_execution_attestation_result_invalid"
        )
    verified = dict(rebuilt)
    if reference_content_sha256 is not None:
        reference_hash = _strict_sha256(
            reference_content_sha256,
            reason="completion_execution_reference_content_hash_invalid",
        )
        content_match = (
            verified["content_sha256"] == reference_hash
        )
        if expected_exact and not content_match:
            raise FrozenPostprocessError(
                "completion_execution_same_hotloop_content_mismatch"
            )
        verified["reference_content_sha256"] = reference_hash
        verified["reference_content_match"] = content_match
    return verified


class DetectionCompletionRuntime:
    """Execute and attest one fully materialized detection tail per work unit."""

    def __init__(
        self,
        execution_contract: Mapping[str, Any],
        *,
        observation_relation: str = "same_hotloop_sentinel",
    ) -> None:
        self.execution_contract = (
            verify_detection_completion_execution_contract(
                execution_contract
            )
        )
        self.observation_relation = str(
            observation_relation or ""
        ).strip().lower()
        if self.observation_relation not in (
            COMPLETION_OBSERVATION_RELATIONS
        ):
            raise FrozenPostprocessError(
                "completion_execution_observation_relation_invalid"
            )
        self._lock = threading.Lock()
        mode = self.execution_contract["completion_mode"]
        if mode in {"raw_head_frozen_decode_nms", "decoded_pre_nms_frozen_nms"}:
            self._processor: Any = FrozenDetectionPostprocessor(
                self.execution_contract["processor_contract"]
            )
        else:
            self._processor = _AttestedDecodedNmsMaterializer(
                self.execution_contract["processor_contract"]
            )
        self.completed_count = 0
        self.last_result: dict[str, Any] = {}
        self.last_detections: list[dict[str, Any]] = []
        self.invocation_chain_sha256 = _EMPTY_INVOCATION_CHAIN_SHA256

    def process(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            source = self.execution_contract["source_endpoint"]
            if tensor_signature(outputs) != source["tensor_signature"]:
                raise FrozenPostprocessError(
                    "completion_runtime_source_tensor_signature_mismatch"
                )
            mode = self.execution_contract["completion_mode"]
            if mode in {"raw_head_frozen_decode_nms", "decoded_pre_nms_frozen_nms"}:
                if self.execution_contract["model_family"] == "yolov7":
                    processor_outputs, observed_mapping = (
                        _canonicalize_yolov7_heads(
                            outputs,
                            input_hw=self.execution_contract[
                                "processor_contract"
                            ]["input_hw"],
                            activation_mode=self.execution_contract[
                                "yolov7_head_mapping"
                            ]["activation_mode"],
                        )
                    )
                    if observed_mapping != self.execution_contract[
                        "yolov7_head_mapping"
                    ]:
                        raise FrozenPostprocessError(
                            "completion_runtime_yolov7_head_mapping_drift"
                        )
                else:
                    processor_outputs = {
                        str(name): np.asarray(value)
                        for name, value in outputs.items()
                    }
                source_content_hash = _tensor_content_sha256(
                    processor_outputs
                )
                self._processor.process(
                    processor_outputs,
                    original_wh=self.execution_contract[
                        "processor_contract"
                    ]["original_wh"],
                )
                detections = _canonical_detection_records(
                    self._processor.last_detections,
                    max_detections=int(
                        self.execution_contract["processor_contract"][
                            "max_detections"
                        ]
                    ),
                )
            else:
                source_content_hash = _tensor_content_sha256(outputs)
                self._processor.process(outputs)
                detections = _canonical_detection_records(
                    self._processor.last_detections,
                    max_detections=int(
                        self.execution_contract["processor_contract"][
                            "max_detections"
                        ]
                    ),
                )

            content_identity = {
                "schema": "onnx-splitpoint/completed-detection-content",
                "schema_version": 1,
                "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
                "coordinate_space": "original_image_xyxy_pixels",
                "sort_policy": CANONICAL_DETECTION_SORT_POLICY,
                "detections": detections,
            }
            content_hash = canonical_json_sha256(content_identity)
            invocation_index = self.completed_count + 1
            artifact = _completion_result_artifact(
                execution_contract=self.execution_contract,
                observation_relation=self.observation_relation,
                invocation_index=invocation_index,
                source_content_sha256=source_content_hash,
                content_sha256=content_hash,
                detections=detections,
            )
            artifact_hash = canonical_json_sha256(artifact)
            invocation_identity = {
                "schema": (
                    "onnx-splitpoint/detection-completion-invocation"
                ),
                "schema_version": 1,
                "execution_contract_sha256": self.execution_contract[
                    "contract_sha256"
                ],
                "observation_relation": self.observation_relation,
                "invocation_index": invocation_index,
                "source_content_sha256": source_content_hash,
                "content_sha256": content_hash,
                "artifact_sha256": artifact_hash,
                "implementation_sha256": self.execution_contract[
                    "implementation_sha256"
                ],
                "schema_sha256": self.execution_contract["schema_sha256"],
                "relation_sha256": self.execution_contract[
                    "relation_sha256"
                ],
            }
            invocation_hash = canonical_json_sha256(invocation_identity)
            next_chain = canonical_json_sha256({
                "schema": (
                    "onnx-splitpoint/detection-completion-invocation-chain"
                ),
                "schema_version": 1,
                "previous_sha256": self.invocation_chain_sha256,
                "invocation_index": invocation_index,
                "invocation_sha256": invocation_hash,
            })
            result = {
                "task": "detection",
                "contract_family": "decoded_nms",
                "observation_relation": self.observation_relation,
                "completion_mode": mode,
                "coordinate_space": "original_image_xyxy_pixels",
                "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
                "detection_count": len(detections),
                "detections": detections,
                "source_content_sha256": source_content_hash,
                "content_sha256": content_hash,
                "artifact": artifact,
                "artifact_sha256": artifact_hash,
                "implementation_sha256": self.execution_contract[
                    "implementation_sha256"
                ],
                "invocation_index": invocation_index,
                "invocation_sha256": invocation_hash,
                "relation_sha256": self.execution_contract[
                    "relation_sha256"
                ],
                "schema_sha256": self.execution_contract["schema_sha256"],
                "execution_contract_sha256": self.execution_contract[
                    "contract_sha256"
                ],
                "source_nms_attested": (
                    mode
                    == "decoded_nms_attested_materialization_no_second_nms"
                ),
                "host_nms_applied": (
                    mode in {"raw_head_frozen_decode_nms", "decoded_pre_nms_frozen_nms"}
                ),
            }
            if mode == "decoded_pre_nms_frozen_nms":
                result["decoded_pre_nms_score_normalization"] = dict(
                    self._processor.last_result.get("decoded_pre_nms_score_normalization") or {}
                )
            # The public completion counter and timestamp boundary may advance
            # only after every record and every hash layer was materialized.
            self.last_detections = detections
            self.last_result = result
            self.invocation_chain_sha256 = next_chain
            self.completed_count = invocation_index
            return dict(result)

    def attestation(
        self,
        *,
        completed_work_units: int | None = None,
    ) -> dict[str, Any]:
        count = (
            self.completed_count
            if completed_work_units is None
            else completed_work_units
        )
        return build_detection_completion_execution_attestation(
            self.execution_contract,
            self.last_result,
            completed_work_units=count,
            completion_count=self.completed_count,
            invocation_chain_sha256=self.invocation_chain_sha256,
            observation_relation=self.observation_relation,
        )


def build_detection_completion_runtime(
    *,
    model_id: str,
    outputs: Mapping[str, Any],
    input_hw: Sequence[int],
    original_wh: Sequence[int],
    source_endpoint_contract: Mapping[str, Any],
    preprocess: Mapping[str, Any] | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    observation_relation: str = "same_hotloop_sentinel",
) -> DetectionCompletionRuntime:
    """Create a runtime from one untimed, physically attested output probe."""

    contract = build_detection_completion_execution_contract(
        model_id=model_id,
        outputs=outputs,
        input_hw=input_hw,
        original_wh=original_wh,
        source_endpoint_contract=source_endpoint_contract,
        preprocess=preprocess,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    return DetectionCompletionRuntime(
        contract,
        observation_relation=observation_relation,
    )
