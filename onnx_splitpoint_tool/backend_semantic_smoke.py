"""Read-only semantic smoke audits for accelerator backend adapters.

The checks in this module deliberately stop before compiler or accelerator
invocation.  They consume already materialized images, ONNX models and output
dumps and answer three narrow questions:

* did all producers bind the same canonical prepared RGB image;
* does a Hailo compiler-fixup ONNX retain the CPU-visible model semantics; and
* do YOLO11/YOLO26 raw heads complete to the same detections as the float
  reference endpoint, independent of the vendor output order where required.

Optional dependencies and evidence are represented as ``SKIP`` rather than a
false ``PASS``.  Malformed or numerically different available evidence is a
``FAIL``.  No input file is opened for writing and no hardware runtime is
loaded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_frozen_postprocess_contract,
)
from .preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
    preprocessing_contract_sha256,
)
from .runners.harness.base import postprocess_result_to_dict
from .runners.harness.yolo import YoloHarness
from .runners.harness import yolo as _yolo_module


SPEC_SCHEMA = "onnx-splitpoint/backend-semantic-smoke-spec/v1"
RESULT_SCHEMA = "onnx-splitpoint/backend-semantic-smoke-result/v1"
CASE_KINDS = {
    "prepared_input_identity",
    "onnx_cpu_pair",
    "detection_host_tail",
}


class SemanticSmokeSkip(RuntimeError):
    """Evidence or one optional dependency is not available."""


class SemanticSmokeFailure(RuntimeError):
    """Available evidence violates the declared semantic contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    identity = {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "data_sha256": _sha256_bytes(array.tobytes(order="C")),
    }
    return _sha256_bytes(_canonical_json_bytes(identity))


def _case_result(
    case_id: str,
    kind: str,
    status: str,
    reason: str,
    details: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "case_id": str(case_id),
        "kind": str(kind),
        "status": str(status),
        "reason": str(reason),
        "details": dict(details or {}),
    }


def _resolve_file(base_dir: Path, raw: Any, *, label: str) -> Path:
    token = str(raw or "").strip()
    if not token:
        raise SemanticSmokeSkip(f"{label}_not_declared")
    path = Path(token).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = Path(os.path.abspath(os.fspath(path)))
    if not os.path.lexists(path):
        raise SemanticSmokeSkip(f"{label}_missing")
    if path.is_symlink() or not path.is_file():
        raise SemanticSmokeFailure(f"{label}_unsafe_or_not_regular")
    return path


def _load_array_file(
    base_dir: Path,
    raw: Any,
    *,
    label: str,
    key: Any = None,
) -> np.ndarray:
    path = _resolve_file(base_dir, raw, label=label)
    suffix = path.suffix.lower()
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"{label}_numpy_load_failed:{type(exc).__name__}"
        ) from exc
    if suffix == ".npz":
        try:
            names = list(loaded.files)
            selected = str(key or "").strip()
            if not selected:
                if len(names) != 1:
                    raise SemanticSmokeFailure(
                        f"{label}_npz_key_required"
                    )
                selected = names[0]
            if selected not in names:
                raise SemanticSmokeFailure(f"{label}_npz_key_missing")
            value = np.asarray(loaded[selected])
        finally:
            loaded.close()
    else:
        value = np.asarray(loaded)
    if value.dtype.kind not in "fiu" or value.size <= 0:
        raise SemanticSmokeFailure(f"{label}_numeric_array_required")
    if not bool(np.isfinite(value).all()):
        raise SemanticSmokeFailure(f"{label}_nonfinite")
    return np.ascontiguousarray(value)


def _load_output_dump(
    base_dir: Path,
    raw: Any,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    path = _resolve_file(base_dir, raw, label=label)
    if path.suffix.lower() != ".npz":
        raise SemanticSmokeFailure(f"{label}_npz_required")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"{label}_numpy_load_failed:{type(exc).__name__}"
        ) from exc
    try:
        outputs = {
            str(name): np.ascontiguousarray(np.asarray(loaded[name]))
            for name in loaded.files
        }
    finally:
        loaded.close()
    if not outputs:
        raise SemanticSmokeFailure(f"{label}_empty")
    if any(value.size <= 0 or value.dtype.kind not in "fiu" for value in outputs.values()):
        raise SemanticSmokeFailure(f"{label}_numeric_outputs_required")
    if any(not bool(np.isfinite(value).all()) for value in outputs.values()):
        raise SemanticSmokeFailure(f"{label}_nonfinite")
    return outputs


def _load_source_image(base_dir: Path, case: Mapping[str, Any]) -> np.ndarray:
    path = _resolve_file(
        base_dir, case.get("source_image"), label="source_image"
    )
    if path.suffix.lower() in {".npy", ".npz"}:
        return _load_array_file(
            base_dir,
            case.get("source_image"),
            label="source_image",
            key=case.get("source_image_key"),
        )
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError as exc:
        raise SemanticSmokeSkip("pillow_unavailable") from exc
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"source_image_decode_failed:{type(exc).__name__}"
        ) from exc


def audit_prepared_input_identity(
    case: Mapping[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    """Compare producer dumps against one canonical prepared RGB image."""

    case_id = str(case.get("id") or "prepared_input_identity")
    task = str(case.get("task") or "").strip()
    target_hw = case.get("target_hw")
    observations = case.get("observations")
    if not isinstance(observations, list) or not observations:
        raise SemanticSmokeSkip("prepared_input_observations_unavailable")
    if len(observations) < 2:
        raise SemanticSmokeFailure(
            "prepared_input_at_least_two_producers_required"
        )
    try:
        contract = canonical_image_preprocessing_contract(task, target_hw)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"preprocessing_contract_invalid:{type(exc).__name__}"
        ) from exc
    source = _load_source_image(base_dir, case)
    try:
        expected, geometry = prepare_rgb_uint8_image(source, contract)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"canonical_preprocessing_failed:{type(exc).__name__}"
        ) from exc
    expected = np.ascontiguousarray(expected)
    expected_sha = _array_sha256(expected)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen_producers: set[str] = set()
    for index, raw_observation in enumerate(observations):
        if not isinstance(raw_observation, Mapping):
            raise SemanticSmokeFailure(
                "prepared_input_observation_invalid"
            )
        producer = str(raw_observation.get("producer") or "").strip()
        if not producer:
            raise SemanticSmokeFailure(
                "prepared_input_producer_missing"
            )
        if producer in seen_producers:
            raise SemanticSmokeFailure(
                f"prepared_input_producer_duplicate:{producer}"
            )
        seen_producers.add(producer)
        try:
            observed = _load_array_file(
                base_dir,
                raw_observation.get("path"),
                label=f"prepared_input_{producer}",
                key=raw_observation.get("key"),
            )
        except SemanticSmokeSkip as exc:
            skipped.append(str(exc))
            rows.append({
                "producer": producer,
                "status": "SKIP",
                "reason": str(exc),
                "exact_match": None,
            })
            continue
        observed_sha = _array_sha256(observed)
        exact = (
            observed.dtype == np.dtype(np.uint8)
            and observed.shape == expected.shape
            and observed_sha == expected_sha
            and np.array_equal(observed, expected)
        )
        rows.append({
            "producer": producer,
            "status": "PASS" if exact else "FAIL",
            "dtype": str(observed.dtype),
            "shape": [int(dim) for dim in observed.shape],
            "array_identity_sha256": observed_sha,
            "exact_match": bool(exact),
        })
    exact_match = all(
        row.get("exact_match") is True
        for row in rows
        if row.get("status") != "SKIP"
    )
    mismatch = any(row.get("status") == "FAIL" for row in rows)
    if skipped and not mismatch:
        raise SemanticSmokeSkip(
            "prepared_input_evidence_incomplete:" + "|".join(skipped)
        )
    return _case_result(
        case_id,
        "prepared_input_identity",
        "PASS" if exact_match and not skipped else "FAIL",
        (
            "all_prepared_rgb_uint8_inputs_exact"
            if exact_match and not skipped
            else "prepared_input_identity_mismatch"
        ),
        {
            "task": contract["task"],
            "target_hw": list(contract["target_hw"]),
            "preprocessing_contract_sha256": preprocessing_contract_sha256(
                contract
            ),
            "canonical_prepared_array_sha256": expected_sha,
            "geometry": geometry,
            "observations": rows,
        },
    )


def _ort_dtype(type_token: str) -> np.dtype:
    token = str(type_token or "").strip().lower()
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int64)": np.dtype(np.int64),
    }
    if token not in mapping:
        raise SemanticSmokeSkip(f"onnx_input_dtype_unsupported:{token}")
    return mapping[token]


def _static_shape(raw: Sequence[Any]) -> tuple[int, ...]:
    shape: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SemanticSmokeSkip("onnx_dynamic_input_requires_input_dump")
        shape.append(int(value))
    if not shape:
        raise SemanticSmokeSkip("onnx_input_shape_missing")
    return tuple(shape)


def _seeded_input(shape: Sequence[int], dtype: np.dtype, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        high = min(int(info.max), 255) + 1
        return generator.integers(
            max(int(info.min), 0), high, size=tuple(shape), dtype=dtype
        )
    return generator.uniform(0.0, 1.0, size=tuple(shape)).astype(dtype)


def _session_options(runtime: Any) -> Any:
    try:
        options = runtime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        if hasattr(runtime, "ExecutionMode"):
            options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
        if hasattr(runtime, "GraphOptimizationLevel"):
            options.graph_optimization_level = (
                runtime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        return options
    except Exception:
        return None


def _create_ort_session(runtime: Any, path: Path) -> Any:
    options = _session_options(runtime)
    kwargs: dict[str, Any] = {"providers": ["CPUExecutionProvider"]}
    if options is not None:
        kwargs["sess_options"] = options
    return runtime.InferenceSession(str(path), **kwargs)


def _onnx_input_contract(inputs: Sequence[Any]) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    for value in inputs:
        name = str(value.name or "").strip()
        if not name or name in contract:
            raise SemanticSmokeFailure("onnx_pair_input_name_invalid")
        shape = list(value.shape)
        if not shape:
            raise SemanticSmokeFailure("onnx_pair_input_shape_missing")
        contract[name] = {
            "type": str(value.type or "").strip().lower(),
            "shape": shape,
        }
    return contract


def _verify_onnx_input_contract(
    original_inputs: Sequence[Any], fixed_inputs: Sequence[Any]
) -> dict[str, dict[str, Any]]:
    original = _onnx_input_contract(original_inputs)
    fixed = _onnx_input_contract(fixed_inputs)
    if set(original) != set(fixed):
        raise SemanticSmokeFailure("onnx_pair_input_names_mismatch")
    for name in original:
        if original[name]["type"] != fixed[name]["type"]:
            raise SemanticSmokeFailure(
                f"onnx_pair_input_dtype_mismatch:{name}"
            )
        if original[name]["shape"] != fixed[name]["shape"]:
            raise SemanticSmokeFailure(
                f"onnx_pair_input_shape_contract_mismatch:{name}"
            )
        _ort_dtype(original[name]["type"])
    return original


def _load_onnx_inputs(
    case: Mapping[str, Any],
    *,
    base_dir: Path,
    original_inputs: Sequence[Any],
) -> tuple[dict[str, np.ndarray], str]:
    raw_path = str(case.get("input_dump") or "").strip()
    if not raw_path:
        seed = int(case.get("seed", 27713))
        arrays = {
            str(value.name): _seeded_input(
                _static_shape(list(value.shape)),
                _ort_dtype(value.type),
                seed + index,
            )
            for index, value in enumerate(original_inputs)
        }
        return arrays, "deterministic_seeded_static_shape"

    path = _resolve_file(base_dir, raw_path, label="onnx_input_dump")
    if path.suffix.lower() not in {".npy", ".npz"}:
        raise SemanticSmokeFailure("onnx_input_dump_numpy_required")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"onnx_input_dump_load_failed:{type(exc).__name__}"
        ) from exc
    if path.suffix.lower() == ".npz":
        try:
            names = list(loaded.files)
            expected_names = [str(value.name) for value in original_inputs]
            if len(names) != len(set(names)) or set(names) != set(expected_names):
                raise SemanticSmokeFailure(
                    "onnx_input_dump_binding_ambiguous"
                )
            arrays = {
                name: np.ascontiguousarray(np.asarray(loaded[name]))
                for name in expected_names
            }
        finally:
            loaded.close()
    else:
        if len(original_inputs) != 1:
            raise SemanticSmokeFailure(
                "onnx_input_dump_npz_required_for_multi_input_model"
            )
        arrays = {
            str(original_inputs[0].name): np.ascontiguousarray(
                np.asarray(loaded)
            )
        }
    if any(
        value.size <= 0
        or value.dtype.kind not in "fiu"
        or not bool(np.isfinite(value).all())
        for value in arrays.values()
    ):
        raise SemanticSmokeFailure("onnx_input_dump_numeric_finite_required")
    return arrays, "declared_numpy_dump"


def _feed_for_session(
    session: Any, arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    inputs = list(session.get_inputs())
    if len(inputs) != len(arrays):
        raise SemanticSmokeFailure("onnx_pair_input_count_mismatch")
    feed: dict[str, np.ndarray] = {}
    if {str(meta.name) for meta in inputs} != set(arrays):
        raise SemanticSmokeFailure("onnx_pair_input_names_mismatch")
    for meta in inputs:
        name = str(meta.name)
        raw = np.ascontiguousarray(np.asarray(arrays[name]))
        expected_shape = list(meta.shape)
        if len(expected_shape) != raw.ndim:
            raise SemanticSmokeFailure("onnx_pair_input_shape_mismatch")
        for expected, observed in zip(expected_shape, raw.shape):
            if isinstance(expected, int) and not isinstance(expected, bool):
                if expected > 0 and int(expected) != int(observed):
                    raise SemanticSmokeFailure(
                        "onnx_pair_input_shape_mismatch"
                    )
        expected_dtype = _ort_dtype(meta.type)
        if raw.dtype != expected_dtype:
            raise SemanticSmokeFailure(
                f"onnx_pair_input_dump_dtype_mismatch:{name}"
            )
        if not bool(np.isfinite(raw).all()):
            raise SemanticSmokeFailure(
                f"onnx_pair_input_dump_nonfinite:{name}"
            )
        if all(isinstance(value, int) and value > 0 for value in expected_shape):
            if tuple(int(value) for value in expected_shape) != raw.shape:
                raise SemanticSmokeFailure("onnx_pair_input_shape_mismatch")
        feed[name] = raw
    return feed


def _onnx_output_pairs(
    case: Mapping[str, Any], original_session: Any, fixed_session: Any
) -> list[tuple[str, str]]:
    original_names = [str(value.name) for value in original_session.get_outputs()]
    fixed_names = [str(value.name) for value in fixed_session.get_outputs()]
    if (
        not original_names
        or not fixed_names
        or any(not name for name in original_names + fixed_names)
        or len(original_names) != len(set(original_names))
        or len(fixed_names) != len(set(fixed_names))
    ):
        raise SemanticSmokeFailure("onnx_pair_output_names_invalid")
    declared = case.get("output_pairs")
    if declared is not None:
        if not isinstance(declared, list) or not declared:
            raise SemanticSmokeFailure("onnx_output_pairs_invalid")
        pairs: list[tuple[str, str]] = []
        for row in declared:
            if not isinstance(row, Mapping):
                raise SemanticSmokeFailure("onnx_output_pair_invalid")
            pairs.append((str(row.get("original") or ""), str(row.get("fixed") or "")))
        if any(not left or not right for left, right in pairs):
            raise SemanticSmokeFailure("onnx_output_pair_name_missing")
        left_names = [left for left, _right in pairs]
        right_names = [right for _left, right in pairs]
        if (
            len(left_names) != len(set(left_names))
            or len(right_names) != len(set(right_names))
            or set(left_names) != set(original_names)
            or set(right_names) != set(fixed_names)
        ):
            raise SemanticSmokeFailure(
                "onnx_output_pairs_not_complete_bijection"
            )
        return pairs
    if len(original_names) != len(fixed_names):
        raise SemanticSmokeFailure("onnx_pair_output_count_mismatch")
    if set(original_names) == set(fixed_names):
        return [(name, name) for name in original_names]
    raise SemanticSmokeFailure(
        "onnx_output_pairs_required_for_renamed_outputs"
    )


def audit_onnx_cpu_pair(
    case: Mapping[str, Any],
    *,
    base_dir: Path,
    runtime_module: Any = None,
) -> dict[str, Any]:
    """Run an original/fixed ONNX pair with the same CPU input."""

    case_id = str(case.get("id") or "onnx_cpu_pair")
    original_path = _resolve_file(
        base_dir, case.get("original_model"), label="original_onnx"
    )
    fixed_path = _resolve_file(
        base_dir, case.get("fixed_model"), label="fixed_onnx"
    )
    original_model_sha256 = _file_sha256(original_path)
    fixed_model_sha256 = _file_sha256(fixed_path)
    if (
        original_path == fixed_path
        or original_model_sha256 == fixed_model_sha256
    ):
        raise SemanticSmokeFailure("onnx_pair_models_not_independent")
    if runtime_module is None:
        try:
            runtime_module = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise SemanticSmokeSkip("onnxruntime_unavailable") from exc
    try:
        original_session = _create_ort_session(runtime_module, original_path)
        fixed_session = _create_ort_session(runtime_module, fixed_path)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"onnx_cpu_session_failed:{type(exc).__name__}"
        ) from exc
    original_inputs = list(original_session.get_inputs())
    fixed_inputs = list(fixed_session.get_inputs())
    if not original_inputs or len(original_inputs) != len(fixed_inputs):
        raise SemanticSmokeFailure("onnx_pair_input_contract_mismatch")
    input_contract = _verify_onnx_input_contract(
        original_inputs, fixed_inputs
    )
    arrays, input_source = _load_onnx_inputs(
        case,
        base_dir=base_dir,
        original_inputs=original_inputs,
    )
    original_feed = _feed_for_session(original_session, arrays)
    fixed_feed = _feed_for_session(fixed_session, arrays)
    pairs = _onnx_output_pairs(case, original_session, fixed_session)
    original_names = [left for left, _right in pairs]
    fixed_names = [right for _left, right in pairs]
    original_output_meta = {
        str(value.name): value for value in original_session.get_outputs()
    }
    fixed_output_meta = {
        str(value.name): value for value in fixed_session.get_outputs()
    }
    output_contract: list[dict[str, Any]] = []
    for original_name, fixed_name in pairs:
        original_type = str(
            original_output_meta[original_name].type or ""
        ).strip().lower()
        fixed_type = str(
            fixed_output_meta[fixed_name].type or ""
        ).strip().lower()
        if original_type != fixed_type:
            raise SemanticSmokeFailure(
                "onnx_pair_output_dtype_contract_mismatch:"
                f"{original_name}:{fixed_name}"
            )
        expected_dtype = _ort_dtype(original_type)
        output_contract.append({
            "original_output": original_name,
            "fixed_output": fixed_name,
            "ort_type": original_type,
            "dtype": str(expected_dtype),
        })
    try:
        original_values = original_session.run(original_names, original_feed)
        fixed_values = fixed_session.run(fixed_names, fixed_feed)
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"onnx_cpu_inference_failed:{type(exc).__name__}"
        ) from exc
    if len(original_values) != len(pairs) or len(fixed_values) != len(pairs):
        raise SemanticSmokeFailure("onnx_pair_output_result_count_mismatch")
    if (
        _file_sha256(original_path) != original_model_sha256
        or _file_sha256(fixed_path) != fixed_model_sha256
    ):
        raise SemanticSmokeFailure("onnx_model_changed_during_audit")
    atol = float(case.get("atol", 1e-5))
    rtol = float(case.get("rtol", 1e-4))
    if atol < 0.0 or rtol < 0.0 or not math.isfinite(atol + rtol):
        raise SemanticSmokeFailure("onnx_pair_tolerance_invalid")
    rows: list[dict[str, Any]] = []
    for (original_name, fixed_name), original, fixed, contract_row in zip(
        pairs, original_values, fixed_values, output_contract
    ):
        left = np.asarray(original)
        right = np.asarray(fixed)
        expected_dtype = np.dtype(contract_row["dtype"])
        if left.dtype != expected_dtype or right.dtype != expected_dtype:
            raise SemanticSmokeFailure(
                "onnx_pair_output_dtype_mismatch:"
                f"{original_name}:{fixed_name}"
            )
        shape_match = left.shape == right.shape
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        allclose = bool(
            shape_match
            and finite
            and np.allclose(left, right, atol=atol, rtol=rtol)
        )
        maximum = (
            float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            if shape_match and left.size and finite else None
        )
        row: dict[str, Any] = {
            "original_output": original_name,
            "fixed_output": fixed_name,
            "shape": [int(dim) for dim in left.shape],
            "dtype": str(left.dtype),
            "shape_match": bool(shape_match),
            "finite": finite,
            "allclose": allclose,
            "max_abs_error": maximum,
            "original_array_sha256": _array_sha256(left),
            "fixed_array_sha256": _array_sha256(right),
        }
        if (
            left.ndim == 2
            and left.shape[0] == 1
            and left.shape[-1] > 1
            and shape_match
            and finite
        ):
            row["argmax_match"] = bool(
                int(np.argmax(left[0])) == int(np.argmax(right[0]))
            )
        rows.append(row)
    numerical_match = bool(rows) and all(row["allclose"] for row in rows)
    return _case_result(
        case_id,
        "onnx_cpu_pair",
        "PASS" if numerical_match else "FAIL",
        (
            "original_and_hailo_fixed_cpu_outputs_match"
            if numerical_match else "onnx_cpu_pair_numerical_mismatch"
        ),
        {
            "input_source": input_source,
            "input_contract": input_contract,
            "output_contract": output_contract,
            "input_array_sha256": {
                name: _array_sha256(value)
                for name, value in sorted(arrays.items())
            },
            "original_model_sha256": original_model_sha256,
            "fixed_model_sha256": fixed_model_sha256,
            "atol": atol,
            "rtol": rtol,
            "outputs": rows,
        },
    )


def _harness_result(
    outputs: Mapping[str, Any],
    *,
    input_hw: Sequence[int],
    original_wh: Sequence[int],
    model_id: str,
) -> dict[str, Any]:
    harness = YoloHarness(
        conf_thresh=0.25,
        iou_thresh=0.45,
        max_det=300,
        model_id=str(model_id),
    )
    result = harness.postprocess(
        dict(outputs),
        {
            "input_hw": [int(value) for value in input_hw],
            "original_wh": [int(value) for value in original_wh],
            "model_id": str(model_id),
            "variant": "backend_semantic_smoke",
        },
    )
    normalized = postprocess_result_to_dict(result)
    payload = normalized.get("json")
    if not isinstance(payload, Mapping):
        raise SemanticSmokeFailure("detection_postprocess_payload_missing")
    return dict(payload)


def _canonical_detection_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    detections = raw.get("detections")
    if not isinstance(detections, list):
        raise SemanticSmokeFailure("detection_result_rows_missing")
    rows: list[dict[str, Any]] = []
    for value in detections:
        if not isinstance(value, Mapping):
            raise SemanticSmokeFailure("detection_result_row_invalid")
        try:
            row = {
                "class_id": int(value["class_id"]),
                "score": float(value["score"]),
                "x1": float(value["x1"]),
                "y1": float(value["y1"]),
                "x2": float(value["x2"]),
                "y2": float(value["y2"]),
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SemanticSmokeFailure("detection_result_row_invalid") from exc
        if not all(math.isfinite(float(item)) for item in row.values()):
            raise SemanticSmokeFailure("detection_result_row_nonfinite")
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -row["score"],
            row["class_id"],
            row["x1"],
            row["y1"],
            row["x2"],
            row["y2"],
        ),
    )


def _compare_detection_rows(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    score_atol: float,
    coordinate_atol: float,
) -> dict[str, Any]:
    if len(left) != len(right):
        return {
            "match": False,
            "left_count": len(left),
            "right_count": len(right),
            "reason": "detection_count_mismatch",
        }
    maximum_score = 0.0
    maximum_coordinate = 0.0
    for left_row, right_row in zip(left, right):
        if int(left_row["class_id"]) != int(right_row["class_id"]):
            return {
                "match": False,
                "left_count": len(left),
                "right_count": len(right),
                "reason": "detection_class_mismatch",
            }
        maximum_score = max(
            maximum_score,
            abs(float(left_row["score"]) - float(right_row["score"])),
        )
        maximum_coordinate = max(
            maximum_coordinate,
            *(
                abs(float(left_row[key]) - float(right_row[key]))
                for key in ("x1", "y1", "x2", "y2")
            ),
        )
    match = maximum_score <= score_atol and maximum_coordinate <= coordinate_atol
    return {
        "match": bool(match),
        "left_count": len(left),
        "right_count": len(right),
        "max_score_error": maximum_score,
        "max_coordinate_error": maximum_coordinate,
        "reason": "match" if match else "detection_numeric_mismatch",
    }


def _raw_order_variants(
    outputs: Mapping[str, np.ndarray], model_id: str
) -> list[tuple[str, dict[str, np.ndarray]]]:
    items = list(outputs.items())
    variants = [("observed", dict(items))]
    if "yolo26" not in str(model_id).lower().replace("_", "").replace("-", ""):
        return variants
    variants.append(("reversed", dict(reversed(items))))

    def geometry_key(item: tuple[str, np.ndarray]) -> tuple[int, int, str]:
        name, value = item
        channels, spatial = _yolo_module._infer_ch_sp_any(np.asarray(value))
        if channels is None or spatial is None:
            return (0, 2, str(name))
        role = 0 if int(channels) == 4 else 1 if int(channels) == 80 else 2
        return (-int(spatial), role, str(name))

    variants.append(("spatial_80_40_20_reg_cls", dict(sorted(items, key=geometry_key))))
    variants.append(("spatial_20_40_80_cls_reg", dict(reversed(sorted(items, key=geometry_key)))))
    return variants


def _verify_yolo26_geometry(outputs: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    names = list(outputs.keys())
    values = list(outputs.values())
    pairs = _yolo_module._get_ultralytics_regcls_pairs(names, values)
    if len(pairs) != 3:
        raise SemanticSmokeFailure("yolo26_exact_three_regcls_pairs_required")
    rows: list[dict[str, Any]] = []
    observed: set[tuple[int, int, int]] = set()
    for _level, reg_index, cls_index in pairs:
        reg_channels, reg_spatial = _yolo_module._infer_ch_sp_any(values[reg_index])
        cls_channels, cls_spatial = _yolo_module._infer_ch_sp_any(values[cls_index])
        if (
            reg_channels != 4
            or cls_channels != 80
            or reg_spatial is None
            or cls_spatial != reg_spatial
        ):
            raise SemanticSmokeFailure("yolo26_ltrb_coco_pair_geometry_invalid")
        side = int(round(math.sqrt(int(reg_spatial))))
        if side * side != int(reg_spatial):
            raise SemanticSmokeFailure("yolo26_grid_not_square")
        observed.add((side, int(reg_channels), int(cls_channels)))
        rows.append({
            "grid_hw": [side, side],
            "regression_output": names[reg_index],
            "classification_output": names[cls_index],
            "regression_channels": int(reg_channels),
            "classification_channels": int(cls_channels),
        })
    if observed != {(80, 4, 80), (40, 4, 80), (20, 4, 80)}:
        raise SemanticSmokeFailure("yolo26_ltrb_coco_three_level_contract_required")
    return sorted(rows, key=lambda row: -int(row["grid_hw"][0]))


def _verify_yolo11_geometry(
    outputs: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    names = list(outputs.keys())
    values = list(outputs.values())
    if len(values) != 6:
        raise SemanticSmokeFailure(
            "yolo11_exact_six_regcls_tensors_required"
        )
    pairs = _yolo_module._get_ultralytics_regcls_pairs(names, values)
    if len(pairs) != 3:
        raise SemanticSmokeFailure(
            "yolo11_exact_three_regcls_pairs_required"
        )
    rows: list[dict[str, Any]] = []
    observed: set[tuple[int, int, int]] = set()
    for _level, reg_index, cls_index in pairs:
        reg_channels, reg_spatial = _yolo_module._infer_ch_sp_any(
            values[reg_index]
        )
        cls_channels, cls_spatial = _yolo_module._infer_ch_sp_any(
            values[cls_index]
        )
        if (
            reg_channels != 64
            or cls_channels != 80
            or reg_spatial is None
            or cls_spatial != reg_spatial
        ):
            raise SemanticSmokeFailure(
                "yolo11_dfl16_coco_pair_geometry_invalid"
            )
        side = int(round(math.sqrt(int(reg_spatial))))
        if side * side != int(reg_spatial):
            raise SemanticSmokeFailure("yolo11_grid_not_square")
        observed.add((side, int(reg_channels), int(cls_channels)))
        rows.append({
            "grid_hw": [side, side],
            "regression_output": names[reg_index],
            "classification_output": names[cls_index],
            "regression_channels": int(reg_channels),
            "classification_channels": int(cls_channels),
        })
    if observed != {(80, 64, 80), (40, 64, 80), (20, 64, 80)}:
        raise SemanticSmokeFailure(
            "yolo11_dfl16_coco_three_level_contract_required"
        )
    return sorted(rows, key=lambda row: -int(row["grid_hw"][0]))


def audit_detection_host_tail(
    case: Mapping[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    """Compare a frozen raw-head host tail with one float reference dump."""

    case_id = str(case.get("id") or "detection_host_tail")
    model_id = str(case.get("model_id") or "").strip()
    family = model_id.lower().replace("_", "").replace("-", "")
    if "yolo11" not in family and "yolo26" not in family:
        raise SemanticSmokeFailure("detection_smoke_model_family_unsupported")
    raw_path = _resolve_file(
        base_dir, case.get("raw_outputs"), label="raw_output_dump"
    )
    raw_file_sha256 = _file_sha256(raw_path)
    raw_outputs = _load_output_dump(
        base_dir, raw_path, label="raw_output_dump"
    )
    if _file_sha256(raw_path) != raw_file_sha256:
        raise SemanticSmokeFailure("raw_output_dump_changed_during_audit")
    input_hw = [int(value) for value in list(case.get("input_hw") or [640, 640])]
    original_wh = [int(value) for value in list(case.get("original_wh") or input_hw[::-1])]
    if len(input_hw) != 2 or len(original_wh) != 2 or any(
        value <= 0 for value in input_hw + original_wh
    ):
        raise SemanticSmokeFailure("detection_smoke_geometry_invalid")
    if input_hw != [640, 640]:
        raise SemanticSmokeFailure("detection_smoke_640_input_required")
    geometry_rows: list[dict[str, Any]] = []
    if "yolo11" in family:
        geometry_rows = _verify_yolo11_geometry(raw_outputs)
    elif "yolo26" in family:
        geometry_rows = _verify_yolo26_geometry(raw_outputs)
    try:
        contract = build_frozen_postprocess_contract(
            model_id=model_id,
            outputs=raw_outputs,
            input_hw=input_hw,
            original_wh=original_wh,
        )
        raw_result = FrozenDetectionPostprocessor(contract).process(
            raw_outputs, original_wh=original_wh
        )
    except Exception as exc:
        raise SemanticSmokeFailure(
            f"frozen_host_tail_failed:{type(exc).__name__}:{exc}"
        ) from exc
    if "yolo11" in family and (
        str(contract.get("decoder_id") or "")
        != "yolo11_regcls_dfl16_classaware_nms_v1"
        or str(contract.get("decoder_format") or "")
        != "ultralytics_regcls"
    ):
        raise SemanticSmokeFailure(
            "yolo11_dfl16_decoder_contract_required"
        )
    raw_rows = _canonical_detection_rows(raw_result)
    raw_minimum = case.get("minimum_detections", 1)
    if isinstance(raw_minimum, bool):
        raise SemanticSmokeFailure("minimum_detections_invalid")
    minimum_detections = int(raw_minimum)
    if minimum_detections < 1:
        raise SemanticSmokeFailure("minimum_detections_invalid")
    if len(raw_rows) < minimum_detections:
        raise SemanticSmokeFailure("raw_host_tail_detection_count_below_minimum")
    score_atol = float(case.get("score_atol", 1e-5))
    coordinate_atol = float(case.get("coordinate_atol", 1e-3))
    if (
        score_atol < 0.0
        or coordinate_atol < 0.0
        or not math.isfinite(score_atol)
        or not math.isfinite(coordinate_atol)
    ):
        raise SemanticSmokeFailure("detection_smoke_tolerance_invalid")

    order_checks: list[dict[str, Any]] = []
    for variant, ordered in _raw_order_variants(raw_outputs, model_id):
        decoded = _canonical_detection_rows(
            _harness_result(
                ordered,
                input_hw=input_hw,
                original_wh=original_wh,
                model_id=model_id,
            )
        )
        comparison = _compare_detection_rows(
            raw_rows,
            decoded,
            score_atol=score_atol,
            coordinate_atol=coordinate_atol,
        )
        order_checks.append({"order": variant, **comparison})
    order_invariant = all(row["match"] for row in order_checks)

    reference_path = _resolve_file(
        base_dir,
        case.get("reference_outputs"),
        label="reference_output_dump",
    )
    reference_file_sha256 = _file_sha256(reference_path)
    if raw_file_sha256 == reference_file_sha256:
        raise SemanticSmokeFailure(
            "detection_reference_not_independent"
        )
    reference_outputs = _load_output_dump(
        base_dir,
        reference_path,
        label="reference_output_dump",
    )
    if _file_sha256(reference_path) != reference_file_sha256:
        raise SemanticSmokeFailure(
            "reference_output_dump_changed_during_audit"
        )
    reference_result = _harness_result(
        reference_outputs,
        input_hw=input_hw,
        original_wh=original_wh,
        model_id=model_id,
    )
    reference_format = str(reference_result.get("format") or "")
    if reference_format != "bn6_detections":
        raise SemanticSmokeFailure(
            "completed_bn6_detection_reference_required"
        )
    reference_rows = _canonical_detection_rows(reference_result)
    parity = _compare_detection_rows(
        raw_rows,
        reference_rows,
        score_atol=score_atol,
        coordinate_atol=coordinate_atol,
    )
    passed = bool(order_invariant and parity["match"])
    reason = (
        "frozen_raw_head_host_tail_matches_float_reference"
        if passed
        else "raw_head_output_order_changes_detections"
        if not order_invariant
        else f"raw_head_reference_parity_failed:{parity['reason']}"
    )
    return _case_result(
        case_id,
        "detection_host_tail",
        "PASS" if passed else "FAIL",
        reason,
        {
            "model_id": model_id,
            "decoder_id": contract["decoder_id"],
            "decoder_format": contract["decoder_format"],
            "frozen_contract_sha256": contract["contract_sha256"],
            "raw_output_dump_sha256": raw_file_sha256,
            "reference_output_dump_sha256": reference_file_sha256,
            "raw_output_order": list(raw_outputs.keys()),
            "raw_detection_count": len(raw_rows),
            "raw_detections_sha256": _sha256_bytes(
                _canonical_json_bytes(raw_rows)
            ),
            "reference_format": reference_format,
            "reference_detection_count": len(reference_rows),
            "reference_detections_sha256": _sha256_bytes(
                _canonical_json_bytes(reference_rows)
            ),
            "parity": parity,
            "order_checks": order_checks,
            "yolo11_regcls_geometry": (
                geometry_rows if "yolo11" in family else []
            ),
            "yolo26_regcls_geometry": (
                geometry_rows if "yolo26" in family else []
            ),
        },
    )


def run_backend_semantic_smokes(
    spec: Mapping[str, Any], *, base_dir: str | Path = "."
) -> dict[str, Any]:
    """Execute all declared cases without mutating their evidence."""

    if not isinstance(spec, Mapping) or spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("backend_semantic_smoke_spec_schema_invalid")
    cases = spec.get("cases")
    if not isinstance(cases, list):
        raise ValueError("backend_semantic_smoke_spec_cases_invalid")
    root = Path(base_dir).expanduser()
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, Mapping):
            results.append(_case_result(
                f"case_{index}", "invalid", "FAIL", "case_not_an_object"
            ))
            continue
        case = dict(raw_case)
        case_id = str(case.get("id") or f"case_{index}")
        kind = str(case.get("kind") or "")
        if case_id in seen_ids:
            results.append(_case_result(
                case_id, kind, "FAIL", "duplicate_case_id"
            ))
            continue
        seen_ids.add(case_id)
        if kind not in CASE_KINDS:
            results.append(_case_result(
                case_id, kind, "FAIL", "unsupported_case_kind"
            ))
            continue
        try:
            if kind == "prepared_input_identity":
                result = audit_prepared_input_identity(case, base_dir=root)
            elif kind == "onnx_cpu_pair":
                result = audit_onnx_cpu_pair(case, base_dir=root)
            else:
                result = audit_detection_host_tail(case, base_dir=root)
        except SemanticSmokeSkip as exc:
            result = _case_result(case_id, kind, "SKIP", str(exc))
        except SemanticSmokeFailure as exc:
            result = _case_result(case_id, kind, "FAIL", str(exc))
        except Exception as exc:  # fail closed, keep CLI result structured
            result = _case_result(
                case_id,
                kind,
                "FAIL",
                f"unexpected_{type(exc).__name__}:{exc}",
            )
        results.append(result)
    counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("PASS", "FAIL", "SKIP")
    }
    status = (
        "FAIL"
        if counts["FAIL"]
        else "SKIP"
        if counts["SKIP"]
        else "PASS"
        if counts["PASS"] and counts["PASS"] == len(results)
        else "SKIP"
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "read_only": True,
        "hardware_invoked": False,
        "case_count": len(results),
        "counts": counts,
        "cases": results,
    }
    payload["result_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return payload


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("backend_semantic_smoke_spec_unsafe_or_missing")
    duplicate = False

    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs
    )
    if duplicate:
        raise ValueError("backend_semantic_smoke_spec_duplicate_json_key")
    return payload


def example_spec() -> dict[str, Any]:
    """Return a path-neutral example for real dump/artifact canaries."""

    return {
        "schema": SPEC_SCHEMA,
        "cases": [
            {
                "id": "classification_prepared_input",
                "kind": "prepared_input_identity",
                "task": "classification",
                "target_hw": [224, 224],
                "source_image": "evidence/source_image.jpg",
                "observations": [
                    {"producer": "cpu", "path": "evidence/cpu_prepared.npy"},
                    {"producer": "tensorrt", "path": "evidence/trt_prepared.npy"},
                    {"producer": "hailo", "path": "evidence/hailo_prepared.npy"},
                ],
            },
            {
                "id": "mobilenet_hailo_fixup",
                "kind": "onnx_cpu_pair",
                "original_model": "models/mobilenet.onnx",
                "fixed_model": "artifacts/mobilenet_hailo_fixed.onnx",
                "input_dump": "evidence/mobilenet_input.npy",
            },
            {
                "id": "yolo11_host_tail",
                "kind": "detection_host_tail",
                "model_id": "yolo11l",
                "raw_outputs": "evidence/yolo11_hailo_raw_heads.npz",
                "reference_outputs": "evidence/yolo11_float_bn6.npz",
                "input_hw": [640, 640],
                "original_wh": [640, 480],
            },
            {
                "id": "yolo26_host_tail",
                "kind": "detection_host_tail",
                "model_id": "yolo26m",
                "raw_outputs": "evidence/yolo26_hailo_raw_heads.npz",
                "reference_outputs": "evidence/yolo26_float_bn6.npz",
                "input_hw": [640, 640],
                "original_wh": [640, 480],
            },
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only CPU/preprocess/decoder semantic smokes over "
            "existing backend artifacts and NumPy dumps."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", help="JSON smoke specification")
    group.add_argument(
        "--example", action="store_true", help="print a path-neutral example"
    )
    parser.add_argument(
        "--compact", action="store_true", help="emit compact JSON"
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return nonzero for SKIP as well as FAIL",
    )
    ns = parser.parse_args(list(argv) if argv is not None else None)
    if ns.example:
        print(json.dumps(example_spec(), indent=2, sort_keys=True))
        return 0
    spec_path = Path(str(ns.spec)).expanduser().resolve()
    try:
        payload = run_backend_semantic_smokes(
            _load_json(spec_path), base_dir=spec_path.parent
        )
    except Exception as exc:
        payload = {
            "schema": RESULT_SCHEMA,
            "status": "FAIL",
            "read_only": True,
            "hardware_invoked": False,
            "case_count": 0,
            "counts": {"PASS": 0, "FAIL": 1, "SKIP": 0},
            "cases": [],
            "error": f"{type(exc).__name__}:{exc}",
        }
        payload["result_sha256"] = _sha256_bytes(
            _canonical_json_bytes(payload)
        )
    print(
        json.dumps(
            payload,
            indent=None if ns.compact else 2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    status = str(payload.get("status") or "")
    return 1 if status == "FAIL" or (ns.require_pass and status != "PASS") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
