"""Deterministic Quality-FIRST contracts for Native accelerator -> TensorRT splits.

The management quality pass is scheduled before the Native performance stage.
Consequently a model/setup/case must select its Native boundary variant before
either side builds an engine.  This module is the single, side-effect-free
registry for that selection and the strict join used after both producers have
finished.  It deliberately never aliases precision tags: a quality result for
``float32_layout_fp16`` cannot attest ``uint8_dequant_fp16``.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Mapping
import hashlib
import json
import os
import re
import stat
from pathlib import Path

if (__package__ or "").split(".", 1)[0] == "splitpoint_runners":
    # A generated BenchmarkSet is self-contained.  Prefer its byte-for-byte
    # vendored sibling even when an older onnx_splitpoint_tool happens to be
    # installed globally on the accelerator host.
    from .native_command_contract import (  # type: ignore
        canonical_json_sha256,
        verify_native_command_contract,
    )
else:
    from onnx_splitpoint_tool.native_command_contract import (
        canonical_json_sha256,
        verify_native_command_contract,
    )


SELECTION_SCHEMA = "onnx-splitpoint/native-split-quality-preselection"
SELECTION_VERSION = 1
POLICY_SCHEMA = "onnx-splitpoint/native-split-quality-policy"
POLICY_VERSION = 1
BINDING_SCHEMA = "onnx-splitpoint/native-split-quality-binding"
BINDING_VERSION = 1
CENTRAL_SELECTION_SCHEMA = (
    "onnx-splitpoint/native-split-quality-central-selection"
)
CENTRAL_SELECTION_VERSION = 1
LOCAL_VERIFICATION_SCHEMA = (
    "onnx-splitpoint/native-split-local-artifact-verification"
)
LOCAL_VERIFICATION_VERSION = 1
TRT_ENGINE_BUILD_RECEIPT_SCHEMA = (
    "onnx-splitpoint/tensorrt-engine-build-receipt"
)
TRT_ENGINE_BUILD_RECEIPT_VERSION = 1
BOUNDARY_METADATA_SCHEMA = "onnx-splitpoint/native-part1-boundary-metadata"
BOUNDARY_METADATA_VERSION = 1
CONSUMER_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/native-split-quality-consumer-attestation"
)
CONSUMER_ATTESTATION_VERSION = 1

_REQUIRED_BINDING_ARTIFACTS = (
    "part1_runtime", "boundary_metadata", "source_part2_onnx",
    "build_part2_onnx", "engine", "native_trt_meta",
    "engine_build_receipt", "trtexec",
)
_EMBEDDED_JSON_ARTIFACTS = (
    "boundary_metadata", "native_trt_meta", "engine_build_receipt",
)


class _BindingValidationError(ValueError):
    """Internal exception carrying one stable fail-closed status token."""


def probability_quantization_suitability(semantic: Mapping[str, Any], tensor: Mapping[str, Any]) -> dict[str, Any]:
    """Prove complete loss of declared probabilities, never infer it from samples.

    Native affine encoding is monotone. Checking both endpoints therefore
    covers the entire declared interval. Unknown rounding includes directed
    rounding; it must not silently acquire the host's numpy rounding rule.
    """
    import math
    result = {"status": "UNKNOWN", "reason": "score_semantics_unproven"}
    if semantic.get("score_semantics") != "probability_0_1":
        return result
    result["required"] = True
    try:
        shape = list(semantic["shape"])
        axis = semantic["channel_axis"]
        channels = list(semantic["score_channels"])
        if (not shape or any(type(v) is not int or v <= 0 for v in shape)
                or type(axis) is not int or not 0 <= axis < len(shape)
                or not channels or len(set(channels)) != len(channels)
                or any(type(c) is not int or not 0 <= c < shape[axis] for c in channels)
                or list(tensor["canonical_part2_shape"]) != shape):
            raise ValueError("shape_or_channel_mapping_invalid")
        dtype = tensor.get("hef_native_storage_dtype")
        bounds = {"uint8": (0, 255), "uint16": (0, 65535), "int8": (-128, 127)}
        if dtype not in bounds:
            raise ValueError("native_integer_representation_unproven")
        quant = tensor["quantization"]
        if quant.get("source") != "hailort_hef_output_vstream_info":
            raise ValueError("quantization_source_unproven")
        scale, zp = quant["scale"], quant["zero_point"]
        if isinstance(scale, (list, tuple)) or isinstance(zp, (list, tuple)):
            if (quant.get("canonical_channel_axis") != axis
                    or len(scale) != shape[axis] or len(zp) != shape[axis]):
                raise ValueError("per_channel_mapping_unproven")
            all_pairs = [(float(s), float(z)) for s, z in zip(scale, zp)]
            pairs = [all_pairs[c] for c in channels]
        else:
            pairs = [(float(scale), float(zp))]
            all_pairs = pairs
        lo, hi = bounds[dtype]
        if any(not math.isfinite(s) or s <= 0 or not math.isfinite(z)
               or z != int(z) or not lo <= z <= hi for s, z in all_pairs):
            raise ValueError("quantization_parameters_invalid")
        rounding = quant.get("rounding", "unknown")
        methods = {"nearest_even": (round,), "nearest_unspecified_ties": (round, lambda x: math.floor(x + .5), lambda x: math.ceil(x - .5)),
                   "floor": (math.floor,), "ceil": (math.ceil,), "truncate": (math.trunc,)}
        functions = methods.get(rounding, (round, math.floor, math.ceil, math.trunc, lambda x: math.floor(x + .5)))
        collapsed, distinct = [], []
        for s, z in pairs:
            upper = z + 1.0 / s
            if not math.isfinite(upper) or upper == z:
                raise ValueError("quantization_arithmetic_unresolved")
            endpoints = [(max(lo, min(hi, f(z))), max(lo, min(hi, f(upper)))) for f in functions]
            collapsed.append(all(a == b == z for a, b in endpoints))
            distinct.append(all(a != b for a, b in endpoints))
        if all(collapsed):
            result.update(status="INCOMPATIBLE", reason="probability_range_collapses_to_zero")
        elif any(distinct):
            result.update(status="NOT_COLLAPSED", reason="probability_range_has_distinct_codes")
        else:
            result.update(reason="complete_collapse_not_proven")
        result.update(native_dtype=dtype, probability_interval=[0.0, 1.0], score_channel_count=len(channels), rounding=rounding)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = str(exc)
    return result


def detection_score_graph_contract(part1: Any, part2: Any, *, task: str) -> dict[str, Any]:
    """Recognize a probability branch consumed as scores by a selection tail.

    Names and model sizes carry no semantics. A Sigmoid feature map alone is
    insufficient: the continuation must split the same channels, rank their
    maximum, and pass score values to an output without learned arithmetic.
    Unsupported graph forms remain UNKNOWN.
    """
    unknown = {"status": "UNKNOWN", "reason": "score_graph_semantics_unproven"}
    if task != "detection":
        return unknown
    import onnx
    import numpy as np
    try:
        if len(part1.graph.output) != 1 or len(part2.graph.input) != 1:
            return unknown
        if any(n.domain not in ('', 'ai.onnx') for g in (part1.graph, part2.graph) for n in g.node):
            return unknown
        p1 = onnx.shape_inference.infer_shapes(part1)
        shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                  for v in list(p1.graph.input) + list(p1.graph.value_info) + list(p1.graph.output)}
        producers = {o: n for n in p1.graph.node for o in n.output}
        output = p1.graph.output[0].name
        if output != part2.graph.input[0].name:
            return unknown
        node = producers[output]
        trailing = []
        while node.op_type in {"Identity", "Transpose"}:
            trailing.append(node)
            node = producers[node.input[0]]
        if node.op_type != "Concat":
            return unknown
        def attr(n, key, default=None):
            return next((onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == key), default)
        concat_shape, shape = shapes[node.output[0]], shapes[output]
        if not shape or any(d <= 0 for d in shape + concat_shape):
            return unknown
        axis = int(attr(node, "axis")) % len(concat_shape)
        widths = [shapes[i][axis] for i in node.input]
        if any(w <= 0 for w in widths) or sum(widths) != concat_shape[axis]:
            return unknown
        for transform in reversed(trailing):
            if transform.op_type == "Transpose":
                axis = list(attr(transform, "perm", list(reversed(range(len(shape)))))).index(axis)
        score_branches = [i for i, name in enumerate(node.input) if producers.get(name) is not None and producers[name].op_type == "Sigmoid"]
        if len(score_branches) != 1:
            return unknown
        branch = score_branches[0]
        consumers = {}
        for n in part2.graph.node:
            for name in n.input:
                consumers.setdefault(name, []).append(n)
        current, mapped_axis = output, axis
        while len(consumers.get(current, [])) == 1 and consumers[current][0].op_type in {"Identity", "Transpose"}:
            n = consumers[current][0]
            if n.op_type == "Transpose":
                perm = list(attr(n, "perm", list(reversed(range(len(shape))))))
                mapped_axis = perm.index(mapped_axis)
            current = n.output[0]
        split = consumers.get(current, [])
        if len(split) != 1 or split[0].op_type != "Split":
            return unknown
        split = split[0]
        constants = {t.name: onnx.numpy_helper.to_array(t) for t in part2.graph.initializer}
        for n in part2.graph.node:
            if n.op_type == "Constant":
                value = attr(n, "value")
                if value is not None:
                    constants[n.output[0]] = onnx.numpy_helper.to_array(value)
        sizes = constants.get(split.input[1]) if len(split.input) > 1 else attr(split, "split")
        if (sizes is None or list(np.asarray(sizes).reshape(-1)) != widths
                or int(attr(split, "axis", 0)) % len(shape) != mapped_axis):
            return unknown
        score = split.output[branch]
        reductions = [n for n in consumers.get(score, []) if n.op_type == "ReduceMax"]
        rank_indices = set()
        for n in reductions:
            axes = constants.get(n.input[1]) if len(n.input) > 1 else attr(n, "axes")
            if axes is not None and [int(a) % len(shape) for a in np.asarray(axes).reshape(-1)] == [mapped_axis]:
                for c in consumers.get(n.output[0], []):
                    if c.op_type == "TopK" and c.input[0] == n.output[0] and len(c.output) == 2:
                        rank_indices.add(c.output[1])
        if not rank_indices:
            return unknown
        # Follow data values, never a TopK index, shape input or gather index.
        values, selected_values = {score}, set()
        passthrough = {"Identity", "Transpose", "Reshape", "Flatten", "Unsqueeze", "Squeeze", "Gather", "GatherElements", "TopK"}
        for n in part2.graph.node:
            if n.op_type in passthrough | {"Tile", "Expand", "Cast"} and n.input[0] in rank_indices:
                rank_indices.add(n.output[0])
            if n.op_type in passthrough and n.input[0] in values:
                values.add(n.output[0])
                if (n.input[0] in selected_values or n.op_type == "TopK"
                        or n.op_type in {"Gather", "GatherElements"} and n.input[1] in rank_indices):
                    selected_values.add(n.output[0])
            elif n.op_type == "Concat" and any(i in values for i in n.input):
                values.add(n.output[0])
                if any(i in selected_values for i in n.input):
                    selected_values.add(n.output[0])
        if not any(o.name in selected_values for o in part2.graph.output):
            return unknown
        start = sum(widths[:branch])
        return {"status": "PROVEN", "source": "onnx_sigmoid_split_ranked_score_output",
                "score_semantics": "probability_0_1", "shape": shape, "channel_axis": axis,
                "score_channels": list(range(start, start + widths[branch]))}
    except (KeyError, TypeError, ValueError, IndexError, onnx.shape_inference.InferenceError):
        return unknown


def _reject(status: str) -> None:
    raise _BindingValidationError(status)


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def _case(value: Any) -> str:
    token = _token(value)
    digits = token[1:] if token.startswith("b") else token
    return f"b{int(digits):03d}" if digits.isdigit() else token


def _backend(value: Any, setup_id: Any = "") -> str:
    token = _token(value)
    setup = _token(setup_id)
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
    # The explicit token is authoritative.  Setup fallback is allowed only
    # when no source/backend identity was supplied; it must never turn a
    # foreign explicit backend into a same-setup alias.
    if not token:
        if "hailo8" in setup:
            return "hailo8_to_trt"
        if "hailo10" in setup:
            return "hailo10h_to_trt"
        if "deepx" in setup:
            return "deepx_to_trt"
    return token


def canonical_native_split_backend(value: Any, setup_id: Any = "") -> str:
    """Canonicalise historical and current Native split backend spellings."""

    return _backend(value, setup_id)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


_CENTRAL_SELECTION_BINDING_FIELDS = (
    "producer_binding_sha256",
    "source_request_sha256",
    "central_result_sha256",
    "central_quality_selection",
    "central_quality_selection_sha256",
)


def _producer_binding_sha256_from_selected_binding(
    binding: Mapping[str, Any],
) -> str:
    """Reconstruct the exact pre-selection producer binding identity."""

    producer = deepcopy(dict(binding))
    producer.pop("binding_sha256", None)
    for field in _CENTRAL_SELECTION_BINDING_FIELDS:
        producer.pop(field, None)
    return canonical_json_sha256(producer)


def validate_central_native_split_quality_selection(
    binding: Mapping[str, Any],
    *,
    required: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Validate management's immutable selection of one Central result.

    The producer binding is created before Central Quality writes its request
    and result identities, so including the request hash in that original seal
    would be circular.  Management therefore derives one second, fully sealed
    binding.  Its receipt binds the original producer seal, the exact Central
    request, the canonical Central result and every scientific selection key.
    """

    if not isinstance(binding, Mapping):
        return None, "native_split_quality_central_selection_binding_missing"
    present = [field in binding for field in _CENTRAL_SELECTION_BINDING_FIELDS]
    if not any(present):
        return (
            (None, "native_split_quality_central_selection_missing")
            if required
            else (None, "native_split_quality_central_selection_not_present")
        )
    if not all(present):
        return None, "native_split_quality_central_selection_fields_incomplete"
    receipt_raw = binding.get("central_quality_selection")
    if not isinstance(receipt_raw, Mapping):
        return None, "native_split_quality_central_selection_receipt_missing"
    receipt = deepcopy(dict(receipt_raw))
    declared = _sha256(receipt.pop("receipt_sha256", ""))
    if not declared or canonical_json_sha256(receipt) != declared:
        return None, "native_split_quality_central_selection_receipt_sha256_mismatch"
    receipt["receipt_sha256"] = declared
    if (
        receipt.get("schema") != CENTRAL_SELECTION_SCHEMA
        or int(receipt.get("schema_version") or 0) != CENTRAL_SELECTION_VERSION
    ):
        return None, "native_split_quality_central_selection_schema_invalid"

    producer_sha = _sha256(binding.get("producer_binding_sha256"))
    request_sha = _sha256(binding.get("source_request_sha256"))
    result_sha = _sha256(binding.get("central_result_sha256"))
    if (
        not producer_sha
        or producer_sha != _producer_binding_sha256_from_selected_binding(binding)
    ):
        return None, "native_split_quality_central_selection_producer_binding_mismatch"
    if not request_sha:
        return None, "native_split_quality_central_selection_request_sha256_invalid"
    if not result_sha:
        return None, "native_split_quality_central_selection_result_sha256_invalid"
    if _sha256(binding.get("central_quality_selection_sha256")) != declared:
        return None, "native_split_quality_central_selection_duplicate_sha256_mismatch"
    for field, expected in (
        ("producer_binding_sha256", producer_sha),
        ("source_request_sha256", request_sha),
        ("central_result_sha256", result_sha),
    ):
        if _sha256(receipt.get(field)) != expected:
            return None, f"native_split_quality_central_selection_{field}_mismatch"

    preselection = binding.get("preselection")
    if not isinstance(preselection, Mapping):
        return None, "native_split_quality_central_selection_preselection_missing"
    exact = {
        "eval_run_id": str(binding.get("eval_run_id") or "").strip(),
        "source_run_id": _backend(
            binding.get("source_run_id"), preselection.get("setup_id"),
        ),
        "model_id": _token(preselection.get("model_id")),
        "case_id": _case(preselection.get("case_id")),
        "setup_id": _token(preselection.get("setup_id")),
        "backend": _backend(preselection.get("backend")),
        "task": _token(preselection.get("task")),
        "precision": _token(preselection.get("precision")),
        "runtime_precision_identity": _token(preselection.get("precision")),
        "variant": "composed",
    }
    for field, expected in exact.items():
        observed = str(receipt.get(field) or "").strip()
        if field in {
            "model_id", "setup_id", "task", "precision",
            "runtime_precision_identity",
        }:
            observed = _token(observed)
        elif field == "case_id":
            observed = _case(observed)
        elif field == "backend":
            observed = _backend(observed, exact["setup_id"])
        elif field == "source_run_id":
            # Historical aliases are accepted only in the immutable producer
            # binding used to derive ``expected``.  A newly created Central
            # receipt must itself carry the canonical spelling exactly.
            observed = _token(observed)
        if not expected or observed != expected:
            return None, f"native_split_quality_central_selection_{field}_mismatch"
    return receipt, "central_quality_selection_exactly_verified"


def select_central_native_split_quality_binding(
    binding: Mapping[str, Any],
    *,
    source_request_sha256: Any,
    central_result_sha256: Any,
    central_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the binding consumed by Native from one Central result."""

    producer, status = validate_native_split_quality_binding(
        binding, verification_mode="portable",
    )
    if producer is None:
        raise ValueError(status)
    existing, existing_status = validate_central_native_split_quality_selection(
        producer,
    )
    if existing is not None or existing_status != (
        "native_split_quality_central_selection_not_present"
    ):
        raise ValueError(
            "native_split_quality_central_selection_already_present_or_invalid"
        )
    request_sha = _sha256(source_request_sha256)
    result_sha = _sha256(central_result_sha256)
    producer_sha = _sha256(producer.get("binding_sha256"))
    if not request_sha or not result_sha or not producer_sha:
        raise ValueError("native_split_quality_central_selection_hash_invalid")
    preselection = producer.get("preselection")
    if not isinstance(preselection, Mapping):
        raise ValueError("native_split_quality_central_selection_preselection_missing")
    receipt: dict[str, Any] = {
        "schema": CENTRAL_SELECTION_SCHEMA,
        "schema_version": CENTRAL_SELECTION_VERSION,
        "eval_run_id": str(central_identity.get("eval_run_id") or "").strip(),
        "source_run_id": _backend(
            central_identity.get("source_run_id"),
            central_identity.get("setup_id") or preselection.get("setup_id"),
        ),
        "model_id": _token(central_identity.get("model_id")),
        "case_id": _case(central_identity.get("case_id")),
        "setup_id": _token(central_identity.get("setup_id")),
        "backend": _backend(central_identity.get("source_run_id")),
        "task": _token(central_identity.get("task")),
        "variant": _token(central_identity.get("variant")),
        "precision": _token(preselection.get("precision")),
        "runtime_precision_identity": _token(
            central_identity.get("runtime_precision_identity")
        ),
        "producer_binding_sha256": producer_sha,
        "source_request_sha256": request_sha,
        "central_result_sha256": result_sha,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    selected = deepcopy(producer)
    selected.pop("binding_sha256", None)
    selected.update({
        "producer_binding_sha256": producer_sha,
        "source_request_sha256": request_sha,
        "central_result_sha256": result_sha,
        "central_quality_selection": receipt,
        "central_quality_selection_sha256": receipt["receipt_sha256"],
    })
    selected["binding_sha256"] = canonical_json_sha256(selected)
    verified, verified_status = validate_native_split_quality_binding(
        selected, verification_mode="portable",
    )
    if verified is None:
        raise ValueError(verified_status)
    selected_receipt, selected_status = (
        validate_central_native_split_quality_selection(verified, required=True)
    )
    if selected_receipt is None:
        raise ValueError(selected_status)
    return verified


def native_split_quality_selection_duplicates(
    binding: Mapping[str, Any],
) -> dict[str, str]:
    """Return the mandatory Command/Result/Attestation selection duplicates."""

    receipt, status = validate_central_native_split_quality_selection(
        binding, required=True,
    )
    if receipt is None:
        raise ValueError(status)
    request_sha = _sha256(binding.get("source_request_sha256"))
    return {
        "source_request_sha256": request_sha,
        "native_split_quality_source_request_sha256": request_sha,
        "native_split_quality_central_result_sha256": _sha256(
            binding.get("central_result_sha256")
        ),
        "native_split_quality_selection_sha256": _sha256(
            binding.get("central_quality_selection_sha256")
        ),
    }


def _strict_json_object(raw: bytes, *, role: str) -> dict[str, Any]:
    """Decode one JSON object while rejecting duplicate keys and non-UTF-8."""

    def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _reject(f"native_split_quality_{role}_duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
        )
    except _BindingValidationError:
        raise
    except Exception:
        _reject(f"native_split_quality_{role}_json_invalid")
    if not isinstance(value, dict):
        _reject(f"native_split_quality_{role}_json_object_required")
    return value


def _normalised_artifact_rows(
    value: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        _reject("native_split_quality_binding_artifacts_missing")
    rows: dict[str, dict[str, Any]] = {}
    for name in _REQUIRED_BINDING_ARTIFACTS:
        row = value.get(name)
        if not isinstance(row, Mapping):
            _reject(f"native_split_quality_binding_{name}_missing")
        path = str(row.get("path") or "").strip()
        sha = _sha256(row.get("sha256"))
        try:
            size = int(row.get("size_bytes"))
        except (TypeError, ValueError):
            size = 0
        if not path or not sha or size <= 0:
            _reject(f"native_split_quality_binding_{name}_invalid")
        rows[name] = {"path": path, "sha256": sha, "size_bytes": size}
    return rows


def _has_symlink_component(path: Path) -> bool:
    """Return true when any existing component, including the file, is a link."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                return True
        except OSError:
            # The caller reports the more useful inaccessible-file status.
            return False
    return False


def _read_and_hash_local_artifact(
    artifact: Mapping[str, Any], *, role: str, capture: bool,
) -> tuple[dict[str, Any], bytes | None]:
    """Read one immutable regular file without following a terminal symlink.

    Artifact paths are emitted with ``Path.resolve()`` by the producer.  A path
    containing a symlink or lexical indirection therefore cannot be the path
    that was originally sealed and is rejected rather than silently followed.
    The before/after ``fstat`` comparison also catches ordinary replacement or
    mutation while a hash is being computed.
    """

    text = str(artifact.get("path") or "").strip()
    path = Path(text)
    try:
        resolved = path.resolve(strict=True)
    except Exception:
        _reject(f"native_split_quality_local_{role}_inaccessible")
    if (
        not path.is_absolute()
        or str(path) != str(resolved)
        or _has_symlink_component(path)
    ):
        _reject(f"native_split_quality_local_{role}_path_or_symlink_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        _reject(f"native_split_quality_local_{role}_inaccessible")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_size) <= 0:
            _reject(f"native_split_quality_local_{role}_not_regular_nonempty_file")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
    except _BindingValidationError:
        raise
    except OSError:
        _reject(f"native_split_quality_local_{role}_read_failed")
    finally:
        os.close(descriptor)
    stable_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable_before != stable_after:
        _reject(f"native_split_quality_local_{role}_changed_while_hashing")
    expected_sha = _sha256(artifact.get("sha256"))
    expected_size = int(artifact.get("size_bytes") or 0)
    observed_sha = digest.hexdigest()
    if observed_sha != expected_sha:
        _reject(f"native_split_quality_local_{role}_sha256_mismatch")
    if int(after.st_size) != expected_size:
        _reject(f"native_split_quality_local_{role}_size_mismatch")
    if role == "trtexec" and not os.access(path, os.X_OK):
        _reject("native_split_quality_local_trtexec_not_executable")
    identity = {
        "path": str(path), "sha256": observed_sha,
        "size_bytes": int(after.st_size),
    }
    return identity, b"".join(chunks) if chunks is not None else None


def _normalised_dtype(value: Any) -> str:
    aliases = {
        "float": "float32", "fp32": "float32", "float32": "float32",
        "half": "float16", "fp16": "float16", "float16": "float16",
        "uint8": "uint8", "int8": "int8",
    }
    return aliases.get(_token(value), _token(value))


def _positive_shape(value: Any, *, status: str) -> list[int]:
    try:
        shape = [int(dim) for dim in value]
    except (TypeError, ValueError):
        _reject(status)
    if not shape or any(dim <= 0 for dim in shape):
        _reject(status)
    return shape


def _element_count(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def resolve_native_boundary_layout(
    runtime_shape: Any, canonical_part2_shape: Any,
) -> str:
    """Resolve an unambiguous raw-memory layout from exact tensor shapes."""

    try:
        runtime = [int(dim) for dim in list(runtime_shape or [])]
        target = [int(dim) for dim in list(canonical_part2_shape or [])]
    except (TypeError, ValueError):
        runtime = []
        target = []
    if (
        not runtime or not target
        or any(dim <= 0 for dim in runtime + target)
        or len(target) not in {3, 4}
    ):
        raise ValueError(
            "native_split_quality_boundary_layout_shape_invalid:"
            f"{runtime!r}->{target!r}"
        )

    if len(target) == 3:
        # NCW Part2 inputs can receive NWC accelerator output.  Resolve only
        # exact, positive contract shapes; equal element counts do not prove
        # a permutation, and equal C/W dimensions leave the layout ambiguous.
        n, c, w = target
        matches = []
        if runtime == target:
            matches.append("as_input")
        if runtime == [n, w, c]:
            matches.append("memory_nwc_to_ncw")
        if len(matches) != 1:
            status = "ambiguous" if len(matches) > 1 else "unresolved"
            raise ValueError(
                "native_split_quality_boundary_layout_"
                f"{status}:{runtime!r}->{target!r}"
            )
        return matches[0]

    # Hailo may squeeze singleton dimensions from a VStream descriptor.  For
    # channel vectors the canonical Part-2 tensor [1, C, 1, 1] and runtime
    # forms such as [C], [1, C], or [C, 1, 1] have the same contiguous byte
    # order.  Accept only that narrow, byte-layout-invariant case.  Equal
    # element counts alone are intentionally insufficient: spatial tensors
    # must still resolve through the exact NCHW/NHWC checks below.
    if target[0] == 1 and target[2:] == [1, 1]:
        channel_count = target[1]
        runtime_non_singletons = [dim for dim in runtime if dim != 1]
        singleton_squeeze_matches = (
            runtime_non_singletons == [channel_count]
            if channel_count != 1
            else not runtime_non_singletons
        )
        if (
            len(runtime) <= 4
            and singleton_squeeze_matches
            and _element_count(runtime) == channel_count
        ):
            return "as_input"

    if len(runtime) not in {3, 4}:
        raise ValueError(
            "native_split_quality_boundary_layout_shape_invalid:"
            f"{runtime!r}->{target!r}"
        )
    if len(runtime) == 3:
        if target[0] != 1:
            raise ValueError(
                "native_split_quality_boundary_layout_batch_ambiguous"
            )
        runtime4 = [1, *runtime]
    else:
        runtime4 = runtime
    n, c, h, w = target
    matches: list[str] = []
    if runtime4 == target:
        matches.append("as_input")
    if runtime4 == [n, h, w, c]:
        matches.append("memory_nhwc_to_nchw")
    if len(matches) != 1:
        status = "ambiguous" if len(matches) > 1 else "unresolved"
        raise ValueError(
            "native_split_quality_boundary_layout_"
            f"{status}:{runtime!r}->{target!r}"
        )
    return matches[0]


def _resolved_boundary_transform(
    *, layout: str, quantization_policy: Any,
) -> str:
    quantized = _token(quantization_policy) != "none"
    if layout == "as_input":
        return "uint8_dequant" if quantized else "identity"
    return "uint8_dequant_then_layout" if quantized else "layout_only"


def known_native_split_policy(
    *, model_id: Any, case_id: Any, setup_id: Any, backend: Any = "",
) -> dict[str, Any] | None:
    """Return only the static variant policy, never build-specific values.

    Quantization values are intentionally absent.  They must be materialised
    from metadata bound to the concrete Part1 runtime artifact.  This keeps a
    legitimate compiler/HEF change from silently reusing values learned from a
    previous run.
    """

    model = _token(model_id)
    case = _case(case_id)
    setup = _token(setup_id)
    canonical_backend = _backend(backend, setup)
    if not re.fullmatch(r"b[0-9]+", case):
        return None
    if model.startswith("resnet"):
        model_family = "resnet"
        task = "classification"
    elif model.startswith("mobilenet"):
        model_family = "mobilenet"
        task = "classification"
    elif model.startswith("regnet"):
        model_family = "regnet"
        task = "classification"
    elif model.startswith("yolo"):
        model_family = "yolo"
        task = "detection"
    else:
        return None
    common: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_VERSION,
        "model_id": model,
        "model_family": model_family,
        "case_id": case,
        "case_scope": "any_generated_split_case",
        "setup_id": setup,
        "backend": canonical_backend,
        "stage2_backend": "native_tensorrt",
        "engine_rebuild_allowed_after_quality": False,
        "boundary_metadata_required": True,
    }
    resolved: dict[str, Any] | None = None
    if canonical_backend == "hailo8_to_trt" and task == "classification":
        resolved = {
            **common,
            "task": task,
            "precision": "float32_layout_fp16",
            "hailo_format": "float32",
            "boundary_dtype": "float32",
            "boundary_layout": "memory_nhwc_to_nchw",
            "boundary_transform": "layout_only",
            "quantization_policy": "none",
            "preprocess_mode": "resize",
            "letterbox_pad_value": 0,
        }
    elif canonical_backend == "hailo8_to_trt" and model_family == "yolo":
        resolved = {
            **common,
            "task": task,
            "precision": "uint8_dequant_fp16",
            "hailo_format": "uint8",
            "boundary_dtype": "uint8",
            "boundary_layout": "memory_nhwc_to_nchw",
            "boundary_transform": "uint8_dequant_then_layout",
            "quantization_policy": "from_exact_part1_boundary_metadata",
            "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        }
    elif canonical_backend == "hailo10h_to_trt" and task == "classification":
        resolved = {
            **common,
            "task": task,
            "precision": "uint8_dequant_fp16",
            "hailo_format": "uint8",
            "boundary_dtype": "uint8",
            "boundary_layout": "as_input",
            "boundary_transform": "uint8_dequant",
            "quantization_policy": "from_exact_part1_boundary_metadata",
            "preprocess_mode": "resize",
            "letterbox_pad_value": 0,
        }
    elif canonical_backend == "hailo10h_to_trt" and model_family == "yolo":
        resolved = {
            **common,
            "task": task,
            "precision": "uint8_dequant_fp16",
            "hailo_format": "uint8",
            "boundary_dtype": "uint8",
            "boundary_layout": "as_input",
            "boundary_transform": "uint8_dequant",
            "quantization_policy": "from_exact_part1_boundary_metadata",
            "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        }
    elif canonical_backend == "deepx_to_trt":
        resolved = {
            **common,
            "task": task,
            # This is the exact quality-side engine variant.  Native must use
            # these same bytes; the legacy ``fp16`` label is not an alias.
            "precision": "float32_layout_fp16",
            "boundary_dtype": "float32",
            "boundary_layout": "as_input",
            "boundary_transform": "identity",
            "quantization_policy": "none",
            "preprocess_mode": "resize" if task == "classification" else "letterbox",
            "letterbox_pad_value": 0 if task == "classification" else 114,
        }
    if resolved is None:
        return None
    resolved["policy_sha256"] = canonical_json_sha256(resolved)
    return resolved


def _validated_artifact(value: Any, *, role: str) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(value, Mapping):
        return None, f"native_split_quality_{role}_missing"
    artifact = dict(value)
    if not str(artifact.get("path") or "").strip() or not _sha256(artifact.get("sha256")):
        return None, f"native_split_quality_{role}_invalid"
    try:
        size = int(artifact.get("size_bytes") or artifact.get("file_size_bytes"))
    except (TypeError, ValueError):
        return None, f"native_split_quality_{role}_invalid"
    if size <= 0:
        return None, f"native_split_quality_{role}_invalid"
    artifact["sha256"] = _sha256(artifact.get("sha256"))
    artifact["size_bytes"] = size
    return artifact, "artifact_identity_verified"


def materialize_native_split_preselection(
    *,
    policy: Any,
    part1_artifact: Any,
    boundary_metadata: Any,
    boundary_metadata_artifact: Any,
) -> dict[str, Any]:
    """Freeze one build-specific selection from authoritative boundary metadata.

    ``boundary_metadata`` is emitted after inspecting the concrete HEF/DXNN
    boundary and is separate from the TensorRT bridge metadata.  The former
    chooses the variant; the latter must subsequently prove that the engine was
    built from exactly that choice.
    """

    if not isinstance(policy, Mapping):
        raise ValueError("native_split_quality_policy_missing")
    policy_value = dict(policy)
    policy_sha = _sha256(policy_value.pop("policy_sha256", ""))
    if (
        policy_value.get("schema") != POLICY_SCHEMA
        or int(policy_value.get("schema_version") or 0) != POLICY_VERSION
        or not policy_sha
        or canonical_json_sha256(policy_value) != policy_sha
        or policy_value.get("engine_rebuild_allowed_after_quality") is not False
        or policy_value.get("boundary_metadata_required") is not True
    ):
        raise ValueError("native_split_quality_policy_invalid")
    policy_value["policy_sha256"] = policy_sha
    part1, part1_status = _validated_artifact(part1_artifact, role="part1_artifact")
    if part1 is None:
        raise ValueError(part1_status)
    metadata_file, metadata_file_status = _validated_artifact(
        boundary_metadata_artifact, role="boundary_metadata_artifact",
    )
    if metadata_file is None:
        raise ValueError(metadata_file_status)
    if not isinstance(boundary_metadata, Mapping):
        raise ValueError("native_split_quality_boundary_metadata_missing")
    metadata = deepcopy(dict(boundary_metadata))
    declared_metadata_sha = _sha256(metadata.pop("metadata_sha256", ""))
    if (
        metadata.get("schema") != "onnx-splitpoint/native-part1-boundary-metadata"
        or int(metadata.get("schema_version") or 0) != 1
        or not declared_metadata_sha
        or canonical_json_sha256(metadata) != declared_metadata_sha
    ):
        raise ValueError("native_split_quality_boundary_metadata_invalid")
    metadata["metadata_sha256"] = declared_metadata_sha
    expected = {
        "model_id": policy_value.get("model_id"),
        "case_id": policy_value.get("case_id"),
        "setup_id": policy_value.get("setup_id"),
        "backend": policy_value.get("backend"),
    }
    for field, wanted in expected.items():
        observed = metadata.get(field)
        matches = (
            _case(observed) == _case(wanted) if field == "case_id"
            else _backend(observed, metadata.get("setup_id")) == _backend(wanted, policy_value.get("setup_id"))
            if field == "backend" else _token(observed) == _token(wanted)
        )
        if not matches:
            raise ValueError(f"native_split_quality_boundary_metadata_{field}_mismatch")
    if (
        _sha256(metadata.get("part1_artifact_sha256")) != part1["sha256"]
        or int(metadata.get("part1_artifact_size_bytes") or 0) != part1["size_bytes"]
    ):
        raise ValueError("native_split_quality_boundary_metadata_part1_mismatch")
    if int(metadata.get("boundary_tensor_count") or 0) != 1:
        raise ValueError("native_split_quality_boundary_tensor_count_invalid")
    tensor = metadata.get("boundary_tensor")
    if not isinstance(tensor, Mapping):
        raise ValueError("native_split_quality_boundary_tensor_missing")
    if (
        not str(tensor.get("name") or "").strip()
        or not isinstance(tensor.get("shape"), list)
        or not tensor.get("shape")
        or any(int(dim) <= 0 for dim in tensor.get("shape") or [])
    ):
        raise ValueError("native_split_quality_boundary_tensor_invalid")
    dtype = _token(tensor.get("dtype"))
    expected_dtype = _token(policy_value.get("boundary_dtype"))
    if dtype != expected_dtype:
        raise ValueError("native_split_quality_boundary_dtype_mismatch")
    policy_layout = _token(policy_value.get("boundary_layout"))
    derived_layout_policy = (
        _backend(
            policy_value.get("backend"), policy_value.get("setup_id"),
        ) == "hailo10h_to_trt"
    )
    if derived_layout_policy:
        resolved_layout = resolve_native_boundary_layout(
            tensor.get("shape"), tensor.get("canonical_part2_shape"),
        )
        resolved_transform = _resolved_boundary_transform(
            layout=resolved_layout,
            quantization_policy=policy_value.get("quantization_policy"),
        )
    else:
        resolved_layout = policy_layout
        resolved_transform = _token(policy_value.get("boundary_transform"))
    if _token(metadata.get("boundary_layout")) != resolved_layout:
        raise ValueError("native_split_quality_boundary_layout_mismatch")
    metadata_transform = _token(metadata.get("boundary_transform"))
    if (
        derived_layout_policy and not metadata_transform
    ) or (
        metadata_transform and metadata_transform != resolved_transform
    ):
        raise ValueError("native_split_quality_boundary_transform_mismatch")

    scale: float | None = None
    zero_point: float | None = None
    if policy_value.get("quantization_policy") == "from_exact_part1_boundary_metadata":
        quantization = tensor.get("quantization")
        if not isinstance(quantization, Mapping):
            raise ValueError("native_split_quality_boundary_quantization_missing")
        if _token(quantization.get("source")) not in {
            "hailort_hef_output_vstream_info", "hailort_output_vstream_info",
        }:
            raise ValueError("native_split_quality_boundary_quantization_source_invalid")
        try:
            scale = float(quantization.get("scale"))
            zero_point = float(quantization.get("zero_point"))
        except (TypeError, ValueError):
            raise ValueError("native_split_quality_boundary_quantization_invalid")
        if not (scale > 0.0 and abs(zero_point) < 1.0e9):
            raise ValueError("native_split_quality_boundary_quantization_invalid")
    elif policy_value.get("quantization_policy") != "none":
        raise ValueError("native_split_quality_quantization_policy_invalid")

    selection = {
        "schema": SELECTION_SCHEMA,
        "schema_version": SELECTION_VERSION,
        **{
            field: policy_value.get(field)
            for field in (
                "model_id", "model_family", "case_id", "case_scope", "setup_id",
                "backend", "stage2_backend",
                "task", "precision", "hailo_format", "boundary_dtype", "boundary_layout",
                "boundary_transform", "preprocess_mode", "letterbox_pad_value",
                "engine_rebuild_allowed_after_quality", "quantization_policy",
            )
        },
        "policy_sha256": policy_sha,
        "part1_artifact_sha256": part1["sha256"],
        "part1_artifact_size_bytes": part1["size_bytes"],
        "boundary_metadata_sha256": declared_metadata_sha,
        "boundary_metadata_file_sha256": metadata_file["sha256"],
        "boundary_metadata_file_size_bytes": metadata_file["size_bytes"],
        "boundary_tensor_name": str(tensor.get("name")),
        "boundary_tensor_shape": [int(dim) for dim in tensor.get("shape") or []],
        "boundary_tensor_dtype": dtype,
        "dequant_scale": scale,
        "dequant_zero_point": zero_point,
        "canonical_part2_shape": [
            int(dim) for dim in tensor.get("canonical_part2_shape") or []
        ],
    }
    selection["boundary_layout"] = resolved_layout
    selection["boundary_transform"] = resolved_transform
    selection["selection_sha256"] = canonical_json_sha256(selection)
    return selection


def known_native_split_preselection(
    *, model_id: Any, case_id: Any, setup_id: Any, backend: Any = "",
) -> dict[str, Any] | None:
    """Compatibility name returning only the static, unmaterialised policy.

    Callers must pass its result to :func:`materialize_native_split_preselection`
    with the concrete Part1 and boundary-metadata artifacts before execution.
    """

    return known_native_split_policy(
        model_id=model_id, case_id=case_id, setup_id=setup_id, backend=backend,
    )


def validate_native_split_preselection(
    value: Any, *, expected_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Validate a preselection without accepting aliases or partial fields."""

    if not isinstance(value, Mapping):
        return None, "native_split_quality_preselection_missing"
    selection = dict(value)
    declared = _sha256(selection.pop("selection_sha256", ""))
    if not declared or canonical_json_sha256(selection) != declared:
        return None, "native_split_quality_preselection_sha256_mismatch"
    selection["selection_sha256"] = declared
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or int(selection.get("schema_version") or 0) != SELECTION_VERSION
        or selection.get("engine_rebuild_allowed_after_quality") is not False
    ):
        return None, "native_split_quality_preselection_schema_or_rebuild_policy_invalid"
    required = (
        "model_id", "model_family", "case_id", "case_scope", "setup_id",
        "backend", "task", "precision", "boundary_dtype",
        "boundary_layout", "boundary_transform",
        "preprocess_mode", "policy_sha256", "part1_artifact_sha256",
        "boundary_metadata_sha256", "boundary_metadata_file_sha256",
        "boundary_tensor_name", "boundary_tensor_dtype",
    )
    if any(not str(selection.get(name) or "").strip() for name in required):
        return None, "native_split_quality_preselection_incomplete"
    if _backend(selection.get("backend"), selection.get("setup_id")) not in {
        "hailo8_to_trt", "hailo10h_to_trt", "deepx_to_trt",
    }:
        return None, "native_split_quality_preselection_backend_unsupported"
    precision = _token(selection.get("precision"))
    if precision not in {
        "uint8_dequant_fp16", "uint8_cast_fp16", "float32_layout_fp16",
    }:
        return None, "native_split_quality_preselection_precision_unsupported"
    if precision == "uint8_dequant_fp16":
        try:
            scale = float(selection.get("dequant_scale"))
            zero_point = float(selection.get("dequant_zero_point"))
        except (TypeError, ValueError):
            return None, "native_split_quality_preselection_dequant_parameters_invalid"
        if not (scale > 0.0 and abs(zero_point) < 1.0e9):
            return None, "native_split_quality_preselection_dequant_parameters_invalid"
    elif selection.get("dequant_scale") is not None or selection.get("dequant_zero_point") is not None:
        return None, "native_split_quality_preselection_unexpected_dequant_parameters"
    for field in (
        "policy_sha256", "part1_artifact_sha256", "boundary_metadata_sha256",
        "boundary_metadata_file_sha256",
    ):
        if not _sha256(selection.get(field)):
            return None, f"native_split_quality_preselection_{field}_invalid"
    try:
        if (
            int(selection.get("part1_artifact_size_bytes")) <= 0
            or int(selection.get("boundary_metadata_file_size_bytes")) <= 0
        ):
            raise ValueError
        shape = [int(dim) for dim in selection.get("boundary_tensor_shape") or []]
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError
    except (TypeError, ValueError):
        return None, "native_split_quality_preselection_artifact_or_shape_invalid"
    policy = known_native_split_policy(
        model_id=selection.get("model_id"),
        case_id=selection.get("case_id"),
        setup_id=selection.get("setup_id"),
        backend=selection.get("backend"),
    )
    if policy is None or _sha256(policy.get("policy_sha256")) != _sha256(
        selection.get("policy_sha256")
    ):
        return None, "native_split_quality_preselection_policy_mismatch"
    for field in (
        "model_family", "case_scope", "task", "precision", "hailo_format",
        "boundary_dtype", "preprocess_mode", "letterbox_pad_value",
        "quantization_policy", "engine_rebuild_allowed_after_quality",
    ):
        if selection.get(field) != policy.get(field):
            return None, f"native_split_quality_preselection_policy_{field}_mismatch"
    derived_layout_policy = (
        _backend(policy.get("backend"), policy.get("setup_id"))
        == "hailo10h_to_trt"
        and bool(selection.get("canonical_part2_shape"))
    )
    if derived_layout_policy:
        try:
            resolved_layout = resolve_native_boundary_layout(
                selection.get("boundary_tensor_shape"),
                selection.get("canonical_part2_shape"),
            )
        except ValueError as exc:
            return None, str(exc)
        resolved_transform = _resolved_boundary_transform(
            layout=resolved_layout,
            quantization_policy=policy.get("quantization_policy"),
        )
        if _token(selection.get("boundary_layout")) != resolved_layout:
            return None, "native_split_quality_preselection_policy_boundary_layout_mismatch"
        if _token(selection.get("boundary_transform")) != resolved_transform:
            return None, "native_split_quality_preselection_policy_boundary_transform_mismatch"
    else:
        for field in ("boundary_layout", "boundary_transform"):
            if selection.get(field) != policy.get(field):
                return None, f"native_split_quality_preselection_policy_{field}_mismatch"
    if expected_identity:
        aliases = {
            "model": "model_id", "case": "case_id", "precision": "precision",
            "setup_id": "setup_id", "backend": "backend", "task": "task",
        }
        for expected_name, actual_name in aliases.items():
            expected = expected_identity.get(expected_name)
            if expected in (None, ""):
                continue
            observed = selection.get(actual_name)
            if expected_name == "case":
                matches = _case(expected) == _case(observed)
            elif expected_name == "backend":
                matches = _backend(expected, expected_identity.get("setup_id")) == _backend(
                    observed, selection.get("setup_id"),
                )
            else:
                matches = _token(expected) == _token(observed)
            if not matches:
                return None, f"native_split_quality_preselection_{expected_name}_mismatch"
    return selection, "hash_schema_identity_and_variant_verified"


def _validate_boundary_contract(
    binding: Mapping[str, Any], selection: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = binding.get("boundary_contract")
    if not isinstance(boundary, Mapping):
        _reject("native_split_quality_binding_boundary_contract_missing")
    boundary = dict(boundary)
    if _sha256(binding.get("boundary_contract_sha256")) != canonical_json_sha256(boundary):
        _reject("native_split_quality_binding_boundary_contract_sha256_mismatch")
    expected_boundary = {
        "precision": _token(selection.get("precision")),
        "boundary_layout": _token(selection.get("boundary_layout")),
        "boundary_transform": _token(selection.get("boundary_transform")),
        "boundary_tensor_name": str(selection.get("boundary_tensor_name") or "").strip(),
        "boundary_tensor_dtype": _normalised_dtype(selection.get("boundary_tensor_dtype")),
        "boundary_metadata_sha256": _sha256(selection.get("boundary_metadata_sha256")),
        "boundary_metadata_file_sha256": _sha256(
            selection.get("boundary_metadata_file_sha256")
        ),
    }
    for field, expected in expected_boundary.items():
        observed = boundary.get(field)
        if field.endswith("_sha256"):
            matches = _sha256(observed) == expected
        elif field == "boundary_tensor_name":
            matches = str(observed or "").strip() == expected
        elif field == "boundary_tensor_dtype":
            matches = _normalised_dtype(observed) == expected
        else:
            matches = _token(observed) == expected
        if not matches:
            _reject(f"native_split_quality_binding_boundary_{field}_mismatch")
    shape = _positive_shape(
        boundary.get("boundary_tensor_shape") or [],
        status="native_split_quality_binding_boundary_tensor_shape_mismatch",
    )
    selection_shape = _positive_shape(
        selection.get("boundary_tensor_shape") or [],
        status="native_split_quality_binding_boundary_tensor_shape_mismatch",
    )
    if shape != selection_shape:
        _reject("native_split_quality_binding_boundary_tensor_shape_mismatch")
    if selection.get("precision") == "uint8_dequant_fp16":
        for field in ("dequant_scale", "dequant_zero_point"):
            try:
                matches = float(boundary.get(field)) == float(selection.get(field))
            except (TypeError, ValueError):
                matches = False
            if not matches:
                _reject(f"native_split_quality_binding_boundary_{field}_mismatch")
    elif (
        boundary.get("dequant_scale") is not None
        or boundary.get("dequant_zero_point") is not None
    ):
        _reject("native_split_quality_binding_boundary_unexpected_dequant_parameters")
    return boundary


def _validate_boundary_metadata_payload(
    metadata_raw: Any, *, selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(metadata_raw, Mapping):
        _reject("native_split_quality_boundary_metadata_payload_missing")
    metadata = deepcopy(dict(metadata_raw))
    declared = _sha256(metadata.pop("metadata_sha256", ""))
    if not declared or canonical_json_sha256(metadata) != declared:
        _reject("native_split_quality_boundary_metadata_payload_sha256_mismatch")
    metadata["metadata_sha256"] = declared
    if (
        metadata.get("schema") != BOUNDARY_METADATA_SCHEMA
        or int(metadata.get("schema_version") or 0) != BOUNDARY_METADATA_VERSION
    ):
        _reject("native_split_quality_boundary_metadata_payload_schema_invalid")
    expected_identity = {
        "model_id": _token(selection.get("model_id")),
        "case_id": _case(selection.get("case_id")),
        "setup_id": _token(selection.get("setup_id")),
        "backend": _backend(selection.get("backend"), selection.get("setup_id")),
    }
    for field, expected in expected_identity.items():
        observed = metadata.get(field)
        if field == "case_id":
            matches = _case(observed) == expected
        elif field == "backend":
            matches = _backend(observed, metadata.get("setup_id")) == expected
        else:
            matches = _token(observed) == expected
        if not matches:
            _reject(f"native_split_quality_boundary_metadata_payload_{field}_mismatch")
    part1 = artifacts["part1_runtime"]
    if (
        _sha256(metadata.get("part1_artifact_sha256")) != part1["sha256"]
        or int(metadata.get("part1_artifact_size_bytes") or 0) != part1["size_bytes"]
    ):
        _reject("native_split_quality_boundary_metadata_payload_part1_mismatch")
    if declared != _sha256(selection.get("boundary_metadata_sha256")):
        _reject("native_split_quality_boundary_metadata_payload_selection_sha256_mismatch")
    if int(metadata.get("boundary_tensor_count") or 0) != 1:
        _reject("native_split_quality_boundary_metadata_payload_tensor_count_invalid")
    tensor = metadata.get("boundary_tensor")
    if not isinstance(tensor, Mapping):
        _reject("native_split_quality_boundary_metadata_payload_tensor_missing")
    if str(tensor.get("name") or "").strip() != str(
        selection.get("boundary_tensor_name") or ""
    ).strip():
        _reject("native_split_quality_boundary_metadata_payload_tensor_name_mismatch")
    tensor_shape = _positive_shape(
        tensor.get("shape") or [],
        status="native_split_quality_boundary_metadata_payload_tensor_shape_mismatch",
    )
    selection_shape = _positive_shape(
        selection.get("boundary_tensor_shape") or [],
        status="native_split_quality_boundary_metadata_payload_tensor_shape_mismatch",
    )
    if tensor_shape != selection_shape:
        _reject("native_split_quality_boundary_metadata_payload_tensor_shape_mismatch")
    if _normalised_dtype(tensor.get("dtype")) != _normalised_dtype(
        selection.get("boundary_tensor_dtype")
    ):
        _reject("native_split_quality_boundary_metadata_payload_tensor_dtype_mismatch")
    if _token(metadata.get("boundary_layout")) != _token(selection.get("boundary_layout")):
        _reject("native_split_quality_boundary_metadata_payload_layout_mismatch")
    quantization = tensor.get("quantization")
    if selection.get("precision") == "uint8_dequant_fp16":
        if not isinstance(quantization, Mapping):
            _reject("native_split_quality_boundary_metadata_payload_quantization_missing")
        if _token(quantization.get("source")) not in {
            "hailort_hef_output_vstream_info", "hailort_output_vstream_info",
        }:
            _reject("native_split_quality_boundary_metadata_payload_quantization_source_invalid")
        for metadata_field, selection_field in (
            ("scale", "dequant_scale"), ("zero_point", "dequant_zero_point"),
        ):
            try:
                matches = float(quantization.get(metadata_field)) == float(
                    selection.get(selection_field)
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                _reject(
                    "native_split_quality_boundary_metadata_payload_"
                    f"{metadata_field}_mismatch"
                )
    elif quantization not in (None, {}):
        _reject("native_split_quality_boundary_metadata_payload_unexpected_quantization")
    return metadata


def _validate_engine_build_receipt_payload(
    receipt_raw: Any, *, selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(receipt_raw, Mapping):
        _reject("native_split_quality_engine_build_receipt_payload_missing")
    receipt = deepcopy(dict(receipt_raw))
    inner_sha = _sha256(receipt.pop("receipt_sha256", ""))
    if not inner_sha or canonical_json_sha256(receipt) != inner_sha:
        _reject("native_split_quality_engine_build_receipt_inner_sha256_mismatch")
    receipt["receipt_sha256"] = inner_sha
    if (
        receipt.get("schema") != TRT_ENGINE_BUILD_RECEIPT_SCHEMA
        or int(receipt.get("schema_version") or 0) != TRT_ENGINE_BUILD_RECEIPT_VERSION
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
    ):
        _reject("native_split_quality_engine_build_receipt_schema_or_status_invalid")
    expected = {
        "source_onnx": artifacts["build_part2_onnx"]["path"],
        "source_onnx_sha256": artifacts["build_part2_onnx"]["sha256"],
        "engine": artifacts["engine"]["path"],
        "engine_sha256": artifacts["engine"]["sha256"],
        "trtexec": artifacts["trtexec"]["path"],
        "trtexec_sha256": artifacts["trtexec"]["sha256"],
    }
    for field, wanted in expected.items():
        observed = receipt.get(field)
        matches = _sha256(observed) == wanted if field.endswith("_sha256") else str(
            observed or ""
        ) == wanted
        if not matches:
            _reject(f"native_split_quality_engine_build_receipt_{field}_mismatch")
    command = receipt.get("command")
    if (
        not isinstance(command, list) or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        _reject("native_split_quality_engine_build_receipt_command_invalid")
    argv = [str(item) for item in command]
    if argv[0] != artifacts["trtexec"]["path"]:
        _reject("native_split_quality_engine_build_receipt_command_trtexec_mismatch")
    onnx_flags = [item for item in argv[1:] if item.startswith("--onnx=")]
    engine_flags = [item for item in argv[1:] if item.startswith("--saveEngine=")]
    if onnx_flags != [f"--onnx={artifacts['build_part2_onnx']['path']}"]:
        _reject("native_split_quality_engine_build_receipt_command_source_mismatch")
    if engine_flags != [f"--saveEngine={artifacts['engine']['path']}"]:
        _reject("native_split_quality_engine_build_receipt_command_engine_mismatch")
    precision = _token(selection.get("precision"))
    if precision in {
        "uint8_dequant_fp16", "uint8_cast_fp16", "float32_layout_fp16",
    }:
        if argv[1:].count("--fp16") != 1 or "--int8" in argv[1:]:
            _reject("native_split_quality_engine_build_receipt_command_precision_mismatch")
    else:
        _reject("native_split_quality_engine_build_receipt_precision_unsupported")
    return receipt, inner_sha, canonical_json_sha256(receipt)


def _validate_native_trt_meta_payload(
    meta_raw: Any, *, selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]], receipt: Mapping[str, Any],
    build_command_sha256: str,
) -> dict[str, Any]:
    """Validate the builder's strict entry schema and every artifact edge."""

    if not isinstance(meta_raw, Mapping):
        _reject("native_split_quality_native_trt_meta_payload_missing")
    meta = deepcopy(dict(meta_raw))
    if (
        meta.get("schema") != "onnx-splitpoint/native-trt-meta"
        or int(meta.get("schema_version") or 0) != 1
        or _case(meta.get("case")) != _case(selection.get("case_id"))
        or _token(meta.get("variant")) != "part2"
        or meta.get("build_ok") is not True
        or meta.get("inputs_static") is not True
    ):
        _reject("native_split_quality_native_trt_meta_entry_schema_invalid")
    for field, artifact_name in (
        ("onnx", "build_part2_onnx"),
        ("source_onnx", "build_part2_onnx"),
        ("engine", "engine"),
        ("engine_build_receipt_path", "engine_build_receipt"),
    ):
        if str(meta.get(field) or "") != artifacts[artifact_name]["path"]:
            _reject(f"native_split_quality_native_trt_meta_{field}_mismatch")
    if (
        _token(meta.get("precision")) != _token(selection.get("precision"))
        or _token(meta.get("requested_precision")) != _token(selection.get("precision"))
    ):
        _reject("native_split_quality_native_trt_meta_precision_mismatch")
    inputs = meta.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], Mapping):
        _reject("native_split_quality_native_trt_meta_input_count_invalid")
    input_row = dict(inputs[0])
    if str(input_row.get("name") or "").strip() != str(
        selection.get("boundary_tensor_name") or ""
    ).strip():
        _reject("native_split_quality_native_trt_meta_input_name_mismatch")
    if bool(input_row.get("has_dynamic")):
        _reject("native_split_quality_native_trt_meta_dynamic_input_forbidden")
    input_shape = _positive_shape(
        input_row.get("shape") or [],
        status="native_split_quality_native_trt_meta_input_shape_invalid",
    )
    boundary_shape = _positive_shape(
        selection.get("boundary_tensor_shape") or [],
        status="native_split_quality_native_trt_meta_input_shape_invalid",
    )
    if _element_count(input_shape) != _element_count(boundary_shape):
        _reject("native_split_quality_native_trt_meta_input_element_count_mismatch")
    expected_input_dtype = (
        "uint8" if selection.get("precision") in {
            "uint8_dequant_fp16", "uint8_cast_fp16",
        } else "float32"
    )
    if _normalised_dtype(input_row.get("elem_type")) != expected_input_dtype:
        _reject("native_split_quality_native_trt_meta_input_dtype_mismatch")

    bridge = meta.get("uint8_cast_bridge")
    if not isinstance(bridge, Mapping):
        _reject("native_split_quality_native_trt_meta_bridge_missing")
    bridge = dict(bridge)
    expected_bridge_schema = {
        "uint8_dequant_fp16": ("onnx-splitpoint/uint8-dequant-bridge", 2, "uint8"),
        "uint8_cast_fp16": ("onnx-splitpoint/uint8-cast-bridge", 1, "uint8"),
        "float32_layout_fp16": ("onnx-splitpoint/float32-layout-bridge", 1, "float32"),
    }.get(_token(selection.get("precision")))
    if expected_bridge_schema is None:
        _reject("native_split_quality_native_trt_meta_bridge_precision_unsupported")
    schema, version, bridge_dtype = expected_bridge_schema
    if (
        bridge.get("schema") != schema
        or int(bridge.get("schema_version") or 0) != version
        or str(bridge.get("source") or "") != artifacts["source_part2_onnx"]["path"]
        or _sha256(bridge.get("source_sha256"))
        != artifacts["source_part2_onnx"]["sha256"]
        or str(bridge.get("bridge") or "") != artifacts["build_part2_onnx"]["path"]
        or _sha256(bridge.get("bridge_sha256"))
        != artifacts["build_part2_onnx"]["sha256"]
        or str(bridge.get("input_name") or "").strip()
        != str(selection.get("boundary_tensor_name") or "").strip()
        or _normalised_dtype(bridge.get("input_dtype")) != bridge_dtype
    ):
        _reject("native_split_quality_native_trt_meta_bridge_schema_or_artifact_mismatch")
    layout = bridge.get("boundary_layout")
    if selection.get("precision") != "uint8_cast_fp16":
        if not isinstance(layout, Mapping) or _token(layout.get("effective")) != _token(
            selection.get("boundary_layout")
        ):
            _reject("native_split_quality_native_trt_meta_bridge_layout_mismatch")
        bridge_input_shape = _positive_shape(
            bridge.get("input_shape") or [],
            status="native_split_quality_native_trt_meta_bridge_input_shape_invalid",
        )
        if bridge_input_shape != input_shape:
            _reject("native_split_quality_native_trt_meta_bridge_input_shape_mismatch")
    if selection.get("precision") == "uint8_dequant_fp16":
        for bridge_field, selection_field in (
            ("scale", "dequant_scale"), ("zero_point", "dequant_zero_point"),
        ):
            try:
                matches = float(bridge.get(bridge_field)) == float(
                    selection.get(selection_field)
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                _reject(
                    "native_split_quality_native_trt_meta_bridge_"
                    f"{bridge_field}_mismatch"
                )
    if meta.get("engine_build_receipt_status") != "engine_build_receipt_verified":
        _reject("native_split_quality_native_trt_meta_receipt_status_invalid")
    if dict(meta.get("engine_build_receipt") or {}) != dict(receipt):
        _reject("native_split_quality_native_trt_meta_receipt_payload_mismatch")
    build = meta.get("build")
    returncode = build.get("returncode") if isinstance(build, Mapping) else None
    if (
        not isinstance(build, Mapping)
        or isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or returncode != 0
    ):
        _reject("native_split_quality_native_trt_meta_build_status_invalid")
    build_command = build.get("cmd")
    if not isinstance(build_command, list) or canonical_json_sha256(build_command) != build_command_sha256:
        _reject("native_split_quality_native_trt_meta_build_command_mismatch")
    if list(build_command) != list(receipt.get("command") or []):
        _reject("native_split_quality_native_trt_meta_receipt_command_mismatch")
    return meta


def _local_evidence_from_files(
    binding: Mapping[str, Any], *, selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-open every artifact and return a portable, hash-bound proof."""

    verified_artifacts: dict[str, dict[str, Any]] = {}
    json_bytes: dict[str, bytes] = {}
    for name in _REQUIRED_BINDING_ARTIFACTS:
        identity, captured = _read_and_hash_local_artifact(
            artifacts[name], role=name, capture=name in _EMBEDDED_JSON_ARTIFACTS,
        )
        verified_artifacts[name] = identity
        if captured is not None:
            json_bytes[name] = captured

    boundary_metadata = _strict_json_object(
        json_bytes["boundary_metadata"], role="boundary_metadata",
    )
    native_meta_payload = _strict_json_object(
        json_bytes["native_trt_meta"], role="native_trt_meta",
    )
    receipt = _strict_json_object(
        json_bytes["engine_build_receipt"], role="engine_build_receipt",
    )
    boundary_metadata = _validate_boundary_metadata_payload(
        boundary_metadata, selection=selection, artifacts=artifacts,
    )
    receipt, receipt_inner_sha, receipt_outer_sha = (
        _validate_engine_build_receipt_payload(
            receipt, selection=selection, artifacts=artifacts,
        )
    )
    build_command_sha = canonical_json_sha256(receipt["command"])
    native_meta_payload = _validate_native_trt_meta_payload(
        native_meta_payload, selection=selection, artifacts=artifacts,
        receipt=receipt, build_command_sha256=build_command_sha,
    )

    build_summary_row = binding.get("native_trt_meta")
    if not isinstance(build_summary_row, Mapping):
        _reject("native_split_quality_build_summary_row_missing")
    # Today the builder writes this exact entry both to the summary and to
    # native_trt_meta.json.  Keep the roles separate in the binding anyway so a
    # future wrapper cannot silently change which value was actually parsed.
    if dict(build_summary_row) != native_meta_payload:
        _reject("native_split_quality_build_summary_native_meta_payload_mismatch")
    embedded_receipt = binding.get("engine_build_receipt")
    if not isinstance(embedded_receipt, Mapping) or dict(embedded_receipt) != receipt:
        _reject("native_split_quality_embedded_engine_build_receipt_mismatch")

    json_evidence: dict[str, dict[str, Any]] = {}
    parsed_values = {
        "boundary_metadata": boundary_metadata,
        "native_trt_meta": native_meta_payload,
        "engine_build_receipt": receipt,
    }
    for name in _EMBEDDED_JSON_ARTIFACTS:
        raw = json_bytes[name]
        parsed = parsed_values[name]
        json_evidence[name] = {
            "encoding": "base64",
            "content_base64": base64.b64encode(raw).decode("ascii"),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "file_size_bytes": len(raw),
            "canonical_value_sha256": canonical_json_sha256(parsed),
        }
    proof: dict[str, Any] = {
        "schema": LOCAL_VERIFICATION_SCHEMA,
        "schema_version": LOCAL_VERIFICATION_VERSION,
        "verification_kind": "producer_local_file_rehash",
        "artifact_names": list(_REQUIRED_BINDING_ARTIFACTS),
        "artifacts": verified_artifacts,
        "artifact_set_sha256": canonical_json_sha256(verified_artifacts),
        "embedded_json_files": json_evidence,
        "boundary_metadata_value_sha256": canonical_json_sha256(boundary_metadata),
        "native_trt_meta_payload_sha256": canonical_json_sha256(native_meta_payload),
        "build_summary_row_sha256": canonical_json_sha256(dict(build_summary_row)),
        "engine_build_receipt_inner_sha256": receipt_inner_sha,
        "engine_build_receipt_outer_sha256": receipt_outer_sha,
        "engine_build_receipt_canonical_size_bytes": len(
            _canonical_json_bytes(receipt)
        ),
        "trt_build_command_sha256": build_command_sha,
    }
    proof["proof_sha256"] = canonical_json_sha256(proof)
    return {
        "boundary_metadata_payload": boundary_metadata,
        "boundary_metadata_value_sha256": canonical_json_sha256(boundary_metadata),
        "native_trt_meta_payload": native_meta_payload,
        "native_trt_meta_payload_sha256": canonical_json_sha256(native_meta_payload),
        "native_trt_meta_file_sha256": artifacts["native_trt_meta"]["sha256"],
        "native_trt_meta_file_size_bytes": artifacts["native_trt_meta"]["size_bytes"],
        "engine_build_receipt_sha256": receipt_inner_sha,
        "engine_build_receipt_outer_sha256": receipt_outer_sha,
        "engine_build_receipt_canonical_size_bytes": len(
            _canonical_json_bytes(receipt)
        ),
        "engine_build_receipt_file_sha256": artifacts["engine_build_receipt"]["sha256"],
        "engine_build_receipt_file_size_bytes": artifacts["engine_build_receipt"]["size_bytes"],
        "trt_build_command_sha256": build_command_sha,
        "local_artifact_verification": proof,
    }


def _portable_embedded_evidence(
    binding: Mapping[str, Any], *, selection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate copied evidence without claiming that remote paths were read."""

    proof_raw = binding.get("local_artifact_verification")
    if not isinstance(proof_raw, Mapping):
        _reject("native_split_quality_local_artifact_verification_missing")
    proof = deepcopy(dict(proof_raw))
    declared_proof_sha = _sha256(proof.pop("proof_sha256", ""))
    if not declared_proof_sha or canonical_json_sha256(proof) != declared_proof_sha:
        _reject("native_split_quality_local_artifact_verification_sha256_mismatch")
    proof["proof_sha256"] = declared_proof_sha
    if (
        proof.get("schema") != LOCAL_VERIFICATION_SCHEMA
        or int(proof.get("schema_version") or 0) != LOCAL_VERIFICATION_VERSION
        or proof.get("verification_kind") != "producer_local_file_rehash"
        or proof.get("artifact_names") != list(_REQUIRED_BINDING_ARTIFACTS)
    ):
        _reject("native_split_quality_local_artifact_verification_schema_invalid")
    proof_artifacts = proof.get("artifacts")
    if not isinstance(proof_artifacts, Mapping):
        _reject("native_split_quality_local_artifact_verification_artifacts_missing")
    normalised_proof_artifacts = _normalised_artifact_rows(proof_artifacts)
    if normalised_proof_artifacts != dict(artifacts):
        _reject("native_split_quality_local_artifact_verification_artifacts_mismatch")
    if _sha256(proof.get("artifact_set_sha256")) != canonical_json_sha256(
        normalised_proof_artifacts
    ):
        _reject("native_split_quality_local_artifact_verification_artifact_set_sha256_mismatch")

    evidence = proof.get("embedded_json_files")
    if not isinstance(evidence, Mapping) or set(evidence) != set(_EMBEDDED_JSON_ARTIFACTS):
        _reject("native_split_quality_embedded_json_evidence_incomplete")
    parsed: dict[str, dict[str, Any]] = {}
    for name in _EMBEDDED_JSON_ARTIFACTS:
        row = evidence.get(name)
        if not isinstance(row, Mapping) or row.get("encoding") != "base64":
            _reject(f"native_split_quality_embedded_{name}_encoding_invalid")
        try:
            raw = base64.b64decode(
                str(row.get("content_base64") or ""), validate=True,
            )
        except Exception:
            _reject(f"native_split_quality_embedded_{name}_base64_invalid")
        if (
            len(raw) != int(row.get("file_size_bytes") or 0)
            or len(raw) != artifacts[name]["size_bytes"]
        ):
            _reject(f"native_split_quality_embedded_{name}_size_mismatch")
        file_sha = hashlib.sha256(raw).hexdigest()
        if (
            file_sha != _sha256(row.get("file_sha256"))
            or file_sha != artifacts[name]["sha256"]
        ):
            _reject(f"native_split_quality_embedded_{name}_file_sha256_mismatch")
        parsed[name] = _strict_json_object(raw, role=f"embedded_{name}")
        if canonical_json_sha256(parsed[name]) != _sha256(
            row.get("canonical_value_sha256")
        ):
            _reject(f"native_split_quality_embedded_{name}_canonical_sha256_mismatch")

    boundary_metadata = _validate_boundary_metadata_payload(
        parsed["boundary_metadata"], selection=selection, artifacts=artifacts,
    )
    receipt, receipt_inner_sha, receipt_outer_sha = (
        _validate_engine_build_receipt_payload(
            parsed["engine_build_receipt"], selection=selection,
            artifacts=artifacts,
        )
    )
    build_command_sha = canonical_json_sha256(receipt["command"])
    native_meta = _validate_native_trt_meta_payload(
        parsed["native_trt_meta"], selection=selection, artifacts=artifacts,
        receipt=receipt, build_command_sha256=build_command_sha,
    )
    expected_values = {
        "boundary_metadata_payload": boundary_metadata,
        "native_trt_meta_payload": native_meta,
        "engine_build_receipt": receipt,
    }
    for field, expected in expected_values.items():
        observed = binding.get(field)
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            _reject(f"native_split_quality_{field}_embedded_bytes_mismatch")
    summary = binding.get("native_trt_meta")
    if not isinstance(summary, Mapping) or dict(summary) != native_meta:
        _reject("native_split_quality_build_summary_native_meta_payload_mismatch")

    duplicate_hashes = {
        "boundary_metadata_value_sha256": canonical_json_sha256(boundary_metadata),
        "native_trt_meta_payload_sha256": canonical_json_sha256(native_meta),
        "native_trt_meta_sha256": canonical_json_sha256(dict(summary)),
        "native_trt_meta_file_sha256": artifacts["native_trt_meta"]["sha256"],
        "engine_build_receipt_sha256": receipt_inner_sha,
        "engine_build_receipt_outer_sha256": receipt_outer_sha,
        "engine_build_receipt_file_sha256": artifacts["engine_build_receipt"]["sha256"],
        "trt_build_command_sha256": build_command_sha,
    }
    for field, expected in duplicate_hashes.items():
        if _sha256(binding.get(field)) != expected:
            _reject(f"native_split_quality_{field}_mismatch")
    for field, expected in (
        ("native_trt_meta_file_size_bytes", artifacts["native_trt_meta"]["size_bytes"]),
        (
            "engine_build_receipt_canonical_size_bytes",
            len(_canonical_json_bytes(receipt)),
        ),
        (
            "engine_build_receipt_file_size_bytes",
            artifacts["engine_build_receipt"]["size_bytes"],
        ),
    ):
        try:
            observed_size = int(binding.get(field))
        except (TypeError, ValueError):
            observed_size = 0
        if observed_size != expected:
            _reject(f"native_split_quality_{field}_mismatch")
    proof_duplicates = {
        "boundary_metadata_value_sha256": duplicate_hashes[
            "boundary_metadata_value_sha256"
        ],
        "native_trt_meta_payload_sha256": duplicate_hashes[
            "native_trt_meta_payload_sha256"
        ],
        "build_summary_row_sha256": duplicate_hashes["native_trt_meta_sha256"],
        "engine_build_receipt_inner_sha256": receipt_inner_sha,
        "engine_build_receipt_outer_sha256": receipt_outer_sha,
        "engine_build_receipt_canonical_size_bytes": len(
            _canonical_json_bytes(receipt)
        ),
        "trt_build_command_sha256": build_command_sha,
    }
    for field, expected in proof_duplicates.items():
        if field.endswith("_size_bytes"):
            try:
                matches = int(proof.get(field)) == int(expected)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = _sha256(proof.get(field)) == expected
        if not matches:
            _reject(
                f"native_split_quality_local_artifact_verification_{field}_mismatch"
            )


def seal_native_split_quality_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a producer binding only after re-opening every local artifact.

    The returned object contains the exact bytes of its three JSON contracts,
    but only content identities for large binaries.  This lets management
    perform a genuinely portable value audit while reserving the phrase
    "locally re-hashed" for this producer step and the later Native bind.
    """

    binding = deepcopy(dict(payload))
    binding.pop("binding_sha256", None)
    binding.pop("local_artifact_verification", None)
    binding["schema"] = BINDING_SCHEMA
    binding["schema_version"] = BINDING_VERSION
    selection, status = validate_native_split_preselection(binding.get("preselection"))
    if selection is None:
        raise ValueError(status)
    try:
        if _backend(binding.get("source_run_id")) != _backend(
            selection.get("backend")
        ):
            _reject("native_split_quality_binding_source_run_id_mismatch")
        artifacts = _normalised_artifact_rows(binding.get("artifacts"))
        _validate_boundary_contract(binding, selection)
        if _sha256(binding.get("preselection_sha256")) != selection["selection_sha256"]:
            _reject("native_split_quality_binding_preselection_duplicate_mismatch")
        for artifact_name, sha_field, size_field in (
            ("part1_runtime", "part1_artifact_sha256", "part1_artifact_size_bytes"),
            (
                "boundary_metadata", "boundary_metadata_file_sha256",
                "boundary_metadata_file_size_bytes",
            ),
        ):
            if (
                artifacts[artifact_name]["sha256"] != _sha256(selection.get(sha_field))
                or artifacts[artifact_name]["size_bytes"]
                != int(selection.get(size_field) or 0)
            ):
                _reject(
                    f"native_split_quality_binding_{artifact_name}_selection_mismatch"
                )
        binding.update(_local_evidence_from_files(
            binding, selection=selection, artifacts=artifacts,
        ))
    except _BindingValidationError as exc:
        raise ValueError(str(exc)) from exc
    binding["binding_sha256"] = canonical_json_sha256(binding)
    verified, verified_status = validate_native_split_quality_binding(
        binding, verification_mode="local",
    )
    if verified is None:
        raise ValueError(verified_status)
    return verified


def validate_native_split_quality_binding(
    value: Any, *, expected_identity: Mapping[str, Any] | None = None,
    verification_mode: str = "portable",
) -> tuple[dict[str, Any] | None, str]:
    """Validate a quality binding in explicit portable or local mode.

    ``portable`` verifies the exact embedded JSON bytes, hashes and cross-links
    and deliberately does not inspect paths that belong to a remote host.
    ``local`` additionally re-opens and re-hashes all eight artifacts.  It is
    mandatory both when sealing the producer and immediately before Native
    execution.
    """

    mode = _token(verification_mode)
    if mode not in {"portable", "local"}:
        return None, "native_split_quality_binding_verification_mode_invalid"
    if not isinstance(value, Mapping):
        return None, "native_split_quality_binding_missing"
    binding = deepcopy(dict(value))
    declared = _sha256(binding.pop("binding_sha256", ""))
    if not declared or canonical_json_sha256(binding) != declared:
        return None, "native_split_quality_binding_sha256_mismatch"
    binding["binding_sha256"] = declared
    if (
        binding.get("schema") != BINDING_SCHEMA
        or int(binding.get("schema_version") or 0) != BINDING_VERSION
        or binding.get("quality_completed") is not True
        or binding.get("performance_claims_emitted") is not False
        or not str(binding.get("eval_run_id") or "").strip()
    ):
        return None, "native_split_quality_binding_schema_or_role_invalid"
    selection, status = validate_native_split_preselection(
        binding.get("preselection"), expected_identity=expected_identity,
    )
    if selection is None:
        return None, status
    central_selection, central_selection_status = (
        validate_central_native_split_quality_selection(binding)
    )
    if central_selection is None and central_selection_status != (
        "native_split_quality_central_selection_not_present"
    ):
        return None, central_selection_status
    try:
        if _backend(binding.get("source_run_id")) != _backend(
            selection.get("backend")
        ):
            _reject("native_split_quality_binding_source_run_id_mismatch")
        if _sha256(binding.get("preselection_sha256")) != selection["selection_sha256"]:
            _reject("native_split_quality_binding_preselection_duplicate_mismatch")
        artifacts = _normalised_artifact_rows(binding.get("artifacts"))
        for artifact_name, sha_field, size_field in (
            ("part1_runtime", "part1_artifact_sha256", "part1_artifact_size_bytes"),
            (
                "boundary_metadata", "boundary_metadata_file_sha256",
                "boundary_metadata_file_size_bytes",
            ),
        ):
            if (
                artifacts[artifact_name]["sha256"] != _sha256(selection.get(sha_field))
                or artifacts[artifact_name]["size_bytes"]
                != int(selection.get(size_field) or 0)
            ):
                _reject(
                    f"native_split_quality_binding_{artifact_name}_selection_mismatch"
                )
        _validate_boundary_contract(binding, selection)
        _portable_embedded_evidence(
            binding, selection=selection, artifacts=artifacts,
        )
        if mode == "local":
            local_fields = _local_evidence_from_files(
                binding, selection=selection, artifacts=artifacts,
            )
            for field, expected in local_fields.items():
                observed = binding.get(field)
                if observed != expected:
                    _reject(f"native_split_quality_local_{field}_drift")
    except _BindingValidationError as exc:
        return None, str(exc)
    if mode == "local":
        return binding, "local_files_rehashed_and_exact_cross_links_verified"
    return (
        binding,
        "portable_embedded_evidence_and_cross_links_verified_without_local_rehash",
    )


def _validate_portable_consumer_attestation(
    *, native_row: Mapping[str, Any], binding: Mapping[str, Any],
    command: Mapping[str, Any], expected_identity: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Verify the remote consumer's sealed local-rehash/command attestation."""

    raw = native_row.get("native_split_quality_consumer_attestation")
    if not isinstance(raw, Mapping):
        return None, "native_split_quality_consumer_attestation_missing"
    attestation = deepcopy(dict(raw))
    declared = _sha256(attestation.pop("attestation_sha256", ""))
    if not declared or canonical_json_sha256(attestation) != declared:
        return None, "native_split_quality_consumer_attestation_sha256_mismatch"
    attestation["attestation_sha256"] = declared
    if (
        attestation.get("schema") != CONSUMER_ATTESTATION_SCHEMA
        or int(attestation.get("schema_version") or 0)
        != CONSUMER_ATTESTATION_VERSION
        or attestation.get("status")
        != "local_files_rehashed_and_exact_command_join_verified"
    ):
        return None, "native_split_quality_consumer_attestation_schema_or_status_invalid"
    expected_values = {
        "binding_sha256": _sha256(binding.get("binding_sha256")),
        "command_contract_sha256": _sha256(command.get("contract_sha256")),
        "eval_run_id": str(binding.get("eval_run_id") or "").strip(),
        "source_run_id": str(binding.get("source_run_id") or "").strip(),
        "backend": _backend(expected_identity.get("backend")),
        "model_id": _token(expected_identity.get("model")),
        "case_id": _case(expected_identity.get("case")),
        "setup_id": _token(expected_identity.get("setup_id")),
        "task": _token(expected_identity.get("task")),
        "precision": _token(expected_identity.get("precision")),
        "local_artifact_verification_sha256": canonical_json_sha256(
            binding.get("local_artifact_verification")
        ),
    }
    central_selection, central_selection_status = (
        validate_central_native_split_quality_selection(binding)
    )
    if central_selection is None and central_selection_status != (
        "native_split_quality_central_selection_not_present"
    ):
        return None, central_selection_status
    if central_selection is not None:
        expected_values.update({
            "native_split_quality_source_request_sha256": _sha256(
                binding.get("source_request_sha256")
            ),
            "native_split_quality_central_result_sha256": _sha256(
                binding.get("central_result_sha256")
            ),
            "native_split_quality_selection_sha256": _sha256(
                binding.get("central_quality_selection_sha256")
            ),
        })
    for field, expected in expected_values.items():
        observed = attestation.get(field)
        if field.endswith("_sha256"):
            matches = _sha256(observed) == expected
        elif field == "backend":
            matches = _backend(observed) == expected
        elif field == "case_id":
            matches = _case(observed) == expected
        elif field == "source_run_id":
            matches = _backend(
                observed, expected_identity.get("setup_id"),
            ) == _backend(expected, expected_identity.get("setup_id"))
        elif field == "eval_run_id":
            matches = str(observed or "").strip() == expected
        else:
            matches = _token(observed) == expected
        if not expected or not matches:
            return None, f"native_split_quality_consumer_attestation_{field}_mismatch"
    command_artifacts = command.get("artifacts")
    command_artifacts = (
        command_artifacts if isinstance(command_artifacts, Mapping) else {}
    )
    for field, artifact_name in (
        ("semantic_output_manifest_sha256", "semantic_output_manifest"),
        ("semantic_boundary_manifest_sha256", "semantic_boundary_manifest"),
    ):
        artifact = command_artifacts.get(artifact_name)
        if not isinstance(artifact, Mapping):
            return None, (
                "native_split_quality_consumer_attestation_"
                f"{artifact_name}_missing"
            )
        artifact_sha = _sha256(artifact.get("sha256"))
        if not artifact_sha or _sha256(attestation.get(field)) != artifact_sha:
            return None, f"native_split_quality_consumer_attestation_{field}_mismatch"
    return attestation, "portable_consumer_attestation_exactly_verified"


def bind_quality_to_native_split(
    *, native_row: Mapping[str, Any], quality_binding: Any,
    verification_mode: str = "local",
) -> tuple[dict[str, Any] | None, str]:
    """Prove that Quality and Native used one selected engine/boundary variant.

    The Native command contract is independently sealed.  This function does
    not trust convenience booleans from either producer; it verifies the command
    hash, then compares the actual artifact hashes and boundary identity.
    """

    mode = _token(verification_mode)
    if mode not in {"local", "portable"}:
        return None, "native_split_quality_binding_verification_mode_invalid"
    expected_identity = {
        "backend": native_row.get("backend"),
        "model": native_row.get("model") or native_row.get("model_id"),
        "case": native_row.get("case") or native_row.get("case_id"),
        "precision": (
            native_row.get("runtime_precision_identity")
            or native_row.get("execution_precision")
            or native_row.get("precision")
        ),
        "setup_id": native_row.get("setup_id"),
        "task": native_row.get("task"),
        "comparison_backend": native_row.get("comparison_backend"),
    }
    binding, status = validate_native_split_quality_binding(
        quality_binding, expected_identity=expected_identity,
        verification_mode=mode,
    )
    if binding is None:
        return None, status
    central_selection, central_selection_status = (
        validate_central_native_split_quality_selection(binding)
    )
    if central_selection is None and central_selection_status != (
        "native_split_quality_central_selection_not_present"
    ):
        return None, central_selection_status
    command_raw = native_row.get("native_command_contract")
    command, command_status = verify_native_command_contract(
        command_raw, expected_identity=expected_identity,
    )
    if command is None:
        return None, f"native_split_quality_{command_status}"
    declared_command_sha = _sha256(native_row.get("native_command_contract_sha256"))
    if not declared_command_sha or declared_command_sha != _sha256(command.get("contract_sha256")):
        return None, "native_split_quality_native_command_contract_duplicate_mismatch"
    binding_sha = _sha256(binding.get("binding_sha256"))
    if (
        not binding_sha
        or _sha256(command.get("native_split_quality_binding_sha256"))
        != binding_sha
    ):
        return None, "native_split_quality_command_binding_sha256_mismatch"
    command_binding = command.get("native_split_quality_binding")
    if not isinstance(command_binding, Mapping) or dict(command_binding) != binding:
        return None, "native_split_quality_command_binding_payload_mismatch"
    command_local_verification = command.get(
        "native_split_quality_local_verification"
    )
    if (
        not isinstance(command_local_verification, Mapping)
        or dict(command_local_verification)
        != dict(binding.get("local_artifact_verification") or {})
    ):
        return None, "native_split_quality_command_local_verification_mismatch"
    binding_eval_run_id = str(binding.get("eval_run_id") or "").strip()
    binding_source_run_id = str(binding.get("source_run_id") or "").strip()
    if str(command.get("native_split_quality_eval_run_id") or "").strip() != binding_eval_run_id:
        return None, "native_split_quality_command_eval_run_id_mismatch"
    if _backend(
        command.get("native_split_quality_source_run_id"),
        expected_identity.get("setup_id"),
    ) != _backend(binding_source_run_id, expected_identity.get("setup_id")):
        return None, "native_split_quality_command_source_run_id_mismatch"
    if central_selection is not None:
        selected_hashes = {
            "native_split_quality_source_request_sha256": _sha256(
                binding.get("source_request_sha256")
            ),
            "native_split_quality_central_result_sha256": _sha256(
                binding.get("central_result_sha256")
            ),
            "native_split_quality_selection_sha256": _sha256(
                binding.get("central_quality_selection_sha256")
            ),
        }
        for field, expected in selected_hashes.items():
            if not expected or _sha256(command.get(field)) != expected:
                return None, f"native_split_quality_command_{field}_mismatch"
        # Once management selects a Central result these Native-result
        # duplicates are mandatory, not convenience metadata.
        required_native_values = {
            **selected_hashes,
            "eval_run_id": binding_eval_run_id,
            "native_split_quality_eval_run_id": binding_eval_run_id,
            "source_request_sha256": selected_hashes[
                "native_split_quality_source_request_sha256"
            ],
        }
        for field, expected in required_native_values.items():
            observed = native_row.get(field)
            matches = (
                _sha256(observed) == expected
                if field.endswith("_sha256")
                else str(observed or "").strip() == expected
            )
            if not matches:
                return None, f"native_split_quality_native_row_{field}_mismatch"
    selection = binding["preselection"]
    canonical_backends = {
        _backend(native_row.get("backend")),
        _backend(command.get("backend")),
        _backend(selection.get("backend")),
        _backend(binding_source_run_id),
    }
    if "" in canonical_backends or len(canonical_backends) != 1:
        return None, "native_split_quality_command_backend_canonical_mismatch"
    for field, expected in (
        ("native_split_quality_binding_sha256", binding_sha),
        ("native_split_quality_eval_run_id", binding_eval_run_id),
        ("native_split_quality_source_run_id", binding_source_run_id),
    ):
        if field not in native_row:
            continue
        observed = native_row.get(field)
        if field.endswith("_sha256"):
            matches = _sha256(observed) == expected
        elif field == "native_split_quality_source_run_id":
            matches = _backend(
                observed, expected_identity.get("setup_id"),
            ) == _backend(expected, expected_identity.get("setup_id"))
        else:
            matches = str(observed or "").strip() == expected
        if not matches:
            return None, f"native_split_quality_native_row_{field}_mismatch"
    command_selection, command_selection_status = validate_native_split_preselection(
        command.get("quality_preselection"), expected_identity=expected_identity,
    )
    if command_selection is None:
        return None, f"native_split_quality_command_{command_selection_status}"
    if command_selection != selection:
        return None, "native_split_quality_preselection_drift"
    if _sha256(command.get("quality_preselection_sha256")) != selection["selection_sha256"]:
        return None, "native_split_quality_command_preselection_duplicate_mismatch"

    command_artifacts = command.get("artifacts")
    command_artifacts = command_artifacts if isinstance(command_artifacts, Mapping) else {}
    quality_artifacts = binding["artifacts"]
    backend = _backend(native_row.get("backend"), native_row.get("setup_id"))
    artifact_pairs = {
        "part1_runtime": "dxnn" if backend == "deepx_to_trt" else "hef",
        "boundary_metadata": "boundary_metadata",
        "source_part2_onnx": "source_part2_onnx",
        "build_part2_onnx": "build_part2_onnx",
        "engine": "engine",
        "native_trt_meta": "native_trt_meta",
        "engine_build_receipt": "engine_build_receipt",
        "trtexec": "trtexec",
    }
    for quality_name, command_name in artifact_pairs.items():
        quality_artifact = quality_artifacts.get(quality_name)
        command_artifact = command_artifacts.get(command_name)
        if not isinstance(command_artifact, Mapping):
            return None, f"native_split_quality_command_{command_name}_missing"
        if str(command_artifact.get("path") or "") != str(
            quality_artifact.get("path") or ""
        ):
            return None, f"native_split_quality_{quality_name}_path_mismatch"
        if _sha256(command_artifact.get("sha256")) != _sha256(quality_artifact.get("sha256")):
            return None, f"native_split_quality_{quality_name}_sha256_mismatch"
        try:
            command_size = int(
                command_artifact.get("size_bytes")
                or command_artifact.get("file_size_bytes")
            )
            quality_size = int(quality_artifact.get("size_bytes"))
        except (TypeError, ValueError):
            return None, f"native_split_quality_{quality_name}_size_invalid"
        if command_size != quality_size:
            return None, f"native_split_quality_{quality_name}_size_mismatch"
    command_boundary = command.get("quality_boundary_contract")
    if not isinstance(command_boundary, Mapping):
        return None, "native_split_quality_command_quality_boundary_contract_missing"
    if canonical_json_sha256(command_boundary) != binding["boundary_contract_sha256"]:
        return None, "native_split_quality_boundary_contract_drift"
    if mode == "portable":
        attestation, attestation_status = _validate_portable_consumer_attestation(
            native_row=native_row, binding=binding, command=command,
            expected_identity=expected_identity,
        )
        if attestation is None:
            return None, attestation_status
        return binding, (
            "portable_binding_command_and_consumer_attestation_exact_match"
        )
    return binding, "exact_quality_native_engine_command_and_boundary_match"


def selection_fingerprint_for_log(selection: Mapping[str, Any]) -> str:
    """Short non-authoritative token for diagnostics only."""

    digest = _sha256(selection.get("selection_sha256")) or canonical_json_sha256(selection)
    return hashlib.sha256(digest.encode("ascii")).hexdigest()[:16]
