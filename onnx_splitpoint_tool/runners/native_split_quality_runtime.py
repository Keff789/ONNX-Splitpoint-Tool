"""Live producer for Quality-FIRST Native accelerator -> TensorRT bindings.

This module is vendored into every generated BenchmarkSet.  It intentionally
does the filesystem and hardware-metadata work that the pure contract module
does not: inspect the concrete HEF/DXNN boundary, copy the exact inputs into a
persistent setup-local cache, build one Part2 engine, and seal its identities.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_CACHE_BINDING_ARTIFACT_ROLES = (
    "part1_runtime", "boundary_metadata", "source_part2_onnx",
    "build_part2_onnx", "engine", "native_trt_meta",
    "engine_build_receipt", "trtexec",
)

try:
    from splitpoint_runners.native_split_quality import (
        canonical_json_sha256,
        known_native_split_policy,
        materialize_native_split_preselection,
        resolve_native_boundary_layout,
        seal_native_split_quality_binding,
        validate_native_split_quality_binding,
    )
except ImportError:  # normal installed-package execution
    from onnx_splitpoint_tool.native_split_quality import (
        canonical_json_sha256,
        known_native_split_policy,
        materialize_native_split_preselection,
        resolve_native_boundary_layout,
        seal_native_split_quality_binding,
        validate_native_split_quality_binding,
    )


def _cache_verify_only() -> bool:
    """Return whether this producer may only replay sealed cache evidence."""

    return str(
        os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY") or ""
    ).strip().lower() == "cache_verify_only"


def trt_build_forbidden(kind: str, *, case_id: str = "", generic_part2: bool = False) -> bool:
    """Enforce the preflight's existing model/setup/role/item expectations.

    This is compiler admission, not a cache key. Explicitly expected cold builds
    stay enabled; ordinary profiles carry no guard at all.
    """
    raw = str(os.environ.get("ONNX_SPLITPOINT_TRT_BUILD_GUARD") or "").strip()
    if not raw:
        return False
    try:
        guard = json.loads(raw)
        policy = guard["policy"]
        model = str(guard["model_id"]).strip()
        setup = str(guard.get("setup_id") or "").strip()
        if not isinstance(policy, dict) or not model:
            raise ValueError("incomplete identity")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("strict_warm_cache_guard_invalid") from exc
    if not policy.get("enabled") or not policy.get("block_on_unexpected_cold_builds"):
        return False
    role = {"full": "trt_full", "part1": "trt_p1", "part2": "trt_p2"}.get(str(kind))
    if role is None:
        raise RuntimeError("strict_warm_cache_guard_unknown_trt_role:" + str(kind))
    case = str(case_id or os.environ.get("ONNX_SPLITPOINT_TRT_BUILD_CASE") or "").strip()
    item = "full" if kind == "full" else case
    if kind == "part2" and generic_part2 and item:
        item += ":generic"
    if not item:
        raise RuntimeError("strict_warm_cache_guard_missing_case")
    scoped_item = f"{setup}/{item}" if setup else item
    expectation = str(policy.get("default_expectation") or "unspecified")
    for row in reversed(list(policy.get("declarations") or [])):
        if not isinstance(row, dict):
            raise RuntimeError("strict_warm_cache_guard_invalid_declaration")
        if row.get("model_id", "*") not in {"*", model}:
            continue
        if row.get("role", "*") not in {"*", role}:
            continue
        declared_item = str(row.get("item_id") or "*")
        if kind == "full" and declared_item.rsplit("/", 1)[-1].startswith("full:"):
            if "/" not in declared_item or declared_item.split("/", 1)[0] == setup:
                # Multi-source Full declarations require a source-specific
                # dispatch context. Never silently turn unresolved identity
                # into permission for a cold build.
                raise RuntimeError("strict_warm_cache_full_source_scope_unresolved")
        if row.get("item_id", "*") not in {"*", item, scoped_item, item.split(":", 1)[0]}:
            continue
        expectation = str(row.get("expectation") or "unspecified")
        break
    return expectation == "warm"


def trt_warm_cache_block(*, compiler: str, artifact: str) -> RuntimeError:
    return RuntimeError(
        "cache_miss_blocked:artifact_policy=strict_warm_cache:"
        f"compiler={compiler}:artifact={artifact}"
    )


def _cache_verify_block(*, artifact: str, reason: str = "missing") -> RuntimeError:
    prefix = (
        "cache_miss_blocked:artifact_policy=cache_verify_only:"
        if _cache_verify_only()
        else "cache_miss_blocked:artifact_policy=strict_warm_cache:"
    )
    return RuntimeError(
        prefix +
        "compiler=trtexec:"
        f"artifact={artifact}:reason={reason}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"native_split_quality_artifact_missing:{resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _regular_file_without_symlink_components(root: Path, path: Path) -> bool:
    """Reject terminal and ancestor symlinks inside a cache namespace."""

    lexical_root = Path(root).expanduser().absolute()
    lexical_path = Path(path).expanduser().absolute()
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        return False
    current = lexical_root
    if current.is_symlink():
        return False
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    try:
        lexical_path.resolve(strict=True).relative_to(
            lexical_root.resolve(strict=True)
        )
    except (FileNotFoundError, ValueError):
        return False
    return lexical_path.is_file()


def prepare_quality_first_boundary_input(
    value: Any,
    *,
    target_shape: Any,
    target_dtype: Any,
    boundary_layout: Mapping[str, Any],
    boundary_transform: str = "",
    dequant_scale: Any = None,
    dequant_zero_point: Any = None,
) -> np.ndarray:
    """Prepare a Generic-runner feed for an internally bridged Native engine.

    Quality-FIRST Hailo engines deliberately bind the accelerator's physical
    memory order to a TensorRT input whose first graph nodes perform the
    declared layout/dequant bridge.  ``HailoSession`` exposes that same buffer
    in canonical ONNX order.  Feeding the canonical array directly makes the
    engine apply NHWC->NCHW a second time.  Reconstruct the physical memory
    view, then reshape it to the TensorRT binding without reordering bytes.

    The helper is intentionally fail-closed: only a sealed permutation and its
    exact memory/target shapes are accepted.
    """

    layout = dict(boundary_layout or {})
    target = tuple(int(dim) for dim in tuple(target_shape or ()))
    if not target or any(dim <= 0 for dim in target):
        raise ValueError("native_split_quality_boundary_target_shape_invalid")
    dtype = np.dtype(target_dtype)
    array = np.asarray(value)

    if layout.get("applied") is not True:
        if tuple(array.shape) != target:
            if int(array.size) != int(np.prod(target, dtype=np.int64)):
                raise ValueError(
                    "native_split_quality_boundary_unbridged_shape_mismatch"
                )
            array = np.reshape(array, target)
        return np.ascontiguousarray(array, dtype=dtype)

    memory_shape = tuple(int(dim) for dim in tuple(layout.get("memory_shape") or ()))
    perm = tuple(int(index) for index in tuple(layout.get("perm") or ()))
    if (
        len(memory_shape) != len(target)
        or len(perm) != len(target)
        or sorted(perm) != list(range(len(target)))
        or any(dim <= 0 for dim in memory_shape)
        or int(np.prod(memory_shape, dtype=np.int64))
        != int(np.prod(target, dtype=np.int64))
        or tuple(memory_shape[index] for index in perm) != target
    ):
        raise ValueError("native_split_quality_boundary_layout_contract_invalid")
    if memory_shape == target and perm != tuple(range(len(target))):
        raise ValueError(
            "native_split_quality_boundary_layout_shape_ambiguous"
        )

    if tuple(array.shape) == memory_shape:
        memory = array
    elif tuple(array.shape) == target:
        inverse_perm = tuple(int(index) for index in np.argsort(perm))
        memory = np.transpose(array, inverse_perm)
    elif (
        len(memory_shape) == array.ndim + 1
        and memory_shape[0] == 1
        and tuple(array.shape) == memory_shape[1:]
    ):
        memory = array[None, ...]
    else:
        raise ValueError(
            "native_split_quality_boundary_source_shape_not_declared"
        )
    if tuple(memory.shape) != memory_shape:
        raise ValueError(
            "native_split_quality_boundary_inverse_layout_shape_mismatch"
        )

    transform = str(boundary_transform or "").strip().lower()
    if (
        np.issubdtype(dtype, np.integer)
        and np.issubdtype(memory.dtype, np.floating)
        and transform == "uint8_dequant_then_layout"
    ):
        try:
            scale = float(dequant_scale)
            zero_point = float(dequant_zero_point)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "native_split_quality_boundary_quantization_parameters_invalid"
            ) from exc
        if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(zero_point):
            raise ValueError(
                "native_split_quality_boundary_quantization_parameters_invalid"
            )
        limits = np.iinfo(dtype)
        memory = np.clip(
            np.rint(np.asarray(memory, dtype=np.float32) / scale + zero_point),
            limits.min,
            limits.max,
        ).astype(dtype)
    elif memory.dtype != dtype:
        memory = memory.astype(dtype, copy=False)

    # TensorRT exposes the canonical binding dimensions even though the bridge
    # consumes the bytes in ``memory_shape`` order.  Reshape is therefore
    # intentional; a transpose here would duplicate the graph bridge.
    return np.ascontiguousarray(memory).reshape(target)


def _case_id(value: Any) -> str:
    token = str(value or "").strip().lower()
    digits = token[1:] if token.startswith("b") else token
    return f"b{int(digits):03d}" if digits.isdigit() else token


def _canonical_backend(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
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
    return token


def _unique_preferred(paths: list[Path], *, role: str) -> Path:
    existing: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved.is_file() and str(resolved) not in seen:
            seen.add(str(resolved))
            existing.append(resolved)
    if not existing:
        raise RuntimeError(f"native_split_quality_{role}_missing")
    # The first entry is the generated contract path.  Additional discovered
    # copies may exist, but a differing byte identity is ambiguous and blocked.
    selected = existing[0]
    selected_sha = _sha256_file(selected)
    conflicting = [p for p in existing[1:] if _sha256_file(p) != selected_sha]
    if conflicting:
        raise RuntimeError(
            f"native_split_quality_{role}_ambiguous:" + ",".join(str(p) for p in conflicting)
        )
    return selected


def _find_part1(case_dir: Path, backend: str) -> Path:
    if backend == "hailo8_to_trt":
        return _unique_preferred([
            case_dir / "hailo/hailo8/part1/compiled.hef",
            case_dir / "hailo/hailo8/part1" / f"{case_dir.name}_part1.hef",
            *sorted(case_dir.glob("hailo/hailo8/part1/**/*.hef")),
        ], role="part1_hef")
    if backend == "hailo10h_to_trt":
        return _unique_preferred([
            case_dir / "hailo/hailo10h/part1/compiled.hef",
            case_dir / "hailo/hailo10/part1/compiled.hef",
            case_dir / "hailo/hailo10n/part1/compiled.hef",
            *sorted(case_dir.glob("hailo/hailo10*/part1/**/*.hef")),
        ], role="part1_hef")
    if backend == "deepx_to_trt":
        return _unique_preferred([
            case_dir / "deepx/deepx_m1/part1/model.dxnn",
            *sorted(case_dir.glob("deepx/deepx_m1/part1/**/*.dxnn")),
        ], role="part1_dxnn")
    raise RuntimeError(f"native_split_quality_backend_unsupported:{backend}")


def _find_part2(case_dir: Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("*_part2_*.onnx", "*part2*.onnx"):
        candidates.extend(sorted(case_dir.glob(pattern)))
    return _unique_preferred(candidates, role="source_part2_onnx")


def _onnx_input(path: Path) -> dict[str, Any]:
    try:
        import onnx  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"native_split_quality_onnx_import_failed:{exc}") from exc
    model = onnx.load(str(path), load_external_data=True)
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1:
        raise RuntimeError(f"native_split_quality_part2_input_count:{len(inputs)}")
    item = inputs[0]
    tensor = item.type.tensor_type
    dims: list[int] = []
    for dim in tensor.shape.dim:
        value = int(getattr(dim, "dim_value", 0) or 0)
        if value <= 0:
            raise RuntimeError("native_split_quality_part2_dynamic_input")
        dims.append(value)
    dtype_map = {
        int(onnx.TensorProto.FLOAT): "float32",
        int(onnx.TensorProto.FLOAT16): "float16",
        int(onnx.TensorProto.UINT8): "uint8",
        int(onnx.TensorProto.INT8): "int8",
    }
    dtype = dtype_map.get(int(tensor.elem_type), "")
    if not dtype:
        raise RuntimeError("native_split_quality_part2_input_dtype_unsupported")
    return {"name": str(item.name), "shape": dims, "dtype": dtype}


def _hailo_info_format_type_name(info: Any) -> str:
    return str(
        getattr(getattr(info, "format", None), "type", "") or ""
    ).split(".")[-1].strip().lower()


def _hailo_native_output_format_type_name(hef: Any, info: Any) -> str:
    """Resolve the unique device-native stream type behind one VStream."""

    required = (
        "get_network_group_names",
        "get_output_stream_infos",
        "get_stream_names_from_vstream_name",
    )
    if any(not callable(getattr(hef, name, None)) for name in required):
        # Older HailoRT releases expose only VStream metadata.  In that API
        # shape an integer VStream type is the strongest available native
        # width; never use a host-transformed FLOAT32 type as such a fallback.
        exposed = _hailo_info_format_type_name(info)
        return exposed if exposed in {"uint8", "uint16"} else ""
    try:
        groups = [str(value) for value in list(hef.get_network_group_names())]
        vstream_name = str(getattr(info, "name", "") or "")
        if len(groups) != 1 or not vstream_name:
            return ""
        group = groups[0]
        low_level_infos = list(hef.get_output_stream_infos(group))
        by_name = {
            str(getattr(row, "name", "") or ""): row
            for row in low_level_infos
        }
        stream_names = [
            str(value) for value in list(
                hef.get_stream_names_from_vstream_name(vstream_name, group)
            )
        ]
        if len(stream_names) != 1 or stream_names[0] not in by_name:
            return ""
        native_type = _hailo_info_format_type_name(by_name[stream_names[0]])
        return native_type if native_type in {"uint8", "uint16"} else ""
    except Exception:
        return ""


def _hailo_metadata(
    *, part1: Path, part2_input: Mapping[str, Any], policy: Mapping[str, Any],
) -> dict[str, Any]:
    hailo_python = str(os.environ.get("HAILO_PY") or "").strip()
    if hailo_python:
        # RUN_PY deliberately remains the TensorRT/CUDA interpreter on mixed
        # Hailo -> TensorRT hosts.  Read only the HEF metadata in the Hailo
        # environment and return plain JSON to the parent process.  Do not
        # resolve or samefile-compare this path: a venv interpreter is often a
        # symlink, while its invocation path is what activates the venv.
        executable = Path(hailo_python).expanduser()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(
                f"native_split_quality_hailo_python_invalid:{hailo_python}"
            )
        probe = r'''
import json
import sys

import hailo_platform as hpf

hef = hpf.HEF(sys.argv[1])
def format_type_name(info):
    return str(
        getattr(getattr(info, "format", None), "type", "") or ""
    ).split(".")[-1].strip().lower()

def native_output_format_type_name(info):
    required = (
        "get_network_group_names",
        "get_output_stream_infos",
        "get_stream_names_from_vstream_name",
    )
    if any(not callable(getattr(hef, name, None)) for name in required):
        exposed = format_type_name(info)
        return exposed if exposed in {"uint8", "uint16"} else ""
    try:
        groups = [str(value) for value in list(hef.get_network_group_names())]
        name = str(getattr(info, "name", "") or "")
        if len(groups) != 1 or not name:
            return ""
        group = groups[0]
        low_level = {
            str(getattr(row, "name", "") or ""): row
            for row in list(hef.get_output_stream_infos(group))
        }
        stream_names = [
            str(value) for value in list(
                hef.get_stream_names_from_vstream_name(name, group)
            )
        ]
        if len(stream_names) != 1 or stream_names[0] not in low_level:
            return ""
        native_type = format_type_name(low_level[stream_names[0]])
        return native_type if native_type in {"uint8", "uint16"} else ""
    except Exception:
        return ""

rows = []
for info in list(hef.get_output_vstream_infos()):
    quant = getattr(info, "quant_info", None)
    scale = getattr(quant, "qp_scale", None)
    zero_point = getattr(quant, "qp_zp", None)
    rows.append({
        "name": str(getattr(info, "name", "") or ""),
        "shape": [int(dim) for dim in list(getattr(info, "shape", ()) or ())],
        "format_type": format_type_name(info),
        "native_format_type": native_output_format_type_name(info),
        "quantization": (
            {"scale": float(scale), "zero_point": float(zero_point)}
            if scale is not None and zero_point is not None else None
        ),
    })
print(json.dumps({"outputs": rows}, sort_keys=True))
'''
        try:
            completed = subprocess.run(
                [str(executable), "-c", probe, str(part1)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "native_split_quality_hailort_probe_timeout"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
            raise RuntimeError(
                "native_split_quality_hailort_probe_failed:"
                f"rc={completed.returncode}:{detail}"
            )
        try:
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            payload = json.loads(lines[-1] if lines else "")
            outputs = list(payload.get("outputs") or [])
        except Exception as exc:
            raise RuntimeError(
                f"native_split_quality_hailort_probe_json_invalid:{exc}"
            ) from exc
        if any(not isinstance(row, Mapping) for row in outputs):
            raise RuntimeError("native_split_quality_hailort_probe_json_invalid")
    else:
        try:
            import hailo_platform as hpf  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"native_split_quality_hailort_import_failed:{exc}"
            ) from exc
        hef = hpf.HEF(str(part1))
        outputs = list(hef.get_output_vstream_infos())
    if len(outputs) != 1:
        raise RuntimeError(f"native_split_quality_hef_output_count:{len(outputs)}")
    info = outputs[0]
    runtime_shape = [
        int(dim) for dim in list(
            (info.get("shape") if isinstance(info, Mapping) else getattr(info, "shape", ()))
            or ()
        )
    ]
    target_shape = [int(dim) for dim in list(part2_input.get("shape") or [])]
    runtime_elements = 1
    for dim in runtime_shape:
        runtime_elements *= dim
    target_elements = 1
    for dim in target_shape:
        target_elements *= dim
    if not runtime_shape or runtime_elements != target_elements:
        raise RuntimeError("native_split_quality_hef_part2_element_count_mismatch")
    tensor: dict[str, Any] = {
        "name": str(part2_input.get("name") or ""),
        "runtime_name": str(
            (info.get("name") if isinstance(info, Mapping) else getattr(info, "name", ""))
            or ""
        ),
        "shape": runtime_shape,
        "canonical_part2_shape": target_shape,
        "dtype": str(policy.get("boundary_dtype") or ""),
    }
    format_type = str(
        (info.get("format_type") if isinstance(info, Mapping) else "")
        or getattr(getattr(info, "format", None), "type", "")
        or ""
    ).split(".")[-1].lower()
    native_format_type = str(
        info.get("native_format_type")
        if isinstance(info, Mapping)
        else _hailo_native_output_format_type_name(hef, info)
    ).strip().lower()
    expected_dtype = str(policy.get("boundary_dtype") or "").strip().lower()
    # InferModel may expose a host-transformed FLOAT32 VStream while the HEF's
    # underlying device stream is UINT8 or UINT16.  A FLOAT32 runtime policy
    # may legitimately request that transform.  A raw quantized policy must
    # bind the exact native integer width because TensorRT consumes those bytes
    # directly; an unresolved native type therefore fails closed.
    if (
        expected_dtype in {"uint8", "uint16"}
        and native_format_type != expected_dtype
    ):
        raise RuntimeError(
            "native_split_quality_hef_boundary_dtype_mismatch:"
            f"expected={expected_dtype}:observed="
            f"{native_format_type or 'unresolved'}"
        )
    tensor["hef_vstream_format_type"] = format_type
    tensor["hef_native_storage_dtype"] = native_format_type
    if str(policy.get("quantization_policy") or "") != "none":
        if isinstance(info, Mapping):
            quant = info.get("quantization")
            quant = quant if isinstance(quant, Mapping) else {}
            scale = quant.get("scale")
            zero_point = quant.get("zero_point")
        else:
            quant = getattr(info, "quant_info", None)
            scale = getattr(quant, "qp_scale", None)
            zero_point = getattr(quant, "qp_zp", None)
        if scale is None or zero_point is None:
            raise RuntimeError("native_split_quality_hef_quant_info_missing")
        tensor["quantization"] = {
            "source": "hailort_hef_output_vstream_info",
            "scale": float(scale),
            "zero_point": float(zero_point),
        }
    return tensor


def _deepx_metadata(
    *, case_dir: Path, part2_input: Mapping[str, Any], policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = case_dir / "deepx/deepx_m1/part1/output_contract.json"
    if not contract_path.is_file():
        raise RuntimeError("native_split_quality_deepx_output_contract_missing")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"native_split_quality_deepx_output_contract_invalid:{exc}") from exc
    names = [str(value).strip() for value in list(contract.get("output_names") or []) if str(value).strip()]
    if not names and isinstance(contract.get("outputs"), list):
        names = [str(row.get("name") or "").strip() for row in contract["outputs"] if isinstance(row, Mapping) and str(row.get("name") or "").strip()]
    if len(names) != 1 or names[0] != str(part2_input.get("name") or ""):
        raise RuntimeError("native_split_quality_deepx_output_part2_name_mismatch")
    input_contract = contract.get("input")
    if not isinstance(input_contract, Mapping):
        raise RuntimeError("native_split_quality_deepx_input_contract_missing")
    input_shape = [int(dim) for dim in list(input_contract.get("shape") or [])]
    input_dtype = str(input_contract.get("dtype") or "").strip().lower()
    input_layout = str(input_contract.get("layout") or "").strip().upper()
    if (
        not input_shape or any(dim <= 0 for dim in input_shape)
        or input_dtype not in {"uint8", "float32"}
        or input_layout not in {"HWC", "NHWC", "CHW", "NCHW"}
    ):
        raise RuntimeError("native_split_quality_deepx_input_contract_invalid")
    tensor = {
        "name": names[0],
        "runtime_name": names[0],
        "output_names": names,
        "shape": [int(dim) for dim in list(part2_input.get("shape") or [])],
        "canonical_part2_shape": [int(dim) for dim in list(part2_input.get("shape") or [])],
        "dtype": str(policy.get("boundary_dtype") or ""),
        "contract_source": "deepx_part1_output_contract_plus_source_part2_input",
        "deepx_output_contract_sha256": _sha256_file(contract_path),
        "deepx_input_contract": dict(input_contract),
    }
    return tensor, _artifact(contract_path)


def _persistent_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256_file(source)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_sha:
            raise RuntimeError(f"native_split_quality_persistent_cache_conflict:{destination}")
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    if _sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("native_split_quality_persistent_copy_sha256_mismatch")
    os.replace(temporary, destination)
    return destination


def _builder_script() -> Path:
    local = Path(__file__).resolve().parent / "native_trt_from_benchmarkset.py"
    if local.is_file():
        return local
    package = (
        Path(__file__).resolve().parents[1]
        / "resources/remote_scripts/native_trt_from_benchmarkset.py"
    )
    if package.is_file():
        return package
    raise RuntimeError("native_split_quality_trt_builder_missing")


def _cache_binding_identity_mismatch_axes(
    binding: Mapping[str, Any],
    *,
    part1_sha256: str,
    source_part2_sha256: str,
    policy_sha256: str,
) -> set[str]:
    """Compare only the three immutable cache-identity axes."""

    artifacts = dict(binding.get("artifacts") or {})
    part1 = dict(artifacts.get("part1_runtime") or {})
    part2 = dict(artifacts.get("source_part2_onnx") or {})
    selection = dict(binding.get("preselection") or {})
    return {
        axis
        for axis, observed, expected in (
            (
                "part1_runtime_sha256",
                part1.get("sha256"),
                part1_sha256,
            ),
            (
                "source_part2_onnx_sha256",
                part2.get("sha256"),
                source_part2_sha256,
            ),
            (
                "policy_sha256",
                selection.get("policy_sha256"),
                policy_sha256,
            ),
        )
        if str(observed or "").strip().lower()
        != str(expected or "").strip().lower()
    }


def _cache_binding_artifact_set_sha256(binding: Mapping[str, Any]) -> str:
    """Hash the complete fixed-role content identity without filesystem paths."""

    artifacts = binding.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(_CACHE_BINDING_ARTIFACT_ROLES)
    ):
        return ""
    artifact_identities: dict[str, dict[str, Any]] = {}
    for role in _CACHE_BINDING_ARTIFACT_ROLES:
        raw = artifacts.get(role)
        if not isinstance(raw, Mapping):
            return ""
        sha256 = str(raw.get("sha256") or "").strip().lower()
        try:
            size_bytes = int(raw.get("size_bytes"))
        except (TypeError, ValueError):
            return ""
        if not re.fullmatch(r"[0-9a-f]{64}", sha256) or size_bytes <= 0:
            return ""
        artifact_identities[str(role)] = {
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return canonical_json_sha256(artifact_identities)


def _cache_binding_equivalence_key(binding: Mapping[str, Any]) -> str:
    """Identify byte-identical sealed bindings independent of discovery path."""

    binding_sha256 = str(binding.get("binding_sha256") or "").strip().lower()
    artifact_set_sha256 = _cache_binding_artifact_set_sha256(binding)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", binding_sha256)
        or not artifact_set_sha256
    ):
        return ""
    return canonical_json_sha256({
        "binding_sha256": binding_sha256,
        "artifact_set_sha256": artifact_set_sha256,
    })


def _replay_cache_verified_native_split_binding(
    *,
    cache_root: str | Path,
    part1_source: Path,
    part2_source: Path,
    policy: Mapping[str, Any],
    setup: str,
    model: str,
    case: str,
    backend: str,
    eval_run_id: str,
    source_run_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select one exact sealed cache binding without parsing model formats.

    Cache verification is intentionally dependency-minimal: current Part1 and
    Part2 bytes are hashed, then already sealed bindings are revalidated with
    their complete local artifact/receipt chain.  No ONNX, HailoRT, ORT,
    compiler helper, persistent copy or subprocess is reachable from here.
    """

    part1_sha = _sha256_file(part1_source)
    part2_sha = _sha256_file(part2_source)
    resolved_cache_root = Path(cache_root).expanduser().resolve()
    binding_root = (
        resolved_cache_root
        / "native_split_quality" / setup / model / case / backend
    )
    binding_paths = (
        sorted(binding_root.glob("*/native_split_quality_binding.json"))
        if binding_root.is_dir() else []
    )
    exact: list[tuple[Path, dict[str, Any], str, str]] = []
    invalid: list[str] = []
    mismatch_axes: set[str] = set()
    for binding_path in binding_paths:
        if not _regular_file_without_symlink_components(
            resolved_cache_root, binding_path,
        ):
            invalid.append("binding_path_not_regular_file")
            continue
        try:
            cached_raw = json.loads(
                binding_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except Exception as exc:
            invalid.append(f"binding_json_invalid_{type(exc).__name__}")
            continue
        cached, cached_status = validate_native_split_quality_binding(
            cached_raw,
            expected_identity={
                "model": model,
                "case": case,
                "setup_id": setup,
                "backend": backend,
                "task": policy["task"],
                "precision": policy["precision"],
            },
            verification_mode="local",
        )
        if cached is None:
            invalid.append(
                "binding_validation_failed_"
                + str(cached_status or "unknown").strip().replace(":", "_")
            )
            continue
        candidate_mismatches = _cache_binding_identity_mismatch_axes(
            cached,
            part1_sha256=part1_sha,
            source_part2_sha256=part2_sha,
            policy_sha256=str(policy.get("policy_sha256") or ""),
        )
        if candidate_mismatches:
            mismatch_axes.update(candidate_mismatches)
            continue
        equivalence_key = _cache_binding_equivalence_key(cached)
        if not equivalence_key:
            invalid.append("binding_equivalence_identity_invalid")
            continue
        exact.append((
            binding_path, dict(cached), str(cached_status), equivalence_key,
        ))

    if not binding_paths:
        raise _cache_verify_block(
            artifact="native_split_quality_binding", reason="missing",
        )
    if invalid:
        raise _cache_verify_block(
            artifact="native_split_quality_binding",
            reason=(
                f"invalid_binding_count_{len(invalid)}_"
                f"first_{invalid[0]}"
            ),
        )
    if not exact:
        if mismatch_axes:
            raise RuntimeError(
                "cache_miss_blocked:artifact_policy="
                + ("cache_verify_only" if _cache_verify_only() else "strict_warm_cache")
                + ":compiler=trtexec:mismatch_axes="
                + ",".join(sorted(mismatch_axes))
                + ":artifact=native_split_quality_binding:"
                "reason=exact_hit_count_0"
            )
        raise _cache_verify_block(
            artifact="native_split_quality_binding", reason="exact_hit_count_0",
        )
    # Every row above is already a complete, locally re-hashed artifact set
    # with the requested semantic identity.  For this diagnostic-only mode it
    # is therefore safe to select one whole set deterministically.  Hashes
    # distinguish the candidates for diagnostics, but do not turn two valid
    # cache generations into an artificial ambiguity error.  No artifacts are
    # ever mixed between rows.
    distinct_equivalence_keys = {row[3] for row in exact}
    persistent_binding_path, cached, cached_status, equivalence_key = min(
        exact, key=lambda row: str(row[0])
    )
    compatible_binding_paths = sorted(str(row[0]) for row in exact)
    equivalent_binding_paths = sorted(
        str(row[0]) for row in exact if row[3] == equivalence_key
    )
    source_binding_sha256 = str(cached.get("binding_sha256") or "").strip().lower()
    source_artifact_set_sha256 = _cache_binding_artifact_set_sha256(cached)

    replay_payload = dict(cached)
    previous_binding_sha256 = str(replay_payload.get("binding_sha256") or "")
    # This diagnostic creates no new Central Quality selection.  Remove only
    # old run-bound selection fields while retaining all compiler artifacts.
    for field in (
        "producer_binding_sha256",
        "source_request_sha256",
        "central_result_sha256",
        "central_quality_selection",
        "central_quality_selection_sha256",
    ):
        replay_payload.pop(field, None)
    replay_payload["eval_run_id"] = eval_run_id
    replay_payload["source_run_id"] = source_run_id
    replay_payload["cache_verify_replay"] = {
        "artifact_policy": "cache_verify_only" if _cache_verify_only() else "strict_warm_cache",
        "source_binding_sha256": previous_binding_sha256,
        "local_validation_status": cached_status,
        "compiler_dispatched": False,
    }
    replay = seal_native_split_quality_binding(replay_payload)
    requested_output = Path(output_path).expanduser().resolve()
    _write_json(requested_output, replay)
    replay_artifacts = dict(replay.get("artifacts") or {})
    replay_selection = dict(replay.get("preselection") or {})
    return {
        "binding": replay,
        "binding_path": str(requested_output),
        "persistent_binding_path": str(persistent_binding_path),
        "precision": str(replay_selection.get("precision") or ""),
        "boundary_mode": (
            "raw_uint8_hailo"
            if str(replay_selection.get("boundary_dtype") or "") == "uint8"
            else "canonical_float32"
        ),
        "engine_path": str(
            dict(replay_artifacts.get("engine") or {}).get("path") or ""
        ),
        "build_onnx_path": str(
            dict(replay_artifacts.get("build_part2_onnx") or {}).get("path")
            or ""
        ),
        "receipt_path": str(
            dict(replay_artifacts.get("engine_build_receipt") or {}).get("path")
            or ""
        ),
        "cache_verify_reused": True,
        "cache_verify_source_binding_sha256": source_binding_sha256,
        "cache_verify_source_artifact_set_sha256": (
            source_artifact_set_sha256
        ),
        "cache_verify_equivalence_key_sha256": equivalence_key,
        "cache_verify_exact_binding_count": len(equivalent_binding_paths),
        "cache_verify_equivalent_binding_paths": equivalent_binding_paths,
        "cache_verify_compatible_binding_count": len(exact),
        "cache_verify_compatible_binding_paths": compatible_binding_paths,
        "cache_verify_distinct_artifact_set_count": len(
            distinct_equivalence_keys
        ),
        "cache_verify_selection_rule": (
            "lexicographic_semantically_compatible_complete_artifact_set"
        ),
    }


def prepare_native_split_quality_binding(
    *, benchmark_set: str | Path, case_id: Any, model_id: Any,
    setup_id: Any, backend: Any, eval_run_id: Any, source_run_id: Any,
    cache_root: str | Path, output_path: str | Path,
    workspace_mb: int = 4096, timeout_s: int = 7200,
) -> dict[str, Any]:
    """Materialize and seal the exact split engine before semantic Quality.

    The returned binding is self-contained metadata, while all large artifacts
    live in ``cache_root`` and are addressed by content hash.  Any unknown case,
    missing runtime metadata, or ambiguous artifact fails before inference.
    """

    root = Path(benchmark_set).expanduser().resolve()
    case = _case_id(case_id)
    case_dir = root / case
    setup = str(setup_id or "").strip()
    model = str(model_id or "").strip().lower()
    eval_id = str(eval_run_id or "").strip()
    run_id = str(source_run_id or "").strip()
    canonical_backend = _canonical_backend(backend)
    if not root.is_dir() or not case_dir.is_dir() or not all((setup, model, eval_id, run_id)):
        raise RuntimeError("native_split_quality_producer_identity_incomplete")
    policy = known_native_split_policy(
        model_id=model, case_id=case, setup_id=setup, backend=canonical_backend,
    )
    if policy is None:
        raise RuntimeError("native_split_quality_policy_unavailable")
    part1_source = _find_part1(case_dir, canonical_backend)
    part2_source = _find_part2(case_dir)
    if _cache_verify_only():
        return _replay_cache_verified_native_split_binding(
            cache_root=cache_root,
            part1_source=part1_source,
            part2_source=part2_source,
            policy=policy,
            setup=setup,
            model=model,
            case=case,
            backend=canonical_backend,
            eval_run_id=eval_id,
            source_run_id=run_id,
            output_path=output_path,
        )
    part2_input = _onnx_input(part2_source)
    if canonical_backend.startswith("hailo"):
        boundary_tensor = _hailo_metadata(
            part1=part1_source, part2_input=part2_input, policy=policy,
        )
        extra_boundary: dict[str, Any] = {}
    else:
        boundary_tensor, deepx_contract_artifact = _deepx_metadata(
            case_dir=case_dir, part2_input=part2_input, policy=policy,
        )
        extra_boundary = {"deepx_output_contract": deepx_contract_artifact}

    policy_layout = str(policy.get("boundary_layout") or "")
    if canonical_backend == "hailo10h_to_trt":
        boundary_layout = resolve_native_boundary_layout(
            boundary_tensor.get("shape"),
            boundary_tensor.get("canonical_part2_shape"),
        )
        quantized_boundary = (
            str(policy.get("quantization_policy") or "").strip().lower()
            != "none"
        )
        boundary_transform = (
            "uint8_dequant" if quantized_boundary else "identity"
        ) if boundary_layout == "as_input" else (
            "uint8_dequant_then_layout"
            if quantized_boundary else "layout_only"
        )
    else:
        boundary_layout = policy_layout
        boundary_transform = str(policy.get("boundary_transform") or "")

    part1_sha = _sha256_file(part1_source)
    part2_sha = _sha256_file(part2_source)
    key = canonical_json_sha256({
        "policy_sha256": policy["policy_sha256"],
        "part1_sha256": part1_sha,
        "source_part2_sha256": part2_sha,
        "boundary_tensor": boundary_tensor,
        "resolved_boundary_layout": boundary_layout,
        "resolved_boundary_transform": boundary_transform,
    })
    persistent = (
        Path(cache_root).expanduser().resolve() / "native_split_quality" / setup
        / model / case / canonical_backend / key
    )
    part1_name = "part1.dxnn" if canonical_backend == "deepx_to_trt" else "part1.hef"
    part1 = _persistent_copy(part1_source, persistent / part1_name)
    persistent_suite = persistent / "benchmark_set"
    persistent_case = persistent_suite / case
    source_part2 = _persistent_copy(part2_source, persistent_case / "source_part2.onnx")

    boundary_metadata: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-part1-boundary-metadata",
        "schema_version": 1,
        "model_id": model,
        "case_id": case,
        "setup_id": setup,
        "backend": canonical_backend,
        "source_run_id": run_id,
        "part1_artifact_sha256": _sha256_file(part1),
        "part1_artifact_size_bytes": int(part1.stat().st_size),
        "boundary_tensor_count": 1,
        "boundary_tensor": boundary_tensor,
        "boundary_layout": boundary_layout,
        "boundary_transform": boundary_transform,
        **extra_boundary,
    }
    boundary_metadata["metadata_sha256"] = canonical_json_sha256(boundary_metadata)
    boundary_metadata_path = _write_json(
        persistent / "part1_boundary_metadata.json", boundary_metadata,
    )
    preselection = materialize_native_split_preselection(
        policy=policy,
        part1_artifact=_artifact(part1),
        boundary_metadata=boundary_metadata,
        boundary_metadata_artifact=_artifact(boundary_metadata_path),
    )

    engine_root = persistent / "engine_cache"
    summary_path = persistent / "native_trt_summary.json"
    command = [
        sys.executable, str(_builder_script()),
        "--benchmark-set", str(persistent_suite),
        "--case", case, "--variants", "part2",
        "--precision", str(preselection["precision"]),
        "--out-dir", str(engine_root),
        "--boundary-layout", str(preselection["boundary_layout"]),
        "--workspace-mb", str(int(workspace_mb)),
        "--workspace-mode", "auto", "--no-run-smoke",
        "--json-out", str(summary_path),
    ]
    if preselection.get("dequant_scale") is not None:
        command += ["--dequant-scale", repr(float(preselection["dequant_scale"]))]
        command += ["--dequant-zero-point", repr(float(preselection["dequant_zero_point"]))]
    strict_warm_cache = trt_build_forbidden("part2", case_id=case)
    if strict_warm_cache:
        command.append("--no-build")
    if _cache_verify_only():
        # Last-process fence: even a future refactor that reaches this point
        # must not start native_trt_from_benchmarkset.py.
        raise _cache_verify_block(
            artifact="native_split_quality_engine",
            reason="builder_dispatch_forbidden",
        )
    def run_builder(builder_command: list[str]) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
        if strict_warm_cache and "--no-build" not in builder_command:
            raise _cache_verify_block(artifact="native_split_quality_engine", reason="builder_dispatch_forbidden")
        try:
            summary_path.unlink(missing_ok=True)
        except OSError:
            pass
        completed = subprocess.run(
            builder_command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=max(60, int(timeout_s)),
        )
        summary_payload: dict[str, Any] = {}
        if completed.returncode == 0:
            try:
                parsed = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    summary_payload = parsed
            except Exception:
                summary_payload = {}
        summary_rows = [
            dict(row) for row in list(summary_payload.get("artifacts") or [])
            if isinstance(row, Mapping)
        ]
        return completed, summary_payload, summary_rows

    # A migrated stable namespace already contains a self-hashed build receipt
    # and engine.  Ask the existing builder to verify that exact ONNX/engine/
    # trtexec binding before it gets any opportunity to delete or rebuild it.
    # An absent, ambiguous or invalid candidate falls through to the original
    # build path; no weak engine-exists shortcut is accepted.
    receipt_candidates = [
        path for path in sorted(engine_root.rglob("engine_build_receipt.json"))
        if _regular_file_without_symlink_components(
            Path(cache_root).expanduser().resolve(), path,
        )
    ]
    reuse_candidate = False
    if len(receipt_candidates) == 1:
        try:
            candidate_receipt = json.loads(
                receipt_candidates[0].read_text(encoding="utf-8")
            )
            candidate_engine = Path(
                str(dict(candidate_receipt).get("engine") or "")
            ).expanduser().resolve()
            reuse_candidate = _regular_file_without_symlink_components(
                Path(cache_root).expanduser().resolve(), candidate_engine,
            )
        except Exception:
            reuse_candidate = False

    executed_command = list(command)
    proc: Any
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    verified_reuse = False
    if reuse_candidate:
        verify_command = list(command) if "--no-build" in command else [*command, "--no-build"]
        proc, summary, rows = run_builder(verify_command)
        verified_reuse = bool(
            proc.returncode == 0
            and len(rows) == 1
            and rows[0].get("build_ok") is True
            and rows[0].get("engine_build_receipt_status")
            == "engine_build_receipt_verified"
            and isinstance(rows[0].get("engine_build_receipt"), Mapping)
            and str(rows[0].get("engine_build_receipt_path") or "").strip()
            == str(receipt_candidates[0].resolve())
        )
        if verified_reuse:
            executed_command = verify_command
    if not verified_reuse and not (strict_warm_cache and reuse_candidate):
        proc, summary, rows = run_builder(command)
        executed_command = list(command)
    if proc.returncode != 0:
        if strict_warm_cache:
            raise _cache_verify_block(artifact="native_split_quality_engine", reason="receipt_verified_reuse_failed")
        raise RuntimeError(
            "native_split_quality_part2_build_failed:" + (proc.stdout or "")[-4000:]
        )
    if len(rows) != 1 or rows[0].get("build_ok") is not True:
        raise RuntimeError("native_split_quality_build_not_uniquely_successful")
    if verified_reuse and rows[0].get("engine_build_receipt_status") != "engine_build_receipt_verified":
        raise RuntimeError("native_split_quality_cached_receipt_not_verified")
    build_row = dict(rows[0])
    engine = Path(str(build_row.get("engine") or "")).resolve()
    build_part2 = Path(str(build_row.get("onnx") or "")).resolve()
    native_trt_meta = engine.parent / "native_trt_meta.json"
    receipt_path = Path(str(build_row.get("engine_build_receipt_path") or "")).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    trtexec = Path(str(receipt.get("trtexec") or "")).resolve()
    if (
        str(receipt.get("source_onnx") or "") != str(build_part2)
        or str(receipt.get("source_onnx_sha256") or "") != _sha256_file(build_part2)
        or str(receipt.get("engine") or "") != str(engine)
        or str(receipt.get("engine_sha256") or "") != _sha256_file(engine)
        or str(receipt.get("trtexec_sha256") or "") != _sha256_file(trtexec)
    ):
        raise RuntimeError("native_split_quality_build_receipt_cross_binding_invalid")
    bridge = build_row.get("uint8_cast_bridge")
    bridge = dict(bridge) if isinstance(bridge, Mapping) else {}
    layout = bridge.get("boundary_layout")
    layout = dict(layout) if isinstance(layout, Mapping) else {}
    if str(layout.get("effective") or "as_input") != str(preselection["boundary_layout"]):
        raise RuntimeError("native_split_quality_bridge_layout_mismatch")
    if preselection["precision"] == "uint8_dequant_fp16" and (
        float(bridge.get("scale")) != float(preselection["dequant_scale"])
        or float(bridge.get("zero_point")) != float(preselection["dequant_zero_point"])
    ):
        raise RuntimeError("native_split_quality_bridge_dequant_mismatch")

    artifacts = {
        "part1_runtime": _artifact(part1),
        "boundary_metadata": _artifact(boundary_metadata_path),
        "source_part2_onnx": _artifact(source_part2),
        "build_part2_onnx": _artifact(build_part2),
        "engine": _artifact(engine),
        "native_trt_meta": _artifact(native_trt_meta),
        "engine_build_receipt": _artifact(receipt_path),
        "trtexec": _artifact(trtexec),
    }
    boundary_contract = {
        "precision": preselection["precision"],
        "boundary_layout": preselection["boundary_layout"],
        "boundary_transform": preselection["boundary_transform"],
        "boundary_tensor_name": preselection["boundary_tensor_name"],
        "boundary_tensor_shape": list(preselection["boundary_tensor_shape"]),
        "boundary_tensor_dtype": preselection["boundary_tensor_dtype"],
        "boundary_metadata_sha256": preselection["boundary_metadata_sha256"],
        "boundary_metadata_file_sha256": preselection["boundary_metadata_file_sha256"],
        "dequant_scale": preselection.get("dequant_scale"),
        "dequant_zero_point": preselection.get("dequant_zero_point"),
    }
    binding = seal_native_split_quality_binding({
        "eval_run_id": eval_id,
        "source_run_id": run_id,
        "quality_completed": True,
        "performance_claims_emitted": False,
        "preselection": preselection,
        "preselection_sha256": preselection["selection_sha256"],
        "artifacts": artifacts,
        "boundary_contract": boundary_contract,
        "boundary_contract_sha256": canonical_json_sha256(boundary_contract),
        "engine_build_receipt": receipt,
        "engine_build_receipt_sha256": str(receipt.get("receipt_sha256") or ""),
        "native_trt_meta": build_row,
        "native_trt_meta_sha256": canonical_json_sha256(build_row),
        "producer_command": executed_command,
    })
    verified, status = validate_native_split_quality_binding(
        binding,
        expected_identity={
            "model": model, "case": case, "setup_id": setup,
            "backend": canonical_backend, "task": policy["task"],
            "precision": policy["precision"],
        },
    )
    if verified is None:
        raise RuntimeError(f"native_split_quality_binding_self_validation_failed:{status}")
    persistent_binding_path = _write_json(persistent / "native_split_quality_binding.json", binding)
    requested_output = Path(output_path).expanduser().resolve()
    _write_json(requested_output, binding)
    return {
        "binding": binding,
        "binding_path": str(requested_output),
        "persistent_binding_path": str(persistent_binding_path),
        "precision": str(preselection["precision"]),
        "boundary_mode": (
            "raw_uint8_hailo"
            if str(preselection["boundary_dtype"]) == "uint8"
            else "canonical_float32"
        ),
        "engine_path": str(engine),
        "build_onnx_path": str(build_part2),
        "receipt_path": str(receipt_path),
    }
