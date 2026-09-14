from __future__ import annotations

"""Remote-safe Hailo Full artifact-contract promotion.

This module is intentionally independent of workflow planning and management
reference code.  Native accelerator nodes receive it as part of the exact
runtime closure, so importing it must never require central-management modules.
"""

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .preprocessing_contract import (
    preprocessing_contract_sha256,
    resolve_image_preprocessing_contract,
)


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def sha256_payload(value: Any) -> str:
    """Preserve the workflow artifact identity serialization exactly."""
    return hashlib.sha256(_stable_json_dumps(value).encode("utf-8")).hexdigest()


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read tolerant runtime metadata without importing workflow.artifacts."""
    candidate = Path(path)
    if not candidate.is_file():
        return default
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return default


_HAILO_HEF_RECEIPT_NAME = "hailo_hef_build_receipt.json"
_HAILO_HEF_RECEIPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-receipt/v2"
_HAILO_HEF_CACHE_SCHEMAS = {
    "onnx-splitpoint/hailo-hef-cache-key-v2",
    "onnx-splitpoint/hailo-hef-cache-key-v3",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _strict_positive_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value > 0 else None


def _case_id_for(candidate: Mapping[str, Any], idx: int) -> str:
    raw = str(candidate.get("case_id") or candidate.get("id") or "").strip()
    if raw:
        return raw
    split = _safe_int(candidate.get("split_index") or candidate.get("boundary") or candidate.get("boundary_index"))
    if split is not None:
        return f"b{split:03d}"
    return f"case_{idx:03d}"


def _target_backend_label(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return "unknown"
    if s in {"cpu", "cpu_ort", "ort_cpu"}:
        return "cpu_ort"
    if s in {"cuda", "cuda_ort", "ort_cuda", "gpu"}:
        return "cuda_ort"
    if "deepx" in s or "dx_m1" in s or "dxm1" in s:
        if "trt" in s or "tensorrt" in s:
            return "deepx_m1_to_tensorrt"
        return "deepx_m1"
    if "trt" in s or "tensorrt" in s:
        return "tensorrt"
    if "hailo10h" in s:
        return "hailo10h"
    if "hailo10" in s:
        return "hailo10"
    if "hailo8r" in s:
        return "hailo8r"
    if "hailo8l" in s:
        return "hailo8l"
    if "hailo8" in s or "hailo" in s:
        return "hailo8"
    return s


def _hailo_contract_backend(value: Any) -> str:
    token = _target_backend_label(value)
    if token in {"hailo10", "hailo10h"}:
        return "hailo10"
    if token in {"hailo8", "hailo8l", "hailo8r"}:
        return "hailo8"
    return token


def _hailo_hw_matches(expected: Any, actual: Any) -> bool:
    expected_hw = _target_backend_label(expected)
    actual_hw = _target_backend_label(actual)
    if expected_hw in {"hailo10", "hailo10h"}:
        return actual_hw in {"hailo10", "hailo10h"}
    return bool(expected_hw) and actual_hw == expected_hw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        return ""
    return token


def _existing_path_candidates(
    values: Sequence[Any], *, roots: Sequence[Path]
) -> List[Path]:
    found: List[Path] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        attempts = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
        for attempt in attempts:
            try:
                resolved = attempt.resolve()
            except OSError:
                continue
            key = str(resolved)
            if key in seen or not resolved.is_file():
                continue
            seen.add(key)
            found.append(resolved)
    return found


def _receipt_path_hints(
    *,
    artifact: Path,
    baseline: Mapping[str, Any] | None = None,
    full_baseline_plan: Mapping[str, Any] | None = None,
    suite_bench: Mapping[str, Any] | None = None,
    backend_meta: Mapping[str, Any] | None = None,
) -> Tuple[List[Path], List[Path]]:
    """Collect explicit ONNX paths plus the receipt-signed compiler sibling.

    ``compiler_onnx_filename`` is part of the canonical build receipt and is
    later required to match both basename and SHA-256.  Treating that signed
    sibling as a candidate is therefore not filename guessing; it closes the
    cache-hit path where no separate build-result JSON was emitted.
    """

    baseline = baseline or {}
    full_baseline_plan = full_baseline_plan or {}
    suite_bench = suite_bench or {}
    backend_meta = backend_meta or {}
    prepared = _as_dict(full_baseline_plan.get("prepared_full_hailo_baseline"))
    prepared_raw = _as_dict(prepared.get("raw"))
    prepared_raw_raw = _as_dict(prepared_raw.get("raw"))
    # A pinned atomic generation contains HEF/Receipt/Meta; the compiler
    # ONNX remains beside the public aliases. Keep both existing locations
    # as hints, still requiring the receipt's exact filename and byte hashes.
    container = artifact.parent
    if artifact.parent.parent.name == ".hailo-generations":
        container = artifact.parent.parent.parent
    logical_artifact = container / artifact.name
    artifact_roots: List[Path] = [artifact.parent, container, container.parent]
    if len(logical_artifact.parents) > 3:
        artifact_roots.append(logical_artifact.parents[3])
    artifact_roots.append(Path.cwd())
    result_payloads = [
        _as_dict(prepared.get("result_json")),
        _as_dict(prepared_raw.get("result_json")),
        _as_dict(prepared_raw_raw.get("result_json")),
        _as_dict(backend_meta.get("full_build")),
    ]
    result_paths: List[Any] = [
        baseline.get("result_json_path"),
        prepared.get("result_json_path"),
        prepared_raw.get("result_json_path"),
        backend_meta.get("full_build_result_path"),
        artifact.parent / "hailo_hef_build_result.json",
    ]
    for result_path in _existing_path_candidates(
        result_paths,
        roots=artifact_roots,
    ):
        payload = read_json(result_path, default={}) or {}
        if isinstance(payload, Mapping):
            result_payloads.append(dict(payload))

    source_values: List[Any] = []
    compiler_values: List[Any] = []
    mappings = [baseline, prepared, prepared_raw, prepared_raw_raw, backend_meta, suite_bench]
    for payload in mappings:
        source_values.extend(
            payload.get(key)
            for key in (
                "model_source",
                "model",
                "source_onnx_path",
                "selected_model_path",
                "source_model_path",
                "model_path",
                "onnx_path",
                "full_model_source",
                "full_model",
            )
        )
        compiler_values.extend(
            payload.get(key)
            for key in (
                "compiler_onnx_path",
                "fixed_onnx_path",
                "compiler_model_path",
            )
        )
    for payload in result_payloads:
        source_values.extend(
            payload.get(key)
            for key in ("source_onnx_path", "onnx_path", "model_path")
        )
        compiler_values.extend(
            payload.get(key)
            for key in ("compiler_onnx_path", "fixed_onnx_path")
        )
    receipt_path = artifact.parent / _HAILO_HEF_RECEIPT_NAME
    receipt = read_json(receipt_path, default={}) or {}
    if isinstance(receipt, Mapping):
        compiler_filename = str(
            receipt.get("compiler_onnx_filename") or ""
        ).strip()
        if (
            compiler_filename
            and Path(compiler_filename).name == compiler_filename
            and compiler_filename.lower().endswith(".onnx")
        ):
            compiler_values.extend([
                artifact.parent / compiler_filename,
                container / compiler_filename,
            ])
    roots = tuple(artifact_roots)
    return (
        _existing_path_candidates(source_values, roots=roots),
        _existing_path_candidates(compiler_values, roots=roots),
    )


def _validate_hailo_build_receipt(
    artifact: Path,
    *,
    receipt_path: Path | None = None,
    source_onnx_candidates: Sequence[Path] = (),
    compiler_onnx_candidates: Sequence[Path] = (),
    expected_backend: str = "",
    expected_task: str = "",
) -> Dict[str, Any]:
    """Revalidate the complete canonical Hailo build receipt chain."""

    artifact = artifact.expanduser().resolve()
    receipt_path = (
        receipt_path.expanduser().resolve()
        if receipt_path is not None
        else artifact.parent / _HAILO_HEF_RECEIPT_NAME
    )
    evidence: Dict[str, Any] = {
        "valid": False,
        "path": str(artifact),
        "receipt_path": str(receipt_path),
        "errors": [],
    }
    errors: List[str] = evidence["errors"]
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        errors.append("hailo_hef_missing_or_empty")
        return evidence
    if not receipt_path.is_file():
        errors.append("hailo_build_receipt_missing")
        return evidence
    evidence["sha256"] = _file_sha256(artifact)
    evidence["size_bytes"] = int(artifact.stat().st_size)
    evidence["receipt_file_sha256"] = _file_sha256(receipt_path)
    evidence["receipt_size_bytes"] = int(receipt_path.stat().st_size)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"hailo_build_receipt_json_invalid:{type(exc).__name__}")
        return evidence
    if not isinstance(receipt, Mapping):
        errors.append("hailo_build_receipt_not_object")
        return evidence
    receipt = dict(receipt)
    evidence["receipt"] = receipt
    evidence["receipt_identity_sha256"] = sha256_payload(receipt)
    if receipt.get("schema") != _HAILO_HEF_RECEIPT_SCHEMA:
        errors.append("hailo_build_receipt_schema_invalid")
    if expected_backend:
        # ``hailo10`` is the historical BenchmarkSet family key for a 10H
        # artifact. Hailo-8, 8L and 8R are distinct compiler targets and must
        # never be accepted merely because they share a family prefix.
        if not _hailo_hw_matches(expected_backend, receipt.get("hw_arch")):
            errors.append("hailo_build_receipt_hw_arch_mismatch")
    if _sha256_token(receipt.get("hef_sha256")) != evidence["sha256"]:
        errors.append("hailo_build_receipt_hef_sha256_mismatch")
    if _strict_positive_int(receipt.get("hef_size_bytes")) != evidence["size_bytes"]:
        errors.append("hailo_build_receipt_hef_size_mismatch")

    source_sha = _sha256_token(receipt.get("source_onnx_sha256"))
    compiler_sha = _sha256_token(receipt.get("compiler_onnx_sha256"))
    compiler_filename = str(
        receipt.get("compiler_onnx_filename") or ""
    ).strip()
    if not source_sha:
        errors.append("hailo_build_receipt_source_onnx_sha256_invalid")
    if not compiler_sha:
        errors.append("hailo_build_receipt_compiler_onnx_sha256_invalid")
    if (
        not compiler_filename
        or Path(compiler_filename).name != compiler_filename
        or not compiler_filename.lower().endswith(".onnx")
    ):
        errors.append("hailo_build_receipt_compiler_onnx_filename_invalid")

    contract = receipt.get("preprocessing_contract")
    preprocessing_sha = _sha256_token(
        receipt.get("preprocessing_contract_sha256")
    )
    if not isinstance(contract, Mapping):
        errors.append("hailo_build_receipt_preprocessing_contract_missing")
        contract = {}
    if expected_task and str(contract.get("task") or "").strip().lower() != str(
        expected_task
    ).strip().lower():
        errors.append("hailo_build_receipt_task_mismatch")
    if not preprocessing_sha:
        errors.append("hailo_build_receipt_preprocessing_sha256_invalid")
    elif isinstance(contract, Mapping):
        try:
            resolved_contract, resolved_sha = resolve_image_preprocessing_contract(
                task=contract.get("task"),
                target_hw=contract.get("target_hw"),
                declared=contract,
            )
            if dict(resolved_contract) != dict(contract):
                errors.append("hailo_build_receipt_preprocessing_contract_noncanonical")
            if resolved_sha != preprocessing_sha:
                errors.append("hailo_build_receipt_preprocessing_sha256_mismatch")
            if preprocessing_contract_sha256(contract) != preprocessing_sha:
                errors.append("hailo_build_receipt_preprocessing_identity_mismatch")
        except Exception:
            errors.append("hailo_build_receipt_preprocessing_contract_invalid")

    cache_payload = receipt.get("cache_payload")
    cache_key = _sha256_token(receipt.get("cache_key"))
    if not isinstance(cache_payload, Mapping):
        errors.append("hailo_build_receipt_cache_payload_missing")
        cache_payload = {}
    else:
        cache_payload = dict(cache_payload)
    if not cache_key:
        errors.append("hailo_build_receipt_cache_key_invalid")
    elif _compact_json_sha256(cache_payload) != cache_key:
        errors.append("hailo_build_receipt_cache_key_payload_mismatch")
    if cache_payload.get("schema") not in _HAILO_HEF_CACHE_SCHEMAS:
        errors.append("hailo_build_receipt_cache_payload_schema_invalid")
    if cache_payload.get("schema") == "onnx-splitpoint/hailo-hef-cache-key-v3":
        shapes = cache_payload.get("net_input_shapes")
        shapes_valid = (
            shapes is None
            or (
                isinstance(shapes, list)
                and bool(shapes)
                and all(type(value) is int for value in shapes)
            )
            or (
                isinstance(shapes, Mapping)
                and bool(shapes)
                and all(
                    isinstance(name, str) and bool(name)
                    and isinstance(values, list) and bool(values)
                    and all(type(value) is int for value in values)
                    for name, values in shapes.items()
                )
            )
        )
        if (
            str(cache_payload.get("net_name") or "").strip()
            != str(receipt.get("net_name") or "").strip()
            or type(cache_payload.get("disable_rt_metadata_extraction"))
            is not bool
            or not shapes_valid
        ):
            errors.append("hailo_build_receipt_cache_translate_identity_invalid")
    if dict(cache_payload.get("preprocessing_contract") or {}) != dict(contract):
        errors.append("hailo_build_receipt_cache_preprocessing_contract_mismatch")
    if _sha256_token(cache_payload.get("preprocessing_contract_sha256")) != preprocessing_sha:
        errors.append("hailo_build_receipt_cache_preprocessing_sha256_mismatch")
    # The cache is sealed against the exact graph passed to the compiler.
    # With ONNX fixups this intentionally differs from the user-selected
    # source model recorded separately in the receipt.
    if _sha256_token(cache_payload.get("model_sha256")) != compiler_sha:
        errors.append("hailo_build_receipt_cache_compiler_onnx_mismatch")
    if str(cache_payload.get("hw_arch") or "") != str(receipt.get("hw_arch") or ""):
        errors.append("hailo_build_receipt_cache_hw_arch_mismatch")
    receipt_sdk = str(receipt.get("hailo_sdk_version") or "").strip()
    cache_sdk = str(cache_payload.get("hailo_sdk_version") or "").strip()
    if not receipt_sdk or receipt_sdk != cache_sdk:
        errors.append("hailo_build_receipt_cache_sdk_version_mismatch")
    cache_calibration_count = _strict_positive_int(
        cache_payload.get("calibration_count")
    )
    receipt_calibration_count = _strict_positive_int(
        receipt.get("calibration_count")
    )
    cache_requested_calibration_count = _strict_positive_int(
        cache_payload.get("requested_calibration_count")
    )
    receipt_requested_calibration_count = _strict_positive_int(
        receipt.get("requested_calibration_count")
    )
    if (
        cache_calibration_count is None
        or cache_calibration_count <= 0
        or receipt_calibration_count is None
        or receipt_calibration_count <= 0
    ):
        errors.append("hailo_build_receipt_calibration_count_invalid")
    if cache_calibration_count != receipt_calibration_count:
        errors.append("hailo_build_receipt_cache_calibration_count_mismatch")
    if (
        cache_requested_calibration_count is None
        or receipt_requested_calibration_count is None
        or cache_requested_calibration_count
        != receipt_requested_calibration_count
        or (
            cache_calibration_count is not None
            and cache_requested_calibration_count < cache_calibration_count
        )
    ):
        errors.append(
            "hailo_build_receipt_requested_calibration_count_mismatch"
        )
    cache_storage = str(
        cache_payload.get("calibration_storage") or ""
    ).strip().lower()
    receipt_storage = str(
        receipt.get("calibration_storage") or ""
    ).strip().lower()
    if (
        cache_storage not in {"memory", "memmap"}
        or cache_storage != receipt_storage
    ):
        errors.append("hailo_build_receipt_calibration_storage_mismatch")
    cache_memory_cap = _strict_positive_int(
        cache_payload.get("calibration_memory_cap_bytes")
    )
    receipt_memory_cap = _strict_positive_int(
        receipt.get("calibration_memory_cap_bytes")
    )
    if (
        cache_memory_cap is None
        or receipt_memory_cap is None
        or cache_memory_cap != receipt_memory_cap
    ):
        errors.append("hailo_build_receipt_calibration_memory_cap_mismatch")
    calibration_identity = str(
        cache_payload.get("calibration_identity") or ""
    ).strip()
    receipt_calibration_identity = str(
        receipt.get("calibration_identity") or ""
    ).strip()
    if not calibration_identity or not receipt_calibration_identity:
        errors.append("hailo_build_receipt_calibration_identity_missing")
    prepared_calibration_identity = _compact_json_sha256({
        "calibration_identity": calibration_identity,
        "preprocessing_contract_sha256": preprocessing_sha,
    })
    if _sha256_token(cache_payload.get("prepared_calibration_identity_sha256")) != prepared_calibration_identity:
        errors.append("hailo_build_receipt_cache_calibration_identity_mismatch")
    if _sha256_token(receipt.get("prepared_calibration_identity_sha256")) != prepared_calibration_identity:
        errors.append("hailo_build_receipt_calibration_identity_mismatch")
    if receipt_calibration_identity != calibration_identity:
        errors.append("hailo_build_receipt_calibration_source_mismatch")

    def _match_onnx(
        candidates: Sequence[Path], expected_sha: str, label: str,
        expected_filename: str = "",
    ) -> str:
        existing = [path.resolve() for path in candidates if path.is_file()]
        if not existing:
            errors.append(f"hailo_build_receipt_{label}_file_missing")
            return ""
        for path in existing:
            if (
                _file_sha256(path) == expected_sha
                and (
                    not expected_filename
                    or path.name == expected_filename
                )
            ):
                return str(path)
        errors.append(f"hailo_build_receipt_{label}_file_mismatch")
        return ""

    matched_source = _match_onnx(
        source_onnx_candidates, source_sha, "source_onnx"
    )
    compiler_candidates = list(compiler_onnx_candidates)
    if not compiler_candidates and source_sha and source_sha == compiler_sha:
        compiler_candidates = list(source_onnx_candidates)
    matched_compiler = _match_onnx(
        compiler_candidates, compiler_sha, "compiler_onnx",
        compiler_filename,
    )
    evidence.update({
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": compiler_sha,
        "compiler_onnx_filename": compiler_filename,
        "matched_source_onnx_path": matched_source,
        "matched_compiler_onnx_path": matched_compiler,
        "preprocessing_contract": dict(contract),
        "preprocessing_contract_sha256": preprocessing_sha,
        "cache_key": cache_key,
        "cache_payload_sha256": _compact_json_sha256(cache_payload),
        "hw_arch": str(receipt.get("hw_arch") or ""),
        "hailo_sdk_version": receipt_sdk,
        "calibration_identity": receipt_calibration_identity,
        "prepared_calibration_identity_sha256": str(
            receipt.get("prepared_calibration_identity_sha256") or ""
        ),
        "calibration_count": int(receipt_calibration_count or 0),
        "requested_calibration_count": int(
            receipt_requested_calibration_count or 0
        ),
        "calibration_storage": receipt_storage,
        "calibration_memory_cap_bytes": int(receipt_memory_cap or 0),
        "compiler_end_nodes": [
            str(value).strip()
            for value in _as_list(cache_payload.get("end_nodes"))
        ],
    })
    evidence["valid"] = not errors
    return evidence


def _hailo_receipt_binding_fields(
    evidence: Mapping[str, Any], *, suite_dir: Path
) -> Dict[str, Any]:
    receipt_path = Path(str(evidence.get("receipt_path") or ""))
    try:
        receipt_path_value = receipt_path.resolve().relative_to(
            suite_dir.resolve()
        ).as_posix()
    except (OSError, ValueError):
        receipt_path_value = str(receipt_path)
    compiler_path = Path(str(evidence.get("matched_compiler_onnx_path") or ""))
    try:
        compiler_path_value = compiler_path.resolve().relative_to(
            suite_dir.resolve()
        ).as_posix()
    except (OSError, ValueError):
        compiler_path_value = str(compiler_path)
    fields = {
        "hailo_build_receipt_path": receipt_path_value,
        "hailo_build_receipt_file_sha256": str(
            evidence.get("receipt_file_sha256") or ""
        ),
        "hailo_build_receipt_identity_sha256": str(
            evidence.get("receipt_identity_sha256") or ""
        ),
        "hailo_build_receipt_size_bytes": int(
            evidence.get("receipt_size_bytes") or 0
        ),
        "hailo_build_receipt_schema": _HAILO_HEF_RECEIPT_SCHEMA,
        "hailo_build_receipt_cache_key": str(evidence.get("cache_key") or ""),
        "hailo_build_receipt_cache_payload_sha256": str(
            evidence.get("cache_payload_sha256") or ""
        ),
        "hailo_build_receipt_hw_arch": str(evidence.get("hw_arch") or ""),
        "hailo_build_receipt_sdk_version": str(
            evidence.get("hailo_sdk_version") or ""
        ),
        "hailo_build_receipt_calibration_identity": str(
            evidence.get("calibration_identity") or ""
        ),
        "hailo_build_receipt_prepared_calibration_identity_sha256": str(
            evidence.get("prepared_calibration_identity_sha256") or ""
        ),
        "hailo_build_receipt_calibration_count": int(
            evidence.get("calibration_count") or 0
        ),
        "hailo_build_receipt_requested_calibration_count": int(
            evidence.get("requested_calibration_count") or 0
        ),
        "hailo_build_receipt_calibration_storage": str(
            evidence.get("calibration_storage") or ""
        ),
        "hailo_build_receipt_calibration_memory_cap_bytes": int(
            evidence.get("calibration_memory_cap_bytes") or 0
        ),
        "hailo_build_receipt_end_nodes": list(
            evidence.get("compiler_end_nodes") or []
        ),
        "preprocessing_contract": dict(
            evidence.get("preprocessing_contract") or {}
        ),
        "preprocessing_contract_sha256": str(
            evidence.get("preprocessing_contract_sha256") or ""
        ),
        "source_onnx_sha256": str(evidence.get("source_onnx_sha256") or ""),
        "compiler_onnx_sha256": str(
            evidence.get("compiler_onnx_sha256") or ""
        ),
        "compiler_onnx_filename": str(
            evidence.get("compiler_onnx_filename") or ""
        ),
        "compiler_onnx_path": compiler_path_value,
    }
    # Generic aliases make the path/file/content distinction explicit for
    # consumers that already use build-receipt terminology across backends.
    fields.update({
        "build_receipt_path": fields["hailo_build_receipt_path"],
        "build_receipt_file_sha256": fields[
            "hailo_build_receipt_file_sha256"
        ],
        "build_receipt_identity_sha256": fields[
            "hailo_build_receipt_identity_sha256"
        ],
    })
    return fields


_HAILO_STALE_ARTIFACT_CLAIM_KEYS = {
    "artifact_path",
    "artifact_sha256",
    "artifact_size_bytes",
    "recorded_artifact_path",
    "recorded_artifact_sha256",
    "recorded_artifact_size_bytes",
    "artifact_binding_sha256",
    "artifact_binding_source",
    "contract_resolution_status",
    "hailo_build_receipt_path",
    "hailo_build_receipt_file_sha256",
    "hailo_build_receipt_identity_sha256",
    "hailo_build_receipt_size_bytes",
    "hailo_build_receipt_schema",
    "hailo_build_receipt_cache_key",
    "hailo_build_receipt_cache_payload_sha256",
    "hailo_build_receipt_hw_arch",
    "hailo_build_receipt_sdk_version",
    "hailo_build_receipt_calibration_identity",
    "hailo_build_receipt_prepared_calibration_identity_sha256",
    "hailo_build_receipt_calibration_count",
    "hailo_build_receipt_requested_calibration_count",
    "hailo_build_receipt_calibration_storage",
    "hailo_build_receipt_calibration_memory_cap_bytes",
    "hailo_build_receipt_end_nodes",
    "build_receipt_path",
    "build_receipt_file_sha256",
    "build_receipt_identity_sha256",
    "source_onnx_sha256",
    "compiler_onnx_sha256",
    "compiler_onnx_filename",
    "compiler_onnx_path",
}


def _demote_hailo_full_contract_claim(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove inherited success claims until current bytes are re-attested."""

    demoted = dict(contract)
    for key in _HAILO_STALE_ARTIFACT_CLAIM_KEYS:
        demoted.pop(key, None)
    demoted.update({
        "contract_status": "pending_build_or_prepare",
        "artifact_binding_status": "pending_receipt_validation",
        "artifact_binding_error": "hailo_full_receipt_not_verified",
    })
    return demoted


def _demote_hailo_suite_full_claims(
    suite_bench: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed on stale BenchmarkSet HEF success/receipt projections."""

    payload = dict(suite_bench)
    hailo = dict(payload.get("hailo") or {})
    hefs = dict(hailo.get("hefs") or {})
    for alias, raw_meta in list(hefs.items()):
        if not _target_backend_label(alias).startswith("hailo"):
            continue
        meta = dict(raw_meta or {}) if isinstance(raw_meta, Mapping) else {}
        full_build = dict(meta.get("full_build") or {})
        for key in _HAILO_STALE_ARTIFACT_CLAIM_KEYS | {
            "artifact_hash", "preprocessing_contract",
            "preprocessing_contract_sha256",
        }:
            full_build.pop(key, None)
        full_build.update({
            "ok": False,
            "receipt_validated": False,
            "status": "pending_receipt_validation",
        })
        meta["full_build"] = full_build
        meta["full_prepared_baseline"] = False
        meta["full_receipt_attested"] = False
        meta.pop("full_build_receipt", None)
        if isinstance(meta.get("full_output_contract"), Mapping):
            meta["full_output_contract"] = _demote_hailo_full_contract_claim(
                meta["full_output_contract"]
            )
        hefs[alias] = meta
    hailo["hefs"] = hefs
    payload["hailo"] = hailo
    return payload


def _suite_artifact_path(suite_dir: Path, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = suite_dir / path
    try:
        path = path.resolve()
    except OSError:
        return None
    return path if path.is_file() and path.stat().st_size > 0 else None


def _hailo_full_artifact_evidence(
    suite_dir: Path,
    suite_bench: Mapping[str, Any],
    *,
    backend: str,
    task: str,
    copied_verified: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return only receipt-attested, byte-addressed Full-HEF evidence."""
    backend_exact = _target_backend_label(backend)
    backend_key = _hailo_contract_backend(backend_exact)
    copied = copied_verified.get(backend_exact)
    if copied is None and backend_exact in {"hailo10", "hailo10h"}:
        copied = copied_verified.get(backend_key)
    if isinstance(copied, Mapping):
        copied_path = Path(str(copied.get("path") or ""))
        copied_receipt_raw = str(copied.get("receipt_path") or "").strip()
        copied_receipt = Path(copied_receipt_raw) if copied_receipt_raw else None
        source_hints, compiler_hints = _receipt_path_hints(
            artifact=copied_path,
            suite_bench=suite_bench,
        )
        for key, hints in (
            ("matched_source_onnx_path", source_hints),
            ("matched_compiler_onnx_path", compiler_hints),
        ):
            value = str(copied.get(key) or "").strip()
            if value:
                hints.append(Path(value))
        validated = _validate_hailo_build_receipt(
            copied_path,
            receipt_path=copied_receipt,
            source_onnx_candidates=source_hints,
            compiler_onnx_candidates=compiler_hints,
            expected_backend=backend,
            expected_task=task,
        )
        if (
            validated.get("valid") is True
            and str(copied.get("sha256") or "") == str(validated.get("sha256") or "")
            and str(copied.get("receipt_file_sha256") or "")
            == str(validated.get("receipt_file_sha256") or "")
            and str(copied.get("receipt_identity_sha256") or "")
            == str(validated.get("receipt_identity_sha256") or "")
        ):
            validated["source"] = str(
                copied.get("source") or "verified_prepared_artifact_copy"
            )
            return validated

    aliases = (
        ("hailo10", "hailo10h")
        if backend_exact in {"hailo10", "hailo10h"}
        else (backend_exact,)
    )
    hefs = (
        ((suite_bench.get("hailo") or {}).get("hefs") or {})
        if isinstance(suite_bench.get("hailo"), Mapping) else {}
    )
    candidates: List[Tuple[Path, str, Mapping[str, Any]]] = []
    for alias in aliases:
        meta = hefs.get(alias) if isinstance(hefs, Mapping) else None
        if isinstance(meta, Mapping):
            artifact = _suite_artifact_path(suite_dir, meta.get("full"))
            if artifact is not None:
                # Success markers are neither necessary nor sufficient. The
                # sibling receipt and current ONNX/HEF bytes are authoritative.
                candidates.append((
                    artifact,
                    "benchmark_set_declared_full_artifact",
                    meta,
                ))
        for name in ("compiled.hef", "model.hef"):
            artifact = suite_dir / "hailo" / alias / "full" / name
            result = artifact.parent / "hailo_hef_build_result.json"
            result_payload = read_json(result, default={}) or {}
            if not isinstance(result_payload, Mapping):
                result_payload = {}
            if artifact.is_file() and artifact.stat().st_size > 0:
                result_hef = _suite_artifact_path(
                    suite_dir, result_payload.get("hef_path") or artifact,
                )
                if result_hef is not None and result_hef.resolve() == artifact.resolve():
                    candidates.append((
                        artifact.resolve(),
                        "receipt_attested_full_artifact",
                        (meta if isinstance(meta, Mapping) else {}),
                    ))
    validated_candidates: List[Dict[str, Any]] = []
    for artifact, source, meta in candidates:
        source_hints, compiler_hints = _receipt_path_hints(
            artifact=artifact,
            suite_bench=suite_bench,
            backend_meta=meta,
        )
        evidence = _validate_hailo_build_receipt(
            artifact,
            source_onnx_candidates=source_hints,
            compiler_onnx_candidates=compiler_hints,
            expected_backend=backend,
            expected_task=task,
        )
        if evidence.get("valid") is True:
            evidence["source"] = source
            validated_candidates.append(evidence)
    unique_paths = {
        str(Path(str(item.get("path") or "")).resolve())
        for item in validated_candidates
    }
    if len(unique_paths) != 1:
        return {}
    path = next(iter(unique_paths))
    matches = [
        item for item in validated_candidates
        if str(Path(str(item.get("path") or "")).resolve()) == path
    ]
    if not matches:
        return {}
    evidence = dict(matches[0])
    evidence["source"] = "+".join(sorted({
        str(item.get("source") or "") for item in matches
        if str(item.get("source") or "")
    }))
    return evidence


def _hailo_suite_raw_head_reconciliation(
    suite_dir: Path,
    suite_bench: Mapping[str, Any],
    *,
    backend: str,
    evidence: Mapping[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Reconcile a planned endpoint with byte-bound BenchmarkSet metadata.

    Some compilers lower a decoded source ONNX endpoint to raw accelerator
    heads.  That fact may only replace the planned endpoint when one exact HEF,
    its successful build receipt, and one non-ambiguous node contract all bind
    to the same bytes.  ``conflict`` is true when raw-head metadata exists but
    cannot satisfy that complete chain.
    """
    backend_exact = _target_backend_label(backend)
    aliases = (
        ("hailo10", "hailo10h")
        if backend_exact in {"hailo10", "hailo10h"}
        else (backend_exact,)
    )
    hefs = (
        ((suite_bench.get("hailo") or {}).get("hefs") or {})
        if isinstance(suite_bench.get("hailo"), Mapping) else {}
    )
    evidence_path = Path(str(evidence.get("path") or ""))
    evidence_sha = str(evidence.get("sha256") or "").lower().removeprefix("sha256:")
    raw_metadata_seen = False
    invalid_raw_metadata_seen = False
    mixed_endpoint_metadata_seen = False
    candidates: List[Dict[str, Any]] = []

    def _endpoint_family(value: Any) -> str:
        token = str(value or "").strip().lower()
        if not token:
            return ""
        if token in {"raw_head", "raw_detection_head"} or token.endswith(
            "_raw_head"
        ):
            return "raw"
        if token in {"decoded", "decoded_nms"}:
            return "decoded"
        return "invalid"

    def _node_alias(
        container: Mapping[str, Any], key: str,
    ) -> Tuple[List[str], bool]:
        if key not in container:
            return [], False
        raw = container.get(key)
        if not isinstance(raw, list):
            return [], True
        values = [str(value).strip() for value in raw]
        return values, bool(any(not value for value in values))

    for alias in aliases:
        meta = hefs.get(alias) if isinstance(hefs, Mapping) else None
        if not isinstance(meta, Mapping):
            continue
        output_contract = (
            meta.get("full_output_contract")
            if isinstance(meta.get("full_output_contract"), Mapping)
            else {}
        )
        mode_values = [
            str(value).strip().lower()
            for value in (
                meta.get("full_endpoint_mode"),
                output_contract.get("mode"),
                output_contract.get("endpoint_mode"),
            )
            if str(value or "").strip()
        ]
        mode_families = {_endpoint_family(value) for value in mode_values}
        mode_conflict = bool(
            "invalid" in mode_families or len(mode_families) > 1
        )
        is_raw = mode_families == {"raw"}
        output_mode = "raw_detection_head" if is_raw else (
            "decoded" if mode_families == {"decoded"} else ""
        )
        if mode_conflict:
            raw_metadata_seen = True
            invalid_raw_metadata_seen = True
            continue
        if not is_raw:
            artifact = _suite_artifact_path(suite_dir, meta.get("full"))
            if (
                mode_values
                and artifact is not None
                and evidence_path.is_file()
                and artifact.resolve() == evidence_path.resolve()
            ):
                mixed_endpoint_metadata_seen = True
            continue
        raw_metadata_seen = True
        artifact = _suite_artifact_path(suite_dir, meta.get("full"))
        if artifact is None or not evidence_path.is_file():
            invalid_raw_metadata_seen = True
            continue
        if artifact.resolve() != evidence_path.resolve():
            invalid_raw_metadata_seen = True
            continue
        build = meta.get("full_build")
        if isinstance(build, Mapping) and build.get("artifact_hash"):
            declared_sha = str(
                build.get("artifact_hash") or ""
            ).lower().removeprefix("sha256:")
            if declared_sha != evidence_sha:
                invalid_raw_metadata_seen = True
                continue
        if output_contract.get("requires_external_postprocess") is not True:
            invalid_raw_metadata_seen = True
            continue
        node_aliases: List[List[str]] = []
        node_alias_invalid = False
        for container, key in (
            (meta, "full_end_node_names"),
            (output_contract, "end_node_names"),
            (output_contract, "full_end_node_names"),
        ):
            values, invalid = _node_alias(container, key)
            if key in container:
                node_aliases.append(values)
            node_alias_invalid = bool(node_alias_invalid or invalid)
        if (
            node_alias_invalid
            or not node_aliases
            or any(values != node_aliases[0] for values in node_aliases[1:])
        ):
            invalid_raw_metadata_seen = True
            continue
        nodes = list(node_aliases[0])
        receipt_nodes = [
            str(value).strip()
            for value in _as_list(evidence.get("compiler_end_nodes"))
        ]
        source_flag_values: List[bool] = []
        source_flag_invalid = False
        for container in (meta, output_contract):
            if "source_onnx_multiscale_raw_head" not in container:
                continue
            value = container.get("source_onnx_multiscale_raw_head")
            if not isinstance(value, bool):
                source_flag_invalid = True
                continue
            source_flag_values.append(value)
        if source_flag_invalid or len(set(source_flag_values)) > 1:
            invalid_raw_metadata_seen = True
            continue
        source_graph_raw = bool(
            source_flag_values and source_flag_values[0]
        )
        if nodes:
            raw_endpoint_origin = "compiler_end_nodes"
            node_contract_valid = bool(
                not any(not value for value in nodes)
                and nodes == receipt_nodes
                and not source_graph_raw
            )
        else:
            raw_endpoint_origin = "source_onnx_graph_outputs"
            node_contract_valid = bool(
                source_graph_raw
                and not receipt_nodes
            )
        if not node_contract_valid:
            invalid_raw_metadata_seen = True
            continue
        candidates.append({
            "metadata_alias": alias,
            "endpoint_mode": "raw_detection_head",
            "full_end_node_names": nodes,
            "raw_endpoint_origin": raw_endpoint_origin,
            "source_onnx_multiscale_raw_head": source_graph_raw,
            "output_contract_mode": output_mode,
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": evidence_sha,
            "source": "verified_benchmark_set_full_raw_head_contract",
        })
    if not raw_metadata_seen:
        return {}, False
    if invalid_raw_metadata_seen or mixed_endpoint_metadata_seen:
        return {}, True
    unique: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        identity = _compact_json_sha256({
            "endpoint_mode": candidate["endpoint_mode"],
            "full_end_node_names": candidate["full_end_node_names"],
            "raw_endpoint_origin": candidate["raw_endpoint_origin"],
            "source_onnx_multiscale_raw_head": candidate[
                "source_onnx_multiscale_raw_head"
            ],
            "output_contract_mode": candidate["output_contract_mode"],
            "artifact_path": candidate["artifact_path"],
            "artifact_sha256": candidate["artifact_sha256"],
        })
        unique.setdefault(identity, candidate)
    if len(unique) != 1:
        return {}, True
    return next(iter(unique.values())), False


_DECODED_DETECTION_METADATA_FIELDS = (
    "output_record_format",
    "coordinate_format",
    "coordinate_space",
    "source_coordinate_space",
    "score_semantics",
    "class_id_semantics",
)


def _normalize_reconciled_raw_detection_contract(
    contract: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Replace a provisional decoded declaration with one atomic raw contract."""
    normalized = dict(contract)
    for field in _DECODED_DETECTION_METADATA_FIELDS:
        normalized.pop(field, None)
    # Persist exactly one canonical alias family.  Leaving legacy aliases in
    # the same row would make an otherwise valid second materialization
    # appear contradictory after the normalized fields are written.
    normalized.pop("mode", None)
    normalized.pop("end_node_names", None)
    normalized.update({
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_mode": "raw_detection_head",
        "output_format": "raw_detection_tensors",
        "compiled_artifact_raw_head": True,
        "requires_external_postprocess": True,
        "host_tail_required": True,
        "postprocessing_required": True,
        "full_end_node_names": list(
            reconciliation["full_end_node_names"]
        ),
        "raw_endpoint_origin": str(
            reconciliation["raw_endpoint_origin"]
        ),
        "source_onnx_multiscale_raw_head": bool(
            reconciliation["source_onnx_multiscale_raw_head"]
        ),
        "contract_reconciliation_status": (
            "verified_suite_artifact_raw_head"
        ),
        "contract_reconciliation_source": str(
            reconciliation["source"]
        ),
        "contract_reconciliation_metadata_alias": str(
            reconciliation["metadata_alias"]
        ),
        "warning": (
            "Accelerator output is a byte-bound raw detection head; "
            "frozen host decode/NMS is required."
        ),
    })
    return normalized


def promote_verified_hailo_full_contracts(
    *,
    suite_dir: Path,
    model_id: str,
    task: str,
    suite_bench: Mapping[str, Any],
    contracts: List[Dict[str, Any]],
    copied_verified: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Mark Full Hailo contracts recorded only after exact HEF verification."""
    promotions: List[Dict[str, Any]] = []
    groups: Dict[str, List[int]] = {}
    for index, contract in enumerate(contracts):
        if contract.get("requested") is False:
            continue
        backend = _target_backend_label(contract.get("backend"))
        if not backend.startswith("hailo"):
            continue
        if str(contract.get("variant") or "full").strip().lower() != "full":
            continue
        if str(contract.get("model_id") or "").strip() != str(model_id):
            continue
        groups.setdefault(backend, []).append(index)

    for backend, indexes in groups.items():
        # Duplicate exact contracts are ambiguous; do not bless either copy.
        if len(indexes) != 1:
            continue
        index = indexes[0]
        contract = dict(contracts[index])
        exact_backend = _target_backend_label(contract.get("backend"))
        contract_task = str(contract.get("task") or "").strip().lower()
        if contract_task != str(task or "").strip().lower() or contract_task not in {
            "classification", "detection",
        }:
            continue
        evidence = _hailo_full_artifact_evidence(
            suite_dir, suite_bench, backend=exact_backend,
            task=contract_task,
            copied_verified=copied_verified,
        )
        artifact = Path(str(evidence.get("path") or ""))
        if (
            evidence.get("valid") is not True
            or not artifact.is_file()
            or not str(evidence.get("sha256") or "")
            or not str(evidence.get("receipt_file_sha256") or "")
            or not str(evidence.get("receipt_identity_sha256") or "")
        ):
            continue
        reconciliation: Dict[str, Any] = {}
        reconciliation_conflict = False
        if contract_task == "detection":
            receipt_end_nodes = [
                str(value).strip()
                for value in _as_list(evidence.get("compiler_end_nodes"))
            ]
            reconciliation, reconciliation_conflict = (
                _hailo_suite_raw_head_reconciliation(
                    suite_dir, suite_bench, backend=exact_backend,
                    evidence=evidence,
                )
            )
            if reconciliation_conflict:
                continue
            if reconciliation:
                contract = _normalize_reconciled_raw_detection_contract(
                    contract, reconciliation,
                )
            elif receipt_end_nodes:
                declared_end_nodes = [
                    str(value).strip()
                    for value in _as_list(
                        contract.get("full_end_node_names")
                        or contract.get("end_node_names")
                    )
                ]
                if declared_end_nodes != receipt_end_nodes:
                    # A provisional decoded plan may not erase compiler-sealed
                    # output heads. Without exact metadata this endpoint is
                    # unknown and cannot be promoted.
                    continue
        mode = str(contract.get("endpoint_mode") or "").strip().lower()
        raw = mode in {"raw_head", "raw_detection_head"}
        if mode not in {"decoded", "decoded_nms", "classification_logits", "classification_probabilities", "raw_head", "raw_detection_head"}:
            continue
        if contract.get("host_tail_required") is not raw:
            continue
        if contract.get("postprocessing_required") is not raw:
            continue
        artifact_rel = artifact.resolve().relative_to(suite_dir.resolve()).as_posix()
        receipt_fields = _hailo_receipt_binding_fields(
            evidence, suite_dir=suite_dir
        )
        binding_payload = {
            "model_id": str(model_id), "task": contract_task,
            "backend": exact_backend, "variant": "full",
            "endpoint_mode": mode,
            "host_tail_required": bool(contract.get("host_tail_required")),
            "postprocessing_required": bool(contract.get("postprocessing_required")),
            "artifact_path": artifact_rel,
            "artifact_sha256": str(evidence["sha256"]),
            "artifact_size_bytes": int(evidence["size_bytes"]),
            "full_end_node_names": list(contract.get("full_end_node_names") or []),
            "contract_reconciliation_status": str(
                contract.get("contract_reconciliation_status") or ""
            ),
            "contract_reconciliation_source": str(
                contract.get("contract_reconciliation_source") or ""
            ),
            **receipt_fields,
        }
        contract.pop("artifact_binding_error", None)
        contract.update({
            "task": contract_task,
            "contract_status": "recorded",
            "artifact_path": artifact_rel,
            "artifact_sha256": binding_payload["artifact_sha256"],
            "artifact_size_bytes": binding_payload["artifact_size_bytes"],
            "recorded_artifact_path": artifact_rel,
            "recorded_artifact_sha256": binding_payload["artifact_sha256"],
            "recorded_artifact_size_bytes": binding_payload["artifact_size_bytes"],
            "artifact_binding_status": "verified",
            "artifact_binding_source": str(evidence.get("source") or ""),
            "artifact_binding_sha256": sha256_payload(binding_payload),
            **receipt_fields,
        })
        contracts[index] = contract
        promotions.append({
            "backend": exact_backend,
            "artifact_path": artifact_rel,
            "artifact_sha256": binding_payload["artifact_sha256"],
            "artifact_size_bytes": binding_payload["artifact_size_bytes"],
            "artifact_binding_sha256": contract["artifact_binding_sha256"],
            "source": contract["artifact_binding_source"],
            **receipt_fields,
            "contract_reconciliation_status": str(
                contract.get("contract_reconciliation_status") or ""
            ),
        })
    return promotions

def remote_import_preflight() -> Dict[str, Any]:
    """Import and invoke the promoter without touching hardware or artifacts."""
    contracts: List[Dict[str, Any]] = [{
        "backend": "hailo8",
        "variant": "full",
        "model_id": "__remote_contract_preflight__",
        "task": "classification",
        "endpoint_mode": "classification_logits",
        "host_tail_required": False,
        "postprocessing_required": False,
    }]
    with tempfile.TemporaryDirectory(
        prefix="onnx_splitpoint_hailo_contract_preflight_"
    ) as empty_suite:
        promotions = promote_verified_hailo_full_contracts(
            suite_dir=Path(empty_suite),
            model_id="__remote_contract_preflight__",
            task="classification",
            suite_bench={},
            contracts=contracts,
            copied_verified={},
        )
    if promotions:
        raise RuntimeError("remote_contract_preflight_unexpected_promotion")
    return {
        "ok": True,
        "module": __name__,
        "helper_invoked": True,
        "fail_closed_result": promotions,
    }
