from __future__ import annotations

"""Formal benchmark-generator binding for the Evaluation Workflow.

The existing GUI benchmark generator is still the component that can perform
heavy graph splitting and Hailo builds interactively. This module provides the
formal v49c contract layer that the EvaluationWorkflowRunner can execute in a
headless/CLI-safe way:

* consume ``analysis/final_candidate_plan.json`` as the authoritative source;
* materialize a benchmark-suite handoff folder with benchmark_set.json,
  benchmark_plan.json, per-case manifests and a runnable-status helper;
* record generation decisions, accepted/rejected cases and policy backfills;
* reuse/copy prepared full-Hailo artifacts when they already exist;
* write structured backend build decisions instead of hiding them in logs.

No synthetic benchmark measurements are produced here. Runtime measurements
must come from local/remote benchmark execution and are then normalized by
``workflow.results``.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..preprocessing_contract import (
    preprocessing_contract_sha256,
    resolve_image_preprocessing_contract,
)
from ..hailo_full_contract_promotion import (
    promote_verified_hailo_full_contracts as _promote_verified_hailo_full_contracts,
)
from ..cache_verify_policy import cache_verify_guard
from ..management_reference import (
    bind_management_cpu_reference_runs,
    is_cpu_reference_recipe,
    management_cpu_reference_required,
)
from ..native_full_quality import enabled_run_profiles
from .full_only_quality_canary import project_full_only_quality_plan_rows
from .artifacts import now_iso, read_json, relpath, sha256_payload, write_csv, write_json, write_text


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


def _stage_descriptor_for_backend(backend: str) -> Dict[str, Any]:
    b = str(backend or "").strip().lower().replace("-", "_")
    if b in {"cpu", "cpu_ort", "ort_cpu"}:
        return {"type": "onnxruntime", "provider": "cpu"}
    if b in {"cuda", "cuda_ort", "ort_cuda", "gpu"}:
        return {"type": "onnxruntime", "provider": "cuda"}
    if b in {"trt", "tensorrt", "tensor_rt"}:
        return {"type": "onnxruntime", "provider": "tensorrt"}
    if "deepx" in b or "dx_m1" in b or "dxm1" in b:
        return {"type": "deepx", "target": "deepx_m1", "artifact_kind": "dxnn"}
    if b.startswith("hailo"):
        return {"type": "hailo", "hw_arch": b}
    return {"type": "onnxruntime", "provider": b or "auto"}


def _benchmark_runs_from_profile(profile_payload: Mapping[str, Any], targets: Sequence[str]) -> List[Dict[str, Any]]:
    """Return the benchmark plan to expose in the formal suite handoff."""

    runs: List[Dict[str, Any]] = []
    profile_runs = profile_payload.get("run_profiles") if isinstance(profile_payload, Mapping) else None
    if isinstance(profile_runs, list) and profile_runs:
        for idx, raw in enumerate(
            enabled_run_profiles(profile_runs), start=1,
        ):
            enabled = raw.get("enabled")
            if enabled is False or str(enabled or "").strip().lower() in {
                "0", "false", "no", "off", "disabled",
            }:
                continue
            run = dict(raw)
            run.setdefault("id", f"profile_run_{idx:02d}")
            run.setdefault("source", "evaluation_profile.run_profiles")
            runs.append(run)
        if runs and (
            management_cpu_reference_required(profile_payload)
            and not cache_verify_guard(profile_payload)
        ):
            automatic = not any(
                is_cpu_reference_recipe(row) for row in runs
            )
            runs = bind_management_cpu_reference_runs(
                runs,
                automatic=automatic,
                require_existing=False,
            )
        # An explicit run_profiles list is authoritative even when every row is
        # disabled.  Falling through to target-derived Cartesian runs would
        # silently resurrect those disabled selections.
        return project_full_only_quality_plan_rows(profile_payload, runs)

    backends: List[str] = []
    for raw in targets or []:
        b = _target_backend_label(raw)
        if b and b not in backends:
            backends.append(b)
    if not backends:
        backends = ["cpu_ort"]

    for backend in backends:
        runs.append({
            "id": f"{backend}_full",
            "variant": "full",
            "backend": backend,
            "stage1": _stage_descriptor_for_backend(backend),
            "stage2": _stage_descriptor_for_backend(backend),
            "source": "derived_from_targets",
        })

    if "hailo8" in backends:
        host_order = [b for b in backends if b != "hailo8"] or ["cpu_ort"]
        for host in host_order:
            runs.append({
                "id": f"hailo8_to_{host}_split",
                "variant": "split",
                "backend": f"hailo8_to_{host}",
                "stage1": _stage_descriptor_for_backend("hailo8"),
                "stage2": _stage_descriptor_for_backend(host),
                "source": "derived_from_targets",
            })
    elif len(backends) >= 2:
        runs.append({
            "id": f"{backends[0]}_to_{backends[1]}_split",
            "variant": "split",
            "backend": f"{backends[0]}_to_{backends[1]}",
            "stage1": _stage_descriptor_for_backend(backends[0]),
            "stage2": _stage_descriptor_for_backend(backends[1]),
            "source": "derived_from_targets",
        })
    else:
        runs.append({
            "id": f"{backends[0]}_split",
            "variant": "split",
            "backend": backends[0],
            "stage1": _stage_descriptor_for_backend(backends[0]),
            "stage2": _stage_descriptor_for_backend(backends[0]),
            "source": "derived_from_targets",
        })
    if (
        management_cpu_reference_required(profile_payload)
        and not cache_verify_guard(profile_payload)
    ):
        automatic = not any(is_cpu_reference_recipe(row) for row in runs)
        runs = bind_management_cpu_reference_runs(
            runs,
            automatic=automatic,
            require_existing=False,
        )
    return runs


def _normalized_cases(final_candidate_plan: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_raw = _as_list(final_candidate_plan.get("selected_candidates"))
    excluded_raw = _as_list(final_candidate_plan.get("excluded_candidates"))
    cases: List[Dict[str, Any]] = []
    for idx, raw in enumerate(selected_raw, start=1):
        if not isinstance(raw, Mapping):
            continue
        case = dict(raw)
        split = _safe_int(case.get("split_index") or case.get("boundary") or case.get("boundary_index"))
        cid = _case_id_for(case, idx)
        case.update({
            "case_id": cid,
            "case_dir": cid,
            "folder": cid,
            "split_index": split,
            "boundary": split,
            "boundary_index": split,
            "prediction_rank": case.get("prediction_rank", case.get("source_rank", case.get("rank", idx))),
            "accepted_by_workflow_plan": True,
            "generation_status": "accepted_by_formal_candidate_plan",
            "generation_reason": case.get("selection_reason") or "selected from final_candidate_plan.json",
        })
        cases.append(case)
    rejected: List[Dict[str, Any]] = []
    for raw in excluded_raw:
        if isinstance(raw, Mapping):
            rec = dict(raw)
            rec.setdefault("generation_status", "excluded_before_generation")
            rejected.append(rec)
    return cases, rejected


def materialize_benchmark_generator_binding(
    *,
    run_dir: str | Path,
    model_id: str,
    model_path: str,
    profile_id: str,
    profile_payload: Mapping[str, Any],
    targets: Sequence[str],
    prediction: Mapping[str, Any],
    final_candidate_plan: Mapping[str, Any],
    full_baseline_plan: Mapping[str, Any],
    output_contracts: Mapping[str, Any],
    skip_benchmarks: bool,
    no_remote: bool,
) -> Dict[str, Any]:
    """Create a contract-only benchmark-generator handoff.

    v49p no longer materializes the reduced formal mini-suite in the normal
    workflow.  Real runs must use the existing BenchmarkSet generator; this
    function is only a cheap contract for dry/skip/diagnostic runs and therefore
    writes no benchmark_suite.py, no case directories and no generated_suite.
    """

    root = Path(run_dir)
    model_dir = root / "models" / str(model_id)
    base = model_dir / "benchmark_set"
    base.mkdir(parents=True, exist_ok=True)

    cases, rejected = _normalized_cases(final_candidate_plan)
    runs = _benchmark_runs_from_profile(profile_payload, targets)
    candidate_plan_rel = relpath(model_dir / "analysis" / "final_candidate_plan.json", root)
    prediction_rel = relpath(model_dir / "analysis" / "prediction.json", root)
    candidate_plan_hash = sha256_payload(final_candidate_plan)
    objective = str(_as_dict(profile_payload.get("benchmark") if isinstance(profile_payload, Mapping) else {}).get("objective") or "latency")

    benchmark_plan = {
        "schema": "onnx-splitpoint/benchmark-plan",
        "schema_version": 3,
        "created_at": now_iso(),
        "profile_id": profile_id,
        "model_id": model_id,
        "model_path": str(model_path or ""),
        "source_prediction_path": prediction_rel,
        "source_prediction_artifact_id": prediction.get("artifact_id", ""),
        "source_candidate_plan_path": candidate_plan_rel,
        "source_candidate_plan_artifact_id": final_candidate_plan.get("artifact_id", ""),
        "candidate_plan_sha256": "sha256:" + candidate_plan_hash,
        "candidate_plan_is_authoritative": True,
        "runs": runs,
        "planned_runs": [dict(run) for run in runs],
        "cases": cases,
        "matrix": [],
        "objective": objective,
        "materialization_scope": "contract_only_no_suite",
        "execution_policy": {
            "skip_benchmarks": bool(skip_benchmarks),
            "no_remote": bool(no_remote),
            "measurements_must_be_ingested_from": "benchmark_results_*.json/csv",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    }
    p_plan = write_json(base / "benchmark_plan.json", benchmark_plan)

    decisions = {
        "schema": "onnx-splitpoint/benchmark-generation-decisions",
        "schema_version": 5,
        "created_at": now_iso(),
        "model_id": model_id,
        "profile_id": profile_id,
        "source_candidate_plan_path": candidate_plan_rel,
        "source_candidate_plan_artifact_id": final_candidate_plan.get("artifact_id", ""),
        "candidate_plan_sha256": "sha256:" + candidate_plan_hash,
        "candidate_plan_consumed": True,
        "accepted_cases": cases,
        "rejected_cases": rejected,
        "policy_promotions": _as_list(final_candidate_plan.get("policy_promotions")),
        "policy_backfills": _as_list(final_candidate_plan.get("policy_backfills")),
        "generator_overrides": [],
        "split_graphs_materialized": False,
        "backend_artifacts_materialized": False,
        "runtime_measurements_materialized": False,
        "status": "contract_only_no_suite",
        "next_service": "legacy_benchmarkset_generator",
        "note": "v49p keeps this as a contract only. Real evaluation runs delegate to the existing BenchmarkSet pipeline rather than writing the old reduced formal mini-suite.",
    }
    generator_input = {
        "schema": "onnx-splitpoint/benchmark-generator-input",
        "schema_version": 3,
        "created_at": now_iso(),
        "model_id": model_id,
        "model_path": str(model_path or ""),
        "profile_id": profile_id,
        "candidate_plan_path": candidate_plan_rel,
        "candidate_plan_artifact_id": final_candidate_plan.get("artifact_id", ""),
        "candidate_plan_sha256": "sha256:" + candidate_plan_hash,
        "targets": [str(t) for t in targets],
        "requested_cases": final_candidate_plan.get("requested_cases"),
        "cases": cases,
        "force_candidate_plan_first": True,
        "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        "contract_only": True,
    }
    binding = {
        "schema": "onnx-splitpoint/benchmark-generator-binding",
        "schema_version": 2,
        "created_at": now_iso(),
        "model_id": model_id,
        "profile_id": profile_id,
        "candidate_plan_path": candidate_plan_rel,
        "candidate_plan_is_authoritative": True,
        "accepted_case_count": len(cases),
        "rejected_case_count": len(rejected),
        "execution_ready": False,
        "runtime_execution_skipped_by_option": bool(skip_benchmarks),
        "remote_execution_disabled": bool(no_remote),
        "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
    }
    p_input = write_json(base / "generator_input.json", generator_input)
    p_dec = write_json(base / "generation_decisions.json", decisions)
    p_binding = write_json(base / "generator_binding.json", binding)
    p_set = write_json(base / "benchmark_set.json", {
        "schema": "onnx-splitpoint/benchmark-set-contract",
        "schema_version": 5,
        "model_id": model_id,
        "profile_id": profile_id,
        "suite_dir": "",
        "legacy_suite_dir": "",
        "benchmark_plan_json": relpath(p_plan, root),
        "generation_decisions_path": "generation_decisions.json",
        "generator_input_path": "generator_input.json",
        "generator_binding_path": "generator_binding.json",
        "cases": cases,
        "materialized": False,
        "materialization_scope": "contract_only_no_suite",
        "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
    })
    p_cases = write_csv(base / "benchmark_cases.csv", [
        {
            "model_id": model_id,
            "case_id": c.get("case_id", ""),
            "split_index": c.get("split_index", ""),
            "prediction_rank": c.get("prediction_rank", ""),
            "predicted_total_latency_ms": c.get("predicted_total_latency_ms", ""),
            "predicted_transfer_latency_ms": c.get("predicted_transfer_latency_ms", ""),
            "predicted_hailo_feasible": c.get("predicted_hailo_feasible", ""),
            "generation_status": c.get("generation_status", ""),
        }
        for c in cases
    ], ["model_id", "case_id", "split_index", "prediction_rank", "predicted_total_latency_ms", "predicted_transfer_latency_ms", "predicted_hailo_feasible", "generation_status"])
    artifacts: Dict[str, Path] = {
        "benchmark_plan_json": p_plan,
        "generator_input_json": p_input,
        "generation_decisions_json": p_dec,
        "benchmark_set_json": p_set,
        "generator_binding_json": p_binding,
        "benchmark_cases_csv": p_cases,
    }
    return {
        "suite_dir": None,
        "cases": cases,
        "rejected_cases": rejected,
        "runs": runs,
        "artifacts": artifacts,
        "metrics": {
            "planned_cases": len(cases),
            "rejected_cases": len(rejected),
            "benchmark_plan_runs": len(runs),
            "candidate_plan_is_authoritative": True,
            "formal_generator_binding": True,
            "contract_only_no_suite": True,
        },
    }


def _resolve_suite_dir(root: Path, model_dir: Path, base: Path, benchmark_set_contract: Mapping[str, Any]) -> Path:
    # v49p: contract-only handoffs must not recreate the old reduced direct
    # ``benchmark_set/suite`` folder.  Real runs either point at ``legacy_suite``
    # explicitly or use an imported existing suite.  Contract-only backend
    # decisions are written at benchmark_set/ level.
    materialized = bool(benchmark_set_contract.get("materialized"))
    scope = str(benchmark_set_contract.get("materialization_scope") or "").lower()
    if (not materialized) and (
        "contract_only" in scope
        or benchmark_set_contract.get("source_of_truth_for_real_runs") == "legacy_benchmarkset_generator"
        or benchmark_set_contract.get("legacy_benchmarkset_required")
        or benchmark_set_contract.get("direct_suite_disabled")
    ):
        return base
    raw = str(benchmark_set_contract.get("legacy_suite_dir") or benchmark_set_contract.get("suite_dir") or benchmark_set_contract.get("generated_suite_dir") or "").strip()
    if raw:
        p = Path(raw)
        if p.is_absolute():
            suite_dir = p
        else:
            suite_dir = root / p
            if not suite_dir.exists():
                suite_dir = model_dir / p
        return suite_dir
    legacy = base / "legacy_suite"
    if legacy.exists():
        return legacy
    # Last-resort compatibility for explicit debug_formal_direct only.
    return base / "suite"


def materialize_backend_artifact_decisions(
    *,
    run_dir: str | Path,
    model_id: str,
    targets: Sequence[str],
    full_baseline_plan: Mapping[str, Any],
    output_contracts: Mapping[str, Any],
    benchmark_set_contract: Mapping[str, Any],
    hailo_full_requested: bool = True,
) -> Dict[str, Any]:
    """Record backend build/reuse decisions and copy reusable full-Hailo HEFs."""

    root = Path(run_dir)
    model_dir = root / "models" / str(model_id)
    base = model_dir / "benchmark_set"
    suite_dir = _resolve_suite_dir(root, model_dir, base, benchmark_set_contract)
    suite_dir.mkdir(parents=True, exist_ok=True)

    source_contracts = [
        dict(item) for item in _as_list(output_contracts.get("contracts"))
        if isinstance(item, Mapping)
    ]
    contracts = [
        _demote_hailo_full_contract_claim(item)
        if (
            _target_backend_label(item.get("backend")).startswith("hailo")
            and str(item.get("variant") or "full").strip().lower() == "full"
        )
        else dict(item)
        for item in source_contracts
    ]
    baselines = _as_list(full_baseline_plan.get("baselines"))
    explicitly_unrequested_hailo_backends = {
        _hailo_contract_backend(row.get("backend"))
        for row in baselines
        if (
            isinstance(row, Mapping)
            and row.get("requested") is False
            and _target_backend_label(row.get("backend")).startswith("hailo")
        )
    }
    for contract in contracts:
        contract_backend = _target_backend_label(contract.get("backend"))
        if (
            contract_backend.startswith("hailo")
            and str(contract.get("variant") or "full").strip().lower()
            == "full"
            and (
                not hailo_full_requested
                or _hailo_contract_backend(contract_backend)
                in explicitly_unrequested_hailo_backends
            )
        ):
            contract.update({
                "requested": False,
                "request_status": "not_requested_by_profile",
            })
    decisions: List[Dict[str, Any]] = []
    copied: List[Dict[str, Any]] = []
    copied_verified: Dict[str, Dict[str, Any]] = {}

    source_contract_by_backend = {
        str(c.get("backend") or ""): dict(c)
        for c in source_contracts
        if isinstance(c, Mapping)
    }
    contract_by_backend = {
        str(c.get("backend") or ""): dict(c)
        for c in contracts
        if isinstance(c, Mapping)
    }
    for raw in baselines:
        if not isinstance(raw, Mapping):
            continue
        backend = str(raw.get("backend") or "").strip() or "unknown"
        backend_contract = source_contract_by_backend.get(backend, {})
        endpoint_mode = str(raw.get("endpoint_mode") or backend_contract.get("endpoint_mode") or "decoded")
        artifact = str(raw.get("artifact_path") or backend_contract.get("artifact_path") or "").strip()
        expected_task = str(
            raw.get("task")
            or backend_contract.get("task")
            or full_baseline_plan.get("task")
            or output_contracts.get("task")
            or ""
        ).strip().lower()
        src = Path(artifact).expanduser() if artifact else None
        is_hailo = "hailo" in backend.lower()
        explicit_requested = raw.get("requested")
        if explicit_requested is not False:
            explicit_requested = backend_contract.get("requested")
        full_requested = (
            False
            if explicit_requested is False
            else (bool(hailo_full_requested) if is_hailo else True)
        )
        dec: Dict[str, Any] = {
            "model_id": model_id,
            "backend": backend,
            "variant": raw.get("variant") or "full",
            "endpoint_mode": endpoint_mode,
            "source_artifact_path": artifact,
            "reuse_policy": raw.get("reuse_policy") or "reuse_existing_prepared_artifact_before_rebuild",
            "decision": "pending_build",
            "copied_to_suite": False,
            # Prepared Hailo metadata is only a claim until its sibling build
            # receipt is revalidated below.
            "compile_ok": (None if is_hailo else raw.get("compile_ok")),
            "error_class": "",
            "requested": full_requested,
        }
        if is_hailo and not full_requested:
            dec.update({
                "decision": "not_requested_by_profile",
                "request_reason": "hailo_build_full_false",
            })
            decisions.append(dec)
            continue
        if src is not None and src.is_file():
            source_receipt_evidence: Dict[str, Any] = {}
            source_hints: List[Path] = []
            compiler_hints: List[Path] = []
            if is_hailo:
                source_hints, compiler_hints = _receipt_path_hints(
                    artifact=src,
                    baseline=raw,
                    full_baseline_plan=full_baseline_plan,
                )
                source_receipt_evidence = _validate_hailo_build_receipt(
                    src,
                    source_onnx_candidates=source_hints,
                    compiler_onnx_candidates=compiler_hints,
                    expected_backend=backend,
                    expected_task=expected_task,
                )
                dec.update({
                    "source_hailo_build_receipt_path": str(
                        source_receipt_evidence.get("receipt_path") or ""
                    ),
                    "source_hailo_build_receipt_file_sha256": str(
                        source_receipt_evidence.get("receipt_file_sha256") or ""
                    ),
                    "source_hailo_build_receipt_identity_sha256": str(
                        source_receipt_evidence.get("receipt_identity_sha256") or ""
                    ),
                    "hailo_build_receipt_validation_errors": list(
                        source_receipt_evidence.get("errors") or []
                    ),
                })
                if source_receipt_evidence.get("valid") is not True:
                    errors = list(source_receipt_evidence.get("errors") or [])
                    missing = "hailo_build_receipt_missing" in errors
                    dec.update({
                        "decision": (
                            "reuse_build_receipt_missing"
                            if missing else "reuse_build_receipt_invalid"
                        ),
                        "error_class": (
                            "missing_hailo_build_receipt"
                            if missing else "invalid_hailo_build_receipt"
                        ),
                        "error_detail": "; ".join(errors),
                        "artifact_copy_verified": False,
                        "compile_ok": False,
                    })
                    decisions.append(dec)
                    continue
                # Pin the verified source generation before reading any more
                # bytes. A concurrent publisher may advance the public alias.
                src = Path(str(source_receipt_evidence["path"]))
                dest = suite_dir / "hailo" / backend / "full" / "compiled.hef"
            else:
                dest = suite_dir / "artifacts" / backend / "full" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination_compiler_path: Path | None = None
                source_sha256 = _file_sha256(src)
                destination_evidence: Dict[str, Any] = {}
                receipt_fields: Dict[str, Any] = {}
                if is_hailo:
                    source_compiler_path = Path(str(
                        source_receipt_evidence.get(
                            "matched_compiler_onnx_path"
                        ) or ""
                    ))
                    if not source_compiler_path.is_file():
                        raise RuntimeError(
                            "verified Hailo compiler ONNX disappeared before copy"
                        )
                    destination_compiler_path = (
                        dest.parent / source_compiler_path.name
                    )
                    if (
                        not destination_compiler_path.is_file()
                        or _file_sha256(destination_compiler_path)
                        != _file_sha256(source_compiler_path)
                    ):
                        shutil.copy2(
                            source_compiler_path,
                            destination_compiler_path,
                        )
                    from ..hailo_backend import (
                        _load_valid_hailo_receipt,
                        _publish_hailo_bundle,
                    )

                    source_receipt = dict(source_receipt_evidence["receipt"])
                    committed_hef = dest.resolve()
                    already_published = bool(
                        dest.is_symlink()
                        and _load_valid_hailo_receipt(committed_hef)
                        == source_receipt
                    )
                    if not already_published:
                        committed_hef = _publish_hailo_bundle(
                            source_hef=src,
                            destination=dest,
                            receipt=source_receipt,
                            source="verified_prepared_full_baseline",
                        )
                    # All three files are visible through one committed
                    # generation. Never copy through a public HEF/receipt link
                    # or overwrite bytes belonging to an earlier generation.
                    destination_receipt_path = committed_hef.parent / _HAILO_HEF_RECEIPT_NAME
                    destination_sha256 = _file_sha256(committed_hef)
                    destination_evidence = _validate_hailo_build_receipt(
                        committed_hef,
                        receipt_path=destination_receipt_path,
                        source_onnx_candidates=source_hints,
                        compiler_onnx_candidates=(
                            [destination_compiler_path]
                            if destination_compiler_path is not None else []
                        ),
                        expected_backend=backend,
                        expected_task=expected_task,
                    )
                    if destination_evidence.get("valid") is not True:
                        raise RuntimeError(
                            "copied Hailo build receipt validation failed: "
                            + "; ".join(destination_evidence.get("errors") or [])
                        )
                    destination_evidence["source"] = (
                        "verified_prepared_artifact_copy_with_canonical_receipt"
                    )
                    receipt_fields = _hailo_receipt_binding_fields(
                        destination_evidence, suite_dir=suite_dir
                    )
                    destination_evidence.update(receipt_fields)
                    copied_verified[_target_backend_label(backend)] = dict(
                        destination_evidence
                    )
                else:
                    if (
                        not dest.is_file()
                        or src.stat().st_size != dest.stat().st_size
                        or _file_sha256(dest) != source_sha256
                    ):
                        shutil.copy2(src, dest)
                    destination_sha256 = _file_sha256(dest)
                if source_sha256 != destination_sha256:
                    raise RuntimeError("copied artifact SHA-256 mismatch")
                dec.update({
                    "decision": "reused_existing_artifact_copied_to_suite",
                    "copied_to_suite": True,
                    "suite_artifact_path": relpath(dest, root),
                    "source_artifact_sha256": source_sha256,
                    "suite_artifact_sha256": destination_sha256,
                    "artifact_size_bytes": int(dest.stat().st_size),
                    "artifact_copy_verified": True,
                    "compile_ok": True,
                    **receipt_fields,
                })
                copied_item = {
                    "backend": backend, "source": str(src),
                    "destination": relpath(dest, root),
                    "suite_destination": dest.resolve().relative_to(
                        suite_dir.resolve()
                    ).as_posix(),
                    "endpoint_mode": endpoint_mode,
                    "source_artifact_sha256": source_sha256,
                    "suite_artifact_sha256": destination_sha256,
                    "artifact_size_bytes": int(dest.stat().st_size),
                    "artifact_copy_verified": True,
                    **receipt_fields,
                }
                copied.append(copied_item)
            except Exception as exc:
                dec.update({
                    "decision": "reuse_copy_failed",
                    "error_class": (
                        "invalid_hailo_build_receipt" if is_hailo
                        else "missing_artifact"
                    ),
                    "error_detail": f"{type(exc).__name__}: {exc}",
                    "artifact_copy_verified": False,
                    "compile_ok": False if is_hailo else dec.get("compile_ok"),
                })
        elif artifact:
            dec.update({"decision": "reuse_artifact_missing_on_disk", "error_class": "missing_artifact"})
        decisions.append(dec)

    case_build_requests: List[Dict[str, Any]] = []
    suite_bench = read_json(suite_dir / "benchmark_set.json", default={}) or {}
    if not isinstance(suite_bench, Mapping):
        suite_bench = {}
    suite_bench = _demote_hailo_suite_full_claims(suite_bench)
    from .hailo_artifact_scope import (
        hailo_scope_backend, selected_hailo_artifact_stages,
    )
    generated_plan = read_json(suite_dir / "benchmark_plan.json", default={}) or {}
    hailo_stage_scope = selected_hailo_artifact_stages(
        generated_plan if isinstance(generated_plan, Mapping) else {},
        _as_list(suite_bench.get("cases")),
    )
    for case in _as_list(suite_bench.get("cases")):
        if not isinstance(case, Mapping):
            continue
        boundary = case.get("split_index") or case.get("boundary")
        cid = str(
            case.get("case_id")
            or case.get("case_dir")
            or case.get("folder")
            or (f"b{int(boundary):03d}" if isinstance(boundary, (int, float)) else "")
            or ""
        )
        availability_all = case.get("hailo_case_variant_availability")
        if not isinstance(availability_all, Mapping):
            availability_all = {}
        for target in targets or []:
            backend = _target_backend_label(target)
            if not backend:
                continue
            backend_key = backend
            availability = availability_all.get(backend_key)
            if not isinstance(availability, Mapping):
                # Common aliases used across BenchmarkSet metadata.
                availability = availability_all.get(str(backend_key).replace("_", "-"))
            if not isinstance(availability, Mapping) and backend_key.startswith("hailo"):
                availability = availability_all.get("hailo") or availability_all.get("hailo8")
            if not isinstance(availability, Mapping):
                availability = {}

            is_hailo = bool("hailo" in backend)
            build_part1 = True
            build_part2 = True
            part2_expected_unsupported = False
            expected_reason = ""
            if is_hailo and availability:
                # The existing BenchmarkSet generator is the source of truth.
                # If it materialized a Hailo Part1/Full raw-head case but no Hailo
                # Part2, this is an intentional host-tail / Hailo->TRT case, not a
                # missing HEF that the formal workflow should queue as a warning.
                build_part1 = bool(availability.get("part1", True))
                build_part2 = bool(availability.get("part2", True))
                if not build_part2 and (availability.get("full") or availability.get("part1")):
                    part2_expected_unsupported = True
                    expected_reason = (
                        "BenchmarkSet generator marked Hailo Part2 unavailable for this "
                        "case; use the ready Hailo Part1/raw-head plus host-tail or "
                        "Hailo->TensorRT path instead of queuing a decoded-tail HEF build."
                    )
            if is_hailo and hailo_stage_scope is not None:
                scope_backend = hailo_scope_backend(backend)
                build_part1 = bool(build_part1 and (scope_backend, cid, "part1") in hailo_stage_scope)
                build_part2 = bool(build_part2 and (scope_backend, cid, "part2") in hailo_stage_scope)
            req = {
                "model_id": model_id,
                "case_id": cid,
                "case_dir": str(case.get("case_dir") or case.get("folder") or cid),
                "backend": backend,
                "variant": "split",
                "split_index": boundary,
                "build_part1": bool(build_part1),
                "build_part2": bool(build_part2),
                "decision": "pending_build_or_runtime_provider",
                "requires_heavy_service": bool(is_hailo),
            }
            if is_hailo and availability:
                req["hailo_case_variant_availability"] = dict(availability)
            if part2_expected_unsupported:
                req["part2_expected_unsupported"] = True
                req["expected_unsupported_reason"] = expected_reason
            case_build_requests.append(req)

    if copied:
        hailo = dict(suite_bench.get("hailo") or {}) if isinstance(suite_bench, Mapping) else {}
        hefs = dict(hailo.get("hefs") or {}) if isinstance(hailo.get("hefs"), Mapping) else {}
        for item in copied:
            backend = str(item.get("backend") or "")
            if "hailo" not in backend.lower():
                continue
            hw = backend or "hailo8"
            hw_meta = dict(hefs.get(hw) or {})
            hw_meta["full"] = str(
                item.get("suite_destination") or item.get("destination") or ""
            )
            if not str(hw_meta.get("full_endpoint_mode") or "").strip():
                hw_meta["full_endpoint_mode"] = str(
                    item.get("endpoint_mode") or ""
                )
            receipt_fields = {
                key: item.get(key)
                for key in (
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
                    "preprocessing_contract",
                    "preprocessing_contract_sha256",
                    "source_onnx_sha256",
                    "compiler_onnx_sha256",
                    "compiler_onnx_filename",
                    "compiler_onnx_path",
                )
                if item.get(key) not in (None, "", {})
            }
            hw_meta["full_prepared_baseline"] = True
            full_build = dict(hw_meta.get("full_build") or {})
            full_build.update({
                "ok": True,
                "artifact_hash": str(item.get("suite_artifact_sha256") or ""),
                "receipt_validated": True,
                **receipt_fields,
            })
            hw_meta["full_build"] = full_build
            hw_meta["full_build_receipt"] = dict(receipt_fields)
            contract = dict(
                contract_by_backend.get(backend)
                or contract_by_backend.get("hailo8")
                or {}
            )
            if contract and not isinstance(
                hw_meta.get("full_output_contract"), Mapping
            ):
                contract.update(receipt_fields)
                hw_meta["full_output_contract"] = contract
            hefs[hw] = hw_meta
        hailo["hefs"] = hefs
        suite_bench["hailo"] = hailo
        mat = dict(suite_bench.get("materialization") or {})
        mat["backend_artifacts_materialized"] = True
        mat["backend_artifact_scope"] = "reused_full_baseline_artifacts"
        suite_bench["materialization"] = mat
        write_json(suite_dir / "benchmark_set.json", suite_bench)

    task = str(
        full_baseline_plan.get("task")
        or output_contracts.get("task")
        or ""
    ).strip().lower()
    promotions = (
        _promote_verified_hailo_full_contracts(
            suite_dir=suite_dir,
            model_id=model_id,
            task=task,
            suite_bench=suite_bench,
            contracts=contracts,
            copied_verified=copied_verified,
        )
        if hailo_full_requested else []
    )
    if promotions:
        hailo = dict(suite_bench.get("hailo") or {})
        hefs = dict(hailo.get("hefs") or {})
        for contract in contracts:
            if contract.get("contract_status") != "recorded":
                continue
            contract_backend = _target_backend_label(contract.get("backend"))
            if not contract_backend.startswith("hailo"):
                continue
            matching_aliases = [
                alias for alias in hefs
                if _hailo_hw_matches(contract_backend, alias)
            ]
            for alias in matching_aliases:
                hw_meta = dict(hefs.get(alias) or {})
                hw_meta["full_output_contract"] = dict(contract)
                hw_meta["full_receipt_attested"] = True
                full_build = dict(hw_meta.get("full_build") or {})
                receipt_fields = {
                    key: contract.get(key)
                    for key in (
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
                        "preprocessing_contract",
                        "preprocessing_contract_sha256",
                        "source_onnx_sha256",
                        "compiler_onnx_sha256",
                        "compiler_onnx_filename",
                        "compiler_onnx_path",
                    )
                    if contract.get(key) not in (None, "", {})
                }
                full_build.update({
                    "ok": True,
                    "status": "receipt_attested",
                    "receipt_validated": True,
                    "artifact_hash": str(
                        contract.get("recorded_artifact_sha256") or ""
                    ),
                    **receipt_fields,
                })
                hw_meta["full_build"] = full_build
                hw_meta["full_build_receipt"] = dict(receipt_fields)
                hefs[alias] = hw_meta
        hailo["hefs"] = hefs
        suite_bench["hailo"] = hailo
    if suite_bench:
        # Persist demotion even when no artifact was promotable. Otherwise a
        # stale ``ok``/``recorded`` claim would survive a failed receipt check.
        write_json(suite_dir / "benchmark_set.json", suite_bench)

    effective_hailo_full_requested = bool(promotions) or any(
        "hailo" in str(row.get("backend") or "").lower()
        and row.get("requested") is not False
        for row in decisions
    )
    p_dec = write_json(base / "backend_artifact_decisions.json", {
        "schema": "onnx-splitpoint/backend-artifact-decisions",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, root),
        "hailo_full_requested": effective_hailo_full_requested,
        "baseline_decisions": decisions,
        "case_build_requests": case_build_requests,
        "copied_artifacts": copied,
        "recorded_hailo_full_contracts": promotions,
        "status": "reuse_recorded" if copied else "pending_builds",
    })
    p_reuse = write_json(base / "artifact_reuse_manifest.json", {
        "schema": "onnx-splitpoint/artifact-reuse-manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "copied_artifact_count": len(copied),
        "copied_artifacts": copied,
        "recorded_hailo_full_contracts": promotions,
        "baseline_decision_count": len(decisions),
    })
    p_contracts = write_json(suite_dir / "output_contracts.json", {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": model_id,
        "task": task,
        "contracts": contracts,
    })
    # Always replace the resume source as well. A failed revalidation must not
    # leave an older recorded/verified contract available to the next resume.
    p_formal_contracts = write_json(
        model_dir / "full_baselines" / "output_contracts.json",
        {
            "schema": "onnx-splitpoint/output-contracts",
            "schema_version": 1,
            "model_id": model_id,
            "task": task,
            "contracts": contracts,
        },
    )
    p_request = write_json(base / "build_backend_artifacts_request.json", {
        "schema": "onnx-splitpoint/backend-build-request",
        "schema_version": 3,
        "created_at": now_iso(),
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, root),
        "hailo_full_requested": effective_hailo_full_requested,
        "target_cases": _as_list(suite_bench.get("cases")),
        "full_baselines": baselines,
        "case_build_requests_path": "backend_artifact_decisions.json",
        "status": "reuse_recorded" if copied else "pending_service_execution",
        "reuse_policy": "reuse_existing_prepared_or_compiled_artifacts_before_rebuild",
    })
    return {
        "artifacts": {
            "build_backend_artifacts_request_json": p_request,
            "backend_artifact_decisions_json": p_dec,
            "artifact_reuse_manifest_json": p_reuse,
            "suite_output_contracts_json": p_contracts,
            "formal_output_contracts_json": p_formal_contracts,
        },
        "metrics": {
            "baseline_decisions": len(decisions),
            "copied_artifacts": len(copied),
            "recorded_hailo_full_contracts": len(promotions),
            "case_build_requests": len(case_build_requests),
            "hailo_full_not_requested": sum(
                1 for row in decisions
                if row.get("requested") is False
            ),
        },
        "status": "ok" if copied or decisions or case_build_requests else "partial",
        "message": "Backend build/reuse decisions recorded; reusable full-baseline artifacts copied when present.",
    }


def benchmark_suite_result_sources(run_dir: str | Path, model_id: str) -> List[str]:
    root = Path(run_dir)
    model_dir = root / "models" / str(model_id)
    base = model_dir / "benchmark_set"
    # v49p: do not scan stale v49c-v49m direct-suite folders by default.
    # Authoritative result sources are the per-model result directory and suite
    # directories explicitly recorded by the legacy/imported BenchmarkSet path.
    sources: List[Path] = [model_dir / "benchmark_results", base]
    binding = read_json(base / "generator_binding.json", default={}) or {}
    contract = read_json(base / "benchmark_set.json", default={}) or {}
    for raw in (
        binding.get("legacy_suite_dir"),
        binding.get("suite_dir"),
        contract.get("legacy_suite_dir"),
        contract.get("suite_dir"),
        contract.get("generated_suite_dir"),
    ):
        suite_dir = str(raw or "").strip()
        if not suite_dir:
            continue
        p = Path(suite_dir)
        if not p.is_absolute():
            candidate = root / p
            if candidate.exists():
                p = candidate
            else:
                p = model_dir / p
        sources.append(p)
        sources.append(p / "results")
    out: List[str] = []
    seen: set[str] = set()
    for p in sources:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _benchmark_suite_stub_text() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
bench = ROOT / 'benchmark_set.json'
print('[evaluation-workflow] benchmark suite binding:', bench)
print('[evaluation-workflow] This v49c suite is the formal handoff from final_candidate_plan.json.')
print('[evaluation-workflow] It does not fabricate measurements. Materialize split graphs / run the existing benchmark executor,')
print('[evaluation-workflow] then place benchmark_results_*.json/csv in this folder or results/.')
if bench.is_file():
    payload = json.loads(bench.read_text(encoding='utf-8'))
    print('[evaluation-workflow] cases:', len(payload.get('cases') or []))
    print('[evaluation-workflow] candidate_plan:', payload.get('candidate_plan_path', ''))
"""


def _suite_readme_text() -> str:
    return """# Formal Evaluation Workflow benchmark suite

This folder was generated by v49c from `analysis/final_candidate_plan.json`.

Important points:

- `benchmark_set.json` and `benchmark_plan.json` are the formal handoff to the benchmark executor.
- Per-case folders contain `case_manifest.json` and `split_request.json`.
- Runtime measurements are not fabricated. After local/remote execution, put `benchmark_results_*.json` or `.csv` here or in `results/`, then resume the Evaluation Workflow to normalize and aggregate them.
- Existing prepared full-Hailo raw-head artifacts are copied into `hailo/<hw>/full/compiled.hef` when discoverable; the output contract is recorded in `output_contracts.json`.
"""

# ---------------------------------------------------------------------------
# v49c runner compatibility wrappers
# ---------------------------------------------------------------------------

def materialize_benchmark_binding(
    *,
    run_root: str | Path,
    model_id: str,
    model_dir: str | Path,
    row: Mapping[str, Any],
    profile_id: str,
    profile_payload: Mapping[str, Any],
    targets: Sequence[str],
    skip_benchmarks: bool,
    no_remote: bool,
    dry_run: bool,
) -> Tuple[Mapping[str, Path], Mapping[str, Any], str, str]:
    """Compatibility entry point used by the v49c runner.

    It writes the same formal benchmark-generator binding as
    ``materialize_benchmark_generator_binding`` and returns the four-tuple stage
    functions expect: artifacts, metrics, message, status.
    """

    mdir = Path(model_dir)
    prediction = read_json(mdir / "analysis" / "prediction.json", default={}) or {}
    candidate_plan = read_json(mdir / "analysis" / "final_candidate_plan.json", default={}) or {}
    full_plan = read_json(mdir / "full_baselines" / "full_baseline_plan.json", default={}) or {}
    output_contracts = read_json(mdir / "full_baselines" / "output_contracts.json", default={}) or {}
    result = materialize_benchmark_generator_binding(
        run_dir=run_root,
        model_id=model_id,
        model_path=str(row.get("resolved_path") or row.get("onnx") or row.get("path") or ""),
        profile_id=profile_id,
        profile_payload=profile_payload,
        targets=targets,
        prediction=prediction,
        final_candidate_plan=candidate_plan,
        full_baseline_plan=full_plan,
        output_contracts=output_contracts,
        skip_benchmarks=skip_benchmarks,
        no_remote=no_remote,
    )
    metrics = dict(result.get("metrics") or {})
    metrics["dry_run"] = bool(dry_run)
    message = "Benchmark generator binding written from authoritative final_candidate_plan.json."
    status = "ok" if metrics.get("planned_cases") else "partial"
    if not skip_benchmarks and not dry_run:
        status = "partial"
        message += " Heavy split/HEF/runtime execution is not fabricated and waits for the generator/executor service."
    return dict(result.get("artifacts") or {}), metrics, message, status


def planned_result_rows_from_plan(*, model_id: str, benchmark_plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand a benchmark plan into expected normalized-result rows."""

    rows: List[Dict[str, Any]] = []
    planned = benchmark_plan.get("planned_runs") if isinstance(benchmark_plan, Mapping) else None
    if isinstance(planned, list) and planned:
        for run in planned:
            if not isinstance(run, Mapping):
                continue
            rows.append({
                "schema": "onnx-splitpoint/planned-benchmark-result",
                "schema_version": 1,
                "model_id": model_id,
                "case_id": str(run.get("case_id") or ""),
                "backend": str(run.get("backend") or "unknown"),
                "variant": str(run.get("variant") or "split"),
                "part1_latency_ms": None,
                "part2_latency_ms": None,
                "transfer_latency_ms": None,
                "total_latency_ms": None,
                "compile_ok": None,
                "runtime_ok": None,
                "validation_ok": None,
                "error_class": "missing_artifact",
                "endpoint_mode": str(run.get("endpoint_mode") or "decoded"),
                "status": "pending_benchmark_execution",
                "prediction_rank": run.get("prediction_rank", ""),
                "split_index": run.get("split_index", ""),
                "run_id": run.get("run_id", ""),
            })
        return rows

    runs = _as_list(benchmark_plan.get("runs")) if isinstance(benchmark_plan, Mapping) else []
    cases = _as_list(benchmark_plan.get("cases")) if isinstance(benchmark_plan, Mapping) else []
    if not runs and cases:
        runs = [{"id": "cpu_ort_split", "backend": "cpu_ort", "variant": "split"}]
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        variant = str(run.get("variant") or "split").lower()
        backend = str(run.get("backend") or run.get("id") or "unknown")
        if variant == "full":
            rows.append({
                "schema": "onnx-splitpoint/planned-benchmark-result",
                "schema_version": 1,
                "model_id": model_id,
                "case_id": "full",
                "backend": backend,
                "variant": "full",
                "part1_latency_ms": None,
                "part2_latency_ms": None,
                "transfer_latency_ms": None,
                "total_latency_ms": None,
                "compile_ok": None,
                "runtime_ok": None,
                "validation_ok": None,
                "error_class": "missing_artifact",
                "endpoint_mode": str(run.get("endpoint_mode") or "decoded"),
                "status": "pending_benchmark_execution",
                "prediction_rank": "",
                "split_index": "",
                "run_id": run.get("id", ""),
            })
            continue
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            rows.append({
                "schema": "onnx-splitpoint/planned-benchmark-result",
                "schema_version": 1,
                "model_id": model_id,
                "case_id": str(case.get("case_id") or _case_id_for(case, 0)),
                "backend": backend,
                "variant": "split",
                "part1_latency_ms": None,
                "part2_latency_ms": None,
                "transfer_latency_ms": None,
                "total_latency_ms": None,
                "compile_ok": None,
                "runtime_ok": None,
                "validation_ok": None,
                "error_class": "missing_artifact",
                "endpoint_mode": str(case.get("endpoint_mode") or run.get("endpoint_mode") or "decoded"),
                "status": "pending_benchmark_execution",
                "prediction_rank": case.get("prediction_rank", ""),
                "split_index": case.get("split_index", case.get("boundary", "")),
                "run_id": run.get("id", ""),
            })
    return rows
