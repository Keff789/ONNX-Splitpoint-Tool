#!/usr/bin/env python3
"""Read-only terminal verifier for Native Full performance/quality rows.

The verifier never discovers success from logs.  It derives the expected Full
matrix from ``reports/native_expected_matrix.json`` and checks every selected
line against the final performance summary and its exact validation join.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_SCHEMA = "onnx-splitpoint/native-expected-matrix"
PERFORMANCE_SCHEMA = "onnx-splitpoint/native-producer-combined-summary"
VALIDATION_SCHEMA = "onnx-splitpoint/native-producer-validation-summary"
OUTPUT_SCHEMA = "onnx-splitpoint/native-full-row-verification"

LINE_IDENTITY_NAMES = (
    "backend", "model", "case", "setup_id", "comparison_backend",
)
VENDOR_FULL_MODELS = ("resnet50", "yolo26s", "yolov7_paper")
VENDOR_FULL_SETUPS = (
    ("native_full_hailo8", "orin_nx_hailo8_01", "hailo8"),
    ("native_full_hailo10h", "orin_nx_hailo10_01", "hailo10h"),
    ("native_full_deepx", "orin_nx_deepx_m1_01", "deepx"),
)
EXPECTED_VENDOR_FULL_IDENTITIES = frozenset(
    (backend, model, "full", setup_id, comparison_backend)
    for backend, setup_id, comparison_backend in VENDOR_FULL_SETUPS
    for model in VENDOR_FULL_MODELS
)
EXPECTED_ALL_FULL_IDENTITIES = EXPECTED_VENDOR_FULL_IDENTITIES | frozenset(
    (
        "native_full_tensorrt", model, "full", setup_id,
        comparison_backend,
    )
    for _backend, setup_id, comparison_backend in VENDOR_FULL_SETUPS
    for model in VENDOR_FULL_MODELS
)
QUALITY_IDENTITY_NAMES = (
    "backend", "model", "case", "runtime_precision", "setup_id",
    "comparison_backend", "output_endpoint_id",
)
DIAGNOSTIC_FIELDS = (
    "failure_reason", "status_detail", "error", "returncode", "timed_out",
    "stderr_tail", "stdout_tail", "report",
)
MAX_DIAGNOSTIC_TEXT = 1200


class VerificationInputError(RuntimeError):
    """The requested verification cannot be performed from authoritative data."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationInputError(f"{label}_missing:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationInputError(
            f"{label}_unreadable:{path}:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VerificationInputError(f"{label}_not_object:{path}")
    return payload


def _require_summary(
    payload: Mapping[str, Any], *, schema: str, label: str,
) -> list[dict[str, Any]]:
    if payload.get("schema") != schema:
        raise VerificationInputError(
            f"{label}_schema_invalid:{payload.get('schema')!r}"
        )
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise VerificationInputError(
            f"{label}_schema_version_invalid:{version!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise VerificationInputError(f"{label}_rows_invalid")
    declared = payload.get("row_count")
    if declared is not None and declared != len(rows):
        raise VerificationInputError(
            f"{label}_row_count_mismatch:{declared!r}!={len(rows)}"
        )
    return [dict(row) for row in rows]


def _line_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("backend") or "").strip().lower(),
        str(row.get("model") or "").strip().lower(),
        str(row.get("case") or row.get("case_id") or "").strip().lower(),
        str(row.get("setup_id") or "").strip().lower(),
        str(row.get("comparison_backend") or "").strip().lower(),
    )


def _identity_dict(identity: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(LINE_IDENTITY_NAMES, identity))


def _expected_full_rows(
    matrix: Mapping[str, Any], *, scope: str,
) -> list[dict[str, Any]]:
    if matrix.get("schema") != EXPECTED_SCHEMA:
        raise VerificationInputError(
            f"expected_matrix_schema_invalid:{matrix.get('schema')!r}"
        )
    version = matrix.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise VerificationInputError(
            f"expected_matrix_schema_version_invalid:{version!r}"
        )

    raw_expected = matrix.get("expected_rows")
    if isinstance(raw_expected, list):
        raw_rows = raw_expected
    else:
        present = matrix.get("present_expected_rows")
        missing = matrix.get("missing_expected_rows")
        if not isinstance(present, list) or not isinstance(missing, list):
            raise VerificationInputError("expected_matrix_rows_missing")
        raw_rows = [*present, *missing]
    if any(not isinstance(row, dict) for row in raw_rows):
        raise VerificationInputError("expected_matrix_rows_invalid")

    all_identities: set[tuple[str, ...]] = set()
    for row in raw_rows:
        identity = _line_identity(row)
        if not all(identity):
            raise VerificationInputError(
                f"expected_matrix_identity_incomplete:{identity!r}"
            )
        if identity in all_identities:
            raise VerificationInputError(
                f"expected_matrix_identity_duplicate:{identity!r}"
            )
        all_identities.add(identity)
    declared_count = matrix.get("expected_row_count")
    if declared_count is not None and declared_count != len(all_identities):
        raise VerificationInputError(
            "expected_matrix_count_mismatch:"
            f"{declared_count!r}!={len(all_identities)}"
        )

    selected: list[dict[str, Any]] = []
    for row in raw_rows:
        backend, _model, case, _setup, _comparison = _line_identity(row)
        if case != "full" or not backend.startswith("native_full_"):
            continue
        if scope == "vendor" and backend == "native_full_tensorrt":
            continue
        selected.append(dict(row))
    if not selected:
        raise VerificationInputError(
            f"expected_matrix_has_no_{scope}_native_full_rows"
        )
    selected_identities = {_line_identity(row) for row in selected}
    required_identities = (
        EXPECTED_VENDOR_FULL_IDENTITIES
        if scope == "vendor"
        else EXPECTED_ALL_FULL_IDENTITIES
    )
    if selected_identities != required_identities:
        missing = sorted(required_identities - selected_identities)
        unexpected = sorted(selected_identities - required_identities)
        raise VerificationInputError(
            f"expected_{scope}_full_matrix_not_exact:"
            f"required={len(required_identities)}:"
            f"actual={len(selected_identities)}:"
            f"missing={missing!r}:unexpected={unexpected!r}"
        )
    return selected


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _returncode_ok(value: Any) -> bool:
    parsed = _strict_int(value)
    return parsed == 0


def _timed_out_ok(value: Any) -> bool:
    return value is False


def _first_present(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _short(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) <= MAX_DIAGNOSTIC_TEXT:
        return normalized
    return normalized[-MAX_DIAGNOSTIC_TEXT:]


def _diagnostic_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: _short(source.get(field))
        for field in DIAGNOSTIC_FIELDS
        if source.get(field) not in (None, "", False)
    }


def _repetition_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    for field in ("performance_repetitions", "repetition_records"):
        records = row.get(field)
        if isinstance(records, list) and records:
            return [dict(record) for record in records if isinstance(record, Mapping)]
    return []


def _first_failed_record(row: Mapping[str, Any]) -> dict[str, Any]:
    for record in _repetition_records(row):
        if (
            record.get("ok") is not True
            or str(record.get("status") or "").strip().lower() != "ok"
            or not _returncode_ok(record.get("returncode"))
            or not _timed_out_ok(record.get("timed_out"))
        ):
            projected = _diagnostic_projection(record)
            projected["repetition_index"] = record.get("repetition_index")
            return projected
    return {}


def _first_failed_step(row: Mapping[str, Any]) -> dict[str, Any]:
    steps = row.get("steps")
    if not isinstance(steps, list):
        return {}
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        rc = _first_present(step, ("rc", "returncode"))
        timed_out_present = "timed_out" in step
        if (
            (rc not in (None, "") and not _returncode_ok(rc))
            or (timed_out_present and not _timed_out_ok(step.get("timed_out")))
            or step.get("error") not in (None, "")
        ):
            projected = _diagnostic_projection(step)
            if rc not in (None, ""):
                projected["returncode"] = rc
            return projected
    return {}


def _load_raw_full_rows(
    run_dir: Path, performance_rows: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    candidates: set[Path] = set()
    for row in performance_rows:
        source_root = str(row.get("source_root") or "").strip()
        if source_root:
            path = Path(source_root).expanduser()
            if path.is_dir():
                candidates.add(
                    path / "analysis_tables" / "native_full_baseline_eval.json"
                )
    for pattern in (
        "analysis_tables/native_full_baseline_eval.json",
        "native_producers/**/analysis_tables/native_full_baseline_eval.json",
        "reports/native_producer_variants/**/analysis_tables/native_full_baseline_eval.json",
    ):
        candidates.update(run_dir.glob(pattern))

    index: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(candidates):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            identity = _line_identity(row)
            if all(identity) and identity[2] == "full":
                projected = dict(row)
                projected["_source_file"] = str(path)
                index[identity].append(projected)
    return index


def _child_failure(
    row: Mapping[str, Any], raw_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _diagnostic_projection(row)
    repetition = _first_failed_record(row)
    if repetition:
        result["failed_repetition"] = repetition
    step = _first_failed_step(row)
    if step:
        result["failed_step"] = step
    for raw in raw_candidates:
        raw_repetition = _first_failed_record(raw)
        raw_step = _first_failed_step(raw)
        raw_projection = _diagnostic_projection(raw)
        if raw_repetition or raw_step or raw_projection:
            result["raw_source"] = {
                "path": raw.get("_source_file"),
                **raw_projection,
            }
            if raw_repetition:
                result["raw_source"]["failed_repetition"] = raw_repetition
            if raw_step:
                result["raw_source"]["failed_step"] = raw_step
            break
    return result


def _quality_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Recompute all seven join axes from raw row evidence.

    ``quality_identity`` is deliberately not an input here.  It is a useful
    final-report projection, but allowing it to define its own identity would
    make the terminal verifier compare two copies of the same assertion.
    """
    identity = _line_identity(row)
    return (
        identity[0], identity[1], identity[2],
        _runtime_precision_identity(row), identity[3], identity[4],
        _explicit_output_endpoint(row),
    )


def _strict_sha256_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def _is_full_row(row: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("execution_mode") or "") == "native_full_baseline"
        or str(row.get("backend") or "").startswith("native_full_")
        or str(row.get("case") or "").strip().lower() == "full"
    )


def _runtime_precision_identity(row: Mapping[str, Any]) -> str:
    """Mirror the final reporter's fail-closed runtime-precision identity."""
    backend = str(row.get("backend") or "").strip().lower()
    scalar = {"fp16", "fp32", "int8"}

    def valid(value: Any) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "")
        if normalized in scalar or re.fullmatch(
            r"(?:uint8_cast|uint8_dequant|float32_layout)_(?:fp16|fp32|int8)",
            normalized,
        ):
            return normalized
        artifact = re.fullmatch(
            r"(deepx_dxnn_sha256|hailo_hef_sha256):([0-9a-f]{64})",
            normalized,
        )
        if artifact:
            prefix = artifact.group(1)
            if prefix == "deepx_dxnn_sha256" and "deepx" in backend:
                return normalized
            if prefix == "hailo_hef_sha256" and "hailo" in backend:
                return normalized
        return ""

    explicit_value = row.get("runtime_precision_identity")
    explicit = valid(explicit_value)
    if explicit_value not in (None, "") and not explicit:
        return ""
    if _is_full_row(row):
        primary_value = (
            row.get("full_runtime_precision") or row.get("execution_precision")
        )
    else:
        primary_value = row.get("execution_precision") or row.get("precision")
    primary = valid(primary_value)
    if primary_value not in (None, ""):
        return primary
    if explicit:
        return explicit

    command = row.get("full_command_contract")
    command = command if isinstance(command, Mapping) else {}
    artifacts = command.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    artifact_name = ""
    token_prefix = ""
    if backend == "native_full_deepx":
        artifact_name, token_prefix = "dxnn", "deepx_dxnn_sha256:"
    elif backend in {"native_full_hailo8", "native_full_hailo10h"}:
        artifact_name, token_prefix = "hef", "hailo_hef_sha256:"
    artifact = artifacts.get(artifact_name)
    artifact = artifact if isinstance(artifact, Mapping) else {}
    digest = _strict_sha256_token(artifact.get("sha256"))
    return token_prefix + digest if token_prefix and digest else ""


def _explicit_output_endpoint(row: Mapping[str, Any]) -> str:
    """Mirror the final reporter's complete endpoint-attestation gate."""
    task = str(row.get("task") or "").strip().lower()
    stage = str(row.get("stage") or "").strip().lower()
    digest = _strict_sha256_token(row.get("endpoint_contract_hash"))
    attestation = row.get("output_endpoint_attestation")
    if (
        task not in {"classification", "detection"}
        or not stage
        or row.get("endpoint_contract_complete") is not True
        or not digest
        or not isinstance(attestation, Mapping)
    ):
        return ""
    if (
        attestation.get("attested") is not True
        or str(attestation.get("status") or "").strip().lower() != "passed"
        or _strict_sha256_token(attestation.get("endpoint_contract_hash"))
        != digest
    ):
        return ""
    allowed_stages = {
        "classification": {
            "classification_logits", "classification_probabilities",
        },
        "detection": {"raw_head", "decoded_pre_nms", "decoded_nms"},
    }
    if stage not in allowed_stages[task]:
        return ""
    if (
        str(attestation.get("stage") or "").strip().lower() != stage
        or str(attestation.get("endpoint") or "").strip().lower() != stage
    ):
        return ""
    attested_task = str(attestation.get("task") or "").strip().lower()
    if attested_task and attested_task != task:
        return ""
    contract_family = str(row.get("contract_family") or "").strip().lower()
    if contract_family and contract_family != stage:
        return ""
    expected = f"{task}:{stage}:{digest}"
    for explicit in (
        row.get("output_endpoint_id"), attestation.get("output_endpoint_id"),
    ):
        if (
            explicit not in (None, "")
            and str(explicit).strip().lower() != expected
        ):
            return ""
    return expected


def _stored_quality_identity(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    stored = row.get("quality_identity")
    if not isinstance(stored, Mapping):
        return None
    return tuple(
        str(stored.get(name) or "").strip().lower()
        for name in QUALITY_IDENTITY_NAMES
    )


def _nearest_quality_candidates(
    performance: Mapping[str, Any], validation_rows: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    # Recompute diagnostics from the validation rows as well.  Stored nearest
    # candidates are presentation data and must not become verifier inputs.
    performance_identity = _quality_identity(performance)
    base = performance_identity[:3] + performance_identity[4:6]
    candidates: list[dict[str, Any]] = []
    for row_index, quality_row in enumerate(validation_rows):
        candidate_identity = _quality_identity(quality_row)
        candidate_base = candidate_identity[:3] + candidate_identity[4:6]
        if candidate_base != base or candidate_identity == performance_identity:
            continue
        differing = {
            name: {
                "performance": performance_identity[index],
                "quality": candidate_identity[index],
            }
            for index, name in enumerate(QUALITY_IDENTITY_NAMES)
            if performance_identity[index] != candidate_identity[index]
        }
        candidates.append({
            "quality_row_index": row_index,
            "quality_identity": dict(zip(
                QUALITY_IDENTITY_NAMES, candidate_identity,
            )),
            "differing_axes": differing,
        })
    return len(candidates), candidates[:8]


def _runtime_check(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    status = str(row.get("status") or "").strip().lower()
    fps = _positive_number(row.get("fps_makespan"))
    if row.get("ok") is not True:
        issues.append("runtime_ok_not_true")
    if status != "ok":
        issues.append(f"runtime_status_not_ok:{status or 'missing'}")
    if fps is None:
        issues.append("runtime_fps_not_positive_finite")
    if not _returncode_ok(row.get("returncode")):
        issues.append(f"runtime_returncode_nonzero_or_invalid:{row.get('returncode')!r}")
    if not _timed_out_ok(row.get("timed_out")):
        issues.append(f"runtime_timed_out_or_invalid:{row.get('timed_out')!r}")
    return {
        "ok": not issues,
        "status": status,
        "fps_makespan": fps,
        "returncode": row.get("returncode"),
        "timed_out": row.get("timed_out"),
    }, issues


def _repetition_check(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    requested = _strict_int(_first_present(
        row, ("repetition_count_requested", "repetitions_requested"),
    ))
    attempted = _strict_int(_first_present(
        row, ("repetition_count_attempted", "repetitions_attempted"),
    ))
    valid = _strict_int(_first_present(
        row, ("repetition_count_valid", "repetitions_completed"),
    ))
    status = str(row.get("repetition_status") or "").strip().lower()
    records = _repetition_records(row)
    runtime_scope = str(row.get("repetition_runtime_scope") or "").strip()
    issues: list[str] = []
    if requested is None or requested < 1:
        issues.append("repetition_requested_invalid")
    if requested is None or attempted != requested:
        issues.append("repetition_attempted_mismatch")
    if requested is None or valid != requested:
        issues.append("repetition_valid_mismatch")
    if status != "complete":
        issues.append(f"repetition_status_not_complete:{status or 'missing'}")
    if row.get("repetition_independence_verified") is not True:
        issues.append("repetition_independence_not_verified")
    if runtime_scope != "fresh_process_per_repetition":
        issues.append(
            f"repetition_runtime_scope_not_fresh_process:{runtime_scope or 'missing'}"
        )
    if requested is None or len(records) != requested:
        issues.append("repetition_record_count_mismatch")
    runtime_ids: list[str] = []
    repetition_indexes: list[int | None] = []
    completed_values: list[int] = []
    for index, record in enumerate(records, start=1):
        repetition_index = _strict_int(record.get("repetition_index"))
        repetition_indexes.append(repetition_index)
        runtime_id = str(record.get("runtime_instance_id") or "").strip()
        runtime_ids.append(runtime_id)
        if record.get("ok") is not True:
            issues.append(f"repetition_{index}_ok_not_true")
        if str(record.get("status") or "").strip().lower() != "ok":
            issues.append(f"repetition_{index}_status_not_ok")
        if not _returncode_ok(record.get("returncode")):
            issues.append(f"repetition_{index}_returncode_nonzero_or_invalid")
        if not _timed_out_ok(record.get("timed_out")):
            issues.append(f"repetition_{index}_timed_out_or_invalid")
        if _positive_number(record.get("fps_makespan")) is None:
            issues.append(f"repetition_{index}_fps_not_positive_finite")
        if not runtime_id:
            issues.append(f"repetition_{index}_runtime_instance_id_missing")
        completed = _strict_int(_first_present(
            record, ("completed_work_units", "completed_frames", "frames"),
        ))
        if completed is None or completed < 1:
            issues.append(f"repetition_{index}_completed_work_units_invalid")
        else:
            completed_values.append(completed)
    if requested is not None and repetition_indexes != list(range(1, requested + 1)):
        issues.append("repetition_indexes_not_exact_sequence")
    if runtime_ids and len(set(runtime_ids)) != len(runtime_ids):
        issues.append("repetition_runtime_instance_ids_not_unique")
    if completed_values and len(set(completed_values)) != 1:
        issues.append("repetition_completed_work_units_mismatch")
    declared_ids = row.get("repetition_runtime_instance_ids")
    if isinstance(declared_ids, list) and [str(value) for value in declared_ids] != runtime_ids:
        issues.append("repetition_runtime_instance_id_projection_mismatch")
    return {
        "ok": not issues,
        "requested": requested,
        "attempted": attempted,
        "valid": valid,
        "status": status,
        "independence_verified": row.get("repetition_independence_verified"),
        "runtime_scope": runtime_scope,
        "runtime_instance_ids": runtime_ids,
        "record_count": len(records),
    }, issues


def _quality_check(
    row: Mapping[str, Any], validation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], str | None]:
    match_status = str(row.get("quality_match_status") or "").strip()
    match_count = _strict_int(row.get("quality_match_count"))
    row_index = _strict_int(row.get("quality_row_index"))
    binding = row.get("precision_quality_binding_verified")
    nearest_count, nearest = _nearest_quality_candidates(row, validation_rows)
    issues: list[str] = []
    failure_class: str | None = None
    performance_identity = _quality_identity(row)
    performance_stored_identity = _stored_quality_identity(row)
    if not all(performance_identity):
        issues.append("performance_quality_identity_incomplete")
        failure_class = "QUALITY_JOIN_FAIL"
    if (
        performance_stored_identity is not None
        and performance_stored_identity != performance_identity
    ):
        issues.append("performance_quality_identity_projection_drift")
        failure_class = "QUALITY_JOIN_FAIL"
    if match_count != 1:
        issues.append(f"quality_exact_match_count_not_one:{match_count!r}")
        failure_class = "QUALITY_JOIN_FAIL"
    if row_index is None or not 0 <= row_index < len(validation_rows):
        issues.append(f"quality_row_index_invalid:{row_index!r}")
        failure_class = "QUALITY_JOIN_FAIL"
    else:
        selected_quality_row = validation_rows[row_index]
        selected_identity = _quality_identity(selected_quality_row)
        selected_stored_identity = _stored_quality_identity(selected_quality_row)
        if _line_identity(selected_quality_row) != _line_identity(row):
            issues.append("quality_row_stable_identity_mismatch")
            failure_class = "QUALITY_JOIN_FAIL"
        if not all(selected_identity):
            issues.append("quality_row_identity_incomplete")
            failure_class = "QUALITY_JOIN_FAIL"
        if (
            selected_stored_identity is not None
            and selected_stored_identity != selected_identity
        ):
            issues.append("quality_row_identity_projection_drift")
            failure_class = "QUALITY_JOIN_FAIL"
        if selected_identity != performance_identity:
            issues.append("quality_row_exact_identity_mismatch")
            failure_class = "QUALITY_JOIN_FAIL"
    if match_status != "exact_identity_match":
        issues.append(f"quality_match_status_not_exact:{match_status or 'missing'}")
        failure_class = failure_class or "QUALITY_BINDING_FAIL"
    if binding is not True:
        issues.append("precision_quality_binding_not_verified")
        failure_class = failure_class or "QUALITY_BINDING_FAIL"
    return {
        "ok": not issues,
        "match_status": match_status,
        "match_count": match_count,
        "quality_row_index": row_index,
        "binding_verified": binding,
        "nearest_candidate_count": nearest_count,
        "nearest_candidates": nearest,
        "task_quality_observation_valid": row.get(
            "task_quality_observation_valid"
        ),
        "quality_claim_result_verified": row.get(
            "quality_claim_result_verified"
        ),
    }, issues, failure_class


def _semantic_check(
    row: Mapping[str, Any], validation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Recheck the selected Full semantic row instead of trusting aliases."""
    row_index = _strict_int(row.get("quality_row_index"))
    selected = (
        validation_rows[row_index]
        if row_index is not None and 0 <= row_index < len(validation_rows)
        else None
    )
    issues: list[str] = []
    if selected is None:
        issues.append("semantic_validation_row_unavailable")
        return {
            "ok": False,
            "quality_row_index": row_index,
            "available": None,
            "semantic_ok": None,
            "status": "unavailable",
            "input_binding_status": "unavailable",
        }, issues

    semantic_available = selected.get("semantic_available")
    semantic_ok = selected.get("semantic_ok")
    status = str(
        selected.get("semantic_validation_status") or ""
    ).strip().lower()
    input_binding_status = str(
        selected.get("semantic_input_binding_status") or ""
    ).strip().lower()
    self_reference_available = selected.get("self_reference_available")
    self_reference_ok = selected.get("self_reference_ok")
    input_manifest_kind = str(
        selected.get("self_reference_input_manifest_kind") or ""
    ).strip().lower()
    if semantic_available is not True:
        issues.append("semantic_evidence_not_available")
    if semantic_ok is not True:
        issues.append("semantic_result_not_passed")
    if status != "passed":
        issues.append(
            f"semantic_validation_status_not_passed:{status or 'missing'}"
        )
    if input_binding_status != "passed":
        issues.append(
            "semantic_input_binding_status_not_passed:"
            f"{input_binding_status or 'missing'}"
        )
    if self_reference_available is not True:
        issues.append("full_self_reference_not_available")
    if self_reference_ok is not True:
        issues.append("full_self_reference_not_passed")
    if input_manifest_kind != "native_full_input_manifest":
        issues.append(
            "full_self_reference_input_manifest_kind_invalid:"
            f"{input_manifest_kind or 'missing'}"
        )
    if selected.get("strict_tensor_ok") is not True:
        issues.append("semantic_tensor_dump_not_strictly_valid")
    return {
        "ok": not issues,
        "quality_row_index": row_index,
        "available": semantic_available,
        "semantic_ok": semantic_ok,
        "status": status,
        "input_binding_status": input_binding_status,
        "self_reference_available": self_reference_available,
        "self_reference_ok": self_reference_ok,
        "input_manifest_kind": input_manifest_kind,
        "reason": _short(selected.get("self_reference_reason")),
        "diagnosis": _short(selected.get("self_reference_diagnosis")),
        "strict_tensor_ok": selected.get("strict_tensor_ok"),
    }, issues


def verify_run(run_dir: Path, *, scope: str = "vendor") -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    reports = run_dir / "reports"
    expected_path = reports / "native_expected_matrix.json"
    performance_path = reports / "native_producer_combined_summary.json"
    validation_path = (
        reports / "native_validation" / "native_producer_validation_summary.json"
    )
    expected_payload = _load_json_object(expected_path, label="expected_matrix")
    performance_payload = _load_json_object(
        performance_path, label="performance_summary",
    )
    validation_payload = _load_json_object(
        validation_path, label="validation_summary",
    )
    expected_rows = _expected_full_rows(expected_payload, scope=scope)
    performance_rows = _require_summary(
        performance_payload, schema=PERFORMANCE_SCHEMA,
        label="performance_summary",
    )
    validation_rows = _require_summary(
        validation_payload, schema=VALIDATION_SCHEMA,
        label="validation_summary",
    )

    performance_index: dict[
        tuple[str, ...], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in performance_rows:
        identity = _line_identity(row)
        if identity[2] == "full" and identity[0].startswith("native_full_"):
            performance_index[identity].append(row)
    raw_index = _load_raw_full_rows(run_dir, performance_rows)

    output_rows: list[dict[str, Any]] = []
    for expected in expected_rows:
        identity = _line_identity(expected)
        matches = performance_index.get(identity, [])
        line: dict[str, Any] = {
            "identity": _identity_dict(identity),
            "ok": False,
            "status": "",
            "issues": [],
        }
        if not matches:
            line.update({
                "status": "MISSING",
                "issues": ["performance_row_missing"],
            })
            output_rows.append(line)
            continue
        if len(matches) > 1:
            line.update({
                "status": "DUPLICATE",
                "issues": [f"performance_row_duplicate:{len(matches)}"],
            })
            output_rows.append(line)
            continue

        row = matches[0]
        runtime, runtime_issues = _runtime_check(row)
        repetitions, repetition_issues = _repetition_check(row)
        quality, quality_issues, quality_failure_class = _quality_check(
            row, validation_rows,
        )
        semantic, semantic_issues = _semantic_check(row, validation_rows)
        issues = [
            *runtime_issues, *repetition_issues, *quality_issues,
            *semantic_issues,
        ]
        if runtime_issues:
            status = "RUNTIME_FAIL"
        elif repetition_issues:
            status = "REPETITION_FAIL"
        elif quality_failure_class:
            status = quality_failure_class
        elif semantic_issues:
            status = "SEMANTIC_FAIL"
        else:
            status = "PASS"
        line.update({
            "ok": status == "PASS",
            "status": status,
            "issues": issues,
            "runtime": runtime,
            "repetitions": repetitions,
            "quality_join": quality,
            "semantic": semantic,
            "child_failure": _child_failure(
                row, raw_index.get(identity, []),
            ) if status != "PASS" else {},
        })
        output_rows.append(line)

    passed = sum(row.get("ok") is True for row in output_rows)
    result = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 2,
        "ok": passed == len(output_rows),
        "status": "ok" if passed == len(output_rows) else "failed",
        "scope": scope,
        "run_dir": str(run_dir),
        "expected_line_count": len(output_rows),
        "passed_line_count": passed,
        "failed_line_count": len(output_rows) - passed,
        "sources": {
            "expected_matrix": str(expected_path),
            "performance_summary": str(performance_path),
            "validation_summary": str(validation_path),
        },
        "rows": output_rows,
    }
    return result


def _table_cell(value: Any, *, limit: int = 80) -> str:
    text = str(value if value not in (None, "") else "-")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _render_table(result: Mapping[str, Any]) -> str:
    headers = (
        "STATUS", "BACKEND", "MODEL", "SETUP", "RUNTIME", "REPS",
        "QUALITY", "SEMANTIC", "DIFF", "CHILD",
    )
    table_rows: list[tuple[str, ...]] = []
    for row in result.get("rows") or []:
        identity = row.get("identity") or {}
        runtime = row.get("runtime") or {}
        repetitions = row.get("repetitions") or {}
        quality = row.get("quality_join") or {}
        semantic = row.get("semantic") or {}
        nearest = quality.get("nearest_candidates") or []
        axes: list[str] = []
        for candidate in nearest:
            if isinstance(candidate, Mapping):
                axes.extend((candidate.get("differing_axes") or {}).keys())
        child = row.get("child_failure") or {}
        child_text = (
            child.get("error") or child.get("status_detail")
            or child.get("failure_reason")
            or ((child.get("failed_repetition") or {}).get("error"))
            or ((child.get("raw_source") or {}).get("error")) or "-"
        )
        table_rows.append(tuple(map(_table_cell, (
            row.get("status"), identity.get("backend"), identity.get("model"),
            identity.get("setup_id"), runtime.get("status"),
            f"{repetitions.get('valid')}/{repetitions.get('requested')}",
            f"{quality.get('match_status')}/{quality.get('binding_verified')}",
            f"{semantic.get('status')}/{semantic.get('input_binding_status')}",
            ",".join(sorted(set(axes))) or "-", child_text,
        ))))
    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: Iterable[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    lines = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    lines.extend(format_row(row) for row in table_rows)
    lines.append(
        f"status={result.get('status')} scope={result.get('scope')} "
        f"passed={result.get('passed_line_count')}/"
        f"{result.get('expected_line_count')}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--scope", choices=("vendor", "all"), default="vendor",
        help="vendor checks accelerator Full rows; all also checks TRT Full rows",
    )
    parser.add_argument(
        "--format", choices=("json", "table"), default="table",
    )
    ns = parser.parse_args(argv)
    try:
        result = verify_run(ns.run_dir, scope=ns.scope)
    except VerificationInputError as exc:
        error = {
            "schema": OUTPUT_SCHEMA,
            "schema_version": 2,
            "ok": False,
            "status": "input_error",
            "error": str(exc),
            "run_dir": str(ns.run_dir.expanduser()),
            "scope": ns.scope,
        }
        if ns.format == "json":
            print(json.dumps(error, indent=2, sort_keys=True))
        else:
            print(f"INPUT_ERROR {exc}", file=sys.stderr)
        return 3
    if ns.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_render_table(result))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
