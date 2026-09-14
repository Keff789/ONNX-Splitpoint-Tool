#!/usr/bin/env python3
"""Run configured native-producer variants for an existing EvaluationRun.

This is the GUI-facing native fastpath coordinator.  The normal EvalRun profile
can declare ``native_producers.variants`` so different models/cases can use
different Native FIFO contracts, for example:

  - yolo26s/b038 as uint8_dequant_fp16 with explicit layout/dequant params
  - resnet50/b052 as float32_layout_fp16

The script delegates each variant to update_evalset_native_producers.py, then
rebuilds the combined native report and semantic self-reference validation using
a report Python that has onnxruntime available when possible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_THIS_FILE = Path(__file__).resolve()
ROOT = (
    _THIS_FILE.parents[1]
    if _THIS_FILE.parent.name == "scripts"
    else _THIS_FILE.parents[3]
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_progress import NativeProgressJournal, run_streaming
from onnx_splitpoint_tool.cache_verify_policy import (
    CACHE_VERIFY_ONLY,
    CacheVerifyPolicyError,
    validate_cache_verify_only_profile,
)
from onnx_splitpoint_tool.energy.config import energy_ab_cli_args
from onnx_splitpoint_tool.quality_service import _validate_candidate_execution_contract
from onnx_splitpoint_tool.trt_quality_chain import (
    PRODUCER_SET_SCHEMA,
    SPLIT_BINDING_SET_SCHEMA,
    TensorRTQualityChainError,
    load_split_binding_set_from_central_quality_summary,
    producer_set_from_central_quality_summary,
)
from onnx_splitpoint_tool.native_split_quality import (
    canonical_native_split_backend,
    canonical_json_sha256,
    validate_central_native_split_quality_selection,
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.native_split_quality_authority import (
    native_split_quality_required_for_row,
    resolve_native_split_quality_authority,
)
from onnx_splitpoint_tool.native_performance_identity import (
    canonical_native_backend as _shared_canonical_native_backend,
    native_performance_alias as _shared_native_performance_alias,
    native_performance_backend_value as _shared_native_performance_backend_value,
    native_performance_identity as _shared_native_performance_identity,
    native_performance_mode as _shared_native_performance_mode,
    native_performance_token as _shared_native_performance_token,
)
from onnx_splitpoint_tool.native_execution_contract import (
    NATIVE_EXECUTION_FIELDS,
    build_native_execution_contract,
    enforce_native_variant_execution_contract,
    verify_native_execution_contract,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.native_energy_preflight import (
    build_native_energy_preflight,
    native_energy_preflight_blocks_streaming,
)
from onnx_splitpoint_tool.workflow.checkpoints import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256 as checkpoint_json_sha256,
    load_stage_checkpoint,
    load_reusable_stage_checkpoint,
    native_coordinator_input_hash,
    write_stage_checkpoint,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _script(name: str) -> Path:
    source = ROOT / "scripts" / name
    if source.is_file():
        return source
    packaged = _THIS_FILE.parent / name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(name)


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_yaml_or_json(p: Path) -> dict[str, Any]:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return {}
    if yaml is not None and p.suffix.lower() in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text) or {}
            return dict(data) if isinstance(data, Mapping) else {}
        except Exception:
            pass
    try:
        data = json.loads(text)
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _write_json(p: Path, data: Mapping[str, Any]) -> None:
    atomic_write_json(p, dict(data))


def _native_performance_checkpoint_paths(
    run_dir: Path,
) -> tuple[Path, Path]:
    root = (
        run_dir / "stages" / "run_native_producers" / "native_performance"
    )
    return root / "stage_result.json", root / "stage_snapshot.json"


def _native_coordinator_checkpoint_path(run_dir: Path) -> Path:
    return (
        run_dir / "stages" / "run_native_producers"
        / "native_coordinator" / "stage_result.json"
    )


def _native_coordinator_artifacts(
    run_dir: Path, reports: Path,
) -> list[Path]:
    candidates = [
        reports / "native_producer_stage.json",
        reports / "native_producer_summary.json",
        reports / "native_producer_combined_summary.json",
        reports / "native_expected_matrix.json",
        reports / "native_validation"
        / "native_producer_validation_summary.json",
        reports / "native_energy_measurements"
        / "native_producer_energy_results.json",
        reports / "native_energy_measurements"
        / "stages" / "native_energy" / "stage_result.json",
        *_native_performance_artifacts(
            _native_performance_checkpoint_paths(run_dir)[1]
        ),
    ]
    return [
        path for path in candidates
        if path.is_file() and not path.is_symlink()
    ]


def _write_native_coordinator_terminal(
    *,
    run_dir: Path,
    reports: Path,
    stage: Mapping[str, Any],
    input_hash: str,
    return_code: int,
) -> None:
    status = str(stage.get("status") or "failed").strip().lower()
    write_stage_checkpoint(
        _native_coordinator_checkpoint_path(run_dir),
        stage="native_coordinator",
        state="failed" if status == "failed" else "completed",
        complete=True,
        input_hash=input_hash,
        run_root=run_dir,
        artifacts=_native_coordinator_artifacts(run_dir, reports),
        details={
            "return_code": int(return_code),
            "stage_status": status,
            "stage_state": str(stage.get("state") or ""),
            "stage_complete": stage.get("complete") is True,
        },
        error=(
            str(stage.get("error") or stage.get("failure_reason") or "")
            if status == "failed" else ""
        ),
        started_at=str(stage.get("started_at") or ""),
    )


def _native_performance_input_hash(
    *,
    run_dir: Path,
    cfg: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    expected_rows: list[Mapping[str, Any]],
    split_quality_authority: Mapping[str, Any],
) -> str:
    manifest = _read_json(run_dir / "run_manifest.json", {}) or {}
    resume_contract = (
        manifest.get("resume_contract")
        if isinstance(manifest, Mapping) else {}
    )
    return checkpoint_json_sha256({
        "schema": "onnx-splitpoint/native-performance-checkpoint-input",
        "schema_version": 1,
        "run_id": run_dir.name,
        "config": dict(cfg),
        "variants": [dict(value) for value in variants],
        "ordered_expected_rows": [dict(value) for value in expected_rows],
        "split_quality_authority": dict(split_quality_authority),
        "resume_contract_sha256": str(
            (resume_contract or {}).get("resume_contract_sha256")
        ),
    })


def _native_performance_expected_matrix(
    expected_rows: list[Mapping[str, Any]],
    actual_rows: list[Mapping[str, Any]],
    *,
    setup_local_tensorrt_errors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the configured Native denominator to the imported report rows."""

    expected_identities = [
        _native_performance_identity(row) for row in expected_rows
    ]
    actual_identities = [
        _native_performance_identity(row) for row in actual_rows
    ]
    actual_index: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    invalid_actual: list[dict[str, Any]] = []
    for row, identity in zip(actual_rows, actual_identities):
        if identity is None:
            invalid_actual.append(dict(row))
            continue
        actual_index.setdefault(identity, []).append(row)

    expected_identity_counts = Counter(
        identity for identity in expected_identities if identity is not None
    )
    duplicate_expected = {
        identity: count
        for identity, count in expected_identity_counts.items()
        if count != 1
    }
    duplicate_actual = {
        identity: len(rows)
        for identity, rows in actual_index.items()
        if len(rows) != 1
    }

    present: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    matched_identities: set[tuple[str, ...]] = set()
    trt_errors = {
        str(key): str(value)
        for key, value in dict(
            setup_local_tensorrt_errors or {}
        ).items()
        if str(key).strip() and str(value).strip()
    }
    for raw_expected, identity in zip(expected_rows, expected_identities):
        expected = dict(raw_expected)
        matches = actual_index.get(identity, []) if identity is not None else []
        if (
            identity is None
            or expected_identity_counts.get(identity, 0) != 1
            or len(matches) != 1
        ):
            reason = (
                "native_expected_identity_invalid"
                if identity is None else
                "native_expected_identity_duplicate"
                if expected_identity_counts.get(identity, 0) != 1 else
                "native_actual_identity_duplicate"
                if len(matches) > 1 else
                "native_expected_row_missing"
            )
            missing_row = {
                **expected,
                "ok": False,
                "result_ok": False,
                "status": "missing_expected_native_row",
                "runtime_status": "missing",
                "failure_reason": reason,
            }
            setup_id = str(expected.get("setup_id") or "").strip()
            is_setup_local_trt = bool(
                str(expected.get("execution_mode") or "")
                == "native_full_baseline"
                and str(expected.get("backend") or "")
                == "native_full_tensorrt"
                and setup_id in trt_errors
                and reason == "native_expected_row_missing"
            )
            if is_setup_local_trt:
                missing_row.update({
                    "status": "blocked_upstream_quality",
                    "runtime_status": "not_started",
                    "failure_class": "upstream_quality_evidence",
                    "failure_reason": (
                        "setup_local_tensorrt_quality_producer_missing"
                    ),
                    "status_detail": trt_errors[setup_id],
                    "upstream_stage": "central_quality",
                    "execution_attempted": False,
                    "transfer_attempted": False,
                })
            missing.append(missing_row)
            continue
        actual = matches[0]
        matched_identities.add(identity)
        actual_ok = (
            actual.get("ok")
            if "ok" in actual else actual.get("result_ok")
        ) is True
        observed = {
            **expected,
            "actual_ok": actual_ok,
            "actual_status": str(
                actual.get("status")
                or actual.get("runtime_status")
                or ("ok" if actual_ok else "failed")
            ),
            "failure_reason": str(actual.get("failure_reason") or ""),
            "status_detail": str(
                actual.get("status_detail")
                or actual.get("error")
                or actual.get("failure_reason")
                or ""
            ),
            "error": str(actual.get("error") or ""),
        }
        present.append(observed)
        (successful if actual_ok else failed).append(observed)

    unexpected_actual = [
        dict(row)
        for identity, rows in actual_index.items()
        if identity not in expected_identity_counts
        for row in rows
    ]
    expected_setup_counts = _native_performance_setup_counts(expected_rows)
    present_setup_counts = _native_performance_setup_counts(present)
    actual_setup_counts = _native_performance_setup_counts(actual_rows)
    standard_63_expected_shape = _native_performance_campaign63_shape(
        expected_rows,
    )
    standard_63_actual_shape = _native_performance_campaign63_shape(
        actual_rows,
    )
    exact_identity_import = bool(
        expected_rows
        and not missing
        and not duplicate_expected
        and not duplicate_actual
        and not invalid_actual
        and not unexpected_actual
        and len(actual_rows) == len(expected_rows)
        and len(matched_identities) == len(expected_rows)
    )

    return {
        "schema": "onnx-splitpoint/native-expected-matrix",
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "expected_row_count": len(expected_rows),
        "present_expected_row_count": len(present),
        "successful_expected_row_count": len(successful),
        "failed_expected_row_count": len(failed),
        "missing_expected_row_count": len(missing),
        "row_presence_complete": exact_identity_import,
        "execution_success_complete": bool(
            exact_identity_import and not failed
        ),
        "expected_rows_by_setup": expected_setup_counts,
        "present_rows_by_setup": present_setup_counts,
        "actual_rows_by_setup": actual_setup_counts,
        "setup_distribution_match": bool(
            expected_setup_counts
            and expected_setup_counts == present_setup_counts
            and expected_setup_counts == actual_setup_counts
        ),
        "standard_63_campaign_shape_complete": (
            bool(standard_63_expected_shape and standard_63_actual_shape)
            if len(expected_rows) == 63 else None
        ),
        "duplicate_expected_identity_count": sum(
            count - 1 for count in duplicate_expected.values()
        ),
        "duplicate_actual_identity_count": sum(
            count - 1 for count in duplicate_actual.values()
        ),
        "invalid_actual_identity_count": len(invalid_actual),
        "unexpected_actual_row_count": len(unexpected_actual),
        "expected_rows": [dict(row) for row in expected_rows],
        "present_expected_rows": present,
        "successful_expected_rows": successful,
        "failed_expected_rows": failed,
        "missing_expected_rows": missing,
        "invalid_actual_rows": invalid_actual,
        "unexpected_actual_rows": unexpected_actual,
    }


def _native_performance_token(value: Any) -> str:
    return _shared_native_performance_token(value)


def _native_performance_mode(row: Mapping[str, Any]) -> str:
    return _shared_native_performance_mode(row)


def _native_performance_backend_value(value: Any, mode: str) -> str:
    return _shared_native_performance_backend_value(value, mode)


def _native_performance_alias(
    row: Mapping[str, Any], *fields: str,
) -> str | None:
    return _shared_native_performance_alias(row, *fields)


def _native_performance_identity(
    row: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Return one identity shared by planned and final-report row schemas."""
    return _shared_native_performance_identity(row)


def _native_performance_setup_counts(
    rows: list[Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(
        _native_performance_token(row.get("setup_id") or row.get("setup"))
        for row in rows
        if isinstance(row, Mapping)
    )
    counts.pop("", None)
    return dict(sorted(counts.items()))


def _native_performance_campaign63_shape(
    rows: list[Mapping[str, Any]],
) -> bool:
    """Validate the current 63-row campaign without constraining old plans."""

    identities = [
        _native_performance_identity(row)
        for row in rows if isinstance(row, Mapping)
    ]
    if (
        len(rows) != 63
        or len(identities) != 63
        or None in identities
        or len(set(identities)) != 63
    ):
        return False
    by_setup: dict[str, list[tuple[str, ...]]] = {}
    for identity in identities:
        assert identity is not None
        by_setup.setdefault(identity[4], []).append(identity)
    if len(by_setup) != 3 or any(len(values) != 21 for values in by_setup.values()):
        return False
    split_backend_by_producer = {
        "hailo8": "hailo8_to_trt",
        "hailo10h": "hailo10h_to_trt",
        "deepx": "deepx_to_trt",
    }
    global_models: set[str] | None = None
    observed_producers: set[str] = set()
    for setup_rows in by_setup.values():
        comparisons = {identity[5] for identity in setup_rows}
        if len(comparisons) != 1:
            return False
        producer = next(iter(comparisons))
        observed_producers.add(producer)
        split_backend = split_backend_by_producer.get(producer)
        if not split_backend:
            return False
        models = {identity[2] for identity in setup_rows}
        if len(models) != 3:
            return False
        if global_models is None:
            global_models = models
        elif models != global_models:
            return False
        for model in models:
            model_rows = [
                identity for identity in setup_rows if identity[2] == model
            ]
            split_rows = [
                identity for identity in model_rows
                if identity[0] == "native_split"
            ]
            full_rows = [
                identity for identity in model_rows
                if identity[0] == "native_full_baseline"
            ]
            if (
                len(model_rows) != 7
                or len(split_rows) != 5
                or len({identity[3] for identity in split_rows}) != 5
                or {identity[1] for identity in split_rows} != {split_backend}
                or len(full_rows) != 2
                or {identity[1] for identity in full_rows} != {
                    f"native_full_{producer}",
                    "native_full_tensorrt",
                }
            ):
                return False
    return bool(
        global_models
        and observed_producers == {"deepx", "hailo8", "hailo10h"}
    )


_NATIVE_PERFORMANCE_RESULT_NAMES = (
    "native_producer_summary.json",
    "native_producer_combined_summary.json",
    "native_expected_matrix.json",
)


def _native_performance_artifacts(snapshot: Path) -> list[Path]:
    artifacts_root = snapshot.parent / "artifacts"
    return [
        snapshot,
        *(
            artifacts_root / name
            for name in _NATIVE_PERFORMANCE_RESULT_NAMES
        ),
    ]


def _freeze_native_performance_artifacts(
    reports: Path, snapshot: Path,
) -> None:
    artifacts_root = snapshot.parent / "artifacts"
    for name in _NATIVE_PERFORMANCE_RESULT_NAMES:
        source = reports / name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"Native Performance result is missing: {source}"
            )
        atomic_write_text(
            artifacts_root / name,
            source.read_text(encoding="utf-8"),
        )


def _restore_native_performance_artifacts(
    reports: Path, snapshot: Path,
) -> None:
    artifacts_root = snapshot.parent / "artifacts"
    for name in _NATIVE_PERFORMANCE_RESULT_NAMES:
        source = artifacts_root / name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"frozen Native Performance result is missing: {source}"
            )
        atomic_write_text(
            reports / name,
            source.read_text(encoding="utf-8"),
        )


def _native_performance_completion(
    *,
    reports: Path,
    expected_rows: list[Mapping[str, Any]],
    variant_count: int,
    stage: Mapping[str, Any],
    final_report: Mapping[str, Any],
    required_campaign_rows: int | None = 63,
) -> dict[str, Any]:
    matrix = _read_json(reports / "native_expected_matrix.json", {}) or {}
    summary = _read_json(reports / "native_producer_summary.json", {}) or {}
    summary_rows = (
        list(summary.get("rows") or [])
        if isinstance(summary, Mapping) else []
    )
    present_rows = (
        list(matrix.get("present_expected_rows") or [])
        if isinstance(matrix, Mapping) else []
    )
    configured_expected = len(expected_rows)
    def _strict_count(source: Mapping[str, Any], field: str) -> int:
        value = source.get(field)
        return value if type(value) is int and value >= 0 else -1

    expected = _strict_count(matrix, "expected_row_count")
    present = _strict_count(matrix, "present_expected_row_count")
    successful = _strict_count(matrix, "successful_expected_row_count")
    present_identities = [
        _native_performance_identity(row)
        for row in present_rows if isinstance(row, Mapping)
    ]
    expected_identities = [
        _native_performance_identity(row)
        for row in expected_rows if isinstance(row, Mapping)
    ]
    summary_identities = [
        _native_performance_identity(row)
        for row in summary_rows if isinstance(row, Mapping)
    ]
    duplicate_present_identity_count = (
        len(present_identities) - len(set(present_identities))
    )
    duplicate_summary_identity_count = (
        len(summary_identities) - len(set(summary_identities))
    )
    missing_expected_count = _strict_count(
        matrix, "missing_expected_row_count",
    )
    matrix_duplicate_expected = _strict_count(
        matrix, "duplicate_expected_identity_count",
    )
    matrix_duplicate_actual = _strict_count(
        matrix, "duplicate_actual_identity_count",
    )
    matrix_invalid_actual = _strict_count(
        matrix, "invalid_actual_identity_count",
    )
    matrix_unexpected_actual = _strict_count(
        matrix, "unexpected_actual_row_count",
    )
    expected_setup_counts = _native_performance_setup_counts(expected_rows)
    present_setup_counts = _native_performance_setup_counts(present_rows)
    summary_setup_counts = _native_performance_setup_counts(summary_rows)
    standard_63_expected_shape = _native_performance_campaign63_shape(
        expected_rows,
    )
    standard_63_present_shape = _native_performance_campaign63_shape(
        present_rows,
    )
    standard_63_summary_shape = _native_performance_campaign63_shape(
        summary_rows,
    )
    standard_63_setup_distribution_complete = bool(
        configured_expected == 63
        and len(expected_setup_counts) == 3
        and set(expected_setup_counts.values()) == {21}
        and present_setup_counts == expected_setup_counts
        and summary_setup_counts == expected_setup_counts
    )
    if required_campaign_rows is not None and (
        type(required_campaign_rows) is not int
        or required_campaign_rows <= 0
    ):
        raise ValueError("required Native Performance row count is invalid")
    configured_shape_complete = bool(
        required_campaign_rows is None
        or (
            configured_expected == required_campaign_rows
            and (
                required_campaign_rows != 63
                or (
                    standard_63_setup_distribution_complete
                    and standard_63_expected_shape
                    and standard_63_present_shape
                    and standard_63_summary_shape
                    and matrix.get(
                        "standard_63_campaign_shape_complete"
                    ) is True
                )
            )
        )
    )
    variant_results = [
        value for value in list(stage.get("variant_results") or [])
        if isinstance(value, Mapping)
    ]
    variants_terminal = bool(
        len(variant_results) == variant_count
        and all(value.get("rc") is not None for value in variant_results)
    )
    row_presence_complete = bool(
        configured_expected > 0
        and expected == configured_expected
        and len(expected_identities) == configured_expected
        and None not in expected_identities
        and len(expected_identities) == len(set(expected_identities))
        and present == expected
        and len(present_rows) == expected
        and missing_expected_count == 0
        and matrix_duplicate_expected == 0
        and matrix_duplicate_actual == 0
        and matrix_invalid_actual == 0
        and matrix_unexpected_actual == 0
        and duplicate_present_identity_count == 0
        and set(present_identities) == set(expected_identities)
        and matrix.get("row_presence_complete") is True
        and matrix.get("setup_distribution_match") is True
        and present_setup_counts == expected_setup_counts
        and configured_shape_complete
    )
    summary_import_complete = bool(
        len(summary_rows) == expected
        and None not in summary_identities
        and len(summary_identities) == len(set(summary_identities))
        and set(summary_identities) == set(present_identities)
        and summary_setup_counts == expected_setup_counts
    )
    imported = bool(
        final_report.get("rc") == 0
        and summary_import_complete
        and all(
            (reports / name).is_file()
            for name in _NATIVE_PERFORMANCE_RESULT_NAMES
        )
    )
    parent_import_terminal = bool(
        final_report.get("rc") == 0
        and all(
            (reports / name).is_file()
            for name in _NATIVE_PERFORMANCE_RESULT_NAMES
        )
        and None not in summary_identities
        and duplicate_summary_identity_count == 0
        and matrix_duplicate_actual == 0
        and matrix_invalid_actual == 0
        and matrix_unexpected_actual == 0
        and set(summary_identities).issubset(set(expected_identities))
    )
    performance_matrix_complete = bool(
        row_presence_complete and variants_terminal and imported
    )
    checkpoint_terminal_valid = bool(
        variants_terminal and parent_import_terminal
    )
    return {
        "configured_expected_row_count": configured_expected,
        "required_campaign_row_count": required_campaign_rows,
        "configured_campaign_shape_complete": configured_shape_complete,
        "expected_row_count": expected,
        "present_row_count": present,
        "successful_row_count": successful,
        "row_presence_complete": row_presence_complete,
        "missing_expected_row_count": missing_expected_count,
        "duplicate_present_identity_count": duplicate_present_identity_count,
        "duplicate_summary_identity_count": duplicate_summary_identity_count,
        "planned_identity_match": bool(
            set(present_identities) == set(expected_identities)
        ),
        "expected_rows_by_setup": expected_setup_counts,
        "present_rows_by_setup": present_setup_counts,
        "summary_rows_by_setup": summary_setup_counts,
        "setup_distribution_match": bool(
            expected_setup_counts == present_setup_counts
            and expected_setup_counts == summary_setup_counts
        ),
        "standard_63_setup_distribution_complete": (
            standard_63_setup_distribution_complete
            if configured_expected == 63 else None
        ),
        "standard_63_campaign_shape_complete": (
            bool(
                standard_63_expected_shape
                and standard_63_present_shape
                and standard_63_summary_shape
                and matrix.get(
                    "standard_63_campaign_shape_complete"
                ) is True
            )
            if configured_expected == 63 else None
        ),
        "summary_row_count": len(summary_rows),
        "summary_import_complete": summary_import_complete,
        "variant_count": variant_count,
        "variant_terminal_count": len(variant_results),
        "variants_terminal": variants_terminal,
        "parent_import_complete": imported,
        "parent_import_terminal": parent_import_terminal,
        "checkpoint_terminal_valid": checkpoint_terminal_valid,
        "performance_matrix_complete": performance_matrix_complete,
        "scientific_coverage_complete": performance_matrix_complete,
        # Compatibility alias: historically ``complete`` meant full matrix,
        # not merely a durable terminal checkpoint.
        "complete": performance_matrix_complete,
    }


def _strict_json(path: Path) -> Any:
    """Read security/scientific identity JSON without duplicate-key collapse."""

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise TensorRTQualityChainError(
                    f"duplicate JSON key in TensorRT quality producer input: {key!r}"
                )
            out[key] = value
        return out

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except TensorRTQualityChainError:
        raise
    except Exception as exc:
        raise TensorRTQualityChainError(
            f"TensorRT quality producer input is not valid JSON: {path}: {exc}"
        ) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _safe_component(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ) or "unknown"


def _sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(value: Any, *, base: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _remote_value(cfg: Mapping[str, Any], backend: str, key: str) -> str:
    remotes = cfg.get("remotes") if isinstance(cfg.get("remotes"), Mapping) else {}
    row = remotes.get(backend) if isinstance(remotes, Mapping) else {}
    if isinstance(row, str):
        return row if key == "ssh" else ""
    return str(row.get(key) or "") if isinstance(row, Mapping) else ""


def _workflow_context(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = cfg.get("_workflow_context")
    return dict(value) if isinstance(value, Mapping) else {}


def _smoke_diagnostic_policy(cfg: Mapping[str, Any]) -> bool:
    if _as_bool(cfg.get("cache_verify_only"), False):
        return True
    context = _workflow_context(cfg)
    preset = context.get("execution_preset")
    preset = dict(preset) if isinstance(preset, Mapping) else {}
    preset_id = str(preset.get("id") or "").strip().lower()
    if preset_id in {"standard", "final"}:
        return False
    if preset_id == "smoke":
        return True
    quality = cfg.get("quality_gate_policy")
    quality = dict(quality) if isinstance(quality, Mapping) else {}
    enforcement = quality.get("enforcement")
    enforcement = dict(enforcement) if isinstance(enforcement, Mapping) else {}
    return bool(
        quality.get("diagnostic_only") is True
        and str(enforcement.get("technical_quality_error") or "").startswith("partial_continue")
    )


def _standard_quality_enforced_policy(cfg: Mapping[str, Any]) -> bool:
    if _as_bool(cfg.get("cache_verify_only"), False):
        return False
    context = _workflow_context(cfg)
    preset = context.get("execution_preset")
    preset = dict(preset) if isinstance(preset, Mapping) else {}
    if str(preset.get("id") or "").strip().lower() in {"standard", "final"}:
        return True
    quality = cfg.get("quality_gate_policy")
    quality = dict(quality) if isinstance(quality, Mapping) else {}
    enforcement = quality.get("enforcement")
    enforcement = dict(enforcement) if isinstance(enforcement, Mapping) else {}
    return str(enforcement.get("technical_quality_error") or "").strip() in {
        "hard_fail", "fail", "enforced",
    }


def _bind_evalrun_quality_gate_policy(
    run_dir: Path, cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Bind standalone Native validation to the EvalRun's frozen policy.

    Native producer configs are derived execution plans.  They may omit the
    top-level Evaluation Profile ``quality_gate`` block, but they must never
    silently fall back to a different default policy.  Existing explicit
    config policy remains accepted only when its normalized identity agrees
    with the EvalRun profile.
    """

    effective = copy.deepcopy(dict(cfg))
    if _as_bool(effective.get("cache_verify_only"), False):
        return effective, "not_applicable_cache_verify_only"

    profile_path = run_dir / "profile.yaml"
    profile = _read_yaml_or_json(profile_path) if profile_path.is_file() else {}
    profile_policy = (
        dict(profile.get("quality_gate"))
        if isinstance(profile.get("quality_gate"), Mapping) else {}
    )
    configured_policy = (
        dict(effective.get("quality_gate_policy"))
        if isinstance(effective.get("quality_gate_policy"), Mapping) else {}
    )
    validation = (
        dict(effective.get("validation"))
        if isinstance(effective.get("validation"), Mapping) else {}
    )
    policy_required = bool(
        _standard_quality_enforced_policy(effective)
        or _as_bool(
            validation.get("enabled")
            or effective.get("native_validation_enabled")
            or effective.get("native_validation"),
            False,
        )
    )

    if profile_policy and configured_policy:
        profile_sha = AccuracyGatePolicy.from_mapping(profile_policy).sha256()
        configured_sha = AccuracyGatePolicy.from_mapping(
            configured_policy
        ).sha256()
        if profile_sha != configured_sha:
            raise ValueError(
                "native quality_gate_policy drifts from EvalRun/profile.yaml"
            )
        effective["quality_gate_policy"] = profile_policy
        return effective, "profile_verified_against_explicit_config"
    if profile_policy:
        effective["quality_gate_policy"] = profile_policy
        return effective, "inherited_from_evalrun_profile"
    if configured_policy:
        if policy_required:
            raise ValueError(
                "EvalRun/profile.yaml quality_gate is required to verify the "
                "Native quality policy"
            )
        return effective, "explicit_config_without_profile_development_only"
    if policy_required:
        raise ValueError(
            "Native validation requires EvalRun/profile.yaml quality_gate"
        )
    return effective, "not_requested"


def _native_performance_required_campaign_rows(
    cfg: Mapping[str, Any],
    *, expected_row_count: int | None = None,
) -> int | None:
    contract = cfg.get("native_performance_checkpoint")
    contract = dict(contract) if isinstance(contract, Mapping) else {}
    explicit = contract.get("required_row_count")
    explicit_count: int | None = None
    if explicit not in (None, ""):
        if type(explicit) is not int or explicit <= 0:
            raise ValueError(
                "native_performance_checkpoint.required_row_count is invalid"
            )
        explicit_count = explicit
    if _as_bool(cfg.get("cache_verify_only"), False):
        expected_plan = cfg.get("cache_verify_expected_plan")
        expected_plan = (
            dict(expected_plan) if isinstance(expected_plan, Mapping) else {}
        )
        expected_rows = [
            row for row in list(expected_plan.get("native_rows") or [])
            if isinstance(row, Mapping)
        ]
        expected_count = len(expected_rows)
        if expected_count <= 0:
            raise ValueError(
                "cache_verify_only requires an attested non-empty native_rows plan"
            )
        if explicit_count is not None and explicit_count != expected_count:
            raise ValueError(
                "native_performance_checkpoint.required_row_count drifts from "
                "the cache_verify_only native_rows plan"
            )
        return expected_count
    context = _workflow_context(cfg)
    preset = context.get("execution_preset")
    preset = dict(preset) if isinstance(preset, Mapping) else {}
    campaign = context.get("campaign")
    campaign = dict(campaign) if isinstance(campaign, Mapping) else {}
    campaign_modes = {
        str(preset.get("id") or "").strip().lower(),
        str(campaign.get("mode") or "").strip().lower(),
    }
    if campaign_modes.intersection({"standard", "final"}):
        if explicit_count in {None, 63}:
            return 63
        # A targeted Native supplement deliberately measures a frozen subset
        # of already generated Single-Tensor cases.  Admit that bounded plan
        # only when its explicit denominator agrees with the independently
        # materialized row matrix and Full baselines are disabled.  This keeps
        # the 63-row fail-closed contract for every ordinary Standard/Final
        # campaign while allowing the historical 24-row supplement to commit.
        full_cfg = (
            dict(cfg.get("full_baselines"))
            if isinstance(cfg.get("full_baselines"), Mapping) else {}
        )
        variants = [
            dict(row) for row in list(cfg.get("variants") or [])
            if isinstance(row, Mapping)
        ]
        variants_disable_full = bool(variants) and all(
            not _as_bool(
                (
                    row.get("full_baselines", {}).get("enabled")
                    if isinstance(row.get("full_baselines"), Mapping)
                    else False
                ),
                False,
            )
            for row in variants
        )
        bounded_supplement = bool(
            str(contract.get("scope") or "").strip().lower()
            == "bounded_supplement"
            and
            str(cfg.get("case_policy") or "").strip().lower()
            == "case_map_only"
            and not _as_bool(full_cfg.get("enabled"), False)
            and variants_disable_full
            and expected_row_count is not None
            and expected_row_count > 0
        )
        if not bounded_supplement:
            raise ValueError(
                "native_performance_checkpoint.required_row_count must be "
                "63 for standard/final campaigns unless an exact bounded "
                "case_map_only supplement matrix with scope=bounded_supplement "
                "is materialized"
            )
        if explicit_count != expected_row_count:
            raise ValueError(
                "native_performance_checkpoint.required_row_count drifts "
                "from the bounded supplement matrix"
            )
        return explicit_count
    return explicit_count


def _window_probe_final_campaign(cfg: Mapping[str, Any]) -> bool:
    """Compatibility helper: the v2.67 sensitivity probe never blocks."""
    del cfg
    return False


def _energy_repeat_count(cfg: Mapping[str, Any], ecfg: Mapping[str, Any]) -> int:
    context = _workflow_context(cfg)
    top_energy = context.get("energy") if isinstance(context.get("energy"), Mapping) else {}
    preset = context.get("execution_preset") if isinstance(context.get("execution_preset"), Mapping) else {}
    snapshot = preset.get("snapshot") if isinstance(preset.get("snapshot"), Mapping) else {}
    snapshot_energy = snapshot.get("energy") if isinstance(snapshot.get("energy"), Mapping) else {}
    for value in (
        ecfg.get("repeat_override"), ecfg.get("repeats"),
        top_energy.get("repeat_override"), top_energy.get("repeats"),
        snapshot_energy.get("repeat_override"), snapshot_energy.get("repeats"),
    ):
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 1


def _final_all_split_energy_requested(
    cfg: Mapping[str, Any],
    ecfg: Mapping[str, Any],
) -> bool:
    context = _workflow_context(cfg)
    top_energy = (
        context.get("energy")
        if isinstance(context.get("energy"), Mapping) else {}
    )
    return _configured_bool(
        (ecfg, "final_all_split_energy"),
        (top_energy, "final_all_split_energy"),
        default=False,
    )


def _energy_window_ab_cli_args(
    cfg: Mapping[str, Any], ecfg: Mapping[str, Any]
) -> list[str]:
    context = _workflow_context(cfg)
    top_energy = context.get("energy") if isinstance(context.get("energy"), Mapping) else {}
    preset = context.get("execution_preset") if isinstance(context.get("execution_preset"), Mapping) else {}
    snapshot = preset.get("snapshot") if isinstance(preset.get("snapshot"), Mapping) else {}
    snapshot_energy = snapshot.get("energy") if isinstance(snapshot.get("energy"), Mapping) else {}
    raw = next(
        (
            dict(value)
            for value in (
                top_energy.get("window_method_ab"),
                ecfg.get("window_method_ab"),
                snapshot_energy.get("window_method_ab"),
            )
            if isinstance(value, Mapping)
        ),
        {},
    )
    if not raw:
        return []
    return energy_ab_cli_args(raw)


def _energy_contract_context(
    cfg: Mapping[str, Any], ecfg: Mapping[str, Any], *, run_dir: Path, reports: Path,
) -> dict[str, Any]:
    """Resolve the same campaign bindings used by the non-variant runner path."""
    context = _workflow_context(cfg)
    campaign = context.get("campaign") if isinstance(context.get("campaign"), Mapping) else {}
    measurement = context.get("measurement_campaign") if isinstance(context.get("measurement_campaign"), Mapping) else {}
    system_power = measurement.get("system_power") if isinstance(measurement.get("system_power"), Mapping) else {}
    profile_path = _resolved_path(context.get("profile_path"), base=run_dir)
    profile_base = profile_path.parent if profile_path is not None else run_dir

    calibration_raw = str(campaign.get("energy_calibration_manifest") or ecfg.get("calibration_manifest") or "").strip()
    calibration_path = _resolved_path(calibration_raw, base=profile_base)
    pipeline_raw = str(campaign.get("pipeline_contract_manifest") or ecfg.get("pipeline_contract_manifest") or "").strip()
    pipeline_path = _resolved_path(pipeline_raw, base=profile_base)
    scope = str(system_power.get("scope") or ecfg.get("physical_scope") or "FS").strip()
    window = str(system_power.get("window") or system_power.get("measurement_window") or ecfg.get("window_label") or "command").strip()
    final = _as_bool(
        context.get("native_energy_final_contract_requested"),
        str(campaign.get("mode") or "development").strip().lower() == "final",
    )

    model_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for manifest_path in sorted((run_dir / "models").glob("*/model_manifest.json")):
        manifest = _read_json(manifest_path, {}) or {}
        model = str(manifest.get("model_id") or "").strip()
        model_path = _resolved_path(manifest.get("resolved_path"), base=manifest_path.parent)
        model_hash = _sha256(model_path)
        manifest_hash = _sha256(manifest_path)
        if not model or model_path is None or not model_hash or not manifest_hash:
            excluded.append({"model": model, "model_manifest": str(manifest_path), "reason": "model_identity_path_or_hash_missing"})
            continue
        model_rows.append({
            "model": model,
            "model_path": str(model_path),
            "model_sha256": model_hash,
            "model_manifest": str(manifest_path.resolve()),
            "model_manifest_sha256": manifest_hash,
        })
    model_map = reports / "native_energy_model_hash_map.json"
    _write_json(model_map, {
        "schema": "onnx-splitpoint/native-energy-model-hash-map",
        "schema_version": 1,
        "rows": model_rows,
        "excluded": excluded,
    })
    return {
        "physical_scope": scope,
        "window_label": window,
        "calibration_manifest": str(calibration_path or calibration_raw),
        "calibration_sha256": _sha256(calibration_path),
        "pipeline_contract_manifest": str(pipeline_path or pipeline_raw),
        "pipeline_contract_sha256": _sha256(pipeline_path),
        "model_hash_map": str(model_map),
        "model_hash_map_sha256": _sha256(model_map),
        "verified_model_hash_count": len(model_rows),
        "excluded_model_hash_count": len(excluded),
        "final_energy_contract_enforced": final,
    }


def _common_energy_args(
    cfg: Mapping[str, Any], *, run_dir: Path, duration_s: float, timeout_s: int,
) -> list[str]:
    return [
        "--hailo8-ssh", _remote_value(cfg, "hailo8", "ssh"),
        "--hailo10-ssh", _remote_value(cfg, "hailo10h", "ssh"),
        "--deepx-ssh", _remote_value(cfg, "deepx", "ssh"),
        "--hailo8-env", _remote_value(cfg, "hailo8", "env"),
        "--hailo10-env", _remote_value(cfg, "hailo10h", "env"),
        "--deepx-env", _remote_value(cfg, "deepx", "env"),
        "--engine-build-python", str(cfg.get("engine_build_python") or "auto"),
        "--remote-tool-dir", str(cfg.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool"),
        "--remote-root", str(cfg.get("remote_root") or "/home/nx/native_fifo_evalsets").rstrip("/") + "/" + run_dir.name,
        "--duration-s", str(duration_s),
        "--timeout", str(timeout_s),
    ]


def _run_window_method_probe(
    *, cfg: Mapping[str, Any], summary_json: Path, validation_summary: Path,
    run_dir: Path, reports: Path,
) -> tuple[dict[str, Any], bool]:
    ecfg = cfg.get("energy") if isinstance(cfg.get("energy"), Mapping) else {}
    pcfg = ecfg.get("window_method_validation_probe") if isinstance(ecfg.get("window_method_validation_probe"), Mapping) else {}
    enabled = _as_bool(pcfg.get("enabled"), False)
    strict = _as_bool(pcfg.get("strict"), True)
    repeats = max(1, int(pcfg.get("repeats") or 3))
    include_raw = _as_bool(pcfg.get("include_raw_parquet"), True)
    out = reports / "window_method_validation_probe"
    if not enabled:
        return {
            "enabled": False, "requested": False, "status": "skipped",
            "strict_requested": strict, "strict_failure": False,
            "strict_validation_failure": False,
            "workflow_blocking_requested": bool(strict and _window_probe_final_campaign(cfg)),
            "include_raw_parquet": include_raw, "requested_repeats": repeats,
        }, False
    try:
        from onnx_splitpoint_tool.energy.config import load_energy_defaults as _load_energy_defaults
        energy_defaults = _load_energy_defaults()
        default_duration = float(getattr(energy_defaults, "native_energy_duration_s", 60.0) or 60.0)
        default_acquisition_retries = int(
            getattr(energy_defaults, "invalid_repeat_max_retries", 1) or 0
        )
        default_acquisition_retry_backoff_s = float(
            getattr(
                energy_defaults, "invalid_repeat_reconnect_backoff_s", 5.0
            )
            or 0.0
        )
    except Exception:
        default_duration = 60.0
        default_acquisition_retries = 1
        default_acquisition_retry_backoff_s = 5.0
    max_acquisition_retries = max(
        0,
        min(
            1,
            int(
                pcfg.get("max_acquisition_retries")
                if pcfg.get("max_acquisition_retries") not in (None, "")
                else default_acquisition_retries
            ),
        ),
    )
    acquisition_retry_backoff_s = max(
        0.0,
        min(
            60.0,
            float(
                pcfg.get("acquisition_retry_backoff_s")
                if pcfg.get("acquisition_retry_backoff_s") not in (None, "")
                else default_acquisition_retry_backoff_s
            ),
        ),
    )
    duration = max(1.0, float(pcfg.get("duration_s") or ecfg.get("duration_s") or ecfg.get("measurement_duration_s") or default_duration))
    timeout_s = int(pcfg.get("timeout_s") or ecfg.get("timeout_s") or ecfg.get("timeout") or max(900, int(duration + 600)))
    contract = _energy_contract_context(cfg, ecfg, run_dir=run_dir, reports=reports)
    cmd = [
        sys.executable, "-u", str(_script("run_window_method_validation_probe.py")),
        "--summary", str(summary_json),
        "--out-dir", str(out),
        "--validation-summary", str(validation_summary),
        *_common_energy_args(cfg, run_dir=run_dir, duration_s=duration, timeout_s=timeout_s),
        "--repeats", str(repeats),
        "--max-reconnect-retries", str(max_acquisition_retries),
        "--reconnect-backoff-s", str(acquisition_retry_backoff_s),
        "--physical-scope", str(contract["physical_scope"]),
        "--window-label", str(contract["window_label"]),
        "--calibration-manifest", str(contract["calibration_manifest"]),
        "--calibration-sha256", str(contract["calibration_sha256"]),
        "--include-raw-parquet" if include_raw else "--no-include-raw-parquet",
        "--strict" if strict else "--no-strict",
    ]
    result = _run(
        cmd,
        timeout=max(
            1800,
            repeats * (1 + max_acquisition_retries) * (timeout_s + 420) + 300,
        ),
        cwd=ROOT,
        label="window-method-validation-probe",
    )
    report_path = out / "window_method_validation_probe.json"
    report = _read_json(report_path, {}) or {}
    complete = report.get("complete") is True
    strict_validation_failure = bool(
        strict and (int(result.get("rc") or 0) != 0 or not complete)
    )
    workflow_blocking_requested = bool(strict and _window_probe_final_campaign(cfg))
    strict_failure = bool(
        strict_validation_failure and workflow_blocking_requested
    )
    block = {
        **result,
        "enabled": True,
        "requested": True,
        "screening_only": True,
        "diagnostic_only": True,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_claim": False,
        "affects_final_energy_gate": False,
        "workflow_blocking_scope": "none",
        "workflow_blocking_requested": workflow_blocking_requested,
        "include_raw_parquet": include_raw,
        "requested_repeats": repeats,
        "strict_requested": strict,
        "strict_validation_failure": strict_validation_failure,
        "strict_failure": strict_failure,
        "status": str(report.get("status") or ("probe_orchestration_failed" if result.get("rc") else "probe_report_missing")),
        "ok": report.get("ok") is True,
        "complete": complete,
        "decision_capable": report.get("decision_capable") is True,
        "started_measurement_count": int(report.get("started_repeat_count") or 0),
        "successful_measurement_count": int(report.get("successful_comparison_count") or 0),
        "reason": str(report.get("blocked_reason") or report.get("decision_capability_reason") or ""),
        "report_path": str(report_path),
        "out_dir": str(out),
        "physical_scope": contract["physical_scope"],
        "window_label": contract["window_label"],
        "calibration_manifest": contract["calibration_manifest"],
        "calibration_sha256": contract["calibration_sha256"],
    }
    return block, strict_failure


def _run(cmd: list[str], *, timeout: int | float | None = None, cwd: Path | None = None, label: str = "native-child") -> dict[str, Any]:
    journal=NativeProgressJournal.from_env()
    cp=run_streaming(cmd,timeout=timeout,cwd=cwd,label=label,journal=journal,heartbeat_s=30.0)
    return {"cmd":cmd,"rc":cp.returncode,"elapsed_s":cp.elapsed_s,"stdout_tail":cp.stdout[-8000:],"stderr_tail":cp.stderr[-8000:]}


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [v.strip() for v in x.replace(";", ",").split(",") if v.strip()]
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [str(x).strip()] if str(x).strip() else []


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"", "none", "null"}:
        return default
    return s in {"1", "true", "yes", "y", "on"}


def _configured_bool(
    *sources: tuple[Mapping[str, Any], str],
    default: bool = False,
) -> bool:
    """Resolve an explicit bool without letting ``False`` fall through."""
    for source, key in sources:
        if key in source and source.get(key) is not None:
            return _as_bool(source.get(key), default)
    return default


def _case_map_from_variant(v: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = v.get("case_map") or v.get("model_case_map")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    out: dict[str, list[str]] = {}
    if isinstance(raw, Mapping):
        for k, val in raw.items():
            out[str(k)] = _as_list(val)
    if not out:
        models = _as_list(v.get("models") or v.get("model"))
        cases = _as_list(v.get("cases") or v.get("case") or v.get("case_id"))
        for m in models:
            out[m] = cases or []
    return {k: [c for c in vals if c] for k, vals in out.items() if k and vals}


def _select_report_python(tool_root: Path, requested: str = "auto") -> tuple[str, dict[str, Any]]:
    def can_import(py: str) -> tuple[bool, str]:
        cp = subprocess.run([py, "-c", "import onnxruntime as ort; print(ort.__version__)"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return cp.returncode == 0, (cp.stdout.strip() or cp.stderr.strip())[-1000:]

    if requested and requested != "auto":
        ok, detail = can_import(requested)
        return requested, {"requested": requested, "selected": requested, "onnxruntime_ok": ok, "detail": detail}
    candidates = [sys.executable]
    venv_py = tool_root / ".venv-report" / "bin" / "python"
    if venv_py.is_file():
        candidates.append(str(venv_py))
    for py in candidates:
        ok, detail = can_import(py)
        if ok:
            return py, {"requested": requested, "selected": py, "onnxruntime_ok": True, "detail": detail}
    return sys.executable, {"requested": requested, "selected": sys.executable, "onnxruntime_ok": False, "detail": "onnxruntime unavailable in checked interpreters"}


def _variant_label(i: int, v: Mapping[str, Any]) -> str:
    return str(v.get("id") or v.get("name") or f"variant_{i:02d}")


def _variant_namespace(i: int, v: Mapping[str, Any]) -> str:
    """Return one collision-free filesystem token for a variant invocation."""

    return f"v{i:03d}_{_safe_component(_variant_label(i, v))}"


def _merged_variant(base: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    raw_contract = base.get("_native_execution_contract")
    if isinstance(raw_contract, Mapping):
        try:
            execution_contract = enforce_native_variant_execution_contract(
                variant, raw_contract,
            )
        except ValueError as exc:
            raise TensorRTQualityChainError(str(exc)) from exc
    else:
        execution_contract = build_native_execution_contract(
            base,
            run_mode=str(
                (_workflow_context(base).get("execution_preset") or {}).get(
                    "id"
                )
                if isinstance(
                    _workflow_context(base).get("execution_preset"), Mapping,
                )
                else ""
            ),
        )
        try:
            enforce_native_variant_execution_contract(
                variant, execution_contract,
            )
        except ValueError as exc:
            raise TensorRTQualityChainError(str(exc)) from exc
    out = dict(base)
    out.update(dict(variant))
    # Runtime effort is stage-immutable.  Precision, cases and backend-specific
    # layout may vary, but these five values always come from the one sealed
    # parent contract.
    for field in NATIVE_EXECUTION_FIELDS:
        out[field] = execution_contract[field]
    out["_native_execution_contract"] = execution_contract
    # Preserve shared remotes unless variant overrides them explicitly.
    if "remotes" not in variant and isinstance(base.get("remotes"), Mapping):
        out["remotes"] = dict(base.get("remotes") or {})
    if "validation" not in variant and isinstance(base.get("validation"), Mapping):
        out["validation"] = dict(base.get("validation") or {})
    if "energy" not in variant and isinstance(base.get("energy"), Mapping):
        out["energy"] = dict(base.get("energy") or {})
    return out


def _canonical_native_backend(value: Any) -> str:
    return _shared_canonical_native_backend(value)


def _default_setup_id(backend: str) -> str:
    return {
        "hailo8": "orin_nx_hailo8_01",
        "hailo10h": "orin_nx_hailo10_01",
        "deepx": "orin_nx_deepx_m1_01",
    }.get(backend, "")


def _variant_backend_bindings(
    base: Mapping[str, Any], variant: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Resolve physical setup bindings and reject alias/remote drift."""

    merged = _merged_variant(base, variant)
    backends = [
        _canonical_native_backend(value)
        for value in _as_list(merged.get("backends") or ["hailo8"])
    ]
    if any(not value for value in backends) or len(set(backends)) != len(backends):
        raise TensorRTQualityChainError(
            "a Native variant contains empty or duplicate physical backend aliases"
        )
    remotes = merged.get("remotes") if isinstance(merged.get("remotes"), Mapping) else {}
    out: list[dict[str, str]] = []
    for backend in backends:
        remote = remotes.get(backend) if isinstance(remotes, Mapping) else {}
        if isinstance(remote, str):
            remote = {"ssh": remote}
        if not isinstance(remote, Mapping):
            remote = {}
        setup_id = str(remote.get("setup_id") or _default_setup_id(backend)).strip()
        ssh = str(remote.get("ssh") or "").strip()
        if not setup_id:
            raise TensorRTQualityChainError(
                f"physical setup identity is missing for backend={backend!r}"
            )
        out.append({"backend": backend, "setup_id": setup_id, "ssh": ssh})
    return out


def _final_report_remote_context_args(
    run_dir: Path, base: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
) -> list[str]:
    """Freeze every DeepX variant dispatch root into a reporter allowlist."""
    records: list[dict[str, Any]] = []
    scopes: dict[tuple[str, str], str] = {}
    for variant in variants:
        merged = _merged_variant(base, variant)
        remote_root = (
            str(
                merged.get("remote_root")
                or "/home/nx/native_fifo_evalsets"
            ).rstrip("/")
            + "/" + run_dir.name
        )
        namespace = str(merged.get("artifact_namespace") or "").strip()
        if namespace:
            remote_root += "/variants/" + namespace
        remote_tool_dir = str(
            merged.get("remote_tool_dir")
            or "/home/nx/ONNX-Splitpoint-Tool"
        ).rstrip("/")
        for binding in _variant_backend_bindings(base, variant):
            if binding["backend"] != "deepx":
                continue
            setup_id = binding["setup_id"]
            scope = (setup_id, remote_root)
            previous = scopes.get(scope)
            if previous is not None and previous != remote_tool_dir:
                raise TensorRTQualityChainError(
                    "DeepX reporter remote context drift for "
                    f"setup={setup_id!r} root={remote_root!r}"
                )
            scopes[scope] = remote_tool_dir
            record = {
                "schema": (
                    "onnx-splitpoint/"
                    "native-final-report-remote-execution-context"
                ),
                "schema_version": 1,
                "setup_id": setup_id,
                "remote_root": remote_root,
                "remote_tool_dir": remote_tool_dir,
            }
            if record not in records:
                records.append(record)
    args: list[str] = []
    for record in records:
        args.extend([
            "--remote-execution-context-json",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        ])
    return args


def _full_backends_for_variant(
    merged: Mapping[str, Any], backend: str,
) -> list[str]:
    full_cfg = (
        merged.get("full_baselines")
        if isinstance(merged.get("full_baselines"), Mapping) else {}
    )
    if not _as_bool(
        full_cfg.get("enabled") or merged.get("native_full_baselines_enabled"),
        False,
    ):
        return []
    producer_map = (
        full_cfg.get("backends_by_producer")
        if isinstance(full_cfg.get("backends_by_producer"), Mapping) else {}
    )
    source = producer_map.get(backend) if backend in producer_map else full_cfg.get("backends")
    defaults = {
        "hailo8": ["hailo8", "tensorrt"],
        "hailo10h": ["hailo10h", "tensorrt"],
        "deepx": ["deepx", "tensorrt"],
    }.get(backend, [])
    if not source:
        return list(defaults)
    aliases = {
        "trt": "tensorrt", "ort_tensorrt": "tensorrt",
        "hailo10": "hailo10h", "hailo_10": "hailo10h",
        "deepx_m1": "deepx", "dx_m1": "deepx",
    }
    out: list[str] = []
    for raw in _as_list(source):
        token = str(raw).strip().lower().replace("-", "_")
        token = aliases.get(token, token)
        if token in defaults and token not in out:
            out.append(token)
    return out


def _evalrun_models(run_dir: Path) -> list[str]:
    models_root = run_dir / "models"
    models = sorted(
        path.name
        for path in models_root.iterdir()
        if path.is_dir() and (path / "benchmark_set").is_dir()
    ) if models_root.is_dir() else []
    if not models:
        raise TensorRTQualityChainError(
            f"no generated BenchmarkSets found below {models_root}"
        )
    return models


def _benchmark_task(run_dir: Path, model_id: str) -> str:
    benchmark_set = run_dir / "models" / model_id / "benchmark_set"
    payload = _read_json(benchmark_set / "benchmark_set.json", {}) or {}
    for value in (
        payload.get("benchmark_task"), payload.get("task"),
        payload.get("model_task"),
        (payload.get("model") or {}).get("task")
        if isinstance(payload.get("model"), Mapping) else "",
    ):
        task = str(value or "").strip().lower()
        if task in {"classification", "detection"}:
            return task
    model = str(model_id).strip().lower()
    if model.startswith("resnet"):
        return "classification"
    if model.startswith("yolo"):
        return "detection"
    raise TensorRTQualityChainError(
        f"benchmark task is missing for Native split model={model_id!r}"
    )


def _all_evalrun_cases(run_dir: Path, model_id: str) -> list[str]:
    """Return cases from the runnable BenchmarkSet, not its container.

    Fresh EvaluationRuns materialize the authoritative suite below
    ``benchmark_set/legacy_suite``.  Older imported runs may still keep case
    directories directly below ``benchmark_set`` (or in ``generated_suite``).
    The Native updater already resolves those layouts; the variant coordinator
    must use the same physical roots while freezing its case map.
    """

    benchmark_set = run_dir / "models" / model_id / "benchmark_set"
    roots = (
        benchmark_set / "legacy_suite",
        benchmark_set / "generated_suite",
        benchmark_set / "suite",
        benchmark_set,
    )
    inspected: list[str] = []
    for root in roots:
        inspected.append(str(root))
        try:
            cases = sorted(
                path.name for path in root.iterdir()
                if path.is_dir() and path.name.startswith("b")
            )
        except OSError:
            cases = []
        if cases:
            return cases
    raise TensorRTQualityChainError(
        "no generated Native split cases found for "
        f"model={model_id!r}; inspected={inspected!r}"
    )


def _effective_variant_case_map(
    run_dir: Path, merged: Mapping[str, Any],
) -> dict[str, list[str]]:
    explicit = _case_map_from_variant(merged)
    model_ids = _evalrun_models(run_dir)
    if explicit:
        unknown = sorted(set(explicit).difference(model_ids))
        if unknown:
            raise TensorRTQualityChainError(
                f"Native variant selects unknown models: {unknown!r}"
            )
        out: dict[str, list[str]] = {}
        for model_id, cases in explicit.items():
            available = set(_all_evalrun_cases(run_dir, model_id))
            selected = [str(case).strip().lower() for case in cases]
            if not selected or any(case not in available for case in selected):
                raise TensorRTQualityChainError(
                    f"Native variant selects missing/empty cases for "
                    f"model={model_id!r}: {selected!r}"
                )
            if len(set(selected)) != len(selected):
                raise TensorRTQualityChainError(
                    f"Native variant contains duplicate cases for model={model_id!r}"
                )
            out[model_id] = selected
        return out

    requested_models = _as_list(merged.get("models") or merged.get("model"))
    if requested_models:
        unknown = sorted(set(requested_models).difference(model_ids))
        if unknown:
            raise TensorRTQualityChainError(
                f"Native variant selects unknown models: {unknown!r}"
            )
        model_ids = requested_models
    return {
        model_id: _all_evalrun_cases(run_dir, model_id)
        for model_id in model_ids
    }


def _split_backend_for_physical(backend: str) -> str:
    canonical = _canonical_native_backend(backend)
    try:
        return {
            "hailo8": "hailo8_to_trt",
            "hailo10h": "hailo10h_to_trt",
            "deepx": "deepx_to_trt",
        }[canonical]
    except KeyError as exc:
        raise TensorRTQualityChainError(
            f"unsupported Native split backend={backend!r}"
        ) from exc


def _split_selected_for_physical(
    merged: Mapping[str, Any], backend: str,
) -> bool:
    """Return whether the sealed profile selected Split on this setup.

    Older direct invocations did not carry ``split_backends`` and retain the
    historical all-physical-backends behaviour.  An explicitly materialised
    empty list, however, is the authoritative Full-only selection and must not
    be expanded back into synthetic Split rows.
    """
    if "split_backends" not in merged:
        return True
    selected = {
        _canonical_native_backend(value)
        for value in _as_list(merged.get("split_backends"))
    }
    return _canonical_native_backend(backend) in selected


def _split_selections_for_variant(
    run_dir: Path, merged: Mapping[str, Any], binding: Mapping[str, str],
) -> list[dict[str, str]]:
    if not _split_selected_for_physical(merged, binding["backend"]):
        return []
    precision = str(merged.get("precision") or "uint8_cast_fp16").strip()
    if not precision:
        raise TensorRTQualityChainError("Native split precision is missing")
    backend = _split_backend_for_physical(binding["backend"])
    selections: list[dict[str, str]] = []
    for model_id, cases in _effective_variant_case_map(run_dir, merged).items():
        task = _benchmark_task(run_dir, model_id)
        for case_id in cases:
            selections.append({
                "model_id": model_id,
                "case_id": case_id,
                "backend": backend,
                "precision": precision,
                "task": task,
            })
    return selections


def _validate_supplied_split_binding_set(
    payload: Any,
    *,
    eval_run_id: str,
    setup_id: str,
    selections: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact setup/selection coverage before remote performance."""

    if not isinstance(payload, Mapping):
        raise TensorRTQualityChainError(
            f"Native split binding set for setup={setup_id!r} is not an object"
        )
    value = copy.deepcopy(dict(payload))
    bindings = value.get("bindings_by_model_case_backend")
    expected_keys = {
        "|".join((
            str(row.get("model_id") or "").strip(),
            str(row.get("case_id") or "").strip().lower(),
            str(row.get("backend") or "").strip(),
        ))
        for row in selections
    }
    if (
        value.get("schema") != SPLIT_BINDING_SET_SCHEMA
        or int(value.get("schema_version") or 0) != 2
        or str(value.get("eval_run_id") or "") != eval_run_id
        or str(value.get("setup_id") or "") != setup_id
        or not isinstance(bindings, Mapping)
        or set(str(key) for key in bindings) != expected_keys
    ):
        raise TensorRTQualityChainError(
            f"Native split binding-set identity/coverage mismatch for "
            f"setup={setup_id!r}"
        )
    declared_set_sha = str(value.get("binding_set_sha256") or "").strip().lower()
    unhashed_set = copy.deepcopy(value)
    unhashed_set.pop("binding_set_sha256", None)
    if (
        len(declared_set_sha) != 64
        or canonical_json_sha256(unhashed_set) != declared_set_sha
        or len(str(value.get("central_quality_summary_sha256") or "").strip())
        != 64
    ):
        raise TensorRTQualityChainError(
            f"Native split binding-set seal mismatch for setup={setup_id!r}"
        )
    by_key = {
        "|".join((
            str(row.get("model_id") or "").strip(),
            str(row.get("case_id") or "").strip().lower(),
            str(row.get("backend") or "").strip(),
        )): row
        for row in selections
    }
    for key in sorted(expected_keys):
        selection = by_key[key]
        raw_binding = bindings.get(key)
        expected_identity = {
            "model": selection.get("model_id"),
            "case": selection.get("case_id"),
            "backend": selection.get("backend"),
            "precision": selection.get("precision"),
            "task": selection.get("task"),
            "setup_id": setup_id,
        }
        validated, status = validate_native_split_quality_binding(
            raw_binding, expected_identity=expected_identity,
            verification_mode="portable",
        )
        if validated is None:
            raise TensorRTQualityChainError(
                f"Native split binding invalid for {key!r}: {status}"
            )
        central_selection, selection_status = (
            validate_central_native_split_quality_selection(
                validated, required=True,
            )
        )
        if central_selection is None:
            raise TensorRTQualityChainError(
                f"Native split Central selection invalid for {key!r}: "
                f"{selection_status}"
            )
        preselection = validated.get("preselection")
        if not isinstance(preselection, Mapping):
            raise TensorRTQualityChainError(
                f"Native split preselection missing for {key!r}"
            )
        exact = {
            "eval_run_id": (validated.get("eval_run_id"), eval_run_id),
            "source_run_id": (
                canonical_native_split_backend(
                    central_selection.get("source_run_id"), setup_id,
                ),
                canonical_native_split_backend(selection.get("backend"), setup_id),
            ),
            "setup_id": (preselection.get("setup_id"), setup_id),
        }
        drift = [name for name, pair in exact.items() if pair[0] != pair[1]]
        if drift:
            raise TensorRTQualityChainError(
                f"Native split exact identity mismatch for {key!r}: {drift!r}"
            )
        bindings[key] = copy.deepcopy(validated)
    value["bindings_by_model_case_backend"] = dict(bindings)
    return value


def _configured_split_binding_set(
    cfg: Mapping[str, Any], *, namespace: str, label: str, setup_id: str,
) -> Any:
    context = _workflow_context(cfg)
    by_variant = cfg.get("native_split_quality_binding_sets_by_variant")
    if not isinstance(by_variant, Mapping):
        by_variant = context.get("native_split_quality_binding_sets_by_variant")
    if isinstance(by_variant, Mapping):
        variant_sets = by_variant.get(namespace)
        if variant_sets is None:
            variant_sets = by_variant.get(label)
        if isinstance(variant_sets, Mapping) and setup_id in variant_sets:
            return variant_sets.get(setup_id)
    by_setup = cfg.get("native_split_quality_binding_sets_by_setup")
    if not isinstance(by_setup, Mapping):
        by_setup = context.get("native_split_quality_binding_sets_by_setup")
    return by_setup.get(setup_id) if isinstance(by_setup, Mapping) else None


def _materialize_native_split_quality_binding_sets(
    run_dir: Path,
    cfg: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    *, allow_missing: bool = False,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Create one canonical, exact-coverage split set per variant/setup."""

    # Resolve applicability before requiring Central Quality input.  A
    # Full-only profile explicitly materialises ``split_backends: []``.  That
    # is a valid empty Split matrix, not a request to rediscover every adapter
    # backend and not a missing-quality error.
    namespaces = {
        _variant_namespace(index, variant): {}
        for index, variant in enumerate(variants)
    }
    split_applicable = any(
        _split_selected_for_physical(
            _merged_variant(cfg, variant), binding["backend"],
        )
        for variant in variants
        for binding in _variant_backend_bindings(cfg, variant)
    )
    if not split_applicable:
        return namespaces, {
            "required": True,
            "applicable": False,
            "status": "not_applicable_full_only",
            "central_quality_summary": "",
            "quality_before_performance": True,
            "engine_rebuild_allowed": False,
            "diagnostic_continue": False,
            "errors_by_variant_setup": {},
            "variants": [],
        }

    context = _workflow_context(cfg)
    summary_text = str(
        cfg.get("central_quality_summary")
        or context.get("central_quality_summary")
        or context.get("central_quality_summary_json")
        or ""
    ).strip()
    summary_path = (
        Path(summary_text).expanduser().resolve()
        if summary_text
        else run_dir / "quality_management" / "central_quality_summary.json"
    )
    if not summary_path.is_file() and not allow_missing:
        raise TensorRTQualityChainError(
            f"central quality summary is missing for Native split: {summary_path}"
        )

    paths_by_variant: dict[str, dict[str, str]] = {}
    plan_rows: list[dict[str, Any]] = []
    errors_by_variant_setup: dict[str, dict[str, str]] = {}
    output_root = run_dir / "reports" / "native_split_quality_binding_sets"
    for index, variant in enumerate(variants):
        namespace = _variant_namespace(index, variant)
        label = _variant_label(index, variant)
        merged = _merged_variant(cfg, variant)
        setup_paths: dict[str, str] = {}
        for physical in _variant_backend_bindings(cfg, variant):
            if not _split_selected_for_physical(
                merged, physical["backend"],
            ):
                continue
            setup_id = physical["setup_id"]
            selections: list[dict[str, Any]] = []
            try:
                selections = _split_selections_for_variant(run_dir, merged, physical)
                expected = load_split_binding_set_from_central_quality_summary(
                    summary_path,
                    eval_run_id=run_dir.name,
                    setup_id=setup_id,
                    selections=selections,
                )
                expected = _validate_supplied_split_binding_set(
                    expected,
                    eval_run_id=run_dir.name,
                    setup_id=setup_id,
                    selections=selections,
                )
                supplied_value = _configured_split_binding_set(
                    cfg, namespace=namespace, label=label, setup_id=setup_id,
                )
                if supplied_value is not None:
                    if isinstance(supplied_value, Mapping):
                        observed = copy.deepcopy(dict(supplied_value))
                    else:
                        supplied_path = Path(str(supplied_value)).expanduser().resolve()
                        if not supplied_path.is_file():
                            raise TensorRTQualityChainError(
                                f"configured Native split binding set does not exist: "
                                f"{supplied_path}"
                            )
                        observed = _strict_json(supplied_path)
                    observed = _validate_supplied_split_binding_set(
                        observed,
                        eval_run_id=run_dir.name,
                        setup_id=setup_id,
                        selections=selections,
                    )
                    if _canonical_json_bytes(expected) != _canonical_json_bytes(observed):
                        raise TensorRTQualityChainError(
                            f"parent Native split binding set drifts from central "
                            f"quality: variant={namespace!r} setup={setup_id!r}"
                        )
                out_path = (
                    output_root / namespace / _safe_component(setup_id)
                    / "native_split_quality_binding_set.json"
                )
                _write_json(out_path, expected)
                setup_paths[setup_id] = str(out_path.resolve())
                plan_rows.append({
                    "variant_index": index,
                    "variant_namespace": namespace,
                    "variant_label": label,
                    "backend": physical["backend"],
                    "setup_id": setup_id,
                    "selection_count": len(selections),
                    "selection_keys": sorted(
                        "|".join((
                            row["model_id"], row["case_id"], row["backend"],
                        ))
                        for row in selections
                    ),
                    "path": str(out_path.resolve()),
                    "sha256": _sha256(out_path),
                    "status": "ready",
                })
            except Exception as exc:
                if not allow_missing:
                    raise
                detail = f"{type(exc).__name__}: {exc}"
                errors_by_variant_setup.setdefault(namespace, {})[setup_id] = detail
                plan_rows.append({
                    "variant_index": index,
                    "variant_namespace": namespace,
                    "variant_label": label,
                    "backend": physical["backend"],
                    "setup_id": setup_id,
                    "selection_count": len(selections),
                    "path": "",
                    "status": "blocked_upstream_quality",
                    "failure_class": "upstream_quality_evidence",
                    "failure_reason": "upstream_central_quality_binding_missing",
                    "error": detail,
                    "transfer_attempted": False,
                })
        if not setup_paths and not allow_missing:
            raise TensorRTQualityChainError(
                f"Native variant={namespace!r} has no physical split setup"
            )
        paths_by_variant[namespace] = setup_paths
    return paths_by_variant, {
        "required": True,
        "applicable": True,
        "status": "required",
        "central_quality_summary": str(summary_path),
        "quality_before_performance": True,
        "engine_rebuild_allowed": False,
        "diagnostic_continue": bool(allow_missing),
        "errors_by_variant_setup": errors_by_variant_setup,
        "variants": plan_rows,
    }


def _validate_supplied_producer_set(
    payload: Any, *, eval_run_id: str, setup_id: str, model_ids: list[str],
) -> dict[str, Any]:
    """Validate a parent-materialised producer set before a variant can use it."""

    if not isinstance(payload, Mapping):
        raise TensorRTQualityChainError(
            f"TensorRT producer set for setup={setup_id!r} is not an object"
        )
    payload = copy.deepcopy(dict(payload))
    producers = payload.get("producers_by_model")
    if (
        payload.get("schema") != PRODUCER_SET_SCHEMA
        or int(payload.get("schema_version") or 0) != 1
        or str(payload.get("eval_run_id") or "") != eval_run_id
        or str(payload.get("setup_id") or "") != setup_id
        or not isinstance(producers, Mapping)
        or set(str(key) for key in producers) != set(model_ids)
    ):
        raise TensorRTQualityChainError(
            f"TensorRT producer set identity/model coverage mismatch for "
            f"setup={setup_id!r}"
        )
    for model_id in model_ids:
        raw = producers.get(model_id)
        if not isinstance(raw, Mapping):
            raise TensorRTQualityChainError(
                f"TensorRT producer missing for setup={setup_id!r} model={model_id!r}"
            )
        task = str(raw.get("task") or "").strip().lower()
        try:
            validated, producer_sha = _validate_candidate_execution_contract(
                raw, role="Native variant TensorRT hand-off", task=task,
            )
        except Exception as exc:
            raise TensorRTQualityChainError(
                f"TensorRT producer is invalid for setup={setup_id!r} "
                f"model={model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        exact = {
            "eval_run_id": eval_run_id,
            "setup_id": setup_id,
            "model_id": model_id,
            "source_run_id": "native_full_tensorrt",
            "case_id": "full",
            "execution_role": "full_quality_only",
            "backend": "native_tensorrt",
            "variant": "full",
            "performance_claims_emitted": False,
        }
        if any(validated.get(key) != value for key, value in exact.items()):
            raise TensorRTQualityChainError(
                f"TensorRT producer role/identity mismatch for "
                f"setup={setup_id!r} model={model_id!r}"
            )
        if str(validated.get("producer_identity_sha256") or "").lower() != str(producer_sha).lower():
            raise TensorRTQualityChainError(
                f"TensorRT producer SHA mismatch for setup={setup_id!r} "
                f"model={model_id!r}"
            )
        producers[model_id] = copy.deepcopy(dict(validated))
    payload["producers_by_model"] = dict(producers)
    return payload


def _materialize_trt_quality_producer_sets(
    run_dir: Path,
    cfg: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    *, allow_missing: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Create exactly one immutable multi-model producer set per setup."""

    eval_run_id = run_dir.name
    setup_requirements: dict[str, dict[str, str]] = {}
    ssh_to_setup: dict[str, str] = {}
    backend_to_remote: dict[str, tuple[str, str]] = {}
    setup_to_backend: dict[str, str] = {}
    occurrences: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        merged = _merged_variant(cfg, variant)
        for binding in _variant_backend_bindings(cfg, variant):
            backend = binding["backend"]
            setup_id = binding["setup_id"]
            ssh = binding["ssh"]
            if ssh and ssh in ssh_to_setup and ssh_to_setup[ssh] != setup_id:
                raise TensorRTQualityChainError(
                    f"physical remote {ssh!r} changes setup identity across variants"
                )
            if ssh:
                ssh_to_setup[ssh] = setup_id
            previous_remote = backend_to_remote.get(backend)
            current_remote = (setup_id, ssh)
            if previous_remote is not None and previous_remote != current_remote:
                raise TensorRTQualityChainError(
                    f"backend={backend!r} changes physical setup/SSH across variants"
                )
            backend_to_remote[backend] = current_remote
            previous_backend = setup_to_backend.get(setup_id)
            if previous_backend is not None and previous_backend != backend:
                raise TensorRTQualityChainError(
                    f"physical setup={setup_id!r} is assigned to multiple Native "
                    f"backends ({previous_backend!r}, {backend!r})"
                )
            setup_to_backend[setup_id] = backend
            full_backends = _full_backends_for_variant(merged, backend)
            if "tensorrt" in full_backends:
                setup_requirements.setdefault(setup_id, binding)
                occurrences.append({
                    "variant_index": index,
                    "backend": backend,
                    "setup_id": setup_id,
                })

    if not setup_requirements:
        return {}, {"required_setups": [], "owners": [], "models": []}

    model_ids = _evalrun_models(run_dir)

    context = _workflow_context(cfg)
    supplied_raw = cfg.get("trt_quality_producer_sets_by_setup")
    if not isinstance(supplied_raw, Mapping):
        supplied_raw = context.get("trt_quality_producer_sets_by_setup")
    supplied = dict(supplied_raw) if isinstance(supplied_raw, Mapping) else {}

    summary_text = str(
        cfg.get("central_quality_summary")
        or context.get("central_quality_summary")
        or context.get("central_quality_summary_json")
        or ""
    ).strip()
    summary_path = (
        Path(summary_text).expanduser().resolve()
        if summary_text else run_dir / "quality_management" / "central_quality_summary.json"
    )
    summary_payload = _strict_json(summary_path) if summary_path.is_file() else None
    if (
        summary_payload is None
        and not set(setup_requirements).issubset(set(supplied))
        and not allow_missing
    ):
        raise TensorRTQualityChainError(
            "central quality summary is missing and the parent did not provide "
            "one producer set for every physical setup"
        )

    output_dir = run_dir / "reports" / "trt_quality_producer_sets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    errors_by_setup: dict[str, str] = {}
    for setup_id in sorted(setup_requirements):
        try:
            expected = None
            if summary_payload is not None:
                expected = producer_set_from_central_quality_summary(
                    summary_payload,
                    eval_run_id=eval_run_id,
                    setup_id=setup_id,
                    model_ids=model_ids,
                )
            supplied_value = supplied.get(setup_id)
            observed = None
            if supplied_value is not None:
                if isinstance(supplied_value, Mapping):
                    observed = copy.deepcopy(dict(supplied_value))
                else:
                    supplied_path = Path(str(supplied_value)).expanduser().resolve()
                    if not supplied_path.is_file():
                        raise TensorRTQualityChainError(
                            f"configured TensorRT producer set does not exist: {supplied_path}"
                        )
                    observed = _strict_json(supplied_path)
                observed = _validate_supplied_producer_set(
                    observed, eval_run_id=eval_run_id, setup_id=setup_id,
                    model_ids=model_ids,
                )
            if expected is not None and observed is not None:
                if _canonical_json_bytes(expected) != _canonical_json_bytes(observed):
                    raise TensorRTQualityChainError(
                        f"parent TensorRT producer set drifts from central quality: "
                        f"setup={setup_id!r}"
                    )
            selected = expected if expected is not None else observed
            selected = _validate_supplied_producer_set(
                selected, eval_run_id=eval_run_id, setup_id=setup_id,
                model_ids=model_ids,
            )
            out_path = output_dir / f"{_safe_component(setup_id)}.json"
            _write_json(out_path, selected)
            output_paths[setup_id] = str(out_path.resolve())
        except Exception as exc:
            if not allow_missing:
                raise
            errors_by_setup[setup_id] = f"{type(exc).__name__}: {exc}"

    # Run TensorRT Full once per setup.  The last variant is the owner so its
    # remote collection cannot be overwritten by a later variant invocation.
    owner_by_setup = {
        setup_id: max(
            occurrence["variant_index"]
            for occurrence in occurrences
            if occurrence["setup_id"] == setup_id
        )
        for setup_id in output_paths
    }
    owners = [
        {**occurrence, "owner": occurrence["variant_index"] == owner_by_setup[occurrence["setup_id"]]}
        for occurrence in occurrences
        if occurrence["setup_id"] in owner_by_setup
    ]
    return output_paths, {
        "required_setups": sorted(setup_requirements),
        "owners": owners,
        "owner_by_setup": owner_by_setup,
        "models": model_ids,
        "central_quality_summary": str(summary_path) if summary_payload is not None else "",
        "diagnostic_continue": bool(allow_missing),
        "errors_by_setup": errors_by_setup,
    }


def _quality_first_variant_plan(
    cfg: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
    producer_paths: Mapping[str, str],
    plan: Mapping[str, Any],
    split_paths_by_variant: Mapping[str, Mapping[str, str]] | None = None,
    split_plan: Mapping[str, Any] | None = None,
    *, allow_missing: bool = False,
) -> list[dict[str, Any]]:
    """Suppress duplicate TRT Full runs while preserving all vendor baselines."""

    owner_by_setup = {
        str(key): int(value)
        for key, value in (plan.get("owner_by_setup") or {}).items()
    }
    prepared: list[dict[str, Any]] = []
    trt_errors = (
        dict(plan.get("errors_by_setup") or {})
        if isinstance(plan.get("errors_by_setup"), Mapping) else {}
    )
    split_applicable = not (
        isinstance(split_plan, Mapping)
        and split_plan.get("applicable") is False
    )
    for index, variant in enumerate(variants):
        namespace = _variant_namespace(index, variant)
        merged = _merged_variant(cfg, variant)
        explicit_map: dict[str, list[str]] = {}
        for binding in _variant_backend_bindings(cfg, variant):
            backend = binding["backend"]
            setup_id = binding["setup_id"]
            backends = _full_backends_for_variant(merged, backend)
            if (
                "tensorrt" in backends
                and (
                    setup_id not in producer_paths
                    or owner_by_setup.get(setup_id) != index
                )
            ):
                backends = [value for value in backends if value != "tensorrt"]
            explicit_map[backend] = backends
        item = copy.deepcopy(dict(variant))
        if explicit_map:
            full_cfg = copy.deepcopy(
                dict(merged.get("full_baselines") or {})
                if isinstance(merged.get("full_baselines"), Mapping) else {}
            )
            full_cfg["enabled"] = bool(
                full_cfg.get("enabled")
                or merged.get("native_full_baselines_enabled")
            )
            full_cfg["backends_by_producer"] = explicit_map
            full_cfg.pop("backends", None)
            item["full_baselines"] = full_cfg
        item["trt_quality_producer_sets_by_setup"] = dict(producer_paths)
        upstream_errors: dict[str, str] = {}
        for binding in _variant_backend_bindings(cfg, variant):
            setup_id = binding["setup_id"]
            if setup_id in trt_errors:
                upstream_errors[f"{setup_id}:tensorrt_full"] = str(trt_errors[setup_id])
        if split_paths_by_variant is not None and split_applicable:
            split_paths = split_paths_by_variant.get(namespace)
            if (not isinstance(split_paths, Mapping) or not split_paths) and not allow_missing:
                raise TensorRTQualityChainError(
                    f"Native split binding sets missing for variant={namespace!r}"
                )
            split_paths = dict(split_paths) if isinstance(split_paths, Mapping) else {}
            item["native_split_quality_binding_sets_by_setup"] = split_paths
            if split_paths:
                item["native_split_quality_required"] = True
            else:
                item["native_split_quality_required"] = False
                item["smoke_diagnostic_skip_native_split"] = bool(allow_missing)
                item["diagnostic_only"] = bool(allow_missing)
                split_plan_errors = (
                    split_plan.get("errors_by_variant_setup")
                    if isinstance(split_plan, Mapping) else None
                )
                if isinstance(split_plan_errors, Mapping):
                    for setup_id, detail in dict(
                        split_plan_errors.get(namespace) or {}
                    ).items():
                        upstream_errors[f"{setup_id}:native_split"] = str(detail)
        elif isinstance(split_plan, Mapping) and not split_applicable:
            item["native_split_quality_required"] = False
            item["native_split_quality_applicable"] = False
            item["native_split_execution_enabled"] = False
        if upstream_errors:
            item["upstream_quality_errors"] = upstream_errors
            item["claim_eligible"] = False
        if allow_missing:
            item["smoke_diagnostic_quality_continue"] = True
            item["diagnostic_only"] = True
            item["claim_eligible"] = False
        item["artifact_namespace"] = namespace
        prepared.append(item)
    return prepared


def _performance_repetitions(cfg: Mapping[str, Any]) -> int:
    """Resolve independent performance repeats; Native Energy is separate."""
    try:
        return max(1, int(cfg.get("repetitions") or 1))
    except (TypeError, ValueError):
        return 1


def _variant_expected_energy_rows(
    run_dir: Path,
    cfg: Mapping[str, Any],
    variants: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], bool, bool]:
    """Flatten the resolved variants into a theoretical Native Energy matrix."""
    rows: list[dict[str, Any]] = []
    setup_ids: dict[str, str] = {}
    full_seen: set[tuple[str, str, str]] = set()
    validation_requested = False
    full_baselines_enabled = False
    for index, variant in enumerate(variants):
        merged = _merged_variant(cfg, variant)
        validation = (
            merged.get("validation")
            if isinstance(merged.get("validation"), Mapping) else {}
        )
        validation_requested = bool(
            validation_requested
            or _as_bool(
                validation.get("enabled")
                or merged.get("native_validation_enabled"),
                False,
            )
        )
        case_map = _effective_variant_case_map(run_dir, merged)
        variant_id = _variant_label(index, variant)
        for physical in _variant_backend_bindings(cfg, variant):
            producer = physical["backend"]
            setup_id = physical["setup_id"]
            setup_ids[producer] = setup_id
            full_backends = _full_backends_for_variant(merged, producer)
            full_baselines_enabled = bool(
                full_baselines_enabled or full_backends
            )
            for model, cases in sorted(case_map.items()):
                if _split_selected_for_physical(merged, producer):
                    for case in cases:
                        rows.append({
                            "execution_mode": "native_split",
                            "backend_key": producer,
                            "backend": producer,
                            "setup_id": setup_id,
                            "model": model,
                            "case": case,
                            "variant": variant_id,
                        })
                for full_backend in full_backends:
                    full_key = (setup_id, model, full_backend)
                    if full_key in full_seen:
                        continue
                    full_seen.add(full_key)
                    rows.append({
                        "execution_mode": "native_full_baseline",
                        "backend_key": producer,
                        "backend": f"native_full_{full_backend}",
                        "setup_id": setup_id,
                        "comparison_backend": producer,
                        "model": model,
                        "case": "full",
                        "variant": variant_id,
                    })
    return (
        rows,
        setup_ids,
        validation_requested,
        full_baselines_enabled,
    )


def _require_native_force_off(cfg: Mapping[str, Any]) -> None:
    for force_key in ("force_rebuild_engines", "native_force_rebuild_engines", "force_rebuild_native_engines"):
        value = cfg.get(force_key, False)
        if type(value) is not bool or value:
            raise TensorRTQualityChainError("productive_force_build_disabled: normal execution forbids force-rebuild flags: " + force_key)
    for variant in cfg.get("variants") or []:
        if isinstance(variant, Mapping):
            _require_native_force_off(variant)


def _build_update_cmd(run_dir: Path, cfg: Mapping[str, Any], variant: Mapping[str, Any], *, refresh_suites: bool, timeout_s: int) -> list[str]:
    merged = _merged_variant(cfg, variant)
    _require_native_force_off(merged)
    execution_contract = verify_native_execution_contract(
        merged["_native_execution_contract"]
    )
    repetitions = int(execution_contract["repetitions"])
    remotes = merged.get("remotes") if isinstance(merged.get("remotes"), Mapping) else {}
    def rem(backend: str, key: str, default: str = "") -> str:
        val = remotes.get(backend) if isinstance(remotes, Mapping) else None
        if isinstance(val, str) and key == "ssh":
            return val
        if isinstance(val, Mapping):
            return str(val.get(key) or default)
        return default

    # ``models`` without an explicit case map is still an exact selection:
    # expand it to the frozen EvalRun cases before invoking the coordinator.
    # Passing ``{}`` together with ``case_map_only`` used to rediscover every
    # model/case and silently enlarge the configured denominator.
    case_map = _effective_variant_case_map(run_dir, merged)
    backends = [
        _canonical_native_backend(value)
        for value in _as_list(merged.get("backends") or ["hailo8"])
    ]
    if any(not value for value in backends) or len(set(backends)) != len(backends):
        raise TensorRTQualityChainError(
            "a Native variant contains empty or duplicate physical backend aliases"
        )
    cmd = [
        sys.executable, "-u", str(_script("update_evalset_native_producers.py")),
        "--eval-run-dir", str(run_dir),
        "--run-native-producers",
        "--backends", ",".join(backends),
        "--case-policy", str(merged.get("case_policy") or "case_map_only"),
        "--case-map", json.dumps(case_map, separators=(",", ":")),
        "--remote-tool-dir", str(merged.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool"),
        "--remote-root", str(merged.get("remote_root") or "/home/nx/native_fifo_evalsets"),
        "--precision", str(merged.get("precision") or "uint8_cast_fp16"),
        "--hailo-format", str(merged.get("hailo_format") or ("float32" if str(merged.get("precision") or "") == "float32_layout_fp16" else "uint8")),
        "--engine-build-python", str(merged.get("engine_build_python") or "auto"),
        "--frames", str(int(execution_contract["frames"])),
        "--warmup", str(int(execution_contract["warmup"])),
        "--repetitions", str(repetitions),
        "--queue-depth", str(int(execution_contract["queue_depth"])),
        "--inflight", str(int(execution_contract["inflight"])),
        "--native-execution-contract-json",
        json.dumps(execution_contract, sort_keys=True, separators=(",", ":")),
        "--timeout", str(timeout_s),
        "--native-telemetry-label", _variant_label(0, variant),
    ]
    if isinstance(merged.get("quality_gate_policy"), Mapping):
        cmd += [
            "--quality-gate-json",
            json.dumps(
                dict(merged.get("quality_gate_policy") or {}),
                sort_keys=True, separators=(",", ":"),
            ),
        ]
    cache_verify_only = _as_bool(merged.get("cache_verify_only"), False)
    if cache_verify_only:
        if _as_bool(merged.get("build_missing_engines"), False):
            raise TensorRTQualityChainError(
                "cache_verify_only forbids build_missing_engines"
            )
        if _as_bool(
            merged.get("force_rebuild_engines")
            or merged.get("native_force_rebuild_engines"), False,
        ):
            raise TensorRTQualityChainError(
                "cache_verify_only forbids force-rebuild flags"
            )
        cmd += ["--artifact-policy", CACHE_VERIFY_ONLY]
    artifact_namespace = str(merged.get("artifact_namespace") or "").strip()
    if artifact_namespace:
        cmd += ["--artifact-namespace", artifact_namespace]
    if _as_bool(merged.get("smoke_diagnostic_quality_continue"), False):
        cmd.append("--smoke-diagnostic-quality-continue")
    for physical in _variant_backend_bindings(cfg, variant):
        setup_flag = {
            "hailo8": "--hailo8-setup-id",
            "hailo10h": "--hailo10-setup-id",
            "deepx": "--deepx-setup-id",
        }.get(physical["backend"])
        if setup_flag:
            cmd += [setup_flag, physical["setup_id"]]
    if refresh_suites:
        cmd.append("--refresh-suites")
    if rem("hailo8", "ssh"):
        cmd += ["--hailo8-ssh", rem("hailo8", "ssh")]
    if rem("hailo10h", "ssh"):
        cmd += ["--hailo10-ssh", rem("hailo10h", "ssh")]
    if rem("deepx", "ssh"):
        cmd += ["--deepx-ssh", rem("deepx", "ssh")]
    if rem("hailo8", "env"):
        cmd += ["--hailo8-env", rem("hailo8", "env")]
    if rem("hailo10h", "env"):
        cmd += ["--hailo10-env", rem("hailo10h", "env")]
    if rem("deepx", "env"):
        cmd += ["--deepx-env", rem("deepx", "env")]
    if rem("hailo8", "remote_base_dir"):
        cmd += [
            "--hailo8-remote-base-dir",
            rem("hailo8", "remote_base_dir"),
        ]
    if rem("hailo10h", "remote_base_dir"):
        cmd += [
            "--hailo10-remote-base-dir",
            rem("hailo10h", "remote_base_dir"),
        ]
    if rem("deepx", "remote_base_dir"):
        cmd += [
            "--deepx-remote-base-dir",
            rem("deepx", "remote_base_dir"),
        ]
    if not _as_bool(merged.get("copy_benchmarksets"), True):
        cmd.append("--no-copy")
    split_quality_sets = merged.get("native_split_quality_binding_sets_by_setup")
    split_quality_applicable = _as_bool(
        merged.get("native_split_quality_applicable"), True,
    )
    if not split_quality_applicable:
        cmd.append("--native-split-quality-not-applicable")
    split_quality_required = _as_bool(
        merged.get("native_split_quality_required"),
        isinstance(split_quality_sets, Mapping) and bool(split_quality_sets),
    )
    if split_quality_required:
        if not isinstance(split_quality_sets, Mapping) or not split_quality_sets:
            raise TensorRTQualityChainError(
                "Quality-FIRST Native split requires setup-local binding sets"
            )
        if _as_bool(
            merged.get("force_rebuild_engines")
            or merged.get("native_force_rebuild_engines"), False,
        ):
            raise TensorRTQualityChainError(
                "Quality-FIRST Native split forbids force-rebuild flags"
            )
        cmd += [
            "--native-split-quality-required",
            "--native-split-quality-binding-sets",
            json.dumps(
                {str(key): str(value) for key, value in split_quality_sets.items()},
                sort_keys=True, separators=(",", ":"),
            ),
        ]
        cmd.append("--no-build-missing-engines")
    elif not _as_bool(merged.get("build_missing_engines"), True):
        cmd.append("--no-build-missing-engines")
    if (
        not split_quality_required
        and _as_bool(
            merged.get("force_rebuild_engines")
            or merged.get("native_force_rebuild_engines"), False,
        )
    ):
        cmd.append("--native-force-rebuild-engines")
    full_cfg = merged.get("full_baselines") if isinstance(merged.get("full_baselines"), Mapping) else {}
    if _as_bool((full_cfg or {}).get("enabled") or merged.get("native_full_baselines_enabled"), False):
        cmd.append("--native-full-baselines")
        full_map = (full_cfg or {}).get("backends_by_producer")
        if not (isinstance(full_map, Mapping) and full_map):
            full_backends = (full_cfg or {}).get("backends")
            if full_backends:
                cmd += ["--native-full-backends", ",".join(_as_list(full_backends))]
        producer_map = (full_cfg or {}).get("backends_by_producer")
        if isinstance(producer_map, Mapping) and producer_map:
            cmd += ["--native-full-backends-by-producer", json.dumps(dict(producer_map), sort_keys=True, separators=(",", ":"))]
        quality_sets = merged.get("trt_quality_producer_sets_by_setup")
        if isinstance(quality_sets, Mapping) and quality_sets:
            cmd += [
                "--trt-quality-producer-sets",
                json.dumps(
                    {str(key): str(value) for key, value in quality_sets.items()},
                    sort_keys=True, separators=(",", ":"),
                ),
            ]
    if _as_bool(merged.get("dump_outputs"), False):
        cmd.append("--dump-outputs")
    vcfg = merged.get("validation") if isinstance(merged.get("validation"), Mapping) else {}
    if _as_bool((vcfg or {}).get("enabled") or merged.get("native_validation"), False):
        cmd.append("--native-validation")
    # v59dz: GUI/native validation needs the exact native input dump for
    # Full-ONNX self-reference.  For Hailo8 variants request boundary dumps
    # automatically whenever native validation is enabled.
    if _as_bool(merged.get("dump_boundary") or merged.get("native_boundary_debug") or (vcfg or {}).get("dump_boundary") or (vcfg or {}).get("enabled"), False):
        cmd.append("--native-boundary-debug")
    if "letterbox_pad_value" in merged or "native_letterbox_pad_value" in merged:
        cmd += ["--native-letterbox-pad-value", str(int(merged.get("native_letterbox_pad_value", merged.get("letterbox_pad_value") or 0) or 0))]
    if "dequant_scale" in merged or "native_dequant_scale" in merged:
        cmd += ["--native-dequant-scale", str(float(merged.get("native_dequant_scale", merged.get("dequant_scale") or 0.0) or 0.0))]
    if "dequant_zero_point" in merged or "native_dequant_zero_point" in merged:
        cmd += ["--native-dequant-zero-point", str(float(merged.get("native_dequant_zero_point", merged.get("dequant_zero_point") or 0.0) or 0.0))]
    if str(merged.get("boundary_layout") or merged.get("native_boundary_layout") or "").strip():
        cmd += ["--native-boundary-layout", str(merged.get("native_boundary_layout", merged.get("boundary_layout")))]
    # v60i: Native energy is executed once after all variants have been
    # combined and semantically validated.  Per-variant energy caused duplicate
    # measurements and allowed unvalidated aliases into the plan.
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-run-dir", required=True)
    ap.add_argument("--config", required=True, help="JSON/YAML native_producers config containing variants.")
    ap.add_argument("--refresh-suites", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse only an exact, completed Native-Performance checkpoint and "
            "continue its managed Native-Energy row journal."
        ),
    )
    ap.add_argument("--report-python", default="auto", help="Python for final native validation. auto prefers current interpreter if onnxruntime works, else .venv-report/bin/python.")
    ap.add_argument("--timeout", type=int, default=7200)
    ns = ap.parse_args()

    run_dir = Path(ns.eval_run_dir).expanduser().resolve()
    cfg_path = Path(ns.config).expanduser().resolve()
    cfg = _read_yaml_or_json(cfg_path)
    try:
        _require_native_force_off(cfg)
    except TensorRTQualityChainError as exc:
        ap.error(str(exc))
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    os.environ["ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR"] = str(reports)
    progress = NativeProgressJournal.from_dir(reports)
    progress.emit("PLAN", phase="native_variant_coordinator", variant_count=0)
    os.environ["ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR"] = str(reports)
    try:
        cfg, quality_gate_policy_source = _bind_evalrun_quality_gate_policy(
            run_dir, cfg,
        )
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "status": "native_quality_policy_preflight_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "transfer_attempted": False,
            "started_remote_count": 0,
        }, indent=2), file=sys.stderr)
        return 2
    cache_verify_only = _as_bool(cfg.get("cache_verify_only"), False)
    if cache_verify_only:
        if ns.resume or ns.refresh_suites:
            raise SystemExit(
                "cache_verify_only requires a fresh, non-refresh Native canary"
            )
        os.environ["ONNX_SPLITPOINT_ARTIFACT_POLICY"] = CACHE_VERIFY_ONLY
        profile_path = run_dir / "profile.yaml"
        if not profile_path.is_file():
            raise SystemExit(
                "cache_verify_only matrix attestation requires EvalRun/profile.yaml"
            )
        profile_payload = _read_yaml_or_json(profile_path)
        profile_payload["native_producers"] = copy.deepcopy(dict(cfg))
        try:
            validate_cache_verify_only_profile(profile_payload)
        except CacheVerifyPolicyError as exc:
            raise SystemExit(str(exc)) from exc
    canonical_stage_path = reports / "native_producer_stage.json"
    coordinator_checkpoint_path = _native_coordinator_checkpoint_path(
        run_dir
    )
    coordinator_input_hash = native_coordinator_input_hash(
        run_dir, cfg_path,
        quality_gate_policy_sha256=(
            AccuracyGatePolicy.from_mapping(
                cfg.get("quality_gate_policy")
            ).sha256()
            if isinstance(cfg.get("quality_gate_policy"), Mapping) else ""
        ),
    )
    if coordinator_checkpoint_path.is_file():
        existing_coordinator, coordinator_reason = load_stage_checkpoint(
            coordinator_checkpoint_path,
            stage="native_coordinator",
            input_hash=coordinator_input_hash,
            run_root=run_dir,
        )
        if existing_coordinator is None:
            print(json.dumps({
                "ok": False,
                "status": "native_coordinator_checkpoint_invalid",
                "reason": coordinator_reason,
            }, indent=2), file=sys.stderr)
            return 70
        if (
            existing_coordinator.get("complete") is True
            and str(existing_coordinator.get("state") or "")
            in {"completed", "failed"}
        ):
            return int(
                (existing_coordinator.get("details") or {}).get(
                    "return_code"
                )
                or (
                    0
                    if existing_coordinator.get("state") == "completed"
                    else 2
                )
            )
        if not ns.resume:
            print(json.dumps({
                "ok": False,
                "status": "native_coordinator_checkpoint_incomplete",
                "reason": "resume_required",
            }, indent=2), file=sys.stderr)
            return 70
    write_stage_checkpoint(
        coordinator_checkpoint_path,
        stage="native_coordinator",
        state="running",
        complete=False,
        input_hash=coordinator_input_hash,
        run_root=run_dir,
        details={"resume_requested": bool(ns.resume)},
    )
    raw_variants = cfg.get("variants") if isinstance(cfg.get("variants"), list) else []
    if not raw_variants:
        raw_variants = [{"id": "default"}]
    if any(not isinstance(value, Mapping) for value in raw_variants):
        failure = {
            "schema": "onnx-splitpoint/native-producer-variant-stage",
            "schema_version": 2,
            "status": "failed",
            "state": "failed",
            "complete": True,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "eval_run_dir": str(run_dir),
            "error": "native producer variants must all be objects",
            "variant_results": [],
        }
        _write_json(canonical_stage_path, failure)
        _write_native_coordinator_terminal(
            run_dir=run_dir,
            reports=reports,
            stage=failure,
            input_hash=coordinator_input_hash,
            return_code=2,
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    variants_input = [dict(value) for value in raw_variants]
    smoke_diagnostic = _smoke_diagnostic_policy(cfg)
    standard_quality_enforced = _standard_quality_enforced_policy(cfg)
    try:
        generated_sets_exist = any(
            path.is_dir() and (path / "benchmark_set").is_dir()
            for path in (run_dir / "models").iterdir()
        ) if (run_dir / "models").is_dir() else False
        if generated_sets_exist and not cache_verify_only:
            split_paths_by_variant, split_quality_plan = (
                _materialize_native_split_quality_binding_sets(
                    run_dir, cfg, variants_input,
                    allow_missing=smoke_diagnostic,
                )
            )
        elif generated_sets_exist:
            # This diagnostic canary proves cache-bound runtime dispatch only.
            # Dataset semantics and Central Quality are deliberately disabled
            # in its frozen profile and must not be re-enabled by the general
            # Quality-FIRST authority resolver.
            split_paths_by_variant = {}
            split_quality_plan = {
                "required": False,
                "applicable": False,
                "status": "not_applicable_cache_verify_only",
                "quality_before_performance": False,
                "engine_rebuild_allowed": False,
                "diagnostic_continue": False,
                "errors_by_variant_setup": {},
                "variants": [],
            }
        else:
            # Compatibility for report/probe-only reprocessing of an already
            # collected native_producers tree. No remote split execution is
            # possible without generated BenchmarkSets.
            split_paths_by_variant = {}
            split_quality_plan = {
                "required": False,
                "applicable": False,
                "status": "report_only_no_generated_benchmark_sets",
                "quality_before_performance": True,
                "engine_rebuild_allowed": False,
                "variants": [],
            }
        producer_paths, quality_first_plan = _materialize_trt_quality_producer_sets(
            run_dir, cfg, variants_input,
            allow_missing=smoke_diagnostic,
        )
        variants = _quality_first_variant_plan(
            cfg, variants_input, producer_paths, quality_first_plan,
            (
                split_paths_by_variant
                if generated_sets_exist and not cache_verify_only
                else None
            ),
            split_quality_plan,
            allow_missing=smoke_diagnostic,
        )
    except Exception as exc:
        failure = {
            "schema": "onnx-splitpoint/native-producer-variant-stage",
            "schema_version": 2,
            "status": "failed",
            "state": "failed",
            "complete": True,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "eval_run_dir": str(run_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "failure_class": "upstream_quality_evidence",
            "failure_reason": "upstream_central_quality_binding_missing",
            "upstream_stage": "central_quality",
            "transfer_attempted": False,
            "variant_results": [],
        }
        _write_json(canonical_stage_path, failure)
        _write_native_coordinator_terminal(
            run_dir=run_dir,
            reports=reports,
            stage=failure,
            input_hash=coordinator_input_hash,
            return_code=2,
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    cfg = copy.deepcopy(dict(cfg))
    cfg["trt_quality_producer_sets_by_setup"] = producer_paths
    cfg["native_split_quality_binding_sets_by_variant"] = split_paths_by_variant
    cfg["variants"] = variants
    initial_split_authority = resolve_native_split_quality_authority(
        run_manifest_path=run_dir / "run_manifest.json",
        stage_path=canonical_stage_path,
    )
    authority_requires_split_quality = bool(
        not cache_verify_only
        and native_split_quality_required_for_row(
            {"backend": "hailo8_to_trt"}, initial_split_authority,
        )
    )
    if authority_requires_split_quality:
        split_quality_plan = dict(split_quality_plan)
        split_quality_plan["required"] = True
    progress.emit("PLAN", phase="native_variant_coordinator", variant_count=len(variants), energy_enabled=_as_bool((cfg.get("energy") or {}).get("enabled") if isinstance(cfg.get("energy"), Mapping) else False))

    stage: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-producer-variant-stage",
        "schema_version": 2,
        "status": "running",
        "state": "running",
        "complete": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_dir.name,
        "eval_run_dir": str(run_dir),
        "workflow_version": str(
            initial_split_authority.get("workflow_version") or ""
        ),
        "tool_version": str(initial_split_authority.get("tool_version") or ""),
        "profile_start_snapshot_sha256": str(
            initial_split_authority.get("profile_start_snapshot_sha256") or ""
        ),
        "profile_selection_snapshot_sha256": str(
            initial_split_authority.get("profile_selection_snapshot_sha256") or ""
        ),
        "profile_selection_fingerprint": str(
            initial_split_authority.get("profile_selection_fingerprint") or ""
        ),
        "config": cfg,
        "tensorrt_quality_first": quality_first_plan,
        "native_split_quality_first": split_quality_plan,
        "trt_quality_producer_sets_by_setup": producer_paths,
        "native_split_quality_binding_sets_by_variant": split_paths_by_variant,
        "quality_error_downstream_policy": (
            "partial_continue_diagnostic" if smoke_diagnostic else "hard_fail"
        ),
        "quality_gate_policy_source": quality_gate_policy_source,
        "diagnostic_only": bool(smoke_diagnostic),
        "claim_eligible": False if smoke_diagnostic else None,
        "artifact_lifecycle": {
            "remote_roots": "variant_isolated",
            "local_roots": "variant_isolated_append_only",
            "cleanup": "retained_through_combined_semantics_and_energy",
        },
        "variant_results": [],
    }
    _write_json(reports / "native_producer_stage_config.json", cfg)
    _write_json(canonical_stage_path, stage)

    phase1_energy_cfg = (
        cfg.get("energy") if isinstance(cfg.get("energy"), Mapping) else {}
    )
    phase1_energy_requested = _as_bool(
        phase1_energy_cfg.get("enabled"), False,
    )
    phase1_context = _workflow_context(cfg)
    phase1_campaign = (
        phase1_context.get("campaign")
        if isinstance(phase1_context.get("campaign"), Mapping) else {}
    )
    phase1_preset = (
        phase1_context.get("execution_preset")
        if isinstance(phase1_context.get("execution_preset"), Mapping) else {}
    )
    phase1_final_requested = _as_bool(
        phase1_context.get("native_energy_final_contract_requested"),
        str(phase1_campaign.get("mode") or "").strip().lower() == "final",
    )
    phase1_run_mode = str(
        phase1_preset.get("id") or phase1_campaign.get("mode") or ""
    ).strip().lower()
    phase1_standard_or_final = phase1_run_mode in {"standard", "final"}
    phase1_top_energy = (
        phase1_context.get("energy")
        if isinstance(phase1_context.get("energy"), Mapping) else {}
    )
    phase1_final_all_split_energy = (
        _final_all_split_energy_requested(
            cfg,
            phase1_energy_cfg,
        )
    )
    phase1_strict_requested = bool(
        not smoke_diagnostic
        and (
            phase1_final_requested
            or _configured_bool(
                (phase1_energy_cfg, "strict"),
                (phase1_top_energy, "strict"),
                default=False,
            )
        )
    )
    phase1_tier = (
        "smoke_diagnostic" if smoke_diagnostic
        else "final_claim" if phase1_final_requested
        else "screening"
    )
    try:
        (
            phase1_expected_rows,
            phase1_setup_ids,
            phase1_validation_requested,
            phase1_full_enabled,
        ) = _variant_expected_energy_rows(run_dir, cfg, variants_input)
    except Exception as exc:
        phase1_expected_rows = []
        phase1_setup_ids = {}
        phase1_validation_requested = False
        phase1_full_enabled = False
        phase1_matrix_error = f"{type(exc).__name__}: {exc}"
    else:
        phase1_matrix_error = ""
    try:
        required_campaign_rows = _native_performance_required_campaign_rows(
            cfg,
            expected_row_count=(
                len(phase1_expected_rows) if not phase1_matrix_error else None
            ),
        )
    except Exception as exc:
        required_campaign_rows = None
        if not phase1_matrix_error:
            phase1_matrix_error = (
                "native_performance_checkpoint_contract_failed:"
                f"{type(exc).__name__}: {exc}"
            )
    native_energy_preflight = build_native_energy_preflight(
        expected_rows=phase1_expected_rows,
        energy_requested=phase1_energy_requested,
        energy_evidence_tier=phase1_tier,
        strict_requested=phase1_strict_requested,
        validation_requested=phase1_validation_requested,
        full_baselines_enabled=phase1_full_enabled,
        setup_ids_by_producer=phase1_setup_ids,
    )
    if phase1_matrix_error:
        native_energy_preflight["plan_viable"] = False
        native_energy_preflight["status"] = (
            "blocked_structural_contradiction"
        )
        native_energy_preflight["errors"] = list(
            native_energy_preflight.get("errors") or []
        ) + [f"variant_matrix_resolution_failed:{phase1_matrix_error}"]
    native_energy_preflight_path = reports / "native_energy_preflight.json"
    _write_json(native_energy_preflight_path, native_energy_preflight)
    stage["native_energy_preflight"] = native_energy_preflight
    stage["native_energy_preflight_json"] = str(
        native_energy_preflight_path
    )
    if phase1_matrix_error:
        # Case/variant discovery is a performance pre-dispatch contract even
        # when Energy is disabled.  Previously the error was only annotated in
        # the Energy preflight, then escaped later as a traceback and left the
        # coordinator checkpoint in ``running``.
        stage["native_energy"] = {
            "enabled": phase1_energy_requested,
            "requested": phase1_energy_requested,
            "status": (
                "blocked_structural_contradiction"
                if phase1_energy_requested else "not_applicable"
            ),
            "complete": True,
            "started_remote_count": 0,
            "started_measurement_count": 0,
        }
        stage.update({
            "status": "failed",
            "state": "failed",
            "complete": True,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "failure_class": "native_selection_preflight",
            "failure_reason": "variant_matrix_resolution_failed",
            "error": phase1_matrix_error,
            "transfer_attempted": False,
            "started_remote_count": 0,
            "started_performance_count": 0,
        })
        _write_json(canonical_stage_path, stage)
        _write_native_coordinator_terminal(
            run_dir=run_dir,
            reports=reports,
            stage=stage,
            input_hash=coordinator_input_hash,
            return_code=2,
        )
        print(json.dumps(stage, indent=2), file=sys.stderr)
        return 2
    if native_energy_preflight_blocks_streaming(
        native_energy_preflight
    ):
        phase1_failed = bool(
            phase1_standard_or_final or phase1_strict_requested
        )
        phase1_status = "failed" if phase1_failed else "partial"
        stage["native_energy"] = {
            "enabled": True,
            "requested": True,
            "status": (
                "failed" if phase1_failed
                else str(native_energy_preflight.get("status") or "")
            ),
            "preflight_status": str(
                native_energy_preflight.get("status") or ""
            ),
            "phase1_preflight": True,
            "plan_viable": False,
            "strict_requested": phase1_strict_requested,
            "strict_failure": bool(
                phase1_failed and phase1_strict_requested
            ),
            "energy_evidence_tier": phase1_tier,
            "energy_tier": phase1_tier,
            "screening_only": phase1_tier == "screening",
            "diagnostic_only": phase1_tier != "final_claim",
            "claim_eligible": False
            if phase1_tier != "final_claim" else None,
            "energy_claim_eligible": False
            if phase1_tier != "final_claim" else None,
            "eligible_for_energy_results_import": False
            if phase1_tier != "final_claim" else None,
            "eligible_for_scientific_claim": False
            if phase1_tier != "final_claim" else None,
            "final_energy_contract_enforced":
                phase1_final_requested,
            "final_all_split_energy_required":
                phase1_final_all_split_energy,
            "started_remote_count": 0,
            "started_performance_count": 0,
            "started_measurement_count": 0,
        }
        stage.update({
            "status": phase1_status,
            "state": "failed" if phase1_failed else "completed",
            "complete": True,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "failure_class": "native_energy_preflight",
            "failure_reason": "native_energy_theoretical_plan_unviable",
            "upstream_stage": "native_energy_phase1_preflight",
            "transfer_attempted": False,
            "started_remote_count": 0,
            "started_performance_count": 0,
        })
        _write_json(canonical_stage_path, stage)
        phase1_return_code = 2 if phase1_failed else 0
        _write_native_coordinator_terminal(
            run_dir=run_dir,
            reports=reports,
            stage=stage,
            input_hash=coordinator_input_hash,
            return_code=phase1_return_code,
        )
        print(json.dumps(stage, indent=2), file=sys.stderr)
        return phase1_return_code

    split_quality_authority = resolve_native_split_quality_authority(
        run_manifest_path=run_dir / "run_manifest.json",
        stage_path=canonical_stage_path,
    )
    stage["native_split_quality_authority"] = split_quality_authority
    if (
        authority_requires_split_quality
        and split_quality_authority.get("valid") is not True
    ):
        if smoke_diagnostic:
            stage.setdefault("warnings", []).append(
                "Native split Quality authority is invalid; affected split paths "
                "remain blocked while setup-local Full diagnostics continue"
            )
            stage["upstream_quality_error"] = {
                "failure_class": "upstream_quality_evidence",
                "failure_reason": "upstream_central_quality_binding_missing",
                "upstream_stage": "central_quality",
                "transfer_attempted": False,
                "errors": list(split_quality_authority.get("errors") or []),
            }
        else:
            stage.update({
                "status": "failed",
                "state": "failed",
                "complete": True,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "failure_class": "upstream_quality_evidence",
                "failure_reason": "upstream_central_quality_binding_missing",
                "upstream_stage": "central_quality",
                "transfer_attempted": False,
                "error": ";".join(
                    str(value)
                    for value in list(split_quality_authority.get("errors") or [])
                ) or "native_split_quality_authority_invalid",
            })
            _write_json(canonical_stage_path, stage)
            _write_native_coordinator_terminal(
                run_dir=run_dir,
                reports=reports,
                stage=stage,
                input_hash=coordinator_input_hash,
                return_code=2,
            )
            print(json.dumps(stage, indent=2), file=sys.stderr)
            return 2

    performance_checkpoint_path, performance_snapshot_path = (
        _native_performance_checkpoint_paths(run_dir)
    )
    performance_input_hash = _native_performance_input_hash(
        run_dir=run_dir,
        cfg=cfg,
        variants=variants,
        expected_rows=phase1_expected_rows,
        split_quality_authority=split_quality_authority,
    )
    performance_reused = False
    performance_resume_reason = "fresh_execution"
    if ns.resume and performance_checkpoint_path.is_file():
        reusable_checkpoint, performance_resume_reason = (
            load_reusable_stage_checkpoint(
                performance_checkpoint_path,
                stage="native_performance",
                input_hash=performance_input_hash,
                run_root=run_dir,
            )
        )
        if reusable_checkpoint is None:
            stage.update({
                "status": "failed",
                "state": "failed",
                "complete": True,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "failure_class": "native_performance_checkpoint",
                "failure_reason": performance_resume_reason,
                "transfer_attempted": False,
            })
            _write_json(canonical_stage_path, stage)
            _write_native_coordinator_terminal(
                run_dir=run_dir,
                reports=reports,
                stage=stage,
                input_hash=coordinator_input_hash,
                return_code=2,
            )
            print(json.dumps(stage, indent=2), file=sys.stderr)
            return 2
        snapshot_payload = _read_json(performance_snapshot_path, {}) or {}
        frozen_stage = (
            snapshot_payload.get("stage")
            if isinstance(snapshot_payload, Mapping) else {}
        )
        if not isinstance(frozen_stage, Mapping):
            stage.update({
                "status": "failed",
                "state": "failed",
                "complete": True,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "failure_class": "native_performance_checkpoint",
                "failure_reason": "native_performance_snapshot_invalid",
                "transfer_attempted": False,
            })
            _write_json(canonical_stage_path, stage)
            _write_native_coordinator_terminal(
                run_dir=run_dir,
                reports=reports,
                stage=stage,
                input_hash=coordinator_input_hash,
                return_code=2,
            )
            return 2
        stage = copy.deepcopy(dict(frozen_stage))
        stage["status"] = "running"
        stage["state"] = "running"
        stage["complete"] = False
        stage.pop("finished_at", None)
        stage["native_performance_checkpoint"] = {
            "reused": True,
            "reason": performance_resume_reason,
            "path": str(performance_checkpoint_path),
            "input_hash": performance_input_hash,
        }
        performance_reused = True
        _restore_native_performance_artifacts(
            reports, performance_snapshot_path,
        )
        _write_json(canonical_stage_path, stage)
    else:
        write_stage_checkpoint(
            performance_checkpoint_path,
            stage="native_performance",
            state="running",
            complete=False,
            input_hash=performance_input_hash,
            run_root=run_dir,
            details={
                "expected_row_count": len(phase1_expected_rows),
                "variant_count": len(variants),
                "resume_requested": bool(ns.resume),
                "resume_reason": (
                    "checkpoint_missing" if ns.resume else "fresh_execution"
                ),
            },
        )

    for i, v0 in enumerate([] if performance_reused else variants):
        if not isinstance(v0, Mapping):
            continue
        label = _variant_label(i, v0)
        merged = _merged_variant(cfg, v0)
        repetitions = _performance_repetitions(merged)
        cmd = _build_update_cmd(run_dir, cfg, v0, refresh_suites=bool(ns.refresh_suites and i == 0), timeout_s=int(ns.timeout))
        rec = {"id": label, "case_map": _effective_variant_case_map(run_dir, merged), "performance_repetitions": repetitions, "native_execution_contract": dict(merged["_native_execution_contract"]), "native_execution_contract_sha256": str(merged["_native_execution_contract"]["contract_sha256"]), "cmd": cmd, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        rr = _run(cmd, timeout=max(300, int(ns.timeout) * repetitions + 120), cwd=ROOT, label=f"variant:{label}")
        rec.update(rr)
        rec["ok"] = rr.get("rc") == 0
        rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        stage["variant_results"].append(rec)
        _write_json(reports / "native_producer_variant_stage.partial.json", stage)
        _write_json(canonical_stage_path, stage)
        # A terminal child failure is row-local.  Every remaining variant must
        # still reach its own terminal record so successful rows and Full
        # baselines are preserved for partial Energy observation.

    # Build final recursive report over all collected backend roots.
    _write_json(canonical_stage_path, stage)
    roots = []
    native_root = run_dir / "native_producers"
    if native_root.exists():
        for child in sorted(native_root.iterdir()):
            if child.is_dir():
                roots.append(child)
    report_cmd = [sys.executable, "-u", str(_script("native_producer_final_report.py"))]
    for r in roots:
        report_cmd += ["--root", str(r)]
    report_cmd += ["--recursive", "--out-dir", str(reports)]
    report_cmd += _final_report_remote_context_args(
        run_dir, cfg, variants,
    )
    if performance_reused:
        fr = dict(stage.get("final_report") or {})
    else:
        fr = _run(report_cmd, timeout=300, cwd=ROOT, label="final_report") if roots else {"rc": 0, "stdout_tail": "no native_producers roots", "stderr_tail": ""}
        stage["final_report"] = fr
    # Keep legacy names synchronized for downstream dashboard/report code.
    for ext in ("json", "csv", "md"):
        src = reports / f"native_producer_combined_summary.{ext}"
        dst = reports / f"native_producer_summary.{ext}"
        if src.is_file():
            dst.write_bytes(src.read_bytes())

    # Native Performance has its own commit point.  Validation, probe and
    # Energy are downstream consumers and can never erase or downgrade this
    # exact expected-row import once it has been committed.
    if not performance_reused:
        imported_summary = (
            _read_json(reports / "native_producer_summary.json", {}) or {}
        )
        imported_rows = [
            row for row in list(imported_summary.get("rows") or [])
            if isinstance(row, Mapping)
        ]
        performance_matrix = _native_performance_expected_matrix(
            phase1_expected_rows,
            imported_rows,
            setup_local_tensorrt_errors=(
                quality_first_plan.get("errors_by_setup")
                if isinstance(quality_first_plan, Mapping) else None
            ),
        )
        _write_json(
            reports / "native_expected_matrix.json",
            performance_matrix,
        )
        stage["expected_matrix"] = performance_matrix
        # The first report discovers actual rows.  Rebuild it immediately
        # against the now-sealed expected denominator so a collected 12/18
        # subset cannot publish itself as complete merely because all twelve
        # observed rows succeeded.  This also keeps JSON and Markdown mirrors
        # in lockstep before validation consumes them.
        matrix_bound_cmd = [
            *report_cmd,
            "--expected-matrix",
            str(reports / "native_expected_matrix.json"),
        ]
        matrix_bound_report = (
            _run(
                matrix_bound_cmd, timeout=300, cwd=ROOT,
                label="matrix_bound_final_report",
            )
            if roots else {
                "rc": 0,
                "stdout_tail": "no native_producers roots",
                "stderr_tail": "",
            }
        )
        stage["unbound_final_report"] = fr
        stage["matrix_bound_final_report"] = matrix_bound_report
        stage["final_report"] = matrix_bound_report
        fr = matrix_bound_report
        for ext in ("json", "csv", "md"):
            src = reports / f"native_producer_combined_summary.{ext}"
            dst = reports / f"native_producer_summary.{ext}"
            if src.is_file():
                dst.write_bytes(src.read_bytes())
        performance_completion = _native_performance_completion(
            reports=reports,
            expected_rows=phase1_expected_rows,
            variant_count=len(variants),
            stage=stage,
            final_report=fr,
            required_campaign_rows=(
                required_campaign_rows
            ),
        )
        stage["native_performance_checkpoint"] = {
            "reused": False,
            "reason": "fresh_execution",
            "path": str(performance_checkpoint_path),
            "input_hash": performance_input_hash,
            **performance_completion,
        }
        _write_json(
            performance_snapshot_path,
            {
                "schema": "onnx-splitpoint/native-performance-stage-snapshot",
                "schema_version": 1,
                "input_hash": performance_input_hash,
                "completion": performance_completion,
                "stage": stage,
            },
        )
        checkpoint_terminal = bool(
            performance_completion.get("checkpoint_terminal_valid")
        )
        performance_matrix_complete = bool(
            performance_completion.get("performance_matrix_complete")
        )
        performance_artifacts = [performance_snapshot_path]
        if checkpoint_terminal:
            _freeze_native_performance_artifacts(
                reports, performance_snapshot_path,
            )
            performance_artifacts = _native_performance_artifacts(
                performance_snapshot_path,
            )
        write_stage_checkpoint(
            performance_checkpoint_path,
            stage="native_performance",
            state="completed" if checkpoint_terminal else "failed",
            complete=True,
            input_hash=performance_input_hash,
            run_root=run_dir,
            artifacts=performance_artifacts,
            details={
                **performance_completion,
                "snapshot": str(
                    performance_snapshot_path.relative_to(run_dir)
                ),
            },
            error=(
                "" if checkpoint_terminal
                else "native performance parent import was not terminal and unambiguous"
            ),
            started_at=str(stage.get("started_at") or ""),
        )
        if not checkpoint_terminal:
            stage.update({
                "status": "failed",
                "state": "failed",
                "complete": True,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "failure_class": "native_performance_checkpoint",
                "failure_reason": "native_performance_matrix_or_import_incomplete",
                "upstream_stage": "native_performance",
            })
            _write_json(canonical_stage_path, stage)
            _write_native_coordinator_terminal(
                run_dir=run_dir,
                reports=reports,
                stage=stage,
                input_hash=coordinator_input_hash,
                return_code=2,
            )
            print(json.dumps(stage, indent=2), file=sys.stderr)
            return 2
        if not performance_matrix_complete:
            stage.update({
                "status": "partial",
                "state": "running",
                "complete": False,
                "performance_matrix_complete": False,
                "scientific_coverage_complete": False,
                "claim_eligible": False,
                "performance_claim_eligible": False,
                "energy_claim_eligible": False,
                "scientific_claim_eligible": False,
                "partial_runtime_policy": (
                    "continue_validation_and_energy_runtime_observations"
                ),
            })
            stage.setdefault("warnings", []).append(
                "Native Performance reached a durable terminal partial "
                "matrix; successful rows continue to validation/Energy, "
                "while all cohort claims remain disabled"
            )
            _write_json(canonical_stage_path, stage)

    # Output dumps are runtime evidence, not an implicit request for semantic
    # validation.  In particular the cache canary freezes validation=false and
    # must not launch ORT after its one Native row.
    validation_cfg = (
        cfg.get("validation")
        if isinstance(cfg.get("validation"), Mapping) else {}
    )
    native_validation_requested = _as_bool(
        validation_cfg.get("enabled")
        or cfg.get("native_validation_enabled")
        or cfg.get("native_validation"),
        False,
    )
    summary_json = reports / "native_producer_combined_summary.json"
    if not summary_json.is_file():
        summary_json = reports / "native_producer_summary.json"
    if native_validation_requested and summary_json.is_file():
        report_py, report_py_meta = _select_report_python(
            ROOT, ns.report_python,
        )
        stage["report_python"] = report_py_meta
        vout = reports / "native_validation"
        val_cmd = [report_py, "-u", str(_script("native_producer_validate_visualize.py")), "--summary", str(summary_json), "--out-dir", str(vout), "--topk", str(int(validation_cfg.get("topk") or 5))]
        if isinstance(cfg.get("quality_gate_policy"), Mapping):
            val_cmd += ["--quality-gate-json", json.dumps(dict(cfg.get("quality_gate_policy") or {}), sort_keys=True, separators=(",", ":"))]
        central_quality_summary = (
            run_dir / "quality_management" / "central_quality_summary.json"
        )
        if central_quality_summary.is_file():
            val_cmd += [
                "--central-quality-summary", str(central_quality_summary),
            ]
        for r in roots:
            val_cmd += ["--root", str(r)]
        vr = _run(val_cmd, timeout=int(validation_cfg.get("timeout_s") or 600), cwd=ROOT, label="native_validation")
        stage["native_validation"] = vr
    elif native_validation_requested:
        stage["report_python"] = {
            "status": "not_selected",
            "reason": "native_producer_summary.json missing",
        }
        stage["native_validation"] = {"rc": 1, "status": "skipped", "error": "native_producer_summary.json missing"}
    else:
        stage["report_python"] = {
            "status": "not_applicable",
            "reason": "native_validation_not_requested",
        }
        stage["native_validation"] = {
            "enabled": False,
            "requested": False,
            "rc": 0,
            "status": "not_applicable",
            "complete": True,
            "technical_error_count": 0,
            "empty_output_error": False,
        }

    # Bind report claim eligibility to the exact validation result.  The
    # ungated report remains fail-closed when validation did not complete.
    quality_summary = reports / "native_validation" / "native_producer_validation_summary.json"
    native_validation_payload = _read_json(quality_summary, {}) or {}
    native_validation_technical_errors = int(
        native_validation_payload.get("technical_error_count") or 0
    )
    validation_rows = list(native_validation_payload.get("rows") or [])
    native_validation_empty_output = bool(
        quality_summary.is_file()
        and (
            native_validation_payload.get("empty_output_error") is True
            or int(native_validation_payload.get("row_count") or len(validation_rows)) == 0
        )
    )
    if native_validation_empty_output:
        native_validation_technical_errors = max(
            1, native_validation_technical_errors,
        )
    native_validation_failed = bool(
        native_validation_requested
        and (
            int((stage.get("native_validation") or {}).get("rc") or 0) != 0
            or not quality_summary.is_file()
        )
    )
    stage["native_validation"]["technical_error_count"] = (
        native_validation_technical_errors
    )
    stage["native_validation"]["empty_output_error"] = (
        native_validation_empty_output
    )
    if (
        native_validation_requested
        and roots
        and (stage.get("native_validation") or {}).get("rc") == 0
        and quality_summary.is_file()
    ):
        gated_cmd = [sys.executable, "-u", str(_script("native_producer_final_report.py"))]
        for root in roots:
            gated_cmd += ["--root", str(root)]
        gated_cmd += [
            "--recursive", "--out-dir", str(reports),
            "--quality-summary", str(quality_summary),
            "--expected-matrix",
            str(reports / "native_expected_matrix.json"),
        ]
        gated_cmd += _final_report_remote_context_args(
            run_dir, cfg, variants,
        )
        gated = _run(gated_cmd, timeout=300, cwd=ROOT, label="quality_gated_final_report")
        stage["quality_gated_final_report"] = gated
        if gated.get("rc") == 0:
            for ext in ("json", "csv", "md"):
                src = reports / f"native_producer_combined_summary.{ext}"
                dst = reports / f"native_producer_summary.{ext}"
                if src.is_file():
                    dst.write_bytes(src.read_bytes())
    else:
        stage["quality_gated_final_report"] = {
            "rc": None,
            "status": (
                "not_applicable" if not native_validation_requested
                else "skipped_fail_closed"
            ),
            "reason": (
                "native_validation_not_requested"
                if not native_validation_requested
                else "native_validation_not_successful_or_summary_missing"
            ),
            "quality_summary": str(quality_summary),
        }

    # v2.62: The marker-v2 versus historical-window A/B probe is deliberately
    # independent of semantic validation and Native Energy pairing.  Run it
    # after the validation attempt even when validation failed, because its
    # only prerequisite is one successful representative native runtime row.
    validation_summary = reports / "native_validation" / "native_producer_validation_summary.json"
    try:
        probe_block, probe_strict_failure = _run_window_method_probe(
            cfg=cfg,
            summary_json=summary_json,
            validation_summary=validation_summary,
            run_dir=run_dir,
            reports=reports,
        )
    except Exception as exc:
        ecfg_for_probe = cfg.get("energy") if isinstance(cfg.get("energy"), Mapping) else {}
        pcfg = ecfg_for_probe.get("window_method_validation_probe") if isinstance(ecfg_for_probe.get("window_method_validation_probe"), Mapping) else {}
        probe_enabled = _as_bool(pcfg.get("enabled"), False)
        probe_strict = _as_bool(pcfg.get("strict"), True)
        probe_strict_validation_failure = bool(probe_enabled and probe_strict)
        probe_workflow_blocking = bool(
            probe_strict_validation_failure and _window_probe_final_campaign(cfg)
        )
        probe_strict_failure = probe_workflow_blocking
        probe_block = {
            "enabled": probe_enabled,
            "requested": probe_enabled,
            "screening_only": True,
            "diagnostic_only": True,
            "eligible_for_energy_results_import": False,
            "eligible_for_scientific_claim": False,
            "affects_final_energy_gate": False,
            "workflow_blocking_scope": "none",
            "workflow_blocking_requested": bool(
                probe_enabled and probe_strict and _window_probe_final_campaign(cfg)
            ),
            "status": "probe_orchestration_failed",
            "ok": False,
            "complete": False,
            "decision_capable": False,
            "started_measurement_count": 0,
            "successful_measurement_count": 0,
            "strict_requested": probe_strict,
            "strict_validation_failure": probe_strict_validation_failure,
            "strict_failure": probe_strict_failure,
            "error": f"{type(exc).__name__}: {exc}",
        }
    stage["window_method_validation_probe"] = probe_block
    _write_json(canonical_stage_path, stage)

    # Execute Native Energy once, after the final combined semantic validation.
    ecfg = cfg.get("energy") if isinstance(cfg.get("energy"), Mapping) else {}
    native_energy_strict_failure = False
    if _as_bool((ecfg or {}).get("enabled"), False):
        mode = str((ecfg or {}).get("mode") or "plan").strip().lower()
        if mode not in {"plan", "measure"}:
            mode = "plan"
        try:
            from onnx_splitpoint_tool.energy.config import load_energy_defaults as _load_energy_defaults
            default_duration = float(getattr(_load_energy_defaults(), "native_energy_duration_s", 60.0) or 60.0)
        except Exception:
            default_duration = 60.0
        duration_s = max(1.0, float((ecfg or {}).get("duration_s") or default_duration))
        timeout_s = int((ecfg or {}).get("timeout") or 900)
        eout = reports / ("native_energy_measurements" if mode == "measure" else "native_energy_plan")
        contract = _energy_contract_context(cfg, ecfg, run_dir=run_dir, reports=reports)
        energy_runs = _energy_repeat_count(cfg, ecfg)
        energy_limit = int(ecfg.get("limit") or ecfg.get("max_rows") or cfg.get("native_energy_limit") or 0)
        allow_unpaired = bool(
            smoke_diagnostic
            or _as_bool(ecfg.get("allow_unpaired") or ecfg.get("diagnostic_unpaired"), False)
        )
        context = _workflow_context(cfg)
        top_energy = context.get("energy") if isinstance(context.get("energy"), Mapping) else {}
        final_all_split_energy = (
            _final_all_split_energy_requested(cfg, ecfg)
        )
        screening_energy = bool(
            not smoke_diagnostic
            and not contract["final_energy_contract_enforced"]
        )
        configured_strict = _configured_bool(
            (ecfg, "strict"),
            (top_energy, "strict"),
            default=False,
        )
        native_energy_strict = bool(
            not smoke_diagnostic
            and (
                contract["final_energy_contract_enforced"]
                or configured_strict
            )
        )
        script = "run_native_producer_energy_from_summary.py" if mode == "measure" else "native_producer_energy_plan.py"
        energy_cmd = [
            sys.executable, "-u", str(_script(script)),
            "--summary", str(summary_json),
            "--validation-summary", str(validation_summary),
            "--out-dir", str(eout),
            *_common_energy_args(cfg, run_dir=run_dir, duration_s=duration_s, timeout_s=timeout_s),
            "--runs", str(energy_runs),
            "--physical-scope", str(contract["physical_scope"]),
            "--window-label", str(contract["window_label"]),
            "--calibration-manifest", str(contract["calibration_manifest"]),
            "--calibration-sha256", str(contract["calibration_sha256"]),
            "--pipeline-contract-manifest", str(contract["pipeline_contract_manifest"]),
            "--pipeline-contract-sha256", str(contract["pipeline_contract_sha256"]),
            "--model-hash-map", str(contract["model_hash_map"]),
            "--model-hash-map-sha256", str(contract["model_hash_map_sha256"]),
        ]
        energy_cmd += _energy_window_ab_cli_args(cfg, ecfg)
        measure_all_runtime_successful = True
        energy_cmd.append("--measure-all-runtime-successful")
        if contract["final_energy_contract_enforced"]:
            energy_cmd += ["--require-runtime-work-units", "--require-command-window-alignment"]
        if allow_unpaired:
            energy_cmd.append("--allow-unpaired")
        if smoke_diagnostic:
            energy_cmd.append("--smoke-diagnostic")
        if screening_energy:
            energy_cmd.append("--screening-energy")
        if final_all_split_energy:
            energy_cmd.append("--final-all-split-energy")
        if mode == "measure" and energy_limit > 0:
            energy_cmd += ["--limit", str(energy_limit)]
        if mode == "measure" and ns.resume:
            energy_cmd.append("--resume-checkpoint")
        energy_result = _run(energy_cmd, timeout=(max(1800, timeout_s * max(1, len(variants)) + 900) if mode == "measure" else 300), cwd=ROOT, label=f"native-energy:{mode}")
        energy_transport_rc = int(energy_result.get("rc") or 0)
        energy_stage_payload: dict[str, Any] = {}
        if mode == "measure":
            energy_stage_path = (
                eout / "stages" / "native_energy" / "stage_result.json"
            )
            raw_energy_stage = _read_json(energy_stage_path, {}) or {}
            if isinstance(raw_energy_stage, Mapping):
                recorded_input_hash = str(
                    raw_energy_stage.get("input_hash") or ""
                )
                if recorded_input_hash:
                    loaded_energy_stage, _energy_stage_reason = (
                        load_stage_checkpoint(
                            energy_stage_path,
                            stage="native_energy",
                            input_hash=recorded_input_hash,
                            run_root=eout,
                        )
                    )
                    if loaded_energy_stage is not None:
                        energy_stage_payload = dict(loaded_energy_stage)
        if (
            energy_stage_payload.get("complete") is True
            and str(energy_stage_payload.get("state") or "")
            in {"completed", "failed"}
        ):
            terminal_rc = int(
                (energy_stage_payload.get("details") or {}).get(
                    "return_code"
                )
                or (
                    0
                    if energy_stage_payload.get("state") == "completed"
                    else 2
                )
            )
            energy_result = {
                **energy_result,
                "transport_rc": energy_transport_rc,
                "rc": terminal_rc,
                "returncode": terminal_rc,
                "terminal_stage_imported": True,
                "terminal_stage": energy_stage_payload,
            }
        plan_path = eout / "plan" / "native_producer_energy_plan.json" if mode == "measure" else eout / "native_producer_energy_plan.json"
        plan_payload = _read_json(plan_path, {}) or {}
        result_path = eout / "native_producer_energy_results.json"
        result_payload = _read_json(result_path, {}) or {}
        plan_preflight = (
            dict(plan_payload.get("preflight") or {})
            if isinstance(plan_payload, Mapping) else {}
        )
        plan_preflight_status = str(
            plan_payload.get("preflight_status")
            or plan_preflight.get("status")
            or (
                "passed" if plan_payload.get("rows")
                else "blocked_no_runtime_constructible_rows"
                if measure_all_runtime_successful
                else "blocked_no_complete_pairs"
            )
        )
        started = int(result_payload.get("started_measurement_count") or 0)
        complete = result_payload.get("complete") is True if mode == "measure" else int(energy_result.get("rc") or 0) == 0
        energy_status = (
            plan_preflight_status
            if plan_preflight_status != "passed"
            else "failed" if int(energy_result.get("rc") or 0) != 0
            else "blocked_zero_measurements_started" if mode == "measure" and started == 0
            else "ok"
        )
        native_energy_strict_failure = bool(
            mode == "measure" and native_energy_strict and (
                int(energy_result.get("rc") or 0) != 0
                or started == 0
                or not complete
                or result_payload.get("ok") is not True
            )
        )
        stage["native_energy"] = {
            **energy_result,
            "enabled": True,
            "mode": mode,
            "out_dir": str(eout),
            "validation_summary": str(validation_summary),
            "semantic_gate_required": False,
            "quality_claim_gate_required": True,
            "measure_all_runtime_successful": (
                measure_all_runtime_successful
            ),
            "measurement_admission_policy": (
                "all_runtime_successful_constructible_native_rows"
            ),
            "status": energy_status,
            "started_measurement_count": started,
            "complete": complete,
            "strict_requested": native_energy_strict,
            "strict_failure": native_energy_strict_failure,
            "final_all_split_energy_required":
                final_all_split_energy,
            "screening_only": screening_energy,
            "screening_energy": screening_energy,
            "energy_evidence_tier": (
                "screening" if screening_energy else
                "smoke_diagnostic" if smoke_diagnostic else "final_claim"
            ),
            "energy_tier": (
                "screening" if screening_energy else
                "smoke_diagnostic" if smoke_diagnostic else "final_claim"
            ),
            "diagnostic_only": bool(
                screening_energy or smoke_diagnostic
            ),
            "claim_eligible": (
                False if screening_energy or smoke_diagnostic else None
            ),
            "energy_claim_eligible": (
                False if screening_energy or smoke_diagnostic else None
            ),
            "eligible_for_energy_results_import": (
                False if screening_energy or smoke_diagnostic else None
            ),
            "eligible_for_scientific_claim": (
                False if screening_energy or smoke_diagnostic else None
            ),
            "plan_preflight_status": plan_preflight_status,
            "plan_preflight": plan_preflight,
            "runs_per_row": energy_runs,
            "limit": energy_limit,
            "pairing_policy": str(plan_payload.get("pairing_policy") or ("allow_unpaired" if allow_unpaired else "paired_only")),
            "planned_rows": len(list(plan_payload.get("rows") or [])),
            "missing_pair_count": len(list(plan_payload.get("paired_missing_rows") or [])),
            "plan_json": str(plan_path) if plan_path.is_file() else "",
            "results_json": str(result_path) if result_path.is_file() else "",
            **contract,
        }
    else:
        stage["native_energy"] = {
            "enabled": False,
            "requested": False,
            "status": "not_applicable",
            "ok": True,
            "complete": True,
            "strict_requested": False,
            "strict_failure": False,
            "reason": "not_requested",
        }

    ok_variants = sum(1 for r in stage.get("variant_results", []) if r.get("ok"))
    total_variants = len(stage.get("variant_results", []))
    has_upstream_quality_gap = bool(
        stage.get("upstream_quality_error")
        or (split_quality_plan.get("errors_by_variant_setup") or {})
        or (quality_first_plan.get("errors_by_setup") or {})
    )
    validation_json = _read_json(reports / "native_validation" / "native_producer_validation_summary.json", {}) or {}
    stage["summary"] = {
        "variant_count": total_variants,
        "variant_ok_count": ok_variants,
        "native_report_rc": fr.get("rc"),
        "native_validation_rc": (stage.get("native_validation") or {}).get("rc"),
        "native_validation_technical_error_count": native_validation_technical_errors,
        "claim_ok_count": validation_json.get("claim_ok_count"),
        "semantic_ok_count": validation_json.get("semantic_ok_count"),
        "window_method_validation_probe_status": (stage.get("window_method_validation_probe") or {}).get("status"),
        "window_method_validation_probe_strict_failure": probe_strict_failure,
        "native_energy_status": (stage.get("native_energy") or {}).get("status"),
        "native_energy_strict_failure": native_energy_strict_failure,
        "upstream_quality_gap": has_upstream_quality_gap,
    }
    stage["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    energy_ok = not bool((stage.get("native_energy") or {}).get("enabled")) or int((stage.get("native_energy") or {}).get("rc") or 0) == 0
    base_ok = bool(ok_variants == total_variants and fr.get("rc") == 0 and (stage.get("native_validation") or {}).get("rc") == 0 and energy_ok)
    stage["status"] = (
        "failed" if (
            probe_strict_failure or native_energy_strict_failure
            or (
                (native_validation_technical_errors > 0 or native_validation_failed)
                and standard_quality_enforced
            )
        )
        else "partial" if (
            has_upstream_quality_gap
            or native_validation_technical_errors > 0
            or native_validation_failed
        )
        else "ok" if base_ok
        else "partial"
    )
    stage["state"] = "failed" if stage["status"] == "failed" else "completed"
    stage["complete"] = True
    if (
        stage["status"] == "failed"
        and (native_validation_technical_errors > 0 or native_validation_failed)
        and standard_quality_enforced
    ):
        stage["failure_class"] = "technical_quality_error"
        stage["failure_reason"] = (
            "native_quality_chain_empty"
            if native_validation_empty_output
            else "native_quality_chain_invalid"
        )
        stage["upstream_stage"] = "native_validation"
        stage["claim_eligible"] = False
    final_return_code = 0 if stage["status"] == "ok" else 2
    _write_json(canonical_stage_path, stage)
    _write_native_coordinator_terminal(
        run_dir=run_dir,
        reports=reports,
        stage=stage,
        input_hash=coordinator_input_hash,
        return_code=final_return_code,
    )
    print(json.dumps({"ok": stage["status"] == "ok", "status": stage["status"], "eval_run_dir": str(run_dir), "stage": str(reports / "native_producer_stage.json"), "summary": stage.get("summary")}, indent=2))
    return final_return_code


if __name__ == "__main__":
    raise SystemExit(main())
