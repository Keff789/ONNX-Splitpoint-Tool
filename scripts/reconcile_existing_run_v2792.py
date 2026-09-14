#!/usr/bin/env python3
from __future__ import annotations

"""Read-only reconciliation of an existing EvaluationRun with v2.79.2 rules.

The source run is never modified.  The tool retains every provenance
representation, groups only exact request/artifact mirrors, joins Central
Quality primarily by source-request SHA, distinguishes Quality-N/A P2 rows
from blocked or missing completed-task rows, and writes a new projection beside
rather than inside the source run.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.validation.detection_records import (
    normalize_detection_records,
)
from onnx_splitpoint_tool.workflow.central_quality_join import (
    is_companion_result,
    join_quality_results_by_request_sha,
)
from onnx_splitpoint_tool.workflow.evidence_state_model import (
    project_evidence_state,
    summarize_evidence_states,
)
from onnx_splitpoint_tool.workflow.logical_measurement import (
    annotate_logical_measurements,
    canonical_backend,
    canonical_run_id,
    direct_setup_ids,
    selected_variant,
    summarize_logical_measurements,
)


class ReconciliationError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    )


def _safe_file(root: Path, path: Path, label: str) -> Path:
    need(path.is_file() and not path.is_symlink(), f"{label}_missing_or_symlink:{path}")
    resolved = path.resolve(strict=True)
    need(resolved.is_relative_to(root), f"{label}_escapes_run:{path}")
    return resolved


def _safe_relative(root: Path, token: Any, label: str) -> Path:
    logical = Path(str(token or "").strip())
    need(str(logical) not in {"", "."}, f"{label}_path_empty")
    need(not logical.is_absolute() and ".." not in logical.parts, f"{label}_path_unsafe:{logical}")
    return _safe_file(root, root / logical, label)


class EvidenceReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}

    def json(self, path: Path, *, label: str, default: Any = None) -> Any:
        if not path.exists() and default is not None:
            return default
        path = _safe_file(self.root, path, label)
        logical = path.relative_to(self.root).as_posix()
        raw = path.read_bytes()
        self.hashes[logical] = hashlib.sha256(raw).hexdigest()
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ReconciliationError(f"{label}_invalid_json:{logical}:{exc}") from exc
        return value

    def text(self, path: Path, *, label: str, default: str = "") -> str:
        if not path.exists():
            return default
        path = _safe_file(self.root, path, label)
        logical = path.relative_to(self.root).as_posix()
        raw = path.read_bytes()
        self.hashes[logical] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8", "replace")


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("results", "rows", "required_results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def discover_normalized_rows(run_dir: Path, reader: EvidenceReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("models/*/benchmark_results/normalized_results.json")):
        payload = reader.json(path, label="normalized_results")
        model_id = path.parents[1].name
        for row in _rows_from_payload(payload):
            row.setdefault("model_id", model_id)
            row.setdefault("source_path", path.relative_to(run_dir).as_posix())
            rows.append(row)
    need(rows, "no_normalized_results_found")
    return rows


def discover_required_rows(run_dir: Path, reader: EvidenceReader) -> list[dict[str, Any]]:
    required: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("models/*/benchmark_results/required_profile_matrix.json")):
        payload = reader.json(path, label="required_profile_matrix")
        model_id = path.parents[1].name
        for row in list(payload.get("required_results") or []):
            if isinstance(row, Mapping):
                item = dict(row)
                item.setdefault("model_id", model_id)
                required.append(item)
    if required:
        return required
    # v2.79.2 scopes are the fallback for future runs.
    for path in sorted(run_dir.glob("models/*/benchmark_set/required_run_scope.json")):
        payload = reader.json(path, label="required_run_scope")
        for row in list(payload.get("identities") or []):
            if isinstance(row, Mapping):
                required.append(dict(row))
    need(required, "no_required_matrix_or_scope_found")
    return required


def discover_quality_results(run_dir: Path, reader: EvidenceReader) -> list[dict[str, Any]]:
    central = run_dir / "quality_management" / "central_quality_summary.json"
    payload = reader.json(central, label="central_quality_summary")
    rows = _rows_from_payload(payload)
    need(rows, "central_quality_results_missing")
    return rows


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _legacy_part2_projection(
    row: Mapping[str, Any], required: Mapping[str, Any],
) -> bool:
    """Recognize the documented v2.78.4 split-to-P2 representation.

    The fallback is deliberately impossible for sealed/new scopes: it accepts
    only the schema-v1 profile matrix, only when no physical setup or endpoint
    identity was declared, and only when the measured row contains an explicit
    variant-local ``not_applicable``/technical-only Quality gate.  It therefore
    reconstructs historical state without importing historical rows or weaker
    identity rules into a newly bound run.
    """

    try:
        schema_version = int(required.get("schema_version") or 0)
    except (TypeError, ValueError):
        return False
    if (
        str(required.get("schema") or "")
        != "onnx-splitpoint/required-profile-measurement"
        or schema_version != 1
    ):
        return False
    if any(
        str(required.get(key) or "").strip()
        for key in (
            "expected_setup_id", "setup_id", "measurement_endpoint",
            "quality_endpoint", "logical_identity_sha256",
        )
    ):
        return False
    if _token(required.get("variant")) not in {"split", "composed", "complete"}:
        return False
    if selected_variant(row) != "part2":
        return False
    gates = row.get("task_quality_gates_by_variant")
    gate = gates.get("part2") if isinstance(gates, Mapping) else None
    if not isinstance(gate, Mapping):
        return False
    return bool(
        _token(gate.get("decision")) == "not_applicable"
        or _token(gate.get("status"))
        in {"not_applicable", "technical_only", "technical_validation_only"}
    )


def _matches_required(row: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
    if _token(row.get("model_id")) != _token(required.get("model_id")):
        return False
    if _token(row.get("case_id") or "full") != _token(required.get("case_id") or "full"):
        return False
    if canonical_backend(row.get("backend")) != canonical_backend(required.get("backend")):
        return False
    if (
        selected_variant(row) != selected_variant(required)
        and not _legacy_part2_projection(row, required)
    ):
        return False
    expected_run = canonical_run_id(required.get("run_id"))
    observed_run = canonical_run_id(
        row.get("quality_source_run_id") or row.get("source_run_id") or row.get("run_id")
    )
    if expected_run and observed_run and expected_run != observed_run:
        return False
    expected_setup = _token(required.get("expected_setup_id") or required.get("setup_id"))
    if expected_setup:
        direct = direct_setup_ids(row)
        mirrors = {_token(value) for value in list(row.get("mirror_setup_ids") or [])}
        if direct == [expected_setup]:
            return True
        return bool(
            not direct and row.get("mirror_provenance_verified") is True
            and expected_setup in mirrors
        )
    return True


def _select_primary(
    candidates: Sequence[Mapping[str, Any]],
    *, required: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    if not candidates:
        return {}, False
    groups = {str(row.get("logical_measurement_id") or "") for row in candidates}
    groups.discard("")
    if len(groups) > 1:
        # v2.78.4 retained an unhashed setup-less serialization beside one
        # setup-bound technical P2 row.  It cannot be promoted to a proven
        # mirror, but the sole direct representation can satisfy the sole
        # schema-v1 requirement.  New/sealed scopes can never enter this path.
        if required is not None and all(
            _legacy_part2_projection(row, required) for row in candidates
        ):
            direct = [row for row in candidates if len(direct_setup_ids(row)) == 1]
            setup_less = [row for row in candidates if not direct_setup_ids(row)]
            if len(direct) == 1 and len(direct) + len(setup_less) == len(candidates):
                return dict(direct[0]), False
        return {}, True
    rows = [dict(row) for row in candidates]
    rows.sort(key=lambda row: (
        0 if row.get("logical_measurement_primary") is True else 1,
        0 if direct_setup_ids(row) else 1,
        str(row.get("source_path") or ""),
    ))
    return rows[0], False


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: stable_json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _yolov7_deepx_schema_audit(
    run_dir: Path, reader: EvidenceReader,
    quality_results: Sequence[Mapping[str, Any]],
    normalized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result_rows = [
        row for row in quality_results
        if _token(row.get("model_id")) == "yolov7_paper"
        and _token(row.get("case_id") or "full") == "full"
        and canonical_run_id(row.get("source_run_id")) == "deepx_m1_full"
    ]
    if len(result_rows) != 1:
        return {"status": "not_uniquely_available", "candidate_count": len(result_rows)}
    result = dict(result_rows[0])
    request_path = _safe_relative(run_dir, result.get("source_request"), "yolov7_deepx_request")
    request = reader.json(request_path, label="yolov7_deepx_request")
    candidate_meta = request.get("candidate") if isinstance(request, Mapping) else None
    need(isinstance(candidate_meta, Mapping), "yolov7_deepx_candidate_metadata_missing")
    candidate_path = _safe_file(
        run_dir,
        request_path.parent / str(candidate_meta.get("path") or ""),
        "yolov7_deepx_candidate",
    )
    candidate = reader.json(candidate_path, label="yolov7_deepx_candidate")
    records = candidate.get("records") if isinstance(candidate, Mapping) else None
    need(isinstance(records, list), "yolov7_deepx_records_missing")
    total = 0
    nonempty = 0
    schemas: set[str] = set()
    for record_index, record in enumerate(records):
        need(isinstance(record, Mapping), f"yolov7_deepx_record_not_object:{record_index}")
        detections = record.get("candidate")
        need(isinstance(detections, list), f"yolov7_deepx_candidate_not_array:{record_index}")
        parsed = normalize_detection_records(
            detections, context=f"yolov7_deepx.records[{record_index}]",
        )
        total += len(parsed)
        nonempty += int(bool(parsed))
        schemas.update(item["source_schema"] for item in parsed)
    primary = result.get("primary") if isinstance(result.get("primary"), Mapping) else {}
    old_rows = [
        row for row in normalized_rows
        if _token(row.get("model_id")) == "yolov7_paper"
        and _token(row.get("case_id") or "full") == "full"
        and canonical_run_id(row.get("source_run_id") or row.get("run_id")) == "deepx_m1_full"
    ]
    old = old_rows[0] if len(old_rows) == 1 else {}
    gate = old.get("semantic_validation_metric_gate") if isinstance(old.get("semantic_validation_metric_gate"), Mapping) else {}
    return {
        "status": "validator_schema_conflict_not_hardware_semantic_failure",
        "record_count": len(records),
        "total_detections": total,
        "nonempty_records": nonempty,
        "source_schemas": sorted(schemas),
        "legacy_parser_expected": ["box_xyxy", "confidence", "class_id"],
        "canonical_parser_supported": ["x1", "y1", "x2", "y2", "score", "class_id"],
        "legacy_predictions_accepted": gate.get("predictions"),
        "legacy_ap50": old.get("mini_coco_ap50_primary"),
        "central_decision": result.get("decision"),
        "central_metric": primary.get("metric"),
        "central_candidate": primary.get("candidate"),
        "central_reference": primary.get("reference"),
        "source_request": request_path.relative_to(run_dir).as_posix(),
        "candidate_path": candidate_path.relative_to(run_dir).as_posix(),
        "candidate_sha256": sha256_file(candidate_path),
    }


def _scope_discrepancy_audit(run_dir: Path, reader: EvidenceReader) -> dict[str, Any]:
    plan = reader.json(
        run_dir / "effective_execution_plan.json",
        label="effective_execution_plan", default={},
    )
    workflow_log = reader.text(
        run_dir / "evaluation_workflow.log",
        label="evaluation_workflow_log", default="",
    )
    model_plans: dict[str, list[str]] = {}
    for path in sorted(run_dir.glob("models/*/benchmark_set/benchmark_plan.json")):
        payload = reader.json(path, label="benchmark_plan")
        ids = [
            canonical_run_id(row.get("id") or row.get("run_id"))
            for row in list(payload.get("runs") or payload.get("planned_runs") or [])
            if isinstance(row, Mapping)
        ]
        model_plans[path.parents[1].name] = ids
    yolo11 = model_plans.get("yolo11l", [])
    effective = [canonical_run_id(value) for value in list(plan.get("effective_generic_run_ids") or [])]
    return {
        "status": (
            "frozen_scope_planning_and_provenance_discrepancy"
            if "hailo8" in effective and "hailo8" not in yolo11 else "not_observed"
        ),
        "global_effective_run_ids": effective,
        "yolo11_model_plan_run_ids": yolo11,
        "global_contains_hailo8": "hailo8" in effective,
        "global_contains_hailo10": "hailo10" in effective,
        "yolo11_plan_contains_hailo8": "hailo8" in yolo11,
        "yolo11_plan_contains_hailo10": "hailo10" in yolo11,
        "hailo8_hard_timeout_9000_observed": bool(
            "9000" in workflow_log and "compile_prep" in workflow_log
        ),
        "hailo10_hard_timeout_9000_observed": bool(
            "9000" in workflow_log and "bias_correction" in workflow_log
        ),
        "historical_matrix_rewritten": False,
        "future_authority": "sealed_global_required_run_scope_before_compiler_dispatch",
    }


def stable_zip(path: Path, members: Sequence[Path], root: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for member in sorted(members, key=lambda value: value.name):
            info = zipfile.ZipInfo(member.name, date_time=(2026, 8, 31, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.read_bytes())
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        need(bad is None, f"output_zip_crc_failure:{bad}")


def reconcile(
    run_dir: Path, out_dir: Path, *, expect_v2784_b500: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve(strict=True)
    out_dir = out_dir.expanduser().resolve(strict=False)
    need(run_dir.is_dir() and not run_dir.is_symlink(), "source_run_missing_or_symlink")
    need(not out_dir.exists(), "output_directory_already_exists")
    need(out_dir.parent.is_dir() and not out_dir.parent.is_symlink(), "output_parent_unsafe")
    need(not out_dir.is_relative_to(run_dir), "output_must_be_outside_source_run")
    need(not run_dir.is_relative_to(out_dir), "source_run_must_not_be_below_output")

    reader = EvidenceReader(run_dir)
    normalized_raw = discover_normalized_rows(run_dir, reader)
    normalized = annotate_logical_measurements(normalized_raw)
    required = discover_required_rows(run_dir, reader)
    quality_results = discover_quality_results(run_dir, reader)
    join = join_quality_results_by_request_sha(rows=normalized, results=quality_results)
    quality_for_row = {
        int(item["row_index"]): dict(quality_results[int(item["result_index"])])
        for item in join["joins"]
    }

    matrix_rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    selected_indices: dict[int, int] = {}
    for ordinal, requirement in enumerate(required):
        candidates_with_index = [
            (index, row) for index, row in enumerate(normalized)
            if _matches_required(row, requirement)
        ]
        selected, ambiguous = _select_primary(
            [row for _, row in candidates_with_index], required=requirement,
        )
        selected_index = -1
        if selected:
            selected_index = next(
                index for index, row in candidates_with_index
                if row.get("source_path") == selected.get("source_path")
                and row.get("logical_measurement_id") == selected.get("logical_measurement_id")
                and row.get("logical_measurement_primary") == selected.get("logical_measurement_primary")
            )
            selected_indices[ordinal] = selected_index
        state = project_evidence_state(
            selected,
            scope=requirement,
            quality_result=quality_for_row.get(selected_index, {}),
        )
        legacy_part2 = bool(
            selected and _legacy_part2_projection(selected, requirement)
        )
        state.update({
            "ordinal": ordinal,
            "model_id": str(requirement.get("model_id") or ""),
            "case_id": str(requirement.get("case_id") or "full"),
            "run_id": canonical_run_id(requirement.get("run_id")),
            "backend": canonical_backend(requirement.get("backend")),
            "variant": selected_variant(requirement),
            "representation_count": len(candidates_with_index),
            "representation_ambiguous": ambiguous,
            "legacy_part2_projection": legacy_part2,
            "selected_normalized_row_index": selected_index if selected_index >= 0 else "",
        })
        states.append(state)
        matrix_rows.append({**dict(requirement), **state})

    coverage = summarize_evidence_states(states)
    coverage.update({
        "companions": int(join["companion_count"]),
        "unmatched": int(join["unmatched_count"]),
        "ambiguous": int(join["ambiguous_count"]),
        "quality_generated": len(quality_results),
        "quality_primary_joined": int(join["matched_primary_count"]),
        "legacy_part2_projection_count": sum(
            bool(state.get("legacy_part2_projection")) for state in states
        ),
    })
    mirror = summarize_logical_measurements(normalized_raw)
    yolo7 = _yolov7_deepx_schema_audit(
        run_dir, reader, quality_results, normalized,
    )
    scope_discrepancy = _scope_discrepancy_audit(run_dir, reader)

    if expect_v2784_b500:
        expected = {
            "matrix_required": 551,
            "matrix_present": 550,
            "quality_not_applicable": 163,
            "quality_applicable": 388,
            "quality_completed": 386,
            "quality_blocked": 1,
            "quality_missing": 1,
            "companions": 21,
            "unmatched": 0,
            "ambiguous": 0,
            "quality_generated": 407,
            "quality_primary_joined": 386,
            "legacy_part2_projection_count": 163,
        }
        mismatches = {
            key: {"expected": value, "observed": coverage.get(key)}
            for key, value in expected.items()
            if coverage.get(key) != value
        }
        need(not mismatches, f"v2784_b500_expected_counts_mismatch:{stable_json(mismatches)}")
        need(yolo7.get("record_count") == 500, "yolov7_deepx_record_count_not_500")
        need(yolo7.get("total_detections") == 3753, "yolov7_deepx_detection_count_not_3753")
        need(yolo7.get("central_decision") == "pass", "yolov7_deepx_central_quality_not_pass")

    status = "RECONCILED_WITH_REAL_GAPS" if any((
        coverage["quality_blocked"], coverage["quality_missing"],
        coverage["unmatched"], coverage["ambiguous"],
        scope_discrepancy.get("status") != "not_observed",
    )) else "RECONCILED"
    result = {
        "schema": "onnx-splitpoint/read-only-evidence-reconciliation",
        "schema_version": 2,
        "algorithm_version": "request-sha-logical-state-legacy-closure-2",
        "status": status,
        "source_run": str(run_dir),
        "source_run_unchanged": True,
        "official_workflow_status_changed": False,
        "claim_eligible": False if status != "RECONCILED" else None,
        "coverage": coverage,
        "logical_measurements": mirror,
        "quality_join": {key: value for key, value in join.items() if key not in {"joins", "companions", "unmatched_results", "ambiguous_results"}},
        "legacy_compatibility": {
            "mode": (
                "schema_v1_explicit_part2_quality_na"
                if coverage["legacy_part2_projection_count"] else "not_used"
            ),
            "legacy_part2_projection_count": coverage[
                "legacy_part2_projection_count"
            ],
            "new_scope_rows_eligible_for_legacy_projection": False,
            "historical_rows_imported_into_new_run": False,
        },
        "yolov7_deepx_validator_schema_audit": yolo7,
        "frozen_scope_audit": scope_discrepancy,
        "historical_decisions_recomputed": False,
        "historical_metrics_recomputed": False,
        "hardware_invoked": False,
        "compiler_invoked": False,
    }

    temp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        summary_path = temp / "reconciliation_summary.json"
        matrix_path = temp / "logical_matrix.csv"
        join_path = temp / "quality_join.csv"
        analysis_path = temp / "ANALYSIS.md"
        provenance_path = temp / "provenance_manifest.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _csv_write(matrix_path, matrix_rows)
        join_rows = [
            {
                **item,
                "result": quality_results[int(item["result_index"])] if "result_index" in item else {},
            }
            for item in join["joins"] + join["unmatched_results"] + join["ambiguous_results"]
        ]
        join_rows.extend({**item, "status": "companion"} for item in join["companions"])
        _csv_write(join_path, join_rows)
        analysis_lines = [
            "# EvaluationRun evidence reconciliation (v2.79.2)", "",
            f"**Status:** `{status}`", "",
            "The original run remains unchanged. No metric or decision was recomputed.", "",
            "## Matrix and Quality state", "",
        ]
        for key in (
            "matrix_required", "matrix_present", "quality_applicable",
            "quality_completed", "quality_blocked", "quality_not_applicable",
            "quality_missing", "companions", "quality_generated",
            "quality_primary_joined", "unmatched", "ambiguous",
        ):
            analysis_lines.append(f"- `{key}`: **{coverage.get(key)}**")
        analysis_lines += [
            "", "## YOLOv7 / DeepX validator audit", "",
            f"- Status: `{yolo7.get('status')}`",
            f"- Records: `{yolo7.get('record_count')}`",
            f"- Accepted canonical detections: `{yolo7.get('total_detections')}`",
            f"- Central decision: `{yolo7.get('central_decision')}`",
            "", "## Frozen-scope audit", "",
            f"- Status: `{scope_discrepancy.get('status')}`",
            "- Historical materialized matrix rewritten: `false`",
            "- Future authority: sealed global scope before compiler dispatch",
        ]
        analysis_path.write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")

        before = dict(sorted(reader.hashes.items()))
        after = {logical: sha256_file(run_dir / logical) for logical in sorted(before)}
        need(before == after, "source_evidence_changed_during_reconciliation")
        provenance = {
            "schema": "onnx-splitpoint/read-only-reconciliation-provenance",
            "schema_version": 2,
            "source_run": str(run_dir),
            "input_file_count": len(before),
            "input_sha256_before": before,
            "input_sha256_after": after,
            "source_unchanged": True,
            "output_sha256": {
                path.name: sha256_file(path)
                for path in (summary_path, matrix_path, join_path, analysis_path)
            },
        }
        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        bundle_path = temp / "evidence_reconciliation_v2792_bundle.zip"
        stable_zip(bundle_path, (summary_path, matrix_path, join_path, analysis_path, provenance_path), temp)
        temp.rename(out_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--expect-v2784-seven-model-b500", action="store_true",
        help="Fail closed unless the known v2.78.4 audit decomposes to 551/550, 163 N/A, 386 completed, one blocked, one missing and 21 companions.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = reconcile(
            args.run_dir, args.out_dir,
            expect_v2784_b500=bool(args.expect_v2784_seven_model_b500),
        )
    except ReconciliationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("V2792_EVIDENCE_RECONCILIATION=PASS")
    print("OFFICIAL_WORKFLOW_STATUS_CHANGED=NO")
    print("STATUS=" + str(result.get("status") or ""))
    print("OUTPUT=" + str(Path(args.out_dir).expanduser().resolve()))
    print("BUNDLE=" + str(Path(args.out_dir).expanduser().resolve() / "evidence_reconciliation_v2792_bundle.zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
