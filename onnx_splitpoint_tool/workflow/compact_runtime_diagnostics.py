"""Bounded, non-authoritative projections of already parsed runtime results.

This module never reads models, recursively searches metadata, or calculates
quality/performance/energy. Companion files are diagnostics, not measurements.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

SUMMARY_SCHEMA = "onnx-splitpoint/compact-runtime-diagnostic"
SUMMARY_VERSION = 1
SUMMARY_MAX_BYTES = 2 * 1024 * 1024
SUMMARY_TOTAL_MAX_BYTES = 16 * 1024 * 1024
from .debug_pack_policy import STRUCTURED_RESULT_MAX_FILE_BYTES
FALLBACK_PARSE_MAX_BYTES = STRUCTURED_RESULT_MAX_FILE_BYTES

# Explicit scalar fields only. Unknown dictionaries, raw bodies and recursive
# details/metadata are deliberately never traversed.
IDENTITY_FIELDS = frozenset("""
model_id model_name model case_id run_id evaluation_run_id setup_id hardware_setup_id
hardware_target_id target_id backend provider stage1_provider stage2_provider full_provider
full_backend variant primary_variant precision precision_mode execution_precision runtime_precision_identity source_run_id quality_source_run_id
quality_source_variant endpoint_contract_hash task benchmark_task
benchmark_task_requested benchmark_task_used measurement_endpoint full_measurement_endpoint
part1_measurement_endpoint part2_measurement_endpoint composed_measurement_endpoint quality_endpoint
endpoint_mode full_endpoint_mode part2_endpoint_mode boundary boundary_index source_path source_tag
role evidence_role quality_role evaluation_role population population_role schema schema_version
created_at timestamp version tool_version workflow_version build_id
""".split())
STATUS_FIELDS = frozenset("""
status ok enabled compile_ok runtime_ok validation_ok runtime_pass build_pass task_quality_pass
returncode latency_ok semantic_validation_ok semantic_validation_status semantic_validation_enabled
semantic_validation_intentionally_skipped diagnostic_run_model_status diagnostic_run_model_ok
completed_frames requested_frames completed_work_units completed_work_units_source completed_work_units_status
io_contract_pass execution_ok compile_status runtime_status validation_status error error_class
error_code error_detail error_message primary_error primary_failure_reason failure_reason failure_stage error_stage blocker_reason reason
prerequisite_status runtime_success repetition_count_requested repetition_count_attempted
repetition_count_valid repetition_status completion_expected completion_observed completion_count
completion_requested_frames completion_measured_frames completion_contract_status
blocked_reason block_reason attempts attempt_count attempted_repetitions completed_repetitions
successful_repetitions expected_repetitions repetitions completed completion_status started
buildable runtime_executable contract_consistent task_valid accuracy_gate_pass accuracy_gate_source
accuracy_gate_reason eligible_for_ranking ranking_eligible performance_eligible energy_eligible
pareto_eligible final_pass final_pass_all task_quality_gate_status task_quality_gate_decision
quality_status quality_valid quality_evaluation_pending quality_execution_location
quality_reference_source quality_reference_role reference_role reference_source semantic_reference_only
exclusion_reason ranking_exclusion_reason evidence_complete semantic_validation_passed
runtime_contract_decision quantized_interface_gate_decision contract_gate_reason interface_valid
interface_status throughput_kind pipeline_applicable performance_matrix_applicable
performance_matrix_status matrix_complete executor_status result_count planned_result_count
missing_measurement_count quality_evidence_only_complete quality_evidence_count expected_full_quality_count
""".split())
PERFORMANCE_FIELDS = frozenset("""
fps fps_makespan fps_median throughput_fps latency_ms mean_ms std_ms min_ms max_ms median_ms p50_ms p90_ms p95_ms p99_ms
latency_mean_ms latency_p50_ms latency_p95_ms latency_semantics measured_makespan_s makespan_s
fps_semantics diagnostic_full_e2e_latency_ms npu_processing_time_ms performance_benchmark_source
dxrt_tool_fps dxrt_tool_fps_semantics backend_tool_fps_semantics
cycle_ms frames measured_frames frame_count n_frames duration_s elapsed_s warmup runs
part1_latency_ms part2_latency_ms transfer_latency_ms total_latency_ms full_latency_ms
full_e2e_latency_ms full_host_tail_latency_ms full_raw_accelerator_latency_ms
full_mean_ms full_std_ms part1_mean_ms part1_std_ms part2_mean_ms part2_std_ms
composed_mean_ms composed_std_ms full_e2e_mean_ms full_e2e_std_ms full_host_tail_mean_ms
full_host_tail_std_ms part2_prefix_mean_ms part2_prefix_std_ms part2_host_tail_mean_ms
part2_host_tail_std_ms split_latency_e2e_ms pipeline_model pipeline_stage1_lane_ms
pipeline_stage2_lane_ms pipeline_transfer_est_ms pipeline_cycle_no_transfer_ms
pipeline_cycle_with_transfer_ms pipeline_cycle_selected_ms pipeline_fps_selected
pipeline_cycle_source pipeline_note pipeline_speedup_full_over_cycle
full_backend_cycle_ms full_backend_throughput_fps full_backend_throughput_source
heterogeneous_pipeline_fps same_backend_composed_fps backend_tool_fps throughput_primary_fps
throughput_primary_metric throughput_primary_source unit units samples count n mean median std min max
""".split())
QUALITY_FIELDS = frozenset("""
classification_top1_accuracy classification_top5_accuracy classification_samples classification_labeled_samples
labeled_samples validated_image_count semantic_validation_image_count detection_ap50 coco_ap50 map50
ap50_gt_instances ap50_predictions ap50_images bootstrap_repetitions_requested bootstrap_skipped_reason
accuracy top1_accuracy top5_accuracy top1 top5 ap ap50 ap75 candidate reference delta ci_low ci_high
margin n population_size requested_images n_images unique_images evaluated_images reference_images
image_count sample_count requested_samples evaluated_samples bootstrap_repetitions confidence_level
metric primary_metric decision gate_status pass passed tier dataset_tier dataset_id dataset_role
reference_id policy_id policy_sha256 quality_request_id protocol role scope
mini_coco_ap50_enabled mini_coco_ap50_status mini_coco_ap50_reference_source mini_coco_ap50_full
mini_coco_ap50_primary_variant mini_coco_ap50_primary mini_coco_ap50_delta_primary_minus_full
mini_classification_enabled mini_classification_status mini_classification_top1_accuracy
mini_classification_top5_accuracy task_quality_gate_tier task_quality_primary_metric task_quality_candidate
task_quality_reference task_quality_delta task_quality_ci_low task_quality_ci_high task_quality_margin
task_quality_n task_quality_reference_source task_quality_policy_sha256 accuracy_gate_policy_sha256
validation_dataset_task validation_dataset_primary_pass validation_dataset_primary_image_pass_ratio
validation_dataset_primary_top1_agreement validation_dataset_primary_top5_agreement
validation_dataset_primary_top1_accuracy validation_dataset_primary_top5_accuracy
backend_drift_enabled backend_drift_status backend_drift_mode backend_drift_variant
backend_drift_provider backend_drift_reference_source backend_drift_dataset_pass
backend_drift_dataset_requested_images backend_drift_dataset_n_images
backend_drift_dataset_top1_agreement backend_drift_dataset_top5_agreement
backend_drift_dataset_top1_accuracy backend_drift_dataset_top5_accuracy
benchmark_input_mode benchmark_loop_count benchmark_unique_dataset_images
max_abs_error mean_abs_error cosine_similarity output_shape_match
""".split())
ENERGY_FIELDS = frozenset("""
energy_j energy_mj energy_per_frame_j energy_per_frame_mj energy_per_inference_j energy_per_inference_mj
power_w mean_power_w median_power_w idle_power_w gross_energy_j net_energy_j duration_s
window_duration_s window_s window_label window_scope physical_scope scope qualification
qualification_status qualification_reason qualified energy_status energy_scope energy_physical_scope
energy_window_label energy_duration_s energy_repetitions energy_qualification energy_qualification_status
energy_row_level_source energy_coverage_status energy_target_status energy_target_phase_count
energy_target_window_count energy_target_valid_window_count energy_target_valid_window_ratio
energy_target_id energy_target_ok energy_target_error_summary energy_merge_source
measured_work_units work_units frames repetitions repeats aggregation statistic unit units
""".split())
SCALAR_FIELDS = IDENTITY_FIELDS | STATUS_FIELDS | PERFORMANCE_FIELDS | QUALITY_FIELDS | ENERGY_FIELDS
# Each known subobject has a fixed scalar allowlist, never an arbitrary subtree.
NESTED_FIELDS = {
    "throughput": IDENTITY_FIELDS | STATUS_FIELDS | PERFORMANCE_FIELDS,
    "performance": IDENTITY_FIELDS | STATUS_FIELDS | PERFORMANCE_FIELDS,
    "timings": PERFORMANCE_FIELDS | STATUS_FIELDS,
    "quality": IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    "task_quality_gate": IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    "mini_classification_eval": IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    "mini_coco_eval": IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    "energy": IDENTITY_FIELDS | STATUS_FIELDS | ENERGY_FIELDS | PERFORMANCE_FIELDS,
    "energy_summary": IDENTITY_FIELDS | STATUS_FIELDS | ENERGY_FIELDS,
    "completion": IDENTITY_FIELDS | STATUS_FIELDS | PERFORMANCE_FIELDS,
    "semantic_validation_metric_gate": IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    "deepx_semantic_validation": IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    "deepx_prepared_feed_benchmark": IDENTITY_FIELDS | STATUS_FIELDS | PERFORMANCE_FIELDS,
    "benchmark_input_policy": IDENTITY_FIELDS | STATUS_FIELDS | PERFORMANCE_FIELDS,
    "variant_status": frozenset({"full", "part1", "part2", "composed", "split"}),
    "task_quality_gates_by_variant": frozenset(),
}
# Fixed paths for known aggregate objects. This is not a metadata tree walk:
# only these named aggregate slots can be reached, never predictions/images.
NESTED_DESCENDANT_FIELDS = {
    ("task_quality_gate", "primary"): STATUS_FIELDS | QUALITY_FIELDS,
    ("task_quality_gate", "policy"): IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    ("task_quality_gate", "quality_input_request"): IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    ("task_quality_gate", "quality_input_request", "reference"): IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS,
    ("task_quality_gate", "guardrails"): frozenset(),
}
for _metric in ("top5_accuracy", "ap50", "ap75", "coco_ap50", "coco_ap75"):
    NESTED_DESCENDANT_FIELDS[("task_quality_gate", "guardrails", _metric)] = STATUS_FIELDS | QUALITY_FIELDS
for _variant in ("full", "part1", "part2", "composed", "split"):
    NESTED_DESCENDANT_FIELDS[("task_quality_gates_by_variant", _variant)] = IDENTITY_FIELDS | STATUS_FIELDS | QUALITY_FIELDS
    NESTED_DESCENDANT_FIELDS[("task_quality_gates_by_variant", _variant, "primary")] = STATUS_FIELDS | QUALITY_FIELDS


def summary_bytes(summary: Mapping[str, Any]) -> bytes:
    return (json.dumps(summary, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def source_observation(path: Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"observed_size_bytes": stat.st_size, "observed_mtime_ns": stat.st_mtime_ns}


def companion_path(source: Path) -> Path:
    source = Path(source)
    return source.parent / "diagnostic_summaries" / (source.stem + ".summary.json")


def _relative_source(value: str) -> bool:
    p = PurePosixPath(value)
    return bool(value and not p.is_absolute() and ".." not in p.parts
                and str(p) == value and "\\" not in value)


def companion_matches(summary: Mapping[str, Any], *, source_path: str,
                      source_stat: Mapping[str, Any]) -> tuple[bool, str]:
    if not isinstance(summary, Mapping) or summary.get("schema") != SUMMARY_SCHEMA or summary.get("schema_version") != SUMMARY_VERSION:
        return False, "companion_schema_unavailable"
    if not _valid_summary_structure(summary):
        return False, "companion_projection_invalid"
    source = summary.get("source")
    if not isinstance(source, Mapping) or source.get("path") != source_path:
        return False, "source_identity_mismatch"
    for key in ("observed_size_bytes", "observed_mtime_ns"):
        if source.get(key) is None or source_stat.get(key) is None or source.get(key) != source_stat.get(key):
            return False, "source_observation_mismatch"
    return True, "matching_source_observation_not_content_reverified"


def _valid_scalar(value: Any) -> bool:
    return (value is None or isinstance(value, (bool, int))
            or (isinstance(value, float) and math.isfinite(value))
            or (isinstance(value, str) and len(value.encode("utf-8")) <= 16384))


def _valid_nested_projection(value: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    fields = NESTED_FIELDS.get(path[0], frozenset()) if len(path) == 1 else NESTED_DESCENDANT_FIELDS.get(path, frozenset())
    children = {parts[len(path)] for parts in NESTED_DESCENDANT_FIELDS if len(parts) == len(path) + 1 and parts[:len(path)] == path}
    if value.keys() - (fields | children):
        return False
    for key, item in value.items():
        if key in children and isinstance(item, Mapping):
            if not _valid_nested_projection(item, path + (key,)):
                return False
        elif key not in fields or not _valid_scalar(item):
            return False
    return True


def _valid_projection_row(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    fields = SCALAR_FIELDS | NESTED_FIELDS.keys() | {"source_pointer", "unavailable_fields", "omitted_fields"}
    if row.keys() - fields:
        return False
    for key, value in row.items():
        if key in NESTED_FIELDS and isinstance(value, Mapping):
            if not _valid_nested_projection(value, (key,)):
                return False
        elif key in {"unavailable_fields", "omitted_fields"}:
            if not isinstance(value, list) or not all(isinstance(v, str) and _valid_scalar(v) for v in value):
                return False
        elif not _valid_scalar(value):
            return False
    return isinstance(row.get("source_pointer"), str)


def _valid_summary_structure(summary: Mapping[str, Any]) -> bool:
    allowed = {"schema", "schema_version", "projection_role", "original_bytes", "full_per_image_replay",
               "source", "status", "rows", "source_row_count", "projected_row_count", "raw_bodies_omitted",
               "limitations", "source_schema", "source_context", "reason", "identity_failure_overview",
               "identity_failure_overview_count", "identity_failure_overview_complete"}
    if summary.keys() - allowed or summary.get("projection_role") != "derived_summary":
        return False
    if summary.get("original_bytes") is not False or summary.get("full_per_image_replay") is not False or summary.get("raw_bodies_omitted") is not True:
        return False
    if summary.get("status") not in {"projected", "summary_unavailable"}:
        return False
    source = summary.get("source")
    source_fields = {"path", "observed_size_bytes", "observed_mtime_ns", "declared_sha256", "hash_verification"}
    if not isinstance(source, Mapping) or source.keys() - source_fields or not all(_valid_scalar(v) for v in source.values()):
        return False
    if not _relative_source(str(source.get("path") or "")):
        return False
    if source.get("hash_verification") not in {"declared_not_reverified", "unavailable"}:
        return False
    rows = summary.get("rows")
    if not isinstance(rows, list) or not all(_valid_projection_row(row) for row in rows):
        return False
    if summary.get("projected_row_count") != len(rows):
        return False
    if "source_context" in summary and not _valid_projection_row(summary["source_context"]):
        return False
    overview = summary.get("identity_failure_overview", [])
    if not isinstance(overview, list) or len(overview) > 64 or not all(_valid_projection_row(row) for row in overview):
        return False
    limitations = summary.get("limitations")
    if not isinstance(limitations, list) or not all(isinstance(v, str) and _valid_scalar(v) for v in limitations):
        return False
    for key in allowed - {"source", "rows", "source_context", "identity_failure_overview", "limitations"}:
        if key in summary and not _valid_scalar(summary[key]):
            return False
    try:
        return len(summary_bytes(summary)) <= SUMMARY_MAX_BYTES
    except (ValueError, TypeError):
        return False


def _scalars(value: Mapping[str, Any], fields: frozenset[str], omitted: list[str], pointer: str) -> dict[str, Any]:
    result = {}
    for key in sorted(fields.intersection(value)):
        item = value[key]
        if isinstance(item, float) and not math.isfinite(item):
            omitted.append(pointer + "/" + key + ":nonfinite")
        elif item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, str) and len(item.encode("utf-8")) > 16384:
                omitted.append(pointer + "/" + key + ":scalar_size_limit")
            else:
                result[key] = item
        else:
            omitted.append(pointer + "/" + key + ":non_scalar")
    return result


def _row(value: Mapping[str, Any], pointer: str) -> dict[str, Any]:
    unavailable: list[str] = []
    result = _scalars(value, SCALAR_FIELDS, unavailable, pointer)
    for key, fields in NESTED_FIELDS.items():
        item = value.get(key)
        if isinstance(item, Mapping):
            result[key] = _scalars(item, fields, unavailable, pointer + "/" + key)
    for parts, fields in sorted(NESTED_DESCENDANT_FIELDS.items()):
        item: Any = value
        for part in parts:
            item = item.get(part) if isinstance(item, Mapping) else None
        if isinstance(item, Mapping):
            target = result
            for part in parts[:-1]:
                current = target.get(part)
                if not isinstance(current, dict):
                    current = {}
                    target[part] = current
                target = current
            target[parts[-1]] = _scalars(item, fields, unavailable, pointer + "/" + "/".join(parts))
    # No directory name supplies a model/setup identity. Explicit absence is
    # visible even for backend payloads preceding authoritative scope binding.
    result["source_pointer"] = pointer
    result["unavailable_fields"] = [key for key in ("model_id", "setup_id", "case_id", "precision", "measurement_endpoint") if key not in result]
    result["omitted_fields"] = unavailable
    return result


def _rows(payload: Any) -> tuple[list[Any] | None, str, str]:
    if isinstance(payload, list):
        return payload, "", "canonical_row_list"
    if not isinstance(payload, Mapping):
        return None, "", "unknown_payload_type"
    schema = payload.get("schema")
    if schema not in (None, "", "onnx-splitpoint/normalized-benchmark-results", "onnx-splitpoint/benchmark-results"):
        return None, "", "unknown_schema"
    if schema == "onnx-splitpoint/normalized-benchmark-results" and payload.get("schema_version") != 2:
        return None, "", "unknown_schema_version"
    for key in ("results", "rows", "benchmark_results"):
        if isinstance(payload.get(key), list):
            return payload[key], "/" + key, str(schema or "legacy_canonical_rows")
    return None, "", "unknown_rows_schema"


def project_runtime_payload(payload: Any, *, source_path: str,
                            source_stat: Mapping[str, Any] | None = None,
                            declared_source_sha256: str | None = None,
                            max_bytes: int = SUMMARY_MAX_BYTES) -> dict[str, Any]:
    """Project original aggregate fields; no inference from neighboring rows."""
    source = {"path": source_path, **{k: v for k, v in dict(source_stat or {}).items()
               if k in {"observed_size_bytes", "observed_mtime_ns"} and isinstance(v, int)}}
    if declared_source_sha256:
        source["declared_sha256"] = str(declared_source_sha256)
    source["hash_verification"] = "declared_not_reverified" if declared_source_sha256 else "unavailable"
    result: dict[str, Any] = {"schema": SUMMARY_SCHEMA, "schema_version": SUMMARY_VERSION,
        "projection_role": "derived_summary", "original_bytes": False,
        "full_per_image_replay": False, "source": source,
        "status": "projected", "rows": [], "source_row_count": None,
        "projected_row_count": 0, "raw_bodies_omitted": True,
        "limitations": ["allowlisted_original_aggregates_only", "no_quality_recalculation",
                        "source_observation_is_not_content_verification",
                        "missing_setup_identity_not_inferred_from_filename"]}
    rows, pointer, schema = _rows(payload)
    result["source_schema"] = schema
    if not _relative_source(source_path):
        result.update(status="summary_unavailable", reason="unsafe_source_path")
    elif rows is None:
        result.update(status="summary_unavailable", reason=schema)
    elif any(not isinstance(row, Mapping) or not (SCALAR_FIELDS.intersection(row) or NESTED_FIELDS.keys() & row.keys()) for row in rows):
        result.update(status="summary_unavailable", reason="unknown_row_schema", source_row_count=len(rows))
    else:
        result["source_row_count"] = len(rows)
        if isinstance(payload, Mapping):
            result["source_context"] = _row(payload, "")
        row_bytes = 0
        for index, row in enumerate(rows):
            projected = _row(row, pointer + "/" + str(index))
            row_bytes += len(summary_bytes(projected))
            if row_bytes > max_bytes:
                break
            result["rows"].append(projected)
        result["projected_row_count"] = len(result["rows"])
    if len(summary_bytes(result)) > max_bytes or (rows is not None and result["status"] == "projected" and result["projected_row_count"] != len(rows)):
        result.update(status="summary_unavailable", reason="summary_size_limit", rows=[], projected_row_count=0)
        result.pop("source_context", None)
        overview = []
        for index, row in enumerate(rows or []):
            if index >= 64:
                break
            item = _scalars(row, IDENTITY_FIELDS | STATUS_FIELDS, [], pointer + "/" + str(index))
            item["source_pointer"] = pointer + "/" + str(index)
            overview.append(item)
            result["identity_failure_overview"] = overview
            if len(summary_bytes(result)) > max_bytes:
                overview.pop()
                break
        result["identity_failure_overview_count"] = len(overview)
        result["identity_failure_overview_complete"] = len(overview) == len(rows or [])
    return result


def _production_root(source: Path) -> Path | None:
    source = Path(source).absolute()
    if source.resolve() != source:
        return None
    if source.suffix != ".json" or not (source.name.startswith("benchmark_results_") or source.name in {"benchmark_results.json", "normalized_results.json", "results.json"}):
        return None
    for directory in source.parents:
        if directory.name == "benchmark_results" and directory.parent.parent.name == "models":
            root = directory.parent.parent.parent
            for part in [source, *source.parents]:
                if part.is_symlink():
                    return None
                if part == root:
                    break
            return root
    return None


def write_runtime_companion(source: Path, payload: Any, *,
                            observed_before: Mapping[str, Any] | None = None,
                            declared_source_sha256: str | None = None) -> dict[str, Any]:
    """Write once per observed final state; diagnostic errors never fail runtime."""
    tmp: Path | None = None
    try:
        source = Path(source).absolute()
        root = _production_root(source)
        if root is None:
            return {"status": "summary_unavailable", "reason": "outside_canonical_runtime_source"}
        observation = source_observation(source)
        if observed_before is not None and dict(observed_before) != observation:
            return {"status": "summary_unavailable", "reason": "source_changed_during_parse"}
        relative = source.relative_to(root).as_posix()
        dest = companion_path(source)
        if dest.is_symlink() or dest.parent.is_symlink():
            return {"status": "summary_unavailable", "reason": "unsafe_companion_path"}
        if dest.is_file() and dest.stat().st_size <= SUMMARY_MAX_BYTES:
            existing = json.loads(dest.read_text(encoding="utf-8"))
            matches, _ = companion_matches(existing, source_path=relative, source_stat=observation)
            if matches:
                return {"status": "unchanged", "path": str(dest), "source_path": relative}
        projected = project_runtime_payload(payload, source_path=relative, source_stat=observation,
                                             declared_source_sha256=declared_source_sha256)
        body = summary_bytes(projected)
        if len(body) > SUMMARY_MAX_BYTES:
            return {"status": "summary_unavailable", "reason": "summary_envelope_size_limit"}
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tempname = tempfile.mkstemp(prefix="." + dest.name, suffix=".tmp", dir=str(dest.parent))
        tmp = Path(tempname)
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
        if source_observation(source) != observation:
            return {"status": "summary_unavailable", "reason": "source_changed_before_publish"}
        os.replace(tmp, dest)
        tmp = None
        return {"status": projected["status"], "path": str(dest), "source_path": relative,
                "reason": projected.get("reason", "")}
    except Exception as exc:
        return {"status": "summary_unavailable", "reason": "companion_write_failed",
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
