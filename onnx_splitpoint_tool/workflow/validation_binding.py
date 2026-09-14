from __future__ import annotations

"""Formal validation binding for Evaluation Workflow runs.

v49i does not try to invent new task metrics.  It binds the validation stage to
normalized benchmark rows, records which adapter would be responsible for each
model/case, and writes thesis-ready validity contracts even when heavy dataset
validation still has to be produced by the benchmark/remote service.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .artifacts import now_iso, read_json, relpath, write_csv, write_json, write_text


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        return None
    return None


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "ok", "pass", "passed", "success"}:
        return True
    if s in {"0", "false", "no", "n", "fail", "failed", "error"}:
        return False
    return None


def _mean(vals: Sequence[Any]) -> Optional[float]:
    xs = [_float(v) for v in vals]
    ys = [x for x in xs if x is not None]
    if not ys:
        return None
    return round(sum(ys) / len(ys), 8)


def _max(vals: Sequence[Any]) -> Optional[float]:
    xs = [_float(v) for v in vals]
    ys = [x for x in xs if x is not None]
    if not ys:
        return None
    return round(max(ys), 8)


def _rate(vals: Sequence[Any]) -> Optional[float]:
    known = [_bool(v) for v in vals]
    known = [v for v in known if v is not None]
    if not known:
        return None
    return round(sum(1 for v in known if v) / len(known), 8)





def _task_score_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarise detection score/confidence semantics across normalized rows.

    u.RECS/report rows can carry raw detector logits or unnormalised scores.
    Do not aggregate such values as calibrated confidence without explicitly
    preserving the score semantics.
    """
    sems: List[str] = []
    raw_vals: List[float] = []
    clipped_vals: List[float] = []
    conf_vals: List[float] = []
    for row in rows:
        sem = str(row.get("score_semantics") or row.get("confidence_score_semantics") or "").strip()
        cm = _float(row.get("confidence_mean"))
        raw = _float(row.get("raw_score_mean"))
        clipped = _float(row.get("confidence_mean_clipped_0_1"))
        if cm is not None:
            conf_vals.append(cm)
            if not sem:
                sem = "sigmoid_confidence" if 0.0 <= cm <= 1.0 else "raw_or_unnormalized_score"
            if raw is None and (cm < 0.0 or cm > 1.0):
                raw = cm
            if clipped is None:
                clipped = max(0.0, min(1.0, cm))
        if raw is not None:
            raw_vals.append(raw)
        if clipped is not None:
            clipped_vals.append(clipped)
        if sem:
            sems.append(sem)
    uniq = sorted(set(sems))
    return {
        "confidence_mean": _mean(conf_vals),
        "confidence_mean_clipped_0_1": _mean(clipped_vals),
        "raw_score_mean": _mean(raw_vals),
        "score_semantics": (uniq[0] if len(uniq) == 1 else ("mixed:" + ";".join(uniq) if uniq else None)),
    }

def _row_final_pass(row: Mapping[str, Any]) -> Optional[bool]:
    for key in ("final_pass", "validation_ok", "semantic_validation_ok", "runtime_ok"):
        b = _bool(row.get(key))
        if b is not None:
            return b
    return None


def _row_latency(row: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "total_latency_ms", "composed_mean_ms", "composed_latency_ms",
        "split_latency_e2e_ms", "pipeline_cycle_selected_ms",
    ):
        v = _float(row.get(key))
        if v is not None:
            return v
    p1 = _float(row.get("part1_mean_ms") or row.get("part1_latency_ms"))
    p2 = _float(row.get("part2_mean_ms") or row.get("part2_latency_ms"))
    if p1 is not None and p2 is not None:
        tr = _float(row.get("transfer_mean_ms") or row.get("transfer_latency_ms")) or 0.0
        return p1 + p2 + tr
    return None


def _is_complete_split_row(row: Mapping[str, Any]) -> bool:
    if str(row.get("variant") or "").lower() != "split":
        return False
    status = str(row.get("component_measurement_status") or "").lower()
    if status in {"part1_only", "part2_only", "none", "component_only", "missing"}:
        return False
    if str(row.get("error_class") or "").lower() in {"requires_stage2_accelerator_artifact", "requires_feature_tensor_calibration"}:
        return False
    return _row_latency(row) is not None


def _is_contract_rejected(row: Mapping[str, Any]) -> bool:
    vals = [
        row.get("deepx_stage2_contract_status"),
        row.get("stage2_contract_status"),
        row.get("interface_check_status"),
        row.get("error_class"),
        row.get("skip_reason"),
    ]
    s = " ".join(str(v or "").lower() for v in vals)
    return any(tok in s for tok in ("all_candidates_rejected", "contract_rejected", "stage2_contract_failed", "interface_rejected", "native_unstable", "deepx_stage2_native_unstable"))

def _profile_validation(profile_payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(profile_payload, Mapping):
        return {}
    raw = profile_payload.get("validation") or {}
    return dict(raw or {}) if isinstance(raw, Mapping) else {}


def _validation_options(options: Any, profile_payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    pv = _profile_validation(profile_payload)

    def _opt(name: str, default: Any) -> Any:
        return getattr(options, name, pv.get(name, default))

    cosine_thr = getattr(options, "validation_cosine_similarity_threshold", None)
    if cosine_thr is None:
        cosine_thr = getattr(options, "validation_cosine_threshold", pv.get("cosine_similarity_threshold", None))
    require_explicit = getattr(options, "validation_require_explicit", pv.get("require_explicit", False))
    require_task = getattr(options, "validation_require_task_metrics", pv.get("require_task_metrics", False))
    mode = str(_opt("validation_mode", pv.get("mode", "summary_only")) or "summary_only").strip().lower()
    if mode in {"off", "none", "disable"}:
        mode = "disabled"
    return {
        "mode": mode,
        "max_abs_error_threshold": _float(_opt("validation_max_abs_error_threshold", pv.get("max_abs_error_threshold", None))),
        "mean_abs_error_threshold": _float(_opt("validation_mean_abs_error_threshold", pv.get("mean_abs_error_threshold", None))),
        "cosine_similarity_threshold": _float(cosine_thr),
        "require_explicit": bool(require_explicit),
        "require_task_metrics": bool(require_task),
        "split_fidelity_reference_mode": str(pv.get("split_fidelity_reference_mode") or "auto"),
        "classification_metrics": _as_list(pv.get("classification_metrics")),
        "detection_metrics": _as_list(pv.get("detection_metrics")),
    }

def _contracts(output_contracts: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(x) for x in _as_list(output_contracts.get("contracts")) if isinstance(x, Mapping)]


def _has_raw_head_contract(output_contracts: Mapping[str, Any]) -> bool:
    return any(str(c.get("endpoint_mode") or "").lower() == "raw_detection_head" for c in _contracts(output_contracts))


def _adapter_plan(*, model_id: str, task: str, validation_preset: str, output_contracts: Mapping[str, Any], profile_payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    task_l = str(task or "auto").lower()
    has_raw_head = _has_raw_head_contract(output_contracts)
    adapters: List[Dict[str, Any]] = []
    adapters.append({
        "adapter_id": "numeric_output_comparison",
        "level": "numeric",
        "required_inputs": ["reference_outputs", "candidate_outputs"],
        "reported_metrics": ["max_abs_error", "mean_abs_error", "cosine_similarity", "output_shape_match"],
        "status": "bound_to_normalized_results",
    })
    if task_l == "classification":
        adapters.append({
            "adapter_id": "classification_adapter",
            "level": "task",
            "validation_preset": validation_preset,
            "reported_metrics": ["top1", "top5", "logits_comparison"],
            "status": "bound_to_normalized_results_when_emitted_by_benchmark_service",
        })
    elif task_l == "detection":
        adapters.append({
            "adapter_id": "detection_adapter_raw_yolo_head" if has_raw_head else "detection_adapter_decoded",
            "level": "task",
            "validation_preset": validation_preset,
            "reported_metrics": ["map_light", "detection_count_stability", "confidence_stability"],
            "postprocessing_required": bool(has_raw_head),
            "status": "raw_head_host_tail_contract" if has_raw_head else "decoded_output_contract",
        })
    else:
        adapters.append({
            "adapter_id": "auto_task_adapter",
            "level": "task",
            "validation_preset": validation_preset,
            "reported_metrics": ["task_proxy_metric_when_available"],
            "status": "pending_task_inference",
        })
    return {
        "schema": "onnx-splitpoint/validation-adapter-plan",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "task": task_l or "auto",
        "validation_preset": validation_preset,
        "output_contracts_summary": {
            "raw_head_contract_present": has_raw_head,
            "contract_count": len(_contracts(output_contracts)),
        },
        "profile_validation": _profile_validation(profile_payload),
        "adapters": adapters,
    }


def _row_numeric_evidence(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "max_abs_error": _float(row.get("max_abs_error")),
        "mean_abs_error": _float(row.get("mean_abs_error")),
        "cosine_similarity": _float(row.get("cosine_similarity")),
        "output_shape_match": _bool(row.get("output_shape_match")),
    }


def _row_task_evidence(row: Mapping[str, Any], task: str) -> Dict[str, Any]:
    task_l = str(task or "").lower()
    if task_l == "classification":
        return {
            "top1": _float(row.get("classification_top1")),
            "top5": _float(row.get("classification_top5")),
        }
    if task_l == "detection":
        return {
            "map_light": _float(row.get("mini_coco_ap50_primary")),
            "detection_count": _float(row.get("detection_count")),
            "confidence_mean": _float(row.get("confidence_mean")),
        }
    return {
        "top1": _float(row.get("classification_top1")),
        "top5": _float(row.get("classification_top5")),
        "map_light": _float(row.get("mini_coco_ap50_primary")),
    }


def _row_interface_evidence(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Evidence that a split interface actually matches the reference cut tensor.

    Wrong Hailo/host composed outputs often still have timing numbers.  These
    fields are emitted by newer runners when they compare part1/cut tensors or
    task proxies.  Treat explicit failures as validation failures instead of
    allowing bad Top-k/box outputs to pass as mere runtime measurements.
    """
    return {
        "raw_head_drift_pass": _bool(row.get("raw_head_drift_pass")),
        "backend_drift_single_pass": _bool(row.get("backend_drift_single_pass")),
        "backend_drift_dataset_pass": _bool(row.get("backend_drift_dataset_pass")),
        "backend_drift_single_top1_match": _bool(row.get("backend_drift_single_top1_match")),
        "backend_drift_single_cosine_similarity": _float(row.get("backend_drift_single_cosine_similarity")),
        "backend_drift_dataset_mean_cosine_similarity": _float(row.get("backend_drift_dataset_mean_cosine_similarity")),
        "backend_drift_dataset_top1_agreement": _float(row.get("backend_drift_dataset_top1_agreement")),
        "raw_head_drift_max_abs": _float(row.get("raw_head_drift_max_abs")),
    }


def _row_looks_like_mixed_hardware_split(row: Mapping[str, Any]) -> bool:
    blob = " ".join(str(row.get(k) or "") for k in ("backend", "provider", "run_id", "stage1_provider", "stage2_provider", "variant", "component_measurement_status"))
    low = blob.lower()
    return ("hailo" in low or "deepx" in low or "dx_m1" in low) and ("to" in low or "composed" in low or "part" in low)



def _row_has_measurement(row: Mapping[str, Any]) -> bool:
    """Return True if a normalized result row contains real benchmark evidence.

    BenchmarkSet outputs may emit planned/diagnostic placeholder rows, for
    example a Hailo case listed in a result file with no runtime, latency,
    validation or error evidence yet. Those rows are useful for diagnostics, but
    they must not make validate_outputs partial when all actually measured rows
    are valid.
    """
    latency_fields = (
        "total_latency_ms",
        "full_latency_ms",
        "part1_latency_ms",
        "part2_latency_ms",
        "transfer_latency_ms",
        "latency_ms",
        "mean_latency_ms",
        "composed_latency_ms",
    )
    if any(_float(row.get(k)) is not None for k in latency_fields):
        return True
    evidence_bool_fields = (
        "runtime_ok",
        "validation_ok",
        "compile_ok",
        "output_shape_match",
        "raw_head_decode_ok",
        "nms_ok",
    )
    if any(_bool(row.get(k)) is not None for k in evidence_bool_fields):
        return True
    error_class = str(row.get("error_class") or row.get("failure_class") or "").strip()
    error_detail = str(row.get("error_detail") or row.get("error") or "").strip()
    if error_class or error_detail:
        return True
    return False

def _decide_validation_status(row: Mapping[str, Any], task: str, opts: Mapping[str, Any]) -> Tuple[str, Optional[bool], str]:
    explicit = _bool(row.get("validation_ok"))
    reasons: List[str] = []
    numeric = _row_numeric_evidence(row)
    task_ev = _row_task_evidence(row, task)
    interface_ev = _row_interface_evidence(row)
    mode = str(opts.get("mode") or "summary_only").lower()

    # v52d hard guard: if the split interface check says part1/cut tensor
    # compatibility failed, do not let the row count as a valid benchmark.
    # This catches bad composed classification outputs such as plausible timing
    # with wrong Top-1 classes.
    if _row_looks_like_mixed_hardware_split(row):
        if interface_ev.get("raw_head_drift_pass") is False:
            return "invalid", False, "split_interface_raw_head_drift_failed"
        if interface_ev.get("backend_drift_single_pass") is False:
            return "invalid", False, "split_interface_backend_drift_failed"
        if interface_ev.get("backend_drift_dataset_pass") is False:
            return "invalid", False, "split_interface_dataset_drift_failed"
        if interface_ev.get("backend_drift_single_top1_match") is False:
            return "invalid", False, "classification_interface_top1_mismatch"

    # Raw detection-head Hailo full baselines deliberately do not expose the
    # decoded ONNX output contract.  Some Benchmark runner rows therefore carry
    # validation_ok=False even though the runtime measurement itself is valid
    # and decoded/host-tail validation is reported separately. In summary mode
    # this is a validation gap, not a hard model failure. Strict mode still
    # respects explicit false as a real validation failure.
    endpoint = str(row.get("endpoint_mode") or "").lower()
    hkind = str(row.get("hailo_runtime_kind") or "").lower()
    cstatus = str(row.get("component_measurement_status") or "").lower()
    if (
        explicit is False
        and mode != "strict"
        and "raw" in endpoint
        and "head" in endpoint
        and hkind == "full"
        and _bool(row.get("runtime_ok")) is not False
        and cstatus in {"full", "full_raw_head_e2e", "full_e2e"}
    ):
        return "runtime_contract_only", None, "raw_detection_head_full_runtime_is_not_a_decoded_task_validation_gate"

    if explicit is not None:
        if explicit is False and str(row.get("task_quality_status") or "").lower() == "pass":
            # A task-quality PASS must remain visible even when an independent
            # physical/interface validation fails. It cannot repair that gate.
            return "invalid", False, "measured_row_validation_failed; task_quality=pass"
        return ("valid" if explicit else "invalid"), explicit, "explicit_validation_ok"
    max_thr = _float(opts.get("max_abs_error_threshold"))
    mean_thr = _float(opts.get("mean_abs_error_threshold"))
    cos_thr = _float(opts.get("cosine_similarity_threshold"))
    numeric_known = False
    ok = True
    if numeric.get("max_abs_error") is not None and max_thr is not None:
        numeric_known = True
        if float(numeric["max_abs_error"]) > max_thr:
            ok = False
            reasons.append("max_abs_error_above_threshold")
    if numeric.get("mean_abs_error") is not None and mean_thr is not None:
        numeric_known = True
        if float(numeric["mean_abs_error"]) > mean_thr:
            ok = False
            reasons.append("mean_abs_error_above_threshold")
    if numeric.get("cosine_similarity") is not None and cos_thr is not None:
        numeric_known = True
        if float(numeric["cosine_similarity"]) < cos_thr:
            ok = False
            reasons.append("cosine_similarity_below_threshold")
    if numeric.get("output_shape_match") is not None:
        numeric_known = True
        if bool(numeric["output_shape_match"]) is False:
            ok = False
            reasons.append("output_shape_mismatch")
    task_known = any(v is not None for v in task_ev.values())
    if bool(opts.get("require_explicit")) and explicit is None:
        return "unvalidated", None, "explicit_validation_required_but_missing"
    if bool(opts.get("require_task_metrics")) and not task_known:
        return "unvalidated", None, "task_metrics_required_but_missing"
    if numeric_known:
        return ("valid" if ok else "invalid"), ok, ";".join(reasons) if reasons else "numeric_thresholds_passed"
    if task_known:
        return "task_metric_observed", None, "task_proxy_present_without_validation_gate"
    return "unvalidated", None, "no_validation_evidence_in_normalized_row"




def _raw_head_hailo_full_soft_gate(row: Mapping[str, Any]) -> bool:
    try:
        backend = str(row.get("backend") or "").lower()
        variant = str(row.get("variant") or "").lower()
        endpoint = str(row.get("endpoint_mode") or "").lower()
        status = str(row.get("component_measurement_status") or "").lower()
        val = str(row.get("validation_ok") or "").lower()
        return ("hailo" in backend and variant == "full" and ("raw" in endpoint or "raw_head" in status) and row.get("total_latency_ms") not in (None, "") and val in {"false", "0", "no", "fail", "failed"})
    except Exception:
        return False

def materialize_validation_binding(
    *,
    run_dir: str | Path,
    model_id: str,
    task: str = "auto",
    validation_preset: str = "",
    profile_payload: Mapping[str, Any] | None = None,
    options: Any = None,
    output_contracts: Mapping[str, Any] | None = None,
    normalized_results: Mapping[str, Any] | None = None,
    profile_row: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    root = Path(run_dir)
    model_dir = root / "models" / str(model_id)
    validation_dir = model_dir / "validation"
    normalized = dict(normalized_results or {}) if isinstance(normalized_results, Mapping) else (read_json(model_dir / "benchmark_results" / "normalized_results.json", default={}) or {})
    output_contracts = dict(output_contracts or {}) if isinstance(output_contracts, Mapping) else (read_json(model_dir / "full_baselines" / "output_contracts.json", default={}) or {})
    assets_status = read_json(validation_dir / "assets_status.json", default={}) or {}
    if not validation_preset and isinstance(profile_row, Mapping):
        validation_preset = str(profile_row.get("validation_preset") or profile_row.get("semantic_dataset") or "")
    if (not task or str(task).lower() == "auto") and isinstance(profile_row, Mapping):
        task = str(profile_row.get("task") or task or "auto")
    rows = [dict(x or {}) for x in _as_list(normalized.get("results")) if isinstance(x, Mapping)]
    opts = _validation_options(options, profile_payload)
    mode = str(opts.get("mode") or "summary_only").lower()
    plan = _adapter_plan(model_id=model_id, task=task, validation_preset=validation_preset, output_contracts=output_contracts, profile_payload=profile_payload)
    plan["validation_options"] = dict(opts)
    plan["assets_status_path"] = relpath(validation_dir / "assets_status.json", root)
    plan["normalized_results_path"] = relpath(model_dir / "benchmark_results" / "normalized_results.json", root)

    if str(opts.get("mode") or "").lower() == "disabled":
        summary = {
            "schema": "onnx-splitpoint/validation-summary",
            "schema_version": 3,
            "created_at": now_iso(),
            "model_id": model_id,
            "task": str(task or "auto").lower(),
            "validation_preset": validation_preset,
            "status": "disabled",
            "result_count": len(rows),
            "validated_result_count": 0,
            "valid_result_count": 0,
            "invalid_result_count": 0,
            "unvalidated_result_count": len(rows),
            "validation_ok": None,
            "numeric_validation": {},
            "task_validation": {},
            "notes": ["Validation summaries disabled by workflow option."],
        }
        p_plan = write_json(validation_dir / "validation_adapter_plan.json", plan)
        p_summary = write_json(validation_dir / "validation_summary.json", summary)
        p_md = write_text(validation_dir / "validation_binding.md", _validation_markdown(summary, plan, {"status": "disabled", "raw_head_contract_present": plan.get("output_contracts_summary", {}).get("raw_head_contract_present")}))
        return {
            "artifacts": {"validation_adapter_plan_json": p_plan, "validation_summary_json": p_summary, "validation_binding_md": p_md},
            "metrics": {"result_count": len(rows), "validation_status": "disabled"},
            "status": "skipped",
            "message": "Validation binding disabled by workflow option.",
        }

    case_rows: List[Dict[str, Any]] = []
    for row in rows:
        measured = _row_has_measurement(row)
        if measured:
            status, ok, reason = _decide_validation_status(row, task, opts)
        else:
            status, ok, reason = "unmeasured_placeholder", None, "diagnostic_or_planned_row_without_runtime_latency_validation_or_error_evidence"
        numeric = _row_numeric_evidence(row)
        task_ev = _row_task_evidence(row, task)
        interface_ev = _row_interface_evidence(row)
        case_rows.append({
            "model_id": model_id,
            "case_id": row.get("case_id", ""),
            "backend": row.get("backend", ""),
            "variant": row.get("variant", ""),
            "endpoint_mode": row.get("endpoint_mode", ""),
            "runtime_ok": row.get("runtime_ok", ""),
            "has_measurement": bool(measured),
            "validation_status": status,
            "validation_ok": ok if ok is not None else "",
            "validation_reason": reason,
            "component_measurement_status": row.get("component_measurement_status", ""),
            "total_latency_ms": row.get("total_latency_ms", ""),
            "full_raw_head_latency_ms": row.get("full_raw_head_latency_ms", ""),
            "source_validation_ok": row.get("validation_ok", ""),
            "runtime_contract_decision": row.get("runtime_contract_decision", ""),
            "structural_contract_status": row.get("structural_contract_status", ""),
            "task_quality_status": row.get("task_quality_status", "unavailable"),
            "task_quality_reason": row.get("task_quality_reason", ""),
            "central_quality_technical_status": row.get("central_quality_technical_status", ""),
            "max_abs_error": numeric.get("max_abs_error"),
            "mean_abs_error": numeric.get("mean_abs_error"),
            "cosine_similarity": numeric.get("cosine_similarity"),
            "output_shape_match": numeric.get("output_shape_match") if numeric.get("output_shape_match") is not None else "",
            "classification_top1": row.get("classification_top1", ""),
            "classification_top5": row.get("classification_top5", ""),
            "map_light": row.get("mini_coco_ap50_primary", ""),
            "detection_count": row.get("detection_count", ""),
            "confidence_mean": row.get("confidence_mean", ""),
            "source_path": row.get("source_path", ""),
            "interface_raw_head_drift_pass": interface_ev.get("raw_head_drift_pass") if interface_ev.get("raw_head_drift_pass") is not None else "",
            "interface_backend_drift_single_pass": interface_ev.get("backend_drift_single_pass") if interface_ev.get("backend_drift_single_pass") is not None else "",
            "interface_backend_drift_dataset_pass": interface_ev.get("backend_drift_dataset_pass") if interface_ev.get("backend_drift_dataset_pass") is not None else "",
            "interface_backend_drift_single_top1_match": interface_ev.get("backend_drift_single_top1_match") if interface_ev.get("backend_drift_single_top1_match") is not None else "",
            "interface_backend_drift_single_cosine": interface_ev.get("backend_drift_single_cosine_similarity"),
        })

    measured_case_rows = [r for r in case_rows if bool(r.get("has_measurement"))]
    diagnostic_case_rows = [r for r in case_rows if not bool(r.get("has_measurement"))]
    # v49p: planned/diagnostic placeholder rows from BenchmarkSets are not
    # validation gates.  They remain in validation_case_matrix for debugging, but
    # only rows with actual timing/runtime/validation evidence decide status.
    soft_validation_rows = [r for r in measured_case_rows if str(r.get("validation_status") or "") == "runtime_contract_only" or _raw_head_hailo_full_soft_gate(r)]
    validation_values = [r["validation_ok"] for r in measured_case_rows if r.get("validation_ok") != "" and r not in soft_validation_rows]
    valid_count = sum(1 for v in validation_values if bool(v))
    invalid_count = sum(1 for v in validation_values if not bool(v))
    soft_validation_count = len(soft_validation_rows)
    validated_count = len(validation_values)
    measured_count = len(measured_case_rows)
    raw_row_count = len(case_rows)
    diagnostic_placeholder_count = len(diagnostic_case_rows)
    unvalidated_count = measured_count - validated_count
    complete_split_rows = [r for r in measured_case_rows if _is_complete_split_row(r)]
    valid_complete_split_rows = [r for r in complete_split_rows if _row_final_pass(r) is True]
    contract_rejected_split_rows = [r for r in complete_split_rows if _is_contract_rejected(r)]
    component_or_optional_invalid_rows = [r for r in measured_case_rows if r not in complete_split_rows and _bool(r.get("validation_ok")) is False]
    complete_split_count = len(complete_split_rows)
    valid_complete_split_count = len(valid_complete_split_rows)
    contract_rejected_split_count = len(contract_rejected_split_rows)
    optional_invalid_count = len(component_or_optional_invalid_rows)
    shape_vals = [r.get("output_shape_match") for r in measured_case_rows if r.get("output_shape_match") != ""]
    strict_mode = str(opts.get("mode") or "summary_only").lower() == "strict"
    if not rows:
        status = "pending_benchmark_execution"
    elif not measured_case_rows:
        status = "diagnostic_rows_only"
    elif valid_complete_split_count and not strict_mode:
        # v55o: once thesis-relevant complete split rows validated, classify the
        # model as having valid split evidence even if optional/component rows are
        # invalid or not explicitly validated.  Those rows remain visible in the
        # matrix and summary counters, but they should not keep the workflow/model
        # job in a perpetual partial state.
        status = "valid_complete_splits_present"
    elif invalid_count:
        status = "validation_failed"
    elif validated_count and not unvalidated_count:
        status = "validated_with_diagnostic_placeholders" if diagnostic_placeholder_count else "validated"
    else:
        status = "partially_validated"
    task_l = str(task or "auto").lower()
    summary = {
        "schema": "onnx-splitpoint/validation-summary",
        "schema_version": 5,
        "created_at": now_iso(),
        "model_id": model_id,
        "task": task_l,
        "validation_preset": validation_preset,
        "status": status,
        "adapter_plan_path": "validation_adapter_plan.json",
        "case_matrix_path": "validation_case_matrix.json",
        "case_matrix_csv_path": "validation_case_matrix.csv",
        "assets_status": assets_status.get("assets", assets_status),
        "validation_scope": "measured_rows_only",
        "raw_result_row_count": raw_row_count,
        "diagnostic_placeholder_count": diagnostic_placeholder_count,
        "result_count": measured_count,
        "measured_result_count": measured_count,
        "validated_result_count": validated_count,
        "valid_result_count": valid_count,
        "invalid_result_count": invalid_count,
        "unvalidated_result_count": unvalidated_count,
        "soft_contract_pending_count": soft_validation_count,
        "validation_ok": (True if status == "valid_complete_splits_present" else ((invalid_count == 0 and (validated_count + soft_validation_count) == measured_count) if measured_count and (validated_count or soft_validation_count) else (None if measured_count else None))),
        "validation_quality": (
            ("valid_complete_splits_with_optional_invalid" if invalid_count else ("valid_complete_splits_with_optional_unvalidated" if unvalidated_count else "valid_complete_splits"))
            if status == "valid_complete_splits_present" else ("strict_invalid_rows" if status == "validation_failed" else status)
        ),
        "complete_split_count": complete_split_count,
        "valid_complete_split_count": valid_complete_split_count,
        "contract_rejected_split_count": contract_rejected_split_count,
        "optional_or_component_invalid_count": optional_invalid_count,
        "numeric_validation": {
            "max_abs_error": _max([r.get("max_abs_error") for r in measured_case_rows]),
            "mean_abs_error": _mean([r.get("mean_abs_error") for r in measured_case_rows]),
            "cosine_similarity": _mean([r.get("cosine_similarity") for r in measured_case_rows]),
            "output_shape_match_rate": _rate(shape_vals),
        },
        "task_validation": {
            "top1": _mean([r.get("classification_top1") for r in measured_case_rows]),
            "top5": _mean([r.get("classification_top5") for r in measured_case_rows]),
            "map_light": _mean([r.get("map_light") for r in measured_case_rows]),
            "detection_count_mean": _mean([r.get("detection_count") for r in measured_case_rows]),
            **_task_score_summary(measured_case_rows),
        },
        "notes": [
            "v49p validates at the workflow level by consuming normalized benchmark rows with actual measurement evidence.",
            "Diagnostic/planned placeholder rows stay in validation_case_matrix but do not force the validation stage to partial.",
            "Task adapters are bound and recorded; heavy dataset validation still has to emit metrics through the benchmark/remote service.",
            "v52d gates mixed accelerator split rows on explicit interface drift evidence before accepting composed accuracy.",
            "Raw-head Hailo full baselines with runtime/e2e latency are non-gating in summary_only mode until a decoded/task validation contract emits comparable metrics.",
            "In summary_only mode, validation_failed marks invalid benchmark cases but does not mean the workflow is incomplete; strict mode turns it into a blocking gate.",
            "v55g separates thesis-relevant valid complete splits from optional/component invalid rows; status valid_complete_splits_present means complete split evidence exists despite auxiliary failures.",
        ],
    }
    readiness = {
        "schema": "onnx-splitpoint/validation-readiness",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "status": "ready_to_validate_results" if measured_case_rows else ("diagnostic_rows_only" if rows else "waiting_for_benchmark_results"),
        "normalized_result_count": len(rows),
        "measured_result_count": measured_count,
        "diagnostic_placeholder_count": diagnostic_placeholder_count,
        "validation_preset": validation_preset,
        "assets_status_path": relpath(validation_dir / "assets_status.json", root),
        "adapter_ids": [a.get("adapter_id") for a in plan.get("adapters", [])],
        "raw_head_contract_present": plan.get("output_contracts_summary", {}).get("raw_head_contract_present"),
    }

    p_plan = write_json(validation_dir / "validation_adapter_plan.json", plan)
    p_matrix = write_json(validation_dir / "validation_case_matrix.json", {
        "schema": "onnx-splitpoint/validation-case-matrix",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "rows": case_rows,
        "row_count": len(case_rows),
    })
    fields = [
        "model_id", "case_id", "backend", "variant", "endpoint_mode", "runtime_ok", "has_measurement",
        "validation_status", "validation_ok", "validation_reason", "component_measurement_status",
        "total_latency_ms", "full_raw_head_latency_ms", "source_validation_ok", "max_abs_error",
        "runtime_contract_decision", "structural_contract_status", "task_quality_status",
        "task_quality_reason", "central_quality_technical_status",
        "mean_abs_error", "cosine_similarity", "output_shape_match", "classification_top1",
        "classification_top5", "map_light", "detection_count", "confidence_mean",
        "confidence_mean_clipped_0_1", "raw_score_mean", "score_semantics", "source_path",
        "interface_raw_head_drift_pass", "interface_backend_drift_single_pass",
        "interface_backend_drift_dataset_pass", "interface_backend_drift_single_top1_match",
        "interface_backend_drift_single_cosine",
    ]
    p_csv = write_csv(validation_dir / "validation_case_matrix.csv", case_rows, fields)
    gap_rows = [r for r in measured_case_rows if str(r.get("validation_status") or "") in {"unvalidated", "invalid"} and not _raw_head_hailo_full_soft_gate(r)]
    p_gaps = write_json(validation_dir / "validation_gaps.json", {
        "schema": "onnx-splitpoint/validation-gaps",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "gaps": gap_rows,
        "gap_count": len(gap_rows),
    })
    p_readiness = write_json(validation_dir / "validation_readiness.json", readiness)
    p_summary = write_json(validation_dir / "validation_summary.json", summary)
    p_md = write_text(validation_dir / "validation_binding.md", _validation_markdown(summary, plan, readiness))
    return {
        "artifacts": {
            "validation_adapter_plan_json": p_plan,
            "validation_case_matrix_json": p_matrix,
            "validation_case_matrix_csv": p_csv,
            "validation_gaps_json": p_gaps,
            "validation_readiness_json": p_readiness,
            "validation_summary_json": p_summary,
            "validation_binding_md": p_md,
        },
        "metrics": {
            "result_count": measured_count,
            "raw_result_row_count": raw_row_count,
            "diagnostic_placeholder_count": diagnostic_placeholder_count,
            "validated_result_count": validated_count,
            "valid_result_count": valid_count,
            "invalid_result_count": invalid_count,
            "soft_contract_pending_count": soft_validation_count,
            "unvalidated_result_count": unvalidated_count,
            "validation_status": status,
        },
        "status": (
            "ok" if status in {"validated", "validated_with_diagnostic_placeholders", "valid_complete_splits_present"}
            else ("skipped" if not rows else ("warn" if status in {"diagnostic_rows_only", "partially_validated", "validation_failed"} and not strict_mode else "partial"))
        ),
        "message": (
            "Validation binding summarized measured normalized results." if status == "validated"
            else ("Validation binding summarized measured results; diagnostic placeholders were ignored as non-gating." if status == "validated_with_diagnostic_placeholders"
            else ("Validation binding found valid complete split evidence; optional invalid/unvalidated rows were kept as non-blocking diagnostics." if status == "valid_complete_splits_present"
            else ("Validation produced invalid measured rows; runtime/interface validation and central task-quality decisions are reported separately. This validation warning is non-blocking in summary_only mode. Use validation.mode=strict to make this a blocking gate." if status == "validation_failed" and not strict_mode
            else ("Validation binding has measured rows but incomplete explicit validation metrics; non-blocking in summary_only mode." if status == "partially_validated" and not strict_mode
            else ("Only diagnostic/planned rows are present; waiting for measured benchmark rows." if status == "diagnostic_rows_only" else "Validation adapters/readiness recorded; waiting for benchmark results.")))))
        ),
    }


def _validation_markdown(summary: Mapping[str, Any], plan: Mapping[str, Any], readiness: Mapping[str, Any]) -> str:
    lines = [
        "# Validation Binding",
        "",
        f"Model: `{summary.get('model_id')}`",
        f"Task: `{summary.get('task')}`",
        f"Status: `{summary.get('status')}`",
        f"Result count: {summary.get('result_count')}",
        f"Validated: {summary.get('validated_result_count')} / {summary.get('result_count')}",
        "",
        "## Bound adapters",
        "",
    ]
    for adapter in _as_list(plan.get("adapters")):
        if isinstance(adapter, Mapping):
            lines.append(f"- `{adapter.get('adapter_id')}` ({adapter.get('level')}) — {adapter.get('status')}")
    lines.extend([
        "",
        "## Readiness",
        "",
        f"- `{readiness.get('status')}`",
        f"- Raw-head contract present: `{readiness.get('raw_head_contract_present')}`",
        "",
        "The workflow does not fabricate validation metrics. Missing values stay unvalidated until a local/remote benchmark service emits numeric or task metrics.",
        "",
    ])
    return "\n".join(lines)


def materialize_validation_summary(
    *,
    run_dir: str | Path,
    model_id: str,
    task: str = "auto",
    options: Any = None,
    validation_preset: str = "",
    profile_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compatibility wrapper used by older callers.

    The v49i binding reads normalized_results.json and output_contracts.json from
    the formal EvaluationRuns layout, so callers only need to provide the run
    directory and model id.
    """
    run_root = Path(run_dir)
    model_dir = run_root / "models" / str(model_id)
    if not validation_preset:
        manifest = read_json(model_dir / "model_manifest.json", default={}) or {}
        if isinstance(manifest, Mapping):
            validation_preset = str(manifest.get("validation_preset") or "")
            task = str(task or manifest.get("task") or "auto")
    return materialize_validation_binding(
        run_dir=run_root,
        model_id=model_id,
        task=task or "auto",
        validation_preset=validation_preset,
        profile_payload=profile_payload or {},
        options=options,
    )
