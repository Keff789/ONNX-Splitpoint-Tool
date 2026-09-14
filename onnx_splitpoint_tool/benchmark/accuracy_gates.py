"""Accuracy/eligibility gate helpers for benchmark, full, generic and native rows.

v59ek keeps a common gate vocabulary used across the tool:

- buildable
- runtime_executable
- contract_consistent
- task_valid
- accuracy_gate_pass
- eligible_for_ranking

The policy is intentionally conservative.  Contract/self-reference validation is a
separate gate and does not by itself satisfy a dataset accuracy gate.  Rows that
only have single-input self-reference evidence can be reported as contract
consistent, but are not eligible for accuracy-gated ranking unless dataset-level
accuracy metrics are present and within the configured threshold.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

from ..validation.host_postprocess import resolve_host_postprocess_evidence

DEFAULT_ACCURACY_GATE_POLICY: dict[str, Any] = {
    "schema": "onnx-splitpoint/accuracy-gate-policy",
    "schema_version": 1,
    "name": "strict_delta_1pp_v1",
    "classification": {
        "top1_max_drop_pp": 1.0,
        "top5_max_drop_pp": 1.0,
        "require_dataset_accuracy_for_ranking": True,
    },
    "detection": {
        "ap50_max_drop_pp": 1.0,
        "ap_max_drop_pp": 1.0,
        "require_dataset_accuracy_for_ranking": True,
    },
    "native_contract": {
        "classification_top1_required": True,
        "classification_min_top5_overlap": 1,
        "detection_min_match_ratio": 0.90,
        "detection_min_class_agnostic_match_ratio": 0.90,
        "self_reference_counts_as_accuracy_gate": False,
    },
}

_TRUE = {"1", "true", "yes", "ok", "pass", "passed", "claim_ok"}
_FALSE = {"0", "false", "no", "fail", "failed", "semantic_fail", "missing_dump"}
_NA = {"", "none", "null", "n/a", "na", "unavailable", "disabled"}


def default_policy() -> dict[str, Any]:
    return deepcopy(DEFAULT_ACCURACY_GATE_POLICY)


def merge_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = default_policy()
    if not isinstance(overrides, Mapping):
        return base

    def rec(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
                rec(dst[k], v)  # type: ignore[index]
            else:
                dst[k] = deepcopy(v)

    rec(base, overrides)
    return base


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    if s in _NA:
        return None
    return None


def as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        if isinstance(v, str) and v.strip().lower() in _NA:
            return None
        f = float(v)
        if f != f:
            return None
        return f
    except Exception:
        return None


def as_int(v: Any) -> int | None:
    f = as_float(v)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def percent_points_delta(primary: float | None, full: float | None) -> float | None:
    if primary is None or full is None:
        return None
    return (float(primary) - float(full)) * 100.0


def _first_number(row: Mapping[str, Any], keys: list[str]) -> float | None:
    for k in keys:
        v = as_float(row.get(k))
        if v is not None:
            return v
    return None


def infer_task(row: Mapping[str, Any]) -> str:
    raw = str(
        row.get("task")
        or row.get("benchmark_task")
        or row.get("benchmark_task_used")
        or row.get("benchmark_task_requested")
        or row.get("validation_dataset_task")
        or ""
    ).strip().lower()
    if raw in {"classification", "detection"}:
        return raw
    model = str(row.get("model") or row.get("model_name") or "").lower()
    if "yolo" in model or any(row.get(k) is not None for k in ("mini_coco_ap50_primary", "mini_coco_ap50_full", "ap50_proxy")):
        return "detection"
    return "classification"


def _classification_primary(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, [
        "mini_classification_eval_primary_top1",
        "mini_classification_eval_composed_top1",
        "validation_dataset_primary_top1_accuracy",
        "validation_dataset_primary_top1_agreement",
        "top1_accuracy",
        "top1",
    ])


def _classification_full(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, [
        "mini_classification_eval_full_top1",
        "validation_dataset_full_top1_accuracy",
        "full_top1_accuracy",
        "full_top1",
    ])


def _classification_primary_top5(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, [
        "mini_classification_eval_primary_top5",
        "mini_classification_eval_composed_top5",
        "validation_dataset_primary_top5_accuracy",
        "top5_accuracy",
        "top5",
    ])


def _classification_full_top5(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, [
        "mini_classification_eval_full_top5",
        "validation_dataset_full_top5_accuracy",
        "full_top5_accuracy",
        "full_top5",
    ])


def _detection_primary(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, [
        "mini_coco_ap50_primary",
        "mini_coco_ap50_composed",
        "detection_ap50",
        "coco_ap50",
        "map50",
    ])


def _detection_full(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, ["mini_coco_ap50_full", "full_ap50", "full_coco_ap50"])


def _already_has_dataset_accuracy(row: Mapping[str, Any], task: str) -> bool:
    if task == "classification":
        return _classification_primary(row) is not None and _classification_full(row) is not None
    return _detection_primary(row) is not None and _detection_full(row) is not None


def _metric_status_from_delta(delta_pp: float | None, max_drop_pp: float) -> tuple[bool | None, str]:
    if delta_pp is None:
        return None, "missing_dataset_delta"
    # Negative delta is a drop. Positive delta is allowed.
    if delta_pp >= -float(max_drop_pp):
        return True, "within_declared_delta"
    return False, "exceeds_declared_delta"


def _native_contract_consistent(row: Mapping[str, Any], policy: Mapping[str, Any], task: str) -> tuple[bool | None, str]:
    if task == "detection":
        family = str(row.get("contract_family") or row.get("output_format") or "").lower()
        raw_status = str(row.get("raw_head_contract_status") or "").lower()
        if "raw_head" in family or "raw_head" in raw_status or as_bool(row.get("raw_head_contract_present")) is True:
            host_evidence = resolve_host_postprocess_evidence(row)
            host_ok = host_evidence.get("available")
            decoder_ok = as_bool(row.get("decoder_contract_pass"))
            if decoder_ok is None:
                decoder_ok = as_bool(row.get("raw_head_decode_ok"))
            if decoder_ok is None and host_ok is True:
                decoder_ok = True
            nms_ok = as_bool(row.get("nms_ok"))
            if nms_ok is None and host_ok is True:
                nms_ok = True
            decoder_id = str(
                row.get("decoder_id")
                or row.get("decoder_family")
                or row.get("host_tail_model")
                or host_evidence.get("decoder_id")
                or ""
            ).strip()
            if not (decoder_ok is True and nms_ok is True and host_ok is True and decoder_id):
                return False, "raw_head_decoder_postprocess_contract_unresolved"
    semantic = as_bool(row.get("semantic_ok"))
    self_ref_ok = as_bool(row.get("self_reference_ok"))
    diag = str(row.get("self_reference_diagnosis") or "")
    source = str(row.get("semantic_reference_source") or "")
    if self_ref_ok is True or diag.startswith("native_") and "matches_full_self_reference" in diag:
        if task == "classification":
            if bool(policy.get("native_contract", {}).get("classification_top1_required", True)):
                return (True if as_bool(row.get("top1_match")) is True else False), "full_onnx_self_reference_top1"
            return True, "full_onnx_self_reference"
        if task == "detection":
            ratio = as_float(row.get("ap50_proxy"))
            min_ratio = float(policy.get("native_contract", {}).get("detection_min_match_ratio", 0.90))
            if ratio is None:
                # If the row only stored the diagnosis but not the ratio, trust the probe diagnosis.
                return True, "full_onnx_self_reference_probe"
            return bool(ratio >= min_ratio), "full_onnx_self_reference_match_ratio"
    if semantic is True and source:
        return True, "semantic_reference"
    if semantic is False:
        return False, "semantic_fail"
    return None, "contract_unavailable"


def apply_accuracy_gates(row: MutableMapping[str, Any], *, policy: Mapping[str, Any] | None = None, row_family: str = "generic") -> MutableMapping[str, Any]:
    """Annotate a result row with standardized gate fields.

    This function is intentionally side-effectful and returns *row* for convenience.
    """
    pol = merge_policy(policy)
    task = infer_task(row)
    row["task"] = row.get("task") or task

    # Build/runtime gates use the best available existing semantics for each row family.
    if row_family == "native":
        buildable = as_bool(row.get("native_row_ok"))
        if buildable is None:
            buildable = as_bool(row.get("report_ok"))
        if buildable is None:
            buildable = as_bool(row.get("row_ok"))
        if buildable is None:
            buildable = as_bool(row.get("status") == "native_row_not_ok") is False if row.get("status") else None
        runtime_executable = as_bool(row.get("tensor_ok"))
        if runtime_executable is None:
            runtime_executable = as_bool(row.get("ok"))
        contract_consistent, contract_reason = _native_contract_consistent(row, pol, task)
        semantic = as_bool(row.get("semantic_ok"))
        task_valid = semantic if semantic is not None else contract_consistent
    else:
        buildable = as_bool(row.get("buildable"))
        if buildable is None:
            buildable = as_bool(row.get("build_ok"))
        if buildable is None:
            buildable = as_bool(row.get("final_pass_all"))
        if buildable is None:
            buildable = as_bool(row.get("status")) is not False if row.get("status") else None
        runtime_executable = as_bool(row.get("runtime_executable"))
        if runtime_executable is None:
            runtime_executable = as_bool(row.get("run_ok"))
        if runtime_executable is None:
            runtime_executable = as_bool(row.get("final_pass_all"))
        if runtime_executable is None:
            runtime_executable = as_float(row.get("composed_mean_ms")) is not None or as_float(row.get("full_mean_ms")) is not None
        backend_key = str(row.get("backend") or row.get("pipeline") or "").lower().replace("-", "_")
        stage1 = str(row.get("stage1_provider") or "").lower()
        stage2 = str(row.get("stage2_provider") or "").lower()
        hailo_to_trt = (stage1.startswith("hailo") and stage2 in {"trt", "tensorrt", "ort_tensorrt"}) or ("hailo" in backend_key and "_to_trt" in backend_key)
        interface = as_bool(row.get("interface_contract_pass"))
        if interface is None:
            interface = as_bool(row.get("interface_check_pass"))
        drift_status = str(row.get("backend_drift_status") or "").strip().lower()
        if hailo_to_trt:
            contract_consistent = interface
            contract_reason = "hailo_trt_interface_contract" if interface is not None else "hailo_trt_interface_contract_missing"
        elif drift_status:
            contract_consistent = drift_status in {"ok", "pass", "passed", "disabled", "n/a", "na"}
            contract_reason = f"backend_drift_status:{drift_status}"
        else:
            contract_consistent = as_bool(row.get("semantic_validation_passed"))
            if contract_consistent is None:
                contract_consistent = as_bool(row.get("final_pass_all"))
            contract_reason = "semantic_validation_or_final_pass" if contract_consistent is not None else "contract_unavailable"
        task_valid = as_bool(row.get("final_pass_all"))
        if task_valid is None:
            task_valid = as_bool(row.get("semantic_validation_passed"))

    # Accuracy gate is dataset-level; self-reference is contract only unless explicitly allowed.
    accuracy_gate_pass: bool | None
    accuracy_reason: str
    accuracy_metric: str = ""
    accuracy_delta_pp: float | None = None
    threshold_pp: float | None = None
    primary: float | None = None
    full: float | None = None

    if task == "classification":
        primary = _classification_primary(row)
        full = _classification_full(row)
        top5_primary = _classification_primary_top5(row)
        top5_full = _classification_full_top5(row)
        threshold_pp = float(pol.get("classification", {}).get("top1_max_drop_pp", 1.0))
        accuracy_delta_pp = percent_points_delta(primary, full)
        accuracy_gate_pass, accuracy_reason = _metric_status_from_delta(accuracy_delta_pp, threshold_pp)
        accuracy_metric = "top1_delta_vs_full_pp"
        if accuracy_gate_pass is not False and top5_primary is not None and top5_full is not None:
            top5_thr = float(pol.get("classification", {}).get("top5_max_drop_pp", 1.0))
            top5_delta = percent_points_delta(top5_primary, top5_full)
            top5_pass, top5_reason = _metric_status_from_delta(top5_delta, top5_thr)
            row["accuracy_gate_top5_delta_pp"] = top5_delta
            row["accuracy_gate_top5_threshold_pp"] = top5_thr
            row["accuracy_gate_top5_pass"] = top5_pass
            if top5_pass is False:
                accuracy_gate_pass = False
                accuracy_reason = "top5_" + top5_reason
    else:
        primary = _detection_primary(row)
        full = _detection_full(row)
        threshold_pp = float(pol.get("detection", {}).get("ap50_max_drop_pp", 1.0))
        accuracy_delta_pp = percent_points_delta(primary, full)
        accuracy_gate_pass, accuracy_reason = _metric_status_from_delta(accuracy_delta_pp, threshold_pp)
        accuracy_metric = "ap50_delta_vs_full_pp"

    # Full rows with a quality metric and no delta are acceptable baselines.
    if accuracy_gate_pass is None and str(row.get("case") or row.get("kind") or "").lower() == "full":
        if primary is not None or full is not None:
            accuracy_gate_pass = True
            accuracy_reason = "full_baseline_quality_present"
            accuracy_delta_pp = 0.0

    # Explicit self-reference contract can optionally satisfy the accuracy gate, but default is False/None.
    if accuracy_gate_pass is None and row_family == "native" and as_bool(row.get("self_reference_ok")) is True:
        if bool(pol.get("native_contract", {}).get("self_reference_counts_as_accuracy_gate", False)):
            accuracy_gate_pass = True
            accuracy_reason = "self_reference_contract_allowed_by_policy"
        else:
            accuracy_reason = "no_dataset_accuracy_self_reference_only"

    # Normalize None => False only for ranking; keep None in the report to distinguish unavailable vs failed.
    backend_id = str(row.get("backend") or row.get("provider") or "").lower().replace("-", "_")
    semantic_reference_only = bool(as_bool(row.get("canonical_reference_row")) is True or backend_id in {"cpu", "cpu_ort", "ort_cpu", "canonical_full_onnx"})
    eligible = all(v is True for v in (buildable, runtime_executable, contract_consistent, task_valid, accuracy_gate_pass)) and not semantic_reference_only

    row["buildable"] = buildable
    row["runtime_executable"] = runtime_executable
    row["contract_consistent"] = contract_consistent
    row["contract_gate_reason"] = contract_reason
    row["task_valid"] = task_valid
    row["accuracy_gate_pass"] = accuracy_gate_pass
    row["accuracy_gate_reason"] = accuracy_reason
    row["accuracy_gate_metric"] = accuracy_metric
    row["accuracy_gate_delta_pp"] = accuracy_delta_pp
    row["accuracy_gate_threshold_pp"] = threshold_pp
    row["accuracy_gate_reference_value"] = full
    row["accuracy_gate_candidate_value"] = primary
    row["accuracy_gate_policy"] = pol.get("name")
    row["eligible_for_ranking"] = bool(eligible)
    if not eligible:
        blockers = []
        for name, val in [
            ("buildable", buildable),
            ("runtime_executable", runtime_executable),
            ("contract_consistent", contract_consistent),
            ("task_valid", task_valid),
            ("accuracy_gate_pass", accuracy_gate_pass),
        ]:
            if val is not True:
                blockers.append(name)
        row["ranking_exclusion_reason"] = ",".join(blockers)
    else:
        row["ranking_exclusion_reason"] = ""
    if semantic_reference_only:
        row["ranking_exclusion_reason"] = "semantic_reference_only"
    row["execution_ok"] = bool(buildable is True and runtime_executable is True)
    row["interface_valid"] = bool(contract_consistent is True)
    row["interface_status"] = "pass" if contract_consistent is True else ("fail" if contract_consistent is False else "unavailable")
    row["quality_valid"] = bool(task_valid is True and accuracy_gate_pass is True)
    row["evidence_complete"] = bool(row["execution_ok"] and contract_consistent in {True, False} and accuracy_gate_pass in {True, False})
    row["semantic_reference_only"] = semantic_reference_only
    row["ranking_eligible"] = bool(eligible)
    row["performance_eligible"] = bool(eligible)
    row["energy_eligible"] = False
    row["pareto_eligible"] = bool(eligible)
    row["thesis_valid"] = bool(eligible)
    return row


def summarize_gate_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "buildable_count": sum(1 for r in rows if r.get("buildable") is True),
        "runtime_executable_count": sum(1 for r in rows if r.get("runtime_executable") is True),
        "contract_consistent_count": sum(1 for r in rows if r.get("contract_consistent") is True),
        "task_valid_count": sum(1 for r in rows if r.get("task_valid") is True),
        "accuracy_gate_pass_count": sum(1 for r in rows if r.get("accuracy_gate_pass") is True),
        "eligible_for_ranking_count": sum(1 for r in rows if r.get("eligible_for_ranking") is True),
    }
