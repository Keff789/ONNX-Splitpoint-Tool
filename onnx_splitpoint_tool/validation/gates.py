"""Central task/contract/accuracy gate policy.

This module intentionally separates three concepts that were previously easy
for reports to conflate:

* runtime/build success
* contract consistency (tensor/self-reference/native-output consistency)
* dataset task validity / accuracy gate

A row is eligible for performance ranking only if all gates are satisfied.
Native Full-ONNX self-reference is a strong contract check, but by design it is
not a dataset-level accuracy check and therefore does not by itself make a row
eligible for ranking.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

from .host_postprocess import resolve_host_postprocess_evidence

Tri = Any

DEFAULT_GATE_POLICY: Dict[str, Any] = {
    "schema": "onnx-splitpoint/accuracy-gate-policy",
    "schema_version": 1,
    "name": "strict_task_gate_v1",
    "description": (
        "Rows are rank-eligible only when build/runtime, contract consistency, "
        "and dataset task-accuracy gates pass. Native self-reference is a contract "
        "gate, not a dataset accuracy gate."
    ),
    "classification": {
        "max_top1_drop_pp": 1.0,
        "max_top5_drop_pp": 1.0,
        "require_dataset_accuracy": True,
    },
    "detection": {
        "max_ap_drop_pp": 1.0,
        "max_ap50_drop_pp": 1.0,
        "require_dataset_accuracy": True,
    },
    "contract": {
        "native_detection_selfref_min_match_ratio": 0.90,
        "native_classification_require_top1_match": True,
        "backend_drift_min_match_ratio": 0.90,
        "backend_drift_min_top1_agreement": 0.99,
    },
}


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    s = str(v).strip()
    if not s or s.lower() in {"none", "nan", "n/a", "na", "unavailable"}:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        if float(v) == 1.0:
            return True
        if float(v) == 0.0:
            return False
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "ok", "pass", "passed", "claim_ok"}:
        return True
    if s in {"0", "false", "no", "n", "off", "fail", "failed", "semantic_fail"}:
        return False
    if s in {"", "none", "n/a", "na", "unavailable"}:
        return None
    return None


def _is_true(v: Any) -> bool:
    return _as_bool(v) is True


def _is_false(v: Any) -> bool:
    return _as_bool(v) is False


def _first_float(row: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        x = _as_float(row.get(k))
        if x is not None:
            return x
    return None


def _scale_threshold(value_a: Optional[float], value_b: Optional[float], pp: float) -> float:
    """Return a threshold in the same units as values.

    Metrics in this project can appear either as fractions [0,1] or as percent
    points [0,100].  We infer the scale from observed values.
    """
    vals = [abs(x) for x in (value_a, value_b) if x is not None]
    if vals and max(vals) > 1.5:
        return float(pp)
    return float(pp) / 100.0


def _infer_task(row: Mapping[str, Any]) -> str:
    raw = str(
        row.get("task")
        or row.get("benchmark_task")
        or row.get("benchmark_task_used")
        or row.get("validation_dataset_task")
        or ""
    ).strip().lower()
    if raw in {"classification", "detection"}:
        return raw
    if any(row.get(k) is not None for k in (
        "top1_match", "top5_overlap", "mini_classification_eval_primary_top1",
        "mini_classification_eval_full_top1", "validation_dataset_primary_top1_accuracy",
    )):
        return "classification"
    if any(row.get(k) is not None for k in (
        "ap50_proxy", "mini_coco_ap50_primary", "mini_coco_ap50_full", "mini_coco_ap50_composed",
    )):
        return "detection"
    model = str(row.get("model") or "").lower()
    if "yolo" in model or "detr" in model:
        return "detection"
    return "classification"


def _native_backend(row: Mapping[str, Any]) -> bool:
    b = str(row.get("backend") or row.get("setup") or row.get("pipeline") or "").lower()
    return any(tok in b for tok in ("_to_trt", "native_", "hailo", "deepx")) or bool(row.get("native_report"))


def _dataset_quality_values(row: Mapping[str, Any], task: str) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Return (candidate, reference, direct_delta, source)."""
    if task == "classification":
        cand = _first_float(row, [
            "mini_classification_eval_primary_top1",
            "mini_classification_eval_composed_top1",
            "validation_dataset_primary_top1_accuracy",
            "validation_dataset_primary_top1_agreement",
        ])
        ref = _first_float(row, [
            "mini_classification_eval_full_top1",
            "mini_classification_eval_cpu_full_top1",
            "validation_dataset_full_top1_accuracy",
        ])
        delta = _first_float(row, [
            "mini_classification_eval_delta_primary_minus_full_top1",
            "validation_dataset_delta_primary_minus_full_top1",
        ])
        return cand, ref, delta, "classification_top1_dataset"
    cand = _first_float(row, [
        "mini_coco_ap50_primary",
        "mini_coco_ap50_composed",
        "validation_dataset_primary_ap50",
        "validation_dataset_primary_ap50_proxy",
    ])
    ref = _first_float(row, [
        "mini_coco_ap50_full",
        "validation_dataset_full_ap50",
        "validation_dataset_full_ap50_proxy",
    ])
    delta = _first_float(row, [
        "mini_coco_ap50_delta_primary_minus_full",
        "validation_dataset_delta_primary_minus_full_ap50",
    ])
    return cand, ref, delta, "detection_ap50_dataset"


def _task_gate(row: Mapping[str, Any], task: str, policy: Mapping[str, Any]) -> Dict[str, Any]:
    cand, ref, direct_delta, source = _dataset_quality_values(row, task)
    out: Dict[str, Any] = {
        "task_gate_source": source,
        "accuracy_candidate_metric": cand,
        "accuracy_reference_metric": ref,
        "accuracy_delta": None,
        "accuracy_gate_threshold": None,
        "task_valid": "unavailable",
        "accuracy_gate_pass": False,
        "accuracy_gate_reason": "dataset_accuracy_metric_missing",
    }
    if cand is None or ref is None:
        # A full baseline row may only carry the full metric.  Treat it as its
        # own reference if the row is explicitly full.
        kind = str(row.get("kind") or row.get("case") or "").lower()
        backend = str(row.get("backend") or row.get("setup") or "").lower()
        is_full = kind == "full" or "full" in backend or str(row.get("case") or "").lower() == "full"
        if is_full and (cand is not None or ref is not None):
            x = cand if cand is not None else ref
            cand = x
            ref = x
        else:
            return out
    delta = direct_delta if direct_delta is not None else (float(cand) - float(ref))
    if task == "classification":
        pp = float(((policy.get("classification") or {}).get("max_top1_drop_pp", 1.0)))
    else:
        pp = float(((policy.get("detection") or {}).get("max_ap50_drop_pp", 1.0)))
    thr = _scale_threshold(cand, ref, pp)
    passed = bool(delta >= -thr)
    out.update({
        "accuracy_delta": delta,
        "accuracy_gate_threshold": thr,
        "task_valid": passed,
        "accuracy_gate_pass": passed,
        "accuracy_gate_reason": "pass" if passed else "accuracy_drop_exceeds_threshold",
    })
    return out


def _contract_gate(row: Mapping[str, Any], task: str, policy: Mapping[str, Any]) -> Dict[str, Any]:
    contract_policy = policy.get("contract") or {}
    source = str(row.get("semantic_reference_source") or "").strip().lower()
    diag = str(row.get("self_reference_diagnosis") or row.get("diagnosis") or "").strip().lower()
    backend = str(row.get("backend") or row.get("setup") or row.get("pipeline") or "").strip().lower()
    sem_ok = row.get("semantic_ok")
    tensor_ok = row.get("tensor_ok")
    out: Dict[str, Any] = {
        "contract_consistent": "unavailable",
        "contract_gate_source": "none",
        "contract_gate_reason": "contract_metric_missing",
    }
    # v2.63: detection tensors explicitly archived as a raw head are not a
    # decoded task result.  A self-reference match or a successful runtime may
    # not hide a missing/failing decoder and NMS tail.
    if task == "detection":
        family = str(row.get("contract_family") or row.get("output_format") or "").lower()
        raw_status = str(row.get("raw_head_contract_status") or "").lower()
        raw = bool("raw_head" in family or "raw_head" in raw_status or _is_true(row.get("raw_head_contract_present")))
        if raw:
            host_evidence = resolve_host_postprocess_evidence(row)
            host_available = host_evidence.get("available")
            decoder = _as_bool(row.get("decoder_contract_pass"))
            if decoder is None:
                decoder = _as_bool(row.get("raw_head_decode_ok"))
            if decoder is None and host_available is True:
                decoder = True
            nms = _as_bool(row.get("nms_ok"))
            if nms is None and host_available is True:
                nms = True
            decoder_id = str(
                row.get("decoder_id")
                or row.get("decoder_family")
                or row.get("host_tail_model")
                or host_evidence.get("decoder_id")
                or ""
            ).strip()
            if not (decoder is True and nms is True and host_available is True and decoder_id):
                out.update({
                    "contract_consistent": False,
                    "contract_gate_source": "raw_head_postprocess_contract",
                    "contract_gate_reason": "raw_head_decoder_postprocess_contract_unresolved",
                })
                return out
    backend_key = str(row.get("backend") or row.get("pipeline") or "").lower().replace("-", "_")
    stage1 = str(row.get("stage1_provider") or "").lower()
    stage2 = str(row.get("stage2_provider") or "").lower()
    hailo_to_trt = (stage1.startswith("hailo") and stage2 in {"trt", "tensorrt", "ort_tensorrt"}) or ("hailo" in backend_key and "_to_trt" in backend_key)
    if hailo_to_trt:
        iface = _as_bool(row.get("interface_contract_pass"))
        if iface is None:
            iface = _as_bool(row.get("interface_check_pass"))
        if iface is not True:
            out.update({
                "contract_consistent": False if iface is False else "unavailable",
                "contract_gate_source": "hailo_trt_interface_contract",
                "contract_gate_reason": "interface_contract_failed" if iface is False else "interface_contract_missing",
            })
            return out
    # Native self-reference is a contract gate.  It checks a concrete boundary,
    # output decoder, and input image; it is not a dataset accuracy gate.
    if source == "full_onnx_self_reference" or "self_reference" in diag:
        if task == "detection":
            m = _first_float(row, ["ap50_proxy", "best_match_ratio", "match_ratio"])
            min_match = float(contract_policy.get("native_detection_selfref_min_match_ratio", 0.90))
            ok = bool(_is_true(sem_ok) or (m is not None and m >= min_match) or "matches" in diag)
            out.update({
                "contract_consistent": ok,
                "contract_gate_source": "full_onnx_self_reference_detection",
                "contract_gate_reason": "pass" if ok else "self_reference_match_below_threshold",
            })
            return out
        top1 = _as_bool(row.get("top1_match"))
        top5ov = _as_float(row.get("top5_overlap"))
        ok = bool(top1 is True or (top5ov is not None and top5ov >= 3) or "matches" in diag)
        out.update({
            "contract_consistent": ok,
            "contract_gate_source": "full_onnx_self_reference_classification",
            "contract_gate_reason": "pass" if ok else "self_reference_topk_mismatch",
        })
        return out
    # Generic rows often expose final_pass_all or backend drift metrics.
    final_pass = _as_bool(row.get("final_pass_all"))
    if final_pass is not None:
        out.update({
            "contract_consistent": final_pass,
            "contract_gate_source": "final_pass_all",
            "contract_gate_reason": "pass" if final_pass else "final_pass_all_false",
        })
        return out
    drift = _first_float(row, [
        "backend_drift_dataset_global_match_ratio",
        "backend_drift_single_match_ratio",
        "backend_drift_dataset_top1_agreement",
        "backend_drift_single_top1_match",
    ])
    if drift is not None:
        minv = float(contract_policy.get("backend_drift_min_top1_agreement" if task == "classification" else "backend_drift_min_match_ratio", 0.90))
        ok = bool(drift >= minv)
        out.update({
            "contract_consistent": ok,
            "contract_gate_source": "backend_drift_metric",
            "contract_gate_reason": "pass" if ok else "backend_drift_below_threshold",
        })
        return out
    if _is_true(tensor_ok):
        out.update({
            "contract_consistent": True,
            "contract_gate_source": "tensor_ok",
            "contract_gate_reason": "pass_tensor_only",
        })
    return out


def apply_accuracy_gates(row: Mapping[str, Any], *, policy: Optional[Mapping[str, Any]] = None, runner_kind: str = "auto") -> Dict[str, Any]:
    """Return a copy of *row* augmented with gate fields.

    The function is conservative: missing dataset accuracy metrics yield
    ``task_valid='unavailable'`` and ``eligible_for_ranking=False``.
    """
    pol = copy.deepcopy(DEFAULT_GATE_POLICY)
    if policy:
        # shallow recursive-ish merge for known sections
        for k, v in policy.items():
            if isinstance(v, Mapping) and isinstance(pol.get(k), dict):
                pol[k].update(v)  # type: ignore[index]
            else:
                pol[k] = v
    r: Dict[str, Any] = dict(row)
    task = _infer_task(r)
    r["task"] = r.get("task") or task

    # Build/runtime gates are intentionally broad to cover generic, full and native rows.
    status = str(r.get("status") or "").strip().lower()
    row_ok = _as_bool(r.get("ok"))
    build_ok = _as_bool(r.get("build_ok"))
    tensor_ok = _as_bool(r.get("tensor_ok"))
    native_ok = _as_bool(r.get("native_ok"))
    failed_status = any(tok in status for tok in ("fail", "error", "missing", "not_ok"))
    buildable = bool((build_ok is True) or (row_ok is True) or (native_ok is True) or (status in {"ok", "claim_ok"})) and not failed_status
    runtime_executable = bool((row_ok is True) or (native_ok is True) or (tensor_ok is True) or status in {"ok", "claim_ok"}) and not failed_status
    # If a validation row is semantic-unavailable but produced tensor/artifact,
    # it remains executable but not rank-eligible.
    if tensor_ok is True and not failed_status:
        runtime_executable = True

    contract = _contract_gate(r, task, pol)
    task_gate = _task_gate(r, task, pol)
    r.update(contract)
    r.update(task_gate)
    r["buildable"] = bool(buildable)
    r["runtime_executable"] = bool(runtime_executable)
    # Explicitly document that self-reference is not a dataset accuracy gate.
    if str(r.get("semantic_reference_source") or "").lower() == "full_onnx_self_reference":
        if r.get("task_valid") == "unavailable":
            r["accuracy_gate_reason"] = "self_reference_is_contract_not_dataset_accuracy"
    eligible = bool(
        r.get("buildable") is True
        and r.get("runtime_executable") is True
        and r.get("contract_consistent") is True
        and r.get("task_valid") is True
        and r.get("accuracy_gate_pass") is True
    )
    backend_key = str(r.get("backend") or r.get("provider") or "").lower().replace("-", "_")
    semantic_reference_only = bool(
        _is_true(r.get("canonical_reference_row"))
        or backend_key in {"cpu", "cpu_ort", "ort_cpu", "canonical_full_onnx"}
    )
    if semantic_reference_only:
        eligible = False
    r["eligible_for_ranking"] = eligible
    if semantic_reference_only:
        r["gate_status"] = "semantic_reference_only"
    elif eligible:
        r["gate_status"] = "eligible_for_ranking"
    elif r.get("contract_consistent") is True and r.get("task_valid") == "unavailable":
        r["gate_status"] = "contract_consistent_task_accuracy_unavailable"
    elif r.get("accuracy_gate_pass") is False and r.get("task_valid") is False:
        r["gate_status"] = "accuracy_gate_fail"
    elif r.get("runtime_executable") is not True:
        r["gate_status"] = "runtime_not_executable"
    elif r.get("buildable") is not True:
        r["gate_status"] = "not_buildable"
    elif r.get("contract_consistent") is not True:
        r["gate_status"] = "contract_gate_fail_or_unavailable"
    else:
        r["gate_status"] = "not_eligible"
    r["accuracy_gate_policy"] = pol.get("name")
    r["execution_ok"] = bool(buildable and runtime_executable)
    r["interface_valid"] = bool(r.get("contract_consistent") is True)
    r["interface_status"] = "pass" if r.get("contract_consistent") is True else ("fail" if r.get("contract_consistent") is False else "unavailable")
    r["quality_valid"] = bool(r.get("task_valid") is True and r.get("accuracy_gate_pass") is True)
    r["evidence_complete"] = bool(r["execution_ok"] and r.get("contract_consistent") in {True, False} and r.get("task_valid") in {True, False})
    r["semantic_reference_only"] = semantic_reference_only
    r["ranking_eligible"] = bool(eligible)
    r["performance_eligible"] = bool(eligible)
    r["energy_eligible"] = False
    r["pareto_eligible"] = bool(eligible)
    r["thesis_valid"] = bool(eligible)
    if semantic_reference_only:
        r["ranking_exclusion_reason"] = "semantic_reference_only"
    return r


def apply_accuracy_gates_to_rows(rows: Iterable[Mapping[str, Any]], *, policy: Optional[Mapping[str, Any]] = None, runner_kind: str = "auto") -> list[Dict[str, Any]]:
    return [apply_accuracy_gates(r, policy=policy, runner_kind=runner_kind) for r in rows]
