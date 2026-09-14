"""Unified task-validity and ranking eligibility gates.

The project deliberately separates three concepts that were historically
conflated in some reports:

* build/runtime success: the artifact can be built and executed.
* contract consistency: a native/generic hand-off reproduces the selected full
  reference on a concrete tensor/image contract.
* task accuracy gate: a dataset-level task metric is within the declared
  tolerance of the full reference/baseline.

Only rows that pass the task accuracy gate are eligible for speed/energy ranking.
Native one-image Full-ONNX self-reference checks are recorded as contract gates;
they do not, by themselves, satisfy the dataset-level accuracy gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..validation.host_postprocess import resolve_host_postprocess_evidence


@dataclass(frozen=True)
class AccuracyGateConfig:
    schema: str = "onnx-splitpoint/accuracy-gate-config"
    schema_version: int = 1
    # Metrics in reports are normalized fractions, e.g. 0.82.  One percentage
    # point is therefore 0.01.
    classification_max_top1_drop: float = 0.01
    classification_max_top5_drop: float = 0.01
    detection_max_ap50_drop: float = 0.01
    detection_max_ap_drop: float = 0.01
    contract_min_detection_match_ratio: float = 0.90
    contract_min_classification_top5_overlap: int = 1
    require_dataset_accuracy_for_ranking: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "AccuracyGateConfig":
        if not raw:
            return cls()
        data = {k: v for k, v in dict(raw).items() if k in cls.__dataclass_fields__}
        for k in list(data):
            try:
                if k.startswith("classification_max") or k.startswith("detection_max") or k.startswith("contract_min_detection"):
                    data[k] = float(data[k])
                elif k == "contract_min_classification_top5_overlap":
                    data[k] = int(data[k])
                elif k == "require_dataset_accuracy_for_ranking":
                    data[k] = _as_bool(data[k])
            except Exception:
                data.pop(k, None)
        return cls(**data)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        if isinstance(v, bool):
            return None
        x = float(v)
        if x != x:
            return None
        # If a report uses percentage points, normalize to a fraction.  Values
        # in [0, 1.5] are treated as fractions; larger magnitudes are assumed
        # to be percentages.
        if abs(x) > 1.5:
            return x / 100.0
        return x
    except Exception:
        return None


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "ok", "pass", "passed", "claim_ok"}


def _state(v: Any) -> str:
    if v is True:
        return "pass"
    if v is False:
        return "fail"
    s = str(v or "").strip().lower()
    if s in {"pass", "passed", "true", "ok", "claim_ok"}:
        return "pass"
    if s in {"fail", "failed", "false", "semantic_fail"}:
        return "fail"
    if s in {"unavailable", "not_evaluated", "n/a", "na", "none", ""}:
        return "not_evaluated"
    return s


def task_from_row(row: Mapping[str, Any]) -> str:
    task = str(row.get("task") or row.get("benchmark_task") or row.get("validation_dataset_task") or "").lower()
    model = str(row.get("model") or "").lower()
    if task in {"classification", "detection"}:
        return task
    return "detection" if "yolo" in model else "classification"


def infer_buildable(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or row.get("build_status") or "").lower()
    if status in {"failed", "fail", "error", "missing_dump", "native_row_not_ok"}:
        return False
    if any(k in row for k in ("build_ok", "ok")):
        return bool(_as_bool(row.get("build_ok", row.get("ok"))))
    return status not in {"failed", "fail", "error"}


def infer_runtime_executable(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    if status in {"missing_dump", "native_row_not_ok", "failed", "fail", "error"}:
        return False
    if "runtime_executable" in row:
        return _as_bool(row.get("runtime_executable"))
    if "tensor_ok" in row:
        return _as_bool(row.get("tensor_ok"))
    if "ok" in row:
        return _as_bool(row.get("ok"))
    return status not in {"failed", "fail", "error"}


def infer_contract_consistent(row: Mapping[str, Any], cfg: AccuracyGateConfig | None = None) -> str:
    cfg = cfg or AccuracyGateConfig()
    diag = str(row.get("self_reference_diagnosis") or row.get("diagnosis") or "").lower()
    source = str(row.get("semantic_reference_source") or "").lower()
    task = task_from_row(row)
    if task == "detection":
        family = str(row.get("contract_family") or row.get("output_format") or "").lower()
        raw_status = str(row.get("raw_head_contract_status") or "").lower()
        if "raw_head" in family or "raw_head" in raw_status or _as_bool(row.get("raw_head_contract_present")):
            host_evidence = resolve_host_postprocess_evidence(row)
            host_ok = host_evidence.get("available")
            decoder_value = (
                row.get("decoder_contract_pass")
                if row.get("decoder_contract_pass") is not None
                else row.get("raw_head_decode_ok")
            )
            decoder_ok = (
                _as_bool(decoder_value)
                if decoder_value is not None
                else host_ok is True
            )
            if not decoder_ok and decoder_value is None and host_ok is True:
                decoder_ok = True
            nms_value = row.get("nms_ok")
            nms_ok = (
                _as_bool(nms_value)
                if nms_value is not None
                else host_ok is True
            )
            if not nms_ok and nms_value is None and host_ok is True:
                nms_ok = True
            decoder_id = str(
                row.get("decoder_id")
                or row.get("decoder_family")
                or row.get("host_tail_model")
                or host_evidence.get("decoder_id")
                or ""
            ).strip()
            if not (decoder_ok and nms_ok and host_ok and decoder_id):
                return "fail"
    backend = str(row.get("backend") or row.get("pipeline") or "").lower().replace("-", "_")
    stage1 = str(row.get("stage1_provider") or "").lower()
    stage2 = str(row.get("stage2_provider") or "").lower()
    if (stage1.startswith("hailo") and stage2 in {"trt", "tensorrt", "ort_tensorrt"}) or ("hailo" in backend and "_to_trt" in backend):
        interface = row.get("interface_contract_pass")
        if interface is None:
            interface = row.get("interface_check_pass")
        return "pass" if _as_bool(interface) else ("fail" if interface is not None else "not_evaluated")
    if "matches_full_self_reference" in diag or "native_semantic_matches_full_self_reference" in diag:
        return "pass"
    if "full_onnx_self_reference" in source:
        if task == "detection":
            r = _as_float(row.get("ap50_proxy") or row.get("match_ratio") or row.get("best_match_ratio"))
            if r is not None:
                return "pass" if r >= cfg.contract_min_detection_match_ratio else "fail"
        else:
            if _as_bool(row.get("top1_match")):
                return "pass"
            ov = row.get("top5_overlap")
            try:
                if ov is not None and int(ov) >= cfg.contract_min_classification_top5_overlap:
                    return "pass"
            except Exception:
                pass
    # A dataset task-valid row also implies contract consistency at the task level.
    if _state(row.get("task_valid")) == "pass" or _state(row.get("semantic_ok")) == "pass" and "external" in source:
        return "pass"
    if _state(row.get("semantic_ok")) == "fail":
        return "fail"
    return "not_evaluated"


def dataset_accuracy_gate_from_row(row: Mapping[str, Any], cfg: AccuracyGateConfig | None = None) -> dict[str, Any]:
    """Evaluate only dataset-level accuracy evidence.

    One-sample native Full-ONNX self-reference is intentionally *not* accepted
    here.  It is handled by infer_contract_consistent().
    """
    cfg = cfg or AccuracyGateConfig()
    task = task_from_row(row)
    source = str(row.get("semantic_reference_source") or "").lower()

    # Native self-reference checks are contract evidence only.
    if "full_onnx_self_reference" in source:
        return {
            "task_valid": "not_evaluated",
            "accuracy_gate_pass": "not_evaluated",
            "accuracy_gate_reason": "self_reference_is_contract_only",
            "accuracy_gate_metric": "none",
            "accuracy_gate_delta": None,
            "accuracy_gate_threshold": cfg.classification_max_top1_drop if task == "classification" else cfg.detection_max_ap50_drop,
        }

    if task == "classification":
        # Prefer explicit deltas, else compute primary-full.  Metric values are
        # fractions where 0.01 == one percentage point.
        delta = _as_float(row.get("classification_top1_delta") or row.get("top1_delta") or row.get("mini_classification_eval_delta_primary_minus_full_top1"))
        primary = _as_float(row.get("primary_top1") or row.get("top1") or row.get("mini_classification_eval_primary_top1") or row.get("validation_dataset_primary_top1_accuracy"))
        full = _as_float(row.get("full_top1") or row.get("reference_top1") or row.get("mini_classification_eval_full_top1"))
        if delta is None and primary is not None and full is not None:
            delta = primary - full
        if delta is not None:
            ok = delta >= -abs(cfg.classification_max_top1_drop)
            return {
                "task_valid": bool(ok),
                "accuracy_gate_pass": bool(ok),
                "accuracy_gate_reason": "classification_top1_delta_gate",
                "accuracy_gate_metric": "top1_delta_vs_full",
                "accuracy_gate_delta": delta,
                "accuracy_gate_threshold": cfg.classification_max_top1_drop,
            }
        return {
            "task_valid": "not_evaluated",
            "accuracy_gate_pass": "not_evaluated",
            "accuracy_gate_reason": "classification_dataset_metric_missing",
            "accuracy_gate_metric": "top1_delta_vs_full",
            "accuracy_gate_delta": None,
            "accuracy_gate_threshold": cfg.classification_max_top1_drop,
        }

    # Detection
    delta = _as_float(row.get("detection_ap50_delta") or row.get("ap50_delta") or row.get("mini_coco_ap50_delta_primary_minus_full"))
    primary = _as_float(row.get("ap50") or row.get("ap50_proxy") or row.get("mini_coco_ap50_primary"))
    full = _as_float(row.get("full_ap50") or row.get("reference_ap50") or row.get("mini_coco_ap50_full"))
    if delta is None and primary is not None and full is not None:
        delta = primary - full
    # A metric called ap50_proxy with self-reference removed is still not a
    # dataset AP50 unless it has a full/reference comparator.  Require delta.
    if delta is not None:
        ok = delta >= -abs(cfg.detection_max_ap50_drop)
        return {
            "task_valid": bool(ok),
            "accuracy_gate_pass": bool(ok),
            "accuracy_gate_reason": "detection_ap50_delta_gate",
            "accuracy_gate_metric": "ap50_delta_vs_full",
            "accuracy_gate_delta": delta,
            "accuracy_gate_threshold": cfg.detection_max_ap50_drop,
        }
    return {
        "task_valid": "not_evaluated",
        "accuracy_gate_pass": "not_evaluated",
        "accuracy_gate_reason": "detection_dataset_metric_missing",
        "accuracy_gate_metric": "ap50_delta_vs_full",
        "accuracy_gate_delta": None,
        "accuracy_gate_threshold": cfg.detection_max_ap50_drop,
    }


def apply_accuracy_gates(row: dict[str, Any], cfg: AccuracyGateConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AccuracyGateConfig()
    buildable = infer_buildable(row)
    runtime_executable = infer_runtime_executable(row)
    contract = infer_contract_consistent(row, cfg)
    dataset = dataset_accuracy_gate_from_row(row, cfg)
    accuracy_state = _state(dataset.get("accuracy_gate_pass"))
    eligible = bool(
        buildable
        and runtime_executable
        and _state(contract) == "pass"
        and accuracy_state == "pass"
    )
    reason = "eligible" if eligible else ""
    if not reason:
        if not buildable:
            reason = "not_buildable"
        elif not runtime_executable:
            reason = "not_runtime_executable"
        elif _state(contract) != "pass":
            reason = f"contract_{_state(contract)}"
        elif accuracy_state != "pass":
            reason = f"accuracy_gate_{accuracy_state}:{dataset.get('accuracy_gate_reason')}"
        else:
            reason = "not_eligible"
    backend = str(row.get("backend") or row.get("provider") or "").lower().replace("-", "_")
    semantic_reference_only = bool(_as_bool(row.get("canonical_reference_row")) or backend in {"cpu", "cpu_ort", "ort_cpu", "canonical_full_onnx"})
    if semantic_reference_only:
        eligible = False
        reason = "semantic_reference_only"
    execution_ok = bool(buildable and runtime_executable)
    interface_valid = _state(contract) == "pass"
    quality_valid = accuracy_state == "pass"
    row.update({
        "buildable": bool(buildable),
        "runtime_executable": bool(runtime_executable),
        "execution_ok": execution_ok,
        "contract_consistent": contract,
        "interface_valid": interface_valid,
        "interface_status": _state(contract),
        **dataset,
        "quality_valid": quality_valid,
        "evidence_complete": bool(execution_ok and _state(contract) in {"pass", "fail"} and accuracy_state in {"pass", "fail"}),
        "eligible_for_ranking": bool(eligible),
        "ranking_eligible": bool(eligible),
        "performance_eligible": bool(eligible),
        "energy_eligible": False,
        "pareto_eligible": bool(eligible),
        "thesis_valid": bool(eligible),
        "semantic_reference_only": semantic_reference_only,
        "ranking_exclusion_reason": reason,
        "gate_config": cfg.to_json(),
    })
    return row


def apply_decision_row_gates(row: dict[str, Any], cfg: AccuracyGateConfig | None = None) -> dict[str, Any]:
    """Apply gates to generic/full decision rows.

    Decision rows already expose quality_delta_vs_full for the selected task.
    This helper maps that value into the same gate vocabulary used by native
    validation summaries.
    """
    cfg = cfg or AccuracyGateConfig()
    task = str(row.get("task") or "detection")
    delta = _as_float(row.get("quality_delta_vs_full") or row.get("ap50_delta_vs_full"))
    contract = "pass" if _as_float(row.get("backend_match_vs_cpu")) is None or (_as_float(row.get("backend_match_vs_cpu")) or 0.0) >= 0.5 else "fail"
    row.update({
        "buildable": True,
        "runtime_executable": _as_float(row.get("streaming_fps")) is not None or _as_float(row.get("latency_ms")) is not None,
        "contract_consistent": contract,
    })
    if delta is None:
        # Full baselines are task-valid only if they have a quality metric; they
        # do not have a delta against themselves in some legacy reports.
        if str(row.get("kind")) == "full" and _as_float(row.get("quality")) is not None:
            row.update({
                "task_valid": True,
                "accuracy_gate_pass": True,
                "accuracy_gate_reason": "full_baseline_reference_metric_present",
                "accuracy_gate_metric": "full_quality",
                "accuracy_gate_delta": 0.0,
                "accuracy_gate_threshold": cfg.classification_max_top1_drop if task == "classification" else cfg.detection_max_ap50_drop,
            })
        else:
            row.update({
                "task_valid": "not_evaluated",
                "accuracy_gate_pass": "not_evaluated",
                "accuracy_gate_reason": "quality_delta_missing",
                "accuracy_gate_metric": "quality_delta_vs_full",
                "accuracy_gate_delta": None,
                "accuracy_gate_threshold": cfg.classification_max_top1_drop if task == "classification" else cfg.detection_max_ap50_drop,
            })
    else:
        threshold = cfg.classification_max_top1_drop if task == "classification" else cfg.detection_max_ap50_drop
        ok = delta >= -abs(threshold)
        row.update({
            "task_valid": bool(ok),
            "accuracy_gate_pass": bool(ok),
            "accuracy_gate_reason": f"{task}_quality_delta_gate",
            "accuracy_gate_metric": "quality_delta_vs_full",
            "accuracy_gate_delta": delta,
            "accuracy_gate_threshold": threshold,
        })
    eligible = bool(row.get("buildable") and row.get("runtime_executable") and _state(row.get("contract_consistent")) == "pass" and _state(row.get("accuracy_gate_pass")) == "pass")
    row["eligible_for_ranking"] = eligible
    if eligible:
        row["ranking_exclusion_reason"] = "eligible"
    elif _state(row.get("accuracy_gate_pass")) != "pass":
        row["ranking_exclusion_reason"] = f"accuracy_gate_{_state(row.get('accuracy_gate_pass'))}:{row.get('accuracy_gate_reason')}"
    elif _state(row.get("contract_consistent")) != "pass":
        row["ranking_exclusion_reason"] = f"contract_{_state(row.get('contract_consistent'))}"
    else:
        row["ranking_exclusion_reason"] = "not_eligible"
    row["gate_config"] = cfg.to_json()
    backend = str(row.get("backend") or row.get("provider") or "").lower().replace("-", "_")
    semantic_reference_only = backend in {"cpu", "cpu_ort", "ort_cpu", "canonical_full_onnx"} or _as_bool(row.get("canonical_reference_row"))
    if semantic_reference_only:
        row["eligible_for_ranking"] = False
        row["ranking_exclusion_reason"] = "semantic_reference_only"
    row["execution_ok"] = bool(row.get("buildable") and row.get("runtime_executable"))
    row["interface_valid"] = _state(row.get("contract_consistent")) == "pass"
    row["interface_status"] = _state(row.get("contract_consistent"))
    row["quality_valid"] = _state(row.get("accuracy_gate_pass")) == "pass"
    row["evidence_complete"] = bool(row["execution_ok"] and row["interface_status"] in {"pass", "fail"} and _state(row.get("accuracy_gate_pass")) in {"pass", "fail"})
    row["semantic_reference_only"] = bool(semantic_reference_only)
    row["ranking_eligible"] = bool(row.get("eligible_for_ranking"))
    row["performance_eligible"] = bool(row.get("eligible_for_ranking"))
    row["energy_eligible"] = False
    row["pareto_eligible"] = bool(row.get("eligible_for_ranking"))
    row["thesis_valid"] = bool(row.get("eligible_for_ranking"))
    return row
