"""Pre-registered task-quality and row-eligibility gates.

The canonical gate is a dataset-level non-inferiority decision against a
canonical full-model reference.  Runtime self-reference remains an I/O contract
gate and never substitutes for task quality unless an explicit compatibility
policy opts into that weaker interpretation.

This module intentionally keeps backwards-compatible function names used by the
older dashboard and benchmark code.  New reports should consume the explicit
``task_quality_gate`` block and the ``*_eligible`` fields written here.
"""
from __future__ import annotations

from ..quality_result_contract import UNCERTAINTY_FIELDS, project_quality_result, project_flat_quality_uncertainty

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from .host_postprocess import (
    apply_host_postprocess_aliases,
    resolve_host_postprocess_evidence,
)


@dataclass(frozen=True)
class AccuracyGatePolicy:
    schema: str = "onnx-splitpoint/task-quality-policy"
    schema_version: int = 3
    name: str = "thesis_task_quality_v2"
    profile_id: str = "thesis_task_quality_v2"
    frozen_before_final_campaign: bool = False
    dataset_tier: str = "screening"  # screening | final
    canonical_reference: str = "canonical_full_onnx"
    confidence_level: float = 0.95
    # Standard/development profiles use 500 repetitions.  Final campaign
    # profiles carry their (normally larger) value explicitly.  Keeping 2,000
    # here used to create a hidden normaliser policy whenever a runtime row did
    # not forward its embedded policy, invalidating otherwise sound evidence.
    bootstrap_repetitions: int = 500
    bootstrap_seed: int = 20260710
    decision_rule: str = "lower_one_sided_bound"
    classification_primary_metric: str = "top1_accuracy"
    classification_max_top1_drop: float = 0.01
    classification_max_top5_drop: float = 0.01
    detection_primary_metric: str = "coco_ap_50_95"
    detection_max_ap_drop: float = 0.01
    detection_max_ap50_drop: float = 0.01
    # v2 makes the Native detection similarity rule explicit and hash-bound.
    # It replaces the historical hidden ``0.80 or 75%-count`` implementation.
    # Schema-v2 snapshots retain their archived 0.90 match-ratio rule; they
    # predate the separate mean-IoU guard.
    native_self_reference_policy_id: str = "class_aware_iou50_postnms_v2"
    native_self_reference_min_match: float = 0.80
    native_self_reference_min_mean_iou: float = 0.85
    native_self_reference_iou_threshold: float = 0.50
    native_self_reference_confidence_threshold: float = 0.25
    native_self_reference_denominator: str = "reference_detections"
    native_self_reference_class_aware: bool = True
    numerical_similarity_required_for_claim: bool = True
    contract_only_eligible_for_ranking: bool = False
    screening_eligible_for_ranking: bool = False
    legacy_point_estimate_eligible_for_ranking: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | str | Path | None = None) -> "AccuracyGatePolicy":
        if data is None or data == "":
            return cls()
        if isinstance(data, (str, Path)):
            s = str(data).strip()
            if s in {"", "default", "strict", "thesis_task_quality_v2", "thesis_default", "thesis_default_v2"}:
                return cls()
            if s in {"thesis_task_quality_v1", "thesis_default_v1"}:
                return cls(
                    schema_version=2,
                    name="thesis_task_quality_v1",
                    profile_id="thesis_task_quality_v1",
                    native_self_reference_policy_id="legacy_match_ratio_v1",
                    native_self_reference_min_match=0.90,
                    native_self_reference_min_mean_iou=0.0,
                )
            # Inline JSON can be much longer than the filesystem filename
            # limit.  Parse JSON-looking strings before touching pathlib;
            # otherwise Path.is_file() can raise ENAMETOOLONG and silently force
            # a runtime policy fallback.
            if s[:1] in {"{", "["}:
                data = json.loads(s)
            else:
                p = Path(s).expanduser()
                if p.is_file():
                    text = p.read_text(encoding="utf-8")
                    try:
                        data = json.loads(text)
                    except Exception:
                        try:
                            import yaml  # optional runtime dependency in the full tool
                            data = yaml.safe_load(text) or {}
                        except Exception as exc:
                            raise ValueError(f"Could not parse task-quality policy {p}: {exc}") from exc
                else:
                    data = json.loads(s)
        if not isinstance(data, Mapping):
            return cls()

        # Accept either a direct policy block or a complete evaluation profile.
        if isinstance(data.get("quality_gate"), Mapping):
            data = data["quality_gate"]  # type: ignore[index]

        flat: dict[str, Any] = {}
        for key in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if key in data:
                flat[key] = data.get(key)

        statistics = data.get("statistics") if isinstance(data.get("statistics"), Mapping) else {}
        flat.setdefault("confidence_level", statistics.get("confidence_level"))
        flat.setdefault("bootstrap_repetitions", statistics.get("bootstrap_repetitions", statistics.get("repetitions")))
        flat.setdefault("bootstrap_seed", statistics.get("seed"))
        flat.setdefault("decision_rule", statistics.get("decision", statistics.get("decision_rule")))

        c = data.get("classification") if isinstance(data.get("classification"), Mapping) else {}
        flat.setdefault("classification_primary_metric", c.get("primary_metric"))
        flat.setdefault("classification_max_top1_drop", c.get("non_inferiority_margin", c.get("non_inferiority_margin_fraction", c.get("max_top1_drop", c.get("top1_max_drop")))))
        guard = c.get("guardrails") if isinstance(c.get("guardrails"), Mapping) else {}
        flat.setdefault("classification_max_top5_drop", guard.get("top5_accuracy_margin", guard.get("top5_accuracy_margin_fraction", c.get("max_top5_drop", c.get("top5_max_drop")))))

        d = data.get("detection") if isinstance(data.get("detection"), Mapping) else {}
        flat.setdefault("detection_primary_metric", d.get("primary_metric"))
        flat.setdefault("detection_max_ap_drop", d.get("non_inferiority_margin", d.get("non_inferiority_margin_fraction", d.get("max_ap_drop", d.get("ap_max_drop")))))
        dguard = d.get("guardrails") if isinstance(d.get("guardrails"), Mapping) else {}
        flat.setdefault("detection_max_ap50_drop", dguard.get("ap50_margin", dguard.get("ap50_margin_fraction", d.get("max_ap50_drop", d.get("ap50_max_drop")))))

        n = data.get("native_contract") if isinstance(data.get("native_contract"), Mapping) else {}
        detection_self_reference = (
            n.get("detection_self_reference")
            if isinstance(n.get("detection_self_reference"), Mapping)
            else {}
        )
        flat.setdefault(
            "native_self_reference_policy_id",
            detection_self_reference.get("policy_id", n.get("detection_self_reference_policy_id")),
        )
        flat.setdefault(
            "native_self_reference_min_match",
            detection_self_reference.get(
                "min_reference_match_ratio",
                n.get("detection_self_reference_min_match_ratio", n.get("native_self_reference_min_match")),
            ),
        )
        flat.setdefault(
            "native_self_reference_min_mean_iou",
            detection_self_reference.get(
                "min_mean_matched_iou",
                n.get("detection_self_reference_min_mean_iou"),
            ),
        )
        flat.setdefault(
            "native_self_reference_iou_threshold",
            detection_self_reference.get(
                "iou_threshold",
                n.get("detection_self_reference_iou_threshold"),
            ),
        )
        flat.setdefault(
            "native_self_reference_confidence_threshold",
            detection_self_reference.get(
                "confidence_threshold",
                n.get("detection_self_reference_confidence_threshold"),
            ),
        )
        flat.setdefault(
            "native_self_reference_denominator",
            detection_self_reference.get(
                "denominator",
                n.get("detection_self_reference_denominator"),
            ),
        )
        flat.setdefault(
            "native_self_reference_class_aware",
            detection_self_reference.get(
                "class_aware",
                n.get("detection_self_reference_class_aware"),
            ),
        )
        flat.setdefault(
            "numerical_similarity_required_for_claim",
            n.get("numerical_similarity_required_for_claim"),
        )
        flat.setdefault("contract_only_eligible_for_ranking", n.get("self_reference_counts_as_task_accuracy", n.get("contract_only_eligible_for_ranking")))

        aliases = {
            "classification_top1_max_abs_drop": "classification_max_top1_drop",
            "classification_top5_max_abs_drop": "classification_max_top5_drop",
            "detection_ap50_max_abs_drop": "detection_max_ap50_drop",
            "detection_ap_max_abs_drop": "detection_max_ap_drop",
            "max_top1_drop": "classification_max_top1_drop",
            "max_top5_drop": "classification_max_top5_drop",
            "max_ap_drop": "detection_max_ap_drop",
            "max_ap50_drop": "detection_max_ap50_drop",
            "native_min_match": "native_self_reference_min_match",
            "native_min_mean_iou": "native_self_reference_min_mean_iou",
            "self_reference_is_task_gate": "contract_only_eligible_for_ranking",
            "allow_contract_only_ranking": "contract_only_eligible_for_ranking",
            "tier": "dataset_tier",
            "frozen": "frozen_before_final_campaign",
        }
        for k, v in data.items():
            if k in aliases:
                flat[aliases[k]] = v

        kwargs: dict[str, Any] = {}
        bool_fields = {
            "frozen_before_final_campaign",
            "contract_only_eligible_for_ranking",
            "screening_eligible_for_ranking",
            "legacy_point_estimate_eligible_for_ranking",
            "native_self_reference_class_aware",
            "numerical_similarity_required_for_claim",
        }
        int_fields = {"schema_version", "bootstrap_repetitions", "bootstrap_seed"}
        str_fields = {
            "schema", "name", "profile_id", "dataset_tier", "canonical_reference",
            "decision_rule", "classification_primary_metric", "detection_primary_metric",
            "native_self_reference_policy_id",
            "native_self_reference_denominator",
        }
        for k in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            v = flat.get(k)
            if v is None:
                continue
            if k in bool_fields:
                kwargs[k] = _as_bool(v) is True
            elif k in int_fields:
                try:
                    kwargs[k] = int(v)
                except Exception:
                    pass
            elif k in str_fields:
                kwargs[k] = str(v)
            else:
                f = _as_float(v)
                if f is not None:
                    kwargs[k] = f
        out = cls(**kwargs)
        if int(out.schema_version) < 3:
            # Preserve the exact semantics and stable policy hash of archived
            # schema-v2 snapshots.  New v2-only fields are omitted by
            # ``as_dict`` below and therefore cannot silently reinterpret an
            # older 0.90 match-ratio decision under the new 0.80 + IoU guard
            # contract.
            if "native_self_reference_min_match" not in kwargs:
                object.__setattr__(
                    out, "native_self_reference_min_match", 0.90
                )
            if "native_self_reference_policy_id" not in kwargs:
                object.__setattr__(
                    out, "native_self_reference_policy_id",
                    "legacy_match_ratio_v1",
                )
            if "native_self_reference_min_mean_iou" not in kwargs:
                object.__setattr__(out, "native_self_reference_min_mean_iou", 0.0)
        tier = str(out.dataset_tier or "screening").strip().lower()
        if tier not in {"screening", "final"}:
            object.__setattr__(out, "dataset_tier", "screening")
        return out

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if int(self.schema_version) < 3:
            for key in (
                "native_self_reference_policy_id",
                "native_self_reference_min_mean_iou",
                "native_self_reference_iou_threshold",
                "native_self_reference_confidence_threshold",
                "native_self_reference_denominator",
                "native_self_reference_class_aware",
                "numerical_similarity_required_for_claim",
            ):
                payload.pop(key, None)
        return payload

    def to_json(self) -> dict[str, Any]:
        return self.as_dict()

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def sha256(self) -> str:
        """Return a stable hash of the effective, normalised policy.

        The runtime template, normaliser and scientific reporter all use the
        same effective-policy hash.  This makes accidental fallback from the
        profile value (for example 5,000 bootstrap repetitions) to a runner
        default (for example 500 repetitions) visible and claim-blocking.
        """
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_POLICY = AccuracyGatePolicy()

_TRUE = {"1", "true", "yes", "y", "ok", "pass", "passed", "claim_ok", "valid", "eligible"}
_FALSE = {"0", "false", "no", "n", "fail", "failed", "invalid", "semantic_fail", "error"}
_EMPTY = {"", "none", "null", "na", "n/a", "unavailable", "unknown"}


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if float(v) == 1.0:
            return True
        if float(v) == 0.0:
            return False
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            if s.lower() in _EMPTY:
                return None
            if s.endswith("%"):
                return float(s[:-1].strip()) / 100.0
        x = float(v)
        return x if x == x else None
    except Exception:
        return None


def evaluate_detection_similarity(
    match: Mapping[str, Any] | None,
    class_agnostic_match: Mapping[str, Any] | None = None,
    policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one class-aware post-NMS self-reference comparison.

    The result is a numerical-similarity axis only.  It deliberately says
    nothing about the structural endpoint contract or dataset task quality.
    """

    pol = (
        policy
        if isinstance(policy, AccuracyGatePolicy)
        else AccuracyGatePolicy.from_mapping(policy)
    )
    observed = match if isinstance(match, Mapping) else {}
    diagnostic = (
        class_agnostic_match
        if isinstance(class_agnostic_match, Mapping)
        else {}
    )
    try:
        ref_count = max(0, int(observed.get("ref_count") or 0))
        matched = max(0, int(observed.get("matched") or 0))
    except (TypeError, ValueError):
        ref_count, matched = 0, 0
    ratio = (matched / ref_count) if ref_count > 0 else None
    mean_iou = _as_float(observed.get("mean_iou"))
    observed_iou_threshold = _as_float(observed.get("iou_threshold"))
    if observed_iou_threshold is None:
        observed_iou_threshold = pol.native_self_reference_iou_threshold
    ratio_pass = bool(
        ratio is not None
        and ratio >= float(pol.native_self_reference_min_match)
    )
    mean_iou_required = float(pol.native_self_reference_min_mean_iou) > 0.0
    mean_iou_pass = bool(
        not mean_iou_required
        or (
            mean_iou is not None
            and mean_iou >= float(pol.native_self_reference_min_mean_iou)
        )
    )
    iou_contract_pass = math.isclose(
        float(observed_iou_threshold),
        float(pol.native_self_reference_iou_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    available = bool(
        ref_count > 0
        and (not mean_iou_required or mean_iou is not None)
    )
    passed = bool(
        available
        and pol.native_self_reference_class_aware
        and ratio_pass
        and mean_iou_pass
        and iou_contract_pass
    )
    if not available:
        status = "unavailable"
        reason = "class_aware_reference_matches_unavailable"
        pass_value: Any = "unavailable"
    elif not pol.native_self_reference_class_aware:
        status = "failed"
        reason = "class_aware_matching_required"
        pass_value = False
    elif not iou_contract_pass:
        status = "failed"
        reason = "matching_iou_threshold_policy_mismatch"
        pass_value = False
    elif not ratio_pass:
        status = "failed"
        reason = "reference_match_ratio_below_threshold"
        pass_value = False
    elif not mean_iou_pass:
        status = "failed"
        reason = "mean_matched_iou_below_threshold"
        pass_value = False
    else:
        status = "passed"
        reason = "class_aware_match_ratio_and_mean_iou_passed"
        pass_value = True
    return {
        "numerical_similarity_pass": pass_value,
        "numerical_similarity_status": status,
        "numerical_similarity_reason": reason,
        "numerical_similarity_scope": "class_aware_postnms_detection",
        "numerical_similarity_metric": (
            "reference_match_ratio_and_mean_matched_iou"
        ),
        "numerical_similarity_value": ratio,
        "numerical_similarity_threshold": float(
            pol.native_self_reference_min_match
        ),
        "numerical_similarity_mean_iou": mean_iou,
        "numerical_similarity_mean_iou_threshold": float(
            pol.native_self_reference_min_mean_iou
        ),
        "numerical_similarity_policy_id": str(
            pol.native_self_reference_policy_id
        ),
        "numerical_similarity_iou_threshold": float(
            pol.native_self_reference_iou_threshold
        ),
        "numerical_similarity_confidence_threshold": float(
            pol.native_self_reference_confidence_threshold
        ),
        "numerical_similarity_denominator": str(
            pol.native_self_reference_denominator
        ),
        "numerical_similarity_class_aware": bool(
            pol.native_self_reference_class_aware
        ),
        "numerical_similarity_reference_count": ref_count,
        "numerical_similarity_matched_count": matched,
        "class_agnostic_match_ratio_diagnostic": _as_float(
            diagnostic.get("match_ratio")
        ),
    }


def _first(row: Mapping[str, Any], keys: tuple[str, ...] | list[str]) -> Any:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def _embedded_policy_mapping(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the policy that was actually used by the runtime, if archived.

    Runtime evidence is authoritative.  In particular, callers that omit the
    optional ``policy=`` argument must not silently replace an archived
    500-repetition Standard policy with a module default.  The accepted paths
    cover both current rows and older validation reports.
    """
    for key in ("task_quality_policy", "effective_quality_policy"):
        value = row.get(key)
        if isinstance(value, Mapping) and value:
            effective = value.get("effective_policy")
            return effective if isinstance(effective, Mapping) and effective else value
    for key in ("task_quality_gate", "quality_gate", "non_inferiority_gate"):
        gate = row.get(key)
        if not isinstance(gate, Mapping):
            continue
        value = gate.get("policy")
        if isinstance(value, Mapping) and value:
            effective = value.get("effective_policy")
            return effective if isinstance(effective, Mapping) and effective else value
    value = row.get("accuracy_gate_policy")
    if isinstance(value, Mapping) and value:
        return value
    return None


def resolve_effective_policy(
    row: Mapping[str, Any],
    policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None,
) -> tuple[AccuracyGatePolicy, str]:
    """Resolve exactly one effective policy and report its provenance."""
    if isinstance(policy, AccuracyGatePolicy):
        return policy, "explicit"
    if policy not in (None, ""):
        return AccuracyGatePolicy.from_mapping(policy), "explicit"
    embedded = _embedded_policy_mapping(row)
    if embedded is not None:
        return AccuracyGatePolicy.from_mapping(embedded), "runtime_embedded"
    return DEFAULT_POLICY, "compatibility_default_500"


def _infer_task(row: Mapping[str, Any]) -> str:
    raw = str(row.get("task") or row.get("benchmark_task") or row.get("benchmark_task_used") or row.get("validation_dataset_task") or "").strip().lower()
    if raw in {"classification", "detection"}:
        return raw
    model = str(row.get("model") or row.get("model_id") or "").lower()
    if "yolo" in model or "detr" in model:
        return "detection"
    if any(row.get(k) is not None for k in ("classification_top1", "mini_classification_eval_primary_top1", "top1_match")):
        return "classification"
    return "detection" if any(row.get(k) is not None for k in ("mini_coco_ap50_primary", "ap50_proxy", "detection_ap50")) else "classification"


def _is_cpu_semantic_reference(row: Mapping[str, Any]) -> bool:
    backend = str(row.get("backend") or row.get("full_backend") or row.get("provider") or "").strip().lower().replace("-", "_")
    reference = str(row.get("canonical_reference") or row.get("semantic_reference_source") or "").strip().lower()
    explicit = _as_bool(_first(row, ("canonical_reference_row", "task_quality_reference_row", "is_canonical_full_reference")))
    return bool(
        explicit is True
        or backend in {"cpu", "cpu_ort", "ort_cpu", "onnxruntime_cpu", "canonical_full_onnx"}
        or ("canonical" in reference and "full" in reference and "cpu" in reference)
    )


def _pipeline_requires_explicit_interface(row: Mapping[str, Any]) -> bool:
    backend = str(row.get("backend") or row.get("pipeline") or row.get("run_id") or "").strip().lower().replace("-", "_")
    stage1 = str(row.get("stage1_provider") or "").strip().lower()
    stage2 = str(row.get("stage2_provider") or "").strip().lower()
    return bool(
        (stage1.startswith("hailo") and stage2 in {"trt", "tensorrt", "ort_tensorrt"})
        or ("hailo" in backend and any(tok in backend for tok in ("_to_trt", "_to_tensorrt")))
    )


def _explicit_interface_result(row: Mapping[str, Any]) -> tuple[Optional[bool], str]:
    nested = row.get("interface_checks") if isinstance(row.get("interface_checks"), Mapping) else {}
    candidates = (
        row.get("structural_contract_pass"),
        row.get("interface_contract_pass"),
        row.get("interface_check_pass"),
        nested.get("pass") if isinstance(nested, Mapping) else None,
    )
    for value in candidates:
        parsed = _as_bool(value)
        if parsed is not None:
            return parsed, "explicit_hailo_trt_interface_contract"
    return None, "hailo_trt_interface_contract_missing"


def _detection_postprocess_contract(row: Mapping[str, Any]) -> tuple[Optional[bool], str]:
    """Fail closed when raw detection tensors lack a frozen decoder/NMS tail."""
    summary = row.get("deployment_contract_summary") if isinstance(row.get("deployment_contract_summary"), Mapping) else {}
    contract_family = str(row.get("contract_family") or summary.get("contract_family") or "").strip().lower()
    output_format = str(row.get("output_format") or summary.get("output_format") or "").strip().lower()
    raw_status = str(row.get("raw_head_contract_status") or summary.get("raw_head_contract_status") or "").strip().lower()
    raw_present = _as_bool(_first(row, ("raw_head_contract_present", "requires_host_decode_nms", "requires_external_postprocess")))
    if raw_present is None:
        raw_present = _as_bool(summary.get("raw_head_contract_present"))
    raw_markers = {
        "raw_head", "raw_detection_head", "raw_detection_tensors",
        "yolo26_one2one_raw_head", "raw_head_only", "missing_host_tail",
    }
    is_raw = bool(
        raw_present is True
        or contract_family in raw_markers
        or output_format in raw_markers
        or raw_status in raw_markers
        or "raw_head" in contract_family
        or "raw_head" in raw_status
    )

    host_evidence = resolve_host_postprocess_evidence(row, summary)
    host_available = host_evidence.get("available")
    explicit_decoder_pass = _as_bool(_first(row, (
        "decoder_contract_pass", "postprocess_contract_pass", "raw_head_decode_ok", "host_tail_decode_ok",
    )))
    if explicit_decoder_pass is None and host_available is True:
        explicit_decoder_pass = True
    if explicit_decoder_pass is False:
        return False, "detection_decoder_contract_failed"
    if not is_raw:
        explicit_post = _as_bool(_first(row, ("postprocess_included", "postprocess_ok", "nms_ok")))
        if explicit_post is False:
            return False, "detection_postprocess_explicitly_absent"
        return None, "not_a_raw_head_contract"

    # A raw-head artefact and a decoded [B,N,6] declaration are contradictory
    # unless a separate, frozen host-tail output is explicitly identified.
    decoder_id = str(_first(row, (
        "decoder_id", "decoder_family", "postprocess_contract_id", "host_tail_model",
    )) or summary.get("host_tail_model") or host_evidence.get("decoder_id") or "").strip()
    host_required = _as_bool(_first(row, ("host_tail_required", "requires_host_decode_nms", "requires_external_postprocess")))
    if host_required is None:
        host_required = _as_bool(summary.get("host_tail_required"))
    nms_ok = _as_bool(_first(row, ("nms_ok", "postprocess_included", "postprocess_ok")))
    if nms_ok is None and host_available is True:
        nms_ok = True
    if output_format == "bn6_detections" and not decoder_id:
        return False, "raw_head_misdeclared_as_bn6_detections"
    if host_required is True and host_available is not True:
        return False, "raw_head_host_tail_missing"
    if explicit_decoder_pass is not True or not decoder_id or nms_ok is not True:
        return False, "raw_head_decoder_postprocess_contract_unresolved"
    return True, "raw_head_frozen_decoder_and_nms_contract"


def _buildable(row: Mapping[str, Any]) -> bool:
    b = _as_bool(_first(row, ("buildable", "build_ok", "engine_build_ok", "compile_ok", "compiled_ok", "producer_ready")))
    if b is not None:
        return b
    status = str(row.get("status") or "").lower()
    if any(x in status for x in ("build_fail", "missing_engine", "compile_fail", "native_row_not_ok", "error")):
        return False
    if status in {"ok", "claim_ok", "eligible_for_ranking"}:
        return True
    ok = _as_bool(row.get("ok"))
    if ok is not None:
        return bool(ok)
    # Normalised result rows often omit compile_ok for already materialised ORT rows.
    return _runtime(row)


def _runtime(row: Mapping[str, Any]) -> bool:
    r = _as_bool(_first(row, ("runtime_executable", "runtime_ok", "run_ok", "result_ok", "consumer_ready")))
    if r is not None:
        return r
    status = str(row.get("status") or "").lower()
    if any(x in status for x in ("runtime_fail", "not_executable", "missing_dump", "native_row_not_ok", "error")):
        return False
    if _as_bool(row.get("tensor_ok")) is True:
        return True
    if _as_float(_first(row, ("FPS", "fps", "fps_makespan", "paper_equivalent_fps", "streaming_fps", "throughput_primary_fps", "total_latency_ms"))) is not None:
        return True
    ok = _as_bool(row.get("ok"))
    return bool(ok) if ok is not None else False


def _self_reference_active(row: Mapping[str, Any]) -> bool:
    src = str(row.get("semantic_reference_source") or row.get("accuracy_reference_source") or "").lower()
    diag = str(row.get("self_reference_diagnosis") or row.get("diagnosis") or "").lower()
    level = str(row.get("validation_level") or row.get("validation_claim_level") or "").lower()
    # Native-producer summaries do not always carry the verbose reference-source
    # strings.  The explicit availability/result fields are nevertheless an
    # unambiguous indication that this row is governed by a self-reference
    # contract.  Treat both True and False result values as active; False is a
    # failed contract, not an absent contract.
    explicit_available = _as_bool(row.get("self_reference_available"))
    explicit_result = _as_bool(row.get("self_reference_ok"))
    return bool(
        "full_onnx_self_reference" in src
        or "self_reference" in diag
        or "self_reference" in level
        or explicit_available is True
        or explicit_result is not None
    )


def _contract(row: Mapping[str, Any], policy: AccuracyGatePolicy, task: str) -> tuple[Any, str]:
    cardinality = row.get("validation_cardinality_contract") if isinstance(row.get("validation_cardinality_contract"), Mapping) else {}
    if cardinality and bool(cardinality.get("enforced")) and _as_bool(cardinality.get("pass")) is False:
        return False, "validation_cardinality_mismatch"
    if task == "detection":
        postprocess, postprocess_reason = _detection_postprocess_contract(row)
        if postprocess is False:
            return False, postprocess_reason
    if _pipeline_requires_explicit_interface(row):
        interface, interface_reason = _explicit_interface_result(row)
        if interface is not True:
            return (False if interface is False else "unavailable"), interface_reason
    explicit = _as_bool(_first(row, (
        "structural_contract_pass",
        "structure_contract_consistent",
        "interface_contract_pass",
        "contract_ok",
        "contract_consistent",
    )))
    if explicit is not None:
        return explicit, str(
            row.get("structural_contract_reason")
            or row.get("contract_gate_reason")
            or "explicit_structural_contract_field"
        )
    endpoint_complete = _as_bool(row.get("endpoint_contract_complete"))
    if endpoint_complete is not None:
        return endpoint_complete, (
            "endpoint_contract_complete"
            if endpoint_complete else
            "endpoint_contract_incomplete"
        )
    output_shape_match = _as_bool(_first(row, (
        "output_shape_match", "shape_contract_pass", "tensor_structure_pass",
    )))
    if output_shape_match is not None:
        return output_shape_match, "output_shape_structure_contract"
    if _self_reference_active(row):
        return (
            "unavailable",
            "structural_contract_missing_self_reference_is_numerical_only",
        )
    return "unavailable", "structural_contract_metric_missing"


def _numerical_similarity(
    row: Mapping[str, Any],
    policy: AccuracyGatePolicy,
    task: str,
) -> dict[str, Any]:
    explicit = _as_bool(row.get("numerical_similarity_pass"))
    if explicit is not None:
        status = str(row.get("numerical_similarity_status") or (
            "passed" if explicit else "failed"
        ))
        return {
            "numerical_similarity_pass": explicit,
            "numerical_similarity_status": status,
            "numerical_similarity_reason": str(
                row.get("numerical_similarity_reason")
                or "explicit_numerical_similarity_axis"
            ),
            "numerical_similarity_scope": str(
                row.get("numerical_similarity_scope") or ""
            ),
            "numerical_similarity_metric": str(
                row.get("numerical_similarity_metric") or ""
            ),
            "numerical_similarity_value": _as_float(
                row.get("numerical_similarity_value")
            ),
            "numerical_similarity_threshold": _as_float(
                row.get("numerical_similarity_threshold")
            ),
            "numerical_similarity_mean_iou": _as_float(
                row.get("numerical_similarity_mean_iou")
            ),
            "numerical_similarity_mean_iou_threshold": _as_float(
                row.get("numerical_similarity_mean_iou_threshold")
            ),
            "numerical_similarity_policy_id": str(
                row.get("numerical_similarity_policy_id")
                or policy.native_self_reference_policy_id
            ),
        }
    if _self_reference_active(row):
        if task == "detection":
            nested = (
                row.get("self_reference_match")
                if isinstance(row.get("self_reference_match"), Mapping)
                else row.get("match")
                if isinstance(row.get("match"), Mapping)
                else {}
            )
            if not nested:
                ratio = _as_float(_first(row, (
                    "self_reference_match_ratio",
                    "best_match_ratio",
                    "ap50_proxy",
                )))
                ref_count = int(_as_float(_first(row, (
                    "self_reference_reference_count",
                    "reference_detection_count",
                ))) or 0)
                matched = int(_as_float(_first(row, (
                    "self_reference_matched_count",
                    "matched_detection_count",
                ))) or 0)
                if not ref_count and ratio is not None:
                    # A ratio without its denominator is diagnostic only.
                    ref_count = 0
                nested = {
                    "ref_count": ref_count,
                    "matched": matched,
                    "match_ratio": ratio,
                    "mean_iou": _first(row, (
                        "self_reference_mean_iou",
                        "numerical_similarity_mean_iou",
                    )),
                    "iou_threshold": _first(row, (
                        "self_reference_iou_threshold",
                        "numerical_similarity_iou_threshold",
                    )),
                }
            return evaluate_detection_similarity(
                nested,
                row.get("class_agnostic_match")
                if isinstance(row.get("class_agnostic_match"), Mapping)
                else {},
                policy,
            )
        top1 = _as_bool(row.get("top1_match"))
        return {
            "numerical_similarity_pass": (
                top1 if top1 is not None else "unavailable"
            ),
            "numerical_similarity_status": (
                "passed" if top1 is True else
                "failed" if top1 is False else
                "unavailable"
            ),
            "numerical_similarity_reason": (
                "classification_top1_self_reference_match"
                if top1 is True else
                "classification_top1_self_reference_mismatch"
                if top1 is False else
                "classification_top1_self_reference_missing"
            ),
            "numerical_similarity_scope": "classification_top1_self_reference",
            "numerical_similarity_metric": "top1_match",
            "numerical_similarity_value": top1,
            "numerical_similarity_threshold": True,
            "numerical_similarity_policy_id": "classification_top1_exact_v1",
        }
    strict_numeric = _as_bool(row.get("strict_boundary_numeric_pass"))
    if strict_numeric is not None:
        return {
            "numerical_similarity_pass": strict_numeric,
            "numerical_similarity_status": (
                "passed" if strict_numeric else "failed"
            ),
            "numerical_similarity_reason": "strict_boundary_numeric_comparison",
            "numerical_similarity_scope": "split_boundary_tensor_values",
            "numerical_similarity_metric": str(
                row.get("strict_boundary_numeric_metric") or "configured_tolerance"
            ),
            "numerical_similarity_value": _as_float(
                row.get("strict_boundary_numeric_value")
            ),
            "numerical_similarity_threshold": _as_float(
                row.get("strict_boundary_numeric_threshold")
            ),
            "numerical_similarity_policy_id": str(
                row.get("strict_boundary_numeric_policy_id")
                or "strict_boundary_numeric_v1"
            ),
        }
    semantic = _as_bool(_first(row, (
        "semantic_ok", "semantic_validation_ok", "semantic_validation_passed",
    )))
    return {
        "numerical_similarity_pass": (
            semantic if semantic is not None else "unavailable"
        ),
        "numerical_similarity_status": (
            "passed" if semantic is True else
            "failed" if semantic is False else
            "unavailable"
        ),
        "numerical_similarity_reason": (
            "legacy_semantic_numeric_alias"
            if semantic is not None else
            "numerical_similarity_metric_missing"
        ),
        "numerical_similarity_scope": "legacy_semantic_alias",
        "numerical_similarity_metric": "",
        "numerical_similarity_value": None,
        "numerical_similarity_threshold": None,
        "numerical_similarity_policy_id": "legacy_semantic_alias_v1",
    }


def _nested_quality_gate(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    row = project_flat_quality_uncertainty(row)
    for key in ("task_quality_gate", "quality_gate", "non_inferiority_gate"):
        block = row.get(key)
        if isinstance(block, Mapping):
            primary = block.get("primary") if isinstance(block.get("primary"), Mapping) else block
            return project_quality_result({**dict(block), "primary": dict(primary)})
    if row.get("task_quality_gate_status") or row.get("task_quality_gate_decision"):
        return {
            "status": row.get("task_quality_gate_status"),
            "decision": row.get("task_quality_gate_decision"),
            "tier": row.get("task_quality_gate_tier"),
            "primary": {
                "metric": row.get("task_quality_primary_metric"),
                "candidate": row.get("task_quality_candidate"),
                "reference": row.get("task_quality_reference"),
                "delta": row.get("task_quality_delta"),
                "ci_low": row.get("task_quality_ci_low"),
                "ci_high": row.get("task_quality_ci_high"),
                "margin": row.get("task_quality_margin"),
                **{field: row.get(f"task_quality_{field}") for field in (*UNCERTAINTY_FIELDS, "bootstrap_repetitions", "bootstrap_skipped_reason") if f"task_quality_{field}" in row},
            },
        }
    return None


def _legacy_quality(row: Mapping[str, Any], task: str, policy: AccuracyGatePolicy) -> dict[str, Any]:
    if task == "classification":
        delta = _as_float(_first(row, ("mini_classification_eval_delta_primary_minus_full_top1", "quality_delta_vs_full", "top1_delta_vs_full", "delta_top1_vs_full")))
        candidate = _as_float(_first(row, ("mini_classification_eval_primary_top1", "mini_classification_eval_composed_top1", "validation_dataset_primary_top1_accuracy", "classification_top1", "classification_top1_accuracy", "top1_accuracy", "top1")))
        reference = _as_float(_first(row, ("mini_classification_eval_full_top1", "validation_dataset_full_top1_accuracy", "reference_top1", "reference_top1_accuracy", "full_top1", "full_top1_accuracy", "baseline_top1")))
        if delta is None and candidate is not None and reference is not None:
            delta = candidate - reference
        return {"metric": "top1_accuracy", "candidate": candidate, "reference": reference, "delta": delta, "margin": policy.classification_max_top1_drop}
    delta = _as_float(_first(row, ("coco_ap_50_95_delta_primary_minus_full", "mini_coco_ap50_delta_primary_minus_full", "quality_delta_vs_full", "ap50_delta_vs_full", "delta_ap50_vs_full")))
    candidate = _as_float(_first(row, ("coco_ap_50_95_primary", "mini_coco_ap50_primary", "mini_coco_ap50_composed", "validation_dataset_primary_ap50", "ap50", "map50")))
    reference = _as_float(_first(row, ("coco_ap_50_95_full", "mini_coco_ap50_full", "validation_dataset_full_ap50", "reference_ap50", "full_ap50", "baseline_ap50")))
    if delta is None and candidate is not None and reference is not None:
        delta = candidate - reference
    metric = "coco_ap_50_95" if row.get("coco_ap_50_95_primary") is not None else "ap50"
    margin = policy.detection_max_ap_drop if metric == "coco_ap_50_95" else policy.detection_max_ap50_drop
    return {"metric": metric, "candidate": candidate, "reference": reference, "delta": delta, "margin": margin}


def _is_reference_row(row: Mapping[str, Any]) -> bool:
    variant = str(row.get("variant") or row.get("kind") or "").strip().lower()
    backend = str(row.get("backend") or row.get("full_backend") or "").strip().lower()
    ref = _as_bool(_first(row, ("canonical_reference_row", "task_quality_reference_row", "is_canonical_full_reference")))
    return ref is True or _is_cpu_semantic_reference(row) or (variant == "full" and backend in {"cpu_ort", "ort_cpu", "canonical_full_onnx"})


def _quality_decision(row: Mapping[str, Any], task: str, policy: AccuracyGatePolicy) -> dict[str, Any]:
    direct = _nested_quality_gate(row)
    if direct is not None:
        primary = direct.get("primary") if isinstance(direct.get("primary"), Mapping) else {}
        decision = str(direct.get("decision") or direct.get("status") or primary.get("decision") or "").strip().lower()
        if decision in {"passed", "ok", "eligible"}:
            decision = "pass"
        elif decision in {"failed", "invalid"}:
            decision = "fail"
        tier = str(direct.get("tier") or direct.get("dataset_tier") or policy.dataset_tier or "screening").strip().lower()
        metric = str(primary.get("metric") or direct.get("primary_metric") or (policy.classification_primary_metric if task == "classification" else policy.detection_primary_metric))
        candidate = _as_float(primary.get("candidate", direct.get("candidate")))
        reference = _as_float(primary.get("reference", direct.get("reference")))
        delta = _as_float(primary.get("delta", direct.get("delta")))
        ci_low = _as_float(primary.get("ci_low", direct.get("ci_low")))
        ci_high = _as_float(primary.get("ci_high", direct.get("ci_high")))
        margin_default = policy.classification_max_top1_drop if task == "classification" else policy.detection_max_ap_drop
        margin = _as_float(primary.get("margin", direct.get("margin")))
        if margin is None:
            margin = margin_default
        if not decision:
            if ci_low is not None:
                decision = "pass" if ci_low >= -float(margin) else ("fail" if delta is not None and delta < -float(margin) else "inconclusive")
            elif delta is not None:
                decision = "legacy_point_estimate_only"
            else:
                decision = "unavailable"
        return {
            "source": "runtime_task_quality_gate",
            "status": str(direct.get("status") or decision),
            "decision": decision,
            "tier": tier,
            "metric": metric,
            "candidate": candidate,
            "reference": reference,
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            **{field: primary.get(field) for field in UNCERTAINTY_FIELDS},
            "margin": margin,
            "n": direct.get("n") or primary.get("n"),
            "guardrails": direct.get("guardrails") if isinstance(direct.get("guardrails"), Mapping) else {},
            "embedded_policy": direct.get("policy") if isinstance(direct.get("policy"), Mapping) else {},
            "bootstrap_repetitions_requested": primary.get("bootstrap_repetitions_requested", direct.get("bootstrap_repetitions_requested")),
            "bootstrap_repetitions": primary.get("bootstrap_repetitions", direct.get("bootstrap_repetitions")),
            "bootstrap_engine": primary.get("bootstrap_engine", direct.get("bootstrap_engine")),
            "bootstrap_skipped_reason": primary.get("bootstrap_skipped_reason", direct.get("bootstrap_skipped_reason")),
            "bootstrap_elapsed_s": primary.get("bootstrap_elapsed_s", direct.get("bootstrap_elapsed_s")),
            "bootstrap_candidate_event_count": primary.get("bootstrap_candidate_event_count", direct.get("bootstrap_candidate_event_count")),
            "bootstrap_reference_event_count": primary.get("bootstrap_reference_event_count", direct.get("bootstrap_reference_event_count")),
            "execution_location": primary.get("execution_location", direct.get("execution_location")),
            "quality_input_request": direct.get("quality_input_request") if isinstance(direct.get("quality_input_request"), Mapping) else {},
            "raw": direct,
        }

    legacy = _legacy_quality(row, task, policy)
    if _is_reference_row(row):
        return {"source": "canonical_reference_row", "status": "reference", "decision": "reference", "tier": policy.dataset_tier, **legacy}
    if legacy.get("delta") is not None:
        decision = "pass" if policy.legacy_point_estimate_eligible_for_ranking and float(legacy["delta"]) >= -float(legacy["margin"]) else "legacy_point_estimate_only"
        return {"source": "legacy_point_estimate", "status": decision, "decision": decision, "tier": policy.dataset_tier, **legacy}
    return {"source": "missing", "status": "unavailable", "decision": "unavailable", "tier": policy.dataset_tier, **legacy}


def _failed_quality_component_reasons(quality: Mapping[str, Any]) -> list[str]:
    """Explain which runtime quality-gate component actually caused a fail.

    The primary point estimate can be unchanged while a guardrail (for example
    classification Top-5 accuracy) fails.  Reporting only the primary delta in
    that situation makes a correct, conservative failure look contradictory.
    Preserve the runtime decision and expose the component-level trigger instead
    of weakening the gate.
    """
    if str(quality.get("decision") or "").strip().lower() != "fail":
        return []
    raw = quality.get("raw") if isinstance(quality.get("raw"), Mapping) else {}

    def component_reason(prefix: str, component: Mapping[str, Any]) -> str:
        metric = str(component.get("metric") or prefix).strip() or prefix
        explicit = str(component.get("failure_reason") or component.get("reason") or "").strip()
        skipped = str(component.get("bootstrap_skipped_reason") or "").strip()
        delta = _as_float(component.get("delta"))
        ci_low = _as_float(component.get("ci_low"))
        margin = _as_float(component.get("margin"))
        if explicit:
            cause = explicit
        elif skipped == "point_estimate_below_non_inferiority_margin":
            cause = skipped
        elif delta is not None and margin is not None and delta < -margin:
            cause = "point_estimate_below_non_inferiority_margin"
        elif ci_low is not None and margin is not None and ci_low < -margin:
            cause = "confidence_bound_below_non_inferiority_margin"
        elif skipped:
            cause = skipped
        else:
            cause = "decision_fail"
        return f"{prefix}:{metric}:{cause}"

    reasons: list[str] = []
    primary = raw.get("primary") if isinstance(raw.get("primary"), Mapping) else raw
    if isinstance(primary, Mapping) and str(primary.get("decision") or primary.get("status") or "").strip().lower() in {"fail", "failed"}:
        reasons.append(component_reason("primary", primary))
    guardrails = raw.get("guardrails") if isinstance(raw.get("guardrails"), Mapping) else {}
    for name, component in sorted(guardrails.items(), key=lambda item: str(item[0])):
        if not isinstance(component, Mapping):
            continue
        if str(component.get("decision") or component.get("status") or "").strip().lower() in {"fail", "failed"}:
            reasons.append(component_reason(f"guardrail:{name}", component))
    if not reasons:
        explicit = str(raw.get("failure_reason") or raw.get("reason") or "").strip()
        reasons.append(explicit or "runtime_gate_declared_fail_without_component_reason")
    return reasons


def apply_accuracy_gate_to_row(row: MutableMapping[str, Any], policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None) -> MutableMapping[str, Any]:
    apply_host_postprocess_aliases(row)
    pol, policy_source = resolve_effective_policy(row, policy)
    task = _infer_task(row)
    row.setdefault("task", task)
    buildable = _buildable(row)
    runtime = _runtime(row)
    contract, contract_reason = _contract(row, pol, task)
    numerical = _numerical_similarity(row, pol, task)
    numerical_pass = numerical.get("numerical_similarity_pass")
    numerical_active = bool(
        _self_reference_active(row)
        or _as_bool(row.get("numerical_similarity_pass")) is not None
        or _as_bool(row.get("strict_boundary_numeric_pass")) is not None
    )
    numerical_required = bool(
        pol.numerical_similarity_required_for_claim and numerical_active
    )
    numerical_allows_claim = bool(
        not numerical_required or numerical_pass is True
    )
    quality = _quality_decision(row, task, pol)
    decision = str(quality.get("decision") or "unavailable").lower()
    quality_trigger_reasons = _failed_quality_component_reasons(quality)
    tier = str(quality.get("tier") or pol.dataset_tier).lower()
    selfref = _self_reference_active(row)

    configured_policy_sha256 = pol.sha256()
    embedded_policy = quality.get("embedded_policy") if isinstance(quality.get("embedded_policy"), Mapping) else {}
    embedded_policy_sha256 = ""
    policy_match = True
    if embedded_policy:
        try:
            embedded_policy_sha256 = AccuracyGatePolicy.from_mapping(embedded_policy).sha256()
            policy_match = embedded_policy_sha256 == configured_policy_sha256
        except Exception:
            policy_match = False

    if decision == "reference":
        task_valid: Any = True
        accuracy_pass: Any = True
    elif decision == "pass":
        task_valid = True
        accuracy_pass = True
    elif decision == "fail":
        task_valid = False
        accuracy_pass = False
    else:
        task_valid = "inconclusive" if decision in {"inconclusive", "legacy_point_estimate_only", "screening_only", "pending_central_evaluation"} else "unavailable"
        accuracy_pass = False

    if (
        pol.contract_only_eligible_for_ranking
        and selfref
        and contract is True
        and numerical_allows_claim
        and decision in {"unavailable", "legacy_point_estimate_only"}
    ):
        task_valid = True
        accuracy_pass = True
        decision = "contract_only_policy_override"

    tier_allows_ranking = tier == "final" or pol.screening_eligible_for_ranking
    quality_allows_ranking = decision in {"pass", "contract_only_policy_override"}
    cpu_semantic_reference = _is_cpu_semantic_reference(row)
    candidate_eligible = bool(
        buildable and runtime and contract is True and task_valid is True
        and accuracy_pass is True and tier_allows_ranking and quality_allows_ranking
        and numerical_allows_claim and policy_match and not cpu_semantic_reference
    )
    reference_row = _is_reference_row(row)
    performance_eligible = bool(buildable and runtime and contract is True and candidate_eligible and not cpu_semantic_reference)
    energy_trace_pass = _as_bool(_first(row, ("energy_trace_pass", "energy_target_ok")))
    energy_available = _as_float(_first(row, ("row_energy_streaming_j_per_frame", "row_energy_latency_j_per_inference", "energy_j_per_frame", "energy_j"))) is not None
    energy_eligible = bool(performance_eligible and (energy_trace_pass is not False) and energy_available)
    pareto_eligible = bool(candidate_eligible)

    if cpu_semantic_reference:
        verdict = "semantic_reference_only"
    elif not buildable:
        verdict = "not_buildable"
    elif not runtime:
        verdict = "runtime_not_executable"
    elif contract is not True:
        verdict = "contract_fail_or_unavailable"
    elif numerical_required and numerical_pass is False:
        verdict = "numerical_similarity_failed"
    elif numerical_required and numerical_pass is not True:
        verdict = "numerical_similarity_unavailable"
    elif not policy_match:
        verdict = "quality_policy_mismatch"
    elif reference_row:
        verdict = "canonical_reference"
    elif decision == "fail":
        verdict = "quality_failed"
    elif decision == "inconclusive":
        verdict = "quality_inconclusive"
    elif decision == "pending_central_evaluation":
        verdict = "pending_central_evaluation"
    elif decision == "legacy_point_estimate_only":
        verdict = "legacy_point_estimate_only"
    elif decision == "unavailable":
        verdict = "quality_unavailable"
    elif tier != "final" and not pol.screening_eligible_for_ranking:
        verdict = "screening_only"
    elif candidate_eligible:
        verdict = "eligible"
    else:
        verdict = "not_eligible"

    execution_ok = bool(buildable and runtime)
    interface_status = "pass" if contract is True else ("fail" if contract is False else "unavailable")
    quality_valid = bool(task_valid is True and accuracy_pass is True)
    quality_evaluated = decision in {"pass", "fail", "reference"}
    task_quality_reason = (
        ";".join(quality_trigger_reasons)
        if decision == "fail"
        else (
            "task_quality_passed"
            if decision == "pass"
            else (
                "canonical_reference"
                if decision == "reference"
                else f"task_quality_{decision}"
            )
        )
    )
    evidence_complete = bool(
        execution_ok
        and contract in {True, False}
        and quality_evaluated
        and policy_match
        and (not numerical_required or numerical_pass in {True, False})
    )

    if selfref and numerical_pass in {True, False}:
        # ``semantic_ok`` remains a compatibility alias, but the decision is
        # recomputed from the versioned numerical policy rather than trusted
        # from an older hard-coded producer threshold.
        row["semantic_ok"] = bool(numerical_pass)
        row["self_reference_ok"] = bool(numerical_pass)

    row.update({
        "buildable": bool(buildable),
        "runtime_executable": bool(runtime),
        "execution_ok": execution_ok,
        "contract_consistent": contract,
        "contract_gate_reason": contract_reason,
        "structural_contract_pass": contract,
        "structural_contract_status": interface_status,
        "structural_contract_reason": contract_reason,
        "interface_valid": bool(contract is True),
        "interface_status": interface_status,
        "task_valid": task_valid,
        "task_quality_pass": task_valid,
        "task_quality_status": decision,
        "task_quality_reason": task_quality_reason,
        "quality_valid": quality_valid,
        "evidence_complete": evidence_complete,
        "accuracy_gate_pass": bool(accuracy_pass),
        "accuracy_gate_reason": verdict,
        "accuracy_gate_trigger_reason": ";".join(quality_trigger_reasons),
        "accuracy_gate_trigger_reasons": quality_trigger_reasons,
        "accuracy_gate_metric": quality.get("metric"),
        "accuracy_gate_delta": quality.get("delta"),
        "accuracy_gate_ci_low": quality.get("ci_low"),
        "accuracy_gate_ci_high": quality.get("ci_high"),
        "accuracy_gate_threshold": quality.get("margin"),
        "accuracy_gate_tier": tier,
        "accuracy_gate_source": quality.get("source"),
        "accuracy_gate_decision": decision,
        **{f"accuracy_gate_{field}": quality.get(field) for field in UNCERTAINTY_FIELDS},
        "accuracy_gate_metrics": {
            **{field: quality.get(field) for field in UNCERTAINTY_FIELDS},
            "candidate": quality.get("candidate"),
            "reference": quality.get("reference"),
            "delta": quality.get("delta"),
            "ci_low": quality.get("ci_low"),
            "ci_high": quality.get("ci_high"),
            "margin": quality.get("margin"),
            "n": quality.get("n"),
            "guardrails": quality.get("guardrails"),
            "bootstrap_repetitions_requested": quality.get("bootstrap_repetitions_requested"),
            "bootstrap_repetitions": quality.get("bootstrap_repetitions"),
            "bootstrap_engine": quality.get("bootstrap_engine"),
            "bootstrap_skipped_reason": quality.get("bootstrap_skipped_reason"),
            "bootstrap_elapsed_s": quality.get("bootstrap_elapsed_s"),
            "bootstrap_candidate_event_count": quality.get("bootstrap_candidate_event_count"),
            "bootstrap_reference_event_count": quality.get("bootstrap_reference_event_count"),
            "execution_location": quality.get("execution_location"),
        },
        "accuracy_gate_bootstrap_repetitions_requested": quality.get("bootstrap_repetitions_requested"),
        "accuracy_gate_bootstrap_repetitions": quality.get("bootstrap_repetitions"),
        "accuracy_gate_bootstrap_engine": quality.get("bootstrap_engine"),
        "accuracy_gate_bootstrap_skipped_reason": quality.get("bootstrap_skipped_reason"),
        "accuracy_gate_bootstrap_elapsed_s": quality.get("bootstrap_elapsed_s"),
        "quality_execution_location": quality.get("execution_location") or ("central_management" if decision == "pending_central_evaluation" else "local"),
        "quality_evaluation_pending": bool(decision == "pending_central_evaluation"),
        "quality_input_request": quality.get("quality_input_request") if isinstance(quality.get("quality_input_request"), Mapping) else {},
        "accuracy_gate_policy": pol.as_dict(),
        "accuracy_gate_policy_source": policy_source,
        "accuracy_gate_policy_sha256": configured_policy_sha256,
        "runtime_quality_gate_policy_sha256": embedded_policy_sha256,
        "accuracy_gate_policy_match": bool(policy_match),
        "eligible_for_ranking": bool(candidate_eligible),
        "ranking_eligible": bool(candidate_eligible),
        "performance_eligible": bool(performance_eligible),
        "energy_eligible": bool(energy_eligible),
        "pareto_eligible": bool(pareto_eligible),
        "ranking_exclusion_reason": "" if candidate_eligible else verdict,
        "exclusion_reason": "" if candidate_eligible else verdict,
        "gate_status": verdict,
        "validation_verdict": verdict,
        "thesis_valid": bool(candidate_eligible),
        "semantic_reference_only": bool(cpu_semantic_reference),
        **numerical,
        "numerical_similarity_required_for_claim": numerical_required,
        "evidence_axes": {
            "structural_contract": {
                "pass": contract,
                "status": interface_status,
                "reason": contract_reason,
            },
            "numerical_similarity": {
                "pass": numerical_pass,
                "status": numerical.get("numerical_similarity_status"),
                "reason": numerical.get("numerical_similarity_reason"),
                "policy_id": numerical.get("numerical_similarity_policy_id"),
                "value": numerical.get("numerical_similarity_value"),
                "threshold": numerical.get("numerical_similarity_threshold"),
                "required_for_claim": numerical_required,
            },
            "task_quality": {
                "pass": task_valid,
                "status": decision,
                "reason": task_quality_reason,
            },
        },
    })
    current_claim = _as_bool(row.get("claim_ok"))
    source_claim = _as_bool(row.get("claim_ok_source"))
    if current_claim is True:
        # A producer may compute its provisional claim between two gate
        # passes. Never let an earlier default False source marker hide that
        # later True input from the structural clamp.
        source_claim = True
    elif source_claim is None:
        source_claim = current_claim
    previously_clamped = (
        _as_bool(row.get("claim_ok_structural_clamped")) is True
    )
    structurally_clamped = bool(
        previously_clamped
        or (source_claim is True and contract is not True)
    )
    if source_claim is not None:
        row["claim_ok_source"] = bool(source_claim)
        row["claim_structural_gate_pass"] = contract is True
        row["claim_structural_gate_reason"] = contract_reason
        row["claim_ok_structural_clamped"] = structurally_clamped
    if structurally_clamped:
        # A numerical or task-quality pass must never resurrect a claim whose
        # structural interface contract failed or remained unavailable.  The
        # independent evidence axes above intentionally remain unchanged.
        row["claim_ok"] = False
        if str(row.get("status") or "").strip().lower() == "claim_ok":
            row["status"] = (
                "structural_contract_failed"
                if contract is False
                else "structural_contract_unavailable"
            )
    return row


def apply_accuracy_gates(row: MutableMapping[str, Any], policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None, mutate: bool = True) -> MutableMapping[str, Any]:
    target: MutableMapping[str, Any] = row if mutate else dict(row)
    return apply_accuracy_gate_to_row(target, policy)


def apply_gate_fields(row: MutableMapping[str, Any], task: str = "", source: str = "", policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None) -> MutableMapping[str, Any]:
    if task and not row.get("task"):
        row["task"] = task
    if source and not row.get("accuracy_gate_source"):
        row["accuracy_gate_source"] = source
    return apply_accuracy_gate_to_row(row, policy)


def apply_accuracy_gates_to_payload(payload: MutableMapping[str, Any], policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None) -> MutableMapping[str, Any]:
    rows = payload.get("rows") or payload.get("results")
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, MutableMapping):
                apply_accuracy_gate_to_row(r, policy)
        payload.update(gate_counts(rows))
    if policy not in (None, ""):
        pol = policy if isinstance(policy, AccuracyGatePolicy) else AccuracyGatePolicy.from_mapping(policy)
        payload["accuracy_gate_policy"] = pol.as_dict()
    else:
        payload["accuracy_gate_policy"] = "runtime_embedded_per_row"
    return payload


def apply_accuracy_gates_to_rows(rows: list[Mapping[str, Any]], policy: AccuracyGatePolicy | Mapping[str, Any] | str | Path | None = None) -> list[dict[str, Any]]:
    return [dict(apply_accuracy_gate_to_row(dict(r), policy)) for r in rows]


def gate_counts(rows: Any) -> dict[str, int]:
    seq = list(rows or [])
    return {
        "row_count": len(seq),
        "ranking_eligible_count": sum(1 for r in seq if isinstance(r, Mapping) and _as_bool(r.get("ranking_eligible")) is True),
        "performance_eligible_count": sum(1 for r in seq if isinstance(r, Mapping) and _as_bool(r.get("performance_eligible")) is True),
        "energy_eligible_count": sum(1 for r in seq if isinstance(r, Mapping) and _as_bool(r.get("energy_eligible")) is True),
        # v60i: count the task-quality decision independently from later
        # contract/tier eligibility.  A row can have a valid bootstrap `fail`
        # while its overall eligibility status is `contract_fail_or_unavailable`.
        "quality_failed_count": sum(
            1 for r in seq if isinstance(r, Mapping)
            and str(r.get("accuracy_gate_decision") or "").strip().lower() == "fail"
        ),
        "quality_inconclusive_count": sum(
            1 for r in seq if isinstance(r, Mapping)
            and str(r.get("accuracy_gate_decision") or "").strip().lower()
            in {"inconclusive", "legacy_point_estimate_only", "screening_only"}
        ),
    }


def load_policy(value: str | Path | Mapping[str, Any] | None = None) -> AccuracyGatePolicy:
    return AccuracyGatePolicy.from_mapping(value)


__all__ = [
    "AccuracyGatePolicy", "DEFAULT_POLICY", "apply_accuracy_gate_to_row",
    "apply_accuracy_gates", "apply_accuracy_gates_to_payload", "apply_accuracy_gates_to_rows",
    "apply_gate_fields", "gate_counts", "load_policy", "resolve_effective_policy",
]
