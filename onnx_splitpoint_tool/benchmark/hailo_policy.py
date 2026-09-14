from __future__ import annotations

"""Hailo-specific boundary and case policy helpers.

These helpers keep benchmark-set generation logic testable and make a few
Hailo-specific rules explicit:

* structural cut-tensor heuristics for late YOLO/one2one head boundaries
* adaptive skipping of boundary neighborhoods after repeated allocator/layout
  failures (``format_conversion* -> Agent infeasible`` style)
* case-level availability of Hailo variants (full/part1/part2/composed)
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_LATE_HEAD_PREFIXES: Tuple[str, ...] = ("/model.22/", "/model.23/")
_RAW_STAGE_TOKENS: Tuple[str, ...] = (
    "/conv_output_",
    "/add_output_",
    "/mul_output_",
    "/sigmoid_output_",
)
_LAYOUT_FAIL_TOKENS: Tuple[str, ...] = (
    "format_conversion",
    "concat",
    "defuse",
    "feature_splitter",
    "successor name is missing",
    "output shape is ambiguous",
    "agent infeasible",
    "no successful assignments",
    "mapping failed",
)


@dataclass(frozen=True)
class HailoCutTensorProfile:
    cut_tensors: Tuple[str, ...]
    late_head_count: int
    model22_count: int
    model23_count: int
    one2one_count: int
    raw_activation_count: int
    reshape_count: int
    slice_count: int
    attn_count: int
    qkv_count: int
    risky_tensor_count: int
    penalty: float
    high_risk: bool
    clusterable: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class HailoFailureRecord:
    boundary: int
    stage: str
    hw_arch: str
    failure_kind: str
    clusterable: bool
    detail: str
    family: str = ""


@dataclass(frozen=True)
class HailoClusterSkipDecision:
    skip: bool
    nearby_failures: int
    detail: str = ""


def _norm_name(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _bool_from_hef_entry(entry: Mapping[str, Any], key: str) -> bool:
    value = entry.get(key)
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return bool(value)


def _is_late_head_tensor(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _LATE_HEAD_PREFIXES)


def _is_raw_activation_tensor(name: str) -> bool:
    if any(tok in name for tok in _RAW_STAGE_TOKENS):
        return True
    return bool(re.search(r"/(conv|add|mul|sigmoid)_output_\d+$", name))


def analyze_cut_tensors(cut_tensors: Sequence[Any]) -> HailoCutTensorProfile:
    names = tuple(str(x) for x in (cut_tensors or []) if str(x).strip())
    norm = tuple(_norm_name(x) for x in names)

    late_head_count = sum(1 for name in norm if _is_late_head_tensor(name))
    model22_count = sum(1 for name in norm if name.startswith("/model.22/"))
    model23_count = sum(1 for name in norm if name.startswith("/model.23/"))
    one2one_count = sum(1 for name in norm if "one2one_" in name)
    raw_activation_count = sum(1 for name in norm if _is_raw_activation_tensor(name))
    reshape_count = sum(1 for name in norm if "/reshape" in name)
    slice_count = sum(1 for name in norm if "/slice" in name)
    attn_count = sum(1 for name in norm if "/attn/" in name)
    qkv_count = sum(1 for name in norm if "/qkv/" in name)
    risky_tensor_count = sum(
        1
        for name in norm
        if _is_late_head_tensor(name)
        and (
            _is_raw_activation_tensor(name)
            or "/reshape" in name
            or "/slice" in name
            or "one2one_" in name
            or "/attn/" in name
            or "/qkv/" in name
        )
    )

    reasons: List[str] = []
    penalty = 0.0

    if late_head_count:
        penalty += min(0.60, 0.14 * float(late_head_count))
    if len(norm) >= 6:
        penalty += 0.16
        reasons.append("many cut tensors cross the split")
    if len(norm) >= 7:
        penalty += 0.25
    if len(norm) >= 8:
        penalty += 0.35
    if raw_activation_count:
        penalty += min(1.10, 0.32 * float(raw_activation_count))
        reasons.append("raw late-head activation tensors cross the split")
    if one2one_count:
        penalty += min(0.95, 0.38 * float(one2one_count))
        reasons.append("one2one head tensors cross the split")
    if attn_count:
        penalty += min(0.70, 0.22 * float(attn_count))
        reasons.append("attention/qkv tensors cross the split")
    if qkv_count:
        penalty += min(0.50, 0.25 * float(qkv_count))
    if model22_count and model23_count:
        penalty += 0.42
        reasons.append("split mixes /model.22 and /model.23 head branches")
    if raw_activation_count and (reshape_count + slice_count) >= 3:
        penalty += 0.40
        reasons.append("raw activations are mixed with reshape/slice boundary tensors")
    if reshape_count >= 2 and slice_count >= 1:
        penalty += 0.18
    if one2one_count and raw_activation_count and len(norm) >= 7:
        penalty += 0.68
        reasons.append("large late-head cut contains one2one raw activations")
    if attn_count and raw_activation_count and len(norm) >= 6:
        penalty += 0.36
    if len(norm) >= 6 and risky_tensor_count >= 3:
        penalty += 0.30
    if raw_activation_count >= 2 and reshape_count >= 1 and slice_count >= 1:
        penalty += 0.24

    high_risk = bool(
        penalty >= 1.20
        or (one2one_count >= 1 and raw_activation_count >= 2)
        or (raw_activation_count >= 1 and model22_count >= 1 and model23_count >= 1 and len(norm) >= 7)
        or (attn_count >= 1 and raw_activation_count >= 1 and len(norm) >= 6)
    )
    clusterable = bool(
        high_risk
        or (raw_activation_count >= 1 and risky_tensor_count >= 2)
        or (attn_count >= 1 and len(norm) >= 6)
    )

    seen: set[str] = set()
    reasons_out: List[str] = []
    for reason in reasons:
        key = reason.strip().lower()
        if key and key not in seen:
            seen.add(key)
            reasons_out.append(reason)

    return HailoCutTensorProfile(
        cut_tensors=names,
        late_head_count=int(late_head_count),
        model22_count=int(model22_count),
        model23_count=int(model23_count),
        one2one_count=int(one2one_count),
        raw_activation_count=int(raw_activation_count),
        reshape_count=int(reshape_count),
        slice_count=int(slice_count),
        attn_count=int(attn_count),
        qkv_count=int(qkv_count),
        risky_tensor_count=int(risky_tensor_count),
        penalty=float(penalty),
        high_risk=bool(high_risk),
        clusterable=bool(clusterable),
        reasons=tuple(reasons_out),
    )


def boundary_policy_from_candidate_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    boundary = int(row.get("boundary", -1))
    profile = analyze_cut_tensors(row.get("cut_tensors") or [])
    strategy = str(row.get("hailo_parse_strategy") or "").strip()
    used_alt = bool(row.get("hailo_parse_used_suggested_end_nodes")) or strategy == "hailo_parser_suggested_end_nodes"
    parse_checked = bool(row.get("hailo_parse_checked"))
    parse_ok_raw = row.get("hailo_parse_ok")
    parse_ok = bool(parse_ok_raw) if parse_ok_raw is not None else None
    alt_penalty = 0.82 if used_alt else 0.0
    if parse_checked and parse_ok is False:
        alt_penalty += 0.70
    return {
        "boundary": boundary,
        "hailo_structural_penalty": float(profile.penalty),
        "hailo_risky_cut_tensor_count": int(profile.risky_tensor_count),
        "hailo_late_head_cut_count": int(profile.late_head_count),
        "hailo_one2one_cut_count": int(profile.one2one_count),
        "hailo_raw_activation_cut_count": int(profile.raw_activation_count),
        "hailo_attn_cut_count": int(profile.attn_count),
        "hailo_qkv_cut_count": int(profile.qkv_count),
        "hailo_crosses_model22_and_model23": bool(profile.model22_count and profile.model23_count),
        "hailo_high_risk_boundary": bool(profile.high_risk),
        "hailo_clusterable_boundary": bool(profile.clusterable),
        "hailo_structural_reasons": list(profile.reasons),
        "hailo_parse_checked": parse_checked,
        "hailo_parse_ok": parse_ok,
        "hailo_parse_strategy": (strategy or None),
        "hailo_parse_used_suggested_end_nodes": bool(used_alt),
        "hailo_alt_strategy_penalty": float(alt_penalty),
    }


def build_candidate_policy_index(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        try:
            boundary = int(row.get("boundary", -1))
        except Exception:
            continue
        out[boundary] = boundary_policy_from_candidate_row(row)
    return out


def _normalize_failure_family(*, detail: str, stage: str, failure_kind: str) -> str:
    low = str(detail or "").lower()
    stage_low = str(stage or "").strip().lower()
    kind_low = str(failure_kind or "").strip().lower()

    if "concat sanity failed" in low or "concat_shape_mismatch" in low:
        return "concat_shape_mismatch"
    if "validator failed on node:" in low:
        if "concat" in low:
            return "validator_concat"
        if "dw1_defuse" in low or "dw2_defuse" in low or "defuse" in low:
            return "validator_defuse"
    if "dimensions mismatch" in low and "concat" in low:
        return "concat_shape_mismatch"
    if "input shapes" in low and "concat" in low:
        return "concat_shape_mismatch"
    if "format_conversion" in low and "agent infeasible" in low:
        return "format_conversion_agent_infeasible"
    if "feature_splitter" in low and ("agent infeasible" in low or "successor name is missing" in low or "output shape is ambiguous" in low):
        return "feature_splitter_agent_infeasible"
    if "agent infeasible" in low and "concat" in low and stage_low == "part2":
        return "concat_shape_mismatch"
    if kind_low == "allocator_agent_infeasible":
        return "allocator_agent_infeasible"
    if kind_low == "allocator_mapping_failed":
        return "mapping_failed"
    return kind_low or "other"


def classify_hailo_build_failure(res: Any, *, boundary: int, stage: str, hw_arch: str) -> HailoFailureRecord:
    text_parts: List[str] = []
    try:
        err = getattr(res, "error", None)
        if err:
            text_parts.append(str(err))
    except Exception:
        pass
    try:
        details = getattr(res, "details", None) or {}
        if isinstance(details, Mapping):
            proc = details.get("process_summary") if isinstance(details.get("process_summary"), Mapping) else {}
            if isinstance(proc, Mapping):
                sc = proc.get("single_context_failure")
                if sc:
                    text_parts.append(str(sc))
                validator_nodes = list(proc.get("validator_failed_nodes") or [])
                if validator_nodes:
                    text_parts.append("validator_failed_nodes=" + ",".join(str(x) for x in validator_nodes if str(x).strip()))
                detected = proc.get("detected") if isinstance(proc.get("detected"), Mapping) else {}
                if isinstance(detected, Mapping) and detected:
                    text_parts.append(" ".join(f"{k}={v}" for k, v in detected.items()))
    except Exception:
        pass

    detail = "\n".join(part for part in text_parts if str(part).strip())
    low = detail.lower()
    clusterable = any(tok in low for tok in _LAYOUT_FAIL_TOKENS)
    if clusterable and "agent infeasible" in low:
        kind = "allocator_agent_infeasible"
    elif clusterable and "mapping failed" in low:
        kind = "allocator_mapping_failed"
    else:
        kind = "other"
    family = _normalize_failure_family(detail=detail, stage=stage, failure_kind=kind)

    return HailoFailureRecord(
        boundary=int(boundary),
        stage=str(stage or ""),
        hw_arch=str(hw_arch or ""),
        failure_kind=str(kind),
        clusterable=bool(clusterable),
        detail=str(detail or ""),
        family=str(family),
    )


def should_skip_from_failure_cluster(
    boundary: int,
    failures: Sequence[HailoFailureRecord],
    *,
    candidate_policy: Optional[Mapping[str, Any]] = None,
    stage: str = "part1",
    hw_archs: Optional[Iterable[str]] = None,
    radius: int = 12,
    min_failures: int = 2,
) -> HailoClusterSkipDecision:
    b = int(boundary)
    wanted_stage = str(stage or "").strip().lower()
    wanted_archs = {str(x).strip().lower() for x in (hw_archs or []) if str(x).strip()}
    family_counts: Dict[str, int] = {}
    for rec in failures or []:
        if not isinstance(rec, HailoFailureRecord):
            continue
        if not rec.clusterable:
            continue
        if wanted_stage and str(rec.stage or "").strip().lower() != wanted_stage:
            continue
        if wanted_archs and str(rec.hw_arch or "").strip().lower() not in wanted_archs:
            continue
        if abs(int(rec.boundary) - b) > int(radius):
            continue
        family = str(rec.family or rec.failure_kind or "other").strip().lower() or "other"
        family_counts[family] = family_counts.get(family, 0) + 1

    if not family_counts:
        return HailoClusterSkipDecision(skip=False, nearby_failures=0)

    dominant_family, dominant_count = max(family_counts.items(), key=lambda item: (int(item[1]), str(item[0])))
    if int(dominant_count) < int(min_failures):
        return HailoClusterSkipDecision(skip=False, nearby_failures=int(dominant_count))

    policy = dict(candidate_policy or {})
    clusterable_boundary = bool(policy.get("hailo_clusterable_boundary") or policy.get("hailo_high_risk_boundary"))
    alt_strategy = bool(policy.get("hailo_parse_used_suggested_end_nodes"))
    crosses_head = bool(policy.get("hailo_crosses_model22_and_model23"))
    late_head_count = int(policy.get("hailo_late_head_cut_count") or 0)

    should_skip = False
    if dominant_family == "concat_shape_mismatch":
        should_skip = bool(
            clusterable_boundary
            or alt_strategy
            or crosses_head
            or late_head_count >= 4
            or int(dominant_count) >= int(min_failures) + 1
        )
    elif dominant_family in {
        "format_conversion_agent_infeasible",
        "validator_concat",
        "validator_defuse",
        "feature_splitter_agent_infeasible",
        "allocator_agent_infeasible",
    }:
        threshold = int(min_failures) if clusterable_boundary else int(min_failures) + 1
        should_skip = int(dominant_count) >= threshold
    else:
        threshold = int(min_failures) + (0 if clusterable_boundary else 1)
        should_skip = int(dominant_count) >= threshold

    if not should_skip:
        return HailoClusterSkipDecision(skip=False, nearby_failures=int(dominant_count))

    detail = (
        f"nearby Hailo {dominant_family.replace('_', '/')} failures in the same boundary neighborhood "
        f"({int(dominant_count)} failure(s) within ±{int(radius)} boundaries)"
    )
    return HailoClusterSkipDecision(skip=True, nearby_failures=int(dominant_count), detail=detail)


def build_case_hailo_variant_availability(
    suite_hailo_hefs: Mapping[str, Any],
    case_hefs: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    raw_suite = suite_hailo_hefs if isinstance(suite_hailo_hefs, Mapping) else {}
    raw_case = case_hefs if isinstance(case_hefs, Mapping) else {}
    all_arches = {
        hw
        for raw_hw in list(raw_suite.keys()) + list(raw_case.keys())
        for hw in [_normalize_physical_hailo_hw(raw_hw)]
        if hw
    }
    for hw_arch in sorted(all_arches):
        suite_entries = [
            dict(meta)
            for raw_hw, meta in raw_suite.items()
            if _normalize_physical_hailo_hw(raw_hw) == hw_arch
            and isinstance(meta, Mapping)
        ]
        case_entries = [
            dict(meta)
            for raw_hw, meta in raw_case.items()
            if _normalize_physical_hailo_hw(raw_hw) == hw_arch
            and isinstance(meta, Mapping)
        ]
        full_ok = any(
            bool(meta.get("full")) and not bool(meta.get("full_error"))
            for meta in suite_entries
        )
        part1_ok = any(
            bool(meta.get("part1")) and not bool(meta.get("part1_error"))
            for meta in case_entries
        )
        part2_ok = any(
            bool(meta.get("part2")) and not bool(meta.get("part2_error"))
            for meta in case_entries
        )
        full_errors = [
            str(meta.get("full_error") or "").strip()
            for meta in suite_entries
            if str(meta.get("full_error") or "").strip()
        ]
        part1_errors = [
            str(meta.get("part1_error") or "").strip()
            for meta in case_entries
            if str(meta.get("part1_error") or "").strip()
        ]
        part2_errors = [
            str(meta.get("part2_error") or "").strip()
            for meta in case_entries
            if str(meta.get("part2_error") or "").strip()
        ]
        out[str(hw_arch)] = {
            "full": bool(full_ok),
            "part1": bool(part1_ok),
            "part2": bool(part2_ok),
            "composed": bool(part1_ok and part2_ok),
            "part1_failed": bool(part1_errors and not part1_ok),
            "part2_failed": bool(part2_errors and not part2_ok),
            "full_failed": bool(full_errors and not full_ok),
            # Keep the vendor/build reason next to the boolean availability.
            # This is diagnostic metadata, not a second eligibility gate.
            "part1_error": "" if part1_ok else (part1_errors[0] if part1_errors else ""),
            "part2_error": "" if part2_ok else (part2_errors[0] if part2_errors else ""),
            "full_error": "" if full_ok else (full_errors[0] if full_errors else ""),
        }
    return out


def _normalize_run_variants(run: Mapping[str, Any]) -> List[str]:
    variants = run.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes, bytearray)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in variants:
        name = str(value or "").strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _normalize_physical_hailo_hw(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"hailo8", "hailo8r", "hailo8l", "hailo10p"}:
        return text
    if text in {"hailo10", "hailo10h"}:
        return "hailo10h"
    if "hailo8r" in text:
        return "hailo8r"
    if "hailo8l" in text:
        return "hailo8l"
    if "hailo8" in text:
        return "hailo8"
    if "hailo10p" in text:
        return "hailo10p"
    if "hailo10h" in text:
        return "hailo10h"
    if "hailo10" in text:
        return "hailo10h"
    return None


def _canonical_case_hailo_availability(
    case_variant_availability: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Merge policy evidence by physical Hailo architecture.

    ``hailo10`` is a historical family label for the physical DFC target
    ``hailo10h``.  Treating both as separate keys can manufacture a missing
    backend or duplicate a terminal state, so every policy consumer shares
    this canonical projection.
    """
    raw = (
        case_variant_availability
        if isinstance(case_variant_availability, Mapping)
        else {}
    )
    availability: Dict[str, Dict[str, Any]] = {}
    for raw_hw, raw_meta in raw.items():
        hw = _normalize_physical_hailo_hw(raw_hw)
        if not hw:
            continue
        meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        merged = availability.setdefault(hw, {})
        for kind in ("full", "part1", "part2", "composed"):
            merged[kind] = bool(merged.get(kind) or meta.get(kind))
            error = str(meta.get(f"{kind}_error") or "").strip()
            if error and not str(merged.get(f"{kind}_error") or "").strip():
                merged[f"{kind}_error"] = error
            merged[f"{kind}_failed"] = bool(
                merged.get(f"{kind}_failed")
                or meta.get(f"{kind}_failed")
                or error
            )
    return availability


def _hailo_stage_hw(run: Mapping[str, Any], stage_key: str) -> Optional[str]:
    stage = run.get(stage_key)
    if isinstance(stage, Mapping):
        if str(stage.get("type") or "").strip().lower() != "hailo":
            return None
        hw = _normalize_physical_hailo_hw(stage.get("hw_arch") or stage.get("arch") or stage.get("id") or "")
        return hw or None
    return _normalize_physical_hailo_hw(stage)


def run_variant_hailo_requirements(run: Mapping[str, Any], variant: str) -> List[Tuple[str, str]]:
    variant_key = str(variant or "").strip().lower()
    stage1_hw = _hailo_stage_hw(run, "stage1")
    stage2_hw = _hailo_stage_hw(run, "stage2")
    run_type = str(run.get("type") or "").strip().lower()
    run_hw = _normalize_physical_hailo_hw(run.get("hw_arch") or run.get("id") or "")

    req: List[Tuple[str, str]] = []
    if variant_key == "full":
        if run_type == "hailo":
            hw = run_hw or stage1_hw or stage2_hw
            if hw:
                req.append((str(hw), "full"))
        return req
    if variant_key == "part1":
        if stage1_hw:
            req.append((str(stage1_hw), "part1"))
        return req
    if variant_key == "part2":
        if stage2_hw:
            req.append((str(stage2_hw), "part2"))
        return req
    if variant_key == "composed":
        if stage1_hw:
            req.append((str(stage1_hw), "part1"))
        if stage2_hw:
            req.append((str(stage2_hw), "part2"))
        return req
    return req


def case_has_usable_hailo_variant(
    bench_plan_runs: Sequence[Mapping[str, Any]],
    case_variant_availability: Mapping[str, Any],
) -> bool:
    availability = _canonical_case_hailo_availability(
        case_variant_availability
    )
    for run in bench_plan_runs or []:
        if not isinstance(run, Mapping):
            continue
        for variant in _normalize_run_variants(run):
            req = run_variant_hailo_requirements(run, variant)
            if not req:
                continue
            if all(bool((availability.get(str(hw)) or {}).get(kind)) for hw, kind in req):
                return True
    return False


def case_satisfies_all_hailo_requirements(
    bench_plan_runs: Sequence[Mapping[str, Any]],
    case_variant_availability: Mapping[str, Any],
) -> bool:
    """Return whether every requested Hailo artefact is available for a case.

    Interactive benchmark generation historically kept a case when *any*
    Hailo variant survived, which is useful for exploratory work.  A resolved
    evaluation matrix is different: accepting a boundary whose requested
    Part1 failed leaves a permanent row hole even when a suite-level Full HEF
    exists.  The evaluation adapter opts into this stricter predicate so the
    generator rejects that boundary and continues through the candidate pool.
    """
    return not missing_case_hailo_requirements(
        bench_plan_runs,
        case_variant_availability,
    )


def missing_case_hailo_requirements(
    bench_plan_runs: Sequence[Mapping[str, Any]],
    case_variant_availability: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    """Return every requested Hailo artefact absent from one generated case.

    Availability is keyed by physical DFC architecture.  Normalize those keys
    as well as the plan requirements so equivalent labels cannot create either
    a false rejection or a false success.  The returned list is deterministic
    and is suitable for a structured generator-rejection record.
    """

    availability = _canonical_case_hailo_availability(
        case_variant_availability
    )

    requirements: set[Tuple[str, str]] = set()
    for run in bench_plan_runs or []:
        if not isinstance(run, Mapping):
            continue
        for variant in _normalize_run_variants(run):
            requirements.update(run_variant_hailo_requirements(run, variant))
    return sorted(
        (hw, kind)
        for hw, kind in requirements
        if not bool((availability.get(str(hw)) or {}).get(kind))
    )


def case_hailo_backend_terminal_states(
    bench_plan_runs: Sequence[Mapping[str, Any]],
    case_variant_availability: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return one explicit terminal build state per required Hailo artefact.

    A formal audit retains a case when one physical backend succeeds and
    another does not.  The retained case must still say exactly which backend
    artefact is ready, failed, or absent; otherwise partial retention merely
    moves the ambiguity downstream.  Only requirements actually present in the
    resolved run plan are emitted.
    """
    availability = _canonical_case_hailo_availability(
        case_variant_availability
    )

    requirements: set[Tuple[str, str]] = set()
    for run in bench_plan_runs or []:
        if not isinstance(run, Mapping):
            continue
        for variant in _normalize_run_variants(run):
            requirements.update(run_variant_hailo_requirements(run, variant))

    states: List[Dict[str, Any]] = []
    for hw, variant in sorted(requirements):
        meta = availability.get(str(hw)) or {}
        available = bool(meta.get(variant))
        error = str(meta.get(f"{variant}_error") or "").strip()
        failed = bool(meta.get(f"{variant}_failed") or error)
        if available:
            status = "ready"
            reason = "artifact_available"
        elif failed:
            status = "terminal_failed"
            reason = error or "build_failed"
        else:
            status = "terminal_missing"
            reason = "required_artifact_missing_after_build"
        states.append({
            "hw_arch": str(hw),
            "variant": str(variant),
            "status": status,
            "terminal": True,
            "available": available,
            "reason": reason,
        })
    return states
