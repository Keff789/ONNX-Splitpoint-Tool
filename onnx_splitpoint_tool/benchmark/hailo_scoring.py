from __future__ import annotations

"""Heuristic Hailo compile/context-aware scoring.

This is intentionally lightweight and prediction-only. It does **not** replace
actual Hailo compilation/runtime checks; instead it helps order the benchmark
candidate queue so likely easier-to-compile / likely single-context splits are
tried earlier.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, List

from .hailo_policy import build_candidate_policy_index

MiB = 1024.0 ** 2


@dataclass
class HailoCompileHeuristic:
    boundary: int
    base_score: Optional[float]
    cut_mib: Optional[float]
    peak_act_right_mib: Optional[float]
    n_cut_tensors: Optional[int]
    flops_right_ratio: Optional[float]
    strict_ok: Optional[bool]
    compile_risk_score: float
    single_context_probability: float


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _analysis_indexed_value(container: Any, boundary: int) -> Any:
    if container is None:
        return None
    try:
        b = int(boundary)
    except Exception:
        return None
    if isinstance(container, Mapping):
        if b in container:
            return container.get(b)
        return container.get(str(b))
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
        if 0 <= b < len(container):
            return container[b]
    return None


def heuristic_for_boundary(analysis: Mapping[str, Any], boundary: int) -> HailoCompileHeuristic:
    b = int(boundary)
    scores = analysis.get('scores') if isinstance(analysis.get('scores'), Mapping) else {}
    candidate_policy_index = build_candidate_policy_index(analysis.get('_candidate_rows') or analysis.get('candidates') or [])
    candidate_policy = candidate_policy_index.get(b) or {}
    base_score = _to_float(_analysis_indexed_value(scores, b))
    cut_bytes = _to_float(_analysis_indexed_value(analysis.get('costs_bytes') or [], b))
    cut_mib = (cut_bytes / MiB) if cut_bytes is not None else None
    peak_right_b = _to_float(_analysis_indexed_value(analysis.get('peak_act_mem_right_bytes') or [], b))
    peak_right_mib = (peak_right_b / MiB) if peak_right_b is not None else None
    n_cut = _analysis_indexed_value(analysis.get('crossing_counts_all') or analysis.get('crossing_counts_known') or [], b)
    try:
        n_cut_tensors = int(n_cut) if n_cut is not None else None
    except Exception:
        n_cut_tensors = None
    fl_left = _to_float(_analysis_indexed_value(analysis.get('flops_left_prefix') or [], b))
    total_flops = _to_float(analysis.get('total_flops'))
    flops_right_ratio = None
    if fl_left is not None and total_flops not in (None, 0.0):
        try:
            flops_right_ratio = max(0.0, min(1.0, (float(total_flops) - float(fl_left)) / float(total_flops)))
        except Exception:
            flops_right_ratio = None
    strict_ok_raw = _analysis_indexed_value(analysis.get('strict_ok') or [], b)
    strict_ok = (bool(strict_ok_raw) if strict_ok_raw is not None else None)

    risk = 0.0
    if cut_mib is not None:
        risk += 0.56 * math.log1p(max(cut_mib, 0.0))
    if peak_right_mib is not None:
        risk += 0.34 * math.log1p(max(peak_right_mib, 0.0))
    if n_cut_tensors is not None:
        risk += 0.28 * math.log1p(max(int(n_cut_tensors), 0))
        if int(n_cut_tensors) >= 6:
            risk += 0.22
        if int(n_cut_tensors) >= 7:
            risk += 0.34
        if int(n_cut_tensors) >= 8:
            risk += 0.48
    if flops_right_ratio is not None:
        risk += 0.16 * max(float(flops_right_ratio), 0.0) * 5.0
    if strict_ok is False:
        risk += 2.60

    risk += float(candidate_policy.get('hailo_structural_penalty') or 0.0)
    risk += float(candidate_policy.get('hailo_alt_strategy_penalty') or 0.0)
    if bool(candidate_policy.get('hailo_parse_checked')) and candidate_policy.get('hailo_parse_ok') is False:
        risk += 0.70
    if bool(candidate_policy.get('hailo_parse_used_suggested_end_nodes')):
        risk += 0.38
    if bool(candidate_policy.get('hailo_crosses_model22_and_model23')):
        risk += 0.25
    if int(candidate_policy.get('hailo_raw_activation_cut_count') or 0) >= 2:
        risk += 0.24
    if int(candidate_policy.get('hailo_one2one_cut_count') or 0) >= 1:
        risk += 0.22
    if int(candidate_policy.get('hailo_attn_cut_count') or 0) >= 1:
        risk += 0.16
    if int(candidate_policy.get('hailo_qkv_cut_count') or 0) >= 1:
        risk += 0.12
    risk += (b % 17) * 1e-4

    single_prob = 1.0 / (1.0 + math.exp(max(-8.0, min(8.0, (risk - 2.55) * 1.32))))
    return HailoCompileHeuristic(
        boundary=b,
        base_score=base_score,
        cut_mib=cut_mib,
        peak_act_right_mib=peak_right_mib,
        n_cut_tensors=n_cut_tensors,
        flops_right_ratio=flops_right_ratio,
        strict_ok=strict_ok,
        compile_risk_score=float(risk),
        single_context_probability=float(single_prob),
    )


def rerank_candidates_for_hailo(analysis: Mapping[str, Any], boundaries: Iterable[int]) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    candidate_policy_index = build_candidate_policy_index(analysis.get('_candidate_rows') or analysis.get('candidates') or [])
    heuristics: List[HailoCompileHeuristic] = [heuristic_for_boundary(analysis, int(b)) for b in boundaries]

    def _sort_key(h: HailoCompileHeuristic):
        strict_penalty = 0.0 if h.strict_ok is not False else 1.0
        base = h.base_score if h.base_score is not None else (h.cut_mib if h.cut_mib is not None else 1e9)
        return (strict_penalty, h.compile_risk_score, float(base), int(h.boundary))

    heuristics.sort(key=_sort_key)
    meta: Dict[int, Dict[str, Any]] = {
        h.boundary: {
            'hailo_compile_risk_score': h.compile_risk_score,
            'hailo_single_context_probability': h.single_context_probability,
            'hailo_cut_mib': h.cut_mib,
            'hailo_peak_act_right_mib': h.peak_act_right_mib,
            'hailo_n_cut_tensors': h.n_cut_tensors,
            'hailo_flops_right_ratio': h.flops_right_ratio,
            'hailo_strict_ok': h.strict_ok,
            **dict(candidate_policy_index.get(h.boundary) or {}),
        }
        for h in heuristics
    }
    return [h.boundary for h in heuristics], meta
