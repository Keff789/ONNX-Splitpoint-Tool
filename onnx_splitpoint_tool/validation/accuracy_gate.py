"""Compatibility wrapper for v59ej accuracy gates.

Older scripts imported ``accuracy_gate``; the implementation now lives in
``accuracy_gates``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from .accuracy_gates import (
    AccuracyGatePolicy,
    DEFAULT_POLICY,
    apply_accuracy_gate_to_row,
    apply_accuracy_gates_to_payload,
    gate_counts,
)

# Backward compatible names.
GateStatus = dict


def apply_accuracy_gate(row: Mapping[str, Any], *, baseline: Mapping[str, Any] | None = None, policy: Mapping[str, Any] | AccuracyGatePolicy | None = None) -> dict[str, Any]:
    # Baseline comparison is handled when rows expose *_delta or candidate/full
    # metrics; the baseline argument is kept for API compatibility.
    merged = dict(row)
    if baseline:
        # Add common reference metrics if the row has candidate metrics but no full/reference metric.
        if 'mini_classification_eval_full_top1' not in merged and 'mini_classification_eval_primary_top1' in baseline:
            merged['mini_classification_eval_full_top1'] = baseline.get('mini_classification_eval_primary_top1')
        if 'mini_coco_ap50_full' not in merged and 'mini_coco_ap50_primary' in baseline:
            merged['mini_coco_ap50_full'] = baseline.get('mini_coco_ap50_primary')
    apply_accuracy_gate_to_row(merged, policy)
    return merged


def annotate_rows(rows: Sequence[MutableMapping[str, Any]], *, policy: Mapping[str, Any] | AccuracyGatePolicy | None = None) -> list[MutableMapping[str, Any]]:
    out=[]
    for row in rows:
        apply_accuracy_gate_to_row(row, policy)
        row['ranking_status'] = 'eligible' if row.get('eligible_for_ranking') else row.get('ranking_exclusion_reason')
        row['thesis_valid'] = bool(row.get('eligible_for_ranking'))
        out.append(row)
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return gate_counts(list(rows))


def find_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    # Kept for API compatibility. New gating prefers explicit candidate/full or delta fields.
    return {}
