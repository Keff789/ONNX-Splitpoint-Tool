"""Cardinality of a model's explicitly bound selection (not a global cap)."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


def requested_cases_for_model(policy: Mapping[str, Any], model_id: str,
                              *, default: int, audit_scope: bool = False) -> int:
    # Audit forced_cases are deployment anchors inside an independently frozen
    # audit union. Their archived shortlist-cap contract is not the selected
    # measurement population and must remain replayable unchanged.
    if audit_scope or policy.get("score_independent_audit_enabled") is True:
        return default
    raw = (policy.get("forced_cases") or policy.get("fixed_cases")
           or policy.get("case_map") or {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return default
    if not isinstance(raw, Mapping) or model_id not in raw:
        return default
    values = raw[model_id]
    if isinstance(values, str):
        values = [value.strip() for value in values.replace(";", ",").split(",") if value.strip()]
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise ValueError(f"explicit_case_selection_not_sequence:{model_id}")
    canonical = []
    for value in values:
        match = re.fullmatch(r"b?(\d+)", str(value).strip(), flags=re.I)
        if match is None:
            raise ValueError(f"explicit_case_selection_invalid:{model_id}:{value}")
        canonical.append(f"b{int(match.group(1)):03d}")
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError(f"explicit_case_selection_empty_or_duplicate:{model_id}")
    return len(canonical)
