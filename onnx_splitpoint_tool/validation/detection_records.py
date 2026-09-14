from __future__ import annotations

"""Strict, shared detection-record parsing.

The B500 reconciliation exposed a dangerous split between two equivalent
representations: legacy records used ``box_xyxy``/``confidence`` while current
completed-detection artefacts use scalar ``x1``...``y2``/``score`` fields.  A
consumer must support both complete schemas, but must never silently skip a
partially populated, mixed, non-finite, or inverted record.
"""

import math
from typing import Any, Iterable, Mapping, Sequence


class DetectionRecordError(ValueError):
    """Raised when a detection record is not one exact supported schema."""


CANONICAL_KEYS = ("x1", "y1", "x2", "y2", "score", "class_id")
LEGACY_KEYS = ("box_xyxy", "confidence", "class_id")


def _finite_float(value: Any, *, field: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectionRecordError(
            f"detection_record_numeric_invalid:{context}:{field}"
        ) from exc
    if not math.isfinite(result):
        raise DetectionRecordError(
            f"detection_record_nonfinite:{context}:{field}"
        )
    return result


def parse_detection_record(
    record: Mapping[str, Any], *, context: str = "record"
) -> dict[str, Any]:
    """Return one canonical ``xyxy/score/class_id`` record.

    Exactly one complete representation is accepted.  Extra metadata keys are
    allowed, but the two coordinate/score representations may not be mixed.
    Scores are required to be probabilities in ``[0, 1]`` and class IDs must
    be finite integers.
    """

    if not isinstance(record, Mapping):
        raise DetectionRecordError(f"detection_record_not_mapping:{context}")
    canonical_present = any(key in record for key in CANONICAL_KEYS[:-1])
    legacy_present = any(key in record for key in LEGACY_KEYS[:-1])
    if canonical_present and legacy_present:
        raise DetectionRecordError(f"detection_record_schema_mixed:{context}")
    if "class_id" not in record:
        raise DetectionRecordError(f"detection_record_class_missing:{context}")

    if canonical_present:
        missing = [key for key in CANONICAL_KEYS if key not in record]
        if missing:
            raise DetectionRecordError(
                f"detection_record_canonical_partial:{context}:{','.join(missing)}"
            )
        coords = [
            _finite_float(record[key], field=key, context=context)
            for key in ("x1", "y1", "x2", "y2")
        ]
        score = _finite_float(record["score"], field="score", context=context)
        schema = "canonical_xyxy_score"
    elif legacy_present:
        missing = [key for key in LEGACY_KEYS if key not in record]
        if missing:
            raise DetectionRecordError(
                f"detection_record_legacy_partial:{context}:{','.join(missing)}"
            )
        box = record["box_xyxy"]
        if (
            not isinstance(box, Sequence)
            or isinstance(box, (str, bytes, bytearray))
            or len(box) != 4
        ):
            raise DetectionRecordError(
                f"detection_record_legacy_box_invalid:{context}"
            )
        coords = [
            _finite_float(value, field=f"box_xyxy[{idx}]", context=context)
            for idx, value in enumerate(box)
        ]
        score = _finite_float(
            record["confidence"], field="confidence", context=context
        )
        schema = "legacy_box_xyxy_confidence"
    else:
        raise DetectionRecordError(f"detection_record_schema_unknown:{context}")

    class_value = _finite_float(
        record["class_id"], field="class_id", context=context
    )
    if not class_value.is_integer():
        raise DetectionRecordError(f"detection_record_class_noninteger:{context}")
    if not 0.0 <= score <= 1.0:
        raise DetectionRecordError(f"detection_record_score_out_of_range:{context}")
    x1, y1, x2, y2 = coords
    if x2 < x1 or y2 < y1:
        raise DetectionRecordError(f"detection_record_box_inverted:{context}")
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": score,
        "class_id": int(class_value),
        "source_schema": schema,
    }


def normalize_detection_records(
    records: Iterable[Mapping[str, Any]], *, context: str = "records"
) -> list[dict[str, Any]]:
    """Parse every record; no item is silently skipped."""

    out: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        out.append(parse_detection_record(record, context=f"{context}[{index}]"))
    return out


def detection_confidence_summary(
    records: Iterable[Mapping[str, Any]], *, context: str = "records"
) -> dict[str, Any]:
    parsed = normalize_detection_records(records, context=context)
    values = [float(item["score"]) for item in parsed]
    return {
        "record_count": len(parsed),
        "confidence_mean": (sum(values) / len(values)) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "source_schemas": sorted({item["source_schema"] for item in parsed}),
    }
