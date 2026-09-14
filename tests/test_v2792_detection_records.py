from __future__ import annotations

import math
import pytest

from onnx_splitpoint_tool.validation.detection_records import (
    DetectionRecordError,
    detection_confidence_summary,
    normalize_detection_records,
    parse_detection_record,
)


def test_canonical_and_legacy_records_share_one_strict_parser() -> None:
    canonical = parse_detection_record({
        "x1": 1, "y1": 2, "x2": 11, "y2": 12,
        "score": 0.75, "class_id": 3,
    })
    legacy = parse_detection_record({
        "box_xyxy": [1, 2, 11, 12], "confidence": 0.75,
        "class_id": 3,
    })
    assert {k: canonical[k] for k in ("x1", "y1", "x2", "y2", "score", "class_id")} == {
        k: legacy[k] for k in ("x1", "y1", "x2", "y2", "score", "class_id")
    }
    summary = detection_confidence_summary([canonical, legacy])
    assert summary["record_count"] == 2
    assert summary["confidence_mean"] == pytest.approx(0.75)


@pytest.mark.parametrize("record,reason", [
    ({"x1": 1, "y1": 2, "x2": 3, "score": .5, "class_id": 0}, "canonical_partial"),
    ({"box_xyxy": [1, 2, 3, 4], "class_id": 0}, "legacy_partial"),
    ({"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": .5, "confidence": .5, "class_id": 0}, "schema_mixed"),
    ({"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": float("nan"), "class_id": 0}, "nonfinite"),
    ({"x1": 3, "y1": 2, "x2": 1, "y2": 4, "score": .5, "class_id": 0}, "box_inverted"),
])
def test_invalid_records_fail_explicitly(record, reason: str) -> None:
    with pytest.raises(DetectionRecordError, match=reason):
        parse_detection_record(record)


def test_no_existing_detection_can_be_silently_counted_as_zero() -> None:
    records = [
        {"x1": i, "y1": i, "x2": i + 1, "y2": i + 1, "score": .5, "class_id": 0}
        for i in range(3753)
    ]
    parsed = normalize_detection_records(records)
    assert len(parsed) == 3753
