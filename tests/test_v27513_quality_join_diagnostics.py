from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_reporter():
    path = ROOT / "scripts" / "native_producer_final_report.py"
    spec = importlib.util.spec_from_file_location(
        "v27513_quality_join_diagnostics", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _attestation(stage: str, digest: str) -> dict[str, object]:
    return {
        "attested": True,
        "status": "passed",
        "task": "detection",
        "stage": stage,
        "endpoint": stage,
        "endpoint_contract_hash": digest,
    }


def _row(*, precision: str = "fp16", endpoint_hash: str = "a" * 64):
    stage = "raw_head"
    return {
        "backend": "native_full_hailo8",
        "model": "yolov7_paper",
        "case": "full",
        "execution_mode": "native_full_baseline",
        "full_runtime_precision": precision,
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "task": "detection",
        "stage": stage,
        "contract_family": stage,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_attestation": _attestation(stage, endpoint_hash),
        "ok": True,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 5,
        "status": "complete",
        "row_count": len(rows),
        "rows": rows,
    }


def test_nearest_candidate_reports_only_runtime_precision_axis() -> None:
    reporter = _load_reporter()
    performance = _row(precision="fp16")
    quality = _row(precision="int8")

    attached, metadata = reporter._attach_quality_evidence(
        [performance], _summary([quality]), quality_summary_status="loaded",
    )

    result = attached[0]
    assert result["quality_match_status"] == "no_exact_identity_match"
    assert result["quality_match_count"] == 0
    assert result["precision_quality_binding_verified"] is False
    assert result["quality_nearest_candidate_count"] == 1
    candidate = result["quality_nearest_candidates"][0]
    assert candidate["quality_row_index"] == 0
    assert candidate["differing_axes"] == {
        "runtime_precision": {"performance": "fp16", "quality": "int8"},
    }
    assert metadata["missing_performance_row_count"] == 1


def test_nearest_candidate_reports_only_physical_endpoint_axis() -> None:
    reporter = _load_reporter()
    performance = _row(endpoint_hash="a" * 64)
    quality = _row(endpoint_hash="b" * 64)

    attached, _ = reporter._attach_quality_evidence(
        [performance], _summary([quality]), quality_summary_status="loaded",
    )

    result = attached[0]
    assert result["quality_match_status"] == "no_exact_identity_match"
    assert result["quality_match_count"] == 0
    assert result["precision_quality_binding_verified"] is False
    assert result["quality_nearest_candidate_count"] == 1
    candidate = result["quality_nearest_candidates"][0]
    assert candidate["differing_axes"] == {
        "output_endpoint_id": {
            "performance": f"detection:raw_head:{'a' * 64}",
            "quality": f"detection:raw_head:{'b' * 64}",
        },
    }


def test_nearest_candidates_never_cross_stable_workload_axes() -> None:
    reporter = _load_reporter()
    performance = _row()
    other_setup = _row()
    other_setup["setup_id"] = "orin_nx_hailo10_01"

    attached, _ = reporter._attach_quality_evidence(
        [performance], _summary([other_setup]), quality_summary_status="loaded",
    )

    result = attached[0]
    assert result["quality_match_status"] == "no_exact_identity_match"
    assert result["quality_nearest_candidate_count"] == 0
    assert result["quality_nearest_candidates"] == []


def test_exact_join_remains_exact_and_has_no_nearest_diagnostics() -> None:
    reporter = _load_reporter()
    performance = _row()
    quality = _row()

    attached, _ = reporter._attach_quality_evidence(
        [performance], _summary([quality]), quality_summary_status="loaded",
    )

    result = attached[0]
    assert result["quality_match_count"] == 1
    assert result["quality_nearest_candidate_count"] == 0
    assert result["quality_nearest_candidates"] == []

