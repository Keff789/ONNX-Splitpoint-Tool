from __future__ import annotations

import csv

from onnx_splitpoint_tool.workflow.execution_binding import _inspect_canonical_result


def test_canonical_csv_audit_accepts_large_embedded_evidence_and_restores_limit(tmp_path):
    result_path = tmp_path / "benchmark_results_hailo10_to_trt_auto.csv"
    embedded_evidence = "x" * (256 * 1024)
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "case_id", "evidence"])
        writer.writeheader()
        writer.writerow({
            "run_id": "hailo10_to_trt",
            "case_id": "b052",
            "evidence": embedded_evidence,
        })

    previous_limit = csv.field_size_limit()
    audit = _inspect_canonical_result(result_path)

    assert csv.field_size_limit() == previous_limit
    assert audit["parseable"] is True
    assert audit["nonempty"] is True
    assert audit["row_count"] == 1
    assert audit["semantic_status"] == "measured_rows"
    assert "parse_error" not in audit
