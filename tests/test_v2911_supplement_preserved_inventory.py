"""Missing old measurements may not become new dispatcher work on reentry."""
import json

import pytest

from onnx_splitpoint_tool.workflow.supplement import build_supplement_plan, preserve_metadata
from onnx_splitpoint_tool.native_job_identity import missing_native_rows
from onnx_splitpoint_tool.workflow.artifacts import sha256_file
from tests.test_v2911_supplement_workflow import archive, write


@pytest.mark.parametrize("damage", ["missing_generic", "generic_precision", "quality_decision",
                                    "native_precision", "native_missing", "raw_missing"])
def test_reentry_preserves_original_identity_and_payload(archive, damage):
    source = write(archive / "raw_measurement.json", {"fps": 23})
    write(archive / "models/yolo26m/benchmark_results/normalized_results.json", {
        "schema": "onnx-splitpoint/normalized-benchmark-results", "model_id": "yolo26m",
        "evaluation_run_id": archive.name,
        "results": [{"source_path": str(source)}],
    })
    original = preserve_metadata(archive, "first")
    if damage in {"missing_generic", "generic_precision"}:
        path = archive / "reports/scientific/performance_results.json"
        rows = json.loads(path.read_text())
        if damage == "missing_generic":
            rows.clear()
        else:
            rows[0]["runtime_precision_identity"] = "contradictory-artifact"
        write(path, rows)
    elif damage == "quality_decision":
        path = archive / "quality_management/central_quality_summary.json"
        payload = json.loads(path.read_text())
        payload["results"][0]["decision"] = "pass"
        write(path, payload)
    elif damage.startswith("native"):
        path = archive / "reports/native_expected_matrix.json"
        payload = json.loads(path.read_text())
        payload["present_expected_rows"][0].update(
            {"actual_ok": False} if damage == "native_missing" else {"precision": "other"})
        write(path, payload)
    else:
        source.unlink()
    with pytest.raises(ValueError, match="supplement_original_"):
        build_supplement_plan(archive, original_root=original)


def test_unavailable_finalization_does_not_erase_completed_inventory(archive):
    original = preserve_metadata(archive, "first")
    expected = build_supplement_plan(archive, original_root=original)
    write(archive / "run_manifest.json", {"measurement_status": "unavailable", "status": "failed"})
    actual = build_supplement_plan(archive, original_root=original)
    assert actual == expected


@pytest.mark.parametrize("damage", ["missing_row", "missing_summary", "missing_raw_report",
                                    "measurement", "command", "report_binding"])
def test_native_dispatch_authority_cannot_lose_original_measurements(archive, damage):
    summary_path = archive / "reports/native_producer_summary.json"
    summary = json.loads(summary_path.read_text())
    old_row = summary["rows"][0]
    original = preserve_metadata(archive, "first")
    if damage == "missing_summary":
        summary_path.unlink()
    elif damage == "missing_raw_report":
        from pathlib import Path
        Path(old_row["report"]).unlink()
    else:
        if damage == "missing_row":
            summary["rows"] = []
            # This is the actual downstream selector. A stale successful
            # matrix alone would let this old measurement be scheduled again.
            assert missing_native_rows([old_row], summary["rows"]) == [old_row]
        elif damage == "measurement":
            summary["rows"][0]["fps_makespan"] = 99
        elif damage == "command":
            summary["rows"][0]["native_command_contract_sha256"] = "different-command"
        else:
            other = write(archive / "native_producers/other.json", {"fps_makespan": 23})
            summary["rows"][0]["report"] = str(other)
        write(summary_path, summary)
    with pytest.raises(ValueError, match="supplement_original_"):
        build_supplement_plan(archive, original_root=original)


def test_native_annotations_and_new_completed_rows_survive_reentry(archive):
    original = preserve_metadata(archive, "first")
    summary_path = archive / "reports/native_producer_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["rows"][0].update(task_quality_status="fail", claim_ok=False)
    added_report = write(archive / "native_producers/new_measurement.json", {"fps_makespan": 24})
    summary["rows"].append({**summary["rows"][0], "case": "b002", "fps_makespan": 24,
                            "report": str(added_report), "source_reports": [str(added_report)]})
    write(summary_path, summary)
    matrix_path = archive / "reports/native_expected_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["present_expected_rows"][1]["actual_ok"] = True
    write(matrix_path, matrix)
    plan = build_supplement_plan(archive, original_root=original)
    assert plan["counts"]["native"] == {"preserve_and_validate_locally": 2, "not_supported": 1}
    assert missing_native_rows(summary["rows"], summary["rows"]) == []


@pytest.mark.parametrize("damage", ["original_report", "original_summary", "source_same_size",
                                    "generic_source_same_size", "missing_binding"])
def test_preserved_metadata_and_native_source_still_require_original_byte_binding(archive, damage):
    generic_source = write(archive / "generic_measurement.json", {"fps": 23})
    write(archive / "models/yolo26m/benchmark_results/normalized_results.json", {
        "schema": "onnx-splitpoint/normalized-benchmark-results", "model_id": "yolo26m",
        "evaluation_run_id": archive.name, "results": [{"source_path": str(generic_source)}],
    })
    source_index = {path.relative_to(archive).as_posix(): {
        "path": path.relative_to(archive).as_posix(), "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in archive.rglob("*") if path.is_file()}
    original = preserve_metadata(archive, "first")
    build_supplement_plan(archive, original_root=original, source_index=source_index)
    if damage in {"source_same_size", "generic_source_same_size"}:
        path = generic_source if damage == "generic_source_same_size" else archive / "native_producers/original_measurement.json"
        before = path.read_bytes()
        path.write_bytes(before.replace(b"23", b"99"))
        assert path.stat().st_size == len(before)
    elif damage == "missing_binding":
        source_index.pop("reports/native_producer_summary.json")
    else:
        raw = ("reports/native_producer_summary.json" if damage == "original_summary"
               else "reports/scientific/performance_results.json")
        path = original / raw
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="supplement_original_(evidence_changed|binding_missing)"):
        build_supplement_plan(archive, original_root=original, source_index=source_index)


def test_native_remote_report_label_requires_frozen_command_transfer_mapping(archive):
    summary_path = archive / "reports/native_producer_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["rows"][0].update(
        report="/remote/original-run/original_measurement.json",
        source_reports=["/remote/original-run/original_measurement.json"],
        source_root=str(archive / "native_producers"),
        native_command_contract={"root": "/remote/original-run"},
    )
    write(summary_path, summary)
    source_index = {path.relative_to(archive).as_posix(): {
        "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
    } for path in archive.rglob("*") if path.is_file()}
    original = preserve_metadata(archive, "first")
    plan = build_supplement_plan(archive, original_root=original, source_index=source_index)
    assert plan["counts"]["native"]["preserve_and_validate_locally"] == 1
    summary["rows"][0]["native_command_contract"]["root"] = "/other/run"
    write(summary_path, summary)
    with pytest.raises(ValueError, match="supplement_original_native_measurement_missing_or_changed"):
        build_supplement_plan(archive, original_root=original, source_index=source_index)


def test_native_full_remote_cache_label_keeps_actual_collected_measurement_required(archive):
    summary_path = archive / "reports/native_producer_summary.json"
    summary = json.loads(summary_path.read_text())
    source_root = archive / "native_producers/full"
    report = write(source_root / "analysis_tables/native_full_baseline_eval.json", {"rows": [{"fps": 23}]})
    summary["rows"][0].update(
        backend="native_full_tensorrt", comparison_backend="hailo8", case="full",
        execution_mode="native_full_baseline", source_root=str(source_root),
        report="/remote/cache/model/native_trt_meta.json",
        source_reports=["/remote/cache/model/native_trt_meta.json"],
        full_command_contract={"root": "/remote/run"},
    )
    write(summary_path, summary)
    matrix_path = archive / "reports/native_expected_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["present_expected_rows"][0] = {**summary["rows"][0], "actual_ok": True}
    write(matrix_path, matrix)
    source_index = {path.relative_to(archive).as_posix(): {
        "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
    } for path in archive.rglob("*") if path.is_file()}
    original = preserve_metadata(archive, "first")
    build_supplement_plan(archive, original_root=original, source_index=source_index)
    report.unlink()
    with pytest.raises(ValueError, match="supplement_original_evidence_missing"):
        build_supplement_plan(archive, original_root=original, source_index=source_index)
