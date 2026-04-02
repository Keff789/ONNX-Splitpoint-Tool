from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.resume_integrity import reconcile_generation_state



def _write_case(case_dir: Path, *, boundary: int, rejected: bool = False) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"model_part1_b{boundary}.onnx").write_text("p1", encoding="utf-8")
    (case_dir / f"model_part2_b{boundary}.onnx").write_text("p2", encoding="utf-8")
    manifest = {
        "boundary": boundary,
        "part1": f"model_part1_b{boundary}.onnx",
        "part2": f"model_part2_b{boundary}.onnx",
        "predicted": {"comm_bytes": float(boundary)},
    }
    if rejected:
        manifest["benchmark_status"] = "rejected"
        manifest["benchmark_rejection"] = {
            "status": "rejected",
            "boundary": boundary,
            "folder": case_dir.name,
            "reason": "hailo_hef_build_failed",
        }
    (case_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")



def test_reconcile_generation_state_repairs_stale_and_missing_entries(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()

    # Valid accepted case already referenced by state.
    _write_case(suite / "b080", boundary=80)

    # Orphan accepted case present on disk but missing from state.
    _write_case(suite / "b090", boundary=90)

    # Rejected archived case.
    _write_case(suite / "_rejected_cases" / "b110", boundary=110, rejected=True)

    state = {
        "requested_cases": 3,
        "case_entries": [
            {"boundary": 80, "folder": "b080", "manifest": "split_manifest.json"},
            {"boundary": 85, "folder": "b085", "manifest": "split_manifest.json"},
        ],
        "discarded_case_entries": [
            {"boundary": 110, "folder": "b110", "reason": "hailo_hef_build_failed"},
            {"boundary": 120, "folder": "b120", "reason": "hailo_hef_build_failed"},
        ],
    }

    report = reconcile_generation_state(suite, state)

    assert report.changed is True
    assert report.dropped_cases == 1  # stale b085 entry
    assert report.inferred_cases == 1  # orphan b090 recovered
    assert report.inferred_rejections == 0
    assert report.dropped_rejections == 0

    repaired = report.repaired_state
    assert [int(x["boundary"]) for x in repaired["case_entries"]] == [80, 90]
    assert [int(x["boundary"]) for x in repaired["discarded_case_entries"]] == [110, 120]
    assert repaired["accepted_boundaries"] == [80, 90]
    assert repaired["discarded_boundaries"] == [110, 120]
    assert repaired["completed_boundaries"] == [80, 90, 110, 120]
    assert repaired["shortfall"] == 1
    assert any(("missing split_manifest.json" in w) or ("incomplete case artifacts" in w) for w in report.warnings)



def test_reconcile_generation_state_uses_benchmark_json_fallback(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    _write_case(suite / "b050", boundary=50)
    bench = {
        "cases": [{"boundary": 50, "folder": "b050", "manifest": "split_manifest.json"}],
        "discarded_cases": [],
    }
    (suite / "benchmark_set.json").write_text(json.dumps(bench), encoding="utf-8")

    report = reconcile_generation_state(suite, {})
    assert [int(x["boundary"]) for x in report.repaired_state["case_entries"]] == [50]
    assert report.repaired_state["generated_cases"] == 1
