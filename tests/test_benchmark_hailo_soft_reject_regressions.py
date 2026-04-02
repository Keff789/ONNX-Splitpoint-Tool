from __future__ import annotations

from pathlib import Path


ROOT = Path("onnx_splitpoint_tool")


def test_gui_benchmark_worker_tracks_discarded_hailo_cases_and_archives_them() -> None:
    src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    assert "discarded_cases = []" in src
    assert "build_benchmark_case_rejection(" in src
    assert 'manifest_out["benchmark_status"] = "rejected"' in src
    assert 'manifest_out["benchmark_rejection"] = dict(case_rejection)' in src
    assert "archive_benchmark_case(case_dir, out_dir, folder=folder)" in src
    assert "q.put((\"prog\", made, f\"b{b} (reject: Hailo build failed)\"))" in src
    assert "'discarded_cases': discarded_cases" in src
    assert "Shortfall:" in src


def test_benchmark_readme_mentions_rejected_cases_archive() -> None:
    src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    assert "Rejected split attempts:" in src
    assert "_rejected_cases/" in src
