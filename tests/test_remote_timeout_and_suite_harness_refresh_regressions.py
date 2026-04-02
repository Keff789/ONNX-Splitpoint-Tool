from __future__ import annotations

from pathlib import Path
import py_compile



def test_refresh_suite_harness_bootstraps_and_then_becomes_noop(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness

    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "benchmark_set.json").write_text("{}\n", encoding="utf-8")

    case_dir = suite_dir / "b001"
    case_dir.mkdir()
    (case_dir / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    # Intentionally stale/broken runner so the refresh helper has something to repair.
    (case_dir / "run_split_onnxruntime.py").write_text("def broken(:\n", encoding="utf-8")

    first = refresh_suite_harness(suite_dir, benchmark_set_json=suite_dir / "benchmark_set.json")
    assert first["changed"] is True
    assert (suite_dir / "benchmark_suite.py").exists()
    assert (suite_dir / "splitpoint_runners" / "harness" / "yolo.py").exists()
    runner = case_dir / "run_split_onnxruntime.py"
    assert runner.exists()
    runner_text = runner.read_text(encoding="utf-8")
    assert "def _maybe_cast_for_onnx_input" in runner_text
    assert "def _shape_from_ort_input" in runner_text

    second = refresh_suite_harness(suite_dir, benchmark_set_json=suite_dir / "benchmark_set.json")
    assert second["changed"] is False
    assert second["case_count"] == 1
    assert second["bench_json_name"] == "benchmark_set.json"



def test_remote_timeout_and_suite_refresh_controls_are_present_in_sources() -> None:
    app_src = Path("onnx_splitpoint_tool/gui/app.py").read_text(encoding="utf-8")
    panel_src = Path("onnx_splitpoint_tool/gui/panels/panel_validate.py").read_text(encoding="utf-8")
    remote_src = Path("onnx_splitpoint_tool/benchmark/remote_run.py").read_text(encoding="utf-8")

    assert "var_remote_timeout" in app_src
    assert "def _refresh_selected_suite_harness" in app_src
    assert '"Outer timeout (s):"' in panel_src
    assert '"Refresh suite harness"' in panel_src
    assert "timeout_s: Optional[int] = 7200" in remote_src
    assert "outer_timeout_s = None" in remote_src
    assert "timeout=outer_timeout_s" in remote_src
    assert "refresh_suite_harness(" in remote_src



def test_new_remote_timeout_and_suite_refresh_sources_compile() -> None:
    py_compile.compile("onnx_splitpoint_tool/benchmark/suite_refresh.py", doraise=True)
    py_compile.compile("onnx_splitpoint_tool/benchmark/remote_run.py", doraise=True)
    py_compile.compile("onnx_splitpoint_tool/gui/app.py", doraise=True)
    py_compile.compile("onnx_splitpoint_tool/gui/panels/panel_validate.py", doraise=True)
