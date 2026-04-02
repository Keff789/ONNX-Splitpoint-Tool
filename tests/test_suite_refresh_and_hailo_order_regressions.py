from __future__ import annotations

from pathlib import Path


def test_write_benchmark_suite_script_refreshes_runner_lib_even_when_script_is_unchanged(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script

    out_dir = tmp_path / "suite"
    first = Path(write_benchmark_suite_script(out_dir, bench_json_name="benchmark_set.json"))
    assert first.exists()
    vendored = out_dir / "splitpoint_runners" / "harness" / "yolo.py"
    assert vendored.exists()

    original = vendored.read_text(encoding="utf-8")
    vendored.write_text("# stale vendored copy\n", encoding="utf-8")

    second = Path(write_benchmark_suite_script(out_dir, bench_json_name="benchmark_set.json"))
    assert second == first
    refreshed = vendored.read_text(encoding="utf-8")
    assert refreshed == original
    assert "def _yolo_multiscale_name_hint" in refreshed


def test_template_hailo_stream_order_preserves_original_order_for_non_slot_names() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert "def _hailo_stream_names_ordered" in src
    assert "Preserve the original HEF stream order for non-generic names" in src


def test_template_yolo_multiscale_name_hint_uses_local_re_import() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert "def _yolo_multiscale_name_hint" in src
    assert "import re as _re" in src


def test_runner_harness_yolo_name_hint_uses_local_re_import() -> None:
    src = Path("onnx_splitpoint_tool/runners/harness/yolo.py").read_text(encoding="utf-8")
    assert "def _yolo_multiscale_name_hint" in src
    assert "import re as _re" in src
