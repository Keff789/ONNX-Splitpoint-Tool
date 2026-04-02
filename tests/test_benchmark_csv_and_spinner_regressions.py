from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_benchmark_suite_csv_writer_accepts_heterogeneous_rows(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script

    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    suite_script = Path(write_benchmark_suite_script(suite_dir, bench_json_name="benchmark_set.json"))
    mod = _load_module(suite_script, "generated_benchmark_suite_csv_union")

    rows = [
        {"boundary": 298, "full_mean_ms": 1.0, "status": "skipped"},
        {
            "boundary": 80,
            "full_mean_ms": 2.0,
            "part2_mean_ms": 3.0,
            "composed_mean_ms": 4.0,
            "sum_parts_ms": 5.0,
            "overhead_ms": 1.0,
            "speedup_full_over_composed": 0.5,
        },
    ]
    out_csv = suite_dir / "heterogeneous.csv"
    mod._write_results_csv(out_csv, rows)

    with out_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        data = list(reader)

    assert "boundary" in fieldnames
    assert "full_mean_ms" in fieldnames
    assert "part2_mean_ms" in fieldnames
    assert "composed_mean_ms" in fieldnames
    assert "sum_parts_ms" in fieldnames
    assert "overhead_ms" in fieldnames
    assert "speedup_full_over_composed" in fieldnames
    assert len(data) == 2
    assert data[0]["boundary"] == "298"
    assert data[1]["boundary"] == "80"
    assert data[1]["part2_mean_ms"] == "3.0"


def test_runner_template_avoids_shadowing_thread_stop_internal() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert "self._stop_event = threading.Event()" in src
    assert "self._stop = threading.Event()" not in src
    assert "self._stop_event.set()" in src
    assert "self._stop_event.is_set()" in src
