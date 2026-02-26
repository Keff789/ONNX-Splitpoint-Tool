import tarfile
from pathlib import Path


def test_parse_benchmark_suite_progress():
    from onnx_splitpoint_tool.benchmark.remote_run import parse_benchmark_suite_progress

    assert parse_benchmark_suite_progress("[abc] [1/10] boundary=...") == (1, 10)
    assert parse_benchmark_suite_progress("no match") is None


def test_build_suite_bundle_excludes_results(tmp_path: Path):
    from onnx_splitpoint_tool.remote.bundle import build_suite_bundle

    suite = tmp_path / "suite"
    (suite / "cases" / "case1" / "results_cpu").mkdir(parents=True)
    (suite / "cases" / "case1" / "results_cpu" / "report.json").write_text("x")
    (suite / "cases" / "case1" / "part1.onnx").write_text("onnx")
    (suite / "benchmark_set.json").write_text("{}")
    (suite / "benchmark_report_cpu.json").write_text("should_exclude")

    out = tmp_path / "bundle.tar.gz"
    build_suite_bundle(suite, out)
    assert out.exists()

    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    # required
    assert "benchmark_set.json" in names
    assert "cases/case1/part1.onnx" in names
    # excluded
    assert "cases/case1/results_cpu/report.json" not in names
    assert "benchmark_report_cpu.json" not in names
