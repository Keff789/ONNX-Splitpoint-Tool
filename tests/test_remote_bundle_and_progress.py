import tarfile
from pathlib import Path


def test_parse_benchmark_suite_progress():
    from onnx_splitpoint_tool.benchmark.remote_run import parse_benchmark_suite_progress

    p = parse_benchmark_suite_progress("[abc] [1/10] boundary=...")
    assert p is not None
    assert (p.run_id, p.i, p.n, p.pct) == ("abc", 1, 10, 0.1)
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



def test_build_suite_bundle_rebuilds_when_content_changes_but_size_and_mtime_stay_equal(tmp_path: Path):
    import os

    from onnx_splitpoint_tool.remote.bundle import build_suite_bundle

    suite = tmp_path / "suite"
    suite.mkdir()
    payload = suite / "payload.bin"
    payload.write_bytes(b"AAAA")

    out = tmp_path / "bundle.tar.gz"

    first = build_suite_bundle(suite, out)
    assert first.reused is False

    st = payload.stat()
    payload.write_bytes(b"BBBB")  # same size, different bytes
    os.utime(payload, ns=(st.st_atime_ns, st.st_mtime_ns))

    second = build_suite_bundle(suite, out)
    assert second.reused is False
