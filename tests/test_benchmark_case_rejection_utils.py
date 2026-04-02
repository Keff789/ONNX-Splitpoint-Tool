from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.benchmark_case_utils import archive_benchmark_case, build_benchmark_case_rejection


def test_build_benchmark_case_rejection_carries_hailo_failure_metadata() -> None:
    hef = SimpleNamespace(
        ok=False,
        skipped=False,
        timed_out=True,
        timeout_kind="hard",
        last_stage="allocation",
        failure_kind="timeout",
        unsupported_reason=None,
        debug_log="/tmp/debug.log",
        returncode=124,
        backend="venv",
        net_name="demo_part2_b189",
        hw_arch="hailo8",
        error="Resolver didn't find possible solution.",
        details={"probe": "mapping"},
    )

    rec = build_benchmark_case_rejection(
        boundary=189,
        folder="b189",
        reason="hailo_hef_build_failed",
        stage="part2",
        hw_arch="hailo8",
        hef_result=hef,
    )

    assert rec["status"] == "rejected"
    assert rec["boundary"] == 189
    assert rec["folder"] == "b189"
    assert rec["stage"] == "part2"
    assert rec["hw_arch"] == "hailo8"
    assert rec["failure_kind"] == "timeout"
    assert rec["timed_out"] is True
    assert rec["timeout_kind"] == "hard"
    assert rec["last_stage"] == "allocation"
    assert rec["returncode"] == 124
    assert "Resolver didn't find possible solution" in rec["detail"]
    assert rec["details"] == {"probe": "mapping"}


def test_archive_benchmark_case_moves_case_into_rejected_folder(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    case_dir = suite / "b189"
    case_dir.mkdir(parents=True)
    manifest = case_dir / "split_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    info = archive_benchmark_case(case_dir, suite, folder="b189")

    assert info["archive_root"] == "_rejected_cases"
    assert info["archive_dir"] == "_rejected_cases/b189"
    assert not case_dir.exists()
    assert (suite / "_rejected_cases" / "b189" / "split_manifest.json").exists()
