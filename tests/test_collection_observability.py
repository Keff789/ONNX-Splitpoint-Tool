"""Small local Bash/process fixtures; no SSH, datasets, SDK or inference."""
from __future__ import annotations

import errno
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import _remote_result_collect_script
from onnx_splitpoint_tool.remote import ssh_transport


def _collect(suite: Path, out: Path, prefix: str = ""):
    return subprocess.run(
        ["/bin/bash", "-c", prefix + _remote_result_collect_script(
            remote_results_dir=str(out), remote_suite_dir=str(suite))],
        capture_output=True, text=True, timeout=5,
    )


def _end(result):
    return [line for line in result.stderr.splitlines()
            if line.startswith("SPLITPOINT_COLLECT_END ")]


def test_empty_optional_sources_have_start_end_and_original_zero(tmp_path):
    suite = tmp_path / "TEST_ONLY_empty"
    suite.mkdir()
    result = _collect(suite, tmp_path / "new")
    assert result.returncode == 0
    assert result.stderr.splitlines().count("SPLITPOINT_COLLECT_BEGIN") == 1
    assert _end(result) == ["SPLITPOINT_COLLECT_END exit_trap_status=0 copy_failed=0"]
    assert "SPLITPOINT_COLLECT_CP_FAILED" not in result.stderr


def test_silent_cp_failure_records_raw_rc_paths_and_continues_later_copy(tmp_path):
    suite = tmp_path / "TEST_ONLY suite with 'quote"
    suite.mkdir()
    first = suite / "benchmark_plan.json"
    first.write_text("TEST_ONLY")
    later = suite / "b003/results_TEST_ONLY/later.txt"
    later.parent.mkdir(parents=True)
    later.write_text("TEST_ONLY later")
    out = tmp_path / "new"
    result = _collect(suite, out, 'cp() { case "$2" in */benchmark_plan.json) return 23;; esac; command cp "$@"; }; ')
    assert result.returncode == 1
    assert (out / "b003/results_TEST_ONLY/later.txt").read_text() == "TEST_ONLY later"
    quoted = subprocess.run(["/bin/bash", "-c", 'printf "%q" "$1"', "TEST_ONLY", str(first)], capture_output=True, text=True, check=True).stdout
    assert f"SPLITPOINT_COLLECT_CP_FAILED rc=23 source={quoted} " in result.stderr
    assert _end(result) == ["SPLITPOINT_COLLECT_END exit_trap_status=1 copy_failed=1"]
    assert first.read_text() == "TEST_ONLY"


def test_early_set_e_failure_has_real_trap_status_without_claiming_cp_failure(tmp_path):
    suite = tmp_path / "TEST_ONLY_empty"
    suite.mkdir()
    out = tmp_path / "not_a_directory"
    out.write_text("TEST_ONLY preserved")
    result = _collect(suite, out)
    assert result.returncode == 1
    assert _end(result) == ["SPLITPOINT_COLLECT_END exit_trap_status=1 copy_failed=0"]
    assert "SPLITPOINT_COLLECT_CP_FAILED" not in result.stderr
    assert out.read_text() == "TEST_ONLY preserved"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_diagnostic_exit_trap_preserves_actual_signal_returncode(tmp_path, sig):
    suite = tmp_path / "TEST_ONLY_suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text("TEST_ONLY")
    result = _collect(suite, tmp_path / "new", f"cp() {{ kill -{sig.value} $$; }}; ")
    assert result.returncode == -sig.value
    assert "SPLITPOINT_COLLECT_BEGIN" in result.stderr
    # A trap's previous-command status is explicitly not a process exit proof.
    if sig == signal.SIGTERM:
        assert _end(result) == ["SPLITPOINT_COLLECT_END exit_trap_status=0 copy_failed=0"]
    else:
        assert _end(result) == []


def test_failed_diagnostic_write_does_not_replace_copy_status(tmp_path):
    suite = tmp_path / "TEST_ONLY_suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text("TEST_ONLY")
    prefix = "printf() { return 99; }; cp() { return 23; }; "
    result = _collect(suite, tmp_path / "new", prefix)
    assert result.returncode == 1


def _local_capture(tmp_path, monkeypatch, *, exitcode=0, reader_error=False,
                   wait_error=False, cleanup_proven=True):
    """Real normal capture/diagnostic writer; replace only local command leaf."""
    transport = ssh_transport.SSHTransport(
        ssh_transport.HostConfig(id="TEST_ONLY", label="TEST_ONLY", host="invalid.local", user="TEST_ONLY"),
    )
    transport.diagnostics_dir = tmp_path / "diagnostics"
    monkeypatch.setattr(transport, "_begin_remote_lease", lambda **_k: None)
    cleanups = []
    def finish(_op, *, abnormal):
        cleanups.append(abnormal)
        return cleanup_proven
    monkeypatch.setattr(transport, "_finish_remote_lease", finish)
    monkeypatch.setattr(transport, "_ssh_cmd", lambda *_a, **_k: [sys.executable, "-c", f"raise SystemExit({exitcode})"])
    real_popen = subprocess.Popen
    def controlled_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if reader_error:
            original = proc.stdout
            class BrokenReader:
                def read(self, _size): raise OSError(errno.EIO, "TEST_ONLY reader failure")
                def close(self): original.close()
            proc.stdout = BrokenReader()
        if wait_error:
            def broken_wait(timeout=None):
                raise subprocess.TimeoutExpired("TEST_ONLY_wait", timeout)
            proc.wait = broken_wait
        return proc
    monkeypatch.setattr(ssh_transport.subprocess, "Popen", controlled_popen)
    result = transport.run("TEST_ONLY_LOCAL_CAPTURE", timeout_s=5)
    paths = list((tmp_path / "diagnostics").glob("ssh_*.json"))
    assert len(paths) == 1
    return result, json.loads(paths[0].read_text()), cleanups


@pytest.mark.parametrize("exitcode", [0, 7])
def test_public_run_persists_actual_capture_phase_and_rc(tmp_path, monkeypatch, exitcode):
    result, report, cleanups = _local_capture(tmp_path, monkeypatch, exitcode=exitcode)
    assert result == (exitcode, "")
    assert cleanups == [exitcode != 0]
    capture = report["capture"]
    assert capture["phase"] == "completed"
    assert capture["wait_outcome"] == "returned"
    assert capture["returncode"] == capture["local_exit_code"] == exitcode
    assert capture["reader_error_type"] is None


def test_reader_exception_is_persisted_without_changing_returncode(tmp_path, monkeypatch):
    result, report, cleanups = _local_capture(tmp_path, monkeypatch, reader_error=True)
    assert result == (0, "") and cleanups == [False]
    assert report["capture"]["reader_error_type"] == "OSError"
    assert report["capture"]["reader_error_errno"] == errno.EIO
    assert report["capture"]["reader_finished"] is True
    assert report["capture"]["returncode"] == 0


def test_wait_exception_remains_unknown_phase_with_same_fallback_rc(tmp_path, monkeypatch):
    result, report, cleanups = _local_capture(tmp_path, monkeypatch, wait_error=True)
    assert result == (1, "") and cleanups == [True]
    capture = report["capture"]
    assert capture["phase"] == "process_exit_unknown"
    assert capture["wait_outcome"] == "exception"
    assert capture["wait_error_type"] == "TimeoutExpired"
    assert capture["local_exit_code"] == 0 and capture["returncode"] == 1


def test_existing_cleanup_override_keeps_separate_capture_rc(tmp_path, monkeypatch):
    result, report, cleanups = _local_capture(tmp_path, monkeypatch, exitcode=7, cleanup_proven=False)
    assert result[0] == report["returncode"] == 70 and cleanups == [True]
    assert report["cleanup_proven"] is False
    assert report["capture"]["returncode"] == 7
