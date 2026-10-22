"""Synthetic local transport faults through the real remote-run orchestration.

No SSH, SDK, inference, or production paths are used. Collection, tar creation,
download/extraction, status finalization and cleanup execute their normal code.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig


def _orchestrate(tmp_path, monkeypatch, *, collect_rc=0, pack_rc=0,
                 workload_rc=0, cleanup_partial=False, collect_raises=False,
                 report_status="passed", resume=False,
                 checkpoint_before_collection=False):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_set.json").write_text(json.dumps({"model_name": "resnet50"}))
    (suite / "benchmark_plan.json").write_text(json.dumps({
        "model_suite": {"primary": [{"id": "resnet50"}]},
        "runs": [{"id": "ort_cpu", "type": "ort", "provider": "cpu"}],
    }))
    remote_base = tmp_path / "remote" / "splitpoint_runs"
    instances = []

    class LocalTransport:
        def __init__(self, _host, **kwargs):
            self.registry = kwargs["remote_lease_registry"]
            self.commands = []
            self.downloads = []
            self.remote_suite = None
            instances.append(self)

        def resolve_path_read_only(self, *_args, **_kwargs):
            return str(remote_base)

        def run_read_only(self, *_args, **_kwargs):
            return 0, "SPLITPOINT_STORAGE_JSON=" + json.dumps({
                "read_only_mount": False, "permission_bits_allow": True,
                "os_access_allow": True, "free_bytes": 10**15,
                "free_inodes": 10**9,
            })

        def run(self, command, **kwargs):
            self.commands.append((command, kwargs))
            if "[preflight]" in command:
                return 0, "TEST_ONLY controlled runtime preflight leaf"
            if "shopt -s nullglob" in command:
                if collect_raises:
                    raise RuntimeError("TEST_ONLY collection transport exception")
                if collect_rc == 70:
                    self.registry.poison()
                    return 70, "TEST_ONLY exact cleanup not proven; lease session poisoned"
                if collect_rc and collect_rc != 124:
                    return collect_rc, "No space left on device" if collect_rc == 1 else "TEST_ONLY collection fault"
                # Real Bash collection first creates a useful partial tree;
                # the injected return code models an interrupted later copy.
                completed = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)
                assert completed.returncode == 0, completed.stderr
                return collect_rc, "TEST_ONLY collection timeout" if collect_rc else completed.stdout
            if "with tarfile.open" in command and pack_rc:
                return pack_rc, "TEST_ONLY packaging timeout"
            # Only exact temporary local paths can be mutated by normal scripts.
            assert str(tmp_path) in command, command
            completed = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)
            return completed.returncode, completed.stdout + completed.stderr

        def scp_upload(self, local_path, remote_path, **kwargs):
            assert kwargs.get("recursive") is True
            self.remote_suite = Path(remote_path) / Path(local_path).name
            shutil.copytree(local_path, self.remote_suite, dirs_exist_ok=True)
            return 0, "TEST_ONLY local upload"

        def run_streaming(self, _command, **_kwargs):
            assert self.remote_suite is not None
            report = self.remote_suite / "b001" / "results_TEST_ONLY" / "validation_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"status": report_status, "rows": [{"test_only": True}]}))
            if checkpoint_before_collection:
                local_report = (tmp_path / "working/Results/suite/1/TEST_ONLY_collection/results"
                                / "b001/results_TEST_ONLY/validation_report.json")
                local_report.parent.mkdir(parents=True)
                shutil.copy2(report, local_report)
            return workload_rc

        def scp_download(self, remote_path, local_path, recursive=False, **kwargs):
            self.downloads.append((str(remote_path), recursive, kwargs))
            if recursive:
                shutil.copytree(remote_path, Path(local_path) / Path(remote_path).name, dirs_exist_ok=True)
            else:
                shutil.copy2(remote_path, local_path)
            return 0, "TEST_ONLY local download"

    monkeypatch.setattr(remote_run, "SSHTransport", LocalTransport)
    monkeypatch.setattr(remote_run, "refresh_suite_harness", lambda *_a, **_k: {"changed": False})
    registry = RemoteProcessLeaseRegistry()
    logs = []
    result = remote_run.run_remote_benchmark(
        host=HostConfig(id="TEST_ONLY", label="TEST_ONLY", host="invalid.local", user="test"),
        benchmark_set_json=suite / "benchmark_set.json",
        local_working_dir=tmp_path / "working", run_id="TEST_ONLY_collection",
        args=remote_run.RemoteBenchmarkArgs(
            resume=resume, timeout_s=30, transfer_mode="direct", reuse_bundle=False,
            warmup=0, iters=1, cleanup_remote_on_partial=cleanup_partial,
        ),
        log=logs.append, progress=lambda *_args: None,
        cancel_event=threading.Event(), remote_process_registry=registry,
    )
    return result, instances[0], registry, logs


@pytest.mark.parametrize("cleanup_partial", [False, True])
def test_failed_collection_retains_remote_after_successful_download(tmp_path, monkeypatch, cleanup_partial):
    result, transport, _registry, _logs = _orchestrate(
        tmp_path, monkeypatch, collect_rc=124, cleanup_partial=cleanup_partial)
    assert result["remote_rc"] == 0  # capture success is not collection success
    assert result["status"] == "partial"
    assert "collection" in result["error"].lower()
    assert result["remote_cleanup"]["attempted"] is False
    assert result["remote_cleanup"]["policy"]["result_collection_rc"] == 124
    assert Path(result["remote_run_dir"]).is_dir()
    assert transport.downloads and not transport.downloads[0][1]
    assert (Path(result["local_run_dir"]) / "results/b001/results_TEST_ONLY/validation_report.json").is_file()


@pytest.mark.parametrize("report_status", ["passed", "failed"])
def test_complete_collection_keeps_normal_cleanup(tmp_path, monkeypatch, report_status):
    result, _transport, _registry, _logs = _orchestrate(tmp_path, monkeypatch, report_status=report_status)
    assert result["status"] == "ok"
    assert result["remote_cleanup"]["attempted"] is True
    assert result["remote_cleanup"]["ok"] is True
    assert not Path(result["remote_run_dir"]).exists()
    saved = Path(result["local_run_dir"]) / "results/b001/results_TEST_ONLY/validation_report.json"
    assert json.loads(saved.read_text())["status"] == report_status


def test_pack_timeout_keeps_normal_recursive_download_and_cleanup(tmp_path, monkeypatch):
    result, transport, _registry, _logs = _orchestrate(tmp_path, monkeypatch, pack_rc=124)
    assert result["status"] == "ok"
    assert transport.downloads[0][1] is True
    assert result["remote_cleanup"]["ok"] is True
    assert not Path(result["remote_run_dir"]).exists()
    collect = next(kwargs for cmd, kwargs in transport.commands if "shopt -s nullglob" in cmd)
    pack = next(kwargs for cmd, kwargs in transport.commands if "with tarfile.open" in cmd)
    assert collect["timeout_s"] == 60
    assert pack["timeout_s"] == 300


def test_collection_failure_preserves_first_workload_error(tmp_path, monkeypatch):
    result, _transport, _registry, _logs = _orchestrate(
        tmp_path, monkeypatch, collect_rc=124, workload_rc=7, cleanup_partial=True)
    assert result["remote_rc"] == 7
    assert result["status"] == "partial"
    assert "Remote benchmark failed (rc=7)" in result["error"]
    assert result["remote_cleanup"]["attempted"] is False
    assert Path(result["remote_run_dir"]).exists()


def test_collection_exception_retains_remote(tmp_path, monkeypatch):
    result, _transport, _registry, _logs = _orchestrate(tmp_path, monkeypatch, collect_raises=True, cleanup_partial=True)
    assert result["remote_rc"] == 0
    assert result["status"] == "failed"
    assert result["remote_cleanup"]["attempted"] is False
    assert Path(result["remote_run_dir"]).exists()


def test_collection_exception_with_checkpoint_never_advises_workload_rerun(tmp_path, monkeypatch):
    result, _transport, _registry, _logs = _orchestrate(
        tmp_path, monkeypatch, collect_raises=True, cleanup_partial=True,
        resume=True, checkpoint_before_collection=True)
    assert result["remote_rc"] == 0
    assert result["status"] == "partial"
    assert result["remote_cleanup"]["attempted"] is False
    assert result["remote_cleanup"]["policy"]["result_collection_rc"] is None
    assert Path(result["remote_run_dir"]).exists()
    assert "TEST_ONLY collection transport exception" in result["error"]
    assert "rerun" not in result["error"].lower()


@pytest.mark.parametrize("collect_rc", [70, 1])
def test_collection_terminal_cleanup_and_storage_failures_still_suppress_transfer(tmp_path, monkeypatch, collect_rc):
    result, transport, registry, _logs = _orchestrate(tmp_path, monkeypatch, collect_rc=collect_rc)
    assert result["status"] == "failed"
    assert result["terminal_remote_failure"] is True
    assert not transport.downloads
    assert not any("with tarfile.open" in cmd for cmd, _ in transport.commands)
    assert result["remote_cleanup"]["attempted"] is False
    assert Path(result["remote_run_dir"]).exists()
    if collect_rc == 70:
        assert result["remote_rc"] == 70
        assert registry.cancelled is True


@pytest.mark.parametrize("fail_copy", [False, True])
def test_collect_script_attempts_later_sources_and_reports_real_copy_failure(tmp_path, fail_copy):
    suite = tmp_path / "suite"
    later = suite / "b001/results_TEST_ONLY/later.txt"
    later.parent.mkdir(parents=True)
    later.write_text("TEST_ONLY later payload")
    early = suite / "benchmark_plan.json"
    early.write_text('{"TEST_ONLY": true}')
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "cp").write_text(
        "#!/bin/bash\n"
        f"if [ \"${{2:-}}\" = {shlex.quote(str(early))} ] && [ {int(fail_copy)} = 1 ]; then exit 23; fi\n"
        "exec /bin/cp \"$@\"\n")
    (shim / "cp").chmod(0o755)
    out = tmp_path / "collected"
    completed = subprocess.run(
        ["/bin/bash", "-c", remote_run._remote_result_collect_script(remote_results_dir=str(out), remote_suite_dir=str(suite))],
        env={**os.environ, "PATH": str(shim) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True)
    assert (out / "b001/results_TEST_ONLY/later.txt").read_text() == "TEST_ONLY later payload"
    assert completed.returncode == (1 if fail_copy else 0)
    assert (out / "benchmark_plan.json").exists() is not fail_copy


def test_collect_script_optional_missing_sources_are_success(tmp_path):
    suite = tmp_path / "empty_suite"
    suite.mkdir()
    out = tmp_path / "collected"
    completed = subprocess.run(["/bin/bash", "-c", remote_run._remote_result_collect_script(
        remote_results_dir=str(out), remote_suite_dir=str(suite))], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert list(out.iterdir()) == []
