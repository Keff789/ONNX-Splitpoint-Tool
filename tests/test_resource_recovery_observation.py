"""Owner-only recovery checks: controlled local children and transport doubles.

No test opens SSH, starts a collector, or touches a production resource lock.
"""
from contextlib import redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.remote import resource_recovery_observation as observation


def _idle():
    return {"ok": True, "read_only": True, "status": "no_known_competing_work_seen",
            "recorded_owners": [], "processes": [], "live_product_leases": [],
            "visibility_errors": [], "exclusive_start_permission": False,
            "cleanup_proven": False}


def _ticks(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    return text[text.rfind(")") + 2:].split()[19]


@pytest.fixture
def child():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_live_exact_controller_owner_blocks_before_transport(child, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("active local owner must block before remote observation")

    monkeypatch.setattr(observation, "SSHTransport", forbidden)
    result = observation.observe_resource_owners(
        {}, local_owners=[{"pid": child.pid, "start_time_ticks": _ticks(child.pid)}],
        diagnostics_dir=tmp_path)
    assert result["ok"] is False
    assert result["remote"]["status"] == "not_observed"
    assert result["controller"]["recorded_owners"][0]["state"] == "active"


def test_ended_child_is_only_absence_and_never_cleanup_proof(child, tmp_path):
    ticks = _ticks(child.pid)
    child.terminate()
    child.wait(timeout=5)
    result = observation._observe_processes(
        [{"pid": child.pid, "start_time_ticks": ticks}], lease_parent=tmp_path)
    assert result["recorded_owners"][0]["state"] == "absent"
    assert result["cleanup_proven"] is False
    assert result["exclusive_start_permission"] is False


def _proc_fixture(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "1").mkdir()
    (proc / "self/stat").write_text(f"{os.getpid()} (observer) S 0 0\n")
    (proc / "self/mountinfo").write_text(
        f"1 0 0:1 / {proc} rw,nosuid,nodev,noexec - proc proc rw\n")
    (proc / "1/cmdline").write_bytes(b"init\0")
    leases = tmp_path / "leases"
    leases.mkdir()
    return proc, leases


def _fake_process(proc, pid, *, ticks="501", program="sleep", extra=b""):
    directory = proc / str(pid)
    directory.mkdir()
    # Fields begin at proc stat field 3, with starttime at index 19.
    fields = ["S"] + ["0"] * 18 + [ticks]
    (directory / "stat").write_text(f"{pid} (fixture) " + " ".join(fields))
    (directory / "cmdline").write_bytes(program.encode() + b"\0" + extra)
    return directory


@pytest.mark.parametrize("program", ["native_trt_from_benchmarkset.py", "run_evaluation_workflow.py",
                                     "analyse_and_split_gui.py"])
def test_known_foreign_program_blocks_without_exposing_arguments(tmp_path, monkeypatch, program):
    proc, leases = _proc_fixture(tmp_path)
    _fake_process(proc, 902, program="/some/path/" + program,
                  extra=b"--password=do-not-report\0")
    result = observation._observe_processes([], proc_root=proc, lease_parent=leases)
    assert result["status"] == "busy"
    assert result["processes"][0]["matched_programs"] == [program]
    assert "do-not-report" not in json.dumps(result)


def test_live_product_lease_blocks_even_for_unknown_program(tmp_path, monkeypatch):
    proc, leases = _proc_fixture(tmp_path)
    _fake_process(proc, 902)
    root = leases / "onnx_splitpoint-process-leases-1000"
    root.mkdir()
    (root / "owned.lease.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/remote-process-lease", "pid": 902,
        "pgid": 902, "operation_id": "owned", "token": "existing-token",
        "start_time_ticks": "501", "command": "never report this command"}))
    result = observation._observe_processes([], proc_root=proc, lease_parent=leases)
    assert result["ok"] is False
    assert result["live_product_leases"][0]["pid"] == 902
    assert "never report" not in json.dumps(result)


@pytest.mark.parametrize("problem", ["namespace", "hidden", "owner_stat", "cmdline", "lease"])
def test_incomplete_visibility_fails_closed(problem, tmp_path, monkeypatch):
    proc, leases = _proc_fixture(tmp_path)
    process = _fake_process(proc, 902)
    if problem == "namespace":
        (proc / "self/stat").write_text(f"{os.getpid() + 1} (other namespace) S 0 0\n")
    elif problem == "hidden":
        path = proc / "self/mountinfo"
        path.write_text(path.read_text().replace("- proc proc rw", "- proc proc rw,hidepid=2"))
    elif problem == "lease":
        root = leases / "onnx_splitpoint-process-leases-1000"
        root.mkdir()
        (root / "owned.lease.json").write_text("not json")
    else:
        target = process / ("stat" if problem == "owner_stat" else "cmdline")
        original = Path.open

        def denied(path, *args, **kwargs):
            if path == target:
                raise PermissionError("denied")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    result = observation._observe_processes(
        [{"pid": 902, "start_time_ticks": "500"}], proc_root=proc, lease_parent=leases)
    assert result["ok"] is False
    assert result["visibility_errors"]


@pytest.mark.parametrize("drained,reused", [
    (None, False), ("wrong-token", False), ("existing-token", False), ("existing-token", True),
])
def test_exited_lease_root_requires_existing_exact_guardian_drain(tmp_path, drained, reused):
    proc, leases = _proc_fixture(tmp_path)
    root = leases / "onnx_splitpoint-process-leases-1000"
    root.mkdir()
    (root / "owned.lease.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/remote-process-lease", "pid": 902,
        "pgid": 902, "operation_id": "owned", "token": "existing-token",
        "start_time_ticks": "501"}))
    if drained is not None:
        (root / "owned.drained").write_text(drained)
    if reused:
        _fake_process(proc, 902, ticks="502")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    result = observation._observe_processes([], proc_root=proc, lease_parent=leases)
    assert result["ok"] is (drained == "existing-token" and not reused)
    if drained != "existing-token":
        assert any("lease_root_exited_without_drain" in error for error in result["visibility_errors"])
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def _transport_double(monkeypatch, result):
    seen = []
    monkeypatch.setattr(observation, "host_config_from_setup", lambda setup: setup["host"])

    class Transport:
        def __init__(self, host):
            assert host == "exact-registry-host"

        def run_read_only(self, command, *, timeout, env):
            seen.append((command, timeout, env, self.diagnostics_dir))
            assert "python3 -B" in command
            assert "_observe_processes([])" in command
            return result

    monkeypatch.setattr(observation, "SSHTransport", Transport)
    return seen


def test_normal_transport_receives_same_read_only_probe_and_safe_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(observation, "_observe_processes", lambda _owners: _idle())
    # The transport contract is isolated here; the generated body is exercised
    # against a complete proc/lease fixture in the separate test below.
    source = "def _observe_processes(local_owners):\n    return {}\n"
    monkeypatch.setattr(observation.inspect, "getsource", lambda _function: source)
    seen = _transport_double(monkeypatch, (0, "RESOURCE_OWNERS=" + json.dumps(_idle())))
    result = observation.observe_resource_owners(
        {"host": "exact-registry-host"}, local_owners=[{"pid": 9, "start_time_ticks": "10"}],
        diagnostics_dir=tmp_path / "diagnostics")
    assert result["ok"] is True
    assert len(seen) == 1
    assert seen[0][1:] == (30, {"PYTHONDONTWRITEBYTECODE": "1"}, tmp_path / "diagnostics")
    assert result["controller"]["cleanup_proven"] is False


@pytest.mark.parametrize("reply", [
    (124, "timeout with sensitive unexpected output"),
    (0, "no observation"),
    (0, "RESOURCE_OWNERS={}"),
    (0, "RESOURCE_OWNERS=" + json.dumps(dict(_idle(), visibility_errors=["denied"]))),
    (0, "RESOURCE_OWNERS=" + json.dumps(dict(_idle(), live_product_leases=[{"pid": 3}]))),
])
def test_failed_or_busy_remote_observation_blocks(reply, tmp_path, monkeypatch):
    monkeypatch.setattr(observation, "_observe_processes", lambda _owners: _idle())
    monkeypatch.setattr(observation.inspect, "getsource", lambda _function: "def _observe_processes(x): pass\n")
    _transport_double(monkeypatch, reply)
    result = observation.observe_resource_owners(
        {"host": "exact-registry-host"}, local_owners=[{"pid": 9, "start_time_ticks": "10"}],
        diagnostics_dir=tmp_path)
    assert result["ok"] is False
    assert "sensitive" not in json.dumps(result)


def test_missing_recorded_owners_never_reaches_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(observation, "_observe_processes", lambda _owners: _idle())
    monkeypatch.setattr(observation, "SSHTransport", lambda *_: pytest.fail("unexpected SSH"))
    result = observation.observe_resource_owners({}, local_owners=[], diagnostics_dir=tmp_path)
    assert result["ok"] is False
    assert "recorded_owners_required" in result["controller"]["visibility_errors"]


def test_generated_remote_body_is_self_contained_and_observes_exact_fixture(tmp_path, monkeypatch):
    proc, leases = _proc_fixture(tmp_path)
    source = inspect.getsource(observation._observe_processes)
    # Exact old identity is no longer present; the same PID now denotes another
    # process. This observation still does not prove cleanup of old descendants.
    _fake_process(proc, 902)
    local = observation._observe_processes(
        [{"pid": 902, "start_time_ticks": "500"}], proc_root=proc, lease_parent=leases)
    assert local["ok"] is True
    assert local["recorded_owners"][0]["state"] == "pid_reused"
    monkeypatch.setattr(observation, "_observe_processes", lambda _owners: local)
    monkeypatch.setattr(observation.inspect, "getsource", lambda _function: source)
    monkeypatch.setattr(observation, "host_config_from_setup", lambda _setup: object())

    class LocalTransport:
        def __init__(self, _host):
            pass

        def run_read_only(self, command, **_kwargs):
            body = command.split("\n", 1)[1].rsplit("\n", 1)[0]
            body = body.replace("_observe_processes([])",
                                f"_observe_processes([], proc_root={str(proc)!r}, lease_parent={str(leases)!r})")
            output = io.StringIO()
            with redirect_stdout(output):
                exec(compile(body, "<local-test-remote-probe>", "exec"), {})
            return 0, output.getvalue()

    monkeypatch.setattr(observation, "SSHTransport", LocalTransport)
    result = observation.observe_resource_owners(
        {}, local_owners=[{"pid": 902, "start_time_ticks": "500"}], diagnostics_dir=tmp_path)
    assert result["ok"] is True
    assert result["remote"]["cleanup_proven"] is False
    _fake_process(proc, 903, program="/fixture/urecs-data-collector")
    blocked = observation.observe_resource_owners(
        {}, local_owners=[{"pid": 902, "start_time_ticks": "500"}], diagnostics_dir=tmp_path)
    assert blocked["ok"] is False
    assert blocked["remote"]["processes"][0]["pid"] == 903
