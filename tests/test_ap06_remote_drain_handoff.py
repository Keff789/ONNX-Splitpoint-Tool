"""Real local lease processes retain drain proof across launcher/control exit.

The controller driver pauses the unchanged cleanup program at its final proof
read. This controls scheduling only: identities, signals and proof decisions
are executed by the actual product launcher, guardian and cleanup program.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseOperation, RemoteProcessLeaseScope


def _wait(path, *, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.005)


def test_launcher_completion_keeps_exact_drain_token_for_a_later_control_reader(tmp_path):
    scope = RemoteProcessLeaseScope("drain-test", "normal-completion", str(tmp_path / "leases"))
    operation = scope.operation(label="short-leaf", control_argv_prefix=["bash", "-lc"])
    payload = shlex.join([sys.executable, "-B", "-c", "print('OWNED_LEAF_COMPLETE', flush=True)"])
    completed = subprocess.run(shlex.split(operation.wrap_remote_command(payload)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
    assert completed.returncode == 0, completed.stdout
    assert "OWNED_LEAF_COMPLETE" in completed.stdout
    operation_root = Path(scope.lease_root) / scope.run_sha256 / scope.session_sha256
    assert not (operation_root / (operation.operation_id + ".lease.json")).exists()
    proof = operation_root / (operation.operation_id + ".drained")
    assert proof.is_file(), "launcher must not erase proof before a concurrent controller has read it"
    assert proof.read_text(encoding="ascii").strip() == operation.token


_CONTROL_DRIVER = r'''
import pathlib, shlex, sys, time
ready, release, target_statement, command = sys.argv[1:]
argv = shlex.split(command)
source = argv[2]
target = next(i for i, line in enumerate(source.splitlines(), 1)
              if line == target_statement)
paused = False
def trace(frame, event, arg):
    global paused
    if not paused and event == "line" and frame.f_code.co_filename == "<actual-cleanup>" and frame.f_lineno == target:
        paused = True
        pathlib.Path(ready).write_text(target_statement)
        deadline = time.monotonic() + 10
        while not pathlib.Path(release).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("test did not release owned cleanup reader")
            time.sleep(.005)
    return trace
sys.argv = ["-c", *argv[3:]]
sys.settrace(trace)
exec(compile(source, "<actual-cleanup>", "exec"), {"__name__": "__main__"})
'''


def test_concurrent_cleanup_reads_drain_proof_after_owned_launcher_has_exited(tmp_path):
    driver = tmp_path / "control_driver.py"
    driver.write_text(_CONTROL_DRIVER, encoding="utf-8")
    ready, release = tmp_path / "control.ready", tmp_path / "control.release"
    leaf_ready = tmp_path / "leaf.ready"
    scope = RemoteProcessLeaseScope("drain-test", "concurrent-completion", str(tmp_path / "leases"))
    operation = scope.operation(label="cancel-owned-leaf",
        control_argv_prefix=[sys.executable, "-B", str(driver), str(ready), str(release),
                             "remaining = alive_count()"])
    code = ("import pathlib,signal,sys,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)")
    payload = shlex.join([sys.executable, "-B", "-c", code, str(leaf_ready)])
    registry = ProcessTreeRegistry()
    launcher = subprocess.Popen(shlex.split(operation.wrap_remote_command(payload)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    registry.register(launcher)
    try:
        _wait(leaf_ready)
        operation_root = Path(scope.lease_root) / scope.run_sha256 / scope.session_sha256
        lease = operation_root / (operation.operation_id + ".lease.json")
        proof = operation_root / (operation.operation_id + ".drained")
        published = json.loads(lease.read_text())
        assert published["token"] == operation.token
        with ThreadPoolExecutor(max_workers=1) as threads:
            cancellation = threads.submit(operation.cancel_remote,
                grace_s=0.05, launch_wait_s=0.01, timeout_s=8)
            try:
                _wait(ready)
                output, _ = launcher.communicate(timeout=3)
                assert launcher.returncode is not None, output
                assert not lease.exists(), "launcher must have completed its own lease cleanup before control resumes"
                assert proof.is_file(), "launcher erased the exact controller's still-needed drain proof"
                assert proof.read_text(encoding="ascii").strip() == operation.token
                release.touch()
                report = cancellation.result(timeout=3)
                assert report["ok"] is True, report
                assert report["returncode"] == 0, report
                assert "remaining=0 drained=1" in report["output"], report
            finally:
                release.touch()
    finally:
        release.touch()
        if launcher.poll() is None:
            registry.terminate_registered(launcher, grace_s=0.1)
        registry.unregister(launcher)
        registry.assert_quiescent()


def test_two_exact_cleanup_controllers_can_both_consume_the_same_terminal_proof(tmp_path):
    driver = tmp_path / "control_driver.py"
    driver.write_text(_CONTROL_DRIVER, encoding="utf-8")
    ready = [tmp_path / (name + ".ready") for name in ("first", "second")]
    release = [tmp_path / (name + ".release") for name in ("first", "second")]
    scope = RemoteProcessLeaseScope("drain-test", "parallel-controls", str(tmp_path / "leases"))
    prefix = lambda index: [sys.executable, "-B", str(driver), str(ready[index]), str(release[index]),
                            "root_identity = identity(root_pid)"]
    first = scope.operation(label="parallel-owned-leaf", control_argv_prefix=prefix(0))
    second = RemoteProcessLeaseOperation(scope=scope, operation_id=first.operation_id,
        token=first.token, control_argv_prefix=prefix(1))
    leaf_ready = tmp_path / "leaf.ready"
    code = ("import pathlib,signal,sys,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)")
    payload = shlex.join([sys.executable, "-B", "-c", code, str(leaf_ready)])
    registry = ProcessTreeRegistry()
    launcher = subprocess.Popen(shlex.split(first.wrap_remote_command(payload)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    registry.register(launcher)
    try:
        _wait(leaf_ready)
        with ThreadPoolExecutor(max_workers=2) as threads:
            controls = [threads.submit(operation.cancel_remote,
                grace_s=0.05, launch_wait_s=0.01, timeout_s=8) for operation in (first, second)]
            try:
                for marker in ready:
                    _wait(marker)
                # Both controllers have read the same exact lease. The second
                # may be delayed until the first and launcher have finished.
                release[0].touch()
                first_report = controls[0].result(timeout=4)
                assert first_report["ok"] is True, first_report
                launcher.communicate(timeout=3)
                operation_root = Path(scope.lease_root) / scope.run_sha256 / scope.session_sha256
                assert not (operation_root / (first.operation_id + ".lease.json")).exists()
                proof = operation_root / (first.operation_id + ".drained")
                assert proof.read_text(encoding="ascii").strip() == first.token
                release[1].touch()
                second_report = controls[1].result(timeout=3)
                assert second_report["ok"] is True, second_report
                assert second_report["returncode"] == 0, second_report
                assert "already_exited_drained" in second_report["output"], second_report
                assert proof.read_text(encoding="ascii").strip() == first.token
            finally:
                for marker in release:
                    marker.touch()
    finally:
        for marker in release:
            marker.touch()
        if launcher.poll() is None:
            registry.terminate_registered(launcher, grace_s=0.1)
        registry.unregister(launcher)
        registry.assert_quiescent()
