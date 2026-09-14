from __future__ import annotations

"""Short P0.1 regressions for run ownership, cancellation, and Resume drift."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import onnx_splitpoint_tool.process_control as process_control
from onnx_splitpoint_tool.energy.collector import _run_one as run_energy_command
from onnx_splitpoint_tool.energy.collector import check_energy_tools
from onnx_splitpoint_tool.energy.config import EnergyDefaults
from onnx_splitpoint_tool.native_progress import run_streaming
from onnx_splitpoint_tool.process_control import (
    ProcessIdentity,
    ProcessTreeCancellationError,
    ProcessTreeCapabilityError,
    ProcessTreeRegistry,
    bind_process_registry,
    current_process_registry,
    require_nested_session_cancellation,
    terminate_process_tree,
)
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport
from onnx_splitpoint_tool.workflow.run_control import (
    EvaluationRunLock,
    WorkflowRunCancelledError,
    WorkflowRunCleanupQuarantineError,
    WorkflowRunTargetError,
    build_resume_contract,
)


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes() if path.is_file() else None
        )
        for path in sorted(root.rglob("*"))
    }


def _wait_for(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


def _procfs_matches_pid_namespace() -> bool:
    if os.name != "posix":
        return False
    try:
        proc_pid = int(
            Path("/proc/self/stat").read_text(encoding="utf-8").split("(", 1)[0]
        )
    except Exception:
        return False
    return proc_pid == os.getpid()


def _new_fixture_workflow_runner(options: object):
    """Construct a Runner without hashing the whole source tree.

    These tests exercise ownership, Resume, and cancellation contracts.  The
    package-provenance snapshot has its own regression tests and can take many
    seconds on a cold or network-backed checkout, so it must not dominate a
    short process-control fixture.
    """

    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner

    fixture_snapshot = {
        "schema": "onnx-splitpoint/p01-test-build-identity",
        "schema_version": 1,
        "package_version": "p01-fixture",
        "build_id": "p01-fixture",
        "critical_module_set_complete": True,
    }
    with mock.patch(
        "onnx_splitpoint_tool.workflow.runner.package_build_snapshot",
        return_value=fixture_snapshot,
    ):
        return EvaluationWorkflowRunner(options)  # type: ignore[arg-type]


class SingleWriterCancellationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.name == "posix",
        "the executable Energy help fixture requires POSIX",
    )
    def test_energy_tool_help_probe_is_registry_owned_and_pre_cancelled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / "energy-help-fixture"
            helper.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "time.sleep(0.15)\n"
                "print('fixture help')\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)
            defaults = EnergyDefaults(
                collector_binary=str(helper),
                power_calculations_binary=str(helper),
            )

            registry = ProcessTreeRegistry()
            with mock.patch.object(
                registry, "register", wraps=registry.register
            ) as register, mock.patch.object(
                registry, "unregister", wraps=registry.unregister
            ) as unregister:
                with bind_process_registry(registry):
                    live = check_energy_tools(defaults)
            self.assertEqual(live["collector_help_rc"], 0)
            self.assertEqual(live["power_calculations_help_rc"], 0)
            self.assertIn("fixture help", live["collector_help_stdout_tail"])
            self.assertEqual(register.call_count, 2)
            self.assertEqual(unregister.call_count, 2)
            registry.assert_quiescent()

            registry.terminate_all(grace_s=0.05)
            with bind_process_registry(registry):
                with mock.patch(
                    "onnx_splitpoint_tool.energy.collector.subprocess.Popen"
                ) as popen:
                    cancelled = check_energy_tools(defaults)
            popen.assert_not_called()
            self.assertEqual(cancelled["collector_help_rc"], 130)
            self.assertEqual(cancelled["power_calculations_help_rc"], 130)

            # A collector timeout must drain the cross-process remote journal
            # before its local collector/broker/SSH tree is terminated.
            with mock.patch(
                "onnx_splitpoint_tool.energy.collector."
                "cancel_journaled_remote_processes_from_environment"
            ) as remote_barrier:
                timed = run_energy_command(
                    [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                    cwd=None,
                    stdout_path=Path(temporary) / "timeout.stdout",
                    stderr_path=Path(temporary) / "timeout.stderr",
                    timeout=0.05,
                )
            self.assertTrue(timed.get("timeout"))
            remote_barrier.assert_called_once()

    def test_cross_process_lock_is_fail_fast_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary)
            run_dir = out_root / "fixture_run"
            run_dir.mkdir()
            (run_dir / "sentinel.txt").write_text(
                "immutable\n", encoding="utf-8"
            )
            before = _tree_snapshot(run_dir)
            holder = EvaluationRunLock(
                out_root=out_root,
                run_dir=run_dir,
                owner={"session_id": "holder"},
            ).acquire()
            child_code = """
import sys
from pathlib import Path
from onnx_splitpoint_tool.workflow.run_control import (
    EvaluationRunLock, WorkflowRunLockedError,
)
lock = EvaluationRunLock(
    out_root=Path(sys.argv[1]),
    run_dir=Path(sys.argv[2]),
    owner={"session_id": "contender"},
)
try:
    lock.acquire()
except WorkflowRunLockedError as exc:
    assert exc.error_code == "run_already_active"
    assert exc.owner.get("session_id") == "holder"
    print("blocked")
else:
    lock.release()
    raise SystemExit("second writer unexpectedly acquired the run")
"""
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT) + (
                os.pathsep + env["PYTHONPATH"]
                if env.get("PYTHONPATH")
                else ""
            )
            try:
                started = time.monotonic()
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        child_code,
                        str(out_root),
                        str(run_dir),
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=2.0,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), "blocked")
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(_tree_snapshot(run_dir), before)
            finally:
                holder.release()

            with EvaluationRunLock(
                out_root=out_root,
                run_dir=run_dir,
                owner={"session_id": "replacement"},
            ):
                self.assertEqual(_tree_snapshot(run_dir), before)

            class _ReleaseWriteFailure:
                def __init__(self, wrapped: object) -> None:
                    self._wrapped = wrapped

                def truncate(self, *_args: object, **_kwargs: object) -> None:
                    raise OSError("synthetic release metadata failure")

                def __getattr__(self, name: str) -> object:
                    return getattr(self._wrapped, name)

            flaky = EvaluationRunLock(
                out_root=out_root,
                run_dir=run_dir,
                owner={"session_id": "flaky-release"},
            ).acquire()
            flaky._fh = _ReleaseWriteFailure(flaky._fh)
            flaky.release()
            after_flaky = EvaluationRunLock(
                out_root=out_root,
                run_dir=run_dir,
                owner={"session_id": "after-flaky-release"},
            )
            with self.assertRaises(WorkflowRunCleanupQuarantineError):
                after_flaky.acquire()
            self.assertTrue(after_flaky.quarantine_path.is_file())

    def test_sticky_cancel_status_and_safe_procfs_fallback(self) -> None:
        # Namespace capability detection is queried from hot cancellation
        # loops, but the immutable per-process result must hit procfs once.
        process_control._procfs_matches_active_pid_namespace.cache_clear()
        try:
            with mock.patch.object(
                Path,
                "read_text",
                return_value=f"{os.getpid()} (python) R",
            ) as read_stat:
                process_control._procfs_matches_active_pid_namespace()
                process_control._procfs_matches_active_pid_namespace()
            if os.name == "posix":
                self.assertEqual(read_stat.call_count, 1)
        finally:
            process_control._procfs_matches_active_pid_namespace.cache_clear()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "late-register-must-not-write.txt"
            registry = ProcessTreeRegistry()
            registry.terminate_all(grace_s=0.05)
            late = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import pathlib,sys,time;time.sleep(0.3);"
                        "pathlib.Path(sys.argv[1]).write_text('late');"
                        "time.sleep(10)"
                    ),
                    str(marker),
                ],
                start_new_session=(os.name == "posix"),
            )
            registry.register(late, label="registered-after-cancel")
            self.assertIsNotNone(late.poll())
            time.sleep(0.35)
            self.assertFalse(marker.exists())

            cancel = threading.Event()
            live_registry = ProcessTreeRegistry()
            outcome: dict[str, object] = {}

            def _run() -> None:
                outcome["result"] = run_streaming(
                    [sys.executable, "-B", "-c", "import time;time.sleep(10)"],
                    cancel_event=cancel,
                    process_registry=live_registry,
                    heartbeat_s=0,
                    line_callback=lambda _line: None,
                )

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            deadline = time.monotonic() + 1.0
            while not live_registry.active_pids() and time.monotonic() < deadline:
                time.sleep(0.01)
            cancel.set()
            live_registry.terminate_all(grace_s=0.1)
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(getattr(outcome.get("result"), "returncode", None), 130)

            unrelated = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time;time.sleep(10)"],
                start_new_session=(os.name == "posix"),
            )
            owned = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time;time.sleep(10)"],
                start_new_session=(os.name == "posix"),
            )
            try:
                with mock.patch(
                    "onnx_splitpoint_tool.process_control."
                    "_procfs_matches_active_pid_namespace",
                    return_value=False,
                ):
                    report = terminate_process_tree(owned, grace_s=0.05)
                    with self.assertRaises(ProcessTreeCapabilityError):
                        require_nested_session_cancellation()
                self.assertEqual(report["remaining_process_count"], 0)
                self.assertFalse(report["descendant_discovery_available"])
                self.assertEqual(
                    report["cancellation_assurance"],
                    "root_process_group_only",
                )
                self.assertIsNone(unrelated.poll())
                if os.name == "posix":
                    unrelated_identity = ProcessIdentity(
                        pid=int(unrelated.pid),
                        pgid=int(os.getpgid(unrelated.pid)),
                        start_time_ticks="",
                        depth=0,
                    )
                    with mock.patch(
                        "onnx_splitpoint_tool.process_control."
                        "_procfs_matches_active_pid_namespace",
                        return_value=False,
                    ), mock.patch.object(
                        Path,
                        "read_text",
                        side_effect=AssertionError(
                            "numeric procfs path must not be read"
                        ),
                    ):
                        self.assertTrue(
                            process_control._identity_alive(unrelated_identity)
                        )
            finally:
                terminate_process_tree(owned, grace_s=0.05)
                terminate_process_tree(unrelated, grace_s=0.05)

            transport = SSHTransport(
                HostConfig(id="fixture", label="fixture", host="invalid")
            )
            callback_marker = root / "ssh-callback-must-not-write.txt"
            local_command = [
                sys.executable,
                "-B",
                "-c",
                (
                    "import pathlib,sys,time;print('ready',flush=True);"
                    "time.sleep(.3);pathlib.Path(sys.argv[1]).write_text('late');"
                    "time.sleep(10)"
                ),
                str(callback_marker),
            ]
            callback_started = time.monotonic()
            with mock.patch.object(
                transport, "_ssh_cmd", return_value=local_command
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture callback"):
                    transport.run_streaming(
                        "ignored",
                        on_line=lambda _line: (_ for _ in ()).throw(
                            RuntimeError("fixture callback")
                        ),
                        timeout_s=10,
                    )
            self.assertLess(time.monotonic() - callback_started, 1.5)
            time.sleep(0.35)
            self.assertFalse(callback_marker.exists())

    def test_registry_survivor_token_and_pre_cancel_contracts(self) -> None:
        registry = ProcessTreeRegistry()
        registry.terminate_all(grace_s=0.05)
        survivor = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time;time.sleep(10)"],
            start_new_session=(os.name == "posix"),
        )
        synthetic_survivor = {
            "root_pid": survivor.pid,
            "remaining_process_count": 1,
            "termination_error": "synthetic survivor",
        }
        try:
            with mock.patch(
                "onnx_splitpoint_tool.process_control.terminate_process_tree",
                return_value=synthetic_survivor,
            ):
                with self.assertRaises(ProcessTreeCancellationError):
                    registry.register(survivor, label="synthetic-survivor")
                # A helper's finally/unregister must not erase a process tree
                # whose bounded cancellation still reports a survivor.
                registry.unregister(survivor)
            self.assertTrue(registry.has_owned_processes())
            self.assertEqual(registry.remaining_owned_count(), 1)
            with self.assertRaises(ProcessTreeCancellationError):
                registry.assert_quiescent()
        finally:
            registry.terminate_all(grace_s=0.05)
        registry.assert_quiescent()

        class _FakePopen:
            pid = 987654

            def __init__(self) -> None:
                self.returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

        token_registry = ProcessTreeRegistry()
        first = _FakePopen()
        second = _FakePopen()
        token_registry.register(first)  # type: ignore[arg-type]
        token_registry.register(second)  # type: ignore[arg-type]
        self.assertEqual(token_registry.remaining_owned_count(), 2)
        first.returncode = 0
        token_registry.unregister(first)  # type: ignore[arg-type]
        self.assertEqual(token_registry.remaining_owned_count(), 1)
        second.returncode = 0
        token_registry.unregister(second)  # type: ignore[arg-type]
        token_registry.assert_quiescent()

        cancel = threading.Event()
        cancel.set()
        with mock.patch(
            "onnx_splitpoint_tool.native_progress.subprocess.Popen"
        ) as native_popen:
            native_result = run_streaming(
                [sys.executable, "-B", "-c", "raise SystemExit(99)"],
                cancel_event=cancel,
                heartbeat_s=0,
                line_callback=lambda _line: None,
            )
        native_popen.assert_not_called()
        self.assertEqual(native_result.returncode, 130)

        transport = SSHTransport(
            HostConfig(id="fixture", label="fixture", host="invalid")
        )
        with mock.patch(
            "onnx_splitpoint_tool.remote.ssh_transport.subprocess.Popen"
        ) as ssh_popen:
            ssh_result = transport.run_streaming(
                "ignored",
                on_line=lambda _line: None,
                cancel_event=cancel,
            )
        ssh_popen.assert_not_called()
        self.assertEqual(ssh_result, 130)

        # Helpers without an explicit cancel event inherit the workflow's
        # sticky registry and must obey the pre-cancel principle.
        inherited_cancel = ProcessTreeRegistry()
        inherited_cancel.terminate_all(grace_s=0.05)
        with bind_process_registry(inherited_cancel):
            with mock.patch(
                "onnx_splitpoint_tool.native_progress.subprocess.Popen"
            ) as inherited_native_popen:
                inherited_native = run_streaming(
                    [sys.executable, "-B", "-c", "raise SystemExit(99)"],
                    heartbeat_s=0,
                    line_callback=lambda _line: None,
                )
            inherited_native_popen.assert_not_called()
            self.assertEqual(inherited_native.returncode, 130)

            with mock.patch(
                "onnx_splitpoint_tool.remote.ssh_transport.subprocess.Popen"
            ) as inherited_ssh_popen:
                inherited_ssh = transport.run_streaming(
                    "ignored",
                    on_line=lambda _line: None,
                )
            inherited_ssh_popen.assert_not_called()
            self.assertEqual(inherited_ssh, 130)

            with tempfile.TemporaryDirectory() as temporary:
                energy_root = Path(temporary)
                with mock.patch(
                    "onnx_splitpoint_tool.energy.collector.subprocess.Popen"
                ) as inherited_energy_popen:
                    inherited_energy = run_energy_command(
                        [sys.executable, "-B", "-c", "raise SystemExit(99)"],
                        cwd=None,
                        stdout_path=energy_root / "stdout.txt",
                        stderr_path=energy_root / "stderr.txt",
                    )
                inherited_energy_popen.assert_not_called()
                self.assertEqual(inherited_energy["rc"], 130)

            from onnx_splitpoint_tool.hailo_backend import (
                _run_owned_subprocess,
            )

            with mock.patch(
                "onnx_splitpoint_tool.hailo_backend.subprocess.Popen"
            ) as inherited_hailo_popen:
                inherited_hailo = _run_owned_subprocess(
                    [sys.executable, "-B", "-c", "raise SystemExit(99)"],
                    text=True,
                )
            inherited_hailo_popen.assert_not_called()
            self.assertEqual(inherited_hailo.returncode, 130)

        from onnx_splitpoint_tool.build_scheduler import (
            BuildScheduler,
            BuildTaskSpec,
        )

        context_registry = ProcessTreeRegistry()
        scheduler = BuildScheduler(max_workers=1)
        try:
            with bind_process_registry(context_registry):
                future = scheduler.submit(
                    BuildTaskSpec(name="context", family="fixture"),
                    current_process_registry,
                )
            self.assertIs(future.result(timeout=1.0), context_registry)
        finally:
            scheduler.shutdown(True)

    @unittest.skipUnless(
        _procfs_matches_pid_namespace(),
        "process-tree ownership requires /proc in the active PID namespace",
    )
    def test_nested_session_cancel_does_not_touch_independent_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "grandchild.pid"
            delayed_marker = root / "must_not_exist.txt"
            grandchild_code = (
                "import pathlib,sys,time;"
                "time.sleep(0.65);"
                "pathlib.Path(sys.argv[1]).write_text('late',encoding='utf-8');"
                "time.sleep(30)"
            )
            parent_code = """
import pathlib,subprocess,sys
child = subprocess.Popen(
    [sys.executable, "-B", "-c", sys.argv[3], sys.argv[2]],
    start_new_session=True,
)
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
raise SystemExit(child.wait())
"""
            independent = subprocess.Popen(
                [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            cancel = threading.Event()
            registry = ProcessTreeRegistry()
            outcome: dict[str, object] = {}

            def _run_owned_tree() -> None:
                try:
                    outcome["result"] = run_streaming(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            parent_code,
                            str(ready),
                            str(delayed_marker),
                            grandchild_code,
                        ],
                        cancel_event=cancel,
                        process_registry=registry,
                        heartbeat_s=0,
                        line_callback=lambda _line: None,
                    )
                except BaseException as exc:  # pragma: no cover - assertion aid
                    outcome["error"] = exc

            worker = threading.Thread(target=_run_owned_tree, daemon=True)
            started = time.monotonic()
            worker.start()
            try:
                self.assertTrue(_wait_for(ready, 0.8), "grandchild not ready")
                cancel.set()
                worker.join(timeout=1.4)
                self.assertFalse(worker.is_alive(), "cancellation did not finish")
                self.assertNotIn("error", outcome)
                result = outcome.get("result")
                self.assertIsNotNone(result)
                self.assertEqual(getattr(result, "returncode", None), 130)
                self.assertEqual(registry.active_pids(), [])
                self.assertIsNone(
                    independent.poll(),
                    "an unrelated process was terminated",
                )
                time.sleep(0.8)
                self.assertFalse(
                    delayed_marker.exists(),
                    "the cancelled grandchild survived long enough to write",
                )
                self.assertLess(time.monotonic() - started, 3.0)

                late_ready = root / "late-term-handler.ready"
                late_marker = root / "late-term-handler-must-not-write.txt"
                handler_code = """
import pathlib,signal,subprocess,sys,time
ready, marker = pathlib.Path(sys.argv[1]), sys.argv[2]
def stop(_signum, _frame):
    subprocess.Popen(
        [sys.executable, '-B', '-c',
         "import pathlib,sys,time;time.sleep(.45);pathlib.Path(sys.argv[1]).write_text('escaped');time.sleep(10)",
         marker],
        start_new_session=True,
    )
    time.sleep(.15)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
ready.write_text('ready', encoding='utf-8')
while True: time.sleep(.1)
"""
                late_parent = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        handler_code,
                        str(late_ready),
                        str(late_marker),
                    ],
                    start_new_session=True,
                )
                self.assertTrue(_wait_for(late_ready, 0.8))
                late_report = terminate_process_tree(late_parent, grace_s=0.3)
                self.assertEqual(late_report["remaining_process_count"], 0)
                time.sleep(0.55)
                self.assertFalse(late_marker.exists())

                inherited_marker = root / "inherited-pipe-must-not-write.txt"
                inherited_code = """
import subprocess,sys,time
subprocess.Popen(
    [sys.executable, '-B', '-c',
     "import pathlib,sys,time;time.sleep(.65);pathlib.Path(sys.argv[1]).write_text('escaped');time.sleep(10)",
     sys.argv[1]],
    start_new_session=True,
)
time.sleep(.08)
"""
                inherited_started = time.monotonic()
                inherited_result = run_streaming(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        inherited_code,
                        str(inherited_marker),
                    ],
                    heartbeat_s=0,
                    line_callback=lambda _line: None,
                )
                self.assertEqual(inherited_result.returncode, 0)
                self.assertLess(time.monotonic() - inherited_started, 1.0)
                time.sleep(0.7)
                self.assertFalse(inherited_marker.exists())
            finally:
                cancel.set()
                registry.terminate_all(grace_s=0.1)
                worker.join(timeout=0.3)
                terminate_process_tree(independent, grace_s=0.1)

    def test_resume_profile_and_plan_drift_precede_manifest_mutation(
        self,
    ) -> None:
        from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions

        baseline_profile = {
            "name": "resume_fixture",
            "model_suite": {"primary": []},
            "native_producers": {"enabled": False},
        }
        baseline_plan = {"schema": "fixture-plan", "cases": ["b001"]}

        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary)
            rerun_a = WorkflowOptions(profile="fixture.yaml", out=str(out_root))
            rerun_b = WorkflowOptions(profile="fixture.yaml", out=str(out_root))
            rerun_b.rerun_generated_only = True
            rerun_b.remote_reuse_bundle = False
            rerun_b.remote_no_reuse_bundle = True
            rerun_b.remote_resume = False
            rerun_b.remote_no_resume = True
            self.assertEqual(
                build_resume_contract(
                    profile_payload=baseline_profile,
                    effective_execution_plan=baseline_plan,
                    options=rerun_a,
                ),
                build_resume_contract(
                    profile_payload=baseline_profile,
                    effective_execution_plan=baseline_plan,
                    options=rerun_b,
                ),
            )

            cancelled_dir = out_root / "cancelled_before_start"
            cancelled_dir.mkdir()
            (cancelled_dir / "sentinel.txt").write_text(
                "unchanged\n", encoding="utf-8"
            )
            cancelled_before = _tree_snapshot(cancelled_dir)
            cancelled_runner = _new_fixture_workflow_runner(
                WorkflowOptions(
                    profile="fixture.yaml",
                    out=str(out_root),
                    run_id=cancelled_dir.name,
                    resume=True,
                )
            )
            cancelled_runner.request_cancel("fixture_prestart")
            with self.assertRaises(WorkflowRunCancelledError):
                cancelled_runner.run()
            self.assertEqual(_tree_snapshot(cancelled_dir), cancelled_before)

            for drift_kind in ("profile", "plan"):
                with self.subTest(drift=drift_kind):
                    run_dir = out_root / f"resume_{drift_kind}"
                    run_dir.mkdir()
                    options = WorkflowOptions(
                        profile="fixture.yaml",
                        out=str(out_root),
                        run_id=run_dir.name,
                        resume=True,
                    )
                    archived_contract = build_resume_contract(
                        profile_payload=baseline_profile,
                        effective_execution_plan=baseline_plan,
                        options=options,
                    )
                    manifest_path = run_dir / "run_manifest.json"
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "schema": (
                                    "onnx-splitpoint/evaluation-run-manifest"
                                ),
                                "schema_version": 1,
                                "run_id": run_dir.name,
                                "profile_id": "resume_fixture",
                                "status": "partial",
                                "resume_contract": archived_contract,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    (run_dir / "profile.yaml").write_text(
                        json.dumps(baseline_profile, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    (run_dir / "stage-sentinel.txt").write_text(
                        "unchanged\n", encoding="utf-8"
                    )
                    before = _tree_snapshot(run_dir)
                    current_profile = dict(baseline_profile)
                    current_plan = dict(baseline_plan)
                    if drift_kind == "profile":
                        current_profile["quality_gate"] = {"margin": 0.02}
                    else:
                        current_plan["cases"] = ["b002"]

                    runner = _new_fixture_workflow_runner(options)

                    def _load_fixture() -> None:
                        runner.profile_id = "resume_fixture"
                        runner.profile_payload = dict(current_profile)

                    runner._load_profile = _load_fixture  # type: ignore[method-assign]
                    runner._validate_requested_start_contract = (  # type: ignore[method-assign]
                        lambda: None
                    )
                    runner._materialize_window_method_probe_profile = (  # type: ignore[method-assign]
                        lambda: None
                    )
                    runner._validate_resume_profile_snapshot = (  # type: ignore[method-assign]
                        lambda: None
                    )
                    runner._effective_execution_plan_for_resume = (  # type: ignore[method-assign]
                        lambda: dict(current_plan)
                    )
                    runner._run_locked = (  # type: ignore[method-assign]
                        lambda: self.fail("drift reached manifest mutation")
                    )

                    with self.assertRaises(WorkflowRunTargetError):
                        runner.run()
                    self.assertIsNone(runner._run_lock)
                    self.assertEqual(_tree_snapshot(run_dir), before)

                    with EvaluationRunLock(
                        out_root=out_root,
                        run_dir=run_dir,
                        owner={"session_id": f"reuse-{drift_kind}"},
                    ):
                        self.assertEqual(_tree_snapshot(run_dir), before)


if __name__ == "__main__":
    unittest.main()
