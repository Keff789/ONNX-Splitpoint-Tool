from __future__ import annotations

"""Short P0.1 guards for run ownership and GUI shutdown routing."""

from pathlib import Path
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_gui_app_headless():
    """Import GUI orchestration without switching an active headless backend."""

    import matplotlib

    with mock.patch.object(matplotlib, "use", lambda *_args, **_kwargs: None):
        return importlib.import_module("onnx_splitpoint_tool.gui.app")


def _process_start_ticks(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        return str(raw[raw.rfind(")") + 2 :].split()[19])
    except Exception:
        return ""


def _process_can_execute(pid: int, start_ticks: str) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    current = _process_start_ticks(pid)
    if start_ticks and current != start_ticks:
        return False
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        state = raw[raw.rfind(")") + 2 :].split()[0]
        return state != "Z"
    except Exception:
        return True


def _fixture_build_snapshot() -> dict[str, object]:
    """Small provenance stand-in for process-control-only Runner fixtures."""

    return {
        "schema": "onnx-splitpoint/p01-test-build-identity",
        "schema_version": 1,
        "package_version": "p01-fixture",
        "build_id": "p01-fixture",
        "critical_module_set_complete": True,
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fixture_diagnostics(
    *, phase_path: Path, stdout_path: Path, stderr_path: Path
) -> str:
    def _tail(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except Exception as exc:
            return f"<unavailable: {type(exc).__name__}: {exc}>"

    try:
        phase = phase_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
    except Exception as exc:
        phase = f"<unavailable: {type(exc).__name__}: {exc}>"
    return (
        f"startup_phase={phase}\n"
        f"stdout_tail={_tail(stdout_path)}\n"
        f"stderr_tail={_tail(stderr_path)}"
    )


def _signal_fixture_main(
    out_root: str, ready_file: str, status_file: str
) -> int:
    """Run one real Runner signal/lock lifecycle without an Evaluation stage."""

    root = Path(out_root).resolve()
    ready_path = Path(ready_file).resolve()
    status_path = Path(status_file).resolve()
    phase_path = root / "startup_phase.json"
    _write_json_atomic(phase_path, {"phase": "importing_workflow"})

    from onnx_splitpoint_tool.workflow import run_evaluation
    from onnx_splitpoint_tool.workflow.contracts import WorkflowRunResult
    from onnx_splitpoint_tool.workflow import runner as runner_module

    RealEvaluationWorkflowRunner = runner_module.EvaluationWorkflowRunner
    _write_json_atomic(phase_path, {"phase": "workflow_imported"})
    grandchild_path = root / "grandchild.pid"
    holder: dict[str, RealEvaluationWorkflowRunner] = {}

    class _SignalFixtureRunner(RealEvaluationWorkflowRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["runner"] = self
            _write_json_atomic(phase_path, {"phase": "runner_constructed"})

        def _load_profile(self) -> None:
            self.profile_id = "p01_signal_fixture"
            self.profile_payload = {
                "name": "p01_signal_fixture",
                "model_suite": {"primary": []},
                "native_producers": {"enabled": False},
            }

        def _validate_requested_start_contract(self) -> None:
            return None

        def _materialize_window_method_probe_profile(self) -> None:
            return None

        def _effective_execution_plan_for_resume(self):
            return {"schema": "p01-signal-fixture-plan", "cases": []}

        def _run_locked(self) -> WorkflowRunResult:
            _write_json_atomic(phase_path, {"phase": "run_locked_entered"})
            grandchild_code = "import time\nwhile True:\n    time.sleep(1)\n"
            child_code = (
                "from pathlib import Path\n"
                "import subprocess, sys, time\n"
                "grandchild = subprocess.Popen(\n"
                "    [sys.executable, '-B', '-c', sys.argv[2]],\n"
                "    stdin=subprocess.DEVNULL,\n"
                "    stdout=subprocess.DEVNULL,\n"
                "    stderr=subprocess.DEVNULL,\n"
                ")\n"
                "Path(sys.argv[1]).write_text(str(grandchild.pid), encoding='utf-8')\n"
                "while True:\n"
                "    time.sleep(1)\n"
            )
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    child_code,
                    str(grandchild_path),
                    grandchild_code,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._process_registry.register(child, label="p01_cli_signal_fixture")
            deadline = time.monotonic() + 5.0
            grandchild_pid = 0
            while grandchild_pid <= 1:
                if child.poll() is not None:
                    raise RuntimeError("signal fixture child exited before ready")
                if grandchild_path.is_file():
                    try:
                        grandchild_pid = int(
                            grandchild_path.read_text(encoding="utf-8").strip()
                        )
                    except (OSError, TypeError, ValueError):
                        grandchild_pid = 0
                if time.monotonic() >= deadline:
                    raise TimeoutError("signal fixture grandchild did not start")
                if grandchild_pid <= 1:
                    time.sleep(0.01)
            _write_json_atomic(
                ready_path,
                {
                    "run_dir": str(self.run_dir),
                    "child_pid": int(child.pid),
                    "grandchild_pid": grandchild_pid,
                },
            )
            _write_json_atomic(phase_path, {"phase": "ready"})
            while not self._cancel_event.wait(0.02):
                if child.poll() is not None:
                    raise RuntimeError("signal fixture child exited unexpectedly")
            return WorkflowRunResult(
                ok=False,
                status="cancelled",
                run_id=self.run_id,
                run_dir=str(self.run_dir),
                manifest_path=str(self.manifest_path),
                artifact_index_path=str(self.artifact_index_path),
                completed=False,
            )

    run_evaluation.EvaluationWorkflowRunner = _SignalFixtureRunner
    _write_json_atomic(phase_path, {"phase": "starting_runner"})
    # Package provenance is unrelated to this process-control test and is
    # covered independently.  Avoid recursively hashing the complete source
    # tree in each fresh signal subprocess.
    with mock.patch.object(
        runner_module,
        "package_build_snapshot",
        return_value=_fixture_build_snapshot(),
    ):
        exit_code = run_evaluation.main(
            [
                "--profile",
                "p01-signal-fixture.yaml",
                "--out",
                str(root),
                "--run-id",
                "fixture_run",
                "--execution-mode",
                "contracts_only",
                "--json",
            ]
        )
    runner = holder.get("runner")
    _write_json_atomic(
        status_path,
        {
            "exit_code": int(exit_code),
            "status": (
                "cancelled"
                if runner is not None and runner._cancel_event.is_set()
                else "unexpected"
            ),
        },
    )
    return int(exit_code)


class RunnerOwnershipGuardTests(unittest.TestCase):
    @staticmethod
    def _runner(root: Path, *, execution_mode: str = "contracts_only"):
        from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
        from onnx_splitpoint_tool.workflow import runner as runner_module

        with mock.patch.object(
            runner_module,
            "package_build_snapshot",
            return_value=_fixture_build_snapshot(),
        ):
            runner = runner_module.EvaluationWorkflowRunner(
                WorkflowOptions(
                    profile="fixture.yaml",
                    out=str(root),
                    run_id="fixture_run",
                    execution_mode=execution_mode,
                )
            )

        def _load_fixture() -> None:
            runner.profile_id = "fixture"
            runner.profile_payload = {
                "name": "fixture",
                "model_suite": {"primary": []},
                "native_producers": {"enabled": False},
            }

        runner._load_profile = _load_fixture  # type: ignore[method-assign]
        runner._validate_requested_start_contract = (  # type: ignore[method-assign]
            lambda: None
        )
        runner._materialize_window_method_probe_profile = (  # type: ignore[method-assign]
            lambda: None
        )
        runner._effective_execution_plan_for_resume = (  # type: ignore[method-assign]
            lambda: {"schema": "fixture-plan", "cases": []}
        )
        runner._install_signal_handlers = lambda: None  # type: ignore[method-assign]
        runner._restore_signal_handlers = lambda: None  # type: ignore[method-assign]
        runner._start_cancellation_watcher = lambda: None  # type: ignore[method-assign]
        runner._stop_cancellation_watcher = lambda: None  # type: ignore[method-assign]
        runner._shutdown_management_services = lambda: None  # type: ignore[method-assign]
        return runner

    @staticmethod
    def _write_current_resume_fixture(runner, run_dir: Path) -> None:
        """Materialize the minimum current, exact-contract Resume target.

        Resume admission intentionally rejects the old ``{}`` manifest stubs
        before acquiring a lease or recovering prior journals.  Ownership
        ordering tests therefore need a structurally admitted target; bypassing
        the admission guard would no longer exercise the production sequence.
        """

        run_dir.mkdir(parents=True, exist_ok=True)
        runner._load_profile()
        contract = runner._build_current_resume_contract()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "onnx-splitpoint/evaluation-run-manifest",
                    "schema_version": 1,
                    "run_id": run_dir.name,
                    "profile_id": "fixture",
                    "tool_version": "2.75.17",
                    "status": "partial",
                    "resume_contract": contract,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "profile.yaml").write_text(
            "name: fixture\n"
            "model_suite:\n"
            "  primary: []\n"
            "native_producers:\n"
            "  enabled: false\n",
            encoding="utf-8",
        )

    def test_job_cancel_write_requires_active_run_ownership(self) -> None:
        class _JobsSpy:
            def __init__(self) -> None:
                self.reasons: list[str] = []

            def request_cancel(self, reason: str) -> None:
                self.reasons.append(reason)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prestart = self._runner(root)
            prestart.run_id = "prestart"
            prestart.run_dir = root / prestart.run_id
            prestart.run_dir.mkdir()
            prestart_jobs = _JobsSpy()
            prestart.jobs = prestart_jobs

            prestart.request_cancel("before_ownership")
            self.assertEqual(prestart_jobs.reasons, [])
            self.assertFalse((prestart.run_dir / "jobs").exists())

            runner = self._runner(root)
            runner.run_id = "active"
            runner.run_dir = root / runner.run_id
            runner.run_dir.mkdir()
            jobs = _JobsSpy()
            runner.jobs = jobs
            runner._run_lock = object()  # type: ignore[assignment]
            runner._run_control_write_enabled = True
            runner.request_cancel("active_owner")
            self.assertEqual(jobs.reasons, ["active_owner"])
            marker = runner.run_dir / "jobs" / "workflow_control.json"
            self.assertTrue(marker.is_file())

            runner._run_control_write_enabled = False
            runner._run_lock = None
            before = marker.read_bytes()
            runner.request_cancel("after_release")
            self.assertEqual(jobs.reasons, ["active_owner"])
            self.assertEqual(marker.read_bytes(), before)

    def test_surviving_owned_process_quarantines_and_releases_lock(self) -> None:
        from onnx_splitpoint_tool.process_control import (
            ProcessTreeCancellationError,
        )
        from onnx_splitpoint_tool.workflow.run_control import (
            EvaluationRunLock,
            WorkflowRunCleanupQuarantineError,
        )

        class _UnresolvedRegistry:
            def __init__(self) -> None:
                self.terminate_calls = 0
                self.lock_observations: list[bool] = []

            def terminate_all(self, *, grace_s: float):
                self.terminate_calls += 1
                return [
                    {
                        "remaining_process_count": 1,
                        "root_pid": 424242,
                    }
                ]

            def assert_quiescent(self) -> None:
                self.lock_observations.append(runner._run_lock is not None)
                raise ProcessTreeCancellationError(
                    "fixture survivor",
                    reports=[
                        {"remaining_process_count": 1, "root_pid": 424242}
                    ],
                    capabilities={},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runner = self._runner(root)
            registry = _UnresolvedRegistry()
            runner._process_registry = registry  # type: ignore[assignment]
            runner._run_locked = lambda: object()  # type: ignore[method-assign]

            started = time.monotonic()
            with self.assertRaises(WorkflowRunCleanupQuarantineError) as raised:
                runner.run()
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertEqual(
                raised.exception.error_code, "run_cleanup_quarantined"
            )
            self.assertEqual(registry.terminate_calls, 1)
            self.assertTrue(registry.lock_observations)
            self.assertTrue(all(registry.lock_observations))
            self.assertIsNone(runner._run_lock)
            marker = (
                root
                / "fixture_run"
                / "jobs"
                / "p01_unresolved_cleanup_quarantine.json"
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertFalse(payload["local_process_quiescence_proven"])

            # The kernel lock was released, but the durable sidecar fence is
            # checked by the next acquirer before it can replace owner state.
            probe = EvaluationRunLock(
                out_root=root,
                run_dir=root / "fixture_run",
                owner={"session_id": "post-quarantine-probe"},
            )
            with self.assertRaises(WorkflowRunCleanupQuarantineError):
                probe.acquire()

            # The durable marker is checked under the next writer's lock and
            # blocks Resume before the workload callback can run.
            resume = self._runner(root)
            resume.options.resume = True
            self._write_current_resume_fixture(
                resume, root / "fixture_run"
            )
            resume_workload = mock.Mock(return_value=object())
            resume._run_locked = resume_workload  # type: ignore[method-assign]
            with self.assertRaises(WorkflowRunCleanupQuarantineError):
                resume.run()
            resume_workload.assert_not_called()
            self.assertIsNone(resume._run_lock)

    def test_resume_inspects_every_prior_remote_session_and_fails_closed(
        self,
    ) -> None:
        from onnx_splitpoint_tool.workflow.run_control import (
            WorkflowRunCleanupQuarantineError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runner = self._runner(root)
            runner.options.resume = True
            runner.run_id = "fixture_run"
            runner.run_dir = root / runner.run_id
            journals = runner.run_dir / "jobs" / "remote_process_leases"
            invalid = journals / "old-invalid-session"
            clean = journals / "old-clean-session"
            invalid.mkdir(parents=True)
            clean.mkdir(parents=True)
            (invalid / "broken.remote-lease.json").write_text(
                "{not-json\n", encoding="utf-8"
            )

            with self.assertRaises(WorkflowRunCleanupQuarantineError) as raised:
                runner._recover_prior_remote_lease_journals()

            details = dict(raised.exception.owner.get("details") or {})
            self.assertEqual(details["inspected_session_count"], 2)
            inspected = {
                row["session_id"] for row in details.get("journals", [])
            }
            self.assertEqual(
                inspected, {"old-invalid-session", "old-clean-session"}
            )
            # The clean directory was not skipped after the invalid one.
            self.assertTrue((clean / "session.cancelled.json").is_file())
            marker = runner._cleanup_quarantine_path
            self.assertTrue(marker.is_file())

    def test_resume_contract_precedes_prior_journal_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_dir = root / "fixture_run"
            runner = self._runner(root)
            runner.options.resume = True
            self._write_current_resume_fixture(runner, run_dir)
            events: list[str] = []
            runner._validate_resume_profile_snapshot = (  # type: ignore[method-assign]
                lambda: events.append("profile_snapshot")
            )
            runner._validate_resume_contract = (  # type: ignore[method-assign]
                lambda: events.append("resume_contract")
            )
            runner._recover_prior_remote_lease_journals = (  # type: ignore[method-assign]
                lambda: events.append("prior_journals")
            )
            runner._run_locked = lambda: object()  # type: ignore[method-assign]

            self.assertIsNotNone(runner.run())
            self.assertLess(
                events.index("resume_contract"),
                events.index("prior_journals"),
            )

    def test_final_remote_journal_exception_quarantines_after_unlock(self) -> None:
        from onnx_splitpoint_tool.workflow.run_control import (
            EvaluationRunLock,
            WorkflowRunCleanupQuarantineError,
        )

        class _ExplodingRemoteRegistry:
            def configure_journal(self, **_kwargs):
                return None

            def cancel_all(self, *, grace_s: float):
                raise OSError("fixture journal I/O failure")

            def active_count(self) -> int:
                raise OSError("fixture journal scan failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runner = self._runner(root)
            runner._remote_process_registry = _ExplodingRemoteRegistry()  # type: ignore[assignment]
            runner._run_locked = lambda: object()  # type: ignore[method-assign]
            runner._write_cleanup_quarantine = mock.Mock(  # type: ignore[method-assign]
                side_effect=OSError("fixture run-dir marker write failure")
            )

            with self.assertRaises(WorkflowRunCleanupQuarantineError) as raised:
                runner.run()
            self.assertEqual(
                raised.exception.error_code, "run_cleanup_quarantined"
            )
            self.assertIsNone(runner._run_lock)
            self.assertIn("quarantine marker write failed", str(raised.exception))
            self.assertFalse(runner._cleanup_quarantine_path.is_file())

            sentinel = root / "fixture_run" / "untouched.txt"
            sentinel.write_text("before-fenced-acquire\n", encoding="utf-8")
            probe = EvaluationRunLock(
                out_root=root,
                run_dir=root / "fixture_run",
                owner={"session_id": "journal-error-lock-probe"},
            )
            self.assertTrue(probe.quarantine_path.is_file())
            fence_before = probe.quarantine_path.read_bytes()
            with self.assertRaises(WorkflowRunCleanupQuarantineError):
                probe.acquire()
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "before-fenced-acquire\n",
            )
            self.assertEqual(probe.quarantine_path.read_bytes(), fence_before)

    def test_lock_quarantine_falls_back_to_owned_inode(self) -> None:
        from onnx_splitpoint_tool.workflow.run_control import (
            EvaluationRunLock,
            WorkflowRunCleanupQuarantineError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_dir = root / "fixture_run"
            lock = EvaluationRunLock(
                out_root=root,
                run_dir=run_dir,
                owner={"session_id": "fallback-writer"},
            )
            lock.acquire()
            try:
                with mock.patch(
                    "onnx_splitpoint_tool.workflow.run_control.os.replace",
                    side_effect=OSError("fixture sidecar failure"),
                ):
                    fence_path = lock.commit_quarantine_fence(
                        {"reason": "fixture_unproven_cleanup"}
                    )
                self.assertEqual(fence_path, lock.lock_path)
                self.assertFalse(lock.quarantine_path.exists())
            finally:
                lock.release()

            probe = EvaluationRunLock(
                out_root=root,
                run_dir=run_dir,
                owner={"session_id": "fallback-probe"},
            )
            with self.assertRaises(WorkflowRunCleanupQuarantineError):
                probe.acquire()

    def test_unproven_direct_native_ssh_poison_is_terminal_rc70(self) -> None:
        from onnx_splitpoint_tool.workflow.run_control import (
            WorkflowRunCleanupQuarantineError,
        )

        class _Lease:
            operation_id = "fixture-direct-ssh"

            def cancel_remote(self, *, grace_s: float):
                return {"ok": False, "returncode": 70, "output": "survivor"}

        class _Registry:
            def __init__(self) -> None:
                self.poisoned = False
                self.unregistered = False

            def poison(self) -> None:
                self.poisoned = True

            def unregister(self, *_args, **_kwargs) -> None:
                self.unregistered = True

        class _Completed:
            returncode = 255
            stdout = "ssh failed"

        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary).resolve())
            registry = _Registry()
            runner._remote_process_registry = registry  # type: ignore[assignment]
            completed = _Completed()
            with self.assertRaises(WorkflowRunCleanupQuarantineError):
                runner._finish_native_direct_remote_lease(_Lease(), completed)
            self.assertTrue(registry.poisoned)
            self.assertFalse(registry.unregistered)
            self.assertEqual(completed.returncode, 70)
            self.assertIn("rc=70", completed.stdout)

    def test_nested_session_capability_gate_precedes_run_creation(self) -> None:
        from onnx_splitpoint_tool.process_control import ProcessTreeCapabilityError
        from onnx_splitpoint_tool.workflow.run_control import WorkflowRunTargetError

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for mode in (
                "generate_benchmarksets",
                "generate_and_run",
                "legacy_benchmarkset",
                "legacy_profile_campaign",
            ):
                executing = self._runner(root / mode, execution_mode=mode)
                executing.profile_payload = {
                    "native_producers": {"enabled": False}
                }
                self.assertTrue(executing._requires_nested_session_cancellation())

            dry = self._runner(root / "dry", execution_mode="generate_and_run")
            dry.options.dry_run = True
            dry.profile_payload = {"native_producers": {"enabled": True}}
            self.assertFalse(dry._requires_nested_session_cancellation())

            native_contract = self._runner(
                root / "native-contract", execution_mode="contracts_only"
            )
            native_contract.profile_payload = {
                "native_producers": {"enabled": True}
            }
            self.assertTrue(
                native_contract._requires_nested_session_cancellation()
            )

            runner = self._runner(root, execution_mode="generate_and_run")
            failure = ProcessTreeCapabilityError(
                "fixture procfs mismatch",
                capabilities={"nested_session_cancellation_safe": False},
            )
            with mock.patch(
                "onnx_splitpoint_tool.workflow.runner."
                "require_nested_session_cancellation",
                side_effect=failure,
            ):
                with self.assertRaisesRegex(
                    WorkflowRunTargetError, "fixture procfs mismatch"
                ) as raised:
                    runner.run()
            self.assertEqual(
                raised.exception.error_code,
                "nested_process_tree_cancellation_unavailable",
            )
            self.assertFalse((root / "fixture_run").exists())
            self.assertIsNone(runner._run_lock)

            portable = self._runner(root, execution_mode="contracts_only")
            portable._run_locked = lambda: object()  # type: ignore[method-assign]
            with mock.patch(
                "onnx_splitpoint_tool.workflow.runner."
                "require_nested_session_cancellation"
            ) as capability_gate:
                portable.run()
            capability_gate.assert_not_called()

    def test_cli_native_override_capability_gate_precedes_run_mutation(
        self,
    ) -> None:
        from onnx_splitpoint_tool.process_control import ProcessTreeCapabilityError
        from onnx_splitpoint_tool.workflow.run_control import WorkflowRunTargetError

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "not-created-yet"
            runner = self._runner(root, execution_mode="contracts_only")
            runner.options.native_producer_enabled = True
            self.assertTrue(runner._requires_nested_session_cancellation())

            failure = ProcessTreeCapabilityError(
                "fixture CLI-native procfs mismatch",
                capabilities={"nested_session_cancellation_safe": False},
            )
            with mock.patch(
                "onnx_splitpoint_tool.workflow.runner."
                "require_nested_session_cancellation",
                side_effect=failure,
            ):
                with self.assertRaisesRegex(
                    WorkflowRunTargetError, "fixture CLI-native procfs mismatch"
                ) as raised:
                    runner.run()

            self.assertEqual(
                raised.exception.error_code,
                "nested_process_tree_cancellation_unavailable",
            )
            self.assertFalse((root / "fixture_run").exists())
            self.assertIsNone(runner._run_lock)


class GuiShutdownBrokerTests(unittest.TestCase):
    def test_gui_job_ids_do_not_collide_within_one_second(self) -> None:
        gui_app = _import_gui_app_headless()
        _new_evaluation_workflow_job_id = (
            gui_app._new_evaluation_workflow_job_id
        )

        with mock.patch(
            "onnx_splitpoint_tool.gui.app.time.strftime",
            return_value="20260801_120000",
        ):
            ids = {_new_evaluation_workflow_job_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_parallel_workflows_namespace_subjob_and_parent_rows(self) -> None:
        gui_app = _import_gui_app_headless()

        gui = object.__new__(gui_app.SplitPointAnalyserGUI)
        gui._background_jobs = {}
        gui._background_job_order = []
        gui._jobs_refresh_views = lambda: None  # type: ignore[method-assign]
        gui._jobs_status_label = lambda status: status  # type: ignore[method-assign]
        gui._jobs_parse_iso_datetime = lambda _value: None  # type: ignore[method-assign]
        event = {
            "event": "planned",
            "job_id": "stage:run_benchmarks",
            "parent_job_id": "workflow:shared-run-id",
            "status": "queued",
            "title": "run_benchmarks",
        }

        gui._jobs_handle_workflow_job_event(
            event, workflow_scope="evaluation-workflow-launch-a"
        )
        gui._jobs_handle_workflow_job_event(
            event, workflow_scope="evaluation-workflow-launch-b"
        )

        rows = list(gui._background_jobs.values())
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0].job_id, rows[1].job_id)
        self.assertNotEqual(rows[0].parent_job_id, rows[1].parent_job_id)
        self.assertIn("launch-a", rows[0].job_id)
        self.assertIn("launch-b", rows[1].job_id)

    def test_signal_handler_defers_close_to_tk_main_loop(self) -> None:
        gui_app = _import_gui_app_headless()

        class _FakeRoot:
            def __init__(self) -> None:
                self.after_calls: list[tuple[int, object]] = []

            def after(self, delay_ms: int, callback: object) -> str:
                self.after_calls.append((delay_ms, callback))
                return "fixture-after"

        gui = object.__new__(gui_app.SplitPointAnalyserGUI)
        root = _FakeRoot()
        gui.root = root
        gui._on_close = mock.Mock()  # type: ignore[method-assign]
        installed: dict[int, object] = {}

        def _capture_handler(signum: int, handler: object) -> None:
            installed[int(signum)] = handler

        with mock.patch.object(
            gui_app.signal, "getsignal", return_value=signal.SIG_DFL
        ), mock.patch.object(gui_app.signal, "signal", side_effect=_capture_handler):
            gui._install_evaluation_shutdown_signal_broker()
            self.assertIn(int(signal.SIGTERM), installed)
            self.assertEqual(len(root.after_calls), 1)

            handler = installed[int(signal.SIGTERM)]
            assert callable(handler)
            handler(signal.SIGTERM, None)

            # The actual signal callback only records intent.  It neither
            # touches Tk nor calls cancellation/close directly.
            self.assertEqual(len(root.after_calls), 1)
            gui._on_close.assert_not_called()

            poll = root.after_calls[0][1]
            assert callable(poll)
            poll()
            gui._on_close.assert_called_once_with()


@unittest.skipUnless(
    os.name == "posix" and hasattr(signal, "SIGINT") and hasattr(signal, "SIGTERM"),
    "the CLI signal integration fixture requires POSIX signals",
)
class CliSignalIntegrationTests(unittest.TestCase):
    def test_sigint_and_sigterm_cancel_tree_and_release_lock(self) -> None:
        from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock

        repo_root = Path(__file__).resolve().parents[1]
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signal.Signals(signum).name):
                with tempfile.TemporaryDirectory() as temporary:
                    out_root = Path(temporary).resolve()
                    ready_path = out_root / "ready.json"
                    status_path = out_root / "status.json"
                    phase_path = out_root / "startup_phase.json"
                    stdout_path = out_root / "fixture.stdout.txt"
                    stderr_path = out_root / "fixture.stderr.txt"
                    stdout_handle = stdout_path.open("w", encoding="utf-8")
                    stderr_handle = stderr_path.open("w", encoding="utf-8")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            str(Path(__file__).resolve()),
                            "--p01-signal-fixture",
                            str(out_root),
                            str(ready_path),
                            str(status_path),
                        ],
                        cwd=str(repo_root),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                    )
                    child_pid = 0
                    grandchild_pid = 0
                    child_start = ""
                    grandchild_start = ""
                    try:
                        deadline = time.monotonic() + 30.0
                        while not ready_path.is_file():
                            if process.poll() is not None:
                                self.fail(
                                    "signal fixture exited before ready: "
                                    f"rc={process.returncode}\n"
                                    + _fixture_diagnostics(
                                        phase_path=phase_path,
                                        stdout_path=stdout_path,
                                        stderr_path=stderr_path,
                                    )
                                )
                            if time.monotonic() >= deadline:
                                self.fail(
                                    "signal fixture did not become ready\n"
                                    + _fixture_diagnostics(
                                        phase_path=phase_path,
                                        stdout_path=stdout_path,
                                        stderr_path=stderr_path,
                                    )
                                )
                            time.sleep(0.01)

                        ready = json.loads(ready_path.read_text(encoding="utf-8"))
                        child_pid = int(ready["child_pid"])
                        grandchild_pid = int(ready["grandchild_pid"])
                        child_start = _process_start_ticks(child_pid)
                        grandchild_start = _process_start_ticks(grandchild_pid)
                        self.assertTrue(
                            _process_can_execute(child_pid, child_start)
                        )
                        self.assertTrue(
                            _process_can_execute(grandchild_pid, grandchild_start)
                        )

                        os.kill(process.pid, signum)
                        process.wait(timeout=15.0)
                        self.assertEqual(
                            process.returncode,
                            130,
                            msg=_fixture_diagnostics(
                                phase_path=phase_path,
                                stdout_path=stdout_path,
                                stderr_path=stderr_path,
                            ),
                        )
                        status = json.loads(
                            status_path.read_text(encoding="utf-8")
                        )
                        self.assertEqual(status["status"], "cancelled")
                        self.assertEqual(int(status["exit_code"]), 130)

                        quiet_deadline = time.monotonic() + 3.0
                        while time.monotonic() < quiet_deadline and (
                            _process_can_execute(child_pid, child_start)
                            or _process_can_execute(
                                grandchild_pid, grandchild_start
                            )
                        ):
                            time.sleep(0.02)
                        self.assertFalse(
                            _process_can_execute(child_pid, child_start),
                            "registered child survived CLI cancellation",
                        )
                        self.assertFalse(
                            _process_can_execute(
                                grandchild_pid, grandchild_start
                            ),
                            "registered grandchild survived CLI cancellation",
                        )

                        run_dir = Path(str(ready["run_dir"]))
                        with EvaluationRunLock(
                            out_root=out_root,
                            run_dir=run_dir,
                            owner={"session_id": "post-signal-lock-probe"},
                        ):
                            pass
                    finally:
                        if process.poll() is None:
                            # Give the real workflow signal handler a chance to
                            # drain its registered child tree even when a test
                            # assertion fires before the normal signal step.
                            try:
                                os.kill(process.pid, signal.SIGTERM)
                                process.wait(timeout=15.0)
                            except (ProcessLookupError, PermissionError, OSError):
                                pass
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5.0)
                        stdout_handle.close()
                        stderr_handle.close()
                        for pid, start_ticks in (
                            (child_pid, child_start),
                            (grandchild_pid, grandchild_start),
                        ):
                            if pid > 1 and _process_can_execute(pid, start_ticks):
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except (ProcessLookupError, PermissionError, OSError):
                                    pass


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--p01-signal-fixture":
        raise SystemExit(_signal_fixture_main(*sys.argv[2:]))
    unittest.main()
