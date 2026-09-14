from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.deepx import compiler as deepx_compiler
from onnx_splitpoint_tool.deepx import env_status as deepx_env_status
from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool import window_method_validation_probe as window_probe
from onnx_splitpoint_tool.process_control import (
    ProcessTreeCancellationError,
    ProcessTreeRegistry,
    bind_process_registry,
    current_process_registry,
)
from onnx_splitpoint_tool.workflow import execution_binding
from onnx_splitpoint_tool.workflow import legacy_benchmarkset_binding


class _FakeProcess:
    def __init__(self, *, communicate_error: BaseException | None = None) -> None:
        self.pid = 424242
        self.returncode: int | None = None
        self.communicate_error = communicate_error
        self.communicate_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def communicate(self, *args, **kwargs):
        timeout = kwargs.get("timeout")
        self.communicate_timeouts.append(timeout)
        if timeout is None:
            raise AssertionError("unbounded communicate() used")
        if self.communicate_error is not None:
            raise self.communicate_error
        return "captured", ""

    def wait(self, timeout=None) -> int:
        if timeout is None:
            raise AssertionError("unbounded wait() used")
        self.returncode = -9 if self.returncode is None else self.returncode
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class _Registry:
    def __init__(self, *, cancellation_reads: list[bool] | None = None, events=None) -> None:
        self._reads = list(cancellation_reads or [False])
        self.events = events if events is not None else []
        self.uncertainties: list[tuple[str, str]] = []

    @property
    def cancelled(self) -> bool:
        if len(self._reads) > 1:
            return self._reads.pop(0)
        return bool(self._reads[0])

    def register(self, proc, *, label=""):
        self.events.append("register")

    def terminate_registered(self, proc, *, grace_s=0.0):
        self.events.append("local")
        proc.returncode = -9
        return {"remaining_process_count": 0}

    def unregister(self, proc):
        self.events.append("unregister")

    def record_cleanup_uncertainty(self, *, label: str, detail: str = ""):
        self.uncertainties.append((label, detail))


class _SecondReadCancelEvent:
    def __init__(self) -> None:
        self.reads = 0

    def is_set(self) -> bool:
        self.reads += 1
        return self.reads >= 2


class BoundedCallsiteCancellationTests(unittest.TestCase):
    def test_deepx_compiler_and_probe_never_drain_without_timeout(self) -> None:
        for module, function in (
            (deepx_compiler, deepx_compiler._run_owned_compiler),
            (deepx_env_status, deepx_env_status._run_owned_probe),
        ):
            with self.subTest(module=module.__name__):
                proc = _FakeProcess()
                registry = _Registry()
                event = _SecondReadCancelEvent()
                with mock.patch.object(module.subprocess, "Popen", return_value=proc):
                    result = function(
                        ["fixture-child"],
                        timeout_s=30,
                        process_registry=registry,
                        cancel_event=event,
                    )
                self.assertEqual(130, result.returncode)
                self.assertTrue(proc.communicate_timeouts)
                self.assertNotIn(None, proc.communicate_timeouts)

    def test_hailo_cancel_poll_is_bounded(self) -> None:
        proc = _FakeProcess()
        registry = _Registry(cancellation_reads=[False, True])
        with mock.patch.object(hailo_backend, "current_process_registry", return_value=registry), mock.patch.object(
            hailo_backend, "current_remote_process_registry", return_value=None
        ), mock.patch.object(hailo_backend.subprocess, "Popen", return_value=proc):
            result = hailo_backend._run_owned_subprocess(
                ["fixture-child"],
                text=True,
                capture_output=True,
            )
        self.assertEqual(130, result.returncode)
        self.assertNotIn(None, proc.communicate_timeouts)

    def test_hailo_unexpected_leased_ssh_exception_is_remote_first(self) -> None:
        events: list[str] = []
        proc = _FakeProcess(communicate_error=RuntimeError("synthetic pipe failure"))
        local_registry = _Registry(cancellation_reads=[False], events=events)

        class _RemoteRegistry:
            def journal_environment(self):
                return {}

            def cancel_all(self, *, grace_s=0.0):
                events.append("remote")
                return []

        with mock.patch.object(
            hailo_backend, "current_process_registry", return_value=local_registry
        ), mock.patch.object(
            hailo_backend,
            "current_remote_process_registry",
            return_value=_RemoteRegistry(),
        ), mock.patch.object(hailo_backend.subprocess, "Popen", return_value=proc):
            with self.assertRaisesRegex(RuntimeError, "synthetic pipe failure"):
                hailo_backend._run_owned_subprocess(
                    ["ssh", "fixture", "echo ok"],
                    text=True,
                    capture_output=True,
                )
        self.assertLess(events.index("remote"), events.index("local"))

    def test_window_probe_timeout_uses_bounded_output_drain(self) -> None:
        proc = _FakeProcess()
        registry = _Registry()
        with mock.patch.object(window_probe, "ProcessTreeRegistry", return_value=registry), mock.patch.object(
            window_probe.subprocess, "Popen", return_value=proc
        ):
            result = window_probe._run(["fixture-child"], timeout=0.0)
        self.assertEqual(124, result["rc"])
        self.assertNotIn(None, proc.communicate_timeouts)

    def test_remote_pool_submit_propagates_bound_registry_context(self) -> None:
        marker = object()
        with bind_process_registry(marker), concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as pool:
            future = execution_binding._submit_with_context(
                pool,
                current_process_registry,
            )
            self.assertIs(marker, future.result(timeout=2.0))

    def test_sticky_worker_uncertainty_forces_final_quarantine_gate(self) -> None:
        registry = ProcessTreeRegistry()
        registry.record_cleanup_uncertainty(
            label="fixture-worker",
            detail="worker exceeded cancel grace",
        )
        self.assertTrue(registry.cancelled)
        with self.assertRaises(ProcessTreeCancellationError) as raised:
            registry.assert_quiescent()
        self.assertTrue(raised.exception.reports[0]["cleanup_uncertainty"])
        reports = registry.terminate_all(grace_s=0.0)
        self.assertTrue(any(row.get("cleanup_uncertainty") for row in reports))

        second = ProcessTreeRegistry()
        class _Pending:
            def __init__(self) -> None:
                self.cancelled = False

            def done(self):
                return False

            def cancel(self):
                self.cancelled = True
                return False

        pending = _Pending()
        with bind_process_registry(second):
            execution_binding._cancel_pending_workers_and_record_uncertainty(
                [pending],
                label="parallel-fixture",
                detail="synthetic unresolved worker",
            )
        self.assertTrue(pending.cancelled)
        with self.assertRaises(ProcessTreeCancellationError):
            second.assert_quiescent()

        class _CancelledBeforeStart:
            def done(self):
                return False

            def cancel(self):
                return True

        third = ProcessTreeRegistry()
        with bind_process_registry(third):
            unresolved = execution_binding._cancel_pending_workers_and_record_uncertainty(
                [_CancelledBeforeStart()],
                label="queued-fixture",
                detail="must not be recorded",
            )
        self.assertEqual([], unresolved)
        third.assert_quiescent()

    def test_deepx_prefetch_cancel_exit_and_shutdown_are_bounded(self) -> None:
        class _NeverDoneFuture:
            def done(self):
                return False

            def cancel(self):
                return False

            def result(self, timeout=None):
                raise AssertionError("unresolved future must not be awaited")

        class _Scheduler:
            def __init__(self):
                self.wait_values: list[bool] = []

            def shutdown(self, wait=True):
                self.wait_values.append(bool(wait))

        scheduler = _Scheduler()
        registry = _Registry(cancellation_reads=[True])
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="bounded-prefetch-") as tmp, mock.patch.object(
            legacy_benchmarkset_binding,
            "_DEEPX_PREFETCH_CANCEL_GRACE_S",
            0.0,
        ):
            report = legacy_benchmarkset_binding._v60s_finish_deepx_prefetch(
                {"scheduler": scheduler, "future": _NeverDoneFuture()},
                suite_dir=Path(tmp),
                log=lambda _message: None,
                cancel_event=threading.Event(),
                process_registry=registry,
            )
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual("cancelled_cleanup_unresolved", report["status"])
        self.assertEqual([False], scheduler.wait_values)
        self.assertEqual("deepx-prefetch-worker", registry.uncertainties[0][0])

    def test_assigned_modules_have_no_bare_postprocess_waits(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "onnx_splitpoint_tool/deepx/compiler.py",
            root / "onnx_splitpoint_tool/deepx/env_status.py",
            root / "onnx_splitpoint_tool/hailo_backend.py",
            root / "onnx_splitpoint_tool/window_method_validation_probe.py",
            root / "onnx_splitpoint_tool/workflow/execution_binding.py",
            root / "onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn(".communicate()", combined)
        self.assertNotIn(".wait()", combined)
        self.assertNotIn(".result()", combined)
        self.assertNotIn("as_completed(", combined)
        self.assertNotIn("shutdown(True)", combined)


if __name__ == "__main__":
    unittest.main()
