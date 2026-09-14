from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BoundedLocalCleanupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "process-tree fixture requires POSIX")
    def test_terminate_all_times_out_worker_and_retains_survivor(self) -> None:
        from onnx_splitpoint_tool.process_control import (
            ProcessTreeCancellationError,
            ProcessTreeRegistry,
            terminate_process_tree,
        )

        proc = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        registry = ProcessTreeRegistry()
        registry.register(proc, label="bounded-worker-fixture")
        release_worker = threading.Event()
        original_terminate_entry = ProcessTreeRegistry._terminate_entry

        def _stuck_terminate_entry(entry, *, grace_s):
            release_worker.wait(timeout=5.0)
            return original_terminate_entry(entry, grace_s=0.05)

        try:
            with mock.patch.object(
                ProcessTreeRegistry,
                "_termination_worker_join_budget_s",
                return_value=0.05,
            ), mock.patch.object(
                ProcessTreeRegistry,
                "_terminate_entry",
                new=staticmethod(_stuck_terminate_entry),
            ):
                started = time.monotonic()
                reports = registry.terminate_all(grace_s=0.01)
                elapsed = time.monotonic() - started

            self.assertLess(elapsed, 0.5)
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0]["remaining_process_count"], 1)
            self.assertIn("bounded cleanup budget", reports[0]["termination_error"])
            self.assertTrue(registry.has_owned_processes())
            with self.assertRaises(ProcessTreeCancellationError):
                registry.assert_quiescent()
        finally:
            release_worker.set()
            terminate_process_tree(proc, grace_s=0.05)
            registry.terminate_all(grace_s=0.05)
        registry.assert_quiescent()

    @unittest.skipUnless(os.name == "posix", "process-tree fixture requires POSIX")
    def test_native_cancel_returns_without_waiting_for_reported_survivor(self) -> None:
        from onnx_splitpoint_tool.native_progress import run_streaming
        from onnx_splitpoint_tool.process_control import (
            ProcessTreeCancellationError,
            ProcessTreeRegistry,
        )

        class _ReportedSurvivorRegistry(ProcessTreeRegistry):
            def __init__(self) -> None:
                super().__init__()
                self.registered = threading.Event()
                self.proc: subprocess.Popen[str] | None = None
                self.terminate_calls = 0

            def register(self, proc, *, label=""):
                result = super().register(proc, label=label)
                self.proc = proc
                self.registered.set()
                return result

            def terminate_registered(self, proc, *, grace_s=3.0):
                self.terminate_calls += 1
                return {
                    "root_pid": int(proc.pid),
                    "remaining_process_count": 1,
                    "termination_error": "synthetic survivor",
                }

        registry = _ReportedSurvivorRegistry()
        cancel_event = threading.Event()
        outcome: dict[str, object] = {}

        def _run() -> None:
            outcome["result"] = run_streaming(
                [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                cancel_event=cancel_event,
                process_registry=registry,
                heartbeat_s=0,
                line_callback=lambda _line: None,
            )

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        self.assertTrue(registry.registered.wait(timeout=2.0))
        cancel_started = time.monotonic()
        cancel_event.set()
        worker.join(timeout=2.0)
        cancel_elapsed = time.monotonic() - cancel_started

        try:
            self.assertFalse(worker.is_alive())
            self.assertLess(cancel_elapsed, 1.8)
            self.assertEqual(getattr(outcome.get("result"), "returncode", None), 130)
            self.assertGreaterEqual(registry.terminate_calls, 1)
            self.assertTrue(registry.has_owned_processes())
            with self.assertRaises(ProcessTreeCancellationError):
                registry.assert_quiescent()
        finally:
            proc = registry.proc
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)
            if proc is not None and proc.stdout is not None:
                proc.stdout.close()
            if proc is not None:
                ProcessTreeRegistry.unregister(registry, proc)
        registry.assert_quiescent()


if __name__ == "__main__":
    unittest.main()
