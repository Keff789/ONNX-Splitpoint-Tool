from __future__ import annotations

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

from onnx_splitpoint_tool.remote import process_lease
from onnx_splitpoint_tool.remote.process_lease import (
    REMOTE_LEASE_UNPROVEN_RC,
    RemoteProcessLeaseJournal,
    RemoteProcessLeaseOperation,
    RemoteProcessLeaseRegistry,
    RemoteProcessLeaseScope,
    run_journaled_ssh,
)


class _TimedOutSSH:
    pid = 424242

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired(["ssh", "fixture"], timeout)


class RemoteTombstoneIOTests(unittest.TestCase):
    def test_cancel_all_bounds_a_stuck_cleanup_worker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-stuck-worker-") as tmp:
            journal_dir = Path(tmp) / "journal"
            scope = RemoteProcessLeaseScope("fixture-run", "fixture-session")
            registry = RemoteProcessLeaseRegistry()
            registry.configure_journal(scope=scope, journal_dir=journal_dir)
            operation = scope.operation(
                label="stuck", control_argv_prefix=["ssh", "fixture"]
            )
            registry.register(operation)
            started = threading.Event()
            release = threading.Event()
            finished = threading.Event()
            daemon_flags: list[bool] = []

            def _stuck_cleanup(*, grace_s: float = 3.0) -> dict[str, object]:
                daemon_flags.append(threading.current_thread().daemon)
                started.set()
                try:
                    release.wait()
                finally:
                    finished.set()
                return {"operation_id": operation.operation_id, "ok": True}

            before = time.monotonic()
            try:
                with mock.patch.object(
                    operation,
                    "cancel_remote",
                    side_effect=_stuck_cleanup,
                ), mock.patch.object(
                    process_lease,
                    "_remote_process_lease_cancel_join_budget_s",
                    return_value=0.02,
                ):
                    reports = registry.cancel_all(grace_s=0.01)
            finally:
                release.set()
                finished.wait(timeout=1.0)

            self.assertLess(time.monotonic() - before, 0.5)
            self.assertTrue(started.is_set())
            self.assertEqual([True], daemon_flags)
            timeouts = [
                report
                for report in reports
                if report.get("error_code")
                == "remote_lease_cleanup_worker_timeout"
            ]
            self.assertEqual(1, len(timeouts), reports)
            self.assertEqual(REMOTE_LEASE_UNPROVEN_RC, timeouts[0]["returncode"])
            self.assertFalse(timeouts[0]["ok"])
            self.assertEqual(1, registry.active_count())
            self.assertEqual(
                1, len(list(journal_dir.glob("*.remote-lease.json")))
            )

    def test_cancel_all_cleans_registered_and_recovered_leases_after_tombstone_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-tombstone-registry-") as tmp:
            journal_dir = Path(tmp) / "journal"
            scope = RemoteProcessLeaseScope("fixture-run", "fixture-session")
            registry = RemoteProcessLeaseRegistry()
            journal = registry.configure_journal(
                scope=scope,
                journal_dir=journal_dir,
            )
            registered = scope.operation(
                label="registered", control_argv_prefix=["ssh", "fixture"]
            )
            recovered = scope.operation(
                label="recovered", control_argv_prefix=["ssh", "fixture"]
            )
            registry.register(registered)
            journal.record_operation(recovered)
            cancelled: list[str] = []

            def _proven_cleanup(
                operation: RemoteProcessLeaseOperation,
                *,
                grace_s: float = 3.0,
                launch_wait_s: float = 1.0,
                timeout_s: float | None = None,
            ) -> dict[str, object]:
                cancelled.append(operation.operation_id)
                return {
                    "operation_id": operation.operation_id,
                    "returncode": 0,
                    "output": "synthetic exact cleanup proof",
                    "elapsed_s": 0.0,
                    "ok": True,
                }

            with mock.patch.object(
                RemoteProcessLeaseOperation,
                "cancel_remote",
                autospec=True,
                side_effect=_proven_cleanup,
            ):
                with mock.patch.object(
                    journal,
                    "mark_cancelled",
                    side_effect=OSError("synthetic tombstone I/O failure"),
                ):
                    reports = registry.cancel_all(grace_s=0.01)

                self.assertTrue(registry.cancelled)
                self.assertEqual(
                    {registered.operation_id, recovered.operation_id},
                    set(cancelled),
                )
                tombstone_failures = [
                    report
                    for report in reports
                    if report.get("error_code")
                    == "remote_lease_tombstone_write_failed"
                ]
                self.assertEqual(1, len(tombstone_failures), reports)
                self.assertFalse(tombstone_failures[0]["ok"])
                self.assertEqual(2, registry.active_count())
                self.assertEqual(
                    2, len(list(journal_dir.glob("*.remote-lease.json")))
                )

                # A later pass may finish ownership only after the durable
                # session tombstone can be written successfully.
                retry_reports = registry.cancel_all(grace_s=0.01)

            self.assertTrue(all(report.get("ok") for report in retry_reports))
            self.assertEqual(0, registry.active_count())
            self.assertEqual([], list(journal_dir.glob("*.remote-lease.json")))
            self.assertTrue(journal.is_cancelled())

    def test_timeout_tombstone_error_still_exact_cleans_and_stops_local_ssh(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-tombstone-ssh-") as tmp:
            journal = RemoteProcessLeaseJournal(
                scope=RemoteProcessLeaseScope(
                    "fixture-run", "fixture-timeout-session"
                ),
                directory=Path(tmp) / "journal",
            )
            main_ssh = _TimedOutSSH()
            cleanup_report = {
                "operation_id": "synthetic",
                "returncode": 0,
                "output": (
                    "__SPLITPOINT_REMOTE_LEASE_CLEANUP__="
                    "terminated owned=1 remaining=0 drained=1"
                ),
                "elapsed_s": 0.0,
                "ok": True,
            }

            with mock.patch.object(
                RemoteProcessLeaseJournal,
                "mark_cancelled",
                autospec=True,
                side_effect=OSError("synthetic tombstone I/O failure"),
            ) as mark_cancelled, mock.patch.object(
                RemoteProcessLeaseOperation,
                "cancel_remote",
                autospec=True,
                return_value=cleanup_report,
            ) as cancel_remote, mock.patch.object(
                process_lease.subprocess,
                "Popen",
                return_value=main_ssh,
            ), mock.patch.object(
                process_lease,
                "_stop_local_ssh",
            ) as stop_local_ssh:
                returncode = run_journaled_ssh(
                    ["ssh", "fixture", "sleep 30"],
                    label="timeout-fixture",
                    env=journal.environment(),
                    timeout_s=0.0,
                )

            self.assertEqual(REMOTE_LEASE_UNPROVEN_RC, returncode)
            mark_cancelled.assert_called_once()
            cancel_remote.assert_called_once()
            stop_local_ssh.assert_called_once_with(main_ssh)
            self.assertEqual(
                1,
                len(list(journal.directory.glob("*.remote-lease.json"))),
            )
            self.assertFalse(journal.is_cancelled())


if __name__ == "__main__":
    unittest.main()
