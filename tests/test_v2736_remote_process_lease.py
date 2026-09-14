from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.remote.process_lease import (
    REMOTE_LEASE_ENV_JOURNAL_DIR,
    REMOTE_LEASE_ENV_RUN_ID,
    REMOTE_LEASE_ENV_SESSION_ID,
    REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
    RemoteProcessLeaseJournal,
    RemoteProcessLeaseJournalError,
    RemoteProcessLeaseLaunchRejected,
    RemoteProcessLeaseOperation,
    RemoteProcessLeaseRegistry,
    RemoteProcessLeaseScope,
    bind_remote_process_registry,
    current_remote_process_registry,
    harden_remote_process_lease_ssh_argv,
    journaled_ssh_wrapper_argv,
    resolve_remote_process_lease_scope,
    remote_process_lease_cleanup_timeout_s,
    validate_remote_process_lease_ssh_prefix,
)
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport


def _procfs_matches_self() -> bool:
    try:
        recorded = int(Path("/proc/self/stat").read_text().split("(", 1)[0].strip())
    except (OSError, ValueError):
        return False
    return recorded == os.getpid()


def _live_identity(pid: int, start_time: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        close = text.rfind(")")
        fields = text[close + 2 :].split()
        return fields[0] != "Z" and int(fields[19]) == int(start_time)
    except (OSError, ValueError, IndexError):
        return False


class RemoteProcessLeaseTests(unittest.TestCase):
    def test_parent_scope_overrides_remote_benchmark_subrun_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-scope-") as tmp:
            parent_scope = RemoteProcessLeaseScope(
                "evaluation-run", "workflow-session"
            )
            registry = RemoteProcessLeaseRegistry()
            registry.configure_journal(
                scope=parent_scope,
                journal_dir=Path(tmp) / "journal",
            )

            resolved = resolve_remote_process_lease_scope(
                registry=registry,
                fallback_run_id="evaluation-run_model_target",
                workflow_session_id="workflow-session",
            )
            self.assertIs(resolved, parent_scope)
            operation = resolved.operation(  # type: ignore[union-attr]
                label="subrun", control_argv_prefix=["ssh", "fixture"]
            )
            registry.register(operation)
            self.assertEqual(registry.active_count(), 1)
            registry.unregister(operation)
            self.assertEqual(registry.active_count(), 0)

    def test_command_contract_is_exact_and_has_no_broad_kill(self) -> None:
        scope = RemoteProcessLeaseScope("fixture-run", "fixture-session")
        operation = scope.operation(
            label="native-energy",
            control_argv_prefix=["ssh", "fixture-host"],
        )
        launch = operation.wrap_remote_command("echo payload")
        cleanup = operation.cleanup_remote_command(grace_s=0.1)

        self.assertIn(scope.run_sha256, launch)
        self.assertIn(scope.session_sha256, launch)
        self.assertIn(operation.operation_id, cleanup)
        self.assertIn(operation.token, cleanup)
        self.assertIn("start_time_ticks", launch)
        self.assertIn("exact_cancel_requested", launch)
        self.assertIn("PR_SET_CHILD_SUBREAPER", launch)
        self.assertIn("guardian-ready", launch)
        self.assertIn("ready.touch", launch)
        self.assertIn("direct_children", launch)
        self.assertIn("os.waitpid", launch)
        self.assertIn("drained_ok", launch)
        self.assertIn("root_exited_without_drain", cleanup)
        self.assertIn("already_exited_drained", cleanup)
        self.assertIn('current.get("token") == token and drained_ok', launch)
        self.assertIn('.strip() == token', launch)
        self.assertIn('pathlib.Path("/tmp")', launch)
        self.assertIn('pathlib.Path("/tmp")', cleanup)
        self.assertNotIn('os.environ.get("XDG_RUNTIME_DIR")', launch)
        self.assertNotIn('os.environ.get("TMPDIR")', cleanup)
        self.assertIn('pathlib.Path("/proc/%d/stat" % pid)', cleanup)
        self.assertIn('pathlib.Path("/proc/%d/task" % parent_pid)', cleanup)
        self.assertNotIn("pkill", cleanup.lower())
        self.assertNotIn("pgrep", cleanup.lower())
        self.assertNotIn('pathlib.Path("/proc").iterdir', cleanup)
        self.assertIn("def signal_descendants(sig):", cleanup)
        self.assertIn("if pid == root_pid or not exact_alive(item):", cleanup)
        self.assertIn("and drain_proven():", cleanup)
        self.assertNotIn("signal_exact(signal.SIGKILL)", cleanup)
        remaining = cleanup.index("remaining = alive_count()")
        conditional = cleanup.index("if remaining == 0 and drained_ok:", remaining)
        unlink = cleanup.index("lease.unlink()", conditional)
        self.assertLess(conditional, unlink)
        self.assertEqual(
            15.0,
            remote_process_lease_cleanup_timeout_s(
                grace_s=3.0,
                launch_wait_s=1.0,
            ),
        )
        self.assertGreaterEqual(
            REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
            remote_process_lease_cleanup_timeout_s() + 2.0,
        )

    def test_remote_registry_context_binding_is_scoped(self) -> None:
        registry = RemoteProcessLeaseRegistry()
        self.assertIsNone(current_remote_process_registry())
        with bind_remote_process_registry(registry):
            self.assertIs(registry, current_remote_process_registry())
        self.assertIsNone(current_remote_process_registry())

    def test_hailo_owned_raw_ssh_uses_bound_journal_broker(self) -> None:
        from onnx_splitpoint_tool.hailo_backend import _run_owned_subprocess
        from onnx_splitpoint_tool.process_control import (
            ProcessTreeRegistry,
            bind_process_registry,
        )

        with tempfile.TemporaryDirectory(prefix="remote-lease-hailo-ssh-") as tmp:
            fixture = Path(tmp)
            fake_ssh = fixture / "ssh"
            argv_log = fixture / "ssh-argv.json"
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,pathlib,sys\n"
                "pathlib.Path(os.environ['SSH_ARGV_LOG']).write_text(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)
            remote_registry = RemoteProcessLeaseRegistry()
            remote_registry.configure_journal(
                scope=RemoteProcessLeaseScope("fixture-run", "fixture-session"),
                journal_dir=fixture / "journal",
            )
            local_registry = ProcessTreeRegistry()
            with bind_process_registry(
                local_registry
            ), bind_remote_process_registry(remote_registry):
                completed = _run_owned_subprocess(
                    [str(fake_ssh), "fixture-host", "true"],
                    text=True,
                    capture_output=True,
                    timeout=1.0,
                    env={"SSH_ARGV_LOG": str(argv_log)},
                )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn(
                "onnx_splitpoint_tool.remote.process_lease_cli",
                list(completed.args),
            )
            self.assertTrue(argv_log.is_file())
            launched = " ".join(json.loads(argv_log.read_text()))
            self.assertIn("onnx-splitpoint/remote-process-lease", launched)
            self.assertEqual(0, remote_registry.active_count())

    @unittest.skipUnless(
        os.name == "posix" and _procfs_matches_self(),
        "requires Linux procfs mounted for the active PID namespace",
    )
    def test_exact_cancel_kills_separately_sessioned_grandchild(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-fixture-") as tmp:
            fixture = Path(tmp)
            lease_root = fixture / "leases"
            root_marker = fixture / "root.json"
            grandchild_marker = fixture / "grandchild.json"

            grandchild_code = (
                "import json,os,pathlib,signal,time; "
                f"p=pathlib.Path({str(grandchild_marker)!r}); "
                "s=pathlib.Path('/proc/%d/stat'%os.getpid()).read_text(); "
                "f=s[s.rfind(')')+2:].split(); "
                "p.write_text(json.dumps({'pid':os.getpid(),'start':int(f[19])})); "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
            )
            root_code = (
                "import json,os,pathlib,signal,subprocess,sys,time; "
                f"marker=pathlib.Path({str(root_marker)!r}); "
                f"child=subprocess.Popen([{sys.executable!r},'-c',{grandchild_code!r}],start_new_session=True); "
                "s=pathlib.Path('/proc/%d/stat'%os.getpid()).read_text(); "
                "f=s[s.rfind(')')+2:].split(); "
                "marker.write_text(json.dumps({'pid':os.getpid(),'start':int(f[19]),'child':child.pid})); "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
            )
            payload = shlex.join([sys.executable, "-c", root_code])
            scope = RemoteProcessLeaseScope(
                "fixture-run", "fixture-session", str(lease_root)
            )
            operation = scope.operation(
                label="setsid-grandchild",
                control_argv_prefix=["bash", "-lc"],
            )
            registry = RemoteProcessLeaseRegistry()
            registry.register(operation)
            wrong_before_start = RemoteProcessLeaseOperation(
                scope=scope,
                operation_id=operation.operation_id,
                token="f" * 64,
                control_argv_prefix=["bash", "-lc"],
            )
            wrong_pre_report = wrong_before_start.cancel_remote(
                grace_s=0.01, launch_wait_s=0.01, timeout_s=3.0
            )
            self.assertTrue(wrong_pre_report["ok"], wrong_pre_report)
            launcher = subprocess.Popen(
                ["bash", "-lc", operation.wrap_remote_command(payload)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.addCleanup(
                lambda: launcher.poll() is None and os.killpg(launcher.pid, 9)
            )

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if root_marker.is_file() and grandchild_marker.is_file():
                    break
                time.sleep(0.02)
            self.assertTrue(root_marker.is_file())
            self.assertTrue(grandchild_marker.is_file())
            root_identity = json.loads(root_marker.read_text())
            grandchild_identity = json.loads(grandchild_marker.read_text())

            # Same operation path but a different token must never signal the
            # leased processes.
            wrong = RemoteProcessLeaseOperation(
                scope=scope,
                operation_id=operation.operation_id,
                token="0" * 64,
                control_argv_prefix=["bash", "-lc"],
            )
            wrong_report = wrong.cancel_remote(
                grace_s=0.05, launch_wait_s=0.05, timeout_s=3.0
            )
            self.assertEqual(66, wrong_report["returncode"])
            self.assertTrue(
                _live_identity(root_identity["pid"], root_identity["start"])
            )

            report = operation.cancel_remote(
                grace_s=0.15, launch_wait_s=0.1, timeout_s=5.0
            )
            self.assertTrue(report["ok"], report)
            launcher.wait(timeout=5.0)
            self.assertFalse(
                _live_identity(root_identity["pid"], root_identity["start"])
            )
            self.assertFalse(
                _live_identity(
                    grandchild_identity["pid"], grandchild_identity["start"]
                )
            )

    @unittest.skipUnless(
        os.name == "posix" and _procfs_matches_self(),
        "requires Linux procfs mounted for the active PID namespace",
    )
    def test_guardian_survives_term_fork_churn_until_descendants_drain(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-churn-") as tmp:
            fixture = Path(tmp)
            churn_marker = fixture / "churn.json"
            fork_marker = fixture / "late-fork.json"
            churn_code = "\n".join(
                [
                    "import json,os,pathlib,signal,time",
                    "def ident():",
                    "    s=pathlib.Path('/proc/%d/stat'%os.getpid()).read_text()",
                    "    f=s[s.rfind(')')+2:].split()",
                    "    return {'pid':os.getpid(),'start':int(f[19])}",
                    f"pathlib.Path({str(churn_marker)!r}).write_text(json.dumps(ident()))",
                    "def on_term(_signum,_frame):",
                    "    pid=os.fork()",
                    "    if pid == 0:",
                    "        os.setsid()",
                    "        signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                    f"        pathlib.Path({str(fork_marker)!r}).write_text(json.dumps(ident()))",
                    "        time.sleep(30)",
                    "        os._exit(0)",
                    "    os._exit(0)",
                    "signal.signal(signal.SIGTERM,on_term)",
                    "time.sleep(30)",
                ]
            )
            payload_code = "\n".join(
                [
                    "import subprocess,sys,time",
                    f"subprocess.Popen([{sys.executable!r},'-c',{churn_code!r}], start_new_session=True)",
                    "time.sleep(30)",
                ]
            )
            operation = RemoteProcessLeaseScope(
                "fixture-run",
                "fixture-session",
                str(fixture / "leases"),
            ).operation(
                label="term-fork-churn",
                control_argv_prefix=["bash", "-lc"],
            )
            launcher = subprocess.Popen(
                [
                    "bash",
                    "-lc",
                    operation.wrap_remote_command(
                        shlex.join([sys.executable, "-c", payload_code])
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.addCleanup(
                lambda: launcher.poll() is None and os.killpg(launcher.pid, 9)
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not churn_marker.is_file():
                time.sleep(0.02)
            self.assertTrue(churn_marker.is_file())

            report = operation.cancel_remote(
                grace_s=0.3,
                launch_wait_s=0.1,
                timeout_s=8.0,
            )
            self.assertTrue(report["ok"], report)
            launcher.wait(timeout=5.0)
            self.assertTrue(fork_marker.is_file(), report)
            churn_identity = json.loads(churn_marker.read_text())
            fork_identity = json.loads(fork_marker.read_text())
            self.assertFalse(
                _live_identity(churn_identity["pid"], churn_identity["start"])
            )
            self.assertFalse(
                _live_identity(fork_identity["pid"], fork_identity["start"])
            )

    def test_sticky_cancel_tombstones_late_operation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-late-") as tmp:
            marker = Path(tmp) / "must-not-run"
            scope = RemoteProcessLeaseScope(
                "fixture-run", "fixture-session", str(Path(tmp) / "leases")
            )
            operation = scope.operation(
                label="late", control_argv_prefix=["bash", "-lc"]
            )
            registry = RemoteProcessLeaseRegistry()
            registry.cancel_all(grace_s=0.01)
            with self.assertRaises(RemoteProcessLeaseLaunchRejected) as rejected:
                registry.register(operation)
            self.assertEqual(130, rejected.exception.returncode)
            self.assertFalse(marker.exists())

    def test_terminal_register_and_detaching_ssh_options_block_main_launch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-start-barrier-") as tmp:
            fixture = Path(tmp)
            fake_bin = fixture / "bin"
            fake_bin.mkdir()
            main_marker = fixture / "main-launched"
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import os,pathlib,sys\n"
                "joined = ' '.join(sys.argv[1:])\n"
                "if '__SPLITPOINT_REMOTE_LEASE_CLEANUP__' in joined:\n"
                "    raise SystemExit(23)\n"
                "pathlib.Path(os.environ['MAIN_LAUNCH_MARKER']).write_text('launched')\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)
            environment = {
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "MAIN_LAUNCH_MARKER": str(main_marker),
            }

            scope = RemoteProcessLeaseScope("fixture-run", "fixture-session")
            registry = RemoteProcessLeaseRegistry()
            registry.configure_journal(
                scope=scope,
                journal_dir=fixture / "cancelled-journal",
            )
            registry.poison()
            transport = SSHTransport(
                HostConfig(id="fixture", label="fixture", host="fixture.invalid"),
                remote_lease_scope=scope,
                remote_lease_registry=registry,
            )
            with mock.patch.dict(os.environ, environment):
                rc, output = transport.run("printf forbidden")
            self.assertEqual(70, rc, output)
            self.assertFalse(main_marker.exists())
            self.assertEqual(1, registry.active_count())
            self.assertEqual(
                1,
                len(
                    list(
                        (fixture / "cancelled-journal").glob(
                            "*.remote-lease.json"
                        )
                    )
                ),
            )

            option_registry = RemoteProcessLeaseRegistry()
            option_scope = RemoteProcessLeaseScope("option-run", "option-session")
            option_registry.configure_journal(
                scope=option_scope,
                journal_dir=fixture / "option-journal",
            )
            detached = SSHTransport(
                HostConfig(
                    id="fixture-f",
                    label="fixture-f",
                    host="fixture.invalid",
                    ssh_extra_args="-f",
                ),
                remote_lease_scope=option_scope,
                remote_lease_registry=option_registry,
            )
            with mock.patch.dict(os.environ, environment):
                rc, output = detached.run("printf forbidden")
            self.assertEqual(64, rc, output)
            self.assertFalse(main_marker.exists())
            self.assertEqual(0, option_registry.active_count())

            for unsafe in (
                ["ssh", "-N", "fixture"],
                ["ssh", "-o", "ForkAfterAuthentication=yes", "fixture"],
                ["ssh", "-o", "ForkAfterAuthentication yes", "fixture"],
                ["ssh", "-oSessionType=none", "fixture"],
                ["ssh", "-o", "ControlPersist=30s", "fixture"],
                ["ssh", "-oControlMaster=auto", "fixture"],
            ):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(ValueError):
                        validate_remote_process_lease_ssh_prefix(unsafe)
            validate_remote_process_lease_ssh_prefix(
                [
                    "ssh",
                    "-oForkAfterAuthentication=no",
                    "-oSessionType=default",
                    "-oControlPersist=no",
                    "-oControlMaster=no",
                    "fixture",
                ]
            )
            hardened = harden_remote_process_lease_ssh_argv(
                ["ssh", "-F", "/tmp/fixture-config", "fixture"]
            )
            for forced in (
                "ForkAfterAuthentication=no",
                "SessionType=default",
                "ControlMaster=no",
                "ControlPersist=no",
            ):
                self.assertIn(forced, hardened)
            self.assertEqual(hardened, harden_remote_process_lease_ssh_argv(hardened))

    def test_journal_scan_ignores_descriptor_removed_after_glob(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-scan-race-") as tmp:
            journal = RemoteProcessLeaseJournal(
                scope=RemoteProcessLeaseScope("fixture-run", "fixture-session"),
                directory=Path(tmp) / "journal",
            )
            operation = journal.scope.operation(
                label="vanishing", control_argv_prefix=["ssh", "fixture"]
            )
            descriptor = journal.record_operation(operation)

            def _vanish(path: Path):
                path.unlink()
                raise RemoteProcessLeaseJournalError("synthetic ENOENT race")

            with mock.patch.object(journal, "_read_json_exact", side_effect=_vanish):
                self.assertEqual([], journal._scan_with_errors())
            self.assertFalse(descriptor.exists())
            self.assertEqual(0, journal.pending_count())

    def test_failed_control_cleanup_stays_registered_for_final_retry(self) -> None:
        scope = RemoteProcessLeaseScope("fixture-run", "fixture-session")
        unproven = scope.operation(
            label="rc-zero-without-proof",
            control_argv_prefix=[sys.executable, "-c", "print('not-a-proof')"],
        )
        unproven_report = unproven.cancel_remote(timeout_s=2.0)
        self.assertEqual(0, unproven_report["returncode"])
        self.assertFalse(unproven_report["ok"])

        registry = RemoteProcessLeaseRegistry()
        transport = SSHTransport(
            HostConfig(id="fixture", label="fixture", host="fixture.invalid"),
            remote_lease_scope=scope,
            remote_lease_registry=registry,
        )
        operation = scope.operation(
            label="failed-control", control_argv_prefix=["ssh", "fixture.invalid"]
        )
        registry.register(operation)
        with mock.patch.object(
            operation, "cancel_remote", return_value={"ok": False, "returncode": 255}
        ):
            cleanup_proven = transport._finish_remote_lease(
                operation, abnormal=True
            )
        self.assertFalse(cleanup_proven)
        self.assertEqual(1, registry.active_count())
        self.assertTrue(registry.cancelled)

        with mock.patch.object(
            operation, "cancel_remote", return_value={"ok": True, "returncode": 0}
        ):
            cleanup_proven = transport._finish_remote_lease(
                operation, abnormal=True
            )
        self.assertTrue(cleanup_proven)
        self.assertEqual(0, registry.active_count())

    def test_parent_recovers_nested_journal_and_retains_failed_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-journal-") as tmp:
            journal_dir = Path(tmp) / "journal"
            scope = RemoteProcessLeaseScope("fixture-run", "fixture-session")
            registry = RemoteProcessLeaseRegistry()
            registry.configure_journal(scope=scope, journal_dir=journal_dir)
            child_env = dict(os.environ)
            child_env.update(registry.journal_environment())
            child_code = "\n".join(
                [
                    "from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal",
                    "journal = RemoteProcessLeaseJournal.from_environment(required=True)",
                    "operation = journal.scope.operation(label='nested', control_argv_prefix=['ssh','fixture'])",
                    "journal.record_operation(operation)",
                ]
            )
            completed = subprocess.run(
                [sys.executable, "-c", child_code],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5.0,
            )
            self.assertEqual(0, completed.returncode, completed.stdout)
            self.assertEqual(1, registry.active_count())

            failed = {
                "operation_id": "nested",
                "returncode": 255,
                "output": "synthetic control failure",
                "ok": False,
            }
            with mock.patch.object(
                RemoteProcessLeaseOperation,
                "cancel_remote",
                autospec=True,
                return_value=failed,
            ):
                reports = registry.cancel_all(grace_s=0.01)
            self.assertFalse(reports[0]["ok"])
            self.assertEqual(1, registry.active_count())
            self.assertEqual(1, len(list(journal_dir.glob("*.remote-lease.json"))))

            proven = {
                "operation_id": "nested",
                "returncode": 0,
                "output": "terminated",
                "ok": True,
            }
            with mock.patch.object(
                RemoteProcessLeaseOperation,
                "cancel_remote",
                autospec=True,
                return_value=proven,
            ):
                reports = registry.cancel_all(grace_s=0.01)
            self.assertTrue(reports[0]["ok"])
            self.assertEqual(0, registry.active_count())
            self.assertEqual([], list(journal_dir.glob("*.remote-lease.json")))

    @unittest.skipUnless(os.name == "posix", "requires POSIX signals")
    def test_warning_free_cli_cleans_descriptor_before_local_signal_exit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-lease-cli-") as tmp:
            fixture = Path(tmp)
            fake_ssh = fixture / "ssh"
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import sys,time\n"
                "payload = sys.argv[-1] if len(sys.argv) > 1 else ''\n"
                "if '__SPLITPOINT_REMOTE_LEASE_CLEANUP__' in payload:\n"
                "    print('__SPLITPOINT_REMOTE_LEASE_CLEANUP__=terminated owned=1 remaining=0 drained=1')\n"
                "    raise SystemExit(0)\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)
            journal = RemoteProcessLeaseJournal(
                scope=RemoteProcessLeaseScope("fixture-run", "fixture-session"),
                directory=fixture / "journal",
            )
            child_env = dict(os.environ)
            child_env.update(journal.environment())
            wrapper = journaled_ssh_wrapper_argv(
                [str(fake_ssh), "fixture-host", "sleep-payload"],
                label="signal-fixture",
                env=child_env,
            )
            proc = subprocess.Popen(
                wrapper,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.addCleanup(lambda: proc.poll() is None and proc.kill())
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if list(journal.directory.glob("*.remote-lease.json")):
                    break
                time.sleep(0.02)
            self.assertTrue(list(journal.directory.glob("*.remote-lease.json")))
            proc.send_signal(signal.SIGTERM)
            output, _ = proc.communicate(timeout=8.0)
            self.assertEqual(130, proc.returncode, output)
            self.assertNotIn("RuntimeWarning", output)
            self.assertEqual([], list(journal.directory.glob("*.remote-lease.json")))
            self.assertTrue(journal.is_cancelled())

            timeout_journal = RemoteProcessLeaseJournal(
                scope=RemoteProcessLeaseScope(
                    "fixture-run", "fixture-timeout-session"
                ),
                directory=fixture / "timeout-journal",
            )
            timeout_env = dict(os.environ)
            timeout_env.update(timeout_journal.environment())
            timed_wrapper = journaled_ssh_wrapper_argv(
                [str(fake_ssh), "fixture-host", "timeout-payload"],
                label="timeout-fixture",
                env=timeout_env,
                timeout_s=0.05,
            )
            timed = subprocess.run(
                timed_wrapper,
                env=timeout_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=8.0,
            )
            self.assertEqual(124, timed.returncode, timed.stdout)
            self.assertNotIn("RuntimeWarning", timed.stdout)
            self.assertEqual(
                [], list(timeout_journal.directory.glob("*.remote-lease.json"))
            )
            self.assertTrue(timeout_journal.is_cancelled())

    def test_wrapper_is_opt_in_and_partial_environment_fails_closed(self) -> None:
        local = ["rsync", "source", "target"]
        ssh = ["ssh", "fixture", "true"]
        self.assertEqual(local, journaled_ssh_wrapper_argv(local, label="local"))
        self.assertEqual(ssh, journaled_ssh_wrapper_argv(ssh, label="standalone", env={}))
        with self.assertRaisesRegex(Exception, "incomplete"):
            journaled_ssh_wrapper_argv(
                ssh,
                label="partial",
                env={REMOTE_LEASE_ENV_RUN_ID: "run"},
            )


if __name__ == "__main__":
    unittest.main()
