#!/usr/bin/env python3
"""Fast local contract test for Native Energy's nested SSH broker route."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = ROOT / "scripts" / "native_producer_energy_plan.py"
PLAN_SCRIPT_MIRROR = (
    ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
    / "native_producer_energy_plan.py"
)
LEASE_ENV = (
    "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID",
    "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID",
    "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR",
    "ONNX_SPLITPOINT_REMOTE_LEASE_REMOTE_ROOT",
)


def _load_plan_module():
    spec = importlib.util.spec_from_file_location(
        "v2736_native_energy_plan_fixture", PLAN_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PLAN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeEnergyRemoteLeaseTests(unittest.TestCase):
    def test_generic_energy_ssh_uses_parent_journal_and_propagates_environment(self) -> None:
        from onnx_splitpoint_tool.benchmark import remote_run
        from onnx_splitpoint_tool.energy.collector import _run_one
        from onnx_splitpoint_tool.remote.process_lease import (
            RemoteProcessLeaseRegistry,
            RemoteProcessLeaseScope,
        )
        from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport

        with tempfile.TemporaryDirectory(prefix="v2736-generic-energy-lease-") as raw:
            root = Path(raw)
            registry = RemoteProcessLeaseRegistry()
            scope = RemoteProcessLeaseScope("run-fixture", "session-fixture")
            registry.configure_journal(scope=scope, journal_dir=root / "journal")
            transport = SSHTransport(
                HostConfig(id="fixture", label="fixture", host="fixture.invalid"),
                remote_lease_scope=scope,
                remote_lease_registry=registry,
            )
            command, lease_env = remote_run._journaled_energy_ssh_command(
                transport,
                registry,
                "printf workload",
                label="generic-energy-fixture",
                timeout_s=30.0,
            )
            argv = shlex.split(command)
            self.assertIn("onnx_splitpoint_tool.remote.process_lease_cli", argv)
            self.assertIn("--timeout-s", argv)
            self.assertEqual(argv[argv.index("--") + 1], "ssh")
            self.assertEqual(lease_env, registry.journal_environment())

            result = _run_one(
                [sys.executable, "-c", "import os; print(os.environ['P01_ENV'])"],
                cwd=None,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                subprocess_env={**dict(lease_env or {}), "P01_ENV": "visible"},
            )
            self.assertEqual(result["rc"], 0, result)
            self.assertEqual((root / "stdout.log").read_text().strip(), "visible")

        run_source = inspect.getsource(remote_run.run_remote_benchmark)
        self.assertGreaterEqual(run_source.count("_journaled_energy_ssh_command("), 4)
        self.assertGreaterEqual(run_source.count("subprocess_env=energy_lease_env"), 4)
        self.assertNotIn("ssh_argv = transport._ssh_cmd", run_source)

    def test_window_probe_wraps_preflight_and_repeated_workload_ssh(self) -> None:
        from onnx_splitpoint_tool import window_method_validation_probe as probe

        run_source = inspect.getsource(probe._run)
        replay_source = inspect.getsource(probe._hailo8_replay_command)
        self.assertIn("journaled_ssh_wrapper_argv", run_source)
        self.assertIn("journaled_ssh_wrapper_argv", replay_source)
        self.assertIn("lease_broker_command", replay_source)
        with mock.patch.object(
            probe,
            "cancel_journaled_remote_processes_from_environment",
            return_value=[],
        ) as remote_barrier:
            timed = probe._run(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=0.05,
            )
        self.assertEqual(timed["rc"], 124, timed)
        remote_barrier.assert_called_once()

    def test_generated_ssh_uses_broker_per_invocation_and_preserves_stdin(self) -> None:
        self.assertEqual(PLAN_SCRIPT.read_bytes(), PLAN_SCRIPT_MIRROR.read_bytes())
        plan = _load_plan_module()
        with tempfile.TemporaryDirectory(prefix="v2736-energy-lease-") as raw:
            root = Path(raw)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            ssh_log = root / "ssh.jsonl"
            broker_log = root / "broker.jsonl"
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['SSH_LOG'], 'a', encoding='utf-8') as h:\n"
                "    h.write(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()}) + '\\n')\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)

            # Shadow only the broker module.  It records the outer route and
            # then runs argv after '--', leaving fake SSH to prove exact argv
            # and stdin propagation without network access.
            fake_python = root / "fake_python"
            broker_module = (
                fake_python / "onnx_splitpoint_tool" / "remote"
                / "process_lease_cli.py"
            )
            broker_module.parent.mkdir(parents=True)
            (broker_module.parents[1] / "__init__.py").write_text("", encoding="utf-8")
            (broker_module.parent / "__init__.py").write_text("", encoding="utf-8")
            broker_module.write_text(
                "import json, os, subprocess, sys\n"
                "with open(os.environ['BROKER_LOG'], 'a', encoding='utf-8') as h:\n"
                "    h.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "split = sys.argv.index('--')\n"
                "raise SystemExit(subprocess.run(sys.argv[split + 1:]).returncode)\n",
                encoding="utf-8",
            )

            stdin_path = root / "contract with spaces.json"
            stdin_path.write_text('{"contract": true}\n', encoding="utf-8")
            remote_command = "printf '%s\\n' \"$HOME exact payload\""
            body = plan._lease_aware_ssh_script_body(
                ssh_target="nx@example.test",
                remote_command=remote_command,
                operation_label="native-energy-fixture",
                stdin_path=stdin_path,
                timeout_s=12.5,
            )
            generated = root / "generated command.sh"
            generated.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n" + body,
                encoding="utf-8",
            )
            generated.chmod(0o755)

            env = os.environ.copy()
            for name in LEASE_ENV:
                env.pop(name, None)
            env.update({
                "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
                "PYTHONPATH": str(fake_python),
                "SSH_LOG": str(ssh_log),
                "BROKER_LOG": str(broker_log),
            })

            direct = subprocess.run(
                [str(generated)], cwd=root, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertFalse(broker_log.exists())

            leased_env = dict(env)
            leased_env.update({
                "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID": "run-fixture",
                "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID": "session-fixture",
                "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR": str(root / "journal"),
            })
            for _ in range(2):
                leased = subprocess.run(
                    [str(generated)], cwd=root, env=leased_env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(leased.returncode, 0, leased.stderr)

            ssh_rows = [
                json.loads(line) for line in ssh_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(ssh_rows), 3)
            for row in ssh_rows:
                self.assertEqual(row["argv"][-2:], ["nx@example.test", remote_command])
                self.assertEqual(row["stdin"], '{"contract": true}\n')
            broker_rows = [
                json.loads(line)
                for line in broker_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(broker_rows), 2)
            self.assertTrue(all(row[:3] == ["exec", "--label", "native-energy-fixture"] for row in broker_rows))
            self.assertTrue(all(row[3:5] == ["--timeout-s", "12.5"] for row in broker_rows))
            self.assertTrue(all(row[5:7] == ["--", "ssh"] for row in broker_rows))

            partial_env = dict(env)
            partial_env["ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID"] = "run-only"
            partial = subprocess.run(
                [str(generated)], cwd=root, env=partial_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(partial.returncode, 64)
            self.assertIn("incomplete Native Energy remote lease environment", partial.stderr)


if __name__ == "__main__":
    unittest.main()
