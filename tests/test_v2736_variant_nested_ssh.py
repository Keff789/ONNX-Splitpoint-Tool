#!/usr/bin/env python3
"""Short local acceptance test for the Native-variant SSH lease boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_update_module():
    path = ROOT / "scripts" / "update_evalset_native_producers.py"
    spec = importlib.util.spec_from_file_location(
        "_test_update_evalset_native_producers", path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VariantNestedSshLeaseTests(unittest.TestCase):
    def test_variant_ssh_uses_broker_only_with_complete_lease_environment(self) -> None:
        module = _load_update_module()
        packaged_mirror = (
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
            / "update_evalset_native_producers.py"
        )
        self.assertEqual(
            (ROOT / "scripts" / "update_evalset_native_producers.py").read_bytes(),
            packaged_mirror.read_bytes(),
        )
        with tempfile.TemporaryDirectory(prefix="splitpoint-variant-ssh-") as raw:
            temporary = Path(raw)
            fake_bin = temporary / "bin"
            fake_bin.mkdir()
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['FAKE_SSH_LOG'], 'a', encoding='utf-8') as h:\n"
                "    h.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)
            log = temporary / "ssh.log"
            command = ["ssh", "fixture-host", "printf remote-ok"]
            clean_env = {
                key: value for key, value in os.environ.items()
                if not key.startswith("ONNX_SPLITPOINT_REMOTE_LEASE_")
            }
            clean_env.update({
                "PATH": str(fake_bin) + os.pathsep + clean_env.get("PATH", ""),
                "FAKE_SSH_LOG": str(log),
            })

            with patch.dict(os.environ, clean_env, clear=True):
                legacy = module._run(
                    command, timeout=10, cwd=ROOT, label="variant-fixture",
                )
            self.assertEqual(legacy["rc"], 0)
            self.assertEqual(legacy["cmd"], command)

            lease_env = dict(clean_env)
            lease_env.update({
                "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID": "fixture-run",
                "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID": "fixture-session",
                "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR": str(
                    temporary / "journal"
                ),
                "ONNX_SPLITPOINT_REMOTE_LEASE_REMOTE_ROOT": str(
                    temporary / "remote-leases"
                ),
            })
            with patch.dict(os.environ, lease_env, clear=True):
                leased = module._run(
                    command, timeout=10, cwd=ROOT, label="variant-fixture",
                )
            self.assertEqual(leased["rc"], 0)
            self.assertEqual(
                leased["cmd"][:4],
                [
                    sys.executable, "-m",
                    "onnx_splitpoint_tool.remote.process_lease_cli", "exec",
                ],
            )
            timeout_index = leased["cmd"].index("--timeout-s")
            self.assertEqual(leased["cmd"][timeout_index + 1], "10.0")
            self.assertEqual(leased["cmd"][-len(command):], command)
            invocations = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(invocations), 2)
            self.assertEqual(
                invocations[0], ["fixture-host", "printf remote-ok"],
            )
            self.assertIn("fixture-host", invocations[1])
            self.assertIn("remote-process-lease", invocations[1][-1])

            incomplete_env = dict(clean_env)
            incomplete_env[
                "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID"
            ] = "fixture-run"
            with patch.dict(os.environ, incomplete_env, clear=True):
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "(?i)incomplete|missing",
                ):
                    module._run(
                        command, timeout=10, cwd=ROOT,
                        label="variant-fixture",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
