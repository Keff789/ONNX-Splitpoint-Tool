"""Direct unittest entrypoint: no HOME changes or user shell-profile fixtures."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_result_collect_command, _remote_result_collect_script,
)


class CollectorNonloginBoundary(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='TEST_ONLY_nonlogin_collect_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suite = self.root / "TEST_ONLY suite 'quoted'"
        self.suite.mkdir()
        (self.suite / 'benchmark_plan.json').write_bytes(b'{"sample_count":5000}\n')

    def command(self, out):
        argv = shlex.split(_remote_result_collect_command(
            remote_suite_dir=str(self.suite), remote_results_dir=str(out)))
        self.assertEqual(argv[:4], ['bash', '--noprofile', '--norc', '-c'])
        self.assertEqual(argv[4], _remote_result_collect_script(
            remote_suite_dir=str(self.suite), remote_results_dir=str(out)))
        return argv

    def run_shell(self, argv):
        # Do not touch HOME, startup files or the parent environment. Prevent
        # unrelated noninteractive hooks in this own tiny subprocess fixture.
        env = {k:v for k,v in os.environ.items() if k not in ('BASH_ENV', 'ENV')}
        return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=5)

    def test_executed_shell_is_nonlogin_and_original_copy_bytes_are_preserved(self):
        out = self.root / 'copy'
        argv = self.command(out)
        argv[4] = "if shopt -q login_shell; then exit 91; fi; printf 'NONLOGIN_CONFIRMED\\n'; " + argv[4]
        result = self.run_shell(argv)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('NONLOGIN_CONFIRMED', result.stdout)
        self.assertIn('exit_trap_status=0 copy_failed=0', result.stderr)
        self.assertEqual((out/'benchmark_plan.json').read_bytes(),
                         (self.suite/'benchmark_plan.json').read_bytes())

    def test_real_copy_failure_stays_nonzero_and_keeps_diagnostic(self):
        argv = self.command(self.root/'failed_copy')
        argv[4] = 'cp() { return 23; }; ' + argv[4]
        result = self.run_shell(argv)
        self.assertEqual(result.returncode, 1)
        self.assertIn('SPLITPOINT_COLLECT_CP_FAILED rc=23', result.stderr)
        self.assertIn('exit_trap_status=1 copy_failed=1', result.stderr)

    def test_early_shell_failure_stays_nonzero_and_keeps_err_diagnostic(self):
        out=self.root/'not_a_directory';out.write_bytes(b'TEST_ONLY preserved')
        result=self.run_shell(self.command(out))
        self.assertEqual(result.returncode, 1)
        self.assertIn('SPLITPOINT_COLLECT_ERROR rc=1', result.stderr)
        self.assertIn('exit_trap_status=1 copy_failed=0', result.stderr)
        self.assertEqual(out.read_bytes(), b'TEST_ONLY preserved')

    def test_existing_collector_and_recovery_use_the_same_command_builder(self):
        for relative in ('onnx_splitpoint_tool/benchmark/remote_run.py',
                         'onnx_splitpoint_tool/remote/collection_recovery.py'):
            tree=ast.parse((ROOT/relative).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Name) and n.func.id=='_remote_result_collect_command']
            self.assertEqual(len(calls), 1, relative)


if __name__ == '__main__':
    unittest.main()
