"""Real local lease payloads; no SSH connection or accelerator is used."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys

import pytest

from onnx_splitpoint_tool.remote.process_lease import (
    RemoteProcessLeaseRegistry,
    RemoteProcessLeaseScope,
)
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport


ROOT = Path(__file__).resolve().parents[1]


class LocalLeaseTransport(SSHTransport):
    """Keep the product's lease and shell chain, replace only the SSH hop."""

    def __init__(self, root, *, leased=True):
        self.local_ssh = root / 'ssh'
        self.local_ssh.write_text(
            '#!/usr/bin/python3\n'
            'import os,sys\n'
            "start=sys.argv.index('test-user@fixture.invalid')+1\n"
            "os.execv('/bin/bash',['bash','-c',' '.join(sys.argv[start:])])\n"
        )
        self.local_ssh.chmod(0o755)
        scope = RemoteProcessLeaseScope('test-environment-run', 'test-session', str(root / 'remote-leases'))
        registry = RemoteProcessLeaseRegistry()
        registry.configure_journal(scope=scope, journal_dir=root / 'controller-journal')
        super().__init__(HostConfig(id='local-test', label='local-test', host='fixture.invalid', user='test-user'),
                         remote_lease_scope=scope if leased else None,
                         remote_lease_registry=registry if leased else None)
        self.registry = registry
        self.diagnostics_dir = root / 'diagnostics'

    def _ssh_base(self):
        normal_argv = super()._ssh_base()
        return [str(self.local_ssh), *normal_argv[1:]]


def environment_fixture(tmp_path):
    startup = tmp_path / 'controlled-login-startup.sh'
    startup.write_text("export PATH=/usr/bin:/bin\nexport SPLITPOINT_EXTRA_SITES=login-startup-value\n")
    injected_marker = tmp_path / 'must-not-execute'
    values = {
        'PATH': str(tmp_path / "declared directory's name") + ':/usr/bin:/bin',
        'SPLITPOINT_EXTRA_SITES': "/declared/site one:/declared/site'two",
        'BASH_ENV': str(startup),
        'QUOTED': "apostrophe' double\" dollar$(touch " + str(injected_marker) + ") backtick`true`\nnext line",
        'EMPTY': '',
        'ZERO': 0,
        'SKIPPED': None,
    }
    return values, injected_marker


def payload(environment, *, prefix=''):
    expected = {key: str(value) for key, value in environment.items() if value is not None}
    code = (
        'import json,os,sys\n'
        f'sys.path.insert(0,{str(ROOT)!r})\n'
        'from onnx_splitpoint_tool.runners.generic_energy import runtime_execution_identity\n'
        f'expected={expected!r}\n'
        "identity=runtime_execution_identity({'execution_environment':{'python_executable':sys.executable,'environment':expected}})\n"
        "print('OBSERVED='+json.dumps({key:os.environ.get(key) for key in expected},sort_keys=True))\n"
    )
    return prefix + shlex.join([sys.executable, '-B', '-c', code]), expected


def invoke(transport, method, command, environment):
    if method == 'capture':
        return transport.run(command, timeout=20, env=environment)
    lines = []
    rc = transport.run_streaming(command, on_line=lines.append, timeout=20, env=environment)
    return rc, '\n'.join(lines)


@pytest.mark.parametrize('method', ['capture', 'streaming'])
@pytest.mark.parametrize('leased', [True, False])
def test_exact_environment_survives_actual_login_and_lease_payload(tmp_path, method, leased):
    transport = LocalLeaseTransport(tmp_path, leased=leased)
    environment, injected = environment_fixture(tmp_path)
    command, expected = payload(environment)
    rc, output = invoke(transport, method, command, environment)
    assert rc == 0, output
    observed = [json.loads(line.split('=', 1)[1]) for line in output.splitlines() if line.startswith('OBSERVED=')]
    assert observed == [expected]
    assert not injected.exists()
    assert transport.registry.active_count() == 0
    assert not transport.registry.cancelled
    assert not list((tmp_path / 'controller-journal').glob('*.remote-lease.json'))
    assert not list((tmp_path / 'remote-leases').rglob('*.lease.json'))
    diagnostics = [json.loads(p.read_text()) for p in (tmp_path / 'diagnostics').glob('*.json')]
    assert diagnostics and all(x['command'] == command for x in diagnostics)


def test_read_only_keeps_direct_environment_and_creates_no_lease(tmp_path):
    transport = LocalLeaseTransport(tmp_path)
    environment, _ = environment_fixture(tmp_path)
    command, expected = payload(environment)
    rc, output = transport.run_read_only(command, timeout=20, env=environment)
    assert rc == 0, output
    assert json.loads(next(line.split('=', 1)[1] for line in output.splitlines() if line.startswith('OBSERVED='))) == expected
    assert transport.registry.active_count() == 0
    assert not list((tmp_path / 'remote-leases').rglob('*.json'))


@pytest.mark.parametrize('method', ['capture', 'streaming'])
def test_payload_assignments_remain_last_and_no_environment_is_invented(tmp_path, method, monkeypatch):
    monkeypatch.delenv('SKIPPED', raising=False)
    transport = LocalLeaseTransport(tmp_path)
    environment, _ = environment_fixture(tmp_path)
    command, expected = payload({**environment, 'EMPTY': 'payload-wins'})
    command = "export EMPTY=payload-wins; " + command
    rc, output = invoke(transport, method, command, environment)
    assert rc == 0, output
    observed = json.loads(next(line.split('=', 1)[1] for line in output.splitlines() if line.startswith('OBSERVED=')))
    assert observed == expected
    assert transport.registry.active_count() == 0


@pytest.mark.parametrize('method', ['capture', 'streaming'])
def test_no_explicit_environment_preserves_original_payload_and_exit_code(tmp_path, method):
    transport = LocalLeaseTransport(tmp_path)
    command = shlex.join([sys.executable, '-B', '-c', "print('UNCHANGED');raise SystemExit(7)"])
    rc, output = invoke(transport, method, command, None)
    assert rc == 7, output
    assert 'UNCHANGED' in output
    assert transport.registry.active_count() == 0
    assert not transport.registry.cancelled
    assert not list((tmp_path / 'controller-journal').glob('*.remote-lease.json'))
    assert not list((tmp_path / 'remote-leases').rglob('*.lease.json'))


def test_current_resume_command_contract_does_not_change(tmp_path):
    from onnx_splitpoint_tool.workflow.generic_only_cell_policy import expected_command
    row = {'remote_root': '/test/exact-execution', 'remote_python_executable': sys.executable,
           'runtime_config': {'config_directory': '/test/exact-execution/configs/model/b028'},
           'case_output_relative': 'results/setup/model/b028'}
    # This is the normal persisted caller command, not the transport envelope.
    command = expected_command(row, 1)
    transport = LocalLeaseTransport(tmp_path)
    operation = transport._begin_remote_lease(label='test-current-resume')
    try:
        envelope = operation.wrap_remote_command(command)
        assert shlex.split(envelope)[-1] == command
        assert expected_command(row, 1) == command
    finally:
        transport.registry.unregister(operation)
