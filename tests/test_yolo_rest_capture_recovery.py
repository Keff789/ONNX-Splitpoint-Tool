"""The existing physical recovery reads the actual bounded rest journal."""
from pathlib import Path
import json
import shutil

import pytest

from test_physical_resource_recovery import recovery_case, _fences, _write, _recovery_cli_argv
from onnx_splitpoint_tool.remote import resource_recovery, process_lease


@pytest.fixture
def rest_case(recovery_case):
    case = recovery_case
    output = case.run_dir.parent / 'augmentation'
    attempt = output / 'attempts' / case.kwargs['session_id']
    attempt.mkdir(parents=True)
    journal = attempt / 'remote-process-leases'
    shutil.move(case.journal, journal)
    plan = output / 'plan.json'
    _write(plan, {'output_root': str(output), 'augmentation_run': str(case.run_dir)})
    _write(attempt / 'production-start.json', {'pid': 99999993, 'plan': str(plan),
        'started_at': '2026-09-23T00:00:00+00:00', 'automatic_restart': False})
    _write(attempt / 'terminal-status.json', {'status': 'stopped', 'exitcode': 1,
        'finished_at': '2026-09-23T00:05:00+00:00', 'automatic_restart': False})
    _write(attempt / 'cleanup-status.json', {'at': '2026-09-23T00:04:59+00:00',
        'active_owned_remote_operations': 0, 'owned_local_processes': 0,
        'foreign_processes_stopped': False, 'broker_errors': ['source stopped']})
    (attempt / 'exitcode.txt').write_text('1\n')
    manifest = case.run_dir / 'run_manifest.json'
    payload = json.loads(manifest.read_text()); payload['status'] = 'partial'
    _write(manifest, payload)
    case.kwargs['rest_attempt_dir'] = attempt
    case.rest_attempt = attempt
    case.rest_plan = plan
    case.reply_path = journal / case.reply_path.name
    case.journal = journal
    return case


def test_actual_rest_journal_release_preserves_old_partial_manifest_and_stop(rest_case):
    case = rest_case
    manifest = case.run_dir / 'run_manifest.json'
    before = manifest.read_bytes()
    reply_before = json.loads(case.reply_path.read_text())
    result = resource_recovery.recover_capture_resources(**case.kwargs)
    assert result['status'] == 'released'
    reply = json.loads(case.reply_path.read_text())
    assert all(reply[k] == v for k, v in reply_before.items())
    assert manifest.read_bytes() == before
    assert reply['recovery']['old_attempt_validated'] is False
    assert result['measurements_started'] == 0


@pytest.mark.parametrize('kind', ['wrong_run', 'wrong_output', 'wrong_parent_pid', 'not_terminal',
    'wrong_exitcode', 'missing_cleanup', 'pending_process', 'cleanup_after_terminal', 'foreign_session'])
def test_rest_adapter_requires_exact_existing_terminal_bindings(rest_case, kind):
    case = rest_case
    if kind in {'wrong_run', 'wrong_output'}:
        path = case.rest_plan
        value = json.loads(path.read_text())
        value['augmentation_run' if kind == 'wrong_run' else 'output_root'] = '/unrelated'
    elif kind == 'wrong_parent_pid':
        path = case.rest_attempt / 'production-start.json'
        value = json.loads(path.read_text()); value['pid'] = 99999800
    elif kind == 'not_terminal':
        path = case.rest_attempt / 'terminal-status.json'
        value = json.loads(path.read_text()); value['status'] = 'running'
    elif kind == 'wrong_exitcode':
        (case.rest_attempt / 'exitcode.txt').write_text('0')
        path = None
    elif kind == 'missing_cleanup':
        (case.rest_attempt / 'cleanup-status.json').unlink(); path = None
    elif kind == 'foreign_session':
        case.kwargs['session_id'] = 'other'; path = None
    else:
        path = case.rest_attempt / 'cleanup-status.json'
        value = json.loads(path.read_text())
        value['owned_local_processes' if kind == 'pending_process' else 'at'] = (
            1 if kind == 'pending_process' else '2026-09-23T00:06:00+00:00')
    if path:
        _write(path, value)
    before = _fences(case)
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == before
    assert 'recovery' not in json.loads(case.reply_path.read_text())


def test_rest_evidence_change_during_owner_probe_keeps_fences(rest_case, monkeypatch):
    case = rest_case
    def observe(*args, **kwargs):
        path = case.rest_attempt / 'terminal-status.json'
        value = json.loads(path.read_text()); value['exitcode'] = 0; _write(path, value)
        return {'ok': True}
    monkeypatch.setattr(resource_recovery, 'observe_resource_owners', observe)
    before = _fences(case)
    with pytest.raises(Exception, match='changed during observation'):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == before


def test_cli_passes_explicit_rest_attempt_only_when_supplied(recovery_case, monkeypatch):
    calls = []
    monkeypatch.setattr(resource_recovery, 'recover_capture_resources',
                        lambda **kwargs: calls.append(kwargs) or {'status': 'released'})
    argv = _recovery_cli_argv(recovery_case) + ['--confirm-action-performed',
        '--rest-attempt-dir', '/actual/attempt']
    assert process_lease.process_lease_cli_main(argv) == 0
    assert calls[0]['rest_attempt_dir'] == '/actual/attempt'
