"""Pure policy/real local ownership-object tests; no remote execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.remote.process_lease import (
    RemoteProcessLeaseRegistry, RemoteProcessLeaseScope,
)

from onnx_splitpoint_tool.workflow import generic_only_cell_policy as policy


@pytest.fixture
def case(tmp_path):
    root = '/session/original'
    row = {'case_key': 'one-exact-case', 'identity': {'model_id': 'fixture'},
        'runtime_config': {'execution_id': 'original-execution', 'config_directory': root + '/configs',
            'runtime_source_files': [{'path': root + '/runner.py', 'sha256': 'a' * 64}],
            'expected_runtime_artifacts': {'engine': {'path': '/cache/engine', 'sha256': 'b' * 64}},
            'input_binding': {'prepared_input': {'path': '/prepared/0.npy', 'sha256': 'c' * 64}},
            'mixed_input_control': {'path': root + '/control.json', 'sha256': 'd' * 64}},
        'remote_root': root, 'remote_python_executable': '/venv/bin/python',
        'case_output_relative': 'setup/model/case',
        'runtime_environment_comparison': {'status': 'observed_no_known_configuration_conflict'}}
    scope = RemoteProcessLeaseScope('actual-local-test-run', 'actual-local-test-session')
    registry = RemoteProcessLeaseRegistry()
    journal = registry.configure_journal(scope=scope, journal_dir=tmp_path / 'actual-journal')
    def admit(index, attempts=(), valid=(), current=None, control=None):
        return policy.admit_missing_repetition(current or row, row, index, valid, attempts,
            control=control or policy.CONTROL, initial_control=policy.CONTROL,
            registry=registry, journal=journal, expected_scope=scope)
    return row, registry, journal, admit


def attempt(row, index, status='exited'):
    return {'repetition_index': index, 'command': policy.expected_command(row, index),
        'started_at': '2026-10-21T00:00:00Z', 'status': status, **({'returncode': 0} if status == 'exited' else {})}


def test_preserves_valid_and_retries_only_missing_after_real_clean_journal(case):
    row, _, _, admit = case
    valid = [{'repetition_index': 1, 'execution_id': 'original-execution', 'status': 'ok'}]
    prior = [attempt(row, 1), attempt(row, 2)]
    proof = admit(2, prior, valid)
    assert proof['attempt_number'] == 2 and proof['preserved_valid_repetition_indices'] == [1]
    with pytest.raises(policy.InvalidResumeBinding, match='never be repeated'):
        admit(1, prior, valid)
    assert admit(3, prior, valid)['attempt_number'] == 1


def test_persistent_attempt_budget_survives_restart_and_running_reservations(case, tmp_path):
    row, _, _, admit = case
    result = tmp_path / 'result/generic_completion_results.json'
    result.parent.mkdir()
    path = result.with_name('dispatch_attempts.json')
    path.write_text(json.dumps({'attempts': [attempt(row, 2), attempt(row, 2, 'running')]}))
    for _restart in range(2):
        reread = policy.read_dispatch_attempts(result)
        with pytest.raises(policy.CellAttemptLimitReached):
            admit(2, reread)
    path.unlink()
    (result.parent / 'dispatch_001.log').write_text('Preserved normal launch log')
    with pytest.raises(policy.InvalidResumeBinding, match='persistent dispatch budget'):
        policy.read_dispatch_attempts(result)


@pytest.mark.parametrize('field', ['execution_id', 'runtime_source_files', 'expected_runtime_artifacts',
    'input_binding', 'mixed_input_control'])
def test_no_config_source_artifact_input_or_control_substitution(case, field):
    row, _, _, admit = case
    changed = deepcopy(row)
    changed['runtime_config'][field] = {'foreign': 'source'}
    with pytest.raises(policy.InvalidResumeBinding, match='Same execution ID/config/source'):
        admit(2, current=changed)


def test_budget_cannot_be_increased_on_reentry(case):
    _, _, _, admit = case
    with pytest.raises(policy.InvalidResumeBinding, match='immutable bounded cell policy'):
        admit(2, control={**policy.CONTROL, 'attempts_per_repetition_max': 3})


@pytest.mark.parametrize('fault', ['cancelled', 'unexpected-journal-file', 'configuration'])
def test_backend_or_source_failure_never_becomes_local_retry(case, fault):
    row, registry, journal, admit = case
    changed = deepcopy(row)
    if fault == 'cancelled':
        registry.poison()
    elif fault == 'unexpected-journal-file':
        (journal.directory / 'unresolved.json').write_text('{}')
    else:
        changed['runtime_environment_comparison']['status'] = 'conflict'
    expected = policy.InvalidResumeBinding if fault == 'configuration' else policy.UnresolvedOwnedProcesses
    with pytest.raises(expected):
        admit(2, current=changed)
