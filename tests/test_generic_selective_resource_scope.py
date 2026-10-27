"""Selective legacy Generic admission through real held local owner proofs."""
import csv
import json
import tarfile

import pytest

from onnx_splitpoint_tool.workflow import generic_completed_task as entry, run_control
from test_generic_completion_dispatch import LocalTransport, prepared_case
from test_generic_completed_task_resources import workflow, idle, target


def selective_plan(tmp_path, monkeypatch):
    primary = tmp_path / 'primary'
    output = tmp_path / 'output'
    primary.mkdir()
    output.mkdir()
    row = prepared_case(tmp_path, setup='fps')
    entry.write_json(primary / 'run_manifest.json', {'hardware_targets': [target(setup_id='fps')]})
    entry.write_json(primary / 'reports/native_host_telemetry_summary.json', {'captures': []})
    scope = tmp_path / 'scope.csv'
    with scope.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row['identity']))
        writer.writeheader()
        writer.writerow(row['identity'])
    historical = tmp_path / 'historical-plan.json'
    entry.write_json(historical, {'rows': [row]})
    with tarfile.open(output / 'runner-bundle.tar.gz', 'w:gz'):
        pass
    plan = {'scope_mode': 'audited_selective_remeasurement', 'authorized_row_count': 1,
        'rows': [row], 'execution_id': 'new-session', 'original_run': str(primary),
        'output_root': str(output), 'scope_source': str(scope),
        'scope_source_sha256': entry.file_sha256(scope), 'start_case_keys': [row['case_key']],
        'source_execution_plans': [{'path': str(historical), 'sha256': entry.file_sha256(historical)}]}
    path = output / 'plan/execution-plan.json'
    entry.write_json(path, plan)
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda: tmp_path / 'locks/platform.lock')

    class OwnerTransport(LocalTransport):
        def run_read_only(self, command, **kwargs):
            if 'GENERIC_COMPLETION_OWNER_PROBE' in command:
                return 0, 'OWNERS=' + json.dumps(idle())
            return super().run_read_only(command, **kwargs)

    transport = OwnerTransport()
    monkeypatch.setattr(entry, 'transport_for', lambda *_, **__: transport)
    return path, output, row, transport


def test_selective_legacy_runs_beside_proven_independent_h10_and_central_work(
        workflow, tmp_path, monkeypatch):
    _, hardware_owner = workflow(host='192.0.2.10')
    _, central_owner = workflow(host='192.0.2.8', central=True)
    observed = {**idle(), 'ok': False, 'status': 'busy',
        'processes': hardware_owner['processes'] + central_owner['processes']}
    monkeypatch.setattr(entry, '_observe_processes', lambda _: observed)
    path, output, row, transport = selective_plan(tmp_path, monkeypatch)
    assert entry.execute_plan(path) == 0
    assert [index for _, index in transport.calls] == [1, 2, 3]
    owners = entry.read_json(output / 'production-preflight/owners.json')
    assert owners['ok'] is True
    assert len(owners['controller']['normal_quality_owners']) == 2
    assert any(owner['central_only'] for owner in owners['controller']['normal_quality_owners'])
    assert entry.read_json(output / 'start-cases-accepted.json')['valid_repetitions'] == 3
    assert entry.read_json(output / 'terminal-status.json')['status'] == 'complete'


@pytest.mark.parametrize('fault', ['same_dut', 'unknown_owner', 'visibility'])
def test_selective_legacy_still_blocks_unresolved_or_same_dut_owners(
        workflow, tmp_path, monkeypatch, fault):
    run, observed = workflow(host='192.0.2.8' if fault == 'same_dut' else '192.0.2.10')
    if fault == 'unknown_owner':
        control = run / 'jobs/workflow_control.json'
        value = entry.read_json(control)
        value['session_id'] = 'unbound-session'
        entry.write_json(control, value)
    elif fault == 'visibility':
        observed['visibility_errors'] = ['unreadable owner namespace']
    monkeypatch.setattr(entry, '_observe_processes', lambda _: observed)
    path, output, _, transport = selective_plan(tmp_path, monkeypatch)
    assert entry.execute_plan(path) == 1
    assert transport.calls == []
    assert not (output / 'start-cases-accepted.json').exists()
    owners = entry.read_json(output / 'production-preflight/owners.json')
    assert owners['ok'] is False
    assert owners['controller']['scope_errors'] or owners['controller'].get('visibility_errors')
