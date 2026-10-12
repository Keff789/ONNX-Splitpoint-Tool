"""Prepared scientific parents retain their runtime contract on controller refresh."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import generic_energy_parent as parent
from onnx_splitpoint_tool.workflow import generic_completed_task as task


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'report'
    run = root / 'energy/generic_completed/case'
    source = tmp_path / 'source'
    run.mkdir(parents=True)
    source.mkdir()

    def binding(relative, content='original'):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return parent.file_binding(path)

    controller = [binding(name) for name in (
        'onnx_splitpoint_tool/remote/process_lease.py',
        'onnx_splitpoint_tool/workflow/execution_binding.py',
        'scripts/energy_measurement_cli.py')]
    runtime = binding('onnx_splitpoint_tool/runners/generic_energy.py')
    adapter = binding('adapter.py')
    budget = source / 'budget.json'
    parent.write(budget, {'starts': 1164, 'valid_slots': 1022})
    identity = {'setup_id': 'device', 'case_id': 'b001'}
    case = {'identity': identity}
    admission = {'performance_result': {'sha256': 'actual-performance'}}
    normal = {'identity': identity, 'performance_result_sha256': 'actual-performance',
              'execution_environment_restored': True, 'duration_s': 60,
              'warmup': 100, 'tasks': 1000, 'runner': runtime,
              'collector_script': controller[2]['path'], 'collector_script_sha256': controller[2]['sha256']}
    parent.write(run / 'case.json', case)
    parent.write(run / 'normal.json', normal)
    plan = {key: binding(key) for key in (
        'performance_plan', 'registry_source', 'registry_snapshot', 'collector_binary', 'python')}
    parent.write(source / 'performance_plan', {})
    plan['performance_plan'] = parent.file_binding(source / 'performance_plan')
    plan.update(run_dir=str(run), point_count=1, capture_count=3, duration_s=60.,
                campaign_budget_file=str(budget), adapter=adapter,
                controller_sources=controller + [runtime], selected_identity=identity,
                case=parent.file_binding(run / 'case.json'), performance_admission=admission,
                normal_energy_plan=parent.file_binding(run / 'normal.json'), remote_command_prefix=[])
    parent.write(run / 'parent_plan.json', plan)
    parent.write(run / 'PREPARED.json', {'ok': True, 'parent_plan_sha256': parent.sha(run / 'parent_plan.json')})
    monkeypatch.setattr(parent, 'configure', lambda *args: None)
    monkeypatch.setattr(parent, 'R', root)
    monkeypatch.setattr(parent, 'W', source)
    monkeypatch.setattr(parent, 'BUDGET', budget)
    monkeypatch.setattr(parent, 'ADAPTER_PATH', source / 'adapter.py')
    monkeypatch.setattr(parent, 'select_case', lambda *args: ({}, case))
    monkeypatch.setattr(parent, 'performance_admission', lambda *args, **kwargs: admission)
    monkeypatch.setattr(parent, 'verify_calls', lambda *args: None)
    monkeypatch.setattr(parent, 'controller_sources', lambda: [])
    monkeypatch.setattr(task, 'check_energy_campaign', lambda *args, **kwargs: None)
    return run, source, budget, plan


def test_controller_refresh_preserves_scientific_inputs_and_journal(prepared):
    run, source, budget, original = prepared
    budget_before = budget.read_bytes()
    plan_before = (run / 'parent_plan.json').read_bytes()
    for binding in original['controller_sources'][:3]:
        Path(binding['path']).write_text('controller fix')
    parent.refresh_controller(SimpleNamespace(run_dir=run))
    refreshed = parent.read(run / 'parent_plan.json')
    for key in set(original) - {'controller_sources', 'normal_energy_plan'}:
        assert refreshed[key] == original[key]
    assert refreshed['controller_sources'][-1] == original['controller_sources'][-1]
    assert budget.read_bytes() == budget_before
    history, = (run / 'controller_binding_refresh').iterdir()
    assert (history / 'parent_plan.json').read_bytes() == plan_before
    receipt = parent.read(history / 'REFRESH_RECEIPT.json')
    assert len(receipt['refreshed_controller_paths']) == 3
    assert receipt['hardware_started'] is False
    assert receipt['normal_energy_plan_unchanged'] is False
    assert receipt['collector_script_binding_refreshed'] is True
    saved = parent.read(original['normal_energy_plan']['path'])
    current = parent.read(parent.bound_file(refreshed['normal_energy_plan']))
    assert current.pop('collector_script_sha256') == parent.sha(saved['collector_script'])
    saved.pop('collector_script_sha256')
    assert current == saved
    assert not (run / 'production-start.json').exists()


@pytest.mark.parametrize('mutation', ['runner', 'input', 'duration', 'repeats'])
def test_refresh_refuses_changed_scientific_contract(prepared, mutation):
    run, source, budget, original = prepared
    if mutation == 'runner':
        (source / 'onnx_splitpoint_tool/runners/generic_energy.py').write_text('changed runtime')
    elif mutation == 'input':
        (run / 'normal.json').write_text('{}')
    else:
        original['duration_s' if mutation == 'duration' else 'capture_count'] = 1
        parent.write(run / 'parent_plan.json', original)
        parent.write(run / 'PREPARED.json', {'ok': True, 'parent_plan_sha256': parent.sha(run / 'parent_plan.json')})
    before = (run / 'parent_plan.json').read_bytes(), budget.read_bytes()
    with pytest.raises(ValueError, match='Frozen input changed|Parent capture scope changed'):
        parent.refresh_controller(SimpleNamespace(run_dir=run))
    assert before == ((run / 'parent_plan.json').read_bytes(), budget.read_bytes())
    assert not (run / 'controller_binding_refresh').exists()


@pytest.mark.parametrize('already_started', [False, True])
def test_collector_refresh_rejects_unbound_or_attempted_capture(prepared, already_started):
    run, source, budget, original = prepared
    (source / 'scripts/energy_measurement_cli.py').write_text('controller fix')
    if already_started:
        (run / 'measurement').mkdir()
        (run / 'measurement/capture_1_started.json').write_text('{}')
        match = 'unstarted scientific parent'
    else:
        normal = parent.read(run / 'normal.json')
        normal['collector_script_sha256'] = 'foreign'
        parent.write(run / 'normal.json', normal)
        original['normal_energy_plan'] = parent.file_binding(run / 'normal.json')
        parent.write(run / 'parent_plan.json', original)
        parent.write(run / 'PREPARED.json', {'ok': True, 'parent_plan_sha256': parent.sha(run / 'parent_plan.json')})
        match = 'frozen controller source'
    before = (run / 'parent_plan.json').read_bytes(), budget.read_bytes()
    with pytest.raises(ValueError, match=match):
        parent.refresh_controller(SimpleNamespace(run_dir=run))
    assert before == ((run / 'parent_plan.json').read_bytes(), budget.read_bytes())
    assert not (run / 'controller_binding_refresh').exists()
