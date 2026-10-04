"""Bounded remainder admission/dispatch; no accelerator or power source."""
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import json

import pytest

from onnx_splitpoint_tool.workflow import yolo_augmentation_rest as rest
from scripts import update_evalset_native_producers as updater

AUG = (Path(os.environ.get('THESIS20_FIXTURE_HOME', Path.home())) / 'Reports/thesis20-yolo-min3-augmentation-20261002_222904')


@pytest.fixture
def actual_plan():
    path = AUG / 'plan/augmentation-plan.json'
    if not path.is_file(): pytest.skip('actual stored campaign is not installed')
    return json.loads(path.read_text())


def test_actual_observations_preserve_completed_generic_and_eleven_native(actual_plan, tmp_path):
    completed = len(rest.io.read_json(Path(actual_plan['output_root']) / 'completion/execution-plan.json')['rows'])
    inventory = rest.remaining_inventory(actual_plan, energy_continuation=completed == 11)
    assert (inventory['native_cases'], inventory['native_repetitions'], inventory['quality_results']) == (11, 33, 12)
    assert (inventory['generic_cases'], inventory['generic_repetitions']) == (completed, completed * 3)
    assert len(inventory['block1_generic']) == 11 - completed and len(inventory['block1_energy']) == 11
    assert rest.SPECIAL not in inventory['block1_generic'] + inventory['block1_energy']
    (tmp_path / 'actual-admission.json').write_text(json.dumps(inventory, indent=2))


def test_actual_singlecase_retains_command_and_quality_scope(actual_plan, tmp_path):
    run = Path(actual_plan['augmentation_run'])
    cfg = rest.io.read_json(run / 'reports/native_producer_stage_config.json')
    cmd, namespace = rest.singlecase_command(run, cfg, tmp_path / 'distinct_attempt')
    assert json.loads(cmd[cmd.index('--case-map') + 1]) == {'yolov7_paper': ['b066']}
    assert cmd[cmd.index('--backends') + 1] == 'hailo8'
    assert cmd[cmd.index('--artifact-namespace') + 1] == namespace
    assert cmd[cmd.index('--remote-tool-dir') + 1].endswith('/tool/attempts/distinct_attempt')
    assert cmd[cmd.index('--frames') + 1] == '1000'
    assert cmd[cmd.index('--warmup') + 1] == '100'
    assert cmd[cmd.index('--repetitions') + 1] == '3'
    assert '--no-build-missing-engines' in cmd and '--existing-outputs-only' in cmd
    assert '--native-validation' in cmd
    (tmp_path / 'singlecase-command.json').write_text(json.dumps(cmd, indent=2))


@pytest.mark.parametrize('fault', ['runtime_id', 'native_count', 'native_time', 'quality_missing'])
def test_actual_admission_keeps_existing_negative_guards(actual_plan, monkeypatch, fault):
    original = rest.io.read_json
    summary = original(Path(actual_plan['augmentation_run']) / 'reports/native_producer_combined_summary.json')
    source = Path(next(row['report'] for row in summary['rows'] if row.get('ok') is True))
    def read(path):
        value = original(path)
        if Path(path) == source:
            value = copy.deepcopy(value)
            block = (value.get('endpoint_results') or {}).get('completed_task') or value
            if fault == 'runtime_id':
                for row in block['repetition_records']: row['runtime_instance_id'] = 'same-runtime'
            if fault == 'native_count': block['repetition_records'][0]['completed_work_units'] = 999
            if fault == 'native_time': block['repetition_records'][0]['measured_makespan_seconds'] = -1
        return value
    monkeypatch.setattr(rest.io, 'read_json', read)
    if fault == 'quality_missing':
        from onnx_splitpoint_tool.workflow import supplement
        monkeypatch.setattr(supplement, 'completed_quality', lambda run: ([], {}))
    with pytest.raises(ValueError): rest.remaining_inventory(actual_plan, energy_continuation=True)


@pytest.fixture
def phase_fixture(tmp_path, monkeypatch):
    run = tmp_path / 'run'; output = tmp_path / 'augmentation/attempts/new'; output.mkdir(parents=True)
    (output / 'runner-bundle.tar.gz').write_bytes(b'test-only-bundle')
    reports = run / 'reports'; reports.mkdir(parents=True)
    for name in ('native_producer_stage_config.json', 'native_producer_combined_summary.json',
                 'native_producer_stage.json', 'native_expected_matrix.json'):
        (reports / name).write_text('{}')
    validation = reports / 'native_validation/native_producer_validation_summary.json'
    validation.parent.mkdir(); validation.write_text('{}')
    plan = {'augmentation_run': str(run), 'output_root': str(output.parent.parent)}
    monkeypatch.setattr(rest.augmentation, 'load_frozen_plan', lambda path: plan)
    monkeypatch.setattr(rest, 'remaining_inventory', lambda plan: {'block1_energy': [('model','case','setup')]})
    monkeypatch.setattr(rest, 'preserved_sources', lambda plan: {})
    monkeypatch.setattr(rest.augmentation, 'preservation_inventory', lambda plan: {})
    monkeypatch.setattr(rest, 'energy_command', lambda *args: ['energy', str(args[-1])])
    phases = []; delivery = {}; reject = []
    @contextmanager
    def owned(*args):
        def execute(cmd, label, *args, **kwargs):
            phases.append(label)
            if cmd[0] == 'energy':
                rest.io.write_json(Path(cmd[1]) / 'reports/native_energy_measurements/native_producer_energy_results.json', {'rows': []})
            if label in reject: raise RuntimeError('controlled phase failure')
        yield execute, {}, None, None
    monkeypatch.setattr(rest, 'owned_execution', owned)
    monkeypatch.setattr(rest, '_verify_energy', lambda *args: None)
    def final(plan, status, error, rc, **kwargs): delivery.update(status=status, error=error, rc=rc, **kwargs); return rc
    monkeypatch.setattr(rest.augmentation, 'write_final_delivery', final)
    return output, phases, delivery, reject


def test_block1_is_independent_of_unavailable_special_native(phase_fixture, monkeypatch):
    output, phases, delivery, _ = phase_fixture
    monkeypatch.setattr(rest, 'singlecase_command', lambda *args: pytest.fail('b066 must not gate Block 1'))
    assert rest.finish_rest(Path('plan.json'), output) == 2
    assert phases == ['block1-eight-generic', 'block1-eleven-energy']
    assert delivery['status'] == 'partial_b066_pending' and len(delivery['energy_results']) == 1


def test_failed_energy_keeps_real_negative_aggregate_and_stops_next_block(phase_fixture, monkeypatch):
    output, phases, delivery, reject = phase_fixture
    reject.append('block1-eleven-energy')
    monkeypatch.setattr(rest, 'singlecase_command', lambda *args: pytest.fail('no dispatch after failure'))
    assert rest.finish_rest(Path('plan.json'), output, include_b066=True) == 1
    assert phases == ['block1-eight-generic', 'block1-eleven-energy']
    assert len(delivery['energy_results']) == 1 and 'controlled phase failure' in delivery['error']


def test_native_special_is_attempted_only_after_completed_first_block(phase_fixture, monkeypatch):
    output, phases, delivery, _ = phase_fixture
    def special(*args):
        assert phases == ['block1-eight-generic', 'block1-eleven-energy']
        raise ValueError('unavailable actual Native binding')
    monkeypatch.setattr(rest, 'singlecase_command', special)
    assert rest.finish_rest(Path('plan.json'), output, include_b066=True) == 1
    assert len(delivery['energy_results']) == 1 and 'unavailable actual Native binding' in delivery['error']


def test_current_cancel_is_not_reset_or_dispatched(tmp_path, monkeypatch):
    run = tmp_path / 'run'; output = tmp_path / 'current'
    marker = run / 'jobs/workflow_control.json'
    rest.io.write_json(marker, {'run_id': run.name, 'session_id': output.name, 'cancel_requested': True})
    before = marker.read_bytes()
    def lock(**kwargs): return SimpleNamespace(acquire=lambda **kwargs: None, release=lambda: None)
    monkeypatch.setattr(rest.io, 'EvaluationRunLock', lock)
    monkeypatch.setattr(rest.io, 'bound_targets', lambda *args: {})
    monkeypatch.setattr(rest.io, 'observe_owners', lambda *args: pytest.fail('cancelled before owner/transport leaf'))
    with pytest.raises(RuntimeError, match='current rest attempt is cancelled'):
        with rest.owned_execution({'augmentation_run': str(run), 'rows': []}, output, {}): pytest.fail('cancelled')
    assert marker.read_bytes() == before


def test_live_owned_session_replaces_only_old_cancel_then_honors_current_cancel(actual_plan, tmp_path, monkeypatch):
    run = tmp_path / 'isolated_owned_run'; output = tmp_path / 'new_owned_session'; output.mkdir()
    marker = run / 'jobs/workflow_control.json'
    rest.io.write_json(marker, {'run_id': run.name, 'session_id': 'historical', 'cancel_requested': True})
    cfg = rest.io.read_json(Path(actual_plan['augmentation_run']) / 'reports/native_producer_stage_config.json')
    monkeypatch.setattr(rest.io, 'bound_targets', lambda *args: {})
    monkeypatch.setattr(rest.io, 'observe_owners', lambda *args: None)  # physical owner probe leaf only
    from onnx_splitpoint_tool import native_progress
    monkeypatch.setattr(native_progress, 'run_streaming', lambda *args, **kwargs: pytest.fail('cancelled before any child'))
    with rest.owned_execution({'augmentation_run': str(run), 'rows': []}, output, cfg) as (execute, env, registry, cancel):
        current = rest.io.read_json(marker)
        assert current['session_id'] == output.name and current['cancel_requested'] is False
        current['cancel_requested'] = True; rest.io.write_json(marker, current)
        with pytest.raises(RuntimeError, match='cancelled before next dispatch'): execute(['unreachable'], 'forbidden')
    assert rest.io.read_json(marker)['cancel_requested'] is True
    cleanup = rest.io.read_json(output / 'cleanup-status.json')
    assert cleanup['owned_local_processes'] == cleanup['active_owned_remote_operations'] == 0


def reference_cfg(tmp_path):
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps({'schema':'onnx-splitpoint/native-producer-validation-summary','rows':[{'status':'stored'}]}))
    return {'validation': {'enabled': True}, 'existing_outputs_only': True,
            'existing_reference_validation_summary': str(path)}


@pytest.mark.parametrize('fault', ['missing', 'schema', 'without_mode', 'disabled_validator', 'symlink'])
def test_stored_reference_mode_fails_before_transport(tmp_path, monkeypatch, fault):
    cfg = reference_cfg(tmp_path); path = Path(cfg['existing_reference_validation_summary'])
    if fault == 'missing': path.unlink()
    if fault == 'schema': path.write_text('{}')
    if fault == 'without_mode': cfg['existing_outputs_only'] = False
    if fault == 'disabled_validator': cfg['validation']['enabled'] = False
    if fault == 'symlink':
        real = tmp_path / 'real.json'; path.rename(real); path.symlink_to(real)
    monkeypatch.setattr(updater, '_run', lambda *args, **kwargs: pytest.fail('transport must remain unreachable'))
    with pytest.raises(updater.TensorRTQualityChainError): updater._run_native_producers(tmp_path, cfg, timeout=1)


def test_real_remote_full_probe_branch_is_skipped_only_for_stored_mode(tmp_path):
    function = next(n for n in ast.parse(Path(updater.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_run_native_producers')
    branch = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                  and 'probe_yolo_requested' in ast.unparse(n.test))
    cfg = reference_cfg(tmp_path)
    args = updater._existing_reference_validation_args(cfg)
    assert args == ['--existing-outputs-only', '--existing-reference-validation-summary', cfg['existing_reference_validation_summary']]
    # Execute the real unchanged branch condition; replace only its remote leaf.
    check = ast.If(test=branch.test, body=[ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError', ctx=ast.Load()),
        args=[ast.Constant('remote Full ORT dispatch')], keywords=[]))], orelse=[])
    code = compile(ast.fix_missing_locations(ast.Module(body=[check], type_ignores=[])), updater.__file__, 'exec')
    scope = dict(existing_reference_args=args, backend_split_execution_allowed=True, b='hailo8',
        vcfg_probe=cfg['validation'], cfg=cfg, dump_boundary=True, probe_yolo_requested=True)
    exec(code, scope)
    scope['existing_reference_args'] = []
    with pytest.raises(RuntimeError, match='remote Full ORT'): exec(code, scope)


def test_updater_energy_helper_and_mirror_are_present():
    root = Path(updater.__file__).parents[1]
    source = Path(updater.__file__).read_bytes()
    assert b'"native_split_energy_preflight.py"' in source
    assert b'"run_and_report_work_units.py"' in source
    assert source == (root / 'onnx_splitpoint_tool/resources/remote_scripts/update_evalset_native_producers.py').read_bytes()


def test_real_local_validator_command_forwards_original_outputs_only_reference(tmp_path):
    function = next(n for n in ast.parse(Path(updater.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_run_native_producers')
    branch = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "bool(vcfg.get('enabled'))"
                  and 'native_producer_validate_visualize.py' in ast.unparse(n))
    cfg = reference_cfg(tmp_path); reports = tmp_path / 'reports'; reports.mkdir()
    (reports / 'native_producer_combined_summary.json').write_text('{}')
    class StopAtValidator(BaseException): pass
    calls = []
    def stop(argv, **kwargs): calls.append(argv); raise StopAtValidator()
    scope = dict(vars(updater), cfg=cfg, vcfg=cfg['validation'], reports=reports,
        run_dir=tmp_path, collected=[tmp_path / 'selected_native_outputs'], stage={},
        existing_reference_args=updater._existing_reference_validation_args(cfg), _run=stop)
    with pytest.raises(StopAtValidator):
        exec(compile(ast.Module(body=[branch], type_ignores=[]), updater.__file__, 'exec'), scope)
    assert len(calls) == 1 and Path(calls[0][1]).name == 'native_producer_validate_visualize.py'
    assert '--existing-outputs-only' in calls[0]
    assert calls[0][calls[0].index('--existing-reference-validation-summary') + 1] == cfg['existing_reference_validation_summary']
