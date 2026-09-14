"""The shipped launcher binds source cases, recipe and frozen hardware."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import pytest
import yaml
ROOT = Path(__file__).resolve().parents[1]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def gate(): return load(ROOT / 'scripts/reference_workflow_gate_v281.py', 'v281_gate_tests')

def sources(tmp_path):
    fixtures = load(ROOT / 'tests/test_v2803_scope_and_diagnostic_claims.py', 'v281_previous_fixtures')
    path, source = fixtures._prepare_sources(tmp_path)
    p = yaml.safe_load(path.read_text()); selected = copy.deepcopy(p['model_suite']['primary'][0]); p['model_suite']['primary'] = []
    cases = {}
    for scope in gate().SCOPE.values():
        for model, ids in scope['models'].items(): cases.setdefault(model, set()).update(ids)
    for model, ids in cases.items():
        row = copy.deepcopy(selected); row.update(id=model, task='classification' if model == 'regnet_x_1_6gf' else 'detection')
        model_path = tmp_path / (model + '.onnx'); model_path.write_bytes(b'original source fixture')
        row['onnx'] = str(model_path); p['model_suite']['primary'].append(row)
        contract = source / 'models' / model / 'benchmark_set/legacy_suite/benchmark_set.json'; contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text(json.dumps({'model_id': model, 'cases': [{'case_id': c} for c in sorted(ids)]}))
    h8 = copy.deepcopy(p['hardware']['resolved_targets'][0]); h8['id'] = 'orin_nx_hailo8_01'
    if not any(r.get('id') == h8['id'] for r in p['hardware']['resolved_targets']): p['hardware']['resolved_targets'].append(h8)
    from onnx_splitpoint_tool.workflow.start_snapshot import snapshot_payload_sha256
    p['hardware']['resolved_targets_sha256'] = snapshot_payload_sha256(p['hardware']['resolved_targets'])
    for ident, source_id in [('hailo8', 'hailo10'), ('hailo8_to_trt', 'hailo10_to_tensorrt')]:
        r = copy.deepcopy(next(r for r in p['run_profiles'] if r['id'] == source_id)); r['id'] = ident
        for key in ('full', 'stage1', 'stage2'):
            if r.get(key) == 'hailo10': r[key] = 'hailo8'
        if not any(existing.get('id') == ident for existing in p['run_profiles']): p['run_profiles'].append(r)
    for cfg in (p.setdefault('hailo_build', {}), p['execution_preset']['snapshot']['build']['hailo']):
        cfg['compute_by_family'] = {'hailo8': {'device': 'gpu'}, 'hailo10h': {'device': 'gpu'}}
    manifest = tmp_path / 'overlay.json'; manifest.write_text('{}'); (source / 'profile.yaml').write_text(yaml.safe_dump(p))
    return source, manifest

@pytest.mark.parametrize('family', ['hailo8', 'hailo10h'])
def test_real_loader_binds_cases_recipe_hardware_without_source_mutation(tmp_path, family):
    source, manifest = sources(tmp_path); before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}; g = gate()
    profile, report = g.prepare(source, tmp_path / 'prepared', family, hailo8_dependency_manifest=manifest)
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    actual, _ = load_runtime_profile_snapshot(str(profile))
    assert actual['selection_policy']['forced_cases'] == g.SCOPE[family]['models']
    assert actual['hardware']['selected_setups'] == [g.SCOPE[family]['setup']]
    assert actual['hailo_build']['force_build'] is False
    assert actual['native_producers']['energy']['enabled'] is False
    assert actual['hailo_build']['calib_count'] == 500
    assert actual['validation_execution']['max_items'] == {'classification': 16, 'detection': 16}
    assert actual['native_producers']['build_missing_engines'] is (family == 'hailo8')
    assert actual['hailo_build']['compute_by_family']['hailo8']['dependency_manifest'] == str(manifest)
    assert report['status'] == 'prepared_not_executed'
    assert {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before

@pytest.mark.parametrize('failure', ['original_missing', 'case_missing'])
def test_preparation_rejects_missing_required_source_before_build(tmp_path, failure):
    source, manifest = sources(tmp_path)
    if failure == 'original_missing':
        p = yaml.safe_load((source / 'profile.yaml').read_text()); Path(p['model_suite']['primary'][0]['onnx']).unlink()
    elif failure == 'case_missing':
        (source / 'models/yolo11l/benchmark_set/legacy_suite/benchmark_set.json').write_text(json.dumps({'model_id': 'yolo11l', 'cases': []}))
    before = (source / 'profile.yaml').read_bytes()
    with pytest.raises(ValueError, match='original_model_missing|fixed_cases_missing|dependency_manifest_not_selected'):
        gate().prepare(source, tmp_path / 'prepared', 'hailo8', hailo8_dependency_manifest=manifest)
    assert (source / 'profile.yaml').read_bytes() == before
    assert not (tmp_path / 'prepared').exists()

def test_archive_retains_debug_packs_without_bulk_model_data(tmp_path):
    out = tmp_path / 'evidence'; out.mkdir(); (out / 'acceptance.json').write_text('{}')
    for family in gate().SCOPE:
        d = out / family; d.mkdir(); (d / 'debug_pack.zip').write_bytes(b'fixture archive'); (d / 'workflow.log').write_text('workflow')
        (d / 'runs').mkdir(); (d / 'runs/model.onnx').write_bytes(b'large model')
    path = gate().archive_evidence(out)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.endswith('debug_pack.zip')]) == 2
        assert not any('/runs/' in n for n in z.namelist())
    with pytest.raises(FileExistsError): gate().archive_evidence(out)

def test_real_child_nonzero_exit_kept(tmp_path):
    result = gate().run_logged([sys.executable, '-I', '-B', '-c', 'print("real child"); raise SystemExit(7)'], tmp_path, tmp_path / 'child.log', timeout_s=5)
    assert result['returncode'] == 7 and not result['cancelled']
    assert (tmp_path / 'child.log').read_text().strip() == 'real child'

def test_real_child_timeout_terminates(tmp_path):
    result = gate().run_logged([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(30)'], tmp_path, tmp_path / 'child.log', timeout_s=.1)
    assert result['timed_out'] and result['cancelled'] and result['returncode'] != 0
    assert result['elapsed_s'] < 10

def test_family_failure_still_collects_other_family(tmp_path, monkeypatch):
    g = gate(); tool = tmp_path / 'tool'; py = tool / '.venv/bin/python'; py.parent.mkdir(parents=True); py.symlink_to(sys.executable)
    monkeypatch.setattr(g, 'ROOT', tool); out = tmp_path / 'out'; out.mkdir(); calls = []
    def prepare(source, output, family, **kwargs):
        output.mkdir(parents=True); path = output / 'profile.yaml'; path.write_text(family); return path, {'fixture': True}
    def logged(command, cwd, log, **kwargs):
        calls.append(command); log.write_text('fixture execution')
        if 'onnx_splitpoint_tool.workflow.run_evaluation' in command:
            output = Path(command[command.index('--out') + 1]); m = output / 'actual/run_manifest.json'; m.parent.mkdir(parents=True); m.write_text('{}')
            return {'returncode': 2 if 'hailo8' in str(output) else 0, 'cancelled': False, 'timed_out': False}
        Path(command[command.index('--out') + 1]).write_bytes(b'fixture'); return {'returncode': 0, 'cancelled': False, 'timed_out': False}
    monkeypatch.setattr(g, 'prepare', prepare); monkeypatch.setattr(g, 'run_logged', logged)
    monkeypatch.setattr(g, 'inspect_run', lambda run, family: {'technical_acceptance': 'pass', 'quality_decision': 'fail'})
    assert g.execute(tmp_path / 'source', out) == 2
    report = json.loads((out / 'acceptance.json').read_text())
    assert len([c for c in calls if 'onnx_splitpoint_tool.workflow.run_evaluation' in c]) == 2
    assert report['families']['hailo10h']['complete'] and report['quality_decision'] == 'fail'
    assert out.with_suffix('.zip').is_file()


def test_owned_descendant_cleanup_does_not_kill_independent_process(tmp_path):
    import subprocess
    unrelated = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        source = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-I","-B","-c","import time; time.sleep(30)"]); print(p.pid)'
        result = gate().run_logged([sys.executable, '-I', '-B', '-c', source], tmp_path, tmp_path / 'group.log', timeout_s=5)
        assert result['returncode'] == 2
        assert result['process_cleanup']['status'] == 'residual_owned_group_terminated'
        assert result['process_cleanup']['remaining_live_pids'] == []
        assert unrelated.poll() is None
    finally:
        unrelated.terminate(); unrelated.wait(timeout=5)


def inspected_run(tmp_path):
    source, manifest = sources(tmp_path); g = gate(); run = tmp_path / 'generated_run'; run.mkdir()
    p, _ = g.prepare(source, tmp_path / 'prepared', 'hailo10h', hailo8_dependency_manifest=manifest)
    (run / 'profile.yaml').write_bytes(p.read_bytes())
    def write(relative, payload):
        path = run / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload))
    scope = g.SCOPE['hailo10h']; success = []; excluded = []; results = []
    for m, cases in scope['models'].items():
        for case in cases:
            row = {'model': m, 'case': case, 'backend': 'hailo10h_to_trt', 'setup_id': scope['setup'], 'actual_ok': case not in scope['known_exclusions'][m]}
            if not row['actual_ok']:
                from onnx_splitpoint_tool.native_job_identity import apply_known_build_disposition
                historical = json.loads((ROOT/'tests/fixtures/v281_status/night_blocked_native_rows.json').read_text())
                original = next(r for r in historical if r.get('model') == m and r.get('case') == case and r.get('setup_id') == scope['setup'])
                row = apply_known_build_disposition(original)
                row.pop('upstream_build_observation', None)
                row['actual_ok'] = False
            (success if row['actual_ok'] else excluded).append(row)
            if row['actual_ok']:
                results.append({'model_id': m, 'case_id': case, 'source_run_id': 'hailo10h_to_trt', 'variant': 'composed',
                    'technical_status': 'completed', 'n': 16, 'decision': 'fail', 'management_cpu_reference': {'reference_sha256': 'sha256:' + 'a'*64}})
        for b in ['hailo10h', 'tensorrt']:
            success.append({'model': m, 'case': 'full', 'backend': 'native_full_' + b, 'setup_id': scope['setup'], 'actual_ok': True})
        write('quality_management/references/'+m+'/management_cpu_reference_status.json', {'reference_sha256': 'sha256:' + 'a'*64})
    write('run_manifest.json', {'technical_status': 'ok', 'runtime_complete': True, 'aggregate_quality_decision': 'fail'})
    write('reports/native_expected_matrix.json', {'technical_execution_complete': True, 'failed_expected_row_count': 0, 'missing_expected_row_count': 0,
        'successful_expected_rows': success, 'excluded_expected_rows': excluded})
    write('quality_management/central_quality_summary.json', {'results': results})
    write('reports/artifact_cache_preflight.json', {'runtime_dispatch_allowed': True, 'unexpected_cold_builds': 0, 'unknown_count': 0, 'artifact_matrix': []})
    write('reports/artifact_index_closure.json', {'status': 'pass', **{k:0 for k in ('hash_mismatch_count','missing_file_count','unindexed_required_path_count','verification_error_count')}})
    write('reports/scientific/report_manifest.json', {})
    return run, write


def test_quality_fail_does_not_mask_technical_success(tmp_path):
    run, _ = inspected_run(tmp_path); report = gate().inspect_run(run, 'hailo10h')
    assert report['technical_acceptance'] == 'pass' and report['quality_decision'] == 'fail'
    assert len(report['known_compile_exclusions']) == 2


@pytest.mark.parametrize('problem', ['exclusion_dispatched', 'missing_measured_case', 'unexpected_cold', 'quality_reference_changed', 'forged_negative', 'duplicate_success', 'foreign_setup'])
def test_inspection_rejects_real_coverage_and_binding_failures(tmp_path, problem):
    run, write = inspected_run(tmp_path)
    if problem in {'exclusion_dispatched', 'missing_measured_case', 'forged_negative', 'duplicate_success', 'foreign_setup'}:
        path = 'reports/native_expected_matrix.json'; data = json.loads((run/path).read_text())
        if problem == 'exclusion_dispatched': data['excluded_expected_rows'][0]['repetition_count_attempted'] = 1
        elif problem == 'forged_negative': data['excluded_expected_rows'][0]['build_exclusion']['build_evidence']['record']['record_sha256'] = '0'*64
        elif problem == 'duplicate_success': data['successful_expected_rows'].append(copy.deepcopy(data['successful_expected_rows'][0]))
        elif problem == 'foreign_setup': data['successful_expected_rows'][0]['setup_id']='another_setup'
        else: data['successful_expected_rows'].pop()
    elif problem == 'unexpected_cold':
        path = 'reports/artifact_cache_preflight.json'; data = json.loads((run/path).read_text()); data['unexpected_cold_builds'] = 1
    else:
        path = 'quality_management/central_quality_summary.json'; data = json.loads((run/path).read_text()); data['results'][0]['management_cpu_reference']['reference_sha256'] = 'sha256:' + 'b'*64
    write(path, data)
    assert gate().inspect_run(run, 'hailo10h')['technical_acceptance'] == 'incomplete'


def test_chatty_child_cannot_disable_timeout(tmp_path, capsys):
    code = 'import os,time; deadline=time.monotonic()+30\nwhile time.monotonic()<deadline: os.write(1,b"x"*65536)'
    result = gate().run_logged([sys.executable, '-I', '-B', '-c', code], tmp_path, tmp_path/'chatty.log', timeout_s=.1)
    assert result['timed_out'] and result['cancelled'] and result['elapsed_s'] < 5
    assert result['returncode'] != 0


def test_complete_venv_can_use_normal_context_without_forced_overlay(tmp_path):
    source, _ = sources(tmp_path)
    path, _ = gate().prepare(source, tmp_path/'prepared_no_overlay', 'hailo8')
    profile = yaml.safe_load(path.read_text())
    assert 'dependency_manifest' not in profile['hailo_build']['compute_by_family']['hailo8']


@pytest.mark.parametrize('entry', [{'device':'cpu'}, {'device':'gpu','dependency_manifest':''}, {'device':'gpu','dependency_manifest':'/explicit/other.json'}, {'device':'gpu'}])
def test_active_explicit_compute_is_not_overwritten(tmp_path, entry):
    source, manifest = sources(tmp_path)
    current = {'compute_by_family': {'hailo8':entry}}
    path, report = gate().prepare(source, tmp_path/'prepared', 'hailo8', hailo8_dependency_manifest=manifest, build_environment=current)
    profile = yaml.safe_load(path.read_text())
    assert profile['hailo_build']['compute_by_family']['hailo8'] == entry
    assert 'venv_activate' not in profile['hailo_build']
    assert profile['hailo_build']['calib_count'] == 500
    assert report['compiler_recipe_unchanged'] is True


@pytest.mark.parametrize('follow', [True,False])
def test_active_profile_mode_precedence_is_resolved_by_normal_loader(tmp_path, monkeypatch, follow):
    from onnx_splitpoint_tool import run_modes
    source, _ = sources(tmp_path)
    profile = yaml.safe_load((source/'profile.yaml').read_text())
    cfg=run_modes.default_run_modes_config(); mode_id=profile['execution_preset']['id']
    cfg['modes'][mode_id]['build']['hailo']['compute_by_family']={'hailo8':{'device':'cpu'}}
    path=tmp_path/'run_modes.yaml';path.write_text(yaml.safe_dump(cfg));monkeypatch.setenv('ONNX_SPLITPOINT_RUN_MODES_FILE',str(path))
    profile['execution_preset']['follow_tool_config']=follow
    profile['execution_preset']['config_path']=str(path)
    # Provenance identifies the old value as inherited, so the ordinary loader
    # follows the selected registry only when configured to do so.
    profile['execution_preset'].setdefault('build_provenance',{})['hailo_compute_source']='tool_config'
    profile['execution_preset']['build_provenance']['hailo_compute_by_family']=copy.deepcopy(profile['hailo_build']['compute_by_family'])
    active=tmp_path/'active.yaml';active.write_text(yaml.safe_dump(profile))
    environment, proof=gate().load_build_environment(active)
    assert environment['compute_by_family']['hailo8']['device']==('cpu' if follow else 'gpu')
    assert set(environment)=={'compute_by_family'}
    assert proof['profile']==str(active)


def test_frozen_night_recipe_does_not_follow_changed_registry(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import run_modes
    source, manifest=sources(tmp_path); profile=yaml.safe_load((source/'profile.yaml').read_text())
    config=run_modes.default_run_modes_config(); mode=profile['execution_preset']['id']
    config['modes'][mode]['build']['hailo']['calibration_items']=499
    config['modes'][mode]['data']['calibration_items']={'classification':499,'detection':499}
    registry=tmp_path/'changed_modes.yaml';registry.write_text(yaml.safe_dump(config));monkeypatch.setenv('ONNX_SPLITPOINT_RUN_MODES_FILE',str(registry))
    profile['execution_preset']['follow_tool_config']=True;profile['execution_preset']['config_path']=str(registry)
    (source/'profile.yaml').write_text(yaml.safe_dump(profile))
    path, report=gate().prepare(source,tmp_path/'prepared','hailo8',hailo8_dependency_manifest=manifest)
    prepared=yaml.safe_load(path.read_text())
    assert prepared['hailo_build']['calib_count']==500 and report['compiler_recipe_unchanged']


def test_exited_leader_with_chatty_descendant_is_reaped(tmp_path, capsys):
    child='import os,time; deadline=time.monotonic()+30\nwhile time.monotonic()<deadline: os.write(1,b"x"*4096); time.sleep(.0001)'
    leader='import subprocess,sys,time; subprocess.Popen([sys.executable,"-I","-B","-c",'+repr(child)+']); time.sleep(.05)'
    result=gate().run_logged([sys.executable,'-I','-B','-c',leader],tmp_path,tmp_path/'residual_chatty.log',timeout_s=2)
    assert result['returncode']==2 and result['process_cleanup']['status']=='residual_owned_group_terminated'
    assert result['process_cleanup']['remaining_live_pids']==[] and result['elapsed_s']<5
