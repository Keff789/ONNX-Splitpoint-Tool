"""Finite functional capture admission; no transport or accelerator is used."""
import copy
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from onnx_splitpoint_tool import generic_completion_plan
from onnx_splitpoint_tool.workflow import native_functional_reference as subject


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return subject.bound(path)


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / 'archive'
    manifest = dump(root / 'run_manifest.json', {'run_id': 'original_run'})
    request = dump(root / 'quality_request.json', {'existing': True})
    ident = {'model_id': 'yolov7_paper', 'setup_id': 'orin_nx_deepx_m1_01', 'case_id': 'b009'}
    command = {'model': 'yolov7_paper', 'setup_id': ident['setup_id'], 'case': 'b009',
        'backend': 'deepx_to_trt', 'precision': 'float32_layout_fp16', 'comparison_backend': 'deepx',
        'eval_run_id': 'original_run', 'contract_sha256': 'command', 'source_request_sha256': request['sha256'],
        'native_split_quality_binding': {'binding_sha256': 'quality'}}
    raw = {'ok': True, 'native_command_contract': command, 'native_command_contract_sha256': 'command',
        'source_request_sha256': request['sha256'], 'native_split_quality_binding': {'binding_sha256': 'quality'},
        'test_counts': [1000, 1000, 1000], 'test_times': [1., 2., 3.]}
    raw_file = dump(root / 'native.json', raw)
    row = {'identity': ident, 'case_key': 'case', 'model_id': command['model'], 'case_id': 'b009',
        'setup_id': ident['setup_id'], 'direction': command['backend'], 'precision': command['precision'],
        'task': 'detection', 'original_run': str(root), 'native_source': raw_file['path'],
        'native_command_contract_sha256': 'command', 'generic_original_quality_request': request['path'],
        'native_quality_binding_sha256': 'quality',
        'input_binding': {'entries': [{'shape': [640, 640, 3], 'dtype': 'uint8'}]},
        'native_measurement_boundary': {'observed_completed_counts': raw['test_counts'],
                                       'observed_measurement_times_s': raw['test_times']}}
    source = {'rows': [row], 'source_run_bindings': [{'historical_root': '/historical/original_run',
        'archive_root': str(root), 'run_manifest_sha256': manifest['sha256']}]}
    source_file = dump(tmp_path / 'source.json', source)
    plan = {'source_row': row, 'source_plan': source_file, 'original_run_manifest': manifest,
            'historical_native_source': raw_file, 'original_quality_request': request}
    import onnx_splitpoint_tool.native_command_contract as commands
    import onnx_splitpoint_tool.native_split_quality as quality
    import onnx_splitpoint_tool.native_rate_endpoints as rates
    monkeypatch.setattr(subject, 'verify_native_command_contract', lambda value, **_: (value, 'verified'))
    monkeypatch.setattr(quality, 'bind_quality_to_native_split', lambda **kw: (kw['quality_binding'], 'verified'))
    monkeypatch.setattr(rates, 'rate_endpoint_fields', lambda value: {
        'completed_task_work_unit_counts': value['test_counts'],
        'completed_task_measurement_times_s': value['test_times']})
    return plan, row, raw, source


def test_default_paired_path_is_untouched():
    legacy = {'legacy': 'not admitted or repaired by this opt-in hook'}
    assert generic_completion_plan.apply_native_functional_reference(legacy) is legacy


def test_authentic_archive_mapping_and_history_are_not_mutated(history):
    plan, row, raw, _ = history
    before = copy.deepcopy((plan, row, raw))
    assert subject._historical(plan, row)[2] == raw['native_command_contract']
    assert (plan, row, raw) == before


def test_current_case_cannot_replace_original_preprocessing_or_tolerances(history):
    plan, row, _, _ = history
    current = copy.deepcopy(row)
    current['original_run_cfg'] = {'quality_gate_json': {'functional_tolerance_atol': 1000.}}
    with pytest.raises(ValueError, match='scientific/source fields changed'):
        subject._historical(plan, current)


@pytest.mark.parametrize('change', [
    lambda raw: raw.update(ok=False),
    lambda raw: raw.update(native_command_contract_sha256='other'),
    lambda raw: raw.update(source_request_sha256='other'),
    lambda raw: raw.update(model_id='yolo26m'),
    lambda raw: raw.update(case_id='b044'),
    lambda raw: raw.update(setup_id='orin_nx_hailo10_01'),
    lambda raw: raw.update(precision='fp32'),
    lambda raw: raw.update(comparison_backend='hailo10h'),
    lambda raw: raw.update(native_split_quality_binding={'binding_sha256': 'other'}),
    lambda raw: raw.update(test_counts=[1001, 1000, 1000]),
    lambda raw: raw.update(test_times=[1., 2., 3.1]),
])
def test_historical_raw_guards_are_retained(history, change):
    plan, row, raw, _ = history
    change(raw)
    plan['historical_native_source'] = dump(Path(row['native_source']), raw)
    with pytest.raises(ValueError):
        subject._historical(plan, row)


def test_archive_basename_alias_cannot_be_invented(history):
    plan, row, _, source = history
    source['source_run_bindings'][0]['run_manifest_sha256'] = 'other'
    plan['source_plan'] = dump(Path(plan['source_plan']['path']), source)
    with pytest.raises(ValueError, match='manifest-bound source mapping'):
        subject._historical(plan, row)


@pytest.mark.parametrize('key,value', [('scope_mode', 'generic_only_completed_task'), ('task', 'classification')])
def test_explicit_scope_rejects_generic_only_and_other_tasks(history, key, value):
    _, row, _, _ = history
    row[key] = value
    with pytest.raises(ValueError):
        subject.identity(row)


def test_native_remains_single_hwc_input(history):
    _, row, _, _ = history
    row['input_binding']['entries'].append(copy.deepcopy(row['input_binding']['entries'][0]))
    with pytest.raises(ValueError, match='exactly one'):
        subject.identity(row)


def test_finite_recipe_preserves_ordinary_builder_and_source(monkeypatch):
    source = {'backend': 'deepx_to_trt', 'model': 'yolov7_paper', 'setup_id': 'orin_nx_deepx_m1_01',
              'eval_run_id': 'original'}
    seen = []
    def normal_builder(command, **options):
        seen.append((copy.deepcopy(command), options))
        return ['/venv/bin/python', '-u', '/tool/normal.py', '--image', '/unchanged/input.jpg',
                '--expected-engine-sha256', 'original_engine', '--warmup', '100', '--queue-depth', '3',
                '--deepx-runtime-mode', 'A4', '--source-contract-sha256', 'original_contract']
    monkeypatch.setattr(subject, 'successful_runtime_argv', normal_builder)
    argv = subject.functional_argv(source, remote_tool='/private/tool', benchmark='/private/benchmark',
        output='/private/output', runner_sha='current_runner', quality_binding='/private/quality.json')
    assert seen == [(source, {'duration_s': 0., 'fresh_output_root': '/private/output', 'remote_tool_dir': '/private/tool'})]
    values = {flag: argv[argv.index(flag) + 1] for flag in ('--frames', '--warmup', '--repetitions',
        '--completion-runtime-mode', '--deepx-runtime-mode', '--image', '--expected-engine-sha256', '--source-contract-sha256')}
    assert values == {'--frames': '1', '--warmup': '0', '--repetitions': '1', '--completion-runtime-mode': 'strict',
        '--deepx-runtime-mode': 'S1', '--image': '/unchanged/input.jpg', '--expected-engine-sha256': 'original_engine',
        '--source-contract-sha256': 'original_contract'}
    assert '--dump-outputs' in argv and '--dump-boundary' in argv


@pytest.mark.parametrize('forbidden', ['--build', '--build-missing-engine', '--force-rebuild-engine', '--energy-workload-only'])
def test_builder_cannot_enable_build_or_energy(monkeypatch, forbidden):
    monkeypatch.setattr(subject, 'successful_runtime_argv', lambda *_, **__: ['python', forbidden])
    with pytest.raises(ValueError, match='Build/energy'):
        subject.functional_argv({'backend': 'deepx_to_trt', 'model': 'yolov7_paper', 'setup_id': 's', 'eval_run_id': 'r'},
            remote_tool='t', benchmark='b', output='o', runner_sha='h', quality_binding='q')


def test_descriptor_requires_exact_actual_file(tmp_path):
    path = tmp_path / 'input.npy'
    path.write_bytes(b'original')
    descriptor = subject.bound(path)
    assert subject.checked(descriptor) == path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): subject.checked(descriptor)
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(ValueError): subject.checked({'path': str(link), 'sha256': subject.sha(path)})


def test_collected_paths_cannot_escape_actual_capture(tmp_path):
    root = tmp_path / 'benchmark'
    root.mkdir()
    (root / 'output.bin').write_bytes(b'proof')
    assert subject._collected_path('/remote/benchmark/output.bin', '/remote/benchmark', root) == root / 'output.bin'
    with pytest.raises(ValueError): subject._collected_path('/remote/benchmark/../other', '/remote/benchmark', root)
    with pytest.raises(ValueError): subject._collected_path('/remote/other', '/remote/benchmark', root)
    (root / 'outside').symlink_to(tmp_path)
    with pytest.raises(ValueError): subject._collected_path('/remote/benchmark/outside/benchmark/output.bin', '/remote/benchmark', root)


def test_edited_plan_is_rejected_before_owner_or_transport(history, monkeypatch, tmp_path):
    plan, row, _, _ = history
    import onnx_splitpoint_tool.workflow.generic_completed_task as workflow
    monkeypatch.setattr(workflow, 'observe_owners', lambda *_: pytest.fail('Owner probe must not run for an invalid plan'))
    monkeypatch.setattr(workflow, 'transport_for', lambda *_a, **_k: pytest.fail('Transport must not be created for an invalid plan'))
    plan.update(schema=subject.PLAN_SCHEMA, identity=row['identity'],
        purpose='functional_reference_only_not_scientific_native_performance',
        build_allowed=False, fallback_allowed=False, automatic_retry=False, energy_capture=False, hardware_executed=False,
        recipe={'processes': 1, 'completed_tasks': 1000, 'warmup_tasks': 0, 'duration_s': 0., 'completion_runtime_mode': 'strict'})
    descriptor = dump(tmp_path / 'capture/PREPARED.json', plan)
    with pytest.raises(ValueError, match='recipe changed'):
        subject.execute_capture(descriptor['path'])


@pytest.fixture
def finite_plan(tmp_path, monkeypatch):
    """Pure plan schema fixture; no fixture is published as a Native capture."""
    import onnx_splitpoint_tool.workflow.generic_completed_task as workflow
    import onnx_splitpoint_tool.workflow.hardware_acceptance_native as acceptance
    output = tmp_path / 'capture'
    output.mkdir()
    setup = 'orin_nx_deepx_m1_01'
    row = {'identity': {'model_id': 'yolov7_paper', 'case_id': 'b009', 'setup_id': setup},
        'task': 'detection', 'model_id': 'yolov7_paper', 'setup_id': setup, 'case_id': 'b009', 'direction': 'deepx_to_trt',
        'precision': 'float32_layout_fp16', 'original_run': str(tmp_path / 'old'), 'remote_python_executable': '/venv/bin/python3',
        'expected_runtime_artifacts': {'part1_runtime': {'path': '/cache/model.dxnn', 'sha256': 'dxnn'}},
        'input_binding': {'entries': [{'shape': [640,640,3], 'dtype': 'uint8', 'remote_path': '/old/input.npy', 'sha256': 'input'}]}}
    command = {'backend': 'deepx_to_trt', 'model': 'yolov7_paper', 'setup_id': setup, 'eval_run_id': 'original',
        'native_split_quality_binding': {'binding_sha256': 'quality'}, 'benchmark_set': '/old/benchmark',
        'input_image': '/dataset/image.jpg', 'input_image_sha256': 'image', 'python_executable': '/venv/bin/python',
        'artifacts': {'python_executable': {'sha256': 'python'}}}
    monkeypatch.setattr(subject, '_historical', lambda *_: (row, {}, command))
    monkeypatch.setattr(subject, 'successful_runtime_argv', lambda *_, **__: ['/venv/bin/python', '-u', 'normal_runner'])
    monkeypatch.setattr(workflow, 'bound_targets', lambda *_: {setup: {'id': setup}})
    monkeypatch.setattr(acceptance, '_benchmark_case_inputs', lambda *_: [])
    source = tmp_path / 'code.py'
    source.write_text('# ordinary runner fixture\n')
    sources = {name: subject.sha(source) for name in (
        'scripts/native_deepx_trt_e2e_from_benchmarkset.py', 'onnx_splitpoint_tool/native_detection_postprocess.py')}
    archive = output / 'native-product-bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for name in sources: handle.add(source, arcname=name)
    environment = {'matches_original_interpreter': True, 'missing_modules': [], 'executable_sha256': 'python', 'executable': '/venv/bin/python3'}
    env_source = dump(tmp_path / setup / 'runtime-environment.json', environment)
    quality = dump(output / 'native_split_quality_binding.json', command['native_split_quality_binding'])
    session = 'native_functional_' + '1' * 32
    remote = '/home/nx/splitpoint_runs/native_functional_reference/' + session
    benchmark = remote + '/yolov7_paper/benchmark_set'
    native_output = benchmark + '/native_pipeline/b009/deepx_to_trt/float32_layout_fp16'
    argv = subject.functional_argv(command, remote_tool=remote+'/tool', benchmark=benchmark, output=native_output,
        runner_sha=sources['scripts/native_deepx_trt_e2e_from_benchmarkset.py'], quality_binding=remote+'/native_split_quality_binding.json')
    expected = {'/cache/model.dxnn': 'dxnn', '/old/input.npy': 'input', '/dataset/image.jpg': 'image', '/venv/bin/python': 'python'}
    expected.update({remote+'/tool/'+name: value for name,value in sources.items()})
    output_contract = {'source': dump(tmp_path / 'metadata/output_contracts.json', {'fixture': 'sealed declaration'}),
                       'semantic_output_manifest': dump(tmp_path / 'metadata/native_outputs_manifest.json', {'fixture': True}),
                       'declaration': {'model_id': 'yolov7_paper'}}
    monkeypatch.setattr(subject, '_output_contract_source', lambda *_: copy.deepcopy(output_contract))
    expected[benchmark+'/output_contracts.json'] = output_contract['source']['sha256']
    return {'source_row': row, 'schema': subject.PLAN_SCHEMA, 'identity': row['identity'],
        'purpose': 'functional_reference_only_not_scientific_native_performance', 'build_allowed': False,
        'fallback_allowed': False, 'automatic_retry': False, 'energy_capture': False, 'hardware_executed': False,
        'recipe': {'processes': 1, 'completed_tasks': 1, 'warmup_tasks': 0, 'duration_s': 0., 'completion_runtime_mode': 'strict'},
        'execution_id': session, 'remote_root': remote, 'remote_tool_dir': remote+'/tool', 'output_root': str(output),
        'runtime_environment_evidence': env_source, 'expected_interpreter': environment, 'target': {'id': setup},
        'quality_binding_file': quality, 'output_contract_source': output_contract,
        'source_files': sources, 'archive': subject.bound(archive), 'expected_remote_files': expected,
        'item': {'argv': argv, 'env': subject._recipe_environment(row), 'repeat': 1, 'source_benchmark_set': '/old/benchmark',
            'benchmark_case_inputs': [], 'remote_output_root': native_output, 'result_json': native_output+'/deepx_native_fifo_e2e_results.json'}}


def test_rebuilt_finite_plan_passes_without_transport(finite_plan):
    assert subject.validate_capture_plan(finite_plan)['backend'] == 'deepx_to_trt'


@pytest.mark.parametrize('change', [
    lambda p: p['item']['argv'].__setitem__(p['item']['argv'].index('--frames')+1, '1000'),
    lambda p: p['item']['env'].update(ONNX_SPLITPOINT_NATIVE_TRT_BUILD='1'),
    lambda p: p['item']['env'].update(PYTHONPATH='/different/source'),
    lambda p: p['item'].update(repeat=3),
    lambda p: p.update(target={'id': 'other_setup'}),
    lambda p: p['expected_remote_files'].update({'/old/input.npy': 'other'}),
    lambda p: p.update(energy_capture=True),
    lambda p: p.update(build_allowed=True),
    lambda p: p.update(remote_root='/old/benchmark'),
    lambda p: p['expected_interpreter'].update(executable_sha256='other'),
])
def test_plan_tampering_cannot_enable_other_work(finite_plan, change):
    change(finite_plan)
    with pytest.raises(ValueError): subject.validate_capture_plan(finite_plan)
