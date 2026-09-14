"""v2.80.1 reference dispatch: real generated suite, runner and ORT computations."""
from __future__ import annotations

import copy
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime  # Required gate dependency: never silently skip real inference.
from PIL import Image
import pytest

from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness
from onnx_splitpoint_tool.management_reference import generate_management_cpu_reference, _cpu_reference_run

ROOT = Path(__file__).resolve().parents[1]


def make_real_reference_suite(root: Path, task: str) -> Path:
    """Only inputs are synthetic; production writers/runners/inference stay real."""
    suite = root / ('suite_' + task)
    case = suite / 'b001'
    case.mkdir(parents=True)
    dataset = root / ('images_' + task)
    dataset.mkdir()
    samples = []
    for index, rgb in enumerate(((240, 20, 10), (10, 230, 20), (10, 20, 220))):
        filename = f'{index:04d}.png'
        Image.new('RGB', (16, 16), rgb).save(dataset / filename)
        samples.append({'image': filename, 'label_id': index})
        if task == 'detection':
            (dataset / f'{index:04d}.json').write_text(json.dumps({
                'annotations': [{'bbox': [2, 2, 8, 8], 'category_id': 1}],
            }))
    (dataset / 'manifest.json').write_text(json.dumps({'samples': samples}))
    inp = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 16, 16])
    if task == 'classification':
        nodes = [helper.make_node('GlobalAveragePool', ['images'], ['pooled']),
                 helper.make_node('Flatten', ['pooled'], ['output'], axis=1)]
        output_shape = [1, 3]
        initializers = []
    else:
        # A decoded [batch, count, xyxy/score/class] graph. Dependence on the
        # image is real (the sum is multiplied by zero), not a forged JSON.
        detections = np.array([[[2, 2, 10, 10, .95, 0], [0, 0, 1, 1, .01, 0]]], dtype=np.float32)
        initializers = [numpy_helper.from_array(detections, 'detections'),
                        numpy_helper.from_array(np.array(0, np.float32), 'zero')]
        nodes = [helper.make_node('ReduceMean', ['images'], ['mean'], keepdims=0),
                 helper.make_node('Mul', ['mean', 'zero'], ['offset']),
                 helper.make_node('Add', ['detections', 'offset'], ['output'])]
        output_shape = [1, 2, 6]
    graph = helper.make_graph(nodes, 'v2801_real_' + task, [inp],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, case / 'model.onnx')
    (case / 'split_manifest.json').write_text(json.dumps({
        'model_id': 'tiny_' + task, 'full_model': 'model.onnx',
        'part1_model': 'model.onnx', 'part2_model': 'model.onnx',
    }))
    row = {'id': 'ort_cpu', 'provider': 'cpu', 'type': 'onnxruntime',
           'stage1': {'provider': 'cpu', 'type': 'onnxruntime'},
           'stage2': {'provider': 'cpu', 'type': 'onnxruntime'},
           'semantic_reference_only': True, 'canonical_cpu_reference': True,
           'variants': ['full'], 'benchmark_task': task,
           'image_scale': ('imagenet' if task == 'classification' else 'norm'), 'validation_images': str(dataset),
           'validation_max_images': 3, 'validation_budget_authoritative': True, 'validation_items_requested': 3, 'task_quality_gate': {
               'statistics': {'execution_location': 'central_management', 'bootstrap_repetitions': 20},
           }}
    plan = {'model_id': 'tiny_' + task, 'runs': [row]}
    (suite / 'benchmark_plan.json').write_text(json.dumps(plan))
    (suite / 'benchmark_set.json').write_text(json.dumps({
        'model_id': 'tiny_' + task, 'model': 'tiny_' + task,
        'cases': [{'case_id': 'b001', 'folder': 'b001', 'boundary': 1}],
    }))
    refresh_suite_harness(suite, validation_max_images=3)
    # Use a labeled manifest directly; that is the supported classification
    # handover and preserves numeric labels through suite path resolution.
    if task == "classification":
        materialized = json.loads((suite / "benchmark_plan.json").read_text())
        source = Path(materialized["runs"][0]["validation_images"])
        materialized["runs"][0]["validation_images"] = str(source / "manifest.json")
        (suite / "benchmark_plan.json").write_text(json.dumps(materialized))
    return suite


def run_real_reference(root: Path, task: str):
    suite = make_real_reference_suite(root, task)
    model_id = 'tiny_' + task
    run_dir = root / 'evaluation'
    out = run_dir / 'quality_management' / 'references' / model_id
    logs = []
    status = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id=model_id, workers=1, timeout_s=60, log=logs.append)
    return suite, run_dir, out, status, logs


@pytest.mark.parametrize('task', ['classification', 'detection'])
def test_real_cpu_reference_chain(tmp_path: Path, task: str):
    suite, run_dir, out, status, logs = run_real_reference(tmp_path, task)
    assert status['status'] == 'completed', (status, (out / 'management_cpu_reference_stdout.txt').read_text())

    payload = json.loads(Path(status['reference_path']).read_text())
    assert payload['task'] == task and payload['semantic_reference_only'] is True
    assert payload['reference_role'] == 'canonical_cpu_ort'
    assert len(payload['records']) == 3
    assert {r['image_id'] for r in payload['records']} == {'0000.png', '0001.png', '0002.png'}
    if task == 'classification':
        assert all(r['reference']['top1_hit'] is True for r in payload['records'])
    else:
        assert payload['provenance_required'] is True
        assert all(len(r['reference']) == len(r['ground_truth']) == 1 for r in payload['records'])
        for record in payload['records']:
            for field, expected in {'class_id': 0, 'x1': 2., 'y1': 2., 'x2': 10., 'y2': 10.}.items():
                assert record['reference'][0][field] == expected
                assert record['ground_truth'][0][field] == expected
    assert status['return_code'] == 0
    assert status['include_in_latency_fps_energy'] is False
    assert status['include_in_ranking'] is False
    assert status['include_in_pareto'] is False
    assert 'Using providers:' in (out / 'management_cpu_reference_stdout.txt').read_text()
    assert 'CPUExecutionProvider' in (out / 'management_cpu_reference_stdout.txt').read_text()
    assert status.get('setup_id') in (None, '')


@pytest.fixture(scope='module')
def suite_module():
    import runpy
    return runpy.run_path(str(ROOT / 'onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt'))


def reference_context_kwargs():
    row = _cpu_reference_run({'runs': [{'id': 'ort_cpu', 'provider': 'cpu'}]})
    row['task_quality_gate'] = {'statistics': {'execution_location': 'central_management'}}
    binding = {'model_id': 'tiny', 'semantic_reference_only': True, 'excluded_from_performance': True}
    return dict(run_id='management_cpu_reference', provider='cpu', stage1='cpu', stage2='cpu',
        variants=['full'], warmup=0, runs=1, phase_runs=0, throughput_frames=0,
        throughput_warmup_frames=0, measurement_only=False, quality_evidence_only=False,
        native_split_quality_binding='', full_only_quality_identity=None,
        quality_evidence_model_id='tiny', quality_evidence_source_run_id='management_cpu_reference',
        task_quality_gate=row['task_quality_gate'], case_meta={'_run': row,
        '_suite_model_id': 'tiny', '_management_reference_plan': {
            'runs': [copy.deepcopy(row)], 'planned_runs': [copy.deepcopy(row)],
            'management_cpu_reference': binding}})


@pytest.mark.parametrize('marker', ['1', 'true', 'YES', 'on'])
@pytest.mark.parametrize('alias', ['cpu', 'ort_cpu', 'cpu_ort', 'onnxruntime_cpu'])
def test_reference_guard_accepts_existing_cpu_aliases(suite_module, monkeypatch, marker, alias):
    monkeypatch.setenv('ONNX_SPLITPOINT_CPU_REFERENCE_ONLY', marker)
    kw = reference_context_kwargs()
    kw.update(provider=alias, stage1=alias, stage2=alias)
    assert suite_module['_management_cpu_reference_context'](**kw) is True


@pytest.mark.parametrize('change', [
    'cuda', 'hailo8', 'hailo10h', 'deepx', 'tensorrt', 'unknown', 'semantic',
    'variant', 'recipe_variant', 'energy', 'measurement', 'quality_only', 'native_binding',
    'performance', 'ranking', 'pareto', 'phases', 'throughput', 'warmup', 'runs',
    'source_run', 'model', 'plan_model', 'plan_row', 'plan_alias', 'plan_semantics', 'local',
])
def test_reference_guard_cannot_bypass_candidate_contract(suite_module, monkeypatch, change):
    monkeypatch.setenv('ONNX_SPLITPOINT_CPU_REFERENCE_ONLY', '1')
    kw = reference_context_kwargs()
    row = kw['case_meta']['_run']
    if change in {'cuda', 'hailo8', 'hailo10h', 'deepx', 'tensorrt', 'unknown'}:
        kw['provider'] = change
    elif change == 'semantic': row.pop('semantic_reference_only')
    elif change == 'variant': kw['variants'] = ['composed']
    elif change == 'recipe_variant': row['variants'] = ['part1']
    elif change == 'energy': row['energy_eligible'] = True
    elif change == 'measurement': kw['measurement_only'] = True
    elif change == 'quality_only': kw['quality_evidence_only'] = True
    elif change == 'native_binding': kw['native_split_quality_binding'] = '/fixture/binding.json'
    elif change in {'performance', 'ranking', 'pareto'}: row[change + '_eligible'] = True
    elif change == 'phases': kw['phase_runs'] = 1
    elif change == 'throughput': kw['throughput_frames'] = 1
    elif change == 'warmup': kw['warmup'] = 1
    elif change == 'runs': kw['runs'] = 2
    elif change == 'source_run': kw['quality_evidence_source_run_id'] = 'other'
    elif change == 'model': kw['quality_evidence_model_id'] = 'other'
    elif change == 'plan_model': kw['case_meta']['_management_reference_plan']['management_cpu_reference']['model_id'] = 'other'
    elif change == 'plan_row': kw['case_meta']['_management_reference_plan']['runs'][0]['image_scale'] = 'raw'
    elif change == 'plan_alias': kw['case_meta']['_management_reference_plan']['planned_runs'] = []
    elif change == 'plan_semantics': kw['case_meta']['_management_reference_plan']['management_cpu_reference'] = {}
    elif change == 'local': kw['task_quality_gate'] = {'execution_location': 'local'}
    with pytest.raises(RuntimeError, match='management_cpu_reference_context_invalid'):
        suite_module['_management_cpu_reference_context'](**kw)


def test_reference_recipe_without_mode_marker_fails(suite_module, monkeypatch):
    monkeypatch.delenv('ONNX_SPLITPOINT_CPU_REFERENCE_ONLY', raising=False)
    with pytest.raises(RuntimeError, match='management_cpu_reference_context_missing'):
        suite_module['_management_cpu_reference_context'](**reference_context_kwargs())


def test_cpu_provider_alone_does_not_activate_reference_exemption(suite_module, monkeypatch):
    monkeypatch.delenv('ONNX_SPLITPOINT_CPU_REFERENCE_ONLY', raising=False)
    kw = reference_context_kwargs()
    kw['run_id'] = 'ort_cpu'
    kw['case_meta'] = {'_run': {'id': 'ort_cpu', 'provider': 'cpu'}}
    assert suite_module['_management_cpu_reference_context'](**kw) is False


@pytest.mark.parametrize('variant', ['full', 'composed'])
def test_normal_cpu_candidates_still_require_physical_identity(suite_module, monkeypatch, tmp_path, variant):
    monkeypatch.delenv('ONNX_SPLITPOINT_CPU_REFERENCE_ONLY', raising=False)
    case = tmp_path / 'b001'
    case.mkdir()
    # Sentinel is only used to assert rejection BEFORE runner dispatch.
    (case / 'run_split_onnxruntime.py').write_text("raise AssertionError('runner_must_not_start')\n")
    with pytest.raises(RuntimeError, match='identity is incomplete: quality-evidence-eval-id,quality-evidence-setup-id'):
        suite_module['_run_case'](case, run_id='ort_cpu', provider='cpu', stage1='cpu', stage2='cpu',
            image='', preset='', image_scale='imagenet', warmup=0, runs=1, timeout_s=5,
            variants=[variant], task_quality_gate={'execution_location': 'central_management'},
            quality_evidence_model_id='tiny', quality_evidence_source_run_id='ort_cpu')


def test_real_reference_reuse_has_no_second_subprocess_and_new_script_new_generation(tmp_path, monkeypatch):
    import onnx_splitpoint_tool.management_reference as reference_module
    suite, run_dir, out, first, _ = run_real_reference(tmp_path, 'classification')
    assert first['status'] == 'completed', first
    path = Path(first['reference_path'])
    original = path.read_bytes()
    original_popen = reference_module.subprocess.Popen
    def unexpected_spawn(*args, **kwargs):
        raise AssertionError('cache_hit_must_not_spawn_a_process')
    monkeypatch.setattr(reference_module.subprocess, 'Popen', unexpected_spawn)
    cached = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id='tiny_classification', workers=1, timeout_s=60)
    assert cached['status'] == 'cache_hit'
    assert cached['reference_path'] == first['reference_path']
    assert path.read_bytes() == original
    monkeypatch.setattr(reference_module.subprocess, 'Popen', original_popen)
    # Refresh a known old harness via production writer; it repairs stale bytes
    # without rewriting the already published immutable reference generation.
    harness = suite / 'benchmark_suite.py'
    harness.write_text(harness.read_text() + '\n# fixture source contract changed\n')
    changed = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id='tiny_classification', workers=1, timeout_s=60)
    assert changed['status'] == 'completed'
    assert changed['reference_path'] != first['reference_path']
    assert path.read_bytes() == original
    before_refresh = harness.read_bytes()
    refreshed = refresh_suite_harness(suite, validation_max_images=3)
    assert refreshed['suite_script_updated'] is True
    assert harness.read_bytes() != before_refresh
    assert '_management_cpu_reference_context' in harness.read_text()
    assert path.read_bytes() == original


@pytest.mark.parametrize('variant', ['full', 'composed'])
def test_normal_candidate_preserves_all_dispatch_ids(suite_module, monkeypatch, tmp_path, variant):
    monkeypatch.delenv('ONNX_SPLITPOINT_CPU_REFERENCE_ONLY', raising=False)
    case = tmp_path / 'b001'
    case.mkdir()
    (case / 'run_split_onnxruntime.py').write_text(
        "import sys,json\nfrom pathlib import Path\n"
        "Path('dispatched_argv.json').write_text(json.dumps(sys.argv))\n"
        "raise SystemExit(7)  # dispatch observation only, no reference result\n")
    suite_module['_run_case'](case, run_id='ort_cpu', provider='cpu', stage1='cpu', stage2='cpu',
        image='', preset='', image_scale='imagenet', warmup=0, runs=1, timeout_s=5,
        variants=[variant], task_quality_gate={'execution_location': 'central_management'},
        quality_evidence_eval_id='eval_real_context', quality_evidence_model_id='tiny',
        quality_evidence_setup_id='physical_setup', quality_evidence_source_run_id='ort_cpu')
    argv = json.loads((case / 'dispatched_argv.json').read_text())
    for option, expected in {
        '--quality-evidence-eval-id': 'eval_real_context', '--quality-evidence-model-id': 'tiny',
        '--quality-evidence-setup-id': 'physical_setup', '--quality-evidence-source-run-id': 'ort_cpu',
    }.items():
        assert argv[argv.index(option) + 1] == expected
    assert '--quality-evidence-only' not in argv


@pytest.mark.parametrize('rtype', ['hailo', 'deepx', 'matrix'])
def test_flagged_non_ort_recipe_rejected_before_backend_dispatch(tmp_path, rtype):
    import subprocess
    suite = make_real_reference_suite(tmp_path, 'classification')
    plan = json.loads((suite / 'benchmark_plan.json').read_text())
    plan['runs'][0].update(id='management_cpu_reference', type=rtype)
    (suite / 'benchmark_plan.json').write_text(json.dumps(plan))
    env = dict(os.environ, ONNX_SPLITPOINT_CPU_REFERENCE_ONLY='1')
    result = subprocess.run([sys.executable, str(suite / 'benchmark_suite.py'),
        '--plan', str(suite / 'benchmark_plan.json'), '--run-id', 'management_cpu_reference'],
        env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert 'management_cpu_reference_context_invalid:selected_run_or_run_type' in result.stderr
    assert not list(suite.glob('benchmark_results_*.json'))
