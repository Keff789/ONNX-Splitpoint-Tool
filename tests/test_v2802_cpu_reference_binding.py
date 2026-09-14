"""Real legacy Generator → BenchmarkSet → Suite → ORT reference regression.

Only ONNX models/images are synthetic. No manually written BenchmarkSet,
reference output, runner, compiler result, or historical fixture is used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime  # Required: real CPU inference must never silently skip.
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
import pytest

from onnx_splitpoint_tool.management_reference import generate_management_cpu_reference
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import materialize_legacy_benchmark_set


def make_generated_reference_suite(root: Path, task: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model_id = 'tiny_' + task
    dataset = root / 'dataset'
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
    manifest = dataset / 'manifest.json'
    manifest.write_text(json.dumps({'samples': samples}))
    if task == 'classification':
        nodes = [helper.make_node('GlobalAveragePool', ['images'], ['pooled']),
                 helper.make_node('Flatten', ['pooled'], ['flat'], axis=1),
                 helper.make_node('Identity', ['flat'], ['output'])]
        output_shape, initializers = [1, 3], []
    else:
        detections = np.array([[[2, 2, 10, 10, .95, 0], [0, 0, 1, 1, .01, 0]]], dtype=np.float32)
        initializers = [numpy_helper.from_array(detections, 'detections'),
                        numpy_helper.from_array(np.array(0, np.float32), 'zero')]
        nodes = [helper.make_node('ReduceMean', ['images'], ['mean'], keepdims=0),
                 helper.make_node('Mul', ['mean', 'zero'], ['offset']),
                 helper.make_node('Add', ['detections', 'offset'], ['output'])]
        output_shape = [1, 2, 6]
    graph = helper.make_graph(nodes, 'v2802_real_' + task,
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    model_path = root / (model_id + '.onnx')
    onnx.save(model, model_path)
    run_dir = root / 'evaluation'
    model_dir = run_dir / 'models' / model_id
    profile = {
        'id': 'v2802_real_generator',
        'targets': ['cpu_ort'],
        'run_profiles': [{'id': 'cpu_ort', 'full': 'cpu_ort', 'enabled': True}],
        'quality_gate': {'statistics': {'execution_location': 'central_management', 'bootstrap_repetitions': 20}},
        'validation': {'validation_images': str(manifest if task == 'classification' else dataset),
                       'validation_max_images': 3, 'image_scale': 'imagenet' if task == 'classification' else 'norm'},
        'hailo_build': {'force_build': False},
        'deepx_build': {'force_build': False},
    }
    logs = []
    result = materialize_legacy_benchmark_set(
        model_id=model_id, model_path=str(model_path), model_dir=model_dir,
        run_dir=run_dir, profile_id=profile['id'], run_id=run_dir.name,
        prediction={'model_id': model_id, 'ranked_candidates': [{'boundary': 1}]},
        candidate_plan={'model_id': model_id, 'requested_cases': 1,
                        'selected_candidates': [{'case_id': 'b001', 'split_index': 1}]},
        targets=['cpu_ort'], profile_payload=profile,
        row={'id': model_id, 'task': task, 'resolved_path': str(model_path)},
        options=WorkflowOptions(profile='', out=str(root), no_remote=True), log=logs.append,
    )
    (root / 'generation.log').write_text('\n'.join(logs))
    assert result.status == 'ok', (result, logs)
    suite = Path(result.suite_dir)
    contract = json.loads((suite / 'benchmark_set.json').read_text())
    assert contract['schema'] == 'onnx-splitpoint/benchmark-set'
    assert contract['model_name'] == model_id
    assert contract['model'] == 'models/' + model_id + '.onnx'
    assert contract['cases'][0]['boundary'] == 1
    return suite


@pytest.mark.parametrize('task', ['classification', 'detection'])
def test_real_generated_reference_chain(tmp_path, task):
    suite = make_generated_reference_suite(tmp_path, task)
    out = tmp_path / 'references' / ('tiny_' + task)
    status = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id='tiny_' + task, workers=1, timeout_s=60)
    assert status['status'] == 'completed', (status, (out / 'management_cpu_reference_stdout.txt').read_text())
    payload = json.loads(Path(status['reference_path']).read_text())
    assert payload['task'] == task
    assert payload['semantic_reference_only'] is True
    assert payload['reference_role'] == 'canonical_cpu_ort'
    assert len(payload['records']) == 3
    assert {r['image_id'] for r in payload['records']} == {'0000.png', '0001.png', '0002.png'}
    if task == 'classification':
        assert all(r['reference']['top1_hit'] is True for r in payload['records'])
    else:
        assert payload['provenance_required'] is True
        assert all(len(r['reference']) == len(r['ground_truth']) == 1 for r in payload['records'])
    assert status['return_code'] == 0
    assert all(status[field] is False for field in ('include_in_latency_fps_energy', 'include_in_ranking', 'include_in_pareto'))
    assert 'CPUExecutionProvider' in (out / 'management_cpu_reference_stdout.txt').read_text()


@pytest.mark.parametrize('change', [
    'launcher_model', 'suite_model_name', 'suite_explicit_model_id',
    'plan_model_id', 'missing_identity', 'matching_filename_only',
])
def test_generated_reference_rejects_wrong_or_missing_model_binding(tmp_path, change):
    suite = make_generated_reference_suite(tmp_path, 'classification')
    expected_id = 'tiny_classification'
    contract_path = suite / 'benchmark_set.json'
    contract = json.loads(contract_path.read_text())
    if change == 'launcher_model':
        expected_id = 'different_model'
    elif change == 'suite_model_name':
        contract['model_name'] = 'different_model'
    elif change == 'suite_explicit_model_id':
        # An explicit ID must not be replaced by a convenient model_name.
        contract['model_id'] = 'different_model'
    elif change == 'plan_model_id':
        path = suite / 'benchmark_plan.json'
        plan = json.loads(path.read_text())
        plan['model_id'] = 'different_model'
        path.write_text(json.dumps(plan))
    else:
        contract.pop('model_id', None)
        contract.pop('model_name', None)
        if change == 'missing_identity':
            contract.pop('model', None)
        # A path with the matching basename is not a logical model ID.
    contract_path.write_text(json.dumps(contract))
    out = tmp_path / 'reference'
    status = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id=expected_id, workers=1, timeout_s=60)
    assert status['status'] == 'failed'
    assert status['return_code'] == 1
    assert status['failure_stage'] == 'reference_process'
    assert status['exception_type'] == 'RuntimeError'
    message = status['exception_message']
    assert 'management_cpu_reference_context_invalid:' in message
    detail = json.loads(message.split('; model_binding=', 1)[1])
    assert detail['expected_model_id'] == expected_id
    assert detail['actual_quality_evidence_model_id'] == expected_id
    assert detail['benchmark_set_identity'] == {key: contract.get(key) for key in ('model_id', 'model_name', 'model')}
    assert 'CPUExecutionProvider' not in (out / 'management_cpu_reference_stdout.txt').read_text()
    assert not list(out.rglob('canonical_cpu_reference.json'))
    assert not list((out / 'workspaces').iterdir())


def test_generated_reference_cache_cannot_be_rebound_to_other_model(tmp_path, monkeypatch):
    import onnx_splitpoint_tool.management_reference as reference_module
    suite = make_generated_reference_suite(tmp_path, 'classification')
    out = tmp_path / 'reference'
    kwargs = dict(suite_dir=suite, output_dir=out, workers=1, timeout_s=60)
    first = generate_management_cpu_reference(model_id='tiny_classification', **kwargs)
    assert first['status'] == 'completed', first
    immutable_path = Path(first['reference_path'])
    immutable_bytes = immutable_path.read_bytes()
    def no_spawn(*args, **kwargs):
        raise AssertionError('cached reference must not start a second process')
    monkeypatch.setattr(reference_module.subprocess, 'Popen', no_spawn)
    cached = generate_management_cpu_reference(model_id='tiny_classification', **kwargs)
    assert cached['status'] == 'cache_hit'
    assert cached['reference_path'] == first['reference_path']
    wrong = generate_management_cpu_reference(model_id='different_model', **kwargs)
    assert wrong['status'] == 'failed'
    assert 'management_cpu_reference_context_invalid:model_binding' in wrong['error']
    assert '"expected_model_id": "different_model"' in wrong['error']
    assert '"benchmark_set_model_id": "tiny_classification"' in wrong['error']
    assert not wrong.get('reference_path')
    assert immutable_path.read_bytes() == immutable_bytes
    # A rejected request must not poison the valid immutable generation.
    recovered = generate_management_cpu_reference(model_id='tiny_classification', **kwargs)
    assert recovered['status'] == 'cache_hit'
    assert recovered['reference_path'] == first['reference_path']


def test_generated_reference_source_change_publishes_new_immutable_generation(tmp_path):
    suite = make_generated_reference_suite(tmp_path, 'detection')
    kwargs = dict(suite_dir=suite, output_dir=tmp_path / 'reference',
                  model_id='tiny_detection', workers=1, timeout_s=60)
    first = generate_management_cpu_reference(**kwargs)
    assert first['status'] == 'completed', first
    original_path = Path(first['reference_path'])
    original_bytes = original_path.read_bytes()
    contract = json.loads((suite / 'benchmark_set.json').read_text())
    full_path = suite / contract['model']
    model = onnx.load(full_path)
    model.producer_name = 'v2802 changed real ONNX source identity'
    onnx.save(model, full_path)
    second = generate_management_cpu_reference(**kwargs)
    assert second['status'] == 'completed', second
    assert second['source_contract_sha256'] != first['source_contract_sha256']
    assert second['reference_path'] != first['reference_path']
    assert original_path.read_bytes() == original_bytes
    assert json.loads(Path(second['reference_path']).read_text())['records']


def test_generated_reference_cache_resumes_earlier_model_generation(tmp_path):
    suite = make_generated_reference_suite(tmp_path, 'classification')
    kwargs = dict(suite_dir=suite, output_dir=tmp_path / 'reference', workers=1, timeout_s=60)
    first = generate_management_cpu_reference(model_id='tiny_classification', **kwargs)
    assert first['status'] == 'completed', first
    path = Path(first['reference_path'])
    immutable_bytes = path.read_bytes()
    contract_path = suite / 'benchmark_set.json'
    original_contract = contract_path.read_bytes()
    contract = json.loads(original_contract)
    # A separately identified version of the same tiny model shares an output
    # root but has its own immutable source contract and reference generation.
    contract['model_name'] = 'tiny_classification_revision_b'
    contract_path.write_text(json.dumps(contract))
    second = generate_management_cpu_reference(model_id=contract['model_name'], **kwargs)
    assert second['status'] == 'completed', second
    assert second['source_contract_sha256'] != first['source_contract_sha256']
    assert second['reference_path'] != first['reference_path']
    contract_path.write_bytes(original_contract)
    restored = generate_management_cpu_reference(model_id='tiny_classification', **kwargs)
    assert restored['status'] == 'cache_hit', restored
    assert restored['reference_path'] == first['reference_path']
    assert path.read_bytes() == immutable_bytes


@pytest.mark.parametrize('schema', ['onnx-splitpoint/benchmark-set', None])
def test_unbound_reference_cache_has_no_model_identity(schema):
    from onnx_splitpoint_tool.management_reference import _verify_reference_reuse_model_binding
    contract = {'cases': [{'case_id': 'b001'}]}
    if schema:
        contract['schema'] = schema
    with pytest.raises(RuntimeError, match='management_cpu_reference_context_invalid:model_binding'):
        _verify_reference_reuse_model_binding(model_id='tiny_classification', contract=contract,
            plan={}, previous_status={}, source_contract_sha256='a' * 64)
    if schema:
        # A modern malformed contract cannot borrow an ID from cached status.
        with pytest.raises(RuntimeError, match='management_cpu_reference_context_invalid:model_binding'):
            _verify_reference_reuse_model_binding(model_id='tiny_classification', contract=contract,
                plan={}, previous_status={'status': 'completed', 'model_id': 'tiny_classification',
                                        'source_contract_sha256': 'a' * 64},
                source_contract_sha256='a' * 64)
