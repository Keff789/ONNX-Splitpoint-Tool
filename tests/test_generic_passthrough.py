"""Canonical original-image P2 feeds; CPU only, no inference or hardware."""
import ast
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool.runners import generic_passthrough as full
from onnx_splitpoint_tool.runners.generic_mixed_control import run_mixed_control, MixedControlError
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from tests.test_v2912_generic_split_completion import _module, TEMPLATE


def descriptor(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def canonical(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    from onnx_splitpoint_tool.preprocessing_contract import resolve_image_preprocessing_contract
    contract, _ = resolve_image_preprocessing_contract(task="detection", target_hw=(64, 64))
    images, arrays, bindings, entries = [], [], [], []
    source = tmp_path / 'SYNTHETIC_normal_quality_request.json'; source.write_text('{}')
    for index, wh in enumerate(((80, 60), (60, 80))):
        path = tmp_path / (str(index) + '.png')
        Image.new('RGB', wh, color=(31 + index * 100, 53, 79)).save(path)
        value = module._load_image_as_nchw(path, target_hw=(64, 64), dtype=np.float32,
            scale='norm', letterbox=True, preprocessing_contract=contract)
        images.append(path); arrays.append(value)
        entries.append({'image_id': path.name, 'source_image_sha256': descriptor(path)['sha256'],
                        'prepared_inputs': {'images': full._snapshot(value)},
                        'runtime_inputs': {'images': full._snapshot(np.ascontiguousarray(value))}})
    proof = {'per_image': entries, 'image_preprocessing_contract': contract,
             'representative_input': {'image_id': images[0].name},
             'runtime_context': {'boundary_runtime': {'passthrough_inputs': {'images': 'images'}}}}
    for index, path in enumerate(images):
        bindings.append(full.binding_from_quality(proof, path.name, descriptor(path), str(index) * 64, descriptor(source)))
    spec = {'shape': [1, 3, 64, 64], 'dtype': 'float32'}
    row = {'task': 'detection', 'quality_reference': {'request': descriptor(source)},
        'input_binding': {'prepared_input': {'sha256': '0' * 64}},
        'boundary_observation': {'passthrough_inputs': {'images': 'images'}, 'part2_inputs': {'images': spec}},
        'canonical_full_input_bindings': bindings}
    full.validate_quality_bindings(row, proof)
    row['runtime_source_files'] = [descriptor(path) for path in images]
    return module, row, proof, arrays


def test_actual_normal_image_loader_matches_recorded_canonical_quality_before_timer(canonical):
    module, row, _, arrays = canonical
    calls = []
    def load(*args, **kwargs):
        calls.append(args[0]); return module._load_image_as_nchw(*args, **kwargs)
    feeds = full.prepare_full_feeds(row, load)
    assert len(calls) == 2
    for index in range(2):
        np.testing.assert_array_equal(feeds[index]['images'], arrays[index])
        assert not feeds[index]['images'].flags.writeable
    assert not np.array_equal(feeds[0]['images'], feeds[1]['images'])


def test_normal_runtime_projection_stages_bound_source_images_without_changing_source_row(canonical, monkeypatch, tmp_path):
    from tests.test_generic_only_plan import scope_fixture
    from onnx_splitpoint_tool import generic_only_plan
    module, original, _, arrays = canonical
    _, _, _, row = scope_fixture(tmp_path / 'scope', monkeypatch)
    row['canonical_full_input_bindings'] = deepcopy(original['canonical_full_input_bindings'])
    row['quality_reference'] = deepcopy(original['quality_reference'])
    row['boundary_observation'] = deepcopy(original['boundary_observation'])
    before = deepcopy(row)
    config, staged = generic_only_plan.runtime_configuration(row, str(tmp_path / 'REMOTE_FIXTURE'), 'synthetic-session')
    assert row == before
    images = [entry for entry in staged if entry['role'].startswith('canonical_original_input_')]
    assert len(images) == 2
    for item in images:
        target = Path(item['remote']); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item['local'], target)
        assert descriptor(target)['sha256'] == item['sha256']
    config['runtime_source_files'] = [{'path': item['remote'], 'sha256': item['sha256']} for item in images]
    feeds = full.prepare_full_feeds(config, module._load_image_as_nchw)
    for index in range(2):
        np.testing.assert_array_equal(feeds[index]['images'], arrays[index])


@pytest.mark.parametrize('fault', ['normalization', 'source_bytes', 'snapshot', 'mapping', 'prepared_baseline'])
def test_wrong_source_normalization_and_canonical_quality_binding_fail_closed(canonical, fault):
    module, row, proof, arrays = canonical
    if fault == 'normalization':
        def wrong(*args, **kwargs):
            return module._load_image_as_nchw(*args, **{**kwargs, 'scale': 'raw', 'preprocessing_contract': None})
        with pytest.raises(ValueError, match='bytes differ'):
            full.prepare_full_feeds(row, wrong)
        return
    if fault == 'source_bytes':
        Path(row['canonical_full_input_bindings'][1]['source_image']['path']).write_bytes(b'changed source image')
    elif fault == 'snapshot':
        row['canonical_full_input_bindings'][1]['inputs']['images']['tensor_sha256'] = 'f' * 64
    elif fault == 'mapping':
        row['boundary_observation']['passthrough_inputs']['other'] = 'images'
    else:
        row['canonical_full_input_bindings'][0]['prepared_input_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        full.validate_quality_bindings(row, proof)


def mixed_case(wrong_baseline=False):
    rgb = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
    canonical = [np.transpose(value.astype(np.float32) / np.float32(255), (2, 0, 1))[None] for value in rgb]
    specs = [{'path': '/SYNTHETIC/' + str(i) + '.npy', 'sha256': str(i) * 64,
              'shape': [2, 2, 3], 'dtype': 'uint8', 'input_name': 'images'} for i in range(2)]
    sequence = [(i % 2) ^ (i >= 16) for i in range(32)]
    tasks = [{'task_id': str(i), 'prepared_input': specs[index], 'feed': {'images': rgb[index]},
              'full_feed': {'images': canonical[index]}} for i, index in enumerate(sequence)]
    cut = {'shape': [1, 6], 'dtype': 'float32'}
    cut2 = {'shape': [1, 7], 'dtype': 'float32'}
    original = {'shape': [1, 3, 2, 2], 'dtype': 'float32'}
    boundary = {'schema_version': 1, 'minimum_boundary_tensors': 2,
        'part1_outputs': {'a': cut, 'b': cut2}, 'part2_inputs': {'a': cut, 'b': cut2, 'images': original},
        'part2_to_part1': {'a': 'a', 'b': 'b'}, 'passthrough_inputs': {'images': 'images'},
        'transfer_contract': {'boundary_mode': 'canonical_float32'}}
    config = {'task': 'classification', 'measurement_mode': 'generic_only_completed_task',
        'stage1_execution': {'policy': 'synchronous', 'inflight': 1}, 'queue_depth': 2, 'timeout_s': 3,
        'input_binding': {'prepared_input': specs[0]}, 'boundary_observation': boundary}
    observed = []
    def stage1(feed):
        value = int(feed['images'].flat[0])
        return {'a': np.roll(np.arange(6, dtype=np.float32), value)[None].copy(),
                'b': (np.arange(7, dtype=np.float32) + value)[None].copy()}
    def transfer(outputs, feed):
        image = canonical[0] if wrong_baseline else feed['images']
        observed.append(full._snapshot(image))
        return {**outputs, 'images': image}
    def stage2(inputs, feed):
        return {'logits': inputs['a'] + np.float32(inputs['images'].mean())}
    def runtime():
        return {'part1_io': {'canonical_outputs': ['a', 'b']}, 'part2_io': {'inputs': boundary['part2_inputs']},
                **{key: boundary[key] for key in ('part2_to_part1', 'passthrough_inputs', 'transfer_contract')}}
    args = dict(config=config, original_sources={'SYNTHETIC': 'CPU-only control'}, serial_stage1=stage1,
        transfer=transfer, stage2=stage2, processor_factory=ClassificationCompletion,
        boundary_observation=boundary, boundary_runtime=runtime)
    return tasks, args, canonical, observed


def test_all_32_serial_and_pipeline_passthroughs_are_bound_to_each_tasks_original_image():
    tasks, args, canonical, observed = mixed_case()
    report = run_mixed_control(tasks, **args)
    assert report['status'] == 'passed' and len(observed) == 64
    expected = [full._snapshot(canonical[(i % 2) ^ (i >= 16)]) for i in range(32)]
    assert observed[:32] == expected and observed[32:] == expected
    assert all(item['boundary_observation']['passed'] for item in report['observations'])


def test_matching_serial_and_pipeline_wrong_baseline_is_rejected_by_task_bytes():
    tasks, args, _, _ = mixed_case(wrong_baseline=True)
    with pytest.raises(MixedControlError, match="passthrough bytes differ"):
        run_mixed_control(tasks, **args)


@pytest.mark.parametrize('passthrough', [True, False])
def test_single_cut_with_original_image_remains_one_truthful_boundary_tensor(passthrough):
    from onnx_splitpoint_tool.runners.generic_mixed_control import normalize_boundary_observation
    tasks, args, _, _ = mixed_case()
    boundary = args['boundary_observation']
    boundary['minimum_boundary_tensors'] = 1
    boundary['part1_outputs'].pop('b'); boundary['part2_inputs'].pop('b'); boundary['part2_to_part1'].pop('b')
    if not passthrough:
        boundary['passthrough_inputs'] = {}; boundary['part2_inputs'].pop('images')
        with pytest.raises(ValueError, match='at least two'):
            normalize_boundary_observation(boundary)
    else:
        assert len(normalize_boundary_observation(boundary)['part1_outputs']) == 1
        old_stage1, old_runtime = args['serial_stage1'], args['boundary_runtime']
        def stage1(feed):
            return {'a': old_stage1(feed)['a']}
        def transfer(outputs, feed):
            return {'a': outputs['a'], 'images': feed['images']}
        def runtime():
            return {**old_runtime(), 'part1_io': {'canonical_outputs': ['a']}}
        args.update(serial_stage1=stage1, transfer=transfer, boundary_runtime=runtime)
        report = run_mixed_control(tasks, **args)
        assert report['status'] == 'passed'
        assert report['boundary_observation']['observed_boundary_tensor_count'] == 1
        assert report['mixed_completed_count'] == 32


def test_passthrough_without_explicit_task_canonical_feed_cannot_fall_back_to_sdk_bytes():
    tasks, args, _, _ = mixed_case()
    tasks[1].pop('full_feed')
    with pytest.raises(ValueError, match="task's canonical full feed"):
        run_mixed_control(tasks, **args)


def actual_transfer(module, io, cut_names, baseline):
    """Execute the unchanged real transfer with metadata-only TRT leaves."""
    tree = ast.parse(TEMPLATE.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    names = {'_prepare_native_boundary_input', '_canonicalize_hailo10_part1_outputs', '_build_stage2_inputs'}
    functions = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 3
    outer = ast.parse('''def make(io, cut_names, baseline):
    p2_in_map = {name: name for name in cut_names}
    p2_in_map_src_keys = tuple(sorted(cut_names))
    deepx_p2_tuned_map = None
    deepx_p2_contract_report = {}
    feeds_full = {'images': baseline}
    native_p2_sess = SimpleNamespace(input_names=list(io), describe_io=lambda: {'inputs': io})
    hailo_p1 = hailo_p2 = deepx_p2 = ort_p2_sess = None
    args = SimpleNamespace()
    _boundary_mode = 'canonical_float32'
    _record_boundary_event = lambda event: None
    _cached_boundary_buffer = None
''').body[0]
    outer.body.extend(functions)
    outer.body.append(ast.Return(ast.Name('_build_stage2_inputs', ast.Load())))
    namespace = {**vars(module), 'SimpleNamespace': SimpleNamespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])), str(TEMPLATE), 'exec'), namespace)
    transfer = namespace['make'](io, cut_names, baseline)
    completion = next(node for node in main.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'generic_completion_config'
        and 'canonical_full_feeds' in ast.unparse(node))
    callback = next(node for node in ast.walk(completion) if isinstance(node, ast.FunctionDef)
                    and node.name == 'completion_transfer')
    return transfer, callback


@pytest.mark.parametrize('case', ['b236', 'b257', 'b278', 'b280', 'b306'])
def test_actual_fifteen_late_cells_normal_transfer_keeps_canonical_original_image(monkeypatch, tmp_path, case):
    """Actual archived I/O +actual RGB tensors; no TRT/SDK or model inference."""
    module = _module(monkeypatch)
    root = Path('/home/kmika/Reports/generic_detectors20_multitensor_20261021T163124Z')
    with Path('/home/kmika/Reports/cache_audit_420_iprfh635/cache_matrix_420.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['model_id'] == 'yolov7_paper' and row['cut_id'] == case]
    assert {row['backend'] for row in rows} == {'h8', 'h10', 'deepx'}
    binding = json.loads((root / 'cache/smoke_b039_source_bindings.json').read_text())['rows'][0]
    from onnx_splitpoint_tool.preprocessing_contract import resolve_image_preprocessing_contract
    contract, _ = resolve_image_preprocessing_contract(task='detection', target_hw=(640, 640))
    encoded, canonical = [], []
    for index, source in enumerate(binding['prepared_inputs']):
        path = Path(source['path'])
        assert descriptor(path)['sha256'] == source['sha256']
        value = np.load(path, allow_pickle=False); encoded.append(value)
        # Preserve the actual existing prepared RGB raster. Normal numeric
        # conversion is executed by the unchanged image loader, never copied.
        image = tmp_path / (str(index) + '.png'); Image.fromarray(value).save(image)
        canonical.append({'images': module._load_image_as_nchw(image, target_hw=(640, 640),
            dtype=np.float32, scale='norm', letterbox=True, preprocessing_contract=contract)})
    for row in rows:
        io = {entry['name']: {key: entry[key] for key in ('shape', 'dtype')}
              for entry in json.loads(row['part2_inputs'])}
        cut_names = [entry['name'] for entry in json.loads(row['cut_boundary_tensors'])]
        outputs = {name: np.zeros(io[name]['shape'], dtype=io[name]['dtype']) for name in cut_names}
        transfer, callback = actual_transfer(module, io, cut_names, encoded[0])
        namespace = {'_build_stage2_inputs': transfer, 'canonical_full_feeds': canonical}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(TEMPLATE), 'exec'), namespace)
        fixed = namespace['completion_transfer'](outputs)
        second = namespace['completion_transfer'](outputs, canonical[1])
        np.testing.assert_array_equal(fixed['images'], canonical[0]['images'])
        np.testing.assert_array_equal(second['images'], canonical[1]['images'])
        assert not np.array_equal(fixed['images'], second['images'])
        wrong = transfer(outputs)['images']  # Precisely the previous encoded baseline path.
        assert wrong.shape == fixed['images'].shape and wrong.dtype == fixed['images'].dtype
        assert not np.array_equal(wrong, fixed['images'])
        assert wrong.max() > 1 and fixed['images'].max() <= 1


@pytest.mark.parametrize('provider', ['hailo8', 'hailo10h', 'deepx_m1'])
@pytest.mark.parametrize('fault', [None, 'baseline', 'runtime_input', 'saved_bytes'])
def test_bound_control_and_saved_proof_require_the_same_image_for_p1_and_p2(monkeypatch, tmp_path, provider, fault):
    """Real sidecar, adapters, FIFO, completion and saved reader; SDK math synthetic."""
    from tests.test_generic_only_generated import prepared_case, boundary, image_value, HailoSession, DeepXSession
    from tests.test_v272_detection_completion_runtime import _decoded_outputs
    from onnx_splitpoint_tool.runners.generic_async import HailoAsyncStage1, DeepXAsyncStage1
    from onnx_splitpoint_tool.runners import generic_mixed_control_binding as bound
    from onnx_splitpoint_tool.preprocessing_contract import resolve_image_preprocessing_contract
    module = _module(monkeypatch)
    backend = 'hailo' if provider == 'hailo10h' else 'deepx'
    config, output, baseline = prepared_case(tmp_path, backend)
    config['original_run_cfg']['stage1_provider'] = provider
    if provider == 'hailo8':
        config['stage1_execution'] = {'policy': 'synchronous', 'inflight': 1}
    sidecar = Path(config['mixed_input_control']['path'])
    payload = json.loads(sidecar.read_text())
    payload['stage1_execution'] = config['stage1_execution']
    config['boundary_observation']['part2_inputs']['images'] = {'shape': [1, 3, 64, 64], 'dtype': 'float32'}
    config['boundary_observation']['passthrough_inputs'] = {'images': 'images'}
    payload['boundary_observation'] = deepcopy(config['boundary_observation'])
    request = tmp_path / 'SYNTHETIC_SOURCE_REQUEST.json'; request.write_text('{}')
    config['quality_reference'] = {'request': descriptor(request)}
    contract, _ = resolve_image_preprocessing_contract(task='detection', target_hw=(64, 64))
    entries = []
    for item in payload['inputs']:
        source = item['provenance']['synthetic_image']
        value = module._load_image_as_nchw(Path(source['path']), target_hw=(64, 64), dtype=np.float32,
            scale='norm', letterbox=True, preprocessing_contract=contract)
        rgb = np.load(item['prepared_input']['path'], allow_pickle=False)
        entries.append({'image_id': Path(source['path']).name, 'source_image_sha256': source['sha256'],
            'prepared_inputs': {'images': full._snapshot(value)}, 'runtime_inputs': {'images': full._snapshot(rgb)}})
    synthetic_proof = {'image_preprocessing_contract': contract, 'per_image': entries}
    config['canonical_full_input_bindings'] = [full.binding_from_quality(synthetic_proof, entry['image_id'],
        item['provenance']['synthetic_image'], item['prepared_input']['sha256'], descriptor(request))
        for entry, item in zip(entries, payload['inputs'])]
    for item, binding in zip(payload['inputs'], config['canonical_full_input_bindings']):
        item['canonical_full_input_binding'] = deepcopy(binding)
    sidecar.write_text(json.dumps(payload)); config['mixed_input_control'] = descriptor(sidecar)
    for source in config['runtime_source_files']:
        if source['path'] == str(sidecar):
            source['sha256'] = config['mixed_input_control']['sha256']
    config['runtime_source_files'].extend([item['provenance']['synthetic_image'] for item in payload['inputs']])
    if str(Path(full.__file__)) not in {item['path'] for item in config['runtime_source_files']}:
        config['runtime_source_files'].append(descriptor(Path(full.__file__)))
    preparation_calls = []
    def load(*args, **kwargs):
        preparation_calls.append(args[0]); return module._load_image_as_nchw(*args, **kwargs)
    feeds = full.prepare_full_feeds(config, load)
    adapter = None
    if provider != 'hailo8':
        session = HailoSession() if provider == 'hailo10h' else DeepXSession()
        cls = HailoAsyncStage1 if provider == 'hailo10h' else DeepXAsyncStage1
        adapter = cls(session, lambda: {'images': baseline}, inflight=config['stage1_execution']['inflight'])
    def stage1(feed):
        return boundary(image_value(feed['images']))
    def transfer(values, feed):
        return {'trt_a': values['cut_a'], 'trt_b': values['cut_b'],
                'images': feeds[0]['images'] if fault == 'baseline' else feed['images']}
    def stage2(inputs, feed):
        values = _decoded_outputs()
        shift = int(inputs['trt_a'][0, 0]) + int(inputs['trt_b'][0, 0]) - 10 + float(inputs['images'].mean())
        values['detections'][0, 0, (0, 2)] += shift
        return values
    def live():
        boundary_contract = config['boundary_observation']
        return {'part1_io': {'canonical_outputs': ['cut_a', 'cut_b']},
            'part2_io': {'inputs': boundary_contract['part2_inputs']},
            **{key: boundary_contract[key] for key in ('part2_to_part1', 'passthrough_inputs', 'transfer_contract')}}
    options = dict(serial_stage1=stage1, transfer=transfer, stage2=stage2, stage1_async=adapter,
        prepare_async_feed=(lambda feed: {'images': feed['images'] + np.uint8(1)}) if fault == 'runtime_input' else (lambda feed: feed),
        clone_payload=None, runner_source=str(TEMPLATE), boundary_runtime=live, canonical_full_feeds=feeds)
    if fault in {'baseline', 'runtime_input'}:
        with pytest.raises((RuntimeError, ValueError), match='Mixed functional control failed|prepared SDK input'):
            bound.run_bound_mixed_control(config, output, 1, **options)
        if fault == 'baseline':
            failed = json.loads((output / 'mixed_input_control/repetition_1.json').read_text())
            assert 'passthrough bytes differ' in failed['error'] and failed['status'] == 'failed'
        return
    proof = bound.run_bound_mixed_control(config, output, 1, **options)
    assert len(preparation_calls) == 2
    repetition = {'mixed_input_control': proof, 'stage1_execution': config['stage1_execution'], 'handoff_payload_copy': False}
    bound.validate_bound_mixed_result(repetition, config, result_root=output)
    if fault == 'saved_bytes':
        report_path = output / proof['report']['path']
        report = json.loads(report_path.read_text())
        for phase in ('serial', 'mixed'):
            item = report['observations'][1]['boundary_observation'][phase]
            item['passthrough_input_content']['images']['tensor_sha256'] = entries[0]['prepared_inputs']['images']['tensor_sha256']
        report_path.write_text(json.dumps(report))
        proof['report']['sha256'] = descriptor(report_path)['sha256']
        with pytest.raises(ValueError, match='passthrough bytes differ'):
            bound.validate_bound_mixed_result(repetition, config, result_root=output)
