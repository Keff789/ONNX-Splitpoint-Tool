"""Revision 3: real generated Generic code with deterministic offline workers."""
import ast
import copy
import hashlib
import os
import queue
import json
import sys
import threading
import time
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'


def namespace():
    tree = ast.parse(TEMPLATE.read_text())
    names = {'_bench_streaming_pipeline', '_clone_tensor_map', '_percentile',
             'DeepXSession', '_resolve_deepx_stage2_mapping',
             '_completion_validate_prepared_input_contract',
             '_completion_hailo_inputs', '_completion_hailo_quantized_inputs'}
    nodes = [n for n in tree.body if getattr(n, 'name', '') in names]
    env = dict(vars(typing), np=np, queue=queue, threading=threading, time=time,
               os=os, Path=Path, _try_adapt_tensor=lambda a, shape: (tuple(a.shape) == tuple(shape or a.shape), a))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])), str(TEMPLATE), 'exec'), env)
    return env


@pytest.mark.parametrize('impl', ['generic', 'native_fifo'])
@pytest.mark.parametrize('pressure', [False, True])
def test_streaming_lossless_and_fully_drained_warmup(impl, pressure):
    env = namespace(); produced = []; consumed = []
    def p1():
        idx = len(produced)
        if idx == 3:
            assert consumed == [0, 1, 2], 'warmup must finish before measured P1'
        produced.append(idx)
        if not pressure:
            time.sleep(.001)
        return {'first': np.array([idx]), 'second': np.array([1000 + idx])}
    def p2(inputs):
        if pressure:
            time.sleep(.003)
        idx = int(inputs['first'][0]); assert int(inputs['second'][0]) == 1000 + idx
        consumed.append(idx)
    result = env['_bench_streaming_pipeline']('rev3', p1, lambda x: x, p2,
        warmup_frames=3, measured_frames=24, queue_depth=1, streaming_impl=impl)
    assert produced == consumed == list(range(27))
    assert result['completed_frames'] == result['requested_frames'] == result['enqueued_frames'] == 24
    assert result['warmup_completed_frames'] == result['warmup_enqueued_frames'] == 3
    assert result['fps_makespan'] == pytest.approx(24 * 1000 / result['makespan_ms'])
    assert result['measurement_scope'] == 'raw_backend_outputs'
    assert result['postprocess_included'] is False
    assert not any(t.name.startswith('rev3_') for t in threading.enumerate())


@pytest.mark.parametrize('impl', ['generic', 'native_fifo'])
@pytest.mark.parametrize('failure', ['producer', 'consumer'])
def test_streaming_error_terminates_peers_without_success(impl, failure):
    env = namespace(); count = 0
    def p1():
        nonlocal count
        count += 1
        if count == 8 and failure == 'producer':
            raise ValueError('producer fixture failure')
        return {'value': np.array([count])}
    def p2(value):
        if failure == 'consumer' and int(value['value'][0]) == 5:
            raise ValueError('consumer fixture failure')
        time.sleep(.002)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match=f'{failure} fixture failure'):
        env['_bench_streaming_pipeline']('rev3_error', p1, lambda x: x, p2,
            warmup_frames=2, measured_frames=24, queue_depth=1, streaming_impl=impl)
    assert time.monotonic() - start < 3
    assert not any(t.name.startswith('rev3_error_') for t in threading.enumerate())


def session():
    env = namespace(); session = object.__new__(env['DeepXSession'])
    session._input_names = ['input']; session._output_names = ['output']
    session.input_order_override = None; session.output_shapes = {}
    session.runtime_input_shapes = {'input': (1, 2, 4, 3)}
    session.input_shapes = {'input': (1, 3, 2, 4)}
    session._should_use_feature_subprocess = lambda: False
    session.dxnn_path = Path('fixture.dxnn')
    seen = []
    session._engine = SimpleNamespace(run=lambda x: seen.append(x[0].copy()) or [x[0]])
    return session, seen


@pytest.mark.parametrize('domain,value,expected', [('raw', 128., 128), ('norm', .5, 127), ('imagenet', 0., None), ('clip', 0., None)])
@pytest.mark.parametrize('layout', ['NCHW', 'NHWC', 'HWC'])
def test_deepx_explicit_domains_and_layouts(domain, value, expected, layout):
    s, seen = session(); shapes = {'NCHW': (1, 3, 2, 4), 'NHWC': (1, 2, 4, 3), 'HWC': (2, 4, 3)}
    s.image_input_contract = {'source_domain': domain, 'source_layout': layout, 'runtime_layout': 'HWC'}
    prepared = s.prepare_image_inputs({'input': np.full(shapes[layout], value, np.float32)})
    assert prepared['input'].shape == (2, 4, 3) and prepared['input'].dtype == np.uint8
    if expected is not None:
        assert np.all(prepared['input'] == expected)
    s.bind_prepared_inputs(prepared); s.infer(prepared)
    assert np.array_equal(seen[0], prepared['input'])


@pytest.mark.parametrize('bad', ['missing', 'domain', 'layout', 'range', 'dtype'])
def test_deepx_contract_errors_precede_engine_call(bad):
    s, seen = session(); arr = np.full((1, 3, 2, 4), 128, np.float32)
    if bad != 'missing':
        s.image_input_contract = {'source_domain': 'unknown' if bad == 'domain' else 'norm' if bad == 'range' else 'raw',
            'source_layout': 'NHWC' if bad == 'layout' else 'NCHW', 'runtime_layout': 'HWC'}
    if bad == 'dtype': arr = arr.astype(complex)
    with pytest.raises((RuntimeError, ValueError)):
        s.infer({'input': arr})
    assert seen == []


def test_deepx_exact_prepared_contract_and_distinct_boundary_values():
    s, seen = session(); first = {'input': np.ones((1, 16, 2, 4), np.float32)}
    s.bind_prepared_inputs(first); s.infer(first)
    s.infer({'input': first['input'] * 9})
    assert np.all(seen[0] == 1) and np.all(seen[1] == 9)
    with pytest.raises(RuntimeError, match='shape/dtype'):
        s.bind_prepared_inputs({'input': first['input'].astype(np.float64)})
    assert len(seen) == 2


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'missing', 'additional'])
def test_deepx_direct_exact_infer_rejects_changed_signature_before_sdk(bad):
    s, seen = session()
    prepared = {'input': np.ones((1, 16, 2, 4), np.float32)}
    s.bind_prepared_inputs(prepared)
    s.infer(prepared)
    invalid = dict(prepared)
    if bad == 'dtype':
        invalid['input'] = prepared['input'].astype(np.float64)
    elif bad == 'shape':
        invalid['input'] = prepared['input'].reshape(1, 16, 4, 2)
    elif bad == 'missing':
        invalid = {}
    else:
        invalid['other'] = prepared['input']
    with pytest.raises(RuntimeError, match='exact prepared input.*mismatch'):
        s.infer(invalid)
    assert len(seen) == 1


def test_deepx_exact_infer_requires_signature_binding_before_sdk():
    s, seen = session()
    s.exact_prepared_input = True
    with pytest.raises(RuntimeError, match='requires a bound shape/dtype signature'):
        s.infer({'input': np.ones((1, 16, 2, 4), np.float32)})
    assert seen == []


def prepared_config():
    contract = {'shape': [2, 4, 3], 'dtype': 'uint8', 'layout': 'HWC',
                'normalization': 'embedded_dxcom_preprocessing', 'task': 'detection',
                'preprocess_mode_effective': 'letterbox',
                'source_image': 'bound.png', 'source_image_sha256': 'a' * 64}
    return {'task': 'detection', 'input_binding': {
        'prepared_input': {'input_name': 'images', 'shape': [2, 4, 3], 'dtype': 'uint8'},
        'prepared_input_contract': contract, 'source_image': 'bound.png',
        'source_image_sha256': 'a' * 64, 'runtime_input_format': 'uint8'}}


@pytest.mark.parametrize('backend', ['deepx_m1', 'hailo8', 'hailo10h'])
def test_exact_dispatch_accepts_declared_prepared_contract(backend):
    check = namespace()['_completion_validate_prepared_input_contract']
    config = prepared_config()
    if backend == 'hailo10h':
        config['input_binding']['prepared_input_contract']['normalization'] = 'hef_quant_info_from_unit_float32'
    check(config, backend)
    config['input_binding']['prepared_input_contract']['source_input_contract'] = {
        'layout': 'HWC', 'normalization': config['input_binding']['prepared_input_contract']['normalization']}
    config['input_binding']['prepared_input_contract'].pop('layout')
    check(config, backend)


@pytest.mark.parametrize('bad', ['missing_layout', 'layout_shape', 'layout_conflict',
    'missing_domain', 'domain_conflict', 'dtype', 'shape', 'backend_encoding',
    'missing_preprocessing', 'task', 'missing_image', 'image_conflict', 'missing_name',
    'source_shape', 'source_dtype', 'source_preprocessing'])
def test_exact_dispatch_rejects_missing_or_conflicting_contract(bad):
    config = copy.deepcopy(prepared_config())
    binding = config['input_binding']; contract = binding['prepared_input_contract']
    if bad == 'missing_layout': contract.pop('layout')
    if bad == 'layout_shape': contract['layout'] = 'NCHW'
    if bad == 'layout_conflict': contract['source_input_contract'] = {'layout': 'CHW'}
    if bad == 'missing_domain': contract.pop('normalization')
    if bad == 'domain_conflict': contract['source_input_contract'] = {'normalization': 'imagenet'}
    if bad == 'dtype': contract['dtype'] = 'float32'
    if bad == 'shape': contract['shape'] = [4, 2, 3]
    if bad == 'backend_encoding': binding['runtime_input_format'] = 'float32'
    if bad == 'missing_preprocessing': contract.pop('preprocess_mode_effective')
    if bad == 'task': contract['task'] = 'classification'
    if bad == 'missing_image': binding.pop('source_image_sha256')
    if bad == 'image_conflict': contract['source_image'] = 'different.png'
    if bad == 'missing_name': binding['prepared_input'].pop('input_name')
    if bad == 'source_shape': contract['source_input_contract'] = {'shape': [9, 9, 3]}
    if bad == 'source_dtype': contract['source_input_contract'] = {'dtype': 'float32'}
    if bad == 'source_preprocessing': contract['source_input_contract'] = {'preprocess_mode_effective': 'resize'}
    with pytest.raises(ValueError, match='Generic completion prepared input'):
        namespace()['_completion_validate_prepared_input_contract'](config, 'deepx_m1')


@pytest.mark.parametrize('task,dtype,normalization', [
    ('classification', 'uint8', 'hef_quant_info_from_imagenet_float32'),
    ('detection', 'uint8', 'hef_quant_info_from_unit_float32'),
    ('classification', 'float32', 'imagenet_float32'),
    ('detection', 'float32', 'divide_255_float32')])
def test_hailo10_declared_encoding_matches_existing_loader(task, dtype, normalization):
    config = prepared_config(); config['task'] = task
    binding = config['input_binding']; contract = binding['prepared_input_contract']
    binding['runtime_input_format'] = binding['prepared_input']['dtype'] = dtype
    contract.update(task=task, dtype=dtype, normalization=normalization)
    check = namespace()['_completion_validate_prepared_input_contract']
    check(config, 'hailo10h')
    contract['normalization'] = ('divide_255_float32' if dtype == 'uint8'
                               else 'hef_quant_info_from_unit_float32')
    with pytest.raises(ValueError, match='normalization/backend encoding conflict'):
        check(config, 'hailo10h')


def test_current_dispatch_descriptor_preserves_historical_native_contract():
    config = prepared_config(); binding = config['input_binding']
    historical = binding['prepared_input_contract']
    historical.pop('normalization')
    before = copy.deepcopy(historical)
    binding['prepared_input'].update(layout='HWC', normalization='uint8_0_255',
                                     source_domain='uint8_0_255')
    check = namespace()['_completion_validate_prepared_input_contract']
    check(config, 'hailo8')
    assert historical == before
    for key, conflicting in [('layout', 'CHW'), ('dtype', 'float32'), ('shape', [9, 9, 3])]:
        invalid = copy.deepcopy(config)
        invalid['input_binding']['prepared_input'][key] = conflicting
        with pytest.raises(ValueError, match='Generic completion prepared input'):
            check(invalid, 'hailo8')
    historical['normalization'] = 'different_declared_domain'
    with pytest.raises(ValueError, match='normalization contract conflict'):
        check(config, 'hailo8')


@pytest.mark.parametrize('mutation', ['none', 'layout', 'normalization', 'source_domain'])
def test_normal_reporter_checks_current_descriptor_against_original_h8_binding(tmp_path, mutation):
    from onnx_splitpoint_tool.generic_completion_plan import _input_binding
    from onnx_splitpoint_tool.generic_completion_reporting import _verified_pair_prepared_input
    benchmark = tmp_path / 'native_producers/hailo8/fixture/benchmark_set'
    benchmark.mkdir(parents=True)
    source = benchmark / 'native_fifo_results.json'; source.write_text('{}')
    array = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    path = benchmark / 'prepared.bin'; path.write_bytes(array.tobytes())
    command = {'backend': 'hailo8_to_trt', 'model': 'fixture', 'benchmark_set': '/remote/benchmark_set',
        'input_image': '/remote/image.png', 'input_image_sha256': 'a' * 64,
        'runtime_options': {'hailo_format': 'uint8'},
        'prepared_input_contract': {'format': 'raw_runtime_tensor', 'name': 'images',
            'shape': list(array.shape), 'dtype': 'uint8', 'layout': 'HWC',
            'source_image_sha256': 'a' * 64},
        'artifacts': {'prepared_input': {'path': '/remote/benchmark_set/prepared.bin',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    before = copy.deepcopy(command)
    binding = _input_binding(command, {'source_root': str(benchmark.parent.parent),
        'model': 'fixture', 'task': 'detection', 'report': str(source)}, tmp_path)
    descriptor = {**binding['entries'][0], 'path': '/remote/benchmark_set/prepared.bin',
        'format': 'raw', 'layout': 'HWC', 'normalization': 'uint8_0_255', 'source_domain': 'uint8_0_255'}
    if mutation != 'none': descriptor[mutation] = 'CHW' if mutation == 'layout' else 'unit_float32'
    case = {'input_binding': binding, 'runtime_config': {'task': 'detection',
        'input_binding': {**binding, 'prepared_input': descriptor}}}
    if mutation == 'none':
        assert _verified_pair_prepared_input(command, case, source, tmp_path) == binding['entries'][0]
    else:
        with pytest.raises(ValueError, match='native_generic_prepared_input_descriptor_conflict'):
            _verified_pair_prepared_input(command, case, source, tmp_path)
    assert command == before


@pytest.mark.parametrize('layout', ['HWC', 'CHW', 'NHWC', 'NCHW'])
def test_h8_classification_exact_adapter_uses_declared_imagenet_domain(layout):
    env = namespace(); config = prepared_config(); config['task'] = 'classification'
    binding = config['input_binding']; contract = binding['prepared_input_contract']
    contract.update(task='classification', normalization='imagenet_mean_std', layout=layout)
    binding['runtime_input_format'] = 'float32'
    rgb = np.array([[[0, 1, 255], [128, 41, 9]], [[5, 199, 81], [240, 47, 111]]], dtype=np.uint8)
    array = rgb if layout == 'HWC' else rgb.transpose(2, 0, 1) if layout == 'CHW' else rgb[None] if layout == 'NHWC' else rgb.transpose(2, 0, 1)[None]
    binding['prepared_input']['shape'] = contract['shape'] = list(array.shape)
    env['_completion_validate_prepared_input_contract'](config, 'hailo8')
    observed = env['_completion_hailo_inputs'](config, 'hailo8', {'images': array})['images']
    expected = (rgb.astype(np.float32) / np.float32(255.) - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
    if layout == 'CHW': expected = expected.transpose(2, 0, 1)
    if layout == 'NHWC': expected = expected[None]
    if layout == 'NCHW': expected = expected.transpose(2, 0, 1)[None]
    assert observed.dtype == np.float32 and observed.tobytes() == expected.tobytes()
    assert array.tobytes() == (rgb if layout == 'HWC' else rgb.transpose(2, 0, 1) if layout == 'CHW' else rgb[None] if layout == 'NHWC' else rgb.transpose(2, 0, 1)[None]).tobytes()
    contract['normalization'] = 'unit_float32'
    with pytest.raises(ValueError, match='ImageNet numeric contract'):
        env['_completion_validate_prepared_input_contract'](config, 'hailo8')
    with pytest.raises(ValueError, match='ImageNet layout/domain'):
        env['_completion_hailo_inputs'](config, 'hailo8', {'images': array})


def test_h8_detection_exact_adapter_remains_unit_float_or_raw_uint8():
    config = prepared_config(); array = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    env = namespace(); inputs = {'images': array}
    assert env['_completion_hailo_inputs'](config, 'hailo8', inputs) is inputs
    config['input_binding']['runtime_input_format'] = 'float32'
    observed = env['_completion_hailo_inputs'](config, 'hailo8', inputs)['images']
    assert observed.tobytes() == (array.astype(np.float32) / np.float32(255.)).tobytes()


@pytest.mark.parametrize('bad', ['layout', 'domain', 'model_name'])
def test_real_generated_main_rejects_unbound_input_before_sdk(tmp_path, monkeypatch, bad):
    from test_v2912_generic_split_completion import _fixture, _module
    module = _module(monkeypatch)
    config, _ = _fixture(tmp_path)
    binding = config['input_binding']
    if bad == 'model_name':
        binding['prepared_input']['input_name'] = 'another_model_input'
    else:
        binding['prepared_input_contract'].pop('layout' if bad == 'layout' else 'normalization')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('SDK must not initialize for an invalid input contract')
    monkeypatch.setattr(module, 'DeepXSession', forbidden)
    monkeypatch.setattr(module, 'NativeTRTSession', forbidden)
    monkeypatch.setattr(module, '_create_session', forbidden)
    path = tmp_path / 'invalid-config.json'; path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['runner', '--generic-completion-config', str(path),
        '--out-dir', str(tmp_path / 'output'), '--generic-completion-repetition', '1'])
    if bad == 'model_name':
        with pytest.raises(RuntimeError, match='input name differs from model contract'):
            module.main()
    else:
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2
    assert calls == []


def test_deepx_explicit_multiinput_and_ambiguity():
    f = namespace()['_resolve_deepx_stage2_mapping']
    outputs = {'left': np.array([1.]), 'right': np.array([2.])}
    assert f(['a', 'b'], outputs, {'a': 'right', 'b': 'left'}, {'a': (1,), 'b': (1,)}, {}) == {'a': 'right', 'b': 'left'}
    with pytest.raises(ValueError, match='ambiguous'):
        f(['a', 'b'], outputs, {}, {'a': (1,), 'b': (1,)}, {})
    with pytest.raises(ValueError, match='missing output'):
        f(['a'], outputs, {'a': 'absent'}, {'a': (1,)}, {})


def test_evaluation_mapping_selector_cannot_use_validation_scores():
    tree = ast.parse(TEMPLATE.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_maybe_tune_deepx_stage2_mapping')
    source = ast.unparse(function)
    assert '_ensure_ort_part2_reference_sess' not in source
    assert '"0"' in ast.get_source_segment(TEMPLATE.read_text(), function)
    assert 'selection_uses_validation_scores' in source


def test_generated_bundle_contains_exact_fast_completion_closure(tmp_path):
    from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_runner_bundle
    from onnx_splitpoint_tool.native_detection_postprocess import build_detection_completion_execution_contract
    from test_v272_yolov7_head_mapping import _heads_640, _raw_source
    import json
    import subprocess
    import sys
    destination = tmp_path / 'runner-bundle'
    prepare_runner_bundle(destination)
    repo = TEMPLATE.parents[2]
    for relative in ('native_three_stage.py', 'native_detection_postprocess.py', 'preprocessing_contract.py'):
        assert (destination / 'splitpoint_runners' / relative).read_bytes() == (repo / relative).read_bytes()
    outputs = _heads_640()
    outputs['vendor_small'][0, 0, 40, 40, :4] = 0
    outputs['vendor_small'][0, 0, 40, 40, 4:6] = 10
    contract = build_detection_completion_execution_contract(model_id='yolov7_paper', outputs=outputs,
        input_hw=[640, 640], original_wh=[640, 480],
        source_endpoint_contract=_raw_source(outputs, endpoint_hash='a' * 64))
    (destination / 'contract.json').write_text(json.dumps(contract))
    np.savez(destination / 'outputs.npz', **outputs)
    completed = subprocess.run([sys.executable, '-B', '-c',
        'import json, sys, numpy as np; '
        'from splitpoint_runners.native_detection_postprocess import create_detection_completion_runtime, detection_completion_runtime_source_check; '
        'from splitpoint_runners.native_three_stage import verify_fast_completion_attestation; '
        'from splitpoint_runners.task_completion import measure_split_completion; '
        'from splitpoint_runners.harness import yolo as original; '
        'from splitpoint_runners.runners.harness import yolo as alias; '
        'assert original is alias; '
        'outputs = dict(np.load("outputs.npz")); '
        'runtime = create_detection_completion_runtime(json.load(open("contract.json"))); '
        'assert runtime.completion_runtime_mode == "fast_oracle_outside_timing"; '
        'runtime.process(outputs); '
        'assert runtime.completed_count == 1 and len(runtime.last_detections) == 1; '
        'verify_fast_completion_attestation(runtime.attestation(completed_work_units=1), '
        'execution_contract=runtime.execution_contract, outputs=runtime.snapshot_completed_source()); '
        'assert detection_completion_runtime_source_check(runtime)["unchanged"] is True; '
        'assert "onnx_splitpoint_tool" not in sys.modules'],
        cwd=destination, env={**os.environ, 'PYTHONPATH': str(destination), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
