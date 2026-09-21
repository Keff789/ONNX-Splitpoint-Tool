"""Original-buffer/generated-session regression; hardware is explicitly simulated."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import onnx
import pytest

from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime, assert_generated_hailo_layout_current
from onnx_splitpoint_tool.runners.backends.hailo_backend import _adapt_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import hailo10_yolo26_boundary_probe_v282 as probe
import hailo10_yolo26_generic_capture_v282 as capture
from hailo_boundary_diagnostics_v27931 import sha256

FIXTURE = ROOT / 'tests/fixtures/v281_hailo10_nwc'
SAMPLES = json.loads((FIXTURE / 'manifest.json').read_text())['samples']


@pytest.fixture
def generated(tmp_path):
    path = Path(write_runner_skeleton_onnxruntime(str(tmp_path / 'case')))
    spec = importlib.util.spec_from_file_location('generated_v282_test', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _infer_generated(module, kind, sample, raw):
    cls = getattr(module, kind); session = object.__new__(cls)
    session._hef_output_names = [sample['physical_name']]
    session._output_name_hef_to_canonical = {sample['physical_name']: sample['canonical_name']}
    session.output_shapes = {sample['canonical_name']: (1, 84, 8400)}
    session.quantized_outputs = True; session.copy_outputs = True
    session._input_names = []
    if kind == 'HailoSession':
        session._pipe = SimpleNamespace(infer=lambda inputs: {sample['physical_name']: raw})
        session._network_group = object(); session.persistent_activation = True; session._active_handle = object()
    else:
        def execute(bindings, callback):
            callback(SimpleNamespace(exception=None))
            return SimpleNamespace(wait=lambda ms: None)
        session._configured_model = SimpleNamespace(run_async=execute)
        session.timeout_ms = 50; session.hotloop = True
        session._prepare_infer_inputs = lambda inputs: inputs
        session._hotloop_binding_for_inputs = lambda inputs: object()
        session._binding_output = lambda binding, name: SimpleNamespace(get_buffer=lambda: raw)
    return session.infer({})[sample['canonical_name']]


@pytest.mark.parametrize('sample', SAMPLES, ids=lambda sample: sample['key'])
@pytest.mark.parametrize('kind', ['HailoSession', 'HailoInferModelSession'])
def test_original_buffers_through_actual_generated_infer(generated, sample, kind):
    with np.load(FIXTURE / 'physical_outputs.npz', allow_pickle=False) as archive:
        raw = archive[sample['key']]
    before = raw.tobytes()
    for physical in (raw, raw[0]):
        actual = _infer_generated(generated, kind, sample, physical)
        np.testing.assert_array_equal(actual, raw.transpose(0, 2, 1))
        np.testing.assert_array_equal(actual, _adapt_tensor(physical, (1, 84, 8400)))
        assert actual.dtype == raw.dtype and raw.tobytes() == before
        assert not np.array_equal(actual, raw.reshape(1, 84, 8400))


def test_generated_adapter_regression_is_caught_after_export(generated):
    path = Path(generated.__file__)
    assert_generated_hailo_layout_current(path)
    source = path.read_text().replace('return np.transpose(arr, (0, 2, 1))', 'return arr.reshape(tgt)', 1)
    # Do not rely solely on a revision label; changed code is rejected.
    path.write_text(source)
    with pytest.raises(RuntimeError, match='stale'):
        assert_generated_hailo_layout_current(path)


def test_source_closure_and_actual_import_origin_are_verified(generated, tmp_path):
    path = Path(generated.__file__)
    report = generated.generic_runtime_source_binding()
    assert report['generated_runner'] == str(path.resolve())
    assert report['hailo_session']['file'] == str(path.resolve())
    assert report['generic_feed']['file'] == str(path.resolve())
    request = {'source_closure': {'generated_runner': {'path': str(path.relative_to(tmp_path)), 'sha256': sha256(path)}}}
    capture.verify_source_closure(request, tmp_path)
    path.write_text(path.read_text() + '\n# stale transfer\n')
    with pytest.raises(ValueError, match='remote_source_mismatch'):
        capture.verify_source_closure(request, tmp_path)


@pytest.mark.parametrize('model,case', [('yolo26m', 'b398'), ('yolo26s', 'b364')])
def test_actual_failed_generic_row_binds_exact_receipt_without_native_result(model, case):
    root = ROOT / 'tests/fixtures/v282_h10_generic'
    path = root / f'models/{model}/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json'
    row = next(row for row in json.loads(path.read_text()) if row.get('case_id') == case)
    assert not (root / 'native_producers').exists()
    bound = probe.recorded_generic_binding_request(root, model, row)
    assert bound['case'] == case and len(bound['binding_sha256']) == 64
    assert bound['path'] == str(Path(row['deployment_contract']['hailo_io_contracts']['part1']['artifact']).parent / 'native_split_quality_binding.json')
    assert bound['engine_sha256'] == row['native_tensorrt']['sessions']['part2:tensorrt']['engine_sha256']
    endpoint = row['task_quality_input_export_diagnostics_by_variant']['composed']['runtime_endpoint']
    assert endpoint['status'] == 'failed'
    assert 'score_column_not_probability_like' in endpoint['reason']
    row['native_tensorrt']['sessions']['part2:tensorrt']['engine'] += '.wrong'
    with pytest.raises(ValueError, match='anchor_ambiguous_or_missing'):
        probe.recorded_generic_binding_request(root, model, row)


def test_external_onnx_payload_is_required_and_bound(tmp_path):
    tensor = onnx.numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), 'weight')
    graph = onnx.helper.make_graph([onnx.helper.make_node('Add', ['input', 'weight'], ['output'])], 'real_add',
        [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [3, 4])],
        [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [3, 4])], [tensor])
    model = onnx.helper.make_model(graph)
    path = tmp_path / 'real.onnx'
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location='weight.bin', size_threshold=0)
    closure = probe.external_closure(path)
    assert closure[0]['location'] == 'weight.bin'
    assert closure[0]['sha256'] == sha256(tmp_path / 'weight.bin')
    (tmp_path / 'weight.bin').unlink()
    with pytest.raises(ValueError, match='external_data_not_available'):
        probe.external_closure(path)


@pytest.mark.parametrize('mutation', ['nested_proof', 'invalid_binding', 'conflicting_binding'])
def test_resolver_distinguishes_embedded_artifact_proof_from_binding(tmp_path, monkeypatch, mutation):
    """Original R2 binding reaches its real portable validator before file IO."""
    import copy
    from test_v27931_hailo26_boundary_diagnostics import minimal_case
    from onnx_splitpoint_tool.native_split_quality import validate_native_split_quality_binding
    run, _, row = minimal_case(tmp_path)
    binding = json.loads((ROOT / 'tests/fixtures/v283_h10_binding.json').read_text())
    session = row['native_tensorrt']['sessions']['part2:tensorrt']
    session.update(engine_sha256=binding['artifacts']['engine']['sha256'],
                   source_model_sha256=binding['artifacts']['build_part2_onnx']['sha256'],
                   precision=binding['preselection']['precision'])
    assert validate_native_split_quality_binding(binding, verification_mode='portable')[0] is not None
    row['binding'] = binding
    if mutation == 'invalid_binding':
        binding['binding_sha256'] = '0' * 64
    elif mutation == 'conflicting_binding':
        other = copy.deepcopy(binding)
        other['binding_sha256'] = '0' * 64
        row['other_binding'] = other
    path = run / 'models/yolo26m/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json'
    path.write_text(json.dumps([row]))
    # Artifact lookup follows candidate selection and the real binding gate.
    # No real model, SSH or inference is needed for this resolver regression.
    class ReachedArtifactLookup(Exception):
        pass
    def lookup(*args, **kwargs):
        raise ReachedArtifactLookup()
    monkeypatch.setattr(probe.retained, 'exact_file', lookup)
    if mutation == 'nested_proof':
        with pytest.raises(ReachedArtifactLookup):
            probe.resolve_case(run, 'yolo26m')
    else:
        reason = 'binding_invalid' if mutation == 'invalid_binding' else 'binding_ambiguous'
        with pytest.raises(ValueError, match=reason):
            probe.resolve_case(run, 'yolo26m')


def test_generic_feed_keeps_existing_bridge_order_without_double_transpose(generated):
    # The real Generic wrapper calls this exported helper. This is a layout
    # unit test, not a claim that a true YOLO26 endpoint executed successfully.
    for sample in SAMPLES:
        with np.load(FIXTURE / 'physical_outputs.npz', allow_pickle=False) as archive:
            raw = archive[sample['key']]
        canonical = _infer_generated(generated, 'HailoInferModelSession', sample, raw)
        layout = {'applied': True, 'mode': 'memory_nwc_to_ncw',
                  'runtime_shape': [1, 8400, 84], 'source_shape': [1, 84, 8400],
                  'source_layout': 'NWC', 'target_layout': 'NCW', 'perm': [0, 2, 1]}
        # Obtain the production bridge declaration instead of guessing keys.
        from scripts import native_trt_from_benchmarkset as builder
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p2 = Path(tmp) / 'p2.onnx'
            graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', [sample['canonical_name']], ['out'])], 'unit_identity_only',
                [onnx.helper.make_tensor_value_info(sample['canonical_name'], onnx.TensorProto.FLOAT, [1, 84, 8400])],
                [onnx.helper.make_tensor_value_info('out', onnx.TensorProto.FLOAT, [1, 84, 8400])])
            onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 13)]), p2)
            _, metadata = builder._make_uint8_dequant_bridge_onnx(p2, Path(tmp) / 'bridge', root=Path(tmp), case=sample['case'],
                scale=sample['quantization']['scale'], zero_point=sample['quantization']['zero_point'], boundary_layout='memory_nwc_to_ncw')
        args = SimpleNamespace(_native_split_quality_binding={}, _native_split_boundary_layout=metadata['boundary_layout'],
            _native_split_boundary_input_name=sample['canonical_name'], _native_split_boundary_transform='uint8_dequant_then_layout',
            _native_split_boundary_dequant_scale=sample['quantization']['scale'], _native_split_boundary_dequant_zero_point=sample['quantization']['zero_point'])
        result = generated.prepare_generic_native_boundary_input(sample['canonical_name'], canonical, (1, 84, 8400), np.uint8,
            args=args, _boundary_mode='raw_uint8_hailo', _record_boundary_event=lambda value: None,
            _cached_boundary_buffer=lambda name, shape, dtype: np.empty(shape, dtype=dtype))
        np.testing.assert_array_equal(result, raw.reshape(1, 84, 8400))


def test_complete_generated_capture_and_real_nonidentity_onnx_reference(tmp_path, monkeypatch):
    """Full diagnostic flow with device handles simulated; real ONNX Part2.

    This deliberately is not a YOLO26 hardware endpoint or release test.
    """
    import ctypes
    import shutil
    from test_v27931_hailo26_boundary_diagnostics import minimal_case, synthetic_reference_graphs
    import hailo10_yolo26_boundary_probe_v27931 as retained
    import hailo_boundary_diagnostics_v27931 as diag
    from scripts import native_trt_from_benchmarkset as builder
    from onnx.reference import ReferenceEvaluator
    run, files, row = minimal_case(tmp_path)
    real, bridge, constant = synthetic_reference_graphs(tmp_path / 'graphs')
    shutil.copyfile(real['part1'], files['part1']); shutil.copyfile(real['part2'], files['part2']); shutil.copyfile(real['image'], files['image'])
    io = row['deployment_contract']['hailo_io_contracts']['part1']
    io.update(artifact_sha256=sha256(files['hef']), source_onnx_sha256=sha256(files['part1']), runtime_api='infer_model', hef_inputs=['images'],
        onnx_output_shapes={'cut': [1, 6, 2]}, runtime_input_format='uint8',
        runtime_input_quantization={'images': {'scale': 1/255, 'zero_point': 0.}})
    row['benchmark_input_policy'] = {'image_scale': 'norm'}
    row['run_cfg'].update(boundary_mode='raw_uint8_hailo', native_trt_precision='uint8_dequant_fp16')
    diag.write_json(run / 'models/yolo26m/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json', [row])
    shutil.rmtree(run / 'native_producers')
    request, resolved = probe.resolve_case(run, 'yolo26m')
    assert request['artifact_binding_source'] == 'current_generic_row_and_exact_receipts'
    _, metadata = builder._make_uint8_dequant_bridge_onnx(files['part2'], tmp_path / 'meta_bridge', root=tmp_path,
        case='b398', scale=.1, zero_point=0., boundary_layout='memory_nwc_to_ncw')
    request['generic_quality_binding'] = {'native_trt_meta': {'uint8_cast_bridge': metadata},
        'preselection': {'boundary_tensor_name': 'cut', 'boundary_layout': 'memory_nwc_to_ncw',
                        'boundary_transform': 'uint8_dequant_then_layout', 'dequant_scale': .1, 'dequant_zero_point': 0.}}
    original = {'bridge': bridge, 'engine': tmp_path / 'bound.engine', 'receipt': tmp_path / 'bound_receipt.json'}
    original['engine'].write_bytes(b'SIMULATED_ENGINE_HANDLE'); original['receipt'].write_text('{"simulated":true}')
    request['remote_artifacts'] = {k: {'path': str(p), 'sha256': sha256(p)} for k, p in original.items()}
    session = request['engine_io']
    session['precision'] = 'uint8_dequant_fp16'
    session['inputs'][0]['type'] = 'tensor(uint8)'; session['outputs'][0]['type'] = 'tensor(float)'
    stage = probe.prepare_stage(tmp_path / 'stage', ROOT, [(request, resolved)])
    raw = np.round(constant.transpose(0, 2, 1) / .1).astype(np.uint8)
    actual_loader = capture.load_generated_runner
    calls = []
    def simulated_hardware_loader(path):
        module = actual_loader(path)
        def hailo_init(self, *args, **kwargs):
            self._hef_output_names = ['actual_hef_name']; self._input_names = ['images']
            self._output_name_hef_to_canonical = {'actual_hef_name': 'cut'}
            self.output_shapes = {'cut': (1, 6, 2)}; self.runtime_input_shapes = {'images': (2, 2, 3)}
            self.input_shapes = {'images': (1, 3, 2, 2)}
            self.quantized_inputs = True; self.quantized_outputs = True; self.copy_outputs = True
            self.hotloop = True; self.timeout_ms = 50
            self.quantize_input = lambda name, source: np.round(source * 255).astype(np.uint8)
            self.describe_io = lambda: io
            def execute(bindings, callback):
                calls.append('hailo_infer'); callback(SimpleNamespace(exception=None))
                return SimpleNamespace(wait=lambda timeout: None)
            self._configured_model = SimpleNamespace(run_async=execute)
            self._prepare_infer_inputs = lambda inputs: inputs
            def prepare(inputs):
                self._hot_input_buffers = inputs; self._hot_binding = object(); return self._hot_binding
            self._hotloop_binding_for_inputs = prepare
            self._binding_output = lambda binding, name: SimpleNamespace(get_buffer=lambda: raw)
        def trt_init(self, *args, **kwargs):
            assert kwargs['allow_build'] is False
            self._input_names = ['cut']; self._output_names = ['output0']
            self.input_info = [module._NativeTRTTensorInfo('cut', (1, 6, 2), 'tensor(uint8)')]
            self.output_info = [module._NativeTRTTensorInfo('output0', (1, 2, 6), 'tensor(float)')]
            self._tensor_shapes = {'cut': (1, 6, 2), 'output0': (1, 2, 6)}
            self._tensor_dtypes = {'cut': np.dtype('uint8'), 'output0': np.dtype('float32')}
            self._test_device = {'cut': np.empty((1, 6, 2), np.uint8), 'output0': np.empty((1, 2, 6), np.float32)}
            self._host_out = {'output0': np.empty((1, 2, 6), np.float32)}
            self._dev_ptrs = {k: v.ctypes.data for k, v in self._test_device.items()}; self.stream = 0
            def memcpy(dst, src, size, kind, stream): ctypes.memmove(dst, src, size); return 0
            self.cudart = SimpleNamespace(cudaMemcpyAsync=memcpy, cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2), cudaStreamSynchronize=lambda stream: 0)
            def execute(stream):
                calls.append('trt_infer')
                output = ReferenceEvaluator(str(bridge)).run(None, {'cut': self._test_device['cut']})[0]
                output[..., 4] = 622.12  # Controlled runtime corruption stays raw.
                np.copyto(self._test_device['output0'], output); return True
            self.context = SimpleNamespace(execute_async_v3=execute)
        monkeypatch.setattr(module.HailoInferModelSession, '__init__', hailo_init)
        monkeypatch.setattr(module.HailoInferModelSession, 'close', lambda self: calls.append('hailo_cleanup'))
        monkeypatch.setattr(module.NativeTRTSession, '__init__', trt_init)
        monkeypatch.setattr(module.NativeTRTSession, 'close', lambda self: calls.append('trt_cleanup'))
        monkeypatch.setattr(module, '_verify_explicit_native_trt_engine_receipt', lambda **kwargs: {'simulated_receipt': True})
        return module
    monkeypatch.setattr(capture, 'load_generated_runner', simulated_hardware_loader)
    monkeypatch.setenv('ONNX_SPLITPOINT_ARTIFACT_POLICY', 'cache_verify_only')
    output = tmp_path / 'results'
    capture.capture(request, stage / 'yolo26m', output)
    retained.offline_reference(request, resolved, output)
    report, tensors = diag.load_packet(output)
    assert calls == ['hailo_infer', 'trt_infer', 'trt_cleanup', 'hailo_cleanup']
    assert report['capture_pass'] is True and report['endpoint_pass'] is False
    assert report['source_path'] == 'current_generated_generic'
    assert report['source_closure_verified'] is True
    assert report['compiler_invoked'] is False and report['energy_invoked'] is False
    np.testing.assert_array_equal(tensors['C_000'], raw.reshape(1, 6, 2))
    np.testing.assert_allclose(tensors['D_000'], constant)
    assert np.all(tensors['E_000'][..., 4] == np.float32(622.12))
    assert report['part2_same_feed_comparison']['status'] == 'evaluated'
    assert report['regular_path_released'] is False
