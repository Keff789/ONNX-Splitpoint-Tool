"""AP6 synthetic transition fixtures; no claimed Hailo10 hardware repair.

T06.6's original before/after hardware verification is conditional on a proven
fix. These tests instead prove the delivered diagnostic cannot relax the real
endpoint validator, cannot turn a collector error into a pass, and preserves
unmodified invalid output bytes through exporter/loader.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import hailo_boundary_diagnostics_v27931 as diag
import hailo10_yolo26_boundary_probe_v27931 as probe
import hailo10_yolo26_boundary_worker_v27931 as worker
import native_hailo10_trt_e2e_from_benchmarkset as native


def minimal_case(tmp_path, model='yolo26m'):
    case = probe.CASES[model]
    run = tmp_path / 'relocated_original_run'
    suite = run / f'models/{model}/benchmark_set/legacy_suite'
    cdir = suite / case
    cdir.mkdir(parents=True)
    files = {'hef': cdir / 'hailo/hailo10/part1/compiled.hef', 'image': suite / 'resources/validation/403584.jpg',
             'part1': cdir / f'{model}_part1_{case}.onnx', 'part2': cdir / f'{model}_part2_{case}.onnx'}
    for key, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('SYNTHETIC_' + key).encode())
    cache = '/home/nx/splitpoint_runs/_onnx_splitpoint_cache/exact_old_engine'
    io = {'hef_outputs': ['actual_hef_name'], 'canonical_outputs': ['cut'],
          'artifact': '/deleted/old/run/part1.hef', 'output_aliases': {'actual_hef_name': 'cut'},
          'canonical_inputs': ['images'], 'runtime_output_shapes': {'cut': [1, 2, 6]},
          'runtime_input_shapes': {'images': [2, 2, 3]}, 'onnx_input_shapes': {'images': [1, 3, 2, 2]},
          'runtime_output_format': 'uint8', 'quantized_inputs': True, 'quantized_outputs': True,
          'runtime_output_quantization': {'actual_hef_name': {'scale': .1, 'zero_point': 0.}}, 'outputs_dequantized': False}
    session = {'explicit_engine': True, 'cache_hit': True, 'inputs': [{'name': 'cut', 'shape': [1, 6, 2]}],
               'outputs': [{'name': 'output0', 'shape': [1, 2, 6]}], 'engine': cache + '/p2.engine', 'engine_sha256': 'a'*64,
               'source_model': cache + '/bridge.onnx', 'source_model_sha256': 'b'*64,
               'engine_build_receipt_path': cache + '/receipt.json', 'engine_build_receipt_file_sha256': 'c'*64}
    row = {'setup_id': 'orin_nx_hailo10_01', 'case_id': case, 'native_tensorrt': {'sessions': {'part2:tensorrt': session}},
           'deployment_contract': {'hailo_io_contracts': {'part1': io}},
           'run_cfg': {'image': '/deleted/old/run/suite/resources/validation/403584.jpg'},
           'part1': files['part1'].name, 'part2': files['part2'].name}
    diag.write_json(run / f'models/{model}/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json', [row])
    diag.write_json(run / f'native_producers/hailo10h/{model}/benchmark_set/native_pipeline/{case}/hailo10h_to_trt/uint8_dequant_fp16/hailo10_native_fifo_e2e_results.json',
                    {'replay_artifact_verification': {'hef_sha256': diag.sha256(files['hef'])}})
    return run, files, row


@pytest.mark.parametrize('model', list(probe.CASES))
def test_t06_1_deleted_remote_run_resolves_exact_original_and_stages_no_models_or_dataset(tmp_path, model):
    run, files, row = minimal_case(tmp_path, model)
    request, resolved = probe.resolve_case(run, model)
    assert resolved == files
    assert request['case'] == probe.CASES[model]
    assert request['image_sha256'] == diag.sha256(files['image'])
    stage = probe.prepare_stage(tmp_path / 'stage', ROOT, [(request, resolved)])
    assert not list(stage.rglob('*.onnx'))
    assert len(list(stage.rglob('*.jpg'))) == 1
    assert len(list(stage.rglob('*.hef'))) == 1
    assert request['compiler_invoked'] is False
    files['hef'].unlink()
    with pytest.raises(ValueError, match='part1_hef'):
        probe.resolve_case(run, model)


def test_t06_1_ambiguous_local_reference_or_changed_hef_rejected(tmp_path):
    run, files, _ = minimal_case(tmp_path)
    second = tmp_path / 'extra/b398'
    second.mkdir(parents=True)
    (second / files['part1'].name).write_bytes(b'OTHER SAME ROLE')
    with pytest.raises(ValueError, match='ambiguous.*reference_part1'):
        probe.resolve_case(run, 'yolo26m', [second.parent])
    files['hef'].write_bytes(b'CHANGED HEF')
    with pytest.raises(ValueError, match='hash_mismatch.*part1_hef'):
        probe.resolve_case(run, 'yolo26m')


def test_t06_2_diagnostic_packet_never_forges_binding_or_regular_success(tmp_path):
    packet = diag.dump_packet(tmp_path, {'E_000': np.zeros((1, 2, 6), np.float32)}, {'status': 'captured', 'claim_eligible': True})
    loaded, values = diag.load_packet(tmp_path)
    for key, value in diag.FLAGS.items():
        assert loaded[key] is value
    assert 'native_split_quality_binding' not in loaded
    assert probe.collector_exit_code({'status': 'failed', 'collection_complete': True}) == 2
    loaded['counts_as_benchmark'] = True
    diag.write_json(tmp_path / 'diagnostic.json', loaded)
    with pytest.raises(ValueError, match='eligibility_flags'):
        diag.load_packet(tmp_path)


@pytest.mark.parametrize('kind', ['name', 'layout', 'values'])
def test_t06_3_same_shaped_wrong_names_layout_or_order_never_semantic_pass(kind):
    a = np.arange(9).reshape(3, 3)
    actual = {'left': a}
    reference = {'right': a.T}
    mapping = {'left': 'wrong' if kind == 'name' else 'right'}
    result = diag.compare_named(actual, reference, mapping, actual_layout='HWC', reference_layout='CHW' if kind == 'layout' else 'HWC')
    assert result['semantic_pass'] is False
    if kind == 'values':
        assert result['tensors']['left']['max_abs'] == 4
        assert result['tensors']['left']['numerically_close'] is False
    else:
        assert result['status'] == 'rejected'


@pytest.mark.parametrize('dtype,already,count', [('uint8', False, 1), ('float32', True, 0)])
def test_t06_4_float_is_not_dequantized_twice_and_uint8_uses_declared_axis(dtype, already, count):
    a = np.arange(12, dtype=dtype).reshape(2, 2, 3)
    output, meta = diag.dequant_reference(a, {'scale': [.1, .2, .3], 'zero_point': 1., 'axis': 2}, outputs_dequantized=already)
    assert meta['dequantization_count'] == count
    if count:
        np.testing.assert_allclose(output, (a.astype(np.float32) - 1) * np.array([.1, .2, .3], np.float32))
    else:
        np.testing.assert_array_equal(output, a)


@pytest.mark.parametrize('change', ['axis_missing', 'axis_wrong', 'double', 'float_unknown', 'scale_nonfinite'])
def test_t06_4_bad_quantization_is_precise_error(change):
    a = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    q = {'scale': [.1, .2, .3], 'zero_point': 1., 'axis': 2}
    already = False
    if change == 'axis_missing':
        q.pop('axis')
    if change == 'axis_wrong':
        q['axis'] = 1
    if change == 'double':
        already = True
    if change == 'float_unknown':
        a = a.astype(np.float32)
    if change == 'scale_nonfinite':
        q['scale'] = float('nan')
    with pytest.raises(ValueError):
        diag.dequant_reference(a, q, outputs_dequantized=already)


def test_t06_5_npz_safe_nonfinite_json_and_controlled_budget(tmp_path):
    values = np.array([np.nan, np.inf, -np.inf, 3.], np.float32)
    report = diag.dump_packet(tmp_path, {'raw': values}, {'measurement': float('nan')})
    assert report['arrays']['raw']['nonfinite_count'] == 3
    loaded, arrays = diag.load_packet(tmp_path)
    assert loaded['measurement'] is None
    np.testing.assert_array_equal(arrays['raw'], values)
    assert 'NaN' not in (tmp_path / 'diagnostic.json').read_text()
    with pytest.raises(ValueError, match='object_array'):
        diag.dump_packet(tmp_path / 'object', {'bad': np.array([{}], dtype=object)}, {})
    with pytest.raises(ValueError, match='budget_exceeded'):
        diag.dump_packet(tmp_path / 'large', {'big': np.zeros(2000)}, {}, max_bytes=1024)
    assert not (tmp_path / 'large/raw_tensors.npz').exists()


def test_t06_5_payload_tampering_is_collector_error(tmp_path):
    diag.dump_packet(tmp_path, {'raw': np.zeros(4)}, {})
    (tmp_path / 'raw_tensors.npz').write_bytes(b'TRUNCATED')
    with pytest.raises(ValueError, match='hash_mismatch'):
        diag.load_packet(tmp_path)


def test_t06_5_real_supervisor_deadline_cleans_owned_process(tmp_path):
    # The production helper is a standalone subreaper: calling it inside
    # pytest lets it adopt/reap pytest's shared multiprocessing tracker.
    # Keep the real supervision path, but give it its own process tree.
    driver = '''
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from deepx_full_workflow_smoke_worker_v27930 import supervised_run
work = Path(sys.argv[2])
command = [sys.executable, '-I', '-B', '-c',
           'import time; print("WORKER_READY", flush=True); time.sleep(30)']
result = supervised_run(command, cwd=work, log_path=work / 'console.log', timeout=1., grace=.1)
(work / 'supervision.json').write_text(json.dumps(result), encoding='utf-8')
'''
    completed = subprocess.run(
        [sys.executable, '-I', '-B', '-c', driver, str(ROOT / 'scripts'), str(tmp_path)],
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / 'supervision.json').read_text(encoding='utf-8'))
    assert 'WORKER_READY' in (tmp_path / 'console.log').read_text(encoding='utf-8')
    assert result['owned_processes_observed'] >= 1
    assert result['timed_out'] is True
    assert result['cleanup_complete'] is True
    assert result['owned_survivors'] == []
    assert result['returncode'] != 0


def test_t06_2_t06_3_opt_in_capture_observes_raw_names_and_normal_sample_stays_copy_free(monkeypatch):
    raw = np.arange(12, dtype=np.uint8).reshape(1, 2, 6)
    class Session:
        _hef_output_names = ['physical_hef_name']
        def _prepare_infer_inputs(self, inputs): return inputs
        def _create_reusable_binding_slot(self): return {'binding': object()}
        def _fill_reusable_slot_inputs(self, *args): pass
        def _submit_reusable_slot(self, *args, **kwargs): pass
        def _wait_reusable_slot(self, *args): pass
        def _binding_output(self, *args): return SimpleNamespace(get_buffer=lambda: raw)
        def describe_io(self): return {'runtime_output_format': 'uint8'}
    monkeypatch.setattr(native, '_extract_slot_outputs', lambda sess, slot: {'canonical': raw})
    inputs = {'images': np.zeros((2, 2, 3), np.uint8)}
    recording = {}
    result = native._capture_raw_hailo10_sample(Session(), inputs, diagnostic_capture=recording)
    assert set(result) == {'canonical'}
    assert set(recording['raw_outputs']) == {'physical_hef_name'}
    assert not np.shares_memory(recording['raw_outputs']['physical_hef_name'], raw)
    assert native._capture_raw_hailo10_sample(Session(), inputs)['canonical'] is raw


def synthetic_reference_graphs(directory):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    import native_trt_from_benchmarkset as builder
    directory.mkdir()
    constant = np.array([[[0, 0, 1, 1, .5, 2], [1, 1, 2, 2, .6, 3]]], np.float32).transpose(0, 2, 1)
    p1 = h.make_model(h.make_graph([h.make_node('Identity', ['const'], ['cut'])], 'synthetic_part1',
        [h.make_tensor_value_info('images', T.FLOAT, [1, 3, 2, 2])], [h.make_tensor_value_info('cut', T.FLOAT, [1, 6, 2])], [nh.from_array(constant, 'const')]), opset_imports=[h.make_opsetid('', 18)], ir_version=10)
    p2 = h.make_model(h.make_graph([h.make_node('Transpose', ['cut'], ['output0'], perm=[0, 2, 1])], 'synthetic_part2',
        [h.make_tensor_value_info('cut', T.FLOAT, [1, 6, 2])], [h.make_tensor_value_info('output0', T.FLOAT, [1, 2, 6])]), opset_imports=[h.make_opsetid('', 18)], ir_version=10)
    onnx.save(p1, directory / 'p1.onnx')
    onnx.save(p2, directory / 'p2.onnx')
    bridge, _ = builder._make_uint8_dequant_bridge_onnx(directory / 'p2.onnx', directory / 'build', root=directory,
                       case='b398', scale=.1, zero_point=0., boundary_layout='memory_nwc_to_ncw')
    from PIL import Image
    Image.new('RGB', (2, 2), (20, 30, 40)).save(directory / 'image.jpg')
    return {'part1': directory / 'p1.onnx', 'part2': directory / 'p2.onnx', 'image': directory / 'image.jpg'}, bridge, constant


def test_t06_4_t06_6_real_bridge_reference_and_exporter_loader_preserve_invalid_bn6(tmp_path):
    files, bridge, constant = synthetic_reference_graphs(tmp_path / 'graphs')
    output = tmp_path / 'results'
    output.mkdir()
    shutil.copyfile(bridge, output / 'bound_bridge.onnx')
    raw = np.round(constant.transpose(0, 2, 1) / .1).astype(np.uint8)
    invalid = constant.transpose(0, 2, 1).copy()
    invalid[..., 4] = -3.0
    from PIL import Image
    hwc = np.array(Image.open(files['image']).convert('RGB'))
    arrays = {'A_000': hwc, 'B_000': raw, 'C_000': raw.reshape(1, 6, 2), 'E_000': invalid}
    request = {'remote_artifacts': {'bridge': {'sha256': diag.sha256(bridge)}},
               'reference_part1_sha256': diag.sha256(files['part1']), 'reference_part2_sha256': diag.sha256(files['part2']),
               'engine_io': {'inputs': [{'name': 'cut'}]}}
    metadata = {'status': 'captured', 'stages': {'A': {'names': ['images']}, 'B': {'names': ['raw_name']}, 'C': {'names': ['cut']}, 'E': {'names': ['output0']}},
                'runtime_io': {'runtime_output_quantization': {'raw_name': {'scale': .1, 'zero_point': 0.}}, 'outputs_dequantized': False}}
    diag.dump_packet(output, arrays, metadata)
    result = probe.offline_reference(request, files, output)
    loaded, values = diag.load_packet(output)
    np.testing.assert_allclose(values['D_000'], constant, atol=1e-6)
    assert loaded['stages']['D']['direct_engine_observation'] is False
    assert loaded['quantization_comparison']['hef_vs_bound_bridge_parameters_equal'] is True
    np.testing.assert_array_equal(values['E_000'], invalid)
    assert loaded['original_endpoint_validation']['endpoint_contract_complete'] is False
    assert loaded['regular_path_released'] is False
    assert loaded['runtime_fix_verified'] is False
    assert loaded['fresh_central_quality_required'] is True
    assert loaded['comparisons']['E_vs_reference_P2']['tensors']['output0']['max_abs'] > 3


def test_t06_6_large_range_violations_never_clipped_or_relabeled():
    a = np.zeros((1, 300, 6), np.float32)
    a[..., 4] = 622.12
    a[..., 0] = 12
    before = a.copy()
    from onnx_splitpoint_tool.native_output_endpoint import runtime_output_contract
    result = runtime_output_contract('detection', {'output0': a}, raw_fallback=False,
                                      declared_contract={'stage': 'decoded_nms', 'output_format': 'bn6_detections'})
    assert result['endpoint_contract_complete'] is False
    assert diag.bn6_stats(a)['score_violation_count'] == 300
    np.testing.assert_array_equal(a, before)


def test_t06_1_t06_2_real_worker_path_only_device_handles_synthetic(tmp_path, monkeypatch):
    run, files, row = minimal_case(tmp_path)
    # Replace only the fake model artifacts and actual physical device handles;
    # preserve production preparation, slot capture, exact name selection,
    # pinned-input materialization policy, packet exporter and packet loader.
    reference, bridge, constant = synthetic_reference_graphs(tmp_path / 'graphs')
    shutil.copyfile(reference['image'], files['image'])
    request, resolved = probe.resolve_case(run, 'yolo26m')
    folder = tmp_path / 'private_stage'
    folder.mkdir()
    shutil.copyfile(files['hef'], folder / 'part1.hef')
    shutil.copyfile(files['image'], folder / 'image.jpg')
    originals = {'bridge': bridge, 'engine': tmp_path / 'original.engine', 'receipt': tmp_path / 'receipt.json'}
    originals['engine'].write_bytes(b'SYNTHETIC_ENGINE_HARDWARE_HANDLE')
    originals['receipt'].write_text('{"synthetic":true}')
    request['remote_artifacts'] = {k: {'path': str(v), 'sha256': diag.sha256(v)} for k, v in originals.items()}
    raw = np.round(constant.transpose(0, 2, 1) / .1).astype(np.uint8)
    captured_calls = []
    io = request['runtime_io']
    io.update(runtime_output_formats={'actual_hef_name': 'UINT8'})
    class Session:
        _hef_output_names = ['actual_hef_name']
        _output_name_hef_to_canonical = {'actual_hef_name': 'cut'}
        def describe_io(self): return io
        def quantize_input(self, name, source): return np.round(source / (1/255)).astype(np.uint8)
        def _prepare_infer_inputs(self, inputs): return inputs
        def _create_reusable_binding_slot(self): return {'binding': object()}
        def _fill_reusable_slot_inputs(self, *args): pass
        def _submit_reusable_slot(self, *args, **kwargs): captured_calls.append('hailo_inference')
        def _wait_reusable_slot(self, *args): pass
        def _binding_output(self, *args): return SimpleNamespace(get_buffer=lambda: raw)
    class Backend:
        def __init__(self, **kwargs): self.options = kwargs
        def prepare(self, config, artifacts_dir):
            assert config.model_path == folder / 'part1.hef'
            return SimpleNamespace(input_names=['images'], handle=SimpleNamespace(session=Session(), runtime_input_shapes={'images': (2, 2, 3)}, input_shapes={}))
        def cleanup(self, prepared): captured_calls.append('hailo_cleanup')
    class TRT:
        prepare_inputs = native.NativeTRT.prepare_inputs
        def __init__(self, path):
            assert path.read_bytes() == originals['engine'].read_bytes()
            self.inputs, self.outputs = ['cut'], ['output0']
            self.shapes = {'cut': (1, 6, 2), 'output0': (1, 2, 6)}
            self.dtypes = {'cut': np.dtype('uint8'), 'output0': np.dtype('float32')}
            self.host_in = {'cut': np.empty((1, 6, 2), np.uint8)}
        def run_prepared(self):
            captured_calls.append('trt_inference')
            result = constant.transpose(0, 2, 1).copy()
            result[..., 4] = 622.12
            return {'output0': result}
        def close(self): captured_calls.append('trt_cleanup')
    monkeypatch.setattr(native, 'HailoBackend', Backend)
    monkeypatch.setattr(native, 'NativeTRT', TRT)
    output = tmp_path / 'output'
    worker.capture(request, folder, output)
    report, arrays = diag.load_packet(output)
    assert captured_calls == ['hailo_inference', 'trt_inference', 'trt_cleanup', 'hailo_cleanup']
    assert report['quality_binding_created'] is False
    assert report['regular_path_released'] is False
    assert report['stages']['B']['names'] == ['actual_hef_name']
    assert report['stages']['C']['names'] == ['cut']
    np.testing.assert_array_equal(arrays['B_000'].reshape(1, 6, 2), arrays['C_000'])
    assert report['raw_output_checks']['output0']['score_violation_count'] == 2
    for key, value in diag.FLAGS.items():
        assert report[key] is value
    assert all(diag.sha256(originals[k]) == request['remote_artifacts'][k]['sha256'] for k in originals)


def test_t06_5_collector_missing_result_and_timeout_cannot_return_success(tmp_path, monkeypatch):
    for model in probe.CASES:
        minimal_case(tmp_path, model)
    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired('ssh', 30)
    monkeypatch.setattr(probe, 'run_transport', unavailable)
    result = probe.main(['--run-dir', str(tmp_path / 'relocated_original_run'), '--output-dir', str(tmp_path / 'deliveries'), '--source-root', str(ROOT)])
    assert result == 2
    summaries = list((tmp_path / 'deliveries').rglob('collection_summary.json'))
    assert len(summaries) == 1
    report = json.loads(summaries[0].read_text())
    assert report['status'] == 'incomplete'
    assert report['collection_complete'] is False
    assert all(report[k] is v for k, v in diag.FLAGS.items())
    assert list((tmp_path / 'deliveries').glob('*.zip'))


def test_t06_1_t06_6_original_contracts_keep_distinct_generic_and_native_failures(tmp_path):
    fixture = json.loads((ROOT / 'tests/fixtures/v27931_complete_set/hailo26_original_contract_summary.json').read_text())
    assert fixture['hardware_runtime_fix_verified'] is False
    assert fixture['raw_payloads_available'] is False
    for original in fixture['cases']:
        generic = original['row_projection']
        session = generic['native_tensorrt']['sessions']['part2:tensorrt']
        native_row = original['native_result']
        assert session['explicit_engine'] is True
        assert session['cache_hit'] is True
        assert len(session['engine_sha256']) == 64
        assert '/_onnx_splitpoint_cache/' in session['engine']
        assert native_row['engine'] == ''
        assert 'missing_native_trt_part2_engine' in native_row['error']
        assert Path(generic['run_cfg']['image']).name != Path(native_row['input_image']).name
        assert original['case'] == probe.CASES[original['model']]
        diag.write_json(tmp_path / original['source_path'], [generic])
        diag.write_json(tmp_path / original['native_path'], native_row)
        with pytest.raises(ValueError, match='original_generic_image'):
            probe.resolve_case(tmp_path, original['model'])


def test_t06_5_probe_blocks_existing_measurement_without_touching_process(tmp_path):
    proc = tmp_path / 'proc'
    (proc / '12').mkdir(parents=True)
    marker = proc / '12/cmdline'
    marker.write_bytes(b'python\x00native_hailo10_trt_e2e_from_benchmarkset.py\x00--energy-workload-only\x00')
    original = marker.read_bytes()
    assert worker.active_measurements(proc) == ['12']
    assert marker.read_bytes() == original
