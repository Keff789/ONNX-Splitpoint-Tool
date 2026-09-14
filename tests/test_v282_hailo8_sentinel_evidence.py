"""A2: actual FIFO, sparse decoder, frozen oracle, dump and validation linkage.

Only the hardware transport and output-contract discovery are replaced. Outputs
vary by frame and fresh repetition; reusable TRT buffers are invalidated on close.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime, build_detection_completion_runtime,
    canonical_json_sha256, persist_detection_completion_execution_artifacts,
)
from onnx_splitpoint_tool.native_three_stage import (
    FastDetectionCompletionRuntime, NativeThreeStageError,
    verify_fast_completion_attestation, verify_fast_completion_dump_binding,
)
from scripts import native_hailo_trt_fifo_from_benchmarkset as runner
from scripts.native_producer_validate_visualize import _completed_v2_self_reference_detection
from scripts.validate_output_dumps import load_dump
from test_v272_yolov7_head_mapping import _heads_640, _raw_source
from test_v279_native_three_stage import _decoded_completion_contract


def _yolov7_case():
    outputs = _heads_640()
    _set_yolov7_frame(outputs, 1)
    runtime = build_detection_completion_runtime(
        model_id='yolov7_paper', outputs=outputs, input_hw=[640, 640],
        original_wh=[640, 640],
        source_endpoint_contract=_raw_source(outputs, endpoint_hash='b' * 64),
        preprocess={'mode': 'letterbox', 'rgb': True, 'pad_value': 114},
    )
    return outputs, runtime.execution_contract


def _set_yolov7_frame(outputs, frame):
    for value in outputs.values():
        value.fill(-20)
    head = outputs['vendor_small'][0, 0, 30, 20 + int(frame)]
    head[:4] = 0
    head[4] = 20
    head[5 + int(frame) % 8] = 20


@pytest.mark.parametrize('family', ['yolov7', 'decoded'])
def test_postflight_snapshot_survives_reused_return_buffer_without_extra_inference(family):
    outputs, contract = _yolov7_case() if family == 'yolov7' else _decoded_completion_contract()
    runtime = FastDetectionCompletionRuntime(contract)
    observed = []
    for frame in range(1, 4):
        if family == 'yolov7':
            _set_yolov7_frame(outputs, frame)
        else:
            outputs['detections'][0, 0, 0] = 8 + frame
        observed.append(runtime.process(outputs)['detections'])
    assert len({json.dumps(value, sort_keys=True) for value in observed}) == 3
    assert np.shares_memory(next(iter(runtime.last_source_outputs.values())), next(iter(outputs.values())))
    attestation = runtime.attestation(completed_work_units=3)
    snapshot = runtime.last_source_outputs
    assert not np.shares_memory(next(iter(snapshot.values())), next(iter(outputs.values())))
    assert not next(iter(snapshot.values())).flags.writeable
    for value in outputs.values():
        value.fill(0)
    verify_fast_completion_attestation(attestation, execution_contract=contract, outputs=snapshot)
    assert runtime.completed_count == 3
    assert attestation['last_result']['detections'] == observed[-1]
    assert runtime.attestation(completed_work_units=3) == attestation
    with pytest.raises(NativeThreeStageError, match='source_tensor_content_differs'):
        verify_fast_completion_attestation(attestation, execution_contract=contract, outputs=outputs)


@pytest.fixture
def actual_fifo_run(tmp_path, monkeypatch):
    outputs, contract = _yolov7_case()
    hardware = []
    class Backend:
        def __init__(self):
            self.calls = 0
        def run(self, *_):
            self.calls += 1
            return SimpleNamespace(outputs={'boundary': np.full((1, 4), self.calls, dtype=np.float32)})
        def cleanup(self, *_):
            pass
    class TRT:
        inputs = ['boundary']
        shapes = {'boundary': (1, 4)}
        dtypes = {'boundary': np.dtype(np.float32)}
        def __init__(self, number):
            self.number = number
            self.outputs = {name: value.copy() for name, value in outputs.items()}
            self.frames = []
        def run(self, feeds):
            frame = self.number * 5 + int(feeds['boundary'].flat[0])
            _set_yolov7_frame(self.outputs, frame)
            self.frames.append(frame)
            return self.outputs
        def close(self):
            for value in self.outputs.values():
                value.fill(0)
    def open_runtime(**_kw):
        backend = Backend()
        trt = TRT(len(hardware))
        hardware.append((backend, trt))
        return backend, SimpleNamespace(input_names=['images'], handle=SimpleNamespace(runtime_input_shapes={'images': (640, 640, 3)})), trt
    monkeypatch.setattr(runner, '_open_hailo8_python_runtime', open_runtime)
    monkeypatch.setattr(runner, '_hailo8_python_input', lambda *_a, **_kw: {'images': np.full((640, 640, 3), 114, dtype=np.uint8)})
    monkeypatch.setattr(runner, '_hailo8_detection_completion_contract', lambda **_kw: contract)
    # Runtime contract discovery needs installed ONNX/SDK model metadata; the
    # test uses the real verified model-bound contract constructed above.
    monkeypatch.setattr(runner, '_annotate_output_contract', lambda *_a: None)
    image = tmp_path / 'fixed_image.jpg'
    image.write_bytes(b'exact shared original image')
    args = SimpleNamespace(
        model_id='yolov7_paper', hailo_format='uint8', preprocess_mode_effective='letterbox',
        letterbox_pad_value=114, dump_outputs=True, dump_boundary=True, output_dir='', boundary_dir='',
        benchmark_set=str(tmp_path), precision='fp16', repetitions=3,
        completion_runtime_mode='fast_oracle_outside_timing', frames=3, warmup=1, queue_depth=2, duration_s=0,
    )
    payload, _, _ = runner._run_hailo8_python_detection(
        bs=tmp_path, case='b009', hef=tmp_path/'part1.hef', engine=tmp_path/'part2.engine',
        image=image, work=tmp_path, args=args, quality_binding=None,
    )
    payload['task'] = 'detection'
    persist_detection_completion_execution_artifacts(payload, output_path=tmp_path/'sentinel.json')
    return payload, contract, hardware


def test_actual_fifo_preserves_each_repetition_dump_and_last_frame_selection(actual_fifo_run):
    payload, contract, hardware = actual_fifo_run
    assert payload['aggregate_total_completed_frames'] == 9
    assert payload['semantic_evidence_selection'] == 'last_completed_repetition_never_best_of'
    assert payload['semantic_evidence_repetition_index'] == 3
    assert len({row['native_fifo_output_manifest'] for row in payload['repetition_records']}) == 3
    # One contract probe plus one warmup + three counted calls per fresh runtime.
    assert [len(trt.frames) for _, trt in hardware] == [1, 4, 4, 4]
    snapshots = []
    for index, row in enumerate(payload['repetition_records'], start=1):
        manifest = Path(row['native_fifo_output_manifest'])
        tensors, _ = load_dump(str(manifest))
        meta = json.loads(manifest.read_text())
        verify_fast_completion_dump_binding(row['completed_task_endpoint_attestation'], meta, row)
        verified = verify_fast_completion_attestation(row['completed_task_endpoint_attestation'], execution_contract=contract, outputs=tensors)
        assert verified['sentinel_identity']['completed_work_unit_index'] == 3
        assert verified['sentinel_identity']['process_local_repetition_index'] == index
        snapshots.append(verified['artifact']['source_content_sha256'])
    assert len(set(snapshots)) == 3
    tensors, _ = load_dump(payload['native_fifo_output_manifest'])
    meta = json.loads(Path(payload['native_fifo_output_manifest']).read_text())
    result = _completed_v2_self_reference_detection(tensors, tensors, payload, dump_metadata=meta)
    assert result['available'] is True, result
    assert result['native_detections'] == result['reference_detections']
    assert payload['completed_task_endpoint_attestation'] == payload['repetition_records'][-1]['completed_task_endpoint_attestation']
    swapped = copy.deepcopy(payload['repetition_records'])
    swapped[0], swapped[2] = swapped[2], swapped[0]
    with pytest.raises(RuntimeError, match='repetition_identity_mismatch'):
        runner._aggregate_repetition_payloads(swapped)


@pytest.mark.parametrize('mutation', ['repetition', 'image', 'endpoint', 'frame', 'scope', 'attestation'])
def test_actual_dump_binding_rejects_wrong_origin(actual_fifo_run, mutation):
    payload, _, _ = actual_fifo_run
    tensors, _ = load_dump(payload['native_fifo_output_manifest'])
    meta = json.loads(Path(payload['native_fifo_output_manifest']).read_text())
    if mutation == 'repetition':
        payload['semantic_evidence_repetition_index'] = 1
    elif mutation == 'image':
        meta['input_image_sha256'] = 'a' * 64
    elif mutation == 'endpoint':
        meta['completion_sentinel_identity']['source_endpoint_contract_hash'] = 'a' * 64
    elif mutation == 'frame':
        meta['completion_sentinel_identity']['completed_work_unit_index'] = 2
    elif mutation == 'scope':
        meta['dump_inference_scope'] = 'separate_bound_probe_inference'
    elif mutation == 'attestation':
        meta['completion_attestation_sha256'] = payload['repetition_records'][0]['completed_task_endpoint_attestation']['attestation_sha256']
    result = _completed_v2_self_reference_detection(tensors, tensors, payload, dump_metadata=meta)
    assert result['available'] is False
    assert 'fast_oracle_dump_' in result['reason']


def test_same_tensor_with_different_decoding_is_distinguished_from_changed_tensor(monkeypatch):
    outputs, contract = _decoded_completion_contract()
    runtime = FastDetectionCompletionRuntime(contract)
    runtime.process(outputs)
    attestation = runtime.attestation()
    original = DetectionCompletionRuntime.process
    def different_platform_decode(self, values):
        result = copy.deepcopy(original(self, values))
        result['detections'][0]['score'] = float(np.nextafter(np.float32(result['detections'][0]['score']), np.float32(0)))
        return result
    monkeypatch.setattr(DetectionCompletionRuntime, 'process', different_platform_decode)
    # This fault injection tests diagnostic classification only, not a claim to
    # have reproduced the original hardware's floating-point discrepancy.
    with pytest.raises(NativeThreeStageError, match='same_source_decoded_result_differs'):
        verify_fast_completion_attestation(attestation, execution_contract=contract, outputs=outputs)


def test_original_four_yolov7_manifests_declare_same_source_as_oracle():
    fixture = json.loads((Path(__file__).parent/'fixtures/v282_hailo8_yolov7_declared_dump_identity.json').read_text())
    assert [row['case'] for row in fixture['cases']] == ['b009', 'b011', 'b044', 'b063']
    for row in fixture['cases']:
        outputs = sorted(row['dump_manifest']['outputs'], key=lambda value: value['shape'][2], reverse=True)
        # Physical outputs are already the exact float32 canonical layouts. This
        # reconstructs the existing source identity from recorded binary digests;
        # it does not claim to have checked the missing original .bin payloads.
        assert [value['shape'] for value in outputs] == [[1,3,80,80,85],[1,3,40,40,85],[1,3,20,20,85]]
        assert all(value['dtype'] == 'float32' for value in outputs)
        identity = {'schema': 'onnx-splitpoint/canonical-tensor-content', 'schema_version': 1,
                    'tensors': [{'ordinal': index, 'shape': value['shape'], 'dtype': value['dtype'], 'payload_sha256': value['sha256']}
                                for index, value in enumerate(outputs)]}
        assert canonical_json_sha256(identity) == row['attested_source_content_sha256']
        assert set(row['source_hashes_by_repetition']) == {row['attested_source_content_sha256']}
        assert row['top_attestation_equals_last_repetition'] is True
        assert len(set(row['output_paths_by_repetition'])) == 1
        assert row['raw_tensor_files_in_debugpack'] is False


@pytest.mark.parametrize('decoder_version', [1, 2])
def test_actual_numpy_dispatch_replay_uses_versioned_math_without_weakening_oracle(tmp_path, decoder_version):
    """Reproduce numeric portability on actual NumPy paths, no patched decoder.

    Legacy v1 still exposes the error; current v2 must replay exactly. On platforms
    with no alternate NumPy CPU dispatch the legacy reproduction is inapplicable.
    """
    import os
    import subprocess
    import sys
    dispatched = getattr(np._core._multiarray_umath, '__cpu_dispatch__', [])
    if not dispatched:
        pytest.skip('NumPy has no alternate dispatched CPU implementation')
    root = Path(__file__).resolve().parents[1]
    script = tmp_path/'dispatch_replay.py'
    saved = tmp_path/'attestation.json'
    script.write_text('''
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); sys.path[:0] = [str(root), str(root/'tests')]
from test_v282_hailo8_sentinel_evidence import _yolov7_case
from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime, verify_fast_completion_attestation
from onnx_splitpoint_tool.runners.harness import yolo
if sys.argv[3] == '1':
    original_registration = yolo.registered_yolov7_decoder_contract
    def legacy_registration(**kw):
        old = original_registration(**kw)
        old.pop('decoder_contract_sha256'); old.pop('sigmoid_arithmetic')
        old['schema_version'] = 1
        old['decoder_contract_sha256'] = yolo._canonical_contract_sha256(old)
        return yolo.verify_yolov7_decoder_contract(old)
    yolo.registered_yolov7_decoder_contract = legacy_registration
outputs, contract = _yolov7_case()
rng = np.random.default_rng(82)
for value in outputs.values(): value.fill(-20)
for index in range(32):
    head = outputs['vendor_small'][0,0,8+index//8*15,8+index%8*8]
    head[:4] = rng.uniform(-2,2,4).astype(np.float32)
    head[4] = rng.uniform(1,5)
    head[5+index%8] = rng.uniform(2,6)
path = Path(sys.argv[2])
if sys.argv[-1] == 'write':
    runtime = FastDetectionCompletionRuntime(contract); runtime.process(outputs)
    att = runtime.attestation()
    path.write_text(json.dumps({'attestation': att, 'contract': contract}))
    print('local sparse/oracle exact pass')
else:
    data = json.loads(path.read_text())
    try:
        verify_fast_completion_attestation(data['attestation'], execution_contract=data['contract'], outputs=outputs)
        print('identical_replay')
    except Exception as exc:
        print(str(exc))
''')
    env = dict(os.environ)
    env.pop('NPY_DISABLE_CPU_FEATURES', None)
    baseline = subprocess.run([sys.executable, str(script), str(root), str(saved), str(decoder_version), 'write'], env=env, capture_output=True, text=True, timeout=60)
    assert baseline.returncode == 0, baseline.stderr
    assert baseline.stdout.strip() == 'local sparse/oracle exact pass'
    env['NPY_DISABLE_CPU_FEATURES'] = ','.join(dispatched)
    replay = subprocess.run([sys.executable, str(script), str(root), str(saved), str(decoder_version), 'read'], env=env, capture_output=True, text=True, timeout=60)
    assert replay.returncode == 0, replay.stderr
    if decoder_version == 2:
        assert replay.stdout.strip() == 'identical_replay'
    else:
        if replay.stdout.strip() == 'identical_replay':
            pytest.skip('This NumPy platform produces identical legacy values on available dispatch paths')
        assert replay.stdout.strip() == 'fast_oracle_dumped_sentinel_mismatch:same_source_decoded_result_differs'


def test_versioned_arithmetic_preserves_existing_scientific_policy_and_scalar_sigmoid():
    import math
    from onnx_splitpoint_tool.runners.harness import yolo
    args = {'model_id': yolo.YOLOV7_PAPER_MODEL_ID, 'model_sha256': yolo.YOLOV7_PAPER_ONNX_SHA256,
            'activation_mode': 'logits'}
    old = yolo.build_yolov7_decoder_contract(**args, schema_version=1)
    new = yolo.build_yolov7_decoder_contract(**args)
    assert old['decoder_contract_sha256'] != new['decoder_contract_sha256']
    assert yolo.verify_yolov7_decoder_contract(old) == old
    assert yolo.verify_yolov7_decoder_contract(new) == new
    assert new['schema_version'] == 2
    assert new.pop('sigmoid_arithmetic') == yolo.YOLOV7_SIGMOID_ARITHMETIC
    for value in (old, new):
        value.pop('schema_version'); value.pop('decoder_contract_sha256')
    assert old == new  # All anchors, thresholds, geometry and NMS remain exact.
    values = np.concatenate([
        np.array([-100, -80, -20, -np.log(3), 0, np.log(3), 20, 80, 100], dtype=np.float32),
        np.random.default_rng(282).uniform(-80, 80, 10000).astype(np.float32),
    ])
    expected = np.array([1.0 / (1.0 + math.exp(-min(80., max(-80., float(x))))) for x in values], dtype=np.float32)
    assert np.array_equal(yolo._yolov7_sigmoid_float64_to_float32(values), expected)


def test_resealed_new_arithmetic_cannot_claim_old_implementation():
    from onnx_splitpoint_tool.native_detection_postprocess import (
        verify_frozen_postprocess_contract, frozen_postprocess_invariant_identity,
    )
    _, execution = _yolov7_case()
    contract = copy.deepcopy(execution['processor_contract'])
    contract['implementation_artifacts']['yolo_harness']['sha256'] = '31657e2717a8cb6cfda53e7de8e06b7b2286ecb430070811546fdec65327a17a'
    contract.pop('contract_sha256')
    contract['invariant_identity'] = frozen_postprocess_invariant_identity(contract)
    contract['invariant_contract_sha256'] = canonical_json_sha256(contract['invariant_identity'])
    contract['contract_sha256'] = canonical_json_sha256(contract)
    with pytest.raises(Exception, match='yolov7_sigmoid_arithmetic_implementation_mismatch'):
        verify_frozen_postprocess_contract(contract)


def test_current_sigmoid_float32_bytes_match_across_numpy_dispatch(tmp_path):
    import os
    import subprocess
    import sys
    dispatched = getattr(np._core._multiarray_umath, '__cpu_dispatch__', [])
    if not dispatched:
        pytest.skip('NumPy has no alternate dispatched CPU implementation')
    root = Path(__file__).resolve().parents[1]
    code = '''
import hashlib, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from onnx_splitpoint_tool.runners.harness.yolo import _yolov7_sigmoid_float64_to_float32
threshold = np.float32(np.log(3.))
values = np.concatenate([np.random.default_rng(282).uniform(-100,100,1_000_000).astype(np.float32),
    np.array([-80,80,0,-threshold,threshold,np.nextafter(threshold,np.float32(0)),np.nextafter(threshold,np.float32(10))], dtype=np.float32)])
result = _yolov7_sigmoid_float64_to_float32(values)
print(hashlib.sha256(result.tobytes()).hexdigest())
'''
    results = []
    for disabled in ('', ','.join(dispatched)):
        env = dict(os.environ)
        env['NPY_DISABLE_CPU_FEATURES'] = disabled
        result = subprocess.run([sys.executable, '-c', code, str(root)], env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        results.append(result.stdout.strip())
    assert len(results[0]) == 64 and results[0] == results[1]
