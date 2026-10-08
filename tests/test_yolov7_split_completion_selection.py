"""Split selection, real FIFO completion and existing oracle without hardware."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as ndp
from onnx_splitpoint_tool.native_three_stage import (
    FastDetectionCompletionRuntime, verify_fast_completion_dump_binding,
)
from test_v282_hailo8_sentinel_evidence import _yolov7_case
from test_v279_native_three_stage import _decoded_completion_contract
from test_v272_detection_hotloop_integration import (
    _FakeDeepX, _FakeHailo10Session, _FakeTRT,
)
from scripts import native_deepx_trt_e2e_from_benchmarkset as deepx
from scripts import native_hailo10_trt_e2e_from_benchmarkset as hailo10
from scripts import native_producer_e2e_eval_runner as dispatcher


def test_auto_uses_supported_adapters_and_full_explicit_default_stays_strict():
    _, contract = _yolov7_case()
    assert isinstance(ndp.create_detection_completion_runtime(contract), FastDetectionCompletionRuntime)
    _, other = _decoded_completion_contract()
    assert isinstance(ndp.create_detection_completion_runtime(other), FastDetectionCompletionRuntime)
    assert ndp.FrozenDetectionPostprocessor(contract['processor_contract']).runtime_mode == 'strict'
    with pytest.raises(ndp.FrozenPostprocessError, match='optimized_supported'):
        ndp.create_detection_completion_runtime(other, observation_relation='separate_postflight_replay')


@pytest.mark.parametrize('channel', [0, 5, 84])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_discarded_nonfinite_frame_cannot_advance_completion(channel, value):
    outputs, contract = _yolov7_case()
    runtime = ndp.create_detection_completion_runtime(contract)
    malformed = {k: v.copy() for k, v in outputs.items()}
    next(iter(malformed.values()))[0, 0, 0, 0, channel] = value
    with pytest.raises((ndp.FrozenPostprocessError, ValueError), match='nonfinite'):
        runtime.process(malformed)
    assert runtime.completed_count == 0
    runtime.process(outputs)
    assert runtime.attestation()['completed_work_units'] == 1


@pytest.mark.parametrize('empty', [False, True])
def test_strict_bound_dense_and_fast_preserve_records(empty):
    outputs, contract = _yolov7_case()
    if empty:
        for value in outputs.values():
            value.fill(-100)
    records = []
    for mode in ('strict', 'bound_dense', 'fast_oracle_outside_timing'):
        runtime = ndp.create_detection_completion_runtime(contract, runtime_mode=mode)
        records.append(runtime.process(outputs)['detections'])
        assert ndp.detection_completion_runtime_source_check(runtime)['unchanged']
    assert records[0] == records[1] == records[2]
    if empty:
        assert records[0] == []


def test_contract_copy_and_postflight_contract_and_source_checks(tmp_path):
    outputs, contract = _yolov7_case()
    helper = tmp_path / 'bound_helper.py'
    helper.write_text('version = 1\n')
    runtime = ndp.create_detection_completion_runtime(contract, source_paths=(helper,))
    contract['processor_contract']['max_detections'] = 1
    assert runtime.execution_contract['processor_contract']['max_detections'] != 1
    runtime.process(outputs)
    helper.write_text('version = 2\n')
    with pytest.raises(ndp.FrozenPostprocessError, match='implementation_changed'):
        runtime.attestation()
    helper.write_text('version = 1\n')
    runtime.execution_contract['processor_contract']['confidence_threshold'] = .9
    with pytest.raises(ndp.FrozenPostprocessError, match='contract_or_mode_changed'):
        runtime.attestation()


def test_deepx_imported_trt_helper_is_an_explicit_source(monkeypatch):
    _, contract = _yolov7_case()
    helper = str(Path(sys.modules[deepx.NativeTRT.__module__].__file__).absolute())
    runtime = ndp.create_detection_completion_runtime(
        contract, source_paths=(deepx.__file__, helper),
    )
    before = ndp.detection_completion_runtime_source_check(runtime)['before']
    assert helper in before and str(Path(deepx.__file__).absolute()) in before
    actual_reader = ndp.native_full_runtime_source_hashes
    def changed_helper(*args, **kwargs):
        hashes = actual_reader(*args, **kwargs)
        hashes[helper] = '0' * 64
        return hashes
    monkeypatch.setattr(ndp, 'native_full_runtime_source_hashes', changed_helper)
    with pytest.raises(ndp.FrozenPostprocessError, match='implementation_changed'):
        ndp.detection_completion_runtime_source_check(runtime)


@pytest.mark.parametrize('backend', ['hailo10', 'deepx'])
def test_actual_fifo_fast_oracle_snapshot_and_dispatcher_admission(backend, tmp_path):
    outputs, contract = _yolov7_case()
    class ReusableTRT(_FakeTRT):
        def run_prepared(self):
            return outputs
    measured = ndp.create_detection_completion_runtime(contract, sentinel_context={
        'runtime_instance_id': 'fresh-test', 'repetition_id': 'rep1',
        'process_local_repetition_index': 1, 'input_image_sha256': 'a' * 64,
    })
    warmup = ndp.create_detection_completion_runtime(contract)
    kwargs = dict(frames=3, warmup=1, queue_depth=2, task='detection',
                  completion_runtime=measured, warmup_completion_runtime=warmup)
    if backend == 'hailo10':
        result = hailo10._hailo10_async_fifo_run(
            _FakeHailo10Session(), {'images': np.zeros(1, np.float32)},
            ReusableTRT(), inflight=2, **kwargs)
        runner = hailo10
    else:
        result = deepx._deepx_fifo_run(_FakeDeepX(), np.zeros(1, np.float32), ReusableTRT(), **kwargs)
        runner = deepx
    assert measured.completed_count == 3 and warmup.completed_count == 1
    assert result['completion_observation_relation'] == 'postflight_oracle_sentinel'
    assert result['completion_runtime_source_binding']['unchanged']
    snapshot = measured.snapshot_completed_source()
    for key in outputs:
        assert not np.shares_memory(snapshot[key], outputs[key])
        outputs[key].fill(0)
    result['repetition_records'] = [copy.deepcopy(result)]
    ndp.persist_detection_completion_execution_artifacts(result, output_path=tmp_path / 'artifact.json')
    assert dispatcher._detection_completion_error(result) == ''
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    runner._bind_fast_output_manifest(str(manifest), result['completion_execution_attestation'])
    verify_fast_completion_dump_binding(result['completion_execution_attestation'], json.loads(manifest.read_text()), result)
    changed = copy.deepcopy(result)
    changed['completion_execution_attestation']['fast_content_sha256'] = '0' * 64
    assert 'postflight_completion_invalid' in dispatcher._detection_completion_error(changed)
