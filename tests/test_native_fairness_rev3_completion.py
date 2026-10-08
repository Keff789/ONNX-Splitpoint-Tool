"""Task/evidence split, exact format adapters and pre-encoding RGB regressions."""
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool import native_detection_postprocess as ndp
from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime
from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract, prepare_rgb_uint8_image
from onnx_splitpoint_tool.runners import native_full_input
from scripts import native_deepx_trt_e2e_from_benchmarkset as dx
from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10
from scripts import native_hailo_trt_fifo_from_benchmarkset as h8
from scripts import smoke_hailo10_hef_runner as full
from scripts import native_full_semantic_dump as semantic
from test_yolov7_bound_full_postprocess import heads, contract
from test_v2777_yolo11_native_adapter import _yolo11_outputs, _raw_source
from test_v279_native_three_stage import _decoded_completion_contract


@pytest.mark.parametrize('mode', ['bound_dense', 'bound_exact_sparse'])
def test_full_task_only_has_no_crypto_and_finalizes_exact_oracle(mode, monkeypatch):
    outputs = heads()
    frozen = contract(outputs)
    strict = ndp.FrozenDetectionPostprocessor(frozen).process(outputs)
    runtime = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode=mode)
    runtime.assert_implementation_unchanged()
    def forbidden(*args, **kwargs):
        raise AssertionError('evidence work inside timed task')
    with monkeypatch.context() as m:
        m.setattr(ndp, 'canonical_json_sha256', forbidden)
        m.setattr(ndp, '_implementation_artifacts', forbidden)
        m.setattr(Path, 'open', forbidden)
        result = runtime.process_task(outputs)
    assert result['detections'] == strict['detections']
    assert 'completed_result_artifact_sha256' not in result
    assert runtime.completed_count == 1
    evidence = runtime.finalize_evidence()
    assert evidence['completed_result_artifact_sha256'] == strict['completed_result_artifact_sha256']
    assert evidence['postflight_oracle']['per_frame_parity_claimed'] is False
    assert runtime.completed_count == 1


def test_full_postflight_cannot_attest_overwritten_borrowed_outputs():
    outputs = heads()
    runtime = ndp.FrozenDetectionPostprocessor(contract(outputs), runtime_mode='bound_exact_sparse')
    runtime.process_task(outputs)
    for value in outputs.values(): value.fill(-100)
    with pytest.raises(ndp.FrozenPostprocessError, match='postflight_oracle_mismatch'):
        runtime.finalize_evidence()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_yolo11_discarded_raw_value_is_invalid_in_all_modes(bad):
    outputs = _yolo11_outputs()
    c = ndp.build_detection_completion_execution_contract(
        model_id='yolo11l', outputs=outputs, input_hw=[640,640], original_wh=[853,479],
        source_endpoint_contract=_raw_source(outputs),
        preprocess={'mode':'letterbox','rgb':True,'pad_value':114})
    for mode in ('strict', 'auto'):
        runtime = ndp.create_detection_completion_runtime(c, runtime_mode=mode)
        malformed = {k:v.copy() for k,v in outputs.items()}
        next(iter(malformed.values())).flat[-1] = bad
        with pytest.raises(ValueError, match='nonfinite'):
            runtime.process(malformed)
        assert runtime.completed_count == 0


def test_yolo11_raw_adapter_matches_strict_and_keeps_task_thresholds():
    outputs = _yolo11_outputs()
    c = ndp.build_detection_completion_execution_contract(
        model_id='yolo11l', outputs=outputs, input_hw=[640,640], original_wh=[853,479],
        source_endpoint_contract=_raw_source(outputs),
        preprocess={'mode':'letterbox','rgb':True,'pad_value':114})
    fast = ndp.create_detection_completion_runtime(c)
    assert isinstance(fast, FastDetectionCompletionRuntime)
    expected = ndp.DetectionCompletionRuntime(c).process(outputs)['detections']
    assert fast.process(outputs)['detections'] == expected
    assert fast.attestation(completed_work_units=1)['status'] == 'passed'


@pytest.mark.parametrize('task,mode,pad', [('detection','letterbox',114),('classification','resize',0)])
def test_actual_full_split_loaders_share_asymmetric_pre_encoding_rgb(tmp_path, task, mode, pad):
    source = np.arange(7*13*3, dtype=np.uint16).reshape(7,13,3).astype(np.uint8)
    path = tmp_path/'asymmetric.png'; Image.fromarray(source).save(path)
    expected, _ = prepare_rgb_uint8_image(source, canonical_image_preprocessing_contract(task, (32,32)))
    shape = (32,32,3)
    fake = SimpleNamespace(input_names=['images'], handle=SimpleNamespace(runtime_input_shapes={'images':shape}))
    h8_rgb = h8._hailo8_python_input(fake,path,quantized=True,preprocess_mode=mode,letterbox_pad_value=pad)['images']
    h10_binding = {}
    h10_rgb = h10._image_to_shape(str(path),shape,True,preprocess_mode=mode,letterbox_pad_value=pad,task=task,timing_input_binding=h10_binding)
    dx_binding = {}
    dx_rgb = dx._exact_input_from_quality_contract({'shape':list(shape),'dtype':'uint8','layout':'HWC','task':task},str(path),preprocess_mode=mode,letterbox_pad_value=pad,timing_input_binding=dx_binding)
    _, full_rgb, _ = full._image_tensor(path,shape,quantized=True,task=task,preprocess_mode=mode,letterbox_pad=pad)
    _, dx_full_rgb, _ = native_full_input._prepare_tensor(path,shape=shape,dtype=np.dtype('uint8'),layout='HWC',task=task,normalization='embedded_dxcom_preprocessing',preprocess_mode=mode,pad_value=pad)
    _, trt_rgb, _ = semantic._prepare_image_tensor(path,shape,np.dtype('uint8'),task=task,preprocess_mode=mode,letterbox_pad=pad)
    for actual in [h8_rgb,h10_rgb,dx_rgb,full_rgb,dx_full_rgb,trt_rgb]:
        assert np.array_equal(actual, expected)
    assert h10_binding == dx_binding


def test_cpp_loader_writes_the_same_canonical_prepared_payload(tmp_path, monkeypatch):
    import sys
    source = np.arange(5*11*3, dtype=np.uint8).reshape(5,11,3)
    path = tmp_path/'input.png'; Image.fromarray(source).save(path)
    class HEF:
        def __init__(self, path): pass
        def get_input_vstream_infos(self): return [SimpleNamespace(shape=(32,32,3))]
    monkeypatch.setitem(sys.modules, 'hailo_platform', SimpleNamespace(HEF=HEF))
    args=SimpleNamespace(preprocess_mode_effective='letterbox',letterbox_pad_value=114,task='detection')
    h8._prepare_hailo8_cpp_rgb_input(hef=tmp_path/'unused.hef',image=path,work=tmp_path,args=args)
    expected,_=prepare_rgb_uint8_image(source,canonical_image_preprocessing_contract('detection',(32,32)))
    assert Path(args.prepared_input_rgb).read_bytes()==expected.tobytes()
    assert args.expected_prepared_input_sha256==args.timing_input_binding['canonical_rgb_sha256']
