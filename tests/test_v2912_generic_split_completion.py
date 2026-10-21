"""Actual Generic completion hooks: no accelerators or reference inference."""
from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest

from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from onnx_splitpoint_tool.runners.task_completion import (
    measure_split_completion, valid_split_completion_repetitions,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'


def _module(monkeypatch):
    import onnx_splitpoint_tool.runners.task_completion as completion
    import onnx_splitpoint_tool.runners.harness.classification as classification
    import onnx_splitpoint_tool.native_detection_postprocess as detection
    for name, module in [('task_completion', completion), ('harness.classification', classification),
                         ('native_detection_postprocess', detection)]:
        monkeypatch.setitem(sys.modules, 'splitpoint_runners.' + name, module)
    loader = importlib.machinery.SourceFileLoader('_generic_completion_template_test', str(TEMPLATE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, loader.name, module)
    loader.exec_module(module)
    return module


def _descriptor(path, value=None):
    result = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if value is not None:
        result.update(shape=list(value.shape), dtype=str(value.dtype))
    return result


def _fixture(tmp_path):
    tmp_path = tmp_path / 'b001'
    tmp_path.mkdir()
    p1 = tmp_path / 'part1.onnx';p2 = tmp_path / 'part2.onnx'
    for path, source, target in ((p1, 'images', 'cut'), (p2, 'cut', 'logits')):
        graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', [source], [target])], 'metadata',
            [onnx.helper.make_tensor_value_info(source, onnx.TensorProto.FLOAT, [1, 6])],
            [onnx.helper.make_tensor_value_info(target, onnx.TensorProto.FLOAT, [1, 6])])
        onnx.save(onnx.helper.make_model(graph), path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'full_model': 'ABSENT_FULL.onnx', 'part1_model': p1.name,
        'part2_model': p2.name, 'orig_inputs': ['images'], 'orig_outputs': ['logits'],
        'part1_cut_names': ['cut'], 'part2_cut_names': ['cut']}))
    dxnn = tmp_path / 'deepx/deepx_m1/part1/model.dxnn';dxnn.parent.mkdir(parents=True);dxnn.write_bytes(b'existing dxnn')
    engine = tmp_path / 'existing.engine';engine.write_bytes(b'existing engine')
    prepared = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    input_path = tmp_path / 'input.npy';np.save(input_path, prepared)
    from PIL import Image
    image_path = tmp_path / 'input.png'; Image.fromarray(prepared).save(image_path)
    image_sha = _descriptor(image_path)['sha256']
    logits = np.array([[1, 6, 3, 4, 5, 2]], dtype=np.float32)
    reference = tmp_path / 'logits.npy';np.save(reference, logits)
    config = {'identity': {'model_id': 'fixture', 'case_id': 'b001', 'setup_id': 'fixture', 'precision': 'fp16'},
        'execution_id': 'new-session', 'original_run': 'original-run', 'task': 'classification',
        'measurement_boundary': 'prepared_input_before_p1_to_last_completed_task',
        'repetitions': 3, 'warmup_count': 100, 'task_count': 1000, 'queue_depth': 2,
        'input_binding': {'prepared_input': {**_descriptor(input_path, prepared), 'input_name': 'images'},
            'runtime_input_format': 'uint8', 'source_image': str(image_path), 'source_image_sha256': image_sha,
            'prepared_input_contract': {'shape': list(prepared.shape), 'dtype': str(prepared.dtype),
                'layout': 'HWC', 'normalization': 'none', 'task': 'classification',
                'preprocess_mode_effective': 'resize', 'source_image': str(image_path),
                'source_image_sha256': image_sha}},
        'completion_contract': {'task': 'classification', 'operation': 'top1_top5', 'label_offset': 0},
        'expected_runtime_artifacts': {'part1_runtime': _descriptor(dxnn), 'engine': _descriptor(engine)},
        'functional_reference': {'input_sha256': _descriptor(input_path)['sha256'],
            'outputs': {'logits': _descriptor(reference, logits)}},
        'case_dir': str(tmp_path), 'original_run_cfg': {'manifest': str(manifest), 'provider': 'tensorrt',
            'stage1_provider': 'deepx_m1', 'stage2_provider': 'tensorrt', 'model_id': 'fixture',
            'native_trt_precision': 'fp16', 'trt_cache_dir': str(tmp_path / 'cache'),
            'image_scale': 'auto', 'variants': 'composed'}}
    return config, logits


def test_timer_drains_warmup_and_retains_existing_pipeline_overlap():
    processor = ClassificationCompletion()
    first_measured = []
    p1_waiting = threading.Event()
    produced = 0
    consumed = 0
    def p1():
        nonlocal produced
        produced += 1
        if produced == 3:
            first_measured.append(processor.completed_count)
        if produced == 4:
            p1_waiting.set()
        return {'cut': np.array([[1, 2]])}
    def p2(inputs):
        nonlocal consumed
        consumed += 1
        if consumed == 3:
            assert p1_waiting.wait(1), 'P1 and P2 were serialized'
        return inputs
    report = measure_split_completion(p1, lambda x:x, p2, processor, warmup=2, tasks=6)
    assert first_measured == [2]
    assert report['completed_task_count'] == 6
    assert report['warmup_completed_count'] == 2
    assert report['measurement_start_ns'] >= report['warmup_end_ns']
    assert report['completed_task_fps'] == pytest.approx(6/report['measured_makespan_seconds'])


def test_explicit_mixed_control_cannot_be_omitted(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    config, _ = _fixture(tmp_path)
    config['mixed_input_control'] = {'path': '/bound/control.json', 'sha256': 'a' * 64}
    with pytest.raises(ValueError, match='explicit runtime binding'):
        module._run_generic_split_completion(config, tmp_path / 'result',
            lambda: None, lambda value: value, lambda value: value, {}, clone_payload=None)


def test_hailo_mixed_feed_preserves_existing_prepared_codes(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    config, _ = _fixture(tmp_path)
    prepared = config['input_binding']['prepared_input']
    value = np.load(prepared['path'], allow_pickle=False)
    session = SimpleNamespace(input_names=['hef_canonical_input'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Prepared Hailo codes must not be quantized again')
    monkeypatch.setattr(module, '_prepare_hailo_image_input', forbidden)
    mapped = module._completion_hailo_prepared_feed(config, session, {prepared['input_name']: value})
    assert mapped == {'hef_canonical_input': value}
    assert mapped['hef_canonical_input'] is value
    assert module._completion_hailo_inputs(config, 'hailo10h', mapped) is mapped
    with pytest.raises(ValueError, match='shape/dtype/layout differs'):
        module._completion_hailo_prepared_feed(config, session, {prepared['input_name']: value.astype(np.float32)})


def test_failed_mixed_control_stops_before_any_timed_task(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    config, logits = _fixture(tmp_path)
    config['mixed_input_control'] = {'path': '/bound/control.json', 'sha256': 'a' * 64}
    output = tmp_path / 'result'
    output.mkdir()
    submissions = []
    def control(index):
        assert index == 1
        raise RuntimeError('bounded mixed-input mismatch')
    with pytest.raises(RuntimeError, match='bounded mixed-input mismatch'):
        module._run_generic_split_completion(config, output,
            lambda: submissions.append('timed'), lambda value: value, lambda value: {'logits': logits},
            {role: spec['path'] for role, spec in config['expected_runtime_artifacts'].items()},
            clone_payload=None, repetition_index=1, mixed_control=control)
    assert submissions == []
    result = json.loads((output / 'generic_completion_results.json').read_text())
    assert result['status'] == 'failed'
    assert result['repetitions'][0]['status'] == 'failed'
    assert 'completed_task_fps' not in result['repetitions'][0]


@pytest.mark.parametrize('failure', ['counter', 'async', 'stage1', 'stage2'])
def test_partial_or_unfinished_work_is_never_a_rate(failure):
    processor = ClassificationCompletion()
    def fail():
        raise RuntimeError('hardware failure fixture')
    if failure == 'counter':
        processor.process = lambda outputs: None
    handle = SimpleNamespace(result=lambda: None)
    p1 = fail if failure == 'stage1' else lambda: {'cut': np.ones((1, 6))}
    p2 = (lambda x: fail()) if failure == 'stage2' else lambda x: {'logits': handle if failure == 'async' else x['cut']}
    with pytest.raises(RuntimeError):
        measure_split_completion(p1, lambda x:x, p2, processor, warmup=1, tasks=2)


def test_progress_callback_time_is_outside_measured_makespan():
    ticks = 1
    phases = []
    clock_threads = []
    coordinator = threading.current_thread().name
    def clock():
        nonlocal ticks
        ticks += 10
        clock_threads.append(threading.current_thread().name)
        return ticks
    def progress(phase, count):
        nonlocal ticks
        ticks += 1_000_000
        phases.append((phase,count))
    values={'logits':np.ones((1,6))}
    report=measure_split_completion(lambda:values,lambda x:x,lambda x:x,
        ClassificationCompletion(),warmup=1,tasks=2,clock=clock,progress=progress)
    assert report['measurement_end_ns']-report['measurement_start_ns']==10
    assert phases==[('warmup_completed',1),('measured_started',0),('measured_completed',2)]
    assert clock_threads == ['generic-completion-p2', coordinator, 'generic-completion-p2']


def test_real_generated_main_uses_generic_pipeline_and_never_cpu_reference(tmp_path, monkeypatch):
    module = _module(monkeypatch)
    config, logits = _fixture(tmp_path)
    calls = {'p1': 0, 'p2': 0}
    def forbidden(*args, **kwargs):
        raise AssertionError('forbidden reference/build path')
    monkeypatch.setattr(module, '_create_session', forbidden)
    monkeypatch.setattr(module, '_bench', forbidden)
    monkeypatch.setattr(module, '_available_providers', lambda: ['CPUExecutionProvider'])
    monkeypatch.setattr(module, 'ort', SimpleNamespace(SessionOptions=lambda: SimpleNamespace()))
    class DeepX:
        def __init__(self, path, **kwargs):
            self.dxnn_path = path
        def describe_io(self):
            return {}
        def bind_prepared_inputs(self, feeds):
            self.exact_prepared_input = True
        def infer(self, feeds):
            assert self.exact_prepared_input is True
            assert list(feeds) == ['images']
            np.testing.assert_array_equal(feeds['images'], np.arange(12,dtype=np.uint8).reshape(2,2,3))
            calls['p1'] += 1
            return {'cut': logits}
    class TRT:
        def __init__(self, *args, **kwargs):
            assert kwargs['allow_build'] is False
            self.engine_path = Path(config['expected_runtime_artifacts']['engine']['path'])
            self.build_info = {'cache_hit': True, 'build_disabled': True}
            self.input_names = ['cut']
        def describe_io(self):
            return {'inputs': {'cut': {'shape': [1,6], 'dtype': 'float32'}}}
        def get_outputs(self):
            return [SimpleNamespace(name='logits')]
        def run(self, _, feeds):
            calls['p2'] += 1
            return [feeds['cut']]
    monkeypatch.setattr(module, 'DeepXSession', DeepX)
    monkeypatch.setattr(module, 'NativeTRTSession', TRT)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    output=tmp_path/'supplement'
    for index in (1, 2, 3):
        monkeypatch.setattr(module, '_generic_process_identity', lambda index=index: {
            'pid': 1000 + index, 'process_start_ticks': 2000 + index, 'boot_id': 'offline-fixture'})
        monkeypatch.setattr(sys, 'argv', ['runner', '--generic-completion-config', str(path), '--out-dir', str(output),
                                        '--generic-completion-repetition', str(index)])
        assert module.main() == 0
    result=json.loads((output/'generic_completion_results.json').read_text())
    assert calls == {'p1': 3300, 'p2': 3300}
    assert len(valid_split_completion_repetitions(result, config)) == 3
    assert all(r['functional_control']['checked_outside_timer'] for r in result['repetitions'])
    assert module.main() == 0
    assert calls == {'p1': 3300, 'p2': 3300}


def test_abort_preserves_valid_repetition_and_resumes_only_missing(tmp_path, monkeypatch):
    module=_module(monkeypatch);config,logits=_fixture(tmp_path)
    output=tmp_path/'output';output.mkdir()
    artifacts={k:Path(v['path']) for k,v in config['expected_runtime_artifacts'].items()}
    calls=0
    def p1():
        nonlocal calls
        calls+=1
        if calls==1150:
            raise RuntimeError('controlled abort fixture')
        return {'cut':logits}
    def p2(inputs):
        return {'logits':inputs['cut']}
    with pytest.raises(RuntimeError,match='controlled abort'):
        module._run_generic_split_completion(config,output,p1,lambda x:x,p2,artifacts,clone_payload=None)
    before=json.loads((output/'generic_completion_results.json').read_text())
    assert [r['repetition_index'] for r in valid_split_completion_repetitions(before,config)] == [1]
    saved=copy.deepcopy(before['repetitions'][0])
    old_calls=calls
    assert module._run_generic_split_completion(config,output,p1,lambda x:x,p2,artifacts,clone_payload=None)==0
    after=json.loads((output/'generic_completion_results.json').read_text())
    assert calls-old_calls==2200
    assert after['repetitions'][0]==saved
    assert len(after['repetitions'])==4
    assert len(valid_split_completion_repetitions(after,config))==3


@pytest.mark.parametrize('family', ['yolo11', 'yolo26', 'yolov7'])
def test_existing_detection_completion_variants_are_inside_timer(tmp_path, family):
    if family == 'yolo11':
        from test_v27923_decoded_completion import _outputs, _runtime
        outputs = _outputs();processor = _runtime(tmp_path, outputs)
    elif family == 'yolo26':
        from test_v272_detection_completion_runtime import _decoded_outputs, _runtime
        outputs = _decoded_outputs();processor = _runtime(outputs)
    else:
        from test_v272_yolov7_head_mapping import _heads_640, _raw_source
        from onnx_splitpoint_tool.native_detection_postprocess import build_detection_completion_runtime
        outputs = _heads_640()
        processor = build_detection_completion_runtime(model_id='yolov7_paper', outputs=outputs,
            input_hw=[640,640], original_wh=[640,480], source_endpoint_contract=_raw_source(outputs,endpoint_hash='a'*64))
    report = measure_split_completion(lambda:outputs,lambda x:x,lambda x:x,processor,warmup=1,tasks=2)
    assert report['processor_count_delta']==3
    assert processor.completed_count==3
    if family=='yolo26':
        assert processor.last_result['host_nms_applied'] is False
        assert processor.last_result['detection_count']==2
    else:
        assert processor.last_result['host_nms_applied'] is True


def test_detection_writer_materializes_and_preserves_last_timed_output(tmp_path, monkeypatch):
    module=_module(monkeypatch);config,_=_fixture(tmp_path)
    from test_v272_detection_completion_runtime import _decoded_outputs, _runtime
    outputs=_decoded_outputs();processor=_runtime(outputs)
    config['task']='detection';config['completion_contract']=processor.execution_contract
    config['functional_reference']['outputs']={}
    for name,value in outputs.items():
        path=tmp_path/(name+'.npy');np.save(path,value)
        config['functional_reference']['outputs'][name]=_descriptor(path,value)
    config['functional_reference']['tolerances']={'match_iou_threshold':.5,'min_match_ratio':.8,'min_mean_iou':.85}
    output=tmp_path/'output';output.mkdir()
    artifacts={k:Path(v['path']) for k,v in config['expected_runtime_artifacts'].items()}
    assert module._run_generic_split_completion(config,output,lambda:outputs,lambda x:x,lambda x:x,
        artifacts,clone_payload=None,repetition_index=1)==0
    result=json.loads((output/'generic_completion_results.json').read_text())
    assert len(valid_split_completion_repetitions(result,config,result_root=output))==1
    saved=result['repetitions'][0]
    assert saved['last_completed_result']['detection_count']==2
    assert saved['last_completed_result']['host_nms_applied'] is False
    assert saved['functional_control']['metrics']['match_ratio']==1
    raw=output/saved['raw_output_samples']['detections']['path']
    np.testing.assert_array_equal(np.load(raw),outputs['detections'])
    raw.write_bytes(b'changed existing measured output')
    with pytest.raises(ValueError,match='sample changed'):
        valid_split_completion_repetitions(result,config,result_root=output)


def test_bound_input_shape_and_content_are_checked_before_any_runtime(tmp_path,monkeypatch):
    module=_module(monkeypatch);config,_=_fixture(tmp_path)
    spec=config['input_binding']['prepared_input']
    wrong=copy.deepcopy(spec);wrong['shape']=[3,2,2]
    with pytest.raises(ValueError,match='shape/dtype'):
        module._completion_bound_array(wrong)
    Path(spec['path']).write_bytes(b'altered')
    with pytest.raises(ValueError,match='bound tensor changed'):
        module._completion_bound_array(spec)


@pytest.mark.parametrize('backend,runtime_format,expected_dtype', [
    ('hailo8','float32',np.float32), ('hailo8','uint8',np.uint8), ('hailo10h','uint8',np.uint8),
])
def test_hailo_uses_original_native_runtime_format_inside_p1(tmp_path,monkeypatch,backend,runtime_format,expected_dtype):
    module=_module(monkeypatch)
    import ast
    tree=ast.parse(TEMPLATE.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_stage1_map')
    raw=np.array([[[0,127,255]]],dtype=np.uint8)
    config={'task':'detection','input_binding':{'runtime_input_format':runtime_format}}
    observed=[]
    class Hailo:
        def infer(self, inputs):
            observed.append(inputs['input'].copy())
            return {'logits': np.ones((1,6))}
    ns=dict(module.__dict__)
    ns.update(hailo_p1=Hailo(),p1_inputs_hailo={'input':raw},generic_completion_config=config,stage1_tok=backend)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(TEMPLATE),'exec'),ns)
    # The real P1 closure performs the bound numeric adapter for every task.
    measure_split_completion(ns['run_stage1_map'],lambda x:x,lambda x:x,
        ClassificationCompletion(),warmup=1,tasks=2)
    assert len(observed)==3
    assert all(value.dtype==expected_dtype for value in observed)
    expected=raw.astype(np.float32)/np.float32(255) if runtime_format=='float32' else raw
    for value in observed: np.testing.assert_array_equal(value,expected)
    np.testing.assert_array_equal(raw,np.array([[[0,127,255]]],dtype=np.uint8))
    assert module._completion_hailo_quantized_inputs(config) is (runtime_format=='uint8')


def test_hailo_rejects_missing_bound_runtime_format_before_inference(monkeypatch):
    module=_module(monkeypatch)
    with pytest.raises(ValueError,match='bound Native runtime option'):
        module._completion_hailo_quantized_inputs({'input_binding':{}})


def test_new_process_evidence_rejects_same_pid_start_but_keeps_legacy_readable(tmp_path, monkeypatch):
    module = _module(monkeypatch); config, logits = _fixture(tmp_path)
    output = tmp_path / 'process_output'; output.mkdir()
    artifacts = {k: Path(v['path']) for k, v in config['expected_runtime_artifacts'].items()}
    module._run_generic_split_completion(config, output, lambda: {'cut': logits}, lambda x: x,
        lambda x: {'logits': x['cut']}, artifacts, clone_payload=None)
    result = json.loads((output / 'generic_completion_results.json').read_text())
    assert len(valid_split_completion_repetitions(result, config)) == 3  # legacy parser
    result['process_repetition_policy'] = 'separate_processes'
    for row in result['repetitions']:
        row.update(pid=100, process_start_ticks=200, boot_id='fixture')
    with pytest.raises(ValueError, match='share a process'):
        valid_split_completion_repetitions(result, config)
    for index, row in enumerate(result['repetitions']):
        row['pid'] += index
    assert len(valid_split_completion_repetitions(result, config)) == 3
    result['repetitions'][0]['process_start_ticks'] = None
    with pytest.raises(ValueError, match='start identity missing'):
        valid_split_completion_repetitions(result, config)


@pytest.mark.parametrize('task,target', [('detection', (32,32)), ('classification', (32,32))])
def test_shared_raster_loader_asymmetric_pixels_and_actual_hashes(tmp_path, monkeypatch, task, target):
    from PIL import Image
    from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract, prepare_rgb_uint8_image
    module = _module(monkeypatch)
    pixels = np.arange(7*11*3, dtype=np.uint8).reshape(7,11,3)
    path = tmp_path / 'asymmetric.png'; Image.fromarray(pixels).save(path)
    contract = canonical_image_preprocessing_contract(task, target)
    expected, geometry = prepare_rgb_uint8_image(pixels, contract)
    binding = {}
    feed = module._load_image_as_nchw(path, target_hw=target, dtype=np.float32,
        scale=contract['image_scale'], preprocessing_contract=contract, binding_out=binding)
    assert binding['canonical_rgb_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert binding['input_image_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding['geometry'] == geometry
    if task == 'detection':
        np.testing.assert_array_equal(feed, expected.transpose(2,0,1)[None].astype(np.float32)/255)
    assert binding['canonical_preprocessing_executed'] is True
