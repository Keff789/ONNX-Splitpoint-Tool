"""Normal generated Suite -> runner -> central candidate, without inference."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.quality_service import quality_request_from_manifest
from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gates
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
from test_v269e_quality_export_vendored import _load_self_contained_runner


def model(path, input_name, output_name, output_shape):
    # Only metadata is consumed. Every executable session below is a fake
    # inference leaf, and creating/executing Full is an assertion failure.
    graph = helper.make_graph([], path.stem,
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)])
    onnx.save(helper.make_model(graph), path)


@pytest.mark.parametrize('stage1_backend', ['cpu', 'hailo8', 'deepx'])
def test_generated_suite_enters_current_runner_and_exports_n5000_without_cpu_full(tmp_path, monkeypatch, stage1_backend):
    runner, case, full, _ = _load_self_contained_runner(
        tmp_path, model_id='yolo11l', endpoint_mode='decoded_pre_nms', outputs=[('output0', [1, 84, 100])])
    runner_path = write_runner_skeleton_onnxruntime(str(case), target='cpu')
    runner_spec = importlib.util.spec_from_file_location('augmentation_actual_generated_runner', runner_path)
    runner = importlib.util.module_from_spec(runner_spec)
    monkeypatch.setitem(sys.modules, runner_spec.name, runner)
    runner_spec.loader.exec_module(runner)
    p1, p2 = case / 'part1.onnx', case / 'part2.onnx'
    model(full, 'images', 'output0', [1, 84, 100])
    model(p1, 'images', 'boundary', [1, 3, 64, 64])
    model(p2, 'boundary', 'output0', [1, 84, 100])
    (case / 'split_manifest.json').write_text(json.dumps({
        'model_id':'yolo11l', 'full_model':str(full), 'part1_model':str(p1), 'part2_model':str(p2),
        'part1_cut_names':['boundary'], 'part2_cut_names':['boundary'],
    }))
    contracts_path = case.parent / 'output_contracts.json'
    contracts = json.loads(contracts_path.read_text())
    contracts['contracts'][0]['backend'] = 'tensorrt'
    contracts_path.write_text(json.dumps(contracts))
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    image = dataset / '00000.png'; Image.new('RGB', (80, 40), (42, 96, 128)).save(image)
    annotation = dataset / '00000.json'; annotation.write_text('[]')
    for index in range(1, 5000):
        os.link(image, dataset / f'{index:05}.png')
        os.link(annotation, dataset / f'{index:05}.json')
    ids = [f'{index:05}.png' for index in range(5000)]
    (dataset / 'manifest.json').write_text(json.dumps({
        'schema':'onnx-splitpoint/detection-validation-manifest', 'schema_version':2,
        'samples':[{'image':name, 'image_id':index, 'sample_id':str(index),
                    'source_size_bytes':image.stat().st_size} for index,name in enumerate(ids)],
        'filtered_coco_annotations':'instances_subset.json'}))
    (dataset / 'instances_subset.json').write_text(json.dumps({
        'images':[{'id':index,'file_name':name} for index,name in enumerate(ids)],
        'annotations':[], 'categories':[]}))
    old_runner = tmp_path / 'original_runner.py'; old_runner.write_text('original closed reference writer\n')
    quality = runner._build_detection_quality_contract(
        model_path=full, validation_source=dataset, image_ids=ids,
        ground_truth_by_image={image_id:[] for image_id in ids},
        image_scale='norm', letterbox=True, input_hw=(64, 64), input_dtype=np.float32,
        output_format='ultralytics_decoded', output_names=['output0'],
        yolo_conf=.25, yolo_iou=.45, yolo_max_det=300, det_conf=.25, det_iou=.45, det_max_det=300,
        labels=None, runner_path=old_runner, endpoint_attestor_sha256=runner._endpoint_attestor_sha256,
    )
    reference = {'schema':'onnx-splitpoint/task-quality-reference-input','schema_version':1,
        'task':'detection','pairing_key':'image_id','reference_role':'canonical_cpu_ort',
        'semantic_reference_only':True,'provenance_required':True,
        'quality_contract':quality,'quality_contract_sha256':quality['quality_contract_sha256'],
        'records':[{'image_id':key,'reference':[],'ground_truth':[]} for key in ids]}
    encoded = json.dumps(reference,sort_keys=True).encode()
    reference_path = tmp_path / 'original_reference.json'; reference_path.write_bytes(encoded)
    descriptor = {'reference_role':'canonical_cpu_ort','quality_contract':quality,
        'quality_contract_sha256':quality['quality_contract_sha256'],
        'reference_path':str(reference_path),'reference_sha256':hashlib.sha256(encoded).hexdigest(),
        'reference_size_bytes':len(encoded),'source_evaluation_run_id':'closed-original'}
    calls = {'part1':0, 'part2':0, 'full':0}
    original_feeds = []
    class Session(runner._CompletionMetadataSession):
        def __init__(self, kind, path):
            super().__init__(path); self.kind = kind
        def run(self, names, feeds):
            calls[self.kind] += 1
            if self.kind == 'full': raise AssertionError('Full CPU inference is forbidden')
            if self.kind == 'part1':
                if not original_feeds: original_feeds.append(feeds['images'].copy())
                return [np.ascontiguousarray(feeds['images'])]
            return [np.zeros((1, 84, 100), dtype=np.float32)]
        def get_providers(self): return ['CPUExecutionProvider']
    def create(kind, path, providers, provider_options, options):
        assert kind != 'full', 'Full CPU session creation is forbidden'
        return Session(kind, path), runner.SessionBuildInfo(
            name=kind, model_path=str(path), build_seconds=0.,
            providers_requested=list(providers), providers_in_use=list(providers))
    monkeypatch.setattr(runner, '_create_session', create)
    class NativeSession(Session):
        def __init__(self, kind, path, **kwargs):
            assert kind == 'part2', 'Full Native inference is forbidden'
            super().__init__(kind, path)
            self.engine_path = tmp_path / 'fake-part2.engine'
            self.engine_path.write_bytes(b'fake TensorRT inference leaf, never executed')
            self.build_info = {'build_ok':True, 'build_elapsed_s':0., 'precision':'fp16'}
            self.input_names = [item.name for item in self.get_inputs()]
        def describe_io(self):
            return {'inputs':{item.name:{'shape':list(item.shape),'dtype':'float32'} for item in self.get_inputs()},
                    'outputs':{item.name:{'shape':list(item.shape),'dtype':'float32'} for item in self.get_outputs()}}
        def close(self): pass
    monkeypatch.setattr(runner, 'NativeTRTSession', NativeSession)
    if stage1_backend == 'hailo8':
        hef = case / 'hailo/hailo8/part1/compiled.hef'; hef.parent.mkdir(parents=True)
        hef.write_bytes(b'fake Hailo inference leaf, never executed')
        class HailoSession:
            def __init__(self, path, **kwargs):
                self.hef_path = path; self.input_names = ['images']
                self.input_shapes = self.runtime_input_shapes = {'images':(64,64,3)}
                self.quantized_inputs = kwargs['quantized_inputs']
            def infer(self, feeds):
                calls['part1'] += 1
                arr = np.asarray(feeds['images'])
                if arr.dtype == np.uint8: arr = arr.astype(np.float32) / 255.
                return {'boundary':np.ascontiguousarray(arr.transpose(2,0,1)[None])}
            def describe_io(self): return {}
            def close(self): pass
        class Device:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def release(self): pass
        monkeypatch.setattr(runner, 'HailoSession', HailoSession)
        monkeypatch.setitem(sys.modules, 'hailo_platform', SimpleNamespace(
            Device=SimpleNamespace(scan=lambda:['fixture-hailo']), VDevice=Device))
        original_glob = runner.glob.glob
        monkeypatch.setattr(runner.glob, 'glob', lambda pattern,*a,**k:
                            ['/dev/fixture-hailo'] if pattern in ('/dev/hailo*','/dev/h1x-*')
                            else original_glob(pattern,*a,**k))
    elif stage1_backend == 'deepx':
        dxnn = case / 'deepx/deepx_m1/part1/model.dxnn'
        dxnn.parent.mkdir(parents=True)
        dxnn.write_bytes(b'fake DeepX inference leaf, never executed')
        class DeepXSession:
            def __init__(self, path, **kwargs):
                self.dxnn_path = path
                self.input_names = ['images']
            def infer(self, feeds):
                calls['part1'] += 1
                if not original_feeds: original_feeds.append(feeds['images'].copy())
                return {'boundary':np.ascontiguousarray(feeds['images'])}
            def describe_io(self):
                return {'canonical_inputs':['images'], 'canonical_outputs':['boundary'],
                        'runtime_input_shapes':{'images':(1,3,64,64)}}
            def close(self): pass
        monkeypatch.setattr(runner, 'DeepXSession', DeepXSession)
    monkeypatch.setattr(runner, 'ort', SimpleNamespace(SessionOptions=lambda:SimpleNamespace(),
                                                    get_available_providers=lambda:['CPUExecutionProvider']))
    suite_path = Path(write_benchmark_suite_script(case.parent))
    spec = importlib.util.spec_from_file_location('augmentation_generated_suite_integration', suite_path)
    suite = importlib.util.module_from_spec(spec); spec.loader.exec_module(suite)
    monkeypatch.setattr(suite, '_provider_unavailable_reason', lambda *a,**k: None)
    invocations = []
    def child(command, **kwargs):
        if isinstance(command,list) and len(command)>2 and command[1] == '-c' and 'vdevice' in command[2]:
            return SimpleNamespace(returncode=0,stdout='fixture vdevice probe only',stderr='')
        assert len(command) > 1 and Path(command[1]) == case / 'run_split_onnxruntime.py', command
        invocations.append(list(command))
        with monkeypatch.context() as child_patch:
            child_patch.chdir(case)
            child_patch.setattr(sys, 'argv', command[1:])
            return SimpleNamespace(returncode=runner.main() or 0)
    monkeypatch.setattr(suite.subprocess, 'run', child)
    policy = {'statistics':{'execution_location':'central_management','bootstrap_repetitions':1000,
                            'seed':20260710,'confidence_level':.95}}
    expected_policy = runner._load_task_quality_policy(policy)
    run_id = stage1_backend + '_to_trt'
    result = suite._run_case(case, run_id=run_id, provider='cpu', image=str(image),
        preset='balanced', image_scale='auto', warmup=0, runs=1, timeout_s=30,
        stage1=stage1_backend, stage2='tensorrt', variants=['composed'], phase_runs=0,
        throughput_frames=0, throughput_warmup_frames=0,
        validation_images=str(dataset), validation_max_images=5000, benchmark_task='detection',
        task_quality_gate=policy, management_cpu_reference=descriptor,
        quality_evidence_eval_id='new-augmentation', quality_evidence_model_id='yolo11l',
        quality_evidence_setup_id='fixture-setup', quality_evidence_source_run_id=run_id)
    assert len(invocations) == 1
    command = invocations[0]
    assert json.loads(command[command.index('--existing-management-cpu-reference-json')+1]) == descriptor
    assert calls['full'] == 0 and calls['part1'] >= 5000 and calls['part2'] >= 5000
    preprocessing, _ = runner._resolve_image_preprocessing_contract(task='detection', target_hw=(64,64))
    expected_feed = runner._load_image_as_nchw(image,target_hw=(64,64),dtype=np.float32,
                                             scale='norm',letterbox=True,preprocessing_contract=preprocessing)
    assert np.array_equal(original_feeds[0], expected_feed)
    request_path = case / f'results_{run_id}/task_quality_inputs/composed_request.json'
    request = json.loads(request_path.read_text())
    assert request['record_count'] == 5000 and request['eval_run_id'] == 'new-augmentation'
    assert request['reference']['quality_contract_sha256'] == quality['quality_contract_sha256']
    assert request['quality_contract_sha256'] != quality['quality_contract_sha256']
    assert request['policy_sha256'] == expected_policy['policy_sha256']
    loaded = quality_request_from_manifest(request_path, reference_artifact={
        'path':str(reference_path),'size_bytes':len(encoded),'sha256':descriptor['reference_sha256']},
        reference_artifact_bytes=encoded)
    assert len(loaded.reference_records) == 5000
    report = json.loads((case / f'results_{run_id}/validation_report.json').read_text())
    assert report['comparisons']['composed']['elementwise'] is None
    assert runner._load_task_quality_policy(report['task_quality_policy'])['policy_sha256'] == expected_policy['policy_sha256']
    assert request['producer_identity']['implementation_runner_sha256'] == hashlib.sha256(Path(runner_path).read_bytes()).hexdigest()
    assert request['producer_identity']['stage1_backend'] == stage1_backend
    assert reference_path.read_bytes() == encoded
    assert result is not None
    if stage1_backend == 'hailo8':
        assert report['interface_checks']['checks']['stage1_vs_ort']['pass'] is True
    # Exercise normal report ingestion before applying the actual gate. The raw
    # report intentionally lacks legacy flattened shape/interface fields.
    normalized, _ = normalize_benchmark_files(
        model_id='yolo11l', source_paths=[case / f'results_{run_id}/validation_report.json'])
    assert len(normalized) == 1 and normalized[0]['variant'] == 'split'
    gated = apply_accuracy_gates(normalized[0], mutate=False)
    assert gated['contract_consistent'] is True
    assert gated['accuracy_gate_pass'] is False
    assert gated['accuracy_gate_reason'] == 'pending_central_evaluation'
    if stage1_backend == 'hailo8':
        assert normalized[0]['interface_stage1_structural_pass'] is True
        assert normalized[0]['interface_contract_pass'] is True
        assert gated['contract_gate_reason'] == 'explicit_structural_contract_field'
    else:
        assert gated['contract_gate_reason'] == 'endpoint_contract_complete'
