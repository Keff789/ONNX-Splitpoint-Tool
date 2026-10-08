"""Reusable real Suite/Runner/aggregate-writer fixture; SDK tensors are synthetic."""
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
from tests.test_v269e_quality_export_vendored import _load_self_contained_runner


def model(path, input_name, output_name, output_shape):
    # Only metadata is consumed. Every executable session below is a fake
    # inference leaf, and creating/executing Full is an assertion failure.
    graph = helper.make_graph([], path.stem,
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)])
    onnx.save(helper.make_model(graph), path)


from onnx_splitpoint_tool.native_split_quality import seal_native_split_quality_binding, canonical_json_sha256, materialize_native_split_preselection, known_native_split_policy
from tests.test_yolo_augmentation_quality_release import binding as make_binding, write, install_original_reference
from tests.test_v269f_native_split_receipt_validation import _artifact
import copy
import shutil


def write_input_consuming_fixture_part2(path, input_name, input_shape, outputs):
    """A real ONNX dependency with the SDK fixture's unchanged output tensors."""
    nodes = [helper.make_node('ReduceSum', [input_name], ['fixture_input_sum'], keepdims=0),
             helper.make_node('Mul', ['fixture_input_sum', 'fixture_zero'], ['fixture_input_zero'])]
    initializers = [onnx.numpy_helper.from_array(np.asarray(0., np.float32), 'fixture_zero')]
    for index, (name, value) in enumerate(outputs.items()):
        constant = f'fixture_output_{index}'
        initializers.append(onnx.numpy_helper.from_array(np.asarray(value, np.float32), constant))
        nodes.append(helper.make_node('Add', [constant, 'fixture_input_zero'], [name]))
    graph = helper.make_graph(nodes, Path(path).stem,
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
         for name, value in outputs.items()], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def fixture_part1_export_metadata(manifest, case, backend, runtime_artifact):
    """Project only the existing SDK fixture artifact into normal case metadata."""
    result = copy.deepcopy(manifest)
    path = Path(runtime_artifact['path'])
    actual = _artifact(path)
    assert actual['sha256'] == runtime_artifact['sha256']
    relative = os.path.relpath(path, case).replace(os.sep, '/')
    if backend == 'deepx_to_trt':
        result['deepx_part1_dxnn'] = relative
        result.setdefault('deepx', {})['part1_dxnn'] = {
            'dxnn_path': relative, 'dxnn_sha256': actual['sha256'],
            'dxnn_size_bytes': path.stat().st_size}
    else:
        assert backend in {'hailo8_to_trt', 'hailo10h_to_trt'}
        arch = 'hailo8' if backend == 'hailo8_to_trt' else 'hailo10'
        result.setdefault('hailo', {}).setdefault('hefs', {}).setdefault(arch, {})['part1'] = relative
    return result


def reseal_fixture_binding(bound, source_p2, outputs):
    payload=copy.deepcopy(bound)
    paths={k:Path(v['path']) for k,v in payload['artifacts'].items()}
    shutil.copyfile(source_p2,paths['source_part2_onnx'])
    bridge_meta = None
    if payload['preselection']['backend'] in {'hailo8_to_trt', 'hailo10h_to_trt'}:
        from onnx_splitpoint_tool.resources.remote_scripts.native_trt_from_benchmarkset import _make_uint8_dequant_bridge_onnx
        boundary = payload['boundary_contract']
        bridge_path, bridge_meta = _make_uint8_dequant_bridge_onnx(
            paths['source_part2_onnx'], paths['build_part2_onnx'].parent / 'normal_bridge_writer',
            root=paths['source_part2_onnx'].parent, case=None,
            scale=float(boundary['dequant_scale']), zero_point=float(boundary['dequant_zero_point']),
            boundary_layout=boundary['boundary_layout'])
        paths['build_part2_onnx'] = bridge_path
    else:
        shutil.copyfile(source_p2,paths['build_part2_onnx'])
    receipt=json.loads(paths['engine_build_receipt'].read_text())
    receipt['source_onnx'] = str(paths['build_part2_onnx'])
    receipt['command'] = [('--onnx=' + str(paths['build_part2_onnx']))
                          if value.startswith('--onnx=') else value for value in receipt['command']]
    receipt['source_onnx_sha256']=_artifact(paths['build_part2_onnx'])['sha256']
    if '--memPoolSize=workspace:4096' not in receipt['command']:
        receipt['command'].append('--memPoolSize=workspace:4096')
    receipt.pop('receipt_sha256',None)
    receipt['receipt_sha256']=canonical_json_sha256(receipt)
    write(paths['engine_build_receipt'],receipt)
    meta=payload['native_trt_meta']
    if bridge_meta is not None:
        meta['uint8_cast_bridge'] = bridge_meta
        meta['onnx'] = meta['source_onnx'] = str(paths['build_part2_onnx'])
    meta['engine_build_receipt']=receipt
    meta['build']['cmd']=list(receipt['command'])
    meta['uint8_cast_bridge']['source_sha256']=_artifact(paths['source_part2_onnx'])['sha256']
    meta['uint8_cast_bridge']['bridge_sha256']=_artifact(paths['build_part2_onnx'])['sha256']
    meta['outputs']=[{'name':name,'shape':list(arr.shape),'elem_type':'FLOAT','has_dynamic':False} for name,arr in outputs.items()]
    write(paths['native_trt_meta'],meta)
    payload.update(engine_build_receipt=receipt,native_trt_meta=meta,native_trt_meta_sha256=canonical_json_sha256(meta),
        artifacts={name:_artifact(path) for name,path in paths.items()})
    return seal_native_split_quality_binding(payload)


def make_generic_fixture(tmp_path, monkeypatch, row, artifact_inputs=None):
    artifact_inputs=artifact_inputs or {}
    workflow_runner=artifact_inputs.get('workflow_runner')
    model_id,case_id,setup_id,backend=[row[k] for k in ('model_id','case_id','setup_id','direction')]
    eval_run_id=workflow_runner.run_id if workflow_runner is not None else artifact_inputs['eval_run_id']
    stage1_backend=backend.removesuffix('_to_trt').replace('deepx','deepx')
    bound=make_binding(tmp_path/'bound',model_id,case_id,setup_id,backend,eval_run_id)
    if backend=='hailo8_to_trt':
        metadata_path=Path(bound['artifacts']['boundary_metadata']['path'])
        metadata=json.loads(metadata_path.read_text())
        metadata['boundary_tensor'].update(shape=[2,3,4],canonical_part2_shape=[1,4,2,3])
        metadata.pop('metadata_sha256',None)
        metadata['metadata_sha256']=canonical_json_sha256(metadata)
        write(metadata_path,metadata)
        policy=known_native_split_policy(model_id=model_id,case_id=case_id,setup_id=setup_id,backend=backend)
        selected=materialize_native_split_preselection(policy=policy,part1_artifact=bound['artifacts']['part1_runtime'],
            boundary_metadata=metadata,boundary_metadata_artifact=_artifact(metadata_path))
        bound['preselection']=selected;bound['preselection_sha256']=selected['selection_sha256']
        bound['boundary_contract']={name:selected[name] for name in bound['boundary_contract'] if name in selected}
        bound['boundary_contract_sha256']=canonical_json_sha256(bound['boundary_contract'])
        bound['native_trt_meta']['inputs'][0]['shape']=[1,4,2,3]
        bridge=bound['native_trt_meta']['uint8_cast_bridge']
        bridge['input_shape']=[1,4,2,3]
        bridge['boundary_layout'].update(memory_shape=[1,2,3,4],perm=[0,3,1,2])
    target_shape=tuple(bound['native_trt_meta']['inputs'][0]['shape'])
    boundary_shape=tuple(bound['preselection']['boundary_tensor_shape'])
    boundary_dtype=np.dtype(bound['preselection']['boundary_tensor_dtype'])
    output_arrays=artifact_inputs.get('outputs') or {'output0':np.zeros((1,84,100),dtype=np.float32)}
    if model_id!='yolo11l' and not artifact_inputs.get('outputs'):
        raise ValueError('Native-matching SDK outputs are required for this model fixture')
    endpoint_mode=artifact_inputs.get('endpoint_mode','decoded_pre_nms')
    input_hw=tuple(artifact_inputs.get('input_hw',(64,64)))
    original_wh=tuple(artifact_inputs.get('original_wh',(80,40)))
    signatures=[(name,list(arr.shape)) for name,arr in output_arrays.items()]
    runner, case, full, _ = _load_self_contained_runner(
        tmp_path, model_id=model_id, endpoint_mode=endpoint_mode, outputs=signatures)
    if case.name!=case_id:
        case=case.rename(case.parent/case_id)
    runner_path = write_runner_skeleton_onnxruntime(str(case), target='cpu')
    runner_spec = importlib.util.spec_from_file_location('augmentation_actual_generated_runner', runner_path)
    runner = importlib.util.module_from_spec(runner_spec)
    monkeypatch.setitem(sys.modules, runner_spec.name, runner)
    runner_spec.loader.exec_module(runner)
    p1, p2 = case / 'part1.onnx', case / 'part2.onnx'
    def signature(path,input_name,input_shape,outputs,input_type=TensorProto.FLOAT):
        graph=helper.make_graph([],path.stem,[helper.make_tensor_value_info(input_name,input_type,input_shape)],
            [helper.make_tensor_value_info(n,TensorProto.FLOAT,list(a.shape)) for n,a in outputs.items()])
        onnx.save(helper.make_model(graph),path)
    exact_full=artifact_inputs.get('full_model')
    if model_id=='yolov7_paper' and not exact_full:
        exact_full=str(Path(os.environ.get('THESIS20_FIXTURE_HOME', Path.home())) / 'Models/EvaluationRuns/thesis_20splits_n5000_b1000_20260925_20260925_103745/models/yolov7_paper/benchmark_set/legacy_suite/models/yolov7_paper.onnx')
    if exact_full:
        shutil.copyfile(exact_full,full)
    else:
        signature(full,'images',[1,3,*input_hw],output_arrays)
    signature(p1,'images',[1,3,*input_hw],{'cut':np.zeros(target_shape,np.float32)})
    if backend in {'hailo8_to_trt', 'hailo10h_to_trt'}:
        write_input_consuming_fixture_part2(p2, 'cut', target_shape, output_arrays)
    else:
        signature(p2,'cut',list(bound['native_trt_meta']['inputs'][0]['shape']),output_arrays,
            TensorProto.UINT8 if boundary_dtype==np.uint8 else TensorProto.FLOAT)
    bound=reseal_fixture_binding(bound,p2,output_arrays)
    binding_path=tmp_path/'native_split_quality_binding.json';write(binding_path,bound)
    plan={'legacy_suite_dir':str(case.parent),'model_suite':{'primary':[{'id':model_id,'path':str(full)}]}}
    write(case.parent/'benchmark_plan.json',plan)
    (case / 'split_manifest.json').write_text(json.dumps({
        'model_id':model_id, 'full_model':str(full), 'part1_model':str(p1), 'part2_model':str(p2),
        'part1':'part1.onnx','part2':'part2.onnx','orig_inputs':['images'],
        'part1_cut_names':['cut'], 'part2_cut_names':['cut'],
    }))
    contracts_path = case.parent / 'output_contracts.json'
    contracts = json.loads(contracts_path.read_text())
    contracts['contracts'][0]['backend'] = 'tensorrt'
    contracts_path.write_text(json.dumps(contracts))
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    image = dataset / '00000.png'; Image.new('RGB', original_wh, (42, 96, 128)).save(image)
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
        image_scale='norm', letterbox=True, input_hw=input_hw, input_dtype=np.float32,
        output_format=runner._detect_output_format(list(output_arrays),list(output_arrays.values())), output_names=list(output_arrays),
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
    if workflow_runner is not None:
        descriptor=install_original_reference(workflow_runner,model_id,reference,encoded)
        reference_path=Path(descriptor['reference_path'])
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
                return [np.zeros(target_shape,dtype=np.float32)]
            return list(output_arrays.values())
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
            self.engine_path = Path(bound['artifacts']['engine']['path'])
            self.receipt_path=Path(bound['artifacts']['engine_build_receipt']['path'])
            self.build_info = {'build_ok':True, 'build_elapsed_s':0., 'precision':row['precision'],
                'runtime':'native_tensorrt','engine':str(self.engine_path),'engine_sha256':bound['artifacts']['engine']['sha256'],
                'source_model':bound['artifacts']['build_part2_onnx']['path'],
                'source_model_sha256':bound['artifacts']['build_part2_onnx']['sha256'],'fallback_disabled':True}
            self.input_names = [item.name for item in self.get_inputs()]
        def describe_io(self):
            return {'inputs':{item.name:{'shape':list(item.shape),'dtype':str(boundary_dtype)} for item in self.get_inputs()},
                    'outputs':{item.name:{'shape':list(item.shape),'dtype':'float32'} for item in self.get_outputs()}}
        def close(self): pass
    monkeypatch.setattr(runner, 'NativeTRTSession', NativeSession)
    if stage1_backend in ('hailo8','hailo10h'):
        hef = case / f'hailo/{stage1_backend}/part1/compiled.hef'; hef.parent.mkdir(parents=True)
        shutil.copyfile(bound['artifacts']['part1_runtime']['path'],hef)
        class HailoSession:
            def __init__(self, path, **kwargs):
                self.hef_path = path; self.input_names = ['images']
                self.input_shapes = self.runtime_input_shapes = {'images':(*input_hw,3)}
                self.quantized_inputs = kwargs['quantized_inputs']
            def infer(self, feeds):
                calls['part1'] += 1
                arr = np.asarray(feeds['images'])
                if arr.dtype == np.uint8: arr = arr.astype(np.float32) / 255.
                return {'cut':np.zeros(target_shape,dtype=boundary_dtype)}
            def describe_io(self): return {}
            def close(self): pass
        class Device:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def release(self): pass
        monkeypatch.setattr(runner, 'HailoSession', HailoSession)
        monkeypatch.setattr(runner, 'HailoInferModelSession', HailoSession)
        monkeypatch.setitem(sys.modules, 'hailo_platform', SimpleNamespace(
            Device=SimpleNamespace(scan=lambda:['fixture-hailo']), VDevice=Device))
        original_glob = runner.glob.glob
        monkeypatch.setattr(runner.glob, 'glob', lambda pattern,*a,**k:
                            ['/dev/fixture-hailo'] if pattern in ('/dev/hailo*','/dev/h1x-*')
                            else original_glob(pattern,*a,**k))
    elif stage1_backend == 'deepx':
        dxnn = case / 'deepx/deepx_m1/part1/model.dxnn'
        dxnn.parent.mkdir(parents=True)
        shutil.copyfile(bound['artifacts']['part1_runtime']['path'],dxnn)
        deepx_adapter = runner.DeepXSession
        class DeepXSession:
            _candidate_inputs = deepx_adapter._candidate_inputs
            prepare_image_inputs = deepx_adapter.prepare_image_inputs
            bind_prepared_inputs = deepx_adapter.bind_prepared_inputs
            def __init__(self, path, **kwargs):
                self.dxnn_path = path
                self.input_names = ['images']
            def infer(self, feeds):
                calls['part1'] += 1
                if not original_feeds: original_feeds.append(feeds['images'].copy())
                return {'cut':np.zeros(target_shape,dtype=boundary_dtype)}
            def describe_io(self):
                return {'canonical_inputs':['images'], 'canonical_outputs':['cut'],
                        'runtime_input_shapes':{'images':(1,3,*input_hw)}}
            def close(self): pass
        monkeypatch.setattr(runner, 'DeepXSession', DeepXSession)
    monkeypatch.setattr(runner, 'ort', SimpleNamespace(SessionOptions=lambda:SimpleNamespace(),
                                                    get_available_providers=lambda:['CPUExecutionProvider']))
    manifest_path = case / 'split_manifest.json'
    staged_part1 = hef if stage1_backend in ('hailo8', 'hailo10h') else dxnn
    part1_artifact = {**bound['artifacts']['part1_runtime'], 'path': str(staged_part1)}
    write(manifest_path, fixture_part1_export_metadata(
        json.loads(manifest_path.read_text()), case, backend, part1_artifact))
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
    from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
    resolved,_=apply_run_mode({},mode_id='final',config=default_run_modes_config())
    policy=copy.deepcopy(artifact_inputs.get('quality_policy') or resolved['quality_gate'])
    policy['statistics'].update(execution_location='central_management',bootstrap_repetitions=1000,seed=20260710,confidence_level=.95)
    expected_policy = runner._load_task_quality_policy(policy)
    run_id = backend
    class SetupAdmitted(BaseException):
        pass
    if artifact_inputs.get('setup_admission_only'):
        def stop_before_dataset(*args,**kwargs):
            import inspect
            state=inspect.currentframe().f_back.f_locals
            assert state['variant_status']['composed']=='ok', state.get('variant_errors')
            assert calls['full']==0 and calls['part1']>0 and calls['part2']>0
            raise SetupAdmitted()
        monkeypatch.setattr(runner,'_collect_validation_image_paths',stop_before_dataset)
    try:
        result = suite._run_case(case, run_id=run_id, provider='cpu', image=str(image),
            quality_source_only=bool(artifact_inputs.get('quality_source_only')),
            preset='balanced', image_scale='auto', warmup=0, runs=1, timeout_s=30,
            stage1=stage1_backend, stage2='tensorrt', variants=['composed'], phase_runs=0,
            throughput_frames=0, throughput_warmup_frames=0,
            validation_images=str(dataset), validation_max_images=5000, benchmark_task='detection',
            task_quality_gate=policy, management_cpu_reference=descriptor,
            quality_evidence_eval_id=eval_run_id, quality_evidence_model_id=model_id,
            quality_evidence_setup_id=setup_id, quality_evidence_source_run_id=run_id,
            native_trt_precision=row['precision'], native_trt_build=False, native_trt_workspace_mb=0,
            native_split_quality_binding=str(binding_path))
    except SetupAdmitted:
        return {'setup_admission':'passed','identity':dict(row),'calls':dict(calls),
                'stopped_before_validation_inference':True,'full_path':full,'binding':bound}
    assert len(invocations) == 1
    command = invocations[0]
    assert json.loads(command[command.index('--existing-management-cpu-reference-json')+1]) == descriptor
    assert calls['full'] == 0 and calls['part1'] >= 5000 and calls['part2'] >= 5000
    preprocessing, _ = runner._resolve_image_preprocessing_contract(task='detection', target_hw=input_hw)
    expected_feed = runner._load_image_as_nchw(image,target_hw=input_hw,dtype=np.float32,
                                             scale='norm',letterbox=True,preprocessing_contract=preprocessing)
    assert calls['full']==0
    request_path = case / f'results_{run_id}/task_quality_inputs/composed_request.json'
    request = json.loads(request_path.read_text())
    assert request['record_count'] == 5000 and request['eval_run_id'] == eval_run_id
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
    aggregate=case.parent/f'benchmark_results_{backend}_auto.json'
    rows=[result]
    suite._augment_result_rows_throughput_semantics(rows)
    suite._write_json(aggregate,rows)
    if workflow_runner is not None:
        from onnx_splitpoint_tool.workflow.execution_binding import _copy_remote_result_files
        remote=tmp_path/'downloaded_attempt'
        remote_results=remote/'results';remote_results.mkdir(parents=True)
        shutil.copy2(aggregate,remote_results/aggregate.name)
        shutil.copytree(case/f'results_{run_id}',remote_results/case_id/f'results_{run_id}')
        destination=Path(artifact_inputs.get('result_import_dir') or (
            workflow_runner.run_dir/'models'/model_id/'benchmark_results'))
        _copy_remote_result_files(remote,destination,flat_prefix=setup_id)
        aggregate=destination/aggregate.name
        request_path=destination/'quality_inputs'/setup_id/'results'/case_id/f'results_{run_id}'/'task_quality_inputs/composed_request.json'
        assert request_path.is_file()
    normalized,_=normalize_benchmark_files(model_id=model_id,source_paths=[aggregate])
    assert len(normalized)==1
    return dict(raw_report_path=case/f'results_{run_id}/validation_report.json',raw_report=report,
        aggregate_path=aggregate,raw_row=rows[0],normalized_row=normalized[0],
        composed_request_path=request_path,request_bytes=request_path.read_bytes(),request=request,
        generated_runner=runner_path,runner_module=runner,full_path=full,case=case,
        binding=bound,binding_path=binding_path,reference=reference,reference_path=reference_path,
        reference_bytes=encoded,descriptor=descriptor,quality=quality,workflow_runner=workflow_runner,
        suite_module=suite,calls=calls)
