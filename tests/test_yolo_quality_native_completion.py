"""New identities through real Native hotloops and stored-output validators.

All SDK calls return synthetic tensors. No accelerator, compiler or model
session is opened; measured durations are software-fixture observations only.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import native_hailo_trt_fifo_from_benchmarkset as h8
from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10
from scripts import native_deepx_trt_e2e_from_benchmarkset as dx
from scripts import native_producer_final_report as final
from scripts import native_producer_validate_visualize as validator
from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract, verify_native_command_contract
from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime, canonical_json_sha256,
    persist_detection_completion_execution_artifacts)
from onnx_splitpoint_tool.native_split_quality import bind_quality_to_native_split, native_split_quality_selection_duplicates
from onnx_splitpoint_tool.trt_quality_chain import split_binding_set_from_central_quality_summary
from tests.test_v272_detection_hotloop_integration import _FakeHailo10Session, _Buffer
from tests.test_v269f_native_split_receipt_validation import _artifact
from tests.test_yolo_augmentation_quality_release import STARTS, make_runner, emit_request, write


def prepared_energy_fixture(backend, inputs, image, root, prepared_path, raw, boundary_shape):
    """Persist the observed SDK feed and normal replay metadata, without execution.

    The H10 QuantInfo leaf is synthetic (scale 1/255, zero point 0); the
    normal input writer must reproduce the already observed tensor exactly.
    This only describes software fixtures, never a real HEF or executable.
    """
    inputs=np.ascontiguousarray(inputs)
    assert inputs.dtype==np.uint8 and inputs.ndim==3 and inputs.shape[-1]==3
    assert raw['measurement_boundary']=='workers_ready_to_last_completed_task_frame'
    assert raw['warmup_contract']=='fully_drained_before_worker_start'
    preprocess=dict(task='detection',preprocess_mode_requested='letterbox',
        preprocess_mode_effective='letterbox',letterbox_pad_value_requested=114,
        letterbox_pad_value_effective=114,letterbox_pad_value=114)
    prepared={**preprocess,'format':'raw_runtime_tensor','name':'images',
        'shape':list(inputs.shape),'dtype':str(inputs.dtype),'layout':'HWC',
        'pad_value_effective':114,'source_image':str(image),
        'source_image_sha256':hashlib.sha256(Path(image).read_bytes()).hexdigest()}
    options={**preprocess,'prepared_input_bound':True,'copy_outputs':True}
    artifacts={}
    if backend=='hailo8_to_trt':
        assert raw['producer_impl']=='hailo8_python_vstreams_fifo'
        assert np.array_equal(np.fromfile(prepared_path,dtype=np.uint8).reshape(inputs.shape),inputs)
        prepared['preprocess']='pillow_letterbox_rgb_uint8'
        options.update(energy_prepared_feed_capable=True,
            prepared_feed_policy='exact_prepared_feed_reused_for_every_work_unit',
            producer_impl=raw['producer_impl'])
    else:
        prepared_dir=Path(root)/'contract_artifacts';prepared_dir.mkdir(parents=True,exist_ok=True)
        prepared_path=prepared_dir/('hailo10_prepared_input_00_images.npy'
            if backend=='hailo10h_to_trt' else 'deepx_selected_input.npy')
        np.save(prepared_path,inputs,allow_pickle=False)
        assert np.array_equal(np.load(prepared_path,allow_pickle=False),inputs)
        prepared.update(format='numpy_npy_v1',c_contiguous=bool(inputs.flags.c_contiguous))
        assert raw['trt_copy_outputs'] is True
        if backend=='hailo10h_to_trt':
            assert raw['producer_impl']=='hailo10_infermodel_async_fifo'
            session=SimpleNamespace(quantize_input=lambda _name,value:
                np.clip(np.rint(value*255.0),0,255).astype(np.uint8))
            handle=SimpleNamespace(runtime_input_shapes={'images':inputs.shape},session=session)
            reproduced=h10._make_input(SimpleNamespace(handle=handle,input_names=['images']),True,
                image=str(image),preprocess_mode='letterbox',letterbox_pad_value=114,task='detection')
            assert np.array_equal(reproduced['images'],inputs)
            prepared.update(slot_order=['images'],preprocess_mode='letterbox',
                preprocess='exact_performance_prepared_tensor_persisted',
                normalization='hef_quant_info_from_unit_float32',letterbox=True,pad_value=114,
                entries=[{'name':'images','artifact_name':'prepared_input_00',
                    'shape':list(inputs.shape),'dtype':str(inputs.dtype),
                    'c_contiguous':bool(inputs.flags.c_contiguous)}])
            artifacts['prepared_input_00']=_artifact(prepared_path)
            options.update(producer_impl='async_fifo',part1_onnx_used=False,
                canonical_input_slot_names=['images'],canonical_output_slot_names=['cut'],
                runtime_input_shapes={'images':list(inputs.shape)},
                runtime_output_shapes={'cut':list(boundary_shape)},
                quantized_inputs=True,quantized_outputs=True)
        else:
            assert backend=='deepx_to_trt'
            prepared.update(preprocess='exact_performance_letterbox_rgb_tensor_persisted_after_candidate_validation',
                selection_method='software_sdk_exact_input_observed_in_hotloop')
    artifacts['prepared_input']=_artifact(prepared_path)
    return prepared,artifacts,options


def native_observation(selected, root, *, model_path=None, frames=1000, warmup=100, repetition_count=3):
    """Real Native counters/Completion/writers around three fake SDK leaves."""
    scope=selected['preselection'];model,case,backend,precision,setup=[scope[k] for k in ('model_id','case_id','backend','precision','setup_id')]
    root.mkdir(parents=True,exist_ok=True)
    image=root/'input.png'
    from PIL import Image
    suite=root;write(suite/'benchmark_set.json',{'model_id':model,'benchmark_task':'detection'})
    if model=='yolo11l':
        from tests.test_v27923_decoded_completion import _outputs,DECLARATION
        outputs=_outputs()
        (suite/'output_contracts.json').write_bytes(DECLARATION.read_bytes())
        input_hw=(640,640);original_wh=(640,480)
    elif model=='yolo26s':
        from tests.test_v272_detection_completion_runtime import _decoded_outputs
        outputs=_decoded_outputs()
        outputs['detections'][:,:,:4] *= 10
        input_hw=(640,640);original_wh=(640,480)
    else:
        outputs={name:np.full((1,3,640//stride,640//stride,85),-20.,dtype=np.float32)
                 for name,stride in (('output',8),('clone_1',16),('clone_2',32))}
        outputs['output'][0,0,40,40,:4]=0.;outputs['output'][0,0,40,40,4:6]=10.
        input_hw=(640,640);original_wh=(640,480)
    if model!='yolo11l':
        is_raw=model=='yolov7_paper'
        write(suite/'output_contracts.json',{'schema':'onnx-splitpoint/output-contracts','schema_version':1,
            'model_id':model,'task':'detection','contracts':[{
                'schema':'onnx-splitpoint/output-contract','schema_version':1,'model_id':model,'task':'detection',
                'backend':'tensorrt','variant':'full','contract_status':'recorded',
                'endpoint_mode':'raw_head' if is_raw else 'decoded_nms',
                'host_tail_required':is_raw,'postprocessing_required':is_raw,
                **({} if is_raw else {'source_coordinate_space':'model_input_letterbox_xyxy_pixels'})}]})
    Image.new('RGB',original_wh,(42,96,128)).save(image)
    inputs=h10._image_to_shape(str(image),(*input_hw,3),True,
        preprocess_mode='letterbox',letterbox_pad_value=114,task='detection')
    execution=h10._build_detection_completion_contract(benchmark_set=suite,model_id=model,outputs=outputs,
        inputs={'images':inputs},image_path=str(image),preprocess_mode='letterbox',letterbox_pad_value=114)
    dtype=np.dtype(scope['boundary_tensor_dtype']);shape=tuple(scope['boundary_tensor_shape'])
    boundary=np.zeros(shape,dtype=dtype)
    class TRT:
        inputs=['cut'];shapes={'cut':tuple(selected['native_trt_meta']['inputs'][0]['shape'])};dtypes={'cut':dtype}
        host_memory_policy='fixture_owned';output_materialization_policy='fixture_host_outputs'
        def prepare_inputs(self,feeds):self.feeds=feeds
        def run_prepared(self):return outputs
        def run(self,feeds):self.prepare_inputs(feeds);return self.run_prepared()
    class H8:
        def run(self,prepared,feeds):return SimpleNamespace(outputs={'cut':boundary})
    class DX:
        def run(self,feeds):return [boundary]
    class H10(_FakeHailo10Session):
        _hef_output_names=['cut'];_output_name_hef_to_canonical={'cut':'cut'}
        output_shapes={'cut':shape}
        def _binding_output(self,binding,name):return _Buffer(boundary)
    repetitions=[]
    for index in range(repetition_count):
        runtime=DetectionCompletionRuntime(execution);warm=DetectionCompletionRuntime(execution)
        opts=dict(frames=frames,warmup=warmup,queue_depth=3,duration_s=0.,completion_runtime=runtime,warmup_completion_runtime=warm)
        if backend=='hailo8_to_trt':
            raw=h8._hailo8_python_fifo_run(H8(),None,{'images':inputs},TRT(),
                expected_boundary_name='cut',expected_boundary_shape=list(shape),expected_boundary_dtype=str(dtype),**opts)
        elif backend=='hailo10h_to_trt':
            raw=h10._hailo10_async_fifo_run(H10(),{'images':inputs},TRT(),inflight=8,task='detection',**opts)
        else:raw=dx._deepx_fifo_run(DX(),inputs,TRT(),task='detection',**opts)
        assert runtime.completed_count==frames and warm.completed_count==warmup
        raw.update(repetition_id=index+1,runtime_instance_id=f'{model}-{case}-{backend}-{index+1}',
            status='ok' if raw['completed_work_units']==frames else 'incomplete')
        raw.update(persist_detection_completion_execution_artifacts(raw,output_path=root/f'repetition_{index+1}.json'))
        repetitions.append(raw)
    raw=copy.deepcopy(repetitions[-1])
    # Persist tensors already produced by the SDK leaf, with existing dump and
    # payload-seal writers. The same bytes feed stored CPU reference checking.
    output_path=Path(dx._dump_trt_outputs(outputs,root/'native_outputs',producer='software_fixture_sdk',case=case,
        backend=backend,benchmark_set=suite,input_image=str(image),task='detection'))
    raw.update(dx._endpoint_report_fields(output_path))
    runtime_boundary_evidence={}
    if backend=='hailo10h_to_trt':
        metadata=json.loads(Path(selected['artifacts']['boundary_metadata']['path']).read_text())
        previous_boundary=h10._STRICT_SPLIT_BOUNDARY
        previous_evidence=h10._STRICT_SPLIT_BOUNDARY_EVIDENCE
        try:
            h10._STRICT_SPLIT_BOUNDARY={**metadata['boundary_tensor'],
                'metadata_sha256':metadata['metadata_sha256']}
            h10._pick_hailo_output({'cut':boundary},TRT())
            runtime_boundary_evidence=copy.deepcopy(h10._STRICT_SPLIT_BOUNDARY_EVIDENCE)
        finally:
            h10._STRICT_SPLIT_BOUNDARY=previous_boundary
            h10._STRICT_SPLIT_BOUNDARY_EVIDENCE=previous_evidence
        boundary_path=Path(h10._dump_hailo10_boundary(root,boundary_name='cut',boundary=boundary,
            inputs={'images':inputs},trt=TRT(),case=case,precision=precision,
            boundary_layout=scope['boundary_layout'],input_image=str(image),
            preprocess_mode='letterbox',letterbox_pad_value=114,
            quantized_inputs=True,quantized_outputs=True))
    else:
        boundary_path=Path(dx._dump_boundary(boundary,inputs,root/'boundary',producer='software_fixture_sdk',backend=backend,
            case=case,precision=precision,trt_input_name='cut',trt_input_dtype=str(dtype),input_image=str(image),
            boundary_layout=scope['boundary_layout'],preprocess_mode='letterbox',letterbox_pad_value=114))
    dx._seal_manifest_payload_files(output_path);dx._seal_manifest_payload_files(boundary_path)
    boundary_payload=json.loads(boundary_path.read_text())
    prepared_path=Path(boundary_payload['input_dump'])
    prepared,prepared_artifacts,replay_options=prepared_energy_fixture(
        backend,inputs,image,root,prepared_path,raw,shape)
    executable=root/'fake-sdk-leaf';executable.write_bytes(b'fixture SDK leaf; never executed')
    artifacts=copy.deepcopy(selected['artifacts']);artifacts['dxnn' if backend=='deepx_to_trt' else 'hef']=artifacts.pop('part1_runtime')
    for role in ('python_executable','native_executable','generated_cpp','cmake'):
        artifacts[role]=_artifact(executable)
    artifacts.update(semantic_output_manifest=_artifact(output_path),semantic_boundary_manifest=_artifact(boundary_path))
    artifacts.update(prepared_artifacts)
    duplicates=native_split_quality_selection_duplicates(selected)
    command_payload={'complete':True,'backend':backend,'model':model,'case':case,'precision':precision,
        'setup_id':setup,'comparison_backend':backend.removesuffix('_to_trt'),'runner':str(executable),
        'runner_sha256':artifacts['native_executable']['sha256'],'python_executable':str(executable),'benchmark_set':str(suite),
        'input_image':str(image),'input_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'interpreter_identity':{'executable':str(executable),'executable_sha256':artifacts['native_executable']['sha256']},
        'runtime_options':{'warmup':warmup,'frames':frames,'repetitions':repetition_count,'queue_depth':3,'inflight':8,
            'hailo_format':'uint8','quantized_inputs':True,'dump_outputs':True,'dump_boundary':True,
            'completion_execution_contract':execution,'completion_execution_contract_sha256':execution['contract_sha256'],
            **replay_options},
        'prepared_input_contract':prepared,
        'boundary_contract':{'boundary_layout_effective':scope['boundary_layout']},
        'runtime_boundary_evidence':runtime_boundary_evidence,
        'quality_boundary_contract':selected['boundary_contract'],'quality_boundary_contract_sha256':selected['boundary_contract_sha256'],
        'quality_preselection':scope,'quality_preselection_sha256':selected['preselection_sha256'],
        'eval_run_id':selected['eval_run_id'],'source_run_id':backend,'native_split_quality_binding':selected,
        'native_split_quality_binding_sha256':selected['binding_sha256'],'native_split_quality_eval_run_id':selected['eval_run_id'],
        'native_split_quality_source_run_id':backend,'native_split_quality_local_verification':selected['local_artifact_verification'],
        'artifacts':artifacts,**duplicates}
    if backend in {'hailo8_to_trt','hailo10h_to_trt'}:
        writer=h8 if backend=='hailo8_to_trt' else h10
        command_payload['boundary_contract']=writer._engine_boundary_contract(Path(artifacts['engine']['path']))
    else:
        command_payload['boundary_contract'].update(
            boundary_layout_requested=scope['boundary_layout'],
            dequant_scale=None,dequant_zero_point=None,
            metadata_path=artifacts['native_trt_meta']['path'],
            metadata_sha256=artifacts['native_trt_meta']['sha256'])
    command_payload.update(engine=artifacts['engine']['path'],
        engine_sha256=artifacts['engine']['sha256'])
    # The normal script main projects these facts after the hotloop. They come
    # from the same fake SDK input used above, never from a positive gate flag.
    trt_input=TRT.inputs[0]
    raw.update(trt_inputs=list(TRT.inputs),trt_input_dtype=str(TRT.dtypes[trt_input]),
        trt_input_bytes=int(np.prod(TRT.shapes[trt_input])*TRT.dtypes[trt_input].itemsize),
        boundary_layout=scope['boundary_layout'])
    if backend=='hailo8_to_trt':
        # This is the mixed-runtime SDK test leaf; all scientific contract,
        # Completion and consumer checks below remain the normal code.
        site=root/'fake_hailo_site';site.mkdir()
        consumer_source=Path(dx.__file__)
        artifacts['native_trt_consumer_source']=_artifact(consumer_source)
        command_payload['interpreter_identity'].update(resolved_executable=str(executable),
            runtime_mode='system_tensorrt_with_process_local_hailo_sites',process_local_extra_sites=[str(site)])
        command_payload['runtime_options'].update(task='detection',producer_impl='hailo8_python_vstreams_fifo',
            mixed_runtime_site_policy='site.addsitedir_after_system_defaults',process_local_extra_sites=[str(site)])
        command_payload['mixed_runtime_contract']={
            'status':'ready','runtime_mode':'system_tensorrt_with_process_local_hailo_sites',
            'site_policy':'site.addsitedir_after_system_defaults','python_executable':str(executable),
            'resolved_python_executable':str(executable),'process_local_extra_sites':[str(site)],
            'modules':{'tensorrt':str(root/'fake_system_tensorrt.py'),'hailo_platform':str(site/'fake_hailo.py'),
                'numpy':np.__file__,'PIL':Image.__file__},'cudart':'fixture_sdk_leaf',
            'native_trt_consumer_source':str(consumer_source),
            'native_trt_consumer_source_sha256':artifacts['native_trt_consumer_source']['sha256'],
            'source_closure_ok':True}
    contract=seal_native_command_contract(command_payload)
    verified,status=verify_native_command_contract(contract)
    assert verified,status
    raw.update(backend=backend,model=model,case=case,model_id=model,case_id=case,task='detection',setup_id=setup,
        precision=precision,comparison_backend=backend.removesuffix('_to_trt'),eval_run_id=selected['eval_run_id'],source_run_id=backend,
        native_command_contract=contract,native_command_contract_sha256=contract['contract_sha256'],
        workload_contract_sha256=contract['contract_sha256'],native_split_quality_binding=selected,
        native_split_quality_binding_sha256=selected['binding_sha256'],native_split_quality_eval_run_id=selected['eval_run_id'],
        native_split_quality_source_run_id=backend,
        native_fifo_output_manifest=str(output_path),native_fifo_boundary_manifest=str(boundary_path),
        dump_manifest=str(output_path),boundary_manifest=str(boundary_path),
        input_image=str(image),input_image_sha256=contract['input_image_sha256'],completion_execution_contract=execution,**duplicates)
    joined,status=bind_quality_to_native_split(native_row=raw,quality_binding=selected)
    assert joined,status
    args=SimpleNamespace(model_id=model,setup_id=setup,precision=precision)
    module={'hailo8_to_trt':h8,'hailo10h_to_trt':h10,'deepx_to_trt':dx}[backend]
    consumer=(module._seal_split_consumer_attestation(binding=joined,command=contract,args=args,case=case,task='detection')
        if backend=='hailo8_to_trt' else module._seal_split_consumer_attestation(joined,contract,args,case=case,task='detection'))
    raw.update(native_split_quality_consumer_attestation=consumer,native_split_quality_consumer_status=status)
    joined,status=bind_quality_to_native_split(native_row=raw,quality_binding=selected,verification_mode='portable')
    assert joined,status
    for index,rep in enumerate(repetitions):
        rep.update(workload_contract_sha256=contract['contract_sha256'],ok=rep['completed_work_units']==frames)
    aggregation=h10._aggregate_repetition_metrics(repetitions,repetition_runtime_scope='fresh_runtime_per_repetition',
        repetition_independence_verified=True)
    raw.update({k:v for k,v in aggregation.items() if k.startswith(('repetition','fps_','aggregate_'))})
    raw.update(performance_repetitions=aggregation['repetition_records'],frames=frames,warmup=warmup,
        model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else '',
        ok=all(r['ok'] for r in repetitions),status='ok' if all(r['ok'] for r in repetitions) else 'failed',
        fps_makespan=float(np.median([r['fps_makespan'] for r in repetitions])))
    output=root/'native_result.json';raw['native_report']=str(output);write(output,raw)
    return {'row':raw,'result_path':output,'outputs':outputs,'inputs':inputs,'completion_contract':execution,
            'model_path':model_path,
            'output_manifest':output_path,'boundary_manifest':boundary_path,'selected_binding':selected}


def stored_original_reference(observation, root):
    """A separate old-run SDK/CPU-output fixture; all proof readers stay real.

    Synthetic CPU predictions are the sole original-reference output leaf.
    There is no ORT model session, and no positive consumer/validator map is
    injected. The old Split source has its own normal request and consumer.
    """
    row=observation['row'];model_path=Path(observation['model_path'])
    start=(row['model'],'b003',row['setup_id'],row['backend'])
    old=make_runner(root,[start],run_name='closed_original_native')
    try:
        target=old.augmentation_plan['rows'][0]
        request,_=emit_request(old,target,old.run_dir/'producer',quality_model_path=model_path)
        old._queue_central_quality_requests(target['model_id'])
        result=old._central_quality_futures[str(request.resolve())].result(timeout=120)
        summary=json.loads(old._publish_case_quality([result]).read_text())
        sets=split_binding_set_from_central_quality_summary(summary,eval_run_id=old.run_id,setup_id=target['setup_id'],
            selections=[dict(target,backend=target['direction'])])
        selected=next(iter(sets['bindings_by_model_case_backend'].values()))
        original=native_observation(selected,old.run_dir/'native_producers'/row['backend']/row['model'],
            model_path=model_path,frames=1,warmup=0,repetition_count=1)
    finally:old._shutdown_management_services()
    source=original['row'];policy=validator.AccuracyGatePolicy();feed={}
    height,width=original['inputs'].shape[:2]
    assert validator._native_input_dump_feed(original['boundary_manifest'],[1,3,height,width],
        image_scale='norm',target_dtype='float32',evidence=feed) is not None
    # This substitutes only the existing saved CPU output. Its tensors and
    # resulting detections follow the frozen real model-family processor.
    cpu_leaf=DetectionCompletionRuntime(original['completion_contract'])
    detections=cpu_leaf.process(original['outputs'])['detections']
    comparison=original['completion_contract']['comparison_endpoint_contract']
    reference={
        'completed_v2_verified':True,'self_reference_input_manifest_kind':'split_boundary_manifest',
        'full_onnx':str(model_path),'boundary_manifest':str(original['boundary_manifest']),
        'reference_input_evidence':feed,'reference_detections':detections,
        'full_reference_result_sha256':canonical_json_sha256(detections),
        'completed_task_comparison_endpoint_contract':comparison,
        'completed_task_comparison_endpoint_contract_hash':comparison['endpoint_contract_hash'],
        'numerical_similarity_threshold':policy.native_self_reference_min_match,
        'numerical_similarity_mean_iou_threshold':policy.native_self_reference_min_mean_iou,
        'numerical_similarity_policy_id':policy.native_self_reference_policy_id,
        'numerical_similarity_iou_threshold':policy.native_self_reference_iou_threshold,
        'numerical_similarity_confidence_threshold':policy.native_self_reference_confidence_threshold,
    }
    source['task_validation']=str(root/'stored_cpu_validation.json')
    write(Path(source['task_validation']),{'schema':'onnx-splitpoint/native-detection-semantic-validation',
        'manifest':source['dump_manifest'],'self_reference_report':reference})
    return source,policy


@pytest.mark.parametrize('index',range(3),ids=['new_h8_yolo11','new_h10_yolo26s','new_deepx_yolov7'])
def test_real_native_hotloops_and_portable_output_consumer_from_new_quality(tmp_path,index):
    runner=make_runner(tmp_path,[STARTS[index]])
    try:
        target=runner.augmentation_plan['rows'][0]
        original_run=(Path(os.environ.get('THESIS20_FIXTURE_HOME', Path.home())) / 'Models/EvaluationRuns/thesis_20splits_n5000_b1000_20260925_20260925_103745')
        model_path=original_run/'models'/target['model_id']/'benchmark_set/legacy_suite/models'/f"{target['model_id']}.onnx"
        if not model_path.is_file():pytest.skip('local integration requires the immutable original model bytes; no fake model hash')
        request,_=emit_request(runner,target,runner.run_dir/'producer',quality_model_path=model_path)
        runner._queue_central_quality_requests(target['model_id'])
        result=runner._central_quality_futures[str(request.resolve())].result(timeout=120)
        summary=json.loads(runner._publish_case_quality([result]).read_text())
        sets=split_binding_set_from_central_quality_summary(summary,eval_run_id=runner.run_id,setup_id=target['setup_id'],
            selections=[dict(target,backend=target['direction'])])
        selected=next(iter(sets['bindings_by_model_case_backend'].values()))
        fixture=native_observation(selected,runner.run_dir/'native_producers'/target['direction']/target['model_id']/'benchmark_set'/'native_pipeline'/target['case_id']/target['precision'],model_path=model_path)
        row=fixture['row']
        assert len(row['performance_repetitions'])==3
        assert all(r['completed_work_units']==1000 for r in row['performance_repetitions'])
        checked=final._verify_split_semantic_artifacts(result_path=fixture['result_path'],manifest_path=fixture['output_manifest'],
            manifest_sha256=_artifact(fixture['output_manifest'])['sha256'],sources=(row,))
        assert checked['native_split_final_portable_binding_valid'] is True,checked
        assert checked['native_split_semantic_binding_valid'] is True,checked
        reference,policy=stored_original_reference(fixture,tmp_path/'stored_reference')
        before=Path(reference['task_validation']).read_bytes()
        semantic=validator._existing_split_reference_detection(fixture['output_manifest'],row,[reference],policy)
        assert semantic['available'] is True,semantic
        assert semantic['numerical_similarity_pass'] is True,semantic
        assert semantic['reference_eval_run_id']=='closed_original_native'
        assert semantic['reference_case']=='b003' and row['case']==STARTS[index][1]
        assert Path(reference['task_validation']).read_bytes()==before
    finally:runner._shutdown_management_services()
