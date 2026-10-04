"""Normal request writer, management queue/consumer and Native binding handoff.

Only accelerator predictions/build bytes are synthetic. No inference or build
is invoked. Reference files retain their closed source-run identity.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.management_reference import _source_contract
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy, materialize_native_split_preselection,
    seal_native_split_quality_binding)
from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint, json_fingerprint
from onnx_splitpoint_tool.trt_quality_chain import split_binding_set_from_central_quality_summary
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.yolo_augmentation_quality import OriginalCPUReferenceMixin, augmentation_quality_source_rows
from tests.test_v269d_trt_central_quality_producer import _detection_quality_contract
from tests.test_v269f_native_split_receipt_validation import _fixture_payload, _artifact
from tests.test_v27911_composed_provenance_binding import _functions


STARTS = [('yolo11l', 'b120', 'orin_nx_hailo8_01', 'hailo8_to_trt'),
          ('yolo26s', 'b038', 'orin_nx_hailo10_01', 'hailo10h_to_trt'),
          ('yolov7_paper', 'b044', 'orin_nx_deepx_m1_01', 'deepx_to_trt')]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    return _artifact(path)


def binding(root, model, case, setup, backend, run_id):
    payload, paths = _fixture_payload(root)
    policy = known_native_split_policy(model_id=model, case_id=case, setup_id=setup, backend=backend)
    metadata = json.loads(paths['boundary_metadata'].read_text())
    metadata.update(model_id=model, case_id=case, setup_id=setup, backend=backend,
        source_run_id=backend, boundary_layout=policy['boundary_layout'], boundary_transform=policy['boundary_transform'])
    if backend == 'hailo10h_to_trt':
        metadata['boundary_tensor'].update(shape=[2,3,4], canonical_part2_shape=[1,2,3,4])
    if backend == 'deepx_to_trt':
        metadata['boundary_tensor']['dtype'] = 'float32'
        metadata['boundary_tensor'].pop('quantization')
    metadata.pop('metadata_sha256')
    metadata['metadata_sha256'] = canonical_json_sha256(metadata)
    write(paths['boundary_metadata'], metadata)
    selected = materialize_native_split_preselection(policy=policy, part1_artifact=_artifact(paths['part1_runtime']),
        boundary_metadata=metadata, boundary_metadata_artifact=_artifact(paths['boundary_metadata']))
    meta = payload['native_trt_meta']
    meta.update(case=case, precision=policy['precision'], requested_precision=policy['precision'])
    bridge = meta['uint8_cast_bridge']
    if backend == 'hailo10h_to_trt':
        meta['inputs'][0]['shape'] = [1,2,3,4]
        bridge['input_shape'] = [1,2,3,4]
    bridge['boundary_layout'] = {'requested':policy['boundary_layout'], 'effective':policy['boundary_layout'],
        'applied':policy['boundary_layout'] != 'as_input'}
    if backend == 'deepx_to_trt':
        meta['inputs'][0]['elem_type'] = 'FLOAT'
        bridge.update(schema='onnx-splitpoint/float32-layout-bridge', schema_version=1, input_dtype='float32')
        bridge.pop('scale'); bridge.pop('zero_point')
    write(paths['native_trt_meta'], meta)
    boundary = {name:selected[name] for name in payload['boundary_contract'] if name in selected}
    payload.update(eval_run_id=run_id, source_run_id=backend, preselection=selected,
        preselection_sha256=selected['selection_sha256'], boundary_contract=boundary,
        boundary_contract_sha256=canonical_json_sha256(boundary), native_trt_meta=meta,
        native_trt_meta_sha256=canonical_json_sha256(meta), artifacts={name:_artifact(p) for name,p in paths.items()})
    return seal_native_split_quality_binding(payload)


class Runner(OriginalCPUReferenceMixin, EvaluationWorkflowRunner):
    pass


def make_runner(tmp_path, starts=STARTS, *, run_name='new_augmentation'):
    root = tmp_path / run_name; root.mkdir(parents=True)
    runner = Runner(WorkflowOptions(profile='', out=str(tmp_path)), log=lambda *_:None)
    runner.run_dir, runner.run_id = root, root.name
    runner.run_log_path, runner.parent_log_path = root / 'evaluation_workflow.log', tmp_path / 'latest.log'
    runner.profile_payload = {'quality_gate':{'statistics':{'execution_location':'central_management', 'workers':1}},
        'workflow_execution':{'native_release_mode':'per_case'}, 'models':[{'id':row[0], 'task':'detection'} for row in starts]}
    runner.augmentation_reference_run = tmp_path / 'closed_original'
    write(runner.augmentation_reference_run / 'run_manifest.json', {'run_id':'closed_original'})
    rows = [dict(model_id=m, case_id=c, setup_id=s, direction=b, backend=b,
                 precision=known_native_split_policy(model_id=m,case_id=c,setup_id=s,backend=b)['precision'],
                 selection_role='primary') for m,c,s,b in starts]
    runner.augmentation_plan = {'augmentation_run':str(root), 'rows':rows}
    return runner


def install_original_reference(runner, model, reference, encoded=None):
    """Store the SDK factory's exact CPU input fixture for the normal reader.

    Must run before Suite dispatch so the forwarded Original descriptor and
    subsequent management reader name the same immutable source bytes.
    """
    quality=reference['quality_contract']
    suite=runner.augmentation_reference_run/'models'/model/'benchmark_set/legacy_suite'
    plan,contract={'model_id':model,'runs':[]},{'model_id':model}
    write(suite/'benchmark_plan.json',plan);write(suite/'benchmark_set.json',contract)
    (suite/'benchmark_suite.py').write_text('original closed reference writer\n')
    source=_source_contract(suite,plan,contract).removeprefix('sha256:')
    path=runner.augmentation_reference_run/'quality_management/references'/model/'by_source_contract'/source/'canonical_cpu_reference.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encoded if encoded is not None else json.dumps(reference,sort_keys=True).encode())
    descriptor=_artifact(path)
    status={'schema':'onnx-splitpoint/management-cpu-reference-job','schema_version':1,'status':'completed','model_id':model,
        'reference_path':str(path),'reference_sha256':descriptor['sha256'],'reference_size_bytes':descriptor['size_bytes'],
        'execution_location':'central_management','provider':'onnxruntime_cpu','semantic_reference_only':True,
        'include_in_latency_fps_energy':False,'include_in_ranking':False,'include_in_pareto':False,
        'reference_storage':'immutable_source_contract','reference_immutable':True,'source_contract_sha256':source,
        'task':'detection','record_count':len(reference['records']),'image_ids_sha256':quality['dataset']['image_ids_sha256'],
        'quality_contract':quality,'quality_contract_sha256':quality['quality_contract_sha256']}
    write(runner.augmentation_reference_run/'quality_management/references'/model/'management_cpu_reference_status.json',status)
    write(runner.run_dir/'models'/model/'model_manifest.json',{'model_id':model,'task':'detection','model_sha256':quality['model']['sha256']})
    runner._schedule_management_cpu_reference(model,None)
    return dict(status,reference_role='canonical_cpu_ort',source_evaluation_run=str(runner.augmentation_reference_run),
        source_evaluation_run_id=runner.augmentation_reference_run.name)


def emit_request(runner, row, tmp_path, *, accuracy_loss=False, quality_model_path=None):
    """Call the existing generated runner's writer; no hand-built request/result."""
    model, case, setup, backend = [row[k] for k in ('model_id','case_id','setup_id','direction')]
    bound = binding(tmp_path / model, model, case, setup, backend, runner.run_id)
    quality = _detection_quality_contract()
    if quality_model_path is not None:
        quality['model']['sha256']=hashlib.sha256(Path(quality_model_path).read_bytes()).hexdigest()
    ids = [f'{i:05}.jpg' for i in range(5000)]
    box = {'x1':10.,'y1':20.,'x2':40.,'y2':60.,'score':1.,'class_id':0}
    gt = {name:[box] if accuracy_loss and index<10 else [] for index,name in enumerate(ids)}
    records = [{'image_id':i,'reference':gt[i],'ground_truth':gt[i]} for i in ids]
    quality['dataset'].update(image_count=5000, image_ids_sha256=image_ids_fingerprint(ids),
        ground_truth_sha256=json_fingerprint([{'image_id':i,'ground_truth':gt[i]} for i in ids]))
    quality.pop('quality_contract_sha256');quality['quality_contract_sha256']=json_fingerprint(quality)
    suite = runner.augmentation_reference_run / 'models' / model / 'benchmark_set/legacy_suite'
    plan, contract = {'model_id':model,'runs':[]}, {'model_id':model}
    write(suite / 'benchmark_plan.json', plan);write(suite / 'benchmark_set.json',contract)
    (suite / 'benchmark_suite.py').write_text('original closed reference writer\n')
    source = _source_contract(suite,plan,contract).removeprefix('sha256:')
    refpath = runner.augmentation_reference_run / 'quality_management/references' / model / 'by_source_contract' / source / 'canonical_cpu_reference.json'
    refdesc = write(refpath, {'schema':'onnx-splitpoint/task-quality-reference-input','schema_version':1,
        'task':'detection','pairing_key':'image_id','reference_role':'canonical_cpu_ort','semantic_reference_only':True,
        'provenance_required':True,'quality_contract':quality,'quality_contract_sha256':quality['quality_contract_sha256'],'records':records})
    status={'schema':'onnx-splitpoint/management-cpu-reference-job','schema_version':1,'status':'completed','model_id':model,
        'reference_path':str(refpath),'reference_sha256':refdesc['sha256'],'reference_size_bytes':refdesc['size_bytes'],
        'execution_location':'central_management','provider':'onnxruntime_cpu','semantic_reference_only':True,
        'include_in_latency_fps_energy':False,'include_in_ranking':False,'include_in_pareto':False,
        'reference_storage':'immutable_source_contract','reference_immutable':True,'source_contract_sha256':source,
        'task':'detection','record_count':5000,'image_ids_sha256':quality['dataset']['image_ids_sha256'],
        'quality_contract':quality,'quality_contract_sha256':quality['quality_contract_sha256']}
    write(runner.augmentation_reference_run / 'quality_management/references' / model / 'management_cpu_reference_status.json',status)
    write(runner.run_dir / 'models' / model / 'model_manifest.json',{'model_id':model,'task':'detection','model_sha256':quality['model']['sha256']})
    runner._schedule_management_cpu_reference(model,None)
    functions = _functions(['_task_quality_execution_location','_quality_json_safe','_write_stable_quality_json',
        '_quality_file_sha256','_quality_contract_sha256','_export_central_quality_inputs'])
    functions['_verified_suite_vendored_endpoint_attestor_sha256'] = lambda: quality['quality_record_endpoint']['identity']['vendored_endpoint_attestor_sha256']
    out = runner.run_dir / 'models' / model / 'benchmark_results/quality_inputs' / setup / case / ('results_'+backend)
    exported = functions['_export_central_quality_inputs'](out_dir=out,task='detection',variant='composed',
        policy={'statistics':{'execution_location':'central_management','bootstrap_repetitions':1000,'seed':20260710},'policy_sha256':'b'*64},
        gt_by_image=gt,candidate_by_image={i:[] for i in ids},quality_contract=quality,
        endpoint_contract={'endpoint_contract_complete':True,'endpoint_contract_hash':'a'*64,'stage':'raw_head','contract_family':'raw_head'},
        runtime_precision_identity=row['precision'], native_split_quality_binding=bound)
    path = out / 'task_quality_inputs/composed_request.json'
    assert path.is_file() and exported['record_count']==5000
    return path, bound


def test_empty_queue_is_failed_with_exact_missing_bindings_instead_of_skipped(tmp_path):
    runner=make_runner(tmp_path)
    try:
        artifacts, metrics, note, state=runner._stage_evaluate_quality()
        summary=json.loads(artifacts['central_quality_summary_json'].read_text())
        progress=json.loads((runner.run_dir/'quality_management/case_quality_progress.json').read_text())
        assert state=='failed' and len(summary['quality_obligation_errors'])==3
        assert metrics['request_count']==0 and progress['status']=='failed' and progress['complete'] is True
        assert 'Required new Quality bindings are incomplete' in note
    finally: runner._shutdown_management_services()


@pytest.mark.parametrize('damage',['model','case','setup','precision','eval_run','reference',
                                  '50_records','bootstrap','seed','candidate_missing','candidate_changed'])
def test_wrong_or_incomplete_new_request_never_publishes_native_binding(tmp_path,damage):
    runner=make_runner(tmp_path,[STARTS[0]])
    try:
        row=runner.augmentation_plan['rows'][0]
        request,_=emit_request(runner,row,tmp_path/'producer')
        raw=json.loads(request.read_text())
        if damage in {'model','case','setup','precision','eval_run'}:
            field={'model':'model_id','case':'case_id','setup':'setup_id',
                   'precision':'runtime_precision_identity','eval_run':'eval_run_id'}[damage]
            raw[field]={'model':'yolo26s','case':'b003','setup':'foreign_setup',
                        'precision':'fp16','eval_run':'closed_original'}[damage]
        elif damage=='reference':raw['reference']['quality_contract_sha256']='f'*64
        elif damage=='50_records':raw['record_count']=50
        elif damage in {'bootstrap','seed'}:raw['statistics'][{'bootstrap':'bootstrap_repetitions','seed':'seed'}[damage]]=50
        else:
            candidate=request.parent/raw['candidate']['path']
            if damage=='candidate_missing':candidate.unlink()
            else:candidate.write_bytes(candidate.read_bytes()+b' ')
        write(request,raw)
        assert runner._queue_central_quality_requests(row['model_id'])==1
        result=runner._central_quality_futures[str(request.resolve())].result(timeout=120)
        assert result['technical_status']=='failed',result
        assert result.get('producer_binding_eligible') is not True
        summary=json.loads(runner._publish_case_quality([result]).read_text())
        assert runner._quality_obligation_errors(summary['results'])
    finally:runner._shutdown_management_services()


def test_retained_candidate_requires_local_statistics_and_completed_accuracy_loss_is_bindable(tmp_path):
    runner=make_runner(tmp_path,[STARTS[0]])
    try:
        row=runner.augmentation_plan['rows'][0]
        request,_=emit_request(runner,row,tmp_path/'producer',accuracy_loss=True)
        generic=[dict(row,run_id=row['direction'],variant='split',action='preserve')]
        plan={'generic_rows':generic}
        source=augmentation_quality_source_rows(runner,plan)
        assert source[0]['action']=='local_statistics' and generic[0]['action']=='preserve'
        # A merely written request is never a completed Native prerequisite.
        assert runner._quality_obligation_errors([])
        assert runner._queue_central_quality_requests(row['model_id'])==1
        result=runner._central_quality_futures[str(request.resolve())].result(timeout=120)
        assert result['technical_status']=='completed',result
        assert result['decision'] in {'fail','accuracy_loss'},result
        assert not runner._quality_obligation_errors([result])
        plan['completed_quality_requests']=[{'path':str(request.relative_to(runner.run_dir))}]
        assert augmentation_quality_source_rows(runner,plan)[0]['action']=='preserve'
    finally:runner._shutdown_management_services()


def overlay_rows(runner, row, request, bound, tmp_path):
    runner._supplement_original_root=tmp_path/'original_attempt'
    write(runner.run_dir/'models'/row['model_id']/'benchmark_results/remote_runs/case/task_quality_inputs/native_split_quality_binding.json',bound)
    original=dict(row,run_id=row['direction'],variant='split',primary_variant='composed',task='detection',
        quality_source_variant='composed',quality_source_run_id=row['direction'],quality_source_setup_ids=[row['setup_id']],
        source_path='original_performance.json',total_latency_ms=123.45,throughput_primary_fps=7.89,
        timings={'composed':{'mean_ms':123.45}},runtime_ok=True,structural_contract_pass=False,
        endpoint_contract_hash='',runtime_precision_identity='',task_quality_gates_by_variant={},
        quality_request_identities_by_variant={})
    descriptor=json.loads(request.read_text())
    descriptor['request']={'path':str(request),'sha256':hashlib.sha256(request.read_bytes()).hexdigest()}
    gate={'status':'pending_central_evaluation','decision':'pending_central_evaluation','variant':'composed',
          'quality_input_request':descriptor}
    candidate=dict(original,total_latency_ms=.01,throughput_primary_fps=99999.,source_path='new_attempt_report.json',
        endpoint_contract_hash=descriptor['endpoint_contract_hash'],runtime_precision_identity=row['precision'],
        structural_contract_pass=True,task_quality_gate=gate,task_quality_gates_by_variant={'composed':gate},
        quality_evaluation_pending=True)
    return original,candidate


def test_quality_overlay_joins_actual_completed_result_without_upgrading_original_performance(tmp_path):
    runner=make_runner(tmp_path,[STARTS[0]])
    try:
        row=runner.augmentation_plan['rows'][0]
        request,bound=emit_request(runner,row,tmp_path/'producer')
        original,candidate=overlay_rows(runner,row,request,bound,tmp_path)
        before=copy.deepcopy(original)
        merged=runner._merge_supplement_quality_metadata([original],[candidate])[0]
        assert original==before
        allowed={'task_quality_gate','task_quality_gates_by_variant','quality_request_identities_by_variant',
                 'quality_source_variant','quality_source_run_id','quality_source_setup_ids','quality_evaluation_pending',
                 'task_quality_policy','task_quality_policy_sha256','supplement_quality_source'}
        assert {k:v for k,v in merged.items() if k not in allowed}=={k:v for k,v in original.items() if k not in allowed}
        assert merged['structural_contract_pass'] is False and merged['endpoint_contract_hash']==''
        identity=runner._quality_request_identity_for_row_variant(merged,'composed')
        assert identity['source_request_sha256']==hashlib.sha256(request.read_bytes()).hexdigest()
        write(runner.run_dir/'models'/row['model_id']/'benchmark_results/normalized_results.json',{'results':[merged]})
        assert runner._queue_central_quality_requests(row['model_id'])==1
        artifacts,metrics,note,state=runner._stage_evaluate_quality()
        summary=json.loads(artifacts['central_quality_summary_json'].read_text())
        assert state=='ok',note
        assert summary['merge']['matched_primary_result_count']==1 and summary['merge']['unmatched_result_count']==0
        after=json.loads((runner.run_dir/'models'/row['model_id']/'benchmark_results/normalized_results.json').read_text())['results'][0]
        assert after['total_latency_ms']==123.45 and after['throughput_primary_fps']==7.89
        assert after['structural_contract_pass'] is False
    finally:runner._shutdown_management_services()


@pytest.mark.parametrize('damage',['missing_original_binding','other_runtime_bytes','descriptor_precision','request_missing'])
def test_quality_overlay_rejects_unproved_attachment(tmp_path,damage):
    runner=make_runner(tmp_path,[STARTS[0]])
    try:
        row=runner.augmentation_plan['rows'][0]
        request,bound=emit_request(runner,row,tmp_path/'producer')
        original,candidate=overlay_rows(runner,row,request,bound,tmp_path)
        prior=next((runner.run_dir/'models').glob('*/benchmark_results/remote_runs/**/native_split_quality_binding.json'))
        if damage=='missing_original_binding':prior.unlink()
        elif damage=='other_runtime_bytes':
            changed=copy.deepcopy(bound);changed['artifacts']['engine']['sha256']='f'*64;write(prior,changed)
        elif damage=='descriptor_precision':
            candidate['task_quality_gate']['quality_input_request']['runtime_precision_identity']='fp16'
        else:request.unlink()
        with pytest.raises(ValueError):runner._merge_supplement_quality_metadata([original],[candidate])
    finally:runner._shutdown_management_services()


def test_source_rows_preserve_original_performance_and_do_not_invent_requests(tmp_path):
    runner=make_runner(tmp_path)
    generic=[dict(r,run_id=r['direction'],variant='split',action='preserve') for r in runner.augmentation_plan['rows']]
    plan={'generic_rows':generic};before=copy.deepcopy(plan)
    sources=augmentation_quality_source_rows(runner,plan)
    assert plan==before and len(sources)==3
    assert all(r['stage']=='quality_source' and r['action']=='execute' and r['performance_action']=='preserve' for r in sources)


@pytest.mark.parametrize('index',range(3),ids=['hailo8_yolo11_b120','hailo10_yolo26s_b038','deepx_yolov7_b044'])
def test_normal_request_queue_consumer_publication_yields_new_native_binding(tmp_path,index):
    runner=make_runner(tmp_path,[STARTS[index]])
    try:
        row=runner.augmentation_plan['rows'][0]
        request,bound=emit_request(runner,row,tmp_path/'producer')
        before={p:p.read_bytes() for p in runner.augmentation_reference_run.rglob('*') if p.is_file()}
        assert runner._queue_central_quality_requests(row['model_id'])==1
        result=runner._central_quality_futures[str(request.resolve())].result(timeout=120)
        assert result['status']=='completed',result
        assert result['technical_status']=='completed' and result['producer_binding_eligible'] is True,result
        path=runner._publish_case_quality([result])
        summary=json.loads(path.read_text())
        sets=split_binding_set_from_central_quality_summary(summary,eval_run_id=runner.run_id,
            setup_id=row['setup_id'],selections=[dict(row,backend=row['direction'])])
        assert len(sets['bindings_by_model_case_backend'])==1
        selected=next(iter(sets['bindings_by_model_case_backend'].values()))
        assert selected['producer_binding_sha256']==bound['binding_sha256']
        assert selected['source_request_sha256']==hashlib.sha256(request.read_bytes()).hexdigest()
        assert selected['central_quality_selection']['eval_run_id']==runner.run_id
        assert not runner._quality_obligation_errors([result])
        assert before=={p:p.read_bytes() for p in runner.augmentation_reference_run.rglob('*') if p.is_file()}
    finally: runner._shutdown_management_services()
