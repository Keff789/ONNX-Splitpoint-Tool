"""AP1: actual producer/stage; synthetic bytes only at vendor compiler boundaries."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
from onnx_splitpoint_tool.workflow import deferred_hailo_builds as deferred
from onnx_splitpoint_tool.native_job_identity import native_job_prerequisite
from onnx_splitpoint_tool.workflow.required_run_scope import seal_required_run_scope
from test_v27921_final_selection_cache_preflight import _real_stage_runner

FIXTURE = Path(__file__).parent / 'fixtures/v2803_night_regression/evidence'
ORIGINAL = json.loads((FIXTURE / 'original_yolo11l_build_stage.json').read_text())
FAILURE = json.loads((FIXTURE / 'h8_yolo11_b064_job.json').read_text())['error']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def result(ok, output, *, error='', count=None, cache_hit=None, negative=None):
    hef = output / 'compiled.hef'
    if ok:
        hef.write_bytes(b'synthetic-boundary-artifact')
    return SimpleNamespace(ok=ok, hef_path=str(hef) if ok else None,
        error=error, details={'compiler_dispatch_count':count, 'cache_hit':cache_hit,
        **({'build_evidence':negative} if negative else {})}, calib_info={})


def prepared(root, *, only=None, deepx=True):
    model = root / 'models/yolo11l'
    suite = model / 'benchmark_set/legacy_suite'
    cases = [{'case_id':f'b{x:03}', 'case_dir':f'b{x:03}', 'boundary':x} for x in (62,64)]
    write(suite / 'benchmark_set.json', {'model_name':'yolo11l', 'task':'detection', 'cases':cases, 'plan':{'runs':[]}})
    write(model / 'benchmark_set/benchmark_set.json', {'model_id':'yolo11l', 'task':'detection', 'cases':cases})
    for case in cases:
        write(suite / case['case_dir'] / 'split_manifest.json', {'boundary':case['boundary'], 'part1_model':'part1.onnx'})
    contexts=[]
    for old in ORIGINAL['details']['deferred_hailo_builds']['jobs']:
        if not old.get('backend'): continue
        ident = (old['backend'], old['boundary'], old['stage'])
        if only is not None and ident not in only: continue
        output = suite / (old['boundary'] if old['stage']!='full' else '') / 'hailo' / old['backend'] / old['stage']
        output.mkdir(parents=True, exist_ok=True)
        source = output.parent / 'input.onnx'; source.write_bytes(b'synthetic-selected-onnx')
        context = {'model_id':'yolo11l','boundary':None if old['stage']=='full' else int(old['boundary'][1:]),'stage':old['stage']}
        deferred.cache_preflight_builder(lambda *a, **k: result(False, output, error='cache miss'))(
            source, outdir=str(output), hw_arch=old['backend'], net_name=old['net_name'],
            build_evidence_context=context, cache_only=False, force=False,
            end_node_names=['raw-head'] if old['stage']=='full' else None)
        contexts.append(output/deferred.REQUEST_NAME)
    if deepx:
        for case in cases: (suite / case['case_dir'] / 'part1.onnx').write_bytes(b'synthetic-deepx-source')
        deferred.defer_deepx_part1_build(out_dir=suite, bench_plan_runs=[], validation_images='', fallback_calib_dir='')
    seal_required_run_scope(model/'benchmark_set/required_run_scope.json', {'model_id':'yolo11l', 'measurements':[]})
    return model,suite,contexts


def boundary_builders(monkeypatch, calls, *, fail=True, exception=None):
    from onnx_splitpoint_tool.gui import benchmark_workflow
    def builder(source, **args):
        context=args['build_evidence_context']; family=args['hw_arch']
        calls.append((family,context['boundary'],context['stage']))
        bad=fail and family=='hailo8' and context['boundary']==64 and context['stage']=='part1'
        if bad and exception: raise exception
        return result(not bad, Path(args['outdir']), error=FAILURE if bad else '',
                      count=None if bad else (1 if family=='hailo10' and context['boundary']==64 else 0),
                      cache_hit=None if bad else not(family=='hailo10' and context['boundary']==64))
    monkeypatch.setattr(benchmark_workflow, 'resolve_hailo_benchmark_helpers',
                        lambda **k: SimpleNamespace(hailo_build_hef_fn=builder))
    def deepx(**args):
        calls.append(('deepx','collection','part1'))
        return {'status':'ok', 'failed_count':0, 'cases':[]}
    monkeypatch.setattr(benchmark_workflow, '_materialize_manual_deepx_part1_artifacts', deepx)
    return builder


def stage(root, monkeypatch, calls, *, targets=("hailo8", "hailo10h", "tensorrt")):
    workflow=_real_stage_runner(root); workflow.profile_payload={}
    workflow.options.skip_benchmarks=False
    workflow.options.hailo_build_full=True
    workflow._targets=lambda:list(targets)
    workflow.log=calls.append
    return workflow._run_stage('yolo11l','build_backend_artifacts',
        lambda:workflow._stage_build_backend_artifacts('yolo11l', {'id':'yolo11l','task':'detection'}))


def test_t011_actual_original_six_hailo_and_deepx_producer_full_stage(tmp_path,monkeypatch):
    prepared(tmp_path); calls=[];boundary_builders(monkeypatch,calls)
    outcome=stage(tmp_path,monkeypatch,calls)
    assert outcome.status=='partial', outcome.details
    assert outcome.details['blocked_backend_count']==1
    assert outcome.details['backend_build_states']['hailo']=='partial'
    jobs=outcome.details['deferred_hailo_builds']['jobs']
    assert len(jobs)==7 and sum(x['status']=='failed' for x in jobs)==1
    assert ('deepx','collection','part1') in calls
    readiness=outcome.details['deferred_build_readiness']
    assert readiness['required_artifact_job_count']==6
    assert readiness['required_collection_request_count']==1
    assert readiness['failed_artifact_job_count']==1
    blocker=readiness['blocked_jobs'][0]
    assert (blocker['model_id'],blocker['backend'],blocker['boundary'],blocker['stage'])==('yolo11l','hailo8','b064','part1')
    assert blocker['primary_failure_reason']==FAILURE and blocker['compiler_dispatch_count'] is None


def test_t012_all_hit_actual_stage_has_no_cold_dispatch(tmp_path,monkeypatch):
    prepared(tmp_path); calls=[]
    from onnx_splitpoint_tool.gui import benchmark_workflow
    def cached(source,**args):
        assert args['force'] is False
        calls.append((args['hw_arch'],args['build_evidence_context']['stage']))
        return result(True,Path(args['outdir']),cache_hit=True)
    boundary_builders(monkeypatch,calls,fail=False)
    monkeypatch.setattr(benchmark_workflow,'resolve_hailo_benchmark_helpers',lambda **k:SimpleNamespace(hailo_build_hef_fn=cached))
    from onnx_splitpoint_tool import hailo_compiler_context
    monkeypatch.setattr(hailo_compiler_context,'resolve_hailo_compiler_context',lambda *a,**k:pytest.fail('GPU/XLA probe on HIT'))
    outcome=stage(tmp_path,monkeypatch,calls)
    assert outcome.status=='ok' and outcome.details['blocked_backend_count']==0
    assert outcome.details['deferred_build_readiness']['ready_artifact_job_count']==6
    assert all(job['compiler_dispatch_count']==0 for job in outcome.details['deferred_hailo_builds']['jobs'] if job['backend']!='deepx')


def test_t013_exact_negative_keeps_revalidated_decision_without_artifact(tmp_path,monkeypatch):
    from test_v27922_negative_preflight import _negative
    model,suite,paths=prepared(tmp_path,only={('hailo8','b064','part1')},deepx=False)
    old=json.loads(paths[0].read_text());old['status']='known_infeasible';old['build_evidence']=_negative();write(paths[0],old)
    calls=[]
    def negative(source,**args):
        calls.append(args)
        assert args['cache_only'] is True and args['force'] is False
        return result(False,Path(args['outdir']),error='exact negative',negative=_negative())
    outcome=deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={},build_fn=negative)
    assert len(calls)==1
    readiness=outcome['readiness'];assert readiness['status']=='partial'
    assert readiness['known_infeasible_artifact_job_count']==1 and readiness['ready_artifact_job_count']==0
    assert readiness['blocked_jobs'][0]['readiness']=='not_executable'
    assert readiness['blocked_jobs'][0]['build_evidence']['state']=='COMPILE_INFEASIBLE'


def test_t014_only_required_failed_job_not_masked_by_skipped_groups(tmp_path,monkeypatch):
    prepared(tmp_path,only={('hailo8','b064','part1')},deepx=False);calls=[];boundary_builders(monkeypatch,calls)
    outcome=stage(tmp_path,monkeypatch,calls,targets=('tensorrt',))
    assert outcome.status=='partial' and outcome.details['blocked_backend_count']==1
    assert outcome.details['deferred_build_readiness']['required_artifact_job_count']==1
    assert outcome.details['deferred_build_readiness']['ready_artifact_job_count']==0


def context(**changes):
    return dict(request='/bound/request.json',model_id='yolo11l',backend='hailo8',boundary='b064',stage='part1',**changes)


@pytest.mark.parametrize('status',[None,'UNKNOWN','pending','skipped','known_infeasible'])
def test_t015_incomplete_required_observation_cannot_be_ready(status):
    row=context();job={**row,'status':status}
    projected=deferred.project_deferred_build_readiness({'jobs':[job]},selected_requests=[row])
    assert projected['status']=='partial' and projected['ready_artifact_job_count']==0
    assert projected['blocked_jobs'][0]['compiler_dispatch_count'] is None


def test_t015_missing_observation_optional_and_unselected_requests():
    row=context();summary=deferred.project_deferred_build_readiness({'jobs':[]},selected_requests=[row])
    assert summary['blocked_jobs'][0]['status']=='unknown'
    for excluded in ({**row,'selected':False},{**row,'required':False},{**row,'status':'not_applicable'}):
        summary=deferred.project_deferred_build_readiness({'jobs':[{**row,'status':'failed'}]},selected_requests=[excluded])
        assert summary['status']=='ok' and summary['required_artifact_job_count']==0


@pytest.mark.parametrize('field,value',[('boundary','b062'),('stage','full'),('stage','part2'),('backend','hailo10h'),('model_id','other')])
def test_t016_conflicting_identity_cannot_cover_own_failure(field,value):
    selected=context();current={**selected,field:value,'status':'completed'}
    summary=deferred.project_deferred_build_readiness({'jobs':[current]},selected_requests=[selected])
    assert summary['status']=='partial' and field in summary['blocked_jobs'][0]['identity_conflicts']


def test_t016_distinct_boundaries_stages_families_remain_separate():
    contexts=[]
    for case,stage_name in [('full','full'),('b062','part1'),('b064','part1'),('b064','part2')]:
        for family in ('hailo8','hailo10'):
            contexts.append({**context(), 'request':f'/{family}/{case}/{stage_name}', 'backend':family,'boundary':case,'stage':stage_name})
    jobs=[{**row,'status':'completed'} for row in contexts]
    jobs[4].update(status='failed',error=FAILURE)
    summary=deferred.project_deferred_build_readiness({'jobs':jobs},selected_requests=contexts)
    assert summary['required_artifact_job_count']==8 and summary['ready_artifact_job_count']==7
    assert summary['backend_states']=={'hailo8':'partial','hailo10h':'ok'}


@pytest.mark.parametrize('negative_result',[False,True])
def test_t017_local_context_exception_and_negative_return_are_saved(tmp_path,monkeypatch,negative_result):
    from onnx_splitpoint_tool.hailo_compiler_context import CompilerContextError
    model,suite,paths=prepared(tmp_path);calls=[]
    exc=None if negative_result else CompilerContextError('hailo_compiler_components_missing','local preparation failed')
    builder=boundary_builders(monkeypatch,calls,exception=exc)
    outcome=deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={},build_fn=builder)
    assert len(calls)==7 and ('deepx','collection','part1') in calls
    bad=next(row for row in outcome['jobs'] if row['status']=='failed')
    assert 'CompilerContextError' in bad['error'] and bad['request'].endswith('deferred_hailo_build.json')
    stored=json.loads(Path(bad['request']).read_text())
    assert stored['status']=='failed' and stored['build_error']==bad['error']
    assert bad['compiler_dispatch_count'] is None and len(bad['attempt_observations'])==1
    assert stored['build_evidence']=={}


def test_t018_failed_job_does_not_skip_later_full_and_deepx(tmp_path,monkeypatch):
    prepared(tmp_path);calls=[];boundary_builders(monkeypatch,calls)
    outcome=stage(tmp_path,monkeypatch,calls)
    bad_index=calls.index(('hailo8',64,'part1'))
    assert calls.index(('hailo10',None,'full'))>bad_index
    assert calls.index(('hailo8',None,'full'))>bad_index
    assert calls.index(('deepx','collection','part1'))>bad_index
    assert outcome.details['deferred_build_readiness']['ready_artifact_job_count']==5


@pytest.mark.parametrize('protection',['cancel','source','scope','request','unknown_builder','force','compute_mismatch'])
def test_t019_global_protection_stays_fail_closed(tmp_path,monkeypatch,protection):
    from onnx_splitpoint_tool.workflow.required_run_scope import RequiredRunScopeError
    model,suite,paths=prepared(tmp_path,deepx=False);calls=[]
    event=threading.Event()
    if protection=='cancel':event.set()
    payload=json.loads(paths[0].read_text())
    if protection=='source':Path(payload['source_onnx']).write_bytes(b'mutation')
    if protection=='request':payload.pop('kwargs');write(paths[0],payload)
    if protection=='force':payload['kwargs']['force']=True;write(paths[0],payload)
    if protection=='compute_mismatch':payload['kwargs']['compute_by_family']={'hailo8':{'device':'gpu'}};write(paths[0],payload)
    def builder(*a,**k):
        calls.append('external_boundary')
        if protection=='scope':raise RequiredRunScopeError('scope integrity changed')
        if protection=='unknown_builder':raise RuntimeError('unclassified builder failure')
        pytest.fail('protected compiler dispatched')
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={},build_fn=builder,cancel_event=event)
    assert len(calls)==(1 if protection in {'scope','unknown_builder'} else 0)


def test_t019_real_stage_scope_mutation_precedes_producer(tmp_path,monkeypatch):
    model,_,_=prepared(tmp_path);p=model/'benchmark_set/required_run_scope.json'
    payload=json.loads(p.read_text());payload['model_id']='altered';write(p,payload)
    calls=[];boundary_builders(monkeypatch,calls)
    outcome=stage(tmp_path,monkeypatch,calls)
    assert outcome.status=='failed' and not any(isinstance(row,tuple) for row in calls)


def test_t0110_deepx_collection_is_not_a_hailo_artifact(tmp_path,monkeypatch):
    model,suite,_=prepared(tmp_path,only=set());calls=[];boundary_builders(monkeypatch,calls)
    outcome=deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={})
    assert len(outcome['jobs'])==1 and outcome['jobs'][0]['backend']=='deepx'
    assert outcome['readiness']['required_artifact_job_count']==0
    assert outcome['readiness']['required_collection_request_count']==1
    assert outcome['readiness']['backend_states']=={'deepx':'ok'}


@pytest.mark.parametrize('count',[None,0,2,True,'0',-1])
def test_t0111_dispatch_requires_an_observation(count):
    request={'kwargs':{'hw_arch':'hailo8','build_evidence_context':{'boundary':64,'stage':'part1'}}}
    outcome=SimpleNamespace(ok=False,error=FAILURE,details={'compiler_dispatch_count':count},calib_info={})
    observed=deferred._completed_job_observation(request,outcome,model_id='yolo11l')
    summary=deferred.project_deferred_build_readiness({'jobs':[{**context(),**observed,'status':'failed','error':FAILURE}]})
    expected=count if type(count) is int and count>=0 else None
    assert summary['blocked_jobs'][0]['compiler_dispatch_count']==expected


def test_t0112_build_cause_reaches_exact_native_case_and_resume_keeps_history(tmp_path,monkeypatch):
    model,suite,paths=prepared(tmp_path);calls=[];boundary_builders(monkeypatch,calls)
    stage(tmp_path,monkeypatch,calls)
    identity={'model':'yolo11l','backend':'hailo8_to_trt','case':'b064','setup_id':'device-h8','comparison_backend':'hailo8','precision':'fp16'}
    bad=native_job_prerequisite(identity,suite_dir=suite,model_dir=model,quality_error='quality task identity mirror missing')
    assert bad['primary_failure_reason']==FAILURE and bad['failure_stage']=='build_backend_artifacts'
    assert bad['repetition_count_attempted']==0 and bad['secondary_failure_reason']=='quality task identity mirror missing'
    assert bad['upstream_build_observation']['compiler_dispatch_count'] is None
    assert Path(bad['upstream_evidence_path']).is_file()
    for change in ({'case':'b062'},{'backend':'hailo10h_to_trt'},{'model':'other-model'},{'backend':'native_full_hailo8','case':'full'}):
        other=native_job_prerequisite({**identity,**change},suite_dir=suite,model_dir=model,quality_error='mirror missing')
        assert other['primary_failure_reason']=='mirror missing'
    boundary_builders(monkeypatch,calls,fail=False)
    current=stage(tmp_path,monkeypatch,calls)
    assert current.status=='ok'
    assert native_job_prerequisite(identity,suite_dir=suite,model_dir=model)['prerequisite_status']=='ready'
    job=next(row for row in current.details['deferred_hailo_builds']['jobs'] if row.get('backend')=='hailo8' and row.get('boundary')=='b064')
    assert job['previous_build_observations'][0]['error']==FAILURE
    assert current.details['deferred_build_readiness']['required_artifact_job_count']==6
    selected=context();summary=deferred.project_deferred_build_readiness({'jobs':[
        {**selected,'status':'failed','error':FAILURE},{**selected,'status':'completed'}]},selected_requests=[selected,selected])
    assert summary['status']=='ok' and summary['required_artifact_job_count']==1
    assert summary['jobs'][0]['previous_observations'][0]['error']==FAILURE


# Existing real SDK-boundary publication fixture; the selected metadata and
# ordinary receipt/cache path are real, accelerator compilation is synthetic.
from test_v280_artifact_reuse import publication_case


def test_t012_fresh_process_deferred_gpu_preference_reuses_published_cpu_hef(publication_case,tmp_path):
    import subprocess
    import sys
    from test_v280_artifact_reuse import _run_controller, _payload_files
    from onnx_splitpoint_tool import hailo_backend
    first=_run_controller(publication_case,tmp_path/'original_cpu')
    assert first['ok'] and first['details']['compiler_dispatch_count']==1
    model,suite,_=prepared(tmp_path/'run',only=set(),deepx=False)
    output=suite/'hailo/hailo10h/full'
    args={key:str(value) if isinstance(value,Path) else value for key,value in publication_case.items()}
    source=args.pop('onnx_path')
    args.update(outdir=str(output),backend='venv',force=False,compute_device='gpu',
                build_evidence_context={'model_id':'yolo11l','stage':'full','boundary':None})
    deferred.cache_preflight_builder(hailo_backend.hailo_build_hef_auto)(source,**args)
    before_cache=_payload_files(tmp_path/'production_cache');before_store=_payload_files(tmp_path/'production_store')
    program='''
import importlib.abc,json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class BlockSDK(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'hailo_sdk_client','tensorflow'}:
   raise AssertionError('SDK import during reuse: '+fullname)
sys.meta_path.insert(0,BlockSDK())
from onnx_splitpoint_tool import hailo_backend,hailo_compiler_context
from onnx_splitpoint_tool.workflow.deferred_hailo_builds import finalize_deferred_hailo_builds

def forbidden(*a,**k): raise AssertionError('Compiler/GPU probe on verified HIT')
hailo_backend._run_streamed_subprocess=forbidden
hailo_compiler_context.resolve_hailo_compiler_context=forbidden
result=finalize_deferred_hailo_builds(model_dir=Path(sys.argv[2]),run_dir=Path(sys.argv[2]).parents[1],profile_payload={},build_fn=hailo_backend.hailo_build_hef_auto)
result['pid']=os.getpid()
Path(sys.argv[3]).write_text(json.dumps(result))
'''
    outcomes=[]
    for i in range(2):
        path=tmp_path/f'fresh_reuse_{i}.json'
        child=subprocess.run([sys.executable,'-I','-B','-c',program,str(Path(__file__).resolve().parents[1]),str(model),str(path)],text=True,capture_output=True,timeout=45)
        assert child.returncode==0,child.stdout+child.stderr
        outcome=json.loads(path.read_text());outcomes.append(outcome)
        assert outcome['readiness']['status']=='ok'
        assert outcome['readiness']['ready_artifact_job_count']==1
        assert outcome['jobs'][0]['cache_hit'] is True
        assert outcome['jobs'][0]['compiler_dispatch_count']==0
        assert Path(outcome['jobs'][0]['hef_path']).read_bytes()==Path(first['hef_path']).read_bytes()
    assert outcomes[0]['pid']!=outcomes[1]['pid']
    assert _payload_files(tmp_path/'production_cache')==before_cache
    assert _payload_files(tmp_path/'production_store')==before_store


def test_t0110_failed_deepx_collection_has_bound_family_but_no_guessed_artifact_count(tmp_path,monkeypatch):
    from onnx_splitpoint_tool.gui import benchmark_workflow
    model,suite,_=prepared(tmp_path,only=set())
    monkeypatch.setattr(benchmark_workflow,'_materialize_manual_deepx_part1_artifacts',
        lambda **args:{'status':'failed','failed_count':2,'error':'DeepX preparation failed'})
    outcome=deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={})
    assert outcome['readiness']['required_artifact_job_count']==0
    assert outcome['readiness']['blocked_collection_request_count']==1
    assert outcome['readiness']['backend_states']=={'deepx':'partial'}
    assert outcome['jobs'][0]['result']['failed_count']==2


@pytest.mark.parametrize('exclusion',[{'selected':False},{'required':False},{'status':'not_applicable'}])
def test_t015_explicitly_excluded_request_never_dispatches_or_blocks(tmp_path,exclusion):
    model,suite,paths=prepared(tmp_path,only={('hailo8','b064','part1')},deepx=False)
    payload=json.loads(paths[0].read_text());payload.update(exclusion);write(paths[0],payload)
    outcome=deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={},
        build_fn=lambda *a,**k:pytest.fail('excluded request dispatched'))
    assert outcome['jobs']==[] and outcome['readiness']['required_artifact_job_count']==0


@pytest.mark.parametrize('field,value',[('model_id','other'),('stage','full'),('boundary',62)])
def test_t019_request_binding_scope_mismatch_never_dispatches(tmp_path,field,value):
    model,suite,paths=prepared(tmp_path,only={('hailo8','b064','part1')},deepx=False)
    payload=json.loads(paths[0].read_text());payload['kwargs']['build_evidence_context'][field]=value;write(paths[0],payload)
    with pytest.raises(ValueError,match='deferred_request_identity_scope_invalid'):
        deferred.finalize_deferred_hailo_builds(model_dir=model,run_dir=tmp_path,profile_payload={},
            build_fn=lambda *a,**k:pytest.fail('conflicting request dispatched'))


@pytest.mark.parametrize('boundary,stage_name',[('foo','part1'),('b064','full'),('full','part2'),('selected_cases','part1')])
def test_t015_malformed_artifact_identity_is_unconfirmed(boundary,stage_name):
    row={**context(),'boundary':boundary,'stage':stage_name,'status':'completed'}
    summary=deferred.project_deferred_build_readiness({'jobs':[row]})
    assert summary['status']=='partial' and summary['ready_artifact_job_count']==0
    assert summary['blocked_jobs'][0]['primary_failure_reason']=='deferred_build_identity_unresolved'
