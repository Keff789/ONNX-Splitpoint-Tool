"""Existing terminal Cancel checkpoint + completed Quality, without hardware."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import pytest
from tests.test_yolo_augmentation_native_barrier import coordinator
from tests.test_yolo_augmentation_quality_release import make_runner, emit_request, STARTS, write
from tests.test_v269f_variant_native_split_quality_first import _evalrun
from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256, load_stage_checkpoint, write_stage_checkpoint
from onnx_splitpoint_tool.workflow.supplement import preserve_metadata
from onnx_splitpoint_tool.workflow.yolo_augmentation_reentry import (
    COORDINATOR, CANCEL_NONSTART, validate_augmentation_reentry, native_reentry_root,
)

@pytest.fixture(scope='module')
def cancelled_seed(tmp_path_factory):
    from tests.test_v269f_variant_native_split_quality_first import _load_script
    c = _load_script('run_evalrun_native_producer_variants.py')
    tmp = tmp_path_factory.mktemp('cancel-admission-seed')
    runner = make_runner(tmp, starts=STARTS[:1]); run = runner.run_dir
    template = _evalrun(tmp/'metadata')
    manifest = json.loads((template/'run_manifest.json').read_text());manifest['run_id']=run.name
    write(run/'run_manifest.json',manifest)
    (run/'profile.yaml').write_bytes((template/'profile.yaml').read_bytes())
    row = dict(runner.augmentation_plan['rows'][0]);row['backend']='hailo8'
    write(run/'models/yolo11l/benchmark_set/b120/split_manifest.json',{})
    write(run/'models/yolo11l/benchmark_set/benchmark_set.json',{'benchmark_task':'detection'})
    plan = {'cohort_id':'yolo_min3_capability_augmentation','frozen':True,'augmentation_run':str(run),'rows':[row]}
    plan_path = tmp/'plan.json';write(plan_path,plan)
    reference=tmp/'reference.json';write(reference,{'rows':[]})
    summary_path=run/'quality_management/central_quality_summary.json'
    write(summary_path,{'schema':'onnx-splitpoint/central-quality-summary','schema_version':1,'results':[],'complete':True,'status':'skipped'})
    cfg={'native_release_mode':'per_case','backends':['hailo8'],'split_backends':['hailo8'],
        'precision':row['precision'],'frames':1000,'warmup':100,'repetitions':3,'build_missing_engines':False,
        'energy':{'enabled':False},'validation':{'enabled':True},'full_baselines':{'enabled':False},
        'remotes':{'hailo8':{'ssh':'fixture@localhost','setup_id':row['setup_id']}},
        'variants':[{'id':'new','models':['yolo11l'],'case_policy':'case_map_only','case_map':{'yolo11l':['b120']}}],
        '_workflow_context':{'central_quality_summary':str(summary_path),
            'workflow_execution':{'setup_queue_mode':'per_setup'},'execution_preset':{'id':'standard'}},
        'augmentation_startcase_barrier':{'plan_path':str(plan_path),'plan_sha256':canonical_json_sha256(plan),
            'startcases':[row],'original_reference_validation_summary':str(reference)}}
    config=run/'reports/initial-config.json';write(config,cfg)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(c,'_run',lambda *a,**k:pytest.fail('no workload before release'))
        m.setattr(sys,'argv',['coordinator','--eval-run-dir',str(run),'--config',str(config)])
        assert c.main()==2
        old_cp=(run/COORDINATOR).read_bytes()
        original=preserve_metadata(run,'quality-attempt')
        cfg['augmentation_startcase_barrier']['reentry']=validate_augmentation_reentry(run,plan_path,original)
        try:
            request,_=emit_request(runner,runner.augmentation_plan['rows'][0],tmp/'producer')
            assert runner._queue_central_quality_requests('yolo11l')==1
            result=runner._central_quality_futures[str(request.resolve())].result(timeout=120)
            assert result['technical_status']=='completed'
            progress=runner._publish_case_quality([result],complete=True)
            summary_path.write_bytes(progress.read_bytes())
        finally: runner._shutdown_management_services()
        invocation=native_reentry_root(run,cfg)/'native_invocation_config.json';write(invocation,cfg)
        write(run/'jobs/workflow_control.json',{'run_id':run.name,'cancel_requested':True,'session_id':'historical'})
        m.setattr(sys,'argv',['coordinator','--eval-run-dir',str(run),'--config',str(invocation),'--resume'])
        assert c.main()==2
    stage=json.loads((run/'reports/native_producer_stage.json').read_text())
    assert stage['error']==CANCEL_NONSTART and stage['variant_results']==[]
    assert (run/COORDINATOR).read_bytes()==old_cp
    return tmp,run,plan_path,cfg

@pytest.fixture
def cancelled(cancelled_seed):
    tmp,run,plan,cfg=cancelled_seed
    before={p:p.read_bytes() for p in tmp.rglob('*') if p.is_file()}
    yield run,plan,copy.deepcopy(cfg)
    for path in tmp.rglob('*'):
        if path.is_file() and path not in before:path.unlink()
    for path,data in before.items():path.write_bytes(data)
    # Remove only empty directories created by this temporary test.
    for path in sorted(tmp.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if path.is_dir() and not any(path.iterdir()):path.rmdir()


def test_cancel_snapshot_preserves_both_checkpoints_and_finished_quality(cancelled):
    run,plan,cfg=cancelled
    proof=validate_augmentation_reentry(run,plan)
    assert proof['kind']=='augmentation_native_cancel_nonstart'
    native=Path(proof['predecessor_checkpoint']).parent.parent
    preserved={p:p.read_bytes() for p in (run/COORDINATOR,native/'native_coordinator/stage_result.json',
        native/'native_performance/stage_result.json',native/'native_invocation_config.json')}
    quality=(run/'quality_management/central_quality_summary.json').read_bytes()
    original=preserve_metadata(run,'next-native-attempt')
    checked=validate_augmentation_reentry(run,plan,original)
    cfg['augmentation_startcase_barrier']['reentry']=checked
    assert native_reentry_root(run,cfg)==original.parent/'native'
    for path,data in preserved.items():
        assert path.read_bytes()==data
        assert (original/path.relative_to(run)).read_bytes()==data
    assert (original/'quality_management/central_quality_summary.json').read_bytes()==quality
    checkpoint=json.loads(preserved[native/'native_coordinator/stage_result.json'])
    assert load_stage_checkpoint(original/native.relative_to(run)/'native_coordinator/stage_result.json',
        stage='native_coordinator',input_hash=checkpoint['input_hash'],run_root=original)[0] is not None
    # Normal new publication replaces the canonical alias. The descriptor
    # must continue reading its immutable saved attempt, never that new alias.
    write(run/'reports/native_producer_stage.json',{'status':'running','session_id':'next-native-attempt'})
    assert native_reentry_root(run,cfg)==original.parent/'native'

@pytest.mark.parametrize('fault',['invocation','stage','checkpoint','ancestor','plan','performance',
    'quality_missing','quality_failed','request','candidate','native_raw','native_leaf','native_summary',
    'foreign_attempt','recursive_ancestor'])
def test_cancel_admission_rejects_changed_or_measured_evidence(cancelled,fault):
    run,plan,cfg=cancelled
    proof=validate_augmentation_reentry(run,plan)
    native=Path(proof['predecessor_checkpoint']).parent.parent
    summary=run/'quality_management/central_quality_summary.json'
    result=json.loads(summary.read_text())['results'][0]
    if fault=='invocation':
        path=native/'native_invocation_config.json';value=json.loads(path.read_text());value['frames']=1001;write(path,value)
    elif fault=='stage':write(run/'reports/native_producer_stage.json',{})
    elif fault=='checkpoint':
        path=native/'native_coordinator/stage_result.json';value=json.loads(path.read_text());value['input_hash']='f'*64;write(path,value)
    elif fault=='ancestor':
        path=Path(proof['quality_nonstart_ancestor']['predecessor_checkpoint']);value=json.loads(path.read_text());value['error']='other';write(path,value)
    elif fault=='plan':
        value=json.loads(plan.read_text());value['rows'][0]['case_id']='b999';write(plan,value)
    elif fault=='performance':
        path=native/'native_performance/stage_result.json';value=json.loads(path.read_text());value['complete']=True;write(path,value)
    elif fault in ('quality_missing','quality_failed'):
        value=json.loads(summary.read_text())
        if fault=='quality_missing':value['results']=[]
        else:value['results'][0]['technical_status']='failed'
        write(summary,value)
    elif fault=='request':(run/result['source_request']).write_text('{}')
    elif fault=='candidate':
        request=run/result['source_request'];value=json.loads(request.read_text());(request.parent/value['candidate']['path']).write_text('')
    elif fault=='native_raw':write(run/'native_producers/attempt/native_fifo_results.json',{'ok':False,'repetition_count_attempted':1})
    elif fault=='native_leaf':write(run/'reports/native_producer_variants/attempt/performance_stage_result.json',{})
    elif fault=='native_summary':write(run/'reports/native_producer_summary.json',{'rows':[{'ok':False,'repetition_count_attempted':1}]})
    elif fault in ('foreign_attempt','recursive_ancestor'):
        path=run/'reports/native_producer_stage.json';value=json.loads(path.read_text())
        prior=value['config']['augmentation_startcase_barrier']['reentry']
        if fault=='foreign_attempt':prior['attempt_root']=str(run/'jobs/supplement_attempts/foreign')
        else:prior['original_root']=str(run)
        write(path,value)
    with pytest.raises(ValueError):validate_augmentation_reentry(run,plan)
