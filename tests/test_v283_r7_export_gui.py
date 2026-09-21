"""Real debug export and GUI summary->snapshot->queue boundary, no run dispatch."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
import zipfile

import pytest
import yaml
from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack
from onnx_splitpoint_tool.energy.task_budget import campaign_budget_profile_args

FIXTURES=Path(__file__).parent/'fixtures/v283_r7'


def external_fixture(tmp_path,mode):
    run=tmp_path/'completsetdev_20260915_085558';run.mkdir()
    (run/'evaluation_workflow.log').write_text('remote_storage_preflight_failed rc=124\nmanagement_service_shutdown_unresolved\n')
    for setup in ('orin_nx_hailo8_01','orin_nx_hailo10_01'):
        name=f'remote_benchmark_status_{setup}.json'
        payload=json.loads((FIXTURES/name).read_text())
        external=tmp_path/'RemoteBenchmarkRuns/Results/legacy_suite/1'/f'{run.name}_yolo26s_{setup}'
        payload['local_run_dir']=str(external)
        if isinstance(payload.get('remote_output'),dict):payload['remote_output']['local_run_dir']=str(external)
        dest=run/'models/yolo26s/benchmark_results'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if mode=='traversal':payload['local_run_dir']=str(external/'../outside')
        dest.write_text(json.dumps(payload))
        if mode!='missing_root':
            diag=external/'diagnostics';diag.mkdir(parents=True)
            (diag/'pre_mkdir_storage_remote_storage_diagnostics.json').write_text(json.dumps({'rc':0,'password':'hidden_password','output':'token=secret_value'}))
            if mode=='complete':
                (diag/'before_uncached_suite_upload_remote_storage_probe_failure.json').write_text(json.dumps({'rc':124,'stage':'before_uncached_suite_upload','command':'python3 statvfs','extra':{'transport':{'phase':'process_wait_timeout','owner_pid':123}}}))
            if mode=='symlink':
                outside=tmp_path/f'{setup}_secret.json';outside.write_text('DO_NOT_EXPORT')
                (diag/'timeout.json').symlink_to(outside)
            if mode=='oversize':(diag/'timeout.log').write_text('x'*4097)
    return run


@pytest.mark.parametrize('mode',['complete','missing_probe','missing_root','traversal','symlink','oversize'])
def test_two_original_failure_projections_export_bounded_external_evidence(tmp_path,mode):
    root=external_fixture(tmp_path,mode)
    zip_path=tmp_path/'debug.zip'
    create_evaluation_debug_pack(root,zip_path,max_small_file_bytes=4096)
    with zipfile.ZipFile(zip_path) as z:
        manifest=json.loads(z.read('debug_pack_manifest.json'))
        diag=manifest['external_remote_diagnostics']
        assert len(diag['targets'])==2
        assert diag['complete'] is (mode=='complete')
        if mode!='complete':assert manifest['complete'] is False
        members=[name for name in z.namelist() if name.startswith('external_remote_diagnostics/')]
        data=b'\n'.join(z.read(name) for name in members)
        assert b'hidden_password' not in data and b'secret_value' not in data and b'DO_NOT_EXPORT' not in data
        assert not any(name.endswith(('.onnx','.npy','.hef','.parquet')) for name in z.namelist())
        if mode in ('complete','missing_probe','symlink','oversize'):
            assert sum('pre_mkdir_storage' in name for name in members)==2
        if mode in ('missing_probe','missing_root'):assert len(diag['missing'])==2


class Var:
    def __init__(self,value=''):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value


class Widget:
    def configure(self,**kw):pass
    def delete(self,*a):pass
    def insert(self,*a):self.text=a[-1]


def gui_profile(tmp_path,monkeypatch):
    import matplotlib
    monkeypatch.setattr(matplotlib,'use',lambda *a,**k:None)
    gui=importlib.import_module('onnx_splitpoint_tool.gui.app')
    panel=gui.panel_evaluation_workflow
    base=Path(os.environ.get('R7_PRIVATE_PROFILE','/home/kmika/.local/share/onnx-splitpoint-codex/v283_tagreparatur_R6_20260915/CompleteSetDev_v283_R6_private.yaml'))
    profile=yaml.safe_load(base.read_text())
    path=tmp_path/'private.yaml';path.write_text(yaml.safe_dump(profile,sort_keys=False))
    app=SimpleNamespace(var_eval_workflow_profile=Var(str(path)),var_eval_workflow_out_root=Var(str(tmp_path/'new_runs')),
                        var_eval_workflow_status=Var(),_eval_workflow_default_out_root=lambda:tmp_path/'new_runs')
    for name in ('_eval_workflow_load_profile_payload','_eval_workflow_snapshot_options','_eval_workflow_bool','_eval_workflow_remote_host_payload_from_profile'):
        setattr(app,name,MethodType(getattr(gui.SplitPointAnalyserGUI,name),app))
    def refresh():
        lines,snapshot=panel._profile_summary_payload(str(path))
        assert panel._commit_profile_summary(app,Widget(),'\n'.join(lines),snapshot)
        return lines,snapshot
    return gui,panel,app,path,profile,refresh


def test_actual_gui_summary_snapshot_queue_stops_before_run_creation(tmp_path,monkeypatch):
    gui,panel,app,path,profile,refresh=gui_profile(tmp_path,monkeypatch)
    lines,summary=refresh()
    assert summary,lines
    assert any('Native collector:' in line for line in lines)
    assert any('Native campaign budget:' in line and 'enabled' in line for line in lines)
    seen={}
    original=app._eval_workflow_snapshot_options
    def options(**kw):
        result=original(**kw);seen['options']=result;return result
    app._eval_workflow_snapshot_options=options
    class StopBeforeRun(BaseException):pass
    mkdir=Path.mkdir
    def capture(self,*a,**k):
        if self==tmp_path/'new_runs':raise StopBeforeRun()
        return mkdir(self,*a,**k)
    monkeypatch.setattr(Path,'mkdir',capture)
    monkeypatch.setattr(gui.messagebox,'askyesno',lambda *a,**k:True)
    with pytest.raises(StopBeforeRun):gui.SplitPointAnalyserGUI._queue_evaluation_workflow(app,resume=False)
    opts=seen['options'];resolved=opts.profile_start_snapshot['resolved_profile']
    assert opts.resume is False and not (tmp_path/'new_runs').exists()
    assert resolved['execution_preset']['id']=='final'
    assert resolved['execution_preset']['follow_tool_config'] is False
    assert resolved['execution_preset']['effective']['bootstrap_repetitions']==5000
    assert resolved['execution_preset']['effective']['validation_items']=={'classification':5000,'detection':5000}
    assert opts.hardware_setups_file==profile['hardware']['setups_file']
    native=resolved['native_producers']
    assert native['energy']['enabled'] is True
    args=campaign_budget_profile_args(native,tmp_path/'new_runs'/'NEW_RUN')
    assert args[args.index('--campaign-budget-file')+1]==str(tmp_path/'new_runs/NEW_RUN/energy_task_budget.json')
    assert '--campaign-max-transport-failures' in args
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    runner=EvaluationWorkflowRunner(opts);runner._load_profile()
    assert runner.profile_payload['native_producers']['energy']['task_budget']==native['energy']['task_budget']
    assert not (tmp_path/'new_runs').exists()
    evidence=os.environ.get('R7_GUI_EVIDENCE')
    if evidence:
        Path(evidence).write_text(json.dumps({'summary':lines,'options':opts.to_dict(),
                                             'budget_args':args,'dispatch_started':False},indent=2,default=str))


def test_gui_changed_mode_after_visible_summary_blocks_start(tmp_path,monkeypatch):
    gui,panel,app,path,profile,refresh=gui_profile(tmp_path,monkeypatch);refresh()
    profile['execution_preset']['id']='standard'
    path.write_text(yaml.safe_dump(profile,sort_keys=False))
    with pytest.raises(ValueError,match='changed|resolved'):app._eval_workflow_snapshot_options(resume_override=False)


def test_gui_missing_collector_visible_and_start_rejected(tmp_path,monkeypatch):
    gui,panel,app,path,profile,refresh=gui_profile(tmp_path,monkeypatch)
    registry=yaml.safe_load(Path(profile['hardware']['setups_file']).read_text())
    registry['energy_defaults']['collector_binary']=str(tmp_path/'missing-collector')
    registry_path=tmp_path/'registry.yaml';registry_path.write_text(yaml.safe_dump(registry))
    profile['hardware']['setups_file']=str(registry_path);path.write_text(yaml.safe_dump(profile,sort_keys=False))
    lines,visible=refresh()
    assert not visible and 'Native collector unavailable' in '\n'.join(lines)
    with pytest.raises(ValueError):app._eval_workflow_snapshot_options(resume_override=False)


def test_actual_gui_failure_callbacks_show_primary_and_cleanup(tmp_path,monkeypatch):
    from test_v283_r7_remote_shutdown import make_runner
    from test_v27930_terminal_progress import _QueuedRoot,_Monitor,options_for
    import concurrent.futures
    import threading
    gui,panel,unused,path,profile,refresh=gui_profile(tmp_path,monkeypatch)
    runner=make_runner(tmp_path,monkeypatch)
    release=threading.Event();executor=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix='r7-gui-reference')
    runner._management_reference_executor=executor
    runner._management_reference_futures['yolo26s']=executor.submit(release.wait)
    bounded=runner._shutdown_management_services_bounded
    monkeypatch.setattr(runner,'_shutdown_management_services_bounded',lambda **kw:bounded(timeout_s=.05))
    try:
        error=runner._finalize_owned_processes()
    finally:
        release.set();executor.shutdown();runner._management_shutdown_thread.join(2)
    def fail():raise error
    runner.run=fail
    monkeypatch.setattr(gui,'EvaluationWorkflowRunner',lambda *a,**kw:runner)
    dialogs=[]
    for name in ('showinfo','showwarning','showerror'):
        monkeypatch.setattr(gui.messagebox,name,lambda *a,**kw:dialogs.append(a))
    app=SimpleNamespace(root=_QueuedRoot(),_background_jobs={},_background_job_order=[],_gui_closing=False,
        _JOB_STATUS_LABELS=gui.SplitPointAnalyserGUI._JOB_STATUS_LABELS,var_eval_workflow_status=Var(),
        _jobs_refresh_views=lambda:None,_eval_workflow_snapshot_options=lambda **kw:options_for(tmp_path),
        _eval_workflow_command_preview=lambda opts:'controlled no-dispatch GUI test',
        _eval_workflow_text_set=lambda text:None,_eval_workflow_text_append=lambda text:None,
        _eval_workflow_render_result=lambda payload:None)
    for name in ('_jobs_register','_jobs_append_log','_jobs_set_progress','_jobs_finish','_jobs_request_cancel',
                 '_jobs_status_label','_jobs_workflow_status_to_gui','_jobs_handle_workflow_job_event','_jobs_parse_iso_datetime'):
        setattr(app,name,MethodType(getattr(gui.SplitPointAnalyserGUI,name),app))
    app._jobs_open_monitor=lambda job_id:setattr(app._background_jobs[job_id],'monitor',_Monitor())
    job=gui.SplitPointAnalyserGUI._queue_evaluation_workflow(app,resume=False)
    app._background_jobs[job].worker_thread.join(5);app.root.drain()
    assert app._background_jobs[job].status=='error'
    text=str(dialogs)
    for token in ('yolo26s','orin_nx_hailo10_01','before_uncached_suite_upload','rc=124','management_reference_executor.shutdown_join','owner_pid='):
        assert token in text


def test_valid_collector_change_after_summary_is_rejected(tmp_path,monkeypatch):
    gui,panel,app,path,profile,refresh=gui_profile(tmp_path,monkeypatch)
    registry=yaml.safe_load(Path(profile['hardware']['setups_file']).read_text())
    registry_path=tmp_path/'private_registry.yaml';registry_path.write_text(yaml.safe_dump(registry))
    profile['hardware']['setups_file']=str(registry_path);path.write_text(yaml.safe_dump(profile,sort_keys=False))
    assert refresh()[1]
    registry['energy_defaults']['collector_binary']='/bin/true'
    registry_path.write_text(yaml.safe_dump(registry))
    with pytest.raises(ValueError,match='changed'):app._eval_workflow_snapshot_options(resume_override=False)


def test_plan_only_needs_no_collector_and_missing_budget_is_visible(tmp_path,monkeypatch):
    gui,panel,app,path,profile,refresh=gui_profile(tmp_path,monkeypatch)
    registry_path=tmp_path/'private_registry.yaml';registry_path.write_text('energy_defaults:\n  collector_binary: /missing\n')
    profile['hardware']['setups_file']=str(registry_path)
    profile['native_producers']['energy']['mode']='plan'
    profile['native_producers']['energy'].pop('task_budget')
    lines=panel.profile_energy_start_binding(profile)
    assert any('disabled / not configured' in line for line in lines)
    assert campaign_budget_profile_args(profile['native_producers'],tmp_path/'NEW')==[]


@pytest.mark.parametrize('case',['quoted_secrets','empty_placeholder','malformed_extra','output_limit'])
def test_external_export_redaction_and_evidence_completeness(tmp_path,case):
    root=external_fixture(tmp_path,'missing_probe')
    directories=list((tmp_path/'RemoteBenchmarkRuns/Results/legacy_suite/1').glob('*/diagnostics'))
    for diag in directories:
        if case=='quoted_secrets':
            (diag/'timeout.json').write_text(json.dumps({'authorization':'secret-auth','api_key':'secret-api'}))
            (diag/'timeout.log').write_text('"password": "secret-json"\npassword=\'secret with spaces\'')
        elif case=='output_limit':
            (diag/'timeout.json').write_text(json.dumps({'padding':'é'*500},ensure_ascii=False))
        else:
            (diag/'before_uncached_suite_upload_remote_storage_probe_failure.json').write_text(
                '{}' if case=='empty_placeholder' else json.dumps({'extra':[], 'rc':124,'stage':'before_uncached_suite_upload','command':'probe'}))
    zip_path=tmp_path/'debug.zip';create_evaluation_debug_pack(root,zip_path,max_small_file_bytes=2048)
    with zipfile.ZipFile(zip_path) as z:
        manifest=json.loads(z.read('debug_pack_manifest.json'))
        assert manifest['external_remote_diagnostics']['complete'] is False
        data=b''.join(z.read(n) for n in z.namelist() if n.startswith('external_remote_diagnostics/'))
        for secret in (b'secret-auth',b'secret-api',b'secret-json',b'secret with spaces'):assert secret not in data
        if case=='output_limit':assert manifest['external_remote_diagnostics']['rejected']
