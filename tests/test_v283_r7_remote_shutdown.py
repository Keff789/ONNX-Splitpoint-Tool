"""R7 production admission/transport/finalizer regressions; no remote hardware."""
import concurrent.futures
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.process_control import ProcessTreeRegistry, bind_process_registry
from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport, HostConfig
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
from onnx_splitpoint_tool.quality_service import ManagementQualityService, QualityServiceClosedError
from test_v2804_quality_cancel import request
from test_v27539_remote_premutation_dispatch import _run
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry

VALID = dict(read_only_mount=False, permission_bits_allow=True, os_access_allow=True,
             free_bytes=10**15, free_inodes=10**9)


class AnswerTransport:
    def __init__(self, payload=VALID, rc=0, output=None):
        self.rc = rc
        self.output = 'SPLITPOINT_STORAGE_JSON=' + json.dumps(payload) if output is None else output
    def run_read_only(self, command, timeout):
        self.command = command
        return self.rc, self.output


def probe(transport):
    return remote_run._remote_storage_preflight(transport, '/remote', required_free_bytes=100,
                                               required_free_inodes=10, stage='before_uncached_suite_upload')


def test_real_storage_contract_admits_valid_capacity():
    assert probe(AnswerTransport())['ok'] is True


@pytest.mark.parametrize('payload,output,kind', [
    ({**VALID, 'free_bytes': 1}, None, 'capacity_or_access_rejected'),
    ({**VALID, 'free_inodes': 1}, None, 'capacity_or_access_rejected'),
    ({**VALID, 'free_bytes': 'bad'}, None, 'probe_malformed'),
    ({**VALID, 'free_inodes': []}, None, 'probe_malformed'),
    ({k:v for k,v in VALID.items() if k!='read_only_mount'}, None, 'probe_malformed'),
    ({}, '', 'probe_malformed'), ({}, 'nonsense', 'probe_malformed'),
])
def test_storage_rejects_capacity_and_malformed_before_upload(tmp_path, monkeypatch, payload, output, kind):
    class Target(AnswerTransport):
        def __init__(self, *a, **kw):super().__init__(payload, output=output)
        def resolve_path_read_only(self,*a,**kw):return '/remote'
        def __getattr__(self, name):
            if name == 'read_only_diagnostics':return {}
            raise AssertionError('No upload/model/followup permitted: '+name)
    monkeypatch.setattr(remote_run,'SSHTransport',Target)
    monkeypatch.setattr(remote_run,'refresh_suite_harness',lambda *a,**k:{'changed':False})
    result=_run(tmp_path,registry=RemoteProcessLeaseRegistry(),logs=[])
    assert result['failure_kind']=='terminal_remote_storage_failure'
    assert kind in result['primary_failure']['primary_error']
    assert result['remote_dispatched'] is False
    reports=list(tmp_path.rglob('*remote_storage_probe_failure.json'))
    assert len(reports)==1
    assert json.loads(reports[0].read_text())['extra']['failure_kind']==kind


def test_probe_timeout_retains_phase_and_no_capacity_claim():
    with pytest.raises(remote_run.RemoteStoragePreflightError) as caught:
        probe(AnswerTransport(rc=124,output=''))
    diagnostic=caught.value.diagnostic
    assert diagnostic['rc']==124 and diagnostic['stage']=='before_uncached_suite_upload'
    assert diagnostic['remote_phase']=='NOT_RECORDED'
    assert 'statvfs' in diagnostic['command']
    assert 'capacity_or_access_rejected' not in str(caught.value)


def parallel_transport_probe(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True)
    registry=ProcessTreeRegistry()
    gate=threading.Barrier(3)
    # A controlled shared bundle lock stands between simultaneous callers and
    # their real local transport children. Same product lock, no model suite.
    suite=tmp_path/'suite';suite.mkdir()
    (suite/'data.txt').write_text('controlled shared bundle')
    child_marker=tmp_path/'transport_child.pid'
    from onnx_splitpoint_tool.remote.bundle import build_suite_bundle
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'])
    class LocalTransport(SSHTransport):
        def _ssh_cmd(self,command,env=None):
            if self.host.id=='h8':
                return [sys.executable,'-u','-c',
                        f"import subprocess,sys,time;from pathlib import Path;print('SPLITPOINT_STORAGE_PHASE=python_started',flush=True);p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)']);Path({str(child_marker)!r}).write_text(str(p.pid));time.sleep(15)"]
            return ['bash','-c',command.replace('/remote',str(tmp_path))]
        def run_read_only(self,command,timeout=None,**kw):
            return super().run_read_only(command,timeout=.4)
    def call(target):
        transport=LocalTransport(HostConfig(id=target,label=target,host='local'))
        gate.wait(3)
        build_suite_bundle(suite,tmp_path/"shared.tar.gz")
        with bind_process_registry(registry):
            try: result=probe(transport)
            except remote_run.RemoteStoragePreflightError as exc: result=exc.diagnostic
        return result,transport.read_only_diagnostics
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            a=pool.submit(call,'h8');b=pool.submit(call,'h10');gate.wait(3)
            start=time.monotonic();ra,da=a.result(8);rb,db=b.result(8)
        assert time.monotonic()-start<7
        assert ra['rc']==124 and rb['ok'] is True
        assert da['phase']=='process_wait_timeout' and db['phase']=='completed'
        assert da['local_pid']!=db['local_pid']
        assert da['target_id']=='h8' and db['target_id']=='h10'
        assert da['local_completion_proven'] and db['local_completion_proven']
        assert da['local_exit_code'] is not None and da['retry_permitted'] is False
        assert unrelated.poll() is None
        registry.assert_quiescent()
        child_pid=int(child_marker.read_text())
        child_stat=Path(f'/proc/{child_pid}/stat')
        assert not child_stat.exists() or child_stat.read_text().split(') ',1)[1].startswith('Z ')
    finally:
        registry.terminate_all(grace_s=.2)
        unrelated.terminate();unrelated.wait(timeout=3)
    return ra,da


def make_runner(tmp_path, monkeypatch):
    monkeypatch.setattr('onnx_splitpoint_tool.workflow.runner.package_build_snapshot',lambda:{})
    runner=EvaluationWorkflowRunner(WorkflowOptions(profile='',out=str(tmp_path)))
    runner.run_dir=tmp_path/'run';runner.run_dir.mkdir()
    runner.run_id='r7-controlled';runner._stop_requested=True
    (runner.run_dir/'primary_failure.json').write_text(json.dumps({
        'model_id':'yolo26s','hardware_target_id':'orin_nx_hailo10_01',
        'primary_error':'remote_storage_preflight_failed: stage=before_uncached_suite_upload, rc=124'}))
    return runner


def test_terminal_failure_stops_real_quality_workers_and_queued_coordinators(tmp_path,monkeypatch):
    runner=make_runner(tmp_path,monkeypatch)
    module=tmp_path/'r7_quality_worker.py';marker=tmp_path/'worker.json'
    module.write_text('''import json,os,time
from pathlib import Path
class Evaluator:
 def __init__(self,r,c,a,cfg):self.marker=a['marker']
 def evaluate(self,m):
  Path(self.marker).write_text(json.dumps({'pid':os.getpid()}));time.sleep(15)
  return {'candidate':.9,'reference':1.,'delta':-.1}
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    service=ManagementQualityService(tmp_path/'cache',workers=1)
    runner._central_quality_service=service
    coord=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix='r7-coordinator')
    runner._central_quality_coord_executor=coord
    req=replace(request(),evaluator_factory='r7_quality_worker:Evaluator',annotations={'marker':str(marker)})
    running=coord.submit(service.evaluate,req)
    runner._central_quality_futures['running']=running
    try:
        deadline=time.monotonic()+8
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists()
        processes=list(service._executor._processes.values())
        queued=coord.submit(service.evaluate,replace(req,seed=100))
        runner._central_quality_futures['queued']=queued
        # Real concurrent transport/storage failure while the production
        # management pool is busy. Inject only the hardware dispatch result.
        failure,transport=parallel_transport_probe(tmp_path/'transport')
        import onnx_splitpoint_tool.workflow.runner as runner_module
        from onnx_splitpoint_tool.workflow.execution_binding import ExecutionBindingResult
        suite=runner.run_dir/'models/yolo26s/benchmark_set';suite.mkdir(parents=True)
        (suite/'benchmark_set.json').write_text('{}')
        primary={'failure_kind':'terminal_remote_storage_failure','remote_rc':124,
                 'primary_error':f"remote_storage_preflight_failed: stage={failure['stage']}, rc={failure['rc']}"}
        status_path=runner.run_dir/'remote_status.json'
        status_path.write_text(json.dumps({'hardware_target_id':'orin_nx_hailo8_01','remote_output':{
            'terminal_remote_failure':True,'remote_rc':124,'primary_failure':primary}}))
        runner.profile_payload={};runner.manifest={'models':{'yolo26s':{'task':'detection'}}}
        runner.artifact_index={'artifacts':[]};runner.artifact_index_path=runner.run_dir/'artifact_index.json'
        runner._stop_requested=False
        monkeypatch.setattr(runner,'_schedule_management_cpu_reference',lambda *a,**kw:None)
        monkeypatch.setattr(runner_module,'benchmark_set_postcondition_v60v',lambda *a,**kw:{'valid':True,'selected_suite_dir':str(suite)})
        monkeypatch.setattr(runner_module,'finalize_suite_for_runtime',lambda *a,**kw:{'benchmark_plan':{}})
        monkeypatch.setattr(runner_module,'execute_benchmark_suite_if_requested',lambda **kw:ExecutionBindingResult(
            artifacts={'remote_benchmark_status_h8_json':status_path},metrics={},status='failed',message=primary['primary_error']))
        assert runner._stage_run_benchmarks('yolo26s',{'id':'yolo26s'})[-1]=='failed'
        assert runner._stop_requested and runner._management_admission_closed
        start=time.monotonic();error=runner._finalize_owned_processes()
        assert error is None
        assert time.monotonic()-start<7
        assert running.done() and queued.cancelled()
        assert all(not p.is_alive() for p in processes)
        assert service.shutdown_state()['finished']
        assert not runner._cancel_event.is_set()
        assert runner._finalize_owned_processes() is None
        assert runner._shutdown_management_services_bounded()['finished']
        with pytest.raises(QualityServiceClosedError):runner._ensure_central_quality_service()
        assert not runner._cleanup_quarantine_path.exists()
    finally:
        service.shutdown(cancel_futures=True,terminate_workers=True)
        coord.shutdown(cancel_futures=True)


def test_unresolved_owned_thread_keeps_primary_cleanup_and_reentrant_identity(tmp_path,monkeypatch):
    runner=make_runner(tmp_path,monkeypatch)
    release=threading.Event();executor=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix='r7-owned-reference')
    runner._management_reference_executor=executor
    future=executor.submit(release.wait);runner._management_reference_futures['yolo26s']=future
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    monkeypatch.setattr('onnx_splitpoint_tool.workflow.run_control.platform_workflow_interlock_path',lambda:tmp_path/'platform.lock')
    run_lock=EvaluationRunLock(out_root=tmp_path,run_dir=runner.run_dir,owner={'session_id':runner.session_id})
    run_lock.acquire();runner._run_lock=run_lock
    original=runner._shutdown_management_services_bounded
    monkeypatch.setattr(runner,'_shutdown_management_services_bounded',lambda **kw:original(timeout_s=.05))
    try:
        error=runner._finalize_owned_processes()
        text=str(error)
        for expected in ('yolo26s','orin_nx_hailo10_01','before_uncached_suite_upload','rc=124','management_reference_executor.shutdown_join','owner_pid='):
            assert expected in text
        assert error.owner['details']['management_service_shutdown']['finished'] is False
        assert runner._cleanup_quarantine_path.exists() and run_lock.quarantine_path.exists()
        thread=runner._management_shutdown_thread
        assert runner._finalize_owned_processes() is error
        assert runner._management_shutdown_thread is thread
        assert error.owner['details']['management_service_shutdown']['management_reference']['threads'][0]['alive']
    finally:
        release.set();future.result(2);executor.shutdown();runner._management_shutdown_thread.join(2);run_lock.release()
    assert original(timeout_s=1)['finished']


@pytest.mark.parametrize('answer',[124,'malformed'])
def test_post_bundle_admission_failure_suppresses_upload_and_preserves_rc(tmp_path,monkeypatch,answer):
    instances=[]
    class Target:
        def __init__(self,*a,**kw):self.probes=0;self.uploads=0;self.models=0;instances.append(self)
        def resolve_path_read_only(self,*a,**kw):return '/remote'
        def run_read_only(self,command,**kw):
            if 'SPLITPOINT_STORAGE_JSON=' in command:
                self.probes+=1
                if self.probes==2:
                    return (124,'') if answer==124 else (0,'SPLITPOINT_STORAGE_JSON='+json.dumps({**VALID,'free_bytes':'bad'}))
                return 0,'SPLITPOINT_STORAGE_JSON='+json.dumps(VALID)
            return 0,'ok'
        def run(self,*a,**kw):return 0,'ok'
        def scp_upload(self,*a,**kw):self.uploads+=1;raise AssertionError('upload forbidden')
        def scp_download(self,*a,**kw):raise AssertionError('download forbidden')
        def run_streaming(self,*a,**kw):self.models+=1;raise AssertionError('model forbidden')
    monkeypatch.setattr(remote_run,'SSHTransport',Target)
    monkeypatch.setattr(remote_run,'refresh_suite_harness',lambda *a,**kw:{'changed':False})
    result=_run(tmp_path,registry=RemoteProcessLeaseRegistry(),logs=[])
    assert instances[0].probes==2
    assert instances[0].uploads==instances[0].models==0
    assert result['failure_kind']=='terminal_remote_storage_failure'
    assert result['remote_rc']==(124 if answer==124 else 0)
    reports=list(tmp_path.rglob('before_uncached_suite_upload_remote_storage_probe_failure.json'))
    assert len(reports)==1
    assert json.loads(reports[0].read_text())['stage']=='before_uncached_suite_upload'


def test_real_parallel_callers_shared_bundle_and_management_child(tmp_path):
    parallel_transport_probe(tmp_path)


def test_late_quality_callback_preserves_already_written_result_after_seal(tmp_path,monkeypatch):
    runner=make_runner(tmp_path,monkeypatch)
    request_path=runner.run_dir/'full_request.json';request_path.write_text('{}')
    result_path=runner.run_dir/'full_central_result.json';result_path.write_text('{"status":"completed","decision":"fail"}')
    before=result_path.read_bytes()
    runner._close_management_admission();runner._terminal_sealing=True
    monkeypatch.setattr(runner,'_quality_request_identity',lambda *a,**kw:{'identity_valid':False})
    result=runner._evaluate_central_quality_request('yolo26s',request_path)
    assert result_path.read_bytes()==before
    assert result['technical_status']=='failed'
    assert runner._central_quality_service is None
