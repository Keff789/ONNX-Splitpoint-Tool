"""Owned tiny local fixtures only: no SSH, SDK, dataset or real run mutation."""
import json
import os
import shutil
import shlex
import subprocess
from pathlib import Path

import pytest

from onnx_splitpoint_tool.remote import collection_recovery as mod
from onnx_splitpoint_tool.workflow import run_control


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return mod._descriptor(path)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda: tmp_path / 'platform.lock')
    run = tmp_path / 'runs/TEST_ONLY_run'; run.mkdir(parents=True)
    session, setup_id, model = 'a' * 32, 'TEST_ONLY_setup', 'TEST_ONLY_model'
    remote_id = run.name + '_' + model + '_' + setup_id
    remote = tmp_path / 'splitpoint_runs/legacy_suite' / remote_id / '1'
    suite = remote / 'suite'; (suite / 'b071/results_TEST_ONLY').mkdir(parents=True)
    (suite / 'b071/results_TEST_ONLY/raw.json').write_text('TEST_ONLY original raw\n')
    (suite / 'input_TEST_ONLY.bin').write_bytes(bytes(range(32)))
    (suite / 'benchmark_set.json').write_text('{"sample_count":5000}')
    (suite / 'benchmark_plan.json').write_text('{"case_ids":["b071"]}')
    identity = dict(run_id=run.name, run_dir=str(run), session_id=session,
                    setup_id=setup_id, model_id=model)
    pid = 999999999
    original = {}
    original['manifest'] = put(run / 'run_manifest.json', {
        'schema':'onnx-splitpoint/evaluation-run-manifest', 'status':'partial',
        'run_id':run.name, 'run_dir':str(run), 'current_session_id':session,
        'options':{'energy_enabled':False,'native_producer_enabled':False},
        'execution_sessions':[{'session_id':session,'status':'partial','finished_at':'2026-01-01T00:00:00Z',
                               'start_origin':{'pid':pid}}]})
    original['profile'] = put(run / 'profile.yaml', {'run_profiles':[{
        'enabled':True,'quality_source_only':True,'hardware_setup_id':setup_id,'case_ids':['b071']}]})
    result = run / 'models' / model / 'benchmark_results'
    endpoint = dict(host='127.0.0.222', user='TEST_ONLY', port=22)
    original['dispatch'] = put(result / f'remote_benchmark_dispatch_{setup_id}.json', {
        'schema':'onnx-splitpoint/remote-benchmark-dispatch','model_id':model,'hardware_target_id':setup_id,
        'run_id':remote_id,'host':endpoint,
        'args':{'quality_evidence_eval_id':run.name,'validation_max_images':5000}})
    raw = {role:put(run / ('TEST_ONLY_' + role + '.json'), {'sample_count':5000,'role':role})
           for role in ('request','candidate','child_report')}
    for role, desc in raw.items(): shutil.copyfile(desc['path'], suite / ('TEST_ONLY_' + role + '.json'))
    transport_root = run.parent / 'RemoteBenchmarkRuns/Results/legacy_suite/1' / remote_id
    original['transport_metadata'] = put(transport_root/'run_meta.json', {
        'run_id':remote_id,'repeat_idx':'1','host':endpoint})
    for name in ('benchmark_set.json','benchmark_plan.json'):
        p=transport_root/'results'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(suite/name,p);original[name]=mod._descriptor(p)
    original['status'] = put(result / f'remote_benchmark_status_{setup_id}.json', {
        'status':'failed','model_id':model,'hardware_target_id':setup_id,'local_run_dir':str(transport_root),
        'remote_output':{'remote_run_dir':str(remote)},
        'generic_quality_source_capture':{'status':'verified_exact','errors':[], 'count':1,'expected':1,
            'evidence':[{'identity':{'case_id':'b071','setup_id':setup_id,'model_id':model},'image_count':5000,**raw}]}})
    original['artifact_index'] = put(run / 'artifact_index.json', {
        'schema':'onnx-splitpoint/artifact-index','schema_version':2,'run_id':run.name,
        'terminal_closure':{'schema':'onnx-splitpoint/artifact-index-terminal-closure-binding',
                           'run_id':run.name,'session_id':session,'status':'pass',
                           'report_path':'reports/artifact_index_closure.json'},
        'artifacts':[{'path':str(Path(r['path']).relative_to(run)),
                      'sha256':'sha256:'+r['sha256'],'size_bytes':r['size_bytes']} for r in raw.values()]})
    original['closure'] = put(run / 'reports/artifact_index_closure.json', {
        'schema':'onnx-splitpoint/artifact-index-terminal-closure','status':'pass','run_id':run.name,
        'session_id':session,**{k:0 for k in ('missing_file_count','hash_mismatch_count',
                                           'unindexed_required_path_count','verification_error_count')}})
    (run / 'jobs/remote_process_leases' / session).mkdir(parents=True)
    lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run,
        owner={'run_id':run.name,'run_dir':str(run),'session_id':session,'pid':pid})
    lock.acquire(); lock.release()
    original['run_lock'] = mod._descriptor(lock.lock_path)
    resource = 'dut:127.0.0.222'
    physical = run_control.EvaluationRunLock.for_resource(resource, owner={'purpose':'generic'})
    physical.acquire(); physical.release()
    registry = tmp_path / 'registry.json'
    put(registry, {'hardware_setups':[{'id':setup_id,'remote':{
        **endpoint,'remote_base_dir':str(tmp_path / 'splitpoint_runs')}}]})
    monkeypatch.setattr('onnx_splitpoint_tool.platform_power.host_config_from_setup',
        lambda row: mod.SSHTransport.__test_host__)
    from onnx_splitpoint_tool.remote.ssh_transport import HostConfig
    host = HostConfig(id=setup_id,label='TEST_ONLY',**endpoint,remote_base_dir=str(tmp_path / 'splitpoint_runs'))
    monkeypatch.setattr(mod, 'observe_resource_owners', lambda *_a, **_k: {'ok':True,'read_only':True})
    calls = []
    class LocalTransport:
        __test_host__ = host
        def __init__(self, *_a, **_k): pass
        def run(self, command, timeout_s):
            calls.append(command)
            p = subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=10)
            return p.returncode,p.stdout+p.stderr
        def run_read_only(self, command, timeout):
            p=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=10)
            return p.returncode,p.stdout+p.stderr
        def scp_download(self, source, destination, timeout_s):
            shutil.copyfile(source,destination); return 0,''
    monkeypatch.setattr(mod,'SSHTransport',LocalTransport)
    evidence = tmp_path / 'evidence.json'
    put(evidence, {'schema':'onnx-splitpoint/generic-closed-collection-recovery-input',
                  'schema_version':1,**identity,'original':original})
    auth = tmp_path / 'auth.json'
    put(auth, {'schema':'onnx-splitpoint/generic-closed-collection-recovery-authorization',
        'schema_version':1,'authorized':True,'action':'recover_generic_closed_collection',**identity,
        'operator':'TEST_ONLY','source':'TEST_ONLY user instruction','authorized_at':'2026-01-01T00:00:00Z',
        'evidence':mod._descriptor(evidence),'endpoint':endpoint,'remote_run_dir':str(remote),'cleanup':True})
    value=json.loads(auth.read_text());value['capacity_budget']={
        'max_archive_bytes':10*1024*1024,'local_reserve_bytes':0,'remote_reserve_bytes':0};put(auth,value)
    kwargs = dict(run_dir=run,session_id=session,setup_id=setup_id,model_id=model,
                  evidence_path=evidence,authorization_path=auth,registry_path=registry,output_dir=tmp_path/'new-output')
    originals = {p:p.read_bytes() for p in [lock.lock_path,physical.lock_path,*run.rglob('*.json'),run/'profile.yaml']}
    return kwargs, remote, originals, calls


def test_full_owned_archive_stream_verified_then_cleanup_preserves_original_history(case):
    kw, remote, originals, calls = case
    result = mod.recover_generic_closed_collection(**kw, apply=True, cleanup=True)
    assert result['status'] == 'completed' and result['fresh_collection_rc'] == 0
    assert result['remote_cleanup'] == 'removed_exact_verified_run' and not remote.exists()
    assert all(p.read_bytes() == value for p,value in originals.items())
    archive = Path(result['archive']['path'])
    manifest = mod._verify_archive(archive,result['archive'],str(remote))
    assert any(r['path']=='suite/input_TEST_ONLY.bin' for r in manifest['files'])
    assert not (Path(kw['output_dir'])/'preserved').exists()  # no large extraction
    assert len(calls)==4  # preflight, collector, archive, exact cleanup; no runner API exists on the leaf


@pytest.mark.parametrize('fault',['active','wrong_run','changed_bytes','unknown_link',
                                 'foreign_result_link','output_escape','remote_wrong_suite_bytes'])
def test_active_misbound_changed_and_unpreserved_content_never_cleanup(case, fault):
    kw, remote, originals, calls = case
    if fault == 'active':
        manifest=Path(kw['run_dir'])/'run_manifest.json'; value=json.loads(manifest.read_text());value['status']='running'
        put(manifest,value)
    elif fault == 'wrong_run': kw['session_id']='b'*32
    elif fault == 'changed_bytes': (Path(kw['run_dir'])/'TEST_ONLY_request.json').write_text('tampered')
    elif fault == 'unknown_link': (remote/'suite/unpreserved').symlink_to(remote/'suite/input_TEST_ONLY.bin')
    elif fault == 'foreign_result_link':
        foreign=remote.parent/'FOREIGN_TEST_ONLY';foreign.mkdir();(foreign/'keep').write_text('untouched')
        (remote/'results').symlink_to(foreign,target_is_directory=True)
    elif fault == 'output_escape':
        holder=Path(kw['run_dir']).parent/'holder';holder.mkdir()
        kw['output_dir']=holder/'..'/Path(kw['run_dir']).name/'recovery'
    else: (remote/'suite/benchmark_set.json').write_text('different scientific suite')
    with pytest.raises(Exception): mod.recover_generic_closed_collection(**kw,apply=True,cleanup=True)
    assert remote.is_dir()
    assert not any(shlex.split(c)[0] == 'python3' and shlex.split(c)[-5] == 'cleanup' for c in calls)
    if fault in ('unknown_link','foreign_result_link','remote_wrong_suite_bytes'):
        assert len(calls)==1  # owned read-only preflight rejects before collector writes
    if fault == 'foreign_result_link': assert (foreign/'keep').read_text()=='untouched'


def test_default_dryrun_no_transport_and_archive_tamper_rejected(case):
    kw, remote, originals, calls = case
    result = mod.recover_generic_closed_collection(**kw)
    assert result['status']=='dry_run_pass' and calls==[] and remote.is_dir()
    assert all(p.read_bytes()==value for p,value in originals.items())
    archive=remote/'.collection-recovery-TEST_ONLY.tar.gz'
    p=subprocess.run(['bash','-c',mod._archive_command('pack',str(remote),str(archive),
        budget={'max_archive_bytes':10*1024*1024,'remote_reserve_bytes':0})],capture_output=True,text=True)
    assert p.returncode==0
    expected=mod._descriptor(archive);archive.write_bytes(archive.read_bytes()[:-16])
    with pytest.raises(Exception): mod._verify_archive(archive,expected,str(remote))


def test_remote_content_change_after_archive_download_blocks_removal(case, monkeypatch):
    kw, remote, originals, _ = case
    original_run = mod.SSHTransport.run
    def change_before_cleanup(self, command, timeout_s):
        values = shlex.split(command)
        if values[0] == 'python3' and values[-5] == 'cleanup':
            (remote/'suite/input_TEST_ONLY.bin').write_bytes(b'changed after download')
        return original_run(self, command, timeout_s)
    monkeypatch.setattr(mod.SSHTransport, 'run', change_before_cleanup)
    with pytest.raises(Exception): mod.recover_generic_closed_collection(**kw,apply=True,cleanup=True)
    assert remote.is_dir() and all(p.read_bytes()==value for p,value in originals.items())


@pytest.mark.parametrize('where',['local_preflight','remote_writer_cap'])
def test_explicit_capacity_shortfall_blocks_transfer_or_removes_only_own_partial(case, where):
    kw, remote, originals, calls = case
    auth=Path(kw['authorization_path']);value=json.loads(auth.read_text())
    value['capacity_budget']['max_archive_bytes'] = 2**63 if where=='local_preflight' else 1
    put(auth,value)
    with pytest.raises(Exception): mod.recover_generic_closed_collection(**kw,apply=True,cleanup=True)
    assert remote.is_dir() and not list(remote.glob('.collection-recovery-*.tar.gz'))
    assert all(p.read_bytes()==data for p,data in originals.items())
    if where=='local_preflight': assert calls==[]
