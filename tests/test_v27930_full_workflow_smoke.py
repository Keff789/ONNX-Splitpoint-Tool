"""H2: complete normal process chain, only a fake physical DXRT device module."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import zipfile
import pytest
from scripts import deepx_full_workflow_smoke_v2804 as driver
from scripts import deepx_full_workflow_smoke_worker_v27930 as worker
from tests.test_v27930_deepx_semantic_pre_nms import ROOT, FIXTURES, prepare_semantic_case, write_json
from onnx_splitpoint_tool.runners.native_full_input import prepare_and_seal_deepx_native_full_input
from onnx_splitpoint_tool.release_identity import VERSION as CURRENT_VERSION


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def h2_dependency_paths(paths, source_root):
    """Share installed dependencies, including those in Tool/.venv, not source."""
    source_root = Path(source_root).resolve()
    dependencies = []
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if not path.is_dir() or path.resolve() == source_root:
            continue
        if path.name not in {'site-packages', 'dist-packages', 'test_dependencies'}:
            continue
        resolved = str(path.resolve())
        if resolved not in dependencies:
            dependencies.append(resolved)
    return dependencies


def test_h2_inherits_tool_venv_dependencies_without_importing_parent_source(tmp_path):
    project = tmp_path / 'parent_tool'
    parent_site = project / '.venv/lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    parent_site.mkdir(parents=True)
    (parent_site / 'h2_dependency_fixture.py').write_text("VALUE = 'installed_dependency'\n")
    (project / 'h2_parent_source_fixture.py').write_text("raise AssertionError('parent source imported')\n")
    # A dependency directory is inherited as a plain path: the parent's editable
    # installation must not add the unstaged source through its own .pth file.
    (parent_site / 'parent_editable.pth').write_text(str(project) + '\n')
    dependencies = h2_dependency_paths(
        [str(project), str(parent_site), str(parent_site), '', str(project / 'missing')], project)
    runtime = tmp_path / 'isolated_runtime'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    runtime_site = next((runtime / 'lib').glob('python*/site-packages'))
    (runtime_site / 'test_dependencies.pth').write_text('\n'.join(dependencies) + '\n')
    staged_tool = tmp_path / 'staged_tool'
    staged_tool.mkdir()
    (staged_tool / 'h2_product_fixture.py').write_text("ORIGIN = 'staged_tool'\n")
    (project / 'h2_product_fixture.py').write_text("raise AssertionError('unstaged product imported')\n")
    code = '''
import importlib.util, json, sys
from pathlib import Path
import h2_dependency_fixture
assert h2_dependency_fixture.VALUE == 'installed_dependency'
assert importlib.util.find_spec('h2_parent_source_fixture') is None
sys.path.insert(0, sys.argv[1])
import h2_product_fixture
assert h2_product_fixture.ORIGIN == 'staged_tool'
print(json.dumps({'dependency': str(Path(h2_dependency_fixture.__file__).resolve()),
                  'product': str(Path(h2_product_fixture.__file__).resolve()),
                  'prefix': sys.prefix}))
'''
    result = subprocess.run([str(runtime / 'bin/python'), '-I', '-B', '-c', code, str(staged_tool)],
                            cwd=project, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = json.loads(result.stdout)
    assert loaded['dependency'] == str((parent_site / 'h2_dependency_fixture.py').resolve())
    assert loaded['product'] == str((staged_tool / 'h2_product_fixture.py').resolve())
    assert Path(loaded['prefix']).resolve() == runtime.resolve()


def h2_stage(tmp_path, monkeypatch):
    case = prepare_semantic_case(tmp_path / 'original', monkeypatch)
    case.setup_id = 'orin_nx_deepx_m1_01'
    case.contract.update(endpoint_mode='decoded_pre_nms', contract_family='decoded_pre_nms',
        artifact_sha256=driver.probe.sha256(case.full / 'model.dxnn'))
    write_json(case.full / 'output_contract.json', case.contract)
    write_json(case.root / 'benchmark_set.json', {'schema':'onnx-splitpoint/benchmark-set','schema_version':2,
        'model_name':'yolo11l','model_id':'yolo11l','model':'models/yolo11l.onnx','task':'detection',
        'artifact_manifest':{'schema':'onnx-splitpoint/benchmark-set','schema_version':2,
            'files':{'models':['models/yolo11l.onnx']},'counts':{'models':1}},
        'cases':[{'case_id':'b003','folder':'b003'}]})
    write_json(case.root / 'benchmark_plan.json', {'model_id':'yolo11l','task':'detection', 'runs':[{
        'id':'deepx_m1_full', 'model_id':'yolo11l', 'backend':'deepx_m1', 'provider':'deepx_m1',
        'variant':'full', 'variants':['full'], 'full':'deepx_m1', 'type':'deepx',
        'benchmark_task':'detection', 'task':'detection', 'setup_id':case.setup_id,
        'dxnn_path':'deepx/deepx_m1/full/model.dxnn','contract_path':'deepx/deepx_m1/full/output_contract.json',
        'measurement_endpoint':'completed_detection','quality_endpoint':'completed_detection'}]})
    (case.root / 'b003').mkdir()
    (case.root / 'b003/run_split_onnxruntime.py').write_text('# original index marker; H2 never runs a split\n')
    package = ROOT / 'onnx_splitpoint_tool'
    shutil.copytree(package / 'runners', case.root / 'splitpoint_runners', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('native_output_endpoint.py','native_detection_postprocess.py','preprocessing_contract.py'):
        shutil.copyfile(package/name, case.root/'splitpoint_runners'/name)
    prepare_and_seal_deepx_native_full_input(image_path=case.image, input_contract=case.contract,
        task='detection',out_dir=case.root/'results/deepx_m1_full/prepared_input',model=case.model,
        setup_id=case.setup_id,comparison_backend='deepx')
    # Unchanged real D manifest is an independent expected binding, not an
    # expectation reconstructed from this test's freshly prepared bytes.
    shutil.copyfile(FIXTURES/'semantic_original_full_input_manifest.json',
                    case.root/'results/deepx_m1_full/prepared_input/native_full_input_manifest.json')
    runtime = tmp_path / 'runtime_venv'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    site = next((runtime/'lib').glob('python*/site-packages'))
    deps = h2_dependency_paths(sys.path, ROOT)
    (site/'test_dependencies.pth').write_text('\n'.join(deps)+'\n')
    (site/'sitecustomize.py').write_text("import sys\nsys.modules['cv2'] = None\n")
    calls = tmp_path/'dxrt_calls.jsonl'
    (site/'dx_engine.py').write_text(f'''
import json,os
from pathlib import Path
import numpy as np
class InferenceEngine:
    def __init__(self,path):
        assert Path(path).read_bytes().startswith(b'synthetic accelerator handle')
        self.path=path
    def run(self,feeds):
        assert len(feeds)==1 and feeds[0].shape==(640,640,3) and feeds[0].dtype==np.uint8
        with Path({str(calls)!r}).open('a') as f:
            f.write(json.dumps({{'pid':os.getpid(),'model':self.path,'input_sha256':__import__('hashlib').sha256(feeds[0].tobytes()).hexdigest()}})+'\\n')
        with np.load({str(FIXTURES/'semantic_probe_outputs.npz')!r},allow_pickle=False) as saved:
            return [saved['tensor_000']]
''')
    request = {'schema':driver.probe.REQUEST_SCHEMA, 'staging_mode':driver.probe.STAGING_MODE,
        'model_id':case.model,'setup_id':case.setup_id,'original_run_id':'synthetic_v27930_H2_fixture',
        'original_remote_suite':'/nonexistent/deleted/old/suite',
        'staged_dxnn_relative':driver.probe.DXNN_RELATIVE,'staged_image_relative':case.image.name,
        'expected_dxnn_sha256':driver.probe.sha256(case.full/'model.dxnn'),
        'expected_image_sha256':driver.probe.sha256(case.image),'runtime_venv':str(runtime)}
    sources = {'suite':case.root,'package':case.root/'splitpoint_runners','dxnn':case.full/'model.dxnn',
        'image':case.image,'template':package/'resources/templates/benchmark_suite.py.txt',
        'postprocess':package/'native_detection_postprocess.py',
        'worker':ROOT/'scripts/deepx_full_output_probe_worker_v27930.py'}
    files = driver.smoke_staging_files(request,sources,ROOT,tmp_path/'original')
    stage=tmp_path/'fresh_remote'
    stage.mkdir()
    driver.probe.prepare_stage(stage,request,files)
    return case, stage, runtime, calls


def test_h2_actual_normal_cli_semantic_child_benchmark_child_and_aggregation(tmp_path, monkeypatch):
    case, stage, runtime, calls = h2_stage(tmp_path,monkeypatch)
    before=inventory(case.root)
    run=subprocess.run([str(runtime/'bin/python'),'-I','-B',str(stage/'deepx_full_workflow_smoke_worker_v27930.py'),
                        '--request',str(stage/'probe_request.json')],capture_output=True,text=True,timeout=45)
    result=json.loads((stage/'results/deepx_full_workflow_smoke.json').read_text())
    detail=(stage/'results/normal_runner_console.log').read_text() if (stage/'results/normal_runner_console.log').is_file() else run.stdout+run.stderr
    assert run.returncode==0, json.dumps(result,indent=2)+'\n'+detail[-14000:]
    assert result['status']=='pass', result
    assert result['repetition_count_requested']==result['repetition_count_attempted']==result['repetition_count_valid']==1
    assert result['completed_frames']==result['postprocess_completed_frames']==100
    assert result['diagnostic_only'] is True and result['counts_as_benchmark'] is False
    events=[json.loads(x) for x in calls.read_text().splitlines()]
    assert len(events)==112  # semantic1 +untimed structure1 +warmup10 +timed100
    assert len({x['pid'] for x in events})==2
    assert len({x['input_sha256'] for x in events})==1
    assert all(str(stage/'native_root') in x['model'] for x in events)
    assert inventory(case.root)==before
    assert result['process']['owned_survivors']==[]
    assert not list((stage/'results').rglob('*.dxnn'))
    matrix = json.loads((stage/'results/suite_diagnostics/benchmark_suite_status_matrix.json').read_text())
    assert len(matrix) == 1 and matrix[0]['run'] == 'deepx_m1_full'
    assert matrix[0]['full'] == 'ok'
    assert matrix[0]['tag'] == 'deepx_m1_full_auto'


def test_h2_mismatched_original_input_stops_before_any_engine(tmp_path,monkeypatch):
    case,stage,runtime,calls=h2_stage(tmp_path,monkeypatch)
    original=driver.probe.read_json(stage/'original_full_input_manifest.json')
    original['runtime_input_sha256']='f'*64
    write_json(stage/'original_full_input_manifest.json',original)
    run=subprocess.run([str(runtime/'bin/python'),'-I','-B',str(stage/'deepx_full_workflow_smoke_worker_v27930.py'),
                        '--request',str(stage/'probe_request.json')],capture_output=True,text=True,timeout=20)
    result=driver.probe.read_json(stage/'results/deepx_full_workflow_smoke.json')
    assert run.returncode==2
    assert 'original_full_input_binding_mismatch' in result['error']
    assert not calls.exists()


def test_h2_supervisor_kills_and_reaps_grandchild_in_other_session(tmp_path):
    # Readiness is bounded and observed before the unchanged supervisor clock.
    run = subprocess.run([sys.executable, '-I', '-S', '-B',
        str(ROOT/'tests/v27931_supervisor_fixture.py'), '--directory', str(tmp_path)],
        capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((tmp_path/'supervision.json').read_text())
    assert report['grandchild_seen_alive'] is True
    assert report['readiness_status'] == 'ready'
    assert report['timed_out'] is True
    assert report['cleanup_complete'] is True
    assert report['owned_survivors'] == []
    assert report['grandchild_reaped'] is True
    import signal
    assert any(event['signal'] == signal.SIGKILL and report['grandchild_pid'] in event['pids']
               for event in report['signal_trace'])
    with pytest.raises(ProcessLookupError):
        os.kill(report['grandchild_pid'], 0)


def test_v27931_supervisor_never_ready_is_bounded_failure_with_cleanup(tmp_path):
    run = subprocess.run([sys.executable, '-I', '-S', '-B',
        str(ROOT/'tests/v27931_supervisor_fixture.py'), '--directory', str(tmp_path), '--never-ready'],
        capture_output=True, text=True, timeout=15)
    assert run.returncode == 2, run.stdout + run.stderr
    report = json.loads((tmp_path/'supervision.json').read_text())
    assert report['readiness_status'] == 'timeout'
    assert report['grandchild_seen_alive'] is False
    assert report['grandchild_pid'] is not None
    assert report['grandchild_reaped'] is True
    assert report['cleanup_complete'] is True and report['owned_survivors'] == []
    assert report['total_elapsed_s'] < 12


def test_h2_transport_timeout_ends_scp_style_child_group(tmp_path):
    child=tmp_path/'ssh_child.py'
    child.write_text("import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nwhile True: time.sleep(.1)\n")
    parent=tmp_path/'scp_parent.py'
    parent.write_text("import subprocess,sys,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nsubprocess.Popen([sys.executable,"+repr(str(child))+" ])\nwhile True: time.sleep(.1)\n")
    check=tmp_path/'transport_check.py'
    check.write_text("import ctypes,os,sys,subprocess\nsys.path.insert(0,"+repr(str(ROOT/'scripts'))+")\nfrom deepx_full_workflow_smoke_v27930 import run_transport\nctypes.CDLL(None).prctl(36,1,0,0,0)\ntry:\n run_transport([sys.executable,"+repr(str(parent))+"],timeout=.4,capture_output=True)\n raise AssertionError('missing timeout')\nexcept subprocess.TimeoutExpired: pass\nwhile True:\n try: os.waitpid(-1,0)\n except ChildProcessError: break\nprint('all owned transport children reaped')\n")
    result=subprocess.run([sys.executable,'-I','-B',str(check)],capture_output=True,text=True,timeout=12)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'all owned transport children reaped' in result.stdout


@pytest.mark.parametrize('lingering_child', [False, True])
def test_v27931_h2_cli_exit_between_snapshot_and_poll_requires_fresh_descendant_check(tmp_path, lingering_child):
    # A real child exits precisely after the first live /proc observation and
    # before Popen.poll(). WNOWAIT observes exit without consuming poll's child.
    child = tmp_path/'child.py'
    child.write_text('''import os,signal,subprocess,sys,time
from pathlib import Path
directory=Path(sys.argv[1])
if sys.argv[2]=='True':
    subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'],start_new_session=True)
(directory/'ready').write_text('ready')
while not (directory/'release').exists(): time.sleep(.002)
''')
    check = tmp_path/'check.py'
    check.write_text('''import json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import deepx_full_workflow_smoke_worker_v27930 as worker
directory=Path(sys.argv[2]); original=worker._capture_owned; forced=[]
def capture(owner,tracked):
    live=original(owner,tracked)
    if owner!=os.getpid() and owner in live and not forced:
        deadline=time.monotonic()+3
        while not (directory/'ready').exists():
            if time.monotonic()>deadline: raise RuntimeError('test_child_not_ready')
            time.sleep(.002)
        live=original(owner,tracked)
        (directory/'release').write_text('exit')
        os.waitid(os.P_PID,owner,os.WEXITED|os.WNOWAIT)
        forced.append(owner)
    return live
worker._capture_owned=capture
report=worker.supervised_run([sys.executable,str(directory/'child.py'),str(directory),sys.argv[3]],cwd=directory,log_path=directory/'child.log',timeout=5,grace=.1)
report['race_forced']=len(forced)==1
(directory/'report.json').write_text(json.dumps(report))
''')
    proc=subprocess.run([sys.executable,'-I','-S','-B',str(check),str(ROOT/'scripts'),str(tmp_path),str(lingering_child)],capture_output=True,text=True,timeout=15)
    assert proc.returncode==0,proc.stdout+proc.stderr
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['race_forced'] is True
    assert report['returncode']==0 and report['timed_out'] is False
    assert report['lingering_children_after_cli'] is lingering_child
    assert report['cleanup_complete'] is True and report['owned_survivors']==[]


@pytest.mark.parametrize('collected,code,status,expected',[(True,0,'pass',0),(True,0,'failed',2),(True,124,'pass',2),(False,0,'pass',2),(True,0,'setup_or_runtime_failed',2)])
def test_h2_transport_cannot_promote_a_failed_normal_runner(collected,code,status,expected):
    assert driver.result_exit_code(collected,code,{'status':status})==expected


def test_h2_collector_uses_packaged_source_zip_and_fresh_remote_paths(tmp_path,monkeypatch):
    source_tag = 'v' + CURRENT_VERSION.replace('.', '')
    collector_name = 'deepx_full_workflow_smoke_' + source_tag + '.py'
    source_name = 'ONNX-Splitpoint-Tool_v' + CURRENT_VERSION + '_SOURCE.zip'
    remote_prefix = '/tmp/onnx-' + source_tag + '-full-workflow-'
    case, unused_stage, runtime, calls=h2_stage(tmp_path,monkeypatch)
    run_dir=tmp_path/'synthetic_evaluationrun'
    suite=run_dir/'models/yolo11l/benchmark_set/legacy_suite'
    shutil.copytree(case.root,suite)
    old='/unavailable/deleted/old_run/1/suite'
    write_json(run_dir/'models/yolo11l/benchmark_results/benchmark_results_deepx_m1_full_auto.json',[{
        'run_id':'deepx_m1_full','backend':'deepx_m1','variant':'full',
        'dxnn_path':old+'/deepx/deepx_m1/full/model.dxnn',
        'deepx_prepared_feed_benchmark':{'image':old+'/'+case.image.name}}])
    write_json(run_dir/'hardware_matrix.json',{'hardware_targets':[{
        'id':case.setup_id,'accelerator':'deepx_m1','runtime':{'host':'192.168.0.102','user':'nx','port':22},
        'build_environment':{'runtime_venv':str(runtime)}}]})
    bundle=tmp_path/'download_bundle'
    probe=bundle/'probe'
    probe.mkdir(parents=True)
    for name in (collector_name,'deepx_full_workflow_smoke_v27930.py','deepx_full_output_probe_v27930.py',
                 'deepx_full_output_probe_worker_v27930.py'):
        shutil.copyfile(ROOT/'scripts'/name,probe/name)
    # Exercise extraction from a real ZIP containing current source bytes; the
    # collector is physically outside the source root and has no --source-root.
    # Derive the archive identity from the actual source release. Historical
    # wrapper imports may share implementation globals during collection;
    # the fresh collector process below must select its own current wrapper.
    with zipfile.ZipFile(bundle/source_name,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and driver._is_source_payload(str(path.relative_to(ROOT))):
                archive.write(path,'ONNX-Splitpoint-Tool_v'+CURRENT_VERSION+'/'+str(path.relative_to(ROOT)))
    bins=tmp_path/'transport'
    bins.mkdir()
    ssh=bins/'ssh'
    ssh.write_text('#!'+sys.executable+'\n'+'''
import os,secrets,shlex,shutil,string,subprocess,sys
words=shlex.split(sys.argv[-1])
if words[0]=='mktemp':
    path=REMOTE_PREFIX+''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(10))
    os.mkdir(path); print(path)
elif words[0]=='rm':
    assert words[:3]==['rm','-rf','--'] and words[3].startswith(REMOTE_PREFIX)
    shutil.rmtree(words[3])
else:
    assert words[0]=='timeout'
    sys.exit(subprocess.run(words).returncode)
'''.replace('REMOTE_PREFIX', repr(remote_prefix)))
    scp=bins/'scp'
    scp.write_text('#!'+sys.executable+'\n'+'''
from pathlib import Path
import shutil,sys
args=sys.argv[1:]; paths=[]; i=0
while i<len(args):
    if args[i] in ('-o','-P'): i+=2
    elif args[i] in ('-q','-r'): i+=1
    else: paths.append(args[i]); i+=1
def local(value): return Path(value.split(':',1)[1] if '@' in value else value)
destination=local(paths[-1])
for value in paths[:-1]:
    source=local(value)
    target=destination/source.name if destination.is_dir() else destination
    if source.is_dir(): shutil.copytree(source,target)
    else: shutil.copyfile(source,target)
''')
    ssh.chmod(0o755);scp.chmod(0o755)
    output=tmp_path/'collected'
    env=dict(os.environ,PATH=str(bins)+os.pathsep+os.environ['PATH'])
    before=inventory(run_dir)
    completed=subprocess.run([sys.executable,'-I','-B',str(probe/collector_name),
        '--run-dir',str(run_dir),'--bundle-dir',str(bundle),'--output-dir',str(output)],
        env=env,capture_output=True,text=True,timeout=45)
    if completed.returncode:
        logs='\n'.join(p.read_text() for p in output.rglob('*.log'))
    else: logs=''
    assert completed.returncode==0,completed.stdout+completed.stderr+logs[-14000:]
    archives=list(output.glob('*.zip'))
    assert len(archives)==1 and 'DIAGNOSTIC_ZIP=' in completed.stdout
    with zipfile.ZipFile(archives[0]) as archive:
        assert archive.testzip() is None
        summary=json.loads(archive.read('collection_summary.json'))
        assert summary['temporary_remote_directory_removed'] is True
        assert not Path(summary['remote_directory']).exists()
        result=json.loads(archive.read('results/deepx_full_workflow_smoke.json'))
        assert result['status']=='pass' and result['counts_as_benchmark'] is False
        assert not any(n.endswith(('.dxnn','.py')) for n in archive.namelist())
    assert inventory(run_dir)==before
