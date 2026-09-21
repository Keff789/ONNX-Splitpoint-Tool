"""R9B closure: exact warm Part1, bounded wrapper reuse and honest endpoints."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_v27925_deepx_compiler_overlay import compiler_fixture
from tests.test_v27931_classification_routing import _production_run
from tests.test_v27931_compiler_routing import _save_config


@pytest.mark.parametrize('drift', ['none', 'missing', 'receipt', 'boundary', 'preprocessing', 'artifact'])
def test_deferred_deepx_exact_cache_probe_without_compiler(compiler_fixture, tmp_path, monkeypatch, drift):
    from onnx_splitpoint_tool.deepx import compiler, env_status
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
        defer_deepx_part1_build, probe_deferred_deepx_part1_cache, DEEPX_PART1_REQUEST,
    )
    cfg = _save_config(tmp_path, monkeypatch, compiler_fixture)
    _, _, part1, _ = _production_run(tmp_path, compiler_fixture, cfg, 'imagenet_mean_std', 'warm')
    suite = tmp_path / 'warm/suite'
    (suite / 'benchmark_set.json').write_text(json.dumps({'cases': [{'id': 'b135', 'case_dir': 'b135'}]}))
    args = dict(out_dir=suite, bench_plan_runs=[{'type':'matrix','stage1':'deepx_m1','stage2':'tensorrt'}],
                validation_images='', fallback_calib_dir=str(tmp_path/'calibration'), calibration_num=1,
                task_hint='classification', classification_preprocessing='imagenet_mean_std',
                build_config=cfg, calibration_manifest=str(tmp_path/'calibration.json'))
    defer_deepx_part1_build(**args)
    request_before = (suite / DEEPX_PART1_REQUEST).read_bytes()
    cached = Path(cfg['cache_dir']) / part1['cases'][0]['cache_key']
    if drift == 'missing': (cached / 'model.dxnn').unlink()  # own synthetic test fixture only
    if drift == 'receipt': (cached / 'build_manifest.json').write_text('{}')
    if drift == 'artifact': (cached / 'model.dxnn').write_bytes(b'changed fixture')
    if drift in {'boundary', 'preprocessing'}:
        receipt = json.loads((cached/'build_manifest.json').read_text())
        receipt['cache_contract']['case_id' if drift=='boundary' else 'classification_preprocessing'] = 'wrong'
        (cached/'build_manifest.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(compiler, 'compile_dxnn', lambda **kw: pytest.fail('cache probe entered compiler'))
    monkeypatch.setattr(env_status, '_run_owned_probe', lambda *a, **kw: pytest.fail('cache probe entered SDK'))
    inspect = env_status.inspect_deepx_environment
    def unavailable_compiler(**kw):
        assert kw.get('path_only') is True and kw.get('probe_import') is False
        return {**inspect(**kw), 'compiler_ready': False}
    monkeypatch.setattr(env_status, 'inspect_deepx_environment', unavailable_compiler)
    result = probe_deferred_deepx_part1_cache(suite)
    status = json.loads((suite/'b135/deepx/deepx_m1/part1/deepx_part1_artifact_status.json').read_text())
    assert (suite / DEEPX_PART1_REQUEST).read_bytes() == request_before
    assert status['cache_lookup']['outcome'] == ('HIT' if drift == 'none' else 'MISS')
    assert result['ok_count'] == (1 if drift == 'none' else 0)
    assert status['ok'] is (drift == 'none')
    assert status['status'] == ('ok' if drift == 'none' else 'cache_miss_blocked')


def test_deferred_deepx_changed_source_cannot_reuse(tmp_path):
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import defer_deepx_part1_build, probe_deferred_deepx_part1_cache
    suite=tmp_path/'suite';case=suite/'b135';case.mkdir(parents=True)
    (case/'part1.onnx').write_bytes(b'original fixture')
    (case/'split_manifest.json').write_text(json.dumps({'part1_model':'part1.onnx'}))
    (suite/'benchmark_set.json').write_text(json.dumps({'cases':[{'case_dir':'b135'}]}))
    defer_deepx_part1_build(out_dir=suite)
    (case/'part1.onnx').write_bytes(b'changed fixture')
    with pytest.raises(RuntimeError, match='ONNX changed'):
        probe_deferred_deepx_part1_cache(suite)


def wrapper_fixture(tmp_path, monkeypatch, *, fail=False):
    from scripts import native_hailo_trt_fifo_from_benchmarkset as runner
    from onnx_splitpoint_tool import native_progress
    work=tmp_path/'wrapper';work.mkdir()
    cfg={};path=work/'native_fifo_config.json';calls=[]
    def build(command, **kw):
        calls.append((command,kw))
        assert 0 < kw['timeout'] <= 290
        if not fail:
            (work/'build/CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Release\n')
            (work/'build/split_native_hailo_trt_fifo').write_bytes(b'controlled test binary')
        return SimpleNamespace(returncode=17 if fail else 0, elapsed_s=.01)
    monkeypatch.setattr(native_progress,'run_streaming',build)
    return runner,work,path,cfg,calls


def test_cpp_wrapper_exact_reuse_and_no_recompile(tmp_path,monkeypatch):
    runner,work,path,cfg,calls=wrapper_fixture(tmp_path,monkeypatch)
    receipt=runner._prepare_cpp_wrapper(work,path,cfg,{},allow_build=True)
    assert receipt['status']=='ready' and len(calls)==2
    assert calls[1][1]['timeout'] <= calls[0][1]['timeout']
    reused=runner._prepare_cpp_wrapper(work,path,{},json.loads(path.read_text()),allow_build=False)
    assert reused['reused'] and reused['compiler_dispatch_count']==0 and len(calls)==2


@pytest.mark.parametrize('drift',['source','binary','options','architecture','cmake_cache','missing_receipt'])
def test_cpp_wrapper_no_build_rejects_stale_binding(tmp_path,monkeypatch,drift):
    runner,work,path,cfg,calls=wrapper_fixture(tmp_path,monkeypatch)
    runner._prepare_cpp_wrapper(work,path,cfg,{},allow_build=True)
    previous=json.loads(path.read_text())
    if drift=='source':monkeypatch.setattr(runner,'CPP_SOURCE',runner.CPP_SOURCE+'\n// changed')
    if drift=='binary':(work/'build/split_native_hailo_trt_fifo').write_bytes(b'changed')
    if drift=='cmake_cache':(work/'build/CMakeCache.txt').write_text('changed')
    if drift=='options':previous['wrapper_build']['identity']['configure_options']=[]
    if drift=='architecture':previous['wrapper_build']['identity']['target_architecture']='foreign'
    if drift=='missing_receipt':previous={}
    with pytest.raises(RuntimeError,match='reuse_binding'):
        runner._prepare_cpp_wrapper(work,path,{},previous,allow_build=False)
    assert len(calls)==2


def test_cpp_failed_generation_is_never_retried(tmp_path,monkeypatch):
    runner,work,path,cfg,calls=wrapper_fixture(tmp_path,monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='rc=17'):
        runner._prepare_cpp_wrapper(work,path,cfg,{},allow_build=True)
    with pytest.raises(RuntimeError,match='already_attempted'):
        runner._prepare_cpp_wrapper(work,path,{},json.loads(path.read_text()),allow_build=True)
    assert len(calls)==1


def test_wrapper_preparation_dependencies_are_staged_and_execute_in_isolation(tmp_path):
    import shutil,subprocess,sys
    from onnx_splitpoint_tool.remote_runtime_closure import native_remote_package_closure
    root=Path(__file__).resolve().parents[1]
    required={'onnx_splitpoint_tool.process_control','onnx_splitpoint_tool.native_progress'}
    package=tmp_path/'onnx_splitpoint_tool';package.mkdir();(package/'__init__.py').write_text('')
    staged=[]
    for relative,module,tokens in native_remote_package_closure():
        if module in required:
            shutil.copy2(root/relative,tmp_path/relative);staged.append(module)
    assert staged==['onnx_splitpoint_tool.process_control','onnx_splitpoint_tool.native_progress']
    code=("import sys,pathlib; sys.path.insert(0,sys.argv[1]); "
          "from onnx_splitpoint_tool import native_progress,process_control; "
          "assert all(pathlib.Path(m.__file__).is_relative_to(sys.argv[1]) for m in (native_progress,process_control)); "
          "r=native_progress.run_streaming([sys.executable,'-c',\"print('owned-child-complete')\"],timeout=5); "
          "assert r.returncode==0 and 'owned-child-complete' in r.stdout")
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(tmp_path)],cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr


def test_classification_host_output_rate_preserves_raw_fields_and_intervals():
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields,format_rate_endpoints
    from tests.test_v283_r9b_request_latency import sample
    row={'task':'classification','measurement_endpoint':'completed_task',
         'measurement_boundary':'first_task_start_to_last_task_completion',
         'completed_work_units':2,'makespan_ms':20.,'fps_makespan':100.,
         'request_latency':sample([7.,8.],task_complete=False)}
    before=copy.deepcopy(row);fields=rate_endpoint_fields(row)
    assert fields['completed_task_fps'] is None
    assert fields['host_output_fps']==100.
    assert fields['historical_fps']==100.
    assert fields['host_output_latency_mean_ms']==7.5
    assert 'Hostoutput (ohne Task-Postprocessing): 100.000 FPS' in format_rate_endpoints(fields)
    assert row==before


def test_mixed_task_and_host_repetitions_are_not_promoted():
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    from tests.test_v283_r9b_request_latency import sample
    records=[{'task':'classification','measurement_boundary':'first_task_start_to_last_task_completion',
              'completed_work_units':1,'makespan_ms':10.,'fps_makespan':100.,'repetition_id':str(i),
              'request_latency':sample([7.],task_complete=bool(i))} for i in range(2)]
    fields=rate_endpoint_fields({'task':'classification','repetition_records':records,'repetition_count_valid':2})
    assert fields['completed_task_fps'] is None and fields['host_output_fps'] is None


@pytest.mark.parametrize('failure', ['none', 'part1_miss', 'part2_miss', 'invalid_parent', 'changed_source'])
def test_actual_runner_barrier_resolves_full_part1_then_part2(compiler_fixture, tmp_path, monkeypatch, failure):
    """Real local resolvers/collector; only remote cache observations are fixtures."""
    from tests.test_v27931_classification_routing import _model
    from tests.test_v27920_artifact_cache_preflight_runner import _bare_runner
    from onnx_splitpoint_tool.campaign import create_dataset_manifest
    from onnx_splitpoint_tool.gui.benchmark_workflow import _materialize_manual_deepx_part1_artifacts
    from onnx_splitpoint_tool.workflow import execution_binding, deepx_build_binding
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import defer_deepx_part1_build
    from onnx_splitpoint_tool.deepx import compiler, env_status
    from onnx_splitpoint_tool.benchmark import remote_run
    cfg = _save_config(tmp_path, monkeypatch, compiler_fixture)
    model='mobilenet_v3_large';source=_model(tmp_path/'classification.onnx')
    calibration=tmp_path/'calibration';calibration.mkdir();(calibration/'image.jpg').write_bytes(b'fixture only')
    manifest=tmp_path/'calibration.json'
    create_dataset_manifest(task='classification',role='calibration',dataset_id='fixture',split='train',root=calibration,output=manifest,hash_mode='content')
    run=tmp_path/'run';formal=run/'models'/model/'benchmark_set';suite=formal/'suite'
    case=suite/'b135';case.mkdir(parents=True)
    _model(case/'part1.onnx');_model(case/'model_part2_b135.onnx')
    (case/'split_manifest.json').write_text(json.dumps({'part1_model':'part1.onnx','part2_model':'model_part2_b135.onnx'}))
    runs=[{'id':'deepx_m1_to_tensorrt','type':'matrix','stage1':{'type':'deepx_m1'},'stage2':{'provider':'tensorrt'},'variants':['composed']},
          {'id':'ort_tensorrt','type':'onnxruntime','provider':'tensorrt','variants':['full']}]
    (suite/'benchmark_plan.json').write_text(json.dumps({'runs':runs}))
    (suite/'benchmark_set.json').write_text(json.dumps({'model_id':model,'model':str(source),'benchmark_task':'classification','cases':[{'id':'b135','case_dir':'b135'}]}))
    (formal/'benchmark_set.json').write_text(json.dumps({'suite_dir':str(suite)}))
    build={**cfg,'mode':'reuse_and_build_missing','calibration_dir':str(calibration),'calib_count':1,'classification_preprocessing':'imagenet_mean_std'}
    profile={'deepx_build':build,'campaign':{'dataset_manifests':{'classification':{'calibration':str(manifest)}}},
             'quality_gate':{'statistics':{'execution_location':'central_management'}},
             'workflow':{'artifact_cache_preflight':{'enabled':True,'default_expectation':'warm','block_on_unexpected_cold_builds':True}}}
    row={'id':model,'task':'classification','resolved_path':str(source),'input_shape':[1,3,2,2]}
    full=deepx_build_binding.materialize_deepx_artifact_binding(run_dir=run,model_id=model,row=row,targets=['deepx_m1'],profile_payload=profile)
    assert full['status']=='ok'
    args=dict(out_dir=suite,bench_plan_runs=runs,validation_images='',fallback_calib_dir=str(calibration),calibration_num=1,task_hint='classification',build_config=build,profile_payload=profile,calibration_manifest=str(manifest))
    part1=_materialize_manual_deepx_part1_artifacts(**args)
    assert part1['status']=='ok'
    defer_deepx_part1_build(**args)
    if failure=='part1_miss':
        (Path(cfg['cache_dir'])/part1['cases'][0]['cache_key']/'model.dxnn').unlink()
    if failure=='changed_source':
        (case/'part1.onnx').write_bytes(b'changed since generation')
    monkeypatch.setattr(compiler,'compile_dxnn',lambda **kw:pytest.fail('preflight compiled Part1'))
    monkeypatch.setattr(deepx_build_binding,'compile_dxnn',lambda **kw:pytest.fail('preflight compiled Full'))
    monkeypatch.setattr(env_status,'_run_owned_probe',lambda *a,**kw:pytest.fail('preflight probed SDK'))
    runner=_bare_runner();runner.run_dir=run;runner.profile_payload=profile
    runner.options=SimpleNamespace(benchmark_execution_backend='remote');runner.outputs={};runner.report_paths=[]
    runner.log=runner._emit_log=lambda *a,**kw:None
    runner._targets=lambda:['deepx_m1','tensorrt']
    runner._profile_with_cli_hardware_overrides=lambda:profile
    runner._task_for=lambda row:'classification'
    monkeypatch.setattr(execution_binding,'_hardware_targets_for_plan',lambda *a,**kw:[{'id':'orin_nx_deepx_m1_01','accelerator':'deepx','runtime':{}}])
    probe=remote_run.probe_remote_trt_artifact_cache;seen=[]
    def remote_fixture(**kw):
        status=json.loads((case/'deepx/deepx_m1/part1/deepx_part1_artifact_status.json').read_text())
        assert status['cache_lookup']['outcome']==('MISS' if failure=='part1_miss' else 'UNKNOWN' if failure=='changed_source' else 'HIT')
        if failure=='invalid_parent':(case/'deepx/deepx_m1/part1/model.dxnn').write_bytes(b'changed after probe')
        kw['transport']=None
        result=probe(**kw)
        for requirement in result['requirements']:
            if requirement['role']=='trt_p2':
                seen.append(requirement)
                assert bool(requirement.get('part1_artifact_sha256')) is (failure not in {'part1_miss','invalid_parent','changed_source'})
        # No device access: fixture models only the final remote receipt result.
        for observation in result['observations']:
            unresolved = next((r for r in result['requirements']
                               if r['item_id']==observation['item_id'] and r.get('local_error')),None)
            if unresolved:
                observation.update(status='UNKNOWN',reason=unresolved.get('local_error_reason','local_source_identity_unavailable'))
            else:
                observation.update(status='MISS' if failure=='part2_miss' and observation['role']=='trt_p2' else 'HIT',reason='local_remote_receipt_fixture')
        return result
    monkeypatch.setattr(remote_run,'probe_remote_trt_artifact_cache',remote_fixture)
    runner._stage_artifact_cache_preflight([row])
    report=json.loads((run/'reports/artifact_cache_preflight.json').read_text())
    assert seen,report.get('collection_errors')
    assert report['runtime_dispatch_allowed'] is (failure=='none'),report
    assert report['completed_cold_build_count']==0
    assert bool(report['collection_errors']) is (failure=='changed_source')
