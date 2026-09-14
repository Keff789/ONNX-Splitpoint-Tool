"""REV4 AP5: real child dispatch with explicitly simulated vendor modules."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

import pytest
import yaml

from onnx_splitpoint_tool.deepx import compiler, env_status
from onnx_splitpoint_tool import backend_build_environments
from tests.test_v27925_deepx_compiler_overlay import compiler_fixture


def _save_config(tmp_path, monkeypatch, f, **changes):
    config = dict(dx_all_suite_root=str(f['venv'].parent), compiler_venv=str(f['venv']),
                  compiler_overlay=str(f['overlay']), cache_dir=str(tmp_path/'cache'))
    config.update(changes)
    path = tmp_path/'build_environments.yaml'
    path.write_text(yaml.safe_dump({'build_environments': [{'kind':'deepx_dxcom', **config}]}))
    monkeypatch.setattr(backend_build_environments, 'CONFIG_PATH', path)
    monkeypatch.delenv('ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY')
    return config


@pytest.mark.parametrize('isolated', [False, True])
def test_T05_1_saved_normal_context_direct_and_shell(compiler_fixture, tmp_path, monkeypatch, isolated):
    f=compiler_fixture
    _save_config(tmp_path, monkeypatch, f)
    ctx=env_status.resolve_compiler_context()
    assert ctx['compiler_selection_source']=='saved_build_configuration'
    code='import torch; print(torch.__version__)'
    direct=env_status._run_owned_probe(env_status.compiler_python_command(ctx, code=code, isolated=isolated), timeout_s=10, env=env_status.compiler_subprocess_environment(context=ctx))
    shell=env_status._run_owned_probe(['bash','-lc',compiler._compiler_shell_prefix(f['venv']/'bin/activate',ctx)+' '+shlex.join(env_status.compiler_python_command(ctx, code=code, isolated=isolated))], timeout_s=10, env=env_status.compiler_subprocess_environment(context=ctx))
    assert direct.returncode==shell.returncode==0
    assert direct.stdout.strip()==shell.stdout.strip()=='overlay+cu126'


def test_T05_1_known_installation_auto_is_bounded(compiler_fixture, tmp_path, monkeypatch):
    f=compiler_fixture
    cfg=_save_config(tmp_path, monkeypatch,f,compiler_overlay='')
    known=f['venv'].parent/'dx-compiler/pytorch-2.12.0-cu126-overlay'
    known.parent.mkdir()
    known.symlink_to(f['overlay'], target_is_directory=True)
    ctx=env_status.resolve_compiler_context(cfg)
    assert ctx['compiler_selection_source']=='known_dx_all_suite_installation'
    assert ctx['compiler_overlay']==str(f['overlay'])
    (known.parent/'pytorch-2.13.0-cu126-overlay').mkdir()
    second=known.parent/'pytorch-2.13.0-cu126-overlay/torch'
    second.mkdir();(second/'__init__.py').write_text('')
    with pytest.raises(ValueError,match='overlay_ambiguous'):
        env_status.resolve_compiler_context(cfg)


def test_T05_2_explicit_priority_and_invalid_never_fallback(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture
    _save_config(tmp_path,monkeypatch,f)
    monkeypatch.setenv('ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY',str(f['vendor']))
    assert env_status.resolve_compiler_context()['compiler_overlay']==str(f['vendor'])
    monkeypatch.setenv('ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY',str(tmp_path/'absent'))
    with pytest.raises(ValueError,match='overlay_invalid'):
        env_status.resolve_compiler_context()
    monkeypatch.delenv('ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY')
    with pytest.raises(ValueError,match='overlay_invalid'):
        env_status.resolve_compiler_context({'compiler_overlay':str(tmp_path/'absent')})


def test_T05_3_normal_compile_parent_modules_and_other_children_unchanged(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture
    _save_config(tmp_path,monkeypatch,f)
    before=dict(os.environ);modules={name:sys.modules.get(name) for name in ('torch','numpy')}
    result=compiler.compile_dxnn(onnx_path=f['model'],config_path=f['config'],output_dir=tmp_path/'compiled')
    assert result.ok, result.message
    assert dict(os.environ)==before
    assert all(sys.modules.get(name) is value for name,value in modules.items())
    assert subprocess.check_output([str(f['python']),'-c','import torch; print(torch.__version__)'],text=True).strip()=='original+cu130'
    report=json.loads((tmp_path/'compiled/build_manifest.json').read_text())
    assert report['compiler_cuda_preflight']['operations_probe_status']=='pass'
    assert report['compiler_cuda_preflight']['dxcom_compile_status']=='not_run'
    assert report['ok'] is True


def test_T05_4_path_only_never_starts_probe_even_invalid_overlay(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture
    cfg=_save_config(tmp_path,monkeypatch,f)
    monkeypatch.setattr(env_status,'_run_owned_probe',lambda *a,**k:pytest.fail('compiler probe in cache lookup'))
    status=env_status.inspect_deepx_environment(config=cfg,path_only=True,probe_import=True)
    assert status['compiler_cuda_preflight']['status']=='not_probed'
    monkeypatch.setenv('ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY',str(tmp_path/'invalid'))
    status=env_status.inspect_deepx_environment(config=cfg,path_only=True)
    assert status['compiler_ready'] is False


def test_T05_5_operation_pass_does_not_make_missing_or_stale_artifact_success(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture
    _save_config(tmp_path,monkeypatch,f)
    (f['vendor']/'dx_com.py').write_text('def compile(**kwargs): pass\n')
    out=tmp_path/'build';out.mkdir();(out/'old.dxnn').write_bytes(b'previous failed attempt')
    result=compiler.compile_dxnn(onnx_path=f['model'],config_path=f['config'],output_dir=out)
    assert not result.ok and result.status=='dxnn_missing'
    assert (out/'old.dxnn').read_bytes()==b'previous failed attempt'
    report=json.loads((out/'build_manifest.json').read_text())
    assert report['compiler_cuda_preflight']['operations_probe_status']=='pass'


@pytest.mark.parametrize('arches,expected',[(['sm_60'],'compatible'),(['compute_60'],'compatible'),(['sm_75'],'incompatible'),(['unknown'],'unknown')])
def test_T05_6_existing_architecture_compatibility_not_sm61_shortcut(arches,expected):
    status=env_status._cuda_architecture_status({'cuda_available':True,'device_capability':[6,1],'compiled_architectures':arches})
    assert status['status']==expected


def test_T05_6_unknown_probe_is_not_build_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(compiler,'probe_compiler_cuda_architecture',lambda *a,**k:{'status':'unknown','reason':'metadata_unavailable'})
    result=compiler._cuda_preflight_failure(venv=tmp_path,onnx_path=Path('model'),config_path=Path('cfg'),output_dir=tmp_path,log_path=tmp_path/'log')
    assert result is not None and not result.ok


@pytest.mark.parametrize('cancel',[False,True])
def test_T05_7_owned_process_timeout_or_cancel_leaves_no_valid_result(tmp_path,cancel):
    marker=tmp_path/'ready'
    code='from pathlib import Path; import time; Path('+repr(str(marker))+').write_text("ready"); time.sleep(60)'
    event=threading.Event()
    if cancel:
        def trigger():
            limit=time.monotonic()+3
            while not marker.exists() and time.monotonic()<limit: time.sleep(.01)
            event.set()
        thread=threading.Thread(target=trigger);thread.start()
        result=compiler._run_owned_compiler([sys.executable,'-c',code],timeout_s=5,cancel_event=event)
        thread.join(4)
        assert result.returncode==130 and 'CANCELLED' in result.stdout
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            compiler._run_owned_compiler([sys.executable,'-c',code],timeout_s=.4)
    assert not list(tmp_path.glob('*.dxnn'))


def test_T05_1_T05_5_normal_gui_materializers_share_saved_context(compiler_fixture,tmp_path,monkeypatch):
    from tests.test_v27931_classification_routing import _model
    from onnx_splitpoint_tool.gui import benchmark_workflow
    import onnx
    f=compiler_fixture
    _save_config(tmp_path,monkeypatch,f)
    suite=tmp_path/'manual';case=suite/'b135';case.mkdir(parents=True)
    source=_model(tmp_path/'source.onnx');_model(case/'part1.onnx')
    (case/'split_manifest.json').write_text(json.dumps({'part1_model':'part1.onnx'}))
    calibration=tmp_path/'calibration';calibration.mkdir();(calibration/'image.jpg').write_bytes(b'image fixture')
    before=dict(os.environ)
    full=benchmark_workflow._materialize_manual_deepx_full_artifact(out_dir=suite,model_path=str(source),model=onnx.load(source),bench_plan_runs=[{'type':'deepx'}],validation_images='',validation_max_images=1,fallback_calib_dir=str(calibration),calibration_num=1,task_hint='classification')
    part1=benchmark_workflow._materialize_manual_deepx_part1_artifacts(out_dir=suite,bench_plan_runs=[{'type':'matrix','stage1':'deepx_m1','stage2':'tensorrt'}],validation_images='',fallback_calib_dir=str(calibration),calibration_num=1,task_hint='classification')
    assert full['status']==part1['status']=='ok', (full,part1)
    assert full['classification_preprocessing']==part1['classification_preprocessing']=='imagenet_mean_std'
    assert full['environment_status']['compiler_selection_source']==part1['environment_status']['compiler_selection_source']=='saved_build_configuration'
    for file in suite.rglob('observed_environment.json'):
        assert json.loads(file.read_text())['torch']=='overlay+cu126'
    assert len(list(suite.rglob('observed_environment.json')))==2
    assert dict(os.environ)==before


@pytest.mark.parametrize('reason', ['timeout','cancel'])
def test_T05_7_probe_primary_reason_preserved(compiler_fixture,monkeypatch,reason):
    f=compiler_fixture
    def probe(*args,**kwargs):
        if reason=='timeout': raise subprocess.TimeoutExpired(args[0],kwargs['timeout_s'])
        return subprocess.CompletedProcess(args[0],130,'CANCELLED','')
    monkeypatch.setattr(env_status,'_run_owned_probe',probe)
    status=env_status.probe_compiler_cuda_architecture(f['python'],operation_probe=True)
    assert status['status']=='unknown'
    assert status['reason']=='deepx_compiler_probe_'+('timeout' if reason=='timeout' else 'cancelled')


@pytest.mark.parametrize('owner',[compiler,env_status])
def test_T05_7_standalone_calls_use_existing_tracking_registry(monkeypatch,owner):
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
    observed=[]
    class TrackingRegistry(ProcessTreeRegistry):
        def register(self,proc,**kwargs):
            observed.append('registered');return super().register(proc,**kwargs)
        def assert_quiescent(self,**kwargs):
            observed.append('quiescent_checked');return super().assert_quiescent(**kwargs)
    monkeypatch.setattr(owner,'ProcessTreeRegistry',TrackingRegistry)
    monkeypatch.setattr(owner,'current_process_registry',lambda:None)
    run=owner._run_owned_compiler if owner is compiler else owner._run_owned_probe
    result=run([sys.executable,'-c','print("owned")'],timeout_s=5)
    assert result.returncode==0 and result.stdout.strip()=='owned'
    assert observed==['registered','quiescent_checked']
