"""Saved H8 selection to the real normal compiler child; vendor SDK is synthetic.

No test here claims physical GPU execution, HEF validity or model quality.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_compiler_context as cc
from onnx_splitpoint_tool import hailo_dependency_plan as hp
from onnx_splitpoint_tool import run_modes as rm
from onnx_splitpoint_tool.build_dispatch_policy import bind_profile_hailo_builder
from test_v280_hailo8_compute_env import boundary, _local_components
from test_v27934_hailo_backend import managed as _managed

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('manifest', [None, False, True, 7, [], {}])
def test_manifest_invalid_type_is_not_coerced(manifest):
    with pytest.raises(cc.CompilerContextError, match='dependency_manifest_invalid'):
        cc.normalize_compute_by_family({'hailo8': {'device': 'gpu', 'dependency_manifest': manifest}})


@pytest.mark.parametrize('job,family,env_value,expected,source', [
    ({'device': 'gpu', 'dependency_manifest': '/job'}, {'device': 'gpu', 'dependency_manifest': '/family'}, '/env', '/job', 'job_override'),
    ({'device': 'gpu'}, {'device': 'cpu', 'dependency_manifest': '/family'}, '/env', '/family', 'compute_by_family.hailo8'),
    ({'device': 'gpu', 'dependency_manifest': ''}, {'device': 'gpu', 'dependency_manifest': '/family'}, '/env', '', 'job_override'),
    (None, {'device': 'gpu', 'dependency_manifest': ''}, '/env', '', 'compute_by_family.hailo8'),
    (None, {'device': 'gpu'}, '/env', '/env', 'environment:' + cc.DEPENDENCY_MANIFEST_ENV),
    (None, {'device': 'gpu'}, None, None, 'unset'),
])
def test_manifest_precedence_is_independent_of_device(job, family, env_value, expected, source):
    env = {} if env_value is None else {cc.DEPENDENCY_MANIFEST_ENV: env_value}
    result = cc.resolve_compute_selection('hailo8', job_override=job, compute_by_family={'hailo8': family}, env=env)
    assert result['dependency_manifest'] == expected
    assert result['dependency_manifest_source'] == source
    if source == 'job_override':
        assert result['ignored_dependency_manifests'][0]['reason'] == 'ignored_lower_priority'


@pytest.mark.parametrize('value', ['', '/family/overlay.json'])
def test_mode_and_profile_roundtrip_preserves_explicit_selection(tmp_path, monkeypatch, value):
    # This is the configured-GPU roundtrip, not the explicit CPU-mask conflict
    # scenario. Acceptance launchers deliberately mask real hardware with an
    # empty CUDA_VISIBLE_DEVICES. Resolve this metadata-only fixture in its own
    # controlled environment; monkeypatch restores the launcher's guard. No
    # SDK, assembler or GPU process is started by profile_hailo_compute_summary.
    for key in ('CUDA_VISIBLE_DEVICES', cc.MASK_SOURCE_ENV,
                'ONNX_SPLITPOINT_HAILO_COMPUTE_OVERRIDE',
                'ONNX_SPLITPOINT_HAILO_COMPUTE_BY_FAMILY',
                'ONNX_SPLITPOINT_HAILO_COMPUTE',
                'ONNX_SPLITPOINT_HAILO_ALLOW_GPU', 'SPLITPOINT_HAILO_ALLOW_GPU'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(cc.DEPENDENCY_MANIFEST_ENV, '/legacy/should/not/win')
    config = rm.default_run_modes_config()
    for mode in config['modes'].values():
        assert 'dependency_manifest' not in mode['build']['hailo']['compute_by_family']['hailo8']
    config['modes']['standard']['build']['hailo']['compute_by_family']['hailo8'] = {'device': 'gpu', 'dependency_manifest': value}
    registry = tmp_path / 'run_modes.yaml'
    registry.write_text(yaml.safe_dump(config))
    before = registry.read_bytes()
    config = rm.load_run_modes_config(registry)
    resolved, _ = rm.apply_run_mode({}, config=config, mode_id='standard')
    saved = yaml.safe_load(yaml.safe_dump(resolved))
    again, _ = rm.apply_run_mode(saved, config=config, mode_id='standard')
    assert again['hailo_build']['compute_by_family']['hailo8']['dependency_manifest'] == value
    row = rm.profile_hailo_compute_summary(again)['hailo8']
    assert row['status'] == 'selected'
    assert row['dependency_manifest'] == value
    assert row['dependency_manifest_source'] == 'compute_by_family.hailo8'
    assert row['ignored_dependency_manifests'][0]['value'] == '/legacy/should/not/win'
    assert registry.read_bytes() == before


def test_cpu_skips_saved_manifest_and_gpu_checks(boundary, monkeypatch):
    monkeypatch.setattr(hp, 'child_library_environment', lambda *a, **k: pytest.fail('CPU validated overlay'))
    monkeypatch.setattr(cc, '_gpu_target', lambda *a, **k: pytest.fail('CPU probed GPU'))
    result = cc.resolve_hailo_compiler_context(boundary['h8']/'bin/python', 'hailo8',
        compute_by_family={'hailo8': {'device': 'cpu', 'dependency_manifest': '/deleted/manifest.json'}})
    assert result['dependency_manifest_usage'] == 'not_used_for_cpu'
    parent = dict(os.environ)
    with cc.compiler_child_environment(result, parent_env=parent) as (child, effective):
        assert child['CUDA_VISIBLE_DEVICES'] == '-1'
        assert child['LD_LIBRARY_PATH'] == parent['LD_LIBRARY_PATH']
        assert effective['dependency_manifest'] == '/deleted/manifest.json'
    assert dict(os.environ) == parent


def test_explicit_empty_skips_legacy_overlay_and_uses_selected_venv(boundary):
    _local_components(boundary['site'])
    result = cc.resolve_hailo_compiler_context(boundary['h8']/'bin/python', 'hailo8',
        compute_by_family={'hailo8': {'device': 'gpu', 'dependency_manifest': ''}})
    assert result['component_source'] == 'selected_venv_triton'
    with cc.compiler_child_environment(result) as (env, _):
        assert env[cc.DEPENDENCY_MANIFEST_ENV] == ''
        assert env['LD_LIBRARY_PATH'] == '/unchanged/base/libs'


@pytest.mark.parametrize('damage', ['wrong_family', 'missing_manifest', 'stale_venv'])
def test_bad_saved_overlay_blocks_required_build_as_infrastructure(boundary, damage):
    manifest = boundary['manifest']
    if damage == 'wrong_family':
        payload = json.loads(manifest.read_text()); payload['family'] = 'hailo10h'; manifest.write_text(json.dumps(payload))
    elif damage == 'missing_manifest':
        manifest = manifest.with_name('missing.json')
    else:
        boundary['tf'].write_text(boundary['tf'].read_text().replace('2.18.0', '2.18.1'))
    with pytest.raises(cc.CompilerContextError, match='dependency_manifest_invalid') as exc:
        cc.resolve_hailo_compiler_context(boundary['h8']/'bin/python', 'hailo8',
            compute_by_family={'hailo8': {'device': 'gpu', 'dependency_manifest': str(manifest)}})
    assert exc.value.details['family'] == 'hailo8'


def test_normal_builder_saved_profile_reaches_real_pre_sdk_child(boundary, tmp_path, monkeypatch):
    # The actual normal builder/context/supervised child/calibration/publication
    # execute. Only SDK math, driver inventory and ptxas bytes are synthetic.
    inputs = _managed.__wrapped__(tmp_path, monkeypatch)
    inputs['hw_arch'] = 'hailo8'
    monkeypatch.setattr(backend, '_resolve_managed_venv_python', lambda **kw:
        ('synthetic-selected', boundary['h8']/'bin/python', str(boundary['h8']/'bin/activate')))
    sdk = (tmp_path/'sdk_boundary/hailo_sdk_client.py').read_text()
    sdk = ('import os\nfrom onnx_splitpoint_tool.hailo_compiler_context import validate_compiler_child_environment\n'
           'assert validate_compiler_child_environment()["family"] == "hailo8"\n' + sdk)
    (boundary['site']/'hailo_sdk_client.py').write_text(sdk)
    (boundary['site']/'test_dependencies.pth').write_text('\n'.join(p for p in sys.path if p.endswith('site-packages') and Path(p).is_dir())+'\n')
    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    monkeypatch.setenv(cc.DEPENDENCY_MANIFEST_ENV, '/wrong/legacy/manifest.json')
    profile = {'hailo_build': {'force_build': False, 'compute_by_family': {
        'hailo8': {'device': 'gpu', 'dependency_manifest': str(boundary['manifest'])}}}}
    effective, _ = rm.apply_run_mode(profile, config=rm.default_run_modes_config(), mode_id='standard')
    saved = tmp_path/'profile.yaml'; saved.write_text(yaml.safe_dump(effective))
    loaded = yaml.safe_load(saved.read_text())
    # Snapshot/dispatch must not depend on a GUI process environment mutation.
    parent = dict(os.environ)
    result = bind_profile_hailo_builder(backend.hailo_build_hef_auto, loaded)(
        inputs.pop('onnx_path'), **inputs, outdir=tmp_path/'normal_build', backend='venv', force=False)
    assert result.ok, result.error
    assert dict(os.environ) == parent
    context = result.details['compiler_context']
    assert context['dependency_manifest'] == str(boundary['manifest'])
    assert context['dependency_manifest_source'] == 'compute_by_family.hailo8'
    assert context['component_source'] == 'selected_hailo8_dependency_overlay'
    assert not Path(context['view_root']).exists()
    records = list((tmp_path/'production_logs').rglob('sdk_observed.json'))
    assert len(records) == 1
    observed = json.loads(records[0].read_text())
    assert observed['pid'] != os.getpid()
    assert observed['environment'][cc.DEPENDENCY_MANIFEST_ENV] == str(boundary['manifest'])
    assert observed['environment']['LD_LIBRARY_PATH'] == boundary['overlay_lib'] + ':/unchanged/base/libs'
    assert result.calib_info['compiler_dispatch_count'] == 1
    assert result.calib_info['gpu_execution_status'] == 'gpu_execution_unproven'


def test_resolved_manifest_is_immutable_at_later_child_spawn(boundary, monkeypatch):
    selected = cc.resolve_hailo_compiler_context(boundary['h8']/'bin/python', 'hailo8',
        compute_by_family={'hailo8': {'device': 'gpu', 'dependency_manifest': str(boundary['manifest'])}})
    monkeypatch.setenv(cc.DEPENDENCY_MANIFEST_ENV, '/changed/after/resolution')
    with cc.compiler_child_environment(selected) as (env, context):
        assert env[cc.DEPENDENCY_MANIFEST_ENV] == str(boundary['manifest'])
        assert env['LD_LIBRARY_PATH'].startswith(boundary['overlay_lib'] + ':')
        env['LD_LIBRARY_PATH'] = '/wrong'
        with pytest.raises(cc.CompilerContextError, match='Dependency libraries changed'):
            cc.validate_compiler_child_environment(env)


def test_h10_selection_does_not_receive_h8_overlay(boundary):
    parent = dict(os.environ)
    selected = cc.resolve_hailo_compiler_context(boundary['h10']/'bin/python', 'hailo10h', job_override='gpu')
    assert selected.get('dependency_manifest') is None
    with cc.compiler_child_environment(selected) as (env, _):
        assert env['LD_LIBRARY_PATH'] == parent['LD_LIBRARY_PATH']
    assert dict(os.environ) == parent


@pytest.mark.parametrize('style', ['relative', 'home'])
def test_manifest_paths_expand_once_at_context_resolution(boundary, monkeypatch, style):
    if style == 'relative':
        monkeypatch.chdir(boundary['root'])
        supplied = str(boundary['manifest'].relative_to(boundary['root']))
    else:
        monkeypatch.setenv('HOME', str(boundary['root']))
        supplied = '~/' + str(boundary['manifest'].relative_to(boundary['root']))
    selected = cc.resolve_hailo_compiler_context(boundary['h8']/'bin/python', 'hailo8',
        compute_by_family={'hailo8': {'device': 'gpu', 'dependency_manifest': supplied}})
    assert selected['dependency_manifest'] == str(boundary['manifest'])
    assert selected['dependency_manifest_source'] == 'compute_by_family.hailo8'


@pytest.mark.parametrize('stage', ['full', 'part1'])
def test_saved_manifest_survives_scheduler_and_deferred_continuation(tmp_path, stage):
    from concurrent.futures import ThreadPoolExecutor
    from test_v27922_negative_preflight import _suite, _result
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
        REQUEST_NAME, cache_preflight_builder, finalize_deferred_hailo_builds)
    model, suite, output, source = _suite(tmp_path)
    if stage == 'full':
        output = suite/'hailo'/'hailo8'/'full'
        output.mkdir(parents=True, exist_ok=True)
    profile = {'hailo_build': {'force_build': False, 'compute_by_family': {
        'hailo8': {'device': 'gpu', 'dependency_manifest': '/saved/overlay.json'}}}}
    original = copy.deepcopy(profile)
    calls = []
    def builder(source, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        result = _result(ok=True, hef_path=str(output/'compiled.hef'))
        (output/'compiled.hef').write_bytes(b'synthetic existing artifact')
        result.details = {'cache_hit': True, 'compiler_dispatch_count': 0}
        return result
    callback = bind_profile_hailo_builder(cache_preflight_builder(builder), profile)
    profile['hailo_build']['compute_by_family']['hailo8']['dependency_manifest'] = '/later/edit'
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(callback, source, outdir=str(output), hw_arch='hailo8',
            net_name='yolo26s_' + (stage + '_b364' if stage == 'part1' else 'full'), force=False,
            build_evidence_context={'model_id':'yolo26s', 'boundary':364 if stage == 'part1' else None, 'stage':stage}).result()
    stored = json.loads((output/REQUEST_NAME).read_text())
    assert stored['kwargs']['compute_by_family'] == original['hailo_build']['compute_by_family']
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload=original, build_fn=builder)
    assert len(calls) == 2
    assert calls[0]['compute_by_family'] == calls[1]['compute_by_family'] == original['hailo_build']['compute_by_family']
    assert result['jobs'][0]['compiler_dispatch_count'] == 0


def test_real_miss_bad_overlay_never_records_model_infeasibility(boundary, tmp_path, monkeypatch):
    from onnx_splitpoint_tool import build_evidence_store
    inputs = _managed.__wrapped__(tmp_path, monkeypatch)
    inputs['hw_arch'] = 'hailo8'
    monkeypatch.setattr(backend, '_resolve_managed_venv_python', lambda **kw:
        ('synthetic-selected', boundary['h8']/'bin/python', str(boundary['h8']/'bin/activate')))
    monkeypatch.setattr(backend, '_run_streamed_subprocess', lambda *a, **kw: pytest.fail('Invalid context dispatched compiler'))
    monkeypatch.setattr(build_evidence_store.BuildEvidenceStore, 'record', lambda *a, **kw: pytest.fail('Infrastructure recorded as model evidence'))
    result = backend.hailo_build_hef_auto(**inputs, backend='venv', outdir=tmp_path/'failed_context',
        compute_by_family={'hailo8': {'device': 'gpu', 'dependency_manifest': '/deleted/overlay.json'}})
    assert not result.ok
    assert result.failure_kind == 'hailo_dependency_manifest_invalid'
    assert result.last_stage == 'compiler_context'
    assert not result.details.get('build_evidence', {}).get('recorded')
    assert not result.details.get('negative_evidence_hit')
