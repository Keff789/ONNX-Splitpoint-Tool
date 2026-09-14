"""F3: normal compiler contexts, using only simulated vendor modules in children."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from onnx_splitpoint_tool import backend_build_environments
from onnx_splitpoint_tool.deepx import compiler, env_status
from tests.test_v27925_deepx_compiler_overlay import compiler_fixture


ENV_KEYS = ('DEEPX_DX_ALL_SUITE_ROOT', 'DX_ALL_SUITE_ROOT', 'DX_ALL_SUITE',
            'DEEPX_COMPILER_VENV', 'ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY')


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path, monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(backend_build_environments, 'CONFIG_PATH', tmp_path/'saved.yaml')


def candidate(tmp_path, name):
    root = tmp_path/name
    venv = root/'dx-compiler/venv'
    overlay = root/'dx-compiler/pytorch-2.12.0-cu126-overlay'
    (venv/'bin').mkdir(parents=True)
    (venv/'bin/python').symlink_to(sys.executable)
    (overlay/'torch').mkdir(parents=True)
    (overlay/'torch/__init__.py').write_text('# simulated package\n')
    return {'kind': 'deepx_dxcom', 'dx_all_suite_root': str(root),
            'compiler_venv': str(venv), 'compiler_overlay': str(overlay),
            'cache_dir': str(tmp_path/'cache')}


def save(*rows):
    backend_build_environments.CONFIG_PATH.write_text(yaml.safe_dump({'build_environments': list(rows)}))


def selected(context):
    return tuple(context[key] for key in ('dx_all_suite_root', 'compiler_venv', 'compiler_python', 'compiler_overlay', 'cache_dir'))


@pytest.mark.parametrize('entrypoint', ['config', 'arguments', 'environment'])
def test_T32_F3_01_02_complete_context_ignores_unused_alternatives(tmp_path, monkeypatch, entrypoint):
    chosen = candidate(tmp_path, 'chosen')
    other = candidate(tmp_path, 'other')
    cfg = dict(chosen)
    kwargs = {}
    if entrypoint == 'arguments':
        kwargs = {'root': cfg.pop('dx_all_suite_root'), 'python': str(Path(cfg.pop('compiler_venv'))/'bin/python')}
    elif entrypoint == 'environment':
        for key, field in zip((ENV_KEYS[0], ENV_KEYS[3], ENV_KEYS[4]), ('dx_all_suite_root', 'compiler_venv', 'compiler_overlay')):
            monkeypatch.setenv(key, cfg.pop(field))
    save(chosen)
    first = env_status.resolve_compiler_context(cfg, **kwargs)
    save(chosen, other)
    second = env_status.resolve_compiler_context(cfg, **kwargs)
    assert selected(first) == selected(second)
    assert second['compiler_overlay'] == chosen['compiler_overlay']
    assert second['compiler_python'] == str(Path(chosen['compiler_venv'])/'bin/python')
    assert second['compiler_selection_source'] == ('process_environment' if entrypoint == 'environment' else 'explicit_configuration')


def test_T32_F3_03_automatic_ambiguity_and_local_enabled_filter(tmp_path):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    save(a, b)
    with pytest.raises(ValueError, match='deepx_compiler_configuration_ambiguous'):
        env_status.resolve_compiler_context()
    save(a, {**b, 'host': 'remote'}, {**b, 'enabled': False}, {**b, 'kind': 'hailo8_dfc'})
    assert env_status.resolve_compiler_context()['compiler_venv'] == a['compiler_venv']


@pytest.mark.parametrize('field', ['dx_all_suite_root', 'venv', 'python'])
def test_T32_F3_04_partial_selects_one_coherent_candidate(tmp_path, field):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    save(a, b)
    cfg = {'cache_dir': a['cache_dir']}
    kwargs = {}
    if field == 'python':
        kwargs['python'] = str(Path(b['compiler_venv'])/'bin/python')
    else:
        cfg[field] = b['compiler_venv'] if field == 'venv' else b[field]
    ctx = env_status.resolve_compiler_context(cfg, **kwargs)
    for key in ('dx_all_suite_root', 'compiler_venv', 'compiler_overlay'):
        assert ctx[key] == b[key]


def test_T32_F3_04_partial_ambiguous_rest_never_forms_hybrid(tmp_path):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    save(a, {**b, 'dx_all_suite_root': a['dx_all_suite_root']})
    with pytest.raises(ValueError, match='configuration_ambiguous'):
        env_status.resolve_compiler_context({'dx_all_suite_root': a['dx_all_suite_root']})
    save(a, b)
    ctx = env_status.resolve_compiler_context({'dx_all_suite_root': a['dx_all_suite_root'], 'compiler_venv': b['compiler_venv']})
    # Both stored records conflict, so only discovery under the explicitly chosen root is permitted.
    assert ctx['compiler_venv'] == b['compiler_venv']
    assert ctx['compiler_overlay'] == a['compiler_overlay']
    assert ctx['compiler_selection_source'] == 'known_dx_all_suite_installation'


@pytest.mark.parametrize('source', ['config', 'environment'])
def test_T32_F3_05_invalid_explicit_overlay_remains_error(tmp_path, monkeypatch, source):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    save(a, b)
    cfg = dict(a)
    if source == 'environment':
        monkeypatch.setenv(ENV_KEYS[4], str(tmp_path/'missing'))
    else:
        cfg['compiler_overlay'] = str(tmp_path/'missing')
    with pytest.raises(ValueError, match='deepx_compiler_overlay_invalid'):
        env_status.resolve_compiler_context(cfg)


@pytest.mark.parametrize('complete', [False, True])
def test_T32_F3_06_profile_helper_and_resolver(tmp_path, complete):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    save(a, b)
    explicit = dict(b) if complete else {'dx_all_suite_root': b['dx_all_suite_root']}
    profile = {'build_environments': [a, b], 'deepx_build': {**explicit, 'classification_preprocessing': 'imagenet_mean_std'}}
    cfg = env_status.profile_compiler_configuration(profile)
    ctx = env_status.resolve_compiler_context(cfg)
    assert ctx['compiler_venv'] == b['compiler_venv']
    assert ctx['compiler_overlay'] == b['compiler_overlay']
    assert cfg['classification_preprocessing'] == 'imagenet_mean_std'


def test_T32_F3_06_profile_still_rejects_needed_ambiguity(tmp_path):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    with pytest.raises(ValueError, match='configuration_ambiguous'):
        env_status.profile_compiler_configuration({'build_environments': [a, b], 'deepx_build': {'compiler_overlay': a['compiler_overlay']}})


def test_T32_F3_06_unique_profile_keeps_runtime_and_cache_fields(tmp_path):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    entry = {**a, 'runtime_venv': '/configured/runtime-only-venv'}
    explicit = {key: value for key, value in a.items() if key != 'cache_dir'}
    cfg = env_status.profile_compiler_configuration({'build_environments': [entry, b], 'deepx_build': explicit})
    assert cfg['runtime_venv'] == entry['runtime_venv']
    assert cfg['cache_dir'] == a['cache_dir']


@pytest.mark.parametrize('alias', ['compiler_venv', 'venv', 'venv_activate'])
def test_T32_F3_07_aliases_normalized_before_saved_merge(tmp_path, alias):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b with spaces')
    save(a)
    cfg = {'compiler_root': b['dx_all_suite_root'], 'compiler_overlay': b['compiler_overlay']}
    cfg[alias] = ('source '+shlex.quote(str(Path(b['compiler_venv'])/'bin/activate'))) if alias == 'venv_activate' else b['compiler_venv']
    ctx = env_status.resolve_compiler_context(cfg)
    assert ctx['compiler_venv'] == b['compiler_venv']
    assert ctx['dx_all_suite_root'] == b['dx_all_suite_root']
    assert ctx['compiler_python'] == str(Path(b['compiler_venv'])/'bin/python')
    assert Path(ctx['compiler_python']).is_symlink()


@pytest.mark.parametrize('empty', [None, '', 'auto', 'default', 'none'])
def test_T32_F3_07_empty_defaults_do_not_mask_saved_selection(tmp_path, monkeypatch, empty):
    a = candidate(tmp_path, 'a')
    save(a)
    for key in ENV_KEYS:
        monkeypatch.setenv(key, '' if empty is None else empty)
    ctx = env_status.resolve_compiler_context({key: empty for key in a}, root=empty, python=empty)
    assert ctx['compiler_venv'] == a['compiler_venv']
    assert ctx['dx_all_suite_root'] == a['dx_all_suite_root']
    assert ctx['compiler_overlay'] == a['compiler_overlay']


def test_T32_F3_07_environment_root_priority_python_and_parent_unchanged(tmp_path, monkeypatch):
    a, b, c = (candidate(tmp_path, name) for name in ('a', 'b', 'c'))
    save(a, b)
    for key in ENV_KEYS[:3]:
        monkeypatch.setenv(key, c['dx_all_suite_root'] if key == ENV_KEYS[0] else b['dx_all_suite_root'])
    monkeypatch.setenv(ENV_KEYS[3], c['compiler_venv'])
    monkeypatch.setenv(ENV_KEYS[4], c['compiler_overlay'])
    before, path_before = dict(os.environ), list(sys.path)
    modules_before = {key: sys.modules.get(key) for key in ('torch', 'numpy', 'dx_com')}
    ctx = env_status.resolve_compiler_context(a, root=b['dx_all_suite_root'], python=str(Path(b['compiler_venv'])/'bin/python'))
    assert ctx['dx_all_suite_root'] == c['dx_all_suite_root']
    assert ctx['compiler_venv'] == c['compiler_venv']
    assert ctx['compiler_overlay'] == c['compiler_overlay']
    assert ctx['compiler_python'] == str(Path(b['compiler_venv'])/'bin/python')
    assert dict(os.environ) == before and sys.path == path_before
    assert all(sys.modules.get(key) is val for key, val in modules_before.items())


def test_T32_F3_07_cache_selection_is_separate_and_ambiguity_explicit(tmp_path):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    cfg = {key: val for key, val in a.items() if key != 'cache_dir'}
    save(a, {**a, 'compiler_overlay': b['compiler_overlay']})
    assert env_status.resolve_compiler_context(cfg)['cache_dir'] == a['cache_dir']
    save(a, {**a, 'cache_dir': str(tmp_path/'other-cache')})
    with pytest.raises(ValueError, match='deepx_compiler_configuration_ambiguous.*cache_dir'):
        env_status.resolve_compiler_context(cfg)
    assert env_status.resolve_compiler_context({**cfg, 'cache_dir': a['cache_dir']})['cache_dir'] == a['cache_dir']


def test_T32_F3_07_unique_saved_cache_survives_explicit_other_compiler(tmp_path):
    a, b = candidate(tmp_path, 'saved'), candidate(tmp_path, 'explicit')
    a['cache_dir'] = str(tmp_path/'existing-custom-artifacts')
    Path(a['cache_dir']).mkdir()
    (Path(a['cache_dir'])/'existing.dxnn').write_bytes(b'preserved cache fixture')
    save(a)
    cfg = {key: value for key, value in b.items() if key != 'cache_dir'}
    ctx = env_status.resolve_compiler_context(cfg)
    assert ctx['compiler_venv'] == b['compiler_venv']
    assert ctx['cache_dir'] == a['cache_dir']
    assert (Path(ctx['cache_dir'])/'existing.dxnn').read_bytes() == b'preserved cache fixture'


def test_T32_F3_07_unrelated_cache_targets_require_explicit_choice(tmp_path):
    a, b, c = (candidate(tmp_path, name) for name in ('a', 'b', 'explicit'))
    b['cache_dir'] = str(tmp_path/'other-cache')
    save(a, b)
    cfg = {key: value for key, value in c.items() if key != 'cache_dir'}
    with pytest.raises(ValueError, match='deepx_compiler_configuration_ambiguous.*cache_dir'):
        env_status.resolve_compiler_context(cfg)
    assert env_status.resolve_compiler_context({**cfg, 'cache_dir': a['cache_dir']})['cache_dir'] == a['cache_dir']


def test_T32_F3_08_no_overlay_retains_inheritance_without_recursive_discovery(tmp_path):
    a = candidate(tmp_path, 'a')
    Path(a['compiler_overlay']).rename(Path(a['dx_all_suite_root'])/'outside-known-directory')
    save()
    cfg = {key: val for key, val in a.items() if key != 'compiler_overlay'}
    ctx = env_status.resolve_compiler_context(cfg)
    assert ctx['compiler_overlay'] == ''
    assert env_status.compiler_subprocess_environment(context=ctx) is None


def test_T32_F3_10_explicit_invalid_root_is_retained_without_probe_or_fallback(tmp_path, monkeypatch):
    a, b = candidate(tmp_path, 'a'), candidate(tmp_path, 'b')
    save(a, b)
    monkeypatch.setattr(env_status, '_run_owned_probe', lambda *args, **kwargs: pytest.fail('path resolution dispatched a compiler'))
    cfg = {**a, 'dx_all_suite_root': str(tmp_path/'missing-root')}
    ctx = env_status.resolve_compiler_context(cfg)
    assert ctx['dx_all_suite_root'] == cfg['dx_all_suite_root']
    status = env_status.inspect_deepx_environment(config=cfg, path_only=True, probe_import=True)
    assert status['compiler_ready'] is False
    assert status['compiler_cuda_preflight']['status'] == 'not_probed'


@pytest.mark.parametrize('isolated', [False, True])
def test_T32_F3_09_explicit_profile_direct_shell_and_isolated_child(compiler_fixture, tmp_path, monkeypatch, isolated):
    f = compiler_fixture
    monkeypatch.delenv(ENV_KEYS[4], raising=False)
    cfg = {'dx_all_suite_root': str(f['venv'].parent), 'compiler_venv': str(f['venv']), 'compiler_overlay': str(f['overlay']), 'cache_dir': str(tmp_path/'cache')}
    alternative = candidate(tmp_path, 'alternative')
    save({'kind': 'deepx_dxcom', **cfg}, alternative)
    profile = {'build_environments': [alternative, {'kind': 'deepx_dxcom', **cfg}], 'deepx_build': cfg}
    ctx = env_status.resolve_compiler_context(env_status.profile_compiler_configuration(profile))
    before, before_path = dict(os.environ), list(sys.path)
    code = 'import torch; print(torch.__version__)'
    cmd = env_status.compiler_python_command(ctx, code=code, isolated=isolated)
    direct = env_status._run_owned_probe(cmd, timeout_s=10, env=env_status.compiler_subprocess_environment(context=ctx))
    shell = env_status._run_owned_probe(['bash', '-lc', compiler._compiler_shell_prefix(f['venv']/'bin/activate', ctx)+' '+shlex.join(cmd)], timeout_s=10, env=env_status.compiler_subprocess_environment(context=ctx))
    assert direct.returncode == shell.returncode == 0
    assert direct.stdout.strip() == shell.stdout.strip() == 'overlay+cu126'
    assert dict(os.environ) == before and sys.path == before_path
    assert subprocess.check_output([str(f['python']), '-c', code], text=True).strip() == 'original+cu130'


def test_T32_F3_09_normal_profile_full_and_part1_materializers(compiler_fixture, tmp_path, monkeypatch):
    from tests.test_v27931_classification_routing import _model
    from onnx_splitpoint_tool.gui import benchmark_workflow
    import onnx
    f = compiler_fixture
    monkeypatch.delenv(ENV_KEYS[4], raising=False)
    cfg = {'dx_all_suite_root': str(f['venv'].parent), 'compiler_venv': str(f['venv']), 'compiler_overlay': str(f['overlay']), 'cache_dir': str(tmp_path/'cache')}
    alternative = candidate(tmp_path, 'alternative')
    save({'kind': 'deepx_dxcom', **cfg}, alternative)
    profile = {'build_environments': [alternative, {'kind': 'deepx_dxcom', **cfg}], 'deepx_build': cfg}
    suite = tmp_path/'normal'; case = suite/'b135'; case.mkdir(parents=True)
    source = _model(tmp_path/'source.onnx'); _model(case/'part1.onnx')
    (case/'split_manifest.json').write_text(json.dumps({'part1_model': 'part1.onnx'}))
    calibration = tmp_path/'calibration'; calibration.mkdir(); (calibration/'image.jpg').write_bytes(b'image fixture')
    before = dict(os.environ)
    full = benchmark_workflow._materialize_manual_deepx_full_artifact(out_dir=suite, model_path=str(source), model=onnx.load(source), bench_plan_runs=[{'type': 'deepx'}], validation_images='', validation_max_images=1, fallback_calib_dir=str(calibration), calibration_num=1, task_hint='classification', build_config=env_status.profile_compiler_configuration(profile))
    part1 = benchmark_workflow._materialize_manual_deepx_part1_artifacts(out_dir=suite, bench_plan_runs=[{'type': 'matrix', 'stage1': 'deepx_m1', 'stage2': 'tensorrt'}], validation_images='', fallback_calib_dir=str(calibration), calibration_num=1, task_hint='classification', profile_payload=profile)
    assert full['status'] == part1['status'] == 'ok', (full, part1)
    assert full['environment_status']['compiler_selection_source'] == part1['environment_status']['compiler_selection_source'] == 'explicit_configuration'
    observed = list(suite.rglob('observed_environment.json'))
    assert len(observed) == 2
    assert all(json.loads(path.read_text())['torch'] == 'overlay+cu126' for path in observed)
    assert dict(os.environ) == before
