"""Explicit Generic Quality transport guards; no devices, models or inference."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'onnx_splitpoint_tool/resources/templates'


def functions(filename, names):
    source = TEMPLATES / filename
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    ns = dict(Path=Path, Any=Any, Dict=Dict, List=List, Mapping=Mapping,
              Optional=Optional, Sequence=Sequence, Tuple=Tuple,
              json=json, hashlib=hashlib, re=re, os=os, sys=sys)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


@pytest.fixture(autouse=True)
def isolate_artifact_policy(monkeypatch):
    # Register restoration even if the key starts absent: the CLI guard sets
    # os.environ directly, while delenv alone records no undo for an absent key.
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY", ""))
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)


def runner_args(**changes):
    args = dict(quality_source_only=True, variants='composed',
        native_split_quality_binding='', generic_quality_completion_config='quality_runtime/b039.json',
        existing_management_cpu_reference_json='{}', validation_images='frozen-dataset',
        validation_max_images=5000,
        quality_gate_json=json.dumps({'statistics': {'execution_location': 'central_management'}}),
        generic_completion_config='', quality_evidence_only=False, build_only=False,
        energy_measurement_only=False, output_dump_only=False, external_output_dump='',
        full_only_quality_identity_json='', warmup=100, runs=1000, phase_runs=5,
        throughput_frames=24, throughput_warmup_frames=6, native_trt_build=True,
        native_trt_no_fallback=False, viz='auto', no_report_plots=False)
    args.update(changes)
    return SimpleNamespace(**args)


def configure(args):
    ns = functions('run_split_onnxruntime.py.txt', ['_configure_quality_source_only',
                   '_load_task_quality_policy', '_task_quality_execution_location'])
    ns['_configure_quality_source_only'](args, argparse.ArgumentParser())


def test_explicit_generic_quality_preserves_source_and_disables_timing_builds():
    args = runner_args()
    configure(args)
    assert args.native_split_quality_binding == ''
    assert args.generic_quality_completion_config == 'quality_runtime/b039.json'
    assert [args.warmup, args.runs, args.phase_runs, args.throughput_frames,
            args.throughput_warmup_frames] == [0] * 5
    assert args.native_trt_build is False and args.native_trt_no_fallback is True
    assert args.trt_runtime == args.trt_runtime_mode == 'native'
    assert os.environ['ONNX_SPLITPOINT_ARTIFACT_POLICY'] == 'cache_verify_only'


@pytest.mark.parametrize('change', [
    {'native_split_quality_binding': 'fake-native.json'},
    {'generic_quality_completion_config': ''}, {'quality_source_only': False},
    {'variants': 'full'}, {'variants': 'composed,part1'},
    {'quality_evidence_only': True}, {'build_only': True}, {'energy_measurement_only': True},
    {'output_dump_only': True}, {'external_output_dump': 'out'},
    {'generic_completion_config': 'timed.json'}, {'full_only_quality_identity_json': '{}'},
    {'existing_management_cpu_reference_json': ''}, {'validation_images': ''},
    {'validation_max_images': 0},
    {'quality_gate_json': json.dumps({'statistics': {'execution_location': 'local'}})},
])
def test_explicit_generic_quality_rejects_other_roles_and_unbound_inputs(change):
    with pytest.raises(SystemExit):
        configure(runner_args(**change))


def test_native_default_and_ordinary_scope_remain_unchanged():
    native = runner_args(native_split_quality_binding='actual-native.json',
                         generic_quality_completion_config='')
    configure(native)
    assert native.native_split_quality_binding == 'actual-native.json'
    ordinary = runner_args(quality_source_only=False, generic_quality_completion_config='')
    before = dict(vars(ordinary))
    configure(ordinary)
    assert vars(ordinary) == before


@pytest.fixture
def transport(tmp_path, monkeypatch):
    suite = tmp_path / 'suite'
    case = suite / 'b039'
    case.mkdir(parents=True)
    (case / 'run_split_onnxruntime.py').write_text('# --phase-runs')
    path = suite / 'quality_runtime/b039.json'
    path.parent.mkdir()
    path.write_text('{}')
    config = {'identity': {'model_id': 'yolo26s', 'case_id': 'b039',
              'setup_id': 'orin_nx_hailo10_01', 'precision': 'float32_layout_fp16',
              'comparison_backend': 'hailo10h'}}
    calls = []
    module = ModuleType('splitpoint_runners.generic_quality_completion')
    def load(candidate, *, check_files):
        assert check_files is True
        assert Path(candidate) == path
        calls.append(str(candidate))
        return config
    module.load_quality_completion_config = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    accounting = ModuleType('splitpoint_runners.generic_quality_accounting')
    monkeypatch.setitem(sys.modules, accounting.__name__, accounting)
    # The actual loader's contract/artifact/source checks have their own tests;
    # this fixture isolates suite selection, opt-in and forwarding to that API.
    return suite, case, path, config, calls


def selection(transport):
    suite, case, path, config, calls = transport
    run = {'quality_source_only': True, 'case_ids': ['b039'],
           'generic_quality_completion_configs': {'b039': 'quality_runtime/b039.json'},
           'generic_quality_completion_config_sha256': {'b039': hashlib.sha256(path.read_bytes()).hexdigest()}}
    ns = functions('benchmark_suite.py.txt', ['_generic_quality_source_for_case'])
    options = dict(model_id='yolo26s', setup_id='orin_nx_hailo10_01',
                   stage1='hailo10', stage2='tensorrt', variants=['composed'])
    return ns['_generic_quality_source_for_case'], run, options


def test_suite_selects_exact_bound_case_and_checks_real_files_api(transport):
    suite, case, path, config, calls = transport
    select, run, options = selection(transport)
    assert select(suite, run, case, **options) == {'path': str(path), 'config': config}
    assert calls == [str(path)]
    assert select(suite, {}, case, **options) is None
    assert select(suite, {'quality_source_only': True}, case, **options) is None


@pytest.mark.parametrize('change', [
    {'quality_source_only': False}, {'case_ids': ['b039', 'b039']},
    {'case_ids': ['b039', 'b077']}, {'generic_quality_completion_configs': {}},
    {'generic_quality_completion_configs': None},
    {'generic_quality_completion_configs': {'b039': '../escape.json'}},
    {'generic_quality_completion_config_sha256': None},
    {'generic_quality_completion_config_sha256': {'b039': '0' * 64}},
    {'generic_quality_completion_config_sha256': {'b077': '0' * 64}},
])
def test_suite_rejects_mapping_scope_or_profile_byte_substitution(transport, change):
    suite, case, _, _, calls = transport
    select, run, options = selection(transport)
    run.update(change)
    with pytest.raises((ValueError, FileNotFoundError)):
        select(suite, run, case, **options)
    assert calls == []


@pytest.mark.parametrize('field,value', [
    ('model_id', 'other'), ('case_id', 'b077'), ('setup_id', 'other'),
    ('comparison_backend', 'deepx_m1'),
])
def test_suite_rejects_wrong_actual_config_identity(transport, field, value):
    suite, case, _, config, _ = transport
    select, run, options = selection(transport)
    config['identity'][field] = value
    with pytest.raises(ValueError, match='differs'):
        select(suite, run, case, **options)


def child_namespace(transport):
    suite, case, path, config, calls = transport
    ns = functions('benchmark_suite.py.txt', ['_run_case'])
    commands = []
    result_calls = []
    def collect(*args, **kwargs):
        result_calls.append(kwargs)
        return {'observed_child_exit': kwargs['runner_rc']}
    # This transport fixture replaces only leaf collection; the dedicated
    # accounting tests and normal generated-suite fixture verify real proofs.
    sys.modules['splitpoint_runners.generic_quality_accounting'].collect_capture = (
        lambda *args, **kwargs: collect(*args, runner_rc=0))
    def dispatch(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0)
    ns.update(_management_cpu_reference_context=lambda **kw: False,
              _bound_management_validation_source=lambda *args: suite,
              _cache_verify_only=lambda: False,
              _stage2_accelerator_gate=lambda *args: {},
              _provider_unavailable_reason=lambda *args: None,
              _native_trt_ort_preflight_tokens=lambda *args: [],
              _provider_tokens_from_stage=lambda *args: [],
              _slug=lambda value: value,
              _collect_case_result=collect,
              _resolve_semantic_validation_source=lambda *args: suite,
              _select_timing_sample_image=lambda *args: (None, None),
              _stage2_calibration_evidence_env=lambda *args: '',
              subprocess=SimpleNamespace(run=dispatch, TimeoutExpired=TimeoutError))
    kwargs = dict(run_id='hailo10_to_tensorrt', provider='hailo10', image='default',
        preset='default', image_scale='norm', warmup=0, runs=0, timeout_s=10,
        stage1='hailo10', stage2='tensorrt', variants=['composed'],
        quality_source_only=True, generic_quality_completion_config=str(path),
        quality_evidence_eval_id='frozen-run', quality_evidence_model_id='yolo26s',
        quality_evidence_setup_id='orin_nx_hailo10_01',
        quality_evidence_source_run_id='hailo10_to_tensorrt', benchmark_task='detection',
        task_quality_gate={'statistics': {'execution_location': 'central_management'}},
        management_cpu_reference={'reference_sha256': '1' * 64},
        validation_images=str(suite), validation_max_images=5000)
    return ns['_run_case'], kwargs, commands, result_calls


def test_suite_forwards_generic_quality_and_preserves_cpu_dataset_no_build(transport):
    _, case, path, _, _ = transport
    child, kwargs, commands, results = child_namespace(transport)
    # Even with resume requested, a previous raw-endpoint Quality report must
    # not suppress the new exact-completion capture.
    child(case, **kwargs, resume=True)
    assert len(commands) == len(results) == 1
    cmd = commands[0]
    assert cmd[cmd.index('--generic-quality-completion-config') + 1] == str(path)
    assert '--quality-source-only' in cmd and '--native-split-quality-binding' not in cmd
    assert '--no-native-trt-build' in cmd and '--native-trt-build' not in cmd
    assert cmd[cmd.index('--trt-runtime') + 1] == 'native'
    assert cmd[cmd.index('--validation-max-images') + 1] == '5000'
    assert '--existing-management-cpu-reference-json' in cmd
    assert '--quality-evidence-only' not in cmd and '--energy-measurement-only' not in cmd


def test_suite_native_binding_dispatch_remains_native(transport):
    _, case, _, _, calls = transport
    child, kwargs, commands, _ = child_namespace(transport)
    kwargs.update(generic_quality_completion_config='', native_split_quality_binding='native.json')
    child(case, **kwargs)
    assert not calls
    cmd = commands[0]
    assert '--native-split-quality-binding' in cmd
    assert '--generic-quality-completion-config' not in cmd


@pytest.mark.parametrize('change', [
    {'quality_source_only': False}, {'quality_evidence_only': True},
    {'measurement_only': True}, {'native_split_quality_binding': 'fake.json'},
    {'variants': ['full']}, {'management_cpu_reference': None},
])
def test_suite_rejects_bad_generic_child_before_subprocess(transport, change):
    _, case, _, _, _ = transport
    child, kwargs, commands, _ = child_namespace(transport)
    kwargs.update(change)
    with pytest.raises((ValueError, RuntimeError)):
        child(case, **kwargs)
    assert commands == []


@pytest.mark.parametrize('fault', [None, 'artifact', 'source'])
def test_suite_uses_actual_normal_full_config_loader(tmp_path, monkeypatch, fault):
    from tests.test_generic_quality_completion import make_config
    from onnx_splitpoint_tool.runners import generic_quality_completion as loader
    suite = tmp_path / 'suite'
    suite.mkdir()
    fixture = make_config(suite)
    config = fixture['config']
    config['identity']['comparison_backend'] = 'hailo10h'
    fixture['path'].write_text(json.dumps(config))
    if fault == 'artifact':
        Path(config['expected_runtime_artifacts']['engine']['path']).write_bytes(b'changed TEST ONLY')
    elif fault == 'source':
        config['runtime_source_files'][-1]['sha256'] = '0' * 64
        fixture['path'].write_text(json.dumps(config))
    monkeypatch.setitem(sys.modules, 'splitpoint_runners.generic_quality_completion', loader)
    select = functions('benchmark_suite.py.txt', ['_generic_quality_source_for_case'])['_generic_quality_source_for_case']
    run = {'quality_source_only': True, 'case_ids': ['b039'],
           'generic_quality_completion_configs': {'b039': fixture['path'].name},
           'generic_quality_completion_config_sha256': {'b039': hashlib.sha256(fixture['path'].read_bytes()).hexdigest()}}
    kwargs = dict(model_id='yolo26s', setup_id='test_only_h10', stage1='hailo10',
                  stage2='tensorrt', variants=['composed'])
    if fault:
        with pytest.raises(ValueError, match='changed'):
            select(suite, run, suite / 'b039', **kwargs)
    else:
        result = select(suite, run, suite / 'b039', **kwargs)
        assert result['config']['expected_runtime_artifacts'] == config['expected_runtime_artifacts']
        assert set(result['config']['expected_runtime_artifacts']) == set(loader.ARTIFACT_ROLES)
