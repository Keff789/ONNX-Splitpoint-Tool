"""AP0/AP5 provenance and predeclared diagnostic scope; no numerical repair."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import yaml

from onnx_splitpoint_tool.run_modes import apply_run_mode
from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import resolve_artifact_cache_preflight_policy

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/v2803_night_regression'


def _read(path):
    return json.loads(Path(path).read_text())


def _gate():
    spec = importlib.util.spec_from_file_location('v2803_reference_cli', ROOT / 'scripts/reference_workflow_gate_v2803.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _prepare_sources(tmp_path):
    profile = yaml.safe_load((FIXTURES / 'original/profile.yaml').read_text())
    profile['execution_preset']['follow_tool_config'] = False
    # Archived absolute paths may exist on the machine running acceptance.
    # Exercise this file-presence gate exclusively with temporary inputs; these
    # placeholders are never passed to inference or any model parser.
    models = tmp_path / 'original_models'
    models.mkdir()
    for row in profile['model_suite']['primary']:
        model_path = models / (row['id'] + '.onnx')
        row['onnx'] = str(model_path)
        if row['id'] in _gate().MODELS:
            model_path.write_bytes(b'file-presence-only test fixture\n')
    path = tmp_path / 'original_profile.yaml'
    path.write_text(yaml.safe_dump(profile))
    source = tmp_path / 'historical_source'; source.mkdir()
    for model in _gate().MODELS:
        dst = source / 'models' / model / 'benchmark_set/legacy_suite'
        dst.mkdir(parents=True)
        shutil.copy2(FIXTURES / 'evidence/original_benchmark_sets' / (model + '.json'), dst / 'benchmark_set.json')
    return path, source


def test_t04_preparation_accepts_existing_temporary_originals_without_mutation(tmp_path):
    path, source = _prepare_sources(tmp_path)
    before = path.read_bytes()
    originals = {row['id']: Path(row['onnx']) for row in yaml.safe_load(before)['model_suite']['primary']}
    selected = {model: originals[model] for model in _gate().MODELS}
    contents = {model: file.read_bytes() for model, file in selected.items()}
    result = _gate().prepare_profile(path, source, require_originals=True)
    assert {row['id']: row['onnx'] for row in result['model_suite']['primary']} == {model: str(file) for model, file in selected.items()}
    assert path.read_bytes() == before
    assert {model: file.read_bytes() for model, file in selected.items()} == contents
    assert all(file.is_relative_to(tmp_path) for file in originals.values())
    assert all(not file.exists() for model, file in originals.items() if model not in selected)


def test_t00_original_night_fixture_bytes_versions_and_reference_failures():
    provenance = _read(FIXTURES / 'PROVENANCE.json')
    for row in provenance['original_members'] + provenance['derived_members']:
        body = (FIXTURES / row['path']).read_bytes()
        assert len(body) == row['size_bytes']
        assert hashlib.sha256(body).hexdigest() == row['sha256']
    diagnostics = list((FIXTURES / 'original/quality_management/references').rglob('management_cpu_reference_*'))
    assert len(diagnostics) == 14
    assert sum(p.stat().st_size for p in diagnostics) == 28379
    for path in diagnostics:
        if path.suffix == '.json':
            status = _read(path)
            assert status['status'] == 'failed'
            assert 'model_binding' in status['error']
    metadata = FIXTURES / 'evidence/original_benchmark_sets'
    assert len([p for p in metadata.glob('*.json') if p.name != 'INDEX.json']) == 14
    for model in ('mobilenet_v3_large', 'regnet_x_1_6gf', 'resnet50', 'yolo11l', 'yolo26m', 'yolo26s', 'yolov7_paper'):
        data = _read(metadata / (model + '.json'))
        assert data['model_name'] == model and data['model'] == f'models/{model}.onnx'


def test_t04_prepare_fixed16_normal_profile_preserves_recipes_and_blocks_misses(tmp_path):
    path, source = _prepare_sources(tmp_path)
    before = path.read_bytes()
    result = _gate().prepare_profile(path, source, require_originals=False)
    again, _ = apply_run_mode(result, follow_tool_config=False)
    assert path.read_bytes() == before
    assert {r['id'] for r in again['model_suite']['primary']} == {'mobilenet_v3_large', 'yolo11l'}
    assert again['selection_policy']['forced_cases'] == {'mobilenet_v3_large': ['b056'], 'yolo11l': ['b062']}
    assert again['validation_execution']['max_items'] == {'classification': 16, 'detection': 16}
    assert again['quality_gate']['statistics']['bootstrap_repetitions'] == 100
    assert again['hailo_build']['calib_count'] == again['deepx_build']['calib_count'] == 500
    assert again['hailo_build']['preset'] == 'balanced' and again['hailo_build']['optimization_level'] == 1
    assert again['hailo_build']['calib_batch_size'] == 8
    assert again['deepx_build']['opt_level'] == 0 and again['deepx_build']['calibration_method'] == 'ema'
    assert again['deepx_build']['classification_preprocessing'] == 'imagenet_mean_std'
    assert again['hailo_build']['force_build'] is again['deepx_build']['force_build'] is False
    assert again['hailo_build']['compute_by_family'] == yaml.safe_load(before)['hailo_build']['compute_by_family']
    assert again['native_producers']['energy']['enabled'] is again['energy']['enabled'] is False
    assert again['native_producers']['build_missing_engines'] is False
    assert not cache_verify_guard(again)
    cache = resolve_artifact_cache_preflight_policy(again)
    assert cache['default_expectation'] == 'warm' and cache['block_on_unexpected_cold_builds'] is True
    assert again['native_producers']['case_map'] == again['selection_policy']['forced_cases']
    assert again['hardware']['selected_setups'] == ['orin_nx_hailo10_01']


@pytest.mark.parametrize('problem', ['original_model_missing', 'fixed_case_missing', 'bad_recipe'])
def test_t04_preparation_blocks_missing_inputs_without_workflow_or_profile_mutation(tmp_path, problem):
    path, source = _prepare_sources(tmp_path)
    if problem == 'original_model_missing':
        profile = yaml.safe_load(path.read_text())
        missing = Path(next(row['onnx'] for row in profile['model_suite']['primary'] if row['id'] == 'mobilenet_v3_large'))
        assert missing.is_relative_to(tmp_path) and missing.is_file()
        missing.unlink()
        assert not missing.exists()
    elif problem == 'fixed_case_missing':
        contract = source / 'models/yolo11l/benchmark_set/legacy_suite/benchmark_set.json'
        data = _read(contract); data['cases'] = []
        contract.write_text(json.dumps(data))
    elif problem == 'bad_recipe':
        data = yaml.safe_load(path.read_text())
        data['execution_preset']['snapshot']['build']['hailo']['calibration_items'] = 499
        data['execution_preset']['snapshot']['data']['calibration_items']['classification'] = 499
        data['execution_preset']['snapshot']['data']['calibration_items']['detection'] = 499
        path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    with pytest.raises(ValueError, match={'original_model_missing': 'original_model_missing', 'fixed_case_missing': 'fixed_case_missing', 'bad_recipe': 'recipe_differs'}[problem]):
        _gate().prepare_profile(path, source, require_originals=problem == 'original_model_missing')
    assert path.read_bytes() == before
    assert not (tmp_path / 'runs').exists()


def test_t05_original_numerical_diagnostics_keep_early_late_and_negative_cases():
    rows = list(csv.DictReader((FIXTURES / 'evidence/night_mini_classification_by_case.csv').open()))
    # These are the archived Generic mini-evaluation outputs, not new central
    # quality decisions and not interchangeable intermediate tensor shapes.
    mobile = [r for r in rows if r.get('model') == 'mobilenet_v3_large']
    assert mobile
    serial = json.dumps(mobile)
    assert 'b056' in serial and 'b135' in serial
    negative = (FIXTURES / 'evidence/independent_split_blocked_replay.json').read_text()
    assert 'b398' in negative and 'b364' in negative
    assert 'score_column_not_probability_like' in negative and 'coordinates_not_ordered_xyxy' in negative
    policy = _read(ROOT / 'docs/V2803_DIAGNOSTIC_SCOPE.json')
    assert policy['mobilenet']['predeclared_cases'] == ['full', 'b056', 'b135']
    assert policy['mobilenet']['intermediate_comparison'] == 'each_part1_against_its_own_float_part1'
    assert policy['yolo26']['retained_negative_cases'] == {'yolo26m': 'b398', 'yolo26s': 'b364'}
    assert policy['yolo26']['part2_input'] == 'same_captured_handover_tensor_for_tensorrt_and_float'
    assert policy['har_emulation_without_bound_artifact'] == 'not_available'
    assert policy['numerical_product_patch'] == 'none_proven'
    assert policy['new_hardware_jobs'] == policy['device_configuration_writes'] == 0


@pytest.mark.parametrize('failure', ['dependency_missing', 'dependency_timeout'])
def test_t04_dependency_probe_failure_is_blocked_with_evidence(tmp_path, monkeypatch, failure):
    import subprocess
    module = _gate()
    source = tmp_path / 'original'; source.mkdir()
    out = tmp_path / 'new_diagnostic'
    def dependency_boundary(*args, **kwargs):
        assert kwargs['timeout'] == 30
        if failure == 'dependency_timeout':
            raise subprocess.TimeoutExpired(args[0], 30, output=b'probe started\n')
        return subprocess.CompletedProcess(args[0], 1, '', "ModuleNotFoundError: No module named 'onnxruntime'\n")
    monkeypatch.setattr(module.subprocess, 'run', dependency_boundary)
    rc = module.main(['prepare', '--profile', str(tmp_path / 'unused.yaml'),
                      '--source-run', str(source), '--output-dir', str(out)])
    assert rc == 2
    report = _read(out / 'failure.json')
    assert report['status'] == 'blocked' and 'environment_blocked' in report['error']
    assert report['g2_status'] == report['g3_status'] == 'not_accepted'
    assert (out / 'dependencies.log').is_file()
    assert not (out / 'profile_fixed16.yaml').exists()
    assert not list(source.iterdir())
