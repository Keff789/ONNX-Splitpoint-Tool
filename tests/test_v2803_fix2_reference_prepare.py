"""Prepare -> serialized profile -> fresh normal start, without any execution."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from onnx_splitpoint_tool.workflow import hardware_matrix
from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
from onnx_splitpoint_tool.workflow.start_snapshot import snapshot_payload_sha256


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/v2803_night_regression'
SETUP = 'orin_nx_hailo10_01'
MODELS = ('mobilenet_v3_large', 'yolo11l')


def _gate():
    spec = importlib.util.spec_from_file_location('fix2_reference_gate', ROOT / 'scripts/reference_workflow_gate_v2803.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sources(tmp_path):
    profile = yaml.safe_load((FIXTURES / 'original/profile.yaml').read_text())
    profile['execution_preset']['follow_tool_config'] = False
    originals = tmp_path / 'originals'
    originals.mkdir()
    source = tmp_path / 'source'
    source.mkdir()
    for row in profile['model_suite']['primary']:
        model = row['id']
        row['onnx'] = str(originals / f'{model}.onnx')
        if model in MODELS:
            Path(row['onnx']).write_bytes(b'file-presence-only; never parsed or executed\n')
            suite = source / 'models' / model / 'benchmark_set/legacy_suite'
            suite.mkdir(parents=True)
            shutil.copy2(FIXTURES / 'evidence/original_benchmark_sets' / f'{model}.json', suite / 'benchmark_set.json')
    path = tmp_path / 'source.yaml'
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    return path, source, profile


def _forbid_registry(*args, **kwargs):
    raise AssertionError('Frozen target preparation must not reload the mutable hardware registry')


def test_prepare_cli_serialization_passes_fresh_normal_start_with_exact_subset(tmp_path):
    path, source, original = _sources(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    out = tmp_path / 'prepared'
    proc = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts/reference_workflow_gate_v2803.py'),
        'prepare', '--profile', str(path), '--source-run', str(source), '--output-dir', str(out)],
        capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'PREPARATION_STATUS=prepared_not_executed' in proc.stdout
    assert not (out / 'runs').exists()
    assert all(p.read_bytes() == value for p, value in before.items())

    prepared_path = out / 'profile_fixed16.yaml'
    prepared = yaml.safe_load(prepared_path.read_text())
    hardware = prepared['hardware']
    selected = [row for row in original['hardware']['resolved_targets'] if row['id'] == SETUP]
    assert len(original['hardware']['resolved_targets']) == 3
    assert hardware['resolved_targets'] == selected
    assert hardware['resolved_targets_sha256'] == snapshot_payload_sha256(selected)
    assert hardware['resolved_targets_sha256'] != original['hardware']['resolved_targets_sha256']
    assert hardware['selected_setups'] == [SETUP]
    assert hardware['selected_groups'] == []
    assert hardware['resolution_frozen_at_start'] is True
    assert prepared['hailo_build']['force_build'] is prepared['deepx_build']['force_build'] is False
    assert prepared['hailo_build']['calib_count'] == prepared['deepx_build']['calib_count'] == 500
    assert prepared['hailo_build']['compute_by_family'] == original['hailo_build']['compute_by_family']
    assert prepared['native_producers']['build_missing_engines'] is False
    assert prepared['artifact_cache_preflight']['default_expectation'] == 'warm'
    assert prepared['artifact_cache_preflight']['block_on_unexpected_cold_builds'] is True

    # Invoke precisely the configuration-loading functions used by the normal
    # workflow CLI, in another interpreter. No runner, inference or device call.
    normal_load = '''
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from onnx_splitpoint_tool.workflow import hardware_matrix
def forbidden_registry(*args, **kwargs):
    raise AssertionError('mutable registry fallback forbidden')
hardware_matrix._load_hardware_registry = forbidden_registry
from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot, workflow_options_from_profile_snapshot
from onnx_splitpoint_tool.workflow.start_snapshot import validate_profile_start_snapshot
profile, snapshot = load_runtime_profile_snapshot(sys.argv[2])
validate_profile_start_snapshot(snapshot)
opts = workflow_options_from_profile_snapshot(profile_request=sys.argv[2], out_root=sys.argv[3], start_snapshot=snapshot, require_fresh_run=True)
assert opts.require_fresh_run is True
assert profile['hardware']['selected_setups'] == ['orin_nx_hailo10_01']
assert not Path(sys.argv[3]).exists()
print('FRESH_NORMAL_START_CONFIGURATION=PASS')
'''
    loaded = subprocess.run([sys.executable, '-I', '-B', '-c', normal_load,
        str(ROOT), str(prepared_path), str(out / 'runs')], capture_output=True, text=True, timeout=60)
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    assert 'FRESH_NORMAL_START_CONFIGURATION=PASS' in loaded.stdout
    report = json.loads((out / 'preparation.json').read_text())
    assert report['status'] == 'prepared_not_executed'
    assert report['g2_status'] == report['g3_status'] == 'not_run'


@pytest.mark.parametrize('problem', ['invalid_original', 'missing_setup', 'duplicate_setup'])
def test_prepare_rejects_invalid_original_before_rebinding_without_registry(tmp_path, monkeypatch, problem):
    path, source, profile = _sources(tmp_path)
    hardware = profile['hardware']
    if problem == 'invalid_original':
        hardware['resolved_targets'][0]['remote']['host'] = 'changed-source.invalid'
        error = 'Frozen hardware target mapping hash mismatch'
    elif problem == 'missing_setup':
        hardware['resolved_targets'] = [row for row in hardware['resolved_targets'] if row['id'] != SETUP]
        hardware['resolved_targets_sha256'] = snapshot_payload_sha256(hardware['resolved_targets'])
        error = 'required_frozen_hardware_setup_missing'
    else:
        selected = next(row for row in hardware['resolved_targets'] if row['id'] == SETUP)
        hardware['resolved_targets'].append(copy.deepcopy(selected))
        hardware['resolved_targets_sha256'] = snapshot_payload_sha256(hardware['resolved_targets'])
        error = 'duplicate_frozen_hardware_setup'
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    before = path.read_bytes()
    monkeypatch.setattr(hardware_matrix, '_load_hardware_registry', _forbid_registry)
    with pytest.raises(ValueError, match=error):
        _gate().prepare_profile(path, source)
    assert path.read_bytes() == before
    assert not (tmp_path / 'runs').exists()


def test_prepared_mapping_tampering_still_fails_without_registry(tmp_path, monkeypatch):
    path, source, original = _sources(tmp_path)
    monkeypatch.setattr(hardware_matrix, '_load_hardware_registry', _forbid_registry)
    prepared = _gate().prepare_profile(path, source)
    prepared['hardware']['resolved_targets'][0]['remote']['host'] = 'changed-after-prepare.invalid'
    prepared_path = tmp_path / 'tampered.yaml'
    prepared_path.write_text(yaml.safe_dump(prepared, sort_keys=False))
    with pytest.raises(ValueError, match='Frozen hardware target mapping hash mismatch'):
        load_runtime_profile_snapshot(str(prepared_path))
    assert yaml.safe_load(path.read_text()) == original


def test_prepare_checks_persisted_profile_before_announcing_ready(tmp_path, monkeypatch):
    path, source, unused = _sources(tmp_path)
    module = _gate()
    original_prepare = module.prepare_profile
    def corrupt_generated_mapping(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        result['hardware']['resolved_targets'][0]['remote']['host'] = 'generator-regression.invalid'
        return result
    monkeypatch.setattr(module, 'prepare_profile', corrupt_generated_mapping)
    monkeypatch.setattr(hardware_matrix, '_load_hardware_registry', _forbid_registry)
    out = tmp_path / 'blocked'
    assert module.main(['prepare', '--profile', str(path), '--source-run', str(source), '--output-dir', str(out)]) == 2
    assert not (out / 'preparation.json').exists()
    failure = json.loads((out / 'failure.json').read_text())
    assert failure['status'] == 'blocked'
    assert 'Frozen hardware target mapping hash mismatch' in failure['error']
    assert not (out / 'runs').exists()
