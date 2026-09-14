"""AP5: real start/final displays separate requested settings from measured evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from onnx_splitpoint_tool.measurement_configuration import (
    measurement_configuration, measurement_configuration_lines,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/v2803_night_regression'


def _night_inputs():
    return (yaml.safe_load((FIXTURE/'original/profile.yaml').read_bytes()),
            json.loads((FIXTURE/'original/effective_execution_plan.json').read_bytes()))


def test_original_night_configuration_keeps_screening_and_gpu_preference_unqualified():
    profile, plan = _night_inputs()
    before_profile, before_plan = copy.deepcopy(profile), copy.deepcopy(plan)
    config = measurement_configuration(profile, plan)
    assert config['validation_items'] == {'classification': 5000, 'detection': 5000}
    assert config['bootstrap_repetitions'] == 5000
    energy = config['native_energy']
    assert (energy['duration_s'], energy['repetitions']) == (1.0, 3)
    assert (energy['physical_scope'], energy['window_label']) == ('FS', 'command')
    assert energy['physical_scope_source'] == 'execution_preset.snapshot.energy.physical_scope'
    assert energy['window_label_source'] == 'execution_preset.snapshot.energy.window_label'
    assert energy['measured_duration_s'] is None
    assert energy['scientific_qualification'] == 'requires_row_evidence'
    assert config['build_preferences'] == {'hailo8': 'gpu', 'hailo10h': 'gpu'}
    assert config['gpu_execution'] == 'not_established_by_configuration'
    assert config['profile_label_is_scientific_approval'] is False
    assert (profile, plan) == (before_profile, before_plan)
    provenance = json.loads((FIXTURE/'measurement_configuration_provenance.json').read_text())
    for row in provenance['files']:
        data = (FIXTURE/row['fixture_path']).read_bytes()
        assert len(data) == row['size_bytes']
        assert hashlib.sha256(data).hexdigest() == row['sha256']


@pytest.mark.parametrize('value', [None, True, False, '5000', -1, 1.5, float('nan'), float('inf')])
def test_unknown_or_invalid_counts_never_become_default_or_measured_zero(value):
    profile, _ = _night_inputs()
    config = measurement_configuration(profile, {'validation_items': {'classification': value},
                                                  'bootstrap_repetitions': value})
    assert config['validation_items'] == {'classification': None, 'detection': None}
    assert config['bootstrap_repetitions'] is None
    assert 'CLS/DET=unavailable/unavailable; bootstrap=unavailable' in measurement_configuration_lines(config)[0]


def test_explicit_zero_count_and_large_integer_are_preserved_without_float_coercion():
    config = measurement_configuration({}, {'validation_items': {'classification': 0, 'detection': 10**400},
                                           'bootstrap_repetitions': 0})
    assert config['validation_items'] == {'classification': 0, 'detection': 10**400}
    assert config['bootstrap_repetitions'] == 0
    assert config['native_energy']['duration_s'] is None
    assert config['native_energy']['repetitions'] is None


def test_historical_missing_configuration_gets_no_current_scope_or_duration_defaults():
    config = measurement_configuration({'name': 'Final Quality (Standard+)'})
    assert config['validation_items'] == {'classification': None, 'detection': None}
    assert config['bootstrap_repetitions'] is None
    energy = config['native_energy']
    for key in ('physical_scope', 'window_label', 'duration_s', 'repetitions', 'measured_duration_s'):
        assert energy[key] is None, key
    assert config['build_preferences'] == {'hailo8': 'unavailable', 'hailo10h': 'unavailable'}
    assert config['profile_label_is_scientific_approval'] is False
    text = '\n'.join(measurement_configuration_lines(config))
    assert 'scope=unavailable; window=unavailable' in text
    assert 'requires job evidence' in text


def test_explicit_family_configuration_and_energy_override_do_not_change_profile():
    profile, plan = _night_inputs()
    profile['hailo_build']['compute_by_family']['hailo8']['device'] = 'cpu'
    profile['native_producers']['energy'].update(duration_s=30, repeat_override=5,
                                                physical_scope='declared_test_scope', window_label='declared_test_window')
    before = copy.deepcopy(profile)
    config = measurement_configuration(profile, plan)
    assert config['build_preferences'] == {'hailo8': 'cpu', 'hailo10h': 'gpu'}
    assert config['gpu_execution'] == 'not_established_by_configuration'
    energy = config['native_energy']
    assert (energy['duration_s'], energy['repetitions'], energy['physical_scope'], energy['window_label']) == (
        30, 5, 'declared_test_scope', 'declared_test_window')
    assert energy['measured_duration_s'] is None
    assert profile == before
    assert profile['hailo_build']['force_build'] is False
    assert profile['hailo_build']['cache_integrity'] == 'relaxed'


def test_actual_gui_start_summary_in_fresh_process_reads_effective_profile(tmp_path):
    # A derived configuration copy is a new start request, not a historical replay.
    # The entire profile loader/resolver/summary run; only window creation is unnecessary.
    profile, _ = _night_inputs()
    profile['native_producers']['energy']['duration_s'] = 30
    profile['native_producers']['energy']['duration_source'] = 'profile.native_producers.energy.duration_s'
    path = tmp_path/'requested_start.yaml'
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    before = path.read_bytes()
    code = '''
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(root))
from onnx_splitpoint_tool.gui.panels.panel_evaluation_workflow import _profile_summary_payload
lines, visible = _profile_summary_payload(sys.argv[2])
print(json.dumps({"lines": lines, "visible": visible}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT), str(path)],
                            text=True, capture_output=True, timeout=45,
                            env={**os.environ, 'ORT_DISABLE_TELEMETRY': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    text = '\n'.join(payload['lines'])
    assert payload['visible'], text
    assert 'Resolved validation CLS/DET=5000/5000; bootstrap=5000' in text
    assert 'Native energy request: 30.0 s × 3; scope=FS; window=command' in text or 'Native energy request: 30 s × 3; scope=FS; window=command' in text
    assert 'measured duration and qualification require row evidence' in text
    assert 'GPU execution requires job evidence; compatible artifacts remain reusable' in text
    assert 'Profile label is configuration; scientific approval is determined per result.' in text
    assert 'Hailo Force: AUS' in text and 'DeepX Force: AUS' in text
    assert path.read_bytes() == before


@pytest.mark.parametrize('archive_config', [True, False])
def test_actual_final_scientific_report_replay_uses_only_saved_configuration(tmp_path, archive_config):
    from onnx_splitpoint_tool.workflow.scientific_reporting import build_scientific_reports

    run = tmp_path/'historical_run'
    run.mkdir()
    if archive_config:
        for name in ('profile.yaml', 'effective_execution_plan.json'):
            shutil.copyfile(FIXTURE/'original'/name, run/name)
    else:
        (run/'profile.yaml').write_text('name: Final Quality (Standard+)\n')
    before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    # External replay invokes actual loaders, aggregation, writer and publication.
    report_dir = tmp_path/'new_report'
    result = build_scientific_reports(run, cleanup_legacy=False, output_dir=report_dir)
    payload = json.loads((report_dir/'scientific_report.json').read_text())
    text = (report_dir/'scientific_report.md').read_text()
    config = payload['measurement_configuration']
    assert config['gpu_execution'] == 'not_established_by_configuration'
    assert config['native_energy']['measured_duration_s'] is None
    assert config['profile_label_is_scientific_approval'] is False
    assert config['role'] == 'resolved_configuration_only'
    if archive_config:
        assert config['build_preferences'] == {'hailo8': 'gpu', 'hailo10h': 'gpu'}
        assert config['validation_items'] == {'classification': 5000, 'detection': 5000}
        assert 'Native energy request: 1.0 s × 3; scope=FS; window=command' in text
        assert 'New-build preferences: Hailo8=gpu; Hailo10H=gpu' in text
        assert (report_dir/'official_coco/official_coco_index.json').is_file()
    else:
        assert config['validation_items'] == {'classification': None, 'detection': None}
        assert config['native_energy']['physical_scope'] is None
        assert 'scope=unavailable; window=unavailable' in text
    assert 'GPU execution requires job evidence' in text
    assert result['scientific_status'] != 'pass'
    after = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    assert after == before


def test_direct_official_coco_call_keeps_its_existing_default_output(tmp_path):
    from onnx_splitpoint_tool.validation.official_coco_orchestrator import run_official_coco_for_run

    run = tmp_path/'run'
    run.mkdir()
    result = run_official_coco_for_run(run, annotations=tmp_path/'not_present.json',
                                       enabled=True, required=False)
    index = run/'reports/scientific/official_coco/official_coco_index.json'
    assert json.loads(index.read_text()) == result
    assert result['status'] == 'unavailable'
    assert result['rows'] == []
