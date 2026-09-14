"""Review of delivered v282 controller scope and failure reporting.

Profile preparation and process supervision are real. Workflow/SSH boundaries
are explicitly mocked in controller tests; none is a hardware acceptance.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate():
    return load(ROOT / 'scripts/reference_workflow_gate_v282.py', 'v282_gate_controller_test')


def controller():
    candidates = [Path(os.environ.get('V282_BUNDLE_DIR', '/nonexistent')),
                  ROOT.parent, ROOT.parent / 'ONNX-Splitpoint-Tool_v2.82_COMPLETE_DELIVERY_BUNDLE',
                  ROOT / 'scripts/release_bundle']
    bundle = next((path for path in candidates if (path / 'install_and_accept_v282.py').is_file()), None)
    assert bundle is not None, 'Canonical source controller is required by standalone source acceptance'
    canonical = ROOT / 'scripts/release_bundle'
    for name in ('install_and_accept_v282.py', 'acceptance_process.py'):
        assert (bundle / name).read_bytes() == (canonical / name).read_bytes(), 'Delivered controller differs from canonical source: ' + name
    sys.path.insert(0, str(bundle))
    try:
        return load(bundle / 'install_and_accept_v282.py', 'v282_bundle_controller_test')
    finally:
        sys.path.remove(str(bundle))


@pytest.mark.parametrize('group', list(gate().SCOPE))
def test_target_profiles_keep_exact_cases_and_standard_recipe(tmp_path, monkeypatch, group):
    g = gate()
    fixtures = load(ROOT / 'tests/test_v281_workflow_acceptance.py', 'v282_previous_controller_fixtures')
    monkeypatch.setattr(fixtures, 'gate', lambda: g)
    source, manifest = fixtures.sources(tmp_path)
    before = (source / 'profile.yaml').read_bytes()
    path, report = g.prepare(source, tmp_path / 'prepared', group, hailo8_dependency_manifest=manifest)
    actual = yaml.safe_load(path.read_text())
    scope = g.SCOPE[group]
    assert actual['selection_policy']['forced_cases'] == scope['models']
    assert actual['hardware']['selected_setups'] == ['orin_nx_hailo8_01']
    assert actual['hailo_build']['calib_count'] == 500 and actual['hailo_build']['force_build'] is False
    assert actual['validation_execution']['max_items'] == {'classification': 500, 'detection': 500}
    assert actual['native_producers']['case_map'] == scope['models']
    assert actual['native_producers']['energy']['enabled'] is False
    assert actual['native_producers']['full_baselines']['backends'] == ['hailo8', 'tensorrt']
    assert report['scientific_final_claim'] is False
    assert (source / 'profile.yaml').read_bytes() == before


def normal_controller(tmp_path, monkeypatch, first_state):
    g = gate()
    tool = tmp_path / 'tool'
    py = tool / '.venv/bin/python'; py.parent.mkdir(parents=True); py.symlink_to(sys.executable)
    monkeypatch.setattr(g, 'ROOT', tool)
    out = tmp_path / 'normal'; out.mkdir()
    calls = []
    def prepare(source, output, group, **kwargs):
        output.mkdir(parents=True)
        path = output / 'profile.yaml'; path.write_text(group)
        return path, {'status': 'prepared_not_executed', 'fixture_boundary': True}
    def logged(command, cwd, log, **kwargs):
        calls.append(command); log.write_text('mocked external execution boundary')
        if 'onnx_splitpoint_tool.workflow.run_evaluation' in command:
            destination = Path(command[command.index('--out') + 1])
            marker = destination / 'only_run/run_manifest.json'; marker.parent.mkdir(parents=True); marker.write_text('{}')
            if 'h8_yolo11l_b064' in str(destination):
                return dict(first_state)
        elif '--recheck-run' in command:
            Path(command[command.index('--recheck-output') + 1]).write_text(json.dumps({
                'status': 'pass', 'pid': os.getpid() + 100000, 'compiler_dispatch_count': 0,
                'fixture_boundary': True}))
        else:
            Path(command[command.index('--out') + 1]).write_bytes(b'mocked debug archive')
        return {'returncode': 0, 'cancelled': False, 'timed_out': False}
    monkeypatch.setattr(g, 'prepare', prepare)
    monkeypatch.setattr(g, 'run_logged', logged)
    monkeypatch.setattr(g, 'inspect_run', lambda run, group: {'technical_acceptance': 'pass', 'quality_decision': 'fail'})
    return g, out, calls


@pytest.mark.parametrize('state,expected_rc,workflow_count', [
    ({'returncode': 2, 'cancelled': False, 'timed_out': False}, 2, 2),
    ({'returncode': 124, 'cancelled': True, 'timed_out': True}, 2, 2),
    ({'returncode': 130, 'cancelled': True, 'timed_out': False}, 130, 1),
    ({'returncode': 0, 'cancelled': False, 'timed_out': False}, 0, 4),
])
def test_cold_failure_stops_later_colds_keeps_warm_control_and_exports(tmp_path, monkeypatch, state, expected_rc, workflow_count):
    g, out, calls = normal_controller(tmp_path, monkeypatch, state)
    assert g.execute(tmp_path / 'source', out) == expected_rc
    summary = json.loads((out / 'acceptance.json').read_text())
    assert len([c for c in calls if 'onnx_splitpoint_tool.workflow.run_evaluation' in c]) == workflow_count
    assert summary['technical_acceptance'] == ('pass' if expected_rc == 0 else 'incomplete')
    assert summary['quality_decision'] == 'fail'  # Valid quality FAIL never forces technical failure.
    assert summary['scientific_final_claim'] is False
    if expected_rc == 0:
        assert len([c for c in calls if '--recheck-run' in c]) == 3
        assert all(row.get('warm_recheck_pass') is True for group, row in summary['families'].items() if g.SCOPE[group]['allowed_missing'])
    with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
        assert archive.testzip() is None
        assert any(name.endswith('acceptance.json') for name in archive.namelist())
        assert not any('/runs/' in name for name in archive.namelist())


def test_current_controller_preserves_real_child_failure_and_timeout(tmp_path):
    g = gate()
    failure = g.run_logged([sys.executable, '-I', '-B', '-c', 'print("real child"); raise SystemExit(9)'],
                           tmp_path, tmp_path / 'failure.log', timeout_s=5)
    timeout = g.run_logged([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(20)'],
                           tmp_path, tmp_path / 'timeout.log', timeout_s=.1)
    assert failure['returncode'] == 9 and failure['cancelled'] is False
    assert timeout['timed_out'] is True and timeout['returncode'] != 0
    assert timeout['elapsed_s'] < 5


def bundle_context(tmp_path, monkeypatch, *, targeted=False, endpoint=False, released=False, first_state=None):
    c = controller()
    tool = tmp_path / 'tool'; py = tool / '.venv/bin/python'; py.parent.mkdir(parents=True); py.symlink_to(sys.executable)
    source = tmp_path / 'original'; source.mkdir()
    out = tmp_path / 'bundle'; out.mkdir()
    args = argparse.Namespace(tool=tool, source_run=source, skip_install=True, run_targeted=targeted,
                              artifact_root=[], h10_ssh='nx@example.test', h10_python='/venv/python', h10_images_json=None)
    calls = []
    def run_stage(command, cwd, log, environment, **kwargs):
        calls.append(command); log.write_text('mocked hardware/installer boundary')
        if any('hailo10_yolo26_boundary_probe_v282.py' in value for value in command):
            destination = out / 'h10/capture'; destination.mkdir(parents=True)
            (destination / 'collection_summary.json').write_text(json.dumps({
                'collection_complete': True, 'capture_pass': True, 'endpoint_pass': endpoint,
                'regular_path_released': released,
                'cases': {name: {'capture_pass': True, 'endpoint_pass': endpoint} for name in ('yolo26m', 'yolo26s')}}))
        if first_state is not None and len(calls) == 1:
            return dict(first_state)
        return {'returncode': 0, 'cancelled': False, 'timed_out': False}
    monkeypatch.setattr(c, 'run_stage', run_stage)
    monkeypatch.setattr(c, 'select_energy_checkpoints', lambda source: [])
    return c, args, out, calls


@pytest.mark.parametrize('endpoint,released', [(False, False), (True, False)])
def test_successful_h10_capture_does_not_claim_endpoint_release(tmp_path, monkeypatch, endpoint, released):
    c, args, out, calls = bundle_context(tmp_path, monkeypatch, targeted=True, endpoint=endpoint, released=released)
    assert c.execute(args, out) == 2
    summary = json.loads((out / 'acceptance.json').read_text())
    assert summary['status'] == 'PARTIAL_H10_ENDPOINTS_UNRELEASED'
    assert summary['h10']['capture_pass'] is True and summary['h10']['regular_path_released'] is False
    assert summary['h10']['scope'] == 'one_bound_source_image'
    assert 'status_quality_replay' in summary['stages']
    assert summary['scientific_final_claim'] is False
    assert out.with_suffix('.zip').is_file()


def test_default_bundle_only_replays_and_prepares(tmp_path, monkeypatch):
    c, args, out, calls = bundle_context(tmp_path, monkeypatch)
    assert c.execute(args, out) == 0
    assert not any('--execute' in call for call in calls)
    assert not any('hailo10_yolo26_boundary_probe_v282.py' in value for call in calls for value in call)
    summary = json.loads((out / 'acceptance.json').read_text())
    assert summary['status'] == 'SOFTWARE_AND_REPLAY_PASS_HARDWARE_NOT_RUN'
    assert 'generic_replay' in summary['stages'] and 'status_quality_replay' in summary['stages']


@pytest.mark.parametrize('state,expected_rc,expected_status', [
    ({'returncode': 130, 'cancelled': True, 'timed_out': False}, 130, 'CANCELLED'),
    ({'returncode': 124, 'cancelled': True, 'timed_out': True}, 2, 'PARTIAL'),
])
def test_bundle_distinguishes_user_cancel_from_timeout_and_exports(tmp_path, monkeypatch, state, expected_rc, expected_status):
    c, args, out, _ = bundle_context(tmp_path, monkeypatch, first_state=state)
    assert c.execute(args, out) == expected_rc
    summary = json.loads((out / 'acceptance.json').read_text())
    assert summary['status'] == expected_status and summary['exit_code'] == expected_rc
    assert out.with_suffix('.zip').is_file()


def test_bundle_export_failure_keeps_json_and_removes_partial_archive(tmp_path, monkeypatch):
    c, args, out, _ = bundle_context(tmp_path, monkeypatch)
    def fail_link(*args, **kwargs):
        raise OSError('mocked export failure')
    monkeypatch.setattr(c.os, 'link', fail_link)
    assert c.execute(args, out) == 2
    summary = json.loads((out / 'acceptance.json').read_text())
    assert summary['status'] == 'EVIDENCE_EXPORT_FAILED'
    assert not out.with_suffix('.zip.part').exists()
    assert (out / 'generic_replay.log').exists()
