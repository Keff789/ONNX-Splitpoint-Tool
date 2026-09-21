"""Bounded source-closure regressions; local fake processes, no devices."""
import json

import pytest

from onnx_splitpoint_tool.energy.collector import _command_window_binding
from onnx_splitpoint_tool.energy.task_budget import EnergyTaskBudget
from test_energy_command_window_binding_v2 import _binding_fixture
from test_v283_campaign_energy_budget import campaign_rig, measure


def test_original_source_failure_survives_next_row_and_reentry(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['first_sample'])
    failed = measure(rig, 'failed', row='failed')
    reason = 'campaign_source_completion_unresolved'
    assert failed['error'] == reason
    before = json.loads((tmp_path / 'campaign.json').read_text())
    for name in ('next-row', 'reentry'):
        blocked = measure(rig, name, row='next-row')
        assert blocked['execution_status'] == 'NOT_RUN'
        assert blocked['error'] == reason
        assert blocked['task_budget']['chains'] == []
    after = json.loads((tmp_path / 'campaign.json').read_text())
    assert after['tasks']['failed'] == before['tasks']['failed']
    assert after['sources'] == before['sources']
    assert rig.counts() == {'preflight': 1, 'collector': 1, 'workload': 0}


@pytest.mark.parametrize('reason', [
    'campaign_source_completion_unresolved',
    'campaign_source_transport_failure_limit',
])
def test_reentry_keeps_first_stop_and_other_source_counters(tmp_path, reason):
    limits = dict(max_chains=6, max_retries=1, max_transport_failures=2)
    original = {'campaign': True, 'max_transport_failures': 2,
                'sources': {'affected': {'stop_reason': reason, 'transport_failures': 1},
                            'other': {'stop_reason': '', 'transport_failures': 1}},
                'tasks': {'old': {'source_id': 'affected', 'limits': limits,
                                 'stop_reason': reason, 'chains': [{
                                     'finished': True, 'collector_started': True,
                                     'source_completion_verified': False,
                                     'source_completion': {'verified': False, 'protocol_end': []},
                                     'logical_repeat': 'repeat:0'}]}}}
    path = tmp_path / 'budget.json'
    path.write_text(json.dumps(original))
    with path.open('r+') as handle:
        budget = EnergyTaskBudget(handle, limits, campaign_row='next', source_id='affected')
        assert not budget.allowed('repeat:0')
        assert budget.data['stop_reason'] == reason
    after = json.loads(path.read_text())
    assert after['sources'] == original['sources']
    assert after['tasks']['old'] == original['tasks']['old']
    assert after['tasks']['next']['chains'] == []


@pytest.mark.parametrize('uncertainty,verified', [(63, True), (64, True), (65, False), (256, False)])
def test_existing_marker_boundary_limit_stays_exact(tmp_path, uncertainty, verified):
    result, timing, path = _binding_fixture(tmp_path, uncertainty_samples=uncertainty,
                                          maximum_uncertainty_samples=64)
    binding = _command_window_binding(tmp_path, result, timing, result_path=path)
    assert binding['verified'] is verified
    if not verified:
        assert binding['status'] == 'boundary_uncertainty_exceeds_limit'
