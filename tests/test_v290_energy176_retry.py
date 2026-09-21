"""Task-local 2-retry/3-source-failure policy using real fake subprocesses."""
import json
from types import SimpleNamespace
import pytest
from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.energy.task_budget import campaign_budget_forward_args, campaign_budget_policy
from test_v283_campaign_energy_budget import campaign_rig, measure


def run(rig, name='series', **kwargs):
    return measure(rig, name, campaign_max_retries=2, campaign_max_transport_failures=3,
                   invalid_repeat_max_retries=2, **kwargs)


@pytest.mark.parametrize('modes,starts,retries', [(['success'],3,0),
    (['drop','success','success','success'],4,1),
    (['drop','success','success','drop','success'],5,2)])
def test_first_valid_attempt_selected_and_negative_evidence_kept(tmp_path, monkeypatch, modes, starts, retries):
    rig = campaign_rig(tmp_path, monkeypatch, modes)
    result = run(rig)
    assert result['ok'], result.get('error')
    assert rig.counts()['collector'] == starts
    assert result['task_budget']['counts']['valid_logical_repeats'] == 3
    chains = result['task_budget']['chains']
    assert all(c['source_completion_verified'] for c in chains)
    assert sum(not c['valid'] for c in chains) == retries
    assert result['runs'][0]['selected_repeat_attempt_index'] == retries


def test_third_source_failure_stops_other_row_and_persists(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['drop'])
    first = run(rig)
    assert rig.counts()['collector'] == 3
    assert first['error'] == 'campaign_source_transport_failure_limit'
    second = run(rig, 'second', row='second')
    assert rig.counts()['collector'] == 3
    assert second['error'] == first['error']


def test_two_retries_maximum_same_logical_replicate(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['success', 'success', 'coverage'])
    result = run(rig)
    assert rig.counts()['collector'] == 5
    assert sum(c['logical_repeat'] == 'repeat:2' for c in result['task_budget']['chains']) == 3
    assert not result['ok']
    run(rig, 'reentry')
    assert rig.counts()['collector'] == 5


def test_gap_outside_window_alone_has_no_retry(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['success'])
    p = tmp_path/'fake_collector.py'
    p.write_text(p.read_text().replace("'dropped_samples': 0,", "'dropped_samples': 0, 'total_dropped_samples': 8576,"))
    result = run(rig)
    assert result['ok'] and rig.counts()['collector'] == 3
    markers = list((tmp_path/'series').glob('run_*/collector_storage/command_window_markers.json'))
    assert len(markers) == 3
    assert all(json.loads(p.read_text())['stream']['total_dropped_samples'] == 8576 for p in markers)


def test_missing_end_stops_immediately(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['first_sample'])
    result = run(rig)
    assert rig.counts()['collector'] == 1
    assert result['error'] == 'campaign_source_completion_unresolved'
    again = run(rig, 'other', row='other')
    assert again['error'] == result['error'] and rig.counts()['collector'] == 1


def test_failed_preflight_does_not_start_collector(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['success'])
    rig.env['R3_PREFLIGHT_FAIL'] = '1'
    result = run(rig)
    assert rig.counts()['collector'] == 0
    assert not result['ok']


def test_workload_error_is_not_a_transient_marker_retry():
    entry = {'status': 'collector_finished', 'workload_timing': {'rc': 1},
             'command_window_request': {'status': 'marker_contract_invalid'}}
    assert collector._energy_repeat_retry_reasons(entry) == []


def test_explicit_cli_policy_does_not_relax_profile_defaults():
    args = SimpleNamespace(campaign_budget_file='/tmp/new-explicit-task.json', campaign_max_retries=2, campaign_max_transport_failures=3)
    assert campaign_budget_forward_args(args)[-1] == '3'
    with pytest.raises(ValueError):
        campaign_budget_policy({'energy': {'task_budget': {'enabled': True, 'max_retries': 2, 'max_transport_failures': 3}}})
    from onnx_splitpoint_tool.energy.config import EnergyDefaults
    assert EnergyDefaults().invalid_repeat_reconnect_backoff_s == 5
