"""Recovery resource scope requires held normal authority for disjoint work.

Only bounded fixture children and temporary control files are observed. There
is no global setup inventory, remote execution or sampling of live user runs.
"""
from __future__ import annotations

import copy
import json

import pytest

from onnx_splitpoint_tool.remote import resource_recovery_observation as observation
from tests.test_generic_completed_task_resources import idle, workflow


SETUP = {"id": "selected-deepx", "host": {"address": "192.0.2.8", "user": "fixture", "port": 22}}
ORIGINAL = {"pid": 1000000000, "start_time_ticks": "1"}


def _with_ended_original(raw):
    value = copy.deepcopy(raw)
    value["recorded_owners"] = [{**ORIGINAL, "observed_start_time_ticks": None, "state": "absent"}]
    return value


def _scope(raw):
    return observation.scope_local_resource_owners(
        SETUP, local_owners=[ORIGINAL], raw_observation=raw)


def test_disjoint_live_normal_authority_is_allowed_without_discarding_raw_evidence(workflow, monkeypatch):
    from onnx_splitpoint_tool.workflow import generic_completed_task
    _, observed = workflow(host="192.0.2.10")
    raw = _with_ended_original(observed)
    before = copy.deepcopy(raw)
    normal_scope = generic_completed_task._scoped_controller_owners
    selected = []

    def selected_only(targets, observed):
        assert set(targets) == {SETUP["id"]}
        selected.append(copy.deepcopy(targets))
        return normal_scope(targets, observed)

    monkeypatch.setattr(generic_completed_task, "_scoped_controller_owners", selected_only)
    result = _scope(raw)
    assert result["ok"], result
    assert len(selected) == 1
    assert result["scope_errors"] == []
    assert result["raw_observation"] == raw == before
    assert result["recorded_owners"] == raw["recorded_owners"]
    assert result["scoped_observation"]["normal_quality_owners"][0]["resources"] == ["dut:192.0.2.10"]
    assert result["cleanup_proven"] is result["exclusive_start_permission"] is False


@pytest.mark.parametrize("central", [False, True])
def test_same_dut_authority_blocks_recovery_even_when_normal_statistics_are_central_only(workflow, central):
    _, observed = workflow(host="192.0.2.8", central=central)
    result = _scope(_with_ended_original(observed))
    assert result["ok"] is False
    assert result["scope_errors"]


@pytest.mark.parametrize("fault", ["unknown_authority", "visibility_error", "live_unattributed_lease"])
def test_unknown_or_globally_unobservable_work_cannot_be_scoped_away(workflow, fault):
    _, observed = workflow(host="192.0.2.10")
    raw = _with_ended_original(observed)
    if fault == "unknown_authority":
        raw["processes"].append({"pid": 999999999, "start_time_ticks": "1", "matched_programs": ["unknown"]})
    elif fault == "visibility_error":
        raw["visibility_errors"] = ["process_directory_unreadable"]
    else:
        raw["live_product_leases"] = [{"pid": 999999999, "start_time_ticks": "1", "path": "/unknown/lease"}]
    result = _scope(raw)
    assert result["ok"] is False
    assert result["scope_errors"]
    assert result["visibility_errors"] == raw["visibility_errors"]
    assert result["raw_observation"] == raw


@pytest.mark.parametrize("fault", ["active", "unknown", "pid_reused", "missing", "wrong_start", "wrong_zombie_start"])
def test_original_pid_identity_must_be_proven_ended_before_any_foreign_exemption(fault):
    raw = _with_ended_original(idle())
    if fault == "missing":
        raw["recorded_owners"] = []
    elif fault == "wrong_start":
        raw["recorded_owners"][0]["start_time_ticks"] = "2"
    elif fault == "wrong_zombie_start":
        raw["recorded_owners"][0].update(state="zombie", observed_start_time_ticks="2")
    else:
        raw["recorded_owners"][0]["state"] = fault
    result = _scope(raw)
    assert result["ok"] is False
    assert result["scope_errors"]


@pytest.mark.parametrize("resource_scoped", [False, True])
def test_public_observer_uses_disjoint_authority_only_for_explicit_resource_scope(workflow, tmp_path, monkeypatch, resource_scoped):
    _, observed = workflow(host="192.0.2.10")
    raw = _with_ended_original(observed)
    monkeypatch.setattr(observation, "_observe_processes", lambda *_args, **_kwargs: copy.deepcopy(raw))
    calls = []

    class Transport:
        def __init__(self, *_args, **_kwargs):
            calls.append(True)

        def run_read_only(self, *_args, **_kwargs):
            return 0, "RESOURCE_OWNERS=" + json.dumps(idle())

    monkeypatch.setattr(observation, "SSHTransport", Transport)
    result = observation.observe_resource_owners(SETUP, local_owners=[ORIGINAL],
        diagnostics_dir=tmp_path / "diagnostics", resource_scoped=resource_scoped)
    assert result["ok"] is resource_scoped
    assert bool(calls) is resource_scoped
    if not resource_scoped:
        assert result["remote"]["status"] == "not_observed"
