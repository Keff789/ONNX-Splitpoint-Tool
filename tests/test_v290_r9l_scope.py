"""Actual A7 checker replay and explicit scope negatives; no device dispatch."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

OUTPUT = Path('/home/kmika/.local/share/onnx-splitpoint-codex/v290_R9L_Abschluss_20260920_220613_nrl9ktbx')
PARENT = Path('/home/kmika/.local/share/onnx-splitpoint-codex/v283_R9K_Abschluss_20260920_151548_6412w66g')
A7 = Path('/home/kmika/Models/EvaluationRuns/v283_R9K_Abschluss_20260920_151548_6412w66g/a_07_28bte7e9/r9j_a_20260920_204901')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(name, payload):
    if folder := os.environ.get('R9L_REPLAY_DIR'):
        dest = Path(folder); dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def test_actual_parent_postcheck_reproduces_residual_budget_failure(monkeypatch):
    assert (OUTPUT / 'PARENT_postcheck.py').read_bytes() == (PARENT / 'operator/postcheck.py').read_bytes()
    # Execute the exact original file at its original location: its policy path
    # is relative to __file__, so the copied file alone has a different context.
    monkeypatch.syspath_prepend(str(PARENT / 'operator'))
    monkeypatch.setitem(sys.modules, 'scope_contract', load(PARENT / 'operator/scope_contract.py', 'old_scope'))
    monkeypatch.setitem(sys.modules, 'strict_postcheck', load(PARENT / 'operator/strict_postcheck.py', 'old_strict'))
    checker = load(PARENT / 'operator/postcheck.py', 'old_postcheck')
    monkeypatch.setenv('R9J_CELL', 'A')
    results = []
    for remaining in (None, '2', '1'):
        if remaining is None:
            monkeypatch.delenv('R9J_REMAINING_SOURCE_FAILURES', raising=False)
        else:
            monkeypatch.setenv('R9J_REMAINING_SOURCE_FAILURES', remaining)
        result = checker.check_run(A7)
        assert result['pass'] is (remaining == '1')
        assert result['defect_ids'] == ([] if remaining == '1' else ['requested_scope_mismatch'])
        results.append(dict(environment_remaining=remaining, expected_budget=2 if remaining is None else int(remaining), observed_budget=1, result=result))
    evidence('parent_scope.json', dict(source=str(PARENT / 'operator/postcheck.py'), copied_source=str(OUTPUT / 'PARENT_postcheck.py'), run=str(A7), replays=results, historical_acceptance_changed=False))


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.setitem(sys.modules, 'scope_contract', load(OUTPUT / 'operator/scope_contract.py', 'current_scope'))
    return load(OUTPUT / 'operator/report_scope.py', 'current_report_scope')


def inputs():
    return (yaml.safe_load((A7 / 'profile.yaml').read_text()),
            json.loads((A7 / 'effective_execution_plan.json').read_text()),
            json.loads((A7 / 'required_run_scope.json').read_text()))


def test_report_scope_accepts_historical_a7_only_with_bound_budget_and_version(scope):
    positive = scope.audit(A7, 'A', 1, expected_version='2.83')
    wrong_budget = scope.audit(A7, 'A', 2, expected_version='2.83')
    wrong_release = scope.audit(A7, 'A', 1)
    assert positive['pass'] is True, positive
    assert wrong_budget['pass'] is False
    assert 'run_release_version' in wrong_release['defect_ids']
    p, plan, required = inputs()
    p['model_suite']['primary'].reverse(); p['hardware']['selected_setups'].reverse(); p['run_profiles'].reverse()
    required['models'].reverse(); required['hardware_targets'].reverse(); required['logical_run_profiles'].reverse()
    assert scope.audit_profile(p, plan, required, 'A', 1) == []
    evidence('report_scope.json', dict(source=str(OUTPUT / 'operator/report_scope.py'), positive=positive, wrong_budget=wrong_budget, wrong_release=wrong_release, reordered_pass=True))


NEGATIVES = [
 ('model_suite.primary', 'duplicate'), ('model_suite.primary', 'missing'),
 ('hardware.selected_setups', 'duplicate'), ('hardware.selected_setups', ['foreign']),
 ('run_profiles', 'duplicate'), ('run_profiles', 'missing'),
 ('native_producers.frames', 101), ('native_producers.warmup', 9), ('native_producers.repetitions', 2),
 ('native_producers.enabled', False), ('native_producers.full_baselines.enabled', False),
 ('native_producers.split_backends', ['hailo8']), ('native_producers.build_missing_engines', False),
 ('selection_policy.max_accepted_cases_per_model', 2), ('execution_preset.id', 'final'),
 ('validation_execution.max_items.detection', 5000), ('validation_execution.max_items.classification', 500),
 ('quality_gate.statistics.bootstrap_repetitions', 5000),
 ('quality_gate.reporting_policy.relative_loss_threshold', .06),
 ('quality_gate.reporting_policy.confidence_level', .9),
 ('quality_gate.reporting_policy.uncertainty_band_low', .02),
 ('quality_gate.reporting_policy.uncertainty_band_high', .08),
 ('quality_gate.reporting_policy.technical_fail_only', False),
 ('native_producers.energy.enabled', False), ('native_producers.energy.physical_scope', 'accelerator'),
 ('native_producers.energy.duration_s', 2), ('energy.repeat_override', 1),
 ('native_producers.energy.window_label', 'full'), ('energy.generic_enabled', True),
 ('native_producers.energy.window_method_validation_probe.enabled', True),
 ('native_producers.energy.task_budget.max_transport_failures', 2),
 ('native_producers.energy.task_budget.max_transport_failures', 0),
 ('native_producers.energy.task_budget.max_retries', 2),
 ('native_producers.energy.task_budget.enabled', False),
 ('hailo_build.force_build', True), ('deepx_build.force_build', True), ('force', True),
 ('hailo_build.mode', 'disabled'), ('deepx_build.mode', 'disabled'),
 ('selection_policy.backend_backfill.max_cold_builds', 13),
 ('selection_policy.backend_backfill.max_hailo_part1_builds', 7),
 ('selection_policy.backend_backfill.max_trt_part2_builds', 9),
]


@pytest.mark.parametrize('path,value', NEGATIVES)
def test_report_scope_keeps_all_guard_dimensions_negative(scope, path, value):
    profile, plan, required = inputs()
    obj = profile
    parts = path.split('.')
    for part in parts[:-1]: obj = obj.setdefault(part, {})
    key = parts[-1]
    active = [i for i, row in enumerate(obj.get(key, [])) if not isinstance(row, dict) or row.get('enabled', True)] if isinstance(obj.get(key), list) else []
    if value == 'duplicate': obj[key].append(deepcopy(obj[key][active[0]]))
    elif value == 'missing': obj[key].pop(active[0])
    else: obj[key] = value
    assert scope.audit_profile(profile, plan, required, 'A', 1), path


@pytest.mark.parametrize('key', ['models', 'hardware_targets', 'logical_run_profiles'])
@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'foreign'])
def test_required_scope_rejects_duplicates_missing_and_foreign(scope, key, damage):
    profile, plan, required = inputs()
    values = required[key]
    if damage == 'duplicate': values.append(deepcopy(values[0]))
    elif damage == 'missing': values.pop()
    else: values[0] = 'foreign' if key == 'logical_run_profiles' else {('model_id' if key == 'models' else 'id'): 'foreign'}
    assert 'required_scope:' + key in scope.audit_profile(profile, plan, required, 'A', 1)


@pytest.mark.parametrize('key,value', [('native_frames', 101), ('native_warmup', 0), ('native_performance_repetitions', 2), ('cases_per_model', 2), ('bootstrap_repetitions', 5000), ('native_energy_enabled', False), ('native_full_baselines', False)])
def test_effective_plan_cannot_disagree_with_profile(scope, key, value):
    profile, plan, required = inputs(); plan[key] = value
    assert 'effective_plan:' + key in scope.audit_profile(profile, plan, required, 'A', 1)
