"""Retain the actual measured first pair in the explicit external run COW."""
import copy
import json
from pathlib import Path

import pytest

from tests.test_yolo_measured_completion_reentry import measured, edit
from onnx_splitpoint_tool.workflow import yolo_augmentation_reentry as admission


def test_completed_first_pair_keeps_original_native_coordinator(measured):
    run, plan, proof = measured
    ancestor = proof["completed_pair_ancestor"]
    assert proof["predecessor_input_hash"] != ancestor["predecessor_input_hash"]
    assert proof["preserved_native_coordinator_input_hash"] == ancestor["predecessor_input_hash"]
    assert proof["preserved_native_leaves"] == ancestor["preserved_native_leaves"]
    assert len(proof["preserved_generic_result"]["repetitions"]) == 3
    assert [r["completed_task_fps"] for r in proof["preserved_generic_result"]["repetitions"]] == [
        35.537524407015844, 35.33776805174142, 35.17317429619841]


def test_saved_attempt_allows_next_frozen_case_without_rewriting_first_pair(measured, edit):
    run, plan_path, proof = measured
    plan = json.loads(plan_path.read_text())
    execution_path = plan_path.parent.parent / "completion/execution-plan.json"
    execution = json.loads(execution_path.read_text())
    starter = next(row for row in plan['rows'] if row['case_key'] == execution['rows'][0]['case_key'])
    next_case = next(row for row in plan['rows'] if row['selection_role'] == 'primary' and row['case_key'] != starter['case_key'])
    summary = json.loads((run / 'reports/native_producer_summary.json').read_text())
    native = json.loads(Path(summary['rows'][0]['report']).read_text())
    execution['rows'].append({'identity':next_case['identity']})
    edit(execution_path, execution)
    assert admission._completed_first_generic(run, plan_path, plan, starter, native,
        allow_additions=True) == proof['preserved_generic_result']
    with pytest.raises(ValueError, match='generic_scope_changed'):
        admission._completed_first_generic(run, plan_path, plan, starter, native)


@pytest.mark.parametrize("fault", ["partial", "duplicate_runtime", "sample", "native_sha",
    "setup", "precision", "reference", "extra_case"])
def test_completed_pair_reentry_rejects_generic_drift(measured, edit, fault):
    run, plan, proof = measured
    completion = plan.parent.parent / "completion"
    execution_path = completion / "execution-plan.json"
    execution = json.loads(execution_path.read_text())
    case = execution["rows"][0]
    result_path = Path(case["result_path"])
    result = json.loads(result_path.read_text())
    if fault == "partial":
        result["repetitions"] = result["repetitions"][:2]
        edit(result_path, result)
    elif fault == "duplicate_runtime":
        for row in result["repetitions"]:
            row["runtime_instance_id"] = result["repetitions"][0]["runtime_instance_id"]
        edit(result_path, result)
    elif fault == "sample":
        sample = next(iter(result["repetitions"][0]["raw_output_samples"].values()))
        edit(result_path.parent / sample["path"], b"changed original output")
    else:
        if fault == "native_sha": case["native_command_contract_sha256"] = "f" * 64
        elif fault in {"setup", "precision"}:
            case["identity"]["setup_id" if fault == "setup" else "precision"] = "foreign"
        elif fault == "reference":
            case["runtime_config"]["functional_reference"]["input_sha256"] = "f" * 64
        else: execution["rows"].append(copy.deepcopy(case))
        edit(execution_path, execution)
    with pytest.raises(ValueError): admission.validate_augmentation_reentry(run, plan)


@pytest.mark.parametrize("fault", ["recursive", "second_pair_reentry"])
def test_no_general_repeat_or_recursive_admission(measured, monkeypatch, fault):
    run, plan, proof = measured
    original = admission._object
    def changed(path):
        value = original(path)
        if Path(path) == run / "reports/native_producer_stage_config.json":
            prior = value["augmentation_startcase_barrier"]["reentry"]
            if fault == "recursive": prior["original_root"] = str(run)
            else: prior["completed_pair_ancestor"] = {"not": "another retry"}
        return value
    monkeypatch.setattr(admission, "_object", changed)
    with pytest.raises(ValueError, match="completed_pair_ancestor_invalid"):
        admission.validate_augmentation_reentry(run, plan)
