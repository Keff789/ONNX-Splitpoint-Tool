"""Interrupted Native energy publications retain ordinary consumer decisions."""
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_energy_reporting import (
    collect_native_energy, native_energy_results_path, summarize_native_energy_reporting,
)
from onnx_splitpoint_tool.yolo_augmentation_reporting import _energy, write_augmentation_reports
from tests.test_yolo_augmentation_reporting import scope, selected


def result(case, valid=0, *, complete=False, attempts=2, failed=1):
    aggregate = {"run_count": 3, "valid_postprocessed_runs": valid,
        "raw_postprocessed_run_count": max(valid, 2), "repeat_contract_complete": complete,
        "scientific_primary_method_frozen": True, "scientific_primary_method": "command_marker_window",
        "scientific_primary_energy_status": "available" if valid else "unavailable"}
    return {"row": {"model": "yolo11l", "case": case, "setup_id": "orin_nx_deepx_m1_01",
        "backend": "deepx_to_trt", "measurement_effective_repeats": 3,
        "claim_eligible": False, "screening_only": True}, "ok": complete,
        "run": {"rc": 0 if complete else 1, "energy_aggregate": aggregate,
            "energy_aggregate_embedded": True, "energy_aggregate_complete": complete,
            "energy_aggregate_verified": complete, "energy_collector_attempt_count": attempts,
            "energy_failed_collector_attempt_count": failed,
            "measurement_start_observation": {"collector_started_repeat_count": max(valid, 2)},
            "execution_status": "COMPLETED" if complete else "INCOMPLETE",
            "energy_not_started_reason": "" if complete else "task_execution_interrupted"}}


def publish(root, rows, *, partial=True):
    directory = root / "reports/native_energy_measurements"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("native_producer_energy_results.partial.json" if partial else "native_producer_energy_results.json")
    path.write_text(json.dumps({"complete": not partial, "ok": not partial, "rows": rows}))
    return path


def test_partial_is_explicit_and_keeps_complete_incomplete_and_negative_observations(tmp_path):
    root = tmp_path / "block"
    path = publish(root, [result("b030",3,complete=True,attempts=3,failed=0), result("b031",1), result("b032")])
    before = path.read_bytes()
    assert collect_native_energy(root) == []
    rows, raw, source, _ = _energy([root])
    assert [row["energy_repeat_valid_n"] for row in rows] == [3,1,0]
    assert [row["ok"] for row in rows] == [True,False,False]
    assert rows[1]["energy_repeat_status"] == "incomplete_valid_repeats"
    assert all(row["source_result_kind"] == "partial" and row["source_result_complete"] is False for row in rows)
    assert not any(row["claim_eligible"] for row in rows)
    summary = summarize_native_energy_reporting(rows, [])
    assert summary["energy_complete_row_count"] == 1
    assert summary["energy_incomplete_row_count"] == 2
    assert summary["energy_valid_replicate_count"] == 4
    assert summary["energy_physical_collector_attempt_count"] == 7
    assert summary["energy_failed_physical_collector_attempt_count"] == 2
    assert json.loads(source) == [str(path.resolve())]
    assert raw["blocks"][0]["results"] == json.loads(before)
    assert path.read_bytes() == before


@pytest.mark.parametrize("input_kind", ["root", "file", "roots"])
def test_normal_report_projects_partial_without_raising_claim_gates(scope, tmp_path, input_kind):
    plan, out, _ = scope
    root = tmp_path / "block"
    path = publish(root, [result("b030",3,complete=True,attempts=3,failed=0)])
    value = {"root":root,"file":path,"roots":[root]}[input_kind]
    summary = write_augmentation_reports(plan, new_energy_results=value, output_root=out)
    assert summary["energy"]["energy_valid_row_count"] == 1
    assert summary["energy"]["energy_valid_replicate_count"] == 3
    assert summary["energy"]["energy_scientifically_released_row_count"] == 0
    assert json.loads((out / "augmentation_energy.json").read_text())["new_observations"][0]["source"] == str(path)


def test_terminal_file_takes_priority_and_malformed_terminal_never_falls_back(tmp_path):
    root = tmp_path / "block"
    partial = publish(root,[result("b030",3,complete=True)])
    final = publish(root,[result("b031")],partial=False)
    assert native_energy_results_path(root,include_partial=True) == final
    assert _energy([root])[0][0]["case"] == "b031"
    with pytest.raises(ValueError,match="differs from canonical"):
        _energy(partial)
    final.write_text("not JSON")
    with pytest.raises(ValueError):
        _energy([root])
    assert collect_native_energy(root,include_partial=True) == []


def test_interruption_without_aggregate_preserves_requested_and_started_counts(tmp_path):
    row = result("b030")
    row["run"].pop("energy_aggregate")
    row["run"].pop("energy_aggregate_embedded")
    publish(tmp_path,[row])
    observed = collect_native_energy(tmp_path,include_partial=True)[0]
    assert observed["energy_repeat_requested_n"] == 3
    assert observed["energy_repeat_observed_n"] == 2
    assert observed["energy_repeat_valid_n"] == 0
    assert observed["energy_repeat_status"] == "aggregate_unavailable"
    assert observed["energy_not_started_reason"] == "task_execution_interrupted"
    assert observed["ok"] is False and observed["claim_eligible"] is False


def test_missing_physical_attempt_provenance_is_unknown_not_zero(tmp_path):
    row = result("b030")
    row["run"].pop("energy_collector_attempt_count")
    publish(tmp_path,[row])
    summary = summarize_native_energy_reporting(collect_native_energy(tmp_path,include_partial=True),[])
    assert summary["energy_physical_collector_attempt_count"] is None
