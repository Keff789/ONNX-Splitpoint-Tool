from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.yolo_augmentation_reporting import COHORT, write_augmentation_reports
from tests.test_yolo_augmentation_plan import exact_terminal_case, cold_terminal_case


SETUP = "orin_nx_deepx_m1_01"


def pair(case, generic=100.0, native=120.0, *, accuracy="reference_close", endpoint="detection:endpoint"):
    return {"model_id": "yolo11l", "case_id": case, "setup_id": SETUP,
        "direction": "deepx_to_trt", "precision": "float32_layout_fp16",
        "comparison_backend": "deepx", "output_endpoint_id": endpoint,
        "boundary_class": "prepared_input_before_p1_to_last_completed_task",
        "generic_completed_task_fps": generic, "native_completed_task_fps": native,
        "generic_completion_status": "completed", "valid_repetition_count": 3,
        "completed_task_count": 3000, "generic_task_quality_status": accuracy,
        "native_task_quality_status": accuracy, "eligible_for_technical_transfer": True,
        "eligible_for_quality_transfer": True, "eligible_for_claim_transfer": False,
        "evaluation_role": "development", "evaluation_role_source": "original-profile.json",
        "generic_raw_source": "old-generic.json", "native_source": "own-native.json",
        "runtime_conditions_comparable": False, "original_run": "original-run-id"}


def selected(case, role="primary", **extra):
    return {key: value for key, value in {**pair(case), "selection_role": role, **extra}.items()
            if key in ("model_id", "case_id", "setup_id", "direction", "precision",
                       "comparison_backend", "selection_role", "status", "execution_status",
                       "execution_error", "activated", "output_endpoint_id")}


@pytest.fixture
def scope(tmp_path):
    base = tmp_path / "base"
    reports = base / "reports"
    reports.mkdir(parents=True)
    payload = {"pairs": [pair("b003"), pair("b062")], "groups": [{"original_statistic": -0.75}],
               "summary": {"planned_cases": 2, "claim_pairs": 0}}
    (reports / "generic_completion_cross_runner.json").write_text(json.dumps(payload))
    plan = {"cohort_id": COHORT, "original_run": str(tmp_path / "original"),
            "base_completion_root": str(base), "rows": [selected("b030")]}
    return plan, tmp_path / "report", payload


def read(out, name):
    return json.loads((out / name).read_text())


def test_missing_measurements_are_unstarted_and_base_remains_unchanged(scope):
    plan, out, base = scope
    original = copy.deepcopy(plan)
    path = Path(plan["base_completion_root"]) / "reports/generic_completion_cross_runner.json"
    before = path.read_bytes()
    result = write_augmentation_reports(plan, output_root=out)
    assert result["base_pair_count"] == 2
    assert result["new_case_status_counts"] == {"unstarted": 1}
    assert result["new_technical_pair_count"] == 0
    assert read(out, "augmented_union.json")["coverage"][0]["shortfall"] == 1
    assert read(out, "base_reference.json")["groups"] == base["groups"]
    assert path.read_bytes() == before and plan == original


def test_accuracy_loss_counts_technically_but_has_separate_positive_accuracy_view(scope):
    plan, out, _ = scope
    added = {"pairs": [pair("b030", accuracy="accuracy_loss")]}
    before = copy.deepcopy(added)
    summary = write_augmentation_reports(plan, added, output_root=out)
    group = read(out, "augmented_union.json")["groups"][0]
    assert group["technical_candidate_count"] == group["quality_candidate_count"] == 3
    assert group["reference_close_candidate_count"] == 2
    assert group["reference_close_status"] == "insufficient_candidates"
    assert group["claim_candidate_count"] == 0
    assert group["top3_at_n3_is_trivial"] is True
    assert group["top3_is_performance_claim"] is False
    assert group["runtime_conditions_comparable"] is False
    assert summary["yolo_groups_at_minimum_three"] == 1
    assert read(out, "augmented_union.json")["coverage"][0]["new_accuracy_classes"] == {"accuracy_loss": 1}
    assert added == before


def test_top1_uses_existing_cycle_regret_and_stable_ties(scope):
    plan, out, _ = scope
    basefile = Path(plan["base_completion_root"]) / "reports/generic_completion_cross_runner.json"
    basefile.write_text(json.dumps({"pairs": [pair("b003", 100, 300), pair("b062", 200, 100)]}))
    write_augmentation_reports(plan, {"pairs": [pair("b030", 150, 200)]}, output_root=out)
    group = read(out, "augmented_union.json")["groups"][0]
    assert group["technical_native_regret_at_1"] == pytest.approx(2.0)
    assert group["technical_native_best_hit_at_1"] is False
    assert group["technical_spearman_rho"] == pytest.approx(-1.0)
    assert group["reference_close_native_regret_at_1"] == pytest.approx(2.0)
    # Equal Generic rates retain existing input order; no measured Native rate
    # is used as a favourable tie breaker.
    basefile.write_text(json.dumps({"pairs": [pair("b003", 200, 100), pair("b062", 200, 300)]}))
    write_augmentation_reports(plan, {"pairs": [pair("b030", 150, 200)]}, output_root=out)
    assert read(out, "augmented_union.json")["groups"][0]["technical_native_regret_at_1"] == pytest.approx(2.0)


def test_failed_primary_remains_when_reserve_completes(scope):
    plan, out, _ = scope
    plan["rows"] = [selected("b030", execution_status="build_failed", execution_error="compiler infeasible"),
                    selected("b031", "reserve", activated=True), selected("b032", "reserve")]
    write_augmentation_reports(plan, {"pairs": [pair("b031")]}, output_root=out)
    rows = read(out, "augmentation_cases.json")["rows"]
    assert [r["accounting_status"] for r in rows] == ["build_failed", "completed", "standby"]
    assert rows[0]["execution_error"] == "compiler infeasible"
    assert rows[0]["technical_pair_measured"] is False


def test_unstarted_projection_does_not_hide_terminal_plan_failure(scope):
    plan, out, _ = scope
    plan["rows"][0]["execution_status"] = "failed"
    row = pair("b030")
    row.update(generic_completion_status="unstarted", eligible_for_technical_transfer=False,
               eligible_for_quality_transfer=False, generic_completed_task_fps=None, native_completed_task_fps=None)
    write_augmentation_reports(plan, {"pairs": [row]}, output_root=out)
    assert read(out, "augmentation_cases.json")["rows"][0]["accounting_status"] == "failed"


def test_different_endpoint_is_not_pooled_to_reach_three(scope):
    plan, out, _ = scope
    plan["rows"][0].pop("output_endpoint_id")
    write_augmentation_reports(plan, {"pairs": [pair("b030", endpoint="different-endpoint")]}, output_root=out)
    data = read(out, "augmented_union.json")
    assert len(data["groups"]) == 2
    assert data["coverage"][0]["union_technical_natural_boundary_count"] == 3
    assert data["coverage"][0]["largest_exact_comparable_stratum_count"] == 2
    assert data["coverage"][0]["shortfall"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "outside", "precision", "endpoint", "base", "nan"])
def test_conflicting_or_unbound_observations_stop_reporting(scope, mutation):
    plan, out, _ = scope
    added = {"pairs": [pair("b030")]}
    if mutation == "duplicate":
        added["pairs"].append(pair("b030"))
    elif mutation == "outside":
        added["pairs"][0]["case_id"] = "b099"
    elif mutation == "precision":
        added["pairs"][0]["precision"] = "other"
    elif mutation == "endpoint":
        added["pairs"][0]["output_endpoint_id"] = "other"
    elif mutation == "base":
        plan["rows"][0]["case_id"] = "b003"
    elif mutation == "nan":
        added["pairs"][0]["generic_completed_task_fps"] = float("nan")
    with pytest.raises(ValueError):
        write_augmentation_reports(plan, added, output_root=out)
    assert not out.exists()


def test_original_and_completion_run_are_read_only_destinations(scope):
    plan, _, _ = scope
    for key in ("original_run", "base_completion_root"):
        with pytest.raises(ValueError, match="outside original"):
            write_augmentation_reports(plan, output_root=Path(plan[key]) / "new-report")


def test_completion_repetition_statistics_and_own_run_identity_are_preserved(scope):
    plan, out, _ = scope
    obs = {"model_id": "yolo11l", "case_id": "b030", "setup_id": SETUP,
        "completed_task_fps_stddev": 0.6, "valid_repetition_count": 3,
        "repetitions": [{"started_at": "2026-10-03T01:00:00Z", "completed_at": "2026-10-03T01:01:00Z",
                         "runtime_instance_id": "fresh-1", "completed_task_count": 1000}]}
    new = pair("b030")
    new.update(original_run="new-run-id", native_source="new-run/native.json")
    write_augmentation_reports(plan, {"pairs": [new], "observations": [obs]}, output_root=out)
    actual = read(out, "augmentation_completion.json")["pairs"][0]
    assert actual["original_run"] == "new-run-id"
    assert actual["native_source"] == "new-run/native.json"
    assert actual["generic_repetition_statistics"]["completed_task_fps_stddev"] == 0.6
    assert actual["generic_repetition_statistics"]["repetitions"] == obs["repetitions"]
    assert actual["measurement_date_status"] == "recorded"


def energy(full=False):
    row = {"model": "yolo11l", "case": "full" if full else "b030", "setup_id": SETUP,
        "backend": "native_full_deepx" if full else "deepx_to_trt", "direction": "deepx_to_trt",
        "execution_mode": "native_full_baseline" if full else "native_split", "task": "detection",
        "task_source": "declared", "precision": "fp16", "prepared_feed_task": "detection",
        "prepared_feed_preprocess_mode": "letterbox", "prepared_feed_letterbox_pad_value": 0,
        "energy_scope": "full_system", "energy_window_effective": "command_window",
        "target_duration_s": 60, "active_duration_s": 60, "energy_per_work_j": 2.0 if full else 1.0,
        "average_power_w": 10.0, "energy_primary_metric": "raw_input_energy", "energy_raw_primary": True,
        "energy_calibrated_input_unsubtracted": True, "ok": True, "runtime_work_units_exact": True,
        "energy_calibration_verified": True, "endpoint_contract_complete": True, "output_endpoint_match": True,
        "completion_pairing_eligible": True, "semantic_claim_ok": True, "screening_only": True,
        "diagnostic_only": True, "screening_comparable": False, "claim_eligible": False,
        "energy_repeat_valid_n": 3, "evaluation_role": "unknown", "evaluation_role_raw": "",
        "evaluation_role_source": "old-source" if full else "new-source",
        "source": "old-run/results.json" if full else "new-run/results.json"}
    for field in ("model_sha256", "validation_dataset_sha256", "validation_dataset_image_ids_sha256",
                  "validation_dataset_ground_truth_sha256", "validation_input_or_image_sha256",
                  "prepared_feed_source_image_sha256", "preprocessing_contract_sha256",
                  "decoder_contract_sha256", "nms_contract_sha256", "energy_calibration_sha256"):
        row[field] = "a" * 64
    row.update(comparison_output_endpoint_id="detection:decoded_nms:comparison:" + "a" * 64,
               comparison_endpoint_contract_hash="a" * 64, comparison_endpoint_stage="decoded_nms",
               output_endpoint_id="detection:decoded_nms:comparison:" + "a" * 64, endpoint_contract_hash="a" * 64)
    return row


def test_new_energy_uses_old_full_only_and_existing_scope_semantic_claim_decisions(scope):
    plan, out, _ = scope
    full, split = energy(True), energy()
    old_split = energy(); old_split["case"] = "b003"
    plan["base_energy_observations"] = [full, old_split]
    before = copy.deepcopy([full, split])
    write_augmentation_reports(plan, new_energy_results=[split], output_root=out)
    data = read(out, "augmentation_energy.json")
    assert data["new_observations"] == [split]
    assert data["baseline_full_reference_count"] == 1
    assert data["baseline_full_remeasured_count"] == 0
    pairrow = data["pairs_against_original_full"][0]
    assert pairrow["descriptive_energy_ratio"] == 0.5
    assert pairrow["semantic_comparable"] is True
    assert pairrow["claim_comparable"] is False
    assert data["summary"]["energy_recorded_row_count"] == 1
    assert data["summary"]["energy_valid_replicate_count"] == 3
    assert [full, split] == before
    full["decoder_contract_sha256"] = "b" * 64
    write_augmentation_reports(plan, new_energy_results=[split], output_root=out)
    changed = read(out, "augmentation_energy.json")["pairs_against_original_full"][0]
    assert changed["descriptive_energy_ratio"] == 0.5
    assert changed["semantic_comparable"] is False
    full["energy_scope"] = "module"
    write_augmentation_reports(plan, new_energy_results=[split], output_root=out)
    assert read(out, "augmentation_energy.json")["pairs_against_original_full"][0]["descriptive_energy_ratio"] is None


def test_raw_energy_without_canonical_validation_stays_raw(scope):
    plan, out, _ = scope
    raw = {"rows": [{"row": {"model": "yolo11l", "case": "b030", "setup_id": SETUP},
                     "ok": True, "run": {"rc": 0}}]}
    write_augmentation_reports(plan, new_energy_results=raw, output_root=out)
    data = read(out, "augmentation_energy.json")
    assert data["projection_status"] == "raw_results_retained_canonical_run_required_for_projection"
    assert data["new_observations"] == []
    assert read(out, "augmentation_energy_raw_results.json") == raw


def test_canonical_energy_path_uses_normal_collector_without_false_completion(scope, tmp_path):
    plan, out, _ = scope
    root = tmp_path / "new-run"
    path = root / "reports/native_energy_measurements/native_producer_energy_results.json"
    path.parent.mkdir(parents=True)
    raw = {"rows": [{"row": {"model": "yolo11l", "case": "b030", "setup_id": SETUP,
                             "backend": "deepx_to_trt"}, "ok": False}]}
    path.write_text(json.dumps(raw))
    write_augmentation_reports(plan, new_energy_results=path, output_root=out)
    data = read(out, "augmentation_energy.json")
    assert data["projection_status"] == "normal_native_energy_collector"
    assert len(data["new_observations"]) == 1
    assert data["summary"]["energy_valid_row_count"] == 0
    assert data["pairs_against_original_full"] == []


def test_repeated_reporting_does_not_double_import(scope):
    plan, out, _ = scope
    args = (plan, {"pairs": [pair("b030")]})
    first = write_augmentation_reports(*args, output_root=out)
    content = (out / "augmented_union.json").read_bytes()
    second = write_augmentation_reports(*args, output_root=out)
    assert first == second
    assert (out / "augmented_union.json").read_bytes() == content
    assert second["union_pair_row_count"] == 3


def test_new_full_or_out_of_scope_native_energy_rows_are_not_silently_imported(scope):
    plan, out, _ = scope
    plan["augmentation_run"] = str(out.parent / "new-run")
    with pytest.raises(ValueError, match="native row outside"):
        write_augmentation_reports(plan, new_native_summary={"run_id": "new-run", "rows": [energy(True)]}, output_root=out)
    with pytest.raises(ValueError, match="energy observation outside"):
        write_augmentation_reports(plan, new_energy_results=[energy(True)], output_root=out)


def _normal_terminal_summary(cohort, run, monkeypatch):
    """Execute the existing local Finalreport over its normal terminal input."""
    from onnx_splitpoint_tool.yolo_augmentation_plan import collect_technical_terminals
    from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256
    terminals = collect_technical_terminals(cohort, run)
    assert len(terminals) == 1
    reports = run / "reports"
    reports.mkdir(exist_ok=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest", "run_id": run.name,
        "workflow_version": "2.91.2"}))
    (reports / "native_producer_stage.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/native-producer-variant-stage", "run_id": run.name,
        "workflow_version": "2.91.2", "eval_run_dir": str(run),
        "augmentation_technical_terminals": terminals}))
    frozen_plan = {**cohort, "cohort_id": COHORT, "frozen": True}
    plan_path = run.parent / "normal-writer-frozen-plan.json"
    plan_path.write_text(json.dumps(frozen_plan))
    (reports / "native_producer_stage_config.json").write_text(json.dumps({
        "augmentation_startcase_barrier": {"plan_path": str(plan_path),
            "plan_sha256": canonical_json_sha256(frozen_plan), "technical_terminals": terminals}}))
    script = Path(__file__).resolve().parents[1] / "scripts/native_producer_final_report.py"
    spec = importlib.util.spec_from_file_location("augmentation_terminal_report_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), "--root", str(run), "--recursive",
        "--out-dir", str(reports), "--eval-root", str(run)])
    assert module.main() == 0
    path = reports / "native_producer_combined_summary.json"
    summary = json.loads(path.read_text())
    authority = summary["native_split_quality_authority"]
    assert authority["run_id"] == authority["stage_run_id"] == run.name
    assert summary["rows"][0]["comparison_backend"] == terminals[0]["comparison_backend"]
    return path, summary


@pytest.mark.parametrize("terminal_fixture,expected", [
    ("exact_terminal_case", "excluded_known_build"), ("cold_terminal_case", "failed")])
def test_normal_native_build_terminal_without_completion_survives_frozen_plan(
        scope, terminal_fixture, expected, request, monkeypatch):
    cohort, run, *_ = request.getfixturevalue(terminal_fixture)
    plan, out, _ = scope
    plan.update(cohort)
    path, native = _normal_terminal_summary(cohort, run, monkeypatch)
    original_plan, native_bytes = copy.deepcopy(plan), path.read_bytes()
    frozen = out.parent / "frozen-plan.json"
    frozen.write_text(json.dumps(plan))
    frozen_bytes = frozen.read_bytes()
    summary = write_augmentation_reports(frozen, new_native_summary=path, output_root=out)
    rows = read(out, "augmentation_cases.json")["rows"]
    assert summary["new_case_status_counts"] == {expected: 1, "standby": 1}
    assert rows[0]["native_execution_status"] == rows[0]["accounting_status"] == expected
    assert rows[0]["generic_completion_status"] == "unstarted"
    assert rows[0]["native_result_present"] is True
    assert rows[0]["accounting_status_source"] == "normal_native_summary"
    assert rows[0]["native_summary_source"] == str(path)
    assert rows[0]["native_summary_row_index"] == 0
    assert rows[0]["native_result_provenance"]["technical_terminal_source"] == native["rows"][0]["technical_terminal_source"]
    assert rows[0]["native_result_provenance"]["measurement_started"] is False
    if expected == "failed":
        assert rows[0]["native_result_provenance"]["compiler_dispatch_count"] == 1
        assert rows[0]["native_result_provenance"]["actual_compile_failure"] == native["rows"][0]["actual_compile_failure"]
    else:
        assert rows[0]["native_result_provenance"]["build_exclusion"] == native["rows"][0]["build_exclusion"]
    assert not rows[0]["technical_pair_measured"] and not rows[0]["claim_pair_eligible"]
    assert summary["new_technical_pair_count"] == 0
    assert read(out, "augmentation_cases.json")["native_summary"] == native
    assert plan == original_plan and frozen.read_bytes() == frozen_bytes and path.read_bytes() == native_bytes


def _bound_native_payload(plan, out, *, status="ok"):
    run = out.parent / "new-run"
    plan["augmentation_run"] = str(run)
    return {"native_split_quality_authority": {"run_id": run.name, "stage_run_id": run.name},
        "rows": [{"model": "yolo11l", "case": "b030", "setup_id": SETUP,
            "backend": "deepx_to_trt", "comparison_backend": "deepx",
            "precision": "float32_layout_fp16", "status": status, "ok": status == "ok",
            "eval_run_id": run.name, "report": str(run / "reports/native.json")}]}


def test_native_success_without_completion_is_only_partial_case(scope):
    plan, out, _ = scope
    native = _bound_native_payload(plan, out)
    summary = write_augmentation_reports(plan, new_native_summary=native, output_root=out)
    row = read(out, "augmentation_cases.json")["rows"][0]
    assert row["accounting_status"] == "native_succeeded_generic_pending"
    assert row["generic_completion_status"] == "unstarted" and row["native_execution_status"] == "ok"
    assert row["technical_pair_measured"] is False and row["claim_pair_eligible"] is False
    assert summary["new_completion_pair_count"] == summary["new_technical_pair_count"] == 0


def test_unstarted_completion_projection_does_not_hide_native_failure(scope):
    plan, out, _ = scope
    native = _bound_native_payload(plan, out, status="failed")
    projected = pair("b030")
    projected.update(generic_completion_status="unstarted", eligible_for_technical_transfer=False,
        eligible_for_quality_transfer=False, generic_completed_task_fps=None, native_completed_task_fps=None)
    write_augmentation_reports(plan, {"pairs": [projected]}, native, output_root=out)
    assert read(out, "augmentation_cases.json")["rows"][0]["accounting_status"] == "failed"


@pytest.mark.parametrize("fault", ["backend", "comparison", "precision", "summary_run", "stage_run",
    "row_run", "command_run", "source_path", "duplicate", "missing_run", "inactive_reserve"])
def test_native_terminal_projection_rejects_foreign_or_ambiguous_identity(scope, fault):
    plan, out, _ = scope
    native = _bound_native_payload(plan, out, status="failed")
    row = native["rows"][0]
    if fault == "backend": row["backend"] = "hailo8_to_trt"
    elif fault == "comparison": row["comparison_backend"] = "hailo8"
    elif fault == "precision": row["precision"] = "other"
    elif fault == "summary_run": native["native_split_quality_authority"]["run_id"] = "old-run"
    elif fault == "stage_run": native["native_split_quality_authority"]["stage_run_id"] = "old-run"
    elif fault == "row_run": row["eval_run_id"] = "old-run"
    elif fault == "command_run": row["native_command_contract"] = {"eval_run_id": "old-run"}
    elif fault == "source_path": row["report"] = str(Path(plan["original_run"]) / "native.json")
    elif fault == "duplicate": native["rows"].append(copy.deepcopy(row))
    elif fault == "missing_run": native["native_split_quality_authority"] = {}
    elif fault == "inactive_reserve": plan["rows"][0].update(selection_role="reserve", activated=False)
    before = copy.deepcopy((plan, native))
    with pytest.raises(ValueError):
        write_augmentation_reports(plan, new_native_summary=native, output_root=out)
    assert (plan, native) == before and not out.exists()


@pytest.mark.parametrize("fault", ["missing_config", "config_rows", "stage_rows", "plan_sha", "plan_run",
    "not_frozen", "current_evidence", "measured_success", "partial_repetitions", "failed_runtime",
    "nested_runtime", "raw_repetitions", "zero_fps_measurement", "capability_channel"])
def test_normal_terminal_writer_revalidates_source_and_keeps_capability_guard(
        exact_terminal_case, monkeypatch, fault):
    from scripts import native_producer_final_report as report
    from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256
    cohort, run, build_stage_path, build_stage = exact_terminal_case
    summary_path, before = _normal_terminal_summary(cohort, run, monkeypatch)
    summary_bytes = summary_path.read_bytes()
    reports = run / "reports"
    config_path = reports / "native_producer_stage_config.json"
    config = json.loads(config_path.read_text())
    barrier = config["augmentation_startcase_barrier"]
    stage_path = reports / "native_producer_stage.json"
    stage = json.loads(stage_path.read_text())
    plan_path = Path(barrier["plan_path"])
    plan = json.loads(plan_path.read_text())
    if fault == "missing_config":
        config_path.unlink()
    elif fault == "config_rows":
        barrier["technical_terminals"] = []
        config_path.write_text(json.dumps(config))
    elif fault == "stage_rows":
        stage["augmentation_technical_terminals"][0]["precision"] = "foreign"
        stage_path.write_text(json.dumps(stage))
    elif fault == "plan_sha":
        barrier["plan_sha256"] = "a" * 64
        config_path.write_text(json.dumps(config))
    elif fault in {"plan_run", "not_frozen"}:
        plan["augmentation_run" if fault == "plan_run" else "frozen"] = str(run.parent / "other-run") if fault == "plan_run" else False
        plan_path.write_text(json.dumps(plan))
        barrier["plan_sha256"] = canonical_json_sha256(plan)
        config_path.write_text(json.dumps(config))
    elif fault == "current_evidence":
        build_stage["details"]["deferred_build_readiness"]["blocked_jobs"] = []
        build_stage_path.write_text(json.dumps(build_stage))
    elif fault in {"measured_success", "partial_repetitions", "failed_runtime", "nested_runtime", "raw_repetitions", "zero_fps_measurement"}:
        previous = reports / "previous.json"
        row = copy.deepcopy(before["rows"][0])
        if fault == "measured_success": row.update(status="ok", ok=True)
        elif fault == "partial_repetitions": row.update(repetition_count_attempted=1, repetition_count_valid=0)
        elif fault == "failed_runtime": row.update(runtime_started=True, ok=False)
        elif fault == "nested_runtime": row["child_observation"] = {"measurement_started": True, "ok": False}
        elif fault == "raw_repetitions": row["repetition_records"] = [{"ok": False, "runtime_instance_id": "attempt-1"}]
        elif fault == "zero_fps_measurement": row["fps_makespan"] = 0.0
        previous.write_text(json.dumps({"rows": [row]}))
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--previous-summary", str(previous)])
    elif fault == "capability_channel":
        stage["native_not_supported_rows"] = stage.pop("augmentation_technical_terminals")
        stage_path.write_text(json.dumps(stage))
    with pytest.raises(ValueError, match="augmentation_terminal|native_capability_projection_source_invalid"):
        report.main()
    assert summary_path.read_bytes() == summary_bytes


@pytest.mark.parametrize("fixture", ["exact_terminal_case", "cold_terminal_case"])
def test_normal_terminal_reprojection_preserves_existing_terminal_without_measurement(
        fixture, request, monkeypatch):
    from scripts import native_producer_final_report as report
    cohort, run, *_ = request.getfixturevalue(fixture)
    path, initial = _normal_terminal_summary(cohort, run, monkeypatch)
    previous = run / "reports/previous-summary.json"
    previous.write_bytes(path.read_bytes())
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--previous-summary", str(previous)])
    assert report.main() == 0
    current = json.loads(path.read_text())
    assert current["rows"] == initial["rows"]
    assert current["ok_count"] == initial["ok_count"] == 0
