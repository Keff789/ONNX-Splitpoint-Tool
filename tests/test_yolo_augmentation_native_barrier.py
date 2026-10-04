from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256, native_coordinator_input_hash
from tests.test_yolo_augmentation_plan import exact_terminal_case, cold_terminal_case


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def coordinator():
    name = "yolo_augmentation_native_coordinator_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/run_evalrun_native_producer_variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    run = tmp_path / "new-run"
    (run / "reports").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"run_id": run.name}))
    reference = tmp_path / "old-validation.json"
    reference.write_text(json.dumps({"rows": []}))
    rows = [{"model_id": "yolo11l", "case_id": case, "setup_id": setup, "backend": backend,
             "selection_role": "primary"} for case, setup, backend in (
        ("b120", "h8", "hailo8"), ("b120", "h10", "hailo10h"), ("b120", "dx", "deepx"),
        ("b130", "h8", "hailo8"))]
    for row in rows:
        directory = run / "models" / row["model_id"] / "benchmark_set" / row["case_id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "split_manifest.json").write_text("{}")
    plan = {"cohort_id": "yolo_min3_capability_augmentation", "frozen": True, "rows": rows}
    path = tmp_path / "frozen-plan.json"
    path.write_text(json.dumps(plan))
    cfg = {"native_release_mode": "per_case", "frames": 1000, "warmup": 100,
           "repetitions": 3, "queue_depth": 3, "inflight": 8,
           "remotes": {backend: {"setup_id": setup, "ssh": "fixture@" + setup}
                       for setup, backend in (("h8", "hailo8"), ("h10", "hailo10h"), ("dx", "deepx"))},
           "validation": {"enabled": True}, "augmentation_startcase_barrier": {
               "plan_path": str(path), "plan_sha256": canonical_json_sha256(plan),
               "original_reference_validation_summary": str(reference), "startcases": rows[:3]}}
    variants = [{"id": row["case_id"] + row["backend"], "models": [row["model_id"]],
                 "case_map": {row["model_id"]: [row["case_id"]]}, "backends": [row["backend"]],
                 "release_kind": "split", "full_baselines": {"enabled": False}}
                for row in rows]
    return run, cfg, variants, plan


def test_dispatch_order_keeps_full_plan_and_three_initial_physical_identities(coordinator, fixture):
    run, cfg, variants, plan = fixture
    variants = [variants[3], *variants[:3]]
    before = copy.deepcopy((cfg, variants))
    barrier = coordinator._augmentation_barrier_config(cfg)
    order, starts = coordinator._augmentation_dispatch_order(cfg, variants, barrier)
    assert order == [1, 2, 3, 0]
    assert list(starts.values()) == ["yolo11l|b120|h8", "yolo11l|b120|h10", "yolo11l|b120|dx"]
    assert (cfg, variants) == before
    config = run / "reports/config.json"
    config.write_text(json.dumps(cfg))
    digest = native_coordinator_input_hash(run, config)
    # Repeated ordering never substitutes a three-row temporary plan/config.
    coordinator._augmentation_dispatch_order(cfg, variants, barrier)
    assert native_coordinator_input_hash(run, config) == digest
    assert json.loads(Path(barrier["plan_path"]).read_text()) == plan


@pytest.mark.parametrize("mutation", ["unfrozen", "drift", "duplicate_setup", "absent_start", "full", "reserve", "backend"])
def test_start_scope_or_frozen_plan_conflict_blocks_before_measurement(coordinator, fixture, mutation):
    _, cfg, variants, plan = fixture
    if mutation in ("unfrozen", "drift"):
        plan["frozen"] = mutation != "unfrozen"
        plan["modified"] = True
        Path(cfg["augmentation_startcase_barrier"]["plan_path"]).write_text(json.dumps(plan))
    elif mutation == "duplicate_setup":
        cfg["augmentation_startcase_barrier"]["startcases"][1]["setup_id"] = "h8"
    elif mutation == "absent_start":
        cfg["augmentation_startcase_barrier"]["startcases"][0]["case_id"] = "b999"
    elif mutation == "full":
        variants[0]["release_kind"] = "full"
    elif mutation == "reserve":
        plan["rows"][3]["selection_role"] = "reserve"
        Path(cfg["augmentation_startcase_barrier"]["plan_path"]).write_text(json.dumps(plan))
        cfg["augmentation_startcase_barrier"]["plan_sha256"] = canonical_json_sha256(plan)
    else:
        cfg["augmentation_startcase_barrier"]["startcases"][0]["backend"] = "deepx"
    with pytest.raises(coordinator._AugmentationBarrierError):
        barrier = coordinator._augmentation_barrier_config(cfg)
        coordinator._augmentation_dispatch_order(cfg, variants, barrier)


def controlled_reports(coordinator, fixture, monkeypatch, *, fail_at="", accuracy="accuracy_loss", drift=False):
    run, cfg, variants, plan = fixture
    (run / "native_producers/hailo8").mkdir(parents=True, exist_ok=True)
    (run / "quality_management").mkdir(exist_ok=True)
    (run / "quality_management/central_quality_summary.json").write_text('{"complete":true}')
    calls = []
    monkeypatch.setattr(coordinator, "_select_report_python", lambda *args: (sys.executable, {}))

    def execute(command, *, label, **kwargs):
        calls.append((label, command))
        if "native_producer_final_report.py" in " ".join(command):
            (run / "reports/native_producer_combined_summary.json").write_text(json.dumps({"rows": [{"task_quality_status": accuracy}]}))
        if "native_producer_validate_visualize.py" in " ".join(command):
            path = run / "reports/native_validation/native_producer_validation_summary.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"rows": [{"task_quality_status": accuracy}],
                                       "technical_error_count": 1 if fail_at == "technical_validation" else 0}))
        if drift and label.startswith("augmentation_completion:"):
            path = Path(cfg["augmentation_startcase_barrier"]["plan_path"])
            changed = json.loads(path.read_text()); changed["unplanned_change"] = True
            path.write_text(json.dumps(changed))
        return {"rc": 2 if label.endswith(fail_at) and fail_at else 0, "stderr_tail": fail_at}

    monkeypatch.setattr(coordinator, "_run", execute)
    return calls


def test_start_native_report_validation_and_completion_use_normal_clis_in_order(coordinator, fixture, monkeypatch):
    run, cfg, variants, _ = fixture
    calls = controlled_reports(coordinator, fixture, monkeypatch)
    stages = coordinator._augmentation_publish_startcase(run, cfg, variants,
        case_key="yolo11l|b120|h8", report_python=sys.executable)
    assert len(calls) == 4
    assert list(stages) == ["final_report", "native_validation", "quality_gated_final_report", "completion_callback"]
    val = calls[1][1]
    assert "--existing-outputs-only" in val
    assert val[val.index("--existing-reference-validation-summary") + 1] == cfg["augmentation_startcase_barrier"]["original_reference_validation_summary"]
    assert "--central-quality-summary" in val
    callback = calls[3][1]
    assert callback[callback.index("-m") + 1] == "onnx_splitpoint_tool.workflow.yolo_augmentation"
    assert callback[callback.index("--case-key") + 1] == "yolo11l|b120|h8"
    assert all("energy" not in " ".join(cmd) for _, cmd in calls)
    # Valid scientific accuracy loss does not fail this technical admission.
    assert stages["completion_callback"]["rc"] == 0
    assert (run / "reports/native_producer_summary.json").read_bytes() == (run / "reports/native_producer_combined_summary.json").read_bytes()
    matrix = json.loads((run / "reports/native_expected_matrix.json").read_text())
    assert matrix["expected_row_count"] == 4
    assert matrix["present_expected_row_count"] == 0
    assert matrix["execution_success_complete"] is False


@pytest.mark.parametrize("failure,expected_calls", [("final_report", 1), ("native_validation", 2),
    ("technical_validation", 2), ("quality_gated_final_report", 3), ("yolo11l|b120|h8", 4)])
def test_failure_never_releases_later_startcase_steps(coordinator, fixture, monkeypatch, failure, expected_calls):
    run, cfg, variants, _ = fixture
    calls = controlled_reports(coordinator, fixture, monkeypatch, fail_at=failure)
    with pytest.raises(coordinator._AugmentationBarrierError):
        coordinator._augmentation_publish_startcase(run, cfg, variants,
            case_key="yolo11l|b120|h8", report_python=sys.executable)
    assert len(calls) == expected_calls


def test_final_remaining_callback_checks_canonical_reports_and_unchanged_plan(coordinator, fixture, monkeypatch):
    run, cfg, variants, _ = fixture
    calls = controlled_reports(coordinator, fixture, monkeypatch)
    coordinator._augmentation_publish_startcase(run, cfg, variants, case_key="yolo11l|b120|h8", report_python=sys.executable)
    coordinator._augmentation_completion_callback(run, cfg, case_key="remaining")
    assert calls[-1][1][-2:] == ["--case-key", "remaining"]
    controlled_reports(coordinator, fixture, monkeypatch, drift=True)
    with pytest.raises(coordinator._AugmentationBarrierError, match="plan missing or changed"):
        coordinator._augmentation_completion_callback(run, cfg, case_key="remaining")


def test_nonaugmentation_configuration_and_packaged_mirror_unchanged(coordinator):
    assert coordinator._augmentation_barrier_config({}) is None
    assert coordinator._existing_reference_validation_args({}) == []
    assert coordinator._augmentation_dispatch_order({}, [{}, {}], None) == ([0, 1], {})
    assert (ROOT / "scripts/run_evalrun_native_producer_variants.py").read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts/run_evalrun_native_producer_variants.py").read_bytes()


def _normal_main_fixture(tmp_path):
    from tests.test_v269f_variant_native_split_quality_first import _binding_and_summary, _evalrun
    run = _evalrun(tmp_path)
    (run / "models/yolo26s/benchmark_set/b040").mkdir()
    _, quality = _binding_and_summary(tmp_path)
    quality["complete"] = True
    quality_path = run / "quality_management/central_quality_summary.json"
    quality_path.parent.mkdir()
    quality_path.write_text(json.dumps(quality))
    cases = [{"model_id": "yolo26s", "case_id": case, "setup_id": "hailo8_setup",
              "backend": "hailo8", "selection_role": "primary"} for case in ("b038", "b040")]
    plan = {"cohort_id": "yolo_min3_capability_augmentation", "frozen": True, "rows": cases}
    plan_path = tmp_path / "plan.json"; plan_path.write_text(json.dumps(plan))
    reference = tmp_path / "reference.json"; reference.write_text('{"rows":[]}')
    cfg = {"native_release_mode": "per_case", "backends": ["hailo8"], "split_backends": ["hailo8"],
        "precision": "uint8_dequant_fp16", "frames": 1000, "warmup": 100, "repetitions": 3,
        "build_missing_engines": False, "energy": {"enabled": False}, "validation": {"enabled": True},
        "full_baselines": {"enabled": False},
        "remotes": {"hailo8": {"ssh": "fixture@127.0.0.1", "setup_id": "hailo8_setup"}},
        "variants": [{"id": "new", "case_map": {"yolo26s": ["b038", "b040"]}}],
        "_workflow_context": {"central_quality_summary": str(quality_path),
            "workflow_execution": {"setup_queue_mode": "per_setup"}, "execution_preset": {"id": "standard"}},
        "augmentation_startcase_barrier": {"plan_path": str(plan_path), "plan_sha256": canonical_json_sha256(plan),
            "startcases": cases[:1], "original_reference_validation_summary": str(reference)}}
    config_path = run / "reports/config.json"; config_path.write_text(json.dumps(cfg))
    return run, cfg, config_path


def test_normal_main_stops_at_failed_first_native_leaf_without_releasing_rest(coordinator, tmp_path, monkeypatch):
    """Real binder/checkpoints/coordinator, only outer child execution replaced."""
    run, cfg, config_path = _normal_main_fixture(tmp_path)
    before_hash = native_coordinator_input_hash(run, config_path)
    calls = []

    def controlled_native(command, *, label, **kwargs):
        assert label.startswith("variant:"), (label, command)
        calls.append(command)
        assert json.loads(command[command.index("--case-map") + 1]) == {"yolo26s": ["b038"]}
        assert command[command.index("--frames") + 1] == "1000"
        assert command[command.index("--warmup") + 1] == "100"
        assert command[command.index("--repetitions") + 1] == "3"
        namespace = command[command.index("--artifact-namespace") + 1]
        checkpoint = json.loads((run / "reports/native_producer_variants" / namespace / "performance_stage_result.json").read_text())
        assert checkpoint["state"] == "running" and checkpoint["complete"] is False
        return {"rc": 2, "stderr_tail": "controlled technical native failure; no hardware"}

    monkeypatch.setattr(coordinator, "_run", controlled_native)
    monkeypatch.setattr(sys, "argv", ["run_evalrun_native_producer_variants.py", "--eval-run-dir", str(run), "--config", str(config_path)])
    assert coordinator.main() == 2
    assert len(calls) == 1
    stage = json.loads((run / "reports/native_producer_stage.json").read_text())
    assert stage["failure_class"] == "augmentation_startcase_barrier"
    assert stage["native_energy"]["status"] == "not_started_startcase_barrier"
    assert len(stage["variant_results"]) == 1
    assert len(stage["case_release_plan"]["expected_rows"]) == 2
    assert native_coordinator_input_hash(run, config_path) == before_hash


@pytest.mark.parametrize("fault", ["none", "no_resume", "quality_snapshot", "checkpoint_digest", "input_hash", "unfinished_leaf", "ordinary_cohort"])
def test_actual_explicit_partial_reentry_preserves_completed_leaf_and_revalidates_outputs(
        coordinator, tmp_path, monkeypatch, fault):
    """Real main twice; interruption is at the outer report process boundary.

    A completed execution record is sufficient to avoid a duplicate physical
    dispatch, but never substitutes for the following normal output validation.
    """
    from onnx_splitpoint_tool.workflow.checkpoints import write_stage_checkpoint
    run, cfg, config_path = _normal_main_fixture(tmp_path)
    if fault == "ordinary_cohort":
        cfg.pop("augmentation_startcase_barrier")
        cfg["_workflow_context"]["workflow_execution"]["setup_queue_mode"] = "serial"
        config_path.write_text(json.dumps(cfg))
    args = ["coordinator", "--eval-run-dir", str(run), "--config", str(config_path)]
    calls = []

    class ControlledInterruption(BaseException):
        pass

    def first(command, *, label, **kwargs):
        calls.append(label)
        if label.startswith("variant:"):
            (run / "native_producers/hailo8").mkdir(parents=True, exist_ok=True)
            return {"rc": 0, "status": "controlled_native_execution_completed"}
        raise ControlledInterruption(label)

    monkeypatch.setattr(coordinator, "_run", first)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ControlledInterruption):
        coordinator.main()
    assert sum(label.startswith("variant:") for label in calls) == 1
    leaf = next((run / "reports/native_producer_variants").glob("*/performance_stage_result.json"))
    leaf_payload = json.loads(leaf.read_text())
    assert leaf_payload["state"] == "completed"
    leaf_before = leaf.read_bytes()
    snapshot = Path(leaf_payload["details"]["variant"]["central_quality_summary"])
    quality_before = snapshot.read_bytes()
    performance, _ = coordinator._native_performance_checkpoint_paths(run)
    performance_payload = json.loads(performance.read_text())
    assert performance_payload["state"] == "running"
    if fault == "quality_snapshot":
        snapshot.write_text("{}")
    elif fault == "checkpoint_digest":
        performance_payload["details"]["tampered"] = True
        performance.write_text(json.dumps(performance_payload))
    elif fault == "input_hash":
        write_stage_checkpoint(performance, stage="native_performance", state="running", complete=False,
                               input_hash="f" * 64, run_root=run)
    elif fault == "unfinished_leaf":
        write_stage_checkpoint(leaf, stage="native_case_performance", state="running", complete=False,
                               input_hash=leaf_payload["input_hash"], run_root=run,
                               details={"variant": leaf_payload["details"]["variant"]})
    resumed_calls = []

    def second(command, *, label, **kwargs):
        resumed_calls.append(label)
        assert not label.startswith("variant:"), "completed Native must never be dispatched twice"
        # No fake measured outputs: the actual next normal report boundary
        # rejects this metadata-only controlled workload.
        return {"rc": 2, "stderr_tail": "controlled missing actual measured outputs"}

    monkeypatch.setattr(coordinator, "_run", second)
    monkeypatch.setattr(sys, "argv", args + ([] if fault == "no_resume" else ["--resume"]))
    assert coordinator.main() == (70 if fault == "no_resume" else 2)
    assert resumed_calls == (["augmentation_startcase:final_report"] if fault == "none" else [])
    if fault != "unfinished_leaf":
        assert leaf.read_bytes() == leaf_before
    if fault != "quality_snapshot":
        assert snapshot.read_bytes() == quality_before
    if fault == "none":
        stage = json.loads((run / "reports/native_producer_stage.json").read_text())
        assert stage["variant_results"][0]["reused"] is True
        assert stage["native_performance_checkpoint"]["partial_leaf_resume"] is True
        assert stage["native_energy"]["status"] == "not_started_startcase_barrier"
        # Failure is terminal; even another explicit entry never retries it.
        assert coordinator.main() == 2
        assert resumed_calls == ["augmentation_startcase:final_report"]


def _terminal_barrier_fixture(coordinator, exact_terminal_case, tmp_path):
    from onnx_splitpoint_tool.yolo_augmentation_plan import collect_technical_terminals
    plan, run, stage_path, stage = exact_terminal_case
    plan.update(cohort_id="yolo_min3_capability_augmentation", frozen=True)
    path = tmp_path / "terminal-plan.json"; path.write_text(json.dumps(plan))
    original = tmp_path / "original-validation.json"; original.write_text('{"rows": []}')
    row = plan["rows"][0]
    cfg = {"native_release_mode": "per_case", "remotes": {
        row["backend"]: {"setup_id": row["setup_id"], "ssh": "fixture@localhost"}},
        "augmentation_startcase_barrier": {"plan_path": str(path), "plan_sha256": canonical_json_sha256(plan),
            "startcases": [row], "original_reference_validation_summary": str(original),
            "technical_terminals": collect_technical_terminals(plan, run)}}
    return plan, run, cfg, stage_path


def test_exact_build_nonstart_retains_terminal_identity_without_selecting_a_reserve(
        coordinator, exact_terminal_case, tmp_path):
    plan, run, cfg, stage_path = _terminal_barrier_fixture(coordinator, exact_terminal_case, tmp_path)
    before = stage_path.read_bytes()
    barrier = coordinator._augmentation_barrier_config(cfg)
    row = plan["rows"][0]
    leaf = {"models": [row["model_id"]], "case_map": {row["model_id"]: [row["case_id"]]},
            "backends": [row["backend"]], "release_kind": "split", "full_baselines": {"enabled": False}}
    order, required = coordinator._augmentation_dispatch_order(cfg, [leaf], barrier)
    assert order == [0] and required == {}
    assert barrier["technical_terminals"][0]["status"] == "excluded_known_build"
    assert barrier["technical_terminals"][0]["measurement_started"] is False
    assert plan["rows"][1]["activated"] is False
    assert stage_path.read_bytes() == before


@pytest.mark.parametrize("fault", ["missing_proof", "changed_record", "scope_drift"])
def test_local_terminal_never_comes_from_text_or_stale_config(
        coordinator, exact_terminal_case, tmp_path, fault):
    _, _, cfg, stage_path = _terminal_barrier_fixture(coordinator, exact_terminal_case, tmp_path)
    if fault == "missing_proof":
        cfg["augmentation_startcase_barrier"]["technical_terminals"] = []
    elif fault == "changed_record":
        stage_path.write_text("{}")
    else:
        cfg["augmentation_startcase_barrier"]["technical_terminals"][0]["case"] = "b999"
    with pytest.raises(coordinator._AugmentationBarrierError, match="terminal evidence"):
        coordinator._augmentation_barrier_config(cfg)


def test_actual_normal_main_continues_independent_fixed_start_after_exact_build_nonstart(
        coordinator, exact_terminal_case, tmp_path, monkeypatch):
    import shutil
    from onnx_splitpoint_tool.yolo_augmentation_plan import collect_technical_terminals
    run, cfg, config_path = _normal_main_fixture(tmp_path / "normal")
    negative_plan, negative_run, _, _ = exact_terminal_case
    negative = copy.deepcopy(negative_plan["rows"][0])
    shutil.copytree(negative_run / "models" / negative["model_id"], run / "models" / negative["model_id"])
    direct = run / "models" / negative["model_id"] / "benchmark_set" / negative["case_id"]
    direct.mkdir()
    (direct / "split_manifest.json").write_text(json.dumps({"cut_tensors": negative["cut_tensors"]}))
    path = Path(cfg["augmentation_startcase_barrier"]["plan_path"])
    plan = json.loads(path.read_text())
    selected = plan["rows"][0]
    selected.update(model_sha256="a" * 64, case_key="fixed-independent", part2_external_inputs=["cut"], cut_tensors=["cut"])
    plan.update(augmentation_run=str(run), rows=[negative, selected])
    path.write_text(json.dumps(plan))
    cfg["variants"] = [
        {"id": "known-negative", "case_map": {negative["model_id"]: [negative["case_id"]]},
         "remotes": {"hailo8": {"setup_id": negative["setup_id"], "ssh": "fixture@localhost"}}},
        {"id": "independent", "case_map": {"yolo26s": ["b038"]}},
    ]
    barrier = cfg["augmentation_startcase_barrier"]
    barrier.update(plan_sha256=canonical_json_sha256(plan), startcases=[negative, selected],
                   technical_terminals=collect_technical_terminals(plan, run))
    assert len(barrier["technical_terminals"]) == 1
    config_path.write_text(json.dumps(cfg))
    calls = []

    class StopAtPhysicalLeaf(BaseException):
        pass

    def controlled(command, *, label, **kwargs):
        calls.append(label)
        assert label.startswith("variant:")
        assert json.loads(command[command.index("--case-map") + 1]) == {"yolo26s": ["b038"]}
        raise StopAtPhysicalLeaf

    monkeypatch.setattr(coordinator, "_run", controlled)
    monkeypatch.setattr(sys, "argv", ["coordinator", "--eval-run-dir", str(run), "--config", str(config_path)])
    with pytest.raises(StopAtPhysicalLeaf):
        coordinator.main()
    assert len(calls) == 1
    stage = json.loads((run / "reports/native_producer_stage.json").read_text())
    assert len(stage["case_release_plan"]["expected_rows"]) == 2
    assert stage["augmentation_technical_terminals"] == barrier["technical_terminals"]
    assert not stage.get("native_not_supported_rows")


@pytest.mark.parametrize("fault", ["none", "identity", "measured", "output_error", "duplicate", "unrelated_error"])
def test_cold_compile_failure_is_local_but_raw_failure_and_other_blockers_remain(
        coordinator, cold_terminal_case, tmp_path, fault):
    _, _, cfg, _ = _terminal_barrier_fixture(coordinator, cold_terminal_case[:4], tmp_path)
    original = cfg["augmentation_startcase_barrier"]["technical_terminals"][0]
    assert original["status"] == "failed" and original["compiler_dispatch_count"] == 1
    assert "build_exclusion" not in original
    row = {**copy.deepcopy(original), "status": "native_row_not_ok", "dump_status": "skipped",
           "technical_quality_error": True, "ok": False, "semantic_available": False}
    payload = {"rows": [row], "technical_error_count": 1}
    if fault == "identity":
        row["precision"] = "wrong-precision"
    elif fault == "measured":
        row["semantic_available"] = True
    elif fault == "output_error":
        row["status"] = "missing_dump"
    elif fault == "duplicate":
        payload["rows"].append(copy.deepcopy(row)); payload["technical_error_count"] = 2
    elif fault == "unrelated_error":
        payload["rows"].append({**row, "case": "b999", "case_id": "b999", "status": "missing_dump"})
        payload["technical_error_count"] = 2
    before = copy.deepcopy((payload, cfg))
    blocking, local = coordinator._augmentation_validation_blockers(cfg, payload)
    assert local == (1 if fault in {"none", "unrelated_error"} else 0)
    assert blocking == payload["technical_error_count"] - local
    assert (payload, cfg) == before
    assert coordinator._augmentation_validation_blockers({}, payload) == (payload["technical_error_count"], 0)
