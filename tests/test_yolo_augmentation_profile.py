"""Normal profile resolution only; never construct or execute a workflow."""
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.yolo_augmentation_plan import COHORT_ID
from onnx_splitpoint_tool.yolo_augmentation_profile import build_augmentation_profile


@pytest.fixture
def profile_source(tmp_path, monkeypatch):
    profile = copy.deepcopy(load_evaluation_profile("thesis_final_evaluated_matrix_v1").raw_profile)
    template = profile["model_suite"]["primary"][1]
    profile["model_suite"]["primary"] = [dict(copy.deepcopy(template), id=m, onnx=str(tmp_path / (m + ".onnx")))
        for m in ("yolo11l", "yolo26s", "yolov7_paper")]
    for row in profile["model_suite"]["primary"]:
        row.pop("path", None)
    profile["selection_policy"]["backend_backfill"] = {"enabled": True}
    profile["hailo_build"] = {"preset": "balanced", "optimization_level": 1, "calib_count": 500,
        "calib_batch_size": 8, "force_build": False, "cache_root": "~/existing/hefs"}
    profile["deepx_build"] = {"calibration_method": "ema", "opt_level": 0, "calib_count": 500, "force_build": False}
    profile["build_scheduler"] = {"prefetch_deepx_full": True}
    profile["remote_cache"] = {"suite_cache_root": "~/original/suites"}
    profile["hardware"] = {"energy_registry_snapshot": {"energy_defaults": {"collector_binary": str(tmp_path / "absent-collector")}}}
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.hardware_matrix.normalize_hardware_targets", lambda _: [])
    profile["campaign"]["auto_bind_dataset_registry"] = False
    profile["validation_execution"] = {"max_items": {"detection": 5000}}
    profile["quality_gate"]["statistics"].update(bootstrap_repetitions=1000, max_active_requests=1)
    profile["native_producers"].update(frames=1000, warmup=100, repetitions=3, queue_depth=3, inflight=8)
    profile["native_producers"]["energy"].update(duration_s=60.0, physical_scope="FS", window_label="command",
        task_budget={"enabled": True, "max_retries": 2, "max_transport_failures": 20})
    run = tmp_path / "original"
    run.mkdir()
    source = {"options": {"profile_start_snapshot": {"resolved_profile": profile},
        "remote_throughput_queue_depth": 2, "remote_warmup": 3, "remote_iters": 5,
        "hailo_build_full": True, "native_producer_energy_enabled": True,
        "window_method_validation_probe_enabled": True, "result_sources": ["original/aggregate.json"]}}
    (run / "run_manifest.json").write_text(json.dumps(source))
    plan = {"cohort_id": COHORT_ID, "original_run": str(run), "base_completion_root": str(tmp_path / "completed"),
        "augmentation_run": str(tmp_path / "new/runs/unique-yolo"),
        "models": {m: {"primary_cases": c, "source_graph": str(tmp_path / (m + ".onnx")), "model_sha256": "1" * 64}
        for m, c in {"yolo11l": ["b120"], "yolo26s": ["b038"], "yolov7_paper": ["b044", "b066"]}.items()}}
    return source, plan


def test_normal_resolver_preserves_contracts_and_narrows_actual_options(profile_source, tmp_path):
    source, plan = profile_source
    original_file = Path(plan["original_run"]) / "run_manifest.json"
    before = original_file.read_bytes()
    result = build_augmentation_profile(plan, tmp_path / "new/plan")
    p, old = result["profile"], source["options"]["profile_start_snapshot"]["resolved_profile"]
    assert original_file.read_bytes() == before
    assert len(p["run_profiles"]) == 3
    assert all(r["type"] == "mixed_backend" and r["variants"] == ["composed"] for r in p["run_profiles"])
    assert result["composed_only_required"] is True
    assert p["selection_policy"]["forced_cases"] == {m: v["primary_cases"] for m, v in plan["models"].items()}
    assert not p["selection_policy"]["backend_backfill"]["enabled"]
    assert not p["hailo_build"]["build_full"] and not p["hailo_build"]["build_part2"]
    assert p["hailo_build"]["cache_root"] == old["hailo_build"]["cache_root"]
    assert not p["native_producers"]["full_baselines"]["enabled"]
    assert not p["native_producers"]["energy"]["window_method_validation_probe"]["enabled"]
    for field in ["quality_gate", "validation_execution", "official_coco_evaluation"]:
        assert p[field] == old[field]
    assert [result["model_checks"][m]["requested_cases"] for m in plan["models"]] == [1, 1, 2]
    options = WorkflowOptions(**result["options"])
    assert options.profile_start_snapshot == result["snapshot"]
    assert options.profile_start_snapshot != source["options"]["profile_start_snapshot"]
    assert options.result_sources == [] and not options.hailo_build_full
    assert options.remote_throughput_queue_depth == 2 and options.remote_iters == 5
    assert options.window_method_validation_probe_enabled is False
    assert options.require_fresh_run is False and options.resume is False
    assert Path(options.out) / options.run_id == Path(plan["augmentation_run"])
    assert result["energy_retry_contract"]["invalid_repeat_max_retries"] == 2
    assert result["snapshot"]["runtime_bindings"]["runtime_materialized"] is True


def test_normal_runner_reads_runtime_snapshot_and_exact_native_scope(profile_source, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import runner
    _, plan = profile_source
    result = build_augmentation_profile(plan, tmp_path / "new/plan")
    monkeypatch.setattr(runner, "package_build_snapshot", lambda: {})
    workflow = runner.EvaluationWorkflowRunner(WorkflowOptions(**result["options"]))
    workflow._load_profile()
    workflow._validate_requested_start_contract()
    cfg = workflow._native_producer_config()
    assert cfg["models"] == list(plan["models"])
    assert cfg["case_policy"] == "case_map_only"
    assert cfg["case_map"] == {m: r["primary_cases"] for m, r in plan["models"].items()}
    assert not cfg["full_baselines"]["enabled"]
    assert [cfg[k] for k in ["frames", "warmup", "repetitions", "queue_depth", "inflight"]] == [1000, 100, 3, 3, 8]
    assert not cfg["energy"]["window_method_validation_probe"]["enabled"]
    assert workflow.profile_payload["workflow_execution"]["native_release_mode"] == "per_case"
    from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
    effective = build_effective_execution_plan(workflow.profile_payload)
    assert effective["native_enabled"] == cfg["enabled"] is True
    assert effective["native_energy_enabled"] == cfg["energy"]["enabled"] is True
    assert effective["native_split_requires_single_part2_input"] is True
    disabled = copy.deepcopy(workflow.profile_payload)
    disabled["native_producers"]["enabled"] = False
    negative = build_effective_execution_plan(disabled)
    assert not negative["native_enabled"] and not negative["native_energy_enabled"]


def test_actual_normal_plan_writer_and_sealed_scope_are_composed_only(profile_source, tmp_path):
    from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
    from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _apply_declared_run_variants, _infer_run_switches
    from onnx_splitpoint_tool.workflow.required_run_scope import build_global_required_run_scope, seal_required_run_scope
    _, plan = profile_source
    profile = build_augmentation_profile(plan, tmp_path / "new/plan")["profile"]
    switches = _infer_run_switches(profile, [])
    generated = BenchmarkGenerationService().build_run_plan(
        **{k: switches.get(k, False) for k in ["acc_cpu", "acc_cuda", "acc_trt", "acc_h8", "acc_h10", "acc_deepx",
            "matrix_trt_to_hailo", "matrix_hailo_to_trt", "matrix_deepx_to_trt", "matrix_trt_to_deepx"]},
        hailo_preset="Custom", hailo_custom_full=False, hailo_custom_composed=True,
        hailo_custom_part1=True, hailo_custom_part2=False, full_hef_policy="skip")
    runs = _apply_declared_run_variants(generated.bench_plan_runs, profile)
    assert {r["id"] for r in runs} == {r["id"] for r in profile["run_profiles"]}
    assert all(r["variants"] == ["composed"] for r in runs)
    assert not generated.hef_full and generated.hef_part1 and not generated.hef_part2
    effective = build_effective_execution_plan(profile)
    assert effective["generic_rows_total"] == 12
    assert effective["bootstrap_repetitions"] == 1000
    assert effective["validation_items"]["detection"] == 5000
    assert effective["calibration_items"]["detection"] == 500
    scope = build_global_required_run_scope(profile_id="augmentation", model_entries=profile["model_suite"]["primary"],
        effective_plan=effective, run_profiles=profile["run_profiles"], created_at="before-generation",
        hardware_targets=[{"id": s, "accelerator": a} for s, a in [
            ("orin_nx_hailo8_01", "hailo8"), ("orin_nx_hailo10_01", "hailo10"), ("orin_nx_deepx_m1_01", "deepx_m1")]])
    sealed = seal_required_run_scope(tmp_path / "new/required_run_scope.json", scope)
    assert len(sealed["physical_run_descriptors"]) == 9
    assert {r["variant"] for r in sealed["physical_run_descriptors"]} == {"split"}
    historical = copy.deepcopy(profile)
    for row in historical["run_profiles"]:
        row.pop("variants")
    assert "ort_tensorrt" in build_effective_execution_plan(historical)["effective_generic_run_ids"]


def test_composed_profile_validation_budget_reaches_normal_writer(profile_source, tmp_path):
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _validation_defaults
    _, plan = profile_source
    profile = build_augmentation_profile(plan, tmp_path / "new/plan")["profile"]
    observed = _validation_defaults(profile, profile["model_suite"]["primary"][0], "yolo11l")
    assert observed["validation_max_images"] == observed["validation_items_requested"] == 5000
    assert observed["validation_budget_authoritative"] is True


@pytest.mark.parametrize("change", [
    {"variants": ["part1", "composed"]}, {"variants": ["full"]},
    {"stage1": "tensorrt", "stage2": "hailo8"}, {"id": "arbitrary_alias"},
    {"stage1": "hailo10"}, {"type": "same_backend_reference"}])
def test_mixed_variant_schema_exception_is_narrow(profile_source, change):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import validate_evaluation_profile_payload
    source, _ = profile_source
    p = copy.deepcopy(source["options"]["profile_start_snapshot"]["resolved_profile"])
    row = {"id": "hailo8_to_trt", "type": "mixed_backend", "stage1": "hailo8", "stage2": "tensorrt", "variants": ["composed"]}
    p["run_profiles"] = [{**row, **change}]
    with pytest.raises(ValueError, match="Invalid evaluation profile"):
        validate_evaluation_profile_payload(p)


@pytest.mark.parametrize("section,key,value", [
    ("hailo_build", "optimization_level", 2), ("deepx_build", "calib_count", 499),
    ("hailo_build", "force_build", True), ("native_producers", "queue_depth", 2)])
def test_refuses_original_recipe_or_runtime_drift(profile_source, tmp_path, section, key, value):
    source, plan = profile_source
    source["options"]["profile_start_snapshot"]["resolved_profile"][section][key] = value
    (Path(plan["original_run"]) / "run_manifest.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="original augmentation contract mismatch"):
        build_augmentation_profile(plan, tmp_path / "new/plan")
    assert not (tmp_path / "new/plan").exists()


def test_refuses_profile_output_in_preserved_base(profile_source):
    _, plan = profile_source
    with pytest.raises(ValueError, match="outside preserved bases"):
        build_augmentation_profile(plan, Path(plan["original_run"]) / "new")
