"""AP07 leaf planning and real Quality-FIRST binding, without remote dispatch."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.trt_quality_chain import (
    TensorRTQualityChainError,
    producer_set_from_central_quality_summary,
    split_binding_set_from_central_quality_summary,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
from tests.test_v269d_trt_quality_chain import _result, _strict_producer, _summary
from tests.test_v269f_variant_native_split_quality_first import (
    _binding_and_summary, _evalrun, _load_script, _variant_cfg, _write_summary,
)


@pytest.fixture
def coordinator():
    return _load_script("run_evalrun_native_producer_variants.py")


def _matrix_keys(rows):
    return sorted((r["setup_id"], r["model"], r["case"],
                   r["execution_mode"], r["backend"]) for r in rows)


def test_leaf_expansion_preserves_full_matrix_and_deduplicates_full(coordinator, tmp_path):
    run = _evalrun(tmp_path)
    (run / "models/yolo26s/benchmark_set/b039").mkdir()
    cfg = {
        "backends": ["hailo8", "hailo10h"],
        "split_backends": ["hailo8", "hailo10h"],
        "precision": "uint8_dequant_fp16",
        "full_baselines": {"enabled": True, "backends_by_producer": {
            "hailo8": ["hailo8", "tensorrt"],
            "hailo10h": ["hailo10h", "tensorrt"],
        }},
        "remotes": {
            "hailo8": {"setup_id": "h8", "ssh": "nx@h8"},
            "hailo10h": {"setup_id": "h10", "ssh": "nx@h10"},
        },
    }
    variants = [
        {"id": "first", "case_map": {"yolo26s": ["b038"]}},
        {"id": "second", "case_map": {"yolo26s": ["b039"]}},
    ]
    original = copy.deepcopy((cfg, variants))
    expected, _, _, _ = coordinator._variant_expected_energy_rows(run, cfg, variants)
    leaves = coordinator._per_case_variants(run, cfg, variants)
    assert (cfg, variants) == original
    assert leaves == coordinator._per_case_variants(run, cfg, variants)
    assert len(leaves) == 8
    namespaces = [leaf["artifact_namespace"] for leaf in leaves]
    assert len(set(namespaces)) == len(leaves)
    observed = []
    for leaf in leaves:
        merged = coordinator._merged_variant(cfg, leaf)
        assert merged["precision"] == "uint8_dequant_fp16"
        assert len(coordinator._variant_backend_bindings(cfg, leaf)) == 1
        assert set(coordinator._effective_variant_case_map(run, merged)) == {"yolo26s"}
        rows, _, _, _ = coordinator._variant_expected_energy_rows(run, cfg, [leaf])
        assert len(rows) == 1
        observed.extend(rows)
    assert _matrix_keys(observed) == _matrix_keys(expected)
    full = [row for row in observed if row["execution_mode"] == "native_full_baseline"]
    assert len(full) == 4
    assert len(set(_matrix_keys(full))) == 4


def test_full_only_without_split_directories_keeps_two_full_leaves(coordinator, tmp_path):
    run = tmp_path / "full-only"
    suite = run / "models/resnet50/benchmark_set/legacy_suite"
    suite.mkdir(parents=True)
    (suite / "benchmark_set.json").write_text(json.dumps({"benchmark_task": "classification"}))
    cfg = {"models": ["resnet50"], "backends": ["hailo8"], "split_backends": [],
           "precision": "float32_layout_fp16", "full_baselines": {
               "enabled": True, "backends_by_producer": {"hailo8": ["hailo8", "tensorrt"]}}}
    variants = [{"id": "full-only", "models": ["resnet50"], "case_map": {"resnet50": []}}]
    assert coordinator._effective_variant_case_map(run, coordinator._merged_variant(cfg, variants[0])) == {"resnet50": []}
    leaves = coordinator._per_case_variants(run, cfg, variants)
    assert len(leaves) == 2
    rows, _, _, _ = coordinator._variant_expected_energy_rows(run, cfg, leaves)
    assert len(rows) == 2
    assert {r["backend"] for r in rows} == {"native_full_hailo8", "native_full_tensorrt"}
    assert all(r["case"] == "full" for r in rows)
    assert not list(suite.glob("b[0-9]*"))


def _select_split(summary):
    return split_binding_set_from_central_quality_summary(
        summary, eval_run_id="eval-native-split-001", setup_id="hailo8_setup",
        selections=[{"model_id": "yolo26s", "case_id": "b038", "backend": "hailo8_to_trt",
                     "task": "detection", "precision": "uint8_dequant_fp16"}],
    )


def test_accuracy_loss_and_repeat_completion_keep_exact_split_binding(coordinator, tmp_path):
    run = _evalrun(tmp_path)
    _, summary = _binding_and_summary(tmp_path)
    result = summary["results"][0]
    result.update(decision="accuracy_loss", scientific_status="failed", scientific_pass=False)
    summary["results"].append(copy.deepcopy(result))
    selected = _select_split(summary)
    assert len(selected["bindings_by_model_case_backend"]) == 1
    cfg = _variant_cfg(_write_summary(run, summary))
    variants = [{"id": "split", "case_map": {"yolo26s": ["b038"]}, "full_baselines": {"enabled": False}}]
    paths, plan = coordinator._materialize_native_split_quality_binding_sets(run, cfg, variants)
    prepared = coordinator._quality_first_variant_plan(cfg, variants, {}, {"owner_by_setup": {}}, paths, plan)
    command = coordinator._build_update_cmd(run, cfg, prepared[0], refresh_suites=False, timeout_s=60)
    assert "--native-split-quality-required" in command
    assert "--no-build-missing-engines" in command
    assert len(prepared) == 1


@pytest.mark.parametrize("failure", ["missing", "cancelled", "technical_incomplete", "binding_missing", "request_sha_mismatch"])
def test_unfinished_or_unbound_split_cannot_materialize_release(coordinator, tmp_path, failure):
    run = _evalrun(tmp_path)
    _, summary = _binding_and_summary(tmp_path)
    result = summary["results"][0]
    if failure == "missing":
        summary["results"] = []
    elif failure == "cancelled":
        result.update(status="cancelled", technical_status="cancelled")
    elif failure == "technical_incomplete":
        result["technical_status"] = "running"
    elif failure == "binding_missing":
        result.pop("native_split_quality_binding")
    else:
        result["source_request_sha256"] = "f" * 64
    cfg = _variant_cfg(_write_summary(run, summary))
    with pytest.raises(TensorRTQualityChainError):
        coordinator._materialize_native_split_quality_binding_sets(
            run, cfg, [{"id": "split", "case_map": {"yolo26s": ["b038"]}}],
        )
    assert not list((run / "reports/native_split_quality_binding_sets").rglob("native_split_quality_binding_set.json"))


@pytest.mark.parametrize("state", ["accuracy_loss", "missing", "cancelled", "binding_missing"])
def test_full_producer_technical_completion_controls_release(state):
    result = _result(_strict_producer())
    result.update(decision="accuracy_loss", scientific_status="failed", scientific_pass=False)
    summary = _summary([result, copy.deepcopy(result)])
    if state == "missing":
        summary["results"] = []
    elif state == "cancelled":
        for row in summary["results"]:
            row.update(status="cancelled", technical_status="cancelled")
    elif state == "binding_missing":
        for row in summary["results"]:
            row["producer_binding_eligible"] = False
    kwargs = dict(eval_run_id="eval-20260721", setup_id="orin_nx_hailo8_01", model_ids=["resnet50"])
    if state == "accuracy_loss":
        selected = producer_set_from_central_quality_summary(summary, **kwargs)
        assert list(selected["producers_by_model"]) == ["resnet50"]
    else:
        with pytest.raises(TensorRTQualityChainError):
            producer_set_from_central_quality_summary(summary, **kwargs)


def _normal_parent(tmp_path, *, models, full_only=False):
    """Only initialize the state consumed by real read-only planner methods."""
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path / "normal-profile"
    runner.run_id = runner.run_dir.name
    runner.options = WorkflowOptions(profile="", out=str(runner.run_dir))
    runner.manifest = {"models": {model: {} for model in models}}
    runner.profile_payload = {
        "hardware_targets": [
            {"id": "h8", "accelerator": "hailo8", "host": {"address": "h8", "user": "nx"}},
            {"id": "h10", "accelerator": "hailo10h", "host": {"address": "h10", "user": "nx"}},
        ],
    }
    cfg = {
        "backends": ["hailo8", "hailo10h"],
        "split_backends": [] if full_only else ["hailo8", "hailo10h"],
        "models": list(models), "validation": {"enabled": True},
        "frames": 100, "warmup": 10, "repetitions": 1,
        "full_baselines": {"enabled": True, "backends_by_producer": {
            "hailo8": ["hailo8", "tensorrt"], "hailo10h": ["hailo10h", "tensorrt"],
        }},
    }
    runner.profile_payload["native_producers"] = copy.deepcopy(cfg)
    for model, case in models.items():
        suite = runner.run_dir / "models" / model / "benchmark_set/legacy_suite"
        suite.mkdir(parents=True)
        cases = [] if full_only else [{"case_id": case}]
        (suite / "benchmark_set.json").write_text(json.dumps({
            "cases": cases, "benchmark_task": "detection" if model.startswith("yolo") else "classification"}))
        (suite / "benchmark_plan.json").write_text(json.dumps({"runs": [{"id": "hailo8_full"}]}))
        (suite / "benchmark_suite.py").write_text("# Not executed: local selection fixture.\n")
        if not full_only:
            (suite / case).mkdir()
            (suite / case / "split_manifest.json").write_text(json.dumps({"part2_external_inputs": ["boundary"]}))
    return runner, cfg


@pytest.mark.parametrize("models,expected_count", [
    ({"mobilenet_v3_large": "b135", "yolo11l": "b062"}, 12),
    ({"mobilenet_v3_large": "b135"}, 6),
])
def test_normal_profile_translation_keeps_real_model_precision_and_matrix(coordinator, tmp_path, models, expected_count):
    runner, cfg = _normal_parent(tmp_path, models=models)
    original = copy.deepcopy(cfg)
    variants = runner._normal_native_release_variants(cfg)
    assert cfg == original
    assert len(variants) == len(models) * 2
    scoped = {**cfg, "variants": variants}
    scoped["remotes"] = runner._materialize_variant_native_remotes(scoped)
    for variant in variants:
        model = variant["models"][0]
        backend = variant["backends"][0]
        expected_precision = "float32_layout_fp16" if backend == "hailo8" and model == "mobilenet_v3_large" else "uint8_dequant_fp16"
        assert variant["precision"] == expected_precision
        assert variant["case_map"] == {model: [models[model]]}
    leaves = coordinator._per_case_variants(runner.run_dir, scoped, variants)
    rows, _, _, _ = coordinator._variant_expected_energy_rows(runner.run_dir, scoped, leaves)
    assert len(rows) == len(leaves) == expected_count
    assert {r["setup_id"] for r in rows} == {"h8", "h10"}
    assert len(set(_matrix_keys(rows))) == expected_count


def test_normal_profile_full_only_translation_has_no_synthetic_split(coordinator, tmp_path):
    runner, cfg = _normal_parent(tmp_path, models={"mobilenet_v3_large": "b135"}, full_only=True)
    variants = runner._normal_native_release_variants(cfg)
    assert len(variants) == 2
    assert all(v["split_backends"] == [] for v in variants)
    scoped = {**cfg, "variants": variants}
    scoped["remotes"] = runner._materialize_variant_native_remotes(scoped)
    leaves = coordinator._per_case_variants(runner.run_dir, scoped, variants)
    rows, _, _, _ = coordinator._variant_expected_energy_rows(runner.run_dir, scoped, leaves)
    assert len(rows) == len(leaves) == 4
    assert all(r["execution_mode"] == "native_full_baseline" for r in rows)


@pytest.mark.parametrize("models,expected_count", [
    ({"mobilenet_v3_large": "b135", "yolo11l": "b062"}, 12),
    ({"mobilenet_v3_large": "b135"}, 6),
])
def test_bounded_gui_contract_requires_independent_exact_matrix_and_never_final(coordinator, tmp_path, models, expected_count):
    runner, cfg = _normal_parent(tmp_path, models=models)
    variants = runner._normal_native_release_variants(cfg)
    cfg["variants"] = variants
    cfg["remotes"] = runner._materialize_variant_native_remotes(cfg)
    rows, *_ = coordinator._variant_expected_energy_rows(runner.run_dir, cfg, variants)
    assert len(rows) == expected_count
    cfg["_workflow_context"] = {"execution_preset": {"id": "standard"}, "campaign": {"mode": "development"}}
    cfg["native_performance_checkpoint"] = {"scope": "bounded_gui_acceptance", "required_row_count": expected_count}
    assert coordinator._native_performance_required_campaign_rows(cfg, expected_row_count=len(rows)) == expected_count
    with pytest.raises(ValueError, match="frozen matrix"):
        coordinator._native_performance_required_campaign_rows(cfg, expected_row_count=len(rows) + 1)
    with pytest.raises(ValueError):
        coordinator._native_performance_required_campaign_rows(cfg)
    for field in ("campaign", "execution_preset"):
        final_cfg = copy.deepcopy(cfg)
        final_cfg["_workflow_context"][field]["mode" if field == "campaign" else "id"] = "final"
        with pytest.raises(ValueError):
            coordinator._native_performance_required_campaign_rows(final_cfg, expected_row_count=len(rows))
    ordinary = copy.deepcopy(cfg)
    ordinary.pop("native_performance_checkpoint")
    assert coordinator._native_performance_required_campaign_rows(ordinary, expected_row_count=len(rows)) == 63
