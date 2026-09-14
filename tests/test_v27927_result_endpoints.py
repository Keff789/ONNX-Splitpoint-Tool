from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from onnx_splitpoint_tool.workflow.required_run_scope import _endpoint_contract
from onnx_splitpoint_tool.workflow.results import (
    _validation_report_to_row,
    expand_normalized_benchmark_rows,
    normalize_benchmark_files,
    normalize_benchmark_row,
)
from onnx_splitpoint_tool.workflow.runner import _bind_results_to_required_scope_v2796

ROOT = Path(__file__).resolve().parents[1]
SETUP = "orin_nx_deepx_m1_01"


def _runner_endpoint(**kwargs: Any) -> str:
    source = (ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_measured_endpoint_for_variant_v27927")
    namespace = {"Optional": Optional, "Sequence": Sequence}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "runner_endpoint", "exec"), namespace)
    return namespace[node.name](**kwargs)


@pytest.mark.parametrize("scope,expected", [
    ("backend_endpoint", "raw_model_outputs"),
    ("hailo_raw_head_plus_hash_bound_onnx_host_tail", "raw_model_outputs"),
    ("hailo_raw_head_plus_frozen_decode_nms", "completed_detection"),
])
def test_full_endpoint_describes_actual_timed_work(scope: str, expected: str) -> None:
    assert _runner_endpoint(variant="full", task="detection", status="ok", runs_ms=[18.2], full_timing_scope=scope) == expected


@pytest.mark.parametrize("variant,task,expected", [
    ("composed", "detection", "p2_output"),
    ("part2", "detection", "p2_output"),
    ("part1", "classification", "p1_output"),
    ("full", "classification", "classification_logits"),
])
def test_stage_endpoints(variant: str, task: str, expected: str) -> None:
    assert _runner_endpoint(variant=variant, task=task, status="ok", runs_ms=[1.0]) == expected


@pytest.mark.parametrize("status,runs", [("error", [1.0]), ("ok", []), ("missing", None)])
def test_no_physical_endpoint_without_completed_timing(status: str, runs: Any) -> None:
    assert _runner_endpoint(variant="full", task="detection", status=status, runs_ms=runs) == ""


def _trt_row() -> dict:
    # Timings and run layout from the 2.79.26 D pack.  New fields explicitly
    # describe the executed generic intervals rather than the later quality NMS.
    return {
        "case_id": "b003", "run_id": "ort_tensorrt", "primary_variant": "composed",
        "stage1_provider": "tensorrt", "stage2_provider": "tensorrt", "full_provider": "tensorrt",
        "setup_id": SETUP, "task": "detection",
        "composed_mean_ms": 24.585247039794922, "full_mean_ms": 18.24474334716797,
        "part1_mean_ms": 7.301950454711914, "part2_mean_ms": 16.797876358032227,
        "variant_status": {"composed": "ok", "full": "ok"},
        "runtime_contract_decision": "fail", "runtime_ok": True,
        "measurement_endpoint": "p2_output", "full_measurement_endpoint": "raw_model_outputs",
        "measurement_endpoints_by_variant": {"full": "raw_model_outputs", "composed": "p2_output"},
        "quality_endpoint": "completed_detection",
    }


def test_full_companion_uses_full_endpoint_and_preserves_setup() -> None:
    rows = expand_normalized_benchmark_rows(_trt_row(), model_id="yolo11l", source_path=Path("benchmark_results_ort_tensorrt_auto.json"))
    by_variant = {r["variant"]: r for r in rows}
    assert by_variant["split"]["measurement_endpoint"] == "p2_output"
    assert by_variant["full"]["measurement_endpoint"] == "raw_model_outputs"
    assert all(r["setup_id"] == SETUP for r in rows)
    assert all(r["quality_endpoint"] == "completed_detection" for r in rows)
    assert by_variant["full"]["total_latency_ms"] == 18.24474334716797
    # Central quality must not overwrite a recorded physical validation failure.
    assert by_variant["split"]["validation_ok"] is False


def test_full_companion_does_not_borrow_split_only_endpoint() -> None:
    row = _trt_row()
    row.pop("full_measurement_endpoint")
    row.pop("measurement_endpoints_by_variant")
    rows = expand_normalized_benchmark_rows(row, model_id="yolo11l", source_path=Path("benchmark_results_ort_tensorrt_auto.json"))
    assert next(r for r in rows if r["variant"] == "full")["measurement_endpoint"] == ""


def test_conflicting_endpoint_declarations_remain_unbound() -> None:
    row = _trt_row()
    row["measurement_endpoints_by_variant"]["full"] = "completed_detection"
    rows = expand_normalized_benchmark_rows(row, model_id="yolo11l", source_path=Path("benchmark_results_ort_tensorrt_auto.json"))
    assert next(r for r in rows if r["variant"] == "full")["measurement_endpoint"] == ""


def test_compact_validation_report_preserves_timing_endpoints() -> None:
    report = _trt_row()
    report["run_cfg"] = {"provider": "tensorrt", "full_provider": "tensorrt", "stage1_provider": "tensorrt", "stage2_provider": "tensorrt"}
    report["timings"] = {
        "full": {"mean_ms": 18.24474334716797, "measurement_endpoint": "raw_model_outputs"},
        "composed": {"mean_ms": 24.585247039794922, "measurement_endpoint": "p2_output"},
    }
    converted = _validation_report_to_row(report, Path("b003/results_ort_tensorrt/validation_report.json"))
    rows = expand_normalized_benchmark_rows(converted, model_id="yolo11l", source_path=Path("validation_report.json"))
    assert {r["measurement_endpoint"] for r in rows} == {"p2_output", "raw_model_outputs"}
    assert all(r["setup_id"] == SETUP for r in rows)


def _write_pair(tmp_path: Path, json_rows: list[dict], csv_rows: list[dict]) -> None:
    target = tmp_path / "benchmark_results_ort_tensorrt_auto.json"
    target.write_text(json.dumps(json_rows))
    keys = list(dict.fromkeys(key for row in csv_rows for key in row))
    with target.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(csv_rows)


def test_json_wins_over_lossy_sibling_csv_without_losing_csv_only_case(tmp_path: Path) -> None:
    row = _trt_row()
    mirror = dict(row)
    mirror.pop("setup_id")
    extra = {**mirror, "case_id": "b004", "full_mean_ms": None}
    _write_pair(tmp_path, [row], [mirror, extra])
    rows, sources = normalize_benchmark_files(model_id="yolo11l", source_paths=[tmp_path])
    assert len(rows) == 3
    assert sum(r["case_id"] == "b003" for r in rows) == 1
    assert sum(r["case_id"] == "b004" for r in rows) == 1
    assert next(s for s in sources if s["path"].endswith(".csv"))["suppressed_csv_mirror_count"] == 1


@pytest.mark.parametrize("changed", [
    {"setup_id": "another_setup"},
    {"measurement_endpoint": "completed_detection", "full_mean_ms": None, "measurement_endpoints_by_variant": {}},
])
def test_distinct_csv_measurement_survives(tmp_path: Path, changed: dict) -> None:
    row = _trt_row()
    csv_row = {**row, **changed}
    _write_pair(tmp_path, [row], [csv_row])
    rows, sources = normalize_benchmark_files(model_id="yolo11l", source_paths=[tmp_path])
    assert next(s for s in sources if s["path"].endswith(".csv"))["suppressed_csv_mirror_count"] == 0
    assert len([r for r in rows if r["variant"] == "split"]) == 2


def test_failed_deepx_diagnostic_never_becomes_full_timing() -> None:
    row = {
        "case_id": "full", "run_id": "deepx_m1_full", "backend": "deepx_m1",
        "variant": "full", "primary_variant": "full", "task": "detection",
        "runtime_ok": False, "total_latency_ms": None, "full_mean_ms": None,
        "full_e2e_latency_ms": 79.937, "full_e2e_mean_ms": 79.937,
        "performance_benchmark_source": "explicit_deepx_contract_required",
        "latency_semantics": "diagnostic_only", "dxrt_tool_fps": 49.88,
        "variant_status": {"full": "error"}, "error_class": "deepx_runtime_failed",
    }
    normalized = normalize_benchmark_row(row, model_id="yolo11l", source_path=Path("benchmark_results_deepx_m1_full_auto.json"))
    assert normalized["total_latency_ms"] is None
    assert normalized["full_e2e_latency_ms"] is None
    assert normalized["throughput_primary_fps"] is None
    assert normalized["diagnostic_full_e2e_latency_ms"] == 79.937
    assert normalized["backend_tool_fps"] == 49.88
    assert normalized["runtime_ok"] is False


def test_new_generic_scope_matches_raw_model_interval_but_native_requires_completion() -> None:
    assert _endpoint_contract("ort_tensorrt", "detection", "full")[:2] == ("raw_model_outputs", "completed_detection")
    assert _endpoint_contract("ort_cpu", "detection", "full")[0] == "raw_model_outputs"
    assert _endpoint_contract("deepx_m1_full", "detection", "full")[0] == "completed_detection"
    assert _endpoint_contract("hailo8", "detection", "full")[0] == "completed_detection"


def test_old_sealed_completed_scope_rejects_raw_measurement() -> None:
    expected = {
        "model_id": "yolo11l", "case_id": "full", "run_id": "ort_tensorrt",
        "backend": "tensorrt", "variant": "full", "expected_setup_id": SETUP,
        "measurement_endpoint": "completed_detection", "quality_endpoint": "completed_detection",
    }
    measured = {**expected, "setup_id": SETUP, "measurement_endpoint": "raw_model_outputs", "task": "detection"}
    _, errors = _bind_results_to_required_scope_v2796([expected], [measured])
    assert errors and errors[0]["error_class"] == "scope_identity_physical_mismatch"
    assert expected["measurement_endpoint"] == "completed_detection"


def _placeholder_filter():
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text()
    node = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_is_non_actionable_missing"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "missing_plan_filter", "exec"), namespace)
    return namespace[node.name]


def _empty_plan_row(backend: str = "tensorrt") -> dict:
    # Exact shape of both extra missing_results in the actual v2.79.26 D pack.
    return {
        "schema": "onnx-splitpoint/planned-benchmark-result",
        "schema_version": 1, "model_id": "yolo11l", "backend": backend,
        "case_id": "", "split_index": "", "run_id": "", "variant": "split",
        "candidate_plan_join_status": "candidate_missing",
        "candidate_plan_join_complete": False, "error_class": "missing_artifact",
        "status": "pending_benchmark_execution", "total_latency_ms": None,
    }


def test_identityless_trt_and_hybrid_plan_rows_are_not_extra_missing_measurements() -> None:
    is_placeholder = _placeholder_filter()
    assert is_placeholder(_empty_plan_row("tensorrt")) is True
    assert is_placeholder(_empty_plan_row("deepx_m1_to_tensorrt")) is True


@pytest.mark.parametrize("identity", [
    {"case_id": "b003"}, {"split_index": 0}, {"run_id": "ort_tensorrt"},
    {"variant": "full"}, {"requested": True},
    {"schema": "onnx-splitpoint/required-profile-measurement"},
    {"logical_identity_sha256": "recorded-required-identity"},
])
def test_real_required_missing_rows_survive_placeholder_filter(identity: dict) -> None:
    assert _placeholder_filter()({**_empty_plan_row(), **identity}) is False
