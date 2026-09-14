from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script(root: Path):
    path = root / "scripts" / "reconcile_existing_run_v2792.py"
    spec = importlib.util.spec_from_file_location("v2792_reconciler_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimum_run(
    run: Path, *, rows: list[dict], required: list[dict],
    results: list[dict],
) -> None:
    model = run / "models" / "m" / "benchmark_results"
    quality = run / "quality_management"
    model.mkdir(parents=True)
    quality.mkdir(parents=True)
    (model / "normalized_results.json").write_text(
        json.dumps({"results": rows}), encoding="utf-8",
    )
    (model / "required_profile_matrix.json").write_text(
        json.dumps({"required_results": required}), encoding="utf-8",
    )
    (quality / "central_quality_summary.json").write_text(
        json.dumps({"results": results}), encoding="utf-8",
    )
    (run / "effective_execution_plan.json").write_text(
        json.dumps({"effective_generic_run_ids": []}), encoding="utf-8",
    )
    (run / "evaluation_workflow.log").write_text("finished\n", encoding="utf-8")


def _historical_v2784_shape() -> tuple[list[dict], list[dict], list[dict]]:
    """Generate the real seven-model denominator shape without archived data."""

    rows: list[dict] = []
    required: list[dict] = []
    results: list[dict] = []
    for index in range(386):
        case_id = f"b{index:04d}"
        request_sha = f"{index + 1:064x}"
        requirement = {
            "schema": "onnx-splitpoint/required-profile-measurement",
            "schema_version": 1,
            "model_id": "m", "case_id": case_id,
            "run_id": "deepx_m1_to_tensorrt",
            "backend": "deepx_m1_to_tensorrt", "variant": "split",
        }
        required.append(requirement)
        base = {
            "model_id": "m", "case_id": case_id,
            "run_id": "deepx_m1_to_tensorrt",
            "quality_source_run_id": "deepx_m1_to_tensorrt",
            "backend": "deepx_m1_to_tensorrt", "variant": "split",
            "primary_variant": "composed", "quality_source_variant": "composed",
            "task": "detection", "measurement_endpoint": "completed_detection",
            "source_request_sha256": request_sha, "status": "completed",
        }
        rows.extend([
            {**base, "setup_id": "", "source_path": f"mirror/{case_id}.json"},
            {**base, "setup_id": "deepx_setup", "source_path": f"direct/{case_id}.json"},
        ])
        # The historical producer used this alias and omitted setup_id.
        results.append({
            "model_id": "m", "case_id": case_id,
            "source_run_id": "deepx_to_trt", "variant": "composed",
            "source_request_sha256": request_sha,
            "status": "completed", "technical_status": "completed",
            "decision": "pass",
        })

    technical_gate = {
        "part2": {
            "status": "technical_validation_only",
            "decision": "not_applicable",
            "reason": "part1_part2_are_interface_validation_not_end_to_end_quality",
        },
    }
    for index in range(163):
        case_id = f"p{index:04d}"
        required.append({
            "schema": "onnx-splitpoint/required-profile-measurement",
            "schema_version": 1,
            "model_id": "m", "case_id": case_id,
            "run_id": "hailo10_to_trt",
            "backend": "hailo10_to_tensorrt", "variant": "split",
        })
        base = {
            "model_id": "m", "case_id": case_id,
            "run_id": "hailo10_to_trt", "quality_source_run_id": "hailo10_to_trt",
            "backend": "hailo10_to_tensorrt", "variant": "split",
            "primary_variant": "part2", "quality_source_variant": "part2",
            "task": "detection", "runtime_ok": True, "status": "completed",
            "task_quality_gates_by_variant": technical_gate,
        }
        # These old technical rows had no request SHA; the sole direct row is
        # selected for the schema-v1 identity without blessing the other row as
        # a proven mirror.
        rows.extend([
            {**base, "setup_id": "", "source_path": f"legacy_mirror/{case_id}.json"},
            {**base, "setup_id": "hailo10_setup", "source_path": f"legacy_direct/{case_id}.json"},
        ])

    required.append({
        "schema": "onnx-splitpoint/required-profile-measurement",
        "schema_version": 1,
        "model_id": "m", "case_id": "blocked", "run_id": "deepx_m1_full",
        "backend": "deepx_m1", "variant": "full",
    })
    rows.append({
        "model_id": "m", "case_id": "blocked", "run_id": "deepx_m1_full",
        "source_run_id": "deepx_m1_full", "backend": "deepx_m1",
        "variant": "full", "setup_id": "deepx_setup",
        "measurement_endpoint": "completed_detection",
        "status": "runtime_failed", "runtime_ok": False,
        "source_path": "direct/blocked.json",
    })
    required.append({
        "schema": "onnx-splitpoint/required-profile-measurement",
        "schema_version": 1,
        "model_id": "m", "case_id": "full", "run_id": "hailo10",
        "backend": "hailo10", "variant": "full",
    })
    for index in range(21):
        results.append({
            "model_id": "m", "case_id": "full",
            "source_run_id": "native_full_tensorrt", "variant": "full",
            "result_class": "summary_only", "status": "completed",
            "source_request_sha256": f"{10_000 + index:064x}",
        })
    return rows, required, results


def test_reconciler_is_read_only_and_separates_na_quality_companion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    module = _load_script(root)
    run = tmp_path / "run"
    model = run / "models" / "m" / "benchmark_results"
    quality = run / "quality_management"
    model.mkdir(parents=True)
    quality.mkdir(parents=True)
    req_a = "a" * 64
    req_b = "b" * 64
    rows = [
        {
            "model_id": "m", "case_id": "b001", "run_id": "hailo10_to_tensorrt",
            "source_run_id": "hailo10_to_tensorrt", "backend": "hailo10_to_tensorrt",
            "variant": "composed", "task": "detection", "setup_id": "s1",
            "source_request_sha256": req_a, "measurement_endpoint": "p2_output",
            "technical_validation_only": True, "status": "completed",
        },
        {
            "model_id": "m", "case_id": "full", "run_id": "deepx_m1_full",
            "source_run_id": "deepx_m1_full", "backend": "deepx_m1",
            "variant": "full", "task": "detection", "setup_id": "s2",
            "source_request_sha256": req_b, "measurement_endpoint": "completed_detection",
            "status": "completed",
        },
    ]
    (model / "normalized_results.json").write_text(json.dumps({"results": rows}))
    required = [
        {"model_id": "m", "case_id": "b001", "run_id": "hailo10_to_tensorrt", "backend": "hailo10_to_tensorrt", "variant": "composed"},
        {"model_id": "m", "case_id": "full", "run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full"},
    ]
    (model / "required_profile_matrix.json").write_text(json.dumps({"required_results": required}))
    results = [
        {
            "model_id": "m", "case_id": "full", "source_run_id": "deepx_m1_full",
            "variant": "full", "setup_id": "s2", "source_request_sha256": req_b,
            "status": "completed", "technical_status": "completed", "decision": "pass",
        },
        {
            "model_id": "m", "case_id": "full", "source_run_id": "native_full_tensorrt",
            "variant": "full", "result_class": "summary_only", "status": "completed",
        },
    ]
    (quality / "central_quality_summary.json").write_text(json.dumps({"results": results}))
    (run / "effective_execution_plan.json").write_text(json.dumps({"effective_generic_run_ids": []}))
    (run / "evaluation_workflow.log").write_text("finished\n")
    out = tmp_path / "out"
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    result = module.reconcile(run, out)
    after = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert before == after
    assert result["coverage"]["matrix_required"] == 2
    assert result["coverage"]["quality_not_applicable"] == 1
    assert result["coverage"]["quality_completed"] == 1
    assert result["coverage"]["companions"] == 1
    assert result["coverage"]["unmatched"] == 0
    assert (out / "evidence_reconciliation_v2792_bundle.zip").is_file()


def test_historical_v2784_shape_reconstructs_exact_denominators(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    module = _load_script(root)
    run = tmp_path / "legacy_run"
    rows, required, results = _historical_v2784_shape()
    _write_minimum_run(run, rows=rows, required=required, results=results)

    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    result = module.reconcile(run, tmp_path / "legacy_out")
    after = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert before == after
    expected = {
        "matrix_required": 551,
        "matrix_present": 550,
        "quality_applicable": 388,
        "quality_completed": 386,
        "quality_blocked": 1,
        "quality_not_applicable": 163,
        "quality_missing": 1,
        "companions": 21,
        "unmatched": 0,
        "ambiguous": 0,
        "quality_generated": 407,
        "quality_primary_joined": 386,
        "legacy_part2_projection_count": 163,
    }
    for key, value in expected.items():
        assert result["coverage"][key] == value
    assert result["legacy_compatibility"]["historical_rows_imported_into_new_run"] is False


def test_fully_bound_new_run_has_no_legacy_na_or_join_gap(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    module = _load_script(root)
    run = tmp_path / "bound_run"
    rows: list[dict] = []
    required: list[dict] = []
    results: list[dict] = []
    for index in range(4):
        case_id = f"b{index:03d}"
        request_sha = f"{20_000 + index:064x}"
        required.append({
            "schema": "onnx-splitpoint/required-profile-measurement",
            "schema_version": 2,
            "model_id": "m", "case_id": case_id,
            "run_id": "hailo8_to_trt", "backend": "hailo8_to_tensorrt",
            "variant": "composed", "setup_id": "hailo8_setup",
            "measurement_endpoint": "completed_detection",
            "quality_endpoint": "completed_detection",
            "quality_applicability": "applicable",
            "logical_identity_sha256": f"{30_000 + index:064x}",
        })
        rows.append({
            "model_id": "m", "case_id": case_id,
            "run_id": "hailo8_to_trt", "source_run_id": "hailo8_to_trt",
            "backend": "hailo8_to_tensorrt", "variant": "composed",
            "setup_id": "hailo8_setup", "task": "detection",
            "measurement_endpoint": "completed_detection",
            "source_request_sha256": request_sha, "status": "completed",
            "source_path": f"bound/{case_id}.json",
        })
        results.append({
            "model_id": "m", "case_id": case_id,
            "source_run_id": "hailo8_to_trt", "variant": "composed",
            "setup_id": "hailo8_setup", "source_request_sha256": request_sha,
            "status": "completed", "technical_status": "completed",
            "decision": "pass",
        })
    _write_minimum_run(run, rows=rows, required=required, results=results)
    result = module.reconcile(run, tmp_path / "bound_out")
    coverage = result["coverage"]
    assert coverage["matrix_required"] == coverage["matrix_present"] == 4
    assert coverage["quality_applicable"] == coverage["quality_completed"] == 4
    for key in (
        "quality_not_applicable", "quality_missing", "quality_blocked",
        "unmatched", "ambiguous", "legacy_part2_projection_count",
    ):
        assert coverage[key] == 0
    assert result["legacy_compatibility"]["mode"] == "not_used"
