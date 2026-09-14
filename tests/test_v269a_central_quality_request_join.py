from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.execution_binding import _copy_remote_result_files
from onnx_splitpoint_tool.workflow.results import _apply_quality_source_identity_v265


MODEL = "resnet50"
CASE = "b052"


def _pending_gate(variant: str, request_sha256: str) -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/task-quality-gate",
        "schema_version": 2,
        "task": "classification",
        "variant": variant,
        "status": "pending_central_evaluation",
        "decision": "pending_central_evaluation",
        "quality_input_request": {
            "schema": "onnx-splitpoint/central-quality-evaluation-request",
            "schema_version": 1,
            "task": "classification",
            "variant": variant,
            "status": "pending_central_evaluation",
            "request": {
                "path": f"/remote/results/{CASE}/results_hailo10_to_trt/"
                f"task_quality_inputs/{variant}_request.json",
                "sha256": request_sha256,
            },
        },
    }


def _row(
    *,
    run_id: str,
    variant: str = "split",
    composed_sha256: str,
    full_sha256: str | None = None,
) -> dict[str, Any]:
    gates = {"composed": _pending_gate("composed", composed_sha256)}
    if full_sha256:
        gates["full"] = _pending_gate("full", full_sha256)
    primary = "full" if variant == "full" else "composed"
    return {
        "model_id": MODEL,
        "case_id": CASE,
        "run_id": run_id,
        "source_tag": f"{run_id}_auto",
        "quality_source_run_id": run_id,
        # This reproduces the overnight run: the optional >2 MiB validation
        # report was omitted, so no setup could be recovered from source_paths.
        "quality_source_setup_ids": [],
        "variant": variant,
        "primary_variant": primary,
        "quality_source_variant": primary,
        "task": "classification",
        "task_quality_gates_by_variant": gates,
        "task_quality_gate": gates[primary],
        "task_quality_policy": {"dataset_tier": "screening"},
        "quality_evaluation_pending": True,
        "technical_status": "completed",
    }


def _result(
    *,
    run_id: str,
    setup_id: str,
    variant: str,
    request_sha256: str,
) -> dict[str, Any]:
    identity = {
        "schema": "onnx-splitpoint/central-quality-request-identity",
        "schema_version": 2,
        "model_id": MODEL,
        "case_id": CASE,
        "source_run_id": run_id,
        "setup_id": setup_id,
        "variant": variant,
        "task": "classification",
        "source_request_sha256": f"sha256:{request_sha256}",
    }
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "schema_version": 1,
        "model_id": MODEL,
        "case_id": CASE,
        "run_id": run_id,
        "source_run_id": run_id,
        "source_setup_id": setup_id,
        "variant": variant,
        "task": "classification",
        "source_request": f"quality_inputs/{setup_id}/results/{CASE}/results_{run_id}/"
        f"task_quality_inputs/{variant}_request.json",
        "source_request_sha256": request_sha256,
        "request_identity": identity,
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "n": 500,
        "primary": {
            "metric": "top1_accuracy",
            "candidate": 0.8,
            "reference": 0.8,
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "margin": 0.01,
        },
    }


def _runner_with_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> EvaluationWorkflowRunner:
    path = tmp_path / f"models/{MODEL}/benchmark_results/normalized_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"results": rows}), encoding="utf-8")
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "quality_gate": {"statistics": {"execution_location": "central_management", "workers": 4}}
    }
    return runner


def _merged_rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / f"models/{MODEL}/benchmark_results/normalized_results.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def test_request_hash_joins_without_optional_validation_report(tmp_path: Path) -> None:
    composed_sha = "1" * 64
    runner = _runner_with_rows(
        tmp_path,
        [_row(run_id="hailo10_to_trt", composed_sha256=composed_sha)],
    )

    merge = runner._merge_central_quality_results([
        _result(
            run_id="hailo10_to_trt",
            setup_id="orin_nx_hailo10_01",
            variant="composed",
            request_sha256=composed_sha,
        )
    ])

    assert merge["unmatched_result_count"] == 0
    assert merge["matched_primary_result_count"] == 1
    row = _merged_rows(tmp_path)[0]
    assert row["quality_evaluation_pending"] is False
    assert row["quality_source_setup_ids"] == ["orin_nx_hailo10_01"]
    identity = row["central_quality_request_identity"]
    assert identity["source_request_sha256"] == composed_sha
    assert identity["source_run_id"] == "hailo10h_to_trt"
    assert identity["setup_id"] == "orin_nx_hailo10_01"


def test_full_supplemental_request_hash_joins_without_diagnostic_path(tmp_path: Path) -> None:
    composed_sha = "2" * 64
    full_sha = "3" * 64
    runner = _runner_with_rows(
        tmp_path,
        [_row(run_id="ort_tensorrt", composed_sha256=composed_sha, full_sha256=full_sha)],
    )

    merge = runner._merge_central_quality_results([
        _result(
            run_id="ort_tensorrt",
            setup_id="orin_nx_deepx_m1_01",
            variant="full",
            request_sha256=full_sha,
        )
    ])

    assert merge["unmatched_result_count"] == 0
    assert merge["matched_supplemental_result_count"] == 1
    row = _merged_rows(tmp_path)[0]
    assert row["central_quality_supplemental_results"]["full"]["status"] == "completed"
    assert row["quality_request_identities_by_variant"]["full"]["source_request_sha256"] == full_sha


def test_model_scope_full_row_accepts_only_its_hash_bound_owner_case(
    tmp_path: Path,
) -> None:
    request_sha = "a" * 64
    row = _row(
        run_id="ort_tensorrt",
        variant="full",
        composed_sha256="b" * 64,
        full_sha256=request_sha,
    )
    row.update({
        "case_id": "full",
        "source_case_id": CASE,
        "full_source_case_ids": [CASE],
    })
    runner = _runner_with_rows(tmp_path, [row])

    merge = runner._merge_central_quality_results([
        _result(
            run_id="ort_tensorrt",
            setup_id="orin_nx_deepx_m1_01",
            variant="full",
            request_sha256=request_sha,
        )
    ])

    assert merge["unmatched_result_count"] == 0
    assert merge["matched_primary_result_count"] == 1
    merged = _merged_rows(tmp_path)[0]
    assert merged["case_id"] == "full"
    assert merged["quality_evaluation_pending"] is False


def test_model_scope_full_row_rejects_other_case_even_with_same_hash(
    tmp_path: Path,
) -> None:
    request_sha = "c" * 64
    row = _row(
        run_id="ort_tensorrt",
        variant="full",
        composed_sha256="d" * 64,
        full_sha256=request_sha,
    )
    row.update({
        "case_id": "full",
        "source_case_id": CASE,
        "full_source_case_ids": [CASE],
    })
    result = _result(
        run_id="ort_tensorrt",
        setup_id="orin_nx_deepx_m1_01",
        variant="full",
        request_sha256=request_sha,
    )
    result["case_id"] = "b999"
    result["request_identity"]["case_id"] = "b999"
    runner = _runner_with_rows(tmp_path, [row])

    merge = runner._merge_central_quality_results([result])

    assert merge["unmatched_result_count"] == 1
    assert merge["unmatched_results"][0]["join_status"] == "no_exact_row"


def test_hash_mismatch_and_ambiguous_exact_rows_fail_closed(tmp_path: Path) -> None:
    expected_sha = "4" * 64
    wrong_sha = "5" * 64
    runner = _runner_with_rows(
        tmp_path,
        [_row(run_id="hailo8_to_trt", composed_sha256=expected_sha)],
    )
    merge = runner._merge_central_quality_results([
        _result(
            run_id="hailo8_to_trt",
            setup_id="orin_nx_hailo8_01",
            variant="composed",
            request_sha256=wrong_sha,
        )
    ])
    assert merge["unmatched_result_count"] == 1
    assert merge["unmatched_results"][0]["join_status"] == "no_exact_row"

    duplicate_rows = [
        _row(run_id="hailo8_to_trt", composed_sha256=expected_sha),
        _row(run_id="hailo8_to_trt", composed_sha256=expected_sha),
    ]
    runner = _runner_with_rows(tmp_path / "ambiguous", duplicate_rows)
    merge = runner._merge_central_quality_results([
        _result(
            run_id="hailo8_to_trt",
            setup_id="orin_nx_hailo8_01",
            variant="composed",
            request_sha256=expected_sha,
        )
    ])
    assert merge["unmatched_result_count"] == 1
    assert merge["unmatched_results"][0]["join_status"] == "ambiguous"
    assert merge["unmatched_results"][0]["join_candidate_count"] == 2


def test_conflicting_top_level_and_nested_result_identity_fails_closed(tmp_path: Path) -> None:
    request_sha = "6" * 64
    result = _result(
        run_id="hailo10_to_trt",
        setup_id="orin_nx_hailo10_01",
        variant="composed",
        request_sha256=request_sha,
    )
    result["request_identity"]["setup_id"] = "different_setup"
    runner = _runner_with_rows(
        tmp_path,
        [_row(run_id="hailo10_to_trt", composed_sha256=request_sha)],
    )

    merge = runner._merge_central_quality_results([result])

    assert merge["unmatched_result_count"] == 1
    assert merge["unmatched_results"][0]["join_status"] == "invalid_result_identity"
    assert "conflicting_setup_id" in merge["unmatched_results"][0]["join_identity_errors"]


def test_request_identity_marked_invalid_by_path_manifest_check_fails_closed(tmp_path: Path) -> None:
    request_sha = "8" * 64
    result = _result(
        run_id="hailo10_to_trt",
        setup_id="orin_nx_hailo10_01",
        variant="composed",
        request_sha256=request_sha,
    )
    result["request_identity"].update({
        "identity_valid": False,
        "identity_errors": [
            "source_run_id_conflict:declared='hailo8_to_trt':path='hailo10_to_trt'"
        ],
    })
    runner = _runner_with_rows(
        tmp_path,
        [_row(run_id="hailo10_to_trt", composed_sha256=request_sha)],
    )

    merge = runner._merge_central_quality_results([result])

    assert merge["unmatched_result_count"] == 1
    unmatched = merge["unmatched_results"][0]
    assert unmatched["join_status"] == "invalid_result_identity"
    assert "invalid_embedded_request_identity" in unmatched["join_identity_errors"]


def test_normalization_retains_request_hash_without_validation_report() -> None:
    request_sha = "7" * 64
    row = _row(run_id="hailo10_to_trt", composed_sha256=request_sha)
    row.pop("quality_source_run_id")
    row.pop("quality_source_setup_ids")

    _apply_quality_source_identity_v265(row)

    identity = row["quality_request_identities_by_variant"]["composed"]
    assert identity["model_id"] == MODEL
    assert identity["case_id"] == CASE
    assert identity["source_run_id"] == "hailo10_to_trt"
    assert identity["setup_ids"] == []
    assert identity["source_request_sha256"] == request_sha


def test_large_diagnostic_is_optional_but_quality_inputs_are_canonical(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    results = remote / "results"
    quality = results / CASE / "results_hailo10_to_trt" / "task_quality_inputs"
    quality.mkdir(parents=True)
    (results / "benchmark_results_hailo10_to_trt.json").write_text(
        json.dumps([{"case_id": CASE, "run_id": "hailo10_to_trt", "status": "ok"}]),
        encoding="utf-8",
    )
    (results / CASE / "results_hailo10_to_trt" / "validation_report.json").write_text(
        json.dumps({"diagnostic": "x" * (2 * 1024 * 1024 + 1024)}),
        encoding="utf-8",
    )
    request_payload = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "task": "classification",
        "variant": "composed",
    }
    (quality / "composed_request.json").write_text(json.dumps(request_payload), encoding="utf-8")
    (quality / "composed_candidate.json").write_text(
        json.dumps({"records": ["y" * (2 * 1024 * 1024 + 1024)]}),
        encoding="utf-8",
    )

    destination = tmp_path / "collected"
    _copy_remote_result_files(remote, destination, flat_prefix="orin_nx_hailo10_01")

    copied_quality = destination / "quality_inputs/orin_nx_hailo10_01/results"
    assert list(copied_quality.rglob("composed_request.json"))
    candidates = list(copied_quality.rglob("composed_candidate.json"))
    assert candidates and candidates[0].stat().st_size > 2 * 1024 * 1024
    assert not list((destination / "remote_diagnostics").rglob("validation_report.json"))
    manifest = json.loads(
        (destination / "remote_diagnostics/orin_nx_hailo10_01/result_copy_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        item.get("kind") == "case_validation_report" and item.get("reason") == "too_large"
        for item in manifest["skipped"]
    )
