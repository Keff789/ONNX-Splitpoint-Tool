from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "verify_native_full_rows.py"
    spec = importlib.util.spec_from_file_location(
        "v27513_verify_native_full_rows", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODELS = ("resnet50", "yolo26s", "yolov7_paper")
SETUPS = (
    ("orin_nx_hailo8_01", "hailo8", "native_full_hailo8"),
    ("orin_nx_hailo10_01", "hailo10h", "native_full_hailo10h"),
    ("orin_nx_deepx_m1_01", "deepx", "native_full_deepx"),
)


def _identity(
    backend: str, model: str, setup: str, comparison: str,
) -> dict[str, str]:
    return {
        "backend": backend,
        "model": model,
        "case": "full",
        "setup_id": setup,
        "comparison_backend": comparison,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(
    tmp_path: Path,
    mutate: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None]
    | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    expected: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for setup, comparison, vendor_backend in SETUPS:
        for model in MODELS:
            for backend in (vendor_backend, "native_full_tensorrt"):
                identity = _identity(backend, model, setup, comparison)
                quality_index = len(validation)
                quality_identity = {
                    **identity,
                    "runtime_precision": "fp16",
                    "output_endpoint_id": (
                        "classification:classification_logits:"
                        + "a" * 64
                    ),
                }
                expected.append({
                    **identity,
                    "execution_mode": "native_full_baseline",
                })
                validation.append({
                    **identity,
                    "runtime_precision_identity": "fp16",
                    "task": "classification",
                    "stage": "classification_logits",
                    "contract_family": "classification_logits",
                    "endpoint_contract_complete": True,
                    "endpoint_contract_hash": "a" * 64,
                    "output_endpoint_id": quality_identity["output_endpoint_id"],
                    "output_endpoint_attestation": {
                        "attested": True,
                        "status": "passed",
                        "task": "classification",
                        "stage": "classification_logits",
                        "endpoint": "classification_logits",
                        "endpoint_contract_hash": "a" * 64,
                        "output_endpoint_id": quality_identity[
                            "output_endpoint_id"
                        ],
                    },
                    "quality_identity": quality_identity,
                    "precision_quality_binding_verified": True,
                    "semantic_available": True,
                    "semantic_ok": True,
                    "semantic_validation_status": "passed",
                    "semantic_input_binding_status": "passed",
                    "self_reference_available": True,
                    "self_reference_ok": True,
                    "self_reference_input_manifest_kind": (
                        "native_full_input_manifest"
                    ),
                    "self_reference_diagnosis": (
                        "native_semantic_matches_full_self_reference"
                    ),
                    "self_reference_reason": "",
                    "strict_tensor_ok": True,
                })
                performance.append({
                    **identity,
                    "execution_mode": "native_full_baseline",
                    "ok": True,
                    "status": "ok",
                    "fps_makespan": 100.0,
                    "returncode": 0,
                    "timed_out": False,
                    "runtime_precision_identity": "fp16",
                    "task": "classification",
                    "stage": "classification_logits",
                    "contract_family": "classification_logits",
                    "endpoint_contract_complete": True,
                    "endpoint_contract_hash": "a" * 64,
                    "output_endpoint_id": quality_identity["output_endpoint_id"],
                    "output_endpoint_attestation": {
                        "attested": True,
                        "status": "passed",
                        "task": "classification",
                        "stage": "classification_logits",
                        "endpoint": "classification_logits",
                        "endpoint_contract_hash": "a" * 64,
                        "output_endpoint_id": quality_identity[
                            "output_endpoint_id"
                        ],
                    },
                    "repetition_count_requested": 1,
                    "repetition_count_attempted": 1,
                    "repetition_count_valid": 1,
                    "repetition_status": "complete",
                    "repetition_independence_verified": True,
                    "repetition_runtime_scope": "fresh_process_per_repetition",
                    "repetition_records": [{
                        "repetition_index": 1,
                        "runtime_instance_id": (
                            f"fresh_process:{backend}:{model}:{setup}"
                        ),
                        "ok": True,
                        "status": "ok",
                        "returncode": 0,
                        "timed_out": False,
                        "fps_makespan": 100.0,
                        "completed_work_units": 100,
                    }],
                    "quality_identity": quality_identity,
                    "quality_match_status": "exact_identity_match",
                    "quality_match_count": 1,
                    "quality_row_index": quality_index,
                    "precision_quality_binding_verified": True,
                    # A diagnostic Smoke observation may be negative while its
                    # provenance binding is still complete.
                    "task_quality_observation_valid": False,
                    "quality_claim_result_verified": False,
                })
    if mutate is not None:
        mutate(performance, validation)

    reports = run_dir / "reports"
    _write_json(reports / "native_expected_matrix.json", {
        "schema": "onnx-splitpoint/native-expected-matrix",
        "schema_version": 2,
        "expected_row_count": len(expected),
        "present_expected_rows": expected,
        "missing_expected_rows": [],
    })
    _write_json(reports / "native_producer_combined_summary.json", {
        "schema": "onnx-splitpoint/native-producer-combined-summary",
        "schema_version": 10,
        "row_count": len(performance),
        "rows": performance,
    })
    _write_json(
        reports / "native_validation" / "native_producer_validation_summary.json",
        {
            "schema": "onnx-splitpoint/native-producer-validation-summary",
            "schema_version": 10,
            "row_count": len(validation),
            "rows": validation,
        },
    )
    return run_dir


def _find(result: dict[str, Any], backend: str, model: str) -> dict[str, Any]:
    return next(
        row for row in result["rows"]
        if row["identity"]["backend"] == backend
        and row["identity"]["model"] == model
    )


def test_vendor_and_all_scopes_derive_nine_and_eighteen_clean_lines(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    run_dir = _make_run(tmp_path)

    vendor = checker.verify_run(run_dir, scope="vendor")
    all_full = checker.verify_run(run_dir, scope="all")

    assert vendor["ok"] is True
    assert vendor["expected_line_count"] == 9
    assert vendor["passed_line_count"] == 9
    assert all_full["ok"] is True
    assert all_full["expected_line_count"] == 18
    assert all_full["passed_line_count"] == 18
    assert all(row["status"] == "PASS" for row in all_full["rows"])


@pytest.mark.parametrize(
    ("scope", "backend"),
    (("vendor", "native_full_hailo8"), ("all", "native_full_tensorrt")),
)
def test_scope_rejects_a_reduced_expected_full_matrix(
    tmp_path: Path, scope: str, backend: str,
) -> None:
    checker = _load_checker()
    run_dir = _make_run(tmp_path)
    matrix_path = run_dir / "reports" / "native_expected_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = matrix["present_expected_rows"]
    removed = next(
        row for row in rows
        if row["backend"] == backend and row["model"] == "resnet50"
    )
    rows.remove(removed)
    matrix["expected_row_count"] = len(rows)
    _write_json(matrix_path, matrix)

    with pytest.raises(
        checker.VerificationInputError,
        match=rf"expected_{scope}_full_matrix_not_exact:required=",
    ):
        checker.verify_run(run_dir, scope=scope)


def test_vendor_scope_rejects_an_additional_expected_identity(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    run_dir = _make_run(tmp_path)
    matrix_path = run_dir / "reports" / "native_expected_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = matrix["present_expected_rows"]
    rows.append(_identity(
        "native_full_hailo8", "unexpected_model",
        "orin_nx_hailo8_01", "hailo8",
    ))
    matrix["expected_row_count"] = len(rows)
    _write_json(matrix_path, matrix)

    with pytest.raises(
        checker.VerificationInputError,
        match="expected_vendor_full_matrix_not_exact:.*unexpected=",
    ):
        checker.verify_run(run_dir, scope="vendor")


def test_vendor_scope_rejects_a_substituted_expected_identity(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    run_dir = _make_run(tmp_path)
    matrix_path = run_dir / "reports" / "native_expected_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    row = next(
        item for item in matrix["present_expected_rows"]
        if item["backend"] == "native_full_deepx"
        and item["model"] == "yolov7_paper"
    )
    row["setup_id"] = "wrong_deepx_setup"
    _write_json(matrix_path, matrix)

    with pytest.raises(
        checker.VerificationInputError,
        match="expected_vendor_full_matrix_not_exact:.*missing=.*unexpected=",
    ):
        checker.verify_run(run_dir, scope="vendor")


def test_runtime_failure_is_primary_and_keeps_concrete_child_error(
    tmp_path: Path,
) -> None:
    def mutate(performance, _validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "yolo26s"
        ))
        row.update({
            "ok": False,
            "status": "partial_repetitions",
            "fps_makespan": None,
            "failure_reason": "native_full_repetition_set_incomplete",
            "repetition_count_valid": 0,
            "repetition_status": "partial",
            "repetition_independence_verified": False,
            "repetition_records": [{
                "repetition_index": 1,
                "ok": False,
                "status": "failed",
                "completed_work_units": 0,
                "returncode": 4,
                "timed_out": False,
                "error": "authoritative endpoint missing",
                "stderr_tail": "child traceback",
            }],
        })

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo8", "yolo26s")

    assert result["ok"] is False
    assert row["status"] == "RUNTIME_FAIL"
    assert row["child_failure"]["failed_repetition"]["error"] == (
        "authoritative endpoint missing"
    )


def test_missing_diagnostic_policy_hash_does_not_mask_deepx_engine_failure(
    tmp_path: Path,
) -> None:
    error = "FileNotFoundError: Native TRT part2 engine not found"

    def mutate(performance, _validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_deepx"
            and item["model"] == "yolov7_paper"
        ))
        assert "runtime_quality_gate_policy_sha256" not in row
        row.update({
            "ok": False,
            "status": "failed",
            "fps_makespan": None,
            "failure_reason": error,
            "returncode": 1,
            "repetition_count_valid": 0,
            "repetition_status": "failed",
            "repetition_independence_verified": False,
            "repetition_records": [{
                "repetition_index": 1,
                "ok": False,
                "status": "failed",
                "completed_work_units": 0,
                "returncode": 1,
                "timed_out": False,
                "error": error,
                "stderr_tail": error,
            }],
        })

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_deepx", "yolov7_paper")

    assert result["ok"] is False
    assert row["status"] == "RUNTIME_FAIL"
    assert row["child_failure"]["failed_repetition"]["error"] == error


def test_repetition_count_mismatch_is_a_distinct_failure(tmp_path: Path) -> None:
    def mutate(performance, _validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_deepx"
            and item["model"] == "resnet50"
        ))
        row["repetition_count_attempted"] = 0

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_deepx", "resnet50")
    assert row["status"] == "REPETITION_FAIL"
    assert "repetition_attempted_mismatch" in row["issues"]


@pytest.mark.parametrize("field", ("returncode", "timed_out"))
def test_missing_top_level_execution_fields_fail_closed(
    tmp_path: Path, field: str,
) -> None:
    def mutate(performance, _validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "resnet50"
        ))
        row.pop(field)

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo8", "resnet50")

    assert row["status"] == "RUNTIME_FAIL"
    assert any(issue.startswith(f"runtime_{field}") for issue in row["issues"])


@pytest.mark.parametrize("field", ("returncode", "timed_out"))
def test_missing_repetition_execution_fields_fail_closed(
    tmp_path: Path, field: str,
) -> None:
    def mutate(performance, _validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "resnet50"
        ))
        row["repetition_records"][0].pop(field)

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo8", "resnet50")

    assert row["status"] == "REPETITION_FAIL"
    expected_issue = (
        "repetition_1_returncode_nonzero_or_invalid"
        if field == "returncode"
        else "repetition_1_timed_out_or_invalid"
    )
    assert expected_issue in row["issues"]


def test_endpoint_axis_mismatch_stays_failed_and_is_explained(
    tmp_path: Path,
) -> None:
    def mutate(performance, validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo10h"
            and item["model"] == "yolov7_paper"
        ))
        quality = validation[row["quality_row_index"]]
        quality.update({
            "task": "detection",
            "stage": "raw_head",
            "contract_family": "raw_head",
            "endpoint_contract_hash": "b" * 64,
            "output_endpoint_id": "detection:raw_head:" + "b" * 64,
            "output_endpoint_attestation": {
                "attested": True,
                "status": "passed",
                "task": "detection",
                "stage": "raw_head",
                "endpoint": "raw_head",
                "endpoint_contract_hash": "b" * 64,
                "output_endpoint_id": "detection:raw_head:" + "b" * 64,
            },
        })
        quality["quality_identity"] = {
            **quality["quality_identity"],
            "output_endpoint_id": (
                "detection:raw_head:" + "b" * 64
            ),
        }
        row.update({
            "quality_match_status": "no_exact_identity_match",
            "quality_match_count": 0,
            "quality_row_index": None,
            "precision_quality_binding_verified": False,
        })

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo10h", "yolov7_paper")
    assert row["status"] == "QUALITY_JOIN_FAIL"
    assert row["quality_join"]["binding_verified"] is False
    candidate = row["quality_join"]["nearest_candidates"][0]
    assert list(candidate["differing_axes"]) == ["output_endpoint_id"]


def test_runtime_precision_axis_mismatch_stays_failed_and_is_explained(
    tmp_path: Path,
) -> None:
    def mutate(performance, validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "resnet50"
        ))
        quality = validation[row["quality_row_index"]]
        quality["runtime_precision_identity"] = "int8"
        quality["quality_identity"] = {
            **quality["quality_identity"], "runtime_precision": "int8",
        }
        row.update({
            "quality_match_status": "no_exact_identity_match",
            "quality_match_count": 0,
            "quality_row_index": None,
            "precision_quality_binding_verified": False,
        })

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo8", "resnet50")
    assert row["status"] == "QUALITY_JOIN_FAIL"
    candidate = row["quality_join"]["nearest_candidates"][0]
    assert list(candidate["differing_axes"]) == ["runtime_precision"]


def test_declared_exact_join_is_recomputed_from_selected_quality_row(
    tmp_path: Path,
) -> None:
    def mutate(performance, validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "resnet50"
        ))
        quality = validation[row["quality_row_index"]]
        quality["runtime_precision_identity"] = "int8"
        # Keep the stored result green to prove that the verifier recomputes
        # the seven-axis identity rather than trusting these aliases.
        row["quality_match_status"] = "exact_identity_match"
        row["quality_match_count"] = 1
        row["precision_quality_binding_verified"] = True

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo8", "resnet50")

    assert row["status"] == "QUALITY_JOIN_FAIL"
    assert "quality_row_exact_identity_mismatch" in row["issues"]
    assert "quality_row_identity_projection_drift" in row["issues"]


def test_terminal_verifier_rejects_unavailable_full_semantics(
    tmp_path: Path,
) -> None:
    def mutate(performance, validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "yolo26s"
        ))
        quality = validation[row["quality_row_index"]]
        quality.update({
            "semantic_available": False,
            "semantic_ok": "unavailable",
            "semantic_validation_status": "unavailable",
            "semantic_input_binding_status": "not_applicable",
            "self_reference_available": False,
            "self_reference_ok": None,
            "self_reference_input_manifest_kind": "",
            "self_reference_reason": (
                "native_full_input_manifest_missing_or_invalid"
            ),
        })

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_hailo8", "yolo26s")

    assert result["ok"] is False
    assert row["status"] == "SEMANTIC_FAIL"
    assert "semantic_evidence_not_available" in row["issues"]
    assert row["semantic"]["reason"] == (
        "native_full_input_manifest_missing_or_invalid"
    )


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    (
        ("duplicate_runtime", "repetition_runtime_instance_ids_not_unique"),
        ("missing_fps", "repetition_1_fps_not_positive_finite"),
        ("unequal_work", "repetition_completed_work_units_mismatch"),
    ),
)
def test_repetition_evidence_is_independently_recomputed(
    tmp_path: Path, mutation: str, expected_issue: str,
) -> None:
    def mutate(performance, _validation):
        row = next(item for item in performance if (
            item["backend"] == "native_full_deepx"
            and item["model"] == "resnet50"
        ))
        first = row["repetition_records"][0]
        second = dict(first, repetition_index=2)
        row.update({
            "repetition_count_requested": 2,
            "repetition_count_attempted": 2,
            "repetition_count_valid": 2,
            "repetition_records": [first, second],
        })
        if mutation == "duplicate_runtime":
            pass
        elif mutation == "missing_fps":
            first.pop("fps_makespan")
            second["runtime_instance_id"] += ":second"
        elif mutation == "unequal_work":
            second["runtime_instance_id"] += ":second"
            second["completed_work_units"] = 99

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    row = _find(result, "native_full_deepx", "resnet50")

    assert row["status"] == "REPETITION_FAIL"
    assert expected_issue in row["issues"]


def test_missing_and_duplicate_performance_rows_are_distinct(tmp_path: Path) -> None:
    def mutate(performance, _validation):
        missing = next(item for item in performance if (
            item["backend"] == "native_full_hailo8"
            and item["model"] == "resnet50"
        ))
        performance.remove(missing)
        duplicate = next(item for item in performance if (
            item["backend"] == "native_full_hailo10h"
            and item["model"] == "resnet50"
        ))
        performance.append(dict(duplicate))

    checker = _load_checker()
    result = checker.verify_run(_make_run(tmp_path, mutate), scope="vendor")
    assert _find(result, "native_full_hailo8", "resnet50")["status"] == "MISSING"
    assert _find(result, "native_full_hailo10h", "resnet50")["status"] == "DUPLICATE"


def test_cli_exit_codes_distinguish_line_failure_and_input_error(
    tmp_path: Path, capsys,
) -> None:
    checker = _load_checker()
    good_run = _make_run(tmp_path / "good")
    assert checker.main([
        "--run-dir", str(good_run), "--scope", "vendor", "--format", "json",
    ]) == 0
    capsys.readouterr()

    def mutate(performance, _validation):
        performance[0]["precision_quality_binding_verified"] = False

    failed_run = _make_run(tmp_path / "failed", mutate)
    assert checker.main([
        "--run-dir", str(failed_run), "--scope", "vendor", "--format", "json",
    ]) == 2
    capsys.readouterr()

    assert checker.main([
        "--run-dir", str(tmp_path / "absent"), "--format", "json",
    ]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "input_error"
