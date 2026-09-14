from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from onnx_splitpoint_tool.workflow.native_energy_preflight import (
    build_native_energy_preflight,
    native_energy_preflight_blocks_streaming,
)
from onnx_splitpoint_tool.native_command_contract import (
    seal_native_command_contract,
)
from onnx_splitpoint_tool.workflow.runner import (
    _native_energy_final_contract_requested_v61d,
)
from tests.test_energy_final_contract_regressions import (
    _with_runtime_contract,
    _write_model_hash_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "scripts" / "native_producer_energy_plan.py"
RUNNER = ROOT / "scripts" / "run_native_producer_energy_from_summary.py"
_SPLIT_TECHNICAL_CONTRACT = (
    ROOT / "tests" / "fixtures" / "v2725_hailo8_resume"
    / (
        "hailo8_to_trt__resnet50__b052__"
        "orin_nx_hailo8_01__8eb40a232207.command_contract.json"
    )
)

_CLAIM_FIELDS = {
    "claim_ok",
    "semantic_claim_ok",
    "claim_eligible",
    "performance_claim_eligible",
    "energy_claim_eligible",
    "eligible_for_energy_results_import",
    "eligible_for_scientific_claim",
}


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v271_screening_energy_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_nonclaimable(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _CLAIM_FIELDS:
                assert item is False, (key, item)
            _assert_nonclaimable(item)
    elif isinstance(value, list):
        for item in value:
            _assert_nonclaimable(item)


def _with_technical_runtime_contract(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Attach a real Part-2 proof to Split rows used by planner fixtures."""

    contracted = dict(_with_runtime_contract(row))
    if contracted.get("backend") != "hailo8_to_trt":
        return contracted
    template = json.loads(
        _SPLIT_TECHNICAL_CONTRACT.read_text(encoding="utf-8")
    )
    for field in (
        "backend", "model", "case", "precision", "setup_id",
        "comparison_backend",
    ):
        template[field] = contracted[field]
    contracted["native_command_contract"] = (
        seal_native_command_contract(template)
    )
    return contracted


def _write_pair_inputs(
    tmp_path: Path,
    *,
    claim_ok: bool,
    split_semantic_ok: bool = True,
) -> tuple[Path, Path]:
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    rows = [
        {
            "ok": True,
            "backend": "hailo8_to_trt",
            "model": "m",
            "case": "b1",
            "precision": "float32_layout_fp16",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "fps_makespan": 10,
        },
        {
            "ok": True,
            "backend": "native_full_hailo8",
            "model": "m",
            "case": "full",
            "precision": "p",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "fps_makespan": 8,
        },
        {
            "ok": True,
            "backend": "native_full_tensorrt",
            "model": "m",
            "case": "full",
            "precision": "p",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "fps_makespan": 12,
        },
    ]
    contracted_rows = [
        _with_technical_runtime_contract(row) for row in rows
    ]
    summary.write_text(
        json.dumps({"rows": contracted_rows}),
        encoding="utf-8",
    )
    validation_rows = []
    for row in contracted_rows:
        semantic_ok = (
            split_semantic_ok
            if row["backend"] == "hailo8_to_trt" else True
        )
        validation_rows.append({
            **row,
            "backend": row["backend"],
            "model": row["model"],
            "case": row["case"],
            "precision": row["precision"],
            "setup_id": row["setup_id"],
            "comparison_backend": row["comparison_backend"],
            "ok": True,
            "error": "",
            "task": "classification",
            "top1_match": True,
            "contract_consistent": True,
            "semantic_ok": semantic_ok,
            "claim_ok": claim_ok and semantic_ok,
            "central_quality_evidence_verified": True,
            "precision_quality_verified": True,
            "precision_quality_binding_verified": True,
            "task_quality_observation_valid": True,
            "accuracy_gate_pass": bool(claim_ok and semantic_ok),
            "quality_claim_result_verified": bool(
                claim_ok and semantic_ok
            ),
            "status": (
                "semantic_pass_screening_only"
                if semantic_ok else "semantic_threshold_miss"
            ),
        })
    validation.write_text(
        json.dumps({"rows": validation_rows}), encoding="utf-8",
    )
    return summary, validation


def _write_matrix_plan_inputs(
    tmp_path: Path,
    *,
    group_count: int,
) -> tuple[Path, Path]:
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    summary_rows: list[dict[str, Any]] = []
    for index in range(group_count):
        model = f"resnet50_{index:02d}"
        setup = f"h8-{index:02d}"
        summary_rows.extend([
            {
                "ok": True,
                "backend": "hailo8_to_trt",
                "model": model,
                "case": "b1",
                "precision": "float32_layout_fp16",
                "setup_id": setup,
                "comparison_backend": "hailo8",
                "fps_makespan": 10,
            },
            {
                "ok": True,
                "backend": "native_full_hailo8",
                "model": model,
                "case": "full",
                "precision": "p",
                "setup_id": setup,
                "comparison_backend": "hailo8",
                "fps_makespan": 8,
            },
            {
                "ok": True,
                "backend": "native_full_tensorrt",
                "model": model,
                "case": "full",
                "precision": "p",
                "setup_id": setup,
                "comparison_backend": "hailo8",
                "fps_makespan": 12,
            },
        ])
    summary_rows = [
        _with_technical_runtime_contract({
            **row,
            "task": "classification",
            "stage": "classification_logits",
            "contract_family": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "d" * 64,
            "output_endpoint_id": (
                f"classification:classification_logits:{'d' * 64}"
            ),
        })
        for row in summary_rows
    ]
    validation_rows = [
        {
            **row,
            "ok": True,
            "error": "",
            "task": "classification",
            "top1_match": True,
            "contract_consistent": True,
            "semantic_ok": True,
            "claim_ok": False,
            "central_quality_evidence_verified": True,
            "precision_quality_verified": True,
            "precision_quality_binding_verified": True,
            "task_quality_observation_valid": True,
            "accuracy_gate_pass": False,
            "quality_claim_result_verified": False,
        }
        for row in summary_rows
    ]
    summary.write_text(
        json.dumps({"rows": summary_rows}), encoding="utf-8",
    )
    validation.write_text(
        json.dumps({"rows": validation_rows}), encoding="utf-8",
    )
    return summary, validation


def _plan_command(
    summary: Path, validation: Path, out: Path, *extra: str,
) -> list[str]:
    return [
        sys.executable,
        str(PLAN),
        "--summary", str(summary),
        "--validation-summary", str(validation),
        "--out-dir", str(out),
        "--hailo8-ssh", "screening-host",
        "--duration-s", "1",
        *extra,
    ]


def test_default_and_screening_plans_retain_nonclaimable_semantic_pass(
    tmp_path: Path,
) -> None:
    summary, validation = _write_pair_inputs(tmp_path, claim_ok=False)
    final_out = tmp_path / "final-plan"
    final = subprocess.run(
        _plan_command(summary, validation, final_out),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert final.returncode == 0, (final.stdout, final.stderr)
    final_payload = json.loads(
        (final_out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )
    assert len(final_payload["rows"]) == 3
    assert final_payload["preflight_status"] == "passed"
    assert all(
        row["runtime_success"] is True
        and row["energy_command_preflight_ok"] is True
        and row["claim_ok"] is False
        and row["energy_claim_eligible"] is False
        for row in final_payload["rows"]
    )

    out = tmp_path / "screening-plan"
    completed = subprocess.run(
        _plan_command(summary, validation, out, "--screening-energy"),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )
    assert len(payload["rows"]) == 3
    assert {
        (
            row["backend"], row["model"], row["case"],
            row["setup_id"], row["comparison_backend"],
        )
        for row in payload["rows"]
    } == {
        (
            row["backend"], row["model"], row["case"],
            row["setup_id"], row["comparison_backend"],
        )
        for row in final_payload["rows"]
    }
    assert payload["screening_only"] is True
    assert payload["screening_energy"] is True
    assert payload["diagnostic_only"] is True
    assert payload["energy_evidence_tier"] == "screening"
    assert payload["energy_tier"] == "screening"
    assert payload["preflight_status"] == "passed"
    preflight = payload["preflight"]
    assert preflight["schema"] == (
        "onnx-splitpoint/native-energy-plan-preflight"
    )
    assert preflight["status"] == "passed"
    assert preflight["ok"] is True
    assert preflight["measurement_start_allowed"] is True
    assert preflight["measure_all_runtime_successful"] is True
    assert preflight["energy_plan_included_count"] == 3
    assert preflight["energy_plan_excluded_count"] == 0
    assert preflight["energy_plan_coverage_contract_valid"] is True
    assert preflight["runtime_measurement_admitted_rows"] == 3
    assert preflight["planned_rows"] == 3
    assert preflight["pair_count"] == 1
    assert preflight["blocked_reason"] == ""
    assert all(row["semantic_validation_ok"] is True for row in payload["rows"])
    assert all(
        row["energy_evidence_tier"] == "screening"
        and row["energy_tier"] == "screening"
        and row["diagnostic_only"] is True
        and "development_screening_energy_only"
        in row["scientific_claim_exclusion_reasons"]
        and "--diagnostic-only" in row["measure_command"]
        and "--claim-exclusion-reason development_screening_energy_only"
        in row["measure_command"]
        for row in payload["rows"]
    )
    _assert_nonclaimable(payload)


@pytest.fixture
def screening_18_of_54_plan(tmp_path: Path) -> dict[str, Any]:
    summary, validation = _write_matrix_plan_inputs(
        tmp_path, group_count=6,
    )
    present_rows = json.loads(
        summary.read_text(encoding="utf-8")
    )["rows"]
    failed_rows = [
        {
            "backend": "hailo8_to_trt",
            "model": f"missing_{index:02d}",
            "case": f"b{index:03d}",
            "setup_id": f"h8-missing-{index:02d}",
            "comparison_backend": "hailo8",
            "precision": "p",
            "actual_ok": False,
            "actual_status": "failed",
            "failure_reason": "fixture_expected_runtime_missing",
        }
        for index in range(36)
    ]
    (tmp_path / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": 54,
            "present_expected_row_count": 54,
            "successful_expected_row_count": 18,
            "failed_expected_row_count": 36,
            "missing_expected_row_count": 0,
            "present_expected_rows": present_rows,
            "failed_expected_rows": failed_rows,
            "missing_expected_rows": [],
        }),
        encoding="utf-8",
    )
    out = tmp_path / "screening-18-of-54"
    completed = subprocess.run(
        _plan_command(
            summary, validation, out, "--screening-energy",
        ),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout, completed.stderr,
    )
    return json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )


def test_screening_plan_separates_planner_and_matrix_denominators(
    screening_18_of_54_plan: dict[str, Any],
) -> None:
    payload = screening_18_of_54_plan

    assert len(payload["rows"]) == 18
    assert payload["source_ok_rows"] == 18
    assert payload["energy_plan_included_count"] == 18
    assert payload["energy_plan_excluded_count"] == 36
    assert payload["energy_matrix_expected_count"] == 54
    assert payload["energy_matrix_coverage_fraction"] == pytest.approx(
        18 / 54,
    )
    assert payload["preflight"]["planned_rows"] == 18
    assert payload["preflight"]["energy_plan_included_count"] == 18
    assert payload["preflight"]["energy_matrix_expected_count"] == 54


def test_complete_screening_plan_has_zero_claim_eligible_rows(
    screening_18_of_54_plan: dict[str, Any],
) -> None:
    payload = screening_18_of_54_plan

    assert payload["preflight_status"] == "passed"
    assert payload["preflight"]["measurement_start_allowed"] is True
    assert payload["energy_plan_included_count"] == 18
    assert payload["energy_claim_eligible"] is False
    assert sum(
        row["energy_claim_eligible"] is True
        for row in payload["rows"]
    ) == 0
    assert all(
        row["energy_claim_eligible"] is False
        and row["claim_eligible"] is False
        for row in payload["rows"]
    )


@pytest.mark.parametrize(
    "matrix_payload",
    [None, "{not-json", json.dumps({"expected_row_count": 0})],
)
def test_final_all_split_requires_verified_native_expected_matrix(
    tmp_path: Path,
    matrix_payload: str | None,
) -> None:
    summary, validation = _write_pair_inputs(
        tmp_path, claim_ok=False,
    )
    if matrix_payload is not None:
        (tmp_path / "native_expected_matrix.json").write_text(
            matrix_payload, encoding="utf-8",
        )
    out = tmp_path / "final-all-split-invalid-matrix"

    completed = subprocess.run(
        _plan_command(
            summary,
            validation,
            out,
            "--final-all-split-energy",
        ),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    assert (
        "--final-all-split-energy requires a verified run-local "
        "native_expected_matrix.json"
    ) in completed.stderr
    assert not (out / "native_producer_energy_plan.json").exists()


def test_final_all_split_plans_exact_bound_accuracy_fail_as_nonclaim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_script("native_producer_energy_plan.py")
    model_map, model_map_sha, model_sha, _ = (
        _write_model_hash_evidence(
            tmp_path, model_id="resnet50",
        )
    )
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    expected_matrix = tmp_path / "native_expected_matrix.json"
    out = tmp_path / "final-bound-negative"
    central_source_sha = "1" * 64
    central_result_sha = "2" * 64
    central_selection_sha = "3" * 64
    endpoint_sha = "4" * 64
    quality_contract_sha = "5" * 64
    preprocessing_contract_sha = "6" * 64
    validation_dataset_sha = "7" * 64
    validation_image_ids_sha = "8" * 64
    validation_ground_truth_sha = "9" * 64
    quality_policy_sha = "a" * 64
    base_rows = [
        {
            "ok": True,
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b1",
            "precision": "float32_layout_fp16",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "fps_makespan": 10,
            "quality_evidence_verified": True,
            "performance_claim_eligible": False,
        },
        {
            "ok": True,
            "backend": "native_full_hailo8",
            "model": "resnet50",
            "case": "full",
            "precision": "p",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "fps_makespan": 8,
            "quality_evidence_verified": True,
            "performance_claim_eligible": True,
        },
        {
            "ok": True,
            "backend": "native_full_tensorrt",
            "model": "resnet50",
            "case": "full",
            "precision": "p",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "fps_makespan": 12,
            "quality_evidence_verified": True,
            "performance_claim_eligible": True,
        },
    ]
    summary_rows = [
        _with_technical_runtime_contract({
            **row,
            "task": "classification",
            "model_sha256": model_sha,
            "stage": "classification_logits",
            "contract_family": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_sha,
            "output_endpoint_id": (
                "classification:classification_logits:"
                f"{endpoint_sha}"
            ),
            "native_split_quality_source_request_sha256":
                central_source_sha,
            "native_split_quality_central_result_sha256":
                central_result_sha,
            "native_split_quality_selection_sha256":
                central_selection_sha,
            "source_request_sha256": central_source_sha,
        })
        for row in base_rows
    ]
    validation_rows = []
    for row in summary_rows:
        accuracy_pass = row["backend"] != "hailo8_to_trt"
        validation_rows.append({
            **row,
            "ok": True,
            "error": "",
            "task": "classification",
            "top1_match": True,
            "semantic_ok": True,
            "contract_consistent": True,
            "claim_ok": accuracy_pass,
            "central_quality_evidence_verified": True,
            "precision_quality_binding_verified": True,
            "precision_quality_verified": accuracy_pass,
            "task_quality_observation_valid": True,
            "accuracy_gate_pass": accuracy_pass,
            "quality_claim_result_verified": accuracy_pass,
            "runtime_precision_identity": row["precision"],
            "quality_contract_sha256": quality_contract_sha,
            "preprocessing_contract_sha256":
                preprocessing_contract_sha,
            "validation_dataset_sha256": validation_dataset_sha,
            "validation_dataset_image_ids_sha256":
                validation_image_ids_sha,
            "validation_dataset_ground_truth_sha256":
                validation_ground_truth_sha,
            "accuracy_gate_policy_sha256": quality_policy_sha,
            "task_quality_policy_sha256": quality_policy_sha,
            "runtime_quality_gate_policy_sha256": quality_policy_sha,
        })
    summary.write_text(
        json.dumps({"rows": summary_rows}), encoding="utf-8",
    )
    validation.write_text(
        json.dumps({"rows": validation_rows}), encoding="utf-8",
    )
    expected_matrix.write_text(
        json.dumps({
            "expected_row_count": 3,
            "present_expected_row_count": 3,
            "successful_expected_row_count": 3,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
            "present_expected_rows": summary_rows,
            "failed_expected_rows": [],
            "missing_expected_rows": [],
        }),
        encoding="utf-8",
    )

    authority = {
        "schema": "onnx-splitpoint/native-split-quality-authority",
        "schema_version": 1,
        "valid": True,
        "mode": "required",
        "native_split_quality_required": True,
        "workflow_version": "v2.72.0-test",
        "run_id": "test-run",
    }
    monkeypatch.setattr(
        planner,
        "resolve_native_split_quality_authority",
        lambda **_kwargs: dict(authority),
    )

    def apply_authority(
        row: dict[str, Any],
        _authority: Mapping[str, Any],
    ) -> dict[str, Any]:
        row.update({
            "native_split_quality_required": True,
            "native_split_quality_binding_required": True,
            "native_split_quality_authority": dict(authority),
            "native_split_quality_authority_valid": True,
            "native_split_quality_authority_mode": "required",
        })
        return row

    monkeypatch.setattr(
        planner,
        "apply_native_split_quality_authority",
        apply_authority,
    )
    monkeypatch.setattr(
        planner,
        "native_split_quality_required_for_row",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        planner,
        "_split_quality_energy_evidence",
        lambda *_args, **_kwargs: ({
            "native_split_quality_required": True,
            "native_split_energy_binding_valid": True,
            "native_split_energy_binding_status":
                "exact_test_binding_verified",
        }, "exact_test_binding_verified"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(planner.__file__),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo8-ssh", "host",
            "--duration-s", "1",
            "--model-hash-map", str(model_map),
            "--model-hash-map-sha256", model_map_sha,
            "--final-all-split-energy",
        ],
    )

    assert planner.main() == 0
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )
    split = next(
        row for row in payload["rows"]
        if row["backend"] == "hailo8_to_trt"
    )
    assert payload["final_all_split_energy_required"] is True
    assert payload["energy_plan_included_count"] == 3
    assert split["precision_quality_binding_verified"] is True
    assert split["accuracy_gate_pass"] is False
    assert split["screening_comparable"] is False
    assert split["claim_comparable"] is False
    assert split["energy_claim_eligible"] is False
    assert split["eligible_for_scientific_claim"] is False
    assert "accuracy_gate_failed" in (
        split["scientific_claim_exclusion_reasons"]
    )


def test_screening_plan_retains_negative_semantic_decision_as_annotation(
    tmp_path: Path,
) -> None:
    summary, validation = _write_pair_inputs(
        tmp_path, claim_ok=False, split_semantic_ok=False,
    )
    out = tmp_path / "screening-negative"
    completed = subprocess.run(
        _plan_command(summary, validation, out, "--screening-energy"),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )
    assert len(payload["rows"]) == 3
    assert payload["preflight_status"] == "passed"
    assert payload["preflight"]["measurement_start_allowed"] is True
    assert payload["energy_plan_excluded_count"] == 0
    split = next(
        row for row in payload["rows"]
        if row["backend"] == "hailo8_to_trt"
    )
    assert split["semantic_validation_ok"] is False
    assert split["claim_ok"] is False
    assert split["energy_claim_eligible"] is False
    _assert_nonclaimable(payload)


def test_screening_runner_dry_run_clamps_nested_results_and_carries_preflight(
    tmp_path: Path,
) -> None:
    summary, validation = _write_pair_inputs(tmp_path, claim_ok=False)
    out = tmp_path / "screening-runner"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo8-ssh", "screening-host",
            "--duration-s", "1",
            "--screening-energy",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8",
        )
    )
    assert report["status"] == "dry_run"
    assert report["preflight_status"] == "passed"
    assert report["screening_only"] is True
    assert report["diagnostic_only"] is True
    assert report["energy_evidence_tier"] == "screening"
    assert report["energy_tier"] == "screening"
    assert len(report["rows"]) == 3
    assert all(item["row"]["screening_only"] is True for item in report["rows"])
    _assert_nonclaimable(report)

    wrapper = _load_script("run_native_producer_energy_from_summary.py")
    nested = wrapper._clamp_screening_result({
        "row": {"claim_ok": True},
        "run": {
            "energy_claim_eligible": True,
            "energy_aggregate": {"eligible_for_scientific_claim": True},
        },
    })
    _assert_nonclaimable(nested)
    assert nested["run"]["energy_aggregate"]["diagnostic_only"] is True
    assert (
        nested["run"]["energy_aggregate"]["energy_evidence_tier"]
        == "screening"
    )


def test_empty_screening_plan_is_a_zero_start_preflight_block(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "empty-summary.json"
    validation = tmp_path / "empty-validation.json"
    out = tmp_path / "empty-screening"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": []}), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--screening-energy",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 3, (completed.stdout, completed.stderr)
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8",
        )
    )
    assert report["status"] == "blocked_no_runtime_constructible_rows"
    assert (
        report["preflight_status"]
        == "blocked_no_runtime_constructible_rows"
    )
    assert report["preflight"]["measurement_start_allowed"] is False
    assert report["started_measurement_count"] == 0
    assert report["strict_requested"] is False
    assert report["strict_failure"] is False
    _assert_nonclaimable(report)


def test_explicit_false_strict_wins_but_final_campaign_stays_fail_closed() -> None:
    standard = {
        "campaign": {"mode": "development"},
        "energy": {"strict": True},
        "execution_preset": {"id": "standard"},
    }
    assert _native_energy_final_contract_requested_v61d(
        standard, {"strict": False},
    ) is False
    assert _native_energy_final_contract_requested_v61d(
        standard, {"strict": "false"},
    ) is False

    final = {
        "campaign": {"mode": "final"},
        "energy": {"strict": False},
        "execution_preset": {"id": "final"},
    }
    assert _native_energy_final_contract_requested_v61d(
        final, {"strict": False},
    ) is True

    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    assert coordinator._configured_bool(
        ({"strict": False}, "strict"),
        ({"strict": True}, "strict"),
        default=True,
    ) is False


def test_screening_cli_rejects_final_only_runtime_contract_flags(
    tmp_path: Path,
) -> None:
    summary, validation = _write_pair_inputs(tmp_path, claim_ok=False)
    for script in (PLAN, RUNNER):
        out = tmp_path / f"conflict-{script.stem}"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--summary", str(summary),
                "--validation-summary", str(validation),
                "--out-dir", str(out),
                "--screening-energy",
                "--require-runtime-work-units",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 2
        assert "cannot be combined with Final-only" in completed.stderr


def test_energy_remote_mirrors_and_runner_strict_propagation_are_current() -> None:
    for name in (
        "native_producer_energy_plan.py",
        "run_native_producer_energy_from_summary.py",
        "run_evalrun_native_producer_variants.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
        ).read_bytes()

    source = (
        ROOT / "onnx_splitpoint_tool" / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    assert 'cmd.append("--screening-energy")' in source
    assert 'energy_state.get("strict_requested")' in source
    assert 'native_energy_state.get("strict_requested")' in source
    assert 'native_energy_state.get("strict")' not in source


def _complete_phase1_rows() -> list[dict[str, Any]]:
    return [
        {
            "execution_mode": "native_split",
            "backend_key": "hailo8",
            "backend": "hailo8_to_trt",
            "model": "m",
            "case": "b001",
        },
        {
            "execution_mode": "native_full_baseline",
            "backend_key": "hailo8",
            "backend": "native_full_hailo8",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "model": "m",
            "case": "full",
        },
        {
            "execution_mode": "native_full_baseline",
            "backend_key": "hailo8",
            "backend": "native_full_tensorrt",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "model": "m",
            "case": "full",
        },
    ]


def test_phase1_preflight_blocks_stream_callback_for_empty_plan() -> None:
    preflight = build_native_energy_preflight(
        expected_rows=[],
        energy_requested=True,
        energy_evidence_tier="screening",
        strict_requested=False,
        validation_requested=True,
        full_baselines_enabled=True,
    )
    calls: list[str] = []

    def stream() -> None:
        calls.append("called")

    if not native_energy_preflight_blocks_streaming(preflight):
        stream()

    assert calls == []
    assert preflight["theoretical_pair_count"] == 0
    assert preflight["plan_viable"] is False
    assert preflight["status"] == "blocked_structural_contradiction"
    assert preflight["started_remote_count"] == 0
    assert preflight["started_performance_count"] == 0


def test_phase1_preflight_complete_matrix_allows_stream_callback() -> None:
    preflight = build_native_energy_preflight(
        expected_rows=_complete_phase1_rows(),
        energy_requested=True,
        energy_evidence_tier="screening",
        strict_requested=False,
        validation_requested=True,
        full_baselines_enabled=True,
        setup_ids_by_producer={"hailo8": "h8"},
    )
    calls: list[str] = []

    def stream() -> None:
        calls.append("called")

    if not native_energy_preflight_blocks_streaming(preflight):
        stream()

    assert calls == ["called"]
    assert preflight["status"] == "passed"
    assert preflight["plan_viable"] is True
    assert preflight["theoretical_pair_count"] == 1
    assert preflight["setup_model_rows"][0]["plan_viable"] is True


def _variant_phase1_fixture(
    tmp_path: Path, *, full_enabled: bool,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "evalrun"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"resume_contract": {}}), encoding="utf-8",
    )
    benchmark_set = run_dir / "models" / "m" / "benchmark_set"
    (benchmark_set / "b001").mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"benchmark_task": "classification"}), encoding="utf-8",
    )
    config = {
        "energy": {"enabled": True, "mode": "measure", "strict": False},
        "validation": {"enabled": True},
        "quality_gate_policy": {
            "enforcement": {"technical_quality_error": "hard_fail"},
        },
        "_workflow_context": {
            "campaign": {"mode": "development"},
            "execution_preset": {"id": "standard"},
            "native_energy_final_contract_requested": False,
            "energy": {"strict": False},
        },
        "variants": [{
            "id": "v",
            "backends": ["hailo8"],
            "case_map": {"m": ["b001"]},
            "remotes": {
                "hailo8": {
                    "setup_id": "h8",
                    "ssh": "unused.example",
                },
            },
            "validation": {"enabled": True},
            "full_baselines": {
                "enabled": full_enabled,
                "backends_by_producer": {
                    "hailo8": ["hailo8", "tensorrt"],
                },
            },
        }],
    }
    config_path = tmp_path / "native-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return run_dir, config_path


def _patch_variant_preflight_dependencies(
    monkeypatch: pytest.MonkeyPatch, coordinator: Any,
) -> None:
    monkeypatch.setattr(
        coordinator,
        "_materialize_native_split_quality_binding_sets",
        lambda *args, **kwargs: ({}, {
            "required": False,
            "status": "test",
            "variants": [],
        }),
    )
    monkeypatch.setattr(
        coordinator,
        "_materialize_trt_quality_producer_sets",
        lambda *args, **kwargs: ({}, {
            "required": False,
            "status": "test",
            "variants": [],
        }),
    )
    monkeypatch.setattr(
        coordinator,
        "_quality_first_variant_plan",
        lambda cfg, variants, *args, **kwargs: [
            dict(item) for item in variants
        ],
    )
    monkeypatch.setattr(
        coordinator,
        "resolve_native_split_quality_authority",
        lambda **kwargs: {"valid": True, "workflow_version": "test"},
    )
    monkeypatch.setattr(
        coordinator,
        "native_split_quality_required_for_row",
        lambda *args, **kwargs: False,
    )


def test_variant_phase1_unpaired_split_reaches_streaming_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run_dir, config_path = _variant_phase1_fixture(
        tmp_path, full_enabled=False,
    )
    _patch_variant_preflight_dependencies(monkeypatch, coordinator)
    class StreamingReached(RuntimeError):
        pass

    calls: list[list[str]] = []

    def reached(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append(list(cmd))
        raise StreamingReached

    monkeypatch.setattr(coordinator, "_run", reached)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(coordinator.__file__),
            "--eval-run-dir", str(run_dir),
            "--config", str(config_path),
        ],
    )

    with pytest.raises(StreamingReached):
        coordinator.main()
    assert len(calls) == 1
    preflight = json.loads(
        (run_dir / "reports" / "native_energy_preflight.json").read_text(
            encoding="utf-8",
        )
    )
    assert preflight["status"] == "passed"
    assert preflight["plan_viable"] is True
    assert preflight["theoretical_pair_count"] == 0
    assert "native_full_baselines_not_enabled" in (
        preflight["nonblocking_annotations"]
    )
    assert "no_theoretical_setup_local_energy_pair" in (
        preflight["nonblocking_annotations"]
    )


def test_variant_phase1_non_strict_development_unpaired_reaches_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run_dir, config_path = _variant_phase1_fixture(
        tmp_path, full_enabled=False,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_workflow_context"]["execution_preset"] = {"id": "development"}
    config["quality_gate_policy"] = {
        "enforcement": {
            "technical_quality_error": "partial_continue_diagnostic",
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _patch_variant_preflight_dependencies(monkeypatch, coordinator)
    class StreamingReached(RuntimeError):
        pass

    calls: list[list[str]] = []

    def reached(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append(list(cmd))
        raise StreamingReached

    monkeypatch.setattr(coordinator, "_run", reached)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(coordinator.__file__),
            "--eval-run-dir", str(run_dir),
            "--config", str(config_path),
        ],
    )

    with pytest.raises(StreamingReached):
        coordinator.main()
    assert len(calls) == 1
    preflight = json.loads(
        (run_dir / "reports" / "native_energy_preflight.json").read_text(
            encoding="utf-8",
        )
    )
    assert preflight["status"] == "passed"
    assert preflight["plan_viable"] is True
    assert preflight["theoretical_pair_count"] == 0
    assert "native_full_baselines_not_enabled" in (
        preflight["nonblocking_annotations"]
    )
    assert "no_theoretical_setup_local_energy_pair" in (
        preflight["nonblocking_annotations"]
    )


def test_variant_phase1_complete_matrix_reaches_streaming_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run_dir, config_path = _variant_phase1_fixture(
        tmp_path, full_enabled=True,
    )
    _patch_variant_preflight_dependencies(monkeypatch, coordinator)

    class StreamingReached(RuntimeError):
        pass

    calls: list[list[str]] = []

    def reached(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append(list(cmd))
        raise StreamingReached

    monkeypatch.setattr(coordinator, "_run", reached)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(coordinator.__file__),
            "--eval-run-dir", str(run_dir),
            "--config", str(config_path),
        ],
    )
    with pytest.raises(StreamingReached):
        coordinator.main()

    assert len(calls) == 1
    preflight = json.loads(
        (run_dir / "reports" / "native_energy_preflight.json").read_text(
            encoding="utf-8",
        )
    )
    assert preflight["status"] == "passed"
    assert preflight["plan_viable"] is True
    assert preflight["theoretical_pair_count"] == 1


def test_direct_runner_phase1_gate_precedes_backend_streaming_loop() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool" / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    phase1 = source.index(
        "# Phase 1: prove that the configured Native matrix"
    )
    block = source.index(
        "native_energy_preflight_blocks_streaming(", phase1,
    )
    backend_loop = source.index("for backend in backends:", block)
    first_remote_stream = source.index("_stream_native(", backend_loop)
    assert phase1 < block < backend_loop < first_remote_stream
