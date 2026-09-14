from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.energy.collector import (
    _energy_repeat_contract_complete,
)
from onnx_splitpoint_tool.native_energy_reporting import _energy_payload
from tests.test_energy_final_contract_regressions import (
    _with_runtime_contract,
    _write_model_hash_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_energy_planner():
    path = ROOT / "scripts" / "native_producer_energy_plan.py"
    spec = importlib.util.spec_from_file_location(
        f"v2721_energy_contract_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_full_smoke_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    binding_verified: bool,
    accuracy_gate_pass: bool,
    semantic_ok: bool = True,
    task_observation_valid: bool | None = True,
    expected_count: int = 1,
    window_ab_repeats: int | None = None,
) -> dict[str, Any]:
    planner = _load_energy_planner()
    model_map, model_map_sha, model_sha, _ = _write_model_hash_evidence(
        tmp_path, model_id="resnet50",
    )
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    out = tmp_path / "energy-plan"
    endpoint_sha = "1" * 64
    quality_policy_sha = "a" * 64
    row = _with_runtime_contract({
        "ok": True,
        "backend": "native_full_hailo8",
        "model": "resnet50",
        "case": "full",
        "precision": "fp16",
        "execution_precision": "fp16",
        "full_runtime_precision": "fp16",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "fps_makespan": 8.0,
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
    })
    quality = {
        **row,
        "ok": True,
        "error": "",
        "task": "classification",
        "top1_match": True,
        "semantic_ok": semantic_ok,
        "numerical_similarity_reason": (
            "" if semantic_ok else "reference_match_ratio_below_threshold"
        ),
        "contract_consistent": True,
        "claim_ok": accuracy_gate_pass,
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": binding_verified,
        # This compatibility field deliberately follows the historical smoke
        # payload.  The v2.72.2 decision must use the explicit binding axis.
        "precision_quality_verified": accuracy_gate_pass,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": accuracy_gate_pass,
        "quality_claim_result_verified": accuracy_gate_pass,
        "runtime_precision_identity": "fp16",
        "quality_contract_sha256": "2" * 64,
        "preprocessing_contract_sha256": "3" * 64,
        "source_request_sha256": "4" * 64,
        "model_sha256": model_sha,
        "validation_dataset_sha256": "5" * 64,
        "validation_dataset_image_ids_sha256": "6" * 64,
        "validation_dataset_ground_truth_sha256": "7" * 64,
        "accuracy_gate_policy_sha256": quality_policy_sha,
        "task_quality_policy_sha256": quality_policy_sha,
        "runtime_quality_gate_policy_sha256": quality_policy_sha,
    }
    if task_observation_valid is None:
        quality.pop("task_quality_observation_valid")
    else:
        quality["task_quality_observation_valid"] = (
            task_observation_valid
        )
    summary.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    validation.write_text(
        json.dumps({"rows": [quality]}), encoding="utf-8",
    )
    failed_expected_rows = [
        {
            "actual_ok": False,
            "actual_status": "failed",
            "backend": "native_full_tensorrt",
            "model": f"expected-failed-{index}",
            "case": "full",
            "precision": "fp16",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "execution_mode": "native_full_baseline",
            "failure_reason": "native_runtime_failed_before_energy_plan",
            "status_detail": "fixture_expected_runtime_failure",
        }
        for index in range(max(0, expected_count - 1))
    ]
    (tmp_path / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": expected_count,
            "present_expected_row_count": expected_count,
            "successful_expected_row_count": 1,
            "present_expected_rows": [row],
            "failed_expected_row_count": len(failed_expected_rows),
            "failed_expected_rows": failed_expected_rows,
            "missing_expected_row_count": 0,
            "missing_expected_rows": [],
        }),
        encoding="utf-8",
    )
    argv = [
        str(planner.__file__),
        "--summary", str(summary),
        "--validation-summary", str(validation),
        "--out-dir", str(out),
        "--hailo8-ssh", "smoke-host",
        "--duration-s", "1",
        "--model-hash-map", str(model_map),
        "--model-hash-map-sha256", model_map_sha,
        "--allow-unpaired",
        "--smoke-diagnostic",
    ]
    if window_ab_repeats is not None:
        argv += [
            "--window-method-ab-json",
            json.dumps({
                "enabled": True,
                "smoke_repeats": window_ab_repeats,
            }),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    assert planner.main() == 0
    return json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )


def _assert_planned_nonclaimable(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["energy_plan_included_count"] == 1
    assert payload["energy_plan_excluded_count"] == 0
    assert payload["preflight"]["measurement_start_allowed"] is True
    assert payload["excluded_rows"] == []
    [row] = payload["rows"]
    assert row["diagnostic_only"] is True
    for claim_axis in (
        "claim_ok",
        "semantic_claim_ok",
        "claim_eligible",
        "energy_claim_eligible",
        "eligible_for_scientific_claim",
    ):
        assert row[claim_axis] is False
    return row


def test_smoke_quality_binding_is_downstream_nonclaim_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=False,
        accuracy_gate_pass=False,
    )

    row = _assert_planned_nonclaimable(payload)
    assert row["backend"] == "native_full_hailo8"
    assert row["precision_quality_binding_verified"] is False


def test_smoke_allows_bound_accuracy_fail_only_as_diagnostic_nonclaim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=True,
        accuracy_gate_pass=False,
    )

    [row] = payload["rows"]
    assert row["precision_quality_binding_verified"] is True
    assert row["accuracy_gate_pass"] is False
    assert row["screening_comparable"] is False
    assert row["claim_comparable"] is False
    assert row["diagnostic_only"] is True
    assert row["energy_claim_eligible"] is False
    assert row["eligible_for_scientific_claim"] is False
    assert "accuracy_gate_failed" in row[
        "scientific_claim_exclusion_reasons"
    ]


def test_smoke_measures_bound_negative_semantic_observation_as_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=True,
        accuracy_gate_pass=False,
        semantic_ok=False,
    )

    [row] = payload["rows"]
    assert row["precision_quality_binding_verified"] is True
    assert row["diagnostic_only"] is True
    assert row["claim_comparable"] is False
    assert row["energy_claim_eligible"] is False


@pytest.mark.parametrize("task_observation_valid", [None, False])
def test_smoke_missing_or_invalid_task_observation_stays_nonclaimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_observation_valid: bool | None,
) -> None:
    payload = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=True,
        accuracy_gate_pass=False,
        task_observation_valid=task_observation_valid,
    )

    row = _assert_planned_nonclaimable(payload)
    assert row["backend"] == "native_full_hailo8"
    assert row["task_quality_observation_valid"] is False


def test_two_of_three_valid_repeats_cannot_report_final_gate_pass() -> None:
    valid_repeat = {
        "final_energy_gate_status": "pass",
        "scientific_primary_energy_status": "available",
        "scientific_primary_energy_j": 1.0,
    }
    incomplete_repeat = {
        # This reproduces the contradictory row observed in the smoke pack:
        # the row-level gate said pass although no primary result was usable.
        "final_energy_gate_status": "pass",
        "scientific_primary_energy_status": "unavailable",
    }
    runs = [dict(valid_repeat), dict(valid_repeat), incomplete_repeat]
    assert not _energy_repeat_contract_complete(
        runs,
        requested_count=3,
        require_ab=False,
    )

    payload = _energy_payload({
        "run": {
            "energy_aggregate": {
                "run_count": 3,
                "requested_valid_repeat_count": 3,
                "valid_postprocessed_runs": 2,
                "raw_postprocessed_run_count": 3,
                "repeat_contract_complete": False,
                "postprocess_status": "ok",
                "final_energy_gate_status": "pass",
                "scientific_primary_energy_statistics": {
                    "energy_j": {"n": 2, "mean": 1.0},
                },
                "runs": runs,
            },
        },
    })
    assert payload["valid_postprocessed_runs"] == 2
    assert payload["postprocess_status"] == "incomplete_valid_repeats"
    assert payload["final_energy_gate_status"] == "fail"


def test_energy_plan_exclusion_ledger_matches_matrix_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=True,
        accuracy_gate_pass=False,
        expected_count=2,
    )

    assert (
        payload["energy_plan_included_count"]
        + payload["energy_plan_excluded_count"]
        == payload["energy_matrix_expected_count"]
    )
    assert (
        len(payload["excluded_rows"])
        == payload["energy_plan_excluded_count"]
    )
    assert payload["preflight"]["energy_plan_excluded_count"] == len(
        payload["excluded_rows"]
    )


def test_hailo8_detection_energy_replay_exports_only_bound_extra_sites() -> None:
    planner = _load_energy_planner()
    site_dir = "/home/nx/hailo_py/lib/python3.10/site-packages"
    contract = {
        "backend": "hailo8_to_trt",
        "interpreter_identity": {
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "process_local_extra_sites": [site_dir],
        },
        "mixed_runtime_contract": {
            "status": "ready",
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "site_policy": "site.addsitedir_after_system_defaults",
            "process_local_extra_sites": [site_dir],
            "source_closure_ok": True,
        },
        "runtime_options": {
            "producer_impl": "hailo8_python_vstreams_fifo",
            "mixed_runtime_site_policy": (
                "site.addsitedir_after_system_defaults"
            ),
            "process_local_extra_sites": [site_dir],
        },
    }

    process_env = planner._process_local_runtime_environment(contract)
    assert process_env == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "SPLITPOINT_EXTRA_SITES": site_dir,
    }
    prefix = planner._remote_environment_prefix(
        "source /home/nx/hailo_py/bin/activate", process_env,
    )
    assert "source /home/nx/hailo_py/bin/activate && " in prefix
    assert f"export SPLITPOINT_EXTRA_SITES={site_dir}" in prefix
    assert "export PYTHONPATH=" not in prefix

    tampered = json.loads(json.dumps(contract))
    tampered["mixed_runtime_contract"][
        "process_local_extra_sites"
    ] = ["/tmp/not-the-bound-site"]
    with pytest.raises(
        ValueError,
        match="hailo8_process_local_extra_sites_mismatch",
    ):
        planner._process_local_runtime_environment(tampered)


def test_energy_plan_freezes_profile_and_effective_repeat_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=True,
        accuracy_gate_pass=False,
        window_ab_repeats=3,
    )

    assert payload["energy_profile_requested_runs_per_row"] == 1
    assert payload["energy_effective_runs_per_row"] == 3
    assert payload["energy_runs_per_row"] == 3
    assert payload["energy_repeat_expansion_applied"] is True
    assert payload["energy_repeat_expansion_reason"] == (
        "frozen_window_method_ab_minimum"
    )
    [row] = payload["rows"]
    assert row["measurement_profile_requested_repeats"] == 1
    assert row["measurement_effective_repeats"] == 3
    assert row["measurement_requested_repeats"] == 3
    assert row["measurement_repeat_expansion_applied"] is True
    assert " --runs 3 " in row["measure_command"]
