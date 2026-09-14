from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    seal_native_command_contract,
    split_energy_runtime_argv,
    verify_native_command_contract,
)
from tests.test_energy_final_contract_regressions import (
    _with_runtime_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "scripts" / "native_producer_energy_plan.py"


def _runtime_row(case: str) -> dict[str, Any]:
    endpoint_sha = "d" * 64
    return _with_runtime_contract({
        "ok": True,
        "backend": "hailo10h_to_trt",
        "model": "resnet50",
        "case": case,
        "precision": "fp16",
        "setup_id": "orin_nx_hailo10_01",
        "comparison_backend": "hailo10h",
        "fps_makespan": 10.0,
        "part2_input_count": 1,
        "task": "classification",
        "_test_task": "classification",
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_sha,
        "output_endpoint_id": (
            f"classification:classification_logits:{endpoint_sha}"
        ),
    })


def _validation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "ok": True,
        "error": "",
        "task": "classification",
        "top1_match": True,
        "semantic_ok": True,
        "contract_consistent": True,
        "claim_ok": True,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_claim_result_verified": True,
    }


def _write_inputs(
    tmp_path: Path,
    row: dict[str, Any],
) -> tuple[Path, Path]:
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    summary.write_text(
        json.dumps({"rows": [row]}),
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps({"rows": [_validation_row(row)]}),
        encoding="utf-8",
    )
    (tmp_path / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": 1,
            "present_expected_row_count": 1,
            "successful_expected_row_count": 1,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
            "present_expected_rows": [row],
            "failed_expected_rows": [],
            "missing_expected_rows": [],
        }),
        encoding="utf-8",
    )
    return summary, validation


def _run_plan(
    tmp_path: Path,
    row: dict[str, Any],
    *,
    provide_ssh: bool,
) -> dict[str, Any]:
    summary, validation = _write_inputs(tmp_path, row)
    out = tmp_path / "energy-plan"
    command = [
        sys.executable,
        str(PLANNER),
        "--summary",
        str(summary),
        "--validation-summary",
        str(validation),
        "--out-dir",
        str(out),
        "--duration-s",
        "1",
        "--allow-unpaired",
    ]
    if provide_ssh:
        command += ["--hailo10-ssh", "h10-test-host"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout,
        completed.stderr,
    )
    return json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )


def _assert_complete_identity(
    row: dict[str, Any],
    *,
    case: str,
) -> None:
    assert {
        field: row.get(field)
        for field in (
            "backend",
            "model",
            "case",
            "setup_id",
            "comparison_backend",
            "precision",
        )
    } == {
        "backend": "hailo10h_to_trt",
        "model": "resnet50",
        "case": case,
        "setup_id": "orin_nx_hailo10_01",
        "comparison_backend": "hailo10h",
        "precision": "fp16",
    }


def test_missing_ssh_is_one_complete_exclusion_not_a_plan_row(
    tmp_path: Path,
) -> None:
    payload = _run_plan(
        tmp_path,
        _runtime_row("b001"),
        provide_ssh=False,
    )

    assert payload["rows"] == []
    [excluded] = payload["excluded_rows"]
    _assert_complete_identity(excluded, case="b001")
    assert excluded["reason"] == "energy_ssh_target_missing"
    assert payload["energy_plan_included_count"] == 0
    assert payload["energy_plan_excluded_count"] == 1
    assert payload["energy_plan_coverage_contract_valid"] is True
    assert payload["energy_plan_ledger_overlap_identities"] == []
    assert payload["energy_plan_ledger_unexpected_identities"] == []
    assert payload["energy_plan_ledger_missing_identities"] == []


def test_unreplayable_contract_has_one_complete_ledger_exclusion(
    tmp_path: Path,
) -> None:
    row = _runtime_row("b002")
    contract = copy.deepcopy(row["native_command_contract"])
    contract.pop("contract_sha256")
    contract["runtime_options"]["producer_impl"] = "sync"
    contract = seal_native_command_contract(contract)
    row["native_command_contract"] = contract

    verified, status = verify_native_command_contract(
        contract,
        expected_identity={
            field: row[field]
            for field in (
                "backend",
                "model",
                "case",
                "precision",
                "setup_id",
                "comparison_backend",
            )
        },
    )
    assert verified is not None, status
    with pytest.raises(
        ValueError,
        match="probe-free async_fifo contract",
    ):
        split_energy_runtime_argv(
            verified,
            duration_s=1.0,
            fresh_output_root="/tmp/fresh-output",
            remote_tool_dir="/opt/onnx-splitpoint-tool",
            preflight_attestation_path="/tmp/preflight.json",
        )

    payload = _run_plan(
        tmp_path,
        row,
        provide_ssh=True,
    )

    assert payload["rows"] == []
    [excluded] = payload["excluded_rows"]
    _assert_complete_identity(excluded, case="b002")
    assert excluded["reason"] == (
        "verified_energy_command_contract_not_replayable"
    )
    assert "probe-free async_fifo contract" in excluded["detail"]
    assert payload["energy_plan_included_count"] == 0
    assert payload["energy_plan_excluded_count"] == 1
    assert payload["energy_plan_coverage_contract_valid"] is True
    assert payload["energy_plan_ledger_overlap_identities"] == []
    assert payload["energy_plan_ledger_unexpected_identities"] == []
    assert payload["energy_plan_ledger_missing_identities"] == []
