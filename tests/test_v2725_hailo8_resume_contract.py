from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import onnx_splitpoint_tool.resume_cohort_preflight as cohort
import onnx_splitpoint_tool.resume_preparation as preparation
from onnx_splitpoint_tool.resume_artifact_contract import (
    SPLIT_COMMAND_CONTRACT_SCHEMA,
)


REMOTE_ROOT = "/home/nx/native_fifo_evalsets/campaign_gate"
SSH_TARGET = "nx@192.168.0.104"
SELECTOR = "hailo8_to_trt|yolo26s|b038|orin_nx_hailo8_01"
REAL_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "v2725_hailo8_resume"
    / (
        "hailo8_to_trt__resnet50__b052__orin_nx_hailo8_01__"
        "8eb40a232207.command_contract.json"
    )
)
REAL_FIXTURE_BYTES = 141_442
REAL_FIXTURE_FILE_SHA256 = (
    "4e5bac737c9a071eee2bf218c3f76e996f69dd0183c7796bdca088c8c91aaba6"
)
REAL_FIXTURE_LOGICAL_SHA256 = (
    "8eb40a2322078272add157c50b41309f2a5b43be5dff16eae773eb339f106a8b"
)
REAL_REMOTE_ROOT = (
    "/home/nx/native_fifo_evalsets/"
    "resnet_yolo26s_yolo7_20260729_214045"
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("contract_sha256", None)
    return hashlib.sha256(json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _artifact(role: str) -> dict[str, Any]:
    return {
        "path": f"{REMOTE_ROOT}/yolo26s/b038/{role}.bin",
        "sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
        "size_bytes": len(role),
    }


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    attempt = tmp_path / "resume_attempt"
    plan = attempt / "plan"
    plan.mkdir(parents=True)
    reports = tmp_path / "EvaluationRun" / "reports"
    reports.mkdir(parents=True)
    summary = reports / "native_producer_summary.json"
    summary.write_text('{"schema":"test-summary"}\n', encoding="utf-8")
    return attempt, plan, summary


def _real_hailo8_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "schema_version": 1,
        "backend": "hailo8_to_trt",
        "model": "yolo26s",
        "case": "b038",
        "setup_id": "orin_nx_hailo8_01",
        "runtime_options": {
            "prepared_input_bound": True,
            "energy_prepared_feed_capable": True,
        },
        "prepared_input_contract": {
            "format": "raw_runtime_tensor",
            "name": "images",
            "shape": [640, 640, 3],
            "dtype": "uint8",
            "layout": "HWC",
        },
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
        },
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
        "complete": True,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _bound_row(plan: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = plan / "hailo8.command_contract.json"
    raw = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)
    return {
        "backend": "hailo8_to_trt",
        "model": "yolo26s",
        "case": "b038",
        "setup_id": "orin_nx_hailo8_01",
        "command_contract_file": str(path),
        "command_contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "successful_command_contract_sha256": contract["contract_sha256"],
    }


def _context() -> dict[str, str]:
    return {
        "remote_root": REMOTE_ROOT,
        "hailo8_ssh": SSH_TARGET,
        "hailo10_ssh": "",
        "deepx_ssh": "",
    }


def _forbidden(label: str):
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{label} must not be called")

    return fail


@pytest.fixture(autouse=True)
def _block_real_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _forbidden("real subprocess/network/hardware"),
    )


def test_real_hailo8_shaped_contract_reaches_preflight_without_part1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the real v2.72.4 ``part1_onnx`` preparation failure."""

    attempt, plan, summary = _layout(tmp_path)
    row = _bound_row(plan, _real_hailo8_contract())
    observed_roles: list[str] = []

    def probe(requirements, **kwargs):
        assert kwargs["ssh_target"] == SSH_TARGET
        assert kwargs["remote_run_root"] == REMOTE_ROOT
        observed_roles.extend(requirement.role for requirement in requirements)
        entries = [{
            "remote_path": requirement.remote_path,
            "roles": [requirement.role],
            "expected_sha256": requirement.sha256,
            "expected_size_bytes": requirement.size_bytes,
            "remote_status": "exact",
            "remote_sha256": requirement.sha256,
            "remote_size_bytes": requirement.size_bytes,
            "exact": True,
            "safe_to_rehydrate": False,
        } for requirement in requirements]
        return {
            "ok": True,
            "read_only": True,
            "requirement_count": len(entries),
            "exact_count": len(entries),
            "rehydration_required_count": 0,
            "all_exact": True,
            "entries": entries,
        }

    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        probe,
    )
    monkeypatch.setattr(
        preparation,
        "build_resume_artifact_stage_map",
        _forbidden("local rehydration resolver"),
    )
    monkeypatch.setattr(
        preparation,
        "rehydrate_remote_stage_map",
        _forbidden("remote mutation"),
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "status": "pass",
            "measurement_wrapper_allowed": True,
            "measurement_wrapper_started_count": 0,
            "collector_started_repeat_count": 0,
            "workload_started_repeat_count": 0,
        },
    )

    report = preparation.prepare_resume_measurement_cohort(
        [row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-hailo8-real-contract",
    )

    assert report["ok"] is True
    assert report["status"] == "ready_for_measurement"
    assert report["measurement_wrapper_allowed"] is True
    assert report["selected_rows"] == [SELECTOR]
    assert set(observed_roles) == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "input_image",
    }
    assert "part1_onnx" not in observed_roles
    assert "prepared_input_00" not in observed_roles
    assert report["remote_groups"][0]["rehydration"]["status"] == (
        "all_remote_artifacts_already_exact"
    )
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0


def test_hailo8_missing_prepared_input_stays_fail_closed_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    contract = _real_hailo8_contract()
    del contract["artifacts"]["prepared_input"]
    contract["contract_sha256"] = _canonical_sha256(contract)
    row = _bound_row(plan, contract)
    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        _forbidden("remote probe after invalid contract"),
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        _forbidden("cohort after invalid contract"),
    )

    report = preparation.prepare_resume_measurement_cohort(
        [row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-hailo8-missing-input",
    )

    assert report["ok"] is False
    assert report["status"] == "resume_preparation_failed"
    assert report["failure_type"] == "ResumeArtifactContractError"
    assert report["failure"] == (
        "command_contract_artifact_missing:prepared_input"
    )
    assert report["remote_groups"] == []
    assert report["cohort_preflight"] is None
    assert report["measurement_wrapper_allowed"] is False
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0


def test_archived_real_hailo8_contract_reaches_actual_cohort_without_ssh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete preparation/cohort seam missed by v2.72.4."""

    fixture_bytes = REAL_FIXTURE.read_bytes()
    assert len(fixture_bytes) == REAL_FIXTURE_BYTES
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        REAL_FIXTURE_FILE_SHA256
    )
    fixture = json.loads(fixture_bytes)
    assert fixture["contract_sha256"] == REAL_FIXTURE_LOGICAL_SHA256
    assert fixture["backend"] == "hailo8_to_trt"
    assert "prepared_input" in fixture["artifacts"]
    assert "part1_onnx" not in fixture["artifacts"]
    assert "prepared_input_00" not in fixture["artifacts"]

    attempt, plan, summary = _layout(tmp_path)
    contract_path = plan / "real_hailo8.command_contract.json"
    contract_path.write_bytes(fixture_bytes)
    command_path = plan / "real_hailo8.command.sh"
    command_path.write_text(
        "run-workload "
        f"--source-contract-sha256 {REAL_FIXTURE_LOGICAL_SHA256} "
        "--nonce __ONNX_SPLITPOINT_PREFLIGHT_NONCE__ "
        "--attestation __ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__\n",
        encoding="utf-8",
    )
    preflight_path = plan / "real_hailo8.preflight.sh"
    preflight_path.write_text(
        "run-preflight "
        f"--contract-json {contract_path} "
        f"--contract-file-sha256 {REAL_FIXTURE_FILE_SHA256} "
        "--nonce __ONNX_SPLITPOINT_PREFLIGHT_NONCE__ "
        "--attestation __ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__\n",
        encoding="utf-8",
    )
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "command_file": str(command_path),
        "preflight_command_file": str(preflight_path),
        "command_contract_file": str(contract_path),
        "command_contract_file_sha256": REAL_FIXTURE_FILE_SHA256,
        "successful_command_contract_sha256": (
            REAL_FIXTURE_LOGICAL_SHA256
        ),
        "preflight_runtime_attestation_path_template": (
            f"{REAL_REMOTE_ROOT}/resume_preflight/"
            "attestation___ONNX_SPLITPOINT_PREFLIGHT_NONCE__.json"
        ),
    }
    observed_roles: list[str] = []

    def probe(requirements, **kwargs):
        assert kwargs["ssh_target"] == SSH_TARGET
        assert kwargs["remote_run_root"] == REAL_REMOTE_ROOT
        observed_roles.extend(requirement.role for requirement in requirements)
        entries = [{
            "remote_path": requirement.remote_path,
            "roles": [requirement.role],
            "expected_sha256": requirement.sha256,
            "expected_size_bytes": requirement.size_bytes,
            "remote_status": "exact",
            "remote_sha256": requirement.sha256,
            "remote_size_bytes": requirement.size_bytes,
            "exact": True,
            "safe_to_rehydrate": False,
        } for requirement in requirements]
        return {
            "ok": True,
            "read_only": True,
            "requirement_count": len(entries),
            "exact_count": len(entries),
            "rehydration_required_count": 0,
            "all_exact": True,
            "entries": entries,
        }

    def preflight_boundary(
        _command_template: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        rendered = (
            str(kwargs["workload_command_template"])
            .replace(
                "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__",
                "offline-regression-nonce",
            )
            .replace(
                "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__",
                f"{REAL_REMOTE_ROOT}/offline-attestation.json",
            )
        )
        return {
            "ok": True,
            "status": "verified",
            "expected_command_contract_sha256": str(
                kwargs["expected_command_contract_sha256"]
            ),
            "collector_started": False,
            "workload_started": False,
            "validation": {
                "ok": True,
                "status": "verified",
                "reasons": [],
            },
        }, rendered

    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        probe,
    )
    monkeypatch.setattr(
        preparation,
        "build_resume_artifact_stage_map",
        _forbidden("local rehydration resolver"),
    )
    monkeypatch.setattr(
        preparation,
        "rehydrate_remote_stage_map",
        _forbidden("remote mutation"),
    )
    monkeypatch.setattr(
        cohort,
        "_run_energy_preflight",
        preflight_boundary,
    )

    report = preparation.prepare_resume_measurement_cohort(
        [row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context={
            "remote_root": REAL_REMOTE_ROOT,
            "hailo8_ssh": SSH_TARGET,
            "hailo10_ssh": "",
            "deepx_ssh": "",
        },
        resume_attempt_id="resume-real-hailo8-cohort",
    )

    assert report["ok"] is True
    assert report["status"] == "ready_for_measurement"
    assert report["measurement_wrapper_allowed"] is True
    assert report["selected_rows"] == [
        "hailo8_to_trt|resnet50|b052|orin_nx_hailo8_01"
    ]
    assert set(observed_roles) == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "input_image",
    }
    assert report["cohort_preflight"]["ok"] is True
    assert report["cohort_preflight"]["status"] == "verified"
    assert report["cohort_preflight"]["preflight_attempted_count"] == 1
    assert report["cohort_preflight"]["preflight_passed_count"] == 1
    assert report["cohort_preflight"]["preflight_failed_count"] == 0
    assert report["cohort_preflight"]["collector_started"] is False
    assert report["cohort_preflight"]["workload_started"] is False
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0
