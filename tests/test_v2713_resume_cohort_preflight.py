from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import onnx_splitpoint_tool.resume_cohort_preflight as cohort


NONCE_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
ATTESTATION_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__"


def _row(
    plan_root: Path,
    *,
    backend: str,
    model: str,
    case: str,
    setup_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    stem = f"{backend}__{model}__{case}__{setup_id}"
    command_file = plan_root / f"{stem}.sh"
    preflight_file = plan_root / f"{stem}.preflight.sh"
    contract_file = plan_root / f"{stem}.command_contract.json"
    contract_file.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/test-command-contract",
                "contract_sha256": contract_sha256,
                "backend": backend,
                "model": model,
                "case": case,
                "setup_id": setup_id,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    contract_file_sha256 = hashlib.sha256(
        contract_file.read_bytes()
    ).hexdigest()
    command_file.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"runner --contract-sha {contract_sha256} "
        f"--nonce {NONCE_TOKEN} --attestation {ATTESTATION_TOKEN}\n",
        encoding="utf-8",
    )
    preflight_file.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"preflight --serialized-contract-sha {contract_file_sha256} "
        f"--nonce {NONCE_TOKEN} "
        f"--attestation {ATTESTATION_TOKEN} < {contract_file}\n",
        encoding="utf-8",
    )
    # Real Full-YOLOv7 plans bind the local contract bytes and validate the
    # logical SHA only through the returned attestation.  The full logical SHA
    # is therefore deliberately absent from the preflight shell template.
    assert contract_sha256 not in preflight_file.read_text(encoding="utf-8")
    return {
        "backend": backend,
        "model": model,
        "case": case,
        "setup_id": setup_id,
        "command_file": str(command_file),
        "preflight_command_file": str(preflight_file),
        "command_contract_file": str(contract_file),
        "command_contract_file_sha256": contract_file_sha256,
        "preflight_runtime_attestation_path_template": (
            f"/remote/energy/{stem}/preflight_{NONCE_TOKEN}.json"
        ),
        "successful_command_contract_sha256": contract_sha256,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    attempt = tmp_path / "resume_attempt"
    plan = attempt / "plan"
    plan.mkdir(parents=True)
    rows = [
        _row(
            plan,
            backend="hailo10h_to_trt",
            model="yolo26s",
            case="b038",
            setup_id="hailo10",
            contract_sha256="a" * 64,
        ),
        _row(
            plan,
            backend="native_full_hailo10h",
            model="yolov7_paper",
            case="full",
            setup_id="hailo10",
            contract_sha256="b" * 64,
        ),
    ]
    return attempt, plan, rows


def _successful_evidence(expected_sha: str) -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/energy-preflight-evidence",
        "schema_version": 1,
        "ok": True,
        "status": "verified",
        "expected_command_contract_sha256": expected_sha,
        "collector_started": False,
        "validation": {"ok": True, "status": "verified", "reasons": []},
    }


def test_all_selected_preflights_pass_and_enable_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, rows = _fixture(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_preflight(
        command_template: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        calls.append({"command_template": command_template, **kwargs})
        evidence_dir = Path(kwargs["evidence_dir"])
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence = _successful_evidence(
            kwargs["expected_command_contract_sha256"]
        )
        (evidence_dir / "preflight_evidence.json").write_text(
            json.dumps(evidence), encoding="utf-8",
        )
        return evidence, kwargs["workload_command_template"].replace(
            NONCE_TOKEN, "rendered-nonce",
        )

    monkeypatch.setattr(cohort, "_run_energy_preflight", fake_preflight)

    report = cohort.run_resume_cohort_preflight(
        rows,
        attempt_dir=attempt,
        plan_root=plan,
        timeout_s=123,
        max_age_s=45,
    )

    assert report["ok"] is True
    assert report["status"] == "verified"
    assert report["selected_row_count"] == 2
    assert report["prepared_row_count"] == 2
    assert report["preflight_attempted_count"] == 2
    assert report["preflight_passed_count"] == 2
    assert report["preflight_failed_count"] == 0
    assert report["measurement_wrapper_allowed"] is True
    assert report["measurement_wrapper_started"] is False
    assert report["collector_started"] is False
    assert len(calls) == 2
    assert all(call["timeout_s"] == 123 for call in calls)
    assert all(call["max_age_s"] == 45 for call in calls)
    assert all(call["repeat_index"] == "cohort" for call in calls)
    assert {
        call["expected_command_contract_sha256"] for call in calls
    } == {"a" * 64, "b" * 64}
    assert all(
        Path(call["evidence_dir"]).is_relative_to(
            attempt / "resume_cohort_preflight" / "rows"
        )
        for call in calls
    )
    report_path = Path(report["report_path"])
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["ok"] is True


def test_one_failure_runs_entire_cohort_and_blocks_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, rows = _fixture(tmp_path)
    calls: list[str] = []

    def fake_preflight(
        command_template: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        del command_template
        expected = kwargs["expected_command_contract_sha256"]
        calls.append(expected)
        passed = expected == "a" * 64
        evidence = _successful_evidence(expected)
        if not passed:
            evidence.update(
                {
                    "ok": False,
                    "status": "failed",
                    "validation": {
                        "ok": False,
                        "status": "failed",
                        "reasons": ["preflight_artifact_verification_not_pass"],
                    },
                }
            )
        return evidence, kwargs["workload_command_template"]

    monkeypatch.setattr(cohort, "_run_energy_preflight", fake_preflight)

    report = cohort.run_resume_cohort_preflight(
        rows,
        attempt_dir=attempt,
        plan_root=plan,
    )

    assert calls == ["a" * 64, "b" * 64]
    assert report["ok"] is False
    assert report["status"] == "failed"
    assert report["preflight_attempted_count"] == 2
    assert report["preflight_passed_count"] == 1
    assert report["preflight_failed_count"] == 1
    assert report["measurement_wrapper_allowed"] is False
    failed = [row for row in report["rows"] if not row["ok"]]
    assert len(failed) == 1
    assert failed[0]["preflight"]["validation"]["reasons"] == [
        "preflight_artifact_verification_not_pass"
    ]


def test_missing_plan_file_blocks_before_any_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, rows = _fixture(tmp_path)
    Path(rows[1]["preflight_command_file"]).unlink()
    monkeypatch.setattr(
        cohort,
        "_run_energy_preflight",
        lambda *args, **kwargs: pytest.fail(
            "invalid cohort must not execute a preflight"
        ),
    )

    report = cohort.run_resume_cohort_preflight(
        rows,
        attempt_dir=attempt,
        plan_root=plan,
    )

    assert report["ok"] is False
    assert report["status"] == "invalid_selection"
    assert report["prepared_row_count"] == 1
    assert report["preflight_attempted_count"] == 0
    assert report["measurement_wrapper_allowed"] is False
    assert {
        row["reason"] for row in report["rows"] if not row["ok"]
    } == {
        "cohort_selection_invalid",
        "preflight_command_file_not_regular",
    }


def test_duplicate_identity_blocks_before_any_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, rows = _fixture(tmp_path)
    duplicate = dict(rows[0])
    monkeypatch.setattr(
        cohort,
        "_run_energy_preflight",
        lambda *args, **kwargs: pytest.fail(
            "duplicate selection must not execute a preflight"
        ),
    )

    report = cohort.run_resume_cohort_preflight(
        [rows[0], duplicate],
        attempt_dir=attempt,
        plan_root=plan,
    )

    assert report["ok"] is False
    assert report["status"] == "invalid_selection"
    assert report["preflight_attempted_count"] == 0
    assert report["measurement_wrapper_allowed"] is False
    assert report["selection_errors"][0]["reason"] == "duplicate_row_identity"


def test_symlink_command_file_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, rows = _fixture(tmp_path)
    original = Path(rows[0]["command_file"])
    target = plan / "real-command.sh"
    original.replace(target)
    original.symlink_to(target)
    monkeypatch.setattr(
        cohort,
        "_run_energy_preflight",
        lambda *args, **kwargs: pytest.fail(
            "symlinked plan command must not execute a preflight"
        ),
    )

    report = cohort.run_resume_cohort_preflight(
        rows,
        attempt_dir=attempt,
        plan_root=plan,
    )

    assert report["ok"] is False
    assert report["preflight_attempted_count"] == 0
    failed = [row for row in report["rows"] if not row["ok"]]
    assert failed[0]["reason"] == "command_file_not_regular"
