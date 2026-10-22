"""The explicit Generic orphan CLI delegates proof admission to the product API."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.remote import process_lease, resource_recovery
from onnx_splitpoint_tool.workflow.run_control import WorkflowRunCleanupQuarantineError


def arguments():
    return [
        "recover-generic-orphan",
        "--run-dir", "/original/run",
        "--session-id", "original-session",
        "--setup-id", "original-setup",
        "--evidence-path", "/evidence/original.json",
        "--authorization-path", "/authorization/one-run.json",
        "--evidence-output", "/recovery/receipt.json",
    ]


@pytest.fixture
def calls(monkeypatch):
    observed = []

    def recover(**kwargs):
        observed.append(kwargs)
        return {"status": "released", "measurements_started": 0}

    monkeypatch.setattr(resource_recovery, "recover_generic_orphan_run", recover, raising=False)
    return observed


def test_explicit_scope_and_evidence_reach_only_normal_recovery(calls, capsys):
    assert process_lease.process_lease_cli_main(arguments()) == 0
    assert calls == [{
        "run_dir": "/original/run", "session_id": "original-session",
        "setup_id": "original-setup", "evidence_path": "/evidence/original.json",
        "authorization_path": "/authorization/one-run.json",
        "evidence_output": "/recovery/receipt.json", "registry_path": None,
    }]
    assert json.loads(capsys.readouterr().out) == {
        "status": "released", "measurements_started": 0,
    }


def test_explicit_registry_is_forwarded_without_replacing_original_scope(calls):
    assert process_lease.process_lease_cli_main(
        arguments() + ["--registry", "/registry/original.yaml"]
    ) == 0
    assert calls[0]["registry_path"] == "/registry/original.yaml"
    assert calls[0]["session_id"] == "original-session"


@pytest.mark.parametrize("option", [
    "--run-dir", "--session-id", "--setup-id", "--evidence-path",
    "--authorization-path", "--evidence-output",
])
def test_missing_original_scope_or_required_proof_never_calls_recovery(option, calls, capsys):
    argv = arguments()
    index = argv.index(option)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(argv)
    assert error.value.code == 2
    assert option in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("option", ["--session-id", "--evidence-path", "--authorization-path"])
def test_blank_scope_or_proof_is_not_an_authorization(option, calls, capsys):
    argv = arguments()
    argv[argv.index(option) + 1] = "  "
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(argv)
    assert error.value.code == 2
    assert "nonempty" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("option", ["--force", "--unlock", "--session"])
def test_generic_unlock_or_abbreviated_scope_flags_are_rejected(option, calls):
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(arguments() + [option, "override"])
    assert error.value.code == 2
    assert calls == []


@pytest.mark.parametrize("error", [
    WorkflowRunCleanupQuarantineError("original descendants remain unresolved"),
    process_lease.RemoteProcessLeaseJournalError("original journal is malformed"),
    ValueError("authorization belongs to another session"),
    OSError("receipt persistence failed"),
])
def test_normal_proof_failure_is_reported_without_success_receipt(error, monkeypatch, capsys):
    def blocked(**_kwargs):
        raise error

    monkeypatch.setattr(resource_recovery, "recover_generic_orphan_run", blocked, raising=False)
    assert process_lease.process_lease_cli_main(arguments()) == 70
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip() == "Generic orphan recovery blocked: " + str(error)


def test_requested_product_module_has_a_real_recovery_help_entrypoint():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "onnx_splitpoint_tool.remote.process_lease",
         "recover-generic-orphan", "--help"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    for required in ("--session-id", "--evidence-path", "--authorization-path", "--evidence-output"):
        assert required in result.stdout
    assert "--force" not in result.stdout
    assert "--unlock" not in result.stdout
