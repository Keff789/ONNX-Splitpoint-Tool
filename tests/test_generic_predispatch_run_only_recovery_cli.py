"""The narrow pre-dispatch command requires explicit original scope and proof."""
from __future__ import annotations

import json

import pytest

from onnx_splitpoint_tool.remote import process_lease, resource_recovery
from onnx_splitpoint_tool.workflow.run_control import WorkflowRunCleanupQuarantineError


def _arguments():
    return [
        "recover-generic-predispatch-run-only",
        "--run-dir", "/original/run", "--session-id", "original-session",
        "--setup-id", "original-setup", "--evidence-path", "/evidence/original.json",
        "--authorization-path", "/authorization/one-run.json",
        "--evidence-output", "/recovery/receipt.json",
    ]


@pytest.fixture
def calls(monkeypatch):
    observed = []

    def recover(**kwargs):
        observed.append(kwargs)
        return {"status": "committed", "measurements_started": 0}

    def ordinary(*_args, **_kwargs):
        pytest.fail("the pre-dispatch command must not fall through to ordinary Generic recovery")

    monkeypatch.setattr(resource_recovery, "recover_generic_predispatch_run_only", recover, raising=False)
    monkeypatch.setattr(resource_recovery, "recover_generic_orphan_run", ordinary)
    return observed


def test_explicit_predispatch_command_forwards_exact_original_scope(calls, capsys):
    assert process_lease.process_lease_cli_main(_arguments()) == 0
    assert calls == [{"run_dir": "/original/run", "session_id": "original-session",
        "setup_id": "original-setup", "evidence_path": "/evidence/original.json",
        "authorization_path": "/authorization/one-run.json",
        "evidence_output": "/recovery/receipt.json", "registry_path": None}]
    assert json.loads(capsys.readouterr().out) == {"status": "committed", "measurements_started": 0}


def test_predispatch_command_preserves_explicit_registry(calls):
    assert process_lease.process_lease_cli_main(_arguments() + ["--registry", "/registry/original.yaml"]) == 0
    assert calls[0]["registry_path"] == "/registry/original.yaml"


@pytest.mark.parametrize("option", ["--run-dir", "--session-id", "--setup-id", "--evidence-path",
    "--authorization-path", "--evidence-output"])
def test_predispatch_command_requires_every_scope_and_proof_argument(calls, capsys, option):
    argv = _arguments()
    index = argv.index(option)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(argv)
    assert error.value.code == 2
    assert option in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("option", ["--session-id", "--evidence-path", "--authorization-path"])
def test_predispatch_command_rejects_blank_scope_or_proof(calls, capsys, option):
    argv = _arguments()
    argv[argv.index(option) + 1] = "  "
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(argv)
    assert error.value.code == 2
    assert "nonempty" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("option", ["--force", "--unlock", "--session"])
def test_predispatch_command_has_no_force_or_abbreviated_scope(calls, option):
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(_arguments() + [option, "override"])
    assert error.value.code == 2
    assert calls == []


@pytest.mark.parametrize("error", [WorkflowRunCleanupQuarantineError("dispatch exists"),
    ValueError("authorization belongs to another session"), OSError("receipt persistence failed")])
def test_predispatch_proof_failure_has_no_success_output(monkeypatch, capsys, error):
    def blocked(**_kwargs):
        raise error

    monkeypatch.setattr(resource_recovery, "recover_generic_predispatch_run_only", blocked, raising=False)
    assert process_lease.process_lease_cli_main(_arguments()) == 70
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pre-dispatch" in captured.err.lower()
    assert str(error) in captured.err


def test_predispatch_help_exposes_only_narrow_recovery_scope(capsys):
    with pytest.raises(SystemExit) as error:
        process_lease.process_lease_cli_main(["recover-generic-predispatch-run-only", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for option in ("--run-dir", "--session-id", "--setup-id", "--evidence-path",
                   "--authorization-path", "--evidence-output", "--registry"):
        assert option in output
    assert "--force" not in output
    assert "--unlock" not in output
    assert "--skip" not in output
