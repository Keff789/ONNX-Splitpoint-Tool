"""Read-only final-cleanup transport and exact guardian token observations."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.remote import resource_recovery_observation as observation
from onnx_splitpoint_tool.remote.process_lease import process_lease_cli_main


@pytest.fixture
def operation(tmp_path, monkeypatch):
    monkeypatch.setattr(observation, "_observe_processes", lambda _: {"ok": True})
    return {"remote_lease_root": str(tmp_path), "run_sha256": "a" * 64,
            "session_sha256": "b" * 64, "operation_id": "ssh-stream-old", "token": "old-exact-token"}


def test_missing_lease_never_invents_a_guardian_drain(operation):
    value = observation._observe_final_cleanup_remote([operation])
    assert value["read_only"] is True
    assert value["operations"][0]["lease_present"] is False
    assert value["operations"][0]["guardian_drained"] is False
    assert not Path(operation["remote_lease_root"], operation["run_sha256"]).exists()


@pytest.mark.parametrize("fault", [None, "token", "symlink", "foreign_lease"])
def test_only_exact_regular_guardian_token_is_observed(operation, fault):
    root = Path(operation["remote_lease_root"], operation["run_sha256"], operation["session_sha256"])
    root.mkdir(parents=True)
    drained = root / (operation["operation_id"] + ".drained")
    drained.write_text("wrong-token" if fault == "token" else operation["token"])
    if fault == "symlink":
        target = root / "foreign"
        drained.rename(target)
        drained.symlink_to(target)
    elif fault == "foreign_lease":
        lease = {"schema": "onnx-splitpoint/remote-process-lease", **operation, "token": "other"}
        (root / (operation["operation_id"] + ".lease.json")).write_text(json.dumps(lease))
    if fault:
        with pytest.raises(ValueError):
            observation._observe_final_cleanup_remote([operation])
    else:
        value = observation._observe_final_cleanup_remote([operation])
        assert value["operations"][0]["guardian_drained"] is True


@pytest.mark.parametrize("response", [(0, 'GENERIC_FINAL_CLEANUP={"read_only": true}'),
                                     (255, "unavailable"), (0, "missing marker")])
def test_final_observer_uses_one_normal_readonly_transport(tmp_path, monkeypatch, operation, response):
    calls = []

    class Transport:
        def __init__(self, host):
            self.host = host

        def run_read_only(self, command, **kwargs):
            calls.append((command, kwargs))
            return response

    monkeypatch.setattr(observation, "SSHTransport", Transport)
    setup = {"id": "fixture", "host": {"address": "192.0.2.104", "user": "fixture"}}
    if response[0] == 0 and response[1].startswith("GENERIC_FINAL_CLEANUP="):
        result = observation.observe_generic_final_cleanup_remote(setup, operations=[operation], diagnostics_dir=tmp_path)
        assert result["read_only"] is True
        assert result["local_end_unix_ns"] >= result["local_start_unix_ns"]
    else:
        with pytest.raises(ValueError):
            observation.observe_generic_final_cleanup_remote(setup, operations=[operation], diagnostics_dir=tmp_path)
    assert len(calls) == 1
    command, options = calls[0]
    assert "_observe_final_cleanup_remote" in command
    assert "old-exact-token" in command
    assert options == {"timeout": 30, "env": {"PYTHONDONTWRITEBYTECODE": "1"}}


def test_final_cleanup_cli_requires_explicit_registry_and_documents():
    with pytest.raises(SystemExit) as exc:
        process_lease_cli_main(["recover-generic-final-cleanup"])
    assert exc.value.code == 2
