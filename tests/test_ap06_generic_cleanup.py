"""Physical DUT fencing follows remaining owned cleanup, not sticky Cancel."""
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.process_control import (
    bind_workflow_resource_options,
    budget_remote_dut,
)
from onnx_splitpoint_tool.quality_service import ResourcePauseGate
from onnx_splitpoint_tool.remote.process_lease import (
    RemoteProcessLeaseRegistry,
    RemoteProcessLeaseScope,
)
from onnx_splitpoint_tool.workflow import run_control


@pytest.mark.parametrize("invalid_descriptor", [False, True])
def test_generic_cancel_fences_only_remaining_unproven_ownership(tmp_path, monkeypatch, invalid_descriptor):
    # Redirect storage only; use actual journal cancellation, gate, kernel
    # locks and persisted quarantine. No SSH operation is created or launched.
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path",
                        lambda: tmp_path / "locks" / "workflow.lock")
    registry = RemoteProcessLeaseRegistry()
    journal = registry.configure_journal(
        scope=RemoteProcessLeaseScope("local-cleanup-test", "session-one"),
        journal_dir=tmp_path / "journal")
    if invalid_descriptor:
        (journal.directory / "corrupt.remote-lease.json").write_text("{invalid", encoding="utf-8")
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=1024)
    resource = "dut:local-cleanup-fixture"

    @budget_remote_dut
    def controlled_job(*, host, remote_process_registry):
        assert gate.snapshot()["activities"], "DUT must remain held through cleanup"
        reports = remote_process_registry.cancel_all(grace_s=0.01)
        assert remote_process_registry.cancelled is True
        assert remote_process_registry.active_count() == int(invalid_descriptor)
        assert all(bool(report["ok"]) is False for report in reports)
        assert bool(reports) is invalid_descriptor
        return "cleanup-finished"

    with bind_workflow_resource_options(
            controller_gate=gate, controller_physical_protection=True,
            controller_physical_dut_key=resource, controller_run_id="local-cleanup-test"):
        assert controlled_job(host=SimpleNamespace(host="never-contacted"),
                              remote_process_registry=registry) == "cleanup-finished"
    assert gate.snapshot()["activities"] == []
    assert journal.is_cancelled(), "session admission must remain closed even after clean cancellation"

    next_session = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "next-session"})
    if invalid_descriptor:
        assert next_session.quarantine_path.exists()
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            next_session.acquire()
    else:
        assert not next_session.quarantine_path.exists(), (
            "a successfully cancelled empty registry must not durably quarantine a clean DUT"
        )
        try:
            next_session.acquire()
        finally:
            next_session.release()
