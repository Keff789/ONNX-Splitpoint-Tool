from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.benchmark.remote_run import RemoteBenchmarkArgs
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig


def _suite(tmp_path: Path) -> Path:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    benchmark_set = suite_dir / "benchmark_set.json"
    benchmark_set.write_text(
        json.dumps({"model_name": "resnet50"}), encoding="utf-8",
    )
    (suite_dir / "benchmark_plan.json").write_text(
        json.dumps(
            {
                "model_suite": {"primary": [{"id": "resnet50"}]},
                "runs": [
                    {"id": "ort_cpu", "type": "ort", "provider": "cpu"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return benchmark_set


def _run(
    tmp_path: Path,
    *,
    registry: RemoteProcessLeaseRegistry,
    logs: list[str],
    transfer_mode: str = "bundle",
) -> dict:
    return remote_run.run_remote_benchmark(
        host=HostConfig(
            id="deepx-setup",
            label="DeepX setup",
            host="192.168.0.102",
            user="nx",
        ),
        benchmark_set_json=_suite(tmp_path),
        local_working_dir=tmp_path / "working",
        run_id="v27538-dispatch-contract",
        args=RemoteBenchmarkArgs(
            resume=False,
            timeout_s=30,
            transfer_mode=transfer_mode,
            reuse_bundle=False,
            warmup=0,
            iters=1,
        ),
        log=logs.append,
        progress=lambda *_args: None,
        cancel_event=threading.Event(),
        remote_process_registry=registry,
    )


@pytest.mark.parametrize(
    ("failure", "expected_kind", "expected_rc"),
    [
        (
            "resolve_path_read_only failed (rc=255)\n"
            "ssh: connect to host 192.168.0.102 port 22: No route to host",
            "remote_connectivity_unavailable",
            255,
        ),
        (
            "resolve_path_read_only failed (rc=124)\ntimeout after 60s",
            "remote_connectivity_unavailable",
            124,
        ),
        (
            "resolve_path_read_only failed (rc=255)\n"
            "Permission denied (publickey)",
            "pre_remote_admission_failure",
            255,
        ),
    ],
)
def test_resolve_failure_before_first_lease_never_collects_or_quarantines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_kind: str,
    expected_rc: int,
) -> None:
    instances: list[object] = []

    class ResolveFailureTransport:
        def __init__(self, _host: HostConfig, **kwargs: object) -> None:
            self.registry = kwargs.get("remote_lease_registry")
            self.calls: list[str] = []
            instances.append(self)

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            raise RuntimeError(failure)

        def run_read_only(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("run_read_only")
            raise AssertionError("read-only follow-up must not run")

        def run(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("run")
            raise AssertionError("leased SSH must not run")

        def run_streaming(self, *_args: object, **_kwargs: object) -> int:
            self.calls.append("run_streaming")
            raise AssertionError("remote benchmark must not run")

        def scp_upload(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("scp_upload")
            raise AssertionError("upload must not run")

        def scp_download(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("scp_download")
            raise AssertionError("download must not run")

    monkeypatch.setattr(remote_run, "SSHTransport", ResolveFailureTransport)
    monkeypatch.setattr(
        remote_run, "refresh_suite_harness", lambda *_args, **_kwargs: {"changed": False},
    )
    registry = RemoteProcessLeaseRegistry()
    logs: list[str] = []

    result = _run(tmp_path, registry=registry, logs=logs)

    assert result["status"] == "failed"
    assert result["dispatch_status"] == "failed_to_dispatch"
    assert result["failure_kind"] == expected_kind
    assert result["remote_rc"] == expected_rc
    assert result["error"] == failure
    assert result["remote_dispatch_failed"] is True
    assert result["remote_dispatched"] is False
    assert result["pre_remote_mutation"] is True
    assert result["terminal_remote_failure"] is False
    assert result["primary_failure"]["schema"] == (
        "onnx-splitpoint/pre-remote-admission-failure"
    )
    assert result["primary_failure"]["stage"] == "pre_remote_mutation"
    assert result["primary_failure"]["failure_kind"] == expected_kind
    assert result["primary_failure"]["remote_rc"] == expected_rc
    assert result["primary_failure"]["primary_error"] == failure
    assert result["remote_cleanup"]["attempted"] is False
    assert registry.cancelled is False
    assert registry.active_count() == 0
    assert instances[0].calls == ["resolve_path_read_only"]

    status = json.loads(
        (Path(result["local_run_dir"]) / "run_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "failed"
    assert status["fail_reason"]["dispatch_status"] == "failed_to_dispatch"
    assert status["fail_reason"]["failure_kind"] == expected_kind
    assert status["fail_reason"]["primary_error"] == failure
    assert status["fail_reason"]["remote_leased_operation_started"] is False
    assert status["fail_reason"]["remote_mutation_started"] is False
    transcript = "\n".join(logs)
    assert "Collecting results on remote" not in transcript
    assert "Packaging results on remote" not in transcript
    assert "Downloading results" not in transcript


def test_read_only_storage_probe_connectivity_failure_is_not_storage_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StorageProbeFailureTransport:
        calls: list[str] = []

        def __init__(self, _host: HostConfig, **_kwargs: object) -> None:
            pass

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            return "/remote/splitpoint_runs"

        def run_read_only(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("run_read_only")
            return 255, "ssh: connect to host 192.168.0.102: No route to host"

        def run(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("run")
            raise AssertionError("leased SSH must not run")

        def run_streaming(self, *_args: object, **_kwargs: object) -> int:
            self.calls.append("run_streaming")
            raise AssertionError("remote benchmark must not run")

        def scp_upload(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("scp_upload")
            raise AssertionError("upload must not run")

        def scp_download(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("scp_download")
            raise AssertionError("download must not run")

    monkeypatch.setattr(remote_run, "SSHTransport", StorageProbeFailureTransport)
    monkeypatch.setattr(
        remote_run, "refresh_suite_harness", lambda *_args, **_kwargs: {"changed": False},
    )
    registry = RemoteProcessLeaseRegistry()

    result = _run(tmp_path, registry=registry, logs=[])

    assert result["status"] == "failed"
    assert result["dispatch_status"] == "failed_to_dispatch"
    assert result["failure_kind"] == "remote_connectivity_unavailable"
    assert result["terminal_remote_failure"] is False
    assert result["remote_dispatched"] is False
    assert registry.cancelled is False
    assert StorageProbeFailureTransport.calls == [
        "resolve_path_read_only", "run_read_only",
    ]


def test_local_admission_exception_before_mkdir_suppresses_remote_follow_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResolveOnlyTransport:
        calls: list[str] = []

        def __init__(self, _host: HostConfig, **_kwargs: object) -> None:
            pass

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            return "/remote/splitpoint_runs"

        def __getattr__(self, name: str):
            def unexpected(*_args: object, **_kwargs: object):
                self.calls.append(name)
                raise AssertionError(f"remote follow-up must not run: {name}")

            return unexpected

    monkeypatch.setattr(remote_run, "SSHTransport", ResolveOnlyTransport)
    monkeypatch.setattr(
        remote_run, "refresh_suite_harness", lambda *_args, **_kwargs: {"changed": False},
    )
    monkeypatch.setattr(
        remote_run,
        "_remote_run_capacity_requirement_for_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("malformed local capacity contract")
        ),
    )
    registry = RemoteProcessLeaseRegistry()

    result = _run(tmp_path, registry=registry, logs=[])

    assert result["status"] == "failed"
    assert result["dispatch_status"] == "failed_to_dispatch"
    assert result["failure_kind"] == "pre_remote_admission_failure"
    assert result["remote_rc"] == 1
    assert result["error"] == "malformed local capacity contract"
    assert result["terminal_remote_failure"] is False
    assert result["remote_dispatched"] is False
    assert registry.cancelled is False
    assert ResolveOnlyTransport.calls == ["resolve_path_read_only"]


def test_cancel_before_first_lease_has_no_remote_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelBeforeDispatchTransport:
        calls: list[str] = []

        def __init__(self, _host: HostConfig, **_kwargs: object) -> None:
            pass

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            raise remote_run.BundleCancelled("cancelled before remote admission")

        def __getattr__(self, name: str):
            def unexpected(*_args: object, **_kwargs: object):
                self.calls.append(name)
                raise AssertionError(f"remote follow-up must not run: {name}")

            return unexpected

    monkeypatch.setattr(
        remote_run, "SSHTransport", CancelBeforeDispatchTransport,
    )
    monkeypatch.setattr(
        remote_run, "refresh_suite_harness", lambda *_args, **_kwargs: {"changed": False},
    )
    registry = RemoteProcessLeaseRegistry()
    logs: list[str] = []

    result = _run(tmp_path, registry=registry, logs=logs)

    assert result["status"] == "cancelled"
    assert result["dispatch_status"] == "cancelled_before_dispatch"
    assert result["failure_kind"] == "remote_dispatch_cancelled_before_start"
    assert result["remote_rc"] == 130
    assert result["remote_dispatched"] is False
    assert result["remote_dispatch_failed"] is False
    assert result["terminal_remote_failure"] is False
    assert result["primary_failure"]["primary_error"] == (
        "cancelled before remote admission"
    )
    assert registry.cancelled is False
    assert registry.active_count() == 0
    assert CancelBeforeDispatchTransport.calls == ["resolve_path_read_only"]
    transcript = "\n".join(logs)
    assert "Collecting results on remote" not in transcript
    assert "Packaging results on remote" not in transcript
    assert "Downloading results" not in transcript


def test_read_only_probe_rc130_is_canonical_cancel_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadOnlyCancelTransport:
        calls: list[str] = []

        def __init__(self, _host: HostConfig, **_kwargs: object) -> None:
            pass

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            return "/remote/splitpoint_runs"

        def run_read_only(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("run_read_only")
            return 130, "cancelled while waiting for SSH transport"

        def __getattr__(self, name: str):
            def unexpected(*_args: object, **_kwargs: object):
                self.calls.append(name)
                raise AssertionError(f"remote follow-up must not run: {name}")

            return unexpected

    monkeypatch.setattr(remote_run, "SSHTransport", ReadOnlyCancelTransport)
    monkeypatch.setattr(
        remote_run,
        "refresh_suite_harness",
        lambda *_args, **_kwargs: {"changed": False},
    )
    registry = RemoteProcessLeaseRegistry()

    result = _run(tmp_path, registry=registry, logs=[])

    assert result["status"] == "cancelled"
    assert result["dispatch_status"] == "cancelled_before_dispatch"
    assert result["failure_kind"] == "remote_dispatch_cancelled_before_start"
    assert result["remote_rc"] == 130
    assert result["remote_dispatch_failed"] is False
    assert result["remote_dispatched"] is False
    assert result["primary_failure"]["primary_error"] == (
        "cancelled while waiting for SSH transport"
    )
    assert registry.cancelled is False
    assert registry.active_count() == 0
    assert ReadOnlyCancelTransport.calls == [
        "resolve_path_read_only", "run_read_only",
    ]


def test_resolve_rc130_is_canonical_cancel_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResolveCancelTransport:
        calls: list[str] = []

        def __init__(self, _host: HostConfig, **_kwargs: object) -> None:
            pass

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            raise RuntimeError("resolve_path_read_only failed (rc=130): cancelled")

        def __getattr__(self, name: str):
            raise AssertionError(f"remote follow-up must not run: {name}")

    monkeypatch.setattr(remote_run, "SSHTransport", ResolveCancelTransport)
    monkeypatch.setattr(
        remote_run,
        "refresh_suite_harness",
        lambda *_args, **_kwargs: {"changed": False},
    )
    result = _run(
        tmp_path, registry=RemoteProcessLeaseRegistry(), logs=[],
    )

    assert result["status"] == "cancelled"
    assert result["dispatch_status"] == "cancelled_before_dispatch"
    assert result["remote_dispatch_failed"] is False
    assert result["remote_dispatched"] is False
    assert result["remote_rc"] == 130
    assert ResolveCancelTransport.calls == ["resolve_path_read_only"]


def test_storage_admission_failure_preserves_terminal_storage_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadOnlyStorageTransport:
        calls: list[str] = []

        def __init__(self, _host: HostConfig, **_kwargs: object) -> None:
            pass

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            self.calls.append("resolve_path_read_only")
            return "/remote/splitpoint_runs"

        def run_read_only(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.calls.append("run_read_only")
            payload = {
                "read_only_mount": True,
                "permission_bits_allow": False,
                "os_access_allow": False,
                "free_bytes": 10**15,
                "free_inodes": 10**9,
            }
            return 0, "SPLITPOINT_STORAGE_JSON=" + json.dumps(payload)

        def __getattr__(self, name: str):
            def unexpected(*_args: object, **_kwargs: object):
                self.calls.append(name)
                raise AssertionError(f"remote follow-up must not run: {name}")

            return unexpected

    monkeypatch.setattr(remote_run, "SSHTransport", ReadOnlyStorageTransport)
    monkeypatch.setattr(
        remote_run, "refresh_suite_harness", lambda *_args, **_kwargs: {"changed": False},
    )
    registry = RemoteProcessLeaseRegistry()

    result = _run(tmp_path, registry=registry, logs=[])

    assert result["status"] == "failed"
    assert result["dispatch_status"] == "failed_to_dispatch"
    assert result["failure_kind"] == "terminal_remote_storage_failure"
    assert result["terminal_remote_failure"] is True
    assert result["primary_failure"]["failure_kind"] == (
        "terminal_remote_storage_failure"
    )
    assert "remote_storage_preflight_failed" in (
        result["primary_failure"]["primary_error"]
    )
    assert result["remote_dispatched"] is False
    assert registry.cancelled is False
    assert ReadOnlyStorageTransport.calls == [
        "resolve_path_read_only", "run_read_only",
    ]


def test_collect_rc70_after_remote_work_suppresses_pack_and_scp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class CollectRc70Transport:
        def __init__(self, _host: HostConfig, **kwargs: object) -> None:
            self.registry = kwargs.get("remote_lease_registry")
            self.run_commands: list[str] = []
            self.upload_count = 0
            self.download_count = 0
            instances.append(self)

        def resolve_path_read_only(self, *_args: object, **_kwargs: object) -> str:
            return "/remote/splitpoint_runs"

        def run_read_only(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            payload = {
                "read_only_mount": False,
                "permission_bits_allow": True,
                "os_access_allow": True,
                "free_bytes": 10**15,
                "free_inodes": 10**9,
            }
            return 0, "SPLITPOINT_STORAGE_JSON=" + json.dumps(payload)

        def run(self, command: str, **_kwargs: object) -> tuple[int, str]:
            self.run_commands.append(command)
            if len(self.run_commands) == 3:
                assert self.registry is not None
                self.registry.poison()
                return (
                    70,
                    "ssh: connect to host 192.168.0.102 port 22: No route to host\n"
                    "remote cleanup was not proven; lease session poisoned",
                )
            return 0, "ok"

        def run_streaming(self, *_args: object, **_kwargs: object) -> int:
            return 0

        def scp_upload(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.upload_count += 1
            return 0, "ok"

        def scp_download(self, *_args: object, **_kwargs: object) -> tuple[int, str]:
            self.download_count += 1
            raise AssertionError("SCP must be suppressed after collect rc=70")

    monkeypatch.setattr(remote_run, "SSHTransport", CollectRc70Transport)
    monkeypatch.setattr(
        remote_run, "refresh_suite_harness", lambda *_args, **_kwargs: {"changed": False},
    )
    registry = RemoteProcessLeaseRegistry()
    logs: list[str] = []

    result = _run(
        tmp_path, registry=registry, logs=logs, transfer_mode="direct",
    )

    assert result["status"] == "failed"
    assert result["dispatch_status"] == ""
    assert result["remote_rc"] == 70
    assert result["terminal_remote_failure"] is True
    assert result["primary_failure"]["failure_kind"] == (
        "terminal_remote_execution_failure"
    )
    assert result["remote_dispatched"] is True
    assert result["remote_dispatch_failed"] is False
    assert registry.cancelled is True
    transport = instances[0]
    assert len(transport.run_commands) == 3
    assert transport.upload_count == 1
    assert transport.download_count == 0
    transcript = "\n".join(logs)
    assert "Collecting results on remote" in transcript
    assert "result collection returned rc=70" in transcript
    assert "Packaging results on remote" not in transcript
    assert "Downloading results" not in transcript
