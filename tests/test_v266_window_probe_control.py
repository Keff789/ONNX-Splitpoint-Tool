from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy.config import EnergyDefaults
from onnx_splitpoint_tool import window_method_validation_probe as probe


ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_setup_acquisition_is_exactly_one_capture_without_ab(
    tmp_path: Path, monkeypatch,
) -> None:
    cli = _load_script(
        ROOT / "scripts" / "energy_measurement_cli.py",
        "v266_energy_measurement_cli",
    )
    defaults = EnergyDefaults()
    setup = SimpleNamespace(
        enabled=True,
        urecs_address="127.0.0.1:3000",
    )
    captured: dict = {}

    def fake_measurement(*_args, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "status": "ok_caller_managed_repeat_capture"}

    monkeypatch.setattr(cli, "load_energy_defaults", lambda: defaults)
    monkeypatch.setattr(cli, "get_setup_energy", lambda _setup_id: setup)
    monkeypatch.setattr(cli, "check_energy_tools", lambda _defaults: {"collector_found": True})
    monkeypatch.setattr(cli, "run_fast_firmware_measurement", fake_measurement)
    args = SimpleNamespace(
        setup_id="setup",
        acquire=True,
        sleep_s=1.0,
        out=str(tmp_path / "setup_probe"),
        workdir="",
    )

    assert cli.cmd_test_setup(args) == 0
    assert captured["run_count"] == 1
    assert captured["exact_run_count"] is True
    assert captured["compare_legacy_window"] is False


def test_probe_owns_three_exact_repeats_with_unique_outputs(
    tmp_path: Path, monkeypatch,
) -> None:
    summary = tmp_path / "summary.json"
    _write_json(summary, {
        "rows": [{
            "ok": True,
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b052",
            "precision": "fp16",
        }],
    })
    target = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "fp16",
        "setup_id": "hailo8",
        "measure_command": "python energy_measurement_cli.py measure --runs 3 --out stale",
    }
    contract = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "fp16",
        "setup_id": "hailo8",
        "contract_sha256": "a" * 64,
    }
    measurement_commands: list[list[str]] = []
    remote_roots: list[str] = []

    monkeypatch.setattr(
        probe,
        "_verified_native_contract",
        lambda _row, _target: (contract, "verified"),
    )

    def fake_binding(*, attempt_id, repeat_index, reconnect_attempt, command_file, **_kwargs):
        command_file.parent.mkdir(parents=True, exist_ok=True)
        command_file.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        remote = (
            f"/remote/{attempt_id}/repeat_{repeat_index:03d}/"
            f"attempt_{reconnect_attempt:02d}/capture_nonce"
        )
        remote_roots.append(remote)
        return {
            "command_file": str(command_file),
            "command_file_sha256": hashlib.sha256(command_file.read_bytes()).hexdigest(),
            "remote_output_root": remote,
            "source_contract_sha256": "a" * 64,
        }

    monkeypatch.setattr(probe, "_hailo8_replay_command", fake_binding)
    monkeypatch.setattr(probe, "_hailo8_remote_preflight", lambda *_args, **_kwargs: ["remote-preflight"])

    def fake_run(command, *, timeout=None):
        parts = [str(item) for item in command]
        if any(item.endswith("native_producer_energy_plan.py") for item in parts):
            plan_dir = Path(parts[parts.index("--out-dir") + 1])
            _write_json(plan_dir / "native_producer_energy_plan.json", {"rows": [target]})
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if parts == ["remote-preflight"]:
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

        measurement_commands.append(parts)
        assert parts[parts.index("--runs") + 1] == "1"
        assert "--exact-run-count" in parts
        assert parts[parts.index("--invalid-repeat-max-retries") + 1] == "0"
        out = Path(parts[parts.index("--out") + 1])
        run_dir = out / "run_000"
        storage = run_dir / "collector_storage"
        storage.mkdir(parents=True, exist_ok=True)
        trace = storage / "trace.parquet"
        trace.write_bytes(b"PAR1" + b"v266-test" + b"PAR1")
        digest = hashlib.sha256(trace.read_bytes()).hexdigest()
        _write_json(run_dir / "window_method_comparison.json", {
            "status": "ok",
            "same_raw_trace_verified": True,
            "raw_trace": {
                "trace_path": str(trace),
                "request_sha256": digest,
                "sha256_before_legacy_postprocess": digest,
                "sha256_after_legacy_postprocess": digest,
            },
            "legacy_minus_command_window": {
                "energy_j": {"relative_percent": 1.0},
            },
        })
        _write_json(out / "energy_aggregate.json", {
            "runs": [{
                "run_index": 0,
                "started_at": 1,
                "collector_rc": 0,
                "workload_command_rc": 0,
                "storage_dir": str(storage),
            }],
        })
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(probe, "_run", fake_run)
    report, rc = probe.run_probe(probe.build_parser().parse_args([
        "--summary", str(summary),
        "--out-dir", str(tmp_path / "probe"),
        "--hailo8-ssh", "nx@hailo8",
        "--repeats", "3",
        "--max-reconnect-retries", "2",
        "--reconnect-backoff-s", "0",
        "--strict",
    ]))

    assert rc == 0
    assert report["complete"] is True
    assert report["decision_capable"] is True
    assert report["started_repeat_count"] == 3
    assert report["successful_comparison_count"] == 3
    assert len(measurement_commands) == 3
    assert len(remote_roots) == len(set(remote_roots)) == 3
    policy = report["repeat_execution_policy"]
    assert policy["repeat_control"] == "caller_managed_exact"
    assert policy["collector_runs_per_outer_repeat"] == 1
    assert policy["collector_runs_per_physical_attempt"] == 1
    assert policy["max_physical_collector_attempts_per_outer_repeat"] == 2
    assert policy["valid_capture_count_required_per_outer_repeat"] == 1
    assert policy["collector_invalid_repeat_retries"] == 0
    assert policy["collector_runs_per_physical_attempt"] == 1
    assert policy["max_physical_collector_attempts_per_outer_repeat"] == 2
    assert policy["valid_capture_count_required_per_outer_repeat"] == 1
    assert policy["unique_remote_output_root_count"] == 3
    assert policy["requested_reconnect_retries"] == 2
    assert policy["max_reconnect_retries"] == 1
    assert policy["reconnect_retry_clamped_to_one"] is True
    assert policy["reconnect_retry_suppressed_by_exact_repeat_contract"] is False
    assert policy["outer_retry_attempt_count"] == 0


@pytest.mark.parametrize("recover", [True, False], ids=["retry_recovers", "retry_exhausted"])
def test_bounded_outer_acquisition_retry_preserves_every_physical_attempt(
    tmp_path: Path, monkeypatch, recover: bool,
) -> None:
    summary = tmp_path / "summary.json"
    _write_json(summary, {
        "rows": [{
            "ok": True,
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b052",
            "precision": "fp16",
        }],
    })
    target = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "fp16",
        "setup_id": "hailo8",
        "measure_command": "python energy_measurement_cli.py measure --runs 3 --out stale",
    }
    contract = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "fp16",
        "setup_id": "hailo8",
        "contract_sha256": "a" * 64,
    }
    measurement_commands: list[list[str]] = []
    remote_roots: list[str] = []

    monkeypatch.setattr(
        probe,
        "_verified_native_contract",
        lambda _row, _target: (contract, "verified"),
    )

    def fake_binding(*, attempt_id, repeat_index, reconnect_attempt, command_file, **_kwargs):
        command_file.parent.mkdir(parents=True, exist_ok=True)
        command_file.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        remote = (
            f"/remote/{attempt_id}/repeat_{repeat_index:03d}/"
            f"attempt_{reconnect_attempt:02d}/capture_nonce"
        )
        remote_roots.append(remote)
        return {
            "command_file": str(command_file),
            "command_file_sha256": hashlib.sha256(command_file.read_bytes()).hexdigest(),
            "remote_output_root": remote,
            "source_contract_sha256": "a" * 64,
        }

    monkeypatch.setattr(probe, "_hailo8_replay_command", fake_binding)
    monkeypatch.setattr(
        probe, "_hailo8_remote_preflight",
        lambda *_args, **_kwargs: ["remote-preflight"],
    )

    def fake_run(command, *, timeout=None):
        parts = [str(item) for item in command]
        if any(item.endswith("native_producer_energy_plan.py") for item in parts):
            plan_dir = Path(parts[parts.index("--out-dir") + 1])
            _write_json(
                plan_dir / "native_producer_energy_plan.json",
                {"rows": [target]},
            )
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if parts == ["remote-preflight"]:
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

        assert parts[parts.index("--runs") + 1] == "1"
        assert "--exact-run-count" in parts
        assert parts[parts.index("--invalid-repeat-max-retries") + 1] == "0"
        out = Path(parts[parts.index("--out") + 1])
        run_dir = out / "run_000"
        storage = run_dir / "collector_storage"
        storage.mkdir(parents=True, exist_ok=True)
        physical_attempt = len(measurement_commands)
        measurement_commands.append(parts)
        should_fail = not recover or physical_attempt == 0
        row = {
            "run_index": 0,
            "started_at": 1,
            "collector_rc": 0,
            "workload_command_rc": 0,
            "storage_dir": str(storage),
        }
        if should_fail:
            rejection = {
                "status": "marker_contract_invalid",
                "errors": ["marker_dropped_samples_nonzero"],
            }
            row["command_window_request"] = rejection
            _write_json(run_dir / "command_window_request_rejection.json", rejection)
            _write_json(out / "energy_aggregate.json", {"runs": [row]})
            return {
                "rc": 3,
                "stdout_tail": "marker_dropped_samples_nonzero",
                "stderr_tail": "",
            }

        trace = storage / "trace.parquet"
        trace.write_bytes(b"PAR1" + b"v266-recovered" + b"PAR1")
        digest = hashlib.sha256(trace.read_bytes()).hexdigest()
        _write_json(run_dir / "window_method_comparison.json", {
            "status": "ok",
            "same_raw_trace_verified": True,
            "raw_trace": {
                "trace_path": str(trace),
                "request_sha256": digest,
                "sha256_before_legacy_postprocess": digest,
                "sha256_after_legacy_postprocess": digest,
            },
        })
        _write_json(out / "energy_aggregate.json", {"runs": [row]})
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(probe, "_run", fake_run)
    report, rc = probe.run_probe(probe.build_parser().parse_args([
        "--summary", str(summary),
        "--out-dir", str(tmp_path / "probe"),
        "--hailo8-ssh", "nx@hailo8",
        "--repeats", "3",
        "--max-reconnect-retries", "5",
        "--reconnect-backoff-s", "0",
        "--strict",
    ]))

    policy = report["repeat_execution_policy"]
    assert policy["requested_reconnect_retries"] == 5
    assert policy["max_reconnect_retries"] == 1
    assert policy["collector_invalid_repeat_retries"] == 0
    assert policy["reconnect_retry_clamped_to_one"] is True
    assert policy["reconnect_retry_suppressed_by_exact_repeat_contract"] is False
    assert policy["outer_retry_attempt_count"] == 1
    expected_physical_attempts = 4 if recover else 2
    assert len(measurement_commands) == expected_physical_attempts
    assert len(remote_roots) == len(set(remote_roots)) == expected_physical_attempts
    first_history = report["per_repeat"][0]["attempt_history"]
    assert len(first_history) == 2
    assert first_history[0]["acquisition_integrity_retry_reasons"] == [
        "marker_dropped_samples_nonzero"
    ]
    assert first_history[0]["physical_attempt_result"][
        "command_window_request"
    ]["errors"] == ["marker_dropped_samples_nonzero"]
    assert first_history[0]["measurement_directory"] != first_history[1][
        "measurement_directory"
    ]
    if recover:
        assert rc == 0
        assert report["complete"] is True
        assert report["started_repeat_count"] == 3
        assert report["successful_comparison_count"] == 3
        assert policy["outer_retry_recovered_count"] == 1
        assert policy["abort_reason"] == ""
    else:
        assert rc == 4
        assert report["complete"] is False
        assert report["started_repeat_count"] == 1
        assert policy["outer_retry_recovered_count"] == 0
        assert policy["abort_reason"] == (
            "acquisition_integrity_retry_exhausted:"
            "marker_dropped_samples_nonzero"
        )


def test_probe_validation_is_nonblocking_in_all_modes(
    tmp_path: Path, monkeypatch,
) -> None:
    coordinator = _load_script(
        ROOT / "scripts" / "run_evalrun_native_producer_variants.py",
        "v266_variant_coordinator_policy",
    )

    def fake_run(command, **_kwargs):
        out = Path(command[command.index("--out-dir") + 1])
        _write_json(out / "window_method_validation_probe.json", {
            "ok": False,
            "complete": False,
            "status": "incomplete_probe_measurement_or_comparison_failed",
            "started_repeat_count": 1,
            "successful_comparison_count": 1,
        })
        return {"rc": 4, "stdout_tail": "", "stderr_tail": "probe incomplete"}

    monkeypatch.setattr(coordinator, "_run", fake_run)
    monkeypatch.setattr(coordinator, "_energy_contract_context", lambda *_args, **_kwargs: {
        "physical_scope": "MB",
        "window_label": "command",
        "calibration_manifest": "",
        "calibration_sha256": "",
    })

    def cfg(mode: str) -> dict:
        return {
            "_workflow_context": {
                "campaign": {
                    "mode": "development" if mode in {"smoke", "standard"} else "final"
                },
                "execution_preset": {"id": mode},
            },
            "energy": {
                "window_method_validation_probe": {
                    "enabled": True,
                    "strict": True,
                    "repeats": 3,
                },
            },
        }

    smoke, smoke_fatal = coordinator._run_window_method_probe(
        cfg=cfg("smoke"),
        summary_json=tmp_path / "summary.json",
        validation_summary=tmp_path / "validation.json",
        run_dir=tmp_path / "smoke_run",
        reports=tmp_path / "smoke_reports",
    )
    assert smoke["strict_validation_failure"] is True
    assert smoke["workflow_blocking_requested"] is False
    assert smoke["strict_failure"] is False
    assert smoke_fatal is False

    standard, standard_fatal = coordinator._run_window_method_probe(
        cfg=cfg("standard"),
        summary_json=tmp_path / "summary.json",
        validation_summary=tmp_path / "validation.json",
        run_dir=tmp_path / "standard_run",
        reports=tmp_path / "standard_reports",
    )
    assert standard["strict_validation_failure"] is True
    assert standard["workflow_blocking_requested"] is False
    assert standard["strict_failure"] is False
    assert standard_fatal is False

    final, final_fatal = coordinator._run_window_method_probe(
        cfg=cfg("final"),
        summary_json=tmp_path / "summary.json",
        validation_summary=tmp_path / "validation.json",
        run_dir=tmp_path / "final_run",
        reports=tmp_path / "final_reports",
    )
    assert final["strict_validation_failure"] is True
    assert final["workflow_blocking_requested"] is False
    assert final["workflow_blocking_scope"] == "none"
    assert final["strict_failure"] is False
    assert final_fatal is False
