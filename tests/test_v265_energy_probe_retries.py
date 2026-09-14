from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.energy.collector import (
    _energy_repeat_aggregation_eligible,
    _energy_repeat_contract_complete,
    _energy_repeat_retry_reasons,
)
from onnx_splitpoint_tool.energy.config import EnergyDefaults
from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract
from onnx_splitpoint_tool.window_method_validation_probe import _hailo8_replay_command


def _sealed_hailo8_contract() -> dict:
    return seal_native_command_contract({
        "complete": True,
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "float32_layout_fp16",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "runner": "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "runner_sha256": "a" * 64,
        "python_executable": "/home/nx/venv/bin/python",
        "interpreter_identity": {
            "executable": "/home/nx/venv/bin/python",
            "resolved_executable": "/home/nx/venv/bin/python3.8",
            "executable_sha256": "1" * 64,
        },
        "benchmark_set": "/remote/run/resnet50/benchmark_set",
        "input_image": "/home/nx/dataset/resnet50.jpg",
        "input_image_sha256": "b" * 64,
        "artifacts": {
            "python_executable": {
                "path": "/home/nx/venv/bin/python", "sha256": "1" * 64,
            },
            "hef": {"path": "/home/nx/part1.hef", "sha256": "2" * 64},
            "engine": {"path": "/home/nx/part2.engine", "sha256": "3" * 64},
            "native_executable": {
                "path": "/home/nx/native_fifo", "sha256": "4" * 64,
            },
            "generated_cpp": {
                "path": "/home/nx/main.cpp", "sha256": "5" * 64,
            },
            "cmake": {
                "path": "/home/nx/CMakeLists.txt", "sha256": "6" * 64,
            },
            "prepared_input": {
                "path": "/home/nx/prepared.rgb", "sha256": "7" * 64,
            },
        },
        "runtime_options": {
            "frames": 100,
            "duration_s": 0.0,
            "warmup": 10,
            "queue_depth": 3,
            "hailo_format": "uint8",
            "letterbox_pad_value": 0,
            "copy_outputs": True,
            "dump_outputs": False,
            "dump_boundary": False,
            "device_id": "",
            "build": False,
            "energy_prepared_feed_capable": True,
            "prepared_input_bound": True,
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 0,
            "letterbox_pad_value_effective": 0,
        },
        "prepared_input_contract": {
            "format": "raw_rgb_uint8",
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "layout": "HWC",
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 0,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
            "pad_value_effective": 0,
        },
        "boundary_contract": {
            "boundary_layout_requested": "as_input",
            "boundary_layout_effective": "as_input",
        },
    })


def test_probe_materializes_nonce_bound_energy_runtime(tmp_path: Path) -> None:
    command_file = tmp_path / "probe_workload.sh"
    binding = _hailo8_replay_command(
        contract=_sealed_hailo8_contract(),
        ns=SimpleNamespace(
            remote_root="/remote/run",
            remote_tool_dir="/remote/tool",
            duration_s=60.0,
            hailo8_ssh="nx@hailo8",
            hailo10_ssh="",
            deepx_ssh="",
            hailo8_env="source /venv/bin/activate",
            hailo10_env="",
            deepx_env="",
        ),
        attempt_id="attempt",
        repeat_index=0,
        reconnect_attempt=0,
        command_file=command_file,
    )
    text = command_file.read_text(encoding="utf-8")
    assert "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__" in text
    assert "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__" in text
    assert "--energy-preflight-nonce" in text
    assert "--energy-preflight-attestation" in text
    assert "--warmup 0" in text
    assert "--energy-workload-only" in text
    assert "--expected-image-sha256" not in text
    assert binding["runtime_contract_preserved"]["preflight_attested"] is True
    assert binding["remote_output_root"].endswith(
        "capture___ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
    )


def test_retry_allow_list_is_marker_and_first_sample_specific(tmp_path: Path) -> None:
    dropped = {
        "command_window_request": {
            "status": "marker_contract_invalid",
            "errors": ["marker_dropped_samples_nonzero"],
        },
        "postprocess_status": "command_window_marker_contract_invalid",
    }
    reasons = _energy_repeat_retry_reasons(dropped)
    assert "marker_dropped_samples_nonzero" in reasons
    assert "command_window_marker_contract_invalid" in reasons

    missing = {
        "command_window_request": {"status": "marker_missing"},
        "command_window_trace_binding": {"status": "postprocessor_window_result_missing"},
    }
    assert _energy_repeat_retry_reasons(missing)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "collector_stderr.log").write_text(
        "First-sample barrier failed: channel is empty because sending half is closed",
        encoding="utf-8",
    )
    assert "first_sample_barrier_invalid" in _energy_repeat_retry_reasons(
        {"collector_rc": 9}, run_dir,
    )

    unrelated = {
        "final_energy_gate_status": "fail",
        "final_energy_gate_reasons": ["calibration_sha256_mismatch"],
    }
    assert _energy_repeat_retry_reasons(unrelated) == []


def test_invalid_repeat_cannot_satisfy_three_repeat_contract() -> None:
    valid = {
        "final_energy_gate_status": "pass",
        "scientific_primary_energy_status": "available",
        "scientific_primary_energy_j": 1.0,
        "window_method_comparison": {
            "status": "ok", "same_raw_trace_verified": True,
        },
    }
    invalid = {
        "final_energy_gate_status": "fail",
        "window_method_comparison": {
            "status": "ok", "same_raw_trace_verified": True,
        },
    }
    assert _energy_repeat_aggregation_eligible(valid, require_ab=True)
    assert not _energy_repeat_aggregation_eligible(invalid, require_ab=True)
    assert _energy_repeat_contract_complete(
        [dict(valid), dict(valid), dict(valid)], requested_count=3, require_ab=True,
    )
    assert not _energy_repeat_contract_complete(
        [dict(valid), dict(valid), invalid], requested_count=3, require_ab=True,
    )
    assert not _energy_repeat_contract_complete(
        [dict(valid), dict(valid)], requested_count=3, require_ab=True,
    )

    shadow_failed = dict(valid)
    shadow_failed["window_method_comparison"] = {
        "status": "legacy_window_postprocess_failed",
        "same_raw_trace_verified": False,
    }
    assert _energy_repeat_aggregation_eligible(shadow_failed, require_ab=True)


def test_retry_default_is_one_and_bounded() -> None:
    assert EnergyDefaults().invalid_repeat_max_retries == 1
