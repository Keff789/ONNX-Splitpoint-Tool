from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    seal_native_command_contract,
    successful_runtime_argv,
    verify_native_command_contract,
)
from onnx_splitpoint_tool.window_method_validation_probe import (
    _hailo8_replay_command,
    _retryable_first_sample_barrier,
    _valid_parquet_container,
)
from onnx_splitpoint_tool.workflow import runner as workflow_runner
from onnx_splitpoint_tool import resources_utils


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _contract(backend: str = "hailo8_to_trt") -> dict:
    artifacts = {
        "python_executable": {"path": "/remote/venv/bin/python", "sha256": "1" * 64},
        "engine": {"path": "/remote/part2.engine", "sha256": "2" * 64},
    }
    if backend == "hailo8_to_trt":
        artifacts.update({
            "hef": {"path": "/remote/part1.hef", "sha256": "3" * 64},
            "native_executable": {"path": "/remote/native_fifo", "sha256": "4" * 64},
            "generated_cpp": {"path": "/remote/main.cpp", "sha256": "5" * 64},
            "cmake": {"path": "/remote/CMakeLists.txt", "sha256": "6" * 64},
            "prepared_input": {"path": "/remote/prepared.rgb", "sha256": "7" * 64},
        })
        runner = "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
        setup, comparison = "orin_nx_hailo8_01", "hailo8"
        options = {
            "frames": 100, "duration_s": 0.0, "warmup": 11, "queue_depth": 7,
            "hailo_format": "uint8",
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
            "copy_outputs": False, "dump_outputs": True, "dump_boundary": True,
            "device_id": "device-0", "build": True,
            "energy_prepared_feed_capable": True,
            "prepared_input_bound": True,
        }
    elif backend == "hailo10h_to_trt":
        artifacts.update({
            "hef": {"path": "/remote/part1.hef", "sha256": "3" * 64},
            "prepared_input_00": {"path": "/remote/prepared.npy", "sha256": "7" * 64},
        })
        runner = "scripts/native_hailo10_trt_e2e_from_benchmarkset.py"
        setup, comparison = "orin_nx_hailo10_01", "hailo10h"
        options = {
            "frames": 100, "duration_s": 0.0, "warmup": 11, "queue_depth": 7,
            "inflight": 9, "producer_impl": "async_fifo", "quantized_inputs": True,
            "quantized_outputs": True, "copy_outputs": True,
            "dump_outputs": False, "dump_boundary": False, "build": False,
            "part1_onnx_used": False,
            "canonical_input_slot_names": ["images"],
            "canonical_output_slot_names": ["boundary"],
            "prepared_input_bound": True,
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
        }
    else:
        artifacts.update({
            "dxnn": {"path": "/remote/part1.dxnn", "sha256": "3" * 64},
            "prepared_input": {"path": "/remote/prepared_input.npy", "sha256": "7" * 64},
        })
        runner = "scripts/native_deepx_trt_e2e_from_benchmarkset.py"
        setup, comparison = "orin_nx_deepx_m1_01", "deepx"
        options = {
            "frames": 100, "duration_s": 0.0, "warmup": 11, "queue_depth": 7,
            "dump_outputs": False, "dump_boundary": False, "build": False,
            "prepared_input_bound": True,
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
        }
    payload = {
        "complete": True,
        "backend": backend,
        "model": "resnet50",
        "case": "b052",
        "precision": "uint8_cast_fp16",
        "setup_id": setup,
        "comparison_backend": comparison,
        "runner": runner,
        "runner_sha256": "a" * 64,
        "python_executable": "/remote/venv/bin/python",
        "interpreter_identity": {
            "executable": "/remote/venv/bin/python",
            "resolved_executable": "/remote/venv/bin/python3.8",
            "executable_sha256": "1" * 64,
        },
        "benchmark_set": "/remote/resnet50/benchmark_set",
        "input_image": "/remote/dataset/image.jpg",
        "input_image_sha256": "b" * 64,
        "artifacts": artifacts,
        "runtime_options": options,
        "boundary_contract": {
            "boundary_layout_requested": "as_input",
            "boundary_layout_effective": "as_input",
            "dequant_scale": 0.25,
            "dequant_zero_point": 3.0,
        },
    }
    if backend == "deepx_to_trt":
        payload["prepared_input_contract"] = {
            "format": "numpy_npy_v1",
            "shape": [1, 224, 224, 3],
            "dtype": "uint8",
            "layout": "NHWC",
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
        }
    elif backend == "hailo8_to_trt":
        payload["prepared_input_contract"] = {
            "format": "raw_rgb_uint8",
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "layout": "HWC",
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
            "pad_value_effective": 0,
        }
    elif backend == "hailo10h_to_trt":
        payload["prepared_input_contract"] = {
            "format": "numpy_npy_v1",
            "preprocess": "exact_performance_prepared_tensor_persisted",
            "normalization": "hef_quant_info_from_imagenet_float32",
            "entries": [{
                "name": "images", "artifact_name": "prepared_input_00",
                "shape": [1, 224, 224, 3], "dtype": "uint8",
                "c_contiguous": True,
            }],
            "slot_order": ["images"],
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode": "resize",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
            "pad_value_effective": 0,
        }
    return seal_native_command_contract(payload)


def test_contract_hash_and_identity_tampering_fail_closed() -> None:
    contract = _contract()
    verified, reason = verify_native_command_contract(
        contract,
        expected_identity={"backend": "hailo8_to_trt", "model": "resnet50", "case": "b052"},
    )
    assert verified is not None
    assert reason == "hash_schema_identity_and_artifacts_verified"

    tampered = dict(contract)
    tampered["input_image"] = "/remote/dataset/other.jpg"
    rejected, reason = verify_native_command_contract(tampered)
    assert rejected is None
    assert reason == "native_command_contract_sha256_mismatch"


def test_contract_requires_backend_specific_artifact_guards() -> None:
    contract = _contract("deepx_to_trt")
    payload = dict(contract)
    payload.pop("contract_sha256")
    payload["artifacts"] = dict(payload["artifacts"])
    payload["artifacts"].pop("dxnn")
    incomplete = seal_native_command_contract(payload)
    verified, reason = verify_native_command_contract(incomplete)
    assert verified is None
    assert reason == "native_command_contract_required_artifact_missing:dxnn"


def test_hailo8_replay_preserves_runtime_options_and_hash_guards() -> None:
    contract = _contract()
    argv = successful_runtime_argv(
        contract, duration_s=23.5, fresh_output_root="/fresh/repeat_000", remote_tool_dir="/tool",
    )
    command = " ".join(argv)
    assert argv[0] == "/remote/venv/bin/python"
    assert "/tool/scripts/native_hailo_trt_fifo_from_benchmarkset.py" in argv
    assert "--duration-s 23.5" in command
    assert "--warmup 11" in command and "--queue-depth 7" in command
    assert "--letterbox-pad-value 114" in command
    assert "--expected-executable-sha256 " + "4" * 64 in command
    assert "--no-copy-outputs" in argv
    assert "--dump-outputs" in argv and "--dump-boundary" in argv
    assert "/fresh/repeat_000/native_fifo_results.json" in argv
    assert "--no-build" in argv


def test_reconnect_attempts_have_distinct_fail_closed_remote_output_roots(tmp_path: Path) -> None:
    contract = _contract()
    ns = argparse.Namespace(
        remote_root="/remote", remote_tool_dir="/tool", duration_s=10.0,
        hailo8_ssh="nx@hailo8", hailo8_env="source /env",
        hailo10_ssh="", hailo10_env="", deepx_ssh="", deepx_env="",
    )
    first = _hailo8_replay_command(
        contract=contract, ns=ns, attempt_id="probe", repeat_index=0,
        reconnect_attempt=0, command_file=tmp_path / "a0.sh",
    )
    second = _hailo8_replay_command(
        contract=contract, ns=ns, attempt_id="probe", repeat_index=0,
        reconnect_attempt=1, command_file=tmp_path / "a1.sh",
    )
    assert first["remote_output_root"] != second["remote_output_root"]
    assert "/attempt_00/" in first["remote_output_root"]
    assert "/attempt_01/" in second["remote_output_root"]
    assert first["remote_output_root"].endswith(
        "capture___ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
    )
    assert second["remote_output_root"].endswith(
        "capture___ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
    )
    assert "test ! -e" in (tmp_path / "a0.sh").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("backend", "artifact_flag"),
    [("hailo10h_to_trt", "--expected-hef-sha256"), ("deepx_to_trt", "--expected-dxnn-sha256")],
)
def test_python_native_replays_use_fresh_outputs_and_artifact_guards(
    backend: str, artifact_flag: str,
) -> None:
    contract = _contract(backend)
    argv = successful_runtime_argv(
        contract, duration_s=10, fresh_output_root="/fresh/replay", remote_tool_dir="/tool",
    )
    assert argv[argv.index("--out-dir") + 1] == "/fresh/replay"
    assert artifact_flag in argv
    assert argv[argv.index("--expected-engine-sha256") + 1] == "2" * 64
    assert argv[argv.index("--source-contract-sha256") + 1] == contract["contract_sha256"]
    assert argv[argv.index("--task") + 1] == "classification"
    assert argv[argv.index("--preprocess-mode") + 1] == "auto"
    assert argv[argv.index("--letterbox-pad-value") + 1] == "114"


def test_parquet_container_rejects_four_byte_placeholder(tmp_path: Path) -> None:
    placeholder = tmp_path / "placeholder.parquet"
    placeholder.write_bytes(b"PAR1")
    assert _valid_parquet_container(placeholder) == (False, "too_small_for_parquet_footer")

    trace = tmp_path / "trace.parquet"
    trace.write_bytes(b"PAR1" + b"payload" + b"PAR1")
    assert _valid_parquet_container(trace) == (True, "valid_parquet_container")


def test_retry_is_bounded_to_first_sample_barrier_failures() -> None:
    assert _retryable_first_sample_barrier("First-sample barrier failed after reconnect")
    assert _retryable_first_sample_barrier("channel is empty because sending half is closed")
    assert not _retryable_first_sample_barrier("workload exited with return code 9")
    assert not _retryable_first_sample_barrier("parquet footer invalid")


def test_remote_work_unit_wrapper_is_python38_argparse_compatible() -> None:
    source = (ROOT / "scripts" / "run_and_report_work_units.py").read_text(encoding="utf-8")
    assert "argparse.BooleanOptionalAction" not in source
    assert '"--follow-reports"' in source
    assert '"--no-follow-reports"' in source


def test_resources_utils_falls_back_without_traversable_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(resources_utils.importlib_resources, "files", raising=False)
    ref = resources_utils._resource_ref("__init__.py")
    assert isinstance(ref, Path)
    assert ref.is_file()
    with resources_utils.resource_path("__init__.py") as path:
        assert path == ref


def test_capability_nonzero_is_checked_before_stale_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_script(
        "v2621_native_fifo_capability_fail_closed",
        ROOT / "scripts" / "native_fifo_smoke_matrix.py",
    )
    benchmark_set = tmp_path / "benchmark_set"
    analysis = benchmark_set / "analysis_tables"
    analysis.mkdir(parents=True)
    (analysis / "native_fifo_capability_report.json").write_text(
        json.dumps({"cases": [{"case_id": "b052", "native_fifo_supported": True}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod, "_run",
        lambda *_args, **_kwargs: {
            "rc": 9, "timed_out": False, "stdout_tail": "", "stderr_tail": "probe failed",
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "native_fifo_smoke_matrix.py", "--benchmark-set", str(benchmark_set),
    ])
    assert mod.main() == 9
    matrix = json.loads((analysis / "native_fifo_smoke_matrix.json").read_text(encoding="utf-8"))
    assert matrix["ok"] is False
    assert matrix["failure_reason"] == "native_fifo_capability_report_nonzero_exit"
    assert matrix["cases"] == []


def test_hailo8_binary_sha_mismatch_fails_before_inference(tmp_path: Path) -> None:
    mod = _load_script(
        "v2621_hailo8_contract_test",
        ROOT / "scripts" / "native_hailo_trt_fifo_from_benchmarkset.py",
    )
    files = {}
    for name in ("hef", "engine", "image", "executable"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        files[name] = path
    args = argparse.Namespace(
        expected_runner_sha256="", expected_image_sha256="", expected_hef_sha256="",
        expected_engine_sha256="", expected_executable_sha256="0" * 64,
        expected_boundary_layout="", source_contract_sha256="source",
    )
    with pytest.raises(RuntimeError, match="native_executable_sha256"):
        mod._verify_replay_expectations(
            args, hef=files["hef"], engine=files["engine"], image=files["image"],
            executable=files["executable"], boundary={"boundary_layout_effective": "as_input"},
        )


def test_remote_asset_hash_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            return subprocess.CompletedProcess([], 0, "", "")
        return subprocess.CompletedProcess(
            [], 0, json.dumps({"sha256": "0" * 64, "missing_tokens": []}) + "\n", "",
        )

    monkeypatch.setattr(workflow_runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="remote package asset verification failed"):
        workflow_runner._sync_remote_package_asset_v263(
            ssh="nx@host", remote_tool_dir="/remote/tool",
            relative_path="onnx_splitpoint_tool/native_command_contract.py",
        )


@pytest.mark.parametrize(
    "completed",
    [
        subprocess.CompletedProcess([], 8, json.dumps({"ok": False, "imported_path": "/old/module.py"}) + "\n", ""),
        subprocess.CompletedProcess([], 127, "", "python: command not found"),
    ],
)
def test_remote_module_wrong_path_or_missing_interpreter_is_fatal(
    monkeypatch: pytest.MonkeyPatch, completed: subprocess.CompletedProcess,
) -> None:
    monkeypatch.setattr(workflow_runner.subprocess, "run", lambda *_args, **_kwargs: completed)
    with pytest.raises(RuntimeError, match="remote Python module binding verification failed"):
        workflow_runner._verify_remote_module_binding_v263(
            ssh="nx@host", remote_tool_dir="/remote/tool", remote_env="source /remote/env",
            module_name="onnx_splitpoint_tool.native_command_contract",
            relative_path="onnx_splitpoint_tool/native_command_contract.py",
            expected_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "name",
    [
        "native_hailo_trt_fifo_from_benchmarkset.py",
        "native_hailo10_trt_e2e_from_benchmarkset.py",
        "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_fifo_eval_runner.py",
        "native_fifo_smoke_matrix.py",
        "native_fifo_capability_report.py",
        "native_producer_final_report.py",
        "run_and_report_work_units.py",
        "validate_output_dumps.py",
        "update_evalset_native_producers.py",
    ],
)
def test_remote_script_mirrors_are_byte_identical(name: str) -> None:
    assert (ROOT / "scripts" / name).read_bytes() == (
        ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
    ).read_bytes()
