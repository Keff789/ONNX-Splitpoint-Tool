from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from typing import Any

import numpy as np

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
DEEPX_SUITE = (
    ROOT
    / "onnx_splitpoint_tool"
    / "resources"
    / "templates"
    / "benchmark_suite.py.txt"
)
TRT_COMPLETED_RUNNER = ROOT / "scripts" / "native_trt_full_completed_hotloop.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _suite_contract_namespace() -> dict[str, Any]:
    source = DEEPX_SUITE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = {
        "_read_json",
        "_deepx_model_family",
        "_deepx_contract_model_id",
        "_deepx_endpoint_json_sha256",
        "_deepx_authoritative_endpoint_attestation",
        "_deepx_bind_authoritative_endpoint_contract",
    }
    definitions = [
        ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected
    ]
    namespace: dict[str, Any] = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
    }
    exec(
        "from __future__ import annotations\n" + "\n\n".join(definitions),
        namespace,
    )
    assert selected <= namespace.keys()
    return namespace


def _write_deepx_endpoint_contract(root: Path, model_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "output_contracts.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "contracts": [
                    {
                        "backend": "deepx_m1",
                        "variant": "full",
                        "model_id": model_id,
                        "contract_status": "recorded",
                        "endpoint_mode": "raw_detection_head",
                        "host_tail_required": True,
                        "postprocessing_required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _raw_artifact_contract(*, model_id: str = "") -> dict[str, Any]:
    value: dict[str, Any] = {
        "input": {
            "task": "detection",
            "shape": [1, 3, 64, 64],
            "layout": "NCHW",
            "dtype": "float32",
        },
        "outputs": [
            {"name": "output"},
            {"name": "clone_1"},
            {"name": "clone_2"},
        ],
    }
    if model_id:
        value["model_id"] = model_id
    return value


def test_deepx_yolov7_model_identity_comes_from_authoritative_attestation(
    tmp_path: Path,
) -> None:
    namespace = _suite_contract_namespace()
    # The numeric parent reproduces the failing remote attempt layout:
    # .../<run>/1/suite. It must never become a decoder model identity.
    suite_root = tmp_path / "remote_attempts" / "1" / "suite"
    _write_deepx_endpoint_contract(suite_root, "yolov7_paper")

    run = {"benchmark_task": "detection"}
    bound = namespace["_deepx_bind_authoritative_endpoint_contract"](
        suite_root,
        run,
        _raw_artifact_contract(),
    )

    assert bound["endpoint_contract_binding_status"] == "attested"
    assert bound["model_id"] == "yolov7_paper"
    assert bound["endpoint_semantic_attestation"]["model_id"] == "yolov7_paper"
    assert namespace["_deepx_contract_model_id"](
        suite_root, run, bound,
    ) == "yolov7_paper"
    assert "1" not in {
        bound["model_id"],
        namespace["_deepx_contract_model_id"](suite_root, run, bound),
    }


def test_deepx_authoritative_model_identity_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    namespace = _suite_contract_namespace()
    suite_root = tmp_path / "remote_attempts" / "1" / "suite"
    _write_deepx_endpoint_contract(suite_root, "yolov7_paper")

    bound = namespace["_deepx_bind_authoritative_endpoint_contract"](
        suite_root,
        {"benchmark_task": "detection"},
        _raw_artifact_contract(model_id="yolo26s"),
    )

    assert bound["endpoint_contract_binding_status"] == "conflict"
    assert "model_id" in bound["endpoint_contract_binding_conflicts"]
    assert not namespace["_deepx_contract_model_id"](
        suite_root, {"benchmark_task": "detection"}, bound,
    )


def test_deepx_model_identity_never_falls_back_to_multimodel_path(
    tmp_path: Path,
) -> None:
    namespace = _suite_contract_namespace()
    suite_root = (
        tmp_path / "resnet_yolo26s_yolo7_20260805" / "1" / "suite"
    )
    suite_root.mkdir(parents=True)

    assert not namespace["_deepx_contract_model_id"](
        suite_root, {"benchmark_task": "detection"}, {},
    )


def _yolov7_outputs() -> dict[str, np.ndarray]:
    return {
        "output": np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        "clone_1": np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        "clone_2": np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    }


def _frozen_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    outputs = _yolov7_outputs()
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    result = FrozenDetectionPostprocessor(contract).process(
        outputs,
        original_wh=[80, 60],
    )
    return contract, result


def _write_runtime_input(tmp_path: Path) -> tuple[Path, Path, str]:
    tensor = np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)
    tensor_path = tmp_path / "runtime_input.bin"
    tensor_path.write_bytes(np.ascontiguousarray(tensor).tobytes())
    tensor_sha = _sha256(tensor_path)
    manifest = tmp_path / "native_full_input_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/native-full-input-dump",
                "schema_version": 1,
                "case": "full",
                "runtime_input_name": "images",
                "runtime_input_shape": [1, 3, 4, 4],
                "runtime_input_dtype": "float32",
                "runtime_input_file": tensor_path.name,
                "runtime_input_sha256": tensor_sha,
                "runtime_input_bytes": int(tensor.nbytes),
            }
        ),
        encoding="utf-8",
    )
    return manifest, tensor_path, tensor_sha


def _argument(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_tensorrt_completed_hotloop_measures_decoded_nms_once_per_work_unit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    outputs = _yolov7_outputs()
    frozen, _frozen_result = _frozen_evidence()
    instances: list[Any] = []

    class FakeNativeTRT:
        def __init__(self, _engine: Path) -> None:
            self.inputs = ["images"]
            self.shapes = {"images": (1, 3, 4, 4)}
            self.dtypes = {"images": np.dtype(np.float32)}
            self.prepare_count = 0
            self.run_count = 0
            self.closed = False
            instances.append(self)

        def prepare_inputs(self, feeds: dict[str, np.ndarray]) -> None:
            assert list(feeds) == ["images"]
            assert feeds["images"].shape == (1, 3, 4, 4)
            self.prepare_count += 1

        def run_prepared(self) -> dict[str, np.ndarray]:
            self.run_count += 1
            return outputs

        def close(self) -> None:
            self.closed = True

    fake_native_module = ModuleType(
        "native_hailo10_trt_e2e_from_benchmarkset"
    )
    fake_native_module.NativeTRT = FakeNativeTRT
    monkeypatch.setitem(
        sys.modules,
        "native_hailo10_trt_e2e_from_benchmarkset",
        fake_native_module,
    )
    runner = _load_script(
        "_osp_v270i_trt_completed_hotloop",
        TRT_COMPLETED_RUNNER,
    )

    manifest, _tensor_path, tensor_sha = _write_runtime_input(tmp_path)
    engine = tmp_path / "full.engine"
    engine.write_bytes(b"quality-sealed-engine")
    report = tmp_path / "completed_hotloop.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRT_COMPLETED_RUNNER),
            "--engine",
            str(engine),
            "--input-manifest",
            str(manifest),
            "--frozen-postprocess-contract-json",
            json.dumps(frozen, sort_keys=True, separators=(",", ":")),
            "--source-endpoint-contract-hash",
            "a" * 64,
            "--frames",
            "2",
            "--warmup",
            "1",
            "--duration-s",
            "0",
            "--json-out",
            str(report),
            "--expected-runner-sha256",
            _sha256(TRT_COMPLETED_RUNNER),
            "--expected-engine-sha256",
            _sha256(engine),
            "--expected-input-manifest-sha256",
            _sha256(manifest),
            "--expected-runtime-input-sha256",
            tensor_sha,
        ],
    )

    assert runner.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["benchmark_kind"] == (
        "tensorrt_full_completed_task_hotloop"
    )
    assert payload["e2e_scope"] == "full_task_pipeline"
    assert payload["measurement_concurrency"] == 1
    assert payload["completed_task_stage"] == "decoded_nms"
    assert payload["completed_task_contract_family"] == "decoded_nms"
    assert payload["completed_frames"] == 2
    assert payload["postprocess_completed_frames"] == 2
    assert payload["postprocess_completion_verified"] is True
    assert payload["frozen_host_postprocess_contract_sha256"] == (
        frozen["contract_sha256"]
    )
    assert payload["completed_task_endpoint_attestation"]["attested"] is True
    assert payload["latency_semantics"] == (
        "prepared_input_h2d_engine_d2h_sync_decode_class_aware_nms"
    )
    assert len(instances) == 1
    assert instances[0].prepare_count == 1
    # One untimed structural probe + one warmup + two measured work units.
    assert instances[0].run_count == 4
    assert instances[0].closed is True


def _completed_payload_from_command(
    command: list[str],
    *,
    completed: int,
    preflight_verified: bool,
) -> dict[str, Any]:
    frozen = json.loads(
        _argument(command, "--frozen-postprocess-contract-json")
    )
    frozen_result = FrozenDetectionPostprocessor(frozen).process(
        _yolov7_outputs(),
        original_wh=[80, 60],
    )
    source_hash = _argument(
        command,
        "--source-endpoint-contract-hash",
    )
    attestation = build_completed_detection_endpoint_attestation(
        frozen,
        frozen_result,
        completed_frames=completed,
        postprocess_completed_frames=completed,
        source_endpoint_contract_hash=source_hash,
    )
    return {
        "ok": True,
        "status": "ok",
        "benchmark_kind": "tensorrt_full_completed_task_hotloop",
        "e2e_scope": "full_task_pipeline",
        "measurement_concurrency": 1,
        "engine_sha256": _argument(
            command, "--expected-engine-sha256",
        ),
        "input_manifest_sha256": _argument(
            command, "--expected-input-manifest-sha256",
        ),
        "runtime_input_sha256": _argument(
            command, "--expected-runtime-input-sha256",
        ),
        "runtime_input_binding_verified": True,
        "completed_frames": completed,
        "completed_work_units": completed,
        "completed_work_units_source": (
            "tensorrt_sync_output_plus_frozen_decode_nms_success_counter"
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "fps_makespan": 12.5,
        "latency_mean_ms": 80.0,
        "latency_p50_ms": 79.0,
        "latency_p95_ms": 81.0,
        "latency_semantics": (
            "prepared_input_h2d_engine_d2h_sync_decode_class_aware_nms"
        ),
        "minimum_duration_satisfied": True,
        "host_postprocess_frozen": True,
        "postprocess_included": True,
        "postprocess_completed_frames": completed,
        "postprocess_completion_verified": True,
        "frozen_host_postprocess_contract": frozen,
        "frozen_host_postprocess_contract_sha256": frozen[
            "contract_sha256"
        ],
        "frozen_host_postprocess_result": frozen_result,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract_hash": attestation[
            "endpoint_contract_hash"
        ],
        "completed_task_output_endpoint_id": attestation[
            "output_endpoint_id"
        ],
        "completed_task_endpoint_attestation": attestation,
        "preflight_verified": preflight_verified,
    }


def test_tensorrt_performance_and_energy_replay_use_the_same_completed_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import native_full_baseline_eval_runner as full

    frozen, frozen_result = _frozen_evidence()
    manifest, tensor_path, tensor_sha = _write_runtime_input(tmp_path)
    engine = tmp_path / "full.engine"
    engine.write_bytes(b"quality-sealed-engine")
    captured: list[tuple[str, list[str]]] = []

    def fake_run(command, *, label="", **_kwargs):
        command = [str(value) for value in command]
        captured.append((str(label), command))
        report = Path(_argument(command, "--json-out"))
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                _completed_payload_from_command(
                    command,
                    completed=int(_argument(command, "--frames")),
                    preflight_verified="energy" in str(label),
                )
            ),
            encoding="utf-8",
        )
        return {
            "rc": 0,
            "returncode": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(full, "_run", fake_run)
    performance_ns = SimpleNamespace(
        engine_python_selected=sys.executable,
        frames=3,
        warmup=1,
        duration_s=0.0,
        timeout=30,
    )
    performance_row = full._attach_trt_completed_task_hotloop(
        {
            "ok": True,
            "status": "ok",
            "backend": "native_full_tensorrt",
            "model": "yolov7_paper",
            "task": "detection",
            "contract_family": "raw_head",
            "endpoint_contract_hash": "a" * 64,
            "input_manifest": str(manifest),
            "quality_first_producer_identity": {
                "engine": {
                    "path": str(engine),
                    "sha256": _sha256(engine),
                }
            },
            "quality_first_producer_identity_sha256": "b" * 64,
            "frozen_host_postprocess_contract": frozen,
            "frozen_host_postprocess_contract_sha256": frozen[
                "contract_sha256"
            ],
            "frozen_host_postprocess_result": frozen_result,
            "fps_makespan": 777.0,
            "completed_work_units": 3,
        },
        tmp_path,
        "yolov7_paper",
        performance_ns,
    )
    assert performance_row["ok"] is True
    assert performance_row["comparison_endpoint_stratum"] == "decoded_nms"
    assert performance_row["measurement_concurrency"] == 1
    assert performance_row["accelerator_only_diagnostic"][
        "fps_makespan"
    ] == 777.0
    performance_command = captured[-1][1]

    energy_contract = {
        "backend": "native_full_tensorrt",
        "model": "yolov7_paper",
        "contract_sha256": "c" * 64,
        "energy_workload": {
            "kind": "tensorrt_full_completed_task_hotloop",
            "runner_artifact": "hotloop_runner",
            "runtime_python_artifact": "runtime_python",
            "engine_artifact": "engine",
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "frozen_postprocess_contract": frozen,
            "frozen_postprocess_contract_sha256": frozen[
                "contract_sha256"
            ],
            "source_endpoint_contract_hash": "a" * 64,
            "quality_first_producer_identity_sha256": "b" * 64,
            "completed_task_stage": "decoded_nms",
            "e2e_scope": "full_task_pipeline",
            "measurement_concurrency": 1,
            "warmup": 0,
        },
        "artifacts": {
            "hotloop_runner": {
                "path": str(TRT_COMPLETED_RUNNER),
                "sha256": _sha256(TRT_COMPLETED_RUNNER),
            },
            "runtime_python": {
                "path": sys.executable,
                "invocation_path": sys.executable,
            },
            "engine": {
                "path": str(engine),
                "sha256": _sha256(engine),
            },
            "input_manifest": {
                "path": str(manifest),
                "sha256": _sha256(manifest),
            },
            "runtime_input_tensor": {
                "path": str(tensor_path),
                "sha256": tensor_sha,
            },
        },
    }
    monkeypatch.setattr(
        full,
        "_verified_energy_preflight_attestation",
        lambda raw, **_kwargs: (raw, "verified"),
    )
    energy_ns = SimpleNamespace(
        root=str(tmp_path),
        energy_command_contract_file="",
        energy_command_contract_json=json.dumps(energy_contract),
        preflight_attestation=str(tmp_path / "preflight.json"),
        preflight_nonce="nonce",
        preflight_attestation_max_age_s=300.0,
        out_dir=str(tmp_path / "energy"),
        frames=3,
        duration_s=0.0,
        timeout=30,
    )
    assert full._energy_workload_only(energy_ns) == 0
    energy_command = captured[-1][1]

    assert Path(performance_command[1]).resolve() == (
        TRT_COMPLETED_RUNNER.resolve()
    )
    assert Path(energy_command[1]).resolve() == (
        TRT_COMPLETED_RUNNER.resolve()
    )
    for option in (
        "--engine",
        "--input-manifest",
        "--frozen-postprocess-contract-json",
        "--source-endpoint-contract-hash",
        "--expected-engine-sha256",
        "--expected-input-manifest-sha256",
        "--expected-runtime-input-sha256",
    ):
        assert _argument(performance_command, option) == _argument(
            energy_command, option,
        )
    assert "--loadEngine" not in " ".join(energy_command)
    assert "trtexec" not in Path(energy_command[1]).name


def _sealed_completed_energy_contract(
    full,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        f"frozen_postprocess_{name}": {
            "sha256": str(row["sha256"]),
        }
        for name, row in frozen["implementation_artifacts"].items()
    }
    contract: dict[str, Any] = {
        "schema": full.FULL_COMMAND_CONTRACT_SCHEMA,
        "schema_version": full.FULL_COMMAND_CONTRACT_VERSION,
        "complete": True,
        "input_image_sha256": "",
        "energy_workload": {
            "available": True,
            "kind": "tensorrt_full_completed_task_hotloop",
            "e2e_scope": "full_task_pipeline",
            "completed_task_stage": "decoded_nms",
            "measurement_concurrency": 1,
            "postprocess_required": True,
            "postprocess_included": True,
            "host_postprocess_frozen": True,
            "frozen_postprocess_contract": frozen,
            "frozen_postprocess_contract_sha256": frozen["contract_sha256"],
            "frozen_postprocess_implementation_bound": True,
            "successful_run_completed_frames": 3,
            "successful_run_postprocess_completed_frames": 3,
            "original_image_wh": [80, 60],
            "input_image_sha256": "",
        },
        "artifacts": artifacts,
    }
    contract["contract_sha256"] = full._canonical_json_sha256(contract)
    return contract


def test_sealed_tensorrt_completed_energy_contract_enforces_endpoint_and_concurrency(
    monkeypatch,
) -> None:
    from scripts import native_full_baseline_eval_runner as full

    frozen, _result = _frozen_evidence()
    monkeypatch.setattr(
        full,
        "_sealed_trt_source_model_binding",
        lambda _contract: True,
    )
    valid = _sealed_completed_energy_contract(full, frozen)
    sealed, reason = full._sealed_full_energy_contract(valid)
    assert sealed is not None, reason

    for field, invalid in (
        ("e2e_scope", "accelerator_only"),
        ("completed_task_stage", "raw_head"),
        ("measurement_concurrency", 8),
    ):
        tampered = copy.deepcopy(valid)
        tampered.pop("contract_sha256")
        tampered["energy_workload"][field] = invalid
        tampered["contract_sha256"] = full._canonical_json_sha256(tampered)
        sealed, _reason = full._sealed_full_energy_contract(tampered)
        assert sealed is None, (
            "TensorRT completed-task energy contract accepted "
            f"{field}={invalid!r}"
        )


def test_tensorrt_completed_hotloop_remote_copy_is_byte_identical() -> None:
    remote = (
        ROOT
        / "onnx_splitpoint_tool"
        / "resources"
        / "remote_scripts"
        / TRT_COMPLETED_RUNNER.name
    )
    assert remote.is_file()
    assert TRT_COMPLETED_RUNNER.read_bytes() == remote.read_bytes()
