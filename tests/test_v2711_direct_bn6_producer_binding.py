from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import onnx_splitpoint_tool.native_detection_postprocess as native_postprocess
import onnx_splitpoint_tool.native_output_endpoint as native_output_endpoint
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor,
    build_frozen_decoded_nms_normalization_contract,
    build_frozen_postprocess_contract,
    build_normalized_detection_endpoint_attestation,
    tensor_signature,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
    runtime_numeric_input_identity,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v2711_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deepx_hotloop = _load_script(
    "native_deepx_full_energy_hotloop.py"
)
full_runner = _load_script("native_full_baseline_eval_runner.py")
energy_plan = _load_script("native_producer_energy_plan.py")
final_report = _load_script("native_producer_final_report.py")
validator = _load_script("native_producer_validate_visualize.py")


def _load_benchmark_suite_template():
    path = (
        ROOT / "onnx_splitpoint_tool" / "resources"
        / "templates" / "benchmark_suite.py.txt"
    )
    name = "v2711_generated_benchmark_suite_template"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(
        compile(path.read_text(encoding="utf-8"), str(path), "exec"),
        module.__dict__,
    )
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_deepx_runtime_input(
    tmp_path: Path,
    *,
    image: Path,
    task: str,
    shape: list[int],
    layout: str,
    dtype: str = "uint8",
    normalization: str = "none",
    artifact_root: Path | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    artifact_root = artifact_root or tmp_path
    artifact_root.mkdir(parents=True, exist_ok=True)
    layout = layout.upper()
    if layout == "HWC":
        target_hw = [shape[0], shape[1]]
    elif layout == "NHWC":
        target_hw = [shape[1], shape[2]]
    elif layout == "CHW":
        target_hw = [shape[1], shape[2]]
    elif layout == "NCHW":
        target_hw = [shape[2], shape[3]]
    else:  # pragma: no cover - helper misuse
        raise AssertionError(layout)
    array = np.zeros(shape, dtype=np.dtype(dtype))
    tensor = artifact_root / "runtime_input.bin"
    tensor.write_bytes(array.tobytes(order="C"))
    preprocessing = canonical_image_preprocessing_contract(
        task, target_hw,
    )
    preprocessing_sha = preprocessing_contract_sha256(preprocessing)
    numeric, numeric_sha = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task=task,
        preprocessing_contract_sha256_value=preprocessing_sha,
        runtime_input_name="input",
        runtime_input_shape=shape,
        runtime_input_dtype=str(array.dtype),
        runtime_input_layout=layout,
        runtime_color_space="RGB",
        runtime_normalization=normalization,
    )
    payload: dict[str, object] = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": "native_full_deepx",
        "task": task,
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "runtime_input_name": "input",
        "runtime_input_shape": shape,
        "runtime_input_dtype": str(array.dtype),
        "runtime_input_bytes": int(array.nbytes),
        "runtime_input_file": str(tensor),
        "runtime_input_sha256": _sha256(tensor),
        "runtime_input_layout": layout,
        "runtime_color_space": "RGB",
        "runtime_normalization": normalization,
        "runtime_preprocess_mode": preprocessing["preprocess_mode"],
        "runtime_preprocessing_identity": preprocessing,
        "runtime_preprocessing_sha256": preprocessing_sha,
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": numeric_sha,
        "preprocess": {
            "mode": preprocessing["preprocess_mode"],
            "layout": layout,
            "rgb": True,
            "color_space": "RGB",
            "normalization": normalization,
            "letterbox_pad_value": preprocessing[
                "letterbox_pad_value"
            ],
        },
    }
    manifest = artifact_root / "native_full_input_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, tensor, payload


def _direct_evidence() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    outputs = {
        "output": np.asarray(
            [[[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]]],
            dtype=np.float32,
        ),
    }
    source_hash = "a" * 64
    source_attestation = {
        "schema": "onnx-splitpoint/runtime-output-endpoint-attestation",
        "schema_version": 3,
        "attested": True,
        "status": "passed",
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": source_hash,
        "tensor_signature": tensor_signature(outputs),
        "declared_contract": {
            "model_id": "yolo26s",
            "source_coordinate_space":
                "model_input_letterbox_xyxy_pixels",
        },
    }
    direct = build_frozen_decoded_nms_normalization_contract(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[1280, 720],
        preprocess={
            "mode": "letterbox_rgb_uint8",
            "rgb": True,
            "pad_value": 114,
        },
        source_coordinate_space=
            "model_input_letterbox_xyxy_pixels",
        source_endpoint_contract_hash=source_hash,
        source_output_endpoint_attestation=source_attestation,
    )
    result = FrozenDecodedNmsPostprocessor(direct).process(
        outputs, original_wh=[1280, 720],
    )
    completion = build_normalized_detection_endpoint_attestation(
        direct,
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
    )
    row: dict[str, object] = {
        "ok": True,
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_hash": source_hash,
        "output_endpoint_id":
            f"detection:decoded_nms:{source_hash}",
        "output_endpoint_attestation": source_attestation,
        "normalization_frozen": True,
        "host_postprocess_frozen": False,
        "postprocess_included": True,
        "completed_frames": 3,
        "postprocess_completed_frames": 3,
        "postprocess_completion_verified": True,
        "frozen_decoded_nms_normalization_contract": direct,
        "frozen_decoded_nms_normalization_contract_sha256":
            direct["contract_sha256"],
        "frozen_decoded_nms_normalization_result": result,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_completion_mode":
            "integrated_accelerator_plus_frozen_normalization",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_endpoint_contract_hash":
            completion["endpoint_contract_hash"],
        "completed_task_output_endpoint_id":
            completion["output_endpoint_id"],
        "completed_task_comparison_endpoint_contract":
            completion[
                "completed_task_comparison_endpoint_contract"
            ],
        "completed_task_comparison_endpoint_contract_hash":
            completion[
                "completed_task_comparison_endpoint_contract_hash"
            ],
        "completed_task_comparison_output_endpoint_id":
            completion[
                "completed_task_comparison_output_endpoint_id"
            ],
        "completed_task_endpoint_attestation": completion,
    }
    return row, outputs


def _ns(
    *, setup_id: str = "", comparison_backend: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        frames=3,
        duration_s=0.0,
        warmup=0,
        repetitions=1,
        inflight=1,
        trt_precision="fp16",
        workspace_mb=1024,
        engine_build_python=sys.executable,
        engine_python_selected=sys.executable,
        no_shapes=False,
        dump_outputs=True,
        diagnostic_deepx_input_probes=False,
        preprocess_mode="letterbox",
        letterbox_pad_value=114,
        setup_id=setup_id,
        comparison_backend=comparison_backend,
    )


def _raw_yolov7_outputs() -> dict[str, np.ndarray]:
    return {
        "output": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "clone_1": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "clone_2": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }


def _full_bn6_outputs() -> dict[str, np.ndarray]:
    return {
        "detections": np.asarray(
            [[
                [8.0, 16.0, 32.0, 48.0, 0.9, 2.0],
                [0.0, 0.0, 4.0, 4.0, 0.1, 1.0],
            ]],
            dtype=np.float32,
        ),
    }


def _screening_policy() -> AccuracyGatePolicy:
    return AccuracyGatePolicy(
        dataset_tier="screening",
        frozen_before_final_campaign=False,
        screening_eligible_for_ranking=False,
        contract_only_eligible_for_ranking=False,
    )


def test_direct_bn6_binding_is_exact_and_backend_neutral() -> None:
    row, _outputs = _direct_evidence()
    direct = row["frozen_decoded_nms_normalization_contract"]
    assert isinstance(direct, dict)
    binding = full_runner._verified_direct_completed_task_workload(
        row,
        direct,
        completed_frames=3,
        postprocess_completed_frames=3,
    )
    assert binding["normalization_frozen"] is True
    assert binding["host_postprocess_frozen"] is False
    assert (
        binding["completed_task_endpoint_attestation"]
        == row["completed_task_endpoint_attestation"]
    )

    for backend in ("native_full_tensorrt", "native_full_deepx"):
        artifacts = {
            f"frozen_postprocess_{name}": {
                "sha256": artifact["sha256"],
            }
            for name, artifact in direct[
                "implementation_artifacts"
            ].items()
        }
        workload = {
            **binding,
            "postprocess_required": True,
            "postprocess_included": True,
            "frozen_postprocess_implementation_bound": True,
            "successful_run_completed_frames": 3,
            "successful_run_postprocess_completed_frames": 3,
            "original_image_wh": [1280, 720],
        }
        contract = {
            "backend": backend,
            "energy_workload": workload,
            "artifacts": artifacts,
        }
        assert (
            full_runner._sealed_frozen_postprocess_binding(contract)
            is True
        )

        tampered = json.loads(json.dumps(contract))
        tampered["energy_workload"].pop(
            "source_endpoint_contract_hash"
        )
        assert (
            full_runner._sealed_frozen_postprocess_binding(tampered)
            is False
        )

    conflicting = dict(row)
    conflicting["endpoint_contract_hash"] = "b" * 64
    assert not full_runner._verified_direct_completed_task_workload(
        conflicting,
        direct,
        completed_frames=3,
        postprocess_completed_frames=3,
    )


def test_deepx_full_command_seals_direct_bn6_completed_task(
    tmp_path: Path,
) -> None:
    row, _outputs = _direct_evidence()
    run_root = tmp_path / "eval-a"
    benchmark_set = run_root / "yolo26s" / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    source_onnx = benchmark_set / "models" / "yolo26s.onnx"
    source_onnx.parent.mkdir(parents=True)
    source_onnx.write_bytes(b"sealed-source-onnx")
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/benchmark-set",
            "schema_version": 2,
            "model_name": "yolo26s",
            "model": "models/yolo26s.onnx",
            "artifact_manifest": {
                "schema": "onnx-splitpoint/benchmark-set",
                "schema_version": 2,
                "files": {"models": ["models/yolo26s.onnx"]},
                "counts": {"models": 1},
            },
        }, sort_keys=True),
        encoding="utf-8",
    )
    ns = _ns(
        setup_id="deepx-setup", comparison_backend="deepx",
    )
    artifact_root = full_runner._native_full_dump_dir(
        benchmark_set, "yolo26s", "native_full_deepx", ns,
    )
    image = benchmark_set / "input.jpg"
    image.write_bytes(b"image")
    dxnn = (
        benchmark_set / "deepx" / "deepx_m1"
        / "full" / "model.dxnn"
    )
    dxnn.parent.mkdir(parents=True)
    dxnn.write_bytes(b"dxnn")
    input_manifest, runtime_tensor, input_payload = (
        _sealed_deepx_runtime_input(
            benchmark_set,
            image=image,
            task="detection",
            shape=[1, 640, 640, 3],
            layout="NHWC",
            artifact_root=artifact_root,
        )
    )
    direct = row["frozen_decoded_nms_normalization_contract"]
    assert isinstance(direct, dict)
    prepared = {
        "input_contract": {
            "input": {
                "shape": [1, 640, 640, 3],
                "dtype": "uint8",
                "layout": "NHWC",
                "normalization": "none",
                "color_space": "RGB",
                "preprocess_mode": "letterbox",
                "letterbox_pad_value": 114,
            },
        },
        "image": str(image),
        "task": "detection",
        "host_postprocess_frozen": False,
        "normalization_frozen": True,
        "postprocess_included": True,
        "completed_frames": 3,
        "postprocess_completed_frames": 3,
        "original_image_wh": [1280, 720],
        "frozen_decoded_nms_normalization_contract": direct,
        "completed_task_endpoint_attestation":
            row["completed_task_endpoint_attestation"],
        "frozen_decoded_nms_normalization_result":
            row["frozen_decoded_nms_normalization_result"],
        "warmup_count": 0,
        "prepared_input_manifest": str(input_manifest),
        "prepared_input_file": str(runtime_tensor),
        "prepared_input_sha256": input_payload[
            "runtime_input_sha256"
        ],
        "prepared_input_file_sha256": input_payload[
            "runtime_input_sha256"
        ],
        "prepared_input_bytes": input_payload["runtime_input_bytes"],
        "prepared_input_name": input_payload["runtime_input_name"],
        "prepared_input_shape": input_payload["runtime_input_shape"],
        "prepared_input_dtype": input_payload["runtime_input_dtype"],
        "prepared_input_layout": input_payload["runtime_input_layout"],
        "runtime_preprocessing_identity": input_payload[
            "runtime_preprocessing_identity"
        ],
        "runtime_preprocessing_sha256": input_payload[
            "runtime_preprocessing_sha256"
        ],
        "runtime_numeric_input_identity": input_payload[
            "runtime_numeric_input_identity"
        ],
        "runtime_numeric_input_sha256": input_payload[
            "runtime_numeric_input_sha256"
        ],
        "prepared_input_binding_verified": True,
        "prepared_input_source": "sealed_semantic_dump_runtime_tensor",
        "prepared_input_source_image_id": image.name,
        "prepared_input_source_image_sha256": _sha256(image),
        "prepared_feed_contract_version": (
            full_runner.DEEPX_PREPARED_FEED_CONTRACT_VERSION
        ),
    }
    completed_path = (
        benchmark_set / "results" / "deepx_m1_full"
        / "deepx_prepared_feed.completed_task_result_artifact.json"
    )
    completed_path.parent.mkdir(parents=True)
    prepared.update(native_postprocess.persist_completed_result_artifact(
        row["frozen_decoded_nms_normalization_result"][
            "completed_result_artifact"
        ],
        expected_sha256=row[
            "frozen_decoded_nms_normalization_result"
        ]["completed_result_artifact_sha256"],
        output_path=completed_path,
    ))
    prepared["completed_task_result_artifact_verification_status"] = (
        "verified_exact"
    )
    row.update({
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "setup_id": "deepx-setup",
        "comparison_backend": "deepx",
        "run_id": "deepx_m1_full",
        "input_case": "full",
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "input_manifest": str(input_manifest),
        "runtime_python": sys.executable,
        "dxnn_path": str(dxnn),
        "frames": 3,
        "completed_frames": 3,
        "completed_work_units": 3,
        "completed_work_units_source":
            "dx_engine_output_plus_frozen_completion_success_counter",
        "performance_benchmark_source": "dx_engine_prepared_feed",
        "prepared_feed_contract_version":
            full_runner.DEEPX_PREPARED_FEED_CONTRACT_VERSION,
        "deepx_prepared_feed_benchmark": prepared,
    })

    contract = full_runner._full_command_contract(
        row=row,
        root=run_root,
        benchmark_set=benchmark_set,
        model="yolo26s",
        backend_arg="deepx",
        ns=ns,
    )
    workload = contract["energy_workload"]
    assert contract["complete"] is True
    assert workload["available"] is True
    assert workload["normalization_frozen"] is True
    assert workload["host_postprocess_frozen"] is False
    assert "frozen_postprocess_contract" not in workload
    assert (
        workload["frozen_decoded_nms_normalization_contract"]
        == direct
    )
    assert (
        workload["completed_task_endpoint_attestation"]
        == row["completed_task_endpoint_attestation"]
    )
    assert (
        workload["source_endpoint_contract_hash"]
        == row["endpoint_contract_hash"]
    )
    assert full_runner._sealed_frozen_postprocess_binding(contract)

    verified, status = energy_plan._verify_full_command_contract(
        contract,
        expected_identity={
            "backend": "native_full_deepx",
            "model": "yolo26s",
            "case": "full",
            "setup_id": "deepx-setup",
            "comparison_backend": "deepx",
            "remote_root": str(run_root),
            "remote_tool_dir": str(ROOT),
        },
    )
    assert verified is not None, status


def test_hailo_full_command_seals_direct_bn6_completed_task(
    tmp_path: Path,
) -> None:
    row, _outputs = _direct_evidence()
    run_root = tmp_path / "eval-a"
    benchmark_set = run_root / "yolo26s" / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    ns = _ns(
        setup_id="hailo10-setup", comparison_backend="hailo10h",
    )
    artifact_root = full_runner._native_full_dump_dir(
        benchmark_set, "yolo26s", "native_full_hailo10h", ns,
    )
    artifact_root.mkdir(parents=True)
    image = benchmark_set / "input.jpg"
    image.write_bytes(b"image")
    hef = (
        benchmark_set / "hailo" / "hailo10"
        / "full" / "compiled.hef"
    )
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"hef")
    source = benchmark_set / "models" / "yolo26s.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    receipt = hef.parent / "hailo_hef_build_receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    runtime_tensor = artifact_root / "runtime_input.bin"
    runtime_tensor.write_bytes(b"\x01")
    input_manifest = artifact_root / "native_full_input_manifest.json"
    input_manifest.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": "native_full_hailo10h",
        "task": "detection",
        "runtime_input_name": "input",
        "runtime_input_shape": [1, 1, 1, 1],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": 1,
        "runtime_input_file": str(runtime_tensor),
        "runtime_input_sha256": _sha256(runtime_tensor),
        "runtime_input_layout": "NHWC",
        "preprocess": {"layout": "NHWC", "mode": "letterbox"},
    }), encoding="utf-8")
    direct = row["frozen_decoded_nms_normalization_contract"]
    report = artifact_root / "native_hailo_meta.json"
    report.write_text(json.dumps({
        "task": "detection",
        "input_names": ["input"],
        "output_names": ["output"],
        "runtime_input_shape": [1, 1, 1, 1],
        "runtime_input_dtype": "uint8",
        "quantized_inputs": True,
        "normalization_frozen": True,
        "host_postprocess_frozen": False,
        "postprocess_included": True,
        "postprocess_completed_frames": 3,
        "completed_frames": 3,
        "original_image_wh": [1280, 720],
        "frozen_decoded_nms_normalization_contract": direct,
    }), encoding="utf-8")
    row.update({
        "backend": "native_full_hailo10h",
        "model": "yolo26s",
        "setup_id": "hailo10-setup",
        "comparison_backend": "hailo10h",
        "input_case": "full",
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "input_manifest": str(input_manifest),
        "runtime_python": sys.executable,
        "hef_path": str(hef),
        "report": str(report),
        "frames": 3,
        "completed_frames": 3,
        "completed_work_units": 3,
        "completed_task_result_artifact_verification_status": (
            "verified_exact"
        ),
        "direct_source_endpoint_binding_verified": True,
        "hailo_hef_build_receipt_status": (
            "hailo_hef_build_receipt_verified_exact"
        ),
        "hailo_hef_build_receipt_path": str(receipt),
        "hailo_hef_build_receipt_file_sha256": _sha256(receipt),
        "hailo_hef_build_receipt_sha256": (
            full_runner._canonical_json_sha256({})
        ),
        "hailo_hef_build_receipt": {},
        "hailo_hef_source_onnx_path": str(source),
        "source_onnx_sha256": _sha256(source),
    })

    contract = full_runner._full_command_contract(
        row=row,
        root=run_root,
        benchmark_set=benchmark_set,
        model="yolo26s",
        backend_arg="hailo10h",
        ns=ns,
    )
    workload = contract["energy_workload"]
    assert contract["complete"] is True
    assert workload["available"] is True
    assert workload["kind"] == "hailo_full_hotloop"
    assert workload["normalization_frozen"] is True
    assert workload["host_postprocess_frozen"] is False
    assert "frozen_postprocess_contract" not in workload
    assert (
        workload["frozen_decoded_nms_normalization_contract"]
        == direct
    )
    assert (
        workload["completed_task_endpoint_attestation"]
        == row["completed_task_endpoint_attestation"]
    )
    assert full_runner._sealed_frozen_postprocess_binding(contract)


def test_tensorrt_full_command_seals_direct_bn6_completed_task(
    tmp_path: Path,
) -> None:
    fixture_path = (
        ROOT / "tests"
        / "test_v269d_trt_quality_first_runtime.py"
    )
    fixture_spec = importlib.util.spec_from_file_location(
        "v2711_quality_fixture", fixture_path,
    )
    assert fixture_spec is not None
    assert fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    benchmark_set, _producer_file, producer = (
        fixture_module._write_quality_producer_set(
            tmp_path, model="yolo26s",
        )
    )
    run_root = benchmark_set.parent.parent
    ns = _ns(
        setup_id="setup-a", comparison_backend="ort_tensorrt",
    )
    artifact_root = full_runner._native_full_dump_dir(
        benchmark_set, "yolo26s", "native_full_tensorrt", ns,
    )
    artifact_root.mkdir(parents=True)

    row, _outputs = _direct_evidence()
    image = benchmark_set / "input.jpg"
    image.write_bytes(b"image")
    runtime_tensor = artifact_root / "runtime_input.bin"
    runtime_tensor.write_bytes(b"\x01")
    input_manifest = artifact_root / "native_full_input_manifest.json"
    input_manifest.write_text(
        json.dumps({
                "schema": "onnx-splitpoint/native-full-input-dump",
                "schema_version": 2,
                "backend": "native_full_tensorrt",
                "task": "detection",
                "runtime_input_name": "input",
                "runtime_input_shape": [1, 1, 1, 1],
            "runtime_input_dtype": "uint8",
            "runtime_input_bytes": 1,
            "runtime_input_file": str(runtime_tensor),
            "runtime_input_sha256": _sha256(runtime_tensor),
            "runtime_input_layout": "NCHW",
            "preprocess": {
                "layout": "NCHW",
                "mode": "letterbox",
            },
        }),
        encoding="utf-8",
    )
    report = artifact_root / "native_trt_meta.json"
    report.write_text(
        json.dumps({
            "onnx": producer["build_onnx"]["path"],
            "engine_build_receipt_path":
                producer["engine_build_receipt"]["path"],
            "engine_build_receipt":
                producer["engine_build_receipt"]["receipt"],
            "run_smoke": {
                "returncode": 0,
                "cmd": [
                    producer["trtexec"]["path"],
                    "--loadEngine=" + producer["engine"]["path"],
                    "--iterations=3",
                    "--warmUp=0",
                ],
            },
        }),
        encoding="utf-8",
    )
    row.update({
        "backend": "native_full_tensorrt",
        "model": "yolo26s",
        "setup_id": "setup-a",
        "comparison_backend": "ort_tensorrt",
        "input_case": "full",
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "input_manifest": str(input_manifest),
        "report": str(report),
        "frames": 3,
        "completed_frames": 3,
        "completed_work_units": 3,
        "quality_first_producer_identity": producer,
        "quality_first_producer_identity_sha256":
            producer["producer_identity_sha256"],
    })

    contract = full_runner._full_command_contract(
        row=row,
        root=run_root,
        benchmark_set=benchmark_set,
        model="yolo26s",
        backend_arg="tensorrt",
        ns=ns,
    )
    workload = contract["energy_workload"]
    assert contract["complete"] is True
    assert workload["available"] is True
    assert (
        workload["kind"]
        == "tensorrt_full_completed_task_hotloop"
    )
    assert workload["normalization_frozen"] is True
    assert workload["host_postprocess_frozen"] is False
    assert "frozen_postprocess_contract" not in workload
    assert (
        workload["completed_task_endpoint_attestation"]
        == row["completed_task_endpoint_attestation"]
    )
    assert full_runner._sealed_frozen_postprocess_binding(contract)

    verified, status = energy_plan._verify_full_command_contract(
        contract,
        expected_identity={
            "backend": "native_full_tensorrt",
            "model": "yolo26s",
            "case": "full",
            "setup_id": "setup-a",
            "comparison_backend": "ort_tensorrt",
        },
    )
    assert verified is not None, status


def test_deepx_raw_host_tail_cli_binds_source_endpoint_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    outputs = {
        "output": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "clone_1": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "clone_2": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }
    frozen = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"dxnn")
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    manifest, runtime_tensor, input_payload = (
        _sealed_deepx_runtime_input(
            tmp_path,
            image=image,
            task="detection",
            shape=[640, 640, 3],
            layout="HWC",
        )
    )
    assert manifest.is_file()
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    source_hash = "c" * 64

    class FakeEngine:
        def __init__(self, _path: str) -> None:
            pass

        def run(self, _feeds):
            return list(outputs.values())

    fake_dx_engine = types.ModuleType("dx_engine")
    fake_dx_engine.InferenceEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "dx_engine", fake_dx_engine)
    monkeypatch.setattr(
        deepx_hotloop, "_verify_preflight",
        lambda *_args, **_kwargs: True,
    )
    runner_path = Path(deepx_hotloop.__file__).resolve()
    monkeypatch.setattr(sys, "argv", [
        "native_deepx_full_energy_hotloop.py",
        "--dxnn", str(dxnn),
        "--prepared-input-file", str(runtime_tensor),
        "--expected-prepared-input-path", str(runtime_tensor),
        "--expected-prepared-input-root", str(runtime_tensor.parent),
        "--expected-prepared-input-sha256",
        str(input_payload["runtime_input_sha256"]),
        "--expected-prepared-input-bytes",
        str(input_payload["runtime_input_bytes"]),
        "--expected-prepared-input-name",
        str(input_payload["runtime_input_name"]),
        "--expected-prepared-input-shape-json",
        json.dumps(input_payload["runtime_input_shape"]),
        "--expected-prepared-input-dtype",
        str(input_payload["runtime_input_dtype"]),
        "--expected-prepared-input-layout",
        str(input_payload["runtime_input_layout"]),
        "--runtime-preprocessing-identity-json",
        json.dumps(input_payload["runtime_preprocessing_identity"]),
        "--expected-runtime-preprocessing-sha256",
        str(input_payload["runtime_preprocessing_sha256"]),
        "--runtime-numeric-input-identity-json",
        json.dumps(input_payload["runtime_numeric_input_identity"]),
        "--expected-runtime-numeric-input-sha256",
        str(input_payload["runtime_numeric_input_sha256"]),
        "--original-image-wh-json", json.dumps([80, 60]),
        "--prepared-feed-contract-version",
        deepx_hotloop.CONTRACT_VERSION,
        "--frames", "1",
        "--warmup", "0",
        "--task", "detection",
        "--frozen-postprocess-contract-json",
        json.dumps(frozen),
        "--source-endpoint-contract-hash", source_hash,
        "--json-out", str(report),
        "--expected-runner-sha256", "d" * 64,
        "--expected-runner-path", str(runner_path),
        "--expected-runner-root", str(runner_path.parent),
        "--expected-dxnn-sha256", _sha256(dxnn),
        "--expected-dxnn-path", str(dxnn),
        "--expected-dxnn-root", str(dxnn.parent),
        "--source-contract-sha256", "1" * 64,
        "--preflight-attestation", str(preflight),
        "--preflight-nonce", "nonce",
    ])

    assert deepx_hotloop.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["runtime_input_sha256"] == _sha256(runtime_tensor)
    assert payload["image_decode_performed"] is False
    assert payload["preprocessing_performed"] is False
    assert payload["completed_task_result_artifact_saved"] is True
    assert payload["source_endpoint_contract_hash"] == source_hash
    completed = payload["completed_task_endpoint_attestation"]
    assert (
        completed["completed_endpoint_contract"][
            "source_endpoint_contract_hash"
        ]
        == source_hash
    )


def test_deepx_yolov7_optional_model_hash_still_rejects_conflicts() -> None:
    suite = _load_benchmark_suite_template()

    assert suite._deepx_contract_model_sha256({}, {}) == ""
    assert suite._deepx_contract_model_sha256(
        {"model_sha256": "sha256:" + YOLOV7_PAPER_ONNX_SHA256.upper()},
        {},
    ) == YOLOV7_PAPER_ONNX_SHA256

    with pytest.raises(
        RuntimeError,
        match="native_full_raw_detection_model_sha256_conflicting",
    ):
        suite._deepx_contract_model_sha256(
            {"model_sha256": YOLOV7_PAPER_ONNX_SHA256},
            {"source_onnx_sha256": "0" * 64},
        )

    with pytest.raises(
        RuntimeError,
        match="native_full_raw_detection_model_sha256_invalid",
    ):
        suite._deepx_contract_model_sha256(
            {"model_sha256": "not-a-sha256"},
            {},
        )


def test_deepx_yolov7_template_projects_portable_completed_v2_screening(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the generated DeepX producer through final projection.

    The performance template must bind the Completed-v2 host tail to the
    physical raw-head endpoint produced by ``runtime_output_contract``.  A
    separate semantic replay may differ only in its result hash in Standard
    screening, but every claim/ranking eligibility axis must remain closed.
    """
    suite = _load_benchmark_suite_template()
    root = tmp_path / "suite"
    results_dir = root / "results" / "deepx_m1_full"
    results_dir.mkdir(parents=True)
    dxnn = root / "model.dxnn"
    dxnn.write_bytes(b"dxnn")
    image = root / "input.jpg"
    # The producer now uses the real Pillow reader for source geometry.
    # Keep the previous fake-cv2 geometry (80 x 60), but supply an actual JPEG.
    from PIL import Image
    Image.new("RGB", (80, 60)).save(image)
    (root / "output_contracts.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/output-contracts",
            "schema_version": 1,
            "model_id": "yolov7_paper",
            "task": "detection",
            "contracts": [{
                "model_id": "yolov7_paper",
                "backend": "deepx_m1",
                "variant": "full",
                "task": "detection",
                "endpoint_mode": "raw_detection_head",
                "contract_status": "recorded",
                "host_tail_required": True,
                "postprocessing_required": True,
                "requires_external_postprocess": True,
            }],
        }),
        encoding="utf-8",
    )
    input_contract = {
        "model_id": "yolov7_paper",
        # Mirror the v2.75.47 DeepX Full contract: identity is present, but the
        # redundant source/build digests are explicitly null and the vendor
        # contract exposes one generic output declaration for three tensors.
        "source_onnx_sha256": None,
        "build_onnx_sha256": None,
        "contract_family": "raw_head_or_unattested",
        "endpoint_mode": "raw_detection_head",
        "host_tail_required": True,
        "input": {
            "shape": [640, 640, 3],
            "dtype": "uint8",
            "layout": "HWC",
            "normalization": "none",
            "color_space": "RGB",
            "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        },
        "outputs": [{"name": "model_outputs", "shape": None}],
    }
    prepared_input_manifest, runtime_tensor, _input_payload = (
        _sealed_deepx_runtime_input(
            root,
            image=image,
            task="detection",
            shape=[640, 640, 3],
            layout="HWC",
        )
    )
    raw_outputs = _raw_yolov7_outputs()
    deepx_runtime_outputs = {
        "model_outputs": raw_outputs["output"],
        "output_1": raw_outputs["clone_1"],
        "output_2": raw_outputs["clone_2"],
    }

    class FakeEngine:
        def __init__(self, _path: str) -> None:
            pass

        def run(self, _feeds):
            return [
                np.array(raw_outputs[name], copy=True)
                for name in ("output", "clone_1", "clone_2")
            ]

    fake_dx_engine = types.ModuleType("dx_engine")
    fake_dx_engine.InferenceEngine = FakeEngine
    splitpoint_runners = types.ModuleType("splitpoint_runners")
    splitpoint_runners.__path__ = []
    native_full_input = types.ModuleType(
        "splitpoint_runners.native_full_input"
    )
    native_full_input.load_sealed_deepx_native_full_input = (
        lambda *_args, **_kwargs: {
            "runtime_input": np.fromfile(
                runtime_tensor, dtype=np.uint8,
            ).reshape((640, 640, 3))
        }
    )
    native_full_input.prepare_and_seal_deepx_native_full_input = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixture supplies an existing sealed manifest")
        )
    )
    monkeypatch.setitem(sys.modules, "dx_engine", fake_dx_engine)
    monkeypatch.setitem(
        sys.modules, "splitpoint_runners", splitpoint_runners,
    )
    monkeypatch.setitem(
        sys.modules, "splitpoint_runners.native_full_input",
        native_full_input,
    )
    monkeypatch.setitem(
        sys.modules,
        "splitpoint_runners.native_detection_postprocess",
        native_postprocess,
    )
    monkeypatch.setitem(
        sys.modules,
        "splitpoint_runners.native_output_endpoint",
        native_output_endpoint,
    )
    monkeypatch.setattr(
        suite,
        "_deepx_input_size_from_contract",
        lambda *_args, **_kwargs: (640, copy.deepcopy(input_contract)),
    )
    monkeypatch.setattr(
        suite,
        "_deepx_find_prepared_feed_image",
        lambda *_args, **_kwargs: (
            image, "explicit_prepared_feed_image",
        ),
    )
    monkeypatch.setattr(
        suite,
        "_deepx_letterbox_bgr",
        lambda *_args, **_kwargs: (
            np.zeros((640, 640, 3), dtype=np.uint8),
            0.8,
            0,
            8,
        ),
    )
    monkeypatch.setattr(
        suite,
        "_deepx_contract_model_id",
        lambda *_args, **_kwargs: "yolov7_paper",
    )
    prepared = suite._run_deepx_prepared_feed_benchmark(
        root,
        dxnn,
        {
            "benchmark_task": "detection",
            "model_id": "yolov7_paper",
        },
        SimpleNamespace(
            runs=3,
            warmup=0,
            energy_measurement_only=False,
            throughput_frames=0,
            benchmark_task="detection",
            validation_images="",
            prepared_input_manifest=str(prepared_input_manifest),
        ),
        results_dir,
    )

    assert prepared["status"] == "ok", prepared
    assert (
        prepared["frozen_host_postprocess_contract"]["model_sha256"]
        == YOLOV7_PAPER_ONNX_SHA256
    )
    assert prepared["runtime_endpoint_contract_family"] == "raw_head"
    assert prepared["runtime_endpoint_contract_complete"] is True
    source_hash = prepared["source_endpoint_contract_hash"]
    assert len(source_hash) == 64
    assert (
        prepared["source_output_endpoint_attestation"][
            "endpoint_contract_hash"
        ]
        == source_hash
    )
    completion = prepared["completed_task_endpoint_attestation"]
    assert completion["attested"] is True
    assert (
        completion["completed_endpoint_contract"][
            "source_endpoint_contract_hash"
        ]
        == source_hash
    )
    producer_binding = (
        full_runner._verified_raw_completed_task_attestation(
            prepared,
            prepared["frozen_host_postprocess_contract"],
            completed_frames=3,
            postprocess_completed_frames=3,
        )
    )
    assert (
        producer_binding["source_endpoint_contract_hash"]
        == source_hash
    )
    tampered_prepared = copy.deepcopy(prepared)
    tampered_prepared["source_endpoint_contract_hash"] = "d" * 64
    assert not full_runner._verified_raw_completed_task_attestation(
        tampered_prepared,
        tampered_prepared["frozen_host_postprocess_contract"],
        completed_frames=3,
        postprocess_completed_frames=3,
    )

    wrong_hash_results = root / "results" / "deepx_m1_full_wrong_hash"
    wrong_hash_results.mkdir()
    wrong_hash = suite._run_deepx_prepared_feed_benchmark(
        root,
        dxnn,
        {
            "benchmark_task": "detection",
            "model_id": "yolov7_paper",
            "model_sha256": "0" * 64,
        },
        SimpleNamespace(
            runs=1,
            warmup=0,
            energy_measurement_only=False,
            throughput_frames=0,
            benchmark_task="detection",
            validation_images="",
            prepared_input_manifest=str(prepared_input_manifest),
        ),
        wrong_hash_results,
    )
    assert wrong_hash["status"] == "runtime_failed"
    assert (
        "completion_source_yolov7_model_sha256_mismatch"
        in wrong_hash["error"]
    )

    # Model the legitimate Standard-screening split between the measured
    # performance hotloop and a separate semantic replay from an archived V1
    # row that stored only a hash.  New rows persist the exact Completed
    # artifact and must not be downgraded into this portable fallback merely
    # by editing their hash.
    portable_hash = "c" * 64
    legacy_result = completion["frozen_postprocess_result"]
    for field in (
        "coordinate_space",
        "record_schema",
        "canonical_sort_policy",
        "detections",
        "completed_result_artifact",
        "completed_result_artifact_sha256",
    ):
        legacy_result.pop(field, None)
    legacy_result["detections_sha256"] = portable_hash
    prepared["frozen_host_postprocess_result"] = copy.deepcopy(
        completion["frozen_postprocess_result"]
    )
    comparison = completion[
        "completed_task_comparison_endpoint_contract"
    ]
    performance_row = {
        "backend": "native_full_deepx",
        "producer_impl": "native_full_deepx_native_full_suite",
        "model": "yolov7_paper",
        "case": "full",
        "ok": True,
        "status": "ok",
        "fps_makespan": 1.0,
        "outer_makespan_verified": True,
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "output_format": "raw_detection_tensors",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": source_hash,
        "output_endpoint_attestation": copy.deepcopy(
            prepared["source_output_endpoint_attestation"]
        ),
        "host_postprocess_frozen": True,
        "normalization_frozen": False,
        "postprocess_included": True,
        "completed_frames": 3,
        "postprocess_completed_frames": 3,
        "postprocess_completion_verified": True,
        "frozen_host_postprocess_contract": copy.deepcopy(
            prepared["frozen_host_postprocess_contract"]
        ),
        "frozen_host_postprocess_contract_sha256": prepared[
            "frozen_host_postprocess_contract_sha256"
        ],
        "frozen_host_postprocess_result": copy.deepcopy(
            prepared["frozen_host_postprocess_result"]
        ),
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract_hash": completion[
            "endpoint_contract_hash"
        ],
        "completed_task_output_endpoint_id": completion[
            "output_endpoint_id"
        ],
        "completed_task_comparison_endpoint_contract": copy.deepcopy(
            comparison
        ),
        "completed_task_comparison_endpoint_contract_hash": comparison[
            "endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": comparison[
            "output_endpoint_id"
        ],
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_endpoint_attestation": copy.deepcopy(completion),
    }
    analysis = tmp_path / "projection" / "analysis_tables"
    analysis.mkdir(parents=True)
    (analysis / "native_full_baseline_eval.json").write_text(
        json.dumps({"rows": [performance_row]}),
        encoding="utf-8",
    )
    projected = final_report._rows_from_native_full(
        analysis.parent
    )[0]
    assert projected[
        "completed_task_endpoint_projection_status"
    ] == "passed_existing_attestation_verified"
    assert projected["completed_task_endpoint_attested"] is True

    semantic = validator._completed_v2_self_reference_detection(
        _full_bn6_outputs(),
        deepx_runtime_outputs,
        projected,
        policy=_screening_policy(),
    )
    assert semantic["available"] is True
    assert semantic["completed_v2_verified"] is True
    assert semantic["portable_result_hash_mismatch"] is True
    assert semantic["exact_completed_result_identity_bound"] is False
    assert semantic["completed_v2_exact_result_claim_binding"] is False

    claim_axes = (
        "claim_eligible",
        "e2e_claim_eligible",
        "performance_claim_eligible",
        "energy_claim_eligible",
        "scientific_claim_eligible",
        "thesis_claim_eligible",
        "eligible_for_ranking",
        "ranking_eligible",
        "performance_eligible",
        "energy_eligible",
        "pareto_eligible",
        "thesis_comparison_eligible",
        "thesis_valid",
    )
    claim_row = {field: True for field in claim_axes}
    validator._apply_completed_v2_semantic_binding(
        claim_row, semantic,
    )
    assert all(claim_row[field] is False for field in claim_axes)

    tampered = copy.deepcopy(performance_row)
    tampered["completed_task_endpoint_attestation"][
        "completed_endpoint_contract"
    ]["source_endpoint_contract_hash"] = "d" * 64
    (analysis / "native_full_baseline_eval.json").write_text(
        json.dumps({"rows": [tampered]}),
        encoding="utf-8",
    )
    rejected = final_report._rows_from_native_full(
        analysis.parent
    )[0]
    assert rejected["completed_task_endpoint_attested"] is False
    assert rejected[
        "completed_task_endpoint_projection_status"
    ] == (
        "conflict:source_and_reconstructed_attestation_mismatch"
    )
    rejected_semantic = (
        validator._completed_v2_self_reference_detection(
            _full_bn6_outputs(),
            deepx_runtime_outputs,
            rejected,
            policy=_screening_policy(),
        )
    )
    assert rejected_semantic["available"] is False
    assert rejected_semantic["completed_v2_verified"] is False
