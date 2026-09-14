from __future__ import annotations

import base64
import hashlib
import contextlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onnx_splitpoint_tool.energy.collector import (
    _command_window_binding,
    _runtime_work_unit_evidence,
    _verify_calibration_manifest,
    _window_alignment,
)
from onnx_splitpoint_tool.native_energy_reporting import (
    NATIVE_ENERGY_ACTIVE_DURATION_COMPARISON_POLICY,
    NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE,
    build_native_energy_pairs,
    collect_native_energy,
    scientific_energy_rows,
)
from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
    seal_native_command_contract,
)
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
from onnx_splitpoint_tool.workflow.analysis_pack import CANONICAL_RESULT_FILES, CANONICAL_TABLE_FILES
from onnx_splitpoint_tool.workflow.cross_runner_reporting import compute_cross_runner_report
from onnx_splitpoint_tool.workflow.scientific_reporting import _ranking_method_comparison
from scripts.run_and_report_work_units import _count_from_payload
import scripts.native_full_baseline_eval_runner as full_runner
import scripts.native_producer_energy_plan as energy_plan
import scripts.native_producer_final_report as final_report
from scripts.native_producer_energy_plan import (
    _contract_fields_for_task,
    _dedupe_key,
    _semantic_decision,
    _stable_json_sha256,
    _validation_map,
    _verified_contract_manifest,
    _verified_model_hashes,
    _wrap_runtime,
)


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_artifact() -> dict[str, object]:
    digest = "1" * 64
    return {
        "path": sys.executable,
        "invocation_path": sys.executable,
        "resolved_path": str(Path(sys.executable).resolve()),
        "sha256": digest,
        "interpreter_identity": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": "CPython",
        },
    }


def _attach_synthetic_part2_technical_proof(
    contract: dict[str, object],
) -> dict[str, object]:
    """Attach the smallest sealed Part-2 proof accepted by Energy planning.

    These planner fixtures exercise post-technical Quality and pairing
    annotations.  Their synthetic Split commands therefore need the same
    single-static-input proof as a real producer command, without depending on
    producer-host files that do not exist in this local integration test.
    """

    precision = str(contract.get("precision") or "").strip().lower()
    elem_type = {
        "float32_layout_fp16": "FLOAT",
        "uint8_dequant_fp16": "UINT8",
        "uint8_cast_fp16": "UINT8",
    }.get(precision, "")
    if not elem_type:
        return contract

    paths = {
        "part1_runtime": "/model.hef",
        "boundary_metadata": "/boundary_metadata.json",
        "source_part2_onnx": "/source_part2.onnx",
        "build_part2_onnx": "/part2.onnx",
        "engine": "/model.engine",
        "native_trt_meta": "/native_trt_meta.json",
        "engine_build_receipt": "/engine_build_receipt.json",
        "trtexec": "/usr/bin/trtexec",
    }
    if str(contract.get("backend") or "") == "deepx_to_trt":
        paths["part1_runtime"] = "/model.dxnn"

    metadata = {
        "schema": "onnx-splitpoint/native-trt-meta",
        "schema_version": 1,
        "variant": "part2",
        "build_ok": True,
        "inputs_static": True,
        "precision": precision,
        "requested_precision": precision,
        "inputs": [{
            "name": "cut",
            "shape": [1, 8, 8, 16],
            "elem_type": elem_type,
            "has_dynamic": False,
        }],
        "onnx": paths["build_part2_onnx"],
        "source_onnx": paths["build_part2_onnx"],
        "engine": paths["engine"],
        "engine_build_receipt_path": paths["engine_build_receipt"],
    }

    def json_bytes(payload: dict[str, object]) -> bytes:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")

    boundary_payload = {"boundary": "cut"}
    receipt_payload = {"build_returncode": 0}
    artifact_bytes = {
        "part1_runtime": b"part1-runtime",
        "boundary_metadata": json_bytes(boundary_payload),
        "source_part2_onnx": b"source-part2-onnx",
        "build_part2_onnx": b"build-part2-onnx",
        "engine": b"tensorrt-engine",
        "native_trt_meta": json_bytes(metadata),
        "engine_build_receipt": json_bytes(receipt_payload),
        "trtexec": b"trtexec-binary",
    }
    binding_artifacts = {
        name: {
            "path": paths[name],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for name, raw in artifact_bytes.items()
    }

    def embedded_json(
        payload: dict[str, object], raw: bytes,
    ) -> dict[str, object]:
        return {
            "encoding": "base64",
            "content_base64": base64.b64encode(raw).decode("ascii"),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "file_size_bytes": len(raw),
            "canonical_value_sha256": canonical_json_sha256(payload),
        }

    metadata_value_sha = canonical_json_sha256(metadata)
    local_proof: dict[str, object] = {
        "schema": (
            "onnx-splitpoint/native-split-local-artifact-verification"
        ),
        "schema_version": 1,
        "verification_kind": "producer_local_file_rehash",
        "artifact_names": list(binding_artifacts),
        "artifacts": binding_artifacts,
        "artifact_set_sha256": canonical_json_sha256(binding_artifacts),
        "embedded_json_files": {
            "boundary_metadata": embedded_json(
                boundary_payload, artifact_bytes["boundary_metadata"],
            ),
            "native_trt_meta": embedded_json(
                metadata, artifact_bytes["native_trt_meta"],
            ),
            "engine_build_receipt": embedded_json(
                receipt_payload, artifact_bytes["engine_build_receipt"],
            ),
        },
        "native_trt_meta_payload_sha256": metadata_value_sha,
    }
    local_proof["proof_sha256"] = canonical_json_sha256(local_proof)

    binding: dict[str, object] = {
        "schema": "onnx-splitpoint/native-split-quality-binding",
        "schema_version": 1,
        "artifacts": binding_artifacts,
        "local_artifact_verification": local_proof,
        "native_trt_meta_payload_sha256": metadata_value_sha,
        "native_trt_meta_payload": metadata,
        "native_trt_meta": metadata,
        "native_trt_meta_file_sha256": binding_artifacts[
            "native_trt_meta"
        ]["sha256"],
        "native_trt_meta_file_size_bytes": binding_artifacts[
            "native_trt_meta"
        ]["size_bytes"],
    }
    binding["binding_sha256"] = canonical_json_sha256(binding)

    artifacts = contract.get("artifacts")
    assert isinstance(artifacts, dict)
    artifacts.update({
        name: dict(binding_artifacts[name])
        for name in (
            "boundary_metadata", "source_part2_onnx", "build_part2_onnx",
            "engine", "native_trt_meta", "engine_build_receipt", "trtexec",
        )
    })
    if str(contract.get("backend") or "") == "deepx_to_trt":
        artifacts["dxnn"] = dict(binding_artifacts["part1_runtime"])
    else:
        artifacts["hef"] = dict(binding_artifacts["part1_runtime"])
    boundary = contract.get("boundary_contract")
    assert isinstance(boundary, dict)
    boundary.update({
        "metadata_path": paths["native_trt_meta"],
        "metadata_sha256": binding_artifacts["native_trt_meta"]["sha256"],
    })
    contract.update({
        "engine": paths["engine"],
        "engine_sha256": binding_artifacts["engine"]["sha256"],
        "native_split_quality_binding": binding,
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_local_verification": local_proof,
    })
    return contract


def _sealed_runtime_contract(row: dict[str, object]) -> dict[str, object]:
    backend = str(row.get("backend") or "")
    input_sha = str(row.get("_test_input_sha256") or "4" * 64)
    model = str(row.get("model") or "")
    task = str(
        row.get("_test_task")
        or ("classification" if "resnet" in model.lower() else "detection")
    )
    preprocess_mode = "letterbox" if task == "detection" else "resize"
    letterbox_pad_value = 114 if preprocess_mode == "letterbox" else 0
    preprocess = {
        "mode": f"{preprocess_mode}_rgb_uint8",
        "preprocess_mode_effective": preprocess_mode,
        "pad_value": letterbox_pad_value,
        "source_image_sha256": input_sha,
    }
    if backend.startswith("native_full_"):
        artifacts: dict[str, object] = {
            "command_python_executable": _python_artifact(),
        }
        if backend.startswith("native_full_hailo"):
            artifacts.update({
                "runtime_python": _python_artifact(),
                "hotloop_runner": {"path": "/tool/hotloop.py", "sha256": "2" * 64},
                "hef": {"path": "/model.hef", "sha256": "3" * 64},
                "input_manifest": {"path": "/input_manifest.json", "sha256": "6" * 64},
                "runtime_input_tensor": {
                    "path": "/runtime_input.bin", "sha256": "7" * 64,
                    "bytes": 12,
                },
            })
            workload: dict[str, object] = {
                "available": True, "status": "available", "kind": "hailo_full_hotloop",
                "runtime_python_artifact": "runtime_python",
                "runner_artifact": "hotloop_runner", "hef_artifact": "hef",
                "input_image": "/input.png", "input_image_sha256": input_sha,
                "input_manifest_artifact": "input_manifest",
                "runtime_input_artifact": "runtime_input_tensor",
                "runtime_input_name": "images",
                "runtime_input_shape": [2, 2, 3],
                "runtime_input_dtype": "uint8",
                "runtime_input_bytes": 12,
                "runtime_input_layout": "HWC",
                "runtime_input_mode": "exact_semantic_dump_runtime_tensor",
                "input_mode": "exact_semantic_dump_runtime_tensor",
                "canonical_input_slot_names": ["images"],
                "canonical_output_slot_names": ["output"],
                "preprocess": preprocess,
                "runtime_input_contract": {
                    "schema": "onnx-splitpoint/preverified-runtime-input-tensor",
                    "schema_version": 1,
                    "runtime_input_name": "images",
                    "runtime_input_shape": [2, 2, 3],
                    "runtime_input_dtype": "uint8",
                    "runtime_input_bytes": 12,
                    "runtime_input_sha256": "7" * 64,
                    "runtime_input_layout": "HWC",
                    "preprocess": preprocess,
                },
                "task": task,
                "preprocess_mode": preprocess_mode,
                "letterbox_pad_value": letterbox_pad_value,
            }
        elif backend == "native_full_deepx":
            artifacts.update({
                "runtime_python": _python_artifact(),
                "hotloop_runner": {"path": "/tool/deepx_hotloop.py", "sha256": "2" * 64},
                "dxnn": {"path": "/model.dxnn", "sha256": "3" * 64},
            })
            workload = {
                "available": True, "status": "available",
                "kind": "deepx_full_prepared_feed_hotloop",
                "runtime_python_artifact": "runtime_python",
                "runner_artifact": "hotloop_runner", "dxnn_artifact": "dxnn",
                "input_image": "/input.png", "input_image_sha256": input_sha,
                "input_contract": {"input": {
                    "shape": [1, 1, 3],
                    "preprocess_mode": preprocess_mode,
                    "letterbox_pad_value": letterbox_pad_value,
                    "source_image_sha256": input_sha,
                }},
                "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
                "task": task,
            }
        else:
            source_model_sha = str(row.get("model_sha256") or "8" * 64)
            trt_receipt: dict[str, object] = {
                "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
                "schema_version": 1, "build_returncode": 0, "dry_run": False,
                "command": [
                    "/usr/bin/trtexec", "--onnx=/model.onnx",
                    "--saveEngine=/model.engine", "--fp16",
                ],
                "source_onnx": "/model.onnx",
                "source_onnx_sha256": source_model_sha,
                "engine": "/model.engine", "engine_sha256": "3" * 64,
                "trtexec": "/usr/bin/trtexec", "trtexec_sha256": "2" * 64,
            }
            trt_receipt["receipt_sha256"] = canonical_json_sha256(trt_receipt)
            artifacts.update({
                "trtexec": {"path": "/usr/bin/trtexec", "sha256": "2" * 64},
                "source_onnx": {"path": "/model.onnx", "sha256": source_model_sha},
                "engine": {
                    "path": "/model.engine", "sha256": "3" * 64,
                    "compiled_from_source_onnx_sha256": source_model_sha,
                },
                "input_manifest": {"path": "/input_manifest.json", "sha256": "6" * 64},
                "runtime_input_tensor": {"path": "/runtime_input.bin", "sha256": "7" * 64},
                "engine_build_receipt": {
                    "path": "/engine_build_receipt.json", "sha256": "9" * 64,
                },
            })
            workload = {
                "available": True, "status": "available", "kind": "tensorrt_full_hotloop",
                "trtexec_artifact": "trtexec", "engine_artifact": "engine",
                "source_model_artifact": "source_onnx",
                "engine_build_receipt_artifact": "engine_build_receipt",
                "engine_build_receipt_status": "engine_build_receipt_verified",
                "input_manifest_artifact": "input_manifest",
                "runtime_input_artifact": "runtime_input_tensor",
                "runtime_input_name": "input", "input_mode": "exact_semantic_dump_runtime_tensor",
                "runtime_input_shape": [2, 2, 3],
                "runtime_input_dtype": "uint8",
                "runtime_input_layout": "HWC",
                "preprocess": preprocess,
                "task": task,
                "input_image_sha256": input_sha,
                "invariant_args": ["--loadEngine=/model.engine"],
            }
        contract: dict[str, object] = {
            "schema": "onnx-splitpoint/native-full-command-contract",
            "schema_version": 1,
            "backend": backend,
            "backend_arg": backend.removeprefix("native_full_"),
            "model": model,
            "case": str(row.get("case") or "full"),
            "setup_id": str(row.get("setup_id") or ""),
            "comparison_backend": str(row.get("comparison_backend") or ""),
            "comparison_precision": str(row.get("precision") or ""),
            "legacy_comparison_precision": str(row.get("precision") or ""),
            "execution_precision": str(row.get("execution_precision") or ""),
            "full_runtime_precision": str(row.get("full_runtime_precision") or ""),
            "python_executable": sys.executable,
            "runner": "scripts/native_full_baseline_eval_runner.py",
            "runner_sha256": "5" * 64,
            "root": "/evaluation",
            "benchmark_set": "/evaluation/benchmark_set",
            "input_case": "full",
            "input_image": "/input.png",
            "input_image_sha256": input_sha,
            "runtime_options": {
                "frames": 10, "warmup": 1, "inflight": 4,
                "trt_precision": "fp16", "workspace_mb": 1024,
                "engine_build_python": sys.executable, "no_shapes": False,
                "dump_outputs": False, "diagnostic_deepx_input_probes": False,
                "image_map": {},
            },
            "energy_workload": workload,
            "artifacts": artifacts,
            "complete": True,
        }
        if backend == "native_full_tensorrt":
            contract["trt_engine_build_receipt"] = trt_receipt
            contract["trt_engine_build_receipt_status"] = "engine_build_receipt_verified"
            contract["source_model_sha256"] = source_model_sha
            contract["model_binding"] = {
                "source_artifact": "source_onnx",
                "source_onnx_sha256": source_model_sha,
                "compiled_artifact": "engine",
                "compiled_artifact_sha256": "3" * 64,
                "status": "verified_engine_build_receipt_bound",
            }
        contract["contract_sha256"] = canonical_json_sha256(contract)
        return contract

    artifacts = {
        "python_executable": {
            "path": sys.executable, "sha256": "1" * 64,
        },
        "hef": {"path": "/model.hef", "sha256": "2" * 64},
        "engine": {"path": "/model.engine", "sha256": "3" * 64},
        "native_executable": {"path": "/runner", "sha256": "4" * 64},
        "dxnn": {"path": "/model.dxnn", "sha256": "5" * 64},
        "generated_cpp": {"path": "/generated.cpp", "sha256": "8" * 64},
        "cmake": {"path": "/CMakeLists.txt", "sha256": "9" * 64},
        "prepared_input": {"path": "/prepared_input.bin", "sha256": "a" * 64},
    }
    options: dict[str, object] = {
        "frames": 10, "warmup": 1, "queue_depth": 2, "dump_outputs": False,
        "dump_boundary": False, "copy_outputs": True,
        "build": False, "task": task,
        "preprocess_mode_effective": preprocess_mode,
        "letterbox_pad_value": letterbox_pad_value,
        "prepared_input_bound": True,
    }
    prepared_input_contract: dict[str, object]
    if backend == "hailo8_to_trt":
        options.update({
            "hailo_format": "float32", "energy_prepared_feed_capable": True,
            "preprocess_mode_requested": "auto", "device_id": "",
            "letterbox_pad_value_requested": letterbox_pad_value,
            "letterbox_pad_value_effective": letterbox_pad_value,
        })
        prepared_input_contract = {
            "format": "raw_rgb_uint8", "shape": [2, 2, 3], "dtype": "uint8",
            "layout": "HWC", "task": task,
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": preprocess_mode,
            "letterbox_pad_value_requested": letterbox_pad_value,
            "letterbox_pad_value_effective": letterbox_pad_value,
            "letterbox_pad_value": letterbox_pad_value,
            "pad_value_effective": letterbox_pad_value,
            "source_image_sha256": input_sha,
        }
    elif backend == "hailo10h_to_trt":
        artifacts["prepared_input_00"] = {
            "path": "/prepared_input_00.npy", "sha256": "b" * 64,
        }
        options.update({
            "inflight": 4, "producer_impl": "auto",
            "quantized_inputs": True, "quantized_outputs": True,
            "part1_onnx_used": False,
            "canonical_input_slot_names": ["images"],
            "canonical_output_slot_names": ["boundary"],
            "preprocess_mode_requested": "auto",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": letterbox_pad_value,
            "letterbox_pad_value": letterbox_pad_value,
        })
        prepared_input_contract = {
            "format": "numpy_npy_v1", "slot_order": ["images"],
            "preprocess": "exact_performance_prepared_tensor_persisted",
            "normalization": (
                "hef_quant_info_from_imagenet_float32"
                if task == "classification"
                else "hef_quant_info_from_unit_float32"
            ),
            "task": task, "preprocess_mode_requested": "auto",
            "preprocess_mode": preprocess_mode,
            "preprocess_mode_effective": preprocess_mode,
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": letterbox_pad_value,
            "letterbox_pad_value": letterbox_pad_value,
            "pad_value_effective": letterbox_pad_value,
            "source_image_sha256": input_sha,
            "entries": [{
                "name": "images", "artifact_name": "prepared_input_00",
                "shape": [1], "dtype": "uint8", "c_contiguous": True,
            }],
        }
    else:
        options.update({
            "preprocess_mode_requested": "auto",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": letterbox_pad_value,
            "letterbox_pad_value": letterbox_pad_value,
        })
        prepared_input_contract = {
            "format": "numpy_npy_v1", "shape": [2, 2, 3], "dtype": "uint8",
            "layout": "HWC", "task": task,
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": preprocess_mode,
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": letterbox_pad_value,
            "letterbox_pad_value": letterbox_pad_value,
            "source_image_sha256": input_sha,
        }
    contract = {
        "complete": True,
        "backend": backend,
        "model": model,
        "case": str(row.get("case") or ""),
        "precision": str(row.get("precision") or ""),
        "setup_id": str(row.get("setup_id") or ""),
        "comparison_backend": str(row.get("comparison_backend") or ""),
        "runner": {
            "hailo8_to_trt": "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
            "hailo10h_to_trt": "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
            "deepx_to_trt": "scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        }.get(backend, "scripts/native_fifo_eval_runner.py"),
        "runner_sha256": "6" * 64,
        "python_executable": sys.executable,
        "benchmark_set": "/evaluation/benchmark_set",
        "input_image": "/input.png",
        "input_image_sha256": input_sha,
        "artifacts": artifacts,
        "interpreter_identity": {
            "executable": sys.executable,
            "executable_sha256": "1" * 64,
        },
        "runtime_options": options,
        "boundary_contract": {"boundary_layout_effective": "nchw"},
        "prepared_input_contract": prepared_input_contract,
    }
    return seal_native_command_contract(
        _attach_synthetic_part2_technical_proof(contract)
    )


def _with_runtime_contract(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    backend = str(out.get("backend") or "")
    if not backend.startswith("native_full_"):
        precision = str(out.get("precision") or "").strip().lower()
        out["precision"] = {
            "p": "float32_layout_fp16",
            "fp16": "float32_layout_fp16",
            "int8": "uint8_dequant_fp16",
        }.get(precision, precision)
    out.setdefault("comparison_backend", {
        "hailo8_to_trt": "hailo8",
        "hailo10h_to_trt": "hailo10h",
        "deepx_to_trt": "deepx",
    }.get(backend, ""))
    out.update({
        "source_request_sha256": "1" * 64,
        "model_sha256": str(out.get("model_sha256") or "2" * 64),
        "validation_dataset_sha256": "3" * 64,
        "validation_dataset_image_ids_sha256": "4" * 64,
        "validation_dataset_ground_truth_sha256": "5" * 64,
        "accuracy_gate_policy_sha256": "6" * 64,
        "task_quality_policy_sha256": "6" * 64,
        "runtime_quality_gate_policy_sha256": "6" * 64,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_claim_result_verified": True,
    })
    field = "full_command_contract" if str(out.get("backend") or "").startswith("native_full_") else "native_command_contract"
    out[field] = _sealed_runtime_contract(out)
    return out


def _local_full_hotloop_contract(tmp_path: Path, kind: str) -> dict[str, object]:
    image = tmp_path / "input.png"
    image.write_bytes(b"exact-image")
    command_python = full_runner._python_interpreter_artifact(sys.executable)
    assert command_python is not None
    artifacts: dict[str, object] = {
        "command_python_executable": command_python,
    }
    contract_benchmark_set = tmp_path
    backend = "native_full_tensorrt"
    workload: dict[str, object]
    if kind.startswith("hailo"):
        backend = "native_full_hailo8" if kind == "hailo8" else "native_full_hailo10h"
        hef = tmp_path / "model.hef"; hef.write_bytes(b"hef")
        runner = tmp_path / "hailo_hotloop.py"; runner.write_text("# fake\n", encoding="utf-8")
        runtime_input = tmp_path / "runtime_input.bin"
        runtime_input.write_bytes(bytes(range(12)))
        input_manifest = tmp_path / "native_full_input_manifest.json"
        preprocess = {
            "mode": "resize_rgb_uint8", "layout": "HWC", "rgb": True,
            "pad_value": 0, "source_image_sha256": _raw_sha256(image),
        }
        input_manifest.write_text(json.dumps({
            "schema": "onnx-splitpoint/native-full-input-dump",
            "schema_version": 1,
            "runtime_input_name": "images",
            "runtime_input_shape": [2, 2, 3],
            "runtime_input_dtype": "uint8",
            "runtime_input_bytes": 12,
            "runtime_input_file": runtime_input.name,
            "runtime_input_sha256": _raw_sha256(runtime_input),
            "preprocess": preprocess,
        }), encoding="utf-8")
        artifacts.update({
            "runtime_python": command_python,
            "hef": {"path": str(hef), "sha256": _raw_sha256(hef)},
            "hotloop_runner": {"path": str(runner), "sha256": _raw_sha256(runner)},
            "input_manifest": {
                "path": str(input_manifest), "sha256": _raw_sha256(input_manifest),
            },
            "runtime_input_tensor": {
                "path": str(runtime_input), "sha256": _raw_sha256(runtime_input),
                "bytes": 12,
            },
        })
        workload = {
            "available": True, "status": "available", "kind": "hailo_full_hotloop",
            "runtime_python_artifact": "runtime_python", "runner_artifact": "hotloop_runner",
            "hef_artifact": "hef", "input_image": str(image),
            "input_image_sha256": _raw_sha256(image),
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "runtime_input_name": "images",
            "runtime_input_shape": [2, 2, 3],
            "runtime_input_dtype": "uint8", "runtime_input_bytes": 12,
            "runtime_input_layout": "HWC",
            "runtime_input_mode": "exact_semantic_dump_runtime_tensor",
            "input_mode": "exact_semantic_dump_runtime_tensor",
            "canonical_input_slot_names": ["images"],
            "canonical_output_slot_names": ["output"],
            "preprocess": preprocess,
            "runtime_input_contract": {
                "schema": "onnx-splitpoint/preverified-runtime-input-tensor",
                "schema_version": 1,
                "runtime_input_name": "images",
                "runtime_input_shape": [2, 2, 3],
                "runtime_input_dtype": "uint8", "runtime_input_bytes": 12,
                "runtime_input_sha256": _raw_sha256(runtime_input),
                "runtime_input_layout": "HWC", "preprocess": preprocess,
            },
            "hw_arch": kind, "runtime_api": "vstreams" if kind == "hailo8" else "infer_model",
            "task": "classification", "preprocess_mode": "resize", "letterbox_pad_value": 0,
            "warmup": 0, "inflight": 4, "quantized_inputs": True,
            "quantized_outputs": False, "persistent_activation": True,
            "hotloop": True, "copy_inputs": True, "copy_outputs": True,
        }
    elif kind == "deepx":
        backend = "native_full_deepx"
        contract_benchmark_set = tmp_path / "m" / "benchmark_set"
        source_onnx = contract_benchmark_set / "models" / "m.onnx"
        source_onnx.parent.mkdir(parents=True)
        source_onnx.write_bytes(b"sealed-source-onnx")
        (contract_benchmark_set / "benchmark_set.json").write_text(
            json.dumps({
                "schema": "onnx-splitpoint/benchmark-set",
                "schema_version": 2,
                "model_name": "m",
                "model": "models/m.onnx",
                "artifact_manifest": {
                    "schema": "onnx-splitpoint/benchmark-set",
                    "schema_version": 2,
                    "files": {"models": ["models/m.onnx"]},
                    "counts": {"models": 1},
                },
            }, sort_keys=True),
            encoding="utf-8",
        )
        source_artifacts, source_status = (
            full_runner._verified_benchmark_set_source_onnx_artifacts(
                contract_benchmark_set, "m",
            )
        )
        assert source_status == "benchmark_set_source_onnx_verified_exact"
        artifacts.update(source_artifacts)
        semantic_root = (
            contract_benchmark_set / "native_full_outputs"
            / "model=m" / "backend=native_full_deepx"
            / "setup=setup" / "comparison=deepx"
        )
        semantic_root.mkdir(parents=True)
        dxnn = (
            contract_benchmark_set / "deepx" / "deepx_m1"
            / "full" / "model.dxnn"
        )
        dxnn.parent.mkdir(parents=True)
        dxnn.write_bytes(b"dxnn")
        runner = (
            Path(__file__).resolve().parents[1]
            / "scripts/native_deepx_full_energy_hotloop.py"
        )
        runtime_input = semantic_root / "runtime_input.bin"
        runtime_input.write_bytes(bytes((1, 2, 3)))
        input_manifest = semantic_root / "native_full_input_manifest.json"
        input_manifest.write_text("{}\n", encoding="utf-8")
        preprocessing = canonical_image_preprocessing_contract(
            "classification", [1, 1],
        )
        preprocessing_sha = preprocessing_contract_sha256(preprocessing)
        numeric, numeric_sha = runtime_numeric_input_identity(
            backend="native_full_deepx",
            task="classification",
            preprocessing_contract_sha256_value=preprocessing_sha,
            runtime_input_name="input",
            runtime_input_shape=[1, 1, 3],
            runtime_input_dtype="uint8",
            runtime_input_layout="HWC",
            runtime_color_space="RGB",
            runtime_normalization="none",
        )
        artifacts.update({
            "runtime_python": command_python,
            "dxnn": {"path": str(dxnn), "sha256": _raw_sha256(dxnn)},
            "hotloop_runner": {"path": str(runner), "sha256": _raw_sha256(runner)},
            "input_manifest": {
                "path": str(input_manifest),
                "sha256": _raw_sha256(input_manifest),
            },
            "runtime_input_tensor": {
                "path": str(runtime_input),
                "sha256": _raw_sha256(runtime_input),
                "bytes": runtime_input.stat().st_size,
            },
        })
        workload = {
            "available": True, "status": "available",
            "kind": "deepx_full_prepared_feed_hotloop",
            "runtime_python_artifact": "runtime_python", "runner_artifact": "hotloop_runner",
            "dxnn_artifact": "dxnn", "input_image": str(image),
            "source_model_artifact": "source_onnx",
            "benchmark_set_manifest_artifact": "benchmark_set_manifest",
            "source_model_binding_status": source_status,
            "input_image_sha256": _raw_sha256(image),
            "runtime_input_artifact": "runtime_input_tensor",
            "input_manifest_artifact": "input_manifest",
            "runtime_input_name": "input",
            "runtime_input_shape": [1, 1, 3],
            "runtime_input_dtype": "uint8",
            "runtime_input_layout": "HWC",
            "runtime_input_bytes": runtime_input.stat().st_size,
            "runtime_input_sha256": _raw_sha256(runtime_input),
            "runtime_input_mode": "exact_semantic_dump_runtime_tensor",
            "input_mode": "exact_semantic_dump_runtime_tensor",
            "runtime_input_binding_verified": True,
            "runtime_preprocessing_identity": preprocessing,
            "runtime_preprocessing_sha256": preprocessing_sha,
            "runtime_numeric_input_identity": numeric,
            "runtime_numeric_input_sha256": numeric_sha,
            "input_contract": {"input": {"shape": [1, 1, 3], "dtype": "uint8", "layout": "HWC", "normalization": "none", "color_space": "RGB", "preprocess_mode": "resize"}},
            "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
            "original_image_wh": [1, 1],
            "warmup": 0, "task": "classification",
        }
    else:
        trtexec = tmp_path / "trtexec"; trtexec.write_bytes(b"trtexec")
        source_onnx = tmp_path / "model.onnx"; source_onnx.write_bytes(b"onnx")
        source_onnx_sha = _raw_sha256(source_onnx)
        engine = tmp_path / "model.engine"; engine.write_bytes(b"engine")
        runtime_input = tmp_path / "runtime_input.bin"; runtime_input.write_bytes(b"tensor")
        input_manifest = tmp_path / "input_manifest.json"
        input_manifest.write_text(json.dumps({"runtime_input_name": "input"}), encoding="utf-8")
        build_receipt: dict[str, object] = {
            "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
            "schema_version": 1, "build_returncode": 0, "dry_run": False,
            "command": [
                str(trtexec), f"--onnx={source_onnx}",
                f"--saveEngine={engine}", "--fp16",
            ],
            "source_onnx": str(source_onnx.resolve()),
            "source_onnx_sha256": source_onnx_sha,
            "engine": str(engine.resolve()),
            "engine_sha256": _raw_sha256(engine),
            "trtexec": str(trtexec.resolve()),
            "trtexec_sha256": _raw_sha256(trtexec),
        }
        build_receipt["receipt_sha256"] = canonical_json_sha256(build_receipt)
        build_receipt_path = tmp_path / "engine_build_receipt.json"
        build_receipt_path.write_text(
            json.dumps(build_receipt, indent=2, sort_keys=True), encoding="utf-8",
        )
        receipt_outer_sha = canonical_json_sha256(build_receipt)
        receipt_canonical_size = len(json.dumps(
            build_receipt, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8"))
        receipt_file_sha = _raw_sha256(build_receipt_path)
        artifacts.update({
            "trtexec": {
                "path": str(trtexec), "sha256": _raw_sha256(trtexec),
                "size_bytes": trtexec.stat().st_size,
            },
            "source_onnx": {
                "path": str(source_onnx), "sha256": source_onnx_sha,
                "size_bytes": source_onnx.stat().st_size,
            },
            "build_onnx": {
                "path": str(source_onnx), "sha256": source_onnx_sha,
                "size_bytes": source_onnx.stat().st_size,
            },
            "engine": {
                "path": str(engine), "sha256": _raw_sha256(engine),
                "size_bytes": engine.stat().st_size,
                "compiled_from_source_onnx_sha256": source_onnx_sha,
            },
            "input_manifest": {"path": str(input_manifest), "sha256": _raw_sha256(input_manifest)},
            "runtime_input_tensor": {"path": str(runtime_input), "sha256": _raw_sha256(runtime_input)},
            "engine_build_receipt": {
                "path": str(build_receipt_path),
                "sha256": receipt_file_sha,
                "file_sha256": receipt_file_sha,
                "file_size_bytes": build_receipt_path.stat().st_size,
                "canonical_sha256": receipt_outer_sha,
                "canonical_size_bytes": receipt_canonical_size,
                "size_bytes": receipt_canonical_size,
                "receipt_sha256": build_receipt["receipt_sha256"],
            },
        })
        quality_producer = {
            "schema": "onnx-splitpoint/tensorrt-central-quality-producer-identity",
            "schema_version": 1,
            "eval_run_id": "eval", "model_id": "m", "setup_id": "setup",
            "source_run_id": "native_full_tensorrt", "case_id": "full",
            "execution_role": "full_quality_only", "backend": "native_tensorrt",
            "variant": "full", "performance_claims_emitted": False,
            "source_onnx": {
                "path": str(source_onnx), "sha256": source_onnx_sha,
                "size_bytes": source_onnx.stat().st_size,
            },
            "build_onnx": {
                "path": str(source_onnx), "sha256": source_onnx_sha,
                "size_bytes": source_onnx.stat().st_size,
                "source_onnx_sha256": source_onnx_sha,
            },
            "engine": {
                "path": str(engine), "sha256": _raw_sha256(engine),
                "size_bytes": engine.stat().st_size,
                "source_onnx_sha256": source_onnx_sha,
                "build_onnx_sha256": source_onnx_sha,
            },
            "trtexec": {
                "path": str(trtexec), "sha256": _raw_sha256(trtexec),
                "size_bytes": trtexec.stat().st_size,
            },
            "engine_build_receipt": {
                "path": str(build_receipt_path),
                "sha256": receipt_outer_sha,
                "size_bytes": receipt_canonical_size,
                "receipt": build_receipt,
            },
            "engine_build_receipt_file_sha256": receipt_file_sha,
        }
        quality_producer["producer_identity_sha256"] = canonical_json_sha256(
            quality_producer
        )
        workload = {
            "available": True, "status": "available", "kind": "tensorrt_full_hotloop",
            "trtexec_artifact": "trtexec", "engine_artifact": "engine",
            "source_model_artifact": "source_onnx",
            "engine_build_receipt_artifact": "engine_build_receipt",
            "engine_build_receipt_status": "engine_build_receipt_verified",
            "engine_build_receipt_sha256": receipt_outer_sha,
            "engine_build_receipt_file_sha256": receipt_file_sha,
            "trt_engine_build_receipt_sha256": build_receipt["receipt_sha256"],
            "quality_first_producer_identity_sha256": quality_producer[
                "producer_identity_sha256"
            ],
            "quality_first_producer_artifact_match": True,
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "runtime_input_name": "input", "input_mode": "exact_semantic_dump_runtime_tensor",
            "invariant_args": [f"--loadEngine={engine}", "--fp16"],
            "warmup_ms": 0, "exact_iteration_evidence": "trtexec_export_times_record_count",
        }
        if kind == "tensorrt_completed":
            completed_runner = tmp_path / "native_trt_full_completed_hotloop.py"
            completed_runner.write_text("# fake completed runner\n", encoding="utf-8")
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
            artifacts["runtime_python"] = command_python
            artifacts["hotloop_runner"] = {
                "path": str(completed_runner),
                "sha256": _raw_sha256(completed_runner),
            }
            for name, artifact in frozen["implementation_artifacts"].items():
                artifacts[f"frozen_postprocess_{name}"] = {
                    "sha256": artifact["sha256"],
                }
            workload.update({
                "kind": "tensorrt_full_completed_task_hotloop",
                "runtime_python_artifact": "runtime_python",
                "runner_artifact": "hotloop_runner",
                "task": "detection",
                "input_image": str(image),
                "input_image_sha256": _raw_sha256(image),
                "e2e_scope": "full_task_pipeline",
                "completed_task_stage": "decoded_nms",
                "completed_task_contract_family": "decoded_nms",
                "measurement_concurrency": 1,
                "postprocess_required": True,
                "postprocess_included": True,
                "host_postprocess_frozen": True,
                "frozen_postprocess_contract": frozen,
                "frozen_postprocess_contract_sha256":
                    frozen["contract_sha256"],
                "frozen_postprocess_implementation_bound": True,
                "successful_run_completed_frames": 3,
                "successful_run_postprocess_completed_frames": 3,
                "original_image_wh": [80, 60],
            })
        if kind == "tensorrt_direct":
            completed_runner = tmp_path / "native_trt_full_completed_hotloop.py"
            completed_runner.write_text("# fake completed runner\n", encoding="utf-8")
            outputs = {
                "output": np.asarray(
                    [[[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]]],
                    dtype=np.float32,
                ),
            }
            source_endpoint_hash = "a" * 64
            source_attestation = {
                "schema":
                    "onnx-splitpoint/runtime-output-endpoint-attestation",
                "schema_version": 3,
                "attested": True,
                "status": "passed",
                "endpoint": "decoded_nms",
                "stage": "decoded_nms",
                "values_decoded_xyxy_score_class": True,
                "declaration_attested": True,
                "endpoint_contract_hash": source_endpoint_hash,
                "tensor_signature": tensor_signature(outputs),
                "declared_contract": {
                    "model_id": "yolo26s",
                    "source_coordinate_space": (
                        "model_input_letterbox_xyxy_pixels"
                    ),
                },
            }
            frozen = build_frozen_decoded_nms_normalization_contract(
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
                source_endpoint_contract_hash=source_endpoint_hash,
                source_output_endpoint_attestation=source_attestation,
            )
            result = FrozenDecodedNmsPostprocessor(frozen).process(
                outputs, original_wh=[1280, 720],
            )
            completed_attestation = (
                build_normalized_detection_endpoint_attestation(
                    frozen,
                    result,
                    completed_frames=3,
                    postprocess_completed_frames=3,
                )
            )
            artifacts["runtime_python"] = command_python
            artifacts["hotloop_runner"] = {
                "path": str(completed_runner),
                "sha256": _raw_sha256(completed_runner),
            }
            for name, artifact in frozen["implementation_artifacts"].items():
                artifacts[f"frozen_postprocess_{name}"] = {
                    "sha256": artifact["sha256"],
                }
            workload.update({
                "kind": "tensorrt_full_completed_task_hotloop",
                "runtime_python_artifact": "runtime_python",
                "runner_artifact": "hotloop_runner",
                "task": "detection",
                "input_image": str(image),
                "input_image_sha256": _raw_sha256(image),
                "e2e_scope": "full_task_pipeline",
                "completed_task_stage": "decoded_nms",
                "completed_task_contract_family": "decoded_nms",
                "measurement_concurrency": 1,
                "postprocess_required": True,
                "postprocess_included": True,
                "postprocess_completed_frames": 3,
                "postprocess_completion_verified": True,
                "normalization_frozen": True,
                "frozen_decoded_nms_normalization_contract": frozen,
                "frozen_decoded_nms_normalization_contract_sha256":
                    frozen["contract_sha256"],
                "frozen_decoded_nms_normalization_result": result,
                "successful_run_completed_frames": 3,
                "successful_run_postprocess_completed_frames": 3,
                "original_image_wh": [1280, 720],
                "source_endpoint_contract_hash":
                    frozen["source_endpoint_contract_hash"],
                "source_output_endpoint_id":
                    frozen["source_output_endpoint_id"],
                "source_output_tensor_signature":
                    frozen["source_output_tensor_signature"],
                "source_output_endpoint_attestation_sha256":
                    frozen["source_output_endpoint_attestation_sha256"],
                "letterbox_geometry_contract_sha256":
                    frozen["letterbox_geometry_contract_sha256"],
                "completed_task_completion_mode":
                    "integrated_accelerator_plus_frozen_normalization",
                "completed_task_endpoint_attestation":
                    completed_attestation,
            })
    contract: dict[str, object] = {
        "schema": full_runner.FULL_COMMAND_CONTRACT_SCHEMA,
        "schema_version": full_runner.FULL_COMMAND_CONTRACT_VERSION,
        "backend": backend, "backend_arg": kind, "model": "m", "case": "full",
        "setup_id": "setup", "comparison_backend": kind,
        "comparison_precision": "legacy", "legacy_comparison_precision": "legacy",
        "execution_precision": "fp16" if kind.startswith("tensorrt") else "",
        "full_runtime_precision": "fp16" if kind.startswith("tensorrt") else "",
        "python_executable": sys.executable,
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "runner_sha256": _raw_sha256(Path(full_runner.__file__)),
        "root": str(tmp_path),
        "benchmark_set": str(contract_benchmark_set),
        "input_case": "full", "input_image": str(image),
        "input_image_sha256": _raw_sha256(image),
        "runtime_options": {}, "energy_workload": workload,
        "artifacts": artifacts, "complete": True,
    }
    if kind.startswith("tensorrt"):
        contract["trt_engine_build_receipt"] = build_receipt
        contract["trt_engine_build_receipt_status"] = "engine_build_receipt_verified"
        contract["quality_first_producer_identity"] = quality_producer
        contract["quality_first_producer_identity_sha256"] = quality_producer[
            "producer_identity_sha256"
        ]
        contract["engine_build_receipt_path"] = str(build_receipt_path)
        contract["engine_build_receipt_sha256"] = receipt_outer_sha
        contract["engine_build_receipt_file_sha256"] = receipt_file_sha
        contract["trt_engine_build_receipt_sha256"] = build_receipt[
            "receipt_sha256"
        ]
        contract["engine_build_receipt_size_bytes"] = receipt_canonical_size
        contract["engine_build_receipt_file_size_bytes"] = (
            build_receipt_path.stat().st_size
        )
        contract["source_model_sha256"] = source_onnx_sha
        contract["model_binding"] = {
            "source_artifact": "source_onnx",
            "source_onnx_sha256": source_onnx_sha,
            "compiled_artifact": "engine",
            "compiled_artifact_sha256": _raw_sha256(engine),
            "status": "verified_engine_build_receipt_bound",
        }
    elif kind == "deepx":
        source_onnx_sha = str(
            contract["artifacts"]["source_onnx"]["sha256"]
        )
        contract["source_model_sha256"] = source_onnx_sha
        contract["model_binding"] = {
            "source_artifact": "source_onnx",
            "source_onnx_sha256": source_onnx_sha,
            "compiled_artifact": "dxnn",
            "compiled_artifact_sha256": contract["artifacts"]["dxnn"][
                "sha256"
            ],
            "status": "source_and_compiled_artifact_hash_bound",
        }
    contract["contract_sha256"] = full_runner._canonical_json_sha256(contract)
    return contract


def _seal_full_contract(contract: dict[str, object]) -> dict[str, object]:
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = full_runner._canonical_json_sha256(contract)
    return contract


def _planner_runtime_options() -> dict[str, object]:
    return {
        "frames": 3,
        "warmup": 0,
        "inflight": 1,
        "trt_precision": "fp16",
        "workspace_mb": 1024,
        "engine_build_python": sys.executable,
        "no_shapes": False,
        "dump_outputs": True,
        "diagnostic_deepx_input_probes": False,
        "image_map": {},
    }


def _deepx_full_energy_identity(
    contract: dict[str, object], *, remote_root: Path,
) -> dict[str, object]:
    return {
        "backend": "native_full_deepx",
        "model": "m",
        "case": "full",
        "setup_id": "setup",
        "comparison_backend": "deepx",
        "remote_root": str(remote_root),
        "remote_tool_dir": str(Path(__file__).resolve().parents[1]),
    }


def test_v27516_reporter_reaches_real_deepx_strict_validator_with_allowlist(
    tmp_path: Path,
) -> None:
    contract = _local_full_hotloop_contract(tmp_path, "deepx")
    contract["runtime_options"] = _planner_runtime_options()
    _seal_full_contract(contract)
    row = {
        "backend": "native_full_deepx",
        "model": "m",
        "case": "full",
        "setup_id": "setup",
        "comparison_backend": "deepx",
        "full_command_contract": contract,
    }

    def context(
        *, setup_id: str = "setup", remote_root: str = str(tmp_path),
        remote_tool_dir: str = str(Path(__file__).resolve().parents[1]),
    ) -> dict[str, object]:
        return {
            "schema": (
                "onnx-splitpoint/"
                "native-final-report-remote-execution-context"
            ),
            "schema_version": 1,
            "setup_id": setup_id,
            "remote_root": remote_root,
            "remote_tool_dir": remote_tool_dir,
        }

    digest, errors = final_report._verified_performance_command_contract(
        dict(row),
        remote_execution_contexts=(context(),),
    )
    assert digest == contract["contract_sha256"]
    assert errors == []

    rejected_rows = (
        (
            dict(row),
            context(remote_root=str(tmp_path / "other")),
        ),
        (
            {**row, "setup_id": "other-setup"},
            context(setup_id="other-setup"),
        ),
        (
            dict(row),
            context(remote_tool_dir=str(tmp_path / "other-tool")),
        ),
    )
    for rejected_row, rejected_context in rejected_rows:
        digest, errors = final_report._verified_performance_command_contract(
            rejected_row,
            remote_execution_contexts=(rejected_context,),
        )
        assert digest == ""
        assert errors


def test_deepx_full_energy_rejects_resealed_numeric_schema_999(
    tmp_path: Path,
) -> None:
    contract = _local_full_hotloop_contract(tmp_path, "deepx")
    contract["runtime_options"] = _planner_runtime_options()
    _seal_full_contract(contract)
    identity = _deepx_full_energy_identity(
        contract, remote_root=tmp_path,
    )

    verified, status = energy_plan._verify_full_command_contract(
        contract, expected_identity=identity,
    )
    assert verified is not None, status
    verified, status = full_runner._verified_full_energy_contract(
        contract, expected_root=tmp_path,
    )
    assert verified is not None, status

    tampered = json.loads(json.dumps(contract))
    numeric = tampered["energy_workload"][
        "runtime_numeric_input_identity"
    ]
    numeric["schema_version"] = 999
    tampered["energy_workload"][
        "runtime_numeric_input_sha256"
    ] = canonical_json_sha256(numeric)
    _seal_full_contract(tampered)

    rejected, reason = energy_plan._verify_full_command_contract(
        tampered, expected_identity=identity,
    )
    assert rejected is None
    assert reason == (
        "full_command_contract_deepx_runtime_input_binding_invalid"
    )
    rejected, reason = full_runner._verified_full_energy_contract(
        tampered, expected_root=tmp_path,
    )
    assert rejected is None
    assert reason == "full_energy_deepx_runtime_input_binding_invalid"


def test_deepx_full_energy_rejects_resealed_moved_run_root(
    tmp_path: Path,
) -> None:
    original_root = tmp_path / "original"
    moved_root = tmp_path / "moved"
    original_root.mkdir()
    contract = _local_full_hotloop_contract(original_root, "deepx")
    shutil.copytree(original_root, moved_root)

    original_prefix = str(original_root)
    moved_prefix = str(moved_root)

    def rewrite(value: object) -> object:
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and (
            value == original_prefix
            or value.startswith(original_prefix + "/")
        ):
            return moved_prefix + value[len(original_prefix):]
        return value

    moved = rewrite(contract)
    assert isinstance(moved, dict)
    moved["runtime_options"] = _planner_runtime_options()
    _seal_full_contract(moved)

    moved_identity = _deepx_full_energy_identity(
        moved, remote_root=moved_root,
    )
    verified, status = energy_plan._verify_full_command_contract(
        moved, expected_identity=moved_identity,
    )
    assert verified is not None, status
    verified, status = full_runner._verified_full_energy_contract(
        moved, expected_root=moved_root,
    )
    assert verified is not None, status

    original_identity = _deepx_full_energy_identity(
        moved, remote_root=original_root,
    )
    rejected, reason = energy_plan._verify_full_command_contract(
        moved, expected_identity=original_identity,
    )
    assert rejected is None
    assert reason == "full_command_contract_deepx_artifact_role_path_mismatch"
    rejected, reason = full_runner._verified_full_energy_contract(
        moved, expected_root=original_root,
    )
    assert rejected is None
    assert reason == "full_energy_deepx_artifact_role_path_mismatch"


def test_energy_plan_accepts_only_fully_bound_tensorrt_completed_hotloop(
    tmp_path: Path,
) -> None:
    contract = _local_full_hotloop_contract(
        tmp_path, "tensorrt_completed"
    )
    contract["runtime_options"] = _planner_runtime_options()
    _seal_full_contract(contract)
    identity = {
        "backend": "native_full_tensorrt",
        "model": "m",
        "case": "full",
        "setup_id": "setup",
        "comparison_backend": "tensorrt_completed",
    }
    verified, status = energy_plan._verify_full_command_contract(
        contract, expected_identity=identity,
    )
    assert verified is not None, status
    argv = energy_plan._full_runtime_argv(
        verified,
        duration_s=10.0,
        frames=3,
        remote_tool_dir="/remote/tools",
        authoritative_root=str(contract["root"]),
        preflight_nonce="nonce",
        preflight_attestation="/remote/preflight.json",
        preflight_max_age_s=60.0,
    )
    assert "--energy-workload-only" in argv

    for field, value in (
        ("e2e_scope", "accelerator_only"),
        ("completed_task_stage", "raw_head"),
        ("completed_task_contract_family", "raw_head"),
        ("measurement_concurrency", 2),
        ("measurement_concurrency", "1"),
        ("postprocess_required", False),
        ("postprocess_included", False),
        ("host_postprocess_frozen", False),
        ("frozen_postprocess_implementation_bound", False),
        ("successful_run_completed_frames", 0),
        ("successful_run_postprocess_completed_frames", 2),
        ("original_image_wh", [81, 60]),
        ("frozen_postprocess_contract_sha256", "f" * 64),
    ):
        tampered = json.loads(json.dumps(contract))
        tampered["energy_workload"][field] = value
        _seal_full_contract(tampered)
        rejected, reason = energy_plan._verify_full_command_contract(
            tampered, expected_identity=identity,
        )
        assert rejected is None
        assert reason == (
            "full_command_contract_tensorrt_completed_task_contract_invalid"
        )

    tampered = json.loads(json.dumps(contract))
    implementation_name = next(iter(
        tampered["energy_workload"][
            "frozen_postprocess_contract"
        ]["implementation_artifacts"]
    ))
    tampered["artifacts"][
        f"frozen_postprocess_{implementation_name}"
    ]["sha256"] = "f" * 64
    _seal_full_contract(tampered)
    rejected, reason = energy_plan._verify_full_command_contract(
        tampered, expected_identity=identity,
    )
    assert rejected is None
    assert reason == (
        "full_command_contract_tensorrt_completed_task_contract_invalid"
    )

    for field in ("runner_artifact", "runtime_python_artifact"):
        tampered = json.loads(json.dumps(contract))
        tampered["energy_workload"][field] = "missing_artifact"
        _seal_full_contract(tampered)
        rejected, reason = energy_plan._verify_full_command_contract(
            tampered, expected_identity=identity,
        )
        assert rejected is None
        assert "binding_missing" in reason or reason.endswith(
            "_artifact_missing"
        )

    tampered = json.loads(json.dumps(contract))
    tampered["energy_workload"]["kind"] = "unsealed_full_hotloop"
    _seal_full_contract(tampered)
    rejected, reason = energy_plan._verify_full_command_contract(
        tampered, expected_identity=identity,
    )
    assert rejected is None
    assert reason == "full_command_contract_workload_kind_unsupported"


def test_energy_plan_accepts_only_attested_direct_bn6_completed_hotloop(
    tmp_path: Path,
) -> None:
    contract = _local_full_hotloop_contract(
        tmp_path, "tensorrt_direct"
    )
    contract["runtime_options"] = _planner_runtime_options()
    _seal_full_contract(contract)
    identity = {
        "backend": "native_full_tensorrt",
        "model": "m",
        "case": "full",
        "setup_id": "setup",
        "comparison_backend": "tensorrt_direct",
    }
    verified, status = energy_plan._verify_full_command_contract(
        contract, expected_identity=identity,
    )
    assert verified is not None, status
    direct_common_count_only = json.loads(json.dumps(contract))
    direct_common_count_only["energy_workload"].pop(
        "successful_run_postprocess_completed_frames"
    )
    _seal_full_contract(direct_common_count_only)
    verified, status = energy_plan._verify_full_command_contract(
        direct_common_count_only, expected_identity=identity,
    )
    assert verified is not None, status

    for field, value in (
        (
            "completed_task_completion_mode",
            "frozen_host_tail",
        ),
        ("normalization_frozen", False),
        ("postprocess_completion_verified", False),
        ("postprocess_completed_frames", 2),
        ("source_endpoint_contract_hash", "f" * 64),
        ("source_output_endpoint_attestation_sha256", "f" * 64),
        ("letterbox_geometry_contract_sha256", "f" * 64),
        (
            "frozen_decoded_nms_normalization_contract_sha256",
            "f" * 64,
        ),
    ):
        tampered = json.loads(json.dumps(contract))
        tampered["energy_workload"][field] = value
        _seal_full_contract(tampered)
        rejected, reason = energy_plan._verify_full_command_contract(
            tampered, expected_identity=identity,
        )
        assert rejected is None
        assert reason == (
            "full_command_contract_tensorrt_completed_task_contract_invalid"
        )

    tampered = json.loads(json.dumps(contract))
    tampered["energy_workload"][
        "frozen_decoded_nms_normalization_result"
    ]["detections_sha256"] = "f" * 64
    _seal_full_contract(tampered)
    rejected, reason = energy_plan._verify_full_command_contract(
        tampered, expected_identity=identity,
    )
    assert rejected is None
    assert reason == (
        "full_command_contract_tensorrt_completed_task_contract_invalid"
    )

    tampered = json.loads(json.dumps(contract))
    tampered["energy_workload"][
        "completed_task_endpoint_attestation"
    ]["completed_frames"] = 2
    _seal_full_contract(tampered)
    rejected, reason = energy_plan._verify_full_command_contract(
        tampered, expected_identity=identity,
    )
    assert rejected is None
    assert reason == (
        "full_command_contract_tensorrt_completed_task_contract_invalid"
    )

    raw_scope = tmp_path / "raw_mixture"
    raw_scope.mkdir()
    raw_contract = _local_full_hotloop_contract(
        raw_scope, "tensorrt_completed"
    )
    tampered = json.loads(json.dumps(contract))
    tampered["energy_workload"]["frozen_postprocess_contract"] = (
        raw_contract["energy_workload"]["frozen_postprocess_contract"]
    )
    _seal_full_contract(tampered)
    rejected, reason = energy_plan._verify_full_command_contract(
        tampered, expected_identity=identity,
    )
    assert rejected is None
    assert reason == (
        "full_command_contract_tensorrt_completed_task_contract_invalid"
    )


def _preflight_for_contract(tmp_path: Path, contract: dict[str, object], nonce: str = "nonce") -> Path:
    attestation = tmp_path / "preflight.json"
    ns = SimpleNamespace(
        energy_command_contract_json=json.dumps(contract), preflight_nonce=nonce,
        preflight_attestation_out=str(attestation), preflight_attestation_max_age_s=60.0,
        root=str(contract["root"]),
    )
    assert full_runner._energy_preflight_only(ns) == 0
    return attestation


def _write_contract_evidence(root: Path) -> tuple[Path, str, dict[str, str]]:
    contracts_dir = root / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("classification_preprocessing", "preprocessing", "classification"),
        ("detection_preprocessing", "preprocessing", "detection"),
        ("detection_decoder", "decoder", "detection"),
        ("detection_nms", "nms", "detection"),
    )
    rows = []
    hashes: dict[str, str] = {}
    for identifier, kind, task in definitions:
        source = contracts_dir / f"{identifier}.json"
        source.write_text(
            json.dumps({"locked": True, "id": identifier, "kind": kind, "task": task}, sort_keys=True),
            encoding="utf-8",
        )
        digest = _raw_sha256(source)
        hashes[identifier] = f"sha256:{digest}"
        rows.append({
            "id": identifier,
            "kind": kind,
            "task": task,
            "path": str(source.resolve()),
            "sha256": f"sha256:{digest}",
            "locked": True,
        })
    manifest = root / "pipeline_contract_manifest.json"
    manifest.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/pipeline-contract-manifest",
            "schema_version": 1,
            "contracts": rows,
            "contract_set_sha256": f"sha256:{_stable_json_sha256(rows)}",
            "required_kinds": ["preprocessing", "decoder", "nms"],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    hashes["contract_set"] = f"sha256:{_stable_json_sha256(rows)}"
    return manifest, _raw_sha256(manifest), hashes


def _write_model_hash_evidence(root: Path, model_id: str = "m") -> tuple[Path, str, str, Path]:
    model = root / f"{model_id}.onnx"
    model.write_bytes(b"frozen onnx model bytes")
    model_digest = _raw_sha256(model)
    manifest = root / "models" / model_id / "model_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/model-manifest",
            "schema_version": 1,
            "model_id": model_id,
            "resolved_path": str(model.resolve()),
            "file": {"path": str(model.resolve()), "sha256": f"sha256:{model_digest}"},
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model_map = root / "native_energy_model_hash_map.json"
    model_map.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/native-energy-model-hash-map",
            "schema_version": 1,
            "rows": [{
                "model": model_id,
                "model_path": str(model.resolve()),
                "model_sha256": f"sha256:{model_digest}",
                "model_manifest": str(manifest.resolve()),
                "model_manifest_sha256": f"sha256:{_raw_sha256(manifest)}",
            }],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return model_map, _raw_sha256(model_map), model_digest, model


def test_runtime_work_units_require_explicit_exact_evidence(tmp_path: Path) -> None:
    (tmp_path / "workload_stdout.log").write_text(
        "__SPLITPOINT_WORK_UNITS__=17\n__SPLITPOINT_WORK_UNITS_SOURCE__=frames\n"
        "__SPLITPOINT_WORK_UNITS_EXACT__=0\n",
        encoding="utf-8",
    )
    evidence = _runtime_work_unit_evidence(tmp_path)
    assert evidence["count"] == 17
    assert evidence["exact"] is False

    (tmp_path / "runtime_completed_work_units.json").write_text(
        json.dumps({"completed_work_units": 16}), encoding="utf-8"
    )
    evidence = _runtime_work_unit_evidence(tmp_path)
    assert evidence["count"] == 16
    assert evidence["exact"] is True
    assert evidence["transport"] == "sidecar"


def test_full_preflight_is_collector_compatible_and_workload_does_no_heavy_hash_io(
    tmp_path: Path,
) -> None:
    from onnx_splitpoint_tool.energy.collector import _validate_energy_preflight_attestation

    for kind in ("hailo8", "hailo10h", "deepx", "tensorrt"):
        scope = tmp_path / kind; scope.mkdir()
        contract = _local_full_hotloop_contract(scope, kind)
        nonce = f"nonce-{kind}"
        attestation_path = _preflight_for_contract(scope, contract, nonce)
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        validated = _validate_energy_preflight_attestation(
            attestation, nonce=nonce,
            started_at_unix_ns=int(attestation["created_at_unix_ns"]),
            ended_at_unix_ns=time.time_ns(), max_age_s=60.0,
            expected_command_contract_sha256=str(contract["contract_sha256"]),
        )
        assert validated["ok"] is True, validated

        commands: list[list[str]] = []
        original_run = full_runner._run
        original_sha = full_runner._sha256_file
        frames = 7

        def fake_run(cmd: list[str], **_kwargs: object) -> dict[str, object]:
            commands.append([str(value) for value in cmd])
            if kind.startswith("hailo"):
                report = Path(cmd[cmd.index("--json-out") + 1])
                workload = dict(contract["energy_workload"])  # type: ignore[arg-type]
                runtime_contract = dict(workload["runtime_input_contract"])  # type: ignore[arg-type]
                runtime_artifact = dict(
                    contract["artifacts"][workload["runtime_input_artifact"]]  # type: ignore[index]
                )
                report.write_text(json.dumps({
                    "completed_frames": frames,
                    "runtime_input_binding_verified": True,
                    "runtime_input_source": "preflight_bound_runtime_input_tensor",
                    "runtime_input_file": str(Path(str(runtime_artifact["path"])).resolve()),
                    "runtime_input_sha256": runtime_artifact["sha256"],
                    "runtime_input_dtype": runtime_contract["runtime_input_dtype"],
                    "runtime_input_shape": runtime_contract["runtime_input_shape"],
                    "runtime_input_bytes": runtime_contract["runtime_input_bytes"],
                    "image_decode_performed": False,
                    "preprocessing_performed": False,
                    "preprocessing_timed": False,
                }), encoding="utf-8")
            elif kind == "deepx":
                report = Path(cmd[cmd.index("--json-out") + 1])
                workload = dict(
                    contract["energy_workload"]  # type: ignore[arg-type]
                )
                runtime_artifact = dict(
                    contract["artifacts"][
                        workload["runtime_input_artifact"]
                    ]  # type: ignore[index]
                )
                report.write_text(json.dumps({
                    "completed_work_units": frames,
                    "runtime_input_binding_verified": True,
                    "runtime_input_source":
                        "preflight_bound_runtime_input_tensor",
                    "runtime_input_file": str(
                        Path(str(runtime_artifact["path"])).resolve()
                    ),
                    "runtime_input_sha256": workload[
                        "runtime_input_sha256"
                    ],
                    "runtime_input_bytes": workload[
                        "runtime_input_bytes"
                    ],
                    "runtime_input_name": workload["runtime_input_name"],
                    "runtime_input_shape": workload[
                        "runtime_input_shape"
                    ],
                    "runtime_input_dtype": workload[
                        "runtime_input_dtype"
                    ],
                    "runtime_input_layout": workload[
                        "runtime_input_layout"
                    ],
                    "runtime_preprocessing_identity": workload[
                        "runtime_preprocessing_identity"
                    ],
                    "runtime_preprocessing_sha256": workload[
                        "runtime_preprocessing_sha256"
                    ],
                    "runtime_numeric_input_identity": workload[
                        "runtime_numeric_input_identity"
                    ],
                    "runtime_numeric_input_sha256": workload[
                        "runtime_numeric_input_sha256"
                    ],
                    "image_decode_performed": False,
                    "preprocessing_performed": False,
                    "preprocessing_timed": False,
                }), encoding="utf-8")
            else:
                export = next(str(value).split("=", 1)[1] for value in cmd if str(value).startswith("--exportTimes="))
                Path(export).write_text(json.dumps([{"computeMs": 1.0}] * frames), encoding="utf-8")
            return {"rc": 0, "returncode": 0, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

        try:
            full_runner._run = fake_run
            full_runner._sha256_file = lambda _path: (_ for _ in ()).throw(AssertionError("heavy hash in measured workload"))
            out = scope / "out"
            ns = SimpleNamespace(
                energy_command_contract_json=json.dumps(contract),
                preflight_attestation=str(attestation_path), preflight_nonce=nonce,
                preflight_attestation_max_age_s=60.0, out_dir=str(out),
                frames=frames, timeout=30,
                root=str(contract["root"]),
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                assert full_runner._energy_workload_only(ns) == 0
            result = json.loads(stdout.getvalue().splitlines()[-1])
            assert result["completed_work_units"] == frames
        finally:
            full_runner._run = original_run
            full_runner._sha256_file = original_sha

        assert len(commands) == 1
        argv = commands[0]
        assert "--energy-workload-only" not in argv
        assert "--backends" not in argv and "--dump-outputs" not in argv
        if kind == "tensorrt":
            assert f"--iterations={frames}" in argv
            assert "--duration=0" in argv and "--warmUp=0" in argv
            assert any(value.startswith("--exportTimes=") for value in argv)
            assert any(value.startswith("--loadInputs=input:") for value in argv)
        elif kind == "deepx":
            assert argv[1].endswith("native_deepx_full_energy_hotloop.py")
        else:
            assert argv[1].endswith("hailo_hotloop.py")
            assert "--counted-hotloop-only" in argv


def test_full_preflight_sha_mismatch_blocks_before_workload(tmp_path: Path) -> None:
    contract = _local_full_hotloop_contract(tmp_path, "hailo8")
    hef = Path(str(contract["artifacts"]["hef"]["path"]))  # type: ignore[index]
    hef.write_bytes(b"changed")
    ns = SimpleNamespace(
        energy_command_contract_json=json.dumps(contract), preflight_nonce="nonce",
        preflight_attestation_out=str(tmp_path / "attestation.json"),
        preflight_attestation_max_age_s=60.0,
        root=str(contract["root"]),
    )
    assert full_runner._energy_preflight_only(ns) != 0
    assert not Path(ns.preflight_attestation_out).exists()


def test_work_unit_wrapper_emits_exact_full_hotloop_marker() -> None:
    command = [
        sys.executable, "scripts/run_and_report_work_units.py", "--",
        sys.executable, "-c",
        "import json; print(json.dumps({'completed_work_units': 13}))",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "__SPLITPOINT_WORK_UNITS__=13" in completed.stdout
    assert "__SPLITPOINT_WORK_UNITS_EXACT__=1" in completed.stdout


def test_requested_legacy_frames_cannot_be_promoted_to_exact() -> None:
    count, source, exact = _count_from_payload({"frames": 5000}, trust_legacy_frames=True)
    assert count == 5000
    assert source == "json_field:frames"
    assert exact is False
    wrapped = _wrap_runtime("/opt/tool", "python child.py --frames 5000")
    assert "--trust-legacy-frames-as-completed" not in wrapped
    assert wrapped.endswith("-- python child.py --frames 5000")
    assert "--no-follow-reports" in _wrap_runtime(
        "/opt/tool", "python child.py --frames 5000", follow_reports=False
    )

    source = Path("scripts/native_full_baseline_eval_runner.py").read_text(encoding="utf-8")
    generic = source[source.index("def _generic_full_via_suite"):source.index("def _row_for_backend")]
    assert "_deepx_prepared_feed_projection(metric_row)" in generic
    assert 'performance_benchmark_source == "dx_engine_prepared_feed"' in generic
    assert "completed_frames == int(throughput_frames)" in generic
    assert '"completed_frames": completed_frames if exact_deepx_counter else None' in generic
    assert 'prepared_projection.get("completed_work_units_source")' in generic
    assert "deepx_postprocess_completion_verified" in generic


def test_full_semantic_validation_is_exact_and_ambiguous_identities_fail_closed(tmp_path: Path) -> None:
    row = {
        "backend": "native_full_hailo8",
        "model": "m",
        "case": "full",
        "precision": "",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "ok": True,
    }
    validation_row = {
        **row,
        "task": "detection",
        "claim_ok": True,
        "semantic_ok": True,
        "contract_consistent": True,
    }
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps({"rows": [validation_row]}), encoding="utf-8")
    accepted, reason, evidence = _semantic_decision(
        row, _validation_map(validation_path), require_claim=True
    )
    assert accepted is True
    assert reason == "native_full_semantic_gate_pass"
    assert evidence is not None and evidence["task"] == "detection"

    accepted, reason, _ = _semantic_decision(row, {}, require_claim=True)
    assert accepted is False
    assert reason == "native_full_validation_missing"

    validation_path.write_text(
        json.dumps({"rows": [validation_row, dict(validation_row)]}), encoding="utf-8"
    )
    accepted, reason, evidence = _semantic_decision(
        row, _validation_map(validation_path), require_claim=True
    )
    assert accepted is False
    assert reason == "native_validation_identity_ambiguous"
    assert evidence is not None and evidence["_ambiguous_candidate_count"] == 2


def test_contract_and_model_hash_evidence_are_locally_reverified(tmp_path: Path) -> None:
    contract_path, contract_sha, expected = _write_contract_evidence(tmp_path)
    manifest, status = _verified_contract_manifest(str(contract_path), contract_sha)
    assert status == "verified"
    fields, status = _contract_fields_for_task(manifest, "detection")
    assert status == "verified"
    assert fields == {
        "contract_hash": expected["contract_set"],
        "preprocessing_hash": expected["detection_preprocessing"],
        "decoder_hash": expected["detection_decoder"],
        "nms_hash": expected["detection_nms"],
    }

    model_map, model_map_sha, model_digest, model_path = _write_model_hash_evidence(tmp_path)
    hashes, status = _verified_model_hashes(str(model_map), model_map_sha)
    assert status == "verified"
    assert hashes == {"m": model_digest}

    model_path.write_bytes(b"changed after freeze")
    hashes, status = _verified_model_hashes(str(model_map), model_map_sha)
    assert hashes == {}
    assert status == "model_hash_source_unverified"


def test_final_energy_plan_propagates_verified_contract_to_split_and_full_rows(tmp_path: Path) -> None:
    contract_path, contract_sha, expected = _write_contract_evidence(tmp_path)
    model_map, model_map_sha, model_digest, _ = _write_model_hash_evidence(
        tmp_path, model_id="resnet50",
    )
    summary = tmp_path / "summary.json"
    validation = tmp_path / "validation.json"
    out = tmp_path / "plan"
    summary_rows = [
        {
            "ok": True, "backend": "hailo8_to_trt", "model": "resnet50", "case": "b001",
            "precision": "p", "setup_id": "h8", "fps_makespan": 10.0,
        },
        {
            "ok": True, "backend": "native_full_hailo8", "model": "resnet50", "case": "full",
            "precision": "p", "setup_id": "h8", "comparison_backend": "hailo8", "fps_makespan": 9.0,
        },
        {
            "ok": True, "backend": "native_full_tensorrt", "model": "resnet50", "case": "full",
            "precision": "p", "setup_id": "h8", "comparison_backend": "hailo8", "fps_makespan": 12.0,
        },
    ]
    summary_rows = [
        _with_runtime_contract({
            **row,
            "model_sha256": model_digest,
            "task": "classification",
            "stage": "classification_logits",
            "contract_family": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "d" * 64,
            "output_endpoint_id":
                f"classification:classification_logits:{'d' * 64}",
        })
        for row in summary_rows
    ]
    validation_rows = [
        {
            **row,
            "task": "classification",
            "top1_match": True,
            "claim_ok": True,
            "semantic_ok": True,
            "contract_consistent": True,
        }
        for row in summary_rows
    ]
    summary.write_text(json.dumps({"rows": summary_rows}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": validation_rows}), encoding="utf-8")
    command = [
        sys.executable,
        "scripts/native_producer_energy_plan.py",
        "--summary", str(summary),
        "--validation-summary", str(validation),
        "--out-dir", str(out),
        "--hailo8-ssh", "host",
        "--duration-s", "1",
        "--require-runtime-work-units",
        "--require-command-window-alignment",
        "--physical-scope", "MB",
        "--pipeline-contract-manifest", str(contract_path),
        "--pipeline-contract-sha256", contract_sha,
        "--model-hash-map", str(model_map),
        "--model-hash-map-sha256", model_map_sha,
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out / "native_producer_energy_plan.json").read_text(encoding="utf-8"))
    assert payload["pipeline_contract_manifest_status"] == "verified"
    assert payload["model_hash_map_status"] == "verified"
    assert len(payload["rows"]) == 3
    for row in payload["rows"]:
        assert row["semantic_claim_ok"] is False
        assert row["contract_evidence_ok"] is True
        # This fixture intentionally invokes the planner outside a managed
        # EvalRun.  It remains useful as a compatibility diagnostic, but may
        # not emit a scientific claim without a current authority record.
        assert row["claim_ok"] is False
        assert row["historical_diagnostic_only"] is True
        assert row["eligible_for_energy_results_import"] is False
        assert row["eligible_for_scientific_claim"] is False
        assert row["scientific_claim_exclusion_reasons"] == [
            "standalone_unmanaged_native_energy_diagnostic_only"
        ]
        assert row["task"] == "classification"
        assert row["contract_hash"] == expected["contract_set"]
        assert row["preprocessing_hash"] == expected[
            "classification_preprocessing"
        ]
        assert row["decoder_hash"] == ""
        assert row["nms_hash"] == ""
        assert row["model_sha256"] == model_digest
        assert row["prepared_feed_task"] == "classification"
        assert row["prepared_feed_preprocess_mode"] == "resize"
        assert row["prepared_feed_letterbox_pad_value"] == "0"
        assert row["prepared_feed_source_image_sha256"] == "4" * 64
        assert row["preflight_required"] is True
        assert "--preflight-command-file" in row["measure_command"]
        assert "--preflight-runtime-attestation-path" in row["measure_command"]
        assert "--preflight-expected-command-contract-sha256" in row["measure_command"]
        preflight_text = Path(row["preflight_command_file"]).read_text(encoding="utf-8")
        workload_text = Path(row["command_file"]).read_text(encoding="utf-8")
        assert "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__" in preflight_text
        assert "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__" in workload_text
        assert "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__" in preflight_text
        assert "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__" in workload_text
        assert "run_and_report_work_units.py" not in workload_text

    input_mismatch_rows = [
        _with_runtime_contract({
            **{key: value for key, value in row.items() if key not in {"native_command_contract", "full_command_contract"}},
            **({"_test_input_sha256": "9" * 64} if row["backend"] == "native_full_hailo8" else {}),
        })
        for row in summary_rows
    ]
    summary.write_text(json.dumps({"rows": input_mismatch_rows}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": [{
        **row, "task": "classification", "top1_match": True, "claim_ok": True,
        "semantic_ok": True, "contract_consistent": True,
    } for row in input_mismatch_rows]}), encoding="utf-8")
    identity_out = tmp_path / "plan_input_identity_mismatch"
    identity_command = list(command)
    identity_command[identity_command.index("--out-dir") + 1] = str(identity_out)
    completed = subprocess.run(
        identity_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    identity_payload = json.loads(
        (identity_out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )
    assert len(identity_payload["rows"]) == 3
    assert identity_payload["pair_count"] == 0
    identity_failure = identity_payload["paired_missing_rows"][0]
    assert identity_failure["reason"] == "pair_contract_identity_missing_or_mismatch"
    assert "validation_input_or_image_sha256" in identity_failure[
        "mismatched_claim_identity_fields"
    ]
    for row in identity_payload["rows"]:
        assert row["native_energy_planner_admission"]["selected"] is True
        assert row["runtime_success"] is True
        assert row["energy_command_preflight_ok"] is True
        assert row["claim_ok"] is False
        assert row["claim_eligible"] is False
        assert row["energy_claim_eligible"] is False
        assert row["eligible_for_energy_results_import"] is False
        assert row["eligible_for_scientific_claim"] is False

    preprocess_mismatch_rows = json.loads(json.dumps(summary_rows))
    mismatched_full = next(
        row for row in preprocess_mismatch_rows
        if row["backend"] == "native_full_hailo8"
    )
    mismatched_contract = mismatched_full["full_command_contract"]
    mismatched_contract.pop("contract_sha256", None)
    letterbox_preprocess = {
        "mode": "letterbox_rgb_uint8",
        "preprocess_mode_effective": "letterbox",
        "pad_value": 114,
        "source_image_sha256": "4" * 64,
    }
    mismatched_contract["energy_workload"]["preprocess"] = letterbox_preprocess
    mismatched_contract["energy_workload"]["runtime_input_contract"]["preprocess"] = (
        letterbox_preprocess
    )
    mismatched_contract["energy_workload"]["preprocess_mode"] = "letterbox"
    mismatched_contract["energy_workload"]["letterbox_pad_value"] = 114
    mismatched_contract["contract_sha256"] = canonical_json_sha256(mismatched_contract)
    summary.write_text(json.dumps({"rows": preprocess_mismatch_rows}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": [{
        **row, "task": "classification", "top1_match": True, "claim_ok": True,
        "semantic_ok": True, "contract_consistent": True,
    } for row in preprocess_mismatch_rows]}), encoding="utf-8")
    preprocess_out = tmp_path / "plan_preprocess_identity_mismatch"
    preprocess_command = list(command)
    preprocess_command[preprocess_command.index("--out-dir") + 1] = str(preprocess_out)
    completed = subprocess.run(
        preprocess_command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    preprocess_payload = json.loads(
        (preprocess_out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )
    assert len(preprocess_payload["rows"]) == 3
    assert preprocess_payload["pair_count"] == 0
    preprocess_failure = preprocess_payload["paired_missing_rows"][0]
    assert preprocess_failure["reason"] == "pair_contract_identity_missing_or_mismatch"
    assert "prepared_feed_preprocess_mode" in preprocess_failure["mismatched_claim_identity_fields"]
    for row in preprocess_payload["rows"]:
        assert row["native_energy_planner_admission"]["selected"] is True
        assert row["runtime_success"] is True
        assert row["energy_command_preflight_ok"] is True
        assert row["claim_ok"] is False
        assert row["claim_eligible"] is False
        assert row["energy_claim_eligible"] is False
        assert row["eligible_for_energy_results_import"] is False
        assert row["eligible_for_scientific_claim"] is False

    # Full runtimes do not have a Split boundary.  Their runtime precision is
    # reported separately and must not prevent a setup-local pair.
    mismatched_summary = [_with_runtime_contract(dict(summary_rows[0]))] + [
        _with_runtime_contract({**row, "precision": "q"}) for row in summary_rows[1:]
    ]
    mismatched_validation = [
        {
            **row,
            "task": "classification",
            "top1_match": True,
            "claim_ok": True,
            "semantic_ok": True,
            "contract_consistent": True,
        }
        for row in mismatched_summary
    ]
    summary.write_text(json.dumps({"rows": mismatched_summary}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": mismatched_validation}), encoding="utf-8")
    mismatch_out = tmp_path / "plan_mismatched_precision"
    mismatch_command = list(command)
    mismatch_command[mismatch_command.index("--out-dir") + 1] = str(mismatch_out)
    completed = subprocess.run(
        mismatch_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    mismatch_payload = json.loads(
        (mismatch_out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )
    assert len(mismatch_payload["rows"]) == 3
    assert mismatch_payload["pair_count"] == 1
    assert mismatch_payload["paired_missing_rows"] == []
    group = mismatch_payload["paired_groups"][0]
    assert group["split_boundary_precision"] == "float32_layout_fp16"
    assert group["vendor_full_comparison_precision"] == "q"
    assert group["tensorrt_full_comparison_precision"] == "q"
    assert group["vendor_full_runtime_precision"] == ""
    assert group["tensorrt_full_runtime_precision"] == ""
    assert group["comparison_precision_match_required"] is False
    assert group["full_runtime_precision_match_required"] is False
    split_row = next(row for row in mismatch_payload["rows"] if row["backend"] == "hailo8_to_trt")
    full_rows = [row for row in mismatch_payload["rows"] if row["backend"].startswith("native_full_")]
    assert split_row["split_boundary_precision"] == "float32_layout_fp16"
    assert split_row["full_runtime_precision"] == ""
    assert all(row["split_boundary_precision"] == "" for row in full_rows)
    assert all(row["comparison_precision"] == "q" for row in full_rows)
    assert all(row["full_runtime_precision"] == "" for row in full_rows)

    # Keep the replayable command intact while removing only the report-side
    # precision annotation.  P0.3 retains the technically valid measurement
    # and renders its boundary precision from the verified command contract.
    missing_precision = _with_runtime_contract(dict(summary_rows[0]))
    missing_precision["precision"] = ""
    summary.write_text(json.dumps({"rows": [missing_precision]}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": [{
        **missing_precision,
        "task": "classification",
        "top1_match": True,
        "claim_ok": True,
        "semantic_ok": True,
        "contract_consistent": True,
    }]}), encoding="utf-8")
    missing_out = tmp_path / "plan_missing_precision"
    missing_command = list(command)
    missing_command[missing_command.index("--out-dir") + 1] = str(missing_out)
    missing_command.append("--allow-unpaired")
    completed = subprocess.run(
        missing_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    missing_payload = json.loads(
        (missing_out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )
    assert len(missing_payload["rows"]) == 1
    [missing_row] = missing_payload["rows"]
    assert missing_row["backend"] == "hailo8_to_trt"
    assert missing_row["split_boundary_precision"] == "float32_layout_fp16"
    admission = missing_row["native_energy_planner_admission"]
    assert admission["selected"] is True
    assert admission["runtime_success"] is True
    assert admission["energy_command_preflight_ok"] is True
    assert admission["full_baseline"] is False
    assert admission["split_has_valid_part2_input"] is True
    assert not any(
        row.get("reason") == "precision_missing"
        for row in missing_payload["excluded_rows"]
    )


def test_energy_plan_identity_keeps_precision_and_full_comparison_context() -> None:
    split = {
        "backend": "hailo8_to_trt", "model": "m", "case": "b001",
        "precision": "fp16",
    }
    full = {
        "backend": "native_full_tensorrt", "model": "m", "case": "full",
        "precision": "fp16", "comparison_backend": "hailo8",
    }
    assert _dedupe_key(split, "setup-a") != _dedupe_key({**split, "precision": "int8"}, "setup-a")
    assert _dedupe_key(full, "setup-a") != _dedupe_key(
        {**full, "comparison_backend": "deepx"}, "setup-a"
    )


def test_energy_plan_command_files_are_contract_unique(tmp_path: Path) -> None:
    rows = [_with_runtime_contract({
        "ok": True, "backend": "hailo8_to_trt", "model": "m", "case": "b001",
        "precision": precision, "setup_id": "h8", "fps_makespan": 10.0,
    }) for precision in ("fp16", "int8")]
    summary = tmp_path / "summary.json"
    validation = tmp_path / "validation.json"
    summary.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": [{
        **row, "task": "classification", "top1_match": True,
        "claim_ok": True, "semantic_ok": True, "contract_consistent": True,
    } for row in rows]}), encoding="utf-8")
    out = tmp_path / "plan"
    completed = subprocess.run([
        sys.executable, "scripts/native_producer_energy_plan.py",
        "--summary", str(summary), "--validation-summary", str(validation),
        "--out-dir", str(out), "--hailo8-ssh", "host", "--allow-unpaired",
        "--duration-s", "1",
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out / "native_producer_energy_plan.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2
    command_files = [Path(row["command_file"]) for row in payload["rows"]]
    assert len({str(path) for path in command_files}) == 2
    for source, planned in zip(rows, payload["rows"]):
        digest = str(source["native_command_contract"]["contract_sha256"])
        assert digest[:12] in Path(planned["command_file"]).name
        assert digest in Path(planned["command_file"]).read_text(encoding="utf-8")


def test_energy_plan_full_precision_ambiguity_fails_closed(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    validation = tmp_path / "validation.json"
    rows = [
        {
            "ok": True, "backend": "hailo10h_to_trt", "model": "resnet50",
            "case": "b052", "precision": "uint8_dequant_fp16",
            "setup_id": "orin_nx_hailo10_01", "fps_makespan": 136.0,
        },
        {
            "ok": True, "backend": "native_full_hailo10h", "model": "resnet50",
            "case": "full", "precision": "uint8_cast_fp16",
            "setup_id": "orin_nx_hailo10_01", "comparison_backend": "hailo10h",
            "fps_makespan": 264.0,
        },
        {
            "ok": True, "backend": "native_full_tensorrt", "model": "resnet50",
            "case": "full", "precision": "uint8_cast_fp16",
            "setup_id": "orin_nx_hailo10_01", "comparison_backend": "hailo10h",
            "fps_makespan": 585.0,
        },
    ]

    def write_evidence(current_rows: list[dict[str, object]]) -> None:
        contracted_rows = [_with_runtime_contract(row) for row in current_rows]
        summary.write_text(json.dumps({"rows": contracted_rows}), encoding="utf-8")
        validation.write_text(json.dumps({"rows": [
            {
                **row, "task": "classification", "top1_match": True,
                "contract_consistent": True, "claim_ok": True, "semantic_ok": True,
            }
            for row in contracted_rows
        ]}), encoding="utf-8")

    def run_plan(out: Path) -> dict[str, object]:
        completed = subprocess.run([
            sys.executable, "scripts/native_producer_energy_plan.py",
            "--summary", str(summary), "--validation-summary", str(validation),
            "--out-dir", str(out), "--hailo10-ssh", "host", "--duration-s", "1",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        assert completed.returncode == 0, completed.stderr
        return json.loads((out / "native_producer_energy_plan.json").read_text(encoding="utf-8"))

    write_evidence(rows)
    valid = run_plan(tmp_path / "valid")
    assert valid["pair_count"] == 1
    assert len(valid["rows"]) == 3
    assert valid["paired_groups"][0]["split_boundary_precision"] == "uint8_dequant_fp16"
    assert valid["paired_groups"][0]["vendor_full_comparison_precision"] == "uint8_cast_fp16"

    ambiguous_vendor = {
        **rows[1], "precision": "fp16", "fps_makespan": 263.0,
    }
    write_evidence([*rows, ambiguous_vendor])
    ambiguous = run_plan(tmp_path / "ambiguous")
    assert ambiguous["pair_count"] == 0
    assert len(ambiguous["rows"]) == 4
    failure = ambiguous["paired_missing_rows"][0]
    assert failure["reason"] == "pair_baseline_ambiguous"
    assert failure["ambiguous_baselines"] == ["vendor_full"]
    assert failure["baseline_candidate_counts"] == {
        "vendor_full": 2, "tensorrt_full": 1,
    }
    assert failure["baseline_candidate_comparison_precisions"]["vendor_full"] == [
        "fp16", "uint8_cast_fp16",
    ]
    assert failure["baseline_candidate_runtime_precisions"]["vendor_full"] == []
    for row in ambiguous["rows"]:
        assert row["native_energy_planner_admission"]["selected"] is True
        assert row["runtime_success"] is True
        assert row["energy_command_preflight_ok"] is True
        assert row["claim_ok"] is False
        assert row["claim_eligible"] is False
        assert row["energy_claim_eligible"] is False
        assert row["eligible_for_energy_results_import"] is False
        assert row["eligible_for_scientific_claim"] is False


def test_calibration_manifest_is_local_and_sha_verified(tmp_path: Path) -> None:
    manifest = tmp_path / "calibration.json"
    manifest.write_text('{"instrument":"uRECS"}\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert _verify_calibration_manifest(manifest, digest)["verified"] is True
    assert _verify_calibration_manifest(manifest, f"sha256:{digest}")["verified"] is True
    mismatch = _verify_calibration_manifest(manifest, "0" * 64)
    assert mismatch["verified"] is False
    assert mismatch["status"] == "sha256_mismatch"
    external = _verify_calibration_manifest("https://example.invalid/calibration.json", digest)
    assert external["status"] == "external_attestation_not_locally_verifiable"


def test_complete_window_and_matching_duration_do_not_prove_command_alignment(tmp_path: Path) -> None:
    result = _window_alignment(
        {
            "power_calculations_mode": "complete_window",
            "active_duration_s": 10.0,
            "workload_execution_duration_s": 10.0,
            "command_window_trace_binding": {"verified": False, "status": "postprocessor_binding_not_supported"},
        },
        "command_window",
    )
    assert result["energy_window_duration_diagnostic_status"] == "within_tolerance"
    assert result["energy_window_alignment_status"] == "fail"
    assert result["energy_window_alignment_reason"] == "postprocessor_binding_not_supported"


def test_local_trace_command_binding_is_verified(tmp_path: Path) -> None:
    trace = tmp_path / "trace.parquet"
    trace.write_bytes(b"trace bytes")
    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    result_data = {"firmware_results": {"energy": 2.5, "duration": 1.0}}
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(result_data, sort_keys=True), encoding="utf-8")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (tmp_path / "command_window_binding.json").write_text(
        json.dumps(
            {
                "binding_method": "trace_timestamp_crop",
                "trace_path": "trace.parquet",
                "trace_sha256": trace_sha,
                "result_path": "results.json",
                "result_sha256": result_sha,
                "energy_semantics": "raw_input_energy",
                "energy_field": "firmware_results.energy",
                "raw_input_energy_j": 2.5,
                "command_start_ns": 1_000_000_000,
                "command_end_ns": 2_000_000_000,
                "integrated_start_ns": 1_000_000_000,
                "integrated_end_ns": 2_000_000_000,
                "alignment_tolerance_ns": 0,
            }
        ),
        encoding="utf-8",
    )
    binding = _command_window_binding(
        tmp_path,
        result_data,
        {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000},
        result_path=result_path,
    )
    assert binding["verified"] is True
    assert binding["raw_input_energy_verified"] is True
    aligned = _window_alignment(
        {
            "power_calculations_mode": "complete_window",
            "active_duration_s": 1.0,
            "workload_execution_duration_s": 1.0,
            "command_window_trace_binding": binding,
        },
        "command_window",
    )
    assert aligned["energy_window_alignment_status"] == "pass"
    assert aligned["energy_window_effective"] == "command_window"


def test_trace_binding_without_raw_energy_result_provenance_fails_closed(tmp_path: Path) -> None:
    trace = tmp_path / "trace.parquet"
    trace.write_bytes(b"trace bytes")
    (tmp_path / "command_window_binding.json").write_text(
        json.dumps({
            "binding_method": "trace_timestamp_crop",
            "trace_path": "trace.parquet",
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "command_start_ns": 1,
            "command_end_ns": 2,
            "integrated_start_ns": 1,
            "integrated_end_ns": 2,
        }),
        encoding="utf-8",
    )
    binding = _command_window_binding(tmp_path, {}, {"start_ns": 1, "end_ns": 2})
    assert binding["verified"] is False
    assert binding["raw_input_energy_verified"] is False
    assert binding["status"] == "result_path_or_sha256_missing"


def test_measurement_scope_wins_over_requested_full_system_label(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "native_energy_measurements"
    target.mkdir(parents=True)
    data = {
        "rows": [
            {
                "row": {
                    "backend": "hailo8_to_trt",
                    "model": "resnet50",
                    "case": "b001",
                    "setup_id": "h8",
                    "energy_scope": "full_system",
                },
                "ok": True,
                "run": {
                    "rc": 0,
                    "stdout_tail": '{"energy_physical_scope":"MB","energy_total_j":10,"postprocess_status":"ok"}',
                },
            }
        ]
    }
    (target / "native_producer_energy_results.json").write_text(json.dumps(data), encoding="utf-8")
    row = collect_native_energy(tmp_path)[0]
    assert row["energy_scope_requested"] == "full_system"
    assert row["energy_scope"] == "MB"
    assert row["claim_eligible"] is False


def test_native_energy_precision_roles_survive_ingestion_and_canonical_reporting(
    tmp_path: Path,
) -> None:
    target = tmp_path / "reports" / "native_energy_measurements"
    target.mkdir(parents=True)
    plan = {
        "backend": "native_full_hailo10h",
        "model": "resnet50",
        "case": "full",
        "precision": "legacy_split_stratum",
        "execution_precision": "fp16",
        "split_boundary_precision": "",
        "full_runtime_precision": "fp16",
        "comparison_precision": "legacy_split_stratum",
        "setup_id": "orin_nx_hailo10_01",
        "duration_s": 60.0,
    }
    (target / "native_producer_energy_results.json").write_text(json.dumps({
        "rows": [{
            "row": plan,
            "ok": True,
            "run": {"rc": 0},
            "active_duration_s": 59.5,
        }],
    }), encoding="utf-8")

    observation = collect_native_energy(tmp_path)[0]
    assert observation["precision"] == "legacy_split_stratum"
    assert observation["comparison_precision"] == "legacy_split_stratum"
    assert observation["execution_precision"] == "fp16"
    assert observation["split_boundary_precision"] == ""
    assert observation["full_runtime_precision"] == "fp16"
    assert observation["target_duration_s"] == 60.0
    assert observation["active_duration_s"] == 59.5
    assert (
        observation["active_duration_relative_tolerance"]
        == NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE
    )
    assert (
        observation["active_duration_comparison_policy"]
        == NATIVE_ENERGY_ACTIVE_DURATION_COMPARISON_POLICY
    )
    canonical = scientific_energy_rows(tmp_path)[0]
    assert canonical["split_boundary_precision"] == ""
    assert canonical["comparison_precision"] == "legacy_split_stratum"
    assert canonical["full_runtime_precision"] == "fp16"
    assert canonical["target_duration_s"] == 60.0
    assert canonical["active_duration_s"] == 59.5


def _native_pair_row(*, backend: str, mode: str, energy: float) -> dict[str, object]:
    request_sha = "1" * 64 if mode == "native_split" else "2" * 64
    return {
        "backend": backend,
        "model": "yolo26s",
        "case": "full" if mode == "native_full_baseline" else "b101",
        "setup_id": "h8",
        "execution_mode": mode,
        "task": "detection",
        "task_source": "declared",
        "evaluation_role": "holdout",
        "direction": "hailo8_to_trt",
        "precision": "uint8_cast_fp16",
        "contract_hash": "contract",
        "preprocessing_hash": "preproc",
        "decoder_hash": "decoder",
        "nms_hash": "nms",
        "pipeline_contract_sha256": "3" * 64,
        "pipeline_preprocessing_sha256": "4" * 64,
        "pipeline_decoder_sha256": "5" * 64,
        "pipeline_nms_sha256": "6" * 64,
        "quality_contract_sha256": "7" * 64,
        "preprocessing_contract_sha256": "8" * 64,
        "decoder_contract_sha256": "9" * 64,
        "nms_contract_sha256": "a" * 64,
        "source_request_sha256": request_sha,
        "model_sha256": "b" * 64,
        "validation_dataset_sha256": "c" * 64,
        "validation_dataset_image_ids_sha256": "d" * 64,
        "validation_dataset_ground_truth_sha256": "e" * 64,
        "accuracy_gate_policy_sha256": "f" * 64,
        "task_quality_policy_sha256": "f" * 64,
        "runtime_quality_gate_policy_sha256": "f" * 64,
        "quality_provenance_complete": True,
        "validation_input_or_image_sha256": "1" * 64,
        "prepared_feed_task": "detection",
        "prepared_feed_preprocess_mode": "letterbox",
        "prepared_feed_letterbox_pad_value": "114",
        "prepared_feed_source_image_sha256": "1" * 64,
        "energy_scope": "MB",
        "energy_window_effective": "command_window",
        "duration_s": 60.0,
        "target_duration_s": 60.0,
        "active_duration_s": 60.0,
        "claim_eligible": True,
        "semantic_claim_ok": True,
        "contract_consistent": True,
        "runtime_work_units_exact": True,
        "energy_per_work_j": energy,
        "energy_primary_metric": "raw_input_energy",
        "energy_raw_primary": True,
        "average_power_w": 10.0,
        "ok": True,
    }


def test_native_energy_pair_requires_full_contract_and_both_claim_gates() -> None:
    split = _native_pair_row(backend="hailo8_to_trt", mode="native_split", energy=0.10)
    full = _native_pair_row(backend="native_full_hailo8", mode="native_full_baseline", energy=0.12)
    pair = build_native_energy_pairs([split, full])[0]
    assert pair["comparable"] is True
    assert abs(float(pair["energy_delta_j"]) + 0.02) < 1e-12

    incompatible = dict(full, pipeline_preprocessing_sha256="0" * 64)
    pair = build_native_energy_pairs([split, incompatible])[0]
    assert pair["comparable"] is False
    assert "pipeline_preprocessing_mismatch_or_missing" in pair["comparison_reasons"]
    assert pair["energy_delta_j"] is None

    different_input = dict(full, validation_input_or_image_sha256="different-input")
    pair = build_native_energy_pairs([split, different_input])[0]
    assert pair["comparable"] is False
    assert "validation_input_or_image_hash_mismatch_or_missing" in pair["comparison_reasons"]

    different_preprocess = dict(full, prepared_feed_preprocess_mode="resize")
    pair = build_native_energy_pairs([split, different_preprocess])[0]
    assert pair["comparable"] is False
    assert "prepared_feed_preprocess_mode_mismatch_or_missing" in pair["comparison_reasons"]
    assert pair["energy_ratio"] is None

    missing_precision = dict(full, precision="", full_runtime_precision="")
    pair = build_native_energy_pairs([split, missing_precision])[0]
    assert pair["comparable"] is True
    assert pair["baseline_comparison_precision"] == ""
    assert pair["baseline_runtime_precision"] == ""

    pair = build_native_energy_pairs([
        dict(split, precision="", split_boundary_precision=""),
        dict(full, precision="", full_runtime_precision=""),
    ])[0]
    assert pair["comparable"] is False
    assert "split_boundary_precision_missing" in pair["comparison_reasons"]
    assert "baseline_runtime_precision_missing" not in pair["comparison_reasons"]


def test_native_energy_pair_rejects_stale_endpoint_id_with_different_valid_hash() -> None:
    endpoint_hash = "1" * 64
    endpoint_id = f"detection:decoded_nms:{endpoint_hash}"
    endpoint = {
        "stage": "decoded_nms", "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": endpoint_id, "output_endpoint_match": True,
    }
    split = {
        **_native_pair_row(
            backend="hailo8_to_trt", mode="native_split", energy=0.10,
        ),
        **endpoint,
    }
    full = {
        **_native_pair_row(
            backend="native_full_hailo8", mode="native_full_baseline", energy=0.12,
        ),
        **endpoint,
    }
    assert build_native_energy_pairs([split, full])[0]["comparable"] is True

    forged = dict(
        full, endpoint_contract_hash="2" * 64,
        # Deliberately preserve the stale ID from the split row.
        output_endpoint_id=endpoint_id,
    )
    rejected = build_native_energy_pairs([split, forged])[0]
    assert rejected["comparable"] is False
    assert "endpoint_contract_hash_mismatch_or_missing" in rejected["comparison_reasons"]
    assert "output_endpoint_mismatch_or_missing" in rejected["comparison_reasons"]


def test_native_energy_pair_active_duration_tolerance_is_fail_closed() -> None:
    split = _native_pair_row(
        backend="hailo8_to_trt", mode="native_split", energy=0.10,
    )
    within_tolerance = dict(
        _native_pair_row(
            backend="native_full_hailo8",
            mode="native_full_baseline",
            energy=0.12,
        ),
        active_duration_s=62.9,
    )
    pair = build_native_energy_pairs([split, within_tolerance])[0]
    assert pair["comparable"] is True
    assert pair["target_duration_s"] == 60.0
    assert pair["split_active_duration_s"] == 60.0
    assert pair["baseline_active_duration_s"] == 62.9
    assert abs(float(pair["active_duration_relative_delta"]) - 2.9 / 60.0) < 1e-12
    assert (
        pair["active_duration_relative_tolerance"]
        == NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE
        == 0.05
    )
    assert pair["active_duration_within_tolerance"] is True
    assert pair["energy_ratio"] is not None

    outside_tolerance = dict(within_tolerance, active_duration_s=63.1)
    pair = build_native_energy_pairs([split, outside_tolerance])[0]
    assert pair["comparable"] is False
    assert pair["active_duration_within_tolerance"] is False
    assert "active_duration_relative_tolerance_exceeded" in pair["comparison_reasons"]
    assert pair["energy_delta_j"] is None
    assert pair["energy_ratio"] is None
    assert pair["energy_saving_fraction"] is None

    missing_active_duration = dict(within_tolerance)
    missing_active_duration.pop("active_duration_s")
    pair = build_native_energy_pairs([split, missing_active_duration])[0]
    assert pair["comparable"] is False
    assert pair["active_duration_within_tolerance"] is None
    assert "baseline_active_duration_missing_or_invalid" in pair["comparison_reasons"]
    assert pair["energy_ratio"] is None

    missing_target_duration = dict(within_tolerance)
    missing_target_duration.pop("target_duration_s")
    missing_target_duration.pop("duration_s")
    pair = build_native_energy_pairs([split, missing_target_duration])[0]
    assert pair["comparable"] is False
    assert "baseline_target_duration_missing_or_invalid" in pair["comparison_reasons"]
    assert pair["energy_ratio"] is None


def test_native_energy_pair_allows_distinct_split_and_full_precisions() -> None:
    split = dict(
        _native_pair_row(backend="hailo8_to_trt", mode="native_split", energy=0.10),
        precision="float32_layout_fp16",
        split_boundary_precision="float32_layout_fp16",
    )
    full = dict(
        _native_pair_row(backend="native_full_hailo8", mode="native_full_baseline", energy=0.12),
        precision="uint8_cast_fp16",
        full_runtime_precision="uint8_cast_fp16",
    )
    pair = build_native_energy_pairs([split, full])[0]
    assert pair["comparable"] is True
    assert pair["split_boundary_precision"] == "float32_layout_fp16"
    assert pair["baseline_runtime_precision"] == "uint8_cast_fp16"
    assert pair["comparison_precision_match_required"] is False
    assert pair["full_runtime_precision_match_required"] is False


def test_native_full_lookup_is_direction_scoped_and_never_last_write_wins() -> None:
    split = _native_pair_row(backend="hailo8_to_trt", mode="native_split", energy=0.10)
    matching = _native_pair_row(backend="native_full_tensorrt", mode="native_full_baseline", energy=0.15)
    matching.update({
        "host_normalization_role": "tensorrt_full",
        "host_normalization_source_run_id": "native_full_tensorrt",
        "host_normalization_target_variant": "full",
        "host_normalization_identity_verified": True,
        "accelerator_idle_correction_requested": True,
        "accelerator_idle_correction_applied": True,
        "accelerator_idle_correction_statuses": ["applied"],
        "accelerator_idle_w_applied": 2.0,
        "accelerator_idle_calibration_verified": True,
        "accelerator_idle_calibration_status": "verified",
        "accelerator_idle_calibration_binding_sha256": "0" * 64,
        "energy_efficiency_claim_eligible": True,
        "host_normalized_energy_per_work_est_j": 0.12,
        "host_normalized_energy_est_j": 480.0,
        "energy_total_j": 600.0,
        "host_normalized_average_power_est_w": 8.0,
        "work_units": 4000,
    })
    matching["direction"] = "hailo8_to_trt"
    other_direction = dict(matching, direction="deepx_to_trt", energy_per_work_j=9.99)
    pair = [
        row for row in build_native_energy_pairs([split, other_direction, matching])
        if row["baseline_backend"] == "native_full_tensorrt"
    ][0]
    assert pair["comparable"] is True
    assert pair["baseline_energy_per_work_j"] == 0.12
    assert pair["baseline_candidate_count"] == 1

    duplicate = dict(
        matching,
        precision="fp16",
        full_runtime_precision="fp16",
    )
    pair = [
        row for row in build_native_energy_pairs([split, matching, duplicate])
        if row["baseline_backend"] == "native_full_tensorrt"
    ][0]
    assert pair["comparable"] is False
    assert pair["comparison_reason"] == "baseline_ambiguous"
    assert pair["baseline_candidate_count"] == 2
    assert pair["baseline_candidate_comparison_precisions"] == ["fp16", "uint8_cast_fp16"]
    assert pair["baseline_candidate_runtime_precisions"] == ["fp16"]


def _cross_runner_fixture(tmp_path: Path, n: int) -> list[dict[str, object]]:
    reports = tmp_path / "reports"
    validation_dir = reports / "native_validation"
    validation_dir.mkdir(parents=True)
    native = []
    validation = []
    generic = []
    endpoint_hash = "a" * 64
    endpoint = {
        "task": "classification",
        "stage": "logits",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "endpoint_contract_hash": endpoint_hash,
        },
    }
    for index in range(1, n + 1):
        case = f"b{index:03d}"
        native.append({
            "ok": True, "model": "m", "case": case,
            "backend": "hailo8_to_trt", "precision": "p",
            "execution_precision": "p", "cycle_ms": float(index),
            "fps_makespan": 1000.0 / float(index),
            "setup_id": "hailo8-host", "comparison_backend": "hailo8",
            "performance_claim_eligible": True,
            "output_endpoint_match": True,
            "precision_quality_verified": True,
            "quality_evidence_verified": True,
            "repeat_claim_gate_pass": True,
            "comparison_stratum_explicit": True,
            **endpoint,
        })
        validation.append({
            "model": "m", "case": case, "backend": "hailo8_to_trt",
            "precision": "p", "execution_precision": "p",
            "setup_id": "hailo8-host", "comparison_backend": "hailo8",
            "contract_consistent": True, "semantic_ok": True,
            "claim_ok": True, "task_valid": True,
            "accuracy_gate_pass": True, "eligible_for_ranking": True,
            "status": "claim_ok", "gate_status": "eligible",
            **endpoint,
        })
        generic.append({
            "model_id": "m",
            "case_id": case,
            "variant": "split",
            "runner_regime": "generic",
            "direction": "hailo8_to_trt",
            "precision": "p",
            "execution_precision": "p",
            "setup_id": "hailo8-host",
            "comparison_backend": "hailo8",
            "cycle_ms": float(index),
            "contract_consistent": True,
            "task_quality_status": "pass",
            "eligible_for_ranking": True,
            **endpoint,
        })
    (reports / "native_producer_combined_summary.json").write_text(json.dumps({"rows": native}), encoding="utf-8")
    (validation_dir / "native_producer_validation_summary.json").write_text(json.dumps({"rows": validation}), encoding="utf-8")
    return generic


def test_cross_runner_topk_is_na_below_minimum_and_when_k_exceeds_n(tmp_path: Path) -> None:
    one = compute_cross_runner_report(tmp_path / "one", _cross_runner_fixture(tmp_path / "one", 1), minimum_candidates=3)["groups"][0]
    for key in ("spearman_rho", "kendall_tau_b", "pairwise_concordance", "native_best_hit_at_1", "native_regret_at_1", "native_best_hit_at_5", "native_regret_at_5"):
        assert one[key] is None

    three = compute_cross_runner_report(tmp_path / "three", _cross_runner_fixture(tmp_path / "three", 3), minimum_candidates=3)["groups"][0]
    assert three["native_best_hit_at_1"] is True
    assert three["native_best_hit_at_3"] is True
    assert three["native_best_hit_at_5"] is None
    assert three["native_regret_at_5"] is None


def test_scientific_ranking_quality_is_na_for_single_candidate() -> None:
    rows = [{
        "model_id": "m",
        "case_id": "b001",
        "variant": "split",
        "direction": "hailo8_to_tensorrt",
        "runner_regime": "generic",
        "pipeline_cycle_selected_ms": 1.0,
        "ranking_eligible": True,
        "evaluation_role": "holdout",
    }]
    predictions = {"m": {
        "_ranking_method_predictions": [{
            "method_id": "cut_bytes_only",
            "case_id": "b001",
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "prediction_available": True,
            "predicted_value": 1.0,
            "prediction_unit": "bytes",
        }],
        "_prediction_freeze": {"valid": True, "ranking_predictions_valid": True},
    }}
    profile = {"model_suite": {"primary": [{"id": "m", "evaluation_role": "holdout", "candidate_universe_complete": True}]}}
    policy = {
        "k_values": [1, 3, 5],
        "elite_q_values": [1, 3],
        "minimum_candidates_for_correlation": 3,
        "near_optimal_relative_epsilon": 0.01,
        "methods": ["cut_bytes_only"],
        "require_complete_candidate_universe": True,
        "require_frozen_predictions": True,
        "primary_k": 5,
    }
    details, _, _ = _ranking_method_comparison(rows, predictions, profile, policy)
    row = details[0]
    for key in (
        "spearman_rho", "kendall_tau_b", "hit_at_1", "near_optimal_hit_at_1",
        "regret_at_1", "elite_recall_at_1_q1", "hit_at_3", "regret_at_5",
    ):
        assert row[key] is None
    assert row["diagnostic_validity_at_1"] == 1.0
    assert row["validity_at_k_is_diagnostic_only"] is True


def test_analysis_pack_contract_contains_native_energy_evidence() -> None:
    for name in (
        "screening_energy_observations.csv",
        "screening_energy_observations.json",
        "native_energy_observations.csv",
        "native_energy_observations.json",
        "native_energy_pair_comparison.csv",
        "native_energy_pair_comparison.json",
    ):
        assert name in CANONICAL_RESULT_FILES
    for name in (
        "screening_energy_observations.tex",
        "native_energy_observations.tex",
        "native_energy_pair_comparison.tex",
    ):
        assert name in CANONICAL_TABLE_FILES
