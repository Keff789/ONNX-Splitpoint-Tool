from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
    deepx_preprocess_binding,
    seal_native_command_contract,
    seal_split_energy_preflight_attestation,
    split_energy_runtime_argv,
    successful_runtime_argv,
    verify_split_energy_preflight_attestation,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    build_detection_completion_execution_contract,
    tensor_signature,
)
from onnx_splitpoint_tool.resume_artifact_rehydration import (
    ArtifactRequirement,
)
from onnx_splitpoint_tool.resume_hailo8_source_recovery import (
    materialize_hailo8_literal_sources,
)
from scripts import native_deepx_trt_e2e_from_benchmarkset as deepx_runner
from scripts import native_hailo10_trt_e2e_from_benchmarkset as hailo10_runner
from scripts import native_hailo_trt_fifo_from_benchmarkset as hailo8_runner
from scripts.native_split_energy_preflight import build_attestation


ROOT = Path(__file__).resolve().parents[1]


def _decoded_completion_fixture() -> tuple[
    dict[str, Any], dict[str, np.ndarray]
]:
    outputs = {
        "detections": np.asarray(
            [[[8.0, 8.0, 16.0, 16.0, 0.9, 1.0]]],
            dtype=np.float32,
        ),
    }
    signature = tensor_signature(outputs)
    endpoint_hash = "d" * 64
    source = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "output_endpoint_attestation": {
            "schema": (
                "onnx-splitpoint/runtime-output-endpoint-attestation"
            ),
            "schema_version": 3,
            "attested": True,
            "status": "passed",
            "endpoint": "decoded_nms",
            "stage": "decoded_nms",
            "values_decoded_xyxy_score_class": True,
            "declaration_attested": True,
            "endpoint_contract_hash": endpoint_hash,
            "tensor_signature": signature,
            "declared_contract": {
                "model_id": "yolo26s",
                "source_coordinate_space": (
                    "model_input_letterbox_xyxy_pixels"
                ),
            },
        },
    }
    contract = build_detection_completion_execution_contract(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[64, 64],
        preprocess={
            "mode": "letterbox",
            "rgb": True,
            "pad_value": 0,
        },
        source_endpoint_contract=source,
    )
    return contract, outputs


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: bytes = b"test-artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _base_contract(
    tmp_path: Path,
    *,
    backend: str,
    runner: str,
    options: dict[str, Any],
    backend_artifacts: dict[str, Path],
    boundary_layout: str = "as_input",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = _write(tmp_path / "input.png", b"exact-image")
    interpreter = Path(sys.executable)
    runner_path = ROOT / runner
    artifacts = {
        "python_executable": {
            "path": str(interpreter),
            "sha256": _sha(interpreter),
            "size_bytes": interpreter.stat().st_size,
        },
        **{
            name: {
                "path": str(path),
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in backend_artifacts.items()
        },
    }
    payload: dict[str, Any] = {
        "backend": backend,
        "model": "test_model",
        "case": "b001",
        "precision": (
            "uint8_dequant_fp16"
            if backend == "hailo10h_to_trt" else "uint8_cast_fp16"
        ),
        "setup_id": {
            "hailo8_to_trt": "orin_nx_hailo8_01",
            "hailo10h_to_trt": "orin_nx_hailo10_01",
            "deepx_to_trt": "orin_nx_deepx_m1_01",
        }[backend],
        "comparison_backend": {
            "hailo8_to_trt": "hailo8",
            "hailo10h_to_trt": "hailo10h",
            "deepx_to_trt": "deepx",
        }[backend],
        "runner": runner,
        "python_executable": str(interpreter),
        "interpreter_identity": {
            "executable": str(interpreter),
            "resolved_executable": str(interpreter.resolve()),
            "executable_sha256": _sha(interpreter),
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", ""),
            "version": sys.version,
        },
        "runner_sha256": _sha(runner_path),
        "benchmark_set": str(tmp_path / "benchmark_set"),
        "hw_arch": "hailo8" if backend == "hailo8_to_trt" else "hailo10h",
        "input_image": str(image),
        "input_image_source": "exact_file",
        "input_image_sha256": _sha(image),
        "artifacts": artifacts,
        "runtime_options": options,
        "boundary_contract": {
            "boundary_layout_requested": boundary_layout,
            "boundary_layout_effective": boundary_layout,
            "bridge_schema": (
                "onnx-splitpoint/uint8-dequant-bridge"
                if backend == "hailo10h_to_trt" else ""
            ),
            "dequant_scale": (
                0.125 if backend == "hailo10h_to_trt" else None
            ),
            "dequant_zero_point": (
                11.0 if backend == "hailo10h_to_trt" else None
            ),
        },
        "complete": True,
    }
    payload.update(extra or {})
    if (
        backend == "hailo8_to_trt"
        and str(options.get("producer_impl") or "")
        == "hailo8_python_vstreams_fifo"
    ):
        hailo_site = str(
            (tmp_path / "hailo-site-packages").resolve()
        )
        resolved_python = str(interpreter.resolve())
        consumer_source = (
            ROOT
            / "scripts"
            / "native_hailo10_trt_e2e_from_benchmarkset.py"
        )
        options.update({
            "process_local_extra_sites": [hailo_site],
            "mixed_runtime_site_policy": (
                "site.addsitedir_after_system_defaults"
            ),
        })
        payload["interpreter_identity"].update({
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "process_local_extra_sites": [hailo_site],
        })
        payload["artifacts"]["native_executable"] = {
            "path": resolved_python,
            "sha256": _sha(interpreter),
        }
        payload["artifacts"]["native_trt_consumer_source"] = {
            "path": str(consumer_source),
            "sha256": _sha(consumer_source),
        }
        payload["mixed_runtime_contract"] = {
            "status": "ready",
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "site_policy": "site.addsitedir_after_system_defaults",
            "python_executable": str(interpreter),
            "resolved_python_executable": resolved_python,
            "process_local_extra_sites": [hailo_site],
            "modules": {
                "tensorrt": "/usr/lib/tensorrt/__init__.py",
                "hailo_platform": (
                    f"{hailo_site}/hailo_platform/__init__.py"
                ),
                "numpy": "/usr/lib/numpy/__init__.py",
                "PIL": "/usr/lib/PIL/__init__.py",
            },
            "cudart": "libcudart.so",
            "native_trt_consumer_source": str(consumer_source),
            "native_trt_consumer_source_sha256": _sha(
                consumer_source
            ),
            "source_closure_ok": True,
        }

    # Positive preflight fixtures carry the same sealed technical Part-2
    # proof required of production contracts.  Quality/claim roles are not an
    # Energy-preflight authority; only the artifact cross-links and one static
    # Part-2 input are established here.
    technical_paths = {
        "boundary_metadata": _write(
            tmp_path / "part2-proof" / "boundary_metadata.json",
            b'{"fixture":"boundary"}',
        ),
        "source_part2_onnx": _write(
            tmp_path / "part2-proof" / "source_part2.onnx",
            b"fixture-source-part2",
        ),
        "build_part2_onnx": _write(
            tmp_path / "part2-proof" / "build_part2.onnx",
            b"fixture-build-part2",
        ),
        "engine_build_receipt": _write(
            tmp_path / "part2-proof" / "engine_build_receipt.json",
            b'{"fixture":"engine-build"}',
        ),
        "trtexec": _write(
            tmp_path / "part2-proof" / "trtexec",
            b"fixture-trtexec",
        ),
    }
    for name, path in technical_paths.items():
        artifacts[name] = {
            "path": str(path),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    precision = str(payload["precision"])
    metadata = {
        "schema": "onnx-splitpoint/native-trt-meta",
        "schema_version": 1,
        "variant": "part2",
        "build_ok": True,
        "inputs_static": True,
        "precision": precision,
        "requested_precision": precision,
        "onnx": artifacts["build_part2_onnx"]["path"],
        "source_onnx": artifacts["build_part2_onnx"]["path"],
        "engine": artifacts["engine"]["path"],
        "engine_build_receipt_path": artifacts[
            "engine_build_receipt"
        ]["path"],
        "inputs": [{
            "name": "cut",
            "shape": [1, 8, 8, 16],
            "elem_type": (
                "FLOAT" if precision == "float32_layout_fp16" else "UINT8"
            ),
            "has_dynamic": False,
        }],
    }
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    metadata_path = _write(
        tmp_path / "part2-proof" / "native_trt_meta.json",
        metadata_bytes,
    )
    artifacts["native_trt_meta"] = {
        "path": str(metadata_path),
        "sha256": _sha(metadata_path),
        "size_bytes": metadata_path.stat().st_size,
    }

    part1_role = "dxnn" if backend == "deepx_to_trt" else "hef"
    artifact_roles = {
        "part1_runtime": part1_role,
        "boundary_metadata": "boundary_metadata",
        "source_part2_onnx": "source_part2_onnx",
        "build_part2_onnx": "build_part2_onnx",
        "engine": "engine",
        "native_trt_meta": "native_trt_meta",
        "engine_build_receipt": "engine_build_receipt",
        "trtexec": "trtexec",
    }
    proof_artifacts = {
        role: {
            field: artifacts[name][field]
            for field in ("path", "sha256", "size_bytes")
        }
        for role, name in artifact_roles.items()
    }

    def embedded_json(path: Path) -> dict[str, Any]:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
        return {
            "encoding": "base64",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "file_sha256": hashlib.sha256(content).hexdigest(),
            "file_size_bytes": len(content),
            "canonical_value_sha256": canonical_json_sha256(value),
        }

    proof = {
        "schema": (
            "onnx-splitpoint/native-split-local-artifact-verification"
        ),
        "schema_version": 1,
        "verification_kind": "producer_local_file_rehash",
        "artifact_names": list(artifact_roles),
        "artifacts": proof_artifacts,
        "artifact_set_sha256": canonical_json_sha256(proof_artifacts),
        "native_trt_meta_payload_sha256": canonical_json_sha256(metadata),
        "embedded_json_files": {
            "boundary_metadata": embedded_json(
                technical_paths["boundary_metadata"]
            ),
            "native_trt_meta": embedded_json(metadata_path),
            "engine_build_receipt": embedded_json(
                technical_paths["engine_build_receipt"]
            ),
        },
    }
    proof["proof_sha256"] = canonical_json_sha256(proof)
    binding = {
        "schema": "onnx-splitpoint/native-split-quality-binding",
        "schema_version": 1,
        "quality_completed": False,
        "performance_claims_emitted": False,
        "artifacts": proof_artifacts,
        "local_artifact_verification": proof,
        "native_trt_meta": metadata,
        "native_trt_meta_payload": metadata,
        "native_trt_meta_payload_sha256": canonical_json_sha256(metadata),
        "native_trt_meta_file_sha256": artifacts[
            "native_trt_meta"
        ]["sha256"],
        "native_trt_meta_file_size_bytes": artifacts[
            "native_trt_meta"
        ]["size_bytes"],
    }
    binding["binding_sha256"] = canonical_json_sha256(binding)
    payload["native_split_quality_binding"] = binding
    payload["native_split_quality_binding_sha256"] = binding[
        "binding_sha256"
    ]
    payload["boundary_contract"].update({
        "metadata_path": artifacts["native_trt_meta"]["path"],
        "metadata_sha256": artifacts["native_trt_meta"]["sha256"],
    })
    payload["engine"] = artifacts["engine"]["path"]
    payload["engine_sha256"] = artifacts["engine"]["sha256"]
    return seal_native_command_contract(payload)


def _preflight(
    tmp_path: Path, contract: dict[str, Any], *, nonce: str = "fresh-nonce"
) -> Path:
    attestation, rc = build_attestation(
        None,
        contract_payload=contract,
        nonce=nonce,
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256=_sha(ROOT / "scripts" / "native_split_energy_preflight.py"),
        tool_root=ROOT,
        valid_for_s=120.0,
    )
    assert rc == 0, attestation
    path = tmp_path / "preflight_attestation.json"
    path.write_text(json.dumps(attestation), encoding="utf-8")
    return path


def test_hailo8_auto_preprocessing_is_task_bound_and_forwarded() -> None:
    assert hailo8_runner._resolve_preprocess_mode("classification", "auto") == "resize"
    assert hailo8_runner._resolve_preprocess_mode("detection", "auto") == "letterbox"
    assert hailo8_runner._resolve_preprocess_mode("detection", "resize") == "resize"
    with pytest.raises(RuntimeError, match="benchmark task"):
        hailo8_runner._resolve_preprocess_mode("", "auto")

    native_source = (ROOT / "scripts" / "native_hailo_trt_fifo_from_benchmarkset.py").read_text()
    outer_source = (ROOT / "scripts" / "native_fifo_eval_runner.py").read_text()
    matrix_source = (ROOT / "scripts" / "native_fifo_smoke_matrix.py").read_text()
    assert "resize_rgb_uint8" in native_source
    assert "preprocess_rgb_uint8" in native_source
    assert "'--task', task" in outer_source and "'--preprocess-mode'" in outer_source
    assert "'--task', task" in matrix_source and "'--preprocess-mode'" in matrix_source


def test_deepx_task_preprocessing_changes_real_image_geometry(tmp_path: Path) -> None:
    from PIL import Image

    source_rgb = np.asarray([17, 33, 65], dtype=np.uint8)
    source = np.empty((2, 4, 3), dtype=np.uint8)
    source[...] = source_rgb
    image = tmp_path / "rectangular.png"
    Image.fromarray(source, mode="RGB").save(image)

    assert deepx_runner._resolve_preprocess_contract(
        "classification", "auto", 114,
    ) == ("resize", 0)
    assert deepx_runner._resolve_preprocess_contract(
        "detection", "auto", 114,
    ) == ("letterbox", 114)
    with pytest.raises(ValueError, match="out_of_range"):
        deepx_runner._resolve_preprocess_contract("detection", "auto", 256)

    resized = deepx_runner._image_variants_for_shape(
        [8, 8, 3], str(image), preprocess_mode="resize",
        letterbox_pad_value=0,
    )[0]
    assert resized.shape == (8, 8, 3)
    assert resized.dtype == np.uint8 and resized.flags.c_contiguous
    assert np.all(resized == source_rgb)

    letterboxed = deepx_runner._image_variants_for_shape(
        [8, 8, 3], str(image), preprocess_mode="letterbox",
        letterbox_pad_value=114,
    )[0]
    assert letterboxed.shape == (8, 8, 3)
    assert letterboxed.dtype == np.uint8 and letterboxed.flags.c_contiguous
    assert np.all(letterboxed[:2] == 114)
    assert np.all(letterboxed[2:6] == source_rgb)
    assert np.all(letterboxed[6:] == 114)


def test_deepx_part1_preprocess_contract_rejects_legacy_or_wrong_task() -> None:
    expected = {
        "task": "classification",
        "requested_mode": "auto",
        "effective_mode": "resize",
        "requested_pad": 114,
        "effective_pad": 0,
    }
    legacy = {
        "input": {
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "layout": "HWC",
        }
    }
    with pytest.raises(
        RuntimeError, match="deepx_part1_task_preprocess_contract_mismatch"
    ):
        deepx_runner._validate_part1_preprocess_contract(legacy, **expected)

    valid = {
        "input": {
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "layout": "HWC",
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "preprocess_mode": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
        }
    }
    assert deepx_runner._validate_part1_preprocess_contract(
        valid, **expected
    )["letterbox_pad_value_effective"] == 0

    wrong_task = json.loads(json.dumps(valid))
    wrong_task["input"]["task"] = "detection"
    with pytest.raises(
        RuntimeError, match="deepx_part1_task_preprocess_contract_mismatch"
    ):
        deepx_runner._validate_part1_preprocess_contract(wrong_task, **expected)


def test_deepx_detection_contract_survives_preflight_and_both_replays(
    tmp_path: Path,
) -> None:
    completion_contract, _completion_outputs = (
        _decoded_completion_fixture()
    )
    prepared = tmp_path / "detection.npy"
    np.save(prepared, np.zeros((8, 8, 3), dtype=np.uint8), allow_pickle=False)
    options = {
        "frames": 20, "warmup": 2, "queue_depth": 2,
        "dump_outputs": False, "dump_boundary": False, "build": False,
        "prepared_input_bound": True, "task": "detection",
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": 114,
        "letterbox_pad_value": 114,
        "completion_execution_contract": completion_contract,
        "completion_execution_contract_sha256": (
            completion_contract["contract_sha256"]
        ),
    }
    prepared_contract = {
        "format": "numpy_npy_v1", "shape": [8, 8, 3], "dtype": "uint8",
        "c_contiguous": True, "task": "detection",
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": 114,
        "letterbox_pad_value": 114,
    }
    contract = _base_contract(
        tmp_path,
        backend="deepx_to_trt",
        runner="scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        options=options,
        backend_artifacts={
            "dxnn": _write(tmp_path / "model.dxnn"),
            "engine": _write(tmp_path / "part2.engine"),
            "prepared_input": prepared,
        },
        extra={"prepared_input_contract": prepared_contract},
    )
    assert deepx_preprocess_binding(options, prepared_contract)[0] is True

    for argv in (
        successful_runtime_argv(
            contract, duration_s=10, fresh_output_root="/tmp/perf",
            remote_tool_dir="/tool",
        ),
        split_energy_runtime_argv(
            contract, duration_s=10, fresh_output_root="/tmp/energy",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        ),
    ):
        assert argv[argv.index("--task") + 1] == "detection"
        assert argv[argv.index("--preprocess-mode") + 1] == "auto"
        assert argv[argv.index("--letterbox-pad-value") + 1] == "114"

    attestation, rc = build_attestation(
        None,
        contract_payload=contract,
        nonce="deepx-detection",
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256=_sha(ROOT / "scripts" / "native_split_energy_preflight.py"),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rc == 0, attestation
    bound_options = attestation["workload_binding"]["runtime_options"]
    assert bound_options["preprocess_mode_requested"] == "auto"
    assert bound_options["preprocess_mode_effective"] == "letterbox"
    assert bound_options["letterbox_pad_value_requested"] == 114
    assert bound_options["letterbox_pad_value_effective"] == 114
    assert (
        bound_options["completion_execution_contract"]
        == completion_contract
    )
    assert (
        bound_options["completion_execution_contract_sha256"]
        == completion_contract["contract_sha256"]
    )

    tampered = json.loads(json.dumps(contract))
    tampered.pop("contract_sha256", None)
    tampered["prepared_input_contract"]["letterbox_pad_value_effective"] = 0
    tampered["prepared_input_contract"]["letterbox_pad_value"] = 0
    tampered = seal_native_command_contract(tampered)
    rejected, rejected_rc = build_attestation(
        None,
        contract_payload=tampered,
        nonce="deepx-detection",
        expected_contract_sha256=tampered["contract_sha256"],
        expected_preflight_script_sha256=_sha(ROOT / "scripts" / "native_split_energy_preflight.py"),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rejected_rc != 0
    assert (
        rejected["workload_binding"]["unsupported_reason"]
        == "deepx_split_energy_task_preprocess_binding_missing_or_inconsistent"
    )

    completion_tamper = json.loads(json.dumps(contract))
    completion_tamper.pop("contract_sha256", None)
    completion_tamper["runtime_options"][
        "completion_execution_contract"
    ]["processor_contract"]["input_hw"][0] += 1
    completion_tamper = seal_native_command_contract(completion_tamper)
    rejected, rejected_rc = build_attestation(
        None,
        contract_payload=completion_tamper,
        nonce="deepx-detection",
        expected_contract_sha256=completion_tamper["contract_sha256"],
        expected_preflight_script_sha256=_sha(
            ROOT / "scripts" / "native_split_energy_preflight.py"
        ),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rejected_rc != 0
    assert rejected["ok"] is False
    assert rejected["workload_binding"]["unsupported_reason"].startswith(
        "split_energy_detection_completion_contract_invalid:"
    )


def test_inline_preflight_hashes_interpreter_runner_input_and_all_artifacts(
    tmp_path: Path,
) -> None:
    executable = _write(tmp_path / "native_fifo", b"native")
    contract = _base_contract(
        tmp_path,
        backend="hailo8_to_trt",
        runner="scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        options={
            "frames": 100, "warmup": 10, "queue_depth": 2,
            "hailo_format": "uint8", "letterbox_pad_value": 0,
            "copy_outputs": True, "dump_outputs": True,
            "dump_boundary": True, "build": True, "device_id": "",
            "energy_prepared_feed_capable": True,
            "prepared_input_bound": True,
            "task": "classification", "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 0,
            "letterbox_pad_value_effective": 0,
        },
        backend_artifacts={
            "hef": _write(tmp_path / "part1.hef"),
            "engine": _write(tmp_path / "part2.engine"),
            "native_executable": executable,
            "generated_cpp": _write(tmp_path / "main.cpp"),
            "cmake": _write(tmp_path / "CMakeLists.txt"),
            "prepared_input": _write(tmp_path / "prepared.rgb", b"\x00" * 12),
        },
        extra={"prepared_input_contract": {"format": "raw_rgb_uint8", "shape": [2, 2, 3], "dtype": "uint8", "layout": "HWC", "task": "classification", "preprocess": "opencv_resize_rgb_uint8", "preprocess_mode_requested": "auto", "preprocess_mode_effective": "resize", "letterbox_pad_value_requested": 0, "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0, "pad_value_effective": 0}},
    )
    attestation, rc = build_attestation(
        None,
        contract_payload=contract,
        nonce="n-1",
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256=_sha(ROOT / "scripts" / "native_split_energy_preflight.py"),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rc == 0
    assert attestation["ok"] is True
    assert attestation["verified_part2_input_count"] == 1
    assert attestation["semantic_payload_verification_status"] == (
        "downstream_annotation_not_preflighted"
    )
    assert attestation["command_contract_source"] == "inline_verified_contract_object"
    assert len(attestation["command_contract_payload_sha256"]) == 64
    labels = {row["label"] for row in attestation["artifact_verification"]}
    assert {
        "runner", "input_image", "artifact:python_executable",
        "artifact:hef", "artifact:engine", "artifact:native_executable",
        "artifact:generated_cpp", "artifact:cmake",
    } <= labels
    binding = attestation["workload_binding"]
    assert binding["runtime_options"]["warmup"] == 0
    assert binding["runtime_options"]["dump_outputs"] is False
    assert binding["runtime_options"]["dump_boundary"] is False
    assert binding["runtime_options"]["build"] is False
    assert binding["energy_overrides"]["source_warmup"] == 10

    rejected, rejected_rc = build_attestation(
        None,
        contract_payload=contract,
        nonce="n-1",
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256="0" * 64,
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rejected_rc != 0
    assert "preflight_script_sha256_mismatch" in rejected["failure_reason"]


def test_inline_contract_tamper_and_attestation_nonce_stale_or_seal_tamper_fail(
    tmp_path: Path,
) -> None:
    completion_contract, _completion_outputs = (
        _decoded_completion_fixture()
    )
    contract = _base_contract(
        tmp_path,
        backend="hailo10h_to_trt",
        runner="scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        options={
            "frames": 20, "warmup": 5, "queue_depth": 2, "inflight": 4,
            "producer_impl": "auto", "quantized_inputs": True,
            "quantized_outputs": True, "copy_outputs": True,
            "dump_outputs": False, "dump_boundary": False, "build": False,
            "part1_onnx_used": False,
            "canonical_input_slot_names": ["images"],
            "canonical_output_slot_names": ["boundary"],
            "prepared_input_bound": True,
            "task": "detection", "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "letterbox",
                "letterbox_pad_value_requested": 114,
                "letterbox_pad_value_effective": 114, "letterbox_pad_value": 114,
                "completion_execution_contract": completion_contract,
                "completion_execution_contract_sha256": (
                    completion_contract["contract_sha256"]
                ),
        },
        backend_artifacts={
            "hef": _write(tmp_path / "part1.hef"),
            "engine": _write(tmp_path / "part2.engine"),
            "prepared_input_00": _write(tmp_path / "prepared.npy"),
        },
        extra={"prepared_input_contract": {"format": "numpy_npy_v1", "preprocess": "exact_performance_prepared_tensor_persisted", "normalization": "hef_quant_info_from_unit_float32", "entries": [{"name": "images", "artifact_name": "prepared_input_00", "shape": [1], "dtype": "uint8", "c_contiguous": True}], "slot_order": ["images"], "task": "detection", "preprocess_mode_requested": "auto", "preprocess_mode": "letterbox", "preprocess_mode_effective": "letterbox", "letterbox_pad_value_requested": 114, "letterbox_pad_value_effective": 114, "letterbox_pad_value": 114, "pad_value_effective": 114}},
    )
    tampered_contract = json.loads(json.dumps(contract))
    tampered_contract["runtime_options"]["queue_depth"] = 99
    failed, rc = build_attestation(
        None,
        contract_payload=tampered_contract,
        nonce="n-2",
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256=_sha(ROOT / "scripts" / "native_split_energy_preflight.py"),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rc != 0
    assert failed["ok"] is False
    assert "native_command_contract_sha256_mismatch" in failed["failure_reason"]

    valid, rc = build_attestation(
        None,
        contract_payload=contract,
        nonce="n-2",
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256=_sha(ROOT / "scripts" / "native_split_energy_preflight.py"),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rc == 0
    verified, reason = verify_split_energy_preflight_attestation(
        valid,
        expected_nonce="wrong",
        expected_command_contract_sha256=contract["contract_sha256"],
    )
    assert verified is None and reason == "split_energy_preflight_nonce_mismatch"

    stale = dict(valid)
    stale["created_at_unix_ns"] = time.time_ns() - 120_000_000_000
    stale["expires_at_unix_ns"] = time.time_ns() - 60_000_000_000
    stale = seal_split_energy_preflight_attestation(stale)
    verified, reason = verify_split_energy_preflight_attestation(
        stale,
        expected_nonce="n-2",
        expected_command_contract_sha256=contract["contract_sha256"],
    )
    assert verified is None and reason == "split_energy_preflight_expired"

    seal_tamper = json.loads(json.dumps(valid))
    seal_tamper["workload_binding"]["runtime_options"]["queue_depth"] = 9
    verified, reason = verify_split_energy_preflight_attestation(
        seal_tamper,
        expected_nonce="n-2",
        expected_command_contract_sha256=contract["contract_sha256"],
    )
    assert verified is None and reason == "split_energy_preflight_attestation_sha256_mismatch"


def test_hailo8_workload_only_never_hashes_builds_warms_or_dumps_and_reports_exact_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    executable = tmp_path / "fake_native_fifo.py"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "a=sys.argv[1:]; get=lambda k:a[a.index(k)+1]\n"
        "out=pathlib.Path(get('--out')); out.write_text(json.dumps({'completed_frames':7}))\n"
        "(out.parent/'native_argv.json').write_text(json.dumps(a))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    options = {
        "frames": 100, "warmup": 10, "queue_depth": 2,
        "hailo_format": "uint8", "letterbox_pad_value": 0,
        "copy_outputs": True, "dump_outputs": True,
        "dump_boundary": True, "build": True, "device_id": "",
        "energy_prepared_feed_capable": True,
        "prepared_input_bound": True,
        "task": "classification", "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "resize",
        "letterbox_pad_value_requested": 0,
        "letterbox_pad_value_effective": 0,
    }
    contract = _base_contract(
        tmp_path,
        backend="hailo8_to_trt",
        runner="scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        options=options,
        backend_artifacts={
            "hef": _write(tmp_path / "part1.hef"),
            "engine": _write(tmp_path / "part2.engine"),
            "native_executable": executable,
            "generated_cpp": _write(tmp_path / "main.cpp"),
            "cmake": _write(tmp_path / "CMakeLists.txt"),
            "prepared_input": _write(tmp_path / "prepared.rgb", b"\x00" * 12),
        },
        extra={"prepared_input_contract": {"format": "raw_rgb_uint8", "shape": [2, 2, 3], "dtype": "uint8", "layout": "HWC", "task": "classification", "preprocess": "opencv_resize_rgb_uint8", "preprocess_mode_requested": "auto", "preprocess_mode_effective": "resize", "letterbox_pad_value_requested": 0, "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0, "pad_value_effective": 0}},
    )
    attestation = _preflight(tmp_path, contract)
    out = tmp_path / "energy" / "native_fifo_results.json"
    args = argparse.Namespace(
        energy_preflight_attestation=str(attestation),
        energy_preflight_nonce="fresh-nonce",
        source_contract_sha256=contract["contract_sha256"],
        energy_preflight_max_age_s=300.0,
        warmup=0, build=False, dump_outputs=False, dump_boundary=False,
        duration_s=1.0, benchmark_set=contract["benchmark_set"], case="b001",
        precision=contract["precision"], image=contract["input_image"],
        hw_arch="hailo8", queue_depth=2, hailo_format="uint8",
        task="classification", preprocess_mode="auto",
        copy_outputs=True, letterbox_pad_value=0, frames=1,
        result_json=str(out),
    )
    monkeypatch.setattr(
        hailo8_runner, "_sha256_file",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("in-window hash")),
    )
    assert hailo8_runner._energy_workload_only(args) == 0
    argv = json.loads((out.parent / "native_argv.json").read_text())
    assert argv[argv.index("--warmup") + 1] == "0"
    assert argv[argv.index("--dump-outputs") + 1] == "0"
    assert argv[argv.index("--dump-boundary") + 1] == "0"
    assert argv[argv.index("--reuse-preprocessed-input") + 1] == "1"
    assert argv[argv.index("--prepared-input-rgb") + 1].endswith("prepared.rgb")
    assert argv[argv.index("--task") + 1] == "classification"
    assert argv[argv.index("--preprocess-mode") + 1] == "resize"
    assert "__SPLITPOINT_WORK_UNITS__=7" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["completed_frames"] == 7
    assert result["prepared_feed_contract"] == "exact_prepared_feed_loaded_outside_counted_loop"


def test_detection_performance_contract_mutation_is_rejected_before_energy_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    completion_contract, _completion_outputs = (
        _decoded_completion_fixture()
    )
    prepared = _write(
        tmp_path / "prepared.rgb",
        b"\x00" * (64 * 64 * 3),
    )
    recovered = materialize_hailo8_literal_sources(
        [
            ArtifactRequirement(
                role="cmake",
                remote_path=str(tmp_path / "remote" / "CMakeLists.txt"),
                sha256=(
                    "a6734f5ccd8f08d2fabd2bdef17bbaa2"
                    "a567e16a5e8e264ef0d0c128c582cae3"
                ),
                size_bytes=1_665,
            ),
            ArtifactRequirement(
                role="generated_cpp",
                remote_path=str(tmp_path / "remote" / "main.cpp"),
                sha256=(
                    "99dfa223ac4aa4ec2500ad4b5e032396"
                    "e26abec197b7dd20e647eff314aa22ae"
                ),
                size_bytes=41_485,
            ),
        ],
        tool_root=ROOT,
        destination=tmp_path / "derived-sources",
    )
    recovered_sources = {
        row["role"]: Path(row["source_path"])
        for row in recovered["entries"]
    }
    assert recovered["code_executed"] is False
    assert recovered["build_started"] is False
    assert recovered["remote_mutation_performed"] is False
    options = {
        "frames": 20,
        "warmup": 2,
        "queue_depth": 2,
        "hailo_format": "uint8",
        "copy_outputs": True,
        "dump_outputs": False,
        "dump_boundary": False,
        "build": False,
        "device_id": "",
        "energy_prepared_feed_capable": True,
        "prepared_input_bound": True,
        "producer_impl": "hailo8_python_vstreams_fifo",
        "task": "detection",
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 0,
        "letterbox_pad_value_effective": 0,
        "letterbox_pad_value": 0,
        "completion_execution_contract": completion_contract,
        "completion_execution_contract_sha256": (
            completion_contract["contract_sha256"]
        ),
    }
    contract = _base_contract(
        tmp_path,
        backend="hailo8_to_trt",
        runner="scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        options=options,
        backend_artifacts={
            "hef": _write(tmp_path / "part1.hef"),
            "engine": _write(tmp_path / "part2.engine"),
            "native_executable": _write(tmp_path / "native_fifo"),
            "generated_cpp": recovered_sources["generated_cpp"],
            "cmake": recovered_sources["cmake"],
            "prepared_input": prepared,
        },
        extra={
            "prepared_input_contract": {
                "format": "raw_rgb_uint8",
                "shape": [64, 64, 3],
                "dtype": "uint8",
                "layout": "HWC",
                "task": "detection",
                "preprocess": "letterbox_rgb_uint8",
                "preprocess_mode_requested": "auto",
                "preprocess_mode_effective": "letterbox",
                "letterbox_pad_value_requested": 0,
                "letterbox_pad_value_effective": 0,
                "letterbox_pad_value": 0,
                "pad_value_effective": 0,
            },
            "runtime_boundary_evidence": {
                "status": "exact_runtime_boundary_verified",
                "output_count": 1,
                "output_name": "attested_boundary",
                "output_shape": [1, 4],
                "output_dtype": "float32",
            },
        },
    )
    attestation, rc = build_attestation(
        None,
        contract_payload=contract,
        nonce="completion-binding",
        expected_contract_sha256=contract["contract_sha256"],
        expected_preflight_script_sha256=_sha(
            ROOT / "scripts" / "native_split_energy_preflight.py"
        ),
        tool_root=ROOT,
        valid_for_s=60,
    )
    assert rc == 0, attestation
    verified = {
        row["label"]: row
        for row in attestation["artifact_verification"]
    }
    assert verified["artifact:cmake"]["actual_sha256"] == (
        "a6734f5ccd8f08d2fabd2bdef17bbaa2a567e16a5e8e264ef0d0c128c582cae3"
    )
    assert verified["artifact:generated_cpp"]["actual_sha256"] == (
        "99dfa223ac4aa4ec2500ad4b5e032396e26abec197b7dd20e647eff314aa22ae"
    )
    binding = attestation["workload_binding"]
    assert (
        binding["runtime_options"]["completion_execution_contract"]
        == completion_contract
    )
    assert (
        binding["runtime_options"][
            "completion_execution_contract_sha256"
        ]
        == completion_contract["contract_sha256"]
    )

    tampered = json.loads(json.dumps(attestation))
    tampered["workload_binding"]["runtime_options"][
        "completion_execution_contract"
    ]["processor_contract"]["input_hw"][0] += 1
    tampered = seal_split_energy_preflight_attestation(tampered)
    attestation_path = tmp_path / "tampered_preflight.json"
    attestation_path.write_text(
        json.dumps(tampered),
        encoding="utf-8",
    )
    runtime_opened = False

    def _unexpected_runtime_open(**_kwargs: Any) -> Any:
        nonlocal runtime_opened
        runtime_opened = True
        raise AssertionError("energy runtime opened after contract mutation")

    monkeypatch.setattr(
        hailo8_runner,
        "_open_hailo8_python_runtime",
        _unexpected_runtime_open,
    )
    args = argparse.Namespace(
        energy_preflight_attestation=str(attestation_path),
        energy_preflight_nonce="completion-binding",
        source_contract_sha256=contract["contract_sha256"],
        energy_preflight_max_age_s=300.0,
        warmup=0,
        build=False,
        dump_outputs=False,
        dump_boundary=False,
        duration_s=1.0,
        benchmark_set=contract["benchmark_set"],
        case=contract["case"],
        precision=contract["precision"],
        image=contract["input_image"],
        hw_arch="hailo8",
        queue_depth=2,
        hailo_format="uint8",
        task="detection",
        preprocess_mode="auto",
        copy_outputs=True,
        letterbox_pad_value=0,
        result_json=str(tmp_path / "energy_result.json"),
        frames=1,
        device_id="",
    )
    assert hailo8_runner._energy_workload_only(args) == 7
    assert runtime_opened is False
    assert (
        "split_energy_detection_completion_contract_invalid"
        in capsys.readouterr().err
    )


def test_hailo10_workload_only_uses_attested_paths_and_has_no_hash_or_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    completion_contract, completion_outputs = (
        _decoded_completion_fixture()
    )
    options = {
        "frames": 100, "warmup": 5, "queue_depth": 2, "inflight": 4,
        "producer_impl": "auto", "quantized_inputs": True,
        "quantized_outputs": True, "copy_outputs": True,
        "dump_outputs": True, "dump_boundary": True, "build": False,
        "part1_onnx_used": False,
        "canonical_input_slot_names": ["images"],
        "canonical_output_slot_names": ["boundary"],
        "prepared_input_bound": True,
        "task": "detection", "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 0,
        "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0,
        "completion_execution_contract": completion_contract,
        "completion_execution_contract_sha256": (
            completion_contract["contract_sha256"]
        ),
    }
    h10_prepared = tmp_path / "h10_prepared.npy"
    np.save(
        h10_prepared,
        np.zeros((1, 64, 64, 3), dtype=np.uint8),
        allow_pickle=False,
    )
    contract = _base_contract(
        tmp_path,
        backend="hailo10h_to_trt",
        runner="scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        options=options,
        backend_artifacts={
            "hef": _write(tmp_path / "part1.hef"),
            "engine": _write(tmp_path / "part2.engine"),
            "prepared_input_00": h10_prepared,
        },
            extra={"prepared_input_contract": {"format": "numpy_npy_v1", "preprocess": "exact_performance_prepared_tensor_persisted", "normalization": "hef_quant_info_from_unit_float32", "entries": [{"name": "images", "artifact_name": "prepared_input_00", "shape": [1, 64, 64, 3], "dtype": "uint8", "c_contiguous": True}], "slot_order": ["images"], "task": "detection", "preprocess_mode_requested": "auto", "preprocess_mode": "letterbox", "preprocess_mode_effective": "letterbox", "letterbox_pad_value_requested": 0, "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0, "pad_value_effective": 0}},
    )
    Image.new("RGB", (64, 64)).save(contract["input_image"])
    (tmp_path / "native_trt_meta.json").write_text(
        '{"uint8_cast_bridge":{"schema":"onnx-splitpoint/uint8-dequant-bridge",'
        '"scale":0.125,"zero_point":11,"boundary_layout":{"effective":"as_input"}}}',
        encoding="utf-8",
    )
    contract.pop("contract_sha256")
    contract["input_image_sha256"] = _sha(
        Path(contract["input_image"])
    )
    contract = seal_native_command_contract(contract)
    attestation = _preflight(tmp_path, contract)

    class FakeBackend:
        def __init__(self, **_kwargs: Any): pass
        def prepare(self, *_args: Any, **_kwargs: Any) -> Any:
            session = SimpleNamespace(describe_io=lambda: {
                "runtime_output_format": "uint8",
                "runtime_output_formats": {"boundary": "UINT8"},
                "runtime_output_quantization": {
                    "boundary": {"scale": 0.125, "zero_point": 11.0},
                },
            })
            return SimpleNamespace(
                input_names=["images"], output_names=["boundary"],
                handle=SimpleNamespace(session=session),
            )
        def cleanup(self, _prepared: Any) -> None: pass

    class FakeTRT:
        inputs=["input"]; outputs=["output"]
        shapes={"input": (1,)}; dtypes={"input": np.dtype(np.float32)}
        def __init__(self, _path: Path): pass
        def close(self) -> None: pass

    monkeypatch.setattr(hailo10_runner, "HailoBackend", FakeBackend)
    monkeypatch.setattr(hailo10_runner, "NativeTRT", FakeTRT)
    monkeypatch.setattr(
        hailo10_runner, "_make_input",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("in-window preprocessing")),
    )
    def fake_fifo(*_args: Any, **kw: Any) -> dict[str, Any]:
        for _ in range(3):
            kw["completion_runtime"].process(completion_outputs)
        result = {
            "frames": 3,
            "completed_frames": 3,
            "completed_work_units": 3,
            "produced_frames": 3,
            "consumed_frames": 3,
            "warmup": kw["warmup"],
            "fps_makespan": 1.0,
            "paper_equivalent_fps": 1.0,
        }
        result.update(hailo10_runner._completion_attestation_fields(
            kw["completion_runtime"],
            completed_work_units=3,
        ))
        return result

    monkeypatch.setattr(
        hailo10_runner, "_hailo10_async_fifo_run", fake_fifo,
    )
    monkeypatch.setattr(
        hailo10_runner, "_file_sha256",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("in-window hash")),
    )
    monkeypatch.setattr(
        hailo10_runner, "_find_part1_onnx",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("in-window discovery")),
    )
    out = tmp_path / "h10_energy"
    monkeypatch.setattr(sys, "argv", [
        "runner", "--benchmark-set", contract["benchmark_set"], "--case", "b001",
        "--precision", contract["precision"], "--image", contract["input_image"],
        "--frames", "1", "--duration-s", "1", "--warmup", "0",
        "--queue-depth", "2", "--inflight", "4", "--producer-impl", "auto",
            "--quantized-inputs", "--quantized-outputs", "--copy-outputs",
            "--task", "detection", "--preprocess-mode", "auto", "--letterbox-pad-value", "0",
            "--model-id", "yolo26s",
        "--boundary-layout", "as_input", "--out-dir", str(out),
        "--source-contract-sha256", contract["contract_sha256"],
        "--energy-workload-only", "--energy-preflight-attestation", str(attestation),
        "--energy-preflight-nonce", "fresh-nonce",
    ])
    assert hailo10_runner.main() == 0
    report = json.loads((out / "hailo10_native_fifo_e2e_results.json").read_text())
    assert report["completed_frames"] == 3
    assert report["warmup"] == 0
    assert "native_command_contract" not in report
    assert "__SPLITPOINT_WORK_UNITS__=3" in capsys.readouterr().out


def test_deepx_new_contract_uses_bound_prepared_input_without_probe_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared = tmp_path / "selected.npy"
    np.save(prepared, np.ascontiguousarray(np.zeros((8, 8, 3), dtype=np.uint8)), allow_pickle=False)
    options = {
        "frames": 100, "warmup": 5, "queue_depth": 2,
        "dump_outputs": True, "dump_boundary": True, "build": False,
        "prepared_input_bound": True,
        "task": "classification", "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "resize",
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0,
    }
    prepared_contract = {
        "format": "numpy_npy_v1", "shape": [8, 8, 3], "dtype": "uint8",
        "c_contiguous": True, "candidate_index": 0,
        "task": "classification", "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "resize",
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0,
        "preprocess": "exact_performance_selected_tensor_persisted_after_candidate_validation",
    }
    contract = _base_contract(
        tmp_path,
        backend="deepx_to_trt",
        runner="scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        options=options,
        backend_artifacts={
            "dxnn": _write(tmp_path / "model.dxnn"),
            "engine": _write(tmp_path / "part2.engine"),
            "prepared_input": prepared,
        },
        extra={"prepared_input_contract": prepared_contract},
    )
    plan_argv = split_energy_runtime_argv(
        contract, duration_s=10, fresh_output_root="/tmp/deepx",
        remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
    )
    assert plan_argv[plan_argv.index("--task") + 1] == "classification"
    assert plan_argv[plan_argv.index("--preprocess-mode") + 1] == "auto"
    assert plan_argv[plan_argv.index("--letterbox-pad-value") + 1] == "114"
    attestation = _preflight(tmp_path, contract)

    class FakeTRT:
        inputs=["input"]; outputs=["output"]
        shapes={"input": (1,)}; dtypes={"input": np.dtype(np.float32)}
        def __init__(self, _path: Path): pass
        def close(self) -> None: pass

    fake_dx = types.ModuleType("dx_engine")
    fake_dx.InferenceEngine = lambda _path: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dx_engine", fake_dx)
    monkeypatch.setattr(deepx_runner, "NativeTRT", FakeTRT)
    monkeypatch.setattr(
        deepx_runner, "_deepx_fifo_run",
        lambda *_a, **kw: {
            "frames": 4, "completed_frames": 4, "warmup": kw["warmup"],
            "fps_makespan": 1.0, "paper_equivalent_fps": 1.0,
        },
    )
    monkeypatch.setattr(
        deepx_runner, "_select_input_candidate",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("uncounted probe")),
    )
    monkeypatch.setattr(
        deepx_runner, "_file_sha256",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("in-window hash")),
    )
    out = tmp_path / "deepx_energy"
    monkeypatch.setattr(sys, "argv", [
        "runner", "--benchmark-set", contract["benchmark_set"], "--case", "b001",
        "--precision", contract["precision"], "--image", contract["input_image"],
        "--frames", "1", "--duration-s", "1", "--warmup", "0",
        "--queue-depth", "2", "--boundary-layout", "as_input", "--out-dir", str(out),
        "--task", "classification", "--preprocess-mode", "auto", "--letterbox-pad-value", "114",
        "--source-contract-sha256", contract["contract_sha256"],
        "--energy-workload-only", "--energy-preflight-attestation", str(attestation),
        "--energy-preflight-nonce", "fresh-nonce",
    ])
    assert deepx_runner.main() == 0
    report = json.loads((out / "deepx_native_fifo_e2e_results.json").read_text())
    assert report["completed_frames"] == 4
    assert report["warmup"] == 0
    assert report["input_selection"]["selection_method"] == "preflight_attested_prepared_input_no_probe"
    assert "native_command_contract" not in report
    assert "__SPLITPOINT_WORK_UNITS__=4" in capsys.readouterr().out


def test_plan_argv_forces_warmup_zero_and_rejects_legacy_h10_or_deepx_contracts(
    tmp_path: Path,
) -> None:
    h8 = _base_contract(
        tmp_path / "h8",
        backend="hailo8_to_trt",
        runner="scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        options={
            "frames": 10, "warmup": 9, "queue_depth": 2,
            "hailo_format": "uint8", "copy_outputs": True,
            "dump_outputs": False, "dump_boundary": False, "build": False,
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0, "letterbox_pad_value": 0,
            "energy_prepared_feed_capable": True,
            "prepared_input_bound": True, "task": "classification",
            "preprocess_mode_requested": "auto", "preprocess_mode_effective": "resize",
        },
        backend_artifacts={
            "hef": _write(tmp_path / "h8" / "a.hef"),
            "engine": _write(tmp_path / "h8" / "a.engine"),
            "native_executable": _write(tmp_path / "h8" / "fifo"),
            "generated_cpp": _write(tmp_path / "h8" / "main.cpp"),
            "cmake": _write(tmp_path / "h8" / "CMakeLists.txt"),
            "prepared_input": _write(tmp_path / "h8" / "prepared.rgb"),
        },
        extra={"prepared_input_contract": {
            "format": "raw_rgb_uint8", "shape": [2, 2, 3], "dtype": "uint8",
            "layout": "HWC", "task": "classification",
            "preprocess_mode_requested": "auto", "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0, "pad_value_effective": 0,
        }},
    )
    h8_argv = split_energy_runtime_argv(
        h8, duration_s=60, fresh_output_root="/tmp/out",
        remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
    )
    assert h8_argv[h8_argv.index("--task") + 1] == "classification"
    assert h8_argv[h8_argv.index("--preprocess-mode") + 1] == "auto"
    legacy_h8 = json.loads(json.dumps(h8))
    legacy_h8["runtime_options"].pop("preprocess_mode_effective")
    with pytest.raises(ValueError, match="task preprocessing semantics"):
        split_energy_runtime_argv(
            legacy_h8, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )
    tampered_h8_pad = json.loads(json.dumps(h8))
    tampered_h8_pad["prepared_input_contract"]["pad_value_effective"] = 114
    with pytest.raises(ValueError, match="task preprocessing semantics"):
        split_energy_runtime_argv(
            tampered_h8_pad, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )

    h10 = _base_contract(
        tmp_path / "h10",
        backend="hailo10h_to_trt",
        runner="scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        options={
            "frames": 10, "warmup": 9, "queue_depth": 2, "inflight": 4,
            "producer_impl": "auto", "quantized_inputs": True,
            "quantized_outputs": True, "copy_outputs": True,
            "dump_outputs": True, "dump_boundary": True, "build": False,
            "part1_onnx_used": False,
            "canonical_input_slot_names": ["images"],
            "canonical_output_slot_names": ["boundary"],
            "prepared_input_bound": True,
            "task": "detection", "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "letterbox",
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": 114, "letterbox_pad_value": 114,
        },
        backend_artifacts={
            "hef": _write(tmp_path / "h10" / "a.hef"),
            "engine": _write(tmp_path / "h10" / "a.engine"),
            "prepared_input_00": _write(tmp_path / "h10" / "prepared.npy"),
        },
        extra={"prepared_input_contract": {"format": "numpy_npy_v1", "preprocess": "exact_performance_prepared_tensor_persisted", "normalization": "hef_quant_info_from_unit_float32", "entries": [{"name": "images", "artifact_name": "prepared_input_00", "shape": [1], "dtype": "uint8", "c_contiguous": True}], "slot_order": ["images"], "task": "detection", "preprocess_mode_requested": "auto", "preprocess_mode": "letterbox", "preprocess_mode_effective": "letterbox", "letterbox_pad_value_requested": 114, "letterbox_pad_value_effective": 114, "letterbox_pad_value": 114, "pad_value_effective": 114}},
    )
    argv = split_energy_runtime_argv(
        h10, duration_s=60, fresh_output_root="/tmp/out",
        remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
    )
    assert argv[argv.index("--warmup") + 1] == "0"
    assert "--quantized-inputs" in argv
    assert "--quantized-outputs" in argv
    assert "--dump-outputs" not in argv and "--dump-boundary" not in argv
    assert "--energy-workload-only" in argv

    tampered_h10_pad = json.loads(json.dumps(h10))
    tampered_h10_pad["prepared_input_contract"]["letterbox_pad_value_effective"] = 0
    with pytest.raises(ValueError, match="task preprocessing semantics"):
        split_energy_runtime_argv(
            tampered_h10_pad, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )
    tampered_h10_slots = json.loads(json.dumps(h10))
    tampered_h10_slots["prepared_input_contract"]["slot_order"] = ["wrong_slot"]
    with pytest.raises(ValueError, match="task preprocessing semantics"):
        split_energy_runtime_argv(
            tampered_h10_slots, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )
    tampered_h10_entry = json.loads(json.dumps(h10))
    tampered_h10_entry["prepared_input_contract"]["entries"].append("invalid")
    with pytest.raises(ValueError, match="task preprocessing semantics"):
        split_energy_runtime_argv(
            tampered_h10_entry, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )

    legacy_h10 = json.loads(json.dumps(h10))
    legacy_h10["runtime_options"].pop("part1_onnx_used")
    with pytest.raises(ValueError, match="legacy contract"):
        split_energy_runtime_argv(
            legacy_h10, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )

    legacy_deepx = _base_contract(
        tmp_path / "deepx",
        backend="deepx_to_trt",
        runner="scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        options={"frames": 10, "warmup": 2, "queue_depth": 2,
                 "dump_outputs": False, "dump_boundary": False, "build": False,
                 "prepared_input_bound": True},
        backend_artifacts={
            "dxnn": _write(tmp_path / "deepx" / "a.dxnn"),
            "engine": _write(tmp_path / "deepx" / "a.engine"),
            "prepared_input": _write(tmp_path / "deepx" / "prepared.npy"),
        },
        extra={"prepared_input_contract": {"shape": [1], "dtype": "uint8"}},
    )
    with pytest.raises(ValueError, match="task preprocessing semantics"):
        split_energy_runtime_argv(
            legacy_deepx, duration_s=1, fresh_output_root="/tmp/out",
            remote_tool_dir="/tool", preflight_attestation_path="/tmp/a.json",
        )


def test_split_energy_source_and_package_mirrors_match() -> None:
    for name in (
        "native_split_energy_preflight.py",
        "native_hailo_trt_fifo_from_benchmarkset.py",
        "native_hailo10_trt_e2e_from_benchmarkset.py",
        "native_deepx_trt_e2e_from_benchmarkset.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
        ).read_bytes()
