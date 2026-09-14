from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.runners import native_split_quality_runtime as runtime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8",
    )


def _write_source_part2(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["cut"], ["output0"])],
        "native-split-runtime-test",
        [helper.make_tensor_value_info(
            "cut", TensorProto.FLOAT, [1, 2, 2, 2],
        )],
        [helper.make_tensor_value_info(
            "output0", TensorProto.FLOAT, [1, 2, 2, 2],
        )],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(graph), str(path))


def test_prepare_native_split_quality_binding_hailo_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    case_dir = benchmark_set / "b038"
    source_part1 = case_dir / "hailo/hailo8/part1/compiled.hef"
    source_part1.parent.mkdir(parents=True, exist_ok=True)
    source_part1.write_bytes(b"real-temporary-hef-artifact-v1")
    source_part2 = case_dir / "yolo26s_part2_native.onnx"
    _write_source_part2(source_part2)

    metadata_calls: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []

    def fake_hailo_metadata(
        *, part1: Path, part2_input: dict[str, Any], policy: dict[str, Any],
    ) -> dict[str, Any]:
        metadata_calls.append((part1, dict(part2_input), dict(policy)))
        assert part1 == source_part1.resolve()
        assert part2_input == {
            "name": "cut", "shape": [1, 2, 2, 2], "dtype": "float32",
        }
        assert policy["precision"] == "uint8_dequant_fp16"
        return {
            "name": "cut",
            "runtime_name": "cut/hailort",
            "shape": [2, 2, 2],
            "canonical_part2_shape": [1, 2, 2, 2],
            "dtype": "uint8",
            "quantization": {
                "source": "hailort_hef_output_vstream_info",
                "scale": 0.03125,
                "zero_point": 17.0,
            },
        }

    monkeypatch.setattr(runtime, "_hailo_metadata", fake_hailo_metadata)

    trtexec = tmp_path / "bin/trtexec"
    trtexec.parent.mkdir(parents=True, exist_ok=True)
    trtexec.write_bytes(b"#!/bin/sh\nexit 0\n")
    trtexec.chmod(0o755)
    builder_commands: list[list[str]] = []
    builder_build_count = [0]
    reject_next_no_build = [False]

    def option(command: list[str], name: str) -> str:
        positions = [index for index, value in enumerate(command) if value == name]
        assert len(positions) == 1, f"{name} must occur exactly once: {command}"
        return command[positions[0] + 1]

    def fake_builder_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        builder_commands.append(list(command))
        assert kwargs["text"] is True
        assert kwargs["stdout"] == runtime.subprocess.PIPE
        assert kwargs["stderr"] == runtime.subprocess.STDOUT
        assert int(kwargs["timeout"]) >= 60

        persistent_suite = Path(option(command, "--benchmark-set")).resolve()
        assert option(command, "--case") == "b038"
        assert option(command, "--variants") == "part2"
        assert option(command, "--precision") == "uint8_dequant_fp16"
        assert option(command, "--boundary-layout") == "memory_nhwc_to_nchw"
        assert option(command, "--dequant-scale") == repr(0.03125)
        assert option(command, "--dequant-zero-point") == repr(17.0)
        engine_root = Path(option(command, "--out-dir")).resolve()
        summary_path = Path(option(command, "--json-out")).resolve()
        persisted_source = persistent_suite / "b038/source_part2.onnx"

        if "--no-build" in command and reject_next_no_build[0]:
            reject_next_no_build[0] = False
            _write_json(summary_path, {
                "ok": False,
                "artifacts": [{
                    "build_ok": True,
                    "engine_build_receipt_status": "engine_build_receipt_sha256_mismatch",
                }],
            })
            return SimpleNamespace(returncode=0, stdout="mock receipt rejection")

        work_dir = engine_root / "b038/part2_uint8_dequant_fp16"
        if "--no-build" in command:
            # Model the real builder's receipt-only verification path: it emits
            # a summary but must not rewrite the sealed engine leaf.
            existing_meta_path = work_dir / "native_trt_meta.json"
            assert existing_meta_path.is_file()
            existing_meta = json.loads(existing_meta_path.read_text(encoding="utf-8"))
            _write_json(summary_path, {
                "ok": True,
                "artifacts": [existing_meta],
            })
            return SimpleNamespace(returncode=0, stdout="mock receipt verified")

        builder_build_count[0] += 1
        work_dir.mkdir(parents=True, exist_ok=True)
        build_part2 = work_dir / "part2_uint8_dequant_bridge.onnx"
        build_part2.write_bytes(
            persisted_source.read_bytes() + b"\nmock-dequant-layout-bridge-v1"
        )
        engine = work_dir / "part2.engine"
        engine.write_bytes(b"real-temporary-tensorrt-engine-v1")
        receipt_path = work_dir / "engine_build_receipt.json"
        trt_command = [
            str(trtexec.resolve()),
            f"--onnx={build_part2.resolve()}",
            f"--saveEngine={engine.resolve()}",
            "--fp16",
        ]
        receipt: dict[str, Any] = {
            "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
            "schema_version": 1,
            "build_returncode": 0,
            "dry_run": False,
            "command": trt_command,
            "source_onnx": str(build_part2.resolve()),
            "source_onnx_sha256": _sha256(build_part2),
            "engine": str(engine.resolve()),
            "engine_sha256": _sha256(engine),
            "trtexec": str(trtexec.resolve()),
            "trtexec_sha256": _sha256(trtexec),
        }
        receipt["receipt_sha256"] = canonical_json_sha256(receipt)
        _write_json(receipt_path, receipt)

        bridge = {
            "schema": "onnx-splitpoint/uint8-dequant-bridge",
            "schema_version": 2,
            "source": str(persisted_source.resolve()),
            "source_sha256": _sha256(persisted_source),
            "bridge": str(build_part2.resolve()),
            "bridge_sha256": _sha256(build_part2),
            "input_name": "cut",
            "input_dtype": "UINT8",
            "input_shape": [1, 2, 2, 2],
            "boundary_layout": {
                "requested": "memory_nhwc_to_nchw",
                "effective": "memory_nhwc_to_nchw",
                "applied": True,
            },
            "scale": 0.03125,
            "zero_point": 17.0,
        }
        native_meta: dict[str, Any] = {
            "schema": "onnx-splitpoint/native-trt-meta",
            "schema_version": 1,
            "case": "b038",
            "variant": "part2",
            "onnx": str(build_part2.resolve()),
            "source_onnx": str(build_part2.resolve()),
            "engine": str(engine.resolve()),
            "inputs": [{
                "name": "cut", "shape": [1, 2, 2, 2],
                "elem_type": "UINT8", "has_dynamic": False,
            }],
            "outputs": [{
                "name": "output0", "shape": [1, 2, 2, 2],
                "elem_type": "FLOAT", "has_dynamic": False,
            }],
            "inputs_static": True,
            "precision": "uint8_dequant_fp16",
            "requested_precision": "uint8_dequant_fp16",
            "uint8_cast_bridge": bridge,
            "build_ok": True,
            "build": {"returncode": 0, "cmd": trt_command},
            "engine_build_receipt_status": "engine_build_receipt_verified",
            "engine_build_receipt_path": str(receipt_path.resolve()),
            "engine_build_receipt": receipt,
        }
        _write_json(work_dir / "native_trt_meta.json", native_meta)
        _write_json(summary_path, {
            "ok": True,
            "artifacts": [native_meta],
        })
        return SimpleNamespace(returncode=0, stdout="mock builder completed")

    monkeypatch.setattr(runtime.subprocess, "run", fake_builder_run)

    output_path = tmp_path / "collected/native_split_quality_binding.json"
    result = runtime.prepare_native_split_quality_binding(
        benchmark_set=benchmark_set,
        case_id="38",
        model_id="yolo26s",
        setup_id="hailo8_setup",
        backend="hailo8",
        eval_run_id="eval-runtime-e2e",
        source_run_id="hailo8_to_trt",
        cache_root=tmp_path / "persistent_cache",
        output_path=output_path,
        workspace_mb=2048,
        timeout_s=90,
    )

    assert len(metadata_calls) == 1
    assert len(builder_commands) == 1
    assert output_path.is_file()
    binding = result["binding"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == binding
    persistent_binding = Path(result["persistent_binding_path"])
    assert json.loads(persistent_binding.read_text(encoding="utf-8")) == binding

    persistent = persistent_binding.parent
    assert re.fullmatch(r"[0-9a-f]{64}", persistent.name)
    expected_key = canonical_json_sha256({
        "policy_sha256": metadata_calls[0][2]["policy_sha256"],
        "part1_sha256": _sha256(source_part1),
        "source_part2_sha256": _sha256(source_part2),
        "boundary_tensor": {
            "name": "cut",
            "runtime_name": "cut/hailort",
            "shape": [2, 2, 2],
            "canonical_part2_shape": [1, 2, 2, 2],
            "dtype": "uint8",
            "quantization": {
                "source": "hailort_hef_output_vstream_info",
                "scale": 0.03125,
                "zero_point": 17.0,
            },
        },
        "resolved_boundary_layout": "memory_nhwc_to_nchw",
        "resolved_boundary_transform": "uint8_dequant_then_layout",
    })
    assert persistent.name == expected_key
    expected_prefix = (
        tmp_path / "persistent_cache/native_split_quality/hailo8_setup"
        / "yolo26s/b038/hailo8_to_trt"
    ).resolve()
    assert persistent.parent == expected_prefix
    artifacts = binding["artifacts"]
    assert Path(artifacts["part1_runtime"]["path"]) == persistent / "part1.hef"
    assert Path(artifacts["source_part2_onnx"]["path"]) == (
        persistent / "benchmark_set/b038/source_part2.onnx"
    )
    assert Path(artifacts["part1_runtime"]["path"]).read_bytes() == source_part1.read_bytes()
    assert Path(artifacts["source_part2_onnx"]["path"]).read_bytes() == source_part2.read_bytes()

    expected_identity = {
        "model": "yolo26s", "case": "b038", "setup_id": "hailo8_setup",
        "backend": "hailo8_to_trt", "task": "detection",
        "precision": "uint8_dequant_fp16",
    }
    portable, portable_status = validate_native_split_quality_binding(
        binding, verification_mode="portable", expected_identity=expected_identity,
    )
    local, local_status = validate_native_split_quality_binding(
        binding, verification_mode="local", expected_identity=expected_identity,
    )
    assert portable == binding
    assert portable_status == (
        "portable_embedded_evidence_and_cross_links_verified_without_local_rehash"
    )
    assert local == binding
    assert local_status == "local_files_rehashed_and_exact_cross_links_verified"

    receipt = binding["engine_build_receipt"]
    meta = binding["native_trt_meta"]
    bridge = meta["uint8_cast_bridge"]
    assert bridge["source"] == artifacts["source_part2_onnx"]["path"]
    assert bridge["source_sha256"] == artifacts["source_part2_onnx"]["sha256"]
    assert bridge["bridge"] == artifacts["build_part2_onnx"]["path"]
    assert bridge["bridge_sha256"] == artifacts["build_part2_onnx"]["sha256"]
    assert receipt["source_onnx"] == artifacts["build_part2_onnx"]["path"]
    assert receipt["source_onnx_sha256"] == artifacts["build_part2_onnx"]["sha256"]
    assert receipt["engine"] == artifacts["engine"]["path"]
    assert receipt["engine_sha256"] == artifacts["engine"]["sha256"]
    assert receipt["trtexec"] == artifacts["trtexec"]["path"]
    assert receipt["trtexec_sha256"] == artifacts["trtexec"]["sha256"]

    # A normal warm run verifies the exact persisted receipt through the
    # builder's --no-build path.  It must not enter the destructive build path
    # which removes the existing engine/receipt first.
    engine_path = Path(artifacts["engine"]["path"])
    receipt_path = Path(artifacts["engine_build_receipt"]["path"])
    engine_before = (engine_path.read_bytes(), engine_path.stat().st_mtime_ns)
    receipt_before = (receipt_path.read_bytes(), receipt_path.stat().st_mtime_ns)
    reused = runtime.prepare_native_split_quality_binding(
        benchmark_set=benchmark_set,
        case_id="b038",
        model_id="yolo26s",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
        eval_run_id="eval-runtime-e2e",
        source_run_id="hailo8_to_trt",
        cache_root=tmp_path / "persistent_cache",
        output_path=output_path,
        workspace_mb=2048,
        timeout_s=90,
    )
    assert len(builder_commands) == 2
    assert "--no-build" in builder_commands[-1]
    assert "--no-build" in reused["binding"]["producer_command"]
    assert reused["binding"]["engine_build_receipt"] == receipt
    assert builder_build_count[0] == 1
    assert (engine_path.read_bytes(), engine_path.stat().st_mtime_ns) == engine_before
    assert (receipt_path.read_bytes(), receipt_path.stat().st_mtime_ns) == receipt_before

    # Verification failure is not accepted as a hit: retry through the normal
    # builder command, which is the only path allowed to replace the leaf.
    reject_next_no_build[0] = True
    rebuilt = runtime.prepare_native_split_quality_binding(
        benchmark_set=benchmark_set,
        case_id="b038",
        model_id="yolo26s",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
        eval_run_id="eval-runtime-e2e",
        source_run_id="hailo8_to_trt",
        cache_root=tmp_path / "persistent_cache",
        output_path=output_path,
        workspace_mb=2048,
        timeout_s=90,
    )
    assert len(builder_commands) == 4
    assert "--no-build" in builder_commands[-2]
    assert "--no-build" not in builder_commands[-1]
    assert "--no-build" not in rebuilt["binding"]["producer_command"]
    assert builder_build_count[0] == 2

    # A content-addressed destination is immutable: a conflicting cache entry
    # aborts before a second builder invocation can observe or overwrite it.
    Path(artifacts["part1_runtime"]["path"]).write_bytes(b"conflicting-cache-bytes")
    with pytest.raises(
        RuntimeError, match="native_split_quality_persistent_cache_conflict",
    ):
        runtime.prepare_native_split_quality_binding(
            benchmark_set=benchmark_set,
            case_id="b038",
            model_id="yolo26s",
            setup_id="hailo8_setup",
            backend="hailo8_to_trt",
            eval_run_id="eval-runtime-e2e",
            source_run_id="hailo8_to_trt",
            cache_root=tmp_path / "persistent_cache",
            output_path=output_path,
            workspace_mb=2048,
            timeout_s=90,
        )
    assert len(builder_commands) == 4
