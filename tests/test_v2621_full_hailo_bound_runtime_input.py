from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import native_full_baseline_eval_runner as full_runner
from scripts import native_producer_energy_plan as energy_plan
from scripts import smoke_hailo10_hef_runner as hailo_runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    tensor = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    tensor_path = tmp_path / "runtime_input.bin"
    tensor.tofile(tensor_path)
    manifest = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 1,
        "runtime_input_name": "images",
        "runtime_input_shape": [2, 2, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": int(tensor.nbytes),
        "runtime_input_file": tensor_path.name,
        "runtime_input_sha256": _sha256(tensor_path),
        "preprocess": {
            "mode": "letterbox_rgb_uint8",
            "layout": "HWC",
            "rgb": True,
            "pad_value": 114,
        },
    }
    manifest_path = tmp_path / "native_full_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tensor_path, manifest_path, manifest


def _preverified_contract(spec: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/preverified-runtime-input-tensor",
        "schema_version": 1,
        "runtime_input_name": spec["runtime_input_name"],
        "runtime_input_shape": spec["runtime_input_shape"],
        "runtime_input_dtype": spec["runtime_input_dtype"],
        "runtime_input_bytes": spec["runtime_input_bytes"],
        "runtime_input_sha256": spec["runtime_input_sha256"],
        "runtime_input_layout": spec["runtime_input_layout"],
        "preprocess": spec["preprocess"],
        "source_manifest_artifact": "input_manifest",
        "source_runtime_input_artifact": "runtime_input_tensor",
    }


def test_full_manifest_validation_binds_schema_name_shape_dtype_bytes_and_file(
    tmp_path: Path,
) -> None:
    tensor_path, manifest_path, manifest = _runtime_fixture(tmp_path)
    for schema_version in (1, 2):
        accepted = dict(manifest)
        accepted["schema_version"] = schema_version
        manifest_path.write_text(json.dumps(accepted), encoding="utf-8")
        spec, status = full_runner._validated_runtime_input_manifest(
            manifest_path, allowed_root=tmp_path,
        )
        assert status == "runtime_input_manifest_and_tensor_verified"
        assert spec == {
            "runtime_input_file": str(tensor_path.resolve()),
            "runtime_input_name": "images",
            "runtime_input_shape": [2, 2, 3],
            "runtime_input_dtype": "uint8",
            "runtime_input_bytes": 12,
            "runtime_input_sha256": _sha256(tensor_path),
            "runtime_input_layout": "HWC",
            "preprocess": manifest["preprocess"],
            "input_image": "",
            "input_image_sha256": "",
        }

    for field, value, expected_status in (
        ("schema", "other/schema", "runtime_input_manifest_schema_incompatible"),
        (
            "schema_version", 3,
            "runtime_input_manifest_schema_version_incompatible",
        ),
        (
            "schema_version", True,
            "runtime_input_manifest_schema_version_incompatible",
        ),
        (
            "schema_version", "2",
            "runtime_input_manifest_schema_version_incompatible",
        ),
        ("runtime_input_shape", [2, 0, 3], "runtime_input_shape_invalid"),
        ("runtime_input_dtype", "object", "runtime_input_dtype_unsupported"),
        ("runtime_input_bytes", 11, "runtime_input_shape_dtype_byte_count_mismatch"),
    ):
        broken = dict(manifest)
        broken[field] = value
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        rejected, reason = full_runner._validated_runtime_input_manifest(
            manifest_path, allowed_root=tmp_path,
        )
        assert rejected is None and reason == expected_status

    for missing_field, expected_status in (
        ("schema", "runtime_input_manifest_schema_incompatible"),
        (
            "schema_version",
            "runtime_input_manifest_schema_version_incompatible",
        ),
    ):
        broken = dict(manifest)
        broken.pop(missing_field)
        manifest_path.write_text(json.dumps(broken), encoding="utf-8")
        rejected, reason = full_runner._validated_runtime_input_manifest(
            manifest_path, allowed_root=tmp_path,
        )
        assert rejected is None and reason == expected_status


def test_hailo_bound_loader_reads_exact_tensor_without_hash_or_preprocessing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tensor_path, manifest_path, _manifest = _runtime_fixture(tmp_path)
    spec, _ = full_runner._validated_runtime_input_manifest(
        manifest_path, allowed_root=tmp_path,
    )
    assert spec is not None
    contract = hailo_runner._runtime_input_contract(json.dumps(_preverified_contract(spec)))
    monkeypatch.setattr(
        hailo_runner, "_file_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("hashing is forbidden")),
    )
    inputs = hailo_runner._load_preverified_runtime_input(
        tensor_path,
        contract,
        preverified_sha256=str(spec["runtime_input_sha256"]),
        prepared_input_names=["images"],
        prepared_runtime_shapes={"images": (2, 2, 3)},
        quantized_inputs=True,
    )
    np.testing.assert_array_equal(
        inputs["images"], np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
    )

    with pytest.raises(RuntimeError, match="name mismatch"):
        hailo_runner._load_preverified_runtime_input(
            tensor_path, contract,
            preverified_sha256=str(spec["runtime_input_sha256"]),
            prepared_input_names=["wrong"],
            prepared_runtime_shapes={"wrong": (2, 2, 3)},
            quantized_inputs=True,
        )
    with pytest.raises(RuntimeError, match="shape mismatch"):
        hailo_runner._load_preverified_runtime_input(
            tensor_path, contract,
            preverified_sha256=str(spec["runtime_input_sha256"]),
            prepared_input_names=["images"],
            prepared_runtime_shapes={"images": (1, 2, 3)},
            quantized_inputs=True,
        )


def test_full_command_contract_seals_successful_hailo_canonical_runtime_input(
    tmp_path: Path,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    semantic_root = (
        benchmark_set / "native_full_outputs"
        / "model=yolo26s" / "backend=native_full_hailo10h"
        / "setup=hailo10" / "comparison=hailo10h"
    )
    semantic_root.mkdir(parents=True)
    tensor_path, manifest_path, _manifest = _runtime_fixture(semantic_root)
    image = tmp_path / "image.png"
    image.write_bytes(b"semantic-image")
    hef = tmp_path / "model.hef"
    hef.write_bytes(b"hef")
    source_onnx = tmp_path / "model.onnx"
    source_onnx.write_bytes(b"onnx")
    receipt_path = tmp_path / "hailo_hef_build_receipt.json"
    receipt_payload = {"schema": "test-receipt"}
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
    report = tmp_path / "performance.json"
    report.write_text(json.dumps({
        "input_names": ["images"],
        "output_names": ["output"],
        "runtime_input_shape": [2, 2, 3],
        "runtime_input_dtype": "uint8",
        "quantized_inputs": True,
        "quantized_outputs": False,
        "persistent_activation": True,
        "hotloop": True,
        "copy_inputs": True,
        "copy_outputs": True,
        "hw_arch": "hailo10h",
        "runtime_api": "infer_model",
        "task": "detection",
    }), encoding="utf-8")
    row = {
        "ok": True,
        "backend": "native_full_hailo10h",
        "model": "yolo26s",
        "setup_id": "hailo10",
        "comparison_backend": "hailo10h",
        "input_case": "sample",
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "hef_path": str(hef),
        "hailo_hef_build_receipt_status": (
            "hailo_hef_build_receipt_verified_exact"
        ),
        "hailo_hef_build_receipt_path": str(receipt_path),
        "hailo_hef_build_receipt_file_sha256": _sha256(receipt_path),
        "hailo_hef_build_receipt_sha256": (
            full_runner._canonical_json_sha256(receipt_payload)
        ),
        "hailo_hef_build_receipt": receipt_payload,
        "hailo_hef_source_onnx_path": str(source_onnx),
        "source_onnx_sha256": _sha256(source_onnx),
        "report": str(report),
        "input_manifest": str(manifest_path),
        "runtime_python": sys.executable,
        "frames": 50,
    }
    ns = SimpleNamespace(
        frames=50, duration_s=0.0, warmup=5, inflight=4,
        trt_precision="fp16", workspace_mb=1024,
        engine_python_selected="", engine_build_python=sys.executable,
        no_shapes=False, dump_outputs=True,
        diagnostic_deepx_input_probes=False,
        preprocess_mode="letterbox", letterbox_pad_value=114,
        setup_id="hailo10", comparison_backend="hailo10h", out_dir="",
    )
    contract = full_runner._full_command_contract(
        row=row, root=tmp_path, benchmark_set=benchmark_set,
        model="yolo26s", backend_arg="hailo10h", ns=ns,
    )
    assert contract["complete"] is True
    workload = contract["energy_workload"]
    assert workload["available"] is True
    assert workload["runtime_input_mode"] == "exact_semantic_dump_runtime_tensor"
    assert workload["canonical_input_slot_names"] == ["images"]
    assert workload["canonical_output_slot_names"] == ["output"]
    assert workload["runtime_input_shape"] == [2, 2, 3]
    assert workload["runtime_input_dtype"] == "uint8"
    assert workload["runtime_input_bytes"] == tensor_path.stat().st_size == 12
    assert contract["artifacts"]["runtime_input_tensor"]["sha256"] == _sha256(tensor_path)
    verified, status = energy_plan._verify_full_command_contract(
        contract,
        expected_identity={
            "backend": "native_full_hailo10h", "model": "yolo26s",
            "case": "full", "setup_id": "hailo10",
            "comparison_backend": "hailo10h",
        },
    )
    assert verified is not None, status


def test_tensorrt_full_contract_cryptographically_binds_source_onnx_to_engine(
    tmp_path: Path,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    semantic_root = (
        benchmark_set / "native_full_outputs"
        / "model=resnet50" / "backend=native_full_tensorrt"
        / "setup=trt-host" / "comparison=hailo8"
    )
    semantic_root.mkdir(parents=True)
    _tensor_path, manifest_path, _manifest = _runtime_fixture(semantic_root)
    source_onnx = tmp_path / "model.onnx"
    source_onnx.write_bytes(b"source-onnx")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"compiled-engine")
    trtexec = tmp_path / "trtexec"
    trtexec.write_bytes(b"runtime")
    report = tmp_path / "native_trt_meta.json"
    report.write_text(json.dumps({
        "onnx": str(source_onnx),
        "run_smoke": {
            "returncode": 0,
            "cmd": [str(trtexec), f"--loadEngine={engine}", "--fp16"],
        },
    }), encoding="utf-8")
    row = {
        "ok": True, "backend": "native_full_tensorrt", "model": "resnet50",
        "setup_id": "trt-host", "comparison_backend": "hailo8",
        "report": str(report), "input_manifest": str(manifest_path),
        "frames": 50, "task": "classification",
    }
    ns = SimpleNamespace(
        frames=50, duration_s=0.0, warmup=5, inflight=4,
        trt_precision="fp16", workspace_mb=1024,
        engine_python_selected="", engine_build_python=sys.executable,
        no_shapes=False, dump_outputs=True,
        diagnostic_deepx_input_probes=False,
        preprocess_mode="resize", letterbox_pad_value=0,
        setup_id="trt-host", comparison_backend="hailo8", out_dir="",
    )
    contract = full_runner._full_command_contract(
        row=row, root=tmp_path, benchmark_set=benchmark_set,
        model="resnet50", backend_arg="tensorrt", ns=ns,
    )
    # A successful load/run of arbitrary engine bytes is not build evidence.
    assert contract["complete"] is False
    assert full_runner._sealed_trt_source_model_binding(contract) is False

    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1, "build_returncode": 0, "dry_run": False,
        "command": [
            str(trtexec), f"--onnx={source_onnx}",
            f"--saveEngine={engine}", "--fp16",
        ],
        "source_onnx": str(source_onnx.resolve()),
        "source_onnx_sha256": _sha256(source_onnx),
        "engine": str(engine.resolve()), "engine_sha256": _sha256(engine),
        "trtexec": str(trtexec.resolve()), "trtexec_sha256": _sha256(trtexec),
    }
    receipt["receipt_sha256"] = full_runner._canonical_json_sha256(receipt)
    receipt_path = tmp_path / "engine_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload.update({
        "engine_build_receipt": receipt,
        "engine_build_receipt_path": str(receipt_path),
        "engine_build_receipt_status": "engine_build_receipt_verified",
    })
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    source_sha = _sha256(source_onnx)
    engine_sha = _sha256(engine)
    receipt_binding = {
        "path": str(receipt_path.resolve()),
        "sha256": full_runner._canonical_json_sha256(receipt),
        "size_bytes": len(json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")),
        "receipt": receipt,
    }
    producer = {
        "schema": "onnx-splitpoint/tensorrt-central-quality-producer-identity",
        "schema_version": 1,
        "eval_run_id": "eval-test", "model_id": "resnet50",
        "setup_id": "trt-host", "source_run_id": "native_full_tensorrt",
        "case_id": "full", "execution_role": "full_quality_only",
        "backend": "native_tensorrt", "variant": "full",
        "performance_claims_emitted": False,
        "source_onnx": {
            "path": str(source_onnx.resolve()), "sha256": source_sha,
            "size_bytes": source_onnx.stat().st_size,
        },
        "build_onnx": {
            "path": str(source_onnx.resolve()), "sha256": source_sha,
            "size_bytes": source_onnx.stat().st_size,
            "source_onnx_sha256": source_sha,
        },
        "engine": {
            "path": str(engine.resolve()), "sha256": engine_sha,
            "size_bytes": engine.stat().st_size,
            "source_onnx_sha256": source_sha,
            "build_onnx_sha256": source_sha,
        },
        "trtexec": {
            "path": str(trtexec.resolve()), "sha256": _sha256(trtexec),
            "size_bytes": trtexec.stat().st_size,
        },
        "engine_build_receipt": receipt_binding,
        "engine_build_receipt_file_sha256": _sha256(receipt_path),
    }
    producer["producer_identity_sha256"] = full_runner._canonical_json_sha256(
        producer
    )
    row.update({
        "quality_first_producer_identity": producer,
        "quality_first_producer_identity_sha256": producer[
            "producer_identity_sha256"
        ],
    })
    contract = full_runner._full_command_contract(
        row=row, root=tmp_path, benchmark_set=benchmark_set,
        model="resnet50", backend_arg="tensorrt", ns=ns,
    )
    assert contract["complete"] is True
    assert contract["source_model_sha256"] == source_sha
    assert contract["artifacts"]["source_onnx"]["sha256"] == source_sha
    assert contract["artifacts"]["engine"]["sha256"] == engine_sha
    assert (
        contract["artifacts"]["engine"]["compiled_from_source_onnx_sha256"]
        == source_sha
    )
    assert full_runner._sealed_trt_source_model_binding(contract) is True
    verified, status = energy_plan._verify_full_command_contract(
        contract,
        expected_identity={
            "backend": "native_full_tensorrt", "model": "resnet50",
            "case": "full", "setup_id": "trt-host",
            "comparison_backend": "hailo8", "model_sha256": source_sha,
        },
    )
    assert verified is not None, status

    tampered = json.loads(json.dumps(contract))
    tampered["artifacts"]["engine"]["compiled_from_source_onnx_sha256"] = "f" * 64
    tampered.pop("contract_sha256")
    tampered["contract_sha256"] = full_runner._canonical_json_sha256(tampered)
    rejected, status = energy_plan._verify_full_command_contract(
        tampered,
        expected_identity={
            "backend": "native_full_tensorrt", "model": "resnet50",
            "case": "full", "setup_id": "trt-host",
            "comparison_backend": "hailo8", "model_sha256": source_sha,
        },
    )
    assert rejected is None
    assert status == "full_command_contract_tensorrt_source_model_binding_invalid"


def test_full_hailo_energy_command_uses_only_preflight_bound_runtime_tensor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    tensor_path, manifest_path, _manifest = _runtime_fixture(tmp_path)
    spec, _ = full_runner._validated_runtime_input_manifest(
        manifest_path, allowed_root=tmp_path,
    )
    assert spec is not None
    runtime_contract = _preverified_contract(spec)
    runner_path = tmp_path / "hailo_hotloop.py"
    runner_path.write_text("# runner\n", encoding="utf-8")
    hef_path = tmp_path / "model.hef"
    hef_path.write_bytes(b"hef")
    contract = {
        "backend": "native_full_hailo10h",
        "model": "yolo26s",
        "setup_id": "hailo10",
        "comparison_backend": "hailo10h",
        "contract_sha256": "a" * 64,
        "artifacts": {
            "runtime_python": {"invocation_path": "/usr/bin/python3", "sha256": "b" * 64},
            "hotloop_runner": {"path": str(runner_path), "sha256": "c" * 64},
            "hef": {"path": str(hef_path), "sha256": "d" * 64},
            "input_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "runtime_input_tensor": {
                "path": str(tensor_path),
                "sha256": spec["runtime_input_sha256"],
                "bytes": spec["runtime_input_bytes"],
            },
        },
        "energy_workload": {
            "available": True,
            "kind": "hailo_full_hotloop",
            "runtime_python_artifact": "runtime_python",
            "runner_artifact": "hotloop_runner",
            "hef_artifact": "hef",
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "runtime_input_mode": "exact_semantic_dump_runtime_tensor",
            "runtime_input_contract": runtime_contract,
            "runtime_input_name": "images",
            "runtime_input_shape": [2, 2, 3],
            "runtime_input_dtype": "uint8",
            "runtime_input_bytes": 12,
            "canonical_input_slot_names": ["images"],
            "canonical_output_slot_names": ["output"],
            "hw_arch": "hailo10h",
            "runtime_api": "infer_model",
            "warmup": 0,
            "inflight": 4,
            "input_image": str(tmp_path / "image.png"),
            "input_image_sha256": "e" * 64,
            "task": "detection",
            "quantized_inputs": True,
            "quantized_outputs": False,
            "persistent_activation": True,
            "hotloop": True,
            "copy_inputs": True,
            "copy_outputs": True,
        },
    }
    commands: list[list[str]] = []
    frames = 7

    monkeypatch.setattr(
        full_runner, "_verified_energy_preflight_attestation",
        lambda raw, **_kwargs: (raw, "verified"),
    )
    monkeypatch.setattr(
        full_runner, "_sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("hashing is forbidden")),
    )

    def fake_run(cmd: list[str], **_kwargs: object) -> dict[str, object]:
        commands.append([str(value) for value in cmd])
        report_path = Path(cmd[cmd.index("--json-out") + 1])
        report_path.write_text(json.dumps({
            "completed_frames": frames,
            "runtime_input_binding_verified": True,
            "runtime_input_source": "preflight_bound_runtime_input_tensor",
            "runtime_input_file": str(tensor_path.resolve()),
            "runtime_input_sha256": spec["runtime_input_sha256"],
            "runtime_input_dtype": "uint8",
            "runtime_input_shape": [2, 2, 3],
            "runtime_input_bytes": 12,
            "image_decode_performed": False,
            "preprocessing_performed": False,
            "preprocessing_timed": False,
        }), encoding="utf-8")
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(full_runner, "_run", fake_run)
    ns = SimpleNamespace(
        energy_command_contract_json=json.dumps(contract),
        preflight_attestation="unused.json", preflight_nonce="nonce",
        preflight_attestation_max_age_s=60.0,
        out_dir=str(tmp_path / "out"), frames=frames, timeout=30,
        root=str(tmp_path),
    )
    assert full_runner._energy_workload_only(ns) == 0
    stdout = capsys.readouterr().out
    assert f"__SPLITPOINT_WORK_UNITS__={frames}" in stdout
    assert "__SPLITPOINT_WORK_UNITS_EXACT__=1" in stdout
    assert len(commands) == 1
    argv = commands[0]
    for flag in (
        "--runtime-input-bin", "--runtime-input-contract-json",
        "--preverified-runtime-input-sha256",
        "--canonical-input-slot-names-json", "--canonical-output-slot-names-json",
    ):
        assert flag in argv
    assert "--onnx" not in argv
    assert "--dump-outputs" not in argv
    assert "--preprocess-mode" not in argv
    assert "--letterbox-pad-value" not in argv


def test_full_hailo_runtime_runner_and_plan_resource_mirrors_match() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "smoke_hailo10_hef_runner.py",
        "native_full_baseline_eval_runner.py",
        "native_producer_energy_plan.py",
    ):
        assert (root / "scripts" / name).read_bytes() == (
            root / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
        ).read_bytes()
