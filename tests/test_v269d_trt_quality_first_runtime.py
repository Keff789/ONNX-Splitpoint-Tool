from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Tuple


ROOT = Path(__file__).resolve().parents[1]
FULL_RUNNER = ROOT / "scripts" / "native_full_baseline_eval_runner.py"
SUITE_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool" / "resources" / "templates"
    / "benchmark_suite.py.txt"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("v269d_full_runner", FULL_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _component(identity: dict[str, Any]) -> dict[str, Any]:
    return {"identity": identity, "sha256": _digest(identity)}


def _write_quality_producer_set(
    tmp_path: Path, *, model: str = "resnet50",
) -> tuple[Path, Path, dict[str, Any]]:
    run_root = tmp_path / "eval-a"
    suite = run_root / model / "benchmark_set"
    suite.mkdir(parents=True)
    persistent = tmp_path / "persistent_trt_cache" / "setup-a" / model
    persistent.mkdir(parents=True)
    source = persistent / "source.onnx"
    engine = persistent / "full_fp16.engine"
    trtexec = persistent / "trtexec"
    source.write_bytes(b"full-source-onnx")
    engine.write_bytes(b"quality-first-engine")
    trtexec.write_bytes(b"trtexec-binary")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    engine_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    trtexec_sha = hashlib.sha256(trtexec.read_bytes()).hexdigest()
    receipt_payload = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": [
            str(trtexec), f"--onnx={source}", f"--saveEngine={engine}",
            "--fp16",
        ],
        "source_onnx": str(source),
        "source_onnx_sha256": source_sha,
        "engine": str(engine),
        "engine_sha256": engine_sha,
        "trtexec": str(trtexec),
        "trtexec_sha256": trtexec_sha,
    }
    receipt = {**receipt_payload, "receipt_sha256": _digest(receipt_payload)}
    receipt_path = persistent / "engine_build_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8",
    )
    receipt_binding = {
        "path": str(receipt_path),
        "sha256": _digest(receipt),
        "size_bytes": len(json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")),
        "receipt": receipt,
    }
    endpoint = _component({
        "schema": "onnx-splitpoint/output-endpoint-contract",
        "schema_version": 3,
        "task": "classification",
        "stage": "classification_logits",
        "output_format": "classification_logits",
        "tensor_signature": {
            "tensor_count": 1,
            "tensors": [{"index": 0, "rank": 1, "shape": [1000]}],
        },
        "semantic": {},
    })
    authority = _component({
        "schema": "onnx-splitpoint/tensorrt-endpoint-authority",
        "schema_version": 1,
        "graph_binding_source": (
            "authoritative_suite_output_contract_plus_exact_onnx_endpoint:v2"
        ),
        "source_contracts_sha256": hashlib.sha256(b"container").hexdigest(),
        "recorded_contract_sha256": hashlib.sha256(b"row").hexdigest(),
        "full_model_sha256": source_sha,
        "terminal_model_sha256": source_sha,
        "endpoint_contract_hash": endpoint["sha256"],
        "endpoint_contract_complete": True,
        "contract_resolution_status": "attested",
        "stage": "classification_logits",
        "output_endpoint_attestation": {
            "attested": True, "status": "passed",
            "stage": "classification_logits",
            "endpoint_contract_hash": endpoint["sha256"],
        },
    })
    precision = _component({
        "schema": "onnx-splitpoint/tensorrt-runtime-precision-contract",
        "schema_version": 1,
        "runtime_precision_identity": "fp16",
        "source_onnx_sha256": source_sha,
        "build_onnx_sha256": source_sha,
        "engine_sha256": engine_sha,
    })
    quality_identity = {
        "schema": "test/quality-contract", "schema_version": 1,
        "model": {"sha256": source_sha},
        "dataset": {"manifest_sha256": hashlib.sha256(b"dataset").hexdigest()},
        "preprocessing": {"sha256": hashlib.sha256(b"prep").hexdigest()},
    }
    quality = {
        **quality_identity, "quality_contract_sha256": _digest(quality_identity),
    }
    source_artifact = {
        "path": str(source), "sha256": source_sha,
        "size_bytes": source.stat().st_size,
    }
    build_artifact = {
        **source_artifact, "source_onnx_sha256": source_sha,
    }
    engine_artifact = {
        "path": str(engine), "sha256": engine_sha,
        "size_bytes": engine.stat().st_size,
        "source_onnx_sha256": source_sha,
        "build_onnx_sha256": source_sha,
    }
    producer = {
        "schema": "onnx-splitpoint/tensorrt-central-quality-producer-identity",
        "schema_version": 1,
        "eval_run_id": "eval-a", "model_id": model, "setup_id": "setup-a",
        "source_run_id": "native_full_tensorrt",
        "originating_plan_run_id": "hailo8_to_tensorrt",
        "case_id": "full", "execution_role": "full_quality_only",
        "backend": "native_tensorrt", "variant": "full",
        "task": "classification",
        "source_onnx": source_artifact,
        "build_onnx": build_artifact,
        "engine": engine_artifact,
        "trtexec": {
            "path": str(trtexec), "sha256": trtexec_sha,
            "size_bytes": trtexec.stat().st_size,
        },
        "engine_build_receipt": receipt_binding,
        "engine_build_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "model": {
            "source_onnx_sha256": source_sha,
            "source_onnx_size_bytes": source.stat().st_size,
            "build_onnx_sha256": source_sha,
            "build_onnx_size_bytes": source.stat().st_size,
            "runtime_artifact_sha256": engine_sha,
            "runtime_artifact_size_bytes": engine.stat().st_size,
        },
        "dataset": copy.deepcopy(quality["dataset"]),
        "preprocessing": copy.deepcopy(quality["preprocessing"]),
        "endpoint": endpoint,
        "endpoint_authority": authority,
        "precision": precision,
        "quality_record_endpoint": _component({
            "schema": "test/quality-record", "schema_version": 1,
            "canonical_record_endpoint": "classification_topk_hits",
        }),
        "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "preprocessing_contract_sha256": quality["preprocessing"]["sha256"],
        "decoder_contract_sha256": "", "nms_contract_sha256": "",
        "quality_record_endpoint_contract_sha256": hashlib.sha256(
            b"quality-record"
        ).hexdigest(),
        "endpoint_contract_hash": endpoint["sha256"],
        "endpoint_contract_complete": True,
        "runtime_precision_identity": "fp16",
        "policy_sha256": hashlib.sha256(b"policy").hexdigest(),
        "implementation_runner_sha256": hashlib.sha256(b"runner").hexdigest(),
        "performance_claims_emitted": False,
    }
    producer["producer_identity_sha256"] = _digest(producer)
    producer_set = {
        "schema": "onnx-splitpoint/tensorrt-quality-producer-set",
        "schema_version": 1,
        "eval_run_id": "eval-a", "setup_id": "setup-a",
        "producers_by_model": {model: producer},
    }
    producer_file = (
        run_root / "quality_first"
        / "tensorrt_quality_producer_set.json"
    )
    producer_file.parent.mkdir(parents=True)
    producer_file.write_text(json.dumps(producer_set), encoding="utf-8")
    return suite, producer_file, producer


def _namespace(producer_file: Path) -> argparse.Namespace:
    return argparse.Namespace(
        setup_id="setup-a", trt_quality_producer_json=str(producer_file),
        root=str(producer_file.parent.parent),
        trt_precision="fp16", engine_python_selected=sys.executable,
        duration_s=0.0, frames=7, warmup=2, workspace_mb=1024,
        no_shapes=False, timeout=60, repetitions=1, inflight=1,
        engine_build_python="auto", dump_outputs=False,
        diagnostic_deepx_input_probes=False, image_map_data={},
        comparison_backend="ort_tensorrt", comparison_precision="fp16",
    )


def test_full_runtime_reuses_exact_quality_engine_without_build(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    suite, producer_file, producer = _write_quality_producer_set(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        work_dir = Path(producer["engine"]["path"]).parent
        (work_dir / "native_trt_meta.json").write_text(json.dumps({
            "build_ok": True, "run_ok": True,
            "completed_work_units": 7,
            "completed_work_units_status": "exact_runtime_counter",
            "completed_work_units_source": "trtexec_export_times",
            "fps_makespan": 123.5, "latency_mean_ms": 8.1,
            "onnx": producer["build_onnx"]["path"],
            "engine_build_receipt_path": producer[
                "engine_build_receipt"
            ]["path"],
            "engine_build_receipt": producer[
                "engine_build_receipt"
            ]["receipt"],
            "run_smoke": {
                "returncode": 0,
                "cmd": [
                    producer["trtexec"]["path"],
                    "--loadEngine=" + producer["engine"]["path"],
                    "--iterations=7",
                ],
            },
            "quality_first_identity": {
                "quality_first_producer_identity_sha256": producer[
                    "producer_identity_sha256"
                ],
                "paths": {"engine": producer["engine"]["path"]},
                "hashes": {"engine": producer["engine"]["sha256"]},
            },
        }), encoding="utf-8")
        (work_dir / "run_trtexec.log").write_text("Throughput: 123.5 qps\n")
        return {
            "rc": 0, "returncode": 0, "timed_out": False,
            "stdout_tail": "", "stderr_tail": "", "error": "",
        }

    monkeypatch.setattr(runner, "_run", fake_run)
    row = runner._native_trt_full(suite, "resnet50", _namespace(producer_file))

    assert row["ok"] is True
    assert row["quality_first_producer_identity"] == producer
    assert row["quality_first_producer_identity_sha256"] == producer[
        "producer_identity_sha256"
    ]
    assert len(commands) == 1
    command = commands[0]
    assert "--no-build" in command
    assert command[command.index("--explicit-full-engine") + 1] == producer["engine"]["path"]
    assert command[command.index("--explicit-full-build-receipt") + 1] == producer[
        "engine_build_receipt"
    ]["path"]
    assert command[command.index("--quality-first-producer-identity-sha256") + 1] == producer[
        "producer_identity_sha256"
    ]

    contract = runner._full_command_contract(
        row=row, root=tmp_path, benchmark_set=suite,
        model="resnet50", backend_arg="tensorrt",
        ns=_namespace(producer_file),
    )
    assert contract["quality_first_producer_identity"] == producer
    assert contract["quality_first_producer_identity_sha256"] == producer[
        "producer_identity_sha256"
    ]
    assert contract["engine_build_receipt_sha256"] == producer[
        "engine_build_receipt"
    ]["sha256"]
    assert contract["engine_build_receipt_file_sha256"] == producer[
        "engine_build_receipt_file_sha256"
    ]
    assert contract["trt_engine_build_receipt_sha256"] == producer[
        "engine_build_receipt"
    ]["receipt"]["receipt_sha256"]
    for name in ("source_onnx", "build_onnx", "engine", "trtexec"):
        assert contract["artifacts"][name]["size_bytes"] > 0
    assert contract["artifacts"]["source_onnx"]["path"] == producer[
        "build_onnx"
    ]["path"]
    assert contract["artifacts"]["engine_build_receipt"]["size_bytes"] == producer[
        "engine_build_receipt"
    ]["size_bytes"]
    assert runner._sealed_trt_source_model_binding(contract) is True


def test_producer_survives_suite_cleanup_and_rejects_receipt_tamper(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    suite, producer_file, producer = _write_quality_producer_set(tmp_path)
    ns = _namespace(producer_file)
    shutil.rmtree(suite)
    loaded, status, _path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert status == "quality_first_identity_verified_exact"
    assert loaded == producer

    receipt_path = Path(producer["engine_build_receipt"]["path"])
    receipt_path.write_text(receipt_path.read_text() + "\n", encoding="utf-8")
    loaded, status, _path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert loaded is None
    assert status == "quality_first_receipt_binding_mismatch"


def test_producer_set_rejects_duplicate_json_model_key(tmp_path: Path) -> None:
    runner = _load_runner()
    run_root = tmp_path / "eval-a"
    suite = run_root / "resnet50" / "benchmark_set"
    suite.mkdir(parents=True)
    producer_file = (
        run_root / "quality_first"
        / "tensorrt_quality_producer_set.json"
    )
    producer_file.parent.mkdir(parents=True)
    producer_file.write_text(
        '{"schema":"onnx-splitpoint/tensorrt-quality-producer-set",'
        '"schema_version":1,"eval_run_id":"eval-a","setup_id":"setup-a",'
        '"producers_by_model":{"resnet50":{},"resnet50":{}}}',
        encoding="utf-8",
    )
    producer, status, _path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", _namespace(producer_file),
    )
    assert producer is None
    assert status == "quality_first_producer_set_invalid_json"


def test_producer_set_role_path_rejects_symlink_and_cross_run_replay(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    suite, producer_file, _producer = _write_quality_producer_set(
        tmp_path
    )
    ns = _namespace(producer_file)
    original = producer_file.read_bytes()

    real_file = producer_file.with_name("producer-set-real.json")
    producer_file.rename(real_file)
    producer_file.symlink_to(real_file)
    loaded, status, _path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert loaded is None
    assert status == "quality_first_producer_set_role_path_mismatch"
    producer_file.unlink()
    real_file.rename(producer_file)

    quality_dir = producer_file.parent
    real_dir = quality_dir.parent / "quality-first-real"
    quality_dir.rename(real_dir)
    quality_dir.symlink_to(real_dir, target_is_directory=True)
    loaded, status, _path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert loaded is None
    assert status == "quality_first_producer_set_role_path_mismatch"
    quality_dir.unlink()
    real_dir.rename(quality_dir)

    replayed = json.loads(original)
    replayed["eval_run_id"] = "different-run"
    producer_file.write_text(json.dumps(replayed), encoding="utf-8")
    loaded, status, _path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert loaded is None
    assert status == "quality_first_producer_set_identity_invalid"


def test_three_plan_triggers_claim_one_setup_model_producer() -> None:
    tree = ast.parse(SUITE_TEMPLATE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_claim_native_full_trt_quality_companion"
    )
    namespace: dict[str, Any] = {"Tuple": Tuple}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SUITE_TEMPLATE), "exec"), namespace)
    claim = namespace["_claim_native_full_trt_quality_companion"]
    seen: set[tuple[str, str, str]] = set()
    identity = ("eval-a", "setup-a", "resnet50")
    assert [claim(seen, identity) for _ in range(3)] == [True, False, False]


def test_repetition_aggregation_blocks_quality_engine_identity_drift(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    _suite, _producer_file, producer = _write_quality_producer_set(tmp_path)

    def row(identity: dict[str, Any], index: int) -> dict[str, Any]:
        receipt = identity["engine_build_receipt"]
        return {
            "ok": True, "fps_makespan": 100.0 + index,
            "latency_mean_ms": 10.0, "backend": "native_full_tensorrt",
            "model": "resnet50", "case": "full", "setup_id": "setup-a",
            "comparison_backend": "ort_tensorrt",
            "execution_precision": "fp16", "full_runtime_precision": "fp16",
            "source_onnx_sha256": identity["source_onnx"]["sha256"],
            "build_onnx_sha256": identity["build_onnx"]["sha256"],
            "runtime_artifact_sha256": identity["engine"]["sha256"],
            "engine_sha256": identity["engine"]["sha256"],
            "trtexec_sha256": identity["trtexec"]["sha256"],
            "engine_build_receipt_sha256": receipt["sha256"],
            "engine_build_receipt_file_sha256": identity[
                "engine_build_receipt_file_sha256"
            ],
            "trt_engine_build_receipt_sha256": receipt["receipt"][
                "receipt_sha256"
            ],
            "quality_first_producer_identity": identity,
            "quality_first_producer_identity_sha256": identity[
                "producer_identity_sha256"
            ],
            "quality_first_full_command_identity_sha256": identity[
                "producer_identity_sha256"
            ],
            "quality_first_semantic_command_identity_sha256": identity[
                "producer_identity_sha256"
            ],
            "native_full_execution_command_sha256": "a" * 64,
            "endpoint_contract_hash": identity["endpoint_contract_hash"],
            "runtime_instance_id": f"fresh_process:{index}",
            "completed_work_units": 7,
        }

    first = row(copy.deepcopy(producer), 1)
    drifted = copy.deepcopy(producer)
    drifted["engine"]["sha256"] = hashlib.sha256(b"different-engine").hexdigest()
    drifted["model"]["runtime_artifact_sha256"] = drifted["engine"]["sha256"]
    drifted["precision"]["identity"]["engine_sha256"] = drifted["engine"]["sha256"]
    drifted["precision"]["sha256"] = _digest(drifted["precision"]["identity"])
    drifted.pop("producer_identity_sha256")
    drifted["producer_identity_sha256"] = _digest(drifted)
    second = row(drifted, 2)

    result = runner._aggregate_full_repetitions([first, second], requested=2)
    assert result["ok"] is False
    assert result["status"] == "identity_drift"
    assert result["failure_reason"] == "native_full_repetition_identity_drift"
    assert "quality_first_producer_identity_sha256" in result[
        "repetition_identity_drift_fields"
    ]
    assert "engine_sha256" in result["repetition_identity_drift_fields"]


def test_remote_runner_mirrors_are_byte_identical() -> None:
    for name in (
        "native_full_baseline_eval_runner.py",
        "native_trt_from_benchmarkset.py",
        "native_full_semantic_dump.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
        ).read_bytes()
