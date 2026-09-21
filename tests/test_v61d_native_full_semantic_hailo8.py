from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_output_manifest(
    path: Path, image: Path, *, task: str = "classification",
    model: str = "resnet50", backend: str = "native_full_hailo8",
    setup_id: str = "", comparison_backend: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.parent / "output_00.bin"
    out.write_bytes(b"\0\0\0\0")
    inp = path.parent / "input_rgb_uint8.bin"
    inp.write_bytes(b"\0" * 12)
    input_manifest = path.parent / "native_full_input_manifest.json"
    input_manifest.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": backend, "model": model, "setup_id": setup_id,
        "comparison_backend": comparison_backend, "case": "full",
        "task": task,
        "input_dump": str(inp), "input_shape_hwc": [2, 2, 3],
        "preprocess": {"ort_model_scale": "imagenet" if task == "classification" else "norm"},
    }), encoding="utf-8")
    path.write_text(json.dumps({
        "schema": "onnx-splitpoint/runner-output-dump", "schema_version": 3,
        "model": model, "backend": backend, "setup_id": setup_id,
        "comparison_backend": comparison_backend, "case": "full",
        "execution_mode": "native_full_baseline",
        "task": task,
        "output_format": "classification_logits" if task == "classification" else "bn6_detections",
        "contract_family": "classification_logits" if task == "classification" else "decoded_nms",
        "contract_source": "test",
        "input_image": str(image), "boundary_manifest": str(input_manifest),
        "outputs": [{"name": "output", "file": out.name, "dtype": "float32", "shape": [1]}],
    }), encoding="utf-8")


def test_native_full_semantic_companion_is_packaged_and_contract_complete() -> None:
    source = Path("scripts/native_full_semantic_dump.py")
    packaged = Path("onnx_splitpoint_tool/resources/remote_scripts/native_full_semantic_dump.py")
    assert source.is_file()
    assert packaged.is_file()
    assert source.read_bytes() == packaged.read_bytes()
    text = source.read_text(encoding="utf-8")
    assert "native_full_outputs_manifest.json" in text
    assert "native_full_input_manifest.json" in text
    assert "decoded_nms" in text


def test_semantic_full_dump_uses_direct_native_companion(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script("v61d_full_runner_semantic", Path("scripts/native_full_baseline_eval_runner.py"))
    bs = tmp_path / "resnet50" / "benchmark_set"
    case = bs / "b001"
    case.mkdir(parents=True)
    image = bs / "resources" / "validation" / "classification" / "images" / "n0001" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    (bs / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_id": "b001"}]}), encoding="utf-8")
    (case / "run_split_onnxruntime.py").write_text("# marker\n", encoding="utf-8")
    captured = {}
    producer_sha = "a" * 64
    monkeypatch.setattr(
        mod, "_quality_first_trt_producer_identity",
        lambda *_args, **_kwargs: (
            {"producer_identity_sha256": producer_sha},
            "quality_first_identity_verified_exact",
            tmp_path / "producer.json",
        ),
    )
    monkeypatch.setattr(
        mod, "_trt_quality_identity_cli_args",
        lambda *_args, **_kwargs: [],
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = [str(x) for x in cmd]
        dump_dir = Path(captured["cmd"][captured["cmd"].index("--out-dir") + 1])
        manifest = dump_dir / "native_full_outputs_manifest.json"
        _write_output_manifest(
            manifest, image, model="resnet50", backend="native_full_tensorrt",
            setup_id="setup-a", comparison_backend="hailo8",
        )
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_payload["quality_first_producer_identity_sha256"] = producer_sha
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        report = Path(captured["cmd"][captured["cmd"].index("--json-out") + 1])
        report.write_text(json.dumps({
            "ok": True, "output_manifest": str(manifest),
            "input_manifest": str(dump_dir / "native_full_input_manifest.json"),
            "quality_first_producer_identity_sha256": producer_sha,
        }), encoding="utf-8")
        return {"rc": 0, "returncode": 0, "timed_out": False, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(mod, "_run", fake_run)
    ns = Namespace(
        engine_python_selected=sys.executable, engine_python_sites=[],
        image_map_data={"resnet50": {"b001": image.relative_to(bs).as_posix()}},
        setup_id="setup-a", comparison_backend="hailo8", timeout=60,
        trt_precision="fp16", dump_outputs=True,
    )
    result = mod._semantic_full_dump(bs, "resnet50", "native_full_tensorrt", "ort_tensorrt", ns)
    assert result["ok"] is True
    assert "native_full_semantic_dump.py" in " ".join(captured["cmd"])
    assert "--backend" in captured["cmd"] and "tensorrt" in captured["cmd"]
    assert Path(result["output_dump_manifest"]).is_file()
    assert result["contract_family"] == "classification_logits"


def test_hailo8_full_uses_direct_hailo_runtime_and_writes_dump(tmp_path: Path, monkeypatch) -> None:
    _check_hailo_full_adapter(tmp_path, monkeypatch, "hailo8")


def test_hailo10_full_original_throughput_reaches_reader(tmp_path: Path, monkeypatch) -> None:
    _check_hailo_full_adapter(tmp_path, monkeypatch, "hailo10h")


def _check_hailo_full_adapter(tmp_path, monkeypatch, arch):
    mod = _load_script("v61d_full_runner_hailo8", Path("scripts/native_full_baseline_eval_runner.py"))
    bs = tmp_path / "resnet50" / "benchmark_set"
    (bs / "b052").mkdir(parents=True)
    (bs / "b052" / "run_split_onnxruntime.py").write_text("# marker\n", encoding="utf-8")
    (bs / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_id": "b052"}], "benchmark_task": "classification"}), encoding="utf-8")
    hef = bs / "hailo" / arch / "full" / "compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"hef")
    source = bs / "models" / "resnet50.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-onnx")
    preprocessing = mod.canonical_image_preprocessing_contract(
        "classification", (224, 224),
    )
    preprocessing_sha = mod.preprocessing_contract_sha256(preprocessing)
    source_sha = mod._sha256_file(source)
    calibration_identity = "none"
    prepared_calibration_identity = mod._canonical_json_sha256({
        "calibration_identity": calibration_identity,
        "preprocessing_contract_sha256": preprocessing_sha,
    })
    cache_payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache-key-v2",
        "model_sha256": source_sha,
        "activation_part1_sha256": "",
        "hw_arch": arch,
        "hailo_sdk_version": "test-sdk",
        "optimization_level": 1,
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_calibration_identity,
        "calibration_count": 54,
        "requested_calibration_count": 64,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 256 * 1024 * 1024,
        "calibration_batch_size": 8,
        "extra_model_script": "",
        "start_nodes": [],
        "end_nodes": [],
        "integrity": "relaxed",
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
    }
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": source_sha,
        "compiler_onnx_filename": source.name,
        "hef_sha256": mod._sha256_file(hef),
        "hef_size_bytes": hef.stat().st_size,
        "hw_arch": arch,
        "net_name": "resnet50",
        "hailo_sdk_version": "test-sdk",
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_calibration_identity,
        "calibration_count": 54,
        "requested_calibration_count": 64,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 256 * 1024 * 1024,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "cache_key": mod._canonical_json_sha256(cache_payload),
        "cache_payload": cache_payload,
    }
    (hef.parent / "hailo_hef_build_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    benchmark_payload = json.loads(
        (bs / "benchmark_set.json").read_text(encoding="utf-8")
    )
    benchmark_payload.update({
        "model": source.relative_to(bs).as_posix(),
        "model_source": str(source),
        "hailo": {"hefs": {arch: {
            "full": hef.relative_to(bs).as_posix(),
            "full_build": {
                "ok": True,
                "artifact_hash": mod._sha256_file(hef),
                "source_onnx_path": str(source),
                "compiler_onnx_path": str(source),
            },
            "full_endpoint_mode": "decoded",
            "full_end_node_names": [],
            "full_output_contract": {
                "mode": "classification_logits",
                "requires_external_postprocess": False,
                "end_node_names": [],
            },
        }}},
    })
    (bs / "benchmark_set.json").write_text(
        json.dumps(benchmark_payload), encoding="utf-8",
    )
    (bs / "output_contracts.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "resnet50",
        "task": "classification",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": "resnet50",
            "task": "classification",
            "backend": arch,
            "variant": "full",
            "contract_status": "pending_build_or_prepare",
            "artifact_binding_status": "pending_receipt_validation",
            "endpoint_mode": "decoded",
            "host_tail_required": False,
            "postprocessing_required": False,
            "full_end_node_names": [],
            "source_onnx_multiscale_raw_head": False,
        }],
    }), encoding="utf-8")
    image = bs / "resources" / "validation" / "classification" / "images" / "n1" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    captured = {}
    throughput = json.loads((Path(__file__).parent / "fixtures/v283_r9a_nachabnahme" / (arch + "_throughput.json")).read_text())["throughput"]

    monkeypatch.setattr(mod, "_select_hailo_python", lambda arch: (sys.executable, {"selected": sys.executable, "probes": []}))

    def fake_run(cmd, **kwargs):
        captured["cmd"] = [str(x) for x in cmd]
        report_path = Path(captured["cmd"][captured["cmd"].index("--json-out") + 1])
        dump_dir = Path(captured["cmd"][captured["cmd"].index("--dump-dir") + 1])
        manifest = dump_dir / "native_full_outputs_manifest.json"
        _write_output_manifest(
            manifest, image, model="resnet50", backend="native_full_" + arch,
            setup_id="h8", comparison_backend=arch,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "ok": True,
            "copy_outputs": True,
            "claim_copy_outputs_verified": True,
            "throughput": throughput,
            "input_image": str(image), "input_image_sha256": "abc",
            "output_manifest": str(manifest),
            "input_manifest": str(dump_dir / "native_full_input_manifest.json"),
        }), encoding="utf-8")
        return {"rc": 0, "returncode": 0, "timed_out": False, "stdout_tail": "throughput fps=123.0", "stderr_tail": ""}

    monkeypatch.setattr(mod, "_run", fake_run)
    ns = Namespace(
        frames=1000, warmup=100, inflight=8, timeout=60, duration_s=0.0,
        setup_id="h8", image_map_data={"resnet50": {"b052": str(image)}},
        comparison_backend=arch, dump_outputs=True,
    )
    row = mod._native_hailo_full(bs, "resnet50", arch, ns)
    assert row["ok"] is True
    assert row["fps_makespan"] == throughput["fps"]
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields, report_rate_fields
    aggregated = mod._aggregate_full_repetitions([row], requested=1)
    fields = rate_endpoint_fields(aggregated)
    # The measured endpoint is logits, without a timed classification Top-k.
    assert fields["completed_task_fps"] is None
    assert fields["host_output_fps"] == throughput["fps"]
    assert fields["host_output_rate"]["measurement_times_s"] == [throughput["elapsed_s"]]
    # The local nested raw report must not replace the adapter's series.
    assert report_rate_fields(aggregated)["host_output_fps"] == throughput["fps"]
    assert row["semantic_dump_status"] == "ok"
    assert Path(row["output_dump_manifest"]).is_file()
    command = " ".join(captured["cmd"])
    assert "smoke_hailo10_full_from_benchmarkset.py" in command
    assert f"--hw-arch {arch}" in command
    assert ("--runtime-api vstreams" if arch == "hailo8" else "--runtime-api infer_model") in command
    assert "--dump-outputs" in captured["cmd"]
    assert "--model resnet50" in command
    assert "--setup-id h8" in command
    assert f"--comparison-backend {arch}" in command
    assert f"/model=resnet50/backend=native_full_{arch}/setup=h8/comparison={arch}/" in row["output_dump_manifest"]
    assert "run_benchmark_suite_from_set.py" not in command


def test_hailo_full_missing_deferred_artifact_is_auditable(tmp_path: Path) -> None:
    mod = _load_script("v61d_full_runner_deferred", Path("scripts/native_full_baseline_eval_runner.py"))
    bs = tmp_path / "yolo26s" / "benchmark_set"
    bs.mkdir(parents=True)
    (bs / "deferred_full_baselines.json").write_text(json.dumps({"rows": [{"backend": "hailo8", "reason": "cache_only"}]}), encoding="utf-8")
    ns = Namespace(frames=100, duration_s=0.0)
    row = mod._native_hailo_full(bs, "yolo26s", "hailo8", ns)
    assert row["status"] == "deferred"
    assert row["failure_reason"] == "deferred_cold_build"


def test_native_full_report_preserves_semantic_manifest_fields(tmp_path: Path) -> None:
    mod = _load_script("v61d_final_report", Path("scripts/native_producer_final_report.py"))
    root = tmp_path / "eval"
    analysis = root / "analysis_tables"
    analysis.mkdir(parents=True)
    manifest = root / "resnet50" / "benchmark_set" / "native_full_outputs" / "model=resnet50" / "backend=native_full_hailo8" / "setup=setup" / "comparison=hailo8" / "native_full_outputs_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    (analysis / "native_full_baseline_eval.json").write_text(json.dumps({"rows": [{
        "backend": "native_full_hailo8", "model": "resnet50", "ok": True,
        "fps_makespan": 100.0, "output_dump_manifest": str(manifest),
        "input_image": "/tmp/image.jpg", "input_image_sha256": "abc",
        "semantic_dump_status": "ok", "task": "classification",
        "output_format": "classification_logits", "contract_family": "classification_logits",
    }]}), encoding="utf-8")
    rows = mod._rows_from_native_full(root)
    assert len(rows) == 1
    assert rows[0]["output_dump_manifest"] == str(manifest)
    assert rows[0]["semantic_dump_status"] == "ok"
    assert rows[0]["contract_family"] == "classification_logits"


def test_validator_prefers_manifest_recorded_directly_on_full_row(tmp_path: Path) -> None:
    mod = _load_script("v61d_validator", Path("scripts/native_producer_validate_visualize.py"))
    root = tmp_path / "resnet50" / "benchmark_set" / "native_full_outputs" / "model=resnet50" / "backend=native_full_hailo8" / "setup=setup" / "comparison=hailo8"
    root.mkdir(parents=True)
    manifest = root / "native_full_outputs_manifest.json"
    manifest.write_text(json.dumps({
        "model": "resnet50", "backend": "native_full_hailo8",
        "setup_id": "setup", "comparison_backend": "hailo8",
        "case": "full", "execution_mode": "native_full_baseline", "outputs": [],
    }), encoding="utf-8")
    row = {
        "backend": "native_full_hailo8", "model": "resnet50", "case": "full",
        "setup_id": "setup", "comparison_backend": "hailo8",
        "execution_mode": "native_full_baseline", "output_dump_manifest": str(manifest),
    }
    found, status = mod._find_dump(None, [tmp_path], row)
    assert found == manifest
    assert status.startswith("full_row_")


def test_workflow_syncs_full_semantic_helpers_and_requests_dumps() -> None:
    runner = Path("onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    for token in (
        "native_full_semantic_dump.py", "smoke_hailo10_hef_runner.py",
        "smoke_hailo10_full_from_benchmarkset.py", 'fargs.append("--dump-outputs")',
    ):
        assert token in runner
    binding = Path("onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py").read_text(encoding="utf-8")
    assert "explicit Native Full selection overrides Smoke cache_or_defer" in binding


def test_debug_pack_keeps_native_full_descriptors_but_excludes_tensor_bodies() -> None:
    app = Path("onnx_splitpoint_tool/gui/app.py").read_text(encoding="utf-8")
    cli = Path("scripts/create_evaluation_debug_pack.py").read_text(encoding="utf-8")
    builder = Path("onnx_splitpoint_tool/workflow/debug_pack.py").read_text(encoding="utf-8")
    assert "create_evaluation_debug_pack" in app
    assert "create_evaluation_debug_pack" in cli
    assert '".bin"' in builder
    assert '".json"' in builder
    hailo = Path("scripts/smoke_hailo10_hef_runner.py").read_text(encoding="utf-8")
    assert "synchronous_vstreams_native_loop" in hailo
