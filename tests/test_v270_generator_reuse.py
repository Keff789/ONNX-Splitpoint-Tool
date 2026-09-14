from __future__ import annotations

from pathlib import Path


def _run_end_policy_orchestration(tmp_path: Path, monkeypatch, *, fail_full: bool):
    from onnx_splitpoint_tool import api as asc
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionCallbacks,
        BenchmarkGenerationExecutionConfig,
        BenchmarkGenerationOrchestrationConfig,
        BenchmarkGenerationOrchestrationService,
        BenchmarkGenerationRuntime,
    )
    from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult

    monkeypatch.setattr(asc, "cut_tensors_for_boundary", lambda *_a, **_k: ["cut"])
    monkeypatch.setattr(
        asc,
        "split_model_on_cut_tensors",
        lambda *_a, **_k: (object(), object(), {"cut_tensors": ["cut"]}),
    )
    monkeypatch.setattr(asc, "save_model", lambda _model, path: Path(path).write_bytes(b"fake-onnx"))

    def _write_runner(case_dir, **_kwargs):
        runner = Path(case_dir) / "run_benchmark.py"
        runner.write_text("# generated test runner\n", encoding="utf-8")
        return str(runner)

    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", _write_runner)

    out_dir = tmp_path / "suite"
    out_dir.mkdir()
    full_model = tmp_path / "resnet50.onnx"
    full_model.write_bytes(b"fake-full-model")
    runtime = BenchmarkGenerationRuntime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark_generation.log",
        state_path=out_dir / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[39, 52],
        candidate_search_pool=[39, 52],
        model_name="resnet50",
        model_source=str(full_model),
        hef_full_policy="end",
    )
    runs = [
        {"id": "hailo8", "type": "hailo", "hw_arch": "hailo8", "variants": ["full"]},
        {
            "id": "hailo8_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "composed"],
        },
    ]
    calls: list[str] = []

    def _fake_hef_build(_model, **kwargs):
        net_name = str(kwargs.get("net_name") or "")
        calls.append(net_name)
        outdir = Path(kwargs["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        if fail_full and net_name.endswith("_full"):
            return HailoHefBuildResult(
                ok=False,
                elapsed_s=0.01,
                hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
                net_name=net_name,
                error="simulated suite Full failure",
            )
        hef = outdir / "compiled.hef"
        hef.write_bytes(net_name.encode("utf-8"))
        return HailoHefBuildResult(
            ok=True,
            elapsed_s=0.01,
            hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
            net_name=net_name,
            hef_path=str(hef),
            calib_info={"source": "test", "cache_hit": False},
        )

    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda _msg, **_kwargs: None,
        queue_put=lambda _event: None,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )
    execution = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[39, 52],
        candidate_search_pool=[39, 52],
        out_dir=out_dir,
        base="resnet50",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        bench_plan_runs=runs,
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        hef_targets=["hailo8"],
        hef_part1=True,
        hef_part2=False,
        hef_opt_level=1,
        hef_calib_count=500,
        hef_calib_bs=8,
        hailo_build_hef_fn=_fake_hef_build,
        require_complete_hailo_matrix_per_case=True,
    )

    def _write_harness(suite_dir: str, _benchmark_name: str) -> str:
        path = Path(suite_dir) / "benchmark_suite.py"
        path.write_text("# harness\n", encoding="utf-8")
        return str(path)

    orchestration = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=execution,
        execution_callbacks=callbacks,
        target_cases=1,
        preferred_shortlist_original=[39],
        ranked_candidates=[39, 52],
        candidate_search_pool=[39, 52],
        out_dir=out_dir,
        base="resnet50",
        pad=3,
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        analysis_payload={},
        analysis_params_payload={},
        system_spec_payload=None,
        bench_log_path=str(runtime.bench_log_path),
        bench_plan_runs=runs,
        hef_targets=["hailo8"],
        hef_full=True,
        hef_part1=True,
        hef_part2=False,
        hef_backend="test",
        hef_fixup=False,
        hef_opt_level=1,
        hef_calib_dir=None,
        hef_calib_count=500,
        hef_calib_bs=8,
        hef_force=False,
        hef_keep=False,
        hef_wsl_distro=None,
        hef_wsl_venv="",
        hef_timeout_s=60,
        full_hef_policy="end",
        full_model_preflight_policy="skip",
        hailo_build_hef_fn=_fake_hef_build,
        hailo_selected=True,
        write_harness_script=_write_harness,
        tool_gui_version="2.70.0",
        tool_core_version="2.70.0",
        require_complete_hailo_matrix_per_case=True,
    )
    result = BenchmarkGenerationOrchestrationService().run(orchestration)
    return result, runtime, calls


def test_hailo_artifact_contract_is_cross_run_and_cross_release_stable(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.hailo_backend import _v60s_hailo_contract

    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    run_a.mkdir()
    run_b.mkdir()
    for run_dir in (run_a, run_b):
        (run_dir / "part2.onnx").write_bytes(b"same-part2")
        (run_dir / "part1.onnx").write_bytes(b"same-part1")

    common = {
        "hw_arch": "hailo8",
        "opt_level": 1,
        "calib_count": 500,
        "calib_batch_size": 8,
    }
    contract_a = _v60s_hailo_contract({
        **common,
        "onnx_path": str(run_a / "part2.onnx"),
        "activation_part1_onnx": str(run_a / "part1.onnx"),
        "outdir": run_a / "hef",
        "tool_version": "2.69.6",
        "workflow_version": "v2.69f",
        "run_id": "old-run",
    })
    contract_b = _v60s_hailo_contract({
        **common,
        "onnx_path": str(run_b / "part2.onnx"),
        "activation_part1_onnx": str(run_b / "part1.onnx"),
        "outdir": run_b / "hef",
        "tool_version": "2.70.0",
        "workflow_version": "v2.70",
        "run_id": "new-run",
    })

    assert contract_a == contract_b

    (run_b / "part1.onnx").write_bytes(b"changed-part1")
    changed = _v60s_hailo_contract({
        **common,
        "onnx_path": str(run_b / "part2.onnx"),
        "activation_part1_onnx": str(run_b / "part1.onnx"),
    })
    assert changed != contract_a


def test_hailo_artifact_contract_binds_calibration_contents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from onnx_splitpoint_tool.hailo_backend import _v60s_hailo_contract

    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY", "strict")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"same-model")
    calib_a = tmp_path / "run-a" / "calibration"
    calib_b = tmp_path / "run-b" / "calibration"
    calib_a.mkdir(parents=True)
    calib_b.mkdir(parents=True)
    (calib_a / "sample.bin").write_bytes(b"sample-a")
    (calib_b / "sample.bin").write_bytes(b"sample-a")

    base = {
        "onnx_path": str(model),
        "hw_arch": "hailo8",
        "opt_level": 1,
        "calib_count": 500,
        "calib_batch_size": 8,
    }
    contract_a = _v60s_hailo_contract({**base, "calib_dir": str(calib_a)})
    same_contents = _v60s_hailo_contract({**base, "calib_dir": str(calib_b)})
    (calib_b / "sample.bin").write_bytes(b"sample-b")
    changed_contents = _v60s_hailo_contract({**base, "calib_dir": str(calib_b)})

    assert contract_a == same_contents
    assert contract_a != changed_contents


def test_hailo_artifact_store_reuses_hef_across_run_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import onnx_splitpoint_tool.hailo_backend as hailo_backend
    from onnx_splitpoint_tool.artifact_store import ArtifactStore
    from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService

    store_root = tmp_path / "artifact-store"
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(store_root))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_VERIFY", "strict")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(tmp_path / "hailo-cache")
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_MODE", "standard")

    model = tmp_path / "resnet50_part1.onnx"
    model.write_bytes(b"same-part1-onnx")
    run_a = tmp_path / "evaluation-run-a" / "hailo"
    run_b = tmp_path / "evaluation-run-b" / "hailo"
    run_a.mkdir(parents=True)
    (run_a / "compiled.hef").write_bytes(b"known-good-hef")

    auto_kwargs = {
        "backend": "venv",
        "hw_arch": "hailo8",
        "outdir": run_b,
        "net_input_shapes": {"images": [1, 3, 224, 224]},
        "task": "classification",
        "opt_level": 1,
        "calib_count": 500,
        "calib_batch_size": 8,
    }
    first_bound = hailo_backend._v60s_hailo_auto_bound((model,), auto_kwargs)
    artifact_contract = hailo_backend._v60s_hailo_contract(first_bound)
    preprocessing_contract = artifact_contract["preprocessing_contract"]
    preprocessing_sha256 = artifact_contract[
        "preprocessing_contract_sha256"
    ]
    cache_key, cache_payload = hailo_backend._hailo_cache_key(
        model_path=model,
        activation_part1=None,
        hw_arch="hailo8",
        opt_level=1,
        calib_dir=None,
        calib_count=500,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=None,
        preprocessing_contract=preprocessing_contract,
        net_name="resnet50_part1",
        net_input_shapes={"images": [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )
    build_receipt = hailo_backend._write_hailo_receipt(
        hef_path=run_a / "compiled.hef",
        source_onnx=model,
        compiler_onnx=model,
        hw_arch="hailo8",
        net_name="resnet50_part1",
        preprocessing_contract=preprocessing_contract,
        preprocessing_sha256=preprocessing_sha256,
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    ArtifactStore(store_root).register(
        source_path=run_a / "compiled.hef",
        kind="hailo_hef",
        contract=artifact_contract,
        metadata={
            "legacy_cache_key": cache_key,
            "preprocessing_contract_sha256": preprocessing_sha256,
            "build_receipt": build_receipt,
        },
    )

    def _unexpected_venv_dispatch(*_args, **_kwargs):
        raise AssertionError("managed venv dispatcher must not run on an artifact-store hit")

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_via_venv", _unexpected_venv_dispatch)
    second = hailo_backend.hailo_build_hef_auto(model, **auto_kwargs)

    assert second.ok is True
    assert second.skipped is True
    assert second.backend == "artifact_store"
    assert (run_b / "compiled.hef").read_bytes() == b"known-good-hef"
    assert second.calib_info is not None
    assert second.calib_info["cache_hit"] is True
    assert second.calib_info["cache_source"] == "artifact_store_exact_v2"
    assert second.calib_info["contract_hash"]
    assert second.details is not None
    assert second.details["artifact_store_restore"]["materialize_method"]
    assert len(ArtifactStore(store_root).list(kind="hailo_hef")) == 1

    summary = BenchmarkGenerationService().compact_hailo_build_summary(second)
    assert summary["cache_hit"] is True
    assert summary["cache_source"] == "artifact_store_exact_v2"
    assert summary["contract_hash"] == second.calib_info["contract_hash"]


def test_hailo_receipt_loader_rejects_tampered_compiler_identity(
    tmp_path: Path,
) -> None:
    import json

    import onnx_splitpoint_tool.hailo_backend as hailo_backend
    from onnx_splitpoint_tool.preprocessing_contract import (
        canonical_image_preprocessing_contract,
        preprocessing_contract_sha256,
    )

    source = tmp_path / "source.onnx"
    compiler = tmp_path / "compiler.onnx"
    hef = tmp_path / "compiled.hef"
    source.write_bytes(b"source-model")
    compiler.write_bytes(b"compiler-fixed-model")
    hef.write_bytes(b"compiled-hef")
    contract = canonical_image_preprocessing_contract(
        "classification", (224, 224)
    )
    cache_key, cache_payload = hailo_backend._hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch="hailo8",
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=None,
        preprocessing_contract=contract,
    )
    hailo_backend._write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch="hailo8",
        net_name="strict-receipt",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    assert hailo_backend._load_valid_hailo_receipt(
        hef,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        source_onnx_sha256=hailo_backend._bare_file_sha256(source),
        cache_key=cache_key,
        cache_payload=cache_payload,
    ) is not None

    receipt_path = hef.parent / "hailo_hef_build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["compiler_onnx_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert hailo_backend._load_valid_hailo_receipt(
        hef,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        source_onnx_sha256=hailo_backend._bare_file_sha256(source),
        cache_key=cache_key,
        cache_payload=cache_payload,
    ) is None


def test_hailo_artifact_store_rejects_v269f_record_without_v2_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import hashlib

    import onnx_splitpoint_tool.hailo_backend as hailo_backend
    from onnx_splitpoint_tool.artifact_store import ArtifactStore

    store_root = tmp_path / "artifact-store"
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(store_root))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_VERIFY", "strict")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(tmp_path / "hailo-cache")
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_MODE", "standard")

    model = tmp_path / "resnet50_part1.onnx"
    model.write_bytes(b"same-part1-onnx")
    old_hef = tmp_path / "v269f" / "compiled.hef"
    old_hef.parent.mkdir()
    old_hef.write_bytes(b"v269f-known-good-hef")
    legacy_contract = {
        "schema": "onnx-splitpoint/hailo-build-contract/v1",
        "activation_gen_batch": 8,
        "activation_part1_onnx": None,
        "add_conv_defaults": True,
        "calib_batch_size": 8,
        "calib_count": 500,
        "calib_dir": None,
        "disable_rt_metadata_extraction": True,
        "end_node_names": None,
        "extra_model_script": None,
        "fixup": True,
        "hw_arch": "hailo8",
        "keep_artifacts": False,
        "net_input_shapes": None,
        "net_name": None,
        "onnx_path": {
            "name": model.name,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "size": model.stat().st_size,
        },
        "onnx_splitpoint_run_mode": "standard",
        "opt_level": 1,
        "start_node_names": None,
    }
    ArtifactStore(store_root).register(
        source_path=old_hef,
        kind="hailo_hef",
        contract=legacy_contract,
        metadata={"source_release": "2.69f"},
    )

    dispatch_calls = []

    def _required_venv_dispatch(*_args, **kwargs):
        dispatch_calls.append(dict(kwargs))
        return hailo_backend.HailoHefBuildResult(
            ok=False,
            elapsed_s=0.01,
            hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
            net_name=str(kwargs.get("net_name") or "resnet50_part1"),
            backend="venv",
            error="simulated rebuild required for legacy receipt",
            failure_kind="launch_error",
        )

    monkeypatch.setattr(
        hailo_backend,
        "hailo_build_hef_via_venv",
        _required_venv_dispatch,
    )
    destination = tmp_path / "v270" / "hailo"
    result = hailo_backend.hailo_build_hef_auto(
        model,
        backend="venv",
        hw_arch="hailo8",
        outdir=destination,
        net_input_shapes={"images": [1, 3, 224, 224]},
        task="classification",
        opt_level=1,
        calib_count=500,
        calib_batch_size=8,
    )

    assert result.ok is False
    assert result.backend == "venv"
    assert len(dispatch_calls) == 1
    assert not (destination / "compiled.hef").exists()


def test_failed_forced_rebuild_never_registers_stale_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import onnx_splitpoint_tool.hailo_backend as hailo_backend
    from onnx_splitpoint_tool.artifact_store import ArtifactStore

    store_root = tmp_path / "artifact-store"
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(store_root))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_VERIFY", "strict")
    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_MODE", "standard")

    model = tmp_path / "resnet50_part1.onnx"
    model.write_bytes(b"current-part1-onnx")
    outdir = tmp_path / "failed-rebuild" / "hailo"
    outdir.mkdir(parents=True)
    (outdir / "compiled.hef").write_bytes(b"stale-hef-from-an-older-contract")

    def _failed_venv_build(_model, **kwargs):
        return hailo_backend.HailoHefBuildResult(
            ok=False,
            elapsed_s=0.01,
            hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
            net_name=str(kwargs.get("net_name") or "resnet50_part1"),
            backend="venv",
            error="simulated forced rebuild failure",
            failure_kind="launch_error",
        )

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_via_venv", _failed_venv_build)
    common = {
        "backend": "venv",
        "hw_arch": "hailo8",
        "outdir": outdir,
        "opt_level": 1,
        "calib_count": 500,
        "calib_batch_size": 8,
    }

    first = hailo_backend.hailo_build_hef_auto(model, **common, force=True)
    second = hailo_backend.hailo_build_hef_auto(model, **common)

    assert first.ok is False
    assert second.ok is False
    assert second.backend == "venv"
    assert len(ArtifactStore(store_root).list(kind="hailo_hef")) == 0


def test_smoke_and_standard_keep_distinct_hailo_cache_keys(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.hailo_backend import _hailo_cache_key

    model = tmp_path / "resnet50_part1.onnx"
    model.write_bytes(b"same-model")
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    (calib_dir / "selection.json").write_text('{"dataset":"same"}\n', encoding="utf-8")

    common = {
        "model_path": model,
        "activation_part1": None,
        "hw_arch": "hailo8",
        "calib_dir": calib_dir,
        "calib_batch_size": 8,
        "extra_model_script": "",
        "start_nodes": None,
        "end_nodes": None,
    }
    smoke_key, _ = _hailo_cache_key(
        **common,
        opt_level=0,
        calib_count=8,
    )
    standard_key, _ = _hailo_cache_key(
        **common,
        opt_level=1,
        calib_count=500,
    )
    standard_key_again, _ = _hailo_cache_key(
        **common,
        opt_level=1,
        calib_count=500,
    )

    assert smoke_key != standard_key
    assert standard_key_again == standard_key


def test_end_policy_accepts_split_before_single_suite_full_build(tmp_path: Path, monkeypatch) -> None:
    result, runtime, calls = _run_end_policy_orchestration(tmp_path, monkeypatch, fail_full=False)

    assert calls == ["resnet50_part1_b39", "resnet50_full"]
    assert [row["boundary"] for row in runtime.cases] == [39]
    assert runtime.discarded_cases == []
    availability = result.bench_payload["cases"][0]["hailo_case_variant_availability"]["hailo8"]
    assert availability["part1"] is True
    assert availability["full"] is True
    assert result.final_status == "ok"


def test_suite_full_failure_does_not_churn_split_candidate_pool(tmp_path: Path, monkeypatch) -> None:
    result, runtime, calls = _run_end_policy_orchestration(tmp_path, monkeypatch, fail_full=True)

    assert calls == ["resnet50_part1_b39", "resnet50_full"]
    assert [row["boundary"] for row in runtime.cases] == [39]
    assert runtime.discarded_cases == []
    assert not (tmp_path / "suite" / "b052").exists()
    availability = result.bench_payload["cases"][0]["hailo_case_variant_availability"]["hailo8"]
    assert availability["part1"] is True
    assert availability["full"] is False
    assert availability["full_failed"] is True
    assert result.final_status == "warn"
