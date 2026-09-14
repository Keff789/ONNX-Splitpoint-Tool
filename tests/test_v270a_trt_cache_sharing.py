from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import types
import sys

import pytest

from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.split_export_runners import (
    write_runner_skeleton_onnxruntime,
)


def _generated_suite(tmp_path: Path):
    suite = tmp_path / "suite"
    case = suite / "b052"
    case.mkdir(parents=True)
    (suite / "resnet50.onnx").write_bytes(b"canonical-full-onnx")
    write_benchmark_suite_script(suite)
    write_runner_skeleton_onnxruntime(str(case), target="cpu")
    spec = importlib.util.spec_from_file_location(
        f"generated_suite_{id(tmp_path)}", suite / "benchmark_suite.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, suite, case


def _args(cache: Path) -> SimpleNamespace:
    validation = cache.parent / "validation.txt"
    validation.write_text("sample.jpg 0\n", encoding="utf-8")
    return SimpleNamespace(
        quality_evidence_eval_id="eval-smoke",
        quality_evidence_setup_id="deepx_setup",
        quality_evidence_model_id="resnet50",
        trt_cache_root=str(cache),
        native_trt_precision="fp16",
        native_trt_workspace_mb=1024,
        native_trt_build_timeout_s=120,
        validation_images=str(validation),
        validation_max_images=1,
        image="default",
        preset="classification",
        image_scale="auto",
        timeout=120,
        verbose_runs=False,
        progress_every=1,
        validation_reference_mode="cpu_full",
    )


def _run() -> dict:
    return {
        "id": "ort_tensorrt",
        "_native_full_trt_quality_companion": True,
        "quality_canary_endpoint_ids": ["tensorrt_at_deepx_m1_full"],
        "benchmark_task": "classification",
        "task_quality_gate": {"classification_max_top1_drop": 0.1},
    }


def _expected_engine(cache: Path, source: Path) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return (
        cache / "full" / digest / "fp16" / "full_fp16.engine"
    )


def test_quality_companion_populates_exact_generic_content_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, suite, _case = _generated_suite(tmp_path)
    cache = tmp_path / "persistent_trt_cache"
    captured: dict = {}

    def fake_run_case(*_pos, **kwargs):
        captured.update(kwargs)
        Path(kwargs["native_trt_engine_path"]).write_bytes(b"sealed-engine")
        Path(kwargs["native_trt_build_receipt"]).write_text(
            "{}", encoding="utf-8",
        )
        identity = dict(kwargs["full_only_quality_identity"])
        identity_sha = hashlib.sha256(json.dumps(
            identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        duplicates = {
            "full_only_plan_identity_required": True,
            "full_only_plan_identity": identity,
            "full_only_plan_identity_sha256": identity_sha,
            "quality_canary_id": identity["quality_canary_id"],
            "eval_run_id": identity["eval_run_id"],
        }
        quality_dir = Path(kwargs["out_dir_override"]) / "task_quality_inputs"
        quality_dir.mkdir(parents=True)
        candidate = quality_dir / "full_candidate.json"
        candidate.write_text(json.dumps({
            "schema": "onnx-splitpoint/task-quality-candidate-input",
            "schema_version": 1,
            "record_count": 1,
            "records": [{"image_id": "sample.jpg"}],
            **duplicates,
        }), encoding="utf-8")
        request = quality_dir / "full_request.json"
        request_payload = {
            "schema": "onnx-splitpoint/central-quality-evaluation-request",
            "schema_version": 1,
            "record_count": 1,
            "candidate": {
                "path": candidate.name,
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "size_bytes": candidate.stat().st_size,
            },
            **duplicates,
        }
        request.write_text(json.dumps(request_payload), encoding="utf-8")
        return {
            "status": "completed",
            "quality_input_request": {
                **request_payload,
                "request": {"path": str(request)},
            },
        }

    monkeypatch.setattr(module, "_run_case", fake_run_case)
    result = module._run_native_full_trt_quality_companion(
        root=suite,
        bench={"model_id": "resnet50"},
        plan={},
        run=_run(),
        run_cases=[{"case_dir": "b052", "boundary": 52}],
        args=_args(cache),
    )

    expected = _expected_engine(cache, suite / "resnet50.onnx")
    assert result["status"] == "completed"
    assert result["quality_input_request"][
        "full_only_plan_identity_required"
    ] is True
    assert Path(captured["native_trt_engine_path"]) == expected
    assert Path(captured["native_trt_build_receipt"]) == (
        expected.parent / "engine_build_receipt.json"
    )
    assert Path(captured["native_trt_build_onnx_path"]) == (
        expected.parent / "source.onnx"
    )
    assert captured["full_only_quality_identity"] == {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": "tensorrt_at_deepx_m1_full",
        "eval_run_id": "eval-smoke",
        "model_id": "resnet50",
        "setup_id": "deepx_setup",
        "source_run_id": "native_full_tensorrt",
        "backend": "tensorrt",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    # Full uses the stable model root directly; split engines are routed below
    # <root>/splits/<case> by the generated suite.
    assert Path(captured["trt_cache_root"]) == cache

    # Compare against the path function from the *generated* case runner, then
    # instantiate its real cache consumer with engine loading stubbed out.
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    runner_spec = importlib.util.spec_from_file_location(
        f"generated_case_runner_{id(tmp_path)}",
        suite / "b052" / "run_split_onnxruntime.py",
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    monkeypatch.setitem(sys.modules, runner_spec.name, runner)
    runner_spec.loader.exec_module(runner)
    generic_engine = runner._native_trt_engine_path(
        "full", suite / "resnet50.onnx", "fp16", cache,
        canonical_role_layout=True,
    )
    assert generic_engine == expected

    # The fake quality call above only materializes placeholders.  Give the
    # real Generic consumer the exact receipt contract it now requires.
    persistent_source = expected.parent / "source.onnx"
    runner._write_explicit_native_trt_engine_receipt(
        source_onnx=persistent_source,
        engine_path=expected,
        receipt_path=expected.parent / "engine_build_receipt.json",
        trtexec=Path(sys.executable),
        command=[
                str(Path(sys.executable).resolve()),
                f"--onnx={persistent_source}",
                f"--saveEngine={expected}",
                "--fp16",
                "--memPoolSize=workspace:4096",
            ],
        returncode=0,
    )

    monkeypatch.setattr(
        runner.NativeTRTSession,
        "_build_engine",
        lambda *_a, **_k: pytest.fail("warm Generic cache rebuilt the engine"),
    )
    monkeypatch.setattr(
        runner.NativeTRTSession, "_load_engine", lambda *_a, **_k: None,
    )
    session = runner.NativeTRTSession(
        "full", suite / "resnet50.onnx", precision="fp16",
        cache_root=cache, canonical_role_layout=True,
    )
    assert session.engine_path == expected
    assert session.build_info["cache_hit"] is True


def test_failed_unsealed_engine_migration_restores_generic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, suite, _case = _generated_suite(tmp_path)
    cache = tmp_path / "persistent_trt_cache"
    engine = _expected_engine(cache, suite / "resnet50.onnx")
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"old-generic-engine")

    def failed_build(*_pos, **_kwargs):
        raise RuntimeError("synthetic rebuild failure")

    monkeypatch.setattr(module, "_run_case", failed_build)
    with pytest.raises(RuntimeError, match="synthetic rebuild failure"):
        module._run_native_full_trt_quality_companion(
            root=suite,
            bench={"model_id": "resnet50"},
            plan={},
            run=_run(),
            run_cases=[{"case_dir": "b052", "boundary": 52}],
            args=_args(cache),
        )

    assert engine.read_bytes() == b"old-generic-engine"
    assert not engine.with_name(engine.name + ".pre_2_70a_unsealed").exists()
    assert not (engine.parent / "engine_build_receipt.json").exists()


def test_missing_quality_prerequisite_does_not_move_generic_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, suite, _case = _generated_suite(tmp_path)
    cache = tmp_path / "persistent_trt_cache"
    engine = _expected_engine(cache, suite / "resnet50.onnx")
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"old-generic-engine")
    args = _args(cache)
    args.validation_images = ""
    run = _run()
    run.pop("task_quality_gate")

    def must_not_build(*_pos, **_kwargs):
        raise AssertionError("builder must not run without Quality prerequisites")

    monkeypatch.setattr(module, "_run_case", must_not_build)
    result = module._run_native_full_trt_quality_companion(
        root=suite,
        bench={"model_id": "resnet50"},
        plan={},
        run=run,
        run_cases=[{"case_dir": "b052", "boundary": 52}],
        args=args,
    )

    assert result is None
    assert engine.read_bytes() == b"old-generic-engine"
    assert not engine.with_name(engine.name + ".pre_2_70a_unsealed").exists()
