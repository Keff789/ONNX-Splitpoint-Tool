from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationOrchestrationService,
)
from onnx_splitpoint_tool.deepx.compiler import DeepXBuildResult
from onnx_splitpoint_tool.deepx import compiler as deepx_compiler
from onnx_splitpoint_tool.deepx import env_status as deepx_env_status
from onnx_splitpoint_tool.gui import benchmark_workflow
import onnx_splitpoint_tool.workflow.deepx_build_binding as deepx_binding


def _hailo_result(
    *, ok: bool, hef_path: Path | None = None, cache_outcome: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        ok=ok,
        skipped=False,
        timed_out=False,
        failure_kind="" if ok else "unsupported_onnx",
        unsupported_reason="",
        error=None if ok else "decoded tail unsupported",
        elapsed_s=0.01,
        hef_path=str(hef_path) if hef_path is not None else "",
        fixed_onnx_path=None,
        details={},
        calib_info={"cache_outcome": cache_outcome} if cache_outcome else {},
    )


def test_yolo_raw_head_fallback_builds_once_then_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second identical fallback must not run any DFC build phase."""

    model = tmp_path / "yolo11l.onnx"
    model.write_bytes(b"same-yolo11-model")
    persistent_hef = tmp_path / "hailo-cache" / "raw-head.hef"
    calls: list[dict[str, object]] = []
    build_phases: list[str] = []

    def fake_builder(_model: str, **kwargs: object) -> SimpleNamespace:
        end_nodes = list(kwargs.get("end_node_names") or [])
        calls.append({
            "end_nodes": end_nodes,
            "force": kwargs.get("force"),
        })
        if not end_nodes:
            # The decoded-tail failure is a parser/compiler outcome, not a
            # successful cacheable raw-head build.
            return _hailo_result(ok=False)
        assert kwargs.get("force") is False
        if persistent_hef.is_file():
            return _hailo_result(
                ok=True, hef_path=persistent_hef, cache_outcome="HIT",
            )
        persistent_hef.parent.mkdir(parents=True)
        for phase in ("translate", "calibrate", "compile"):
            build_phases.append(phase)
        persistent_hef.write_bytes(b"cached-raw-head-hef")
        return _hailo_result(
            ok=True, hef_path=persistent_hef, cache_outcome="MISS_BUILT",
        )

    execution_cfg = SimpleNamespace(
        benchmark_task="detection",
        build_scheduler_config={},
        hailo_run_mode="standard",
    )
    cfg = SimpleNamespace(
        hef_targets=["hailo8"],
        hef_full=True,
        hef_part1=False,
        hef_part2=False,
        hailo_build_hef_fn=fake_builder,
        hailo_build_unavailable=None,
        should_cancel=None,
        hailo_full_end_node_names=[],
        hailo_full_endpoint_mode="",
        hailo_full_output_contract=None,
        out_dir=tmp_path / "suite",
        full_model_src=str(model),
        full_model_dst=str(model),
        base="yolo11l",
        hailo_full_timeout_explicit=False,
        hailo_full_timeout_s=0,
        hef_timeout_s=60,
        hef_backend="test",
        hef_fixup=False,
        hef_opt_level=1,
        hef_calib_dir=None,
        hef_calib_count=16,
        hef_calib_bs=1,
        hef_force=False,
        hef_keep=True,
        hef_wsl_distro=None,
        hef_wsl_venv="",
        hailo_cache_only=False,
        hailo_full_cache_only=False,
        hailo_run_mode="standard",
        execution_cfg=execution_cfg,
        analysis_payload={},
        analysis_params_payload={},
        bench_log_path=str(tmp_path / "suite" / "benchmark.log"),
    )
    service = BenchmarkGenerationOrchestrationService()
    monkeypatch.setattr(
        service,
        "_infer_suite_raw_head_end_nodes",
        lambda _cfg: ["cv2.2/Conv", "cv3.2/Conv"],
    )

    def run_once() -> tuple[dict[str, dict[str, object]], list[str]]:
        evidence: dict[str, dict[str, object]] = {}
        logs: list[str] = []
        service._build_suite_full_hefs(
            cfg,
            log=lambda message, **_kwargs: logs.append(str(message)),
            queue_put=lambda _event: None,
            errors=[],
            suite_hailo_hefs=evidence,
            publish_hailo_diagnostics=lambda *_args: None,
        )
        return evidence, logs

    first, _ = run_once()
    assert build_phases == ["translate", "calibrate", "compile"]
    first_phase_count = len(build_phases)

    second, second_logs = run_once()
    assert len(build_phases) == first_phase_count
    raw_calls = [call for call in calls if call["end_nodes"]]
    assert [call["force"] for call in raw_calls] == [False, False]
    assert first["hailo8"]["full_endpoint_mode"] == "raw_detection_head"
    assert second["hailo8"]["full_endpoint_mode"] == "raw_detection_head"
    assert first["hailo8"]["full_output_contract"] == second["hailo8"][
        "full_output_contract"
    ]
    assert any("raw detection-head fallback OK" in line for line in second_logs)


def test_deepx_full_persistent_cache_logs_miss_then_hit_and_builds_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "yolo11l.onnx"
    source.write_bytes(b"stable-deepx-source")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "frame.jpg").write_bytes(b"image")
    cache_root = tmp_path / "BackendArtifacts" / "deepx"
    suite = tmp_path / "suite"
    suite.mkdir()
    compiler_calls: list[Path] = []

    monkeypatch.setattr(
        deepx_binding,
        "inspect_deepx_environment",
        lambda **_kwargs: {
            "compiler_ready": True,
            "runtime_ready": True,
            "cache_dir": str(cache_root),
        },
    )
    monkeypatch.setattr(
        deepx_binding,
        "_onnx_first_input_info",
        lambda _path: ("images", [1, 3, 640, 640]),
    )

    def fake_compile_dxnn(**kwargs: object) -> DeepXBuildResult:
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        built = output_dir / "compiler-output.dxnn"
        built.write_bytes(b"expensive-dxcom-output")
        compiler_calls.append(built)
        return DeepXBuildResult(
            ok=True,
            status="ok",
            onnx_path=str(kwargs["onnx_path"]),
            config_path=str(kwargs["config_path"]),
            output_dir=str(output_dir),
            dxnn_path=str(built),
            message="DXNN built",
        )

    monkeypatch.setattr(deepx_binding, "compile_dxnn", fake_compile_dxnn)

    profile = {
        "deepx_build": {
            "mode": "reuse_and_build_missing",
            "cache_dir": str(cache_root),
            "calibration_dir": str(calibration),
        },
    }

    def run_once(name: str) -> tuple[dict[str, object], list[str], dict[str, object]]:
        logs: list[str] = []
        result = deepx_binding.materialize_deepx_build_binding(
            run_dir=tmp_path / name,
            model_id="yolo11l",
            model_path=str(source),
            row={"task": "detection", "input_shape": [1, 3, 640, 640]},
            profile_payload=profile,
            targets=["deepx_m1"],
            benchmark_set_contract={"suite_dir": str(suite)},
            log=logs.append,
        )
        status = json.loads(
            Path(result["artifacts"]["deepx_artifact_status_json"])
            .read_text(encoding="utf-8")
        )
        return result, logs, status

    first, first_logs, first_status = run_once("run-1")
    second, second_logs, second_status = run_once("run-2")

    assert first["status"] == second["status"] == "ok"
    assert len(compiler_calls) == 1
    assert first["metrics"]["deepx_build_status"] == "ready_built"
    assert first["metrics"]["deepx_cache_outcome"] == "MISS"
    assert first["metrics"]["deepx_cache_reason"] == "not_found"
    assert first_status["cache_lookup"]["outcome"] == "MISS"
    assert any(
        "[deepx-cache] MISS role=full model=yolo11l" in line
        and "reason=not_found" in line
        for line in first_logs
    )

    assert second["metrics"]["deepx_build_status"] == "ready_reused"
    assert second["metrics"]["deepx_cache_outcome"] == "HIT"
    assert second["metrics"]["deepx_cache_reason"] == (
        "artifact_identity_verified"
    )
    assert second_status["cache_lookup"]["outcome"] == "HIT"
    assert second_status["cache_lookup"]["artifact"].endswith("model.dxnn")
    assert any(
        "[deepx-cache] HIT role=full model=yolo11l" in line
        and "reason=artifact_identity_verified" in line
        for line in second_logs
    )


def test_deepx_part1_persistent_cache_logs_miss_then_hit_and_builds_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "sample.jpg").write_bytes(b"image")
    cache_root = tmp_path / "BackendArtifacts" / "deepx"
    compiler_calls: list[Path] = []

    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    monkeypatch.setattr(
        deepx_env_status,
        "inspect_deepx_environment",
        lambda **_kwargs: {
            "compiler_ready": True,
            "runtime_ready": True,
            "cache_dir": str(cache_root),
            "compiler_version": "test-dxcom-2.4",
        },
    )

    def fake_compile_dxnn(**kwargs: object) -> SimpleNamespace:
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        built = output_dir / "compiler-output.dxnn"
        built.write_bytes(b"expensive-part1-dxcom-output")
        compiler_calls.append(built)
        return SimpleNamespace(
            ok=True,
            status="ok",
            dxnn_path=str(built),
            log_path="",
            message="DXNN built",
        )

    monkeypatch.setattr(deepx_compiler, "compile_dxnn", fake_compile_dxnn)

    def run_once(name: str) -> tuple[dict[str, object], list[str]]:
        suite = tmp_path / name
        case = suite / "b024"
        case.mkdir(parents=True)
        part1 = case / "part1.onnx"
        # Byte-identical Part1 ONNXs model two independent EvalRuns with the
        # same accepted split contract.
        part1.write_bytes(b"stable-deepx-part1-onnx")
        (case / "split_manifest.json").write_text(
            json.dumps({"part1_model": part1.name}) + "\n",
            encoding="utf-8",
        )
        logs: list[str] = []
        result = benchmark_workflow._materialize_manual_deepx_part1_artifacts(
            out_dir=suite,
            bench_plan_runs=[{
                "id": "deepx_m1_to_tensorrt",
                "type": "matrix",
                "stage1": "deepx_m1",
                "stage2": "tensorrt",
            }],
            validation_images="",
            fallback_calib_dir=str(calibration),
            calibration_num=1,
            task_hint="detection",
            log=logs.append,
        )
        return result, logs

    first, first_logs = run_once("run-1")
    second, second_logs = run_once("run-2")

    assert first["status"] == second["status"] == "ok"
    assert len(compiler_calls) == 1
    first_case = first["cases"][0]
    second_case = second["cases"][0]
    assert first_case["build_status"] == "ready_built"
    assert first_case["cache_lookup"]["outcome"] == "MISS"
    assert first_case["cache_lookup"]["reason"] == "model_missing"
    assert any(
        "[deepx-cache] MISS role=part1" in line
        and "case=b024" in line
        and "reason=model_missing" in line
        for line in first_logs
    )
    assert second_case["build_status"] == "ready_reused"
    assert second_case["cache_lookup"]["outcome"] == "HIT"
    assert second_case["cache_lookup"]["reason"] == "compatible"
    assert any(
        "[deepx-cache] HIT role=part1" in line
        and "case=b024" in line
        and "reason=compatible" in line
        for line in second_logs
    )
    assert (tmp_path / "run-1/b024/deepx/deepx_m1/part1/model.dxnn").is_file()
    assert (tmp_path / "run-2/b024/deepx/deepx_m1/part1/model.dxnn").is_file()


def test_manual_deepx_full_cache_is_persistent_and_logs_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "yolo26s.onnx"
    source.write_bytes(b"same-manual-full-model")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "sample.jpg").write_bytes(b"image")
    cache_root = tmp_path / "BackendArtifacts" / "deepx"
    compiler_calls: list[Path] = []

    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    monkeypatch.setattr(
        deepx_env_status,
        "inspect_deepx_environment",
        lambda **_kwargs: {
            "compiler_ready": True,
            "runtime_ready": True,
            "cache_dir": str(cache_root),
        },
    )

    def fake_compile_dxnn(**kwargs: object) -> SimpleNamespace:
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        built = output_dir / "compiler-output.dxnn"
        built.write_bytes(b"manual-full-dxcom-output")
        compiler_calls.append(built)
        return SimpleNamespace(
            ok=True,
            status="ok",
            dxnn_path=str(built),
            log_path="",
            message="DXNN built",
        )

    monkeypatch.setattr(deepx_compiler, "compile_dxnn", fake_compile_dxnn)

    def run_once(name: str) -> tuple[dict[str, object], list[str]]:
        logs: list[str] = []
        result = benchmark_workflow._materialize_manual_deepx_full_artifact(
            out_dir=tmp_path / name,
            model_path=str(source),
            model=SimpleNamespace(graph=None),
            bench_plan_runs=[{"type": "deepx"}],
            validation_images="",
            validation_max_images=1,
            fallback_calib_dir=str(calibration),
            calibration_num=1,
            task_hint="detection",
            log=logs.append,
        )
        return result, logs

    first, first_logs = run_once("run-1-full")
    second, second_logs = run_once("run-2-full")

    assert first["status"] == second["status"] == "ok"
    assert len(compiler_calls) == 1
    assert first["build_status"] == "ready_built"
    assert first["cache_lookup"]["outcome"] == "MISS"
    assert first["cache_lookup"]["reason"] == "not_found"
    assert any(
        "[deepx-cache] MISS role=full model=yolo26s" in line
        for line in first_logs
    )
    assert second["build_status"] == "ready_reused"
    assert second["cache_lookup"]["outcome"] == "HIT"
    assert second["cache_lookup"]["reason"] == "model_present"
    assert any(
        "[deepx-cache] HIT role=full model=yolo26s" in line
        for line in second_logs
    )
