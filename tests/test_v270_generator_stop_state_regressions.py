from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _install_minimal_split_api(monkeypatch) -> None:
    from onnx_splitpoint_tool import api as asc

    monkeypatch.setattr(asc, "cut_tensors_for_boundary", lambda *_a, **_k: ["cut"])
    monkeypatch.setattr(
        asc,
        "split_model_on_cut_tensors",
        lambda *_a, **_k: (object(), object(), {"cut_tensors": ["cut"]}),
    )
    monkeypatch.setattr(
        asc,
        "save_model",
        lambda _model, path: Path(path).write_bytes(b"fake-onnx"),
    )

    def _write_runner(case_dir, **_kwargs):
        runner = Path(case_dir) / "run_benchmark.py"
        runner.write_text("# generated test runner\n", encoding="utf-8")
        return str(runner)

    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", _write_runner)


def _part1_plan() -> list[dict[str, Any]]:
    return [
        {
            "id": "hailo8_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "composed"],
        }
    ]


def _callbacks(runtime, *, logs: list[str] | None = None):
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionCallbacks,
    )
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
        _benchmark_service_log_adapter,
    )

    captured_logs = logs if logs is not None else []
    return BenchmarkGenerationExecutionCallbacks(
        log=_benchmark_service_log_adapter(captured_logs.append),
        queue_put=lambda _event: None,
        persist_state=runtime.persist,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )


def _execution_config(
    runtime,
    *,
    candidates: list[int],
    full_model: Path,
    build_fn: Callable[..., Any] | None,
    build_unavailable: str | None = None,
):
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionConfig,
    )

    return BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=list(candidates),
        candidate_search_pool=list(candidates),
        out_dir=runtime.out_dir,
        base="resnet50",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        bench_plan_runs=_part1_plan(),
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        hef_targets=["hailo8"],
        hef_part1=True,
        hef_part2=False,
        hef_backend="test",
        hef_fixup=False,
        hef_opt_level=1,
        hef_calib_count=500,
        hef_calib_bs=8,
        hailo_build_hef_fn=build_fn,
        hailo_build_unavailable=build_unavailable,
        require_complete_hailo_matrix_per_case=True,
    )


def _start_runtime(
    tmp_path: Path,
    *,
    candidates: list[int],
    resume_state_hint: dict[str, Any] | None = None,
):
    from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService

    out_dir = tmp_path / "suite"
    out_dir.mkdir()
    full_model = tmp_path / "resnet50.onnx"
    full_model.write_bytes(b"fake-full-model")
    runtime = BenchmarkGenerationService().start_generation_runtime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark_generation.log",
        requested_cases=1,
        ranked_candidates=list(candidates),
        candidate_search_pool=list(candidates),
        hef_full_policy="end",
        model_name="resnet50",
        model_source=str(full_model),
        resume_generation=resume_state_hint is not None,
        resume_state_hint=resume_state_hint,
    )
    return runtime, full_model


def test_resume_honors_persisted_candidate_search_stop(monkeypatch, tmp_path: Path) -> None:
    """A resume must not consume two more candidates after a terminal global stop."""

    _install_minimal_split_api(monkeypatch)
    candidates = [1, 2, 3]
    common = {
        "status": "rejected",
        "reason": "hailo_hef_build_unavailable",
        "stage": "matrix",
        "detail": "Hailo DFC is not installed; missing required Hailo artifacts: hailo8:part1",
        "missing_required_hailo_artifacts": [
            {"hw_arch": "hailo8", "variant": "part1"}
        ],
        "global_rejection": True,
    }
    resume_state = {
        "created_at": "2026-07-22T08:00:00",
        "case_entries": [],
        "discarded_case_entries": [
            {
                **common,
                "boundary": 1,
                "folder": "b001",
                "global_rejection_repeat_count": 1,
            },
            {
                **common,
                "boundary": 2,
                "folder": "b002",
                "global_rejection_repeat_count": 2,
                "candidate_search_stopped": True,
            },
        ],
        "errors": [
            "candidate search stopped after 2 identical global Hailo rejections: "
            "hailo_hef_build_unavailable"
        ],
    }
    runtime, full_model = _start_runtime(
        tmp_path,
        candidates=candidates,
        resume_state_hint=resume_state,
    )
    build_calls: list[str] = []

    def _unexpected_build(*_args, **kwargs):
        build_calls.append(str(kwargs.get("net_name") or ""))
        raise AssertionError("a terminal resumed candidate search must not build another HEF")

    try:
        from onnx_splitpoint_tool.benchmark.services import (
            BenchmarkGenerationExecutionService,
        )

        chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(
            _execution_config(
                runtime,
                candidates=candidates,
                full_model=full_model,
                build_fn=_unexpected_build,
            ),
            _callbacks(runtime),
        )

        assert chosen == []
        assert build_calls == []
        assert runtime.discarded_boundaries == {1, 2}
        assert not (runtime.out_dir / "b003").exists()
    finally:
        runtime.close()


def _run_failure_kind_case_loop(
    monkeypatch,
    tmp_path: Path,
    *,
    failure_kind: str,
):
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionService,
    )
    from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult

    _install_minimal_split_api(monkeypatch)
    candidates = [1, 2, 3]
    runtime, full_model = _start_runtime(tmp_path, candidates=candidates)
    build_calls: list[str] = []

    def _failed_build(_model, **kwargs):
        net_name = str(kwargs.get("net_name") or "")
        build_calls.append(net_name)
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=0.01,
            hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
            net_name=net_name,
            error=(
                "Hailo SDK is not available"
                if failure_kind == "sdk_unavailable"
                else "unsupported splitpoint for this boundary"
            ),
            failure_kind=failure_kind,
        )

    try:
        BenchmarkGenerationExecutionService().execute_case_build_loop(
            _execution_config(
                runtime,
                candidates=candidates,
                full_model=full_model,
                build_fn=_failed_build,
            ),
            _callbacks(runtime),
        )
        return runtime, build_calls
    finally:
        runtime.close()


def test_sdk_unavailable_stops_after_two_identical_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime, build_calls = _run_failure_kind_case_loop(
        monkeypatch,
        tmp_path,
        failure_kind="sdk_unavailable",
    )

    assert len(build_calls) == 2
    assert runtime.discarded_boundaries == {1, 2}
    assert runtime.discarded_cases[-1]["candidate_search_stopped"] is True
    assert not (runtime.out_dir / "b003").exists()


def test_boundary_specific_failure_does_not_trigger_global_stop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime, build_calls = _run_failure_kind_case_loop(
        monkeypatch,
        tmp_path,
        failure_kind="unsupported_splitpoint",
    )

    assert len(build_calls) == 3
    assert runtime.discarded_boundaries == {1, 2, 3}
    assert not any(
        bool(row.get("candidate_search_stopped"))
        for row in runtime.discarded_cases
    )


def test_exact_cache_canary_case_loop_attempts_only_b052(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionService,
    )
    from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult

    _install_minimal_split_api(monkeypatch)
    runtime, full_model = _start_runtime(tmp_path, candidates=[52])
    build_calls: list[str] = []

    def _cache_miss(_model, **kwargs):
        net_name = str(kwargs.get("net_name") or "")
        build_calls.append(net_name)
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=0.0,
            hw_arch="hailo8",
            net_name=net_name,
            skipped=True,
            failure_kind="cache_miss_blocked",
            unsupported_reason="cache_verify_only_policy",
            error="cache_miss_blocked[hailo_dfc]: exact b052 miss",
            last_stage="cache_lookup",
        )

    try:
        chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(
            _execution_config(
                runtime,
                candidates=[52],
                full_model=full_model,
                build_fn=_cache_miss,
            ),
            _callbacks(runtime),
        )
    finally:
        runtime.close()

    assert chosen == []
    assert build_calls == ["resnet50_part1_b52"]
    assert runtime.discarded_boundaries == {52}
    assert not (runtime.out_dir / "b039").exists()


def test_workflow_logger_allows_timeout_rejection_and_candidate_backfill(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionService,
    )
    from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
        _benchmark_service_log_adapter,
    )

    direct_messages: list[str] = []
    workflow_log = _benchmark_service_log_adapter(direct_messages.append)
    workflow_log("warning with metadata", level=30, extra={"candidate": 1})
    workflow_log("positional metadata", 30)
    assert direct_messages == ["warning with metadata", "positional metadata"]

    _install_minimal_split_api(monkeypatch)
    candidates = [1, 2, 3]
    runtime, full_model = _start_runtime(tmp_path, candidates=candidates)
    build_calls: list[str] = []
    logs: list[str] = []

    def _timeout_then_success(_model, **kwargs):
        net_name = str(kwargs.get("net_name") or "")
        hw_arch = str(kwargs.get("hw_arch") or "hailo8")
        build_calls.append(net_name)
        if net_name.endswith("_b1"):
            return HailoHefBuildResult(
                ok=False,
                elapsed_s=900.0,
                hw_arch=hw_arch,
                net_name=net_name,
                error="hard timeout after 900 seconds",
                timed_out=True,
                timeout_kind="hard",
                last_stage="allocation",
                failure_kind="timeout",
            )
        out_dir = Path(kwargs["outdir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        hef = out_dir / "compiled.hef"
        hef.write_bytes(net_name.encode("utf-8"))
        return HailoHefBuildResult(
            ok=True,
            elapsed_s=0.01,
            hw_arch=hw_arch,
            net_name=net_name,
            hef_path=str(hef),
        )

    try:
        chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(
            _execution_config(
                runtime,
                candidates=candidates,
                full_model=full_model,
                build_fn=_timeout_then_success,
            ),
            _callbacks(runtime, logs=logs),
        )

        assert chosen == [2]
        assert build_calls == ["resnet50_part1_b1", "resnet50_part1_b2"]
        assert runtime.discarded_boundaries == {1}
        assert runtime.accepted_boundaries == {2}
        assert any("b1: REJECT" in message for message in logs)
        assert (runtime.out_dir / "b002").is_dir()
    finally:
        runtime.close()


def _run_orchestration(
    monkeypatch,
    tmp_path: Path,
    *,
    full_failure: bool = False,
    builder_unavailable: bool = False,
    full_only_unavailable: bool = False,
):
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationOrchestrationConfig,
        BenchmarkGenerationOrchestrationService,
    )
    from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult

    _install_minimal_split_api(monkeypatch)
    candidates = [39, 52, 82, 91] if builder_unavailable else [39, 52]
    runtime, full_model = _start_runtime(tmp_path, candidates=candidates)
    build_calls: list[str] = []

    def _fake_build(_model, **kwargs):
        net_name = str(kwargs.get("net_name") or "")
        build_calls.append(net_name)
        if full_failure and net_name.endswith("_full"):
            return HailoHefBuildResult(
                ok=False,
                elapsed_s=0.01,
                hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
                net_name=net_name,
                error="simulated suite Full failure",
            )
        outdir = Path(kwargs["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        hef = outdir / "compiled.hef"
        hef.write_bytes(net_name.encode("utf-8"))
        return HailoHefBuildResult(
            ok=True,
            elapsed_s=0.01,
            hw_arch=str(kwargs.get("hw_arch") or "hailo8"),
            net_name=net_name,
            hef_path=str(hef),
        )

    build_fn = None if builder_unavailable else _fake_build
    build_error = "Hailo DFC is not installed" if builder_unavailable else None
    execution = _execution_config(
        runtime,
        candidates=candidates,
        full_model=full_model,
        build_fn=build_fn,
        build_unavailable=build_error,
    )
    if full_only_unavailable:
        execution.hef_part1 = False
        execution.bench_plan_runs = [
            {
                "id": "hailo8",
                "type": "hailo",
                "hw_arch": "hailo8",
                "variants": ["full"],
            }
        ]
    elif not builder_unavailable:
        execution.bench_plan_runs = [
            {
                "id": "hailo8",
                "type": "hailo",
                "hw_arch": "hailo8",
                "variants": ["full"],
            },
            *_part1_plan(),
        ]

    def _write_harness(suite_dir: str, _benchmark_name: str) -> str:
        path = Path(suite_dir) / "benchmark_suite.py"
        path.write_text("# harness\n", encoding="utf-8")
        return str(path)

    callbacks = _callbacks(runtime)
    cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=execution,
        execution_callbacks=callbacks,
        target_cases=1,
        preferred_shortlist_original=[candidates[0]],
        ranked_candidates=list(candidates),
        candidate_search_pool=list(candidates),
        out_dir=runtime.out_dir,
        base="resnet50",
        pad=3,
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        analysis_payload={},
        analysis_params_payload={},
        system_spec_payload=None,
        bench_log_path=str(runtime.bench_log_path),
        bench_plan_runs=list(execution.bench_plan_runs),
        hef_targets=["hailo8"],
        hef_full=bool(full_only_unavailable or not builder_unavailable),
        hef_part1=not full_only_unavailable,
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
        hailo_build_hef_fn=build_fn,
        hailo_build_unavailable=build_error,
        hailo_selected=True,
        write_harness_script=_write_harness,
        tool_gui_version="2.70.0",
        tool_core_version="2.70.0",
        require_complete_hailo_matrix_per_case=True,
    )
    result = BenchmarkGenerationOrchestrationService().run(cfg)
    return result, runtime, build_calls


def test_suite_full_failure_keeps_final_generation_state_incomplete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result, runtime, build_calls = _run_orchestration(
        monkeypatch,
        tmp_path,
        full_failure=True,
    )
    try:
        state = json.loads(runtime.state_path.read_text(encoding="utf-8"))

        assert build_calls == ["resnet50_part1_b39", "resnet50_full"]
        assert result.final_status == "warn"
        assert state["status"] != "complete"
    finally:
        runtime.close()


def test_two_strike_global_stop_is_not_reported_as_search_pool_exhaustion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result, runtime, _build_calls = _run_orchestration(
        monkeypatch,
        tmp_path,
        builder_unavailable=True,
    )
    try:
        summary = result.bench_payload["summary"]

        assert runtime.discarded_boundaries == {39, 52}
        assert runtime.discarded_cases[-1]["candidate_search_stopped"] is True
        assert summary["search_pool_exhausted"] is False
    finally:
        runtime.close()


def test_full_only_missing_builder_is_reported_without_case_churn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result, runtime, build_calls = _run_orchestration(
        monkeypatch,
        tmp_path,
        builder_unavailable=True,
        full_only_unavailable=True,
    )
    try:
        state = json.loads(runtime.state_path.read_text(encoding="utf-8"))
        availability = result.bench_payload["cases"][0]["hailo_case_variant_availability"]["hailo8"]

        assert build_calls == []
        assert [row["boundary"] for row in runtime.cases] == [39]
        assert runtime.discarded_cases == []
        assert result.final_status == "warn"
        assert state["status"] == "partial"
        assert availability["full"] is False
        assert availability["full_failed"] is True
        assert "Hailo DFC is not installed" in result.bench_payload["hailo"]["hefs"]["hailo8"]["full_error"]
    finally:
        runtime.close()
