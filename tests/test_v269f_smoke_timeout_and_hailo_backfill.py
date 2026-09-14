from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.hailo_policy import (
    case_has_usable_hailo_variant,
    case_satisfies_all_hailo_requirements,
    missing_case_hailo_requirements,
)
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationRuntime,
    _global_hailo_rejection_signature,
    _strict_hailo_matrix_case_rejection,
)
from onnx_splitpoint_tool.run_modes import RUN_MODE_SCHEMA_VERSION, default_run_modes_config, validate_run_modes_config
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row


ROOT = Path(__file__).resolve().parents[1]


def _evaluation_runs() -> list[dict]:
    return [
        {
            "id": "hailo8",
            "type": "hailo",
            "hw_arch": "hailo8",
            "variants": ["full"],
        },
        {
            "id": "hailo10",
            "type": "hailo",
            "hw_arch": "hailo10",
            "variants": ["full"],
        },
        {
            "id": "hailo8_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["full", "part1", "part2", "composed"],
        },
        {
            "id": "hailo10_to_tensorrt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo10"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["full", "part1", "part2", "composed"],
        },
    ]


def test_partial_case_is_usable_for_exploration_but_not_complete_evaluation_matrix() -> None:
    # Mirrors YOLOv7 b216: both suite-level Full HEFs exist, while both
    # case-specific Hailo Part1 builds failed.
    availability = {
        "hailo8": {"full": True, "part1": False, "part2": False, "composed": False},
        "hailo10": {"full": True, "part1": False, "part2": False, "composed": False},
    }

    assert case_has_usable_hailo_variant(_evaluation_runs(), availability) is True
    assert case_satisfies_all_hailo_requirements(_evaluation_runs(), availability) is False


def test_complete_case_satisfies_every_requested_hailo_requirement() -> None:
    availability = {
        "hailo8": {"full": True, "part1": True, "part2": False, "composed": False},
        "hailo10": {"full": True, "part1": True, "part2": False, "composed": False},
    }

    assert case_satisfies_all_hailo_requirements(_evaluation_runs(), availability) is True


def test_strict_full_only_missing_is_suite_global_not_case_rejection() -> None:
    runs = [{
        "id": "hailo8",
        "type": "hailo",
        "hw_arch": "hailo8",
        "variants": ["full"],
    }]
    availability = {
        "hailo8": {
            "full": False,
            "part1": False,
            "part2": False,
            "composed": False,
            "full_failed": True,
        },
    }

    rejection, missing = _strict_hailo_matrix_case_rejection(
        boundary=216,
        folder="b216",
        bench_plan_runs=runs,
        case_variant_availability=availability,
    )

    assert missing == []
    assert rejection is None


def test_strict_builder_unavailable_rejects_missing_part1_even_with_full() -> None:
    runs = [
        {
            "id": "hailo8",
            "type": "hailo",
            "hw_arch": "hailo8",
            "variants": ["full"],
        },
        {
            "id": "hailo8_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "composed"],
        },
    ]
    availability = {
        "hailo8": {
            "full": True,
            "part1": False,
            "part2": False,
            "composed": False,
        },
    }

    rejection, missing = _strict_hailo_matrix_case_rejection(
        boundary=216,
        folder="b216",
        bench_plan_runs=runs,
        case_variant_availability=availability,
        builder_error="Hailo HEF build unavailable",
    )

    assert missing == [("hailo8", "part1")]
    assert rejection is not None
    assert rejection["reason"] == "hailo_hef_build_unavailable"
    assert "Hailo HEF build unavailable" in rejection["detail"]
    assert rejection["missing_required_hailo_artifacts"] == [
        {"hw_arch": "hailo8", "variant": "part1"},
    ]


def test_strict_requirement_check_normalizes_physical_architecture_aliases() -> None:
    runs = [{
        "id": "hailo10",
        "type": "hailo",
        "hw_arch": "hailo10h",
        "variants": ["full"],
    }]

    assert missing_case_hailo_requirements(
        runs,
        {"HAILO10H": {"full": True}},
    ) == []


def test_strict_matrix_enriches_actual_part1_failure_before_backfill() -> None:
    availability = {
        "hailo8": {"full": True, "part1": False},
        "hailo10": {"full": True, "part1": False},
    }
    first = {
        "status": "rejected",
        "boundary": 216,
        "folder": "b216",
        "reason": "hailo_hef_build_failed",
        "stage": "part1",
        "hw_arch": "hailo8",
    }

    rejection, missing = _strict_hailo_matrix_case_rejection(
        boundary=216,
        folder="b216",
        bench_plan_runs=_evaluation_runs(),
        case_variant_availability=availability,
        first_rejection=first,
    )

    assert missing == [("hailo10h", "part1"), ("hailo8", "part1")]
    assert rejection is not None
    assert rejection["reason"] == "hailo_hef_build_failed"
    assert rejection["missing_required_hailo_artifacts"] == [
        {"hw_arch": "hailo10h", "variant": "part1"},
        {"hw_arch": "hailo8", "variant": "part1"},
    ]


def test_strict_case_backfill_is_opt_in_and_formal_audit_disables_it() -> None:
    assert BenchmarkGenerationExecutionConfig.__dataclass_fields__[
        "require_complete_hailo_matrix_per_case"
    ].default is False
    assert BenchmarkGenerationOrchestrationConfig.__dataclass_fields__[
        "require_complete_hailo_matrix_per_case"
    ].default is False
    binding = (ROOT / "onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py").read_text(encoding="utf-8")
    assert "_require_complete_hailo_matrix_for_candidate_plan(candidate_plan)" in binding
    assert "require_complete_hailo_matrix_per_case=(" in binding


def test_actual_case_loop_accepts_first_case_when_only_suite_full_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A model-wide Full HEF must not cause split-boundary backfill."""
    from onnx_splitpoint_tool import api as asc

    monkeypatch.setattr(asc, "cut_tensors_for_boundary", lambda *_a, **_k: ["cut"])
    monkeypatch.setattr(
        asc,
        "split_model_on_cut_tensors",
        lambda *_a, **_k: (object(), object(), {"cut_tensors": ["cut"]}),
    )

    def _save_model(_model, path) -> None:
        Path(path).write_bytes(b"fake-onnx")

    monkeypatch.setattr(asc, "save_model", _save_model)

    def _write_runner(case_dir, **_kwargs):
        runner = Path(case_dir) / "run_benchmark.py"
        runner.write_text("# generated test runner\n", encoding="utf-8")
        return str(runner)

    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", _write_runner)

    out_dir = tmp_path / "suite"
    out_dir.mkdir()
    runtime = BenchmarkGenerationRuntime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark.log",
        state_path=out_dir / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[1, 2],
        candidate_search_pool=[1, 2],
        model_name="unit-model",
        model_source="unit-model.onnx",
        hef_full_policy="start",
        suite_hailo_hefs={},
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[1, 2],
        candidate_search_pool=[1, 2],
        out_dir=out_dir,
        base="unit_model",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        bench_plan_runs=[{
            "id": "hailo8",
            "type": "hailo",
            "hw_arch": "hailo8",
            "variants": ["full"],
        }],
        full_model_src="unit-model.onnx",
        hef_targets=["hailo8"],
        hef_part1=False,
        hef_part2=False,
        require_complete_hailo_matrix_per_case=True,
    )
    events: list[tuple] = []
    cb = BenchmarkGenerationExecutionCallbacks(
        log=lambda _msg, **_kwargs: None,
        queue_put=events.append,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )

    chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, cb)

    assert chosen == [1]
    assert [row["boundary"] for row in runtime.cases] == [1]
    assert runtime.discarded_boundaries == set()
    assert runtime.discarded_cases == []
    assert not (out_dir / "b002").exists()
    assert any(len(event) >= 3 and event[2] == "b1" for event in events)


def test_repeated_global_builder_unavailable_rejection_stops_candidate_search(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from onnx_splitpoint_tool import api as asc

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
    runtime = BenchmarkGenerationRuntime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark.log",
        state_path=out_dir / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[1, 2, 3],
        candidate_search_pool=[1, 2, 3],
        model_name="unit-model",
        model_source="unit-model.onnx",
        hef_full_policy="end",
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[1, 2, 3],
        candidate_search_pool=[1, 2, 3],
        out_dir=out_dir,
        base="unit_model",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        bench_plan_runs=[{
            "id": "hailo8_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "composed"],
        }],
        full_model_src="unit-model.onnx",
        hef_targets=["hailo8"],
        hef_part1=True,
        hef_part2=False,
        hailo_build_hef_fn=None,
        hailo_build_unavailable="Hailo DFC is not installed",
        require_complete_hailo_matrix_per_case=True,
    )
    logs: list[str] = []
    events: list[tuple] = []
    cb = BenchmarkGenerationExecutionCallbacks(
        log=lambda msg, **_kwargs: logs.append(str(msg)),
        queue_put=events.append,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )

    chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, cb)

    assert chosen == []
    assert runtime.discarded_boundaries == {1, 2}
    assert not (out_dir / "b003").exists()
    assert all(row["reason"] == "hailo_hef_build_unavailable" for row in runtime.discarded_cases)
    assert runtime.discarded_cases[-1]["candidate_search_stopped"] is True
    assert any("REJECT reason=hailo_hef_build_unavailable" in row for row in logs)
    assert any("candidate search stopped after 2 identical global Hailo rejections" in row for row in logs)
    assert any("reject: hailo_hef_build_unavailable" in str(event[2]) for event in events if len(event) >= 3)


def test_only_boundary_independent_rejections_receive_global_signature() -> None:
    global_rejection = {
        "reason": "hailo_hef_build_unavailable",
        "detail": "Hailo DFC is not installed",
        "missing_required_hailo_artifacts": [{"hw_arch": "hailo8", "variant": "part1"}],
    }
    candidate_rejection = {
        "reason": "hailo_hef_build_failed",
        "detail": "allocator failed for this graph",
    }

    assert _global_hailo_rejection_signature(global_rejection)
    assert _global_hailo_rejection_signature(candidate_rejection) == ""


def test_schema_v9_smoke_timeout_is_3600_and_migration_is_value_sensitive() -> None:
    assert RUN_MODE_SCHEMA_VERSION >= 9
    assert default_run_modes_config()["schema_version"] == RUN_MODE_SCHEMA_VERSION
    assert default_run_modes_config()["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] == 3600

    old = default_run_modes_config()
    old["schema_version"] = 8
    old["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] = 1800
    assert validate_run_modes_config(old)["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] == 3600

    old["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] = 2700
    assert validate_run_modes_config(old)["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] == 2700


def test_partial_validation_report_keeps_logical_dispatch_run_id(
    tmp_path: Path,
) -> None:
    report = (
        tmp_path
        / "remote_diagnostics/case_reports/results/b038/results_ort_tensorrt"
        / "validation_report.json"
    )
    report.parent.mkdir(parents=True)
    raw = {
        "case_id": "b038",
        "run_id": "tensorrt",
        "backend": "tensorrt",
        "variant": "split",
        "primary_variant": "composed",
        "total_latency_ms": 1.0,
        "runtime_ok": True,
    }
    report.write_text(json.dumps(raw), encoding="utf-8")

    row = normalize_benchmark_row(
        raw, model_id="yolo26s", source_path=report, tag="validation_report",
    )

    assert row["run_id"] == "ort_tensorrt"
    assert row["reported_run_id"] == "tensorrt"
