from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService, BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    build_artifact_cache_preflight, render_artifact_cache_preflight_log_lines,
)


def test_generation_accepts_exact_min_gap_and_rejects_closer_candidate(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import api as asc

    exported = []
    monkeypatch.setattr(asc, "cut_tensors_for_boundary", lambda *a, **k: ["cut"])
    monkeypatch.setattr(asc, "split_model_on_cut_tensors", lambda *a, **k: (object(), object(), {"cut_tensors": ["cut"]}))
    def save(model, path):
        path = Path(path)
        exported.append(path.parent.name)
        path.write_bytes(b"selected-part")
    monkeypatch.setattr(asc, "save_model", save)
    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", lambda d, **k: str(Path(d) / "runner.py"))
    suite = tmp_path / "benchmark_set"
    suite.mkdir()
    runtime = BenchmarkGenerationRuntime(
        out_dir=suite, bench_log_path=suite / "benchmark.log",
        state_path=suite / "generation_state.json", requested_cases=2,
        ranked_candidates=[62, 64, 65], candidate_search_pool=[62, 64, 65],
        model_name="unit", model_source="unit.onnx", hef_full_policy="end",
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime, target_cases=2, gap=3, ranked_candidates=[62, 64, 65],
        candidate_search_pool=[62, 64, 65], out_dir=suite, base="unit", pad=3,
        strict_boundary=False, model=object(), nodes=[], order=[], analysis_payload={},
        bench_plan_runs=[], hef_targets=[], hef_part1=False,
    )
    logs = []
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=logs.append, queue_put=lambda *a: None, persist_state=lambda **k: None,
        publish_hailo_diagnostics=lambda *a, **k: None,
        predicted_metrics_for_boundary=lambda *a: {},
        hailo_parse_entry_for_boundary=lambda *a: None,
        hailo_parse_scalar_fields=lambda *a: {},
    )
    assert BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, callbacks) == [62, 65]
    assert set(exported) == {"b062", "b065"}
    assert "b64: skip (min_gap)" in logs
    assert not (suite / "b064").exists()


def _probe(case, *, ready=False, expectation="unspecified"):
    return {
        "model_id": "yolo26m", "role": "hailo10h", "item_id": case + ":part1",
        "status": "MISS", "expectation": expectation,
        "reason": "built_during_preparation_artifact_now_ready" if ready else "exact_cache_artifact_missing",
        "evidence": {"artifact_now_ready": ready, "current_build": ready},
    }


def test_preparation_build_is_history_not_another_expected_compiler_start():
    report = build_artifact_cache_preflight(
        model_ids=["yolo26m"],
        observations=[_probe("b398", ready=True), _probe("b399")],
        applicable_roles={"yolo26m": ["hailo10h"]},
    )
    assert report["confirmed_miss_count"] == 2
    assert report["cold_builds_required"] == report["completed_cold_build_count"] == 1
    assert [row["boundary"] for row in report["cold_build_rows"]] == ["b399"]
    assert report["expected_cold_builds"] == 0  # Separate historical policy counter.
    lines = render_artifact_cache_preflight_log_lines(report)
    assert any("completed_cold_build: model=yolo26m boundary=b398" in line for line in lines)
    assert any("expected_cold_build: model=yolo26m boundary=b399" in line for line in lines)
    assert not any("expected_cold_build:" in line and "boundary=b398 " in line for line in lines)


def test_completed_preparation_build_does_not_bypass_strict_warm_policy():
    report = build_artifact_cache_preflight(
        model_ids=["yolo26m"], observations=[_probe("b398", ready=True, expectation="warm")],
        applicable_roles={"yolo26m": ["hailo10h"]},
        block_on_unexpected_cold_builds=True,
    )
    assert report["cold_builds_required"] == 0
    assert report["unexpected_cold_builds"] == 1
    assert report["runtime_dispatch_allowed"] is False
    assert report["artifact_matrix"][0]["compiler_dispatch_allowed"] is False


def test_all_completed_cold_builds_are_reported_as_completed():
    report = build_artifact_cache_preflight(
        model_ids=["yolo26m"], observations=[_probe("b398", ready=True)],
        applicable_roles={"yolo26m": ["hailo10h"]},
    )
    assert report["status"] == "cold_builds_completed"
    assert report["cold_build_rows"] == []
    assert report["hit_count"] == 0
    assert report["runtime_dispatch_allowed"] is True
