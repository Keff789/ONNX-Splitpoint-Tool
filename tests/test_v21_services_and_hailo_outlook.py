from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationService,
    RemoteBenchmarkService,
)


def _analysis_payload() -> dict:
    return {
        "scores": {"10": 1.2, "20": 0.8, "30": 0.9},
        "costs_bytes": [0] * 40,
        "peak_act_mem_right_bytes": [0] * 40,
        "crossing_counts_all": [0] * 40,
        "flops_left_prefix": [0.0] * 40,
        "total_flops": 100.0,
        "strict_ok": [True] * 40,
    }


def test_benchmark_generation_service_hailo_outlook_and_plan():
    service = BenchmarkGenerationService()
    analysis = _analysis_payload()
    analysis["costs_bytes"][10] = int(2.0 * 1024 * 1024)
    analysis["peak_act_mem_right_bytes"][10] = int(8.0 * 1024 * 1024)
    analysis["crossing_counts_all"][10] = 2
    analysis["flops_left_prefix"][10] = 75.0

    analysis["costs_bytes"][20] = int(0.4 * 1024 * 1024)
    analysis["peak_act_mem_right_bytes"][20] = int(1.0 * 1024 * 1024)
    analysis["crossing_counts_all"][20] = 1
    analysis["flops_left_prefix"][20] = 55.0

    analysis["costs_bytes"][30] = int(3.0 * 1024 * 1024)
    analysis["peak_act_mem_right_bytes"][30] = int(12.0 * 1024 * 1024)
    analysis["crossing_counts_all"][30] = 4
    analysis["flops_left_prefix"][30] = 20.0

    rows, summary, meta = service.build_hailo_outlook(analysis, [10, 20, 30], top_n=3)
    assert summary is not None
    assert rows
    assert rows[0].boundary in {10, 20, 30}
    assert meta

    plan = service.prepare_generation_plan(
        analysis,
        ranked_candidates=[10, 20, 30],
        candidate_search_pool=[10, 20, 30],
        requested_cases=5,
        hailo_selected=True,
    )
    assert plan.requested_cases == 3
    assert plan.hailo_compile_rank_meta
    assert plan.hailo_outlook_summary is not None
    assert len(plan.ranked_candidates) == 3


def test_remote_benchmark_service_rebuild_cached_suite_bundle(tmp_path: Path):
    service = RemoteBenchmarkService()
    dist = tmp_path / "suite" / "dist"
    dist.mkdir(parents=True)
    for name in (
        "suite_bundle.tar.gz",
        "suite_bundle.tar.gz.manifest.json",
        "suite_bundle.tar.gz.tmp",
    ):
        (dist / name).write_text("x", encoding="utf-8")

    dist_dir, removed = service.rebuild_cached_suite_bundle(tmp_path / "suite")
    assert dist_dir == dist
    assert "suite_bundle.tar.gz" in removed
    assert not (dist / "suite_bundle.tar.gz").exists()


def test_benchmark_generation_runtime_and_skip_label(tmp_path: Path):
    service = BenchmarkGenerationService()
    runtime = service.start_generation_runtime(
        out_dir=tmp_path / "suite",
        bench_log_path=tmp_path / "suite" / "benchmark_generation.log",
        requested_cases=3,
        ranked_candidates=[10, 20],
        candidate_search_pool=[10, 20, 30],
        hef_full_policy="end",
        model_name="demo",
        model_source="/tmp/demo.onnx",
    )
    runtime.log("hello", queue_put=None)
    runtime.persist(status="running", current_boundary=10)
    runtime.close()
    assert (tmp_path / "suite" / "generation_state.json").exists()
    label = service.format_benchmark_case_label({
        "boundary": 42,
        "hw_arch": "hailo8",
        "likely_original_inputs": ["images"],
    })
    assert "b42" in label and "images" in label
