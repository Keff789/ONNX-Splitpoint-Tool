from __future__ import annotations

from onnx_splitpoint_tool.objective_scoring import (
    candidate_objective_metrics,
    hailo_feasibility_risk,
    hailo_interface_penalty,
    objective_sort_key,
    predicted_handover_ms,
    predicted_stream_fps,
)


def test_objective_scoring_components_are_separated() -> None:
    feas = hailo_feasibility_risk(
        compile_risk_score=2.4,
        single_context_probability=0.40,
        fallback_used=True,
        parse_ok=True,
    )
    iface = hailo_interface_penalty(
        cut_mib=6.0,
        n_cut_tensors=30,
        unknown_crossing_tensors=2,
        peak_act_right_mib=8.0,
        stage1="hailo8",
        stage2="tensorrt",
    )
    hand = predicted_handover_ms(
        cut_mib=6.0,
        n_cut_tensors=30,
        unknown_crossing_tensors=2,
        peak_act_right_mib=8.0,
        compile_risk_score=2.4,
        single_context_probability=0.40,
        fallback_used=True,
        stage1="hailo8",
        stage2="tensorrt",
    )
    fps = predicted_stream_fps(bottleneck_ms=20.0, handover_ms=hand)
    assert feas is not None and iface is not None and hand is not None and fps is not None
    assert hand > iface
    assert fps > 0.0


def test_throughput_objective_prefers_better_stream_proxy() -> None:
    row_good = {
        "boundary": 10,
        "cut_mb_val": 2.0,
        "n_cut_tensors": 8,
        "unknown_count": 0,
        "peak_right_mib_val": 3.0,
        "hailo_compile_risk_score": 1.2,
        "hailo_single_context_probability": 0.95,
        "pred_latency_total_ms": 8.0,
        "flops_left_abs": 10.0,
        "flops_right_abs": 10.0,
        "score_pred": 0.5,
        "source_rank": 2,
    }
    row_bad = {
        "boundary": 20,
        "cut_mb_val": 6.0,
        "n_cut_tensors": 40,
        "unknown_count": 3,
        "peak_right_mib_val": 12.0,
        "hailo_compile_risk_score": 2.8,
        "hailo_single_context_probability": 0.30,
        "pred_latency_total_ms": 10.0,
        "flops_left_abs": 10.0,
        "flops_right_abs": 10.0,
        "score_pred": 0.3,
        "source_rank": 1,
    }
    m = candidate_objective_metrics(row_good, stage1="hailo8", stage2="tensorrt")
    assert m["predicted_stream_fps"] is not None
    assert m["hailo_feasibility_risk"] is not None
    assert m["hailo_interface_penalty"] is not None
    assert objective_sort_key(row_good, objective="Throughput", stage1="hailo8", stage2="tensorrt") < objective_sort_key(row_bad, objective="Throughput", stage1="hailo8", stage2="tensorrt")
    assert objective_sort_key(row_good, objective="Hailo feasibility", stage1="hailo8", stage2="tensorrt") < objective_sort_key(row_bad, objective="Hailo feasibility", stage1="hailo8", stage2="tensorrt")
