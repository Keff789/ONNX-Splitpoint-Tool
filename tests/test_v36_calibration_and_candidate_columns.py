from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.objective_scoring import (
    THROUGHPUT_CALIBRATION_PROFILE_NAME,
    active_throughput_calibration_profile,
    predicted_handover_ms,
)


def test_active_calibration_profile_is_embedded_default() -> None:
    assert active_throughput_calibration_profile() == THROUGHPUT_CALIBRATION_PROFILE_NAME


def test_calibrated_handover_responds_to_direction_and_imbalance() -> None:
    common = dict(
        cut_mib=5.5,
        n_cut_tensors=3,
        unknown_crossing_tensors=0,
        peak_act_right_mib=22.0,
        compile_risk_score=2.1,
        single_context_probability=0.50,
        fallback_used=False,
        parse_ok=True,
    )
    balanced = predicted_handover_ms(stage1="hailo8", stage2="tensorrt", imbalance=0.23, **common)
    imbalanced = predicted_handover_ms(stage1="hailo8", stage2="tensorrt", imbalance=0.60, **common)
    reverse = predicted_handover_ms(stage1="tensorrt", stage2="hailo8", imbalance=0.60, **common)
    assert balanced is not None and imbalanced is not None and reverse is not None
    assert imbalanced > balanced
    assert reverse < imbalanced


def test_gui_candidate_tree_declares_objective_columns() -> None:
    src = Path("onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8")
    assert '"predicted_stream_fps"' in src
    assert '"hailo_feasibility_risk"' in src
    assert '"hailo_interface_penalty"' in src
