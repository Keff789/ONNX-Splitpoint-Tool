"""A4 and historical S1 evidence must survive ordinary reporting separately."""
import json

from onnx_splitpoint_tool.native_performance_reporting import (
    _deepx_runtime_fields, _detail_for, collect_native_performance_matrix,
)
from onnx_splitpoint_tool.native_energy_reporting import build_native_energy_pairs


def _row(mode=None):
    row = dict(model="resnet50", backend="deepx_to_trt", case="b045",
               precision="float32_layout_fp16", setup_id="deepx", ok=True,
               status="ok", fps_makespan=123.0)
    if mode:
        row["native_command_contract"] = {"runtime_options": {
            "deepx_runtime_mode": mode, "deepx_max_inflight": 4 if mode == "A4" else 1,
        }}
    return row


def test_detail_selection_never_uses_other_runtime_cohort():
    old, new = _row(), _row("A4")
    assert _detail_for(old, [new]) == {}
    assert _detail_for(new, [old]) == {}
    assert _detail_for(old, [new, old]) == old
    assert _detail_for(new, [old, new]) == new


def test_full_nested_contract_and_actual_pending_are_preserved():
    row = dict(_row("A4"), backend="native_full_deepx", case="full", actual_pending_max=3)
    row["full_command_contract"] = row.pop("native_command_contract")
    assert _deepx_runtime_fields(row) == {
        "deepx_runtime_mode": "A4", "deepx_max_inflight": 4, "host_pending_max": 3,
    }
    legacy = dict(row)
    legacy.pop("full_command_contract")
    assert _detail_for(legacy, [row]) == {}


def test_ordinary_matrix_retains_both_modes_without_relabelling_history(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    old, new = _row(), _row("A4")
    (reports / "native_producer_summary.json").write_text(json.dumps({"rows": [old, new]}))
    matrix = collect_native_performance_matrix(tmp_path)
    assert len(matrix["observations"]) == 2
    assert sum(r.get("deepx_runtime_mode") == "A4" for r in matrix["observations"]) == 1
    assert sum("deepx_runtime_mode" not in r for r in matrix["observations"]) == 1
    assert _deepx_runtime_fields(dict(new, backend="hailo8_to_trt")) == {}


def test_a4_split_never_uses_historical_s1_vendor_energy_baseline():
    split = dict(_row(), execution_mode="native_split", deepx_runtime_mode="A4")
    full = dict(split, backend="native_full_deepx", execution_mode="native_full_baseline", case="full")
    full.pop("deepx_runtime_mode")
    pair = next(p for p in build_native_energy_pairs([split, full])
                if p["baseline_backend"] == "native_full_deepx")
    assert not pair["comparable"]
    assert "deepx_runtime_mode_mismatch" in pair["screening_comparison_reasons"]
    assert "deepx_runtime_mode_mismatch" in pair["descriptive_comparison_reasons"]
    full["deepx_runtime_mode"] = "A4"
    pair = next(p for p in build_native_energy_pairs([split, full])
                if p["baseline_backend"] == "native_full_deepx")
    assert "deepx_runtime_mode_mismatch" not in str(pair)
    historical = dict(full)
    historical.pop("deepx_runtime_mode")
    pair = next(p for p in build_native_energy_pairs([split, historical, full])
                if p["baseline_backend"] == "native_full_deepx")
    assert "baseline_ambiguous" not in str(pair)
    assert "deepx_runtime_mode_mismatch" not in str(pair)
