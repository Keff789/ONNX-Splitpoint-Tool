from __future__ import annotations

from pathlib import Path


def test_template_understands_hailo_multiscale_layouts() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'def _canonicalize_yolo_multiscale_output' in src
    assert 'Unsupported multiscale head output rank' in src
    assert 'a.ndim == 3' in src
    assert 'a.reshape(1, gh, gw, na_hint, ch // na_hint).transpose(0, 3, 1, 2, 4)' in src


def test_template_reports_no_shape_matched_outputs_honestly() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'shape_mismatch_outputs' in src
    assert '"reason": ("no shape-matched outputs" if no_shape_match else None)' in src
    assert 'float(max_abs) if compared > 0 else None' in src
    assert 'int(elem.get("compared_outputs_shape_matched") or n_total)' in src


def test_template_marks_layout_mismatch_honestly() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'comp_status = "skipped_layout_mismatch" if no_shape_match else "ok"' in src
    assert '"reason": comp_reason' in src
    assert 'c.get("status") not in ("ok", "skipped_layout_mismatch")' in src


def test_template_records_viz_provenance_per_variant() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'def _finalize_viz_result' in src
    assert '"viz_enabled"' in src
    assert '"viz_error"' in src
    assert '"detections_generated_from_variant"' in src
    assert '"n_detections"' in src


def test_template_primary_variant_is_requested_not_best_survivor() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'for candidate in ("composed", "full", "part2", "part1")' in src
    assert 'if candidate in variants:' in src
    assert 'prim_status = str(prim.get("status") or "missing")' in src
    assert 'final_pass = bool(prim.get("passed")) if prim_status in ("ok", "skipped_layout_mismatch") else False' in src


def test_template_missing_variant_errors_are_not_soft_skips() -> None:
    src = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'st = "error"' in src
    assert 'st = "skipped" if variant_status.get(v) == "skipped" else "error"' not in src
