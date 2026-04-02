from __future__ import annotations

from pathlib import Path


TEMPLATE = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt")


def test_template_detects_detr_like_outputs_for_validation() -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert 'if _is_detr_like(output_names, outputs):' in src
    assert 'return "detr_like"' in src


def test_template_auto_mode_uses_proxy_for_detr_like() -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert 'detection_like = output_format in ("multiscale_head", "ultralytics_regcls", "bn6_detections", "detr_like")' in src
    assert 'if output_format in ("bn6_detections", "ultralytics_regcls", "detr_like"):' in src


def test_template_extracts_detr_proxy_detections_via_decode_detr() -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert 'if output_format == "detr_like":' in src
    assert 'detr_proxy_decode_exception' in src
    assert 'dets = _decode_detr(' in src


def test_template_keeps_detr_visualization_off_yolo_harness_path() -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert 'elif output_format == "detr_like":' in src
    assert "report['viz'] = detr_info" in src
