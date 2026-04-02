from __future__ import annotations

import numpy as np

from onnx_splitpoint_tool.runners.harness.yolo import YoloHarness, _detect_yolo_format


def _make_split_heads():
    outs = []
    names = []
    for idx, hw in enumerate((80, 40, 20), start=1):
        reg = np.full((1, 64, hw, hw), -8.0, dtype=np.float32)
        cls = np.full((1, 80, hw, hw), -20.0, dtype=np.float32)
        # Create a single confident class location.
        cls[0, 0, 0, 0] = 12.0
        outs.extend([reg, cls])
        names.extend([f"reg{idx}", f"cls{idx}"])
    return names, outs


def test_detect_yolo11n_style_split_heads_as_ultralytics_regcls():
    names, outs = _make_split_heads()
    assert _detect_yolo_format(outs, output_names=names) == "ultralytics_regcls"


def test_yolo_harness_postprocess_accepts_split_regcls_heads():
    names, outs = _make_split_heads()
    harness = YoloHarness(conf_thresh=0.9, iou_thresh=0.45, max_det=50)
    payload = {name: out for name, out in zip(names, outs)}
    pp = harness.postprocess(payload, context={"input_hw": (640, 640)})
    assert pp.task == "detection"
    assert isinstance(pp.json, dict)
    assert pp.json.get("format") == "ultralytics_regcls"
    assert pp.json.get("n_detections", 0) >= 1
