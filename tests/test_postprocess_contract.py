import json
from pathlib import Path

import numpy as np

from onnx_splitpoint_tool.runners.harness.base import postprocess_result_to_dict
from onnx_splitpoint_tool.runners.harness.classification import ClassificationHarness
from onnx_splitpoint_tool.runners.harness.yolo import YoloHarness


def _write_dummy_rgb(path: Path, w: int = 320, h: int = 240) -> None:
    try:
        from PIL import Image

        im = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        im.save(path)
    except Exception as e:
        raise RuntimeError("PIL is required for these tests") from e


def test_classification_postprocess_contract_and_files(tmp_path: Path):
    img = tmp_path / "img.png"
    _write_dummy_rgb(img)

    h = ClassificationHarness(topk=3, labels=["a", "b", "c", "d"])
    logits = np.array([0.0, 10.0, -1.0, 9.0], dtype=np.float32)

    res = postprocess_result_to_dict(
        h.postprocess({"logits": logits}, context={"output_dir": tmp_path, "variant": "full", "image_path": img})
    )
    assert res["task"] == "classification"
    assert "topk" in res["json"]
    assert isinstance(res["json"]["topk"], list)

    # JSON serializable
    json.dumps(res)

    # Files exist
    assert (tmp_path / "classification_full.json").is_file()
    assert (tmp_path / "classification_full.png").is_file()


def test_yolo_postprocess_bn6_contract_and_files(tmp_path: Path):
    img = tmp_path / "img.png"
    _write_dummy_rgb(img, w=640, h=480)

    h = YoloHarness(conf_thresh=0.1, iou_thresh=0.5)
    # One box: x1,y1,x2,y2,score,class
    det = np.array([[10, 20, 110, 120, 0.9, 0]], dtype=np.float32)

    res = postprocess_result_to_dict(
        h.postprocess(
            {"output": det},
            context={"output_dir": tmp_path, "variant": "full", "image_path": img, "input_hw": (640, 640)},
        )
    )
    assert res["task"] == "detection"
    assert res["json"]["task"] == "detection"
    assert isinstance(res["json"].get("detections"), list)
    json.dumps(res)

    assert (tmp_path / "detections_full.json").is_file()
    assert (tmp_path / "detections_full.png").is_file()


def test_yolo_postprocess_multiscale_head_does_not_crash(tmp_path: Path):
    img = tmp_path / "img.png"
    _write_dummy_rgb(img, w=640, h=640)

    h = YoloHarness(conf_thresh=0.1, iou_thresh=0.5)

    # Typical YOLOv5/7: 3 scales, each [1,3,gh,gw,85]
    p3 = np.zeros((1, 3, 80, 80, 85), dtype=np.float32)
    p4 = np.zeros((1, 3, 40, 40, 85), dtype=np.float32)
    p5 = np.zeros((1, 3, 20, 20, 85), dtype=np.float32)

    # Inject one confident prediction at p3[anchor0, cell(0,0)]
    p3[0, 0, 0, 0, 0:4] = 0.0  # tx,ty,tw,th
    p3[0, 0, 0, 0, 4] = 10.0  # obj
    p3[0, 0, 0, 0, 5] = 10.0  # class 0

    res = postprocess_result_to_dict(
        h.postprocess(
            {"p3": p3, "p4": p4, "p5": p5},
            context={"output_dir": tmp_path, "variant": "full", "image_path": img, "input_hw": (640, 640)},
        )
    )
    assert res["task"] == "detection"
    assert len(res["json"].get("detections", [])) >= 1
    assert (tmp_path / "detections_full.json").is_file()
    assert (tmp_path / "detections_full.png").is_file()


def test_yolo_postprocess_letterbox_rescale(tmp_path):
    """Boxes should be mapped back correctly when YOLO preprocessing uses letterbox."""
    from PIL import Image
    import numpy as np

    from onnx_splitpoint_tool.runners.harness.yolo import YoloHarness

    # Original image is non-square -> letterbox adds vertical padding for 640x640.
    img_path = tmp_path / "im.png"
    Image.new("RGB", (640, 480), (255, 255, 255)).save(img_path)

    # Letterbox parameters for (640x480) -> (640x640): gain=1.0, pad_top=80, pad_left=0
    x1, y1, x2, y2 = 100.0, 50.0, 200.0, 150.0
    det = np.array([[x1, y1 + 80.0, x2, y2 + 80.0, 0.9, 1.0]], dtype=np.float32)

    h = YoloHarness()
    ctx = {
        "schema_version": 2,
        "image_path": str(img_path),
        "input_hw": [640, 640],
        "out_names": ["output0"],
        "out_format": "bn6",
    }
    res = h.postprocess({"output0": det}, context=ctx)

    assert res.task == "detection"
    assert res.json is not None
    assert "detections" in res.json and len(res.json["detections"]) == 1

    box = res.json["detections"][0]
    assert abs(box["x1"] - x1) < 1e-3
    assert abs(box["y1"] - y1) < 1e-3
    assert abs(box["x2"] - x2) < 1e-3
    assert abs(box["y2"] - y2) < 1e-3


