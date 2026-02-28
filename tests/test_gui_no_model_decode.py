from pathlib import Path


def test_gui_contains_no_model_specific_decode_keywords():
    """The GUI should not contain model-specific decoding logic.

    This is a lightweight, grep-based regression test. If the GUI needs to
    reference these terms for UI labels in the future, update the allowlist.
    """

    repo_root = Path(__file__).resolve().parents[1]
    gui_dir = repo_root / "onnx_splitpoint_tool" / "gui"
    assert gui_dir.is_dir(), "GUI directory not found"

    forbidden = [
        "decode_yolo",
        "nms",
        "sigmoid",
        "xywh",
        "xyxy",
        "non-maximum",
        "imagenet_labels",
        "ultralytics",
    ]

    for py in gui_dir.rglob("*.py"):
        txt = py.read_text(encoding="utf-8", errors="ignore").lower()
        for token in forbidden:
            assert token not in txt, f"Found forbidden token '{token}' in GUI file: {py}"
