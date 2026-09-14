from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/native_detection_completion_tail_canary.py"
SPEC = importlib.util.spec_from_file_location("completion_tail_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_primary_and_remote_canary_sources_are_byte_identical() -> None:
    remote = (
        ROOT
        / "onnx_splitpoint_tool"
        / "resources"
        / "remote_scripts"
        / "native_detection_completion_tail_canary.py"
    )
    assert SCRIPT.read_bytes() == remote.read_bytes()


def test_percentile_and_parity_tolerance() -> None:
    assert MODULE._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    measured = {
        "content_sha256": "different",
        "detections": [
            {"class_id": 1, "score": 0.5, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}
        ],
    }
    reference = {
        "completion_content_sha256": "reference",
        "completed_task_result_artifact": {
            "detections": [
                {"class_id": 1, "score": 0.5000001, "x1": 1.0001, "y1": 2.0, "x2": 3.0, "y2": 4.0}
            ]
        },
    }
    parity = MODULE._parity(measured, reference)
    assert parity["exact_content_sha256"] is False
    assert parity["tolerance_pass"] is True
    assert parity["status"] == "tolerance_pass"
