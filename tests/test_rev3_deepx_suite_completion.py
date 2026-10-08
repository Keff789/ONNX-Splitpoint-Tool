"""The ordinary generated DeepX suite uses timed completion + postflight proof."""
from types import SimpleNamespace
from PIL import Image
import pytest

from test_v27926_deepx_full_decoded_pre_nms import (
    case, _suite, _install_runtime, _decoded_outputs,
)
import onnx_splitpoint_tool.native_detection_postprocess as ndp


@pytest.mark.parametrize("mode", ["auto", "strict"])
def test_suite_keeps_decode_timed_and_finalizes_after_last_task(case, monkeypatch, mode):
    suite = _suite()
    calls = _install_runtime(monkeypatch, _decoded_outputs())
    image = case.root / "image.png"
    Image.new("RGB", (640, 640), "black").save(image)
    results = case.root / "results"
    results.mkdir()
    monkeypatch.setattr(suite, "_deepx_find_prepared_feed_image", lambda *a: (image, "fixture"))
    observed = []
    original_task = ndp.FrozenDetectionPostprocessor.process_task
    original_final = ndp.FrozenDetectionPostprocessor.finalize_evidence

    def task(self, outputs, **kwargs):
        result = original_task(self, outputs, **kwargs)
        observed.append(("task", len(calls), self.completed_count))
        assert result["detection_count"] == 2
        return result

    def final(self):
        observed.append(("final", len(calls), self.completed_count))
        assert len(calls) == 5 and self.completed_count == 4
        return original_final(self)

    monkeypatch.setattr(ndp.FrozenDetectionPostprocessor, "process_task", task)
    monkeypatch.setattr(ndp.FrozenDetectionPostprocessor, "finalize_evidence", final)
    result = suite._run_deepx_prepared_feed_benchmark(
        case.root, case.cached, {"benchmark_task": "detection", "model_id": "yolo11l"},
        SimpleNamespace(runs=3, warmup=1, energy_measurement_only=False,
                        native_full_postprocess_runtime_mode=mode), results,
    )
    assert result["status"] == "ok", result
    assert result["completed_frames"] == result["postprocess_completed_frames"] == 3
    assert result["completed_task_result_artifact_saved"] is True
    assert observed[-1] == ("final", 5, 4)
    assert sum(x[0] == "task" for x in observed) == (4 if mode == "auto" else 0)
    assert result["frozen_host_postprocess_result"]["postflight_oracle"]["status"] == "passed"
