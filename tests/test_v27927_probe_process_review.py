"""Execute the worker plus real suite/postprocessing in a fresh interpreter.

Only the vendor engine and OpenCV reader are substituted. The sealed input,
copied remote package, output contract and NPZ writer execute their real code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts import deepx_full_output_probe_worker as worker
from tests.test_v27926_deepx_full_decoded_pre_nms import case, ROOT


def _inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("invalid", [False, True])
def test_full_worker_process_uses_copied_package_and_real_probe(case, tmp_path, invalid):
    root = case.root
    package = ROOT / "onnx_splitpoint_tool"
    copied = root / "splitpoint_runners"
    shutil.copytree(package / "runners", copied,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("native_output_endpoint.py", "preprocessing_contract.py"):
        shutil.copy2(package / name, copied / name)
    # An unusable old implementation forces the worker's overlay to be used.
    (copied / "native_detection_postprocess.py").write_text("raise RuntimeError('old copy used')\n")
    image = root / "resources/000000212226.png"
    image.parent.mkdir()
    Image.new("RGB", (640, 640), "black").save(image)
    (root / "benchmark_plan.json").write_text(json.dumps({"runs": [{
        "id": "deepx_m1_full", "setup_id": "recorded_deepx", "benchmark_task": "detection",
        "contract_path": "deepx/deepx_m1/full/output_contract.json",
    }]}))
    work = tmp_path / "isolated probe"
    work.mkdir()
    request = {
        "schema": "onnx-splitpoint/deepx-full-output-probe-request/v1",
        "remote_suite": str(root), "dxnn_path": str(case.cached), "image_path": str(image),
        "expected_dxnn_sha256": hashlib.sha256(case.cached.read_bytes()).hexdigest(),
        "model_id": "yolo11l", "setup_id": "recorded_deepx", "original_run_id": "fixture",
    }
    (work / "probe_request.json").write_text(json.dumps(request))
    shutil.copy2(package / "native_detection_postprocess.py", work)
    shutil.copy2(package / "resources/templates/benchmark_suite.py.txt", work)
    bootstrap = work / "bootstrap.py"
    bootstrap.write_text('''
import runpy, sys, types
import numpy as np
from PIL import Image
dx = types.ModuleType("dx_engine")
invalid = sys.argv[3] == "invalid"
class Engine:
    def __init__(self, path):
        self.calls = 0
    def run(self, feeds):
        self.calls += 1
        assert self.calls <= 2
        assert feeds[0].shape == (640, 640, 3) and feeds[0].dtype == np.uint8
        assert not feeds[0].any()
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        output[0, 4, 7] = -0.00001 if invalid else 0.9
        return [output]
dx.InferenceEngine = Engine
cv = types.ModuleType("cv2")
cv.imread = lambda path: np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
sys.modules.update(dx_engine=dx, cv2=cv)
worker, request = sys.argv[1:3]
sys.argv = [worker, "--request", request, "--runtime"]
runpy.run_path(worker, run_name="__main__")
''')
    before = _inventory(root)
    completed = subprocess.run([
        sys.executable, "-I", "-B", str(bootstrap), str(Path(worker.__file__).resolve()),
        str(work / "probe_request.json"), "invalid" if invalid else "valid",
    ], text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = work / "results"
    result = json.loads((results / "deepx_output_value_probe.json").read_text())
    assert result["status"] == ("runtime_failed" if invalid else "diagnostic_pass"), result
    assert result["engine_call_count"] == (1 if invalid else 2)
    assert result["prepared_input_binding_verified"] is True
    assert result["counts_as_benchmark"] is False
    assert not {"fps_makespan", "mean_ms", "completed_frames"}.intersection(result)
    assert result["raw_output_snapshot"]["status"] == "saved"
    with np.load(results / result["raw_output_snapshot"]["file"], allow_pickle=False) as outputs:
        assert outputs["tensor_000"][0, 4, 7] == np.float32(-0.00001 if invalid else 0.9)
    assert (results / "probe_input.png").read_bytes() == image.read_bytes()
    assert _inventory(root) == before
