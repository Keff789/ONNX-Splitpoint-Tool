"""Fresh v28 staged worker + real suite and harness; only DXRT is substituted."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts import deepx_full_output_probe_worker_v27928 as worker
from tests.test_v27926_deepx_full_decoded_pre_nms import case, ROOT


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize("kind", ["roundoff", "invalid"])
def test_staged_current_worker_real_tensor_and_harness(case, tmp_path, kind):
    original_npz = os.environ.get("V27928_REPLAY_NPZ")
    if not original_npz:
        pytest.skip("recorded tensor requires V27928_REPLAY_NPZ from the delivery bundle")
    original_npz = Path(original_npz)
    input_image = original_npz.with_name("probe_input.jpg")
    assert Image.open(input_image).size == (500, 335)
    package = ROOT / "onnx_splitpoint_tool"
    original = case.root
    shutil.copytree(package / "runners", original / "splitpoint_runners",
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('native_output_endpoint.py', 'preprocessing_contract.py', 'native_detection_postprocess.py'):
        shutil.copy2(package / name, original / 'splitpoint_runners' / name)
    shutil.copy2(package / 'resources/templates/benchmark_suite.py.txt', original / 'benchmark_suite.py')
    image = original / 'resources/000000212226.jpg'
    image.parent.mkdir()
    shutil.copy2(input_image, image)
    dxnn = original / 'deepx/deepx_m1/full/model.dxnn'
    # The fixture's actual cache bytes are synthetic, as documented by case.
    shutil.copy2(case.cached, dxnn)
    (original / 'benchmark_plan.json').write_text(json.dumps({'runs': [{
        'id': 'deepx_m1_full', 'setup_id': 'recorded_deepx', 'benchmark_task': 'detection',
        'contract_path': 'deepx/deepx_m1/full/output_contract.json',
    }]}))
    before = inventory(original)
    work = tmp_path / 'fresh staged work'
    work.mkdir()
    shutil.copytree(original, work / 'suite')
    request = {
        'schema': worker.SCHEMA, 'staging_mode': worker.STAGING_MODE,
        'staged_dxnn_relative': 'deepx/deepx_m1/full/model.dxnn',
        'staged_image_relative': 'resources/000000212226.jpg',
        'expected_dxnn_sha256': hashlib.sha256(dxnn.read_bytes()).hexdigest(),
        'expected_image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'model_id': 'yolo11l', 'setup_id': 'recorded_deepx', 'original_run_id': 'recorded_tensor_fixture',
        'original_remote_suite': '/nonexistent/deleted/v26/run/1/suite',
    }
    (work / 'probe_request.json').write_text(json.dumps(request))
    bootstrap = work / 'test_dxrt_engine.py'
    bootstrap.write_text('''
import runpy, sys, types
import numpy as np
worker_path, request_path, npz_path, kind = sys.argv[1:]
with np.load(npz_path, allow_pickle=False) as recorded:
    output = recorded["tensor_000"]
if kind == "invalid":
    output[0,4,17] = -0.01
class Engine:
    def __init__(self, model): self.count = 0
    def run(self, feeds):
        self.count += 1
        assert self.count <= 2
        assert feeds[0].shape == (640,640,3) and feeds[0].dtype == np.uint8
        return [output.copy()]
dx = types.ModuleType("dx_engine")
dx.InferenceEngine = Engine
sys.modules["dx_engine"] = dx
sys.argv = [worker_path,"--request",request_path,"--runtime"]
runpy.run_path(worker_path,run_name="__main__")
''')
    process = subprocess.run([
        sys.executable, '-I', '-B', str(bootstrap), worker.__file__,
        str(work / 'probe_request.json'), str(original_npz), kind,
    ], text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((work / 'results/deepx_output_value_probe.json').read_text())
    assert result['status'] == ('diagnostic_pass' if kind == 'roundoff' else 'runtime_failed'), result
    assert result['engine_call_count'] == (2 if kind == 'roundoff' else 1)
    assert result['counts_as_benchmark'] is False
    assert result['probe_source']['compiler_invoked'] is False
    assert result['probe_source']['remote_suite'] == str(work / 'suite')
    if kind == 'roundoff':
        obs = result['decoded_pre_nms_score_normalization']
        assert obs['corrected_score_count'] == 21 and obs['source_modified'] is False
        with np.load(work / 'results/deepx_output_value_probe_outputs.npz', allow_pickle=False) as captured:
            with np.load(original_npz, allow_pickle=False) as source:
                assert captured['tensor_000'].tobytes() == source['tensor_000'].tobytes()
    else:
        assert 'decoded_pre_nms_values_invalid' in result['error']
    assert inventory(original) == before
