"""A4 is a Native-only extension of actual generated product bundles."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns
from onnx_splitpoint_tool.split_export_runners import write_runner_onnxruntime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generated(tmp_path, monkeypatch):
    suite = tmp_path / 'suite'
    suite.mkdir()
    script = Path(write_benchmark_suite_script(suite))
    case = suite / 'b001'
    case.mkdir()
    runner = Path(write_runner_onnxruntime(str(case)))
    spec = importlib.util.spec_from_file_location('_v295_generated_suite', script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return suite, case, runner, module


@pytest.mark.parametrize('backend', ['deepx_m1', 'hailo8', 'hailo10h'])
@pytest.mark.parametrize('variant', ['full', 'composed'])
def test_generic_backend_requests_keep_existing_effort_and_scheduler(generated, monkeypatch, backend, variant):
    _, case, _, suite = generated
    calls = []
    monkeypatch.setattr(suite, '_provider_unavailable_reason', lambda *_: None)
    monkeypatch.setattr(suite, '_collect_case_result', lambda *a, **k: {'captured': True})
    monkeypatch.setattr(suite.subprocess, 'run', lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0))
    result = suite._run_case(
        case, backend + '_' + variant, provider=backend, image='default', preset='auto', image_scale='auto',
        warmup=100, runs=1000, timeout_s=120, stage1=backend,
        stage2='tensorrt' if variant == 'composed' else backend, variants=[variant],
        native_trt_build=False, streaming_impl='generic', throughput_frames=1000,
        throughput_warmup_frames=100, throughput_queue_depth=2, phase_runs=0, benchmark_task='classification',
    )
    assert result == {'captured': True}
    assert len(calls) == 1
    argv = calls[0]
    for flag, value in (('--warmup', '100'), ('--runs', '1000'), ('--throughput-frames', '1000'),
                        ('--throughput-warmup-frames', '100'), ('--throughput-queue-depth', '2')):
        assert argv[argv.index(flag) + 1] == value
    assert '--deepx-runtime-mode' not in argv
    assert '--deepx-max-inflight' not in argv
    if variant == 'composed':
        assert argv[argv.index('--streaming-impl') + 1] == 'generic'
        assert '--no-native-trt-build' in argv


def test_generic_deepx_session_keeps_synchronous_engine_api(generated):
    _, _, runner, _ = generated
    tree = ast.parse(runner.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepXSession')
    attributes = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    assert 'run' in attributes
    assert not {'run_async', 'wait', 'native_deepx_async'} & attributes


def test_regular_remote_archive_contains_importable_native_adapter(generated, tmp_path):
    suite, _, _, _ = generated
    (suite / 'benchmark_set.json').write_text(json.dumps({'cases': [{'case_id': 'b001', 'case_dir': 'b001'}]}))
    archive = tmp_path / 'product.tar.gz'
    includes, excludes = remote_minimal_bundle_patterns()
    build_suite_bundle(suite, archive, includes=includes, excludes=excludes)
    remote = tmp_path / 'extracted'
    with tarfile.open(archive) as payload:
        member = payload.getmember('splitpoint_runners/native_deepx_async.py')
        assert payload.extractfile(member).read() == (ROOT / 'onnx_splitpoint_tool/native_deepx_async.py').read_bytes()
        payload.extractall(remote, filter='data')
    script = '''import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
m=importlib.import_module('splitpoint_runners.native_deepx_async')
assert Path(m.__file__).resolve()==root/'splitpoint_runners/native_deepx_async.py'
assert m.INFLIGHT_LIMIT==4 and m.FIFO_DEPTH==3
assert callable(m.run_a4)
print(json.dumps({'module':str(m.__file__),'inflight':m.INFLIGHT_LIMIT,'queue':m.FIFO_DEPTH}))
'''
    proof = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(remote)], cwd=tmp_path,
                           text=True, capture_output=True, check=True, timeout=30)
    assert json.loads(proof.stdout)['inflight'] == 4
