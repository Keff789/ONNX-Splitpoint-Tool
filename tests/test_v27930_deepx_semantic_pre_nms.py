"""Actual semantic runner replay; only the physical dx_engine boundary is fake.

The NPZ/image/declaration are unchanged recorded bytes. New test runs use their
own setup and directory; no archived run/session is promoted to successful data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/v27930"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_semantic_module(path=None):
    path = path or ROOT / "scripts/native_full_semantic_dump.py"
    spec = importlib.util.spec_from_file_location("v27930_semantic_" + str(abs(hash(str(path)))), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recorded_values():
    with np.load(FIXTURES / "semantic_probe_outputs.npz", allow_pickle=False) as archive:
        assert archive.files == ["tensor_000"]
        return archive["tensor_000"]


def prepare_semantic_case(tmp_path, monkeypatch, output=None, *, model="yolo11l", task="detection"):
    """Reusable hardware-boundary fixture for full-runner integration tests."""
    root = tmp_path / "synthetic_v27930_run" / model / "benchmark_set"
    full = root / "deepx/deepx_m1/full"
    full.mkdir(parents=True)
    (full / "model.dxnn").write_bytes(b"synthetic accelerator handle; engine boundary is mocked")
    contract = json.loads((FIXTURES / "semantic_input_contract.json").read_text())
    contract["model_id"] = model
    write_json(full / "output_contract.json", contract)
    shutil.copyfile(FIXTURES / "semantic_recorded_output_contracts.json", root / "output_contracts.json")
    image = root / "000000212226.jpg"
    shutil.copyfile(FIXTURES / "semantic_probe_input.jpg", image)
    output = recorded_values() if output is None else output
    values = output if isinstance(output, list) else [output]
    calls = []

    class Engine:
        def __init__(self, path):
            assert Path(path) == full / "model.dxnn"

        def run(self, feeds):
            assert len(feeds) == 1
            assert feeds[0].shape == (640, 640, 3)
            assert feeds[0].dtype == np.uint8
            calls.append(feeds[0].copy())
            # Deliberately share read-only physical buffers. A hidden in-place
            # clipping fix would fail and cannot be concealed by engine copies.
            return values

    for value in values:
        value.setflags(write=False)
    dx = ModuleType("dx_engine")
    dx.InferenceEngine = Engine
    monkeypatch.setitem(sys.modules, "dx_engine", dx)
    return SimpleNamespace(root=root, full=full, image=image, model=model,
                           task=task, setup_id="synthetic_v27930_deepx",
                           values=values, calls=calls, contract=contract)


def run_semantic_case(case, module=None, out_dir=None, prepared_input_manifest=None):
    module = module or load_semantic_module()
    return module._run_deepx(
        case.root, case.image, out_dir or case.root / "semantic", case.model,
        case.task, case.setup_id, "deepx", prepared_input_manifest=prepared_input_manifest,
    )


def _manifest(result):
    return json.loads(Path(result["output_manifest"]).read_text())


def _assert_current_implementation_contract(frozen, archived):
    """Replay old semantics with the current, explicitly verified code bytes.

    v2.82 changed the YOLOv7 arithmetic implementation in these two shared
    modules. The archived YOLO11 contract remains immutable; only the two
    implementation digests and the seals that bind them may differ today.
    """
    from onnx_splitpoint_tool.native_detection_postprocess import (
        canonical_json_sha256, verify_frozen_postprocess_contract,
    )
    expected = copy.deepcopy(archived)
    for artifact, relative in (
        ("native_detection_postprocess", "onnx_splitpoint_tool/native_detection_postprocess.py"),
        ("yolo_harness", "onnx_splitpoint_tool/runners/harness/yolo.py"),
    ):
        actual_sha = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert frozen["implementation_artifacts"][artifact]["sha256"] == actual_sha, (
            "current_runtime_code_sha256_mismatch:" + artifact
        )
        assert frozen["invariant_identity"]["implementation_artifacts"][artifact]["sha256"] == actual_sha, (
            "current_invariant_code_sha256_mismatch:" + artifact
        )
        expected["implementation_artifacts"][artifact]["sha256"] = actual_sha
        expected["invariant_identity"]["implementation_artifacts"][artifact]["sha256"] = actual_sha
    # The real consumer must validate today's complete contract, including its
    # seals. Recompute only these expected derived fields in the test copy.
    verify_frozen_postprocess_contract(frozen)
    expected["invariant_contract_sha256"] = canonical_json_sha256(expected["invariant_identity"])
    expected["contract_sha256"] = canonical_json_sha256({
        key: value for key, value in expected.items() if key != "contract_sha256"
    })
    assert frozen == expected


def _assert_replay(result, case, *, exact_contract=True):
    archived = json.loads((FIXTURES / "semantic_fixture.json").read_text())
    frozen = result["frozen_host_postprocess_contract"]
    processed = result["frozen_host_postprocess_result"]
    assert frozen["source_contract_family"] == "decoded_pre_nms"
    assert frozen["output_contract_family"] == "decoded_nms"
    assert frozen["decoded_pre_nms_score_policy"]["absolute_tolerance"] == 2.0 ** -23
    if exact_contract:
        _assert_current_implementation_contract(frozen, archived["frozen_host_postprocess_contract"])
    assert processed["detections"] == archived["frozen_host_postprocess_result"]["detections"]
    assert processed["detection_count"] == 2
    assert processed["decoded_pre_nms_score_normalization"]["corrected_score_count"] == 21
    manifest = _manifest(result)
    assert manifest["stage"] == manifest["contract_family"] == "decoded_pre_nms"
    assert manifest["endpoint_contract_complete"] is True
    assert manifest["host_postprocess_frozen"] is True
    assert manifest["postprocess_included"] is True
    assert manifest["e2e_scope"] == "full_task_pipeline"
    for entry, value in zip(manifest["outputs"], case.values):
        assert (Path(result["output_manifest"]).parent / entry["file"]).read_bytes() == value.tobytes()
    return frozen, processed


@pytest.mark.parametrize("remote", [False, True])
def test_s01_actual_recorded_semantic_runner_preserves_physical_bytes(tmp_path, monkeypatch, remote):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    before = case.values[0].tobytes()
    path = ROOT / ("onnx_splitpoint_tool/resources/remote_scripts" if remote else "scripts") / "native_full_semantic_dump.py"
    result = run_semantic_case(case, load_semantic_module(path))
    _assert_replay(result, case)
    assert case.values[0].tobytes() == before
    assert len(case.calls) == 1  # semantic inference only; builder may validate NMS


def test_s02_transposed_physical_layout_preserves_detections(tmp_path, monkeypatch):
    a = prepare_semantic_case(tmp_path / "a", monkeypatch)
    original = run_semantic_case(a)
    b = prepare_semantic_case(tmp_path / "b", monkeypatch, recorded_values().transpose(0, 2, 1))
    transposed = run_semantic_case(b)
    frozen, _ = _assert_replay(transposed, b, exact_contract=False)
    assert frozen["contract_sha256"] != original["frozen_host_postprocess_contract_sha256"]
    assert frozen["raw_output_tensor_signature"]["tensors"][0]["shape"] == [1, 8400, 84]


@pytest.mark.parametrize("channel,value", [(4, -0.01), (4, np.nan), (4, np.inf), (2, -0.01), (3, -0.01)])
def test_s03_actual_numeric_violations_block_completion(tmp_path, monkeypatch, channel, value):
    outputs = recorded_values()
    outputs[0, channel, 0] = value
    case = prepare_semantic_case(tmp_path, monkeypatch, outputs)
    before = outputs.tobytes()
    with pytest.raises(RuntimeError, match="FrozenPostprocessError|deepx_full_pre_nms_runtime_endpoint_invalid"):
        run_semantic_case(case)
    assert outputs.tobytes() == before
    assert not (case.root / "semantic/native_full_outputs_manifest.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "wrong_model", "unknown_stage", "contradictory"])
def test_s04_declaration_is_not_inferred_from_tensor_shape(tmp_path, monkeypatch, mutation):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    path = case.root / "output_contracts.json"
    if mutation == "missing":
        path.unlink()
    else:
        data = json.loads(path.read_text())
        if mutation == "wrong_model":
            data["model_id"] = "unrelated"
        else:
            row = next(r for r in data["contracts"] if r["backend"] == "deepx_m1")
            if mutation == "unknown_stage":
                row.update(stage="unknown", contract_family="unknown", endpoint_mode="unknown")
            else:
                row["host_tail_required"] = False
        write_json(path, data)
    result = run_semantic_case(case)
    assert result["frozen_host_postprocess_contract"] == {}
    manifest = _manifest(result)
    assert manifest["stage"] == "unknown"
    assert manifest["endpoint_contract_complete"] is False
    assert manifest["claim_eligible_e2e"] is False


@pytest.mark.parametrize("mutation", ["missing_input_geometry", "unreadable_image"])
def test_s04_missing_geometry_stays_an_error(tmp_path, monkeypatch, mutation):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    if mutation == "missing_input_geometry":
        case.contract["input"].pop("shape")
        write_json(case.full / "output_contract.json", case.contract)
    else:
        case.image.write_bytes(b"not an image")
    with pytest.raises(Exception):
        run_semantic_case(case)
    assert not case.calls


def _set_endpoint(case, stage):
    tail = stage in {"raw_head", "decoded_pre_nms"}
    row = {"schema": "onnx-splitpoint/output-contract", "schema_version": 1,
           "model_id": case.model, "task": case.task, "backend": "deepx_m1", "variant": "full",
           "contract_status": "recorded", "stage": stage, "contract_family": stage,
           "endpoint_mode": stage, "host_tail_required": tail, "postprocessing_required": tail}
    if stage == "decoded_nms":
        row["source_coordinate_space"] = "model_input_letterbox_xyxy_pixels"
    write_json(case.root / "output_contracts.json", {"schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1, "model_id": case.model, "task": case.task, "contracts": [row]})


def test_s05_direct_bn6_keeps_existing_normalization_route(tmp_path, monkeypatch):
    # The existing Direct-BN6 normalization includes its canonical filter/NMS.
    # Preserve that separate historical route, without routing through the newly
    # enabled FrozenDetectionPostprocessor used for physical pre-NMS outputs.
    output = np.array([[[10, 150, 120, 250, .9, 0], [10, 150, 120, 250, .8, 0]]], dtype=np.float32)
    case = prepare_semantic_case(tmp_path, monkeypatch, output, model="yolo26s")
    _set_endpoint(case, "decoded_nms")
    result = run_semantic_case(case)
    assert result["frozen_host_postprocess_contract"] == {}
    direct = result["frozen_decoded_nms_normalization_result"]
    assert direct["detection_count"] == 1
    assert direct["decoder_format"] == "bn6_detections"
    assert _manifest(result)["normalization_frozen"] is True


def test_s05_classification_keeps_logits_without_detection_processing(tmp_path, monkeypatch):
    output = np.linspace(-2, 2, 1000, dtype=np.float32)[None, :]
    case = prepare_semantic_case(tmp_path, monkeypatch, output, model="resnet50", task="classification")
    _set_endpoint(case, "classification_logits")
    result = run_semantic_case(case)
    assert result["frozen_host_postprocess_contract"] == {}
    assert result["frozen_decoded_nms_normalization_contract"] == {}
    assert _manifest(result)["contract_family"] == "classification_logits"
    assert _manifest(result)["endpoint_contract_complete"] is True


def test_s05_raw_yolov7_keeps_existing_frozen_decoder(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.runners.harness.yolo import YOLOV7_PAPER_ONNX_SHA256
    output = [np.full((1, 3, n, n, 85), -20, dtype=np.float32) for n in (80, 40, 20)]
    case = prepare_semantic_case(tmp_path, monkeypatch, output, model="yolov7_paper")
    case.contract["source_onnx_sha256"] = YOLOV7_PAPER_ONNX_SHA256
    case.contract["outputs"] = [{"name": f"raw_{i}"} for i in range(3)]
    write_json(case.full / "output_contract.json", case.contract)
    _set_endpoint(case, "raw_head")
    result = run_semantic_case(case)
    assert result["frozen_host_postprocess_contract"]["source_contract_family"] == "raw_head"
    assert result["frozen_host_postprocess_result"]["contract_family"] == "decoded_nms"
    assert _manifest(result)["stage"] == "raw_head"


def test_r02_prepared_input_is_replayed_exactly_without_opencv(tmp_path, monkeypatch):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "cv2", None)
    first = run_semantic_case(case)
    second = run_semantic_case(case, out_dir=case.root / "prepared_replay", prepared_input_manifest=Path(first["input_manifest"]))
    assert second["input_candidate"] == "shared_pre_timing_sealed_manifest"
    assert first["frozen_host_postprocess_contract"] == second["frozen_host_postprocess_contract"]
    np.testing.assert_array_equal(case.calls[0], case.calls[1])
    _assert_replay(second, case)


@pytest.mark.parametrize("route", ["local_semantic", "vendored_prepared_feed"])
def test_r02_fresh_process_uses_only_copied_runtime_without_opencv(tmp_path, monkeypatch, route):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    prepared = run_semantic_case(case)
    staged = tmp_path / "isolated_runtime"
    staged.mkdir()
    package = ROOT / "onnx_splitpoint_tool"
    for package_name in ("onnx_splitpoint_tool", "splitpoint_runners"):
        copied = staged / package_name
        if package_name == "splitpoint_runners":
            shutil.copytree(package / "runners", copied,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            copied.mkdir()
            (copied / "__init__.py").write_text("")
            shutil.copytree(package / "runners", copied / "runners",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("native_detection_postprocess.py", "native_output_endpoint.py", "preprocessing_contract.py"):
            shutil.copyfile(package / name, copied / name)
    scripts = staged / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/native_full_semantic_dump.py", scripts / "native_full_semantic_dump.py")
    shutil.copyfile(package / "resources/templates/benchmark_suite.py.txt", staged / "benchmark_suite.py")
    shutil.copyfile(FIXTURES / "semantic_probe_outputs.npz", staged / "recorded.npz")
    request = {"route": route, "suite": str(case.root), "image": str(case.image),
               "model": case.model, "setup_id": case.setup_id, "prepared": prepared["input_manifest"]}
    write_json(staged / "request.json", request)
    bootstrap = staged / "bootstrap.py"
    bootstrap.write_text('''
import importlib.util, json, sys, types
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
sys.modules['cv2'] = None
request = json.loads((root / 'request.json').read_text())
values = np.load(root / 'recorded.npz', allow_pickle=False)['tensor_000']
before = values.tobytes()
values.setflags(write=False)
calls = []
dx = types.ModuleType('dx_engine')
class Engine:
    def __init__(self, path):
        assert Path(path).is_file()
    def run(self, feeds):
        assert feeds[0].shape == (640, 640, 3) and feeds[0].dtype == np.uint8
        calls.append(feeds[0].tobytes())
        return [values]
dx.InferenceEngine = Engine
sys.modules['dx_engine'] = dx
def module(path):
    spec = importlib.util.spec_from_file_location('copied_runtime', path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result
suite = Path(request['suite'])
if request['route'] == 'local_semantic':
    runner = module(root / 'scripts/native_full_semantic_dump.py')
    result = runner._run_deepx(suite, Path(request['image']), root / 'results',
        request['model'], 'detection', request['setup_id'], 'deepx',
        prepared_input_manifest=Path(request['prepared']))
    assert result['ok'] is True
    assert result['frozen_host_postprocess_result']['detection_count'] == 2
    expected_count = 1
else:
    runner = module(root / 'benchmark_suite.py')
    args = types.SimpleNamespace(runs=3, warmup=0, energy_measurement_only=False,
        prepared_input_manifest=request['prepared'], prepared_feed_image=request['image'],
        quality_evidence_model_id=request['model'])
    (root / 'results').mkdir()
    result = runner._run_deepx_prepared_feed_benchmark(suite,
        suite / 'deepx/deepx_m1/full/model.dxnn',
        {'id': 'synthetic_v27930_full', 'benchmark_task': 'detection',
         'model_id': request['model'], 'setup_id': request['setup_id'],
         'contract_path': 'deepx/deepx_m1/full/output_contract.json'},
        args, root / 'results')
    assert result['status'] == 'ok', result
    assert result['completed_frames'] == result['postprocess_completed_frames'] == 3
    assert result['frozen_host_postprocess_result']['detection_count'] == 2
    expected_count = 4  # untimed structure +3 actual completed hotloop frames
assert len(calls) == expected_count
assert all(value == calls[0] for value in calls)
assert values.tobytes() == before
borrowed = []
for name, loaded in list(sys.modules.items()):
    if name.startswith(('onnx_splitpoint_tool', 'splitpoint_runners')):
        path = getattr(loaded, '__file__', None)
        if path and not Path(path).resolve().is_relative_to(root):
            borrowed.append((name, path))
assert not borrowed, borrowed
assert sys.modules['cv2'] is None
print(json.dumps({'status':'pass', 'route':request['route'], 'engine_calls':len(calls),
                  'borrowed_modules':borrowed, 'opencv_loaded':False}))
''', encoding="utf-8")
    completed = subprocess.run([sys.executable, "-I", "-B", str(bootstrap)],
                               cwd=staged, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["status"] == "pass"
    assert report["borrowed_modules"] == []
    assert report["opencv_loaded"] is False


def test_fixture_hashes_and_mirrored_scripts():
    recorded = json.loads((FIXTURES / "semantic_fixture.json").read_text())
    for file, digest in recorded["files"].items():
        assert hashlib.sha256((FIXTURES / file).read_bytes()).hexdigest() == digest
    assert (ROOT / "scripts/native_full_semantic_dump.py").read_bytes() == (ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_full_semantic_dump.py").read_bytes()


@pytest.mark.parametrize("artifact", ["native_detection_postprocess", "yolo_harness"])
def test_current_replay_rejects_resealed_foreign_implementation_hash(tmp_path, monkeypatch, artifact):
    from onnx_splitpoint_tool.native_detection_postprocess import canonical_json_sha256
    case = prepare_semantic_case(tmp_path, monkeypatch)
    frozen = copy.deepcopy(run_semantic_case(case)["frozen_host_postprocess_contract"])
    frozen["implementation_artifacts"][artifact]["sha256"] = "0" * 64
    frozen["invariant_identity"]["implementation_artifacts"][artifact]["sha256"] = "0" * 64
    frozen["invariant_contract_sha256"] = canonical_json_sha256(frozen["invariant_identity"])
    frozen["contract_sha256"] = canonical_json_sha256({
        key: value for key, value in frozen.items() if key != "contract_sha256"
    })
    archived = json.loads((FIXTURES / "semantic_fixture.json").read_text())
    with pytest.raises(AssertionError, match="current_runtime_code_sha256_mismatch:" + artifact):
        _assert_current_implementation_contract(frozen, archived["frozen_host_postprocess_contract"])
