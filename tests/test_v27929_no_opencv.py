"""Full geometry/byte regressions with simulated DXRT and a controlled BGR decoder."""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

from tests.test_v27926_deepx_full_decoded_pre_nms import (
    case, ROOT, _suite, _decoded_outputs, _install_runtime,
)
from onnx_splitpoint_tool.runners.native_full_input import (
    load_sealed_deepx_native_full_input,
)


def _forbid_cv2(monkeypatch):
    monkeypatch.delitem(sys.modules, 'cv2', raising=False)
    original = builtins.__import__
    attempts = []
    def blocked(name, *args, **kwargs):
        if name == 'cv2' or name.startswith('cv2.'):
            attempts.append(name)
            raise ModuleNotFoundError('cv2 forbidden in regression', name='cv2')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    return attempts


def _arguments(energy=False):
    return SimpleNamespace(
        runs=2, warmup=1, energy_measurement_only=energy,
        throughput_frames=3 if energy else 0,
        prepared_input_manifest='', quality_evidence_model_id='yolo11l',
    )


def _run(case, monkeypatch, image, *, energy=False):
    suite = _suite()
    count = _install_runtime(monkeypatch, _decoded_outputs())
    attempts = _forbid_cv2(monkeypatch)
    monkeypatch.setattr(suite, '_deepx_find_prepared_feed_image', lambda *a: (image, 'fixture'))
    args = _arguments(energy)
    result = suite._run_deepx_prepared_feed_benchmark(
        case.root, case.cached,
        {'benchmark_task': 'detection', 'model_id': 'yolo11l', 'setup_id': 'test_deepx'},
        args, case.root / 'results',
    )
    assert attempts == []
    return result, count, args


@pytest.mark.parametrize('energy', [False, True])
@pytest.mark.parametrize('image_mode,exif_orientation', [('RGB', None), ('L', None), ('RGB', 6), ('RGB', 8)])
def test_full_and_energy_without_opencv_preserve_encoded_geometry_and_feed(case, monkeypatch, energy, image_mode, exif_orientation):
    image = case.root / 'source.jpg'
    array = np.arange(335 * 500 * 3, dtype=np.uint8).reshape(335, 500, 3)
    pil = Image.fromarray(array).convert(image_mode)
    options = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        options['exif'] = exif
    pil.save(image, **options)
    before = hashlib.sha256(image.read_bytes()).hexdigest()
    result, count, args = _run(case, monkeypatch, image, energy=energy)
    assert result['status'] == 'ok', result
    assert result['original_image_wh'] == [500, 335]
    assert result['preprocessing_contract_audit']['source_shape_hw'] == [335, 500]
    assert result['frozen_host_postprocess_contract']['original_wh'] == [500, 335]
    expected = 3 if energy else 2
    assert result['completed_frames'] == result['postprocess_completed_frames'] == expected
    assert len(count) == expected + 2  # structural + warmup + completed work
    assert hashlib.sha256(image.read_bytes()).hexdigest() == before
    # Existing loader remains the authority, not an OpenCV/Pillow substitute.
    loaded = load_sealed_deepx_native_full_input(
        Path(args.prepared_input_manifest), image_path=image,
        input_contract=case.contract, task='detection', expected_model='yolo11l',
        expected_setup_id='test_deepx', expected_comparison_backend='deepx',
    )
    assert loaded['runtime_input'].shape == (640, 640, 3)
    assert loaded['runtime_input'].dtype == np.uint8
    assert hashlib.sha256(loaded['runtime_input'].tobytes()).hexdigest() == result['prepared_input_sha256']


@pytest.mark.parametrize('image_kind', ['missing', 'empty', 'invalid', 'truncated'])
def test_bad_image_reports_image_read_failed_before_engine(case, monkeypatch, image_kind):
    image = case.root / 'source.jpg'
    if image_kind == 'empty':
        image.write_bytes(b'')
    elif image_kind == 'invalid':
        image.write_bytes(b'not an image')
    elif image_kind == 'truncated':
        Image.new('RGB', (500, 335)).save(image)
        image.write_bytes(image.read_bytes()[:len(image.read_bytes()) // 2])
    result, count, _ = _run(case, monkeypatch, image)
    assert result['status'] == 'image_read_failed', result
    assert result['error']
    assert count == []


def test_cv2_has_not_been_removed_from_unrelated_runtime_paths():
    text = (ROOT / 'onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text()
    tree = ast.parse(text)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_deepx_prepared_feed_benchmark')
    names = [node.id for node in ast.walk(func) if isinstance(node, ast.Name)]
    assert 'cv2' not in names
    assert 'img' not in names
    assert 'original_wh' in names
    # v31 deliberately adds an optional Pillow decoder to the quality path.
    # Its actual RGB bytes/geometry are checked below; the prepared measurement
    # loop must still leave decoding/preparation to its existing sealed loader.


@pytest.mark.parametrize('decoder', ['opencv_api', 'pillow_fallback'])
@pytest.mark.parametrize('source_hw,expected_pad', [((361,640),(0,139)), ((640,320),(160,0)), ((335,500),(0,105))])
def test_quality_optional_decoder_preserves_canonical_rgb_feed_and_geometry(case, monkeypatch, decoder, source_hw, expected_pad):
    """Exercise the real quality path; only vendor inference and cv2.imread
    are controlled. OpenCV itself is not installed in this test environment.
    The fallback really decodes PNG, and both paths must match the unchanged
    native input preparer byte for byte, including channel order and odd pad.
    """
    from onnx_splitpoint_tool.runners.native_full_input import _prepare_tensor
    height,width=source_hw
    yy,xx=np.indices(source_hw)
    rgb=np.empty((height,width,3),np.uint8)
    rgb[...,0]=(xx*3+17)%256;rgb[...,1]=(yy*7+51)%256;rgb[...,2]=(xx+yy+199)%256
    image=case.root/'quality_source.png';Image.fromarray(rgb).save(image)
    original_sha=hashlib.sha256(image.read_bytes()).hexdigest()
    _install_runtime(monkeypatch,_decoded_outputs())
    decoded=[]
    if decoder=='opencv_api':
        def decode_bgr(path):
            assert Path(path)==image
            decoded.append(path)
            # Fixed decoder boundary, independent of the product conversion.
            return np.take(rgb,[2,1,0],axis=2)
        monkeypatch.setattr(sys.modules['cv2'],'imread',decode_bgr)
    else:
        attempts=_forbid_cv2(monkeypatch)
    dx=sys.modules['dx_engine'];base=dx.InferenceEngine;feeds=[]
    class Capture(base):
        def run(self,inputs):
            feeds.append(inputs[0].copy())
            return super().run(inputs)
    monkeypatch.setattr(dx,'InferenceEngine',Capture)
    suite=_suite()
    results=case.root/'quality_result';results.mkdir()
    result=suite._run_deepx_semantic_validation(case.root,case.cached,
        {'id':'deepx_m1_full','model_id':'yolo11l','benchmark_task':'detection','validation_images':str(image)},
        SimpleNamespace(validation_images='',validation_max_images=1),results)
    assert result['status']=='ok' and result['error_count']==0,result
    assert len(feeds)==1
    inp=case.contract['input']
    expected,_,_=_prepare_tensor(image,shape=inp['shape'],dtype=np.dtype(inp['dtype']),layout=inp['layout'],
        task='detection',normalization=inp['normalization'],preprocess_mode=inp['preprocess_mode'],pad_value=inp['letterbox_pad_value'])
    assert np.array_equal(feeds[0],expected)
    assert feeds[0].dtype==np.uint8 and feeds[0].flags.c_contiguous
    record=json.loads((results/'detections.json').read_text())['images'][0]
    audit=record['preprocessing_audit']
    assert audit['source_shape_hw']==list(source_hw)
    assert (audit['pad_x'],audit['pad_y'])==expected_pad
    assert audit['prepared_tensor_binding']['prepared_input_sha256']==hashlib.sha256(expected.tobytes()).hexdigest()
    assert record['decoder_contract']['frozen_postprocess_contract']['original_wh']==[width,height]
    if source_hw==(361,640):
        assert np.array_equal(feeds[0][139:500],rgb)
        assert np.all(feeds[0][:139]==114) and np.all(feeds[0][500:]==114)
    if decoder=='opencv_api':assert decoded==[str(image)]
    else:assert attempts==['cv2']
    assert hashlib.sha256(image.read_bytes()).hexdigest()==original_sha
