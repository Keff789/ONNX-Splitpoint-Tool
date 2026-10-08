"""CPU references must use the bound RGB raster, not H8 quantization codes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    runtime_numeric_input_identity,
)
from scripts import native_producer_validate_visualize as validator


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def encoded_input(tmp_path):
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3) * 10
    codes = np.flip(rgb, axis=1).copy()
    (tmp_path / 'input_rgb_uint8.bin').write_bytes(rgb.tobytes())
    (tmp_path / 'runtime_input.bin').write_bytes(codes.tobytes())
    canonical = canonical_image_preprocessing_contract('classification', [2, 4])
    contract_sha = validator._canonical_json_sha256(canonical)
    numeric, numeric_sha = runtime_numeric_input_identity(
        backend='native_full_hailo8', task='classification',
        preprocessing_contract_sha256_value=contract_sha,
        runtime_input_name='classifier_full/input_layer1',
        runtime_input_shape=[2, 4, 3], runtime_input_dtype='uint8',
        runtime_input_layout='HWC', runtime_color_space='RGB',
        runtime_normalization='hef_quant_info_from_imagenet_mean_std',
    )
    image_sha = _sha(b'actual-source-image')
    payload = {
        'schema': 'onnx-splitpoint/native-full-input-dump', 'schema_version': 2,
        'model': 'classifier', 'backend': 'native_full_hailo8',
        'setup_id': 'h8', 'comparison_backend': 'hailo8', 'case': 'full',
        'task': 'classification', 'input_shape_hwc': [2, 4, 3],
        'runtime_input_file': '/original/remote/runtime_input.bin',
        'runtime_input_sha256': _sha(codes.tobytes()), 'runtime_input_bytes': 24,
        'runtime_input_shape': [2, 4, 3], 'runtime_input_dtype': 'uint8',
        'runtime_input_name': 'classifier_full/input_layer1',
        'runtime_input_layout': 'HWC', 'runtime_color_space': 'RGB',
        'runtime_normalization': 'hef_quant_info_from_imagenet_mean_std',
        'runtime_numeric_input_identity': numeric,
        'runtime_numeric_input_sha256': numeric_sha,
        'runtime_preprocessing_identity': canonical,
        'runtime_preprocessing_sha256': contract_sha,
        'preprocessing_contract': canonical,
        'preprocessing_contract_sha256': contract_sha,
        'image': '/original/image.jpeg', 'input_image': '/original/image.jpeg',
        'image_sha256': image_sha, 'input_image_sha256': image_sha,
        'input_dump': '/original/remote/input_rgb_uint8.bin',
        'input_dump_bytes': 24, 'input_dump_sha256': _sha(rgb.tobytes()),
        'canonical_rgb_sha256': _sha(rgb.tobytes()),
        'preprocess': {
            'layout': 'HWC', 'ort_model_scale': 'imagenet',
            'normalization': 'hef_quant_info_from_imagenet_mean_std',
            'source_domain': 'float_imagenet_mean_std', 'quantized_inputs': True,
            'preprocessing_contract': canonical,
            'preprocessing_contract_sha256': contract_sha,
        },
        'timing_input_binding': {
            'canonical_rgb_sha256': _sha(rgb.tobytes()),
            'input_image_sha256': image_sha,
            'preprocessing_contract_sha256': contract_sha,
        },
    }
    path = tmp_path / 'native_full_input_manifest.json'
    return path, payload, rgb


def _derive(fixture, *, shape=(1, 3, 2, 4), dtype=np.float32, scale='imagenet'):
    path, payload, _ = fixture
    path.write_text(json.dumps(payload))
    evidence = {}
    feed = validator._native_input_dump_feed(
        path, list(shape), image_scale=scale, target_dtype=dtype, evidence=evidence,
    )
    return feed, evidence


@pytest.mark.parametrize('layout,dtype', [('NCHW', np.float32), ('NHWC', np.float32), ('NCHW', np.float16)])
def test_bound_rgb_reference_matches_canonical_pixels_not_quantization_codes(encoded_input, layout, dtype):
    path, payload, rgb = encoded_input
    before = (path.parent / 'runtime_input.bin').read_bytes()
    shape = (1, 3, 2, 4) if layout == 'NCHW' else (1, 2, 4, 3)
    feed, evidence = _derive(encoded_input, shape=shape, dtype=dtype)
    real = rgb.astype(np.float32) / 255.0
    real = (real - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
    expected = (np.transpose(real, (2, 0, 1))[None] if layout == 'NCHW' else real[None]).astype(dtype)
    np.testing.assert_array_equal(feed, expected)
    assert evidence['status'] == 'passed'
    assert evidence['transformation_id'] == 'bound_canonical_rgb_to_onnx_reference_v1'
    assert evidence['canonical_rgb_sha256'] == payload['canonical_rgb_sha256']
    assert evidence['derived_reference_tensor_sha256'] == _sha(np.ascontiguousarray(expected).tobytes())
    assert (path.parent / 'runtime_input.bin').read_bytes() == before


@pytest.mark.parametrize('fault', [
    'rgb_missing', 'rgb_modified', 'rgb_size', 'rgb_sha', 'rgb_mirror', 'rgb_filename',
    'runtime_modified', 'runtime_sha', 'shape', 'model', 'setup', 'backend', 'task',
    'case', 'image', 'image_sha', 'timing_image', 'timing_rgb', 'contract',
    'contract_sha', 'numeric_sha', 'numeric_name', 'normalization',
    'normalization_missing', 'all_normalization_missing', 'all_domain_missing',
    'all_encoding_missing', 'source_domain', 'layout',
    'schema_version',
])
def test_incomplete_or_conflicting_binding_fails_before_reference(encoded_input, fault):
    path, payload, _ = encoded_input
    payload = copy.deepcopy(payload)
    rgb = path.parent / 'input_rgb_uint8.bin'
    if fault == 'rgb_missing': payload.pop('input_dump')
    elif fault == 'rgb_modified': rgb.write_bytes(b'x' * 24)
    elif fault == 'rgb_size': payload['input_dump_bytes'] = 23
    elif fault == 'rgb_sha': payload['input_dump_sha256'] = 'f' * 64
    elif fault == 'rgb_mirror': payload['canonical_rgb_sha256'] = 'f' * 64
    elif fault == 'rgb_filename': payload['input_dump'] = 'runtime_input.bin'
    elif fault == 'runtime_modified': (path.parent / 'runtime_input.bin').write_bytes(b'x' * 24)
    elif fault == 'runtime_sha': payload['runtime_input_sha256'] = 'f' * 64
    elif fault == 'shape': payload['input_shape_hwc'] = [4, 2, 3]
    elif fault == 'model': payload['model'] = 'different'
    elif fault == 'setup': payload.pop('setup_id')
    elif fault == 'backend': payload['backend'] = 'native_full_deepx'
    elif fault == 'task': payload['task'] = 'detection'
    elif fault == 'case': payload['case'] = 'b001'
    elif fault == 'image': payload['image'] = '/other/image.jpeg'
    elif fault == 'image_sha': payload['image_sha256'] = 'f' * 64
    elif fault == 'timing_image': payload['timing_input_binding']['input_image_sha256'] = 'f' * 64
    elif fault == 'timing_rgb': payload['timing_input_binding']['canonical_rgb_sha256'] = 'f' * 64
    elif fault == 'contract': payload['preprocessing_contract']['resize_interpolation'] = 'nearest'
    elif fault == 'contract_sha': payload['runtime_preprocessing_sha256'] = 'f' * 64
    elif fault == 'numeric_sha': payload['runtime_numeric_input_sha256'] = 'f' * 64
    elif fault == 'numeric_name': payload['runtime_numeric_input_identity']['runtime_input_name'] = 'different'
    elif fault == 'normalization': payload['runtime_normalization'] = 'embedded_hailo_quantization'
    elif fault == 'normalization_missing': payload.pop('runtime_normalization')
    elif fault in {'all_normalization_missing', 'all_domain_missing', 'all_encoding_missing'}:
        payload.pop('runtime_normalization')
        payload['preprocess'].pop('normalization')
        payload['runtime_numeric_input_identity'].pop('runtime_normalization')
        if fault != 'all_normalization_missing': payload['preprocess'].pop('source_domain')
        if fault == 'all_encoding_missing': payload['preprocess'].pop('quantized_inputs')
    elif fault == 'source_domain': payload['preprocess'].pop('source_domain')
    elif fault == 'layout': payload['runtime_input_layout'] = 'CHW'
    elif fault == 'schema_version': payload['schema_version'] = 1
    feed, evidence = _derive((path, payload, None))
    assert feed is None
    assert evidence['status'] == 'unavailable'
    assert 'derived_reference_tensor_sha256' not in evidence


@pytest.mark.parametrize('shape,dtype,scale', [
    ((1, 3, 4, 2), np.float32, 'imagenet'),
    ((1, 3, 2, 4), np.uint8, 'imagenet'),
    ((1, 3, 2, 4), np.float32, 'norm'),
    ((2, 4, 3), np.uint8, 'imagenet'),
])
def test_encoded_input_cannot_bypass_reference_shape_dtype_or_scale(encoded_input, shape, dtype, scale):
    feed, evidence = _derive(encoded_input, shape=shape, dtype=dtype, scale=scale)
    assert feed is None
    assert evidence['status'] == 'unavailable'


def test_duplicate_manifest_binding_keys_are_rejected(encoded_input):
    path, payload, _ = encoded_input
    path.write_text(json.dumps(payload)[:-1] + ', "canonical_rgb_sha256": "' + payload['canonical_rgb_sha256'] + '"}')
    evidence = {}
    assert validator._native_input_dump_feed(path, [1, 3, 2, 4], image_scale='imagenet', target_dtype=np.float32, evidence=evidence) is None
    assert evidence['status'] == 'unavailable'
