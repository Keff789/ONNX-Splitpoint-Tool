"""Execute corrected H8 input branches without accelerator inference/builds."""
from pathlib import Path
from types import SimpleNamespace
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from scripts import native_hailo_trt_fifo_from_benchmarkset as split
from scripts import smoke_hailo10_hef_runner as full
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract, preprocessing_contract_sha256,
    runtime_numeric_input_identity, runtime_numeric_input_identity_errors,
)


RGB = np.array([[[0, 128, 255], [255, 0, 128]],
                [[17, 63, 201], [234, 123, 9]]], dtype=np.uint8)


def imagenet(rgb):
    return (rgb.astype(np.float32)/np.float32(255)
            - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)


@pytest.mark.parametrize('shape', [(2, 2, 3), (1, 2, 2, 3), (3, 2, 2), (1, 3, 2, 2)])
def test_split_classifier_channelwise_input_and_detection_bytes(tmp_path, shape):
    source = tmp_path/'rgb.bin'; source.write_bytes(RGB.tobytes())
    prepared = SimpleNamespace(input_names=['images'],
        handle=SimpleNamespace(runtime_input_shapes={'images': shape}))
    before = source.read_bytes()
    for task, quantized in [('classification', False), ('detection', False), ('detection', True)]:
        actual = split._hailo8_python_input(prepared, tmp_path/'unused.png',
            quantized=quantized, task=task, preprocess_mode='resize',
            letterbox_pad_value=114, prepared_input_rgb=source)['images']
        expected = imagenet(RGB) if task == 'classification' else RGB if quantized else RGB.astype(np.float32)/np.float32(255)
        if shape in [(3, 2, 2), (1, 3, 2, 2)]: expected = expected.transpose(2, 0, 1)
        if len(shape) == 4: expected = expected[None]
        assert actual.tobytes() == np.ascontiguousarray(expected).tobytes()
        assert actual.flags.c_contiguous
        assert source.read_bytes() == before


def test_actual_embedded_cpp_pixel_encoder_preserves_detection_and_matches_classifier(tmp_path):
    compiler = shutil.which('g++')
    if not compiler: pytest.skip('local C++ compiler unavailable for the isolated host arithmetic oracle')
    function = re.search(r'static float hailo_rgb_float_input\([^}]+\{.*?\n\}', split.CPP_SOURCE, re.S).group()
    # Only this host arithmetic function is compiled; no Hailo/CUDA/engine code.
    source = tmp_path/'input.cpp'
    source.write_text('#include <cstdint>\n#include <cstddef>\n#include <iostream>\n' + function + '\nint main(int argc,char**) {\n'
        + 'uint8_t rgb[]={' + ','.join(map(str, RGB.ravel())) + '}; float out[12];\n'
        + 'for(size_t i=0;i<12;++i) out[i]=hailo_rgb_float_input(rgb[i],i%3,argc>1);\n'
        + 'std::cout.write(reinterpret_cast<char*>(out),sizeof(out));}\n')
    executable = tmp_path/'input'
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(executable)], check=True, capture_output=True)
    for classification in [False, True]:
        encoded = subprocess.run([str(executable)] + (['classification'] if classification else []), check=True, capture_output=True).stdout
        expected = imagenet(RGB) if classification else RGB.astype(np.float32)/np.float32(255)
        assert encoded == expected.tobytes()
    assert 'hailo_rgb_float_input(src[i], i % 3, classification_input_)' in split.CPP_SOURCE
    assert 'classification_input_ = o.task == "classification"' in split.CPP_SOURCE


def test_full_uses_canonical_meanstd_before_exact_sdk_quantizer(tmp_path, monkeypatch):
    image = tmp_path/'asymmetric.png'; Image.fromarray(RGB).save(image)
    real, raw, preprocess = full._image_tensor(image, (2, 2, 3), quantized=False,
        task='classification', preprocess_mode='resize', letterbox_pad=114)
    assert real.tobytes() == imagenet(RGB).tobytes()
    assert raw.tobytes() == RGB.tobytes()
    quant = SimpleNamespace(qp_scale=.01872340589761734, qp_zp=114.)
    info = SimpleNamespace(name='bound/input_layer1', format=SimpleNamespace(type='FormatType.UINT8'), quant_info=quant)
    calls = []
    def sdk_quantize(source, destination, count, actual_quant):
        assert actual_quant is quant
        assert source.dtype == np.float32 and source.flags.c_contiguous
        assert count == RGB.size
        calls.append(source.copy())
        # SDK stand-in output tests propagation; production uses the SDK method.
        destination[:] = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], np.uint8)
    monkeypatch.setitem(sys.modules, 'hailo_platform', SimpleNamespace(
        HEF=lambda path: SimpleNamespace(get_input_vstream_infos=lambda: [info]),
        HailoRTTransformUtils=SimpleNamespace(quantize_input_buffer=sdk_quantize)))
    encoded, encoding = full._quantize_hailo8_classifier_input(tmp_path/'bound.hef', real, input_names=['images'])
    assert len(calls) == 1 and calls[0].tobytes() == real.tobytes()
    assert encoded.dtype == np.uint8 and encoded.tobytes() == bytes(range(1, 13))
    assert raw.tobytes() == RGB.tobytes()
    assert encoding['normalization'] == 'hef_quant_info_from_imagenet_mean_std'
    assert encoding['source_domain'] == 'float_imagenet_mean_std'
    assert encoding['quant_info']['source'] == 'exact_loaded_hef_quant_info'
    preprocess.update(encoding, quantized_inputs=True)
    output, manifest = full._write_output_dump({'logits': np.array([[.2, .8]], np.float32)}, tmp_path/'dump',
        backend='native_full_hailo8', model='classifier', setup_id='orin_nx_hailo8_01', comparison_backend='hailo8',
        task='classification', image=image, input_hwc=raw, input_tensor=encoded, input_name='images', preprocess=preprocess)
    payload = json.loads(manifest.read_text())
    assert payload['runtime_normalization'] == encoding['normalization']
    assert Path(payload['runtime_input_file']).read_bytes() == encoded.tobytes()
    assert payload['preprocess']['source_domain'] == 'float_imagenet_mean_std'
    assert runtime_numeric_input_identity_errors(payload['runtime_numeric_input_identity'], payload['runtime_preprocessing_identity']) == []


@pytest.mark.parametrize('task,normalization,valid', [
    ('classification', 'hef_quant_info_from_imagenet_mean_std', True),
    ('detection', 'hef_quant_info_from_imagenet_mean_std', False),
    ('detection', 'embedded_hailo_quantization', True),
])
def test_uint8_numeric_domain_validator_limits_classifier_encoding(task, normalization, valid):
    preprocessing = canonical_image_preprocessing_contract(task, (2, 2))
    numeric, digest = runtime_numeric_input_identity(backend='native_full_hailo8', task=task,
        preprocessing_contract_sha256_value=preprocessing_contract_sha256(preprocessing),
        runtime_input_name='images', runtime_input_shape=[2, 2, 3], runtime_input_dtype='uint8',
        runtime_input_layout='HWC', runtime_color_space='RGB', runtime_normalization=normalization)
    errors = runtime_numeric_input_identity_errors(numeric, preprocessing)
    assert ('runtime_numeric_input_normalization_invalid' not in errors) is valid


@pytest.mark.parametrize('task,dtype,format,expected', [
    ('detection', 'uint8', 'uint8', ('uint8_0_255', 'uint8_0_255')),
    ('detection', 'uint8', 'float32', ('uint8_0_255', 'unit_float32')),
    ('detection', 'float32', 'float32', ('unit_float32', 'unit_float32')),
    ('classification', 'uint8', 'float32', ('uint8_0_255', 'imagenet_mean_std')),
    ('classification', 'float32', 'float32', ('imagenet_mean_std', 'imagenet_mean_std')),
])
def test_prepared_input_metadata_declares_storage_and_actual_runtime_domain(task, dtype, format, expected):
    actual = split._hailo8_input_domain(storage_dtype=dtype, hailo_format=format, completed_producer=False, task=task)
    assert (actual['source_domain'], actual['normalization']) == expected
    assert split._hailo8_input_domain(storage_dtype='uint8', hailo_format='auto', completed_producer=False, task=task) == {}


def test_runtime_mirrors_match():
    root = Path(__file__).resolve().parents[1]
    for name in ['native_hailo_trt_fifo_from_benchmarkset.py', 'smoke_hailo10_hef_runner.py']:
        assert (root/'scripts'/name).read_bytes() == (root/'onnx_splitpoint_tool/resources/remote_scripts'/name).read_bytes()


@pytest.mark.parametrize('damage', ['', 'binary', 'cpp', 'receipt', 'expected_sha'])
def test_normal_cpp_wrapper_reuse_verifies_existing_build_before_run(tmp_path, monkeypatch, damage):
    builder = tmp_path/'one-normal-build'; (builder/'build').mkdir(parents=True)
    executable = builder/'build/split_native_hailo_trt_fifo'; executable.write_bytes(b'normal-build-output')
    cache = builder/'build/CMakeCache.txt'; cache.write_bytes(b'normal-configure-output')
    (builder/'main.cpp').write_text(split.CPP_SOURCE)
    (builder/'CMakeLists.txt').write_text(split.CMAKE_TXT)
    expected = split._sha256_file(executable)
    receipt = {'status': 'ready', 'identity': {
        'generated_cpp_sha256': split._sha256_file(builder/'main.cpp'),
        'cmake_sha256': split._sha256_file(builder/'CMakeLists.txt'),
        'target_architecture': split.platform.machine(),
        'configure_options': ['-DCMAKE_BUILD_TYPE=Release'],
    }, 'native_executable_sha256': expected, 'cmake_cache_sha256': split._sha256_file(cache),
        'compiler_dispatch_count': 1}
    if damage == 'binary': executable.write_bytes(b'wrong-executable')
    if damage == 'cpp': (builder/'main.cpp').write_text(split.CPP_SOURCE+'\n// changed source\n')
    if damage == 'receipt': receipt['status'] = 'failed'
    if damage == 'expected_sha': expected = '0'*64
    source_config = builder/'native_fifo_config.json'; source_config.write_text(json.dumps({'wrapper_build': receipt}))
    source_before = source_config.read_bytes()
    case = tmp_path/'fresh-case'; case.mkdir()
    monkeypatch.setattr(split.subprocess, 'run', lambda *a, **kw: pytest.fail('reuse must not build or run'))
    if damage:
        with pytest.raises(RuntimeError, match='reuse_binding_missing_or_mismatched'):
            split._reuse_cpp_wrapper(executable, expected_sha256=expected, config_json=case/'native_fifo_config.json', cfg={})
        assert not (case/'native_fifo_config.json').exists()
    else:
        bound, source = split._reuse_cpp_wrapper(executable, expected_sha256=expected, config_json=case/'native_fifo_config.json', cfg={})
        assert source == builder
        assert bound['reused'] is True and bound['compiler_dispatch_count'] == 0
        assert bound['source_work_dir'] == str(builder)
        assert bound['native_executable_sha256'] == expected
    assert source_config.read_bytes() == source_before


def test_command_contract_uses_shared_actual_wrapper_sources_and_case_prepared_bytes(tmp_path, monkeypatch):
    builder = tmp_path/'one-build'; builder.mkdir()
    (builder/'main.cpp').write_text(split.CPP_SOURCE)
    (builder/'CMakeLists.txt').write_text(split.CMAKE_TXT)
    case = tmp_path/'case'; case.mkdir()
    # The case-local copies are not the sources of the shared executable.
    (case/'main.cpp').write_text('unused case-local source')
    (case/'CMakeLists.txt').write_text('unused case-local cmake')
    executable = builder/'binary'; executable.write_bytes(b'bound wrapper')
    hef = tmp_path/'bound.hef'; hef.write_bytes(b'bound hef')
    engine = tmp_path/'bound.engine'; engine.write_bytes(b'bound engine')
    image = tmp_path/'image.png'; Image.fromarray(RGB).save(image)
    prepared = case/'prepared.bin'; prepared.write_bytes(RGB.tobytes())
    args = SimpleNamespace(model_id='classifier', setup_id='orin_nx_hailo8_01',
        hw_arch='hailo8', precision='float32_layout_fp16', task='classification',
        preprocess_mode='resize', preprocess_mode_effective='resize', letterbox_pad_value=114,
        frames=1000, duration_s=0., warmup=100, repetitions=3, queue_depth=3,
        hailo_format='float32', copy_outputs=True, dump_outputs=True, dump_boundary=True,
        device_id='', build=False, wrapper_source_work=str(builder))
    monkeypatch.setattr(split, '_engine_boundary_contract', lambda path: {'boundary_layout_effective':'HWC'})
    contract = split._native_command_contract(bs=tmp_path, case='b001', args=args,
        hef=hef, engine=engine, image=image, work=case, executable=executable,
        prepared_input=prepared, prepared_input_shape=[2, 2, 3], prepared_input_name='images')
    assert contract['artifacts']['generated_cpp']['path'] == str(builder/'main.cpp')
    assert contract['artifacts']['cmake']['path'] == str(builder/'CMakeLists.txt')
    assert contract['artifacts']['prepared_input']['path'] == str(prepared)
    assert contract['prepared_input_contract']['normalization'] == 'imagenet_mean_std'
    assert contract['prepared_input_contract']['source_domain'] == 'uint8_0_255'
    assert prepared.read_bytes() == RGB.tobytes()
