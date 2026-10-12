"""CPU-only coverage of the historical H8 exact-byte domain projection."""
import ast
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import generic_completed_task as completed


@pytest.fixture
def legacy_input(tmp_path):
    payload = bytes(range(12))
    local = tmp_path / 'prepared.bin'
    local.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    native = tmp_path / 'native_fifo_results.json'
    native.write_text('{"immutable":true}\n')
    image_sha = 'a' * 64
    contract = {
        'format': 'raw_runtime_tensor', 'name': 'part1/input_layer1',
        'shape': [2, 2, 3], 'dtype': 'uint8', 'layout': 'HWC',
        'task': 'detection', 'preprocess': 'opencv_letterbox_rgb_uint8',
        'preprocess_mode_effective': 'letterbox',
        'shared_rgb_source': '/original/shared_rgb_uint8.bin',
        'shared_rgb_source_sha256': sha,
        'source_image': '/original/image.jpg', 'source_image_sha256': image_sha,
    }
    command = {
        'backend': 'hailo8_to_trt', 'contract_sha256': 'b' * 64,
        'input_image_sha256': image_sha,
        'prepared_input_contract': deepcopy(contract),
        'artifacts': {'prepared_input': {'path': '/original/prepared.bin', 'sha256': sha}},
    }
    row = {
        'task': 'detection', 'native_source': str(native),
        'native_command_contract_sha256': command['contract_sha256'],
        'native_quality_binding_sha256': 'c' * 64,
        'input_binding': {
            'source_image': contract['source_image'], 'source_image_sha256': image_sha,
            'prepared_input_contract': deepcopy(contract),
            'entries': [{'local_path': str(local)}],
        },
    }
    prepared = {'path': '/original/prepared.bin', 'sha256': sha, 'shape': [2, 2, 3],
                'dtype': 'uint8', 'format': 'raw', 'input_name': 'images'}
    return row, command, prepared


@pytest.fixture(scope='module')
def normal_input_validator():
    # Run only the actual template's pure pre-SDK validator in this CPU test.
    template = Path(completed.__file__).parents[1] / 'resources/templates/run_split_onnxruntime.py.txt'
    text = template.read_text()
    start = text.index('def _completion_validate_prepared_input_contract(')
    end = text.index('\ndef ', start + 1)
    tree = ast.parse(text[start:end])
    namespace = {'Mapping': Mapping}
    exec(compile(tree, str(template), 'exec'), namespace)
    return namespace['_completion_validate_prepared_input_contract']


def test_projects_domain_with_exact_source_proof_and_no_byte_transform(legacy_input, normal_input_validator):
    row, command, prepared = legacy_input
    unchanged_row, unchanged_command = deepcopy(row), deepcopy(command)
    local = Path(row['input_binding']['entries'][0]['local_path'])
    original_bytes = local.read_bytes()
    native_bytes = Path(row['native_source']).read_bytes()
    config = {'task': 'detection', 'input_binding': {
        **row['input_binding'], 'prepared_input': prepared, 'runtime_input_format': 'uint8'}}
    with pytest.raises(ValueError, match='explicit numeric domain/normalization'):
        normal_input_validator(config, 'hailo8')
    completed._bind_prepared_numeric_domain(row, command, prepared, 'uint8')
    normal_input_validator(config, 'hailo8')
    assert prepared['source_domain'] == 'uint8_0_255'
    proof = prepared['source_domain_binding']
    assert proof['numeric_transform_applied'] is False
    assert proof['prepared_input_sha256'] == proof['shared_rgb_source']['sha256']
    assert proof['native_command_contract_sha256'] == command['contract_sha256']
    assert proof['native_quality_binding_sha256'] == row['native_quality_binding_sha256']
    assert proof['native_source']['sha256'] == hashlib.sha256(native_bytes).hexdigest()
    assert row == unchanged_row and command == unchanged_command
    assert local.read_bytes() == original_bytes
    assert Path(row['native_source']).read_bytes() == native_bytes


@pytest.mark.parametrize('declaration', ['prepared', 'contract', 'nested', 'entry'])
def test_preserves_existing_explicit_numeric_metadata(legacy_input, declaration):
    row, command, prepared = legacy_input
    contract = row['input_binding']['prepared_input_contract']
    if declaration == 'prepared':
        prepared['source_domain'] = 'uint8_0_255'
    elif declaration == 'contract':
        contract['source_domain'] = 'uint8_0_255'
    elif declaration == 'nested':
        contract['source_input_contract'] = {'normalization': 'embedded_dxcom_preprocessing'}
    else:
        contract['entries'] = [{'normalization': 'hef_quant_info_from_unit_float32'}]
    before = deepcopy(prepared)
    completed._bind_prepared_numeric_domain(row, command, prepared, 'uint8')
    assert prepared == before
    assert 'source_domain_binding' not in prepared


@pytest.mark.parametrize('fault', [
    'missing_preprocess', 'unknown_preprocess', 'foreign_contract', 'shared_sha',
    'missing_shared_source', 'backend_h10', 'backend_dx', 'float_format',
    'classification', 'prepared_dtype', 'shape', 'artifact_path', 'artifact_sha',
    'image_sha', 'command_sha', 'missing_local', 'changed_bytes', 'wrong_size',
])
def test_rejects_unproven_legacy_domain(legacy_input, fault):
    row, command, prepared = legacy_input
    contract = row['input_binding']['prepared_input_contract']
    runtime_format = 'uint8'
    if fault == 'missing_preprocess':
        contract.pop('preprocess')
        command['prepared_input_contract'] = deepcopy(contract)
    elif fault == 'unknown_preprocess':
        contract['preprocess'] = 'unspecified_rgb'
        command['prepared_input_contract'] = deepcopy(contract)
    elif fault == 'foreign_contract':
        command['prepared_input_contract']['source_image'] = '/other/image.jpg'
    elif fault == 'shared_sha':
        contract['shared_rgb_source_sha256'] = 'd' * 64
        command['prepared_input_contract'] = deepcopy(contract)
    elif fault == 'missing_shared_source':
        contract.pop('shared_rgb_source')
        command['prepared_input_contract'] = deepcopy(contract)
    elif fault.startswith('backend_'):
        command['backend'] = 'hailo10h_to_trt' if fault == 'backend_h10' else 'deepx_to_trt'
    elif fault == 'float_format':
        runtime_format = 'float32'
    elif fault == 'classification':
        row['task'] = 'classification'
    elif fault == 'prepared_dtype':
        prepared['dtype'] = 'float32'
    elif fault == 'shape':
        prepared['shape'] = [3, 2, 2]
    elif fault == 'artifact_path':
        command['artifacts']['prepared_input']['path'] = '/other/prepared.bin'
    elif fault == 'artifact_sha':
        command['artifacts']['prepared_input']['sha256'] = 'd' * 64
    elif fault == 'image_sha':
        command['input_image_sha256'] = 'd' * 64
    elif fault == 'command_sha':
        command['contract_sha256'] = 'd' * 64
    elif fault == 'missing_local':
        Path(row['input_binding']['entries'][0]['local_path']).unlink()
    elif fault == 'changed_bytes':
        Path(row['input_binding']['entries'][0]['local_path']).write_bytes(b'\xff' * 12)
    elif fault == 'wrong_size':
        Path(row['input_binding']['entries'][0]['local_path']).write_bytes(b'\x00' * 11)
    with pytest.raises(ValueError, match='numeric domain|bytes differ'):
        completed._bind_prepared_numeric_domain(row, command, prepared, runtime_format)
    assert 'source_domain' not in prepared
    assert 'source_domain_binding' not in prepared


@pytest.mark.parametrize('key', ['normalization', 'source_domain'])
def test_rejects_conflicting_explicit_numeric_contracts(legacy_input, key):
    row, command, prepared = legacy_input
    prepared[key] = 'first_domain'
    row['input_binding']['prepared_input_contract'][key] = 'other_domain'
    with pytest.raises(ValueError, match='numeric contract conflict'):
        completed._bind_prepared_numeric_domain(row, command, prepared, 'uint8')
    assert 'source_domain_binding' not in prepared
