"""Completion drops only the active N5000 reference option, retaining its source.

The generated CLI, hydration and strict reference loader are real. A controlled
stop at the accelerator session boundary prevents inference and measurement.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.split_export_runners import write_runner_onnxruntime
from tests import test_v2912_generic_split_completion as fixtures


def descriptor(module):
    preprocessing = {'schema': 'test-preprocessing', 'target_hw': [640, 640]}
    contract = {'task': 'detection', 'model': {'sha256': 'a' * 64},
                'dataset': {'image_count': 5000},
                'preprocessing': {'identity': preprocessing,
                                  'sha256': module._quality_contract_sha256(preprocessing)}}
    digest = module._quality_contract_sha256(contract)
    contract['quality_contract_sha256'] = digest
    return {'quality_contract': contract, 'quality_contract_sha256': digest,
            'reference_role': 'canonical_cpu_ort', 'source_evaluation_run_id': 'closed-original',
            'reference_path': '/closed-original/canonical_cpu_reference.json',
            'reference_sha256': 'b' * 64, 'reference_size_bytes': 123}


class StopBeforeAccelerator(BaseException):
    pass


@pytest.mark.parametrize('inherited', ['json', 'mapping', 'absent_192_compatible'])
def test_normal_generated_cli_reaches_runtime_after_strict_reference_loader(
        tmp_path, monkeypatch, inherited):
    generated = Path(write_runner_onnxruntime(str(tmp_path / 'generated')))
    monkeypatch.setattr(fixtures, 'TEMPLATE', generated)
    module = fixtures._module(monkeypatch)
    config, _ = fixtures._fixture(tmp_path)
    ref = descriptor(module)
    reference_value = json.dumps(ref) if inherited == 'json' else ref if inherited == 'mapping' else ''
    config['original_run_cfg'].update(
        existing_management_cpu_reference_json=reference_value,
        quality_source_only=inherited != 'absent_192_compatible',
        validation_images='original-frozen-N5000', validation_max_images=5000,
        quality_gate_json=json.dumps({'statistics': {'execution_location': 'central_management'}}))
    original = copy.deepcopy(config)
    config_path = tmp_path / 'completion.json'
    config_path.write_text(json.dumps(config))
    encoded = config_path.read_bytes()
    observations = []
    real_loader = module._existing_management_reference

    def observe_loader(value, **kwargs):
        admitted = real_loader(value, **kwargs)
        observations.append((value, kwargs, admitted))
        return admitted

    def forbidden(*args, **kwargs):
        raise AssertionError('No CPU reference, Quality, ordinary timing or model inference allowed')

    def stop_at_runtime(*args, **kwargs):
        assert observations == [('', {'model_sha256': '', 'variants': ['composed'],
                                     'central': False, 'image_count': 0,
                                     'task': config['task']}, None)]
        raise StopBeforeAccelerator('controlled accelerator initialization stop')

    monkeypatch.setattr(module, '_existing_management_reference', observe_loader)
    monkeypatch.setattr(module, '_create_session', forbidden)
    monkeypatch.setattr(module, '_bench', forbidden)
    monkeypatch.setattr(module, '_bench_streaming_pipeline', forbidden)
    monkeypatch.setattr(module, '_available_providers', lambda: ['CPUExecutionProvider'])
    monkeypatch.setattr(module, 'ort', SimpleNamespace(SessionOptions=lambda: SimpleNamespace()))
    # The native TensorRT suffix can initialize before the accelerator prefix.
    # Stop at either first runtime boundary, after the same strict loader check.
    monkeypatch.setattr(module, 'NativeTRTSession', stop_at_runtime)
    monkeypatch.setattr(module, 'DeepXSession', stop_at_runtime)
    # Restore the process environment after exercising the real CLI guard.
    monkeypatch.setenv('ONNX_SPLITPOINT_ARTIFACT_POLICY', 'cache_verify_only')
    monkeypatch.setattr(sys, 'argv', [str(generated), '--generic-completion-config', str(config_path),
                                    '--out-dir', str(tmp_path / 'completion-results'),
                                    '--generic-completion-repetition', '1'])
    with pytest.raises(StopBeforeAccelerator):
        module.main()
    assert config_path.read_bytes() == encoded
    assert json.loads(config_path.read_text()) == original
    assert config['functional_reference'] == original['functional_reference']
    assert not list(tmp_path.rglob('generic_completion_results.json'))


@pytest.mark.parametrize('mutation', ['none', 'model', 'variants', 'central', 'count',
                                     'contract_hash', 'descriptor_hash', 'reference_role',
                                     'source_run', 'reference_path', 'reference_sha',
                                     'reference_size', 'preprocessing'])
def test_ordinary_quality_reference_admission_remains_strict(monkeypatch, mutation):
    module = fixtures._module(monkeypatch)
    ref = descriptor(module)
    context = dict(model_sha256='a' * 64, variants=['composed'], central=True, image_count=5000)
    if mutation in {'model', 'variants', 'central', 'count'}:
        key, value = {'model': ('model_sha256', 'c' * 64), 'variants': ('variants', ['full']),
                      'central': ('central', False), 'count': ('image_count', 50)}[mutation]
        context[key] = value
    elif mutation == 'contract_hash':
        ref['quality_contract']['quality_contract_sha256'] = 'c' * 64
    elif mutation == 'descriptor_hash':
        ref['quality_contract_sha256'] = 'c' * 64
    elif mutation in {'reference_role', 'source_run', 'reference_path', 'reference_sha', 'reference_size'}:
        key = {'source_run': 'source_evaluation_run_id', 'reference_sha': 'reference_sha256',
               'reference_size': 'reference_size_bytes'}.get(mutation, mutation)
        ref[key] = ''
    elif mutation == 'preprocessing':
        contract = ref['quality_contract']
        contract['preprocessing']['identity']['target_hw'] = [320, 320]
        contract.pop('quality_contract_sha256')
        digest = module._quality_contract_sha256(contract)
        contract['quality_contract_sha256'] = ref['quality_contract_sha256'] = digest
    if mutation == 'none':
        assert module._existing_management_reference(ref, **context) == ref
    else:
        with pytest.raises(ValueError, match='reference .*contract mismatch'):
            module._existing_management_reference(ref, **context)
