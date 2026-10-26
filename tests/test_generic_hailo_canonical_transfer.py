"""Actual H10/236 typed metadata; no tensors, inference or hardware execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.runners import generic_passthrough as full


def actual_case():
    fixture = json.loads((Path(__file__).parent / 'fixtures/generic_canonical_transfer/h10_b236_observed_metadata.json').read_text())
    proof = fixture['proof']
    row = {'identity': deepcopy(proof['identity']),
           'quality_reference': {'request': deepcopy(fixture['provenance']['request'])},
           'generic_artifact_binding': {'artifacts': deepcopy(proof['runtime_artifacts']),
                                       'producer_source': deepcopy(fixture['provenance']['request'])}}
    observed = proof['runtime_context']['boundary_runtime']
    logical = proof['runtime_context']['logical_boundary']
    return row, proof, observed, logical['part1_outputs'], logical['part2_inputs'], observed['transfer_contract']['manifest_cut_name_map_p2_to_p1']


def validate(case):
    full._validate_canonical_cut_transfer(*case)


def test_actual_236_bound_own_hef_dequantization_retains_both_observed_domains():
    case = actual_case(); before = deepcopy(case)
    validate(case)
    assert case == before
    assert {s['dtype'] for s in case[3].values()} == {'uint8'}
    assert {case[4][name]['dtype'] for name in case[5]} == {'float32'}


@pytest.mark.parametrize('fault', [
    'false_flag', 'missing_flag', 'raw_mode', 'wrong_backend', 'wrong_api', 'missing_alias',
    'wrong_alias', 'swapped_alias', 'repeated_alias', 'missing_format', 'wrong_format', 'missing_quant',
    'zero_scale', 'infinite_scale', 'bad_zp', 'boolean_scale', 'already_dequantized',
    'wrong_hef', 'unbound_artifact', 'unbound_request', 'missing_receipt', 'wrong_receipt',
    'wrong_raw_type', 'wrong_target_type', 'wrong_shape', 'wrong_source_shape', 'wrong_onnx_shape',
    'missing_backend_source', 'wrong_backend_source', 'wrong_backend_size', 'missing_output',
])
def test_missing_contradictory_or_unbound_canonicalization_is_rejected(fault):
    case = actual_case(); row, proof, observed, outputs, inputs, cut = case
    io = observed['part1_io']; transfer = observed['transfer_contract']
    raw_name = io['hef_outputs'][0]; source = next(iter(outputs)); target = next(iter(cut))
    if fault == 'false_flag': transfer['hailo_exact_dequantization'] = False
    elif fault == 'missing_flag': transfer.pop('hailo_exact_dequantization')
    elif fault == 'raw_mode': transfer['boundary_mode'] = 'raw_uint8'
    elif fault == 'wrong_backend': proof['identity']['comparison_backend'] = 'deepx_m1'
    elif fault == 'wrong_api': io['runtime_api'] = 'vstreams'
    elif fault == 'missing_alias': io['output_aliases'].pop(raw_name)
    elif fault == 'wrong_alias': io['output_aliases'][raw_name] = 'foreign'
    elif fault == 'swapped_alias':
        second = io['hef_outputs'][1]
        io['output_aliases'][raw_name], io['output_aliases'][second] = io['output_aliases'][second], io['output_aliases'][raw_name]
    elif fault == 'repeated_alias': io['output_aliases'][raw_name] = io['output_aliases'][io['hef_outputs'][1]]
    elif fault == 'missing_format': io['runtime_output_formats'].pop(raw_name)
    elif fault == 'wrong_format': io['runtime_output_formats'][raw_name] = 'FLOAT32'
    elif fault == 'missing_quant': io['runtime_output_quantization'].pop(raw_name)
    elif fault == 'zero_scale': io['runtime_output_quantization'][raw_name]['scale'] = 0
    elif fault == 'infinite_scale': io['runtime_output_quantization'][raw_name]['scale'] = float('inf')
    elif fault == 'bad_zp': io['runtime_output_quantization'][raw_name]['zero_point'] = '5'
    elif fault == 'boolean_scale': io['runtime_output_quantization'][raw_name]['scale'] = True
    elif fault == 'already_dequantized': io['outputs_dequantized'] = True
    elif fault == 'wrong_hef': io['artifact'] = '/foreign/compiled.hef'
    elif fault == 'unbound_artifact': row['generic_artifact_binding']['artifacts']['part1_runtime']['sha256'] = 'f' * 64
    elif fault == 'unbound_request': row['generic_artifact_binding']['producer_source']['sha256'] = 'f' * 64
    elif fault == 'missing_receipt': proof['runtime_artifacts'].pop('part1_build_receipt'); row['generic_artifact_binding']['artifacts'].pop('part1_build_receipt')
    elif fault == 'wrong_receipt': proof['runtime_artifacts']['part1_build_receipt']['size_bytes'] = 0; row['generic_artifact_binding']['artifacts'] = deepcopy(proof['runtime_artifacts'])
    elif fault == 'wrong_raw_type': outputs[source]['dtype'] = 'int16'
    elif fault == 'wrong_target_type': inputs[target]['dtype'] = 'float16'
    elif fault == 'wrong_shape': inputs[target]['shape'] = [1, 512, 10, 40]
    elif fault == 'wrong_source_shape': outputs[source]['shape'] = [1, 512, 10, 40]
    elif fault == 'wrong_onnx_shape': io['onnx_output_shapes'][source] = [1, 512, 10, 40]
    elif fault == 'missing_output': io['canonical_outputs'].pop()
    else:
        sources = proof['runtime_source_files']; match = next(s for s in sources if s['path'].endswith('/backends/hailo_backend.py'))
        if fault == 'missing_backend_source': sources.remove(match)
        elif fault == 'wrong_backend_source': match['sha256'] = 'f' * 64
        else: match['size_bytes'] += 1
    with pytest.raises(ValueError, match='exact HEF'):
        validate(case)


def test_unchanged_direct_typed_transfer_does_not_require_a_hailo_relation():
    case = actual_case()
    for target, source in case[5].items():case[3][source] = deepcopy(case[4][target])
    case[2]['transfer_contract'].clear()
    validate(case)


def factory_projected_case(monkeypatch):
    """Captured outputs of the exact held Factory loops, whose real bytes passed."""
    from types import SimpleNamespace
    from onnx_splitpoint_tool import generic_only_plan
    fixture = json.loads((Path(__file__).parent / 'fixtures/generic_canonical_transfer/h10_b236_actual_factory_artifact_projection.json').read_text())
    assert fixture['all_actual_local_bytes_checked_by_unchanged_normal_factory']
    case = actual_case()
    assert fixture['original_artifacts'] == case[1]['runtime_artifacts']
    case[0]['generic_artifact_binding']['artifacts'] = deepcopy(fixture['projected_artifacts'])
    admitted = {v['local_path']: v for v in fixture['projected_artifacts'].values() if v.get('local_path')}
    def checked_actual_descriptor(value):
        expected = admitted.get(value.get('local_path'))
        if expected is None or any(value.get(k) != expected.get(k) for k in ('path', 'sha256', 'size_bytes')):
            raise ValueError('Local source missing or changed')
        return SimpleNamespace(stat=lambda: SimpleNamespace(st_size=expected['size_bytes']))
    monkeypatch.setattr(generic_only_plan, 'read_bound', checked_actual_descriptor)
    return case


def test_actual_factory_local_staging_projection_keeps_source_science(monkeypatch):
    case = factory_projected_case(monkeypatch); before = deepcopy(case)
    validate(case)
    assert case == before


@pytest.mark.parametrize('fault', ['sha', 'size', 'role', 'receipt', 'extra_science', 'stage_type',
                                 'engine_stage', 'relative_path', 'foreign_path', 'missing_local'])
def test_actual_factory_projection_rejects_changed_science_or_unvalidated_local_staging(monkeypatch, fault):
    case = factory_projected_case(monkeypatch); artifacts = case[0]['generic_artifact_binding']['artifacts']
    value = artifacts['source_part1_onnx']
    if fault == 'sha': value['sha256'] = 'f' * 64
    elif fault == 'size': value['size_bytes'] += 1
    elif fault == 'role': artifacts['foreign_role'] = artifacts.pop('source_part1_onnx')
    elif fault == 'receipt': artifacts['part1_build_receipt']['sha256'] = 'f' * 64
    elif fault == 'extra_science': value['layout'] = 'foreign'
    elif fault == 'stage_type': value['stage_local'] = 1
    elif fault == 'engine_stage': artifacts['engine'].update(stage_local=True,local_path='/foreign/engine')
    elif fault == 'relative_path': value['local_path'] = 'relative/part1.onnx'
    elif fault == 'foreign_path': value['local_path'] = '/foreign/part1.onnx'
    else: value.pop('local_path')
    with pytest.raises(ValueError): validate(case)


def test_staging_identity_uses_actual_normal_file_hash_and_size(tmp_path):
    import hashlib
    local = tmp_path / 'part1.onnx'; local.write_bytes(b'original scientific file')
    source = {'path': '/original/part1.onnx', 'sha256': hashlib.sha256(local.read_bytes()).hexdigest(), 'size_bytes': local.stat().st_size}
    projected = {**source, 'local_path': str(local), 'stage_local': True}
    full._validate_staged_artifact_identity({'source_part1_onnx': source}, {'source_part1_onnx': projected})
    local.write_bytes(b'changed scientific file!')
    with pytest.raises(ValueError): full._validate_staged_artifact_identity({'source_part1_onnx': source}, {'source_part1_onnx': projected})
