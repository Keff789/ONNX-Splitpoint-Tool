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
