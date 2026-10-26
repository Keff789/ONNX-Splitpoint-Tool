"""Real own-H8 hint staging metadata; no artifact reads or measurements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.generic_split_manifest import validate_generic_split_manifest_staging

FIXTURE = Path(__file__).parent / 'fixtures/generic_split_manifest'


def actual_arguments():
    original = (FIXTURE / 'b140_hint_original.json').read_bytes()
    staged = (FIXTURE / 'b140_hint_staged.json').read_bytes()
    arguments = json.loads((FIXTURE / 'b140_hint_bindings.json').read_bytes())
    arguments['part1_build_receipt_bytes'] = (FIXTURE / 'b140_hint_part1_build_receipt.json').read_bytes()
    return original, staged, arguments


def test_actual_h8_original_hint_staging_preserves_bound_input_artifacts_and_manifests():
    original, staged, arguments = actual_arguments()
    before = deepcopy(arguments)
    proof = validate_generic_split_manifest_staging(original, staged, **arguments)
    assert proof['scientific_manifest_fields_unchanged'] is True
    assert arguments == before


def encode_receipt(receipt, arguments):
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    descriptor = {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    arguments['part1_build_receipt_bytes'] = raw
    arguments['original_artifacts']['part1_build_receipt'].update(descriptor)
    arguments['row']['generic_artifact_binding']['artifacts']['part1_build_receipt'].update(descriptor)


def encode_staged(document, arguments):
    raw = (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()
    arguments['staged_source'].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return raw


def change_feed(arguments, field, value, *, contract=False):
    for binding in (arguments['row']['input_binding'], arguments['row']['runtime_config']['input_binding']):
        binding['prepared_input_contract' if contract else 'prepared_input'][field] = value


@pytest.mark.parametrize('fault', [
    'missing_receipt', 'receipt_bytes', 'receipt_shape', 'receipt_bool_shape', 'receipt_arch',
    'receipt_source', 'receipt_hef', 'receipt_hef_size', 'receipt_net_name', 'receipt_cache_key',
    'receipt_preprocessing', 'input_name', 'input_layout', 'input_dtype', 'input_size',
    'input_shape', 'input_encoding', 'input_contiguity', 'input_normalization', 'input_preprocessing',
])
def test_own_h8_input_authority_rejects_missing_or_mismatching_bound_identity(fault):
    original, staged, arguments = actual_arguments()
    receipt = json.loads(arguments['part1_build_receipt_bytes'])
    if fault == 'missing_receipt':
        arguments['part1_build_receipt_bytes'] = None
    elif fault == 'receipt_bytes':
        arguments['part1_build_receipt_bytes'] += b'\n'
    elif fault.startswith('receipt_'):
        if fault == 'receipt_shape': receipt['cache_payload']['net_input_shapes'] = [1, 3, 320, 640]
        elif fault == 'receipt_bool_shape': receipt['cache_payload']['net_input_shapes'][0] = True
        elif fault == 'receipt_arch': receipt['hw_arch'] = receipt['cache_payload']['hw_arch'] = 'hailo10h'
        elif fault == 'receipt_source': receipt['source_onnx_sha256'] = 'f' * 64
        elif fault == 'receipt_hef': receipt['hef_sha256'] = 'f' * 64
        elif fault == 'receipt_hef_size': receipt['hef_size_bytes'] += 1
        elif fault == 'receipt_net_name': receipt['net_name'] = receipt['cache_payload']['net_name'] = 'other_part1_b1'
        elif fault == 'receipt_cache_key': receipt['cache_key'] = 'f' * 64
        else: receipt['preprocessing_contract_sha256'] = 'f' * 64
        if fault != 'receipt_cache_key':
            from onnx_splitpoint_tool.build_evidence import canonical_sha256
            receipt['cache_key'] = canonical_sha256(receipt['cache_payload'])
        encode_receipt(receipt, arguments)
    elif fault == 'input_name': change_feed(arguments, 'input_name', 'foreign_image')
    elif fault == 'input_layout':
        change_feed(arguments, 'layout', 'CHW'); change_feed(arguments, 'layout', 'CHW', contract=True)
    elif fault == 'input_dtype':
        change_feed(arguments, 'dtype', 'float32'); change_feed(arguments, 'dtype', 'float32', contract=True)
    elif fault == 'input_size': change_feed(arguments, 'size_bytes', 1)
    elif fault == 'input_shape':
        change_feed(arguments, 'shape', [320, 640, 3]); change_feed(arguments, 'shape', [320, 640, 3], contract=True)
        change_feed(arguments, 'size_bytes', 320 * 640 * 3)
    elif fault == 'input_encoding': change_feed(arguments, 'format', 'npy')
    elif fault == 'input_contiguity': change_feed(arguments, 'c_contiguous', False)
    elif fault == 'input_normalization': change_feed(arguments, 'normalization', 'identity', contract=True)
    else:
        preprocessing = deepcopy(arguments['row']['input_binding']['prepared_input_contract']['image_preprocessing_contract'])
        preprocessing['image_scale'] = 'raw'
        change_feed(arguments, 'image_preprocessing_contract', preprocessing, contract=True)
    with pytest.raises(ValueError, match='split-manifest staging'):
        validate_generic_split_manifest_staging(original, staged, **arguments)


@pytest.mark.parametrize('fault', [
    'hint_receipt_path', 'hint_provenance_path', 'fake_current_h8_build', 'wrong_h8_locator',
    'unused_h10_cache', 'unused_h10_failure', 'unused_h10_unknown_field',
    'manifest_input_name', 'missing_native_input_authority', 'science_output', 'synthetic_deepx',
    'measured_hef', 'measured_graph', 'missing_original_receipt_authority',
])
def test_import_hint_acceptance_does_not_relax_other_manifest_or_artifact_guards(fault):
    original, staged, arguments = actual_arguments()
    current = json.loads(staged)
    h8 = current['hailo']['hefs']['hailo8']
    if fault == 'hint_receipt_path': h8['part1_original_reuse']['receipt'] = '../foreign.json'
    elif fault == 'hint_provenance_path': h8['part1_original_reuse']['provenance'] = 'foreign.json'
    elif fault == 'fake_current_h8_build': h8['part1_build'] = {'ok': True}
    elif fault == 'wrong_h8_locator': h8['part1'] = 'hailo/hailo10/part1/compiled.hef'
    elif fault == 'unused_h10_cache': current['hailo']['hefs']['hailo10']['part1_build']['cache_key'] = 'f' * 64
    elif fault == 'unused_h10_failure': current['hailo']['hefs']['hailo10']['part1_build']['mapping_failed'] = True
    elif fault == 'unused_h10_unknown_field': current['hailo']['hefs']['hailo10']['part1_build']['unknown'] = False
    elif fault == 'manifest_input_name': current['orig_inputs'] = ['foreign_image']
    elif fault == 'missing_native_input_authority': current.pop('part1_external_inputs')
    elif fault == 'science_output': current['orig_outputs'] = ['foreign_output']
    elif fault == 'synthetic_deepx': current['deepx'] = {}
    elif fault == 'measured_hef': arguments['row']['generic_artifact_binding']['artifacts']['part1_runtime']['sha256'] = 'f' * 64
    elif fault == 'measured_graph': arguments['producer_model']['part1_onnx_sha256'] = 'f' * 64
    else: arguments['original_artifacts'].pop('part1_build_receipt')
    with pytest.raises(ValueError, match='split-manifest staging'):
        validate_generic_split_manifest_staging(original, encode_staged(current, arguments), **arguments)


def test_actual_relation_uses_only_bound_metadata_and_preserves_both_documents(monkeypatch):
    from onnx_splitpoint_tool import build_evidence, preprocessing_contract
    assert build_evidence is not None and preprocessing_contract is not None
    original, staged, arguments = actual_arguments()
    before = deepcopy(arguments)
    def no_payloads(*_args, **_kwargs):
        pytest.fail('pure staging relation must not open artifacts or payloads')
    with monkeypatch.context() as guarded:
        guarded.setattr('builtins.open', no_payloads)
        guarded.setattr(Path, 'open', no_payloads)
        proof = validate_generic_split_manifest_staging(original, staged, **arguments)
    native = proof['native_h8_input_authority']
    assert native['authority'] == 'bound_own_hailo8_build_receipt'
    assert native['net_input_shapes'] == [1, 3, 640, 640] and native['input_name'] == 'images'
    assert native['receipt'] == arguments['original_artifacts']['part1_build_receipt']
    assert native['hef'] == arguments['original_artifacts']['part1_runtime']
    assert arguments == before
    assert proof['original_manifest']['sha256'] == hashlib.sha256(original).hexdigest()
    assert proof['staged_manifest']['sha256'] == hashlib.sha256(staged).hexdigest()
    assert proof['omitted_unexecuted_backend_metadata'][0]['backend_metadata'] == 'deepx'
    assert proof['retained_historical_build_metadata'][0]['backend_metadata'] == 'hailo10'
    assert proof['retained_historical_build_metadata'][0]['current_compile_claimed'] is False
    assert 'part1_build' not in json.loads(staged)['hailo']['hefs']['hailo8']


@pytest.mark.parametrize('arch', ['hailo8', 'hailo10'])
def test_original_positive_reuse_keeps_its_history_without_inventing_current_build(arch):
    original, staged, arguments = actual_arguments()
    document = json.loads(original)
    old = document['hailo']['hefs'][arch]['part1_build']
    source = old['source_onnx_path']
    reuse = {'build_receipt_path': old['build_receipt_path'], 'source_onnx_path': source,
             'compiler_onnx_path': source, 'cache_key': old['cache_key'],
             'ok': True, 'cache_hit': True, 'skipped': True,
             'context_mode': 'skipped', 'calib_source': 'validated_existing_artifact', 'elapsed_s': .5,
             **{key: False for key in ['mapping_failed', 'multi_context_used', 'single_context_failed',
                                      'single_context_used', 'timed_out', 'watchdog_expired']}}
    document['hailo']['hefs'][arch]['part1_build'] = reuse
    raw = (json.dumps(document, sort_keys=True, indent=2) + '\n').encode()
    descriptor = {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    arguments['original_source'].update(descriptor)
    arguments['row']['generic_artifact_binding']['artifacts']['boundary_metadata'].update(descriptor)
    proof = validate_generic_split_manifest_staging(raw, staged, **arguments)
    assert proof['scientific_manifest_fields_unchanged'] is True
    assert 'part1_build' not in json.loads(staged)['hailo']['hefs']['hailo8']
    if arch == 'hailo8': assert proof['historical_build_reuse'] == []
    else: assert proof['retained_historical_build_metadata'][0]['historical_reuse_observation'] is True
