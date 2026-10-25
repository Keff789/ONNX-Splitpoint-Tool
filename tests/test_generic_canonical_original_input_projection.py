"""Explicit SourceQuality projection and canonical bytes; no inference/hardware."""
from copy import deepcopy
import json

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_passthrough as full
from tests.test_generic_passthrough import canonical


def legacy_case(canonical, count):
    module, source_row, source_proof, arrays = canonical
    row, proof = deepcopy(source_row), deepcopy(source_proof)
    names = [f'cut_{index}' for index in range(count)]
    outputs = {name: {'shape': [1, index + 2, 4, 4], 'dtype': 'float32'}
               for index, name in enumerate(names)}
    inputs = {**outputs, 'images': {'shape': list(arrays[0].shape), 'dtype': 'float32'}}
    transfer = {'boundary_mode': 'canonical_float32', 'part1_cut_names': names,
                'part2_cut_names': names, 'manifest_cut_name_map_p2_to_p1': {name: name for name in names}}
    boundary = {'schema_version': 1, 'minimum_boundary_tensors': count,
                'part1_outputs': outputs, 'part2_inputs': inputs,
                'part2_to_part1': {**{name: name for name in names}, 'images': names[0]},
                'passthrough_inputs': {}, 'transfer_contract': transfer}
    row['boundary_observation'] = deepcopy(boundary)
    row['generic_artifact_binding'] = {'boundary_contract': deepcopy(boundary),
                                     'producer_source': deepcopy(row['quality_reference']['request'])}
    row['split_manifest'] = {'path': '/SYNTHETIC/selected_manifest.json', 'sha256': 'a' * 64}
    row['input_binding']['prepared_input_contract'] = {
        'image_preprocessing_contract': deepcopy(proof['image_preprocessing_contract'])}
    proof.update(image_count=2, implementation_runner={'sha256': full._LEGACY_STALE_MAP_RUNNER})
    proof['runtime_context'] = {
        'boundary_observation_count': 2,
        'logical_boundary': {'part1_outputs': deepcopy(outputs), 'part2_inputs': deepcopy(inputs)},
        'boundary_runtime': {'part1_io': {'canonical_outputs': names},
            'part2_io': {'inputs': deepcopy(inputs)},
            **{key: deepcopy(boundary[key]) for key in (
                'part2_to_part1', 'passthrough_inputs', 'transfer_contract')}}}
    manifest = {'part1_cut_names': names, 'part2_cut_names': names,
                'orig_inputs': ['images'], 'io': {'part2_external_inputs': [*names, 'images']}}
    return module, row, proof, arrays, manifest


@pytest.mark.parametrize('count', [2, 4])
def test_source_projection_preserves_original_and_uses_two_actual_canonical_images(canonical, count):
    module, row, proof, arrays, manifest = legacy_case(canonical, count)
    original = deepcopy(row)
    projected = full.prepare_canonical_row(row, proof, split_manifest=manifest)
    assert row == original
    assert projected['canonical_original_input_projection']['original_source_boundary'] == row['boundary_observation']
    assert projected['boundary_observation']['passthrough_inputs'] == {'images': 'images'}
    assert 'images' not in projected['boundary_observation']['part2_to_part1']
    full.validate_quality_bindings(projected, proof, split_manifest=manifest)
    actual = full.prepare_full_feeds(projected, module._load_image_as_nchw)
    for index in range(2):
        np.testing.assert_array_equal(actual[index]['images'], arrays[index])
    assert not np.array_equal(actual[0]['images'], actual[1]['images'])


@pytest.mark.parametrize('fault', ['runner', 'manifest', 'logical_shape', 'snapshot_shape', 'original_boundary', 'count'])
def test_unproved_source_projection_rejects(canonical, fault):
    _, row, proof, _, manifest = legacy_case(canonical, 4)
    if fault == 'runner':
        proof['implementation_runner']['sha256'] = 'f' * 64
    elif fault == 'manifest':
        manifest['orig_inputs'] = ['another_image']
    elif fault == 'logical_shape':
        proof['runtime_context']['logical_boundary']['part2_inputs']['images']['shape'] = [1, 3, 32, 32]
    elif fault == 'snapshot_shape':
        proof['per_image'][1]['prepared_inputs']['images']['shape'] = [1, 3, 32, 32]
    elif fault == 'original_boundary':
        row['boundary_observation']['part2_to_part1']['images'] = 'foreign'
    else:
        proof['image_count'] = 5000
    with pytest.raises(ValueError):
        full.prepare_canonical_row(row, proof, split_manifest=manifest)


@pytest.mark.parametrize('fault', ['missing_bindings', 'source', 'normalization', 'second_image', 'projection'])
def test_projected_row_still_requires_exact_source_normalization_and_mixed_image(canonical, fault):
    _, row, proof, _, manifest = legacy_case(canonical, 2)
    projected = full.prepare_canonical_row(row, proof, split_manifest=manifest)
    if fault == 'missing_bindings':
        projected.pop('canonical_full_input_bindings')
    elif fault == 'source':
        projected['canonical_full_input_bindings'][0]['producer_source']['sha256'] = 'f' * 64
    elif fault == 'normalization':
        projected['canonical_full_input_bindings'][0]['preprocessing_contract']['image_scale'] = 'raw'
    elif fault == 'second_image':
        projected['canonical_full_input_bindings'][1]['inputs']['images']['tensor_sha256'] = 'f' * 64
    else:
        projected['canonical_original_input_projection']['source_quality_request']['sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        full.validate_quality_bindings(projected, proof, split_manifest=manifest)


def test_old_measurement_row_cannot_claim_canonical_passthrough_after_execution(canonical):
    _, row, proof, _, manifest = legacy_case(canonical, 4)
    row.pop('canonical_full_input_bindings')
    row['boundary_observation']['passthrough_inputs'] = {'images': 'images'}
    with pytest.raises(ValueError, match='two bound canonical'):
        full.validate_quality_bindings(row, proof, split_manifest=manifest)


def test_feature_only_projection_is_unchanged(canonical):
    _, row, proof, _, manifest = legacy_case(canonical, 2)
    manifest['io']['part2_external_inputs'].remove('images')
    assert full.prepare_canonical_row(row, proof, split_manifest=manifest) == row


def manifest_passthrough_case():
    from tests.test_generic_split_manifest import bindings, rewritten, rewritten_original
    original, staged, args = bindings()
    original, staged = json.loads(original), json.loads(staged)
    image_spec = {'shape': [1, 3, 640, 640], 'dtype': 'float32'}
    for document in (original, staged):
        document['io']['part2_external_inputs'].append('images')
    boundary = args['row']['boundary_observation']
    boundary['part2_inputs']['images'] = image_spec
    boundary['passthrough_inputs'] = {'images': 'images'}
    args['row']['generic_artifact_binding']['boundary_contract'] = deepcopy(boundary)
    args['boundary_inputs'].append({'name': 'images', 'layout': 'NCHW', **image_spec})
    request = {'path': '/SYNTHETIC/normal_quality_request.json', 'sha256': 'a' * 64}
    args['row']['quality_reference'] = {'request': request}
    args['row']['generic_artifact_binding']['producer_source'] = request
    feed = args['row']['input_binding']
    canonical = []
    for index in range(2):
        canonical.append({'schema': full.SCHEMA, 'image_id': str(index) + '.jpg',
            'producer_source': deepcopy(request),
            'source_image': {'path': '/SYNTHETIC/' + str(index) + '.jpg',
                'sha256': feed['source_image_sha256'] if index == 0 else 'b' * 64},
            'prepared_input_sha256': feed['prepared_input']['sha256'] if index == 0 else 'c' * 64,
            'preprocessing_contract': deepcopy(feed['prepared_input_contract']['image_preprocessing_contract']),
            'inputs': {'images': {**image_spec, 'tensor_sha256': str(index) * 64}}})
    args['row']['canonical_full_input_bindings'] = canonical
    return rewritten_original(original, args), rewritten(staged, args), args


def test_complete_manifest_cut_plus_canonical_original_image_preserves_all_artifacts():
    from onnx_splitpoint_tool.generic_split_manifest import validate_generic_split_manifest_staging
    original, staged, args = manifest_passthrough_case()
    before = deepcopy(args)
    result = validate_generic_split_manifest_staging(original, staged, **args)
    assert args == before
    assert result['canonical_original_input_bindings'] == args['row']['canonical_full_input_bindings']
    assert result['scientific_manifest_fields_unchanged']
    assert set(result['unchanged_artifact_bindings']) == set(args['original_artifacts'])


@pytest.mark.parametrize('fault', ['old_stale_row', 'missing', 'wrong_source', 'wrong_norm',
                                 'same_image', 'feature_alias', 'feature_science', 'missing_byte_hash'])
def test_manifest_passthrough_does_not_relax_existing_input_science(fault):
    from onnx_splitpoint_tool.generic_split_manifest import validate_generic_split_manifest_staging
    original, staged, args = manifest_passthrough_case()
    row = args['row']
    boundary = row['boundary_observation']
    if fault == 'old_stale_row':
        boundary['part2_to_part1']['images'] = next(iter(boundary['part1_outputs']))
        boundary['passthrough_inputs'] = {}
        row.pop('canonical_full_input_bindings')
    elif fault == 'missing':
        row.pop('canonical_full_input_bindings')
    elif fault == 'wrong_source':
        row['canonical_full_input_bindings'][1]['producer_source']['sha256'] = 'f' * 64
    elif fault == 'wrong_norm':
        row['canonical_full_input_bindings'][1]['preprocessing_contract']['image_scale'] = 'raw'
    elif fault == 'same_image':
        row['canonical_full_input_bindings'][1]['image_id'] = row['canonical_full_input_bindings'][0]['image_id']
    elif fault == 'feature_alias':
        boundary['passthrough_inputs']['images'] = next(iter(boundary['part1_outputs']))
    elif fault == 'feature_science':
        boundary['part1_outputs'][next(iter(boundary['part1_outputs']))]['dtype'] = 'float16'
    else:
        row['canonical_full_input_bindings'][0]['inputs']['images'].pop('tensor_sha256')
    row['generic_artifact_binding']['boundary_contract'] = deepcopy(boundary)
    with pytest.raises(ValueError):
        validate_generic_split_manifest_staging(original, staged, **args)
