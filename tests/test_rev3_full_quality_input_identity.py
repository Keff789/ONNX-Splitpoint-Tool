from copy import deepcopy

from scripts.native_producer_final_report import _quality_binding_evidence


def full_row():
    return {
        'backend': 'native_full_tensorrt',
        'input_image_sha256': '1' * 64,
        'runtime_input_sha256': '2' * 64,
        'full_command_contract': {
            'artifacts': {'runtime_input_tensor': {'sha256': '2' * 64}},
        },
    }


def test_full_tensor_and_source_image_have_distinct_bound_identities():
    row = full_row()
    before = deepcopy(row)
    values, errors = _quality_binding_evidence(row)
    assert not errors
    assert values['runtime_input_sha256'] == '1' * 64
    assert values['runtime_prepared_tensor_sha256'] == '2' * 64
    assert row == before


def test_full_prepared_tensor_must_match_its_sealed_artifact():
    row = full_row()
    row['runtime_input_sha256'] = '3' * 64
    values, errors = _quality_binding_evidence(row)
    assert errors == ['runtime_prepared_tensor_sha256:contract_mismatch']
    assert values['runtime_prepared_tensor_sha256'] == ''


def test_source_image_alias_conflict_is_still_rejected():
    row = full_row()
    row['prepared_feed_source_image_sha256'] = '3' * 64
    _, errors = _quality_binding_evidence(row)
    assert errors == ['runtime_input_sha256:conflict']


def test_missing_tensor_artifact_preserves_legacy_alias_check():
    row = full_row()
    row['full_command_contract']['artifacts'] = {}
    _, errors = _quality_binding_evidence(row)
    assert errors == ['runtime_input_sha256:conflict']


def test_split_rows_preserve_existing_image_alias_contract():
    row = full_row()
    row['backend'] = 'hailo8_to_trt'
    _, errors = _quality_binding_evidence(row)
    assert errors == ['runtime_input_sha256:conflict']
