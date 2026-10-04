"""Dispatcher labels are not precision; sealed runtime contracts remain strict."""
import copy

import pytest

from scripts import native_producer_final_report as report
from onnx_splitpoint_tool.native_job_identity import complete_historical_identity
from onnx_splitpoint_tool.native_split_quality import bind_quality_to_native_split
from tests.test_v269f_native_split_final_energy_integrity import _quality_first_row


def _observation(tmp_path):
    row, _, _ = _quality_first_row(tmp_path)
    context = {key: row[key] for key in ('backend', 'model', 'case', 'setup_id', 'comparison_backend')}
    context.update(execution_mode='native_split',
        variant='hailo8_yolo26s_selected_single_part2_input_hailo8_setup_yolo26s_b038')
    return row, context


def test_dispatch_label_uses_existing_command_and_consumer_without_mutation(tmp_path):
    row, context = _observation(tmp_path)
    before = copy.deepcopy((row, context))
    result = report._complete_bound_native_identity(row, [context])
    assert not result.get('identity_conflicts')
    assert result == row
    assert (row, context) == before


def test_distinct_dispatch_contexts_remain_unresolved(tmp_path):
    row, context = _observation(tmp_path)
    other = {**context, 'variant': context['variant'] + '_other_selection'}
    result = report._complete_bound_native_identity(row, [context, other])
    assert result['identity_conflicts'] == ['precision']
    assert result == complete_historical_identity(row, [context, other])


def test_exact_duplicate_dispatch_contexts_add_no_ambiguity(tmp_path):
    row, context = _observation(tmp_path)
    assert report._complete_bound_native_identity(row, [context, copy.deepcopy(context)]) == row


@pytest.mark.parametrize('field,value', [
    ('precision', 'uint8_cast_fp16'), ('contract', 'uint8_cast_fp16'),
    ('variant', 'uint8_cast_fp16'), ('variant', 'uint8_cast_broken'),
    ('variant', 'float32_layout_fp16'), ('variant', 'fp16'),
    ('precision', False), ('contract', 0),
    ('runtime_precision_identity', 'uint8_cast_fp16'),
    ('execution_precision', 'uint8_cast_fp16'),
])
def test_explicit_or_precision_like_plan_values_remain_conflicts(tmp_path, field, value):
    row, context = _observation(tmp_path)
    context[field] = value
    result = report._complete_bound_native_identity(row, [context])
    assert result['identity_conflicts'] == ['precision']
    assert result['precision'] == 'uint8_dequant_fp16'


@pytest.mark.parametrize('field,value', [
    ('precision', 'uint8_dequant_fp16'), ('contract', 'uint8_dequant_fp16'),
    ('variant', 'uint8_dequant_fp16'),
])
def test_existing_precision_aliases_keep_their_normal_match(tmp_path, field, value):
    row, context = _observation(tmp_path)
    context[field] = value
    assert report._complete_bound_native_identity(row, [context]) == row


@pytest.mark.parametrize('mutation', ['command_sha', 'request', 'setup', 'backend',
    'precision', 'selection', 'missing_binding', 'existing_conflict', 'context_setup'])
def test_weak_or_conflicting_evidence_cannot_supply_precision(tmp_path, mutation):
    row, context = _observation(tmp_path)
    if mutation == 'command_sha':
        row['native_command_contract']['contract_sha256'] = 'f' * 64
    elif mutation == 'request':
        row['source_request_sha256'] = 'f' * 64
    elif mutation in ('setup', 'backend', 'precision'):
        field, value = {'setup': ('setup_id', 'foreign_device'),
                        'backend': ('backend', 'deepx_to_trt'),
                        'precision': ('precision', 'uint8_cast_fp16')}[mutation]
        row[field] = value
    elif mutation == 'selection':
        row['native_split_quality_binding']['preselection']['precision'] = 'uint8_cast_fp16'
    elif mutation == 'missing_binding':
        row.pop('native_split_quality_binding')
    elif mutation == 'existing_conflict':
        row['identity_conflicts'] = ['precision']
    else:
        context['setup_id'] = 'foreign_device'
    before = copy.deepcopy((row, context))
    result = report._complete_bound_native_identity(row, [context])
    assert result == complete_historical_identity(row, [context])
    if mutation == 'backend':
        assert bind_quality_to_native_split(native_row=result,
            quality_binding=result['native_split_quality_binding'],
            verification_mode='portable')[0] is None
    else:
        assert result.get('identity_conflicts')
    assert (row, context) == before
