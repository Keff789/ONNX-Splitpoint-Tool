"""The explicitly authorized two-row H8 energy remainder, with no GUI test."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import copy
import json

import pytest

from test_yolo_augmentation_rest_blocks import actual_plan, phase_fixture, AUG
from onnx_splitpoint_tool.workflow import yolo_augmentation_rest as rest

PREVIOUS = AUG / 'attempts/energy_authorized_20261004_135903'


def test_actual_h8_reentry_preserves_nine_and_selects_exact_two(actual_plan):
    observed = rest.energy_continuation(actual_plan, PREVIOUS, h8_reentry=True)
    assert observed['reused_energy_rows'] == 9 and observed['remaining_energy_rows'] == 2
    assert observed['inventory']['block1_generic'] == []
    assert observed['inventory']['native_repetitions'] == observed['inventory']['generic_repetitions'] == 33
    assert observed['inventory']['quality_results'] == 12
    with pytest.raises(ValueError, match='exact prior energy cohort'):
        rest.energy_continuation(actual_plan, PREVIOUS)
    with pytest.raises(ValueError, match='exact prior energy cohort'):
        rest.energy_continuation(actual_plan, AUG / 'attempts/restabschluss_20261004_065600', h8_reentry=True)


def test_h8_completed_energy_source_and_failed_repetitions_are_not_rewritten(actual_plan, tmp_path):
    from onnx_splitpoint_tool.native_energy_reporting import collect_native_energy
    observation = rest.energy_continuation(actual_plan, PREVIOUS, h8_reentry=True)
    source = Path(observation['previous_results']); before = source.read_bytes()
    roots = rest._energy_delivery_roots(tmp_path, observation)
    rows = collect_native_energy(roots[0], include_partial=True)
    assert sum(row['ok'] is True for row in rows) == 9
    assert sum(row['energy_repeat_valid_n'] or 0 for row in rows if row['ok'] is True) == 27
    assert source.read_bytes() == before
    assert roots == [Path(observation['previous_block'])]


def test_interrupted_child_without_current_row_keeps_original_valid_partial(actual_plan, tmp_path):
    from onnx_splitpoint_tool.native_energy_reporting import collect_native_energy
    from onnx_splitpoint_tool.resources.remote_scripts.run_native_producer_energy_from_summary import _energy_result_reusable
    observation = rest.energy_continuation(actual_plan, PREVIOUS, h8_reentry=True)
    original = Path(observation['previous_results']); before = original.read_bytes()
    current = copy.deepcopy(rest.io.read_json(original))
    current['rows'] = [row for row in current['rows'] if _energy_result_reusable(row)]
    assert len(current['rows']) == 9
    rest.io.write_json(tmp_path / 'block1/reports/native_energy_measurements/native_producer_energy_results.partial.json', current)
    roots = rest._energy_delivery_roots(tmp_path, observation)
    rows = collect_native_energy(roots[0], include_partial=True)
    assert roots == [Path(observation['previous_block'])]
    assert sum(row['ok'] is True for row in rows) == 9
    assert sum(row['energy_repeat_valid_n'] or 0 for row in rows) == 28
    partial = next(row for row in rows if row['model'] == 'yolo26s' and row['setup_id'] == 'orin_nx_hailo8_01')
    assert partial['ok'] is False and partial['claim_eligible'] is False
    assert partial['energy_repeat_valid_n'] == 1 and partial['energy_repeat_requested_n'] == 3
    assert original.read_bytes() == before


def test_h8_scope_cannot_authorize_different_open_case(actual_plan, monkeypatch):
    from onnx_splitpoint_tool.resources.remote_scripts import run_native_producer_energy_from_summary as consumer
    original = consumer._supplement_previous_results
    def changed(*args, **kwargs):
        reused = copy.deepcopy(original(*args, **kwargs))
        row = next(iter(reused.values()))
        row['row'].update(model='yolo26s', case='b038', setup_id='orin_nx_hailo8_01')
        return reused
    monkeypatch.setattr(consumer, '_supplement_previous_results', changed)
    with pytest.raises(ValueError, match='different open energy scope'):
        rest.energy_continuation(actual_plan, PREVIOUS, h8_reentry=True)


def test_h8_resume_enters_regular_energy_without_source_test_or_performance(phase_fixture, monkeypatch):
    output, _, delivery, _ = phase_fixture
    previous = output.parent / 'old'; block = output.parent / 'old_block'
    descriptor = dict(previous_attempt=str(previous), previous_results=str(block / 'partial.json'),
        previous_block=str(block), reused_energy_rows=9, remaining_energy_rows=2,
        inventory={'block1_energy': [('model', 'case', 'setup')]})
    def continuation(*args, **kwargs):
        assert kwargs == {'h8_reentry': True}; return descriptor
    monkeypatch.setattr(rest, 'energy_continuation', continuation)
    monkeypatch.setattr(rest, '_energy_delivery_roots', lambda *args: [block])
    monkeypatch.setattr(rest, '_run_source_test', lambda *args: pytest.fail('no separate diagnostic capture'))
    monkeypatch.setattr(rest, 'singlecase_command', lambda *args: pytest.fail('Native follows completed Block 1 only'))
    phases = []
    @contextmanager
    def owned(*args, **kwargs):
        assert kwargs == {'previous_attempt': str(previous), 'h8_reentry': True}
        def execute(command, label, *args, **kwargs):
            phases.append(label); raise RuntimeError('controlled regular energy dispatch')
        yield execute, {}, None, None
    monkeypatch.setattr(rest, 'owned_execution', owned)
    monkeypatch.setattr(rest, 'energy_command', lambda *args, **kwargs: ['normal-energy'])
    assert rest.finish_rest('plan.json', output, include_b066=True,
        previous_energy_attempt=previous, h8_reentry=True) == 1
    assert phases == ['block1-remaining-energy']
    start = rest.io.read_json(output / 'production-start.json')
    assert start['authorized_h8_energy_reentry'] is True and start['source_test_requested'] is False
    assert len(delivery['energy_results']) == 1


@pytest.mark.parametrize('arguments', [
    ['--execute', 'plan.json', '--authorize-h8-energy-reentry'],
    ['--finish-rest', 'plan.json', '--authorize-h8-energy-reentry'],
    ['--finish-rest', 'plan.json', '--resume-rest-energy', 'previous', '--authorize-h8-energy-reentry', '--test-recovered-energy-source'],
])
def test_h8_cli_requires_explicit_predecessor_and_forbids_extra_diagnostic(arguments):
    with pytest.raises(SystemExit) as error: rest.augmentation.main(arguments)
    assert error.value.code == 2


def test_actual_original_execution_context_retains_two_for_previous_results(actual_plan, tmp_path):
    run = Path(actual_plan['augmentation_run'])
    cfg = rest.io.read_json(run / 'reports/native_producer_stage_config.json')
    command = rest.energy_command(run, cfg, run / 'reports/native_producer_combined_summary.json',
        run / 'reports/native_validation/native_producer_validation_summary.json', tmp_path / 'block1',
        previous_results=Path(rest.energy_continuation(actual_plan, PREVIOUS, h8_reentry=True)['previous_results']))
    assert command[command.index('--campaign-max-retries') + 1] == '2'
    assert command[command.index('--duration-s') + 1] in {'60', '60.0'}
    assert command[command.index('--runs') + 1] == '3'
    assert '--test-recovered-energy-source' not in command
