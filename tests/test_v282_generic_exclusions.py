"""Original Sept-13 replay plus synthetic adversarial execution evidence.

Original JSON bytes are retained under fixtures/v282_generic. The small
adversarial observations reuse older valid negative records and are explicitly
synthetic software tests, never hardware acceptance.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from onnx_splitpoint_tool.native_job_identity import required_profile_build_exclusions

FIXTURES = Path(__file__).parent / 'fixtures/v281_status'

def fixture(name):
    return json.loads((FIXTURES / name).read_text())

def inputs():
    required = fixture('night_required_missing_rows.json')
    blocked = fixture('night_blocked_native_rows.json')
    readiness = {'blocked_jobs': [r['upstream_build_observation'] for r in blocked if r.get('upstream_build_observation')]}
    components = [dict(r, primary_variant='part2', component_measurement_status='part2_only',
                       measured_variants=['part2'], skipped_variants=['part1', 'composed'],
                       variant_status={'part1': 'skipped', 'part2': 'ok', 'composed': 'skipped'},
                       runtime_ok=True, runtime_executable=False, repetition_count_attempted=None,
                       repetition_count_valid=None, part2_latency_ms=1.25)
                  for r in required if str(r.get('model_id') or '').startswith('yolo26')]
    return required, readiness, components

def test_six_exact_exclusions_accept_explicit_components_without_mutation():
    required, readiness, components = inputs()
    before = copy.deepcopy((required, readiness, components))
    output = required_profile_build_exclusions(required, readiness, components)
    assert len(output) == 6
    assert {r['model_id'] for r in output} == {'yolo26m', 'yolo26s'}
    assert all(r['quality_applicability'] == 'not_applicable_build_excluded' for r in output)
    assert all(r['measurement_values_synthesized'] is False for r in output)
    assert all('part2_latency_ms' not in r for r in output)
    assert (required, readiness, components) == before


@pytest.mark.parametrize('mutation', [
    {'composed_runtime_started': True},
    {'composed_execution_started': True},
    {'composed_measurement_completed': True},
    {'composed_repetition_count_attempted': 1},
    {'composed_completed_work_units': 32},
    {'completed_task_endpoint_attested': True},
    {'composed_completion_verified': True},
    {'measured_variants': ['part2', 'composed']},
    {'variant_status': {'part2': 'ok', 'composed': 'running'}},
    {'timings': {'composed': {'mean': 0.2}}},
    {'variant_results': {'composed': {'ok': False, 'runtime_started': True}}},
    {'variant_results': [{'variant': 'composed', 'completed_work_units': 32}]},
    {'runtime_results_by_variant': {'composed': {'runtime_success': True}}},
    {'identity_conflicts': ['setup_id']},
    {'composed_repetition_count_attempted': '1'},
    {'composed_repetition_count_attempted': -1},
])
def test_conflicting_composed_evidence_overrides_component_flag(mutation):
    required, readiness, components = inputs()
    for row in components:
        row.update(copy.deepcopy(mutation))
    diagnostics = []
    assert required_profile_build_exclusions(required, readiness, components, diagnostics=diagnostics) == []
    assert len(diagnostics) == 6
    assert all(row['reasons'] for row in diagnostics)


@pytest.mark.parametrize('value', [None, '1', True, -1, 'bad'])
def test_unbound_attempt_counts_are_not_zero_or_positive_evidence(value):
    required, readiness, components = inputs()
    ambiguous = [dict(row, primary_variant='', measured_variants=[], component_measurement_status='',
                      variant_status={}, runtime_ok=False, repetition_count_attempted=value)
                 for row in components]
    assert required_profile_build_exclusions(required, readiness, ambiguous) == []
    for row in components:
        row['repetition_count_attempted'] = value
    # These row-wide counters belong to the independent component role.
    assert len(required_profile_build_exclusions(required, readiness, components)) == 6


def test_generic_runtime_success_without_role_stays_unresolved():
    required, readiness, components = inputs()
    rows = [dict(row, primary_variant='', component_measurement_status='', measured_variants=[],
                 variant_status={}, repetition_count_attempted=0) for row in components]
    assert required_profile_build_exclusions(required, readiness, rows) == []


def test_success_required_remains_strict_and_foreign_setup_is_not_relabelled():
    required, readiness, components = inputs()
    assert required_profile_build_exclusions([dict(row, success_required=True) for row in required], readiness, components) == []
    for observation in readiness['blocked_jobs']:
        observation['setup_id'] = 'foreign_setup'
    assert required_profile_build_exclusions(required, readiness, components) == []


@pytest.mark.parametrize('mutation', ['foreign_recipe', 'damaged_record', 'infrastructure', 'duplicate', 'wrong_family', 'wrong_boundary'])
def test_negative_build_authority_remains_exact(mutation):
    required, readiness, components = inputs()
    for observation in readiness['blocked_jobs']:
        evidence = observation.get('build_evidence') or {}
        if not evidence:
            continue
        if mutation == 'foreign_recipe':
            evidence['record']['key']['recipe']['net_name'] = 'foreign_recipe'
        elif mutation == 'damaged_record':
            evidence['record']['record_sha256'] = '0' * 64
        elif mutation == 'infrastructure':
            evidence['state'] = 'TRANSIENT_INFRASTRUCTURE'
        elif mutation == 'wrong_family':
            observation['backend'] = 'deepx'
        elif mutation == 'wrong_boundary':
            observation['boundary'] = 'b001'
    if mutation == 'duplicate':
        readiness['blocked_jobs'] *= 2
    assert required_profile_build_exclusions(required, readiness, components) == []


def test_existing_precision_contract_does_not_borrow_another_runtime():
    required, readiness, components = inputs()
    required = [dict(row, precision='uint8_dequant_fp16') for row in required]
    foreign = [dict(row, precision='fp32', primary_variant='composed', runtime_ok=True) for row in components]
    assert len(required_profile_build_exclusions(required, readiness, foreign)) == 6
    for row in foreign:
        row['precision'] = 'uint8_dequant_fp16'
    assert required_profile_build_exclusions(required, readiness, foreign) == []


@pytest.mark.parametrize('model', ['yolo26m', 'yolo26s'])
def test_original_reader_matrix_applicability_and_reporting_chain(model):
    import hashlib
    from onnx_splitpoint_tool.workflow.results import _rows_from_json
    from onnx_splitpoint_tool.workflow.logical_measurement import select_logical_primary_rows
    from onnx_splitpoint_tool.workflow.runner import required_profile_outcomes_v282
    from onnx_splitpoint_tool.workflow.evidence_status import workflow_completion_projection
    from onnx_splitpoint_tool.workflow.required_run_scope import quality_applicability
    base = FIXTURES.parent / 'v282_generic' / model
    paths = [base / 'normalized_results.json', base / 'stage_result.json']
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    payload = json.loads(paths[0].read_text())
    stage = json.loads(paths[1].read_text())
    rows = _rows_from_json(paths[0])
    original_rows = copy.deepcopy(rows)
    logical, errors = select_logical_primary_rows(rows)
    assert not errors
    outcome = required_profile_outcomes_v282(payload['required_profile_results'], logical, rows,
                                            stage['details']['deferred_build_readiness'])
    assert outcome['excluded_result_count'] == 3
    assert outcome['missing_result_count'] == 1
    assert outcome['duplicate_result_count'] == 0
    assert outcome['build_exclusion_conflict_count'] == 0
    assert all(row['quality_applicability'] == 'not_applicable_build_excluded' for row in outcome['excluded_results'])
    assert all(quality_applicability(row, {}) == 'not_applicable' for row in outcome['excluded_results'])
    assert all(row['performance_claim_eligible'] is False for row in outcome['excluded_results'])
    projection = workflow_completion_projection('partial', generic_excluded_count=outcome['excluded_result_count'])
    assert projection['counts']['generic_excluded'] == 3
    assert rows == original_rows and len(rows) == 20
    assert outcome == required_profile_outcomes_v282(payload['required_profile_results'], logical, rows,
                                                     stage['details']['deferred_build_readiness'])
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_ingestion_keeps_composed_start_evidence_while_preserving_component_latency(tmp_path):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
    required, readiness, components = inputs()
    required = [row for row in required if row.get('model_id') == 'yolo26m']
    components = [row for row in components if row.get('model_id') == 'yolo26m']
    for row in components:
        row.update(composed_runtime_started=True, composed_completed_work_units=32,
                   variant_results={'composed': {'runtime_started': True, 'completed_work_units': 32}})
    source = tmp_path / 'benchmark_results_components.json'
    source.write_text(json.dumps({'results': components}))
    rows, _ = normalize_benchmark_files(model_id='yolo26m', source_paths=[source])
    assert rows and all(row.get('part2_latency_ms') == 1.25 for row in rows if row.get('primary_variant') == 'part2')
    assert all(row.get('composed_runtime_started') is True for row in rows)
    assert all(row.get('composed_completed_work_units') == 32 for row in rows)
    assert required_profile_build_exclusions(required, readiness, rows) == []


def test_readonly_replay_cli_is_idempotent_and_refuses_original_output(tmp_path):
    import subprocess
    import sys
    import hashlib
    import shutil
    source = tmp_path / 'source'
    for model in ('yolo26m', 'yolo26s'):
        old = FIXTURES.parent / 'v282_generic' / model
        base = source / 'models' / model
        (base / 'benchmark_results').mkdir(parents=True)
        (base / 'stages/build_backend_artifacts').mkdir(parents=True)
        shutil.copyfile(old / 'normalized_results.json', base / 'benchmark_results/normalized_results.json')
        shutil.copyfile(old / 'stage_result.json', base / 'stages/build_backend_artifacts/stage_result.json')
    original = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source.rglob('*.json')}
    script = Path(__file__).parents[1] / 'scripts/replay_generic_exclusions_v282.py'
    out = tmp_path / 'projection'
    command = [sys.executable, '-I', '-B', str(script), '--run-root', str(source), '--out', str(out), '--generic-only']
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    report = json.loads((out / 'REPLAY_REPORT.json').read_text())
    assert report['generic']['excluded_count'] == 6 and report['generic']['missing_count'] == 2
    assert report['hardware_execution'] is False and report['quality_recalculation'] is False
    assert report['native']['status'] == 'not_checked_generic_only'
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert original == {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in source.rglob('*.json')}
    rejected = subprocess.run(command[:command.index('--out')] + ['--out', str(source / 'bad'), '--generic-only'],
                              capture_output=True, text=True)
    assert rejected.returncode == 2 and 'outside_original_run' in rejected.stderr
    (out / 'foreign.txt').write_text('keep')
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode == 2 and 'different_content' in rejected.stderr
    assert (out / 'foreign.txt').read_text() == 'keep'
