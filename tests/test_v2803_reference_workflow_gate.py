"""AP4 real generator/launcher/ORT/consumer gates; no accelerator attestation.

Tiny input ONNX/images and candidate records are synthetic. The generator,
reference subprocess, ONNX Runtime inference, quality service and strict loader
run normally. Target-system G2/G3 remain a separate requested acceptance.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from threading import Event

import onnx
import onnxruntime
from pycocotools import coco
import pytest

from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
from onnx_splitpoint_tool.management_reference import generate_management_cpu_reference
from onnx_splitpoint_tool.workflow.artifacts import write_json

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    key = 'v2803_reference_' + name.removesuffix('.py')
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / 'tests' / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _read(path):
    return json.loads(Path(path).read_text())


def _schedule(runner, suite, task):
    model = 'tiny_' + task
    model_dir = runner.run_dir / 'models' / model
    (model_dir / 'benchmark_set').mkdir(parents=True, exist_ok=True)
    canonical = model_dir / 'benchmark_set/legacy_suite'
    if suite != canonical:
        suite = Path(shutil.copytree(suite, canonical))
    for filename in ('benchmark_plan.json', 'benchmark_set.json'):
        shutil.copy2(suite / filename, model_dir / 'benchmark_set' / filename)
    write_json(model_dir / 'model_manifest.json', {
        'model_id': model, 'task': task,
        'resolved_path': str(suite / 'models' / (model + '.onnx')),
    })
    runner.profile_payload['models'].append({'id': model, 'task': task})
    runner.manifest.setdefault('models', {})[model] = {'model_id': model, 'task': task}
    runner._schedule_management_cpu_reference(model, suite)
    return model, runner._management_reference_futures[model].result(timeout=90), suite


def _set_population(suite, count):
    # The real generator already bound a tiny diagnostic dataset. Narrow the
    # saved run contract before dispatch; do not relabel B500 or B5000 evidence.
    path = suite / 'benchmark_plan.json'
    plan = _read(path)
    for key in ('runs', 'planned_runs'):
        for row in plan.get(key, []):
            row['validation_max_images'] = count
            row['validation_items_requested'] = count
    write_json(path, plan)


@pytest.mark.parametrize('task', ['classification', 'detection'])
def test_t04_actual_reference_consumer_own_dataset_and_strict_producer(tmp_path, task):
    integration = _module('test_v2802_workflow_integration.py')
    generator = _module('test_v2802_cpu_reference_binding.py')
    suite = generator.make_generated_reference_suite(tmp_path / 'generated', task)
    _set_population(suite, 2)
    runner, _ = integration._runner(tmp_path)
    try:
        assert not cache_verify_guard(runner.profile_payload)
        model, status, suite = _schedule(runner, suite, task)
        assert status['status'] == 'completed' and status['return_code'] == 0
        records, consumed, encoded = runner._management_reference_records(model)
        assert len(records) == 2
        assert consumed['semantic_reference_only'] is True
        assert all(consumed[k] is False for k in ('include_in_latency_fps_energy', 'include_in_ranking', 'include_in_pareto'))
        reference = _read(status['reference_path'])
        assert len(reference['records']) == 2
        assert 'CPUExecutionProvider' in (runner.run_dir / 'quality_management/references' / model / 'management_cpu_reference_stdout.txt').read_text()
        request_path, candidate = integration._candidate(runner, model, task, reference, bad=True)
        assert _read(request_path)['test_evidence_scope'] == 'synthetic_accelerator_candidate_only'
        assert runner._queue_central_quality_requests(model) == 1
        artifacts, _, _, _ = runner._stage_evaluate_quality()
        summary = _read(artifacts['central_quality_summary_json'])
        assert (summary['completed_count'], summary['failed_count']) == (1, 0)
        result = summary['results'][0]
        # A correctly computed negative comparison is a completed software job.
        assert result['technical_status'] == 'completed'
        assert result['decision'] == 'fail' and result['n'] == 2
        integration._blocked_full_from_actual_summary(runner, model, candidate['setup_id'], artifacts['central_quality_summary_json'])
        import yaml
        (runner.run_dir / 'profile.yaml').write_text(yaml.safe_dump(runner.profile_payload))
        write_json(runner.run_dir / 'run_manifest.json', {'run_id': runner.run_id, 'tool_version': '2.80.3'})
        spec = importlib.util.spec_from_file_location('v2803_inspector', ROOT / 'scripts/reference_workflow_gate_v2803.py')
        gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
        verified = gate.inspect_run(runner.run_dir, models={model: task}, expected_items=2)
        assert verified['g2_status'] == 'verified_existing_reference_and_consumer', verified
        assert verified['producer_binding_status'] == 'blocked'
        assert verified['g3_status'] == 'not_accepted_by_reference_check_alone'
        assert verified['observed_compiler_dispatch_count'] is None
        assert verified['new_inference_started'] is False
        assert verified['models'][0]['quality_decisions'] == ['fail']
        assert verified['models'][0]['producer_checks'][0]['setup_id'] == candidate['setup_id']
        # Exercise the fixed target-scope contract against the actual consumer
        # request bytes. This explicit synthetic setup never becomes physical
        # attestation: the strict Native producer remains blocked.
        gate.SETUP = candidate['setup_id']
        scoped_profile = dict(runner.profile_payload)
        scoped_profile['selection_policy'] = {'forced_cases': {model: ['b001']}}
        scoped_profile['native_producers'] = {'case_map': {model: ['b001']}}
        scoped_profile['hardware'] = {'selected_setups': [candidate['setup_id']]}
        profile_path = runner.run_dir / 'profile.yaml'
        profile_path.write_text(yaml.safe_dump(scoped_profile))
        scoped = gate.inspect_run(runner.run_dir, models={model: (task, 'b001')}, expected_items=2)
        assert scoped['g2_status'] == 'verified_existing_reference_and_consumer', scoped
        scoped_profile['selection_policy']['forced_cases'][model] = ['b002']
        profile_path.write_text(yaml.safe_dump(scoped_profile))
        mismatch = gate.inspect_run(runner.run_dir, models={model: (task, 'b001')}, expected_items=2)
        assert mismatch['g2_status'] == 'blocked' and 'case_profile_scope_mismatch' in mismatch['models'][0]['errors'][0]
        scoped_profile['selection_policy']['forced_cases'][model] = ['b001']
        scoped_profile['hardware']['selected_setups'] = ['other_setup']
        profile_path.write_text(yaml.safe_dump(scoped_profile))
        mismatch = gate.inspect_run(runner.run_dir, models={model: (task, 'b001')}, expected_items=2)
        assert mismatch['g2_status'] == 'blocked' and 'setup_profile_scope_mismatch' in mismatch['models'][0]['errors'][0]
        scoped_profile['hardware']['selected_setups'] = [candidate['setup_id']]
        profile_path.write_text(yaml.safe_dump(scoped_profile))
        # Even with the expected profile scope, editable summary labels cannot
        # override the setup bound in the original quality request.
        saved_summary = Path(artifacts['central_quality_summary_json']).read_bytes()
        foreign = _read(artifacts['central_quality_summary_json'])
        foreign['results'][0]['source_setup_id'] = 'other_setup'
        write_json(artifacts['central_quality_summary_json'], foreign)
        mismatch = gate.inspect_run(runner.run_dir, models={model: (task, 'b001')}, expected_items=2)
        assert mismatch['g2_status'] == 'blocked' and 'setup_consumer_scope_mismatch' in mismatch['models'][0]['errors'][0]
        Path(artifacts['central_quality_summary_json']).write_bytes(saved_summary)

        # A completed result with the right count but a different original
        # reference must not pass the read-only acceptance projection.
        original_summary_bytes = Path(artifacts['central_quality_summary_json']).read_bytes()
        wrong_summary = _read(artifacts['central_quality_summary_json'])
        wrong_summary['results'][0]['management_cpu_reference']['source_contract_sha256'] = '0' * 64
        write_json(artifacts['central_quality_summary_json'], wrong_summary)
        rejected = gate.inspect_run(runner.run_dir, models={model: task}, expected_items=2)
        assert rejected['g2_status'] == 'blocked'
        assert 'reference_binding_mismatch' in rejected['models'][0]['errors'][0]
        Path(artifacts['central_quality_summary_json']).write_bytes(original_summary_bytes)
        # A different diagnostic population creates a different immutable
        # reference generation, even with the same tiny original ONNX.
        before = Path(status['reference_path']).read_bytes()
        _set_population(suite, 3)
        expanded = generate_management_cpu_reference(suite_dir=suite,
            output_dir=runner.run_dir / 'quality_management/references' / model,
            model_id=model, workers=1, timeout_s=60)
        assert expanded['status'] == 'completed'
        assert expanded['source_contract_sha256'] != status['source_contract_sha256']
        assert expanded['reference_path'] != status['reference_path']
        assert len(_read(expanded['reference_path'])['records']) == 3
        assert Path(status['reference_path']).read_bytes() == before
    finally:
        runner._shutdown_management_services()


@pytest.mark.parametrize('damage', ['missing_model', 'invalid_model', 'explicit_wrong_id'])
def test_t04_actual_bad_source_does_not_publish_reference(tmp_path, damage):
    generator = _module('test_v2802_cpu_reference_binding.py')
    suite = generator.make_generated_reference_suite(tmp_path / 'generated', 'classification')
    contract_path = suite / 'benchmark_set.json'
    contract = _read(contract_path)
    if damage == 'missing_model':
        (suite / contract['model']).unlink()
    elif damage == 'invalid_model':
        (suite / contract['model']).write_bytes(b'not an ONNX model')
    else:
        contract['model_id'] = 'contradictory_explicit_id'
        write_json(contract_path, contract)
    out = tmp_path / 'reference'
    status = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id='tiny_classification', workers=1, timeout_s=60)
    assert status['status'] == 'failed'
    assert not status.get('reference_path')
    assert not list(out.rglob('canonical_cpu_reference.json'))
    assert (out / 'management_cpu_reference_status.json').is_file()
    assert (out / 'management_cpu_reference_stdout.txt').is_file()
    assert status.get('error') or status.get('errors')


@pytest.mark.parametrize('stop', ['timeout', 'cancel'])
def test_t04_actual_process_timeout_and_cancel_keep_small_diagnostics(tmp_path, stop):
    generator = _module('test_v2802_cpu_reference_binding.py')
    suite = generator.make_generated_reference_suite(tmp_path / 'generated', 'classification')
    cancel = Event()
    if stop == 'cancel':
        cancel.set()
    else:
        # Actual blocked child at OS boundary; the launcher, timeout, cleanup
        # and status writer run normally. Never replace ORT with a success mock.
        script = suite / 'benchmark_suite.py'
        lines = script.read_text().splitlines(keepends=True)
        function = next(n for n in ast.parse(''.join(lines)).body
                        if isinstance(n, ast.FunctionDef) and n.name == '_run_case')
        lines.insert(function.body[0].lineno - 1, '    import time; time.sleep(30)\n')
        script.write_text(''.join(lines))
    out = tmp_path / 'reference'
    status = generate_management_cpu_reference(suite_dir=suite, output_dir=out,
        model_id='tiny_classification', workers=1, timeout_s=1, cancel_event=cancel)
    assert status['status'] == ('cancelled' if stop == 'cancel' else 'failed')
    assert status['cancelled'] is (stop == 'cancel')
    assert status['timed_out'] is (stop == 'timeout')
    assert 'reference_cancelled' in status['error'] if stop == 'cancel' else 'reference_runner_timeout' in status['error']
    assert (out / 'management_cpu_reference_status.json').is_file()
    assert (out / 'management_cpu_reference_stdout.txt').is_file()
    assert not list(out.rglob('canonical_cpu_reference.json'))
    assert not list((out / 'workspaces').iterdir())
