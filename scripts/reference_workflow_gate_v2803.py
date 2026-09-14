#!/usr/bin/env python3
"""Prepare a bounded normal workflow; inspect its actual reference/consumer evidence.

No alternate inference, quality evaluator, compiler, or device-control path.
Prepare writes a separate profile. Execute uses the ordinary workflow CLI.
Inspect is read-only and delegates reference and producer validation to existing
production consumers. A preparation result is never G2/G3 acceptance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future
import copy
import json
from pathlib import Path
import shlex
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = {'mobilenet_v3_large': ('classification', 'b056'), 'yolo11l': ('detection', 'b062')}
SETUP = 'orin_nx_hailo10_01'


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'source_missing_or_unsafe:{path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def prepare_profile(source_profile, source_run, *, require_originals=True, only_model=None):
    from onnx_splitpoint_tool.run_modes import apply_run_mode
    from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    from onnx_splitpoint_tool.workflow.start_snapshot import snapshot_payload_sha256
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import resolve_artifact_cache_preflight_policy
    original, _ = load_runtime_profile_snapshot(str(source_profile))
    if only_model is not None and only_model not in MODELS:
        raise ValueError('unsupported_fixed_diagnostic_model:' + str(only_model))
    selected_models = {only_model: MODELS[only_model]} if only_model else MODELS
    source_run = Path(source_run).expanduser().resolve(strict=True)
    profile = copy.deepcopy(original)
    primary = profile.get('model_suite', {}).get('primary', [])
    chosen = {row['id']: copy.deepcopy(row) for row in primary if row.get('id') in selected_models}
    if set(chosen) != set(selected_models):
        raise ValueError('profile_must_contain_mobilenet_v3_large_and_yolo11l')
    for model, (task, case) in selected_models.items():
        contract = _read(source_run / 'models' / model / 'benchmark_set/legacy_suite/benchmark_set.json')
        if str(contract.get('model_id') or contract.get('model_name') or '') != model:
            raise ValueError(f'historical_model_binding_invalid:{model}')
        available = {str(row.get('case_id') or row.get('case') or f"b{int(row.get('boundary', -1)):03d}") for row in contract.get('cases', [])}
        if case not in available:
            raise ValueError(f'fixed_case_missing:{model}:{case}')
        model_path = Path(chosen[model].get('onnx') or '').expanduser()
        if require_originals and (not model_path.is_file() or model_path.is_symlink()):
            raise ValueError(f'original_model_missing:{model}:{model_path}')
        if chosen[model].get('task') != task:
            raise ValueError(f'profile_task_mismatch:{model}')
    preset = profile.get('execution_preset') or {}
    snapshot = copy.deepcopy(preset.get('snapshot') or {})
    if not snapshot:
        raise ValueError('resolved_mode_snapshot_missing')
    # These are the existing bound mode fields. The separate diagnostic profile
    # gets its own 16-image dataset/quality contract; calibration stays B500.
    snapshot['label'] = 'v2.80.3 Reference/Native diagnostic 16'
    snapshot['description'] = 'Separate bounded normal workflow; no B500/B5000 quality or final energy claim.'
    snapshot.setdefault('data', {})['validation_items'] = {'classification': 16, 'detection': 16}
    snapshot.setdefault('quality', {}).update(profile_id='v2803_reference_diagnostic_16', dataset_tier='screening', bootstrap_repetitions=100)
    snapshot.setdefault('defaults', {}).update(native_enabled=True, energy_enabled=False)
    snapshot.setdefault('runtime', {}).setdefault('benchmark', {})['timeout_s'] = 300
    snapshot['runtime'].setdefault('parallel', {}).update(remote_setups=False, max_setups=1)
    snapshot['runtime'].setdefault('native', {})['build_missing_engines'] = False
    snapshot.setdefault('build', {}).setdefault('scheduler', {})['max_workers'] = 1
    profile['name'] = 'v2803_reference_native_fixed16' + ('_' + only_model if only_model else '')
    profile['purpose'] = 'Predeclared Hailo10→TRT fixed split plus Hailo10/TRT Full baselines; CPU reference, central consumer, strict Native binding. Generic TRT→TRT split is outside this diagnostic scope.'
    profile['model_suite'] = {'primary': [chosen[m] for m in selected_models], 'reserve': []}
    profile.setdefault('selection_policy', {}).update(max_accepted_cases_per_model=1,
        forced_cases={m: [case] for m, (_, case) in selected_models.items()}, score_independent_audit_enabled=False)
    profile['run_profiles'] = [row for row in profile.get('run_profiles', []) if row.get('id') in {'ort_tensorrt', 'hailo10', 'hailo10_to_tensorrt'}]
    if {row.get('id') for row in profile['run_profiles']} != {'ort_tensorrt', 'hailo10', 'hailo10_to_tensorrt'}:
        raise ValueError('required_hailo10_and_tensorrt_run_profiles_missing')
    for row in profile['run_profiles']:
        if row['id'] == 'ort_tensorrt':
            row['variants'] = ['full']
    hardware = profile.setdefault('hardware', {})
    targets = hardware.get('resolved_targets', [])
    target_ids = [row.get('id') for row in targets]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError('duplicate_frozen_hardware_setup')
    selected_targets = [row for row in targets if row.get('id') == SETUP]
    if len(selected_targets) != 1:
        raise ValueError(f'required_frozen_hardware_setup_missing:{SETUP}')
    # The ordinary loader above has already validated the original mapping.
    # Bind only its exact selected subset, using the existing snapshot contract;
    # never retain the original three-target hash or reload the registry here.
    hardware.update(selected_setups=[SETUP], selected_groups=[],
        resolved_targets=selected_targets,
        resolved_targets_sha256=snapshot_payload_sha256(selected_targets))
    profile.setdefault('hailo_build', {}).update(hw_arch='hailo10h', targets=['hailo10h'])
    profile['artifact_cache_preflight'] = {'enabled': True, 'default_expectation': 'warm', 'block_on_unexpected_cold_builds': True}
    native = profile.setdefault('native_producers', {})
    native.update(enabled=True, backends=['hailo10h'], split_backends=['hailo10h'], models=list(selected_models),
        case_policy='case_map_only', case_map={m: [case] for m, (_, case) in selected_models.items()}, build_missing_engines=False)
    native['full_baselines'] = {'enabled': True, 'backends': ['hailo10h', 'tensorrt']}
    native.setdefault('energy', {})['enabled'] = False
    native['energy'].setdefault('window_method_validation_probe', {})['enabled'] = False
    profile.setdefault('energy', {})['enabled'] = False
    preset.update(follow_tool_config=False, snapshot=snapshot, overrides={'native_enabled': True, 'energy_enabled': False})
    profile['execution_preset'] = preset
    profile, _ = apply_run_mode(profile, follow_tool_config=False)
    if cache_verify_guard(profile):
        raise ValueError('global_cache_verify_only_forbidden')
    hailo, deepx = profile['hailo_build'], profile['deepx_build']
    if (hailo.get('force_build') is not False or deepx.get('force_build') is not False
        or hailo.get('calib_count') != 500 or hailo.get('preset') != 'balanced'
        or hailo.get('optimization_level') != 1 or hailo.get('calib_batch_size') != 8
        or deepx.get('calib_count') != 500 or deepx.get('calibration_method') != 'ema'
        or deepx.get('opt_level') != 0 or deepx.get('classification_preprocessing') != 'imagenet_mean_std'):
        raise ValueError('existing_profile_recipe_differs_from_fixed_B500_contract')
    policy = resolve_artifact_cache_preflight_policy(profile)
    assert policy['default_expectation'] == 'warm' and policy['block_on_unexpected_cold_builds']
    if profile['validation_execution']['max_items'] != {'classification': 16, 'detection': 16}:
        raise ValueError('diagnostic_population_was_overridden')
    return profile


def inspect_run(run_dir, *, models=None, expected_items=16):
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
    from onnx_splitpoint_tool.management_reference import _source_contract
    from onnx_splitpoint_tool.trt_quality_chain import load_producer_set_from_central_quality_summary
    from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
    import yaml
    run = Path(run_dir).expanduser().resolve(strict=True)
    manifest = _read(run / 'run_manifest.json')
    profile = yaml.safe_load((run / 'profile.yaml').read_text())
    if cache_verify_guard(profile):
        raise ValueError('cache_verify_only_cannot_accept_normal_reference_gate')
    chosen = models or MODELS
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile='', out=str(run.parent)))
    runner.run_dir = run
    runner.run_id = str(manifest.get('run_id') or run.name)
    runner.profile_payload = profile
    summary_path = run / 'quality_management/central_quality_summary.json'
    summary = _read(summary_path)
    report = {'schema': 'onnx-splitpoint/reference-workflow-gate', 'schema_version': 1,
        'source_run_id': runner.run_id, 'source_tool_version': manifest.get('tool_version'),
        'role': 'read_only_validation_of_existing_run', 'new_inference_started': False,
        'new_compiler_started': False, 'new_hardware_started': False,
        'expected_validation_items': expected_items, 'expected_compiler_dispatch_count': 0,
        'observed_compiler_dispatch_count': None, 'models': [], 'g2_status': 'blocked',
        'g3_status': 'not_accepted_by_reference_check_alone'}
    for model, value in chosen.items():
        task = value[0] if isinstance(value, (list, tuple)) else value
        expected_case = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else None
        row = {'model_id': model, 'task': task, 'reference_status': 'blocked', 'consumer_status': 'not_checked',
               'producer_status': 'not_checked', 'errors': []}
        try:
            suite = run / 'models' / model / 'benchmark_set/legacy_suite'
            plan, contract = _read(suite / 'benchmark_plan.json'), _read(suite / 'benchmark_set.json')
            if expected_case:
                if profile.get('selection_policy', {}).get('forced_cases', {}).get(model) != [expected_case]:
                    raise ValueError('fixed_case_profile_scope_mismatch')
                if profile.get('native_producers', {}).get('case_map', {}).get(model) != [expected_case]:
                    raise ValueError('fixed_case_native_scope_mismatch')
                if profile.get('hardware', {}).get('selected_setups') != [SETUP]:
                    raise ValueError('fixed_setup_profile_scope_mismatch')
                actual_cases = {str(item.get('case_id') or item.get('case') or f"b{int(item.get('boundary', -1)):03d}") for item in contract.get('cases', [])}
                if actual_cases != {expected_case}:
                    raise ValueError('fixed_case_generated_suite_scope_mismatch')
            status = _read(run / 'quality_management/references' / model / 'management_cpu_reference_status.json')
            future = Future(); future.set_result(status)
            runner._management_reference_futures[model] = future
            runner._management_reference_source_contracts[model] = _source_contract(suite, plan, contract)
            records, consumed, _ = runner._management_reference_records(model)
            if consumed.get('task') != task or len(records) != expected_items:
                raise ValueError(f'reference_task_or_population_mismatch:{task}:{len(records)}')
            row.update(reference_status='verified', record_count=len(records),
                       source_contract_sha256=consumed['source_contract_sha256'])
            results = [r for r in summary.get('results', []) if r.get('model_id') == model]
            if not results:
                raise ValueError('central_consumer_result_missing')
            failures = [r for r in results if r.get('technical_status') != 'completed' or r.get('n') != expected_items]
            if failures:
                raise ValueError('central_consumer_incomplete_or_population_mismatch:' + str(failures[0].get('error') or failures[0].get('technical_status')))
            selected_split_consumed = False
            for result in results:
                reference_used = result.get('management_cpu_reference') or {}
                for field in ('source_contract_sha256', 'reference_sha256', 'quality_contract_sha256'):
                    expected = str(consumed.get(field) or '').removeprefix('sha256:')
                    observed = str(reference_used.get(field) or '').removeprefix('sha256:')
                    # Classification status uses the exact reference bytes;
                    # unlike detection it need not duplicate a quality-contract
                    # digest. Never invent that absent legacy field.
                    optional_absent = field == 'quality_contract_sha256' and task == 'classification' and not expected and not observed
                    if not optional_absent and (not expected or expected != observed):
                        raise ValueError('central_consumer_reference_binding_mismatch:' + field)
                if result.get('task') != task or result.get('eval_run_id') != runner.run_id:
                    raise ValueError('central_consumer_model_task_run_binding_mismatch')
                request_relative = Path(str(result.get('source_request') or ''))
                if request_relative.is_absolute() or '..' in request_relative.parts or not request_relative.parts:
                    raise ValueError('central_consumer_request_path_invalid')
                request_path = run / request_relative
                if request_path.is_symlink() or run not in request_path.resolve().parents:
                    raise ValueError('central_consumer_request_path_unsafe')
                identity = runner._quality_request_identity(request_path, model_id=model)
                if not identity.get('identity_valid') or str(identity.get('source_request_sha256')).removeprefix('sha256:') != str(result.get('source_request_sha256')).removeprefix('sha256:'):
                    raise ValueError('central_consumer_request_binding_invalid')
                if expected_case:
                    request_case = str(identity.get('case_id') or '')
                    request_setup = str(identity.get('setup_id') or '')
                    result_case = str(result.get('case_id') or '')
                    result_setup = str(result.get('source_setup_id') or result.get('setup_id') or '')
                    if request_case not in {expected_case, 'full'} or result_case != request_case:
                        raise ValueError('fixed_case_consumer_scope_mismatch')
                    if request_setup != SETUP or result_setup != request_setup:
                        raise ValueError('fixed_setup_consumer_scope_mismatch')
                    selected_split_consumed |= request_case == expected_case
            if expected_case and not selected_split_consumed:
                raise ValueError('fixed_case_consumer_coverage_missing')
            row.update(consumer_status='completed', quality_decisions=sorted({str(r.get('decision')) for r in results}))
            setups = sorted({str(r.get('source_setup_id') or r.get('setup_id')) for r in results if r.get('source_setup_id') or r.get('setup_id')})
            producer_checks = []
            for setup in setups:
                try:
                    load_producer_set_from_central_quality_summary(summary_path,
                        eval_run_id=runner.run_id, setup_id=setup, model_ids=[model])
                    producer_checks.append({'setup_id': setup, 'status': 'verified'})
                except Exception as exc:
                    producer_checks.append({'setup_id': setup, 'status': 'blocked', 'reason': f'{type(exc).__name__}: {exc}'})
            row['producer_checks'] = producer_checks
            row['producer_status'] = 'verified' if producer_checks and all(r['status'] == 'verified' for r in producer_checks) else 'blocked'
        except Exception as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        report['models'].append(row)
    if report['models'] and all(r['reference_status'] == 'verified' and r['consumer_status'] == 'completed' for r in report['models']):
        report['g2_status'] = 'verified_existing_reference_and_consumer'
    # Producer binding alone is not a completed Native invocation, nor proof
    # that every expected compiler dispatch was zero. Preserve that distinction.
    report['producer_binding_status'] = 'verified' if report['models'] and all(r['producer_status'] == 'verified' for r in report['models']) else 'blocked'
    runner._shutdown_management_services()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--profile', required=True)
    prepare.add_argument('--source-run', required=True)
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--only-model', '--model', choices=sorted(MODELS))
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--run-dir', required=True)
    inspect.add_argument('--output-dir', required=True)
    inspect.add_argument('--only-model', '--model', choices=sorted(MODELS))
    args = parser.parse_args(argv)
    out = Path(args.output_dir).expanduser().absolute()
    source = Path(args.source_run if args.command == 'prepare' else args.run_dir).expanduser().resolve()
    if out.resolve() == source or source in out.resolve().parents:
        parser.error('output-dir must be outside the source run')
    out.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == 'prepare':
            try:
                probe = subprocess.run([sys.executable, '-I', '-B', '-c',
                    "import onnx, onnxruntime; from pycocotools import coco; assert 'CPUExecutionProvider' in onnxruntime.get_available_providers(); print('REFERENCE_DEPENDENCIES=PASS')"],
                    text=True, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or b'') + (exc.stderr or b'') if isinstance(exc.stdout or exc.stderr, bytes) else (exc.stdout or '') + (exc.stderr or '')
                (out / 'dependencies.log').write_text(output.decode(errors='replace') if isinstance(output, bytes) else output)
                raise RuntimeError('environment_blocked:reference_dependency_probe_timeout_30s') from exc
            (out / 'dependencies.log').write_text(probe.stdout + probe.stderr)
            if probe.returncode != 0:
                raise RuntimeError('environment_blocked:ONNX_ORT_PyCOCO_CPU_dependencies; see dependencies.log')
            import yaml
            profile = prepare_profile(args.profile, source, only_model=args.only_model)
            target = out / 'profile_fixed16.yaml'
            target.write_text(yaml.safe_dump(profile, sort_keys=False))
            # Preparation is complete only when the serialized profile passes
            # the same loader used by the subsequent normal workflow CLI.
            from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
            load_runtime_profile_snapshot(str(target))
            command = [str(ROOT / '.venv/bin/python'), '-m', 'onnx_splitpoint_tool.workflow.run_evaluation',
                '--profile-driven', '--profile', str(target), '--out', str(out / 'runs'), '--require-fresh-run']
            report = {'status': 'prepared_not_executed', 'g2_status': 'not_run', 'g3_status': 'not_run',
                'expected_compiler_dispatch_count': 0, 'observed_compiler_dispatch_count': None,
                'fixed_models_and_cases': {args.only_model: MODELS[args.only_model]} if args.only_model else MODELS,
                'validation_items': 16, 'calibration_items': 500,
                'source_run': str(source), 'normal_workflow_command': command}
            _write(out / 'preparation.json', report)
            print('PROFILE=' + str(target))
            print('NORMAL_WORKFLOW_COMMAND=' + shlex.join(command))
            print('PREPARATION_STATUS=prepared_not_executed')
        else:
            report = inspect_run(source, models={args.only_model: MODELS[args.only_model]} if args.only_model else None)
            _write(out / 'reference_gate.json', report)
            print('G2_STATUS=' + report['g2_status'])
            print('PRODUCER_BINDING_STATUS=' + report['producer_binding_status'])
            print('G3_STATUS=' + report['g3_status'])
            return 0 if report['g2_status'] == 'verified_existing_reference_and_consumer' else 2
    except Exception as exc:
        _write(out / 'failure.json', {'status': 'blocked', 'error': f'{type(exc).__name__}: {exc}', 'g2_status': 'not_accepted', 'g3_status': 'not_accepted'})
        print('GATE_STATUS=blocked')
        print(f'ERROR={type(exc).__name__}: {exc}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
