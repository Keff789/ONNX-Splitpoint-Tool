#!/usr/bin/env python3
"""One bounded ordinary workflow acceptance for the four v2.81 corrections.

Two sequential, setup-bound profiles prevent cross-family cold builds. They use
ordinary profile loading, splitting, cache admission, compilation, inference,
quality and Native paths. No alternate inference implementation or tuning.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ITEMS = 16
BOOTSTRAPS = 100
SCOPE = {
    'hailo8': {'setup': 'orin_nx_hailo8_01', 'models': {
        'regnet_x_1_6gf': ['b073'], 'yolo11l': ['b064', 'b067'],
        'yolo26m': ['b040', 'b398', 'b399'], 'yolo26s': ['b023', 'b364', 'b365']},
        'known_exclusions': {'yolo26m': ['b398', 'b399'], 'yolo26s': ['b364', 'b365']},
        'allowed_missing': {'regnet_x_1_6gf': ['b073'], 'yolo11l': ['b064'],
                            'yolo26m': ['b040'], 'yolo26s': ['b023']},
        'full_id': 'hailo8', 'split_id': 'hailo8_to_trt', 'timeout_s': 18000},
    'hailo10h': {'setup': 'orin_nx_hailo10_01', 'models': {
        'yolo26m': ['b398', 'b399'], 'yolo26s': ['b364', 'b365']},
        'known_exclusions': {'yolo26m': ['b399'], 'yolo26s': ['b365']},
        'allowed_missing': {}, 'full_id': 'hailo10',
        'split_id': 'hailo10_to_tensorrt', 'timeout_s': 3600},
}


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing_or_unsafe_file:' + str(path))
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('json_object_required:' + str(path))
    return value


def _write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def run_logged(command, cwd, log, *, timeout_s):
    spec = importlib.util.spec_from_file_location('v281_acceptance_process', ROOT / 'scripts/v281_acceptance_process.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', ORT_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1')
    return module.run_stage(command, cwd, log, environment, timeout_s=timeout_s)


def prepare(source_run, output, family, *, hailo8_dependency_manifest=None, build_environment=None, require_originals=True):
    import yaml
    from onnx_splitpoint_tool.run_modes import apply_run_mode
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    from onnx_splitpoint_tool.workflow.start_snapshot import snapshot_payload_sha256
    from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
    scope = SCOPE[family]
    source = Path(source_run).expanduser().resolve(strict=True)
    output = Path(output).absolute()
    if output.resolve() == source or source in output.resolve().parents:
        raise ValueError('output_must_be_outside_existing_source_run')
    original_path = source / 'profile.yaml'
    before = original_path.read_bytes()
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import validate_evaluation_profile_payload
    from onnx_splitpoint_tool.workflow.start_snapshot import resolve_runtime_profile_start_snapshot
    source_payload = yaml.safe_load(before)
    if not isinstance(source_payload, dict) or not (source_payload.get('execution_preset') or {}).get('snapshot'):
        raise ValueError('resolved_mode_snapshot_missing')
    frozen = copy.deepcopy(source_payload)
    frozen['execution_preset']['follow_tool_config'] = False
    resolved, _ = apply_run_mode(frozen, follow_tool_config=False)
    resolved = validate_evaluation_profile_payload(resolved, source=str(original_path))
    original, _ = resolve_runtime_profile_start_snapshot(profile_request=str(original_path), source_profile=source_payload,
        resolved_profile=resolved, profile_id=str(resolved.get('name') or original_path.stem),
        profile_path=str(original_path), profile_source='file', options=None)
    profile = copy.deepcopy(original)
    chosen = {r['id']: copy.deepcopy(r) for r in profile.get('model_suite', {}).get('primary', [])
              if r.get('id') in scope['models']}
    if set(chosen) != set(scope['models']):
        raise ValueError('original_night_profile_models_missing:' + family)
    for model, cases in scope['models'].items():
        contract_path = source / 'models' / model / 'benchmark_set/legacy_suite/benchmark_set.json'
        if not contract_path.exists():
            contract_path = source / 'native_producers' / family / model / 'benchmark_set/benchmark_set.json'
        contract = _read(contract_path)
        if str(contract.get('model_id') or contract.get('model_name') or '') != model:
            raise ValueError('historical_model_binding_invalid:' + model)
        available = {str(r.get('case_id') or r.get('case') or f"b{int(r.get('boundary', -1)):03d}") for r in contract.get('cases', [])}
        if not set(cases).issubset(available):
            raise ValueError('fixed_cases_missing:' + model + ':' + ','.join(sorted(set(cases) - available)))
        path = Path(chosen[model].get('onnx') or '').expanduser()
        if require_originals and (path.is_symlink() or not path.is_file()):
            raise ValueError('original_model_missing:' + model + ':' + str(path))
        expected_task = 'classification' if model == 'regnet_x_1_6gf' else 'detection'
        if chosen[model].get('task') != expected_task:
            raise ValueError('profile_task_mismatch:' + model)
    preset = profile.get('execution_preset') or {}
    snapshot = copy.deepcopy(preset.get('snapshot') or {})
    if not snapshot:
        raise ValueError('resolved_mode_snapshot_missing')
    original_recipe = copy.deepcopy(original['hailo_build'])
    snapshot['label'] = 'v2.81 targeted ordinary workflow ' + family
    snapshot['description'] = '16 validation images, 100 bootstraps, 32 Native frames; technical acceptance, no final scientific claim.'
    snapshot.setdefault('defaults', {}).update(native_enabled=True, energy_enabled=False)
    snapshot.setdefault('data', {})['validation_items'] = {'classification': ITEMS, 'detection': ITEMS}
    snapshot.setdefault('quality', {}).update(profile_id='v281_corrections_fixed16', dataset_tier='screening', bootstrap_repetitions=BOOTSTRAPS)
    snapshot.setdefault('runtime', {}).setdefault('benchmark', {})['timeout_s'] = 1200
    snapshot['runtime'].setdefault('parallel', {}).update(remote_setups=False, max_setups=1)
    snapshot['runtime'].setdefault('native', {}).update(backends=[family], frames=32, warmup=3,
        repetitions=2, build_missing_engines=bool(scope['allowed_missing']))
    snapshot.setdefault('build', {}).setdefault('scheduler', {}).update(max_workers=1, prefetch_deepx_full=False)
    if build_environment is not None:
        for section in (profile.setdefault('hailo_build', {}), snapshot['build'].setdefault('hailo', {})):
            section['compute_by_family'] = copy.deepcopy(build_environment.get('compute_by_family') or {})
    if hailo8_dependency_manifest is not None and build_environment is None:
        manifest = Path(hailo8_dependency_manifest).expanduser().resolve(strict=True)
        if not manifest.is_file():
            raise ValueError('verified_hailo8_manifest_missing')
        for section in (profile.setdefault('hailo_build', {}), snapshot['build'].setdefault('hailo', {})):
            entry = section.setdefault('compute_by_family', {}).get('hailo8')
            if isinstance(entry, dict) and entry.get('device') == 'gpu' and 'dependency_manifest' not in entry:
                entry['dependency_manifest'] = str(manifest)
    profile['name'] = 'v281_corrections_' + family
    profile['purpose'] = 'Normal workflow acceptance of compiler context, shared Native input, layout and exact known build exclusions; no model tuning.'
    profile['model_suite'] = {'primary': [chosen[m] for m in scope['models']], 'reserve': []}
    profile.setdefault('selection_policy', {}).update(max_accepted_cases_per_model=max(map(len, scope['models'].values())),
        forced_cases=copy.deepcopy(scope['models']), score_independent_audit_enabled=False)
    wanted = {'ort_tensorrt', scope['full_id'], scope['split_id']}
    profile['run_profiles'] = [r for r in profile.get('run_profiles', []) if r.get('id') in wanted]
    if {r.get('id') for r in profile['run_profiles']} != wanted:
        raise ValueError('required_run_profiles_missing:' + family)
    for row in profile['run_profiles']:
        if row['id'] == 'ort_tensorrt':
            row['variants'] = ['full']
    hardware = profile.setdefault('hardware', {})
    targets = hardware.get('resolved_targets', [])
    ids = [r.get('id') for r in targets]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_frozen_hardware_setup')
    selected = [r for r in targets if r.get('id') == scope['setup']]
    if len(selected) != 1:
        raise ValueError('required_frozen_hardware_setup_missing:' + scope['setup'])
    hardware.update(selected_setups=[scope['setup']], selected_groups=[], resolved_targets=selected,
                    resolved_targets_sha256=snapshot_payload_sha256(selected))
    profile['hailo_build'].update(hw_arch=family, targets=[family])
    profile['artifact_cache_preflight'] = {'enabled': True, 'default_expectation': 'warm',
        'block_on_unexpected_cold_builds': True,
        'expected_cold': [{'model_id': m, 'item_id': case} for m, cases in scope['allowed_missing'].items() for case in cases]}
    native = profile.setdefault('native_producers', {})
    native.update(enabled=True, backends=[family], split_backends=[family], models=list(scope['models']),
        case_policy='case_map_only', case_map=copy.deepcopy(scope['models']), frames=32, warmup=3,
        repetitions=2, build_missing_engines=bool(scope['allowed_missing']))
    native['full_baselines'] = {'enabled': True, 'backends': [family, 'tensorrt']}
    native.setdefault('energy', {})['enabled'] = False
    native['energy'].setdefault('window_method_validation_probe', {})['enabled'] = False
    profile.setdefault('energy', {})['enabled'] = False
    preset.update(follow_tool_config=False, snapshot=snapshot, overrides={'native_enabled': True, 'energy_enabled': False})
    profile['execution_preset'] = preset
    profile, _ = apply_run_mode(profile, follow_tool_config=False)
    if cache_verify_guard(profile):
        raise ValueError('global_cache_verify_only_forbidden')
    for field in ('mode', 'preset', 'optimization_level', 'calib_count', 'calib_batch_size', 'cache_integrity'):
        if profile['hailo_build'].get(field) != original_recipe.get(field):
            raise ValueError('compiler_recipe_was_changed:' + field)
    if profile['hailo_build']['force_build'] is not False or profile['deepx_build']['force_build'] is not False:
        raise ValueError('force_build_forbidden')
    if profile['validation_execution']['max_items'] != {'classification': ITEMS, 'detection': ITEMS}:
        raise ValueError('diagnostic_population_was_overridden')
    if original_path.read_bytes() != before:
        raise ValueError('source_profile_changed_during_preparation')
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'profile_fixed16.yaml'
    destination.write_text(yaml.safe_dump(profile, sort_keys=False), encoding='utf-8')
    loaded, _ = load_runtime_profile_snapshot(str(destination))
    if loaded['selection_policy']['forced_cases'] != scope['models']:
        raise ValueError('serialized_profile_changed_boundaries')
    report = {'status': 'prepared_not_executed', 'family': family, 'source_run': str(source),
        'source_profile_sha256': hashlib.sha256(before).hexdigest(), 'profile': str(destination),
        'scope': copy.deepcopy(scope), 'validation_items': ITEMS, 'bootstrap_repetitions': BOOTSTRAPS,
        'calibration_items': profile['hailo_build']['calib_count'], 'compiler_recipe_unchanged': True,
        'force_build': False, 'energy_execution': 'not_requested', 'scientific_final_claim': False}
    _write(output / 'preparation.json', report)
    return destination, report


def _identity(row):
    return str(row.get('model') or row.get('model_id')), str(row.get('backend')), str(row.get('case') or row.get('case_id'))


def inspect_run(run_dir, family):
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    run = Path(run_dir).resolve(strict=True)
    scope = SCOPE[family]
    report = {'technical_acceptance': 'incomplete', 'family': family, 'run_directory': str(run),
              'quality_decision': 'not_evaluated', 'errors': [], 'scientific_final_claim': False}
    try:
        profile, _ = load_runtime_profile_snapshot(str(run / 'profile.yaml'))
        if profile['selection_policy']['forced_cases'] != scope['models'] or profile['hardware']['selected_setups'] != [scope['setup']]:
            raise ValueError('normal_run_scope_mismatch')
        manifest = _read(run / 'run_manifest.json')
        report['quality_decision'] = manifest.get('aggregate_quality_decision', 'not_evaluated')
        matrix = _read(run / 'reports/native_expected_matrix.json')
        split_backend = family + '_to_trt'
        expected = {(m, split_backend, c) for m, cases in scope['models'].items() for c in cases}
        expected |= {(m, 'native_full_' + b, 'full') for m in scope['models'] for b in (family, 'tensorrt')}
        exclusions = {(m, split_backend, c) for m, cases in scope['known_exclusions'].items() for c in cases}
        success_rows = matrix.get('successful_expected_rows', [])
        excluded_rows = matrix.get('excluded_expected_rows', [])
        if any(r.get('setup_id') != scope['setup'] for r in success_rows + excluded_rows):
            raise ValueError('unexpected_native_hardware_setup')
        if matrix.get('failed_expected_rows') or matrix.get('missing_expected_rows'):
            raise ValueError('native_failed_or_missing_observation')
        success = {_identity(r) for r in success_rows if r.get('actual_ok') is True and r.get('setup_id') == scope['setup']}
        excluded = {_identity(r) for r in excluded_rows if r.get('setup_id') == scope['setup']}
        if len(success_rows) != len(success) or len(excluded_rows) != len(excluded):
            raise ValueError('duplicate_or_unsuccessful_native_observation')
        if success != expected - exclusions or excluded != exclusions:
            raise ValueError('native_success_or_exact_exclusion_coverage_mismatch')
        if (matrix.get('technical_execution_complete') is not True or matrix.get('failed_expected_row_count') != 0
                or matrix.get('missing_expected_row_count') != 0):
            raise ValueError('native_technical_terminal_completion_missing')
        from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
        for row in excluded_rows:
            exclusion = known_build_exclusion(row)
            if not exclusion or (exclusion.get('build_evidence') or {}).get('state') != 'COMPILE_INFEASIBLE':
                raise ValueError('known_negative_record_binding_invalid')
            if row.get('repetition_count_attempted', 0) or row.get('repetition_count_valid', 0):
                raise ValueError('known_negative_was_dispatched')
        if family == 'hailo8':
            shared = _read(run / 'native_producers/hailo8/yolo11l/benchmark_set/native_pipeline/b067/hailo_to_trt/uint8_dequant_fp16/native_fifo_results.json')
            relation = shared.get('endpoint_relation') or {}
            if shared.get('endpoint_relation_verified') is not True or any(relation.get(k) is not True for k in ('prepared_input_exact', 'boundary_exact', 'raw_outputs_exact')):
                raise ValueError('hailo8_shared_endpoint_input_or_output_not_exact')
            endpoints = shared.get('endpoint_results') or {}
            raw = (endpoints.get('raw_model_outputs') or {}).get('native_command_contract') or {}
            completed = (endpoints.get('completed_task') or {}).get('native_command_contract') or {}
            digest = str(((raw.get('artifacts') or {}).get('prepared_input') or {}).get('sha256') or '').removeprefix('sha256:')
            consumed = str((completed.get('prepared_input_contract') or {}).get('shared_rgb_source_sha256') or '').removeprefix('sha256:')
            if not digest or digest != consumed:
                raise ValueError('hailo8_completed_endpoint_did_not_consume_shared_raw_feed')
            report['hailo8_shared_rgb_sha256'] = digest
        report['native_successful_rows'] = len(success)
        report['known_compile_exclusions'] = excluded_rows
        quality = _read(run / 'quality_management/central_quality_summary.json')
        results = quality.get('results') or []
        if not results or any(r.get('technical_status') != 'completed' or r.get('n') != ITEMS for r in results):
            raise ValueError('quality_execution_or_population_incomplete')
        if any(r.get('decision') not in {'pass', 'fail', 'inconclusive'} for r in results):
            raise ValueError('quality_decision_unresolved')
        for model, cases in scope['models'].items():
            model_results = [r for r in results if r.get('model_id') == model]
            if not model_results:
                raise ValueError('central_results_missing:' + model)
            for case in set(cases) - set(scope['known_exclusions'].get(model, [])):
                if not any(r.get('case_id') == case and r.get('source_run_id') == split_backend and r.get('variant') == 'composed' for r in model_results):
                    raise ValueError('generic_split_quality_missing:' + model + ':' + case)
            reference = _read(run / 'quality_management/references' / model / 'management_cpu_reference_status.json')
            expected_reference = str(reference.get('reference_sha256') or '').removeprefix('sha256:')
            if not expected_reference:
                raise ValueError('canonical_reference_identity_missing:' + model)
            for row in model_results:
                observed = str((row.get('management_cpu_reference') or {}).get('reference_sha256') or '').removeprefix('sha256:')
                if observed != expected_reference:
                    raise ValueError('canonical_reference_binding_mismatch:' + model)
        preflight = _read(run / 'reports/artifact_cache_preflight.json')
        if preflight.get('runtime_dispatch_allowed') is not True or preflight.get('unexpected_cold_builds') != 0 or preflight.get('unknown_count') != 0:
            raise ValueError('cache_admission_not_within_declared_scope')
        allowed = {(m, case) for m, cases in scope['allowed_missing'].items() for case in cases}
        for row in preflight.get('artifact_matrix') or []:
            if row.get('compiler_dispatch_allowed') is True or (row.get('evidence') or {}).get('current_build') is True:
                model = str(row.get('model_id') or '')
                case = str(row.get('boundary') or row.get('item_id') or '').rsplit('/', 1)[-1].split(':', 1)[0]
                if case.isdigit(): case = 'b' + case.zfill(3)
                if (model, case) not in allowed:
                    raise ValueError('unexpected_compiler_dispatch:' + model + ':' + case)
        report['cache_admission'] = {k: preflight.get(k) for k in ('status', 'hit_count', 'cold_builds_required', 'completed_cold_build_count', 'known_infeasible_count')}
        report['central_quality_result_count'] = len(results)
        report['quality_decisions'] = sorted({r['decision'] for r in results})
        if manifest.get('technical_status') != 'ok' or manifest.get('runtime_complete') is not True:
            raise ValueError('normal_workflow_technical_completion_missing')
        closure = _read(run / 'reports/artifact_index_closure.json')
        if closure.get('status') != 'pass' or any(closure.get(k) != 0 for k in ('hash_mismatch_count', 'missing_file_count', 'unindexed_required_path_count', 'verification_error_count')):
            raise ValueError('terminal_closure_incomplete')
        _read(run / 'reports/scientific/report_manifest.json')
        report.update(terminal_closure='pass', technical_acceptance='pass')
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ':' + str(exc))
    return report


def archive_evidence(out):
    target = out.with_name(out.name + '.zip'); temp = target.with_suffix('.zip.part')
    if target.exists() or target.is_symlink():
        raise FileExistsError('existing_evidence_archive:' + str(target))
    try:
        with zipfile.ZipFile(temp, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(out.rglob('*')):
                relative = path.relative_to(out)
                if 'runs' in relative.parts or path.is_symlink() or not path.is_file():
                    continue
                if path.suffix in {'.json', '.yaml', '.log', '.zip'}:
                    if not path.resolve().is_relative_to(out.resolve()):
                        raise ValueError('evidence_outside_output')
                    archive.write(path, out.name + '/' + relative.as_posix())
        with zipfile.ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise ValueError('evidence_crc_failed')
        os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def load_build_environment(profile_path):
    """Resolve current GUI/CLI compute precedence; import no scientific settings."""
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    path = Path(profile_path).expanduser().resolve(strict=True)
    before = path.read_bytes()
    active, _ = load_runtime_profile_snapshot(str(path))
    if path.read_bytes() != before:
        raise ValueError('active_compute_profile_changed_during_read')
    environment = {k: copy.deepcopy(active.get('hailo_build', {}).get(k)) for k in ('compute_by_family',)}
    return environment, {'profile': str(path), 'profile_sha256': hashlib.sha256(before).hexdigest(), **environment}


def execute(source, out, *, hailo8_dependency_manifest=None, compute_profile=None):
    python = ROOT / '.venv/bin/python'
    summary = {'schema': 'onnx-splitpoint/v281-normal-corrections-acceptance', 'technical_acceptance': 'incomplete',
        'quality_decision': 'not_evaluated', 'stage': 'prepare', 'families': {}, 'errors': [],
        'source_run': str(source), 'energy_execution': 'not_requested', 'scientific_final_claim': False}
    code = 2
    try:
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError('tool_venv_missing')
        build_environment = None
        if compute_profile is not None:
            build_environment, summary['active_compiler_environment'] = load_build_environment(compute_profile)
        # Validate both source selections before any expensive compilation.
        profiles = {}
        for family in SCOPE:
            directory = out / family; directory.mkdir()
            profiles[family], preparation = prepare(source, directory / 'prepared', family,
                hailo8_dependency_manifest=hailo8_dependency_manifest, build_environment=build_environment)
            summary['families'][family] = {'preparation': preparation}
        for family, profile in profiles.items():
            directory = out / family
            record = summary['families'][family]
            summary['stage'] = family + '_normal_workflow'
            command = [str(python), '-B', '-m', 'onnx_splitpoint_tool.workflow.run_evaluation',
                '--profile-driven', '--profile', str(profile), '--out', str(directory / 'runs'), '--require-fresh-run']
            record['workflow_command'] = command
            _write(out / 'acceptance.json', summary)
            print('STAGE=normal_workflow_' + family, flush=True)
            try:
                execution = record['workflow'] = run_logged(command, ROOT, directory / 'workflow.log', timeout_s=SCOPE[family]['timeout_s'])
                manifests = list((directory / 'runs').glob('*/run_manifest.json'))
                if len(manifests) != 1:
                    raise ValueError('expected_one_new_run_manifest:' + str(len(manifests)))
                run = manifests[0].parent
                record['inspection'] = inspect_run(run, family)
                print('STAGE=debug_export_' + family, flush=True)
                record['debug_export'] = run_logged([str(python), '-I', '-B', str(ROOT / 'scripts/create_evaluation_debug_pack.py'),
                    '--eval-run-dir', str(run), '--out', str(directory / 'debug_pack.zip')], ROOT, directory / 'debug_export.log', timeout_s=900)
                record['complete'] = (execution['returncode'] == 0 and not execution['cancelled'] and not execution['timed_out']
                    and record['inspection']['technical_acceptance'] == 'pass'
                    and record['debug_export']['returncode'] == 0 and not record['debug_export']['cancelled'] and not record['debug_export']['timed_out']
                    and (directory / 'debug_pack.zip').is_file())
            except Exception as exc:
                record.update(complete=False, error=type(exc).__name__ + ':' + str(exc))
                if hasattr(exc, 'stage_cleanup'):
                    record['controller_failure_cleanup'] = exc.stage_cleanup
            if record.get('workflow', {}).get('cancelled') or record.get('debug_export', {}).get('cancelled'):
                code = 130
                break
        complete = all(r.get('complete') is True for r in summary['families'].values())
        decisions = [r.get('inspection', {}).get('quality_decision', 'not_evaluated') for r in summary['families'].values()]
        summary['quality_decision'] = 'fail' if 'fail' in decisions else ('inconclusive' if 'inconclusive' in decisions else 'pass' if set(decisions) == {'pass'} else 'not_evaluated')
        summary['technical_acceptance'] = 'pass' if complete else 'incomplete'
        summary['stage'] = 'complete' if complete else 'completed_with_errors'
        if code != 130:
            code = 0 if complete else 2
    except (Exception, KeyboardInterrupt) as exc:
        summary['errors'].append(type(exc).__name__ + ':' + str(exc))
        code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    summary['exit_code'] = code
    try:
        _write(out / 'acceptance.json', summary)
        archive = archive_evidence(out)
        print('EVIDENCE_ZIP=' + str(archive), flush=True)
    except Exception as exc:
        summary.update(technical_acceptance='incomplete', exit_code=2)
        summary['errors'].append('evidence_export:' + repr(exc))
        code = 2
        try: _write(out / 'acceptance.json', summary)
        except Exception: pass
        print('EVIDENCE_EXPORT=FAILED; RESULTS_RETAINED=' + str(out), flush=True)
    print('V281_NORMAL_WORKFLOW=' + summary['technical_acceptance'].upper(), flush=True)
    print('QUALITY_DECISION=' + summary['quality_decision'], flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path.home() / 'Models/EvaluationRuns/completsetdev_20260912_213922')
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--hailo8-dependency-manifest')
    parser.add_argument('--compute-profile', type=Path, help='Read only current effective Hailo compute selection; source recipe/hardware and existing managed DFC venv configuration remain unchanged.')
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().absolute(); root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='v281_normal_workflow_', dir=root))
    print('REPORT_DIR=' + str(out), flush=True)
    return execute(args.source_run.expanduser().absolute(), out, hailo8_dependency_manifest=args.hailo8_dependency_manifest, compute_profile=args.compute_profile)


if __name__ == '__main__': raise SystemExit(main())
