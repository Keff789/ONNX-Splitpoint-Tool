#!/usr/bin/env python3
"""One bounded v2.80.4 normal CLS+DET workflow and its ordinary debug export.

Uses existing profile, cache admission, CPU-reference, producer and Native paths.
There is no alternate inference or quality algorithm. Quality FAIL is a completed
measurement, distinct from a failed or incomplete technical acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = {'mobilenet_v3_large': ('classification', 'b056'), 'yolo11l': ('detection', 'b062')}
SETUP = 'orin_nx_hailo10_01'
ITEMS = 500
BOOTSTRAPS = 500


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing_or_unsafe_file:' + str(path))
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('json_object_required:' + str(path))
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _legacy_gate():
    spec = importlib.util.spec_from_file_location('reference_gate_v2803_base', ROOT / 'scripts/reference_workflow_gate_v2803.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reuse_gate():
    spec = importlib.util.spec_from_file_location('hailo_reuse_v2804', ROOT / 'scripts/hailo_reuse_gate_v2804.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(source_run, output, *, source_profile=None, hailo8_dependency_manifest=None):
    """Prepare both predeclared boundaries after validating their source suites.

    The source is explicit. Creating an output directory can never influence
    source selection; in particular there is no newest-directory heuristic.
    """
    import yaml
    from onnx_splitpoint_tool.run_modes import apply_run_mode
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
    source = Path(source_run).expanduser().resolve(strict=True)
    out = Path(output).expanduser().absolute()
    if not source.is_dir() or source == out.resolve() or source in out.resolve().parents:
        raise ValueError('output_must_be_outside_existing_source_run')
    source_profile = Path(source_profile).expanduser() if source_profile else source / 'profile.yaml'
    before = source_profile.read_bytes()
    original, _ = load_runtime_profile_snapshot(str(source_profile))
    # Reuse the .3 FIX5 model/task/case/original-model, frozen-target and recipe
    # checks. This helper writes nothing and never reloads the mutable registry.
    profile = _legacy_gate().prepare_profile(source_profile, source, require_originals=True)
    mode = copy.deepcopy(profile['execution_preset']['snapshot'])
    mode['label'] = 'v2.80.4 Standard CLS+DET workflow acceptance'
    mode['description'] = '500 validation items and 500 bootstraps; separate normal workflow, no energy or scientific Final claim.'
    mode['data']['validation_items'] = {'classification': ITEMS, 'detection': ITEMS}
    mode['quality'].update(profile_id='v2804_standard_workflow_500', dataset_tier='development', bootstrap_repetitions=BOOTSTRAPS)
    mode['runtime']['benchmark']['timeout_s'] = 3600
    profile['execution_preset']['snapshot'] = mode
    profile['name'] = 'v2804_standard_reference_native'
    profile['purpose'] = 'Normal CPU reference, Hailo10/TRT Full and one existing split per model, central Quality and Native completion; quality outcomes remain unchanged.'
    profile, _ = apply_run_mode(profile, follow_tool_config=False)
    if cache_verify_guard(profile):
        raise ValueError('global_cache_verify_only_forbidden')
    if profile['validation_execution']['max_items'] != {'classification': ITEMS, 'detection': ITEMS}:
        raise ValueError('standard_validation_scope_overridden')
    if profile['quality_gate']['statistics']['bootstrap_repetitions'] != BOOTSTRAPS:
        raise ValueError('standard_bootstrap_scope_overridden')
    for field in ('compute_by_family',):
        if profile['hailo_build'].get(field) != original['hailo_build'].get(field):
            raise ValueError('saved_compute_configuration_was_modified:' + field)
    if hailo8_dependency_manifest is not None:
        families = copy.deepcopy(profile['hailo_build'].get('compute_by_family') or {})
        families.setdefault('hailo8', {}).update(device='gpu', dependency_manifest=str(hailo8_dependency_manifest))
        profile['hailo_build']['compute_by_family'] = families
        profile, _ = apply_run_mode(profile, follow_tool_config=False)
    if source_profile.read_bytes() != before:
        raise ValueError('source_profile_changed_during_preparation')
    out.mkdir(parents=True, exist_ok=False)
    destination = out / 'profile_standard500.yaml'
    destination.write_text(yaml.safe_dump(profile, sort_keys=False), encoding='utf-8')
    loaded, _ = load_runtime_profile_snapshot(str(destination))
    if loaded['selection_policy']['forced_cases'] != {m: [v[1]] for m, v in MODELS.items()}:
        raise ValueError('serialized_profile_changed_selected_boundaries')
    report = {'status': 'prepared_not_executed', 'source_run': str(source), 'source_profile': str(source_profile),
              'profile': str(destination), 'models_and_cases': MODELS, 'setup': SETUP,
              'validation_items_per_model': ITEMS, 'calibration_items': 500, 'bootstrap_repetitions': BOOTSTRAPS,
              'force_build': False, 'unexpected_cold_builds': 'blocked_by_normal_admission',
              'energy_execution': 'not_requested', 'hardware_execution': 'not_run', 'scientific_final_claim': False}
    _write(out / 'preparation.json', report)
    return destination, report


def _safe_member(run, relative):
    relative = Path(str(relative or ''))
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe_run_relative_member:' + str(relative))
    path = run
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError('symlink_run_member:' + str(path))
    if not path.is_file():
        raise ValueError('missing_run_member:' + str(path))
    return path


def require_central_endpoint_coverage(seen, model, case):
    if ('hailo10h_to_trt', case, 'composed') not in seen:
        raise ValueError('bound_split_quality_missing:' + model)
    if not any(('hailo10', address, 'full') in seen for address in (case, 'full')):
        raise ValueError('hailo_full_quality_missing:' + model)
    if ('native_full_tensorrt', 'full', 'full') not in seen:
        raise ValueError('native_tensorrt_full_quality_missing:' + model)


def inspect_run(run_dir):
    """Inspect completed normal evidence; no metric recomputation or inference."""
    from concurrent.futures import Future
    from dataclasses import replace
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
    from onnx_splitpoint_tool.management_reference import _source_contract
    from onnx_splitpoint_tool.trt_quality_chain import load_producer_set_from_central_quality_summary
    from onnx_splitpoint_tool.quality_service import quality_request_from_manifest, prepare_evaluation
    from onnx_splitpoint_tool.quality_replay import _management_reference_member
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    run = Path(run_dir).expanduser().resolve(strict=True)
    manifest = _read(run / 'run_manifest.json')
    profile, _ = load_runtime_profile_snapshot(str(run / 'profile.yaml'))
    if profile['selection_policy']['forced_cases'] != {m: [v[1]] for m, v in MODELS.items()}:
        raise ValueError('normal_run_scope_mismatch')
    if profile['validation_execution']['max_items'] != {'classification': ITEMS, 'detection': ITEMS}:
        raise ValueError('normal_run_population_mismatch')
    report = {'technical_acceptance': 'incomplete', 'quality_decision': manifest.get('aggregate_quality_decision', 'not_evaluated'),
              'scientific_final_claim': False, 'run_directory': str(run), 'models': [], 'errors': [],
              'g6_cold_build': 'not_run_warm_scope_only', 'energy_execution': 'not_requested'}
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile='', out=str(run.parent)))
    runner.run_dir, runner.run_id, runner.profile_payload = run, str(manifest.get('run_id') or run.name), profile
    try:
        summary_path = run / 'quality_management/central_quality_summary.json'
        summary = _read(summary_path)
        for model, (task, case) in MODELS.items():
            suite = run / 'models' / model / 'benchmark_set/legacy_suite'
            plan, contract = _read(suite / 'benchmark_plan.json'), _read(suite / 'benchmark_set.json')
            expected_cases = {str(row.get('case_id') or row.get('case') or f"b{int(row.get('boundary', -1)):03d}") for row in contract.get('cases', [])}
            if expected_cases != {case}:
                raise ValueError('generated_case_scope_mismatch:' + model)
            status = _read(run / 'quality_management/references' / model / 'management_cpu_reference_status.json')
            future = Future(); future.set_result(status)
            runner._management_reference_futures[model] = future
            runner._management_reference_source_contracts[model] = _source_contract(suite, plan, contract)
            records, consumed, _ = runner._management_reference_records(model)
            if consumed.get('task') != task or len(records) != ITEMS:
                raise ValueError('canonical_reference_task_or_population_mismatch:' + model)
            results = [r for r in summary.get('results', []) if r.get('model_id') == model]
            if not results:
                raise ValueError('central_results_missing:' + model)
            seen = set()
            legacy_full = 0
            for result in results:
                if result.get('technical_status') != 'completed' or result.get('n') != ITEMS:
                    raise ValueError('central_result_incomplete:' + model)
                if result.get('task') != task or result.get('decision') not in {'pass', 'fail', 'inconclusive'}:
                    raise ValueError('central_result_task_or_decision_invalid:' + model)
                actual_reference = result.get('management_cpu_reference') or {}
                for field in ('source_contract_sha256', 'reference_sha256', 'quality_contract_sha256'):
                    expected = str(consumed.get(field) or '').removeprefix('sha256:')
                    actual = str(actual_reference.get(field) or '').removeprefix('sha256:')
                    optional = field == 'quality_contract_sha256' and task == 'classification' and not expected and not actual
                    if not optional and (not expected or actual != expected):
                        raise ValueError('central_reference_binding_mismatch:' + model + ':' + field)
                request_path = _safe_member(run, result.get('source_request'))
                if hashlib.sha256(request_path.read_bytes()).hexdigest() != str(result.get('source_request_sha256') or '').removeprefix('sha256:'):
                    raise ValueError('central_request_byte_binding_mismatch:' + model)
                # Production request reader validates candidate bytes, descriptor
                # identities and required Native/quality-first producer contracts.
                reference_path = _management_reference_member(actual_reference, run, model, label='canonical_reference')
                request = quality_request_from_manifest(request_path, verify_artifacts=True,
                    reference_artifact={'path': str(reference_path), 'sha256': actual_reference['reference_sha256'],
                                        'size_bytes': actual_reference['reference_size_bytes']})
                request = replace(request, reference_identity=str(result.get('reference_identity') or request.reference_identity or ''))
                fingerprint, mathematical = prepare_evaluation(request)
                if len(mathematical['image_ids']) != ITEMS or request.repetitions != BOOTSTRAPS:
                    raise ValueError('candidate_population_or_bootstrap_mismatch:' + model)
                for field in ('reference_predictions_sha256', 'candidate_predictions_sha256', 'annotations_sha256'):
                    if str(result.get(field) or '').removeprefix('sha256:') != mathematical[field]:
                        raise ValueError('evaluated_prediction_binding_mismatch:' + model + ':' + field)
                if str(result.get('evaluation_fingerprint') or '').removeprefix('sha256:') != fingerprint:
                    raise ValueError('evaluated_fingerprint_mismatch:' + model)
                result_case = str(result.get('case_id') or '')
                result_setup = str(result.get('source_setup_id') or result.get('setup_id') or '')
                if result_case not in {'full', case} or result_setup != SETUP:
                    raise ValueError('central_endpoint_scope_mismatch:' + model)
                identity = result.get('request_identity') or {}
                run_id = str(result.get('eval_run_id') or '')
                if identity.get('identity_valid') is not True or identity.get('identity_errors'):
                    raise ValueError('central_request_identity_invalid:' + model)
                if run_id != runner.run_id:
                    raw = _read(request_path)
                    if (run_id or result.get('variant') != 'full' or result.get('source_run_id') not in {'hailo10', 'ort_tensorrt'}
                            or result.get('collection_eval_run_id') != runner.run_id
                            or result.get('eval_run_binding_status') != 'workflow_request_collection'
                            or str(raw.get('eval_run_id') or '')
                            or result.get('artifact_provenance_binding_status') != 'not_required'
                            or result.get('artifact_provenance_claim_eligible') is not False
                            or result.get('producer_binding_eligible') is not False
                            or raw.get('producer_identity') or raw.get('native_split_quality_binding')):
                        raise ValueError('central_run_binding_mismatch:' + model)
                    # New .4 explicitly records the collection run; this is not
                    # a fabricated physical producer run attestation.
                    legacy_full += 1
                seen.add((result.get('source_run_id'), result_case, result.get('variant')))
            require_central_endpoint_coverage(seen, model, case)
            load_producer_set_from_central_quality_summary(summary_path, eval_run_id=runner.run_id, setup_id=SETUP, model_ids=[model])
            report['models'].append({'model': model, 'case': case, 'reference_records': len(records),
                                     'central_result_count': len(results), 'producer_binding': 'verified',
                                     'full_workflow_collection_binding_count': legacy_full,
                                     'quality_decisions': sorted({r['decision'] for r in results})})
        preflight = _read(run / 'reports/artifact_cache_preflight.json')
        if (preflight.get('runtime_dispatch_allowed') is not True or preflight.get('cold_builds_required') != 0
                or preflight.get('confirmed_miss_count') != 0 or preflight.get('unknown_count') != 0):
            raise ValueError('normal_admission_not_all_warm')
        report['cache_admission'] = {k: preflight.get(k) for k in ('status', 'hit_count', 'confirmed_miss_count', 'unknown_count', 'cold_builds_required')}
        matrix = _read(run / 'reports/native_expected_matrix.json')
        expected = {(m, backend, case if backend == 'hailo10h_to_trt' else 'full') for m, (_, case) in MODELS.items()
                    for backend in ('hailo10h_to_trt', 'native_full_hailo10h', 'native_full_tensorrt')}
        actual = {(row.get('model'), row.get('backend'), row.get('case')) for row in matrix.get('successful_expected_rows', [])
                  if row.get('actual_ok') is True and row.get('setup_id') == SETUP}
        if matrix.get('execution_success_complete') is not True or actual != expected:
            raise ValueError('required_native_execution_incomplete')
        report['native_successful_rows'] = len(actual)
        if manifest.get('technical_status') != 'ok' or manifest.get('runtime_complete') is not True:
            raise ValueError('normal_workflow_technical_completion_missing')
        closure = _read(run / 'reports/artifact_index_closure.json')
        if closure.get('status') != 'pass' or any(closure.get(k) != 0 for k in ('hash_mismatch_count', 'missing_file_count', 'unindexed_required_path_count', 'verification_error_count')):
            raise ValueError('artifact_index_terminal_closure_incomplete')
        _read(run / 'reports/scientific/report_manifest.json')
        report['terminal_closure'] = 'pass'
        report['technical_acceptance'] = 'pass'
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ':' + str(exc))
    finally:
        runner._shutdown_management_services()
    return report


def run_logged(command, cwd, log, *, timeout_s=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', ORT_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1')
    started = last_output = time.monotonic()
    cancelled_at = None
    termination = None
    timed_out = False
    with Path(log).open('wb', buffering=0) as output, Path(log).open('rb') as reader:
        child = subprocess.Popen(command, cwd=str(cwd), stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        def cancel(signum, frame):
            nonlocal cancelled_at
            if cancelled_at is None:
                cancelled_at = time.monotonic()
                print('CANCELLATION_REQUESTED=YES; waiting for workflow cleanup', flush=True)
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
        old_int, old_term = signal.signal(signal.SIGINT, cancel), signal.signal(signal.SIGTERM, cancel)
        try:
            while True:
                chunk = reader.read(65536)
                if chunk:
                    print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
                    last_output = time.monotonic()
                elif child.poll() is not None:
                    break
                else:
                    now = time.monotonic()
                    if now - last_output >= 60:
                        print(f'WORKFLOW_RUNNING=YES ELAPSED_S={int(now-started)} LIVE_LOG={log}', flush=True)
                        last_output = now
                    if timeout_s is not None and now - started >= timeout_s and cancelled_at is None:
                        timed_out = True
                        cancel(signal.SIGTERM, None)
                    if cancelled_at is not None and child.poll() is None:
                        if now - cancelled_at > 150 and termination != 'kill':
                            os.killpg(child.pid, signal.SIGKILL); termination = 'kill'
                        elif now - cancelled_at > 120 and termination is None:
                            os.killpg(child.pid, signal.SIGTERM); termination = 'term'
                    time.sleep(0.2)
            rc = child.wait()
        except BaseException as primary:
            cleanup = {'owned_process_group': child.pid, 'status': 'not_needed'}
            if child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=15)
                        cleanup['status'] = 'terminated_after_controller_error'
                        try:
                            os.killpg(child.pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            os.killpg(child.pid, signal.SIGKILL)
                            cleanup['status'] = 'killed_residual_owned_group_after_controller_error'
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                        cleanup['status'] = 'killed_owned_group_after_controller_error'
                except Exception as error:
                    cleanup.update(status='failed', error=type(error).__name__ + ':' + str(error))
            primary.workflow_cleanup = cleanup
            raise
        finally:
            signal.signal(signal.SIGINT, old_int); signal.signal(signal.SIGTERM, old_term)
    return {'returncode': rc, 'cancelled': cancelled_at is not None, 'timed_out': timed_out, 'forced_termination': termination,
            'elapsed_s': round(time.monotonic() - started, 3)}


def archive_evidence(out):
    archive = out.with_name(out.name + '.zip')
    partial = archive.with_suffix('.zip.part')
    if archive.exists():
        raise FileExistsError('existing_evidence_archive:' + str(archive))
    members = [p for p in out.iterdir() if p.is_file() and not p.is_symlink()]
    for folder in ('prepared', 'reuse', 'source_export'):
        directory = out / folder
        if directory.is_dir():
            members.extend(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in {'.json', '.yaml', '.log', '.zip'})
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for path in sorted(set(members)):
                if not path.resolve().is_relative_to(out.resolve()):
                    raise ValueError('evidence_path_outside_output')
                zf.write(path, out.name + '/' + path.relative_to(out).as_posix())
        with zipfile.ZipFile(partial) as verification:
            if verification.testzip() is not None:
                raise ValueError('evidence_archive_CRC_failure')
        # Atomic no-overwrite publication within the same output filesystem.
        os.link(partial, archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


def execute(source, out, *, source_profile=None, export_source=False, hailo8_dependency_manifest=None):
    python = ROOT / '.venv/bin/python'  # Keep the venv path; do not resolve its symlink.
    summary = {'schema': 'onnx-splitpoint/v2804-standard-workflow-acceptance', 'technical_acceptance': 'incomplete',
               'quality_decision': 'not_evaluated', 'stage': 'prepare', 'errors': [],
               'source_run': str(source), 'energy_execution': 'not_requested', 'scientific_final_claim': False}
    code = 2
    try:
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError('tool_venv_missing:' + str(python))
        print('STAGE=prepare_standard_CLS_DET', flush=True)
        profile, summary['preparation'] = prepare(source, out / 'prepared', source_profile=source_profile, hailo8_dependency_manifest=hailo8_dependency_manifest)
        if export_source:
            print('STAGE=normal_export_existing_source', flush=True)
            summary['source_debug_export'] = run_logged([str(python), '-I', '-B', str(ROOT / 'scripts/verify_debug_export_v2804.py'),
                '--run', str(source), '--output-dir', str(out / 'source_export')], ROOT, out / 'source_debug_export.log', timeout_s=900)
            source_report = out / 'source_export/debug_export_acceptance.json'
            summary['source_debug_verification'] = _read(source_report) if source_report.is_file() else {'status': 'missing'}
            if summary['source_debug_export']['returncode'] or summary['source_debug_export']['cancelled'] or summary['source_debug_verification'].get('status') != 'PASS':
                raise ValueError('normal_source_debug_export_failed')
        print('STAGE=two_fresh_Hailo8_reuse_processes', flush=True)
        reuse = _reuse_gate().collect(source, profile, out / 'reuse', python)
        summary['hailo8_reuse'] = reuse
        if reuse['status'] != 'two_fresh_process_hits_pass':
            raise ValueError('hailo8_reuse_gate_incomplete; see reuse/summary.json')
        summary['stage'] = 'normal_workflow'
        command = [str(python), '-B', '-m', 'onnx_splitpoint_tool.workflow.run_evaluation', '--profile-driven',
                   '--profile', str(profile), '--out', str(out / 'runs'), '--require-fresh-run']
        summary['workflow_command'] = command
        _write(out / 'acceptance.json', summary)
        print('STAGE=normal_workflow_reference_quality_native_report', flush=True)
        execution = summary['workflow'] = run_logged(command, ROOT, out / 'workflow.log')
        manifests = list((out / 'runs').glob('*/run_manifest.json'))
        if len(manifests) != 1:
            raise ValueError('expected_exactly_one_new_run_manifest:' + str(len(manifests)))
        run = manifests[0].parent
        summary['inspection'] = inspect_run(run)
        print('STAGE=normal_debug_export', flush=True)
        summary['debug_export'] = run_logged([str(python), '-I', '-B', str(ROOT / 'scripts/create_evaluation_debug_pack.py'),
            '--eval-run-dir', str(run), '--out', str(out / 'debug_pack.zip')], ROOT, out / 'debug_export.log')
        complete = (execution['returncode'] == 0 and not execution['cancelled']
                    and summary['inspection']['technical_acceptance'] == 'pass'
                    and summary['debug_export']['returncode'] == 0 and (out / 'debug_pack.zip').is_file())
        summary['technical_acceptance'] = 'pass' if complete else 'incomplete'
        summary['quality_decision'] = summary['inspection']['quality_decision']
        summary['stage'] = 'complete' if complete else 'completed_with_errors'
        code = 0 if complete else (130 if execution['cancelled'] else 2)
    except (Exception, KeyboardInterrupt) as exc:
        if hasattr(exc, 'workflow_cleanup'):
            summary['controller_failure_cleanup'] = exc.workflow_cleanup
        summary['errors'].append(type(exc).__name__ + ':' + str(exc))
        code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    summary['exit_code'] = code
    _write(out / 'acceptance.json', summary)
    try:
        archive = archive_evidence(out)
        print('EVIDENCE_ZIP=' + str(archive), flush=True)
    except Exception as exc:
        summary.update(technical_acceptance='incomplete', exit_code=2)
        summary['errors'].append('evidence_export:' + repr(exc))
        _write(out / 'acceptance.json', summary)
        code = 2
        print('EVIDENCE_EXPORT=FAILED; RESULTS_RETAINED=' + str(out), flush=True)
    print('V2804_NORMAL_WORKFLOW=' + summary['technical_acceptance'].upper(), flush=True)
    print('QUALITY_DECISION=' + str(summary['quality_decision']), flush=True)
    for error in summary['errors']:
        print('ERROR=' + error, flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True, help='Exact existing two-model run; no automatic newest-directory selection.')
    parser.add_argument('--profile', type=Path, help='Optional existing profile; defaults to source-run/profile.yaml.')
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--hailo8-dependency-manifest', help='Explicit optional H8 manifest for the new private acceptance profile; no global settings are saved.')
    parser.add_argument('--export-source', action='store_true', help='Also export the original night run through the ordinary exporter (G1).')
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().absolute(); root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='v2804_normal_workflow_', dir=root))
    print('REPORT_DIR=' + str(out), flush=True)
    return execute(args.source_run.expanduser().absolute(), out, source_profile=args.profile, export_source=args.export_source, hailo8_dependency_manifest=args.hailo8_dependency_manifest)


if __name__ == '__main__':
    raise SystemExit(main())
