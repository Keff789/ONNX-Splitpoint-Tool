"""Bounded post-planned YOLO augmentation using the normal product stages.

The closed campaign is a read-only reference. This entry never resumes it.
Native's existing per-case coordinator gates its first leaves through the
existing Generic completed-task runner before releasing the rest or energy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import traceback

from ..yolo_augmentation_quality import OriginalCPUReferenceMixin
from .runner import EvaluationWorkflowRunner
from .contracts import WorkflowOptions
from . import generic_completed_task as completion


def key(row):
    return (str(row.get('model_id') or row.get('model') or ''),
            str(row.get('case_id') or row.get('case') or ''), str(row.get('setup_id') or ''))


def load_frozen_plan(path: Path):
    plan = completion.read_json(path)
    if (plan.get('schema') != 'onnx-splitpoint/yolo-min3-augmentation-plan'
            or plan.get('cohort_id') != 'yolo_min3_capability_augmentation'
            or plan.get('frozen') is not True or plan.get('force_rebuild') is not False):
        raise ValueError('a frozen, non-force augmentation plan is required')
    rows = plan['rows']; primary = [r for r in rows if r['selection_role'] == 'primary']
    reserve = [r for r in rows if r['selection_role'] == 'reserve']
    if (len(primary) != 12 or len(reserve) > 9 or len(rows) > 21
            or len({key(r) for r in rows}) != len(rows)
            or any(r['model_id'] not in {'yolo11l', 'yolo26s', 'yolov7_paper'} for r in rows)
            or any(len(r['part2_external_inputs']) != 1 for r in rows)):
        raise ValueError('augmentation identity or structural scope drift')
    from ..generic_completion_plan import SETUPS
    expected = {(model, case, setup) for model in ('yolo11l', 'yolo26s', 'yolov7_paper')
        for case in plan['models'][model]['primary_cases'] for setup in SETUPS}
    if ({key(r) for r in primary} != expected
            or any(len(plan['models'][m]['primary_cases']) != n for m, n in
                   (('yolo11l', 1), ('yolo26s', 1), ('yolov7_paper', 2)))):
        raise ValueError('augmentation primary model/setup distribution drift')
    expected_reserves = {(model, case, setup) for model in ('yolo11l', 'yolo26s', 'yolov7_paper')
        for case in plan['models'][model]['reserve_cases'] for setup in SETUPS}
    if ({key(r) for r in reserve} != expected_reserves
            or any(len(plan['models'][m]['reserve_cases']) > 1 for m in ('yolo11l', 'yolo26s', 'yolov7_paper'))
            or any(r['backend'] != SETUPS.get(r['setup_id']) for r in rows)):
        raise ValueError('augmentation reserve/setup distribution drift')
    base = completion.read_json(Path(plan['base_execution_plan']))
    if {key(r) for r in rows} & {key(r) for r in base['rows']}:
        raise ValueError('augmentation would repeat an existing completed case')
    for row in rows:
        if (row['warmup_count'], row['task_count'], row['repetitions']) != (100, 1000, 3):
            raise ValueError('augmentation measurement budget changed')
    starts = [r for r in primary if r['case_key'] in plan['start_case_keys']]
    if len(starts) != 3 or len({r['setup_id'] for r in starts}) != 3:
        raise ValueError('augmentation needs exactly one frozen start case per setup')
    original = Path(plan['original_run']).resolve()
    output = completion.outside_primary(Path(plan['output_root']), original)
    completion.outside_primary(output, Path(plan['base_completion_root']))
    new_run = Path(plan['augmentation_run']).resolve()
    if not new_run.is_relative_to(output) or new_run.name == original.name:
        raise ValueError('augmentation must own a separate new run identity')
    return plan


class YOLOAugmentationRunner(OriginalCPUReferenceMixin, EvaluationWorkflowRunner):
    def __init__(self, options, plan_path, *, augmentation_resume=False, execution_output=None, **kwargs):
        self.augmentation_plan_path = Path(plan_path).resolve()
        self.augmentation_plan = load_frozen_plan(self.augmentation_plan_path)
        self.augmentation_reference_run = Path(self.augmentation_plan['original_run'])
        self.augmentation_resume = augmentation_resume
        self._augmentation_first_quality_ready = Future()
        self.augmentation_execution_output = execution_output_path(self.augmentation_plan, execution_output)
        if augmentation_resume:
            from .yolo_augmentation_reentry import validate_augmentation_reentry
            validate_augmentation_reentry(self.augmentation_plan['augmentation_run'], self.augmentation_plan_path)
            if not options.resume or not options.resume_missing:
                raise ValueError('augmentation reentry requires normal selective completion')
        super().__init__(options, **kwargs)

    def _supplement_compatibility_reviewed(self, original):
        if not self.augmentation_resume or original.get('tool_version') != '2.91.2':
            return False
        from .yolo_augmentation_reentry import validate_augmentation_reentry
        validate_augmentation_reentry(self.run_dir, self.augmentation_plan_path)
        return True

    def _prepare_selective_completion(self):
        super()._prepare_selective_completion()
        if not self.augmentation_resume:
            return
        from ..yolo_augmentation_quality import augmentation_quality_source_rows
        from .supplement import write_plan
        self._supplement_plan['quality_source_rows'] = augmentation_quality_source_rows(self, self._supplement_plan)
        write_plan(self._supplement_plan, self._supplement_original_root.parent)

    def _supplement_dispatch_rows(self, model_id):
        rows = super()._supplement_dispatch_rows(model_id)
        if rows is not None and self.augmentation_resume:
            rows += [dict(row) for row in self._supplement_plan.get('quality_source_rows', [])
                     if row.get('model_id') == model_id]
        return rows

    def _accept_first_quality(self, *, results_root=None, required=False):
        """Publish the first real Quality result and run the normal Native binder."""
        from ..yolo_augmentation_quality import _quality_key
        if self._augmentation_first_quality_ready.done():
            self._augmentation_first_quality_ready.result()
            return True
        first = ('yolo11l', 'b120', 'orin_nx_hailo8_01')
        if first not in {key(row) for row in self.augmentation_plan['rows']
                         if row.get('selection_role') == 'primary'}:
            raise ValueError('augmentation first Quality identity is absent')
        self._queue_central_quality_requests(first[0], results_root=results_root)
        results = list(self._preserved_central_quality_results)
        for request, future in list(self._central_quality_futures.items()):
            identity = self._quality_request_identity(Path(request), model_id=first[0])
            if key(identity) == first:
                results.append(future.result())
        if not any(_quality_key(result)[:3] == first for result in results):
            if required:
                raise ValueError('augmentation first Quality request/result missing after H8 download')
            return False
        summary_path = self._publish_case_quality(results)
        from ..resources.remote_scripts import run_evalrun_native_producer_variants as coordinator
        cfg = self._native_producer_config()
        # Match the normal Native stage's per-case expansion, which happens
        # after _native_producer_config for profiles without explicit variants.
        cfg['native_release_mode'] = 'per_case'
        if not cfg.get('variants'):
            cfg['variants'] = self._normal_native_release_variants(cfg)
            cfg['remotes'] = self._materialize_variant_native_remotes(cfg)
        leaves = coordinator._per_case_variants(self.run_dir, cfg, cfg['variants'])
        selected = [leaf for leaf in leaves
                    if coordinator._augmentation_leaf_key(cfg, leaf) == '|'.join(first)]
        if len(selected) != 1:
            raise ValueError('augmentation first Native leaf is not unique')
        prepared, bindings, split_plan, _, _ = coordinator._prepare_case_release(
            self.run_dir, cfg, selected[0], completion.read_json(summary_path))
        evidence = {'case_key': '|'.join(first), 'central_quality_summary': str(summary_path),
                    'prepared_native_leaf': prepared, 'native_split_quality_bindings': bindings,
                    'native_split_quality_plan': split_plan,
                    'scope': 'Quality publication and Native binding; no Native performance dispatch'}
        completion.write_json(self.augmentation_execution_output / 'first-quality-native-binding.json', evidence)
        self._augmentation_first_quality_ready.set_result(evidence)
        self._emit_log('[augmentation] first H8/yolo11l/b120 Quality and Native binding accepted; '
                       'remaining Quality sources released')
        return True

    def _stage_run_benchmarks_steps(self, model_id, row, *, dispatch_manager=None,
                                    on_setup_complete=None, dispatch_log=None):
        from .execution_binding import PhysicalSetupDispatchManager
        ready = self._augmentation_first_quality_ready
        first_model, first_setup = 'yolo11l', 'orin_nx_hailo8_01'
        owner = dispatch_manager is None
        manager = dispatch_manager or PhysicalSetupDispatchManager(max_workers=3, model_ids=[model_id])
        first_download_completed = False

        def fail(exc):
            if not ready.done():
                ready.set_exception(exc)

        class GatedDispatch:
            first_registered = False

            def submit_group(inner, model, physical, function, *args, **kwargs):
                group = args[0]
                is_first = model == first_model and group.get('target_id') == first_setup
                if is_first:
                    inner.first_registered = True
                    def first_group(*group_args, **group_kwargs):
                        try:
                            output = function(*group_args, **group_kwargs)
                            if not ready.done():
                                if (not first_download_completed or not output.get('statuses')
                                        or any(status != 'ok' for status in output['statuses'])
                                        or self._cancel_event.is_set()):
                                    raise ValueError('augmentation first H8 group failed before Quality release: '
                                                     + str(output.get('records') or output.get('statuses')))
                                root = self.run_dir / 'models' / model / 'benchmark_results'
                                if self.options.resume_missing:
                                    root = root / 'attempts' / self.session_id
                                self._accept_first_quality(results_root=root, required=True)
                            return output
                        except BaseException as exc:
                            fail(exc)
                            raise
                    return manager.submit_group(model, physical, first_group, *args, **kwargs)
                return manager.submit_group(model, physical, function, *args, readiness=ready, **kwargs)

            def finish_registration(inner, model):
                if model == first_model and not inner.first_registered and not ready.done():
                    fail(ValueError('augmentation first H8 Quality group was not selected'))
                return manager.finish_registration(model)

        def completed(model, setup, artifacts):
            nonlocal first_download_completed
            if model == first_model and setup == first_setup:
                first_download_completed = True
            if on_setup_complete is not None:
                on_setup_complete(model, setup, artifacts)

        try:
            # Retained requests use the normal queue/cache reader before any
            # new physical work. Never manufacture a positive release result.
            if model_id == first_model and not ready.done():
                self._accept_first_quality()
            if owner and model_id != first_model and not ready.done():
                raise ValueError('augmentation first Quality must precede standalone later-model dispatch')
            result = yield from super()._stage_run_benchmarks_steps(model_id, row,
                dispatch_manager=GatedDispatch(), on_setup_complete=completed, dispatch_log=dispatch_log)
            if model_id == first_model and not ready.done():
                raise ValueError('augmentation first H8 Quality group did not run or reuse')
            return result
        except BaseException as exc:
            fail(exc)
            raise
        finally:
            if owner:
                manager.shutdown(wait=True, cancel_futures=True)

    def _stage_generate_benchmark_set(self, model_id, row):
        result = super()._stage_generate_benchmark_set(model_id, row)
        _records, status, _bytes = self._augmentation_original_reference(model_id)
        descriptor = {k: status[k] for k in ('quality_contract', 'reference_path',
            'reference_sha256', 'reference_size_bytes', 'source_evaluation_run_id')}
        descriptor.update(reference_role='canonical_cpu_ort',
            quality_contract_sha256=status['quality_contract']['quality_contract_sha256'])
        root = self.run_dir / 'models' / model_id / 'benchmark_set'
        # These are the new normal writer's plans, before the first dispatch.
        # The source artifacts and all old canonical suites remain untouched.
        paths = [root / 'benchmark_plan.json', root / 'legacy_suite/benchmark_plan.json']
        found = 0
        for path in paths:
            if not path.is_file():
                continue
            payload = completion.read_json(path)
            payload['management_cpu_reference'] = descriptor
            from ..management_reference import is_cpu_reference_recipe
            def checked_run(run):
                if is_cpu_reference_recipe(run):
                    if (run.get('semantic_reference_only') is not True
                            or run.get('performance_eligible') is not False
                            or run.get('execution_location') != 'central_management'):
                        raise ValueError('augmentation CPU descriptor is not semantic-only')
                    return
                if 'full' in str(run.get('id', '')).lower():
                    raise ValueError('augmentation generated a forbidden Full run')
                if run.get('variants') != ['composed']:
                    raise ValueError('generated augmentation scope differs from composed-only profile')
            for run in payload.get('runs', []):
                checked_run(run)
            if 'planned_runs' in payload:
                payload['planned_runs'] = list(payload.get('runs') or payload['planned_runs'])
                for run in payload['planned_runs']:
                    checked_run(run)
            completion.write_json(path, payload)
            found += 1
        if found != 2:
            raise ValueError('normal generated augmentation plans missing')
        return result

    def _native_producer_config(self):
        cfg = super()._native_producer_config()
        from .checkpoints import canonical_json_sha256
        from ..yolo_augmentation_plan import collect_technical_terminals
        plan = self.augmentation_plan
        starts = [r for r in plan['rows'] if r['case_key'] in plan['start_case_keys']]
        cfg['augmentation_startcase_barrier'] = {
            'plan_path': str(self.augmentation_plan_path),
            'plan_sha256': canonical_json_sha256(plan),
            'startcases': [{k: r[k] for k in ('model_id', 'case_id', 'setup_id', 'backend')} for r in starts],
            'technical_terminals': collect_technical_terminals(plan, Path(plan['augmentation_run'])),
            'original_reference_validation_summary': plan['original_reference_validation_summary']}
        cfg['augmentation_startcase_barrier']['execution_output'] = str(self.augmentation_execution_output)
        if self.augmentation_resume:
            from .yolo_augmentation_reentry import validate_augmentation_reentry
            cfg['augmentation_startcase_barrier']['reentry'] = validate_augmentation_reentry(
                self.run_dir, self.augmentation_plan_path, self._supplement_original_root)
        return cfg

    def _stage_artifact_cache_preflight(self, rows):
        result = super()._stage_artifact_cache_preflight(rows)
        if not self._artifact_cache_preflight_pending:
            raise RuntimeError('augmentation cache matrix was not prepared behind the compiler fence')
        import shutil
        destination = self.augmentation_execution_output / 'plan/cache-build-matrix'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('artifact_cache_preflight.json', 'artifact_cache_preflight.csv',
                     'artifact_cache_preflight_items.csv', 'artifact_cache_preflight.md',
                     'remote_trt_cache_preflight.json'):
            source = self.run_dir / 'reports' / name
            if source.is_file():
                shutil.copy2(source, destination / name)
        return result


def _normal_pairs(new_run, *, native_run_dir=None):
    from .cross_runner_reporting import compute_cross_runner_report
    from .scientific_reporting import _load_evalrun_rows, _policy_from_profile
    import yaml
    profile = yaml.safe_load((new_run / 'profile.yaml').read_text())
    rows, _, _ = _load_evalrun_rows(new_run, profile, _policy_from_profile(profile))
    pairs = compute_cross_runner_report(new_run, rows, minimum_candidates=3, native_run_dir=native_run_dir,
                                      generic_input_source='normalized_benchmark_results')['pairs']
    for pair in pairs:
        # Use the exact seven-dimensional join performed by the normal
        # reporter, not a second weaker model/case/setup join here.
        source = pair.get('generic_source_path')
        if not source:
            raise ValueError('new Generic pair has no bound original result source')
        pair['original_source'] = source
    return pairs


def _completion_source_pairs(new_run, pairs, native_rows):
    """Use the new bound candidate recipe while retaining old proxy timings.

    The original raw reports deliberately have no new Quality request. Their
    normal metadata overlay names the later candidate report. The ordinary
    completion binder still validates that report's full request, artifacts,
    seven-dimensional identity and Native command before any dispatch.
    """
    from ..generic_completion_plan import _inside
    result = []
    for pair in pairs:
        pair = dict(pair)
        path = new_run / 'models' / pair['model_id'] / 'benchmark_results/normalized_results.json'
        rows = completion.read_json(path).get('results', [])
        matches = [row for row in rows if key(row) == key(pair)
                   and row.get('source_path') == pair['generic_source_path']]
        if len(matches) != 1:
            raise ValueError('augmentation completion has no unique preserved Generic source')
        overlay = matches[0].get('supplement_quality_source')
        if overlay:
            sha = str(overlay.get('source_request_sha256') or '')
            natives = [row for row in native_rows if key(row) == key(pair)
                       and row.get('source_request_sha256') == sha and sha]
            if len(natives) != 1:
                raise ValueError('augmentation completion candidate does not bind its Native request')
            request = _inside(new_run / str(overlay.get('source_request') or ''), new_run)
            if completion.file_sha256(request).removeprefix('sha256:') != sha.removeprefix('sha256:'):
                raise ValueError('augmentation completion candidate request changed')
            source = _inside(str(overlay.get('candidate_report_source') or ''), new_run)
            if overlay.get('preserved_performance_source') != pair['generic_source_path']:
                raise ValueError('augmentation completion lost its original performance source')
            pair['preserved_generic_performance_source'] = pair['generic_source_path']
            pair['completion_generic_source_path'] = str(source)
            pair['original_source'] = str(source)
        result.append(pair)
    return result


@contextmanager
def completion_setup_resources(new_run, setup, target, registry, case_key):
    """Admit Generic through the inherited Native controller broker, if any.

    A broker owns its DUT lock; acquiring another one in this process would
    conflict. Standalone callbacks retain the same physical-resource lock.
    """
    from ..process_control import bind_workflow_resource_options
    from ..remote.process_lease import ControllerResourceClaim
    attempt = (Path(new_run) / 'reports/augmentation_completion' / setup
               / case_key.replace('|', '_')).resolve()
    if not attempt.is_relative_to(Path(new_run).resolve()):
        raise ValueError('completion resource attempt outside augmentation run')
    attempt.mkdir(parents=True, exist_ok=True)
    with bind_workflow_resource_options(controller_setup_id=setup,
                                        controller_attempt_dir=str(attempt)):
        claim = ControllerResourceClaim.from_environment(setup_id=setup,
            attempt_dir=attempt, purpose='dut_job')
        lock = None
        if claim is not None:
            claim.acquire()
        else:
            lock = completion.EvaluationRunLock.for_resource(completion.physical_dut_key(target),
                owner={'run_id': Path(new_run).name, 'session_id': Path(new_run).name,
                       'purpose': 'yolo_min3_completion'})
            lock.acquire()
        try:
            yield
        finally:
            pending = registry.active_count()
            if pending:
                registry.poison()
                evidence = {'reason': 'augmentation_completion_cleanup_unproven',
                            'pending_operations': pending,
                            'broker_claim_left_owned_until_process_exit': claim is not None}
                completion.write_json(attempt / 'cleanup-unproven.json', evidence)
                if lock is not None:
                    lock.commit_quarantine_fence(evidence)
                # Do not finish an unproven broker claim. The callback exits
                # technically; the existing owner-lost path fences this DUT.
            elif claim is not None:
                claim.finish({'process_started': False})
            if lock is not None:
                lock.release()
            if pending:
                raise RuntimeError('augmentation completion cleanup remains unresolved')


def execution_output_path(plan, attempt_dir=None):
    root = Path(plan['output_root']).resolve()
    output = Path(attempt_dir).resolve() if attempt_dir is not None else root
    if output != root and (output.parent != root / 'attempts' or output.is_symlink()):
        raise ValueError('augmentation attempt output must be a direct child of its attempts directory')
    return output


def _same_completion_reference(original, current):
    """Keep the original reference when only its local validation now succeeds.

    The normal binder has rechecked the actual Native command and output bytes.
    A previously unavailable diagnostic may become verified; it does not alter
    the retained reference, runtime configuration, repetitions or claim gates.
    """
    if original == current:
        return True
    if (original.get('historical_semantic_ok') == 'unavailable'
            and current.get('historical_semantic_ok') is True):
        comparable = dict(current)
        comparable['historical_semantic_ok'] = 'unavailable'
        return original == comparable
    return False


def completion_barrier(plan_path, native_summary, validation_summary, case_key, attempt_dir=None,
                       native_report_root=None):
    """Normal coordinator callback: real Native evidence → Generic → pair gate."""
    from ..generic_completion_plan import bind_completion_rows, completion_identity
    from ..generic_completion_reporting import write_completion_reports, completion_pair, validate_completion_case
    plan = load_frozen_plan(plan_path)
    execution_output = execution_output_path(plan, attempt_dir)
    output = Path(plan['output_root']) / 'completion'
    new_run = Path(plan['augmentation_run'])
    allowed = {key(r): r for r in plan['rows'] if r['selection_role'] == 'primary'}
    native = completion.read_json(native_summary)
    measured = {key(r) for r in native['rows'] if r.get('status') == 'ok'}
    if not measured <= set(allowed):
        raise ValueError('Native callback includes a case outside the frozen active plan')
    requested = measured if case_key == 'remaining' else {tuple(case_key.split('|'))}
    if not requested or not requested <= measured:
        raise ValueError('completion barrier lacks its successful Native measurement')
    if native_report_root is not None:
        native_report_root = Path(native_report_root).resolve()
        if (not native_report_root.is_relative_to(new_run.resolve() / 'reports/rest_attempts')
                or Path(native_summary).resolve() != native_report_root / 'reports/native_producer_combined_summary.json'
                or Path(validation_summary).resolve() != native_report_root / 'reports/native_validation/native_producer_validation_summary.json'):
            raise ValueError('completion derived Native reports escape their bound rest attempt')
    normal_pairs = (_normal_pairs(new_run, native_run_dir=native_report_root)
                    if native_report_root is not None else _normal_pairs(new_run))
    pairs = [p for p in normal_pairs if key(p) in requested]
    if {key(p) for p in pairs} != requested:
        raise ValueError('normal reporter could not join every new Native/Generic identity')
    pairs = _completion_source_pairs(new_run, pairs, native['rows'])
    from ..generic_completion_plan import IDENTITY_FIELDS
    scope = [{**{k: p[k] for k in IDENTITY_FIELDS}, 'original_source': p['original_source']} for p in pairs]
    bound = bind_completion_rows(scope, pairs, native_summary, validation_summary, new_run,
        scope_source=str(plan_path), start_selection=lambda rows: [])
    for row in bound['rows']:
        planned = allowed[key(row)]
        if (completion_identity(row) != completion_identity(planned)
                or row['model_sha256'] != planned['model_sha256']):
            raise ValueError('new completion measurement differs from its frozen model/stratum')
    # The whole Native stage writes this projection only after all leaves.
    # The start barrier needs the same normal read-only comparison now.
    from ..resources.remote_scripts.native_host_telemetry import compare_root
    completion.write_json(new_run / 'reports/native_host_telemetry_summary.json',
                          compare_root(new_run / 'native_producers'))
    execution_id = new_run.name
    targets = completion.bound_targets(new_run, bound['rows'])
    runtime_path = output / 'execution-plan.json'
    accumulated = completion.read_json(runtime_path) if runtime_path.is_file() else {
        **bound, 'rows': [], 'execution_id': execution_id, 'output_root': str(output)}
    previous = {r['case_key']: r for r in accumulated['rows']}
    for row in bound['rows']:
        if row['case_key'] in previous:
            old = previous[row['case_key']]
            for field in ('native_command_contract_sha256', 'model_sha256', 'generic_original_quality_request',
                          'native_measurement_boundary', 'measurement_boundary_comparison',
                          'expected_runtime_artifacts', 'input_binding', 'completion_contract', 'functional_reference'):
                unchanged = (_same_completion_reference(old[field], row[field])
                             if field == 'functional_reference' else old[field] == row[field])
                if not unchanged:
                    raise ValueError('new completion reentry evidence changed: ' + field)
            continue
        row['result_path'] = str(output / 'results' / row['case_output_relative'] / 'generic_completion_results.json')
        try:
            runtime = completion.original_generic_environment(row)
            row.update(remote_python_executable=runtime['python'], remote_environment=runtime['environment'],
                generic_environment_source=runtime['source'], generic_environment_original_line=runtime['original_line'])
            target = targets[row['setup_id']]
            remote_root = f"/home/{target['remote']['user']}/splitpoint_runs/yolo_min3_completion/{execution_id}"
            config, files = completion.runtime_configuration(row, remote_root, execution_id)
            row.update(runtime_config=config, staging_files=files, remote_root=remote_root)
        except Exception as exc:
            row.update(execution_status='blocked', dispatch_attempted=False,
                       execution_error=f'{type(exc).__name__}: {exc}')
            previous[row['case_key']] = row
            accumulated['rows'] = list(previous.values())
            _completion_plan_counts(plan, accumulated)
            completion.write_json(runtime_path, accumulated)
            write_completion_reports(accumulated, output / 'results', output / 'reports')
            raise
        previous[row['case_key']] = row
    accumulated['rows'] = list(previous.values())
    _completion_plan_counts(plan, accumulated)
    completion.write_json(runtime_path, accumulated)
    return _run_completion_requests(plan, accumulated, requested, targets, case_key,
                                    execution_output=execution_output)


def _completion_plan_counts(plan, accumulated):
    count = len(accumulated['rows'])
    accumulated.update(row_count=count, planned_repetitions=3 * count,
                       planned_measured_completions=3000 * count,
                       authorized_primary_case_count=plan['primary_count'])


def _run_completion_requests(plan, accumulated, requested, targets, case_key, *, execution_output=None):
    """Preserve completed cases and publish real partial work on every exit."""
    from ..generic_completion_reporting import (
        load_valid_repetitions, write_completion_reports, completion_pair, validate_completion_case,
    )
    output = Path(plan['output_root']) / 'completion'
    new_run = Path(plan['augmentation_run'])
    runtime_path = output / 'execution-plan.json'
    execution_id = new_run.name
    execution_output = execution_output_path(plan, execution_output)
    bundle = execution_output / 'runner-bundle.tar.gz'
    registry = completion.RemoteProcessLeaseRegistry()
    from ..remote.process_lease import RemoteProcessLeaseJournal
    inherited = RemoteProcessLeaseJournal.from_environment(required=False)
    scope_lease = inherited.scope if inherited else completion.RemoteProcessLeaseScope(new_run.name, execution_id + '_completion')
    registry.configure_journal(scope=scope_lease, journal_dir=inherited.directory if inherited else output / 'remote-process-leases' / execution_id)
    current = None
    try:
        pending_rows = []
        completed_count = 0
        for row in accumulated['rows']:
            current = row
            path = Path(row['result_path'])
            valid = load_valid_repetitions(path, row)
            if len(valid) == 3:
                observed = validate_completion_case(completion.read_json(path), row, result_root=path.parent)
                pair = completion_pair(observed, row)
                if not pair['eligible_for_technical_transfer']:
                    raise RuntimeError('preserved Native/Generic completed-task pair is not technically valid: ' + row['case_key'])
                completed_count += 1
                # No new device snapshot may replace the recorded conditions of
                # a completed measurement, even in the final remaining callback.
                continue
            if key(row) in requested:
                pending_rows.append(row)
        current = None
        if pending_rows and not bundle.is_file():
            current = pending_rows[0]
            raise ValueError('tested session-local Generic runner bundle is missing')
        for setup, target in targets.items():
            rows = [r for r in pending_rows if r['setup_id'] == setup]
            if not rows:
                continue
            current = rows[0]
            with completion_setup_resources(new_run, setup, target, registry, case_key):
                transport = completion.transport_for(target, output / 'diagnostics' / setup,
                    scope=scope_lease, registry=registry, cancel=threading.Event())
                snapshot_dir = output / 'configuration' / setup
                ordinal = 1 + sum(1 for _ in snapshot_dir.glob('*.json'))
                snapshot_path = snapshot_dir / (case_key.replace('|', '_') + f'_{ordinal:04d}.json')
                while snapshot_path.exists():
                    ordinal += 1
                    snapshot_path = snapshot_dir / (case_key.replace('|', '_') + f'_{ordinal:04d}.json')
                snapshot = completion.capture_configuration(transport, target, primary=new_run,
                    output=snapshot_path, phase='generic_pre')
                completion.verify_runtime_environment(transport, rows[0], snapshot_path.with_name(snapshot_path.stem + '_runtime.json'))
                remote_root = rows[0]['remote_root']
                # A repaired continuation owns new runner bytes. Keep the old
                # bundle and the bound case/config/result paths unchanged.
                runner_root = remote_root + '/attempts/' + execution_output.name
                completion.upload_new_file(transport, bundle, runner_root + '/runner-bundle.tar.gz')
                completion.checked_remote('tar -xzf ' + shlex.quote(runner_root + '/runner-bundle.tar.gz')
                    + ' -C ' + shlex.quote(runner_root), transport)
                for row in rows:
                    current = row
                    valid = load_valid_repetitions(row['result_path'], row)
                    completed = sorted(rep['repetition_index'] for rep in valid)
                    comparison = completion.configuration_comparison(row, snapshot, snapshot_path)
                    row.setdefault('completion_environment_observations', []).append({
                        'observed_at': completion.now(), 'comparison': comparison,
                        'preserved_repetition_indices': completed,
                        'pending_repetition_indices': [i for i in (1, 2, 3) if i not in completed]})
                    if not completed:
                        row['runtime_environment_comparison'] = comparison
                    retained = row.get('runtime_environment_comparison') or {}
                    if any(item.get('status') in {'conflict', 'incompatible', 'failed'}
                           for item in (retained, comparison)):
                        raise RuntimeError('new or preserved runtime configuration conflicts with Native evidence')
                    if (completed and retained.get('status') in {'compatible', 'verified_compatible'}
                            and comparison.get('status') not in {'compatible', 'verified_compatible'}):
                        raise RuntimeError('partial reentry cannot preserve a verified runtime comparability claim')
                    completion.write_json(runtime_path, accumulated)
                    completion.run_case(row, transport, output, completed_count + 1,
                                        total=plan['primary_count'], runner_root=runner_root)
                    observed = validate_completion_case(completion.read_json(Path(row['result_path'])), row,
                        result_root=Path(row['result_path']).parent)
                    pair = completion_pair(observed, row)
                    completion.write_json(output / 'pair-checks' / (completion.case_name(row) + '.json'), pair)
                    if not pair['eligible_for_technical_transfer']:
                        raise RuntimeError('new Native/Generic completed-task pair failed: ' + row['case_key'])
                    row['execution_status'] = 'completed'
                    row.pop('execution_error', None)
                    row['completion_observed_at'] = completion.now()
                    completed_count += 1
                    completion.write_json(runtime_path, accumulated)
                    summary = write_completion_reports(accumulated, output / 'results', output / 'reports')
                    completion.write_json(execution_output / 'progress.json', {
                        'updated_at': completion.now(), 'phase': 'generic_completion',
                        'last_completed_case': row['case_key'], 'summary': summary})
    except BaseException as exc:
        if current is not None:
            current['execution_status'] = 'failed' if current.get('dispatch_attempted') else 'blocked'
            current['execution_error'] = f'{type(exc).__name__}: {exc}'
            current.setdefault('execution_error_history', []).append({
                'at': completion.now(), 'error': current['execution_error'],
                'dispatch_attempted': current.get('dispatch_attempted', False)})
        raise
    finally:
        pending = registry.active_count()
        completion.write_json(output / 'cleanup-status.json', {'at': completion.now(),
            'active_owned_remote_operations': pending, 'registry_cancelled_or_poisoned': registry.cancelled,
            'foreign_processes_stopped': False})
        _completion_plan_counts(plan, accumulated)
        completion.write_json(runtime_path, accumulated)
        summary = write_completion_reports(accumulated, output / 'results', output / 'reports')
        completion.write_json(execution_output / 'progress.json', {
            'updated_at': completion.now(), 'phase': 'generic_completion', 'summary': summary,
            'callback_case_key': case_key, 'active_owned_remote_operations': pending})
    if pending:
        raise RuntimeError('augmentation completion cleanup remains unresolved')
    return 0


def preservation_inventory(plan):
    original = Path(plan['original_run']); base = Path(plan['base_completion_root'])
    relative = ['run_manifest.json', 'quality_management/central_quality_summary.json',
        'reports/native_producer_summary.json', 'reports/native_producer_combined_summary.json',
        'reports/native_energy_measurements/native_producer_energy_results.json']
    paths = [original / p for p in relative] + [base / 'terminal-status.json', base / 'plan/execution-plan.json',
        base / 'reports/generic_completion_cross_runner.json', base / 'reports/generic_completion_summary.json']
    return {str(p): {'sha256': completion.file_sha256(p), 'size': p.stat().st_size}
            for p in paths if p.is_file()}


def write_final_delivery(plan, status, error, rc, *, execution_output=None,
                         native_summary=None, energy_results=None):
    from ..yolo_augmentation_reporting import write_augmentation_reports
    output = execution_output_path(plan, execution_output); run = Path(plan['augmentation_run'])
    reports = Path(plan['output_root']) / 'completion/reports'
    try:
        summary = write_augmentation_reports(plan,
            reports if reports.is_dir() else None,
            native_summary or (run / 'reports/native_producer_combined_summary.json' if (run / 'reports/native_producer_combined_summary.json').is_file() else None),
            energy_results if energy_results is not None else (run if (run / 'reports/native_energy_measurements/native_producer_energy_results.json').is_file() else None),
            output_root=output / 'reports')
    except Exception as exc:
        traceback.print_exc()
        summary = {'reporting_status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        completion.write_json(output / 'delivery-error.json', {
            **summary, 'workflow_status': status, 'workflow_exitcode': rc})
        status, rc, error = 'delivery_failed', 1, error + '; ' + summary['error']
    terminal = {'finished_at': completion.now(), 'status': status, 'exitcode': rc,
                'summary': summary, 'error': error, 'automatic_restart': False}
    completion.write_json(output / 'terminal-status.json', terminal)
    (output / 'finished_at.txt').write_text(terminal['finished_at'] + '\n')
    (output / 'exitcode.txt').write_text(str(rc) + '\n')
    (output / 'STATUS.txt').write_text(f'THESIS20 YOLO augmentation\nStatus: {status}\n{error}\n' + json.dumps(summary, indent=2) + '\n')
    (output / 'ABSCHLUSSBERICHT.md').write_text(
        '# THESIS20 YOLO-Min3-Augmentierung\n\n'
        f'Status: {status}; Exitcode: {rc}.\n\n{error}\n\n'
        'Post-planned capability/graph cohort; development/screening, kein Hold-out.\n'
        'Basis, neue Kohorte und augmentierte Vereinigung: reports/.\n'
        'Alle Primär- und Reservefälle samt Shortfall: reports/.\n'
        'Ungekürzte FS-Energie bleibt eine eigene Messung; keine Generic-Energie.\n'
        'Kein automatischer Reparatur-/Neustart, kein Release oder Commit.\n\n'
        '```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    paths = ['AUFTRAG.md', str(Path(plan['output_root']) / 'plan/augmentation-plan.json'),
        'STATUS.txt', 'progress.json', 'production-start.json', 'console.log',
        'launch_production.sh', 'launcher.pid', 'cli.pid', 'started_at.txt', 'finished_at.txt', 'exitcode.txt',
        'terminal-status.json', 'cleanup-status.json', 'workflow-session.json', 'result-locations.json',
        str(Path(plan['output_root']) / 'completion'), str(run), 'reports/',
        'ABSCHLUSSBERICHT.md', 'completion-diagnostics.tar.gz']
    (output / 'LIEFERINDEX.txt').write_text('\n'.join(paths) + '\n')
    # Keep the ordinary terminal evidence beside this invocation's log. The
    # underlying requests, measurements and old failed attempts stay in place.
    import shutil
    sources = [run / 'reports/native_producer_stage.json',
               run / 'reports/native_producer_summary.json',
               run / 'reports/native_producer_combined_summary.json',
               run / 'quality_management/case_quality_progress.json',
               run / 'quality_management/central_quality_summary.json',
               Path(plan['output_root']) / 'completion/cleanup-status.json']
    evidence = output / 'reports/product-diagnostics'
    evidence.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.is_file():
            shutil.copy2(source, evidence / source.name)
    completion.write_json(output / 'result-locations.json', {
        'frozen_plan': str(Path(plan['output_root']) / 'plan/augmentation-plan.json'),
        'augmentation_run': str(run), 'completion': str(Path(plan['output_root']) / 'completion'),
        'attempt_output': str(output), 'historical_attempt_output': plan['output_root']})
    completion.terminal_bundle(output)
    return rc


def validated_workflow_options(plan, prepared):
    """Fail before any normal workflow writer can open an unintended run."""
    options = WorkflowOptions(**prepared)
    output = Path(plan['output_root']).resolve()
    run = Path(plan['augmentation_run']).resolve()
    if ((Path(options.out).resolve() / str(options.run_id)) != run
            or Path(options.profile).resolve() != output / 'plan/augmentation-profile.yaml'
            or options.resume or options.resume_missing or options.resume_missing_full_quality_only
            or options.force_stage or options.hailo_force_build or options.hailo_build_full
            or options.hailo_build_part2 or options.result_sources
            or options.execution_mode != 'generate_and_run'
            or options.energy_enabled or options.window_method_validation_probe_enabled is not False):
        raise ValueError('workflow options differ from the separate bounded augmentation')
    return options


def execute(plan_path, *, resume=False, attempt_dir=None):
    plan = load_frozen_plan(plan_path)
    output = execution_output_path(plan, attempt_dir)
    prepared = completion.read_json(Path(plan['output_root']) / 'plan/workflow-options.json')
    options = validated_workflow_options(plan, prepared)
    if resume:
        if attempt_dir is None or output == Path(plan['output_root']).resolve():
            raise ValueError('explicit reentry requires a new attempt directory')
        from .yolo_augmentation_reentry import validate_augmentation_reentry
        predecessor = validate_augmentation_reentry(plan['augmentation_run'], plan_path)
        options.resume = options.resume_missing = True
    elif attempt_dir is not None:
        raise ValueError('attempt directory requires explicit reentry')
    if (output / 'production-start.json').exists():
        raise FileExistsError('this augmentation already entered production; no automatic restart')
    before = preservation_inventory(plan)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'production-start.json').open('x') as stream:
        json.dump({'started_at': completion.now(), 'pid': os.getpid(), 'ppid': os.getppid(),
            'plan': str(plan_path), 'command': sys.argv, 'tmux': os.environ.get('TMUX', ''),
            'explicit_reentry': resume, 'predecessor': predecessor if resume else None}, stream, indent=2)
    completion.write_json(output / 'preservation-before.json', before)
    (output / 'started_at.txt').write_text(completion.now() + '\n')
    def progress(done, total, label):
        completion.write_json(output / 'progress.json', {
            'updated_at': completion.now(), 'phase': 'normal_workflow',
            'done_stages': done, 'total_stages': total, 'last_stage': label,
            'run_dir': plan['augmentation_run'], 'measurement_success_not_implied': True})
        (output / 'STATUS.txt').write_text(
            'THESIS20 YOLO augmentation: running\n' + str(label) + '\n'
            + f'Workflow stages: {done}/{total}; this is not a measurement success count.\n'
            + 'Details: console.log and runs/; no automatic repair or restart.\n')
    progress(0, 0, 'owner preflight pending')
    status, error, rc = 'stopped', '', 1
    runner = None
    try:
        targets = completion.bound_targets(Path(plan['original_run']), [r for r in plan['rows'] if r['selection_role'] == 'primary'])
        completion.observe_owners(targets, output / 'production-preflight')
        runner = YOLOAugmentationRunner(options, plan_path, augmentation_resume=resume,
                                        execution_output=output, log=lambda s: print(s, flush=True),
                                        progress=progress)
        session_id = getattr(runner, 'session_id', '')
        completion.write_json(output / 'workflow-session.json', {
            'workflow_session_id': session_id, 'run_dir': plan['augmentation_run'],
            'native_attempt_directory': str(Path(plan['augmentation_run']) / 'jobs/supplement_attempts'
                                            / session_id / 'native') if resume and session_id else '',
            'historical_checkpoint_retained': resume})
        result = runner.run()
        completion.write_json(output / 'workflow-result.json', result.to_dict() if hasattr(result, 'to_dict') else vars(result))
        status = result.status
        rc = 0 if result.ok else 1
        if not result.ok:
            failures = [row for row in result.stage_results if row.get('status') == 'failed'
                        or row.get('state') == 'failed']
            error = '; '.join(str(row.get('notes') or row.get('error') or row.get('metrics', {}).get('failure_reason')
                                 or row.get('stage') or row) for row in failures)
            if not error:
                error = 'workflow failed: ' + str(result.status)
            print('[augmentation] ' + error, flush=True)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
        completion.write_json(output / 'technical-stop.json', {'at': completion.now(), 'error': error,
            'automatic_restart': False})
    finally:
        after = preservation_inventory(plan)
        completion.write_json(output / 'preservation-after.json', {'unchanged': before == after, 'files': after})
        if before != after:
            status, rc, error = 'preservation_conflict', 1, error + '; protected original inventory changed'
        registry = getattr(runner, '_remote_process_registry', None)
        completion.write_json(output / 'cleanup-status.json', {'at': completion.now(),
            'active_owned_remote_operations': registry.active_count() if registry is not None else None,
            'foreign_processes_stopped': False, 'normal_workflow_finalization': str(Path(plan['augmentation_run']) / 'reports')})
        try:
            rc = write_final_delivery(plan, status, error, rc, execution_output=output)
        except Exception as exc:
            traceback.print_exc()
            completion.write_json(output / 'delivery-error.json', {'error': str(exc), 'workflow_status': status})
            rc = 1
            completion.write_json(output / 'terminal-status.json', {
                'finished_at': completion.now(), 'status': 'delivery_failed', 'exitcode': rc,
                'workflow_status': status, 'error': str(exc), 'automatic_restart': False})
            (output / 'finished_at.txt').write_text(completion.now() + '\n')
            (output / 'exitcode.txt').write_text(str(rc) + '\n')
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute', type=Path)
    group.add_argument('--completion-barrier', type=Path)
    group.add_argument('--finish-rest', type=Path)
    parser.add_argument('--include-b066', action='store_true', help='After block 1, execute the separately admitted H8/b066 block')
    parser.add_argument('--resume-rest-energy', type=Path, help='Explicit failed rest attempt whose normal energy evidence is preserved')
    parser.add_argument('--test-recovered-energy-source', action='store_true', help='One explicitly authorized GUI energy test before the remaining energy')
    parser.add_argument('--authorize-h8-energy-reentry', action='store_true', help='Explicit H8 source/retry authorization; resume its regular energy directly')
    parser.add_argument('--native-report-root', type=Path)
    parser.add_argument('--native-summary', type=Path)
    parser.add_argument('--validation-summary', type=Path)
    parser.add_argument('--case-key')
    parser.add_argument('--resume', action='store_true', help='Explicit selective reentry after the bound Quality nonstart')
    parser.add_argument('--attempt-dir', type=Path)
    args = parser.parse_args(argv)
    if args.finish_rest:
        if args.resume:
            parser.error('--finish-rest uses a separate attempt, not --resume')
        if args.test_recovered_energy_source and args.resume_rest_energy is None:
            parser.error('--test-recovered-energy-source requires --resume-rest-energy')
        if args.authorize_h8_energy_reentry and (args.resume_rest_energy is None or args.test_recovered_energy_source):
            parser.error('--authorize-h8-energy-reentry requires --resume-rest-energy without a separate source test')
        from .yolo_augmentation_rest import finish_rest
        return finish_rest(args.finish_rest, args.attempt_dir, include_b066=args.include_b066,
                           previous_energy_attempt=args.resume_rest_energy, source_test=args.test_recovered_energy_source,
                           h8_reentry=args.authorize_h8_energy_reentry)
    if args.authorize_h8_energy_reentry:
        parser.error('--authorize-h8-energy-reentry requires --finish-rest')
    if args.test_recovered_energy_source:
        parser.error('--test-recovered-energy-source requires --finish-rest')
    if args.resume_rest_energy:
        parser.error('--resume-rest-energy requires --finish-rest and a new --attempt-dir')
    if args.include_b066:
        parser.error('--include-b066 requires --finish-rest')
    if args.execute:
        return execute(args.execute, resume=args.resume, attempt_dir=args.attempt_dir)
    if args.resume:
        parser.error('--resume is only valid with --execute')
    return completion_barrier(args.completion_barrier, args.native_summary, args.validation_summary,
                              args.case_key, attempt_dir=args.attempt_dir, native_report_root=args.native_report_root)


if __name__ == '__main__':
    raise SystemExit(main())
