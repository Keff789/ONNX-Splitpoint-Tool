"""Explicit one-task SingleInput Native references, separate from Native FPS.

This adapter does not implement inference. It invokes the existing Native
runner with its normal command builder, artifact/quality bindings, transport,
physical locks and process leases. Historical paired admission is unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import tarfile
import threading
import uuid

from ..native_command_contract import successful_runtime_argv, verify_native_command_contract

SCHEMA = 'onnx-splitpoint/native-functional-reference/v1'
PLAN_SCHEMA = 'onnx-splitpoint/native-functional-reference-plan/v1'
AUTHORIZED = {('yolov7_paper', 'orin_nx_deepx_m1_01', case) for case in ('b009', 'b044', 'b066')}
AUTHORIZED.add(('yolo26s', 'orin_nx_hailo10_01', 'b038'))
HISTORICAL_FIELDS = ('identity', 'case_key', 'original_run', 'native_source',
    'native_command_contract', 'native_command_contract_sha256', 'native_measurement_boundary', 'historical_pair',
    'input_binding', 'expected_runtime_artifacts', 'completion_contract',
    'functional_reference', 'generic_original_quality_request', 'native_quality_binding_sha256',
    'generic_native_split_quality_binding', 'generic_raw_source', 'original_manifest', 'model_sha256',
    'original_run_cfg', 'original_local_case_dir', 'original_remote_case_dir')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def bound(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def checked(descriptor):
    require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'sha256'}, 'Reference requires path and sha256')
    path = Path(descriptor['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and str(path.resolve()) == str(path) and sha(path) == descriptor['sha256'], 'Reference source missing, noncanonical or changed')
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def identity(case):
    value = case['identity']
    key = (value['model_id'], value['setup_id'], value['case_id'])
    require(key in AUTHORIZED and case.get('task') == 'detection', 'Functional Native reference is limited to four authorized SingleInput cases')
    for context in (case, case.get('runtime_config') or {}):
        require(not str(context.get('scope_mode', '')).startswith('generic_only')
                and context.get('admission_mode') != 'generic_only'
                and context.get('measurement_mode') != 'generic_only_completed_task'
                and 'generic_artifact_binding' not in context
                and (context.get('functional_reference') or {}).get('kind') != 'serial_compiled_subnets',
                'Generic-only cannot enter paired functional-reference admission')
    require(len(case['input_binding']['entries']) == 1, 'Native functional reference requires exactly one prepared feed')
    entry = case['input_binding']['entries'][0]
    require(entry['shape'] == [640, 640, 3] and entry['dtype'] == 'uint8', 'Original HWC uint8 prepared feed must be preserved')
    return value


def replace_argument(argv, flag, value):
    require(argv.count(flag) <= 1, 'Ambiguous Native argument: ' + flag)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend((flag, str(value)))


def functional_argv(command, *, remote_tool, benchmark, output, runner_sha, quality_binding):
    """Use the ordinary replay builder, then explicitly choose the finite control recipe."""
    argv = successful_runtime_argv(command, duration_s=0., fresh_output_root=output, remote_tool_dir=remote_tool)
    for flag, value in (('--benchmark-set', benchmark), ('--frames', 1), ('--warmup', 0),
                        ('--repetitions', 1), ('--completion-runtime-mode', 'strict'),
                        ('--expected-runner-sha256', runner_sha), ('--native-split-quality-binding', quality_binding),
                        ('--model-id', command['model']), ('--setup-id', command['setup_id']),
                        ('--eval-run-id', command['eval_run_id']), ('--source-run-id', command['backend'])):
        replace_argument(argv, flag, value)
    if command['backend'] == 'deepx_to_trt':
        # S1 preserves the original HWC artifact; SDK-internal batching is not
        # permission to substitute a different 1HWC NPY header or source feed.
        replace_argument(argv, '--deepx-runtime-mode', 'S1')
    for flag in ('--dump-outputs', '--dump-boundary'):
        if flag not in argv:
            argv.append(flag)
    require(not any(flag in argv for flag in ('--build', '--build-missing-engine', '--force-rebuild-engine',
        '--force-rebuild', '--energy-workload-only', '--deepx-numerical-only')), 'Build/energy/extra workload is forbidden')
    return argv


def _verified_historical_native(original, raw, expected_sha):
    from ..native_job_identity import planned_native_identity
    from ..native_split_quality import bind_quality_to_native_split
    from ..native_rate_endpoints import rate_endpoint_fields
    command, status = verify_native_command_contract(raw.get('native_command_contract'), expected_identity={
        'model': original['model_id'], 'case': original['case_id'], 'setup_id': original['setup_id'],
        'backend': original['direction'], 'precision': original['precision']})
    require(command is not None and raw.get('ok') is True
            and command['contract_sha256'] == expected_sha == raw.get('native_command_contract_sha256')
            and command['source_request_sha256'] == raw.get('source_request_sha256'), 'Historical Native command invalid: ' + status)
    consumer = dict(raw)
    sealed_identity = planned_native_identity(command)
    for field, aliases in (
        ('backend', ('backend', 'producer_backend')), ('model', ('model', 'model_id')),
        ('case', ('case', 'case_id', 'boundary')), ('setup_id', ('setup_id', 'measurement_setup_id', 'setup')),
        ('precision', ('runtime_precision_identity', 'execution_precision', 'precision', 'contract')),
        ('comparison_backend', ('comparison_backend',))):
        for alias in aliases:
            require(consumer.get(alias) in (None, '') or
                planned_native_identity({field: consumer[alias]})[field] == sealed_identity[field],
                'Historical Native identity assertion conflicts: ' + alias)
        if all(consumer.get(alias) in (None, '') for alias in aliases):
            consumer[field] = command[field]
    selected, quality_status = bind_quality_to_native_split(native_row=consumer,
        quality_binding=raw.get('native_split_quality_binding'), verification_mode='portable')
    require(selected is not None and selected.get('binding_sha256') == original['native_quality_binding_sha256'],
            'Historical Native portable quality consumer invalid: ' + quality_status)
    return command, rate_endpoint_fields(raw)


def _historical(plan, case):
    original = plan['source_row']
    identity(case)
    require(all(case.get(key) == original.get(key) for key in HISTORICAL_FIELDS), 'Historical paired scientific/source fields changed')
    source_plan = read(checked(plan['source_plan']))
    matches = [row for row in source_plan['rows'] if row['case_key'] == original['case_key']]
    require(len(matches) == 1 and matches[0] == original, 'Original selected source row changed')
    manifest_path = checked(plan['original_run_manifest'])
    require(manifest_path == Path(original['original_run']) / 'run_manifest.json', 'Wrong original EvaluationRun manifest')
    manifest = read(manifest_path)
    source = checked(plan['historical_native_source'])
    require(str(source) == original['native_source'], 'Historical Native source substituted')
    raw = read(source)
    command, rates = _verified_historical_native(original, raw, original['native_command_contract_sha256'])
    require(command['eval_run_id'] == manifest['run_id'], 'Historical artifact/quality EvaluationRun identity differs')
    if Path(original['original_run']).name != manifest['run_id']:
        bindings = [value for value in source_plan.get('source_run_bindings', [])
            if Path(value['archive_root']).resolve() == Path(original['original_run']).resolve()
            and value.get('run_manifest_sha256') == sha(manifest_path)
            and Path(value['historical_root']).name == manifest['run_id']]
        require(len(bindings) == 1, 'Archived EvaluationRun requires its original manifest-bound source mapping')
    boundary = original['native_measurement_boundary']
    if boundary.get('process_repetition_policy') == 'separate_processes':
        science = plan.get('historical_scientific_source')
        require(isinstance(science, dict), 'Separate-process Native science requires all three original sources')
        summary_path = checked(science['summary'])
        summary = read(summary_path)
        require(str(summary_path) == boundary['source'] and summary.get('status') == 'complete'
                and summary.get('source_integrity_verified') is True and len(summary['rows']) == 3
                and len(science['reports']) == 3 and summary['rows'][0]['raw_report'] == str(source),
                'Historical separate-process Native summary differs')
        counts, times, ids, command_shas = [], [], [], []
        for index, (entry, descriptor) in enumerate(zip(summary['rows'], science['reports']), 1):
            path = checked(descriptor)
            require(str(path) == entry['raw_report'] and entry['repeat'] == index, 'Historical scientific report origin/order differs')
            record = read(path)
            verified, projected = _verified_historical_native(original, record, boundary['native_command_contract_sha256s'][index-1])
            require(verified['eval_run_id'] == command['eval_run_id']
                    and verified['source_request_sha256'] == command['source_request_sha256']
                    and verified['input_image_sha256'] == command['input_image_sha256']
                    and verified['prepared_input_contract'] == command['prepared_input_contract'], 'Historical scientific input/run differs')
            for role in ('hef' if original['direction'] == 'hailo10h_to_trt' else 'dxnn', 'engine'):
                require(verified['artifacts'][role] == command['artifacts'][role], 'Historical scientific compiled artifact differs')
            require(record.get('runtime_instance_id') == entry.get('runtime_instance_id')
                    and all(record.get(key) is None or record[key] == entry.get(key)
                            for key in ('runner_pid', 'runner_start_monotonic_ns')),
                    'Historical scientific process identity differs')
            # These old runners record the runtime-instance identity in raw
            # JSON; PID/start are the original orchestration summary's evidence.
            ids.append((entry.get('runner_pid'), entry.get('runner_start_monotonic_ns'), record.get('runtime_instance_id')))
            counts.extend(projected['completed_task_work_unit_counts'])
            times.extend(projected['completed_task_measurement_times_s'])
            command_shas.append(verified['contract_sha256'])
        require(len({values[:2] for values in ids}) == 3 and len({values[2] for values in ids}) == 3
                and all(type(values[0]) is int and values[0] > 0 and type(values[1]) is int and values[1] > 0
                        and isinstance(values[2], str) and bool(values[2]) for values in ids)
                and command_shas == boundary['native_command_contract_sha256s'], 'Historical scientific process/command coverage differs')
        rates = {'completed_task_work_unit_counts': counts, 'completed_task_measurement_times_s': times}
    require(rates.get('completed_task_work_unit_counts') == boundary.get('observed_completed_counts')
            and rates.get('completed_task_measurement_times_s') == boundary.get('observed_measurement_times_s'),
            'Historical Native count/time projection differs')
    request = checked(plan['original_quality_request'])
    require(str(request) == original['generic_original_quality_request'] and sha(request) == command['source_request_sha256'], 'Historical quality request changed')
    require(command['native_split_quality_binding']['binding_sha256'] == original['native_quality_binding_sha256'], 'Historical quality binding changed')
    return original, raw, command


def _recipe_environment(row):
    env = {str(key): str(value) for key, value in row.get('remote_environment', {}).items()
           if key not in {'PYTHONHOME', 'PYTHONPATH'}}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', ONNX_SPLITPOINT_NATIVE_TRT_BUILD='0',
               ONNX_SPLITPOINT_ARTIFACT_POLICY='cache_verify_only')
    return env


def _output_contract_source(row, raw):
    """Bind the exact archived declaration already attested by this Native job."""
    from ..native_output_endpoint import load_authoritative_output_contract
    from ..resources.remote_scripts.native_producer_final_report import _resolve_split_semantic_manifest
    manifest, evidence = _resolve_split_semantic_manifest(Path(row['native_source']), (raw,), kind='output')
    require(manifest is not None and evidence.get('manifest_resolution_status') == 'manifest_verified',
            'Functional output declaration requires the original sealed semantic output manifest')
    recorded = read(manifest).get('authoritative_output_contract_resolution') or {}
    require(recorded.get('authoritative_output_contract') is True
            and recorded.get('contract_resolution_status') == 'attested',
            'Historical Native output declaration was not authoritatively resolved')
    # Follow the same original normal staging authority retained by the Native
    # summary. A suite with a merely equivalent row cannot replace its exact
    # historical container, even if other backend rows caused the difference.
    if (row.get('native_measurement_boundary') or {}).get('process_repetition_policy') == 'separate_processes':
        summary = Path(row['native_measurement_boundary']['source'])
        summary_payload = read(summary)
        staging_plan = checked({'path': summary_payload.get('plan'), 'sha256': summary_payload.get('plan_sha256')})
        matches = [item for item in read(staging_plan).get('stage_files', [])
                   if item.get('remote') == recorded.get('declaration_source')
                   and item.get('sha256') == recorded.get('source_contracts_sha256')]
        require(len(matches) == 1, 'Original normal staging plan does not bind the exact output declaration')
        source = Path(matches[0]['local'])
        origin = {'summary': bound(summary), 'staging_plan': bound(staging_plan)}
    else:
        captured = Path(row['native_source']).parent / 'benchmark_set' / 'output_contracts.json'
        source = captured if captured.exists() or captured.is_symlink() else Path(row['original_local_case_dir']).parent / 'output_contracts.json'
        origin = {'original_suite': str(source.parent.resolve())}
    require(source.is_file() and not source.is_symlink(), 'Archived output declaration is missing or symlinked')
    descriptor = bound(source)
    checked(descriptor)
    require(descriptor['sha256'] == recorded.get('source_contracts_sha256'),
            'Archived output declaration differs from its original semantic manifest')
    declaration = load_authoritative_output_contract(source, backend='tensorrt', model_id=row['model_id'],
                                                     variant='full', task='detection')
    fields = ('model_id', 'task', 'backend', 'variant', 'stage', 'source_contract_sha256', 'source_contracts_sha256')
    require(declaration.get('authoritative_output_contract') is True
            and declaration.get('contract_resolution_status') == 'attested'
            and all(declaration.get(key) == recorded.get(key) for key in fields),
            'Archived output declaration no longer resolves to the original model/endpoint authority')
    return {'source': descriptor, 'semantic_output_manifest': bound(manifest), 'original_authority': origin,
            'declaration': {key: declaration[key] for key in fields}}


def _collected_path(remote_path, benchmark, local_root):
    remote_path, benchmark = PurePosixPath(remote_path), PurePosixPath(benchmark)
    require(remote_path.is_absolute() and benchmark.is_absolute() and '..' not in remote_path.parts
            and '..' not in benchmark.parts, 'Collected artifact path is not canonical')
    path = Path(local_root) / str(remote_path.relative_to(benchmark))
    root = Path(local_root).resolve(strict=True)
    require(path.resolve(strict=True).is_relative_to(root) and path.resolve() == path,
            'Collected artifact escapes its actual benchmark origin')
    return path


def validate_capture_plan(plan):
    """Rebuild the finite ordinary command before any staging or hardware work."""
    from .generic_completed_task import bound_targets
    from .hardware_acceptance_native import _benchmark_case_inputs
    row, raw, command = _historical(plan, plan['source_row'])
    require(plan['schema'] == PLAN_SCHEMA and plan['identity'] == identity(row)
            and plan['purpose'] == 'functional_reference_only_not_scientific_native_performance', 'Wrong functional plan role')
    require(all(plan[key] is False for key in ('build_allowed', 'fallback_allowed', 'automatic_retry', 'energy_capture', 'hardware_executed')),
            'Functional plan must disable builds, fallback, retries and energy')
    require(plan['recipe'] == {'processes': 1, 'completed_tasks': 1, 'warmup_tasks': 0, 'duration_s': 0.,
                              'completion_runtime_mode': 'strict'}, 'Functional plan recipe changed')
    require(re.fullmatch(r'native_functional_[0-9a-f]{32}', plan['execution_id']) is not None, 'Invalid functional execution identity')
    remote = '/home/nx/splitpoint_runs/native_functional_reference/' + plan['execution_id']
    require(plan['remote_root'] == remote and plan['remote_tool_dir'] == remote + '/tool', 'Functional staging root changed')
    output = Path(plan['output_root'])
    require(output.is_absolute() and output.resolve() == output
            and not output.is_relative_to(Path(row['original_run']).resolve()), 'Functional output overlaps history')
    environment_path = checked(plan['runtime_environment_evidence'])
    environment = read(environment_path)
    require(environment == plan['expected_interpreter'] and environment_path.parent.name == row['setup_id']
            and environment.get('matches_original_interpreter') is True and environment.get('missing_modules') == [],
            'Installed interpreter observation changed')
    require(environment['executable_sha256'] == command['artifacts']['python_executable']['sha256']
            and environment['executable'] == row['remote_python_executable']
            and PurePosixPath(environment['executable']).parent == PurePosixPath(command['python_executable']).parent
            and PurePosixPath(environment['executable']).name in ('python', 'python3')
            and PurePosixPath(command['python_executable']).name in ('python', 'python3'),
            'Installed interpreter differs from sealed historical Native interpreter')
    require(plan['target'] == bound_targets(Path(row['original_run']), [row])[row['setup_id']], 'Functional target changed')
    quality = checked(plan['quality_binding_file'])
    require(quality == output / 'native_split_quality_binding.json'
            and read(quality) == command['native_split_quality_binding'], 'Functional quality source changed')
    output_contract = _output_contract_source(row, raw)
    require(plan.get('output_contract_source') == output_contract,
            'Functional output declaration source changed or is missing')
    sources = plan['source_files']
    require(isinstance(sources, dict) and sources and all(
        not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        and re.fullmatch('[0-9a-f]{64}', digest) for name, digest in sources.items()), 'Invalid source closure')
    archive = checked(plan['archive'])
    require(archive == output / 'native-product-bundle.tar.gz', 'Functional private bundle moved')
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        require(len(members) == len(sources) and {value.name for value in members} == set(sources)
                and all(value.isfile() for value in members), 'Private bundle source membership differs')
        require(all(hashlib.sha256(bundle.extractfile(value).read()).hexdigest() == sources[value.name]
                    for value in members), 'Private bundle source bytes differ')
    runner = 'scripts/native_' + ('deepx' if row['direction'] == 'deepx_to_trt' else 'hailo10') + '_trt_e2e_from_benchmarkset.py'
    benchmark = remote + '/' + row['model_id'] + '/benchmark_set'
    remote_output = benchmark + '/native_pipeline/' + row['case_id'] + '/' + row['direction'] + '/' + row['precision']
    argv = functional_argv(command, remote_tool=remote + '/tool', benchmark=benchmark, output=remote_output,
        runner_sha=sources[runner], quality_binding=remote + '/native_split_quality_binding.json')
    expected_item = {'argv': argv, 'env': _recipe_environment(row), 'repeat': 1, 'source_benchmark_set': command['benchmark_set'],
        'benchmark_case_inputs': _benchmark_case_inputs(command, {'backend': row['direction'], 'case': row['case_id']}),
        'remote_output_root': remote_output, 'result_json': remote_output + '/' +
        ('deepx_native_fifo_e2e_results.json' if row['direction'] == 'deepx_to_trt' else 'hailo10_native_fifo_e2e_results.json')}
    require(plan['item'] == expected_item, 'Functional command/finite recipe/environment changed')
    expected = {item['path']: item['sha256'] for item in row['expected_runtime_artifacts'].values()}
    prepared = row['input_binding']['entries'][0]
    expected.update({prepared['remote_path']: prepared['sha256'], command['input_image']: command['input_image_sha256'],
                     command['python_executable']: environment['executable_sha256']})
    expected.update({remote + '/tool/' + name: digest for name, digest in sources.items()})
    expected[benchmark + '/output_contracts.json'] = output_contract['source']['sha256']
    require(plan['expected_remote_files'] == expected, 'Functional artifact/input/source closure differs')
    return command


def prepare_capture(source_plan, case_key, output, *, runtime_environment_evidence):
    """Offline preparation only. Actual execution requires a clean published checkout."""
    from ..remote.bundle import build_suite_bundle
    from .generic_completed_task import bound_targets
    from .hardware_acceptance_native import _benchmark_case_inputs
    source_plan = Path(source_plan).resolve(strict=True)
    rows = [row for row in read(source_plan)['rows'] if row['case_key'] == case_key]
    require(len(rows) == 1, 'Exactly one existing paired case is required')
    row = copy.deepcopy(rows[0])
    ident = identity(row)
    native_raw = read(row['native_source'])
    command, status = verify_native_command_contract(native_raw.get('native_command_contract'))
    require(command is not None, 'Historical artifact source command invalid: ' + status)
    output_contract = _output_contract_source(row, native_raw)
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(Path(row['original_run']).resolve()), 'Fresh reference output must not replace historical evidence')
    environment_source = bound(runtime_environment_evidence)
    environment = read(environment_source['path'])
    require(Path(environment_source['path']).parent.name == row['setup_id']
            and environment.get('matches_original_interpreter') is True and environment.get('missing_modules') == [],
            'A genuine setup-local normal installed-runtime observation is required')
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    runner_name = ('native_deepx_trt_e2e_from_benchmarkset.py' if row['direction'] == 'deepx_to_trt'
                   else 'native_hailo10_trt_e2e_from_benchmarkset.py')
    files = sorted(str(path.relative_to(repository)) for path in (repository / 'onnx_splitpoint_tool').rglob('*.py')
                   if '__pycache__' not in path.parts)
    files += ['scripts/native_hailo10_trt_e2e_from_benchmarkset.py', 'scripts/native_deepx_trt_e2e_from_benchmarkset.py']
    sources = {name: sha(repository / name) for name in files}
    archive = output / 'native-product-bundle.tar.gz'
    build_suite_bundle(repository, archive, includes=files, excludes=[], reuse_if_unchanged=False)
    session = 'native_functional_' + uuid.uuid4().hex
    remote = '/home/nx/splitpoint_runs/native_functional_reference/' + session
    benchmark = remote + '/' + row['model_id'] + '/benchmark_set'
    remote_output = benchmark + '/native_pipeline/' + row['case_id'] + '/' + row['direction'] + '/' + row['precision']
    quality = output / 'native_split_quality_binding.json'
    write_new(quality, command['native_split_quality_binding'])
    argv = functional_argv(command, remote_tool=remote + '/tool', benchmark=benchmark, output=remote_output,
        runner_sha=sources['scripts/' + runner_name], quality_binding=remote + '/native_split_quality_binding.json')
    expected = {item['path']: item['sha256'] for item in row['expected_runtime_artifacts'].values()}
    prepared = row['input_binding']['entries'][0]
    expected[prepared['remote_path']] = prepared['sha256']
    expected[command['input_image']] = command['input_image_sha256']
    expected[command['python_executable']] = environment['executable_sha256']
    expected.update({remote + '/tool/' + name: digest for name, digest in sources.items()})
    expected[benchmark + '/output_contracts.json'] = output_contract['source']['sha256']
    env = _recipe_environment(row)
    plan = {'schema': PLAN_SCHEMA, 'purpose': 'functional_reference_only_not_scientific_native_performance',
        'created_at': datetime.now(timezone.utc).isoformat(), 'identity': ident, 'source_row': row,
        'source_plan': bound(source_plan), 'original_run_manifest': bound(Path(row['original_run']) / 'run_manifest.json'),
        'historical_native_source': bound(row['native_source']), 'original_quality_request': bound(row['generic_original_quality_request']),
        'runtime_environment_evidence': environment_source, 'expected_interpreter': environment,
        'controller_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
        'repository': str(repository), 'source_files': sources, 'archive': bound(archive),
        'output_root': str(output), 'remote_root': remote, 'remote_tool_dir': remote + '/tool',
        'execution_id': session, 'target': bound_targets(Path(row['original_run']), [row])[row['setup_id']],
        'expected_remote_files': expected, 'quality_binding_file': bound(quality),
        'output_contract_source': output_contract,
        'item': {'argv': argv, 'env': env, 'repeat': 1, 'source_benchmark_set': command['benchmark_set'],
                 'benchmark_case_inputs': _benchmark_case_inputs(command, {'backend': row['direction'], 'case': row['case_id']}),
                 'remote_output_root': remote_output, 'result_json': remote_output + '/' + ('deepx_native_fifo_e2e_results.json'
                     if row['direction'] == 'deepx_to_trt' else 'hailo10_native_fifo_e2e_results.json')},
        'recipe': {'processes': 1, 'completed_tasks': 1, 'warmup_tasks': 0, 'duration_s': 0., 'completion_runtime_mode': 'strict'},
        'build_allowed': False, 'fallback_allowed': False, 'automatic_retry': False, 'energy_capture': False,
        'hardware_executed': False}
    if row['native_measurement_boundary'].get('process_repetition_policy') == 'separate_processes':
        summary = bound(row['native_measurement_boundary']['source'])
        plan['historical_scientific_source'] = {'summary': summary,
            'reports': [bound(value['raw_report']) for value in read(summary['path'])['rows']]}
    validate_capture_plan(plan)
    write_new(output / 'PREPARED.json', plan)
    return bound(output / 'PREPARED.json')


def _remote_sources(transport, expected, python):
    program = ('import hashlib,json\nfrom pathlib import Path\nexpected=' + repr(expected)
        + '\nactual={}\nfor name in expected:\n p=Path(name)\n with p.open("rb") as h:actual[name]=hashlib.file_digest(h,"sha256").hexdigest()\n'
        + 'print("SOURCE_BINDING="+json.dumps(actual,sort_keys=True))\n')
    # Python3.10 lacks hashlib.file_digest; use the ordinary streaming hash.
    program = program.replace('with p.open("rb") as h:actual[name]=hashlib.file_digest(h,"sha256").hexdigest()',
        'h=hashlib.sha256()\n with p.open("rb") as f:\n  for block in iter(lambda:f.read(8388608),b""):h.update(block)\n actual[name]=h.hexdigest()')
    rc, text = transport.run_read_only(shlex.quote(python) + " -B - <<'FUNCTIONAL_REFERENCE_SOURCES'\n" + program
                                     + 'FUNCTIONAL_REFERENCE_SOURCES', timeout=300)
    lines = [line[len('SOURCE_BINDING='):] for line in text.splitlines() if line.startswith('SOURCE_BINDING=')]
    require(rc == 0 and len(lines) == 1, 'Native functional source verification failed')
    observed = json.loads(lines[0])
    require(observed == expected, 'Native functional artifact/input/source bytes changed')
    return observed


def _functional_command_projection(command, historical, plan):
    """Bind normal runner output to the planned finite invocation's semantics."""
    options, prior = command['runtime_options'], historical['runtime_options']
    for key in ('queue_depth', 'inflight', 'producer_impl', 'quantized_inputs', 'quantized_outputs',
        'copy_outputs', 'task', 'preprocess_mode_requested', 'preprocess_mode_effective',
        'letterbox_pad_value_requested', 'letterbox_pad_value_effective', 'letterbox_pad_value',
        'part1_onnx_used', 'canonical_input_slot_names', 'canonical_output_slot_names',
        'runtime_input_shapes', 'runtime_output_shapes'):
        if key in prior:
            require(options.get(key) == prior[key], 'Functional Native preprocessing/runtime policy differs: ' + key)
    require(command['boundary_contract'] == historical['boundary_contract'], 'Functional Native boundary transformation differs')
    if command['backend'] == 'deepx_to_trt':
        require(options.get('deepx_runtime_mode') == 'S1' and options.get('deepx_max_inflight') == 1,
                'Functional Native must use the original SingleInput S1 policy')
    for role, expected in plan['source_row']['expected_runtime_artifacts'].items():
        native_role = ('dxnn' if command['backend'] == 'deepx_to_trt' else 'hef') if role == 'part1_runtime' else role
        actual = command['artifacts'].get(native_role) or {}
        require(all(actual.get(key) == expected[key] for key in ('path', 'sha256')),
                'Functional Native artifact closure differs: ' + role)


def execute_capture(prepared_path):
    """One explicit leased Native command; never restart an attempted capture."""
    from ..remote.process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
    from .execution_binding import physical_dut_key
    from .run_control import EvaluationRunLock
    from .generic_completed_task import observe_owners, transport_for, upload_new_file, verify_runtime_environment
    from .hardware_acceptance_native import _stage_benchmark_set
    prepared_path = Path(prepared_path).resolve(strict=True)
    plan = read(prepared_path)
    row = plan['source_row']
    validate_capture_plan(plan)
    output = Path(plan['output_root'])
    require(prepared_path == output / 'PREPARED.json' and plan['schema'] == PLAN_SCHEMA, 'Wrong capture plan')
    require(not (output / 'production-start.json').exists(), 'Capture already attempted; preserve its markers')
    repository = Path(plan['repository'])
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip() == plan['controller_commit']
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True).strip(),
            'Publish the reviewed source and reprepare from a clean checkout before capture')
    require(all(sha(repository / name) == digest for name, digest in plan['source_files'].items()), 'Prepared Native source freeze changed')
    expected_names = {str(path.relative_to(repository)) for path in (repository / 'onnx_splitpoint_tool').rglob('*.py')
                      if '__pycache__' not in path.parts}
    expected_names.update(('scripts/native_hailo10_trt_e2e_from_benchmarkset.py', 'scripts/native_deepx_trt_e2e_from_benchmarkset.py'))
    require(set(plan['source_files']) == expected_names, 'Prepared source closure omits current product files')
    checked(plan['archive'])
    registry = RemoteProcessLeaseRegistry()
    scope = RemoteProcessLeaseScope(output.name, plan['execution_id'])
    cancel = threading.Event()
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: cancel.set())
    status = 'failed'
    try:
        with ExitStack() as stack:
            owner = {'run_id': output.name, 'session_id': plan['execution_id'], 'purpose': 'native_functional_reference'}
            stack.enter_context(EvaluationRunLock(out_root=output.parent, run_dir=output, owner=owner))
            lock = stack.enter_context(EvaluationRunLock.for_resource(physical_dut_key(plan['target']), owner=owner))
            def fence():
                if registry.active_count():
                    lock.commit_quarantine_fence({'reason': 'native_functional_reference_cleanup_unproven',
                        'journal': str(output / 'remote-process-leases' / plan['execution_id'])})
            stack.callback(fence)
            registry.configure_journal(scope=scope, journal_dir=output / 'remote-process-leases' / plan['execution_id'])
            write_new(output / 'production-start.json', {'prepared': bound(prepared_path), 'execution_id': plan['execution_id']})
            observe_owners({row['setup_id']: plan['target']}, output / 'preflight')
            transport = transport_for(plan['target'], output / 'diagnostics', scope=scope, registry=registry, cancel=cancel)
            verify_runtime_environment(transport, row, output / 'preflight/runtime-environment.json')
            remote = plan['remote_root']
            upload_new_file(transport, checked(plan['archive']), remote + '/native-product-bundle.tar.gz', expected_sha=plan['archive']['sha256'])
            rc, text = transport.run('mkdir -p ' + shlex.quote(remote + '/tool') + ' && tar -xzf '
                + shlex.quote(remote + '/native-product-bundle.tar.gz') + ' -C ' + shlex.quote(remote + '/tool'), timeout=120)
            require(rc == 0, 'Native private bundle staging failed: ' + text[-1000:])
            upload_new_file(transport, checked(plan['quality_binding_file']), remote + '/native_split_quality_binding.json',
                            expected_sha=plan['quality_binding_file']['sha256'])
            _stage_benchmark_set(transport, plan['item'], output)
            benchmark = plan['item']['argv'][plan['item']['argv'].index('--benchmark-set') + 1]
            upload_new_file(transport, checked(plan['output_contract_source']['source']), benchmark + '/output_contracts.json',
                            expected_sha=plan['output_contract_source']['source']['sha256'])
            before = _remote_sources(transport, plan['expected_remote_files'], row['remote_python_executable'])
            write_new(output / 'source-binding-before.json', before)
            command = 'cd -- ' + shlex.quote(plan['remote_tool_dir']) + ' && ' + shlex.join(plan['item']['argv'])
            rc, text = transport.run(command, timeout=900, env=plan['item']['env'])
            (output / 'dispatch.log').write_text(text)
            write_new(output / 'dispatch.json', {'argv': plan['item']['argv'], 'returncode': rc, 'env': plan['item']['env']})
            require(rc == 0, 'Native functional reference runner failed; no automatic retry')
            after = _remote_sources(transport, plan['expected_remote_files'], row['remote_python_executable'])
            write_new(output / 'source-binding-after.json', after)
            destination = output / 'download' / row['model_id']
            destination.mkdir(parents=True)
            benchmark = plan['item']['argv'][plan['item']['argv'].index('--benchmark-set') + 1]
            rc, _ = transport.scp_download(benchmark, str(destination), recursive=True, timeout=600)
            require(rc == 0, 'Native functional capture download failed')
            original_input = row['input_binding']['entries'][0]
            inputs = output / 'original_input'
            inputs.mkdir()
            rc, _ = transport.scp_download(original_input['remote_path'], str(inputs), recursive=False, timeout=120)
            require(rc == 0, 'Original Generic prepared feed download failed')
            local_input = inputs / PurePosixPath(original_input['remote_path']).name
            require(sha(local_input) == original_input['sha256'], 'Original Generic prepared feed changed')
            raw_path = destination / 'benchmark_set' / str(PurePosixPath(plan['item']['result_json']).relative_to(benchmark))
            require(registry.active_count() == 0 and registry.cancelled is False, 'Native functional transport cleanup unproven')
            write_new(output / 'cleanup-status.json', {'active_owned_remote_operations': 0,
                'registry_cancelled_or_poisoned': False, 'foreign_processes_stopped': False})
            record = {'schema': SCHEMA, 'purpose': plan['purpose'], 'plan': bound(prepared_path),
                'raw_report': bound(raw_path), 'original_prepared_input': bound(local_input),
                'source_binding_before': bound(output / 'source-binding-before.json'),
                'source_binding_after': bound(output / 'source-binding-after.json'),
                'dispatch': bound(output / 'dispatch.json'), 'cleanup': bound(output / 'cleanup-status.json')}
            candidate = output / 'REFERENCE.candidate.json'
            write_new(candidate, record)
            verify_native_functional_reference({**row, 'functional_reference_source': bound(candidate)}, _candidate=True)
            final = output / 'REFERENCE.json'
            write_new(final, record)
            status = 'complete'
            return bound(final)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if not (output / 'cleanup-status.json').exists():
            write_new(output / 'cleanup-status.json', {'active_owned_remote_operations': registry.active_count(),
                'registry_cancelled_or_poisoned': registry.cancelled, 'foreign_processes_stopped': False})
        write_new(output / 'terminal-status.json', {'status': status, 'automatic_retry': False})


def verify_native_functional_reference(case, *, _candidate=False):
    """Verify the explicit functional chain; never reassign historical Native FPS."""
    from ..native_detection_postprocess import verify_detection_completion_execution_contract, verify_detection_completion_execution_attestation
    from ..resources.remote_scripts.native_producer_final_report import _verify_split_semantic_artifacts
    from ..resources.remote_scripts.native_producer_validate_visualize import _verified_execution_completed_result_artifact
    import numpy as np
    descriptor = case.get('functional_reference_source')
    require(descriptor is not None, 'Functional reference descriptor is required')
    reference_path = checked(descriptor)
    require(reference_path.name == ('REFERENCE.candidate.json' if _candidate else 'REFERENCE.json'),
            'Only a completed published functional reference may enter paired admission')
    if not _candidate:
        require(read(reference_path.parent / 'terminal-status.json') == {'status': 'complete', 'automatic_retry': False},
                'Functional capture terminal status is not complete')
    record = read(reference_path)
    require(record.get('schema') == SCHEMA and record.get('purpose') == 'functional_reference_only_not_scientific_native_performance', 'Wrong functional reference role')
    plan_path = checked(record['plan'])
    plan = read(plan_path)
    require(plan['schema'] == PLAN_SCHEMA and Path(plan['output_root']) == reference_path.parent
            and plan_path == reference_path.parent / 'PREPARED.json', 'Functional capture origin differs')
    validate_capture_plan(plan)
    for key, name in (('source_binding_before', 'source-binding-before.json'),
        ('source_binding_after', 'source-binding-after.json'), ('dispatch', 'dispatch.json'), ('cleanup', 'cleanup-status.json')):
        require(checked(record[key]) == reference_path.parent / name, 'Functional proof detached from capture: ' + key)
    started = read(reference_path.parent / 'production-start.json')
    require(started == {'prepared': bound(plan_path), 'execution_id': plan['execution_id']}, 'Functional attempt origin differs')
    original, historical_raw, historical_command = _historical(plan, case)
    expected = plan['expected_remote_files']
    require(read(checked(record['source_binding_before'])) == read(checked(record['source_binding_after'])) == expected,
            'Functional capture source/artifact/input closure changed')
    dispatch = read(checked(record['dispatch']))
    require(dispatch['argv'] == plan['item']['argv'] and dispatch['returncode'] == 0 and dispatch['env'] == plan['item']['env'], 'Functional Native dispatch differs')
    cleanup = read(checked(record['cleanup']))
    require(cleanup == {'active_owned_remote_operations': 0, 'registry_cancelled_or_poisoned': False,
                        'foreign_processes_stopped': False}, 'Functional capture cleanup unproven')
    journal = reference_path.parent / 'remote-process-leases' / plan['execution_id']
    require(journal.is_dir() and not journal.is_symlink() and not list(journal.iterdir()), 'Actual functional capture lease journal not empty')
    raw_path = checked(record['raw_report'])
    benchmark_root = reference_path.parent / 'download' / original['model_id'] / 'benchmark_set'
    benchmark = plan['remote_root'] + '/' + original['model_id'] + '/benchmark_set'
    require(raw_path == _collected_path(plan['item']['result_json'], benchmark, benchmark_root),
            'Functional raw escapes its collected origin')
    raw = read(raw_path)
    command, status = verify_native_command_contract(raw.get('native_command_contract'), expected_identity={
        'model': original['model_id'], 'case': original['case_id'], 'setup_id': original['setup_id'],
        'backend': original['direction'], 'precision': original['precision']})
    require(command is not None and raw.get('ok') is True
            and raw.get('native_command_contract_sha256') == command['contract_sha256']
            and raw.get('source_request_sha256') == command['source_request_sha256'], 'Functional Native command invalid: ' + status)
    require(command['source_contract_sha256'] == historical_command['contract_sha256']
            and command['eval_run_id'] == historical_command['eval_run_id']
            and command['source_request_sha256'] == historical_command['source_request_sha256']
            and command['native_split_quality_binding']['binding_sha256'] == original['native_quality_binding_sha256'],
            'Functional Native original-run/quality/artifact ancestry differs')
    runner = 'scripts/native_' + ('deepx' if original['direction'] == 'deepx_to_trt' else 'hailo10') + '_trt_e2e_from_benchmarkset.py'
    require(command['runner_sha256'] == plan['source_files'][runner]
            and command['artifacts']['python_executable']['sha256'] == plan['expected_interpreter']['executable_sha256']
            and command['python_executable'] == historical_command['python_executable'], 'Functional interpreter/runner freeze differs')
    options = command['runtime_options']
    require(options.get('frames') == 1 and options.get('warmup') == 0 and options.get('repetitions') == 1
            and options.get('duration_s') == 0 and options.get('build') is False
            and options.get('completion_runtime_mode') == 'strict', 'Functional Native finite recipe differs')
    _functional_command_projection(command, historical_command, plan)
    require(raw.get('completed_work_units') == raw.get('completed_frames') == raw.get('postprocess_completed_frames') == 1
            and raw.get('postprocess_completion_verified') is True, 'Functional Native did not complete exactly one full task')
    require(command['input_image_sha256'] == original['input_binding']['source_image_sha256'], 'Functional input image changed')
    for native_role, generic_role in (('dxnn' if original['direction'] == 'deepx_to_trt' else 'hef', 'part1_runtime'), ('engine', 'engine')):
        actual, existing = command['artifacts'][native_role], original['expected_runtime_artifacts'][generic_role]
        require(all(actual[key] == existing[key] for key in ('path', 'sha256')), 'Compiled SingleInput artifact changed: ' + native_role)
    require(command['benchmark_set'] == benchmark, 'Functional command benchmark origin differs')
    manifest_path = _collected_path(command['artifacts']['semantic_output_manifest']['path'], benchmark, benchmark_root)
    require(sha(manifest_path) == command['artifacts']['semantic_output_manifest']['sha256'], 'Functional output manifest missing or changed')
    manifest = read(manifest_path)
    require(manifest.get('input_image_sha256') == command['input_image_sha256'], 'Output manifest input differs')
    proof = _verify_split_semantic_artifacts(result_path=raw_path, manifest_path=manifest_path,
        manifest_sha256=sha(manifest_path), sources=(raw,))
    require(proof.get('native_split_semantic_binding_valid') is True and proof.get('native_split_final_portable_binding_valid') is True,
            'Functional output/boundary/quality consumer proof failed: ' + str(proof))
    contract = verify_detection_completion_execution_contract(raw.get('completion_execution_contract'))
    require(contract == options.get('completion_execution_contract')
            and contract['source_endpoint']['output_endpoint_id'] == original['identity']['output_endpoint_id'], 'Functional full completion/source endpoint differs')
    attestation = verify_detection_completion_execution_attestation(raw.get('completed_task_endpoint_attestation'),
        execution_contract=contract, expected_observation_relation='same_hotloop_sentinel')
    require(attestation['completed_work_units'] == 1, 'Functional completion attestation task count differs')
    _verified_execution_completed_result_artifact(attestation['last_result'], attestation, raw, native_report=raw_path)
    prepared_contract = command['prepared_input_contract']
    entries = prepared_contract.get('entries') or [{'artifact_name': 'prepared_input', 'shape': prepared_contract.get('shape'), 'dtype': prepared_contract.get('dtype')}]
    require(len(entries) == 1 and entries[0]['shape'] == [640, 640, 3] and entries[0]['dtype'] == 'uint8', 'Functional Native persisted feed must retain original HWC')
    spec = command['artifacts'][entries[0]['artifact_name']]
    prepared_path = _collected_path(spec['path'], benchmark, benchmark_root)
    baseline = checked(record['original_prepared_input'])
    require(baseline == reference_path.parent / 'original_input' / PurePosixPath(original['input_binding']['entries'][0]['remote_path']).name,
            'Original Generic prepared input detached from capture')
    require(sha(prepared_path) == spec['sha256'] == sha(baseline) == original['input_binding']['entries'][0]['sha256'], 'Functional Native prepared input bytes differ from Generic')
    arrays = [np.load(path, allow_pickle=False) for path in (prepared_path, baseline)]
    require(all(list(array.shape) == [640, 640, 3] and str(array.dtype) == 'uint8' and array.flags.c_contiguous for array in arrays)
            and np.array_equal(*arrays), 'Functional Native prepared input typed payload differs')
    outputs = []
    for item in manifest['outputs']:
        path = manifest_path.parent / item['file']
        require(path.resolve().parent == manifest_path.parent.resolve() and sha(path) == item['sha256'], 'Functional output payload missing/escaped/changed')
        outputs.append({**item, 'path': str(path), 'local_path': str(path), 'format': 'raw'})
    runtime = copy.deepcopy(case)
    runtime.pop('runtime_config', None)
    reference = {**runtime['functional_reference'], 'source_report': str(raw_path), 'output_manifest': str(manifest_path),
        'output_manifest_sha256': sha(manifest_path), 'outputs': outputs,
        'role': 'fresh_functional_reference_only_not_scientific_native_performance'}
    runtime.update(native_source=str(raw_path), native_command_contract=command,
        native_command_contract_sha256=command['contract_sha256'], completion_contract=contract,
        functional_reference=reference)
    return {'status': 'verified', 'runtime_case': runtime, 'completion_contract': contract,
        'functional_reference': reference, 'version_relation': {
            'same_full_contract_as_historical_generic': contract == original['completion_contract'],
            'historical_contract_sha256': original['completion_contract']['contract_sha256'],
            'functional_contract_sha256': contract['contract_sha256'],
            'same_semantic_comparison_endpoint': contract['comparison_endpoint_contract']['output_endpoint_id']
                == original['native_measurement_boundary']['completion_output_endpoint_id'],
            'scientific_native_performance_unchanged': True},
        'evidence': {'scope': 'fresh_single_input_native_functional_reference', 'capture': descriptor,
            'historical_source': plan['historical_native_source'], 'historical_command_sha256': historical_command['contract_sha256'],
            'original_run_manifest': plan['original_run_manifest'], 'original_quality_request': plan['original_quality_request'],
            'historical_native_counts_times': original['native_measurement_boundary'],
            'historical_scientific_source': plan.get('historical_scientific_source'),
            'functional_raw_report': record['raw_report'], 'functional_native_command_sha256': command['contract_sha256'],
            'functional_output_manifest': bound(manifest_path), 'native_split_semantic_proof': proof,
            'prepared_input_hwc_bytes_exact': True, 'captured_completed_tasks': 1,
            'new_scientific_native_fps': None, 'historical_source_inside_original_run': Path(original['native_source']).resolve().is_relative_to(Path(original['original_run']).resolve())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', type=Path, help='Existing paired source execution plan')
    actions.add_argument('--execute', type=Path, help='Explicitly execute a reviewed PREPARED.json')
    parser.add_argument('--case-key')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--runtime-environment-evidence', type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        require(args.case_key and args.output and args.runtime_environment_evidence, 'Preparation requires case-key/output/runtime evidence')
        result = prepare_capture(args.prepare, args.case_key, args.output,
                                 runtime_environment_evidence=args.runtime_environment_evidence)
    else:
        result = execute_capture(args.execute)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
