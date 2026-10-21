"""Explicit bounded continuation policy for the normal Generic dispatcher.

This helper dispatches nothing; run_case retains normal parsing, transport,
process ownership and durable reservation before each remote invocation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shlex

from onnx_splitpoint_tool.remote.process_lease import (
    RemoteProcessLeaseJournal, RemoteProcessLeaseRegistry,
)


CONTROL = {'mode': 'independent_cells_v1', 'attempts_per_repetition_max': 2}
PROGRESS_FIELDS = frozenset({
    'runtime_environment_comparison', 'dispatch_attempted', 'execution_status',
    'execution_error', 'completion_observed_at', 'dispatch_wall_seconds',
})


class InvalidResumeBinding(ValueError):
    failure_scope = 'branch'


class UnresolvedOwnedProcesses(RuntimeError):
    failure_scope = 'backend'


class CellAttemptLimitReached(RuntimeError):
    failure_scope = 'cell'


class CellCompletionRejected(RuntimeError):
    failure_scope = 'cell'


def require_bound_control(current, initial):
    if current != CONTROL or initial != CONTROL:
        raise InvalidResumeBinding('Explicit immutable bounded cell policy differs')
    return CONTROL['attempts_per_repetition_max']


def plan_binding(plan):
    value = deepcopy(plan)
    value['rows'] = [{key: item for key, item in row.items() if key not in PROGRESS_FIELDS}
                     for row in value['rows']]
    return value


def admit_controller_attempt(plan, plan_path, output, bundle, *, resume, started_at):
    """Preserve the first start/snapshot; explicit reentry adds a new marker."""
    output, plan_path, bundle = Path(output), Path(plan_path).resolve(), Path(bundle).resolve()
    require_bound_control(plan.get('cell_execution_control'), plan.get('cell_execution_control'))
    if plan.get('scope_mode') != 'generic_only_completed_task':
        raise InvalidResumeBinding('Independent cells require explicit Generic-only mode')
    start = output / 'production-start.json'
    initial_path = output / 'plan' / 'initial-execution-plan.json'
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if resume:
        if not start.is_file() or not initial_path.is_file():
            raise InvalidResumeBinding('Resume requires the preserved original start and plan snapshot')
        original = json.loads(start.read_text())
        initial = json.loads(initial_path.read_text())
        if (original.get('execution_id') != plan['execution_id']
                or original.get('plan') != str(plan_path)
                or original.get('runner_bundle_sha256') != digest(bundle)
                or original.get('initial_plan_sha256') != digest(initial_path)
                or plan_binding(plan) != plan_binding(initial)):
            raise InvalidResumeBinding('Original execution/config/source/bundle/input binding changed')
    else:
        if start.exists() or initial_path.exists():
            raise InvalidResumeBinding('Previous start retained; continuation requires explicit --resume')
        for row in plan['rows']:
            result = Path(row['result_path'])
            if result.exists() or (result.parent / 'dispatch_attempts.json').exists() or any(result.parent.glob('dispatch_*.log')):
                raise InvalidResumeBinding('New independent execution cannot adopt existing measurements or dispatches')
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        with initial_path.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, indent=2)
        initial = deepcopy(plan)
        with start.open('x', encoding='utf-8') as stream:
            json.dump({'execution_id': plan['execution_id'], 'started_at': started_at,
                'plan': str(plan_path), 'runner_bundle': str(bundle),
                'runner_bundle_sha256': digest(bundle), 'initial_plan_sha256': digest(initial_path)}, stream, indent=2)
    attempts = output / 'controller-attempts'
    attempts.mkdir(exist_ok=True)
    existing = sorted(path for path in attempts.glob('attempt_*.json')
                      if re.fullmatch(r'attempt_\d{4}\.json', path.name))
    if [path.name for path in existing] != [f'attempt_{index:04d}.json' for index in range(1, len(existing) + 1)]:
        raise InvalidResumeBinding('Controller attempt history has missing markers')
    for path in existing:
        previous = json.loads(path.read_text())
        if (previous.get('execution_id') != plan['execution_id']
                or previous.get('original_start_sha256') != digest(start)
                or previous.get('initial_plan_sha256') != digest(initial_path)):
            raise InvalidResumeBinding('Controller attempt is foreign to the original start')
    marker = attempts / f'attempt_{len(existing) + 1:04d}.json'
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'execution_id': plan['execution_id'], 'started_at': started_at, 'explicit_resume': resume,
            'original_start_sha256': digest(start), 'initial_plan_sha256': digest(initial_path)}, stream, indent=2)
    return initial, marker


def require_same_case(current, initial):
    """The original prepared row is read from an immutable start-bound snapshot."""
    left = {key: value for key, value in current.items() if key not in PROGRESS_FIELDS}
    right = {key: value for key, value in initial.items() if key not in PROGRESS_FIELDS}
    if left != right:
        raise InvalidResumeBinding('Same execution ID/config/source/artifact/input/control binding required')
    configuration = current.get('runtime_environment_comparison') or {}
    if configuration.get('status') != 'observed_no_known_configuration_conflict':
        raise InvalidResumeBinding('Current normal runtime configuration conflict or observation missing')


def expected_command(row, repetition):
    config = row['runtime_config']
    path = config['config_directory'] + '/generic_completion_config.json'
    argv = [row['remote_python_executable'], '-B', '-u',
        row['remote_root'] + '/runner/run_split_onnxruntime.py',
        '--generic-completion-config', path, '--generic-completion-repetition', str(repetition),
        '--out-dir', row['remote_root'] + '/results/' + row['case_output_relative']]
    return 'cd -- ' + shlex.quote(row['remote_root']) + ' && ' + shlex.join(argv)


def read_dispatch_attempts(result_path):
    """A controller restart rereads the existing normal budget; never resets it."""
    root = Path(result_path).parent
    path = root / 'dispatch_attempts.json'
    if not path.exists():
        if Path(result_path).exists() or any(root.glob('dispatch_*.log')):
            raise InvalidResumeBinding('Existing result/logs without their persistent dispatch budget')
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get('attempts'), list):
        raise InvalidResumeBinding('Normal persistent dispatch attempts malformed')
    for logfile in root.glob('dispatch_*.log'):
        match = re.fullmatch(r'dispatch_(\d{3})\.log', logfile.name)
        if match is None or not 1 <= int(match.group(1)) <= len(payload['attempts']):
            raise InvalidResumeBinding('Dispatch log has no matching persistent budget reservation')
    return deepcopy(payload['attempts'])


def validate_dispatch_history(row, valid_repetitions, attempts):
    counts = {index: 0 for index in (1, 2, 3)}
    succeeded = set()
    for attempt in attempts:
        index = attempt.get('repetition_index')
        if (type(index) is not int or index not in counts
                or attempt.get('command') != expected_command(row, index)
                or not isinstance(attempt.get('started_at'), str) or not attempt['started_at']
                or attempt.get('status') not in {'running', 'exited', 'dispatch_error'}):
            raise InvalidResumeBinding('Persistent normal attempt belongs to another dispatch')
        counts[index] += 1
        if attempt.get('status') == 'exited' and type(attempt.get('returncode')) is int and attempt['returncode'] == 0:
            succeeded.add(index)
    if any(count > CONTROL['attempts_per_repetition_max'] for count in counts.values()):
        raise InvalidResumeBinding('Persistent dispatch budget exceeds the original fixed limit')
    valid = set()
    for item in valid_repetitions:
        index = item.get('repetition_index')
        if (type(index) is not int or index not in counts or index in valid
                or item.get('execution_id') != row['runtime_config']['execution_id'] or item.get('status') != 'ok'):
            raise InvalidResumeBinding('Normal validated repetition projection foreign or duplicated')
        if counts[index] == 0:
            raise InvalidResumeBinding('Parsed repetition has no original persistent dispatch authority')
        if index not in succeeded:
            raise CellCompletionRejected('Parsed valid repetition retained, but its normal dispatch has no verified exit zero; no repeat of valid work')
        valid.add(index)
    return counts, valid


def require_no_recorded_integrity_failure(logfile):
    """Known normal guard failures are branch stops, never local retries."""
    text = Path(logfile).read_text(encoding='utf-8')
    guards = (
        'Generic runtime source differs from frozen bundle',
        'Generic runtime source changed during execution',
        'Generic completion runtime artifact mismatch',
        'Generic completion functional reference input mismatch',
        'Generic-only generated model metadata differs from its producer',
        'generic split repetition runtime source binding mismatch',
    )
    matched = next((guard for guard in guards if guard in text), None)
    if matched is not None:
        raise InvalidResumeBinding('Normal runtime integrity guard rejected the bound branch: ' + matched)


def require_clean_ownership(registry, journal, expected_scope):
    """Use actual normal ownership objects, not success flags or an rc alone."""
    if (not isinstance(registry, RemoteProcessLeaseRegistry)
            or not isinstance(journal, RemoteProcessLeaseJournal)
            or registry.configured_scope != expected_scope or journal.scope != expected_scope):
        raise UnresolvedOwnedProcesses('Matching normal process ownership/journal unavailable')
    try:
        if (registry.cancelled or registry.active_count() != 0 or journal.is_cancelled()
                or journal.pending_count() != 0 or not journal.directory.is_dir()
                or journal.directory.is_symlink() or any(journal.directory.iterdir())):
            raise UnresolvedOwnedProcesses('Owned process cleanup or matching empty journal unproven')
    except UnresolvedOwnedProcesses:
        raise
    except Exception as error:
        raise UnresolvedOwnedProcesses('Normal ownership journal cannot be verified') from error


def admit_missing_repetition(row, initial_row, repetition, normal_valid_repetitions, attempts, *,
                             control, initial_control, registry, journal, expected_scope):
    """Intended normal run_case before-dispatch hook; it does not append or run."""
    maximum = require_bound_control(control, initial_control)
    require_same_case(row, initial_row)
    if type(repetition) is not int or repetition not in (1, 2, 3):
        raise InvalidResumeBinding('Only the original three repetition indices may be dispatched')
    counts, valid = validate_dispatch_history(row, normal_valid_repetitions, attempts)
    if repetition in valid:
        raise InvalidResumeBinding('A normally valid repetition must never be repeated')
    if counts[repetition] >= maximum:
        raise CellAttemptLimitReached('Original repetition exhausted its fixed two-attempt budget')
    require_clean_ownership(registry, journal, expected_scope)
    return {'repetition_index': repetition, 'prior_dispatch_attempts': counts[repetition],
        'attempt_number': counts[repetition] + 1, 'attempt_limit': maximum,
        'preserved_valid_repetition_indices': sorted(valid),
        'execution_id': row['runtime_config']['execution_id']}
