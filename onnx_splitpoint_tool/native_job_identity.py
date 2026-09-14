"""Planned Native job identity, independent of runtime/quality attestations.

This is a small projection of the existing job fields, not an identity registry.
A failed observation is present; it is never a successful measurement.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

FIELDS = ('backend', 'model', 'case', 'setup_id', 'comparison_backend', 'precision')

def _text(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_')

def native_backend(value: Any) -> str:
    text = _text(value)
    return {
        'hailo8': 'hailo8_to_trt', 'hailo_to_trt': 'hailo8_to_trt',
        'hailo8_to_tensorrt': 'hailo8_to_trt',
        'hailo10': 'hailo10h_to_trt', 'hailo10h': 'hailo10h_to_trt',
        'hailo10_to_trt': 'hailo10h_to_trt', 'hailo10_to_tensorrt': 'hailo10h_to_trt', 'hailo10h_to_tensorrt': 'hailo10h_to_trt',
        'deepx': 'deepx_to_trt', 'deepx_m1': 'deepx_to_trt',
        'deepx_to_tensorrt': 'deepx_to_trt',
        'native_full_hailo10': 'native_full_hailo10h',
    }.get(text, text)

def native_comparison(value: Any) -> str:
    text = native_backend(value)
    return {'hailo8_to_trt': 'hailo8', 'hailo10h_to_trt': 'hailo10h',
            'deepx_to_trt': 'deepx', 'native_full_hailo8': 'hailo8',
            'native_full_hailo10h': 'hailo10h', 'native_full_deepx': 'deepx'}.get(text, text)

def planned_native_identity(job: Mapping[str, Any], resolved_selection: Mapping[str, Any] | None = None) -> dict[str, str]:
    selection = resolved_selection or {}
    backend = native_backend(job.get('backend') or job.get('producer_backend'))
    comparison = native_comparison(job.get('comparison_backend'))
    if not comparison and backend != 'native_full_tensorrt':
        comparison = native_comparison(backend)
    case = _text(job.get('case') or job.get('case_id') or job.get('boundary'))
    if case.startswith('b') and case[1:].isdigit():
        case = 'b' + case[1:].zfill(3)
    return dict(zip(FIELDS, (
        backend, str(job.get('model') or job.get('model_id') or '').strip(), case,
        str(job.get('setup_id') or job.get('measurement_setup_id') or job.get('setup') or '').strip(),
        comparison,
        _text(selection.get('precision') or job.get('precision') or job.get('contract') or job.get('variant')),
    )))

def native_identity_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = planned_native_identity(row)
    # Full ledger identity is comparison-based. Its runtime precision is retained
    # in the result and must not be forced to the Split comparison precision.
    if value['backend'].startswith('native_full_'):
        value['precision'] = ''
    return tuple(value[field] for field in FIELDS)

def failed_native_result(identity: Mapping[str, Any], *, failure_stage: str,
                         failure_reason: str, repetition_count_attempted: int = 0,
                         **details: Any) -> dict[str, Any]:
    result = dict(details)
    result.update(planned_native_identity(identity))
    result.update(ok=False, result_ok=False, status=str(details.get('status') or ('blocked' if repetition_count_attempted == 0 else 'failed')),
                  failure_stage=failure_stage, failure_reason=failure_reason,
                  primary_failure_reason=failure_reason, status_detail=failure_reason,
                  error=failure_reason, prerequisite_status='blocked',
                  repetition_count_attempted=int(repetition_count_attempted),
                  repetition_count_valid=0, planned_native_identity=planned_native_identity(identity))
    return result


def known_build_exclusion(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project an existing exact negative decision; never infer it from logs.

    The build service owns cache admission. This validates its original record
    and case/family binding before distinguishing an excluded nonstart from a
    failed measurement. The original key and record hashes are reused.
    """
    observation = row.get('upstream_build_observation')
    if not isinstance(observation, Mapping):
        observation = row.get('build_exclusion')
    if not isinstance(observation, Mapping):
        return {}
    if (row.get('identity_conflicts') or row.get('actual_ok') is True
            or row.get('ok') is True or row.get('result_ok') is True
            or any(row.get(field) is True for field in (
                'runtime_success', 'runtime_ok', 'runtime_started', 'execution_started',
                'measurement_started', 'started', 'compiler_dispatched', 'timed_out', 'cancelled',
            ))
            or row.get('failure_stage') not in (None, '', 'build_backend_artifacts')
            or row.get('returncode') not in (None, 0)
            or row.get('compiler_dispatch_count') not in (None, 0)
            or any(row.get(field) not in (None, '', 0) for field in (
                'completed_frames', 'frames_completed', 'repetitions_completed',
            ))
            or any(field in row and type(row[field]) is not int for field in (
                'repetition_count_attempted', 'repetition_count_valid',
            ))
            or row.get('repetition_count_attempted', 0) != 0
            or row.get('repetition_count_valid', 0) != 0
            or observation.get('compiler_dispatched') is not False
            or observation.get('compiler_dispatch_count') not in (None, 0)
            or observation.get('status') != 'known_infeasible'
            or observation.get('readiness') != 'not_executable'
            or observation.get('identity_conflicts')):
        return {}
    decision = observation.get('build_evidence')
    if not isinstance(decision, Mapping):
        return {}
    if (decision.get('negative_evidence_hit') is not True
            or decision.get('compiler_dispatch_allowed') is not False
            or decision.get('reason') != 'exact_deterministic_outcome'
            or decision.get('state') not in {'COMPILE_INFEASIBLE', 'PARSER_UNSUPPORTED'}):
        return {}
    from .build_evidence import (
        validate_build_evidence_record, build_key_sha256, BuildEvidenceError,
        boundary_endpoint_contract_sha256, canonical_build_key_from_hailo_v3_payload,
    )
    try:
        record = validate_build_evidence_record(decision.get('record') or {})
        if (record['state'] != decision['state']
                or record['key'] != decision.get('key')
                or record['key_sha256'] != decision.get('key_sha256')
                or build_key_sha256(record['key']) != decision['key_sha256']):
            return {}
    except (BuildEvidenceError, ValueError, TypeError, KeyError):
        return {}
    identity = planned_native_identity(row)
    context = decision.get('context')
    context = context if isinstance(context, Mapping) else {}
    context_case = str(context.get('boundary') or '')
    if context_case.isdigit():
        context_case = 'b' + context_case.zfill(3)
    try:
        cache_payload = decision.get('cache_payload_v3') or {}
        manifest = context.get('split_manifest') or {}
        boundary_hash = boundary_endpoint_contract_sha256(
            stage=str(observation.get('stage') or ''), cache_payload=cache_payload,
            split_manifest=manifest,
        )
        projected_key = canonical_build_key_from_hailo_v3_payload(
            cache_payload, builder_source_onnx_sha256=record['key']['builder_source_onnx_sha256'],
            full_source_onnx_sha256=context.get('full_source_onnx_sha256'),
            boundary_endpoint_contract_sha256=boundary_hash,
            expected_cache_key=decision.get('cache_key_v3'),
        )
        if projected_key != record['key']:
            return {}
        manifest_case = ('full' if observation.get('stage') == 'full'
                         else 'b' + str(manifest.get('boundary', manifest.get('boundary_index'))).zfill(3))
        if manifest_case != context_case:
            return {}
    except (BuildEvidenceError, ValueError, TypeError, KeyError):
        return {}
    if (not all(identity[field] for field in FIELDS[:5])
            or str(observation.get('model_id') or '') != identity['model']
            or str(observation.get('boundary') or '') != identity['case']
            or native_comparison(observation.get('backend')) != identity['comparison_backend']
            or native_comparison(record['key'].get('hw_arch')) != identity['comparison_backend']
            or observation.get('stage') != ('full' if identity['backend'].startswith('native_full_') else 'part1')
            or context.get('model_id') != identity['model']
            or context_case != identity['case']
            or context.get('stage') != observation.get('stage')
            or context.get('full_source_onnx_sha256') != record['key'].get('full_source_onnx_sha256')
            or observation.get('net_name') != (record['key'].get('recipe') or {}).get('net_name')):
        return {}
    return {
        **{key: observation.get(key) for key in (
            'model_id', 'boundary', 'backend', 'stage', 'status', 'readiness',
            'compiler_dispatched', 'compiler_dispatch_count', 'net_name',
        )},
        'build_evidence': {key: decision[key] for key in (
            'negative_evidence_hit', 'compiler_dispatch_allowed', 'reason',
            'state', 'key', 'key_sha256', 'record', 'cache_payload_v3',
        )} | {'context': {key: context.get(key) for key in (
            'model_id', 'boundary', 'stage', 'full_source_onnx_sha256', 'split_manifest',
        )}},
        'exclusion_reason': str(record['state']).lower(),
        'source': 'existing_exact_build_evidence',
    }


def apply_known_build_disposition(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    exclusion = known_build_exclusion(row)
    if exclusion:
        result.update(
            status='excluded_known_build', prerequisite_status='blocked',
            build_exclusion=exclusion, disposition='excluded_known_build',
            measurement_started=False, runtime_executable=False,
            quality_applicability='not_applicable_build_excluded',
            task_quality_status='not_evaluated', performance_claim_eligible=False,
            execution_terminal=True, compiler_dispatched=False,
        )
    return result


def project_known_build_exclusions(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Keep full coverage honest while making exact exclusions terminal.

    A matching identity alone cannot discard a contradictory observation. Each
    presentation of an exclusion must carry the same valid original decision.
    Exclusion-aware completion requires a distinct, complete row partition;
    archived numeric-only matrices without exclusions retain their old contract.
    """
    result = dict(matrix)
    fields = ('present_expected_rows', 'successful_expected_rows',
              'failed_expected_rows', 'excluded_expected_rows')
    collections = {field: list(matrix.get(field) or []) for field in fields}
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    malformed = False
    for collection in collections.values():
        for row in collection:
            if not isinstance(row, Mapping):
                malformed = True
                continue
            groups.setdefault(native_identity_key(row), []).append(row)
    excluded = {}
    conflicts = set()
    for key, observations in groups.items():
        proofs = [known_build_exclusion(row) for row in observations]
        if not any(proofs):
            continue
        signatures = {
            (proof['build_evidence']['key_sha256'],
             proof['build_evidence']['record']['record_sha256'])
            for proof in proofs if proof
        }
        if not all(proofs) or len(signatures) != 1:
            conflicts.add(key)
            continue
        excluded[key] = apply_known_build_disposition(observations[0])

    exclusion_scope = bool(excluded or conflicts or collections['excluded_expected_rows']
                           or matrix.get('excluded_expected_row_count'))
    if not exclusion_scope:
        # Presence of array fields activates strict identity accounting in
        # historical energy readers. Do not invent those fields for an archive
        # whose complete contract consists only of counts.
        expected = int(matrix.get('expected_row_count') or 0)
        success = int(matrix.get('successful_expected_row_count') or 0)
        terminal = bool(expected and int(matrix.get('present_expected_row_count') or 0) == expected
                        and success == expected and not matrix.get('failed_expected_row_count')
                        and not matrix.get('missing_expected_row_count')
                        and not matrix.get('identity_unresolved_rows'))
        result.update(excluded_expected_row_count=0, executable_expected_row_count=expected,
                      execution_terminal_complete=terminal, technical_execution_complete=terminal,
                      exclusion_accounting_valid=True, exclusion_projection_errors=[])
        return result

    def projected(row):
        if not isinstance(row, Mapping):
            return row
        key = native_identity_key(row)
        if key in conflicts:
            copy = dict(row)
            copy['identity_conflicts'] = sorted(set(row.get('identity_conflicts') or []) | {
                'build_exclusion_observation_conflict'})
            return copy
        return excluded.get(key, dict(row))

    failed = [projected(row) for row in collections['failed_expected_rows']
              if not isinstance(row, Mapping) or native_identity_key(row) not in excluded]
    failed_count = (len(failed) if isinstance(matrix.get('failed_expected_rows'), list)
                    else max(0, int(matrix.get('failed_expected_row_count') or 0) - len(excluded)))
    present = [projected(row) for row in collections['present_expected_rows']]
    result.update(
        failed_expected_rows=failed, failed_expected_row_count=failed_count,
        excluded_expected_rows=list(excluded.values()),
        excluded_expected_row_count=len(excluded), present_expected_rows=present,
    )
    expected = int(matrix.get('expected_row_count') or 0)
    success = int(matrix.get('successful_expected_row_count') or 0)
    exclusion_scope = bool(excluded or conflicts or collections['excluded_expected_rows']
                           or matrix.get('excluded_expected_row_count'))
    errors = []
    if exclusion_scope:
        if malformed:
            errors.append('non_mapping_native_matrix_row')
        if conflicts:
            errors.append('contradictory_build_exclusion_observations')
        partitions = {
            'present': present,
            'successful': collections['successful_expected_rows'],
            'failed': failed,
            'excluded': list(excluded.values()),
        }
        keys = {}
        for label, rows in partitions.items():
            values = [native_identity_key(row) for row in rows if isinstance(row, Mapping)]
            keys[label] = set(values)
            if len(keys[label]) != len(rows):
                errors.append('duplicate_or_invalid_' + label + '_identity')
            if any(not all(key[:5]) for key in values):
                errors.append('incomplete_' + label + '_identity')
        if int(matrix.get('present_expected_row_count') or 0) != len(present):
            errors.append('present_row_count_mismatch')
        if success != len(collections['successful_expected_rows']):
            errors.append('successful_row_count_mismatch')
        if (isinstance(matrix.get('failed_expected_rows'), list)
                and int(matrix.get('failed_expected_row_count') or 0) != len(collections['failed_expected_rows'])):
            errors.append('failed_row_count_mismatch')
        if (keys['successful'] & keys['failed'] or keys['successful'] & keys['excluded']
                or keys['failed'] & keys['excluded']):
            errors.append('overlapping_native_row_dispositions')
        if keys['present'] != keys['successful'] | keys['failed'] | keys['excluded']:
            errors.append('native_row_partition_mismatch')
        if len(keys['present']) != expected:
            errors.append('expected_native_identity_count_mismatch')
        if matrix.get('missing_expected_rows'):
            errors.append('missing_native_rows')
    accounting_valid = not errors
    terminal = bool(expected and int(matrix.get('present_expected_row_count') or 0) == expected
                    and success + len(excluded) == expected and not failed_count
                    and not matrix.get('missing_expected_row_count')
                    and not matrix.get('identity_unresolved_rows') and accounting_valid)
    result.update(executable_expected_row_count=max(0, expected - len(excluded)),
                  execution_terminal_complete=terminal, technical_execution_complete=terminal,
                  exclusion_accounting_valid=accounting_valid,
                  exclusion_projection_errors=sorted(set(errors)))
    # execution_success_complete / matrix_complete keep their original meaning:
    # every requested row was actually measured successfully.
    return result


_RUNTIME_START_FIELDS = (
    'runtime_started', 'execution_started', 'measurement_started', 'started',
    'runtime_completed', 'execution_completed', 'measurement_completed',
    'completion_verified', 'completed_task_endpoint_attested',
)
_RUNTIME_SUCCESS_FIELDS = ('actual_ok', 'ok', 'result_ok', 'runtime_ok', 'runtime_success')
_RUNTIME_COUNT_FIELDS = (
    'repetition_count_attempted', 'repetition_count_valid', 'completed_work_units',
    'completed_frames', 'frames_completed', 'repetitions_completed', 'work_units',
)
_RUNTIME_NONSTART = {'skipped', 'not_started', 'not_executable', 'blocked', 'not_run'}
_RUNTIME_STARTED = {'ok', 'success', 'passed', 'measured', 'running', 'started', 'completed'}


def generic_runtime_observation_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve compact execution evidence through the Generic result reader.

    Missing/null counters stay missing/null. Only scalar timing summaries and
    explicit per-variant execution records are retained, never raw arrays.
    This is a projection of existing producer fields, not a new identity key.
    """
    import copy
    fields = set(_RUNTIME_START_FIELDS + _RUNTIME_COUNT_FIELDS)
    fields.update(('identity_conflicts', 'runtime_executable', 'compiler_dispatched',
                   'compiler_dispatch_count', 'runtime_success', 'timed_out', 'cancelled'))
    for role in ('composed', 'part1', 'part2', 'full'):
        fields.update(role + '_' + field for field in
                      _RUNTIME_START_FIELDS + _RUNTIME_SUCCESS_FIELDS + _RUNTIME_COUNT_FIELDS)
    result = {field: copy.deepcopy(row[field]) for field in fields if field in row}
    nested_fields = set(_RUNTIME_START_FIELDS + _RUNTIME_SUCCESS_FIELDS + _RUNTIME_COUNT_FIELDS)
    nested_fields.update(('variant', 'primary_variant', 'status', 'runtime_executable',
                          'identity_conflicts', 'mean', 'mean_ms', 'count', 'n'))
    for container in ('timings', 'variant_results', 'runtime_results_by_variant'):
        value = row.get(container)
        if isinstance(value, Mapping):
            result[container] = {
                role: {key: copy.deepcopy(item[key]) for key in nested_fields if key in item}
                if isinstance(item, Mapping) else copy.deepcopy(item)
                for role, item in value.items()
                if role in {'composed', 'part1', 'part2', 'full'}
            }
        elif container != 'timings' and isinstance(value, list):
            result[container] = [
                {key: copy.deepcopy(item[key]) for key in nested_fields if key in item}
                if isinstance(item, Mapping) else copy.deepcopy(item) for item in value
            ]
    return result


def _positive_runtime_count(value: Any) -> bool:
    # A null, numeric string, bool or negative number is not a positive start.
    return type(value) is int and value > 0


def _generic_composed_observation_conflict(row: Mapping[str, Any]) -> str:
    """Explain why this observation cannot coexist with a composed nonstart.

    Only an independently delimited component measurement can explain away a
    row-wide success or unknown attempt count. Contradictory composed evidence
    always wins over the part2_only label.
    """
    if row.get('identity_conflicts'):
        return 'generic_build_exclusion_identity_conflict'
    status_map = row.get('variant_status')
    status_map = status_map if isinstance(status_map, Mapping) else {}
    composed_status = _text(status_map.get('composed'))
    primary = _text(row.get('primary_variant'))
    component = _text(row.get('component_measurement_status'))
    measured_value = row.get('measured_variants')
    measured = {_text(value) for value in measured_value} if isinstance(measured_value, list) else set()
    composed_blocks = []
    for container in ('timings', 'variant_results', 'runtime_results_by_variant'):
        value = row.get(container)
        if isinstance(value, Mapping) and 'composed' in value:
            item = value['composed']
            if not isinstance(item, Mapping):
                if item not in (None, ''):
                    return 'generic_composed_execution_evidence_unresolved'
            else:
                composed_blocks.append(item)
        elif isinstance(value, list):
            composed_blocks.extend(item for item in value if isinstance(item, Mapping)
                                   and _text(item.get('variant') or item.get('primary_variant')) in {'composed', 'split'})
    explicit = {field: row['composed_' + field] for field in
                _RUNTIME_START_FIELDS + _RUNTIME_SUCCESS_FIELDS + _RUNTIME_COUNT_FIELDS
                if 'composed_' + field in row}
    composed_blocks.append(explicit)
    if composed_status in _RUNTIME_STARTED or 'composed' in measured:
        return 'generic_composed_execution_conflicts_with_build_exclusion'
    for block in composed_blocks:
        if (block.get('identity_conflicts')
                or any(block.get(field) is True for field in _RUNTIME_START_FIELDS + _RUNTIME_SUCCESS_FIELDS)
                or any(_positive_runtime_count(block.get(field)) for field in _RUNTIME_COUNT_FIELDS)
                or _text(block.get('status')) in _RUNTIME_STARTED
                or any(type(block.get(field)) in (int, float) and block[field] > 0
                       for field in ('mean', 'mean_ms', 'count', 'n'))):
            return 'generic_composed_execution_conflicts_with_build_exclusion'
        if any(field in block and block[field] is not None
               and (type(block[field]) is not int or block[field] < 0)
               for field in _RUNTIME_COUNT_FIELDS):
            return 'generic_composed_execution_evidence_unresolved'
    # Completed-task counters/attestations cannot be supplied by a component
    # timing alone, even when the row also says part2_only.
    if (row.get('completed_task_endpoint_attested') is True
            or row.get('composed_completion_verified') is True
            or any(row.get(field) is True for field in
                   ('hailo_composed_runtime_ok', 'deepx_composed_runtime_ok'))):
        return 'generic_composed_completion_conflicts_with_build_exclusion'

    component_role = {'part2_only': 'part2', 'part1_only': 'part1'}.get(component)
    explicit_component = bool(
        component_role and primary == component_role
        and measured == {component_role}
        and composed_status in _RUNTIME_NONSTART
        and row.get('runtime_executable') is False
    )
    if explicit_component:
        # Row-level runtime_started/counts, including null counts, describe the
        # explicitly selected component. Do not turn them into composed work.
        return ''
    if (any(row.get(field) is True for field in _RUNTIME_START_FIELDS + _RUNTIME_SUCCESS_FIELDS)
            or any(_positive_runtime_count(row.get(field)) for field in _RUNTIME_COUNT_FIELDS)):
        return 'generic_runtime_role_unresolved' if primary not in {'composed', 'split'} else 'generic_composed_execution_conflicts_with_build_exclusion'
    if any(field in row and type(row[field]) is not int for field in
           ('repetition_count_attempted', 'repetition_count_valid')):
        return 'generic_composed_attempt_count_unresolved'
    # An explicitly skipped/nonstarted composed result is admissible even if
    # it has no component success. Preserve the historical exact zero case.
    if (composed_status in _RUNTIME_NONSTART
            or row.get('measurement_started') is False
            or type(row.get('repetition_count_attempted')) is int
            and row['repetition_count_attempted'] == 0):
        return ''
    return 'generic_runtime_role_unresolved'


def required_profile_build_exclusions(required_rows: Sequence[Mapping[str, Any]],
                                      readiness: Mapping[str, Any],
                                      observations: Sequence[Mapping[str, Any]] = (), *,
                                      diagnostics: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Resolve missing requested outcomes from this model's exact build stage.

    These are retained requested rows with a terminal exclusion, never synthetic
    measurements. A success-required request or contradictory measured result
    remains unresolved and therefore continues to fail the existing gate.
    """
    result = []
    for required in required_rows:
        if (required.get('success_required') not in (None, False)
                or required.get('identity_conflicts')
                or not required.get('logical_identity_sha256')):
            continue
        identity = planned_native_identity(required)
        if not all(identity[field] for field in FIELDS[:5]):
            continue
        if required.get('expected_setup_id') and required['expected_setup_id'] != identity['setup_id']:
            continue
        candidates = []
        for observation in readiness.get('blocked_jobs') or []:
            if not isinstance(observation, Mapping):
                continue
            # Existing local readiness can include an explicit setup. A
            # foreign nonempty setup must never be relabelled to this request.
            decision = observation.get('build_evidence')
            decision = decision if isinstance(decision, Mapping) else {}
            context = decision.get('context')
            context = context if isinstance(context, Mapping) else {}
            observed_setups = [str(value) for block in (observation, context)
                               for field in ('setup_id', 'measurement_setup_id', 'expected_setup_id')
                               if (value := block.get(field)) not in (None, '')]
            if any(value != identity['setup_id'] for value in observed_setups):
                continue
            candidate = failed_native_result(identity, failure_stage='build_backend_artifacts',
                failure_reason=str(observation.get('primary_failure_reason') or ''),
                upstream_build_observation=observation,
                upstream_evidence_path=str(observation.get('upstream_evidence_path') or ''))
            if known_build_exclusion(candidate):
                candidates.append(apply_known_build_disposition(candidate))
        if len(candidates) != 1:
            continue
        conflicting = []
        for row in observations:
            if not isinstance(row, Mapping) or native_identity_key(row)[:5] != native_identity_key(identity)[:5]:
                continue
            # Compile evidence belongs to the required contract. A different
            # explicit runtime precision is another observation, not proof of
            # this request's composed success or nonstart.
            required_precision = required.get('precision') or required.get('runtime_precision_identity')
            observed_precision = row.get('precision') or row.get('runtime_precision_identity')
            if required_precision and observed_precision:
                from .workflow.logical_measurement import canonical_runtime_precision_identity
                expected_precision, expected_error = canonical_runtime_precision_identity(required_precision)
                actual_precision, actual_error = canonical_runtime_precision_identity(observed_precision)
                if expected_error or actual_error:
                    conflicting.append('generic_runtime_precision_unresolved')
                    continue
                if expected_precision != actual_precision:
                    continue
            role = _text(required.get('variant') or required.get('primary_variant'))
            if role in {'split', 'composed'}:
                reason = _generic_composed_observation_conflict(row)
            else:
                reason = ('generic_runtime_observation_conflict'
                          if any(row.get(field) is True for field in _RUNTIME_SUCCESS_FIELDS + _RUNTIME_START_FIELDS)
                          or row.get('repetition_count_attempted', 0) != 0 else '')
            if reason:
                conflicting.append(reason)
        if conflicting:
            if diagnostics is not None:
                diagnostics.append({'logical_identity_sha256': required['logical_identity_sha256'],
                                    'planned_native_identity': identity,
                                    'reasons': sorted(set(conflicting))})
            continue
        result.append({**dict(required), **candidates[0],
                       'schema': 'onnx-splitpoint/required-profile-build-exclusion',
                       'measurement_values_synthesized': False})
    return result

def attach_identity_without_conflicts(identity: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    planned = planned_native_identity(identity)
    observed = planned_native_identity(observation)
    # Only explicitly observed fields participate: inferred comparison is useful
    # for matching but is not a physical child assertion.
    aliases = {'model': ('model','model_id'), 'case': ('case','case_id','boundary'),
               'backend': ('backend','producer_backend'), 'setup_id': ('setup_id','measurement_setup_id','setup'),
               'comparison_backend': ('comparison_backend',), 'precision': ('precision','contract')}
    conflicts = [field for field in FIELDS if any(observation.get(k) not in (None,'') for k in aliases[field])
                 and planned[field] and observed[field] != planned[field]]
    result = dict(observation)
    if conflicts:
        return failed_native_result(planned, failure_stage='child_result_identity',
                                    failure_reason='native_job_identity_conflict:' + ','.join(conflicts),
                                    repetition_count_attempted=int(observation.get('repetition_count_attempted') or 0),
                                    child_observation=dict(observation), identity_conflicts=conflicts)
    result.update(planned)
    result['planned_native_identity'] = planned
    return result

def complete_historical_identity(row: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fill empty fields using documented planned context; never alter input.

    Context provenance must come from a dispatcher/plan, not a filename vendor
    guess. A wrong nonempty field cannot be repaired by dropping it.
    """
    result = dict(row)
    current = planned_native_identity(row)
    if all(str(row.get(field) or '').strip() for field in FIELDS[:5]) and (current['precision'] or current['backend'].startswith('native_full_')):
        if contexts and not any(native_identity_key(row) == native_identity_key(candidate) for candidate in contexts):
            related = [planned_native_identity(candidate) for candidate in contexts
                       if all(planned_native_identity(candidate)[field] == current[field] for field in ('backend','model','case'))]
            if related:
                result['identity_status'] = 'identity_conflict'
                result['identity_conflicts'] = sorted({field for candidate in related for field in FIELDS
                                                      if candidate[field] != current[field]})
        return result
    explicit = row.get('planned_native_identity')
    candidates = [explicit] if isinstance(explicit, Mapping) else list(contexts)
    compatible = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        normalized = planned_native_identity(candidate)
        if all(not value or normalized[field] == value for field, value in current.items()):
            if normalized not in compatible:
                compatible.append(normalized)
    if len(compatible) != 1:
        result['identity_status'] = 'identity_ambiguous' if len(compatible) > 1 else 'identity_unresolved'
        result['identity_candidate_count'] = len(compatible)
        return result
    completed = compatible[0]
    fields = [field for field in FIELDS if not str(row.get(field) or '').strip() and completed[field]]
    result.update({field: completed[field] for field in fields})
    result['planned_native_identity'] = completed
    result['identity_completion'] = {'source': 'documented_dispatcher_plan', 'fields': fields,
                                     'measurement_values_modified': False}
    return result

def native_job_prerequisite(job: Mapping[str, Any], *, suite_dir: Any,
                            model_dir: Any, quality_error: str = '') -> dict[str, Any]:
    """Read the selected job's existing build/endpoint decisions before launch.

    Exact negative-cache admission is owned by the existing build service. This
    reader consumes its current selected-case decision, never invents a blacklist.
    """
    import json
    from pathlib import Path
    identity = planned_native_identity(job)
    suite, model = Path(suite_dir), Path(model_dir)
    case, backend = identity['case'], identity['backend']
    def read(path):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}
    def blocked(reason, stage, path):
        return failed_native_result(identity, failure_stage=stage, failure_reason=reason,
                                    upstream_evidence_path=str(path))
    # Only the current model's actual build stage may supersede its later
    # missing quality mirror. No neighbouring model/case/setup search is used.
    stage_path = model / 'stages/build_backend_artifacts/stage_result.json'
    stage_result = read(stage_path)
    readiness = (stage_result.get('details') or {}).get('deferred_build_readiness') or {}
    expected_stage = 'full' if backend.startswith('native_full_') else 'part1'
    if backend != 'native_full_tensorrt':
        for value in readiness.get('blocked_jobs', []):
            if not isinstance(value, Mapping) or value.get('identity_conflicts'):
                continue
            if (str(value.get('model_id') or '') != identity['model']
                    or native_comparison(value.get('backend')) != native_comparison(backend)
                    or str(value.get('boundary') or '') != case
                    or str(value.get('stage') or '') != expected_stage
                    or value.get('job_kind', 'artifact') != 'artifact'
                    or value.get('status') not in {'failed', 'known_infeasible'}
                    or value.get('readiness') not in {'blocked', 'not_executable'}):
                continue
            if value.get('setup_id') and value['setup_id'] != identity['setup_id']:
                continue
            path = str(value.get('upstream_evidence_path') or '')
            reason = str(value.get('primary_failure_reason') or '')
            if path and reason:
                result = blocked(reason, 'build_backend_artifacts', path)
                result['upstream_build_observation'] = dict(value)
                if quality_error:
                    result['secondary_failure_reason'] = quality_error
                return apply_known_build_disposition(result)
    if backend == 'deepx_to_trt':
        path = suite / 'deepx/deepx_m1/part1/deepx_part1_artifact_status.json'
        for value in read(path).get('cases', []):
            if str(value.get('case_id') or '') == case and value.get('ok') is False:
                reason = str(value.get('error') or value.get('failure_reason') or '')
                if reason:
                    return blocked(reason, 'build_backend_artifacts', path)
    if backend == 'hailo8_to_trt':
        path = model / 'benchmark_set/backend_artifact_decisions.json'
        for value in read(path).get('case_build_requests', []):
            if native_backend(value.get('backend')) != backend or str(value.get('case_id') or value.get('case_dir')) != case:
                continue
            availability = value.get('hailo_case_variant_availability') or {}
            reason = str(availability.get('part1_error') or '')
            # This literal is emitted only after the existing exact build-policy
            # lookup; generic timeout/parser failures deliberately do not match.
            if availability.get('part1_failed') and 'Exact build evidence prevents repeated compiler attempt' in reason and 'exact_deterministic_outcome' in reason:
                return blocked(reason, 'build_backend_artifacts', path)
    if quality_error:
        path = model / 'benchmark_results/normalized_results.json'
        for value in read(path).get('results', []):
            if native_backend(value.get('backend')) != backend or str(value.get('case_id') or value.get('case')) != case or str(value.get('setup_id') or '') != identity['setup_id']:
                continue
            pending = [value]
            while pending:
                nested = pending.pop(0)
                if isinstance(nested, dict):
                    reason = str(nested.get('endpoint_attestation_reason') or '')
                    if reason and reason not in {'ok', 'pass', 'attested'}:
                        return blocked(reason, 'evaluate_quality', path)
                    pending.extend(nested.values())
                elif isinstance(nested, list):
                    pending.extend(nested)
        return blocked(quality_error, 'evaluate_quality', path)
    return {**identity, 'planned_native_identity':identity, 'prerequisite_status':'ready'}

def native_build_summary(statuses: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate existing backend decisions without declaring partial work PASS."""
    states = {str(key):str(value or 'partial').lower() for key,value in statuses.items()}
    blocked = {key:value for key,value in states.items() if value not in {'ok','skipped'}}
    return {'backend_build_status':'partial' if blocked else 'ok',
            'ready_backend_count':sum(value == 'ok' for value in states.values()),
            'blocked_backend_count':len(blocked), 'backend_build_states':states}

def select_native_target(targets: Sequence[Mapping[str, Any]], backend: str) -> Mapping[str, Any] | None:
    """Vendor selection alone is insufficient when more than one setup matches."""
    requested = native_comparison(backend)
    candidates = [row for row in targets if row.get('enabled', True)
                  and native_comparison(row.get('accelerator')) == requested]
    if len(candidates) > 1:
        raise ValueError('native_setup_identity_ambiguous:' + ','.join(str(row.get('id') or '') for row in candidates))
    return candidates[0] if candidates else None
