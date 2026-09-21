"""Count/time-bound reporting projections; never change measured workloads.

The saved endpoint owns its samples, interval and units. In particular a P2
rate cannot borrow a completed-task interval from a surrounding summary.
"""
from __future__ import annotations

import math
import statistics
import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .runners.request_latency import latency_fields


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _host_output_only(record: Mapping[str, Any]) -> bool:
    latency = record.get('request_latency')
    return bool(record.get('task') == 'classification' and (
        record.get('task_complete') is False
        or isinstance(latency, Mapping) and latency.get('task_complete') is False
        or record.get('contract_family') == 'classification_logits' and not (
            record.get('task_complete') is True
            and record.get('completed_task_stage') == 'classification_top1_top5'
            and record.get('postprocess_completion_verified') is True
        )
    ))


def _completed(record: Mapping[str, Any]) -> bool:
    if _host_output_only(record):
        return False
    task = str(record.get('task') or '').lower()
    boundary = str(record.get('measurement_boundary') or '')
    endpoint = str(record.get('measurement_endpoint') or record.get('performance_endpoint') or '')
    if task == 'classification':
        return boundary in {'workers_ready_to_last_completed_trt_frame',
                            'workers_ready_to_last_completed_task_frame',
                            'first_task_start_to_last_task_completion'}
    return bool(
        (endpoint == 'completed_task' or 'completed_task' in boundary or boundary == 'first_task_start_to_last_task_completion')
        and record.get('postprocess_completion_verified') is True
        and record.get('completed_task_endpoint_attested') is True
    )


def project_rate_endpoint(payload: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    """Validate one saved series without synthesizing counts or durations."""
    result: dict[str, Any] = {
        'endpoint': endpoint, 'status': 'unavailable', 'reason': 'endpoint_evidence_missing',
        'fps': None, 'ci95_low': None, 'ci95_high': None, 'ci95_method': '',
        'samples': [], 'work_unit_counts': [], 'measurement_times_s': [],
        'repetition_ids': [], 'repetition_count_valid': 0,
    }
    if not payload:
        return result
    declared = str(payload.get('measurement_endpoint') or '')
    allowed_labels = {endpoint, 'model_outputs' if endpoint == 'completed_task' and payload.get('task') == 'classification' else endpoint}
    if endpoint == 'host_outputs' and payload.get('task') == 'classification':
        allowed_labels.update({'completed_task', 'model_outputs', 'raw_model_outputs'})
    if declared and declared not in allowed_labels:
        result['reason'] = 'endpoint_label_mismatch'
        return result
    records = payload.get('repetition_records') or payload.get('repetition_evidence')
    requested = _number(payload.get('repetition_count_valid'))
    if not records:
        if requested is not None and requested > 1:
            result['reason'] = 'repetition_count_time_evidence_missing'
            return result
        records = [payload]
    if not isinstance(records, list) or not records:
        return result
    samples, counts, times, ids = [], [], [], []
    contracts = set()
    for record in records:
        if not isinstance(record, Mapping) or record.get('ok') is False:
            continue
        if not record.get('task') and payload.get('task'):
            record = dict(record, task=payload['task'])
        for key in ('task', 'measurement_boundary', 'completed_task_endpoint_contract_hash',
                    'completion_execution_contract_sha256'):
            if payload.get(key) and record.get(key) and payload[key] != record[key]:
                result['reason'] = 'aggregate_repetition_endpoint_contract_mismatch'
                return result
        # Do not fill an incomplete repetition from the aggregate/last run.
        if endpoint == 'completed_task' and (_host_output_only(payload) or not _completed(record)):
            result['reason'] = 'completion_endpoint_provenance_missing_or_mixed'
            return result
        if endpoint == 'host_outputs' and not (
            (_host_output_only(record) or _host_output_only(payload))
            and record.get('task_complete') is not True
            and not (isinstance(record.get('request_latency'), Mapping)
                     and record['request_latency'].get('task_complete') is True)
            and record.get('measurement_boundary') in {
                'workers_ready_to_last_completed_trt_frame',
                'workers_ready_to_last_completed_task_frame',
                'first_task_start_to_last_task_completion',
            }
        ):
            result['reason'] = 'host_output_endpoint_provenance_missing_or_mixed'
            return result
        if endpoint == 'raw_model_outputs' and (
            record.get('measurement_boundary') != 'workers_ready_to_last_completed_trt_frame'
            or str(record.get('measurement_endpoint') or '') not in {'', 'raw_model_outputs'}):
            result['reason'] = 'raw_output_repetition_endpoint_mismatch'
            return result
        count = _number(record.get('completed_work_units', record.get('completed_frames')))
        if str(record.get('completed_work_units_status') or '').lower() in {'unavailable', 'invalid', 'failed', 'inconclusive'}:
            result['reason'] = 'completed_count_status_invalid'
            return result
        if endpoint == 'completed_task' and (record.get('task') == 'detection' or record.get('completed_task_stage') == 'classification_top1_top5') and record.get('postprocess_completed_frames') is not None and _number(record['postprocess_completed_frames']) != count:
            result['reason'] = 'postprocess_completed_count_conflict'
            return result
        if record.get('completed_work_units') is not None and record.get('completed_frames') is not None and _number(record['completed_frames']) != count:
            result['reason'] = 'completed_count_conflict'
            return result
        contract = (record.get('task'), record.get('measurement_boundary'),
                    record.get('completed_task_endpoint_contract_hash'), record.get('completion_execution_contract_sha256'))
        contracts.add(contract)
        if len(contracts) > 1:
            result['reason'] = 'repetition_endpoint_contract_mismatch'
            return result
        if endpoint == 'completed_task' and record.get('task') == 'detection' and not record.get('completed_task_endpoint_contract_hash'):
            result['reason'] = 'completion_endpoint_contract_missing'
            return result
        milliseconds = _number(record.get('makespan_ms'))
        seconds = milliseconds / 1000 if milliseconds is not None else _number(record.get('measured_duration_s'))
        if count is None or count <= 0 or count != int(count) or seconds is None or seconds <= 0:
            result['reason'] = 'completed_count_or_measurement_time_missing'
            return result
        fps = _number(record.get('fps_makespan', record.get('fps')))
        measured = count / seconds
        if fps is None or not math.isclose(fps, measured, rel_tol=1e-6, abs_tol=1e-6):
            result['reason'] = 'rate_count_time_mismatch'
            return result
        identity = str(record.get('repetition_id') or record.get('runtime_instance_id') or '')
        if len(records) > 1 and (not identity or identity in ids):
            result['reason'] = 'repetition_identity_missing_or_duplicate'
            return result
        ids.append(identity); samples.append(fps); counts.append(int(count)); times.append(seconds)
    if not samples or requested is not None and int(requested) != len(samples):
        result['reason'] = 'repetition_series_incomplete'
        return result
    median = statistics.median(samples)
    declared_samples = payload.get('fps_repetition_samples')
    if declared_samples is not None and (not isinstance(declared_samples, list) or len(declared_samples) != len(samples)
            or any(_number(a) is None or not math.isclose(float(a), b, rel_tol=1e-6) for a, b in zip(declared_samples, samples))):
        result['reason'] = 'aggregate_repetition_series_mismatch'
        return result
    published = _number(payload.get('fps_median', payload.get('fps_makespan')))
    if published is not None and not math.isclose(published, median, rel_tol=1e-6, abs_tol=1e-6):
        result['reason'] = 'aggregate_repetition_series_mismatch'
        return result
    low, high = _number(payload.get('fps_ci95_low')), _number(payload.get('fps_ci95_high'))
    method = str(payload.get('fps_ci95_method') or '')
    if low is not None or high is not None:
        if low is None or high is None or not (min(samples)-1e-6 <= low <= median <= high <= max(samples)+1e-6):
            result['reason'] = 'interval_repetition_series_mismatch'
            return result
    if len(samples) == 1:
        # One repetition cannot estimate between-repetition uncertainty.
        # Preserve its FPS and all raw saved interval fields in the source.
        low = high = None
        method = 'not_estimated_single_repetition'
    result.update(status='available', reason='', fps=median, ci95_low=low,
                  ci95_high=high, ci95_method=method, samples=samples,
                  work_unit_counts=counts, measurement_times_s=times,
                  repetition_ids=ids, repetition_count_valid=len(samples))
    return result


def rate_endpoint_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = payload.get('endpoint_results')
    endpoints = endpoints if isinstance(endpoints, Mapping) else {}
    completed = project_rate_endpoint(endpoints.get('completed_task') or (payload if not endpoints else {}), 'completed_task')
    host = project_rate_endpoint(endpoints.get('completed_task') or (payload if not endpoints else {}), 'host_outputs')
    raw = project_rate_endpoint(endpoints.get('raw_model_outputs') or {}, 'raw_model_outputs')
    fps = completed['fps']
    return {
        **latency_fields(payload),
        'rate_endpoint_projection_version': 1,
        'completed_task_rate': completed, 'p2_output_rate': raw,
        'host_output_rate': host,
        'host_output_fps': host['fps'],
        'host_output_fps_ci95_low': host['ci95_low'],
        'host_output_fps_ci95_high': host['ci95_high'],
        'host_output_fps_unavailable_reason': host['reason'],
        'host_output_rate_endpoint': 'host_outputs_without_task_postprocessing',
        'completed_task_fps': fps, 'completed_task_fps_median': fps,
        'completed_task_fps_ci95_low': completed['ci95_low'],
        'completed_task_fps_ci95_high': completed['ci95_high'],
        'completed_task_fps_unavailable_reason': completed['reason'],
        'fps_makespan': fps, 'fps_median': fps,
        'fps_ci95_low': completed['ci95_low'], 'fps_ci95_high': completed['ci95_high'],
        'fps_ci95_method': completed['ci95_method'],
        'fps_repetition_samples': completed['samples'],
        'performance_endpoint': 'completed_task', 'primary_performance_endpoint': 'completed_task',
        'throughput_primary_fps': fps, 'application_throughput_fps': fps,
        'completed_detection_fps': fps if str(payload.get('task') or '') == 'detection' else None,
        'p2_output_fps': raw['fps'], 'p2_output_fps_ci95_low': raw['ci95_low'],
        'p2_output_fps_ci95_high': raw['ci95_high'],
        'p1_output_fps': None,
        'p1_output_fps_unavailable_reason': 'separate_stage_count_time_not_recorded',
        'estimated_pipeline_cycle_fps': _number(payload.get('paper_equivalent_fps', payload.get('paper_fps'))),
        'estimated_pipeline_cycle_fps_semantics': 'estimated_stage_cycle; not completed task throughput',
        'completed_task_work_unit_counts': completed['work_unit_counts'],
        'completed_task_measurement_times_s': completed['measurement_times_s'],
        'p2_output_work_unit_counts': raw['work_unit_counts'],
        'p2_output_measurement_times_s': raw['measurement_times_s'],
        **_historical_rate_fields(payload, unavailable=fps is None),
    }


def _historical_rate_fields(payload: Mapping[str, Any], *, unavailable: bool) -> dict[str, Any]:
    """Retain a saved rate as diagnosis, without upgrading its evidence."""
    retained = payload.get('historical_rate')
    if isinstance(retained, Mapping):
        diagnostic = dict(retained)
    else:
        diagnostic = {}
        fps = _number(payload.get('fps_median', payload.get('fps_makespan')))
        if unavailable and fps is not None:
            diagnostic = {
                'fps': fps, 'ci95_low': _number(payload.get('fps_ci95_low')),
                'ci95_high': _number(payload.get('fps_ci95_high')),
                'ci95_method': str(payload.get('fps_ci95_method') or payload.get('repetition_aggregation') or ''),
                'endpoint': str(payload.get('measurement_endpoint') or payload.get('performance_endpoint')
                                or payload.get('comparison_endpoint_stratum') or payload.get('contract_family') or 'unrecorded'),
                'source': str(payload.get('fps_source') or payload.get('performance_benchmark_source') or 'saved_native_summary'),
                'e2e_scope': str(payload.get('e2e_scope') or ''),
                'samples': list(payload.get('fps_repetition_samples') or []),
                'status': 'historical_diagnostic_only',
            }
    return {'historical_rate': diagnostic, 'historical_fps': diagnostic.get('fps'),
            'historical_fps_ci95_low': diagnostic.get('ci95_low'),
            'historical_fps_ci95_high': diagnostic.get('ci95_high'),
            'historical_performance_endpoint': diagnostic.get('endpoint', ''),
            'historical_fps_source': diagnostic.get('source', '')}


def _canonical_full_row(row: Mapping[str, Any], base: Path | None) -> tuple[Mapping[str, Any], str]:
    """Read the collected Full aggregate, with the existing exact row identity.

    Its report often names a remote last repetition. Never replace the saved
    series with that leaf or select a collection by modification time.
    """
    if str(row.get('case') or row.get('case_id') or '').lower() not in {'full', 'full_model'}:
        return row, ''
    root = str(row.get('source_root') or '')
    if not root:
        return row, ''
    path = Path(root) / 'analysis_tables/native_full_baseline_eval.json'
    if not path.is_absolute() and base is not None:
        path = base / path
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return row, ''
    keys = ('backend', 'model', 'case', 'setup_id', 'comparison_backend', 'precision')
    matches = [r for r in data.get('rows', []) if isinstance(r, Mapping)
               and all(str(r.get(k) or '') == str(row.get(k) or '') for k in keys)]
    if len(matches) != 1:
        return {}, 'canonical_full_identity_missing_or_ambiguous'
    raw = matches[0]
    for key in ('full_command_contract_sha256', 'quality_first_producer_identity_sha256'):
        if row.get(key) and row.get(key) != raw.get(key):
            return {}, 'raw_report_summary_identity_conflict'
    # Repetition UUIDs survive aggregation with a different repetition_id label.
    def ids(value: Mapping[str, Any]) -> set[str]:
        return {str(r.get('runtime_instance_id')) for r in value.get('repetition_records', [])
                if isinstance(r, Mapping) and r.get('runtime_instance_id')}
    if ids(row) and ids(row) != ids(raw):
        return {}, 'raw_report_summary_identity_conflict'
    return raw, str(path)


def report_rate_fields(row: Mapping[str, Any], base: Path | None = None) -> dict[str, Any]:
    """Replay the saved raw report, or validate the retained summary series."""
    canonical, source = _canonical_full_row(row, base)
    if source:
        fields = rate_endpoint_fields(canonical)
        fields['rate_endpoint_source'] = source
        if not canonical:
            fields['completed_task_rate']['reason'] = source
            fields['completed_task_fps_unavailable_reason'] = source
        return fields
    payload = row
    report = str(row.get('report') or '')
    if report:
        path = Path(report)
        if not path.is_absolute() and base is not None:
            path = base / path
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raw = None
        if isinstance(raw, dict):
            expected = _number(row.get('repetition_count_valid'))
            actual = _number(raw.get('repetition_count_valid')) or 1
            # A Full aggregate often points only to the final repetition.
            count_conflict = expected is not None and expected != actual
            full_summary = str(row.get('case') or row.get('case_id') or '').lower() in {'full', 'full_model'}
            conflict = count_conflict and not (full_summary and expected > 1 and actual == 1)
            for key in ('case', 'setup_id', 'precision', 'native_command_contract_sha256'):
                if row.get(key) and raw.get(key) and row[key] != raw[key]:
                    conflict = True
            raw_model = raw.get('model') or raw.get('model_id') or (raw.get('native_command_contract') or {}).get('model_id')
            if raw_model and (row.get('model') or row.get('model_id')) and raw_model != (row.get('model') or row.get('model_id')):
                conflict = True
            digest = row.get('native_fifo_result_sha256')
            if digest and digest != hashlib.sha256(path.read_bytes()).hexdigest():
                conflict = True
            if conflict:
                fields = rate_endpoint_fields({})
                fields['completed_task_fps_unavailable_reason'] = 'raw_report_summary_identity_conflict'
                fields['completed_task_rate']['reason'] = 'raw_report_summary_identity_conflict'
                fields['rate_endpoint_source'] = report
                return fields
            if (expected is None or expected == actual) and not (
                    full_summary and (row.get('repetition_records') or row.get('repetition_evidence'))):
                payload = raw
    fields = rate_endpoint_fields(payload)
    fields['rate_endpoint_source'] = report if payload is not row else 'summary_repetition_evidence'
    return fields


def format_rate_endpoints(row: Mapping[str, Any]) -> str:
    def rate(value: Any, low: Any, high: Any) -> str:
        if value is None:
            return 'unavailable'
        interval = f' (CI95 {low:.3f}–{high:.3f})' if low is not None and high is not None else ''
        return f'{value:.3f} FPS{interval}'
    task = rate(row.get('completed_task_fps'), row.get('completed_task_fps_ci95_low'), row.get('completed_task_fps_ci95_high'))
    p2 = rate(row.get('p2_output_fps'), row.get('p2_output_fps_ci95_low'), row.get('p2_output_fps_ci95_high'))
    host = ''
    if row.get('host_output_fps') is not None:
        host = '; Hostoutput (ohne Task-Postprocessing): ' + rate(
            row['host_output_fps'], row.get('host_output_fps_ci95_low'), row.get('host_output_fps_ci95_high'))
    p1 = rate(row.get('p1_output_fps'), None, None)
    estimated = rate(row.get('estimated_pipeline_cycle_fps', row.get('native_theoretical_cycle_rate_fps')), None, None)
    latency_text = 'unavailable'
    if row.get('request_latency_status') == 'complete':
        latency_text = (f"Mean {row['request_latency_mean_ms']:.3f} / P50 {row['request_latency_p50_ms']:.3f} / "
                        f"P95 {row['request_latency_p95_ms']:.3f} ms; "
                        f"n={row['request_latency_count']}/{row['request_latency_expected_count']}; "
                        f"ab vorbereitetem Input; {row['request_latency_semantics']}")
    elif row.get('request_latency_unavailable_reason'):
        latency_text += (f" ({row['request_latency_unavailable_reason']}; "
                         f"n={row.get('request_latency_count', 0)}/{row.get('request_latency_expected_count', 0)})")
    if row.get('host_output_latency_mean_ms') is not None:
        latency_text += (f"; Hostoutputlatenz ab vorbereitetem Input: Mean {row['host_output_latency_mean_ms']:.3f} / "
                         f"P50 {row['host_output_latency_p50_ms']:.3f} / P95 {row['host_output_latency_p95_ms']:.3f} ms "
                         "(Task-Postprocessing nicht enthalten)")
    historical = ''
    if row.get('historical_fps') is not None and row.get('host_output_fps') is None:
        historical = ('; Historische Diagnose [' + str(row.get('historical_performance_endpoint') or 'unrecorded')
                      + '; ' + str(row.get('historical_fps_source') or 'saved_native_summary') + ']: '
                      + rate(row['historical_fps'], row.get('historical_fps_ci95_low'), row.get('historical_fps_ci95_high')))
    status = str(row.get('runtime_status') or row.get('status') or 'unavailable')
    semantic = str(row.get('semantic_status') or row.get('task_quality_status') or 'unavailable')
    output_status = 'Ausgabevertrag erfüllt' if semantic == 'claim_ok' else semantic
    from .accuracy_reporting import accuracy_label
    quality = accuracy_label(row.get('accuracy_assessment')) or str(row.get('task_quality_status') or row.get('accuracy_gate_status') or 'unavailable')
    return f'Completed Task: {task}{host}; P1-Ausgabe: {p1}; P2-Ausgabe: {p2}; geschätzter Pipelinezyklus: {estimated}; Einbildlatenz: {latency_text}; Runtime: {status}; Output: {output_status}; Taskqualität: {quality}{historical}'
