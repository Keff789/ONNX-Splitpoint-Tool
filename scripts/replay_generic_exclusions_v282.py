#!/usr/bin/env python3
"""Read-only Generic/AP5 report projection of an existing EvaluationRun.

Uses the same observation reader, required matrix and display projection as the
workflow. Never imports the energy collector, starts SSH, builds artifacts or
runs inference. All output is a new directory outside the historical run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

# Bind scripts launched with -I to their delivered source tree.
_SOURCE = Path(__file__).resolve().parents[1]
if (_SOURCE / 'onnx_splitpoint_tool').is_dir():
    sys.path.insert(0, str(_SOURCE))

from onnx_splitpoint_tool import __version__, __build_id__
from onnx_splitpoint_tool.native_job_identity import native_backend
from onnx_splitpoint_tool.workflow.results import _rows_from_json
from onnx_splitpoint_tool.workflow.logical_measurement import select_logical_primary_rows
from onnx_splitpoint_tool.workflow.runner import required_profile_outcomes_v282, _native_expected_matrix_status_v60y
from onnx_splitpoint_tool.workflow.evidence_status import workflow_completion_projection


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def replay(run_root: Path, out: Path, *, generic_only: bool = False) -> dict[str, Any]:
    root = run_root.resolve(strict=True)
    out = out.resolve()
    if not root.is_dir() or out == root or root in out.parents or out in root.parents:
        raise ValueError('replay_output_must_be_outside_original_run')
    sources: dict[str, bytes] = {}
    files: dict[str, bytes] = {}

    def read_source(path: Path, *, optional: bool = False) -> dict[str, Any]:
        if optional and not path.exists():
            return {}
        actual = path.resolve(strict=True)
        if root not in actual.parents or not actual.is_file():
            raise ValueError('replay_source_path_outside_run:' + str(path))
        data = actual.read_bytes()
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError('replay_source_not_object:' + str(path))
        sources[path.relative_to(root).as_posix()] = data
        return value

    manifest = read_source(root / 'run_manifest.json', optional=generic_only)
    source_identity = {
        'original_run_id': manifest.get('run_id') or root.name,
        'original_run_root': str(root),
        'original_tool_version': manifest.get('current_tool_version') or manifest.get('tool_version'),
        'original_workflow_version': manifest.get('current_workflow_version') or manifest.get('workflow_version'),
        'evaluation_tool_version': __version__, 'evaluation_build_id': __build_id__,
        'source_results_modified': False, 'hardware_execution': False,
        'compiler_dispatch': False, 'quality_recalculation': False,
        'energy_reimport': False,
    }
    model_reports = []
    all_exclusions = []
    all_missing = []
    observations_preserved = 0
    paths = sorted(root.glob('models/*/benchmark_results/normalized_results.json'))
    if not paths:
        raise ValueError('no_normalized_model_results')
    for path in paths:
        model = path.parents[1].name
        payload = read_source(path)
        if payload.get('model_id') != model:
            raise ValueError('normalized_model_identity_mismatch:' + model)
        stage = read_source(path.parents[1] / 'stages/build_backend_artifacts/stage_result.json')
        readiness = (stage.get('details') or {}).get('deferred_build_readiness') or {}
        required = payload.get('required_profile_results')
        if not isinstance(required, list) or any(not isinstance(row, dict) for row in required):
            raise ValueError('required_profile_scope_missing_or_invalid:' + model)
        rows = _rows_from_json(path)
        if rows != payload.get('results'):
            raise ValueError('normalized_source_reader_did_not_preserve_rows:' + model)
        logical, group_errors = select_logical_primary_rows(rows)
        matrix = required_profile_outcomes_v282(required, logical, rows, readiness)
        matrix.update(schema='onnx-splitpoint/required-profile-matrix', schema_version=1,
                      model_id=model, logical_group_errors=group_errors,
                      matrix_complete=not (matrix['missing_result_count'] or matrix['duplicate_result_count'] or group_errors),
                      evaluation_projection=source_identity)
        old_fields = {key: payload[key] for key in (
            'missing_required_profile_results', 'missing_required_profile_result_count',
            'excluded_required_profile_results', 'excluded_required_profile_result_count',
            'executor_status', 'status',
        ) if key in payload}
        projected = dict(payload)
        projected.update(
            missing_required_profile_results=matrix['missing_results'],
            missing_required_profile_result_count=matrix['missing_result_count'],
            excluded_required_profile_results=matrix['excluded_results'],
            excluded_required_profile_result_count=matrix['excluded_result_count'],
            duplicate_required_profile_results=matrix['duplicate_results'],
            duplicate_required_profile_result_count=matrix['duplicate_result_count'],
            build_exclusion_conflicts=matrix['build_exclusion_conflicts'],
            build_exclusion_conflict_count=matrix['build_exclusion_conflict_count'],
            original_projection=old_fields, evaluation_projection=source_identity,
        )
        # Preserve all component metrics, original statuses and quality values.
        if projected['results'] != payload['results']:
            raise ValueError('measurement_projection_changed_original_rows:' + model)
        files[f'models/{model}/benchmark_results/normalized_results.json'] = _encoded(projected)
        files[f'models/{model}/benchmark_results/required_profile_matrix.json'] = _encoded(matrix)
        all_exclusions.extend(matrix['excluded_results'])
        all_missing.extend(matrix['missing_results'])
        observations_preserved += len(rows)
        model_reports.append({
            'model_id': model, 'original_missing_count': payload.get('missing_required_profile_result_count'),
            'missing_count': matrix['missing_result_count'], 'excluded_count': matrix['excluded_result_count'],
            'duplicate_count': matrix['duplicate_result_count'], 'conflict_count': matrix['build_exclusion_conflict_count'],
            'logical_group_error_count': len(group_errors), 'measurement_rows_preserved': len(rows),
        })

    native_counts: dict[str, Any] = {'status': 'not_checked_generic_only'}
    quality_counts: dict[str, Any] = {'status': 'not_checked_generic_only'}
    quality = {}
    native_evidence = {}
    if not generic_only:
        original_matrix = read_source(root / 'reports/native_expected_matrix.json')
        native_summary = read_source(root / 'reports/native_producer_summary.json')
        native_evidence = read_source(root / 'reports/native_evidence_status.json', optional=True)
        # Use original requested identities and actual observations; exclusions
        # remain unmeasured terminal outcomes in the existing matrix contract.
        expected = list(original_matrix.get('present_expected_rows') or []) + list(original_matrix.get('missing_expected_rows') or [])
        native = _native_expected_matrix_status_v60y(expected, native_summary['rows'], [])
        fields = ('expected_row_count', 'present_expected_row_count', 'successful_expected_row_count',
                  'failed_expected_row_count', 'missing_expected_row_count', 'excluded_expected_row_count')
        for field in fields:
            if native.get(field) != original_matrix.get(field):
                raise ValueError('native_partition_changed:' + field)
        native_counts = {field: native[field] for field in fields}
        native_counts['full_success_count'] = sum(
            native_backend(row.get('backend')).startswith('native_full_') and row.get('ok') is True
            for row in native_summary['rows'])
        native_counts['status'] = 'original_partition_preserved'
        files['reports/native_expected_matrix.json'] = _encoded(native)
        quality = read_source(root / 'quality_management/central_quality_summary.json')
        decisions = Counter(str(row.get('quality_decision') or '').lower() for row in quality['results'])
        declared_decisions = quality.get('quality_decision_counts') or {}
        if dict(decisions) != declared_decisions:
            raise ValueError('quality_decision_count_mismatch')
        quality_counts = {
            'status': 'original_results_preserved', 'result_count': len(quality['results']),
            'decision_counts': dict(decisions),
            'primary_result_count': quality.get('matched_primary_result_count'),
            'companion_count': quality.get('companions'),
            'technical_failed_count': quality.get('technical_failed_count'),
            'recalculated': False,
        }
        # Copied verbatim, with every original decision and provenance intact.
        files['quality_management/central_quality_summary.json'] = sources['quality_management/central_quality_summary.json']

    completion = workflow_completion_projection(
        'partial' if all_missing or native_counts.get('failed_expected_row_count') or generic_only else 'ok',
        native_evidence=native_evidence, central_quality=quality,
        generic_excluded_count=len(all_exclusions),
    )
    report = {
        'schema': 'onnx-splitpoint/readonly-generic-replay', 'schema_version': 1,
        **source_identity, 'scope': 'generic_only' if generic_only else 'generic_native_quality_preservation',
        'status': 'projection_complete', 'run_pass_claimed': False,
        'models': model_reports,
        'generic': {
            'original_missing_count': sum(int(item['original_missing_count'] or 0) for item in model_reports),
            'missing_count': len(all_missing), 'excluded_count': len(all_exclusions),
            'observation_count_preserved': observations_preserved,
            'exclusions': all_exclusions, 'remaining_missing': all_missing,
            'component_metrics_preserved': True, 'quality_values_preserved': True,
        },
        'native': native_counts, 'quality': quality_counts, 'completion': completion,
        'limitations': [
            'Existing invalid H10 endpoints remain in the unchanged observation rows.',
            'Remaining workspace-blocked jobs have not been built or measured.',
            'Energy replay is a separate command; this projection does not change its old import decisions.',
            'No new final-scope, performance or energy claim is granted by reporting.',
        ],
        'source_files': [
            {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            for name, data in sorted(sources.items())
        ],
    }
    files['REPLAY_REPORT.json'] = _encoded(report)
    files['reports/run_status_summary.json'] = _encoded({'completion': completion, 'evaluation_projection': source_identity})
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=['model_id', 'case_id', 'backend', 'setup_id', 'disposition',
                                             'quality_applicability', 'performance_claim_eligible'])
    writer.writeheader()
    for row in all_exclusions:
        writer.writerow({key: row.get(key) for key in writer.fieldnames})
    files['reports/generic_build_exclusions.csv'] = table.getvalue().encode('utf-8')
    escaped = html.escape(json.dumps({'generic': {k: v for k, v in report['generic'].items() if k not in {'exclusions', 'remaining_missing'}},
                                    'native': native_counts, 'quality': quality_counts}, indent=2))
    files['REPLAY_REPORT.html'] = (
        '<!doctype html><meta charset="utf-8"><title>v2.82 Read-only Replay</title>'
        '<style>body{max-width:75rem;margin:3rem auto;font:16px system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>Read-only Auswertungsprojektion</h1><p>' + html.escape(str(completion.get('label') or '')) + '</p>'
        '<p>Originalmessungen bleiben unverändert. Dieser Bericht gibt keinen Gesamtlauf frei.</p>'
        '<pre>' + escaped + '</pre><p><a href="reports/generic_build_exclusions.csv">Exakte Generic-Ausschlüsse</a></p>'
    ).encode('utf-8')
    for name, data in sources.items():
        # Retain original negative import/result states and source software.
        files['original/' + name] = data
        if (root / name).read_bytes() != data:
            raise ValueError('source_changed_during_replay:' + name)
    if out.exists():
        actual_names = {path.relative_to(out).as_posix() for path in out.rglob('*') if path.is_file()}
        if actual_names != set(files) or any((out / name).read_bytes() != data for name, data in files.items()):
            raise FileExistsError('replay_output_exists_with_different_content:' + str(out))
        return report
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + out.name + '-', dir=out.parent))
    try:
        for name, data in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        # Rename publishes a completed report directory; never replace another
        # populated output. Existing matching results were checked above.
        if out.exists():
            raise FileExistsError('replay_output_appeared_during_publication')
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--generic-only', action='store_true', help='Only Generic fixture scope; no Native/Quality preservation claim.')
    args = parser.parse_args()
    try:
        report = replay(args.run_root, args.out, generic_only=args.generic_only)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'Replay rejected: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': report['status'], 'generic_missing': report['generic']['missing_count'],
                      'generic_excluded': report['generic']['excluded_count'], 'native': report['native'],
                      'quality': report['quality'], 'output': str(args.out.resolve())}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
