#!/usr/bin/env python3
"""One-step, read-only historical Q5 admission via the normal debug exporter.

The existing run is never modified. This performs no inference, model build,
statistics recomputation or user-configuration update.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnx_splitpoint_tool.filesystem_admission import require_output_outside_source
from onnx_splitpoint_tool.workflow.debug_pack import (
    create_evaluation_debug_pack, discover_central_request_descriptors,
    is_management_reference_diagnostic,
)
from onnx_splitpoint_tool.workflow.zip_utils import require_safe_pack_source


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(run: Path, output_dir: Path, *, expected_requests=123,
           expected_request_bytes=35157419, expected_references=14,
           expected_completed=63):
    if run.is_symlink():
        raise ValueError('source_run_symlink')
    run = run.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().absolute()
    require_output_outside_source(run, output_dir / 'debug_pack.zip', operation='v2.80.4 historical debug acceptance')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = require_safe_pack_source(run / 'quality_management/central_quality_summary.json', run)
    payload = json.loads(summary.read_text())
    central = discover_central_request_descriptors(run)
    if central['failures'] or central['oversized'] or central['missing_source_members']:
        raise ValueError('source_request_contract_incomplete:' + json.dumps({k: central[k] for k in ('failures', 'oversized', 'missing_source_members')}))
    paths = [str(record['path']) for record in central['files']]
    references = sorted(p.relative_to(run).as_posix() for p in (run / 'quality_management/references').glob('*/*')
                        if is_management_reference_diagnostic(p.relative_to(run).as_posix()))
    expected = {'request_count': expected_requests, 'request_bytes': expected_request_bytes,
                'reference_diagnostics': expected_references, 'completed_results': expected_completed}
    observed = {'request_count': len(paths), 'request_bytes': central['total_size_bytes'],
                'reference_diagnostics': len(references),
                'completed_results': sum(row.get('status') == 'completed' for row in payload.get('results', []))}
    if observed != expected:
        raise ValueError('source_scope_mismatch:' + json.dumps({'expected': expected, 'observed': observed}))
    evidence_names = [*paths, *references, 'quality_management/central_quality_summary.json']
    before = {name: _sha(require_safe_pack_source(run / name, run)) for name in evidence_names}
    print('STAGE=normal_debug_export', flush=True)
    result = create_evaluation_debug_pack(run, output_dir / 'debug_pack.zip')
    with zipfile.ZipFile(result['out_zip']) as archive:
        manifest = json.loads(archive.read('debug_pack_manifest.json'))
        for name, expected_hash in before.items():
            actual = hashlib.sha256()
            with archive.open(name) as member:
                for block in iter(lambda: member.read(1024 * 1024), b''):
                    actual.update(block)
            if actual.hexdigest() != expected_hash:
                raise ValueError('archived_source_changed:' + name)
        for name, expected_hash in before.items():
            if _sha(require_safe_pack_source(run / name, run)) != expected_hash:
                raise ValueError('historical_source_changed:' + name)
        if archive.testzip() is not None:
            raise ValueError('archive_crc_failed')
    q = manifest['central_quality_replay_inputs']
    return {'status': 'PASS', 'schema': 'onnx-splitpoint/v2804-historical-debug-export-acceptance',
            'scope': 'original_Q5_requests_and_reference_diagnostics', 'source_run': str(run),
            'expected': expected, 'observed': observed, 'source_bytes_unchanged': True,
            'archived_bytes_identical': True, 'archive': result,
            'decoded_predictions': q.get('decoded_prediction_payloads', {}),
            'debug_complete': manifest.get('complete'),
            'hardware_execution': 'NOT_RUN', 'model_build': 'NOT_RUN',
            'statistics_recomputed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-requests', type=int, default=123)
    parser.add_argument('--expected-request-bytes', type=int, default=35157419)
    parser.add_argument('--expected-references', type=int, default=14)
    parser.add_argument('--expected-completed', type=int, default=63)
    args = parser.parse_args(argv)
    report = {'status': 'FAIL', 'hardware_execution': 'NOT_RUN', 'model_build': 'NOT_RUN'}
    try:
        report = verify(args.run, args.output_dir,
                        expected_requests=args.expected_requests, expected_request_bytes=args.expected_request_bytes,
                        expected_references=args.expected_references, expected_completed=args.expected_completed)
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}:{exc}'
    # Output remains outside the source even on a failed scope check.
    try:
        require_output_outside_source(args.run.expanduser().resolve(), args.output_dir / 'debug_export_acceptance.json',
                                      operation='v2.80.4 debug acceptance report')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / 'debug_export_acceptance.json'
        temporary = path.with_suffix('.json.part')
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)
    except Exception as exc:
        print(f'REPORT_WRITE_ERROR={type(exc).__name__}:{exc}', flush=True)
        return 2
    print('DEBUG_EXPORT_ACCEPTANCE=' + report['status'], flush=True)
    print('ACCEPTANCE_REPORT=' + str(path), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
