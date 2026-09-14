#!/usr/bin/env python3
"""Read-only historical identity replay through production matrix/report code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onnx_splitpoint_tool.native_job_identity import complete_historical_identity
from onnx_splitpoint_tool.workflow.runner import _native_expected_matrix_status_v60y, _native_concise_summary_v60w


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root_fixture_provenance() -> tuple[dict, dict[str, bytes]]:
    """Verify shipped projections, not the omitted tensors or hardware run."""
    fixture = ROOT/'tests/fixtures/v27931_complete_set'
    try:
        manifest_path = fixture/'ROOT_PROJECTION_PROVENANCE.json'
        if manifest_path.is_symlink():
            raise ValueError('symlinked_provenance')
        provenance = json.loads(manifest_path.read_text())
        if (provenance.get('schema') != 'onnx-splitpoint/historical-root-projection-provenance'
                or provenance.get('schema_version') != 1 or provenance.get('synthetic') is not False):
            raise ValueError('invalid_provenance_schema')
        archive = provenance['source_archive']
        original_files = provenance['original_files']
        records = [archive, *original_files.values()]
        for record in records:
            digest = record['sha256']
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in '0123456789abcdef' for char in digest)
                    or type(record['size_bytes']) is not int or record['size_bytes'] <= 0):
                raise ValueError('invalid_original_source_record')
        if archive['path'] != 'complete_set_20260907_161614_debug_pack.zip':
            raise ValueError('wrong_source_archive')
        payloads = {}
        for name in ('original_identity_projection.json', 'original_energy_projection.json',
                     'original_source_profile.yaml', 'original_resolved_profile.yaml'):
            path = fixture/name
            record = provenance['fixtures'][name]
            if path.is_symlink() or not path.is_file():
                raise ValueError('required_fixture_missing_or_symlinked:' + name)
            content = path.read_bytes()
            if (hashlib.sha256(content).hexdigest() != record['sha256']
                    or len(content) != record['size_bytes']
                    or not record['source_files']
                    or any(source not in original_files for source in record['source_files'])):
                raise ValueError('fixture_source_binding_invalid:' + name)
            payloads[name] = content
        # Bind the pre-existing source annotations to the new complete manifest.
        identity_origin = json.loads((fixture/'PROVENANCE.json').read_text())
        if (identity_origin.get('synthetic') is not False
                or identity_origin.get('source_archive') != archive['path']
                or identity_origin.get('identity_projection_sha256')
                    != provenance['fixtures']['original_identity_projection.json']['sha256']):
            raise ValueError('identity_origin_invalid')
        for name, digest in identity_origin['original_files'].items():
            if original_files[name]['sha256'] != digest:
                raise ValueError('identity_origin_mismatch')
        energy_origin = json.loads(payloads['original_energy_projection.json'])['origin']
        if (energy_origin['path'] not in provenance['fixtures']['original_energy_projection.json']['source_files']
                or original_files[energy_origin['path']]['sha256'] != energy_origin['sha256']):
            raise ValueError('energy_origin_mismatch')
        return provenance, payloads
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError('historical_root_fixture_provenance_invalid:' + str(exc)) from exc


def _replay_output_guard(input_run: Path | None, output_dir: Path, *, fresh: bool = False) -> None:
    """The command may only create reports in a separate, empty destination."""
    resolved_output = output_dir.resolve()
    if input_run is not None:
        original = input_run.resolve()
        if resolved_output == original or original in resolved_output.parents:
            raise ValueError('historical_replay_output_must_be_outside_original_run')
    if fresh and (output_dir.is_symlink() or
                  (resolved_output.exists() and (not resolved_output.is_dir()
                                                or any(resolved_output.iterdir())))):
        raise ValueError('historical_replay_output_must_be_new_or_empty_directory')


def replay(input_run: Path | None, output_dir: Path) -> dict:
    _replay_output_guard(input_run, output_dir)
    output_dir = output_dir.resolve()
    if input_run is None:
        source = ROOT/'tests/fixtures/v27931_complete_set/original_identity_projection.json'
        provenance = json.loads((source.parent/'PROVENANCE.json').read_text())
        actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if (provenance.get('synthetic') is not False or not provenance.get('source_archive')
                or not provenance.get('original_files') or not provenance.get('excluded_payloads')
                or provenance.get('identity_projection_sha256') != actual_sha):
            raise ValueError('historical_identity_fixture_provenance_invalid')
        _root_fixture_provenance()
        data = json.loads(source.read_text())
        inputs = [source]
    else:
        input_run = input_run.resolve()
        inputs = [input_run/'reports/native_producer_stage.json', input_run/'reports/native_producer_summary.json']
        stage, summary = [json.loads(path.read_text()) for path in inputs]
        data = dict(stage, rows=summary['rows'])
    before = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    rows, expected = data['rows'], data['expected_native_rows']
    completed = [complete_historical_identity(row, expected) for row in rows]
    matrix = _native_expected_matrix_status_v60y(expected, rows, data.get('backend_results') or [])
    reports = output_dir/'reports'; reports.mkdir(parents=True, exist_ok=True)
    (reports/'native_producer_stage.json').write_text(json.dumps({'expected_native_rows':expected}))
    (reports/'native_producer_summary.json').write_text(json.dumps({'rows':completed}))
    _, concise = _native_concise_summary_v60w(reports, missing_expected_rows=matrix['missing_expected_rows'])
    after = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    report = {**matrix, 'concise_logical_row_count':len(concise),
              'historical_measurements_modified':False, 'original_inputs_byte_unchanged':before == after,
              'input_sha256':before, 'scope':'historical_identity_replay_no_new_hardware_measurement',
              'identity_completions':[row['identity_completion'] for row in completed if row.get('identity_completion')]}
    (output_dir/'complete_set_identity_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def replay_energy(input_run: Path | None, output_dir: Path) -> dict:
    """Project original accounting without changing metrics or admission seals."""
    from copy import deepcopy
    from onnx_splitpoint_tool.workflow.evidence_status import derive_native_evidence_status, project_native_evidence_status
    _replay_output_guard(input_run, output_dir)
    fixture_verification = None
    if input_run is None:
        fixture = ROOT/'tests/fixtures/v27931_complete_set'
        source = fixture/'original_energy_projection.json'
        provenance, payloads = _root_fixture_provenance()
        data = json.loads(payloads['original_energy_projection.json'])
        identity = json.loads(payloads['original_identity_projection.json'])
        inputs = [fixture/name for name in payloads]
        fixture_verification = {
            'status': 'verified_shipped_projection_hashes',
            'provenance_sha256': _sha256_file(fixture/'ROOT_PROJECTION_PROVENANCE.json'),
            'scope': provenance['verification_scope'],
        }
    else:
        input_run = input_run.resolve()
        source = input_run/'reports/native_energy_measurements/native_producer_energy_results.json'
        data = json.loads(source.read_text())
        stage = json.loads((input_run/'reports/native_producer_stage.json').read_text())
        summary = json.loads((input_run/'reports/native_producer_summary.json').read_text())
        identity = dict(stage, rows=summary['rows'])
        inputs = [source, input_run/'reports/native_producer_stage.json', input_run/'reports/native_producer_summary.json']
    before = {str(path): _sha256_file(path) for path in inputs}
    expected = identity['expected_native_rows']
    matrix = _native_expected_matrix_status_v60y(expected, identity['rows'], identity.get('backend_results') or [])
    plan = deepcopy(data['plan_payload'])
    for entry in plan.get('excluded_rows', []):
        nested = entry.get('row') if isinstance(entry.get('row'), dict) else entry
        completed = complete_historical_identity(nested, expected)
        if nested is entry:
            entry.update(completed)
        else:
            entry['row'] = completed
    evidence = derive_native_evidence_status(run_mode='standard', expected_matrix=matrix,
        validation_payload={"rows":identity["rows"]}, validation_requested=False, energy_requested=True,
        energy_plan_payload=plan, energy_results_payload={'rows':data['rows']})
    projection = project_native_evidence_status(evidence)
    report = {'scope':'historical_energy_accounting_only_no_new_measurement_or_claim',
        'original_source_sha256':before[str(source)],
        'input_sha256':before,
        'fixture_provenance_verification':fixture_verification,
        'original_inputs_byte_unchanged':before == {str(path): _sha256_file(path) for path in inputs},
        'measurement_rows':len(data['rows']),
        'valid_repetitions':evidence['energy']['accounting']['valid_repetition_count'],
        'energy_evidence':evidence, 'projection':projection,
        'metric_payload_sha256':hashlib.sha256(json.dumps([row['run']['energy_aggregate'] for row in data['rows']],sort_keys=True).encode()).hexdigest(),
        'fixture_scope':data.get('origin', {'type':'original_full_result_JSON'})}
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'complete_set_energy_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    _replay_output_guard(args.input_run, args.output_dir, fresh=True)
    result=replay(args.input_run,args.output_dir)
    energy=replay_energy(args.input_run,args.output_dir)
    print(json.dumps({key:result[key] for key in ('expected_row_count','present_expected_row_count','successful_expected_row_count','failed_expected_row_count','missing_expected_row_count','concise_logical_row_count','original_inputs_byte_unchanged')},indent=2))
    return 0 if result['original_inputs_byte_unchanged'] and result['missing_expected_row_count']==0 and energy['original_inputs_byte_unchanged'] and energy['projection']['energy_accounting']['status']=='consistent' else 2
if __name__=='__main__':
    raise SystemExit(main())
