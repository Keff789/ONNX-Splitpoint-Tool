#!/usr/bin/env python3
"""Install the exact v2.82 bundle, replay old evidence, and prepare a bounded scope.

Hardware executes only with --run-targeted. Every stage writes into one new
evidence directory; original runs, caches and selected boundaries are retained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from acceptance_process import run_stage

BUNDLE = Path(__file__).resolve().parent
ENERGY_IDENTITIES = {
    "hailo8_to_trt|yolo26m|b038|orin_nx_hailo8_01|hailo8|uint8_dequant_fp16",
    "hailo8_to_trt|yolo26s|b021|orin_nx_hailo8_01|hailo8|uint8_dequant_fp16",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def select_energy_checkpoints(source):
    found = {}
    for path in sorted((source / 'reports/native_energy_measurements/checkpoints/native_energy/rows').glob('*.json')):
        if path.is_symlink():
            raise ValueError('unsafe_checkpoint:' + str(path))
        payload = json.loads(path.read_text())
        identity = payload.get('identity')
        if identity in ENERGY_IDENTITIES:
            if identity in found:
                raise ValueError('duplicate_checkpoint_identity:' + identity)
            found[identity] = path
    if set(found) != ENERGY_IDENTITIES:
        raise ValueError('required_energy_checkpoint_missing:' + ','.join(sorted(ENERGY_IDENTITIES - set(found))))
    return [found[key] for key in sorted(found)]


def inspect_h10_capture(output):
    """A successful collector exit confirms capture, not endpoint validity."""
    paths = sorted(output.glob('*/collection_summary.json'))
    if not paths:
        return {'capture_pass': False, 'endpoint_pass': False,
                'regular_path_released': False, 'reason': 'h10_collection_summary_missing'}
    records = []
    for path in paths:
        if path.is_symlink():
            raise ValueError('unsafe_h10_summary:' + str(path))
        value = json.loads(path.read_text())
        cases = value.get('cases') or {}
        records.append({
            'summary': str(path),
            'capture_pass': value.get('collection_complete') is True and value.get('capture_pass') is True,
            'endpoint_pass': value.get('endpoint_pass') is True,
            'regular_path_released': value.get('regular_path_released') is True,
            'cases': cases,
        })
    return {'capture_pass': all(row['capture_pass'] for row in records),
            'endpoint_pass': all(row['endpoint_pass'] for row in records),
            'regular_path_released': all(row['regular_path_released'] for row in records),
            'capture_count_per_case': len(records), 'records': records,
            'scope': 'fixed4' if len(records) == 4 else 'one_bound_source_image' if len(records) == 1 else 'incomplete_fixed_image_set'}


def execute(args, output):
    summary = {'version': '2.82', 'status': 'INCOMPLETE', 'source_run': str(args.source_run),
               'stages': {}, 'hardware_requested': args.run_targeted, 'errors': [],
               'scientific_final_claim': False}
    environment = dict(os.environ, ONNX_SPLITPOINT_TOOL_DIR=str(args.tool),
                       DL=str(output / 'installation'), PYTHONDONTWRITEBYTECODE='1',
                       ORT_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1')
    (output / 'installation').mkdir()

    def stage(name, command, timeout):
        print('STAGE=' + name, flush=True)
        write_json(output / 'acceptance.json', summary)
        record = run_stage(command, args.tool if args.tool.is_dir() else BUNDLE,
                           output / (name + '.log'), environment, timeout_s=timeout)
        summary['stages'][name] = record
        write_json(output / 'acceptance.json', summary)
        if record.get('cancelled') and not record.get('timed_out'):
            raise KeyboardInterrupt()
        return record.get('returncode') == 0 and not record.get('timed_out')

    rc = 2
    try:
        if not args.skip_install:
            if not stage('install', ['bash', str(BUNDLE / 'install_v282_and_collect_acceptance.sh')], 7200):
                raise RuntimeError('installation_gate_failed')
        python = args.tool / '.venv/bin/python'
        if not python.is_file():
            raise ValueError('tool_venv_missing')
        if not args.source_run.is_dir():
            summary['errors'].append('original_run_not_available; software_install_only')
        else:
            replay_ok = stage('generic_replay', [str(python), '-I', '-B',
                str(args.tool / 'scripts/replay_generic_exclusions_v282.py'),
                '--run-root', str(args.source_run), '--out', str(output / 'generic_replay')], 900)
            replay_ok = stage('status_quality_replay', [str(python), '-I', '-B',
                str(args.tool / 'scripts/replay_status_quality_v282.py'),
                '--run-root', str(args.source_run), '--out', str(output / 'status_quality_replay')], 900) and replay_ok
            try:
                checkpoints = select_energy_checkpoints(args.source_run)
                command = [str(python), '-I', '-B', str(args.tool / 'scripts/replay_selected_energy_attempts_v282.py'),
                           '--run-root', str(args.source_run), '--results', str(args.source_run / 'reports/native_energy_measurements/native_producer_energy_results.json'),
                           '--output', str(output / 'energy_reimport.json')]
                for checkpoint in checkpoints:
                    command += ['--checkpoint', str(checkpoint)]
                replay_ok = stage('energy_reimport', command, 900) and replay_ok
            except ValueError as exc:
                summary['errors'].append(str(exc))
                replay_ok = False
            command = [str(python), '-I', '-B', str(args.tool / 'scripts/reference_workflow_gate_v282.py'),
                       '--source-run', str(args.source_run), '--output-root', str(output / 'targeted')]
            prepared = stage('targeted_scope', command, 120)
            if args.run_targeted and prepared:
                stage('targeted_workflows', command + ['--execute'], 64800)
                # Existing normal run receipt/logs locate the same remote artifacts.
                h10 = [str(python), '-I', '-B', str(args.tool / 'scripts/hailo10_yolo26_boundary_probe_v282.py'),
                       '--run-dir', str(args.source_run), '--output-dir', str(output / 'h10'),
                       '--ssh', args.h10_ssh, '--remote-python', args.h10_python]
                for root in args.artifact_root:
                    h10 += ['--artifact-root', str(root)]
                if getattr(args, 'h10_images_json', None):
                    h10 += ['--images-json', str(args.h10_images_json)]
                stage('h10_endpoint_evidence', h10, 900)
                summary['h10'] = inspect_h10_capture(output / 'h10')
            summary['original_replay_complete'] = replay_ok
        records = summary['stages'].values()
        passed = all(row.get('returncode') == 0 and not row.get('timed_out') for row in records) and not summary['errors']
        if passed and args.run_targeted and (not summary.get('h10', {}).get('endpoint_pass')
                                            or not summary.get('h10', {}).get('regular_path_released')):
            summary['status'] = 'PARTIAL_H10_ENDPOINTS_UNRELEASED'
            passed = False
        else:
            summary['status'] = ('SOFTWARE_AND_REPLAY_PASS_HARDWARE_NOT_RUN' if passed and not args.run_targeted
                                 else 'TARGETED_GATES_PASS' if passed else 'PARTIAL')
        rc = 0 if passed else 2
    except KeyboardInterrupt:
        summary['status'] = 'CANCELLED'
        rc = 130
    except Exception as exc:
        summary['errors'].append(type(exc).__name__ + ':' + str(exc))
    finally:
        summary['exit_code'] = rc
        write_json(output / 'acceptance.json', summary)
        archive = output.with_suffix('.zip')
        temporary = archive.with_suffix('.zip.part')
        try:
            with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as z:
                for path in sorted(output.rglob('*')):
                    relative = path.relative_to(output)
                    if path.is_file() and not path.is_symlink() and 'runs' not in relative.parts and path.suffix in {'.json', '.xml', '.log', '.txt', '.md', '.csv', '.yaml', '.zip'}:
                        z.write(path, relative.as_posix())
            with zipfile.ZipFile(temporary) as z:
                if z.testzip() is not None:
                    raise RuntimeError('evidence_archive_crc_failure')
            os.link(temporary, archive)
            temporary.unlink()
            print('EVIDENCE_ZIP=' + str(archive), flush=True)
        except Exception as exc:
            summary.update(status='EVIDENCE_EXPORT_FAILED', exit_code=2)
            summary['errors'].append(str(exc))
            write_json(output / 'acceptance.json', summary)
            rc = 2
        finally:
            temporary.unlink(missing_ok=True)
        print('V282_ACCEPTANCE=' + summary['status'], flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, default=Path.home() / 'ONNX-Splitpoint-Tool')
    parser.add_argument('--source-run', type=Path, default=Path.home() / 'Models/EvaluationRuns/completsetdev_20260913_123411')
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--skip-install', action='store_true')
    parser.add_argument('--run-targeted', action='store_true')
    parser.add_argument('--artifact-root', type=Path, action='append', default=[])
    parser.add_argument('--h10-ssh', default='nx@192.168.0.145')
    parser.add_argument('--h10-python', default='/home/nx/venvs/hailo10/bin/python')
    parser.add_argument('--h10-images-json', type=Path, help='Optional exact four development image IDs/paths; absent means one bound source image per case.')
    args = parser.parse_args()
    for name in ('tool', 'source_run', 'output_root'):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.source_run == args.output_root or args.source_run in args.output_root.parents:
        parser.error('output-root must be outside the original run')
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='v282_install_and_accept_', dir=args.output_root))
    print('REPORT_DIR=' + str(output), flush=True)
    return execute(args, output)


if __name__ == '__main__':
    raise SystemExit(main())
