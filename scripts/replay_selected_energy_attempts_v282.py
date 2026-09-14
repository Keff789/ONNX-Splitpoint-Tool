#!/usr/bin/env python3
"""Reimport immutable, checkpoint-bound native energy results without hardware.

Usage:
  python scripts/replay_selected_energy_attempts_v282.py \
    --run-root /path/to/original_run --checkpoint /path/to/row.json \
    --output /path/outside/original_run/energy_reimport_v282.json

For one archived job only, add both --original-measurement-root OLD and
--extracted-measurement-root NEW. No source JSON or timestamps are rewritten.
This utility reads checkpoints and aggregates; it does not initialize u.RECS,
run SSH/compiler commands, or recalculate a trace.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from scripts.run_native_producer_energy_from_summary import (
        _reimport_energy_checkpoint, _strict_json_text,
    )
except ImportError:
    from run_native_producer_energy_from_summary import (
        _reimport_energy_checkpoint, _strict_json_text,
    )


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise FileExistsError('output already contains a different result: ' + str(path))
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.energy-reimport-', delete=False) as f:
            temporary = Path(f.name)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            # Atomic create-if-absent, unlike replace which overwrites a race.
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _project_existing_energy_report(
    original: dict[str, Any], records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace only explicitly verified rows in a derived import projection."""
    rows = original.get('rows')
    if not isinstance(rows, list):
        raise ValueError('original report rows missing')
    projected = copy.deepcopy(rows)
    updated = []
    for record in records:
        if record.get('ok') is not True:
            continue
        index = record.get('source_row_index')
        if type(index) is not int or index < 0 or index >= len(rows) or index in updated:
            raise ValueError('checkpoint row index missing, duplicate or outside original report')
        before = rows[index]
        if not isinstance(before, dict) or not isinstance(before.get('row'), dict):
            raise ValueError('original report row invalid')
        for key in ('backend', 'model', 'case', 'setup_id', 'measurement_output_dir', 'native_command_contract'):
            if before['row'].get(key) != record['row'].get(key):
                raise ValueError('original report/checkpoint identity conflict: ' + key)
        if before.get('run', {}).get('energy_aggregate_sha256') != record['run'].get('energy_aggregate_sha256'):
            raise ValueError('original report/checkpoint aggregate digest conflict')
        after = copy.deepcopy(before)
        after['run'].update(copy.deepcopy(record['run']))
        after.update(ok=True, derived_read_only_reimport=True,
                     original_import_decision=copy.deepcopy(record['original_import_decision']),
                     source_checkpoint=record.get('source_checkpoint'))
        projected[index] = after
        updated.append(index)
    successful = lambda values: sum(isinstance(item, dict) and item.get('run', {}).get('energy_aggregate_verified') is True for item in values)
    return {
        'source_report_state': {key: original.get(key) for key in ('state', 'status', 'ok', 'started_measurement_count')},
        'measurement_order_count': len(rows),
        'original_verified_import_count': successful(rows),
        'projected_verified_import_count': successful(projected),
        'unchanged_row_count': len(rows) - len(updated),
        'updated_row_indices': sorted(updated),
        'new_measurement_count': 0, 'claim_eligible': False,
        'rows': projected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, help='optional original native_producer_energy_results.json for a full derived row projection')
    parser.add_argument('--original-measurement-root', type=Path)
    parser.add_argument('--extracted-measurement-root', type=Path)
    args = parser.parse_args(argv)
    root = args.run_root.resolve(strict=True)
    output = args.output.absolute()
    if output.resolve().is_relative_to(root):
        parser.error('--output must be outside the original/extracted run')
    if bool(args.original_measurement_root) != bool(args.extracted_measurement_root):
        parser.error('archive projection requires both original and extracted measurement roots')
    mapping = None
    if args.original_measurement_root:
        if len(args.checkpoint) != 1:
            parser.error('one explicit measurement-root mapping requires exactly one checkpoint')
        extracted = args.extracted_measurement_root.absolute()
        if not extracted.resolve().is_relative_to(root):
            parser.error('extracted measurement root must be inside --run-root')
        mapping = (args.original_measurement_root, extracted)
    records = []
    seen = set()
    for requested in args.checkpoint:
        path = requested.absolute()
        if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
            parser.error('checkpoint must be an existing nonsymlink file inside --run-root')
        if str(path) in seen:
            parser.error('duplicate checkpoint selection: ' + str(path))
        seen.add(str(path))
        raw = path.read_bytes()
        try:
            checkpoint = _strict_json_text(raw.decode('utf-8'), label=str(path))
            if not isinstance(checkpoint, dict):
                raise ValueError('checkpoint is not an object')
            execution = checkpoint.get('execution') or {}
            measurement = Path(str(execution.get('output_dir') or ''))
            if mapping is None and (not measurement.is_absolute() or not measurement.resolve().is_relative_to(root)):
                raise ValueError('bound measurement root is outside --run-root; explicit archive mapping required')
            derived = _reimport_energy_checkpoint(checkpoint, measurement_path_mapping=mapping)
            records.append({'source_checkpoint': str(path), 'source_row_index': checkpoint.get('index'), **derived})
        except (ValueError, TypeError, OSError) as exc:
            records.append({'source_checkpoint': str(path), 'ok': False,
                            'reimport_error': str(exc), 'new_measurement_started': False})
        if path.read_bytes() != raw:
            raise RuntimeError('source checkpoint changed while reading: ' + str(path))
    payload = {
        'schema': 'onnx-splitpoint/selected-energy-attempt-reimport',
        'schema_version': 1, 'version': '2.82',
        'source_run_root': str(root), 'read_only_sources': True,
        'measurement_scope': 'screening_1s', 'new_measurement_count': 0,
        'claim_eligible': False, 'selected_checkpoint_count': len(records),
        'verified_import_count': sum(item.get('ok') is True for item in records),
        'rows': records,
    }
    if args.results is not None:
        source = args.results.absolute()
        if source.resolve() != source or not source.is_relative_to(root) or not source.is_file():
            parser.error('--results must be a nonsymlink original report inside --run-root')
        raw = source.read_bytes()
        original = _strict_json_text(raw.decode('utf-8'), label=str(source))
        if not isinstance(original, dict):
            parser.error('original results must be an object')
        payload['projection'] = _project_existing_energy_report(original, records)
        payload['source_results'] = str(source)
        if source.read_bytes() != raw:
            raise RuntimeError('source results changed while reading')
    _write_once(output, payload)
    print(json.dumps({'output': str(output), 'verified': payload['verified_import_count'],
                      'selected': len(records), 'new_measurements': 0}))
    return 0 if all(item.get('ok') is True for item in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
