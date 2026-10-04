"""Separate real output locations use the existing energy collector."""
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.yolo_augmentation_reporting import _energy


def block(root: Path, case: str) -> Path:
    output = root / 'reports/native_energy_measurements/native_producer_energy_results.json'
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({'rows': [{'row': {
        'model': 'yolov7_paper', 'case': case, 'setup_id': 'orin_nx_hailo8_01',
        'backend': 'hailo8_to_trt'}, 'ok': False}]}))
    return root


def test_separate_blocks_keep_original_sources_and_no_false_measurement(tmp_path):
    roots = [block(tmp_path / 'block1', 'b044'), block(tmp_path / 'block2', 'b066')]
    originals = {p: p.read_bytes() for root in roots for p in root.rglob('*.json')}
    rows, raw, source, status = _energy(roots)
    assert status == 'normal_native_energy_collector_separate_blocks'
    assert {row['case'] for row in rows} == {'b044', 'b066'}
    assert all(row['ok'] is False and row['energy_repeat_valid_n'] == 0 for row in rows)
    assert json.loads(source) == [item['source'] for item in raw['blocks']]
    assert all(Path(item['source']).read_bytes() == originals[Path(item['source'])] for item in raw['blocks'])
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_duplicate_block_identity_is_not_double_counted(tmp_path):
    roots = [block(tmp_path / 'block1', 'b066'), block(tmp_path / 'block2', 'b066')]
    with pytest.raises(ValueError, match='unique complete'):
        _energy(roots)
    with pytest.raises(ValueError, match='duplicate augmentation energy block'):
        _energy([roots[0], roots[0]])


def test_explicit_missing_block_is_not_reported_as_completed(tmp_path):
    with pytest.raises(ValueError, match='aggregate missing'):
        _energy([tmp_path / 'missing'])
