"""Resume the proven augmentation nonstart without inventing Native reports."""
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.artifacts import sha256_file
from onnx_splitpoint_tool.workflow.supplement import _preserved_inventory, preserve_metadata
from onnx_splitpoint_tool.workflow.yolo_augmentation_reentry import (
    COORDINATOR, PERFORMANCE, UNPUBLISHED_NATIVE_BASELINE, unpublished_native_baseline,
)
from tests.test_yolo_quality_release_reentry import coordinator, quality_nonstart


PROOF_PATHS = (COORDINATOR.as_posix(), PERFORMANCE.as_posix(),
              'reports/native_producer_stage.json', 'reports/native_producer_stage_config.json')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def archived_nonstart(quality_nonstart):
    run, _cfg, _config, plan, *_ = quality_nonstart
    _write(run/'reports/scientific/performance_results.json', [])
    _write(run/'quality_management/central_quality_summary.json', {'results': []})
    paths = (*PROOF_PATHS, 'reports/scientific/performance_results.json',
             'quality_management/central_quality_summary.json')
    index = {raw: {'path': raw, 'sha256': sha256_file(run/raw),
                  'size_bytes': (run/raw).stat().st_size} for raw in paths}
    _write(run/'artifact_index.json', {'artifacts': list(index.values())})
    original = preserve_metadata(run, 'first-failed-quality-source-attempt')
    return run, original, index, plan


def test_bound_never_published_native_basis_keeps_first_source_and_all_files(archived_nonstart):
    run, original, index, _plan = archived_nonstart
    paths = [p for p in original.rglob('*') if p.is_file()]
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert unpublished_native_baseline(run, original, index) == UNPUBLISHED_NATIVE_BASELINE
    _preserved_inventory(run, original, [], [], {}, index)
    _preserved_inventory(run, original, [], [], {}, index)
    assert {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths} == before
    assert all(not (root/raw).exists() for root in (run, original) for raw in UNPUBLISHED_NATIVE_BASELINE)


@pytest.mark.parametrize('raw', PROOF_PATHS)
@pytest.mark.parametrize('damage', ['missing_index', 'changed_bytes', 'missing_file'])
def test_every_nonstart_proof_remains_original_index_bound(archived_nonstart, raw, damage):
    run, original, index, _plan = archived_nonstart
    if damage == 'missing_index':
        del index[raw]
    elif damage == 'changed_bytes':
        with (original/raw).open('a') as stream:
            stream.write(' ')
    else:
        (original/raw).unlink()
    with pytest.raises(ValueError):
        _preserved_inventory(run, original, [], [], {}, index)


@pytest.mark.parametrize('raw', sorted(UNPUBLISHED_NATIVE_BASELINE))
def test_previously_indexed_native_report_cannot_be_erased(archived_nonstart, raw):
    run, original, index, _plan = archived_nonstart
    index[raw] = {'path': raw, 'size_bytes': 0, 'sha256': 'e'*64}
    with pytest.raises(ValueError, match='native_basis_previously_bound'):
        _preserved_inventory(run, original, [], [], {}, index)


@pytest.mark.parametrize('raw', sorted(UNPUBLISHED_NATIVE_BASELINE))
@pytest.mark.parametrize('namespace', ['original', 'current'])
@pytest.mark.parametrize('form', ['empty_report', 'dangling_symlink'])
def test_partial_publication_or_symlink_is_not_a_proven_absent_basis(
        archived_nonstart, raw, namespace, form):
    run, original, index, _plan = archived_nonstart
    path = (original if namespace == 'original' else run)/raw
    if form == 'empty_report':
        _write(path, {'rows': []})
    else:
        path.symlink_to(path.parent/'missing-target')
    with pytest.raises(ValueError, match='native_basis_not_absent'):
        _preserved_inventory(run, original, [], [], {}, index)


@pytest.mark.parametrize('damage', ['native_raw', 'native_leaf', 'indexed_native_raw',
                                   'wrong_current_run', 'changed_plan', 'foreign_original'])
def test_current_or_historical_started_native_and_scope_drift_stay_blocked(archived_nonstart, damage):
    run, original, index, plan = archived_nonstart
    if damage == 'native_raw':
        _write(run/'native_producers/h8/native_fifo_results.json', {'ok': False, 'repetition_count_attempted': 1})
    elif damage == 'native_leaf':
        _write(run/'reports/native_producer_variants/case/performance_stage_result.json', {'state': 'running'})
    elif damage == 'indexed_native_raw':
        index['native_producers/h8/native_fifo_results.json'] = {'sha256': 'e'*64}
    elif damage == 'wrong_current_run':
        payload = json.loads((run/'run_manifest.json').read_text())
        payload['run_id'] = 'another-run'
        _write(run/'run_manifest.json', payload)
    elif damage == 'changed_plan':
        payload = json.loads(plan.read_text()); payload['changed'] = True; _write(plan, payload)
    else:
        old = original
        original = run.parent/'foreign-original'
        import shutil
        shutil.copytree(old, original)
    with pytest.raises(ValueError):
        _preserved_inventory(run, original, [], [], {}, index)


@pytest.mark.parametrize('raw', ['reports/scientific/performance_results.json',
                                'quality_management/central_quality_summary.json'])
def test_non_native_inventory_is_still_strict(archived_nonstart, raw):
    run, original, index, _plan = archived_nonstart
    (original/raw).unlink()
    with pytest.raises(ValueError, match='supplement_original_evidence_missing'):
        _preserved_inventory(run, original, [], [], {}, index)


def test_ordinary_supplement_missing_native_basis_still_fails(tmp_path):
    run = tmp_path/'ordinary-run'; original = run/'jobs/supplement_attempts/a/original'
    for raw, value in [('reports/scientific/performance_results.json', []),
                       ('quality_management/central_quality_summary.json', {'results': []})]:
        _write(original/raw, value)
    index = {raw: {'path': raw, 'sha256': sha256_file(original/raw),
                  'size_bytes': (original/raw).stat().st_size}
             for raw in ('reports/scientific/performance_results.json',
                         'quality_management/central_quality_summary.json')}
    with pytest.raises(ValueError, match='supplement_original_evidence_missing:reports/native_expected_matrix.json'):
        _preserved_inventory(run, original, [], [], {}, index)
