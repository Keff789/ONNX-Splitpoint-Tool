"""AP3 actual ZIP export: bounded originals and explicitly derived summaries."""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from onnx_splitpoint_tool.workflow import debug_pack as packs


def _write(run, relative, value):
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def _run(tmp_path, refs=0):
    run = tmp_path / 'historical_run'
    _write(run, 'evaluation_workflow.log', b'historical source remains unchanged\n')
    for i in range(refs):
        _write(run, f'quality_management/references/m{i}/management_cpu_reference_status.json', {'status': 'failed', 'error': 'model_binding', 'tool_version': '2.80.1'})
        _write(run, f'quality_management/references/m{i}/management_cpu_reference_stdout.txt', b'original failure\n')
    return run


def _pack(tmp_path, run):
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'pack.zip')
    with zipfile.ZipFile(result['out_zip']) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        members = {name: z.read(name) for name in z.namelist()}
    return json.loads(members['debug_pack_manifest.json']), members


def test_t0302_large_index_now_checked_and_fourteen_originals_retained(tmp_path):
    run = _run(tmp_path, refs=7)
    index = _write(run, 'artifact_index.json', b'{"artifacts":[],"padding":"' + b'x' * (32 * 1024 * 1024) + b'"}')
    before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    manifest, members = _pack(tmp_path, run)
    refs = manifest['management_cpu_reference_diagnostics']
    assert not any('invalid' in f['reason'] for f in refs['reference_failures'])
    validation = refs['index_validation']
    assert validation['status'] == 'verified'
    assert validation['observed_size_bytes'] == index.stat().st_size
    assert validation['index_coverage_verified'] is True
    assert refs['all_admitted_members_archived'] and len(refs['archived_members']) == 14
    assert refs['complete'] and manifest['complete']
    assert not any('invalid' in f['reason'] for f in refs['reference_failures'])
    for name in refs['archived_members']:
        assert members[name] == before[name]
    assert {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('body,status', [
    (b'{"artifacts":[]}', 'verified'), (b'{', 'invalid_json'),
    (b'{"artifacts":{}}', 'invalid_schema'), (b'[]', 'invalid_schema'),
    (b'{"artifacts":[{}]}', 'invalid_schema'), (None, 'missing'),
])
def test_t0301_t0303_t0304_index_states_remain_distinct(tmp_path, body, status):
    run = _run(tmp_path, refs=1)
    if body is not None:
        _write(run, 'artifact_index.json', body)
    manifest, _ = _pack(tmp_path, run)
    inventory = manifest['management_cpu_reference_diagnostics']
    assert inventory['index_validation']['status'] == status
    assert inventory['complete'] is (status == 'verified')
    assert inventory['all_admitted_members_archived']


def test_t0303_large_damaged_index_not_parsed(tmp_path, monkeypatch):
    run = _run(tmp_path, refs=1)
    _write(run, 'artifact_index.json', b'broken' * 20)
    monkeypatch.setattr(packs, 'EXACT_METADATA_MAX_FILE_BYTES', 100)
    manifest, _ = _pack(tmp_path, run)
    validation = manifest['management_cpu_reference_diagnostics']['index_validation']
    assert validation['status'] == 'size_limit_exceeded' and 'error' not in validation


def test_t0304_unreadable_index_and_total_budget_are_not_json_corruption(tmp_path, monkeypatch):
    run = _run(tmp_path, refs=1)
    index = _write(run, 'artifact_index.json', {'artifacts': []})
    real_open = Path.open
    def unreadable(path, *args, **kwargs):
        if path == index:
            raise PermissionError('controlled read failure')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', unreadable)
    manifest, _ = _pack(tmp_path, run)
    assert manifest['management_cpu_reference_diagnostics']['index_validation']['status'] == 'unreadable'
    monkeypatch.setattr(Path, 'open', real_open)
    monkeypatch.setattr(packs, 'INDEX_VALIDATION_MAX_TOTAL_BYTES', 1)
    manifest, _ = _pack(tmp_path, run)
    assert manifest['management_cpu_reference_diagnostics']['index_validation']['status'] == 'not_checked_budget_limit'


def test_t0305_index_symlink_external_and_traversal_references_never_read(tmp_path):
    run = _run(tmp_path, refs=1)
    outside = _write(tmp_path, 'private.json', b'EXTERNAL CONTENT')
    (run / 'artifact_index.json').symlink_to(outside)
    manifest, members = _pack(tmp_path, run)
    assert manifest['management_cpu_reference_diagnostics']['index_validation']['status'] == 'unsafe_path'
    assert not any(b'EXTERNAL CONTENT' in data for data in members.values())
    (run / 'artifact_index.json').unlink()
    bad = ['../quality_management/references/m/management_cpu_reference_status.json',
           '/outside/quality_management/references/m/management_cpu_reference_status.json',
           'quality_management/references/m/deeper/management_cpu_reference_status.json']
    _write(run, 'artifact_index.json', {'artifacts': [{'path': p} for p in bad]})
    manifest, _ = _pack(tmp_path, run)
    inventory = manifest['management_cpu_reference_diagnostics']
    assert len(inventory['reference_failures']) == 3
    assert not inventory['index_validation']['index_coverage_verified']
    assert not inventory['complete']


def test_t0306_index_missing_registered_source_prevents_complete(tmp_path):
    run = _run(tmp_path, refs=1)
    missing = 'quality_management/references/missing/management_cpu_reference_stdout.txt'
    _write(run, 'artifact_index.json', {'artifacts': [{'path': missing}]})
    manifest, _ = _pack(tmp_path, run)
    inventory = manifest['management_cpu_reference_diagnostics']
    assert inventory['index_validation']['status'] == 'verified'
    assert inventory['missing_source_members'] == [missing]
    assert not inventory['complete'] and inventory['all_admitted_members_archived']
    assert inventory['all_discovered_members_archived']


def _result(index=0, padding=0):
    return {'tool_version': '2.80.1', 'run_id': 'original_night', 'results': [{
        'model_id': 'm0', 'case_id': 'b064', 'setup_id': 'setup_A',
        'backend': 'tensorrt', 'precision': 'fp16', 'status': 'failed',
        'primary_failure_reason': 'quality_rejected', 'failure_stage': 'quality',
        'fps_makespan': 12.5 + index,
        'predictions': 'RAW_SECRET_' + 'x' * padding,
    }]}


def _source(run, index=0, padding=0):
    return _write(run, f'models/m0/benchmark_results/benchmark_results_{index}.json', _result(index, padding))


def _snapshot(run):
    return {p.relative_to(run).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in run.rglob('*') if p.is_file() and not p.is_symlink()}


def test_t0307_t0314_196_sources_43_large_have_export_only_views(tmp_path, monkeypatch):
    # Preserve projection regression with an explicit smaller original budget.
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 2 * 1024 * 1024)
    run = _run(tmp_path)
    for i in range(196):
        _source(run, i, 2 * 1024 * 1024 if i < 43 else 0)
    before = _snapshot(run)
    manifest, members = _pack(tmp_path, run)
    runtime = manifest['runtime_execution_diagnostics']
    assert len(runtime['expected_members']) == 196
    assert runtime['original_archived_count'] == 153
    assert runtime['derived_summary_count'] == 43
    assert len(runtime['source_coverage']) == 196
    assert runtime['compact_view_complete_for_discovered_sources']
    assert not runtime['complete']
    for row in runtime['source_coverage']:
        assert row['original_archived'] != row['derived_summary_archived']
        if row['derived_summary_archived']:
            assert row['summary_origin'] == 'export_only_fallback'
            body = members[row['summary_path']]
            assert b'RAW_SECRET' not in body
            assert json.loads(body)['status'] == 'projected'
    assert _snapshot(run) == before
    assert not (run / 'models/m0/benchmark_results/diagnostic_summaries').exists()


@pytest.mark.parametrize('kind,reason', [('large', 'size_limit'), ('invalid', 'invalid_json'),
                                         ('unknown', 'schema'), ('missing', 'source_missing'),
                                         ('unsafe', 'source_unsafe')])
def test_t0311_t0314_unavailable_sources_are_explicit(tmp_path, monkeypatch, kind, reason):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 64)
    run = _run(tmp_path)
    relative = 'models/m0/benchmark_results/benchmark_results_0.json'
    source = _source(run, padding=400)
    monkeypatch.setattr(packs, 'DEFAULT_MAX_SMALL_FILE_BYTES', 64)
    if kind == 'large':
        from onnx_splitpoint_tool.workflow import compact_runtime_diagnostics as compact
        monkeypatch.setattr(compact, 'FALLBACK_PARSE_MAX_BYTES', 100)
    elif kind == 'invalid':
        source.write_bytes(b'{' + b'x' * 400)
    elif kind == 'unknown':
        source.write_text(json.dumps({'unknown': 'x' * 400}))
    elif kind in {'missing', 'unsafe'}:
        source.unlink()
        _write(run, 'debug_pack_manifest.json', {'runtime_execution_diagnostics': {'expected_members': [relative]}})
        if kind == 'unsafe':
            source.symlink_to(_write(tmp_path, 'outside.json', b'PRIVATE EXTERNAL CONTENT'))
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'pack.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('debug_pack_manifest.json'))
        assert not any(b'PRIVATE EXTERNAL CONTENT' in z.read(name) for name in z.namelist())
    row = manifest['runtime_execution_diagnostics']['source_coverage'][0]
    assert not row['derived_summary_archived'] and reason in row['summary_reason']
    assert not manifest['runtime_execution_diagnostics']['compact_view_complete_for_discovered_sources']


def test_t0313_single_and_total_summary_and_fallback_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 64)
    from onnx_splitpoint_tool.workflow import compact_runtime_diagnostics as compact
    run = _run(tmp_path)
    _source(run, 0, 400)
    _source(run, 1, 400)
    monkeypatch.setattr(compact, 'SUMMARY_TOTAL_MAX_BYTES', 1)
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'total.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        assert runtime['derived_summary_count'] == 0
        assert all('summary_total_budget_limit' in row['summary_reason'] for row in runtime['source_coverage'])
        assert all(row['identity_failure_overview'][0]['primary_failure_reason'] == 'quality_rejected'
                   for row in runtime['source_coverage'])
        assert all(row['identity_failure_overview_complete'] for row in runtime['source_coverage'])
    monkeypatch.setattr(compact, 'SUMMARY_TOTAL_MAX_BYTES', 16 * 1024 * 1024)
    monkeypatch.setattr(packs, 'RUNTIME_FALLBACK_PARSE_MAX_TOTAL_BYTES', 1)
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'parse.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        assert all('fallback_not_checked_budget_limit' in row['summary_reason'] for row in runtime['source_coverage'])
    monkeypatch.setattr(packs, 'RUNTIME_FALLBACK_PARSE_MAX_TOTAL_BYTES', 128 * 1024 * 1024)
    monkeypatch.setattr(compact, 'SUMMARY_MAX_BYTES', 1)
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'single.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        assert runtime['derived_summary_count'] == 0
        for row in runtime['source_coverage']:
            assert row['summary_status'] == 'summary_unavailable'


def test_t0315_declared_hash_not_reverified_and_drift_rejects_companion(tmp_path, monkeypatch):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 64)
    from onnx_splitpoint_tool.workflow import compact_runtime_diagnostics as compact
    run = _run(tmp_path)
    source = _source(run, padding=400)
    relative = source.relative_to(run).as_posix()
    _write(run, 'artifact_index.json', {'artifacts': [{'path': relative, 'sha256': 'sha256:' + 'a' * 64}]})
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'fallback.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        assert runtime['source_coverage'][0]['declared_original_sha256_verification'] == 'declared_not_reverified'
    projection = compact.project_runtime_payload(_result(padding=400), source_path=relative,
                                                 source_stat=compact.source_observation(source))
    companion = compact.companion_path(source)
    _write(run, companion.relative_to(run).as_posix(), compact.summary_bytes(projection))
    source.write_bytes(source.read_bytes() + b' ')
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'drift.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        assert 'companion_unconfirmed' in runtime['source_coverage'][0]['summary_reason']
        assert runtime['derived_summary_count'] == 0


@pytest.mark.parametrize('failure', ['write', 'interrupt', 'source_mutation'])
def test_t0316_atomic_summary_publication_failure_has_no_final_archive(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 64)
    run = _run(tmp_path)
    source = _source(run, padding=400)
    real_write = zipfile.ZipFile.writestr
    def fail(z, name, *args, **kwargs):
        if str(name).endswith('.summary.json'):
            if failure == 'write':
                raise OSError('controlled summary ZIP failure')
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            source.write_bytes(source.read_bytes() + b' ')
        return real_write(z, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'writestr', fail)
    with pytest.raises((OSError, RuntimeError, KeyboardInterrupt)):
        packs.create_evaluation_debug_pack(run, tmp_path / 'failed.zip', max_small_file_bytes=64)
    assert not (tmp_path / 'failed.zip').exists()
    assert not (tmp_path / 'failed.zip.manifest.json').exists()
    assert not list(tmp_path.glob('*.partial*'))


@pytest.mark.parametrize('tamper', ['none', 'source_path', 'unsafe_link', 'raw_body'])
def test_t0305_t0312_t0315_matching_companions_require_safe_source_binding(tmp_path, tamper, monkeypatch):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 64)
    from onnx_splitpoint_tool.workflow import compact_runtime_diagnostics as compact
    run = _run(tmp_path)
    source = _source(run, padding=400)
    relative = source.relative_to(run).as_posix()
    projection = compact.project_runtime_payload(_result(padding=400), source_path=relative,
                                                 source_stat=compact.source_observation(source))
    companion = compact.companion_path(source)
    if tamper == 'source_path':
        projection['source']['path'] = 'models/other/benchmark_results/benchmark_results_0.json'
    if tamper == 'raw_body':
        projection['rows'][0]['predictions'] = 'PRIVATE RAW COMPANION BODY'
    _write(run, companion.relative_to(run).as_posix(), compact.summary_bytes(projection))
    if tamper == 'unsafe_link':
        companion.unlink()
        companion.symlink_to(_write(tmp_path, 'private_companion.json', compact.summary_bytes(projection)))
    before = _snapshot(run)
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'companion.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        assert z.testzip() is None
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        assert not any(b'PRIVATE RAW COMPANION BODY' in z.read(n) for n in z.namelist())
        assert runtime['derived_summary_count'] == int(tamper == 'none')
        if tamper == 'none':
            row = runtime['source_coverage'][0]
            assert row['summary_origin'] == 'existing_companion'
            assert row['derived_summary_archived']
            assert json.loads(z.read(row['summary_path'])) == projection
        else:
            assert runtime['source_coverage'][0]['summary_status'] == 'summary_unavailable'
    assert _snapshot(run) == before


def test_t0316_closed_archive_crc_failure_does_not_publish(tmp_path, monkeypatch):
    run = _run(tmp_path)
    _source(run, padding=400)
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda z: 'damaged_member.json')
    with pytest.raises(zipfile.BadZipFile):
        packs.create_evaluation_debug_pack(run, tmp_path / 'crc.zip', max_small_file_bytes=64)
    assert not (tmp_path / 'crc.zip').exists()
    assert not (tmp_path / 'crc.zip.manifest.json').exists()


def test_t0304_t0314_unreadable_original_fallback_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 64)
    run = _run(tmp_path)
    source = _source(run, padding=400)
    real_open = Path.open
    def fail(path, *args, **kwargs):
        if path == source:
            raise PermissionError('controlled source read failure')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail)
    result = packs.create_evaluation_debug_pack(run, tmp_path / 'unreadable.zip', max_small_file_bytes=64)
    with zipfile.ZipFile(result['out_zip']) as z:
        runtime = json.loads(z.read('debug_pack_manifest.json'))['runtime_execution_diagnostics']
        row = runtime['source_coverage'][0]
        assert row['status'] == 'source_unreadable'
        assert row['summary_reason'] == 'source_unreadable:PermissionError'
        assert runtime['derived_summary_count'] == 0
