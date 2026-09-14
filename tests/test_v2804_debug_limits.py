"""AP1 inclusive boundaries and default complete decoded prediction export."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest
from onnx_splitpoint_tool.workflow import debug_pack as packs
from onnx_splitpoint_tool.workflow import debug_pack_policy as policy


def write(root, name, body):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body if isinstance(body, bytes) else json.dumps(body).encode())
    return p


def run_root(tmp_path):
    root = tmp_path / 'run'
    write(root, 'evaluation_workflow.log', b'completed diagnostics\n')
    write(root, 'artifact_index.json', {'artifacts': []})
    return root


def descriptor(root, index, size=128, body=None):
    name = f'models/m/benchmark_results/quality_inputs/task_quality_inputs/request{index}.json'
    if body is None:
        body = b'{"padding":"' + b'x' * (size - 14) + b'"}'
    path = write(root, name, body)
    return {'source_request': name, 'source_request_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def summary(root, rows):
    return write(root, 'quality_management/central_quality_summary.json', {'results': rows})


def export(tmp_path, root):
    result = packs.create_evaluation_debug_pack(root, tmp_path / 'pack.zip')
    with zipfile.ZipFile(result['out_zip']) as z:
        return json.loads(z.read('debug_pack_manifest.json')), set(z.namelist())


def test_t0101_shared_binary_limits_and_small_cap_independence(tmp_path):
    assert packs.CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES == policy.CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES == 512 * 1024**2
    assert packs.CENTRAL_DESCRIPTOR_MAX_FILE_BYTES == policy.CENTRAL_DESCRIPTOR_MAX_FILE_BYTES == 64 * 1024**2
    assert packs.EXACT_METADATA_MAX_FILE_BYTES == policy.STRUCTURED_RESULT_MAX_FILE_BYTES == 256 * 1024**2
    assert packs.INDEX_VALIDATION_MAX_TOTAL_BYTES >= packs.EXACT_METADATA_MAX_FILE_BYTES
    root = run_root(tmp_path)
    for name in ('artifact_index.json', 'models/m/validation/validation_summary.json',
                 'models/m/benchmark_results/normalized_results.json',
                 'models/m/benchmark_set/backend_artifact_decisions.json'):
        path = write(root, name, b'{}' + b' ' * 128)
        assert packs.should_include_debug_file(root, path, max_bytes=1)[0], name


@pytest.mark.parametrize('delta,accepted', [(-1, True), (0, True), (1, False)])
def test_t0104_request_file_inclusive_boundary(tmp_path, monkeypatch, delta, accepted):
    root = run_root(tmp_path)
    monkeypatch.setattr(packs, 'CENTRAL_DESCRIPTOR_MAX_FILE_BYTES', 128)
    summary(root, [descriptor(root, 1, 128 + delta)])
    found = packs.discover_central_request_descriptors(root)
    assert bool(found['oversized']) is not accepted
    if not accepted:
        assert found['oversized'][0]['reason'] == 'size_limit_exceeded'
        assert not found['failures']
        with pytest.raises(RuntimeError, match='size_limit_exceeded'):
            export(tmp_path, root)
        assert not (tmp_path / 'pack.zip').exists()


@pytest.mark.parametrize('total,accepted', [(255, False), (256, True), (257, True)])
def test_t0105_request_total_inclusive_and_path_dedup(tmp_path, monkeypatch, total, accepted):
    root = run_root(tmp_path)
    monkeypatch.setattr(packs, 'CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES', total)
    first, second = descriptor(root, 1), descriptor(root, 2)
    summary(root, [first, first, second])
    found = packs.discover_central_request_descriptors(root)
    assert len(found['files']) == 2 and found['total_size_bytes'] == 256
    assert (not found['oversized']) is accepted


def test_t0106_real_large_index_normal_export_process(tmp_path):
    root = run_root(tmp_path)
    body = b'{"artifacts":[],"padding":"' + b'x' * (32 * 1024**2 + 1) + b'"}'
    index = write(root, 'artifact_index.json', body)
    out = tmp_path / 'large.zip'
    code = 'from pathlib import Path;from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack;import sys;create_evaluation_debug_pack(Path(sys.argv[1]),Path(sys.argv[2]))'
    subprocess.run([sys.executable, '-B', '-c', code, str(root), str(out)], check=True, timeout=90)
    with zipfile.ZipFile(out) as z:
        assert z.read('artifact_index.json') == body
        manifest = json.loads(z.read('debug_pack_manifest.json'))
        assert manifest['management_cpu_reference_diagnostics']['index_validation']['status'] == 'verified'
        assert manifest['uncompressed_size_bytes'] > 32 * 1024**2
        assert out.stat().st_size < manifest['uncompressed_size_bytes']
    assert index.read_bytes() == body


def test_t0107_bad_json_distinct_from_oversize(tmp_path, monkeypatch):
    root = run_root(tmp_path)
    source = write(root, 'artifact_index.json', b'{')
    found = packs._management_reference_diagnostic_inventory(root, {'artifact_index.json': source}, max_bytes=1)
    assert found['index_validation']['status'] == 'invalid_json'
    monkeypatch.setattr(packs, 'EXACT_METADATA_MAX_FILE_BYTES', 1)
    source.write_bytes(b'{}')
    found = packs._management_reference_diagnostic_inventory(root, {'artifact_index.json': source}, max_bytes=1)
    assert found['index_validation']['status'] == 'size_limit_exceeded'
    assert found['index_validation']['limit_bytes'] == 1


def quality_inputs(root):
    candidate = write(root, 'models/m/benchmark_results/quality_inputs/task_quality_inputs/full_candidate.json', {'records': [{'image_id': 'a', 'detections': []}]})
    reference = write(root, 'quality_management/references/m/by_source_contract/id/canonical_cpu_reference.json', {'records': [{'image_id': 'a', 'detections': []}]})
    request = descriptor(root, 0, body=json.dumps({'candidate': {'path': candidate.name, 'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(), 'size_bytes': candidate.stat().st_size}}).encode())
    request['management_cpu_reference'] = {'reference_path': str(reference), 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(), 'reference_size_bytes': reference.stat().st_size}
    summary(root, [request])
    return candidate, reference


def test_t0110_declared_decoded_bodies_default_unreferenced_and_raw_excluded(tmp_path):
    root = run_root(tmp_path)
    candidate, reference = quality_inputs(root)
    for suffix in ('.hef', '.onnx', '.npy', '.npz', '.jpg', '.whl', '.so'):
        write(root, 'reports/raw' + suffix, b'NEVER EXPORT')
    ignored = write(root, 'quality_management/references/unrequested/canonical_cpu_reference.json', {'records': []})
    manifest, names = export(tmp_path, root)
    payloads = manifest['central_quality_replay_inputs']['decoded_prediction_payloads']
    assert payloads['complete'] and len(payloads['archived_members']) == 2
    for path in (candidate, reference):
        assert path.relative_to(root).as_posix() in names
    assert ignored.relative_to(root).as_posix() not in names
    assert not any(name.startswith('reports/raw') for name in names)


@pytest.mark.parametrize('problem', ['hash', 'symlink', 'outside', 'traversal'])
def test_t0108_unsafe_decoded_references_are_omitted(tmp_path, problem):
    root = run_root(tmp_path)
    candidate, reference = quality_inputs(root)
    request_path = root / 'models/m/benchmark_results/quality_inputs/task_quality_inputs/request0.json'
    data = json.loads(request_path.read_text())
    if problem == 'hash':
        candidate.write_bytes(b'{"records":[{"changed":true}]}')
    else:
        outside = write(tmp_path, 'private_candidate.json', {'records': [{'SECRET': True}]})
        if problem == 'symlink':
            candidate.unlink(); candidate.symlink_to(outside)
        else:
            data['candidate']['path'] = str(outside) if problem == 'outside' else '../../../../../../../../private_candidate.json'
            request_path.write_text(json.dumps(data))
            row = json.loads((root / 'quality_management/central_quality_summary.json').read_text())['results'][0]
            row['source_request_sha256'] = hashlib.sha256(request_path.read_bytes()).hexdigest()
            summary(root, [row])
    manifest, names = export(tmp_path, root)
    payloads = manifest['central_quality_replay_inputs']['decoded_prediction_payloads']
    assert not payloads['complete'] and payloads['omitted']
    assert candidate.relative_to(root).as_posix() not in names
    with zipfile.ZipFile(tmp_path / 'pack.zip') as z:
        assert not any(b'SECRET' in z.read(name) for name in z.namelist())


@pytest.mark.parametrize('error', [OSError(28, 'No space left on device'), KeyboardInterrupt()])
def test_t0109_failed_stream_preserves_prior_archive(tmp_path, monkeypatch, error):
    root = run_root(tmp_path)
    quality_inputs(root)
    destination = tmp_path / 'pack.zip'
    destination.write_bytes(b'previous complete archive')
    original = packs._write_source_member
    def broken(*args, **kwargs):
        if str(args[2]).endswith('full_candidate.json'):
            raise error
        return original(*args, **kwargs)
    monkeypatch.setattr(packs, '_write_source_member', broken)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        packs.create_evaluation_debug_pack(root, destination)
    assert destination.read_bytes() == b'previous complete archive'
    assert not list(tmp_path.glob('*.partial*'))


def test_t0108_duplicate_path_rechecks_each_declaration(tmp_path):
    root = run_root(tmp_path)
    row = descriptor(root, 1)
    summary(root, [row, {**row, 'source_request_sha256': 'f' * 64}])
    found = packs.discover_central_request_descriptors(root)
    assert found['total_size_bytes'] == 128
    assert found['files'][0]['result_indexes'] == [0, 1]
    assert not found['source_contract_ok']
    assert found['failures'] == [{'path': row['source_request'], 'result_index': 1, 'reason': 'declared_hash_mismatch'}]


@pytest.mark.parametrize('offset,accepted', [(-1, True), (0, True), (1, False)])
def test_t0106_structured_index_inclusive_boundary(tmp_path, monkeypatch, offset, accepted):
    root = run_root(tmp_path)
    monkeypatch.setattr(packs, 'EXACT_METADATA_MAX_FILE_BYTES', 64)
    path = write(root, 'artifact_index.json', b'{"artifacts":[]}' + b' ' * (48 + offset))
    assert path.stat().st_size == 64 + offset
    inventory = packs._management_reference_diagnostic_inventory(root, {'artifact_index.json': path}, max_bytes=1)
    assert (inventory['index_validation']['status'] == 'verified') is accepted
    assert packs.should_include_debug_file(root, path, max_bytes=1)[0] is accepted


def test_prediction_aggregate_is_separate_from_descriptor_budget(tmp_path, monkeypatch):
    root = run_root(tmp_path)
    candidate, reference = quality_inputs(root)
    monkeypatch.setattr(packs, 'CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES', 1024)
    needed = candidate.stat().st_size + reference.stat().st_size
    monkeypatch.setattr(packs, 'QUALITY_PREDICTION_MAX_TOTAL_BYTES', needed)
    central = packs.discover_central_request_descriptors(root)
    first = packs._discover_quality_prediction_payloads(root, central)
    assert first['total_size_bytes'] == needed and not first['omitted']
    monkeypatch.setattr(packs, 'QUALITY_PREDICTION_MAX_TOTAL_BYTES', needed - 1)
    second = packs._discover_quality_prediction_payloads(root, central)
    assert second['omitted'][0]['reason'] == 'size_limit_exceeded'
    assert second['omitted'][0]['limit_bytes'] == needed - 1


def test_t0109_partial_optional_member_eio_aborts_publication(tmp_path, monkeypatch):
    root = run_root(tmp_path)
    write(root, 'reports/notes.log', b'original complete log')
    out = tmp_path / 'pack.zip'
    out.write_bytes(b'previous complete ZIP')
    original = packs._write_source_member
    def broken(archive, source, relative, **kwargs):
        if relative == 'reports/notes.log':
            archive.writestr(relative, b'PARTIAL')
            raise OSError(5, 'Input/output error')
        return original(archive, source, relative, **kwargs)
    monkeypatch.setattr(packs, '_write_source_member', broken)
    with pytest.raises(RuntimeError, match='Input/output error'):
        packs.create_evaluation_debug_pack(root, out)
    assert out.read_bytes() == b'previous complete ZIP'


def test_t0102_t0103_synthetic123_request_payload_bytes_and14_diagnostics(tmp_path):
    # Explicitly synthetic shape/size regression; actual Q5 runs on the target.
    root = run_root(tmp_path)
    total = 35157419
    sizes = [total // 123] * 123
    sizes[-1] += total - sum(sizes)
    rows = [{**descriptor(root, i, size), 'status': 'completed' if i < 63 else 'cancelled'} for i, size in enumerate(sizes)]
    summary(root, rows)
    for i in range(7):
        write(root, f'quality_management/references/m{i}/management_cpu_reference_status.json', {'status': 'completed'})
        write(root, f'quality_management/references/m{i}/management_cpu_reference_stdout.txt', b'complete\n')
    import importlib.util
    script = Path(__file__).parents[1] / 'scripts/verify_debug_export_v2804.py'
    spec = importlib.util.spec_from_file_location('v2804_debug_verify', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify(root, tmp_path / 'verified')
    assert result['status'] == 'PASS' and result['observed']['request_bytes'] == total
    assert result['observed']['reference_diagnostics'] == 14
    assert result['source_bytes_unchanged'] and result['archived_bytes_identical']
    assert result['hardware_execution'] == 'NOT_RUN'
