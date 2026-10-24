"""Explicit, non-inference recovery of one closed Generic source collection.

Historical files and lock owners are immutable. An apply uses the existing
physical/run locks and SSH process leases; cleanup additionally requires a full
verified archive of the exact owned remote run, including generated inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .resource_recovery import _generic_orphan_read, _require, _time
from .resource_recovery_observation import observe_resource_owners
from .process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
from .ssh_transport import SSHTransport
from ..workflow.run_control import EvaluationRunLock


def _descriptor(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def _local_capacity(output, archive_bytes, reserve_bytes):
    from ..filesystem_admission import require_write_target
    return require_write_target(output, operation='Closed collection archive',
        minimum_free_bytes=archive_bytes + reserve_bytes, minimum_free_inodes=128).as_dict()


# Uses the same tarfile/full-byte packaging and normal SCP as remote_run.
# The manifest describes storage bytes, not a new scientific result authority.
_ARCHIVE_SCRIPT = r'''
import hashlib, io, json, os, pathlib, shutil, stat, sys, tarfile
mode, root_raw, archive_raw, expected_archive_sha, anchors_raw = sys.argv[1:]
root, archive = pathlib.Path(root_raw), pathlib.Path(archive_raw)
anchors = json.loads(anchors_raw)
budget = anchors.pop('_capacity_budget', {})
if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
    raise RuntimeError('unsafe original remote run path')
if archive.parent != root or not archive.name.startswith('.collection-recovery-'):
    raise RuntimeError('unsafe recovery archive path')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''): h.update(data)
    return h.hexdigest()
def inventory():
    rows = []
    for p in sorted(root.rglob('*')):
        if p == archive: continue
        st = p.lstat()
        if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
            raise RuntimeError('unpreserved symlink/special file: ' + str(p))
        if p.is_file():
            before = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            row = {'path': p.relative_to(root).as_posix(), 'size_bytes': st.st_size, 'sha256': sha(p)}
            after = p.stat()
            if before != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError('remote content changed during hashing')
            rows.append(row)
    return rows
if mode == 'preflight':
    for p in root.rglob('*'):
        st = p.lstat()
        if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
            raise RuntimeError('unpreserved symlink/special file: ' + str(p))
    for logical, expected in anchors.items():
        p = root / logical
        if not p.is_file() or sha(p) != expected['sha256'] or p.stat().st_size != expected['size_bytes']:
            raise RuntimeError('original remote suite binding differs: ' + logical)
    print('COLLECTION_PREFLIGHT=exact_regular_owned_run')
elif mode == 'pack':
    if archive.exists(): raise RuntimeError('archive already exists')
    maximum, reserve = budget.get('max_archive_bytes'), budget.get('remote_reserve_bytes')
    if type(maximum) is not int or maximum <= 0 or type(reserve) is not int or reserve < 0:
        raise RuntimeError('explicit archive capacity budget missing')
    def free_bytes():
        fs = os.statvfs(root)
        return int(fs.f_bavail) * int(fs.f_frsize or fs.f_bsize)
    if free_bytes() < maximum + reserve: raise RuntimeError('fresh remote archive capacity insufficient')
    rows = inventory()
    manifest = json.dumps({'schema':'onnx-splitpoint/closed-run-byte-manifest',
        'schema_version':1, 'remote_run_dir':str(root), 'files':rows}, sort_keys=True).encode()
    created = False
    try:
        with archive.open('xb') as raw:
            created = True
            class BoundedOutput:
                def write(self, data):
                    if raw.tell() + len(data) > maximum or free_bytes() - len(data) < reserve:
                        raise RuntimeError('authorized compressed archive cap/remote reserve reached')
                    return raw.write(data)
                def tell(self): return raw.tell()
                def flush(self): return raw.flush()
            with tarfile.open(fileobj=BoundedOutput(), mode='w:gz') as tf:
                for row in rows: tf.add(root / row['path'], arcname='run/' + row['path'], recursive=False)
                info = tarfile.TarInfo('collection_manifest.json'); info.size = len(manifest)
                tf.addfile(info, io.BytesIO(manifest))
    except BaseException:
        if created: archive.unlink(missing_ok=True)  # only this new operation's partial
        raise
    if inventory() != rows: raise RuntimeError('remote content changed during packaging')
    print('COLLECTION_ARCHIVE=' + json.dumps({'sha256':sha(archive), 'size_bytes':archive.stat().st_size}))
elif mode == 'cleanup':
    if sha(archive) != expected_archive_sha: raise RuntimeError('remote archive changed')
    with tarfile.open(archive, 'r:gz') as tf:
        manifest = json.load(tf.extractfile('collection_manifest.json'))
    if manifest['remote_run_dir'] != str(root) or inventory() != manifest['files']:
        raise RuntimeError('remote content changed since verified download')
    shutil.rmtree(root)
    if root.exists(): raise RuntimeError('remote removal incomplete')
    print('COLLECTION_CLEANUP=removed_exact_verified_run')
else: raise RuntimeError('unknown collection operation')
'''.strip()


def _archive_command(mode, remote, archive, sha='', anchors=None, budget=None):
    control = dict(anchors or {})
    control['_capacity_budget'] = budget or {}
    return ' '.join(shlex.quote(v) for v in
                    ('python3', '-B', '-c', _ARCHIVE_SCRIPT, mode, remote, archive, sha,
                     json.dumps(control, separators=(',', ':'))))


def _verify_archive(archive, expected, remote):
    """Reject tampering, missing members, aliases and unsafe tar entries."""
    actual = _descriptor(archive)
    _require(all(actual[k] == expected[k] for k in ('sha256', 'size_bytes')),
             'downloaded full archive differs from remote digest')
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        _require(len(names) == len(set(names)) and names.count('collection_manifest.json') == 1,
                 'duplicate/missing archive manifest member')
        earlier = {}
        for member in members:
            logical = PurePosixPath(member.name)
            _require(not logical.is_absolute() and '..' not in logical.parts
                     and str(logical) == member.name
                     and (member.name == 'collection_manifest.json'
                          or (len(logical.parts) > 1 and logical.parts[0] == 'run')),
                     'unsafe archive member path: ' + member.name)
            if member.islnk():
                target = PurePosixPath(member.linkname)
                original = earlier.get(member.linkname)
                _require(member.name != 'collection_manifest.json' and member.size == 0
                         and not target.is_absolute() and '..' not in target.parts
                         and str(target) == member.linkname and len(target.parts) > 1
                         and target.parts[0] == 'run' and original is not None and original.isfile(),
                         'unsafe archive hardlink relation: ' + member.name)
            else:
                _require(member.isfile(), 'unsafe/special archive member: ' + member.name)
            earlier[member.name] = member
        manifest = json.load(tf.extractfile('collection_manifest.json'))
        _require(manifest.get('schema') == 'onnx-splitpoint/closed-run-byte-manifest'
                 and manifest.get('schema_version') == 1 and manifest.get('remote_run_dir') == remote,
                 'full archive belongs to a different remote run')
        rows = manifest.get('files')
        _require(isinstance(rows, list) and bool(rows)
                 and all(isinstance(r, dict) and isinstance(r.get('path'), str)
                         and type(r.get('size_bytes')) is int and r['size_bytes'] >= 0
                         and re.fullmatch('[0-9a-f]{64}', r.get('sha256', '')) for r in rows)
                 and len({r['path'] for r in rows}) == len(rows)
                 and set(names) == {'collection_manifest.json', *('run/' + r['path'] for r in rows)},
                 'archive content is incomplete or unmanifested')
        by_name = {'run/' + row['path']: row for row in rows}
        verified = {}
        for member in members:
            if member.name == 'collection_manifest.json':
                continue
            row = by_name[member.name]
            if member.islnk():
                target = by_name.get(member.linkname, {})
                _require((row['sha256'], row['size_bytes']) ==
                         (target.get('sha256'), target.get('size_bytes')) == verified.get(member.linkname),
                         'archive hardlink manifest/content differs: ' + member.name)
                continue
            handle = tf.extractfile(member)
            h = hashlib.sha256(); size = 0
            for data in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(data); size += len(data)
            _require(size == row['size_bytes'] and h.hexdigest() == row['sha256'],
                     'archive member content differs from its manifest')
            verified[member.name] = (h.hexdigest(), size)
    return manifest


def _archive_checkpoint(path, auth, identity, endpoint, remote, snapshots):
    """Bind the completed prior copy, never invent a fresh collection success."""
    descriptor = auth.get('archive_checkpoint')
    _require(isinstance(descriptor, dict), 'authorization lacks its exact archive checkpoint')
    checkpoint = _generic_orphan_read(Path(path), snapshots, descriptor)
    _require(checkpoint.get('schema') == 'onnx-splitpoint/generic-closed-collection-archive-checkpoint'
             and checkpoint.get('schema_version') == 1
             and all(checkpoint.get(k) == v for k, v in identity.items())
             and checkpoint.get('endpoint') == endpoint and checkpoint.get('remote_run_dir') == remote,
             'archive checkpoint scope differs')
    def read(name):
        d = checkpoint.get(name, {})
        return _generic_orphan_read(Path(d.get('path', '')), snapshots, d)
    prior, logger = read('prior_recovery_receipt'), read('prior_logger_result')
    prior_dir = Path(checkpoint['prior_recovery_receipt']['path']).parent
    _require(all(prior.get(k) == v for k, v in identity.items())
             and prior.get('schema') == 'onnx-splitpoint/generic-closed-collection-recovery'
             and prior.get('status') == 'failed' and prior.get('apply') is True
             and prior.get('endpoint') == endpoint and prior.get('remote_run_dir') == remote
             and prior.get('measurements_started') == 0 and prior.get('new_operation_cleanup') == []
             and not prior.get('new_physical_quarantine')
             and 'unsafe/special archive member' in str(prior.get('error', ''))
             and logger.get('returncode') == 1, 'prior failed copy event is not bound/quiescent')
    argv = logger.get('argv', [])
    for flag, expected in (('--run-dir', identity['run_dir']), ('--session-id', identity['session_id']),
                           ('--setup-id', identity['setup_id']), ('--model-id', identity['model_id']),
                           ('--output-dir', str(prior_dir))):
        _require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                 and argv[argv.index(flag) + 1] == expected, 'prior collection logger scope differs')
    archive = checkpoint.get('remote_archive', {})
    local = checkpoint.get('local_archive', {})
    _require(archive == prior.get('remote_archive') and Path(archive.get('path', '')).parent == Path(remote)
             and Path(archive['path']).name.startswith('.collection-recovery-')
             and local.get('path') == str(prior_dir / 'original_remote_run.tar.gz')
             and all(local.get(k) == archive.get(k) for k in ('sha256', 'size_bytes')),
             'prior local/remote archive binding differs')
    from ..benchmark.remote_run import _remote_result_collect_command
    collector, pack = read('collector_diagnostic'), read('pack_diagnostic')
    for name, diagnostic in (('collector_diagnostic', collector), ('pack_diagnostic', pack)):
        _require(Path(checkpoint[name]['path']).parent == prior_dir / 'diagnostics'
                 and diagnostic.get('target_id') == identity['setup_id']
                 and diagnostic.get('returncode') == 0 and diagnostic.get('cleanup_proven') is True,
                 'prior collection/pack diagnostic is not normal completed zero')
    _require(collector.get('command') == _remote_result_collect_command(
        remote_results_dir=remote + '/results', remote_suite_dir=remote + '/suite')
        and collector.get('output') == 'SPLITPOINT_COLLECT_BEGIN\nSPLITPOINT_COLLECT_END exit_trap_status=0 copy_failed=0\n',
        'previous actual collection zero is not proven')
    args = shlex.split(pack.get('command', ''))
    _require(len(args) == 9 and args[:3] == ['python3', '-B', '-c']
             and args[3] == _ARCHIVE_SCRIPT and args[4:7] == ['pack', remote, archive['path']],
             'previous normal archive writer/path differs')
    values = [json.loads(line.split('=', 1)[1]) for line in pack.get('output', '').splitlines()
              if line.startswith('COLLECTION_ARCHIVE=')]
    _require(values == [{k: archive[k] for k in ('sha256', 'size_bytes')}], 'previous pack integrity proof differs')
    return checkpoint


def recover_generic_closed_collection(*, run_dir, session_id, setup_id, model_id,
                                      evidence_path, authorization_path, output_dir,
                                      registry_path=None, apply=False, cleanup=False,
                                      archive_checkpoint_path=None):
    """Dry-run by default. Never run inference or rewrite original completion."""
    from ..energy.config import default_registry_path, _read_hardware_registry_unlocked
    from ..platform_power import host_config_from_setup
    from ..workflow.execution_binding import physical_dut_key
    from ..workflow.hardware_matrix import _setup_to_target
    from ..benchmark.remote_run import _remote_result_collect_command, _remote_storage_preflight
    import yaml

    run = Path(run_dir).expanduser()
    _require(run.is_absolute() and run.is_dir()
             and not any(p.is_symlink() for p in (run, *run.parents)), 'unsafe original run')
    identity = dict(run_id=run.name, run_dir=str(run), session_id=session_id,
                    setup_id=setup_id, model_id=model_id)
    _require(all(isinstance(v, str) and re.fullmatch('[A-Za-z0-9_-]+', v)
                 for v in (setup_id, model_id)), 'unsafe setup/model identifier')
    _require(re.fullmatch('[0-9a-f]{32}', session_id or '') is not None, 'invalid original session')
    snapshots = {}
    evidence = _generic_orphan_read(Path(evidence_path), snapshots)
    auth = _generic_orphan_read(Path(authorization_path), snapshots)
    schema = 'onnx-splitpoint/generic-closed-collection-recovery'
    _require(evidence.get('schema') == schema + '-input' and evidence.get('schema_version') == 1
             and all(evidence.get(k) == v for k, v in identity.items()), 'evidence scope differs')
    _require(auth.get('schema') == schema + '-authorization' and auth.get('schema_version') == 1
             and auth.get('authorized') is True
             and auth.get('action') == 'recover_generic_closed_collection'
             and all(auth.get(k) == v for k, v in identity.items())
             and all(isinstance(auth.get(k), str) and auth[k].strip()
                     for k in ('operator', 'source', 'authorized_at'))
             and _time(auth['authorized_at']) <= datetime.now(timezone.utc), 'explicit authorization differs')
    _require(isinstance(auth.get('evidence'), dict), 'authorization lacks evidence descriptor')
    _generic_orphan_read(Path(evidence_path), snapshots, auth['evidence'])
    _require(not cleanup or (apply and auth.get('cleanup') is True), 'cleanup not explicitly authorized')
    _require(bool(archive_checkpoint_path) == bool(auth.get('archive_checkpoint')),
             'archive checkpoint requires its explicit bound authorization and argument')
    budget = auth.get('capacity_budget', {})
    _require(type(budget.get('max_archive_bytes')) is int and budget['max_archive_bytes'] > 0
             and all(type(budget.get(k)) is int and budget[k] >= 0
                     for k in ('local_reserve_bytes', 'remote_reserve_bytes')),
             'explicit max archive bytes and local/remote reserves required')
    original = evidence.get('original', {})
    def read(name, path):
        return _generic_orphan_read(path, snapshots, original.get(name, {}))
    manifest = read('manifest', run / 'run_manifest.json')
    _require(manifest.get('schema') == 'onnx-splitpoint/evaluation-run-manifest'
             and manifest.get('run_id') == run.name and manifest.get('run_dir') == str(run)
             and manifest.get('status') in ('ok', 'completed', 'partial', 'failed')
             and manifest.get('current_session_id') == session_id, 'original run is not closed/bound')
    sessions = manifest.get('execution_sessions', [])
    _require(bool(sessions) and sessions[-1].get('session_id') == session_id
             and all(s.get('status') not in (None, 'running') and s.get('finished_at') for s in sessions),
             'original sessions remain active or incomplete')
    profile_path = run / 'profile.yaml'
    _require(_descriptor(profile_path) == original.get('profile'), 'original profile changed')
    profile_bytes = profile_path.read_bytes(); snapshots[profile_path] = profile_bytes
    profile = yaml.safe_load(profile_bytes)
    profiles = [p for p in profile.get('run_profiles', []) if p.get('enabled', True)]
    options = manifest.get('options', {})
    _require(len(profiles) == 1 and profiles[0].get('quality_source_only') is True
             and profiles[0].get('hardware_setup_id') == setup_id
             and options.get('energy_enabled') is False and options.get('native_producer_enabled') is False,
             'only Generic quality-source-only runs are eligible')
    result_root = run / 'models' / model_id / 'benchmark_results'
    dispatch = read('dispatch', result_root / ('remote_benchmark_dispatch_' + setup_id + '.json'))
    status = read('status', result_root / ('remote_benchmark_status_' + setup_id + '.json'))
    _require(dispatch.get('schema') == 'onnx-splitpoint/remote-benchmark-dispatch'
             and dispatch.get('model_id') == model_id and dispatch.get('hardware_target_id') == setup_id
             and dispatch.get('run_id') == run.name + '_' + model_id + '_' + setup_id
             and dispatch.get('args', {}).get('quality_evidence_eval_id') == run.name
             and dispatch.get('args', {}).get('validation_max_images') == 5000
             and status.get('model_id') == model_id and status.get('hardware_target_id') == setup_id,
             'original dispatch differs')
    transport_root = run.parent / 'RemoteBenchmarkRuns/Results/legacy_suite/1' / dispatch['run_id']
    _require(status.get('local_run_dir') == str(transport_root), 'original transport directory differs')
    metadata = read('transport_metadata', transport_root / 'run_meta.json')
    _require(metadata.get('run_id') == dispatch['run_id'] and metadata.get('repeat_idx') == '1',
             'original transport run metadata differs')
    anchors = {}
    for name in ('benchmark_set.json', 'benchmark_plan.json'):
        p = transport_root / 'results' / name
        read(name, p)
        anchors['suite/' + name] = _descriptor(p)
    capture = status.get('generic_quality_source_capture', {})
    cases = profiles[0].get('case_ids', [])
    entries = capture.get('evidence', [])
    _require(capture.get('status') == 'verified_exact' and capture.get('errors') == []
             and bool(cases) and capture.get('count') == capture.get('expected') == len(cases)
             and len(entries) == len(cases)
             and {e.get('identity', {}).get('case_id') for e in entries} == set(cases),
             'strict original source capture is not complete')
    index = read('artifact_index', run / 'artifact_index.json')
    closure = read('closure', run / 'reports/artifact_index_closure.json')
    _require(index.get('schema') == 'onnx-splitpoint/artifact-index' and index.get('schema_version') == 2
             and index.get('run_id') == run.name and closure.get('status') == 'pass'
             and closure.get('schema') == 'onnx-splitpoint/artifact-index-terminal-closure'
             and closure.get('run_id') == run.name and closure.get('session_id') == session_id
             and all(closure.get(k) == 0 for k in ('missing_file_count', 'hash_mismatch_count',
                                                 'unindexed_required_path_count', 'verification_error_count')),
             'original terminal artifact closure is not valid')
    terminal = index.get('terminal_closure', {})
    _require(terminal.get('schema') == 'onnx-splitpoint/artifact-index-terminal-closure-binding'
             and terminal.get('run_id') == run.name and terminal.get('session_id') == session_id
             and terminal.get('status') == 'pass'
             and terminal.get('report_path') == 'reports/artifact_index_closure.json',
             'index lacks its original terminal binding')
    indexed = {row.get('path'): row for row in index.get('artifacts', [])}
    scientific = []
    for entry in entries:
        _require(entry.get('image_count') == 5000
                 and entry.get('identity', {}).get('model_id') == model_id
                 and entry.get('identity', {}).get('setup_id') == setup_id, 'source identity differs')
        for role in ('request', 'candidate', 'child_report'):
            desc = entry[role]; p = Path(desc['path'])
            _require(p.is_relative_to(run) and not any(q.is_symlink() for q in (p, *p.parents))
                     and _descriptor(p) == desc, 'original scientific bytes changed')
            record = indexed.get(p.relative_to(run).as_posix(), {})
            _require(record.get('sha256') == 'sha256:' + desc['sha256']
                     and record.get('size_bytes') == desc['size_bytes'],
                     'scientific descriptor is outside the sealed original index')
            scientific.append((p, desc))
    registry_file = Path(registry_path or default_registry_path())
    registry_before = registry_file.read_bytes()
    registry, _ = _read_hardware_registry_unlocked(registry_file)
    setups = [row for row in registry.get('hardware_setups', []) if row.get('id') == setup_id]
    _require(len(setups) == 1, 'unknown/ambiguous setup')
    target = _setup_to_target(setups[0], build_envs=registry.get('build_environments', []),
                              registry_source=str(registry_file))
    # The shared owner observer also accepts the registry's top-level alias.
    # Carry the factory's exact physical identity there, without inventing one.
    target['physical_host_id'] = target['remote'].get('physical_host_id', '')
    host = host_config_from_setup(target)
    resource = physical_dut_key(target)
    endpoint = {'host': host.host, 'user': host.user, 'port': int(host.port)}
    _require(all(dispatch.get('host', {}).get(k) == v for k, v in endpoint.items())
             and all(metadata.get('host', {}).get(k) == v for k, v in endpoint.items())
             and auth.get('endpoint') == endpoint, 'original/current endpoint differs')
    remote = status.get('remote_output', {}).get('remote_run_dir', '')
    remote_base = host.remote_base_dir
    if remote_base.startswith('~/'):
        remote_base = '/home/' + host.user + remote_base[1:]
    expected_remote = str(PurePosixPath(remote_base) / 'legacy_suite' / str(dispatch.get('run_id')) / '1')
    posix = PurePosixPath(remote)
    _require(posix.is_absolute() and '..' not in posix.parts
             and remote_base.endswith('/splitpoint_runs') and remote == expected_remote
             and str(posix) == remote
             and auth.get('remote_run_dir') == remote, 'unsafe/unbound original remote directory')
    checkpoint = (_archive_checkpoint(archive_checkpoint_path, auth, identity, endpoint, remote, snapshots)
                  if archive_checkpoint_path else None)
    if checkpoint:
        _require(0 < checkpoint['local_archive']['size_bytes'] <= budget['max_archive_bytes'],
                 'bound existing archive exceeds operator cap')
    output = Path(output_dir).expanduser()
    _require(output.is_absolute() and not output.exists() and not output.is_relative_to(run)
             and output == output.resolve() and run == run.resolve()
             and not any(p.is_symlink() for p in output.parents), 'recovery requires new separate output')
    new_archive_bytes = 0 if checkpoint else budget['max_archive_bytes']
    local_admission = _local_capacity(output, new_archive_bytes, budget['local_reserve_bytes'])
    run_lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    physical = EvaluationRunLock.for_resource(resource, owner={})
    run_lock.acquire_for_recovery()
    try:
        physical.acquire_for_recovery()
        try:
            old = read('run_lock', run_lock.lock_path)
            _require(old == run_lock.previous_owner and old.get('state') == 'released'
                     and old.get('schema') == 'onnx-splitpoint/evaluation-run-lock'
                     and all(old.get(k) == identity[k] for k in ('run_id', 'run_dir', 'session_id'))
                     and type(old.get('pid')) is int and old['pid'] > 0
                     and old.get('pid') == sessions[-1].get('start_origin', {}).get('pid'),
                     'original released controller binding differs')
            _require(physical.previous_owner.get('state') == 'released'
                     and physical.previous_owner.get('schema') == 'onnx-splitpoint/evaluation-run-lock'
                     and physical.previous_owner.get('physical_resource') == resource
                     and not os.path.lexists(run_lock.quarantine_path) and not os.path.lexists(physical.quarantine_path)
                     and not Path('/proc', str(old['pid'])).exists(), 'active/unknown owner or quarantine')
            journal_root = run / 'jobs/remote_process_leases'
            _require(journal_root.is_dir() and not journal_root.is_symlink()
                     and not any(p.is_symlink() for p in journal_root.rglob('*'))
                     and not list(journal_root.rglob('*.remote-lease.json')),
                     'original remote actor cleanup unresolved')
            output.mkdir(parents=True)
            receipt = {'schema': schema, 'schema_version': 1, **identity, 'remote_run_dir': remote,
                       'endpoint': endpoint, 'physical_resource': resource,
                       'apply': bool(apply), 'cleanup_requested': bool(cleanup),
                       'capacity_budget': budget, 'local_pre_mutation_capacity': local_admission,
                       'collection_mode': 'reuse_bound_previous_archive' if checkpoint else 'fresh_collect_pack_download',
                       'storage_forecast': {
                           'maximum_new_local_compressed_bytes': new_archive_bytes,
                           'minimum_local_free_bytes': new_archive_bytes + budget['local_reserve_bytes'],
                           'minimum_remote_free_bytes': new_archive_bytes + budget['remote_reserve_bytes'],
                           'full_plain_extraction_bytes': 0, 'compression_ratio_assumed': False},
                       'remote_capacity': 'not_probed_in_local_dry_run',
                       'original_status_unchanged': True, 'measurements_started': 0,
                       'evidence': _descriptor(evidence_path), 'authorization': _descriptor(authorization_path)}
            def unchanged(check_scientific=False):
                _require(registry_file.read_bytes() == registry_before
                         and all(p.read_bytes() == b for p, b in snapshots.items())
                         and run_lock._read_owner() == old
                         and physical._read_owner() == physical.previous_owner,
                         'original evidence/registry/lock changed during recovery')
                if check_scientific:
                    _require(all(_descriptor(p) == desc for p, desc in scientific),
                             'original scientific bytes changed during recovery')
            leases = None
            try:
                if apply:
                    observed = observe_resource_owners(target, local_owners=[], absent_owner_pids=[old['pid']],
                        diagnostics_dir=output / 'owner_diagnostics', resource_scoped=True)
                    receipt['ownership_observation'] = observed
                    _require(observed.get('ok') is True, 'associated controller/remote actors not proven stopped')
                    scope = RemoteProcessLeaseScope(run_id=run.name, session_id=session_id)
                    leases = RemoteProcessLeaseRegistry()
                    journal = leases.configure_journal(scope=scope, journal_dir=output / 'remote_process_leases')
                    transport = SSHTransport(host, remote_lease_scope=scope, remote_lease_registry=leases)
                    transport.diagnostics_dir = output / 'diagnostics'
                    def command(cmd, timeout):
                        rc, text = transport.run(cmd, timeout_s=timeout)
                        _require(rc == 0 and leases.active_count() == journal.pending_count() == 0,
                                 'fresh collection operation failed; remote retained')
                        return text
                    unchanged()
                    receipt['remote_pre_mutation_capacity'] = _remote_storage_preflight(
                        transport, remote, required_free_bytes=new_archive_bytes + budget['remote_reserve_bytes'],
                        required_free_inodes=4, stage='closed_collection_before_collect_pack')
                    receipt['remote_capacity'] = 'fresh_normal_preflight_passed'
                    archive_remote = (checkpoint['remote_archive']['path'] if checkpoint else
                                      remote + '/.collection-recovery-' + uuid.uuid4().hex + '.tar.gz')
                    command(_archive_command('preflight', remote, archive_remote, anchors=anchors), 120)
                    if checkpoint:
                        expected = checkpoint['local_archive']
                        local_archive = Path(expected['path'])
                        receipt.update(archive_checkpoint=auth['archive_checkpoint'], previous_collection_rc=0,
                            previous_event={k: checkpoint[k] for k in (
                                'prior_recovery_receipt', 'prior_logger_result', 'collector_diagnostic', 'pack_diagnostic')},
                            fresh_collection_performed=False, fresh_pack_or_download_performed=False)
                    else:
                        command(_remote_result_collect_command(
                            remote_results_dir=remote + '/results', remote_suite_dir=remote + '/suite'), 60)
                        text = command(_archive_command('pack', remote, archive_remote, budget=budget), 600)
                        values = [json.loads(line.split('=', 1)[1]) for line in text.splitlines()
                                  if line.startswith('COLLECTION_ARCHIVE=')]
                        _require(len(values) == 1, 'remote archive integrity descriptor missing')
                        expected = values[0]
                        _require(type(expected.get('size_bytes')) is int
                                 and 0 < expected['size_bytes'] <= budget['max_archive_bytes'],
                                 'actual archive exceeds authorized compressed budget')
                        receipt['local_before_download_capacity'] = _local_capacity(
                            output, expected['size_bytes'], budget['local_reserve_bytes'])
                        local_archive = output / 'original_remote_run.tar.gz'
                        rc, _ = transport.scp_download(archive_remote, str(local_archive), timeout_s=600)
                        _require(rc == 0 and leases.active_count() == journal.pending_count() == 0,
                                 'archive download incomplete')
                        receipt.update(fresh_collection_rc=0, fresh_collection_performed=True,
                                       fresh_pack_or_download_performed=True)
                    receipt['remote_archive'] = {'path': archive_remote, **{
                        k: expected[k] for k in ('sha256', 'size_bytes')}}
                    byte_manifest = _verify_archive(local_archive, expected, remote)
                    preserved = {(r['sha256'], r['size_bytes']) for r in byte_manifest['files']}
                    _require(all((desc['sha256'], desc['size_bytes']) in preserved for _, desc in scientific),
                             'full archive lacks original scientific content')
                    receipt.update(archive=_descriptor(local_archive), preserved_file_count=len(byte_manifest['files']),
                                   complete_archive_verified=True)
                    unchanged(check_scientific=True)
                    if cleanup:
                        command(_archive_command('cleanup', remote, archive_remote, expected['sha256']), 600)
                        receipt['remote_cleanup'] = 'removed_exact_verified_run'
                unchanged()
                receipt['status'] = 'completed' if apply else 'dry_run_pass'
            except BaseException as exc:
                receipt.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                if leases is not None:
                    reports = leases.cancel_all(grace_s=3.0)
                    receipt['new_operation_cleanup'] = reports
                    if leases.active_count() or journal.pending_count() or any(not r.get('ok') for r in reports):
                        fence = physical.commit_quarantine_fence({
                            'reason': 'closed_collection_recovery_remote_cleanup_unproven',
                            'recovery_output': str(output), 'original_run_id': run.name,
                            'session_id': session_id})
                        receipt['new_physical_quarantine'] = str(fence)
                raise
            finally:
                (output / 'COLLECTION_RECOVERY_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
            return receipt
        finally:
            physical.release(write_released_state=False)
    finally:
        run_lock.release(write_released_state=False)
