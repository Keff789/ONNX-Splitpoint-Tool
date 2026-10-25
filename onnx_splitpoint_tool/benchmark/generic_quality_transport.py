"""Bound only explicitly selected Generic source runtime transport binaries.

The catalogue and all shared inputs/metadata remain complete. This does not
authorize execution or replace any cache, artifact, or child Quality checks.
"""
from __future__ import annotations

from collections.abc import Mapping
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .generic_quality_preflight import _contained
from ..runners.generic_quality_accounting import _binding, _stage


def _inline_graphs(suite, excludes, required_graphs):
    """Prove inline storage only; never resolve or follow external blobs."""
    try:
        import onnx
        graphs = set(required_graphs)
        graphs.update(p for p in suite.rglob('*.onnx') if not any(
            fnmatch.fnmatch(p.relative_to(suite).as_posix(), pattern) for pattern in excludes))
        for path in sorted(graphs):
            relative = path.relative_to(suite)
            _contained(suite, relative)
            model = onnx.load(str(path), load_external_data=False)
            if not model.HasField('graph'):
                return 'graph_missing:' + str(relative)
            known = onnx.ModelProto(); known.CopyFrom(model); known.DiscardUnknownFields()
            if known.SerializeToString(deterministic=True) != model.SerializeToString(deterministic=True):
                return 'unsupported_graph_protobuf:' + str(relative)
            pending = [model]
            while pending:
                message = pending.pop()
                if isinstance(message, onnx.TensorProto) and (
                        message.external_data or message.data_location != onnx.TensorProto.DEFAULT):
                    return 'external_or_unknown_tensor_storage:' + str(relative)
                for field, value in message.ListFields():
                    if field.message_type is not None:
                        pending.extend(value if field.label == field.LABEL_REPEATED else (value,))
        return ''
    except Exception as exc:
        return 'inline_graph_proof_unavailable:' + type(exc).__name__ + ':' + str(exc)[:160]


def selected_source_transport_excludes(suite_dir, *, active_run_ids, model_id, log=None):
    """Return one operation-local exclusion tuple, or unchanged transport.

    Only declared unselected case binaries can be omitted. Unknown/mixed run
    selections retain the default; malformed explicit source dependencies fail.
    """
    suite = Path(suite_dir).resolve(strict=True)
    requested = set(active_run_ids or ())
    if not requested:
        return ()
    try:
        plan = json.loads((suite / 'benchmark_plan.json').read_text())
    except (OSError, ValueError):
        return ()  # No explicit source proof: retain ordinary transport.
    return _selected_source_excludes(suite, plan=plan, active_run_ids=requested,
                                    model_id=model_id, log=log)


def selected_source_import_excludes(suite_dir, *, runs, model_id, log=None):
    """Prove the same dependency selection before a new local suite copy.

    Use the current explicitly bound original configs, never the source suite's
    historical executable plan. Sidecars/runtime are materialized normally after
    import; missing old dependencies keep the ordinary complete local copy.
    """
    return selected_source_import_selection(
        suite_dir, runs=runs, model_id=model_id, log=log)[0]


def selected_source_import_selection(suite_dir, *, runs, model_id, log=None):
    """Return binary exclusions and exact selected normal publication targets."""
    suite = Path(suite_dir).resolve(strict=True)
    runs = list(runs)
    staged_paths = set()
    excludes = _selected_source_excludes(
        suite, plan={'runs': runs},
        active_run_ids=[str(row.get('id') or row.get('name') or '')
                        for row in runs if isinstance(row, Mapping)
                        and row.get('semantic_reference_only') is not True],
        model_id=model_id, source_configs=True, staged_paths=staged_paths, log=log)
    return excludes, tuple(sorted(staged_paths))


def _selected_source_excludes(suite, *, plan, active_run_ids, model_id,
                              source_configs=False, staged_paths=None, log=None):
    requested = set(active_run_ids or ())
    if not requested:
        return ()

    def retain_full(reason):
        if log is not None:
            prefix = '[source][local-import]' if source_configs else '[remote][transport]'
            log(prefix + ' retaining full suite: ' + reason)
        return ()

    if not isinstance(plan, Mapping) or not isinstance(plan.get('runs'), list):
        return ()
    rows = [r for r in plan.get('runs', []) if isinstance(r, Mapping)
            and str(r.get('id') or r.get('name') or '') in requested]
    if ({str(r.get('id') or r.get('name') or '') for r in rows} != requested
            or any(r.get('quality_source_only') is not True
                   or 'generic_quality_completion_configs' not in r for r in rows)):
        return ()
    # One model's bundle path is shared by setup workers. Do not give another
    # executable recipe a different payload at that same transport address.
    if any(r not in rows and not (r.get('execution_location') == 'central_management'
                                  and r.get('semantic_reference_only') is True)
           for r in plan.get('runs', []) if isinstance(r, Mapping)):
        return ()

    def require(ok):
        if not ok:
            raise ValueError('generic_quality_transport_dependency_unproved')

    catalogue = json.loads((suite / 'benchmark_set.json').read_text())
    require(isinstance(catalogue, Mapping) and isinstance(catalogue.get('cases'), list))
    declared = set()
    for row in catalogue.get('cases', []):
        require(isinstance(row, Mapping))
        case = row.get('case_dir') or row.get('folder')
        require(isinstance(case, str) and re.fullmatch(r'b[0-9]+', case)
                and row.get('folder', case) == case and case not in declared)
        declared.add(case)
    require(declared and len(rows) == len(requested))
    selected = set(); required_graphs = set(); publications = {}
    for row in rows:
        binding = _binding(row)  # Existing exact source/config/stage scope checks.
        cases = binding['case_ids']; paths = binding['generic_quality_completion_configs']
        hashes = binding['generic_quality_completion_config_sha256']
        if source_configs and row.get('native_trt_build') is not False:
            return retain_full('source_only_build_policy_not_explicit')
        require(row.get('native_trt_build') is False and set(cases) <= declared)
        for case in cases:
            require(paths[case] == f'quality_runtime/{case}.json'
                    and isinstance(hashes[case], str) and re.fullmatch('[0-9a-f]{64}', hashes[case]))
            if source_configs:
                from ..workflow.generic_quality_sidecars import _producer_source_bytes, _producer_source_path
                payload = _producer_source_bytes(binding['generic_quality_completion_sources'][case])
            else:
                target = _contained(suite, paths[case])
                payload = target.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == hashes[case]
                    and len(payload) == binding['generic_quality_completion_sources'][case]['size_bytes'])
            config = json.loads(payload)
            if source_configs and not (isinstance(config, Mapping)
                    and config.get('schema') == 'onnx-splitpoint/generic-composed-quality-completion-config'
                    and type(config.get('schema_version')) is int
                    and config['schema_version'] == 1):
                return retain_full('original_config_schema_unavailable')
            identity = config.get('identity') or {}
            backend = {'hailo8': 'hailo8', 'hailo10': 'hailo10h',
                       'hailo10h': 'hailo10h', 'deepx_m1': 'deepx_m1'}.get(_stage(row.get('stage1')))
            if source_configs and not (isinstance(identity, Mapping)
                    and all(identity.get(key) for key in
                            ('case_id', 'model_id', 'setup_id', 'comparison_backend'))):
                return retain_full('original_config_identity_metadata_unavailable')
            require(config.get('path_root') == 'suite_root' and backend
                    and _stage(row.get('stage2')) == 'tensorrt'
                    and identity.get('case_id') == case and identity.get('model_id') == model_id
                    and identity.get('setup_id') == row.get('hardware_setup_id')
                    and bool(identity.get('setup_id')) and identity.get('comparison_backend') == backend)
            artifacts = config.get('expected_runtime_artifacts')
            sources = config.get('runtime_source_files')
            if source_configs and not (isinstance(artifacts, Mapping)
                    and {'source_part1_onnx', 'source_part2_onnx'} <= artifacts.keys()
                    and isinstance(sources, list) and sources):
                return retain_full('original_config_dependency_metadata_unavailable')
            require(isinstance(artifacts, Mapping) and isinstance(sources, list) and sources
                    and {'source_part1_onnx', 'source_part2_onnx'} <= artifacts.keys())
            staged_assets = {}
            if source_configs:
                from ..workflow.generic_quality_sidecars import _portable_asset_descriptors
                def stage(relative, descriptor, origin):
                    target = _contained(suite, relative)
                    original = _producer_source_path(origin)
                    require(original.stat().st_uid == os.getuid())
                    _producer_source_bytes({**descriptor, 'path': str(original)})
                    require(original.stat().st_uid == os.getuid()
                            and (not target.exists() or (
                                stat.S_ISREG(target.stat().st_mode)
                                and target.stat().st_uid == os.getuid())))
                    identity = (descriptor['sha256'], descriptor['size_bytes'], str(original))
                    require(relative not in publications or publications[relative] == identity)
                    publications[relative] = identity
                    return original
                stage(paths[case], binding['generic_quality_completion_sources'][case],
                      binding['generic_quality_completion_sources'][case]['path'])
                for _role, descriptor in _portable_asset_descriptors(config, case):
                    if descriptor.get('local_path'):
                        relative = descriptor['path']
                        staged_assets[relative] = stage(relative, descriptor, descriptor['local_path'])
            for role in ('source_part1_onnx', 'source_part2_onnx'):
                require(Path(artifacts[role].get('path', '')).parts[:1] == (case,))
                required_graphs.add(_contained(suite, artifacts[role]['path']))
            # All declared suite-relative artifact/runtime dependencies remain
            # present. Persistent absolute DUT cache references are unchanged.
            for role, descriptor in [*artifacts.items(), *((None, d) for d in sources)]:
                require(isinstance(descriptor, Mapping) and isinstance(descriptor.get('path'), str))
                relative = Path(descriptor['path'])
                if relative.is_absolute():
                    require(role in {'engine', 'engine_build_receipt', 'build_part2_onnx'}
                            and '..' not in relative.parts and not relative.is_relative_to(suite))
                    continue
                dependency = _contained(suite, relative)
                require(not (re.fullmatch(r'b[0-9]+', relative.parts[0])
                             and relative.parts[0] != case))
                if source_configs and relative.as_posix() in staged_assets:
                    # Only the normal materializer's exact selected portable
                    # target may use the bound original before the new copy.
                    dependency = staged_assets[relative.as_posix()]
                elif source_configs and not dependency.is_file():
                    return retain_full('dependency_not_materialized:' + str(relative))
                require(dependency.is_file()
                        and ('size_bytes' not in descriptor or dependency.stat().st_size == descriptor['size_bytes']))
            def check_paths(value):
                if isinstance(value, Mapping):
                    token = value.get('path')
                    if isinstance(token, str) and token and not Path(token).is_absolute():
                        relative = Path(token)
                        _contained(suite, relative)
                        require(not (re.fullmatch(r'b[0-9]+', relative.parts[0]) and relative.parts[0] != case))
                    for item in value.values():
                        check_paths(item)
                elif isinstance(value, list):
                    for item in value:
                        check_paths(item)
            check_paths(config)
            selected.add(case)
    suffixes = ('.onnx', '.hef', '.dxnn', '.engine', '.plan', '.data', '.bin')
    excludes = tuple(f'{case}/*{suffix}' for case in sorted(declared - selected) for suffix in suffixes)
    reason = _inline_graphs(suite, excludes, required_graphs) if excludes else ''
    if reason:
        return retain_full(reason)
    if staged_paths is not None:
        staged_paths.update(publications)
    return excludes
