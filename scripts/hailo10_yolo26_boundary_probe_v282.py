#!/usr/bin/env python3
"""Bounded current Generic H10 replay: existing artifacts, Fixed1 or Fixed4.

The retained collector owns transport, process supervision and offline_reference.
This entry closes artifact/source resolution without requiring a Native result.
"""
from __future__ import annotations
import argparse
import ast
import json
import shlex
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import hailo10_yolo26_boundary_probe_v27931 as retained
from hailo_boundary_diagnostics_v27931 import FLAGS, sha256, write_json


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _digest(value, role):
    digest = str(value or '').removeprefix('sha256:').lower()
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('diagnostic_bound_identity_not_available:' + role)
    return digest


_REMOTE_CONFIG = None


def recorded_generic_binding_request(run, model, row):
    """Bind a read-only receipt request to the exact failed Generic row/log."""
    session = row['native_tensorrt']['sessions']['part2:tensorrt']
    io = row['deployment_contract']['hailo_io_contracts']['part1']
    path = str(Path(io['artifact']).parent / 'native_split_quality_binding.json')
    if '/_onnx_splitpoint_cache/' not in path:
        raise ValueError('diagnostic_generic_binding_persistent_path_not_available')
    log_path = Path(run) / f'models/{model}/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log'
    if not log_path.is_file():
        raise ValueError('diagnostic_generic_binding_log_anchor_not_available')
    pattern = r'exact local binding verified sha256=([0-9a-f]{64}) engine=' + re.escape(session['engine']) + r'(?:\s|$)'
    anchors = set(re.findall(pattern, log_path.read_text(errors='replace')))
    if len(anchors) != 1:
        raise ValueError('diagnostic_generic_binding_log_anchor_ambiguous_or_missing')
    return {'path': path, 'binding_sha256': next(iter(anchors)), 'case': row['case_id'],
            'engine_sha256': _digest(session['engine_sha256'], 'engine'),
            'bridge_sha256': _digest(session['source_model_sha256'], 'bridge'),
            'receipt_file_sha256': _digest(session['engine_build_receipt_file_sha256'], 'receipt'),
            'log_path': str(log_path), 'log_sha256': sha256(log_path)}


def _load_recorded_remote_binding(run, model, row):
    """Read one exact persistent receipt via existing transport, no device job."""
    session = row['native_tensorrt']['sessions']['part2:tensorrt']
    io = row['deployment_contract']['hailo_io_contracts']['part1']
    binding_path = str(Path(io['artifact']).parent / 'native_split_quality_binding.json')
    if '/_onnx_splitpoint_cache/' not in binding_path or _REMOTE_CONFIG is None:
        return None
    # The current failed Generic process logged this very sealed binding, even
    # when it could not export a valid endpoint/quality request afterwards.
    recorded = recorded_generic_binding_request(run, model, row)
    anchors = {recorded['binding_sha256']}
    remote, python = _REMOTE_CONFIG
    script = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.stat().st_size <= 2097152; sys.stdout.write(p.read_text())'
    command = shlex.join([python, '-I', '-B', '-c', script, binding_path])
    result = retained.run_transport(retained.transport.ssh_base(remote) + [command],
        timeout=30, check=True, capture_output=True, text=True)
    binding = json.loads(result.stdout)
    from onnx_splitpoint_tool.native_split_quality import validate_native_split_quality_binding
    verified, reason = validate_native_split_quality_binding(binding, verification_mode='portable',
        expected_identity={'model': model, 'case': row['case_id'], 'setup_id': row['setup_id'],
                           'backend': 'hailo10h_to_trt', 'task': 'detection', 'precision': session['precision']})
    if verified is None or binding.get('binding_sha256') not in anchors:
        raise ValueError('diagnostic_generic_remote_binding_not_exact:' + str(reason))
    return binding


def external_closure(path):
    """Resolve the actual graph's external data; never fabricate tensor values."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors
    path = Path(path).resolve()
    graph = onnx.load(str(path), load_external_data=False)
    closure = []
    seen = set()
    for tensor in _get_all_tensors(graph):
        data = {entry.key: entry.value for entry in tensor.external_data}
        if not data:
            continue
        location = data.get('location', '')
        member = (path.parent / location).resolve()
        if not location or Path(location).is_absolute() or '..' in Path(location).parts:
            raise ValueError('diagnostic_external_data_location_unsafe')
        if not member.is_file():
            raise ValueError('diagnostic_external_data_not_available:' + location)
        offset = int(data.get('offset', 0)); length = int(data.get('length', member.stat().st_size - offset))
        if offset < 0 or length < 0 or offset + length > member.stat().st_size:
            raise ValueError('diagnostic_external_data_range_invalid:' + location)
        if location not in seen:
            closure.append({'location': location, 'sha256': sha256(member), 'size_bytes': member.stat().st_size})
            seen.add(location)
    return closure


def resolve_case(run, model, artifact_roots=(), diagnostic_image=None):
    run = Path(run).resolve(); case = retained.CASES[model]
    result_path = run / f'models/{model}/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json'
    rows = retained.read_json(result_path)
    rows = [r for r in rows if r.get('case_id') == case and r.get('setup_id') == 'orin_nx_hailo10_01']
    if len(rows) != 1:
        raise ValueError('diagnostic_original_result_not_unique:' + model)
    row = rows[0]
    session = row['native_tensorrt']['sessions']['part2:tensorrt']
    io = row['deployment_contract']['hailo_io_contracts']['part1']
    if len(io['hef_outputs']) != 1 or len(io['canonical_outputs']) != 1 or len(session['inputs']) != 1:
        raise ValueError('diagnostic_single_boundary_contract_required')
    if session['inputs'][0]['name'] != io['canonical_outputs'][0]:
        raise ValueError('diagnostic_engine_boundary_name_mismatch')
    roots = [run / f'models/{model}/benchmark_set/legacy_suite',
             run / f'native_producers/hailo10h/{model}/benchmark_set'] + [Path(p) for p in artifact_roots]
    # A binding embeds local_artifact_verification with the same artifact
    # roles. That proof is not a second binding and has no binding digest.
    bindings = [obj for obj in _objects(row)
                if obj.get('schema') == 'onnx-splitpoint/native-split-quality-binding'
                and isinstance(obj.get('artifacts'), dict)
                and 'part1_runtime' in obj['artifacts'] and 'engine' in obj['artifacts']]
    declared = row.get('run_cfg', {}).get('native_split_quality_binding')
    if declared:
        rel = str(declared).split('/suite/', 1)[-1]
        candidates = [Path(str(declared))] + [r / rel for r in roots]
        for candidate in candidates:
            if candidate.is_file():
                bindings.append(retained.read_json(candidate))
    # Existing per-case receipt/binding files are admissible only when their
    # engine + HEF roles match this exact Generic row. No successful Native row
    # is invented or needed and no generic recursive "best match" is selected.
    for root in roots:
        for candidate in (root / case).glob('**/*quality_binding*.json'):
            if candidate.is_file():
                bindings.append(retained.read_json(candidate))
    if not bindings:
        remote_binding = _load_recorded_remote_binding(run, model, row)
        if remote_binding is not None:
            bindings.append(remote_binding)
    matches = {}
    for binding in bindings:
        artifacts = binding.get('artifacts', {})
        if artifacts.get('engine', {}).get('sha256') != session.get('engine_sha256'):
            continue
        if artifacts.get('build_part2_onnx', {}).get('sha256') != session.get('source_model_sha256'):
            continue
        matches[json.dumps(binding, sort_keys=True)] = binding
    if len(matches) > 1:
        raise ValueError('diagnostic_generic_binding_ambiguous')
    binding = next(iter(matches.values()), None)
    if binding is not None:
        from onnx_splitpoint_tool.native_split_quality import validate_native_split_quality_binding
        verified, reason = validate_native_split_quality_binding(binding, verification_mode='portable',
            expected_identity={'model': model, 'case': case, 'setup_id': row['setup_id'],
                               'backend': 'hailo10h_to_trt', 'task': 'detection', 'precision': session.get('precision')})
        if verified is None:
            raise ValueError('diagnostic_generic_binding_invalid:' + str(reason))
        binding = verified
    hef_digest = io.get('artifact_sha256') or io.get('hef_sha256')
    if binding is not None:
        observed = binding['artifacts']['part1_runtime']['sha256']
        if hef_digest and _digest(hef_digest, 'hef') != observed:
            raise ValueError('diagnostic_generic_hef_receipt_conflict')
        hef_digest = observed
    # The row's direct HEF digest is a legitimate execution receipt for plain
    # Generic runs. Older unbound rows remain explicitly incomplete.
    hef_digest = _digest(hef_digest, 'generic_hef_receipt')
    hef = retained.exact_file([Path(io['artifact'])] + [r / case / 'hailo/hailo10/part1/compiled.hef' for r in roots], model + ':part1_hef', hef_digest)
    p1 = retained.exact_file([r / case / row['part1'] for r in roots], model + ':reference_part1')
    reference_part1_binding = {'source': 'generic_row', 'sha256': str(io.get('source_onnx_sha256') or '')}
    if reference_part1_binding['sha256']:
        if _digest(reference_part1_binding['sha256'], 'source_part1') != sha256(p1):
            raise ValueError('diagnostic_reference_part1_receipt_mismatch')
    else:
        # Reuse the existing source/HEF/cache receipt validator. A same-shaped
        # local Float-P1 is not sufficient evidence for the recorded HEF.
        from onnx_splitpoint_tool.hailo_backend import _load_valid_hailo_receipt
        receipt = _load_valid_hailo_receipt(hef, source_onnx_sha256=sha256(p1))
        if receipt is None or str(receipt.get('hw_arch')) not in {'hailo10', 'hailo10h'}:
            raise ValueError('diagnostic_reference_part1_exact_hef_receipt_not_available')
        reference_part1_binding = {'source': 'validated_existing_hailo_build_receipt',
                                  'sha256': receipt['source_onnx_sha256']}
    p2_expected = binding['artifacts']['source_part2_onnx']['sha256'] if binding else ''
    p2 = retained.exact_file([r / case / row['part2'] for r in roots], model + ':reference_part2', p2_expected)
    image_declared = Path(row['run_cfg']['image'])
    image_relative = str(image_declared).split('/suite/', 1)[-1]
    image_candidates = [image_declared] + [r / image_relative for r in roots]
    for root in artifact_roots:
        image_candidates += list(Path(root).rglob(image_declared.name))
    image = retained.exact_file(image_candidates, model + ':original_generic_image')
    remote_artifacts = {role: {'path': session[path], 'sha256': _digest(session[digest], role)}
        for role, path, digest in [('engine', 'engine', 'engine_sha256'),
          ('bridge', 'source_model', 'source_model_sha256'),
          ('receipt', 'engine_build_receipt_path', 'engine_build_receipt_file_sha256')]}
    for role, item in remote_artifacts.items():
        if '/_onnx_splitpoint_cache/' not in item['path']:
            raise ValueError('diagnostic_exact_persistent_cache_identity_missing:' + role)
    request = {**FLAGS, 'model': model, 'case': case, 'setup_id': row['setup_id'],
        'capture_path': 'current_generated_generic', 'artifact_binding_source': 'current_generic_row_and_exact_receipts',
        'original_result_sha256': sha256(result_path), 'original_result': row,
        'original_image_path': str(image_declared), 'image_identity_source': 'original_run_cfg_path_and_unique_local_bytes',
        'image_sha256': sha256(image), 'hef_sha256': hef_digest, 'remote_artifacts': remote_artifacts,
        'runtime_io': io, 'engine_io': session, 'selected_image_count': 1,
        'reference_part1_sha256': sha256(p1), 'reference_part2_sha256': sha256(p2),
        'reference_part1_binding': reference_part1_binding,
        'reference_external_data': {'part1': external_closure(p1), 'part2': external_closure(p2)},
        'generic_quality_binding': binding, 'runtime_limit_s': 90, 'term_grace_s': 5,
        'compiler_invoked': False, 'energy_invoked': False,
        'historical_remote_source': row.get('generic_runtime_source_binding') or {'status': 'not_available'}}
    if diagnostic_image is not None:
        image = Path(diagnostic_image['path']).resolve(strict=True)
        if sha256(image) != diagnostic_image['sha256']:
            raise ValueError('diagnostic_fixed_image_changed')
        request.update(diagnostic_image_id=diagnostic_image['id'], diagnostic_image_path=str(image),
            image_sha256=diagnostic_image['sha256'], image_identity_source='explicit_fixed4_development_manifest',
            images_usage='opened_development_diagnostic_not_holdout')
    return request, {'hef': hef, 'image': image, 'part1': p1, 'part2': p2}


_PREPARE_STAGE = retained.prepare_stage

def prepare_stage(destination, source, cases):
    import onnx
    from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime, assert_generated_hailo_layout_current
    destination = _PREPARE_STAGE(destination, source, cases)
    # The diagnostic uses exactly the generated sessions and the same package
    # closure as production. Transfer completes before any remote imports.
    for request, files in cases:
        folder = destination / request['model']
        generated = Path(write_runner_skeleton_onnxruntime(str(folder)))
        assert_generated_hailo_layout_current(generated)
        model = onnx.load(str(files['part1']), load_external_data=False)
        # An IO metadata declaration only, never executed as a continuation.
        interface = onnx.helper.make_model(onnx.helper.make_graph([], 'metadata_only_not_executable', list(model.graph.input), list(model.graph.output)))
        onnx.save(interface, folder / 'part1_interface.onnx')
        request['source_closure'] = {
            'generated_runner': {'path': request['model'] + '/run_split_onnxruntime.py', 'sha256': sha256(generated)},
            'quality_feed': {'path': 'tool/onnx_splitpoint_tool/runners/native_split_quality_runtime.py',
                             'sha256': sha256(destination / 'tool/onnx_splitpoint_tool/runners/native_split_quality_runtime.py')},
            'package_backend': {'path': 'tool/onnx_splitpoint_tool/runners/backends/hailo_backend.py',
                                'sha256': sha256(destination / 'tool/onnx_splitpoint_tool/runners/backends/hailo_backend.py')},
        }
        generator_sources = ast.literal_eval(next(node.value for node in ast.parse(generated.read_text()).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'GENERATED_SOURCE_FILES' for target in node.targets)))
        if generator_sources['package_adapter']['source_sha256'] != request['source_closure']['package_backend']['sha256']:
            raise ValueError('diagnostic_generator_and_staged_package_source_differ')
        request['interface_sha256'] = sha256(folder / 'part1_interface.onnx')
        write_json(folder / 'request.json', request)
    return destination


def main(argv=None):
    from onnx_splitpoint_tool.release_identity import VERSION
    if VERSION != '2.83':
        raise ValueError('installed_v283_required')
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--run-dir' not in arguments and not any(arg.startswith('--run-dir=') for arg in arguments) and not set(arguments).intersection({'--help', '-h'}):
        raise ValueError('diagnostic_explicit_current_run_dir_required')
    global _REMOTE_CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ssh', default='nx@192.168.0.145')
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--remote-python', default='/home/nx/venvs/hailo10/bin/python')
    options, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+', options.ssh) or not options.remote_python.startswith('/'):
        raise ValueError('diagnostic_remote_configuration_invalid')
    user, host = options.ssh.split('@', 1)
    _REMOTE_CONFIG = ({'user': user, 'host': host, 'port': options.port}, options.remote_python)
    previous_resolver, previous_stage = retained.resolve_case, retained.prepare_stage
    retained.resolve_case = resolve_case
    retained.prepare_stage = prepare_stage
    try:
        return retained.main(argv)
    finally:
        retained.resolve_case, retained.prepare_stage = previous_resolver, previous_stage


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
