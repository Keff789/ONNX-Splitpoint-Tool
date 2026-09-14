#!/usr/bin/env python3
"""One original image per YOLO26/Hailo10 split, with existing artifacts only.

The management host resolves local HEF, image and ONNX reference files; only
HEF/image/code are uploaded. Exact TensorRT engine/bridge/receipt files are
copied from the recorded persistent device cache into private temporary
staging. No old temporary remote run, model build, or dataset upload is used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hailo_boundary_diagnostics_v27931 import FLAGS, MAX_BYTES, sha256, write_json, load_packet, dump_packet, compare_named, bn6_stats, dequant_reference, tensor_stats
from deepx_full_workflow_smoke_v27930 import run_transport
import deepx_full_output_probe_v27930 as transport

CASES = {'yolo26m': 'b398', 'yolo26s': 'b364'}
DEFAULT_RUN = 'complete_set_20260907_161614'


def read_json(path):
    return json.loads(Path(path).read_text())


def exact_file(candidates, role, expected=''):
    found = {}
    for candidate in candidates:
        p = Path(candidate).expanduser()
        if p.is_file():
            digest = sha256(p)
            if not expected or digest == expected.removeprefix('sha256:'):
                found.setdefault(digest, p.resolve())
    if len(found) != 1:
        raise ValueError('diagnostic_artifact_' + ('ambiguous:' if found else 'missing_or_hash_mismatch:') + role)
    return next(iter(found.values()))


def resolve_case(run, model, artifact_roots=(), diagnostic_image=None):
    run = Path(run).resolve()
    case = CASES[model]
    result_path = run / f'models/{model}/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json'
    rows = read_json(result_path)
    rows = [r for r in rows if r.get('case_id') == case and r.get('setup_id') == 'orin_nx_hailo10_01']
    if len(rows) != 1:
        raise ValueError('diagnostic_original_result_not_unique:' + model)
    row = rows[0]
    session = row['native_tensorrt']['sessions']['part2:tensorrt']
    if session.get('explicit_engine') is not True or session.get('cache_hit') is not True:
        raise ValueError('diagnostic_explicit_original_engine_missing')
    io = row['deployment_contract']['hailo_io_contracts']['part1']
    if len(io['hef_outputs']) != 1 or len(io['canonical_outputs']) != 1 or len(session['inputs']) != 1:
        raise ValueError('diagnostic_single_boundary_contract_required')
    roots = [run / f'models/{model}/benchmark_set/legacy_suite', run / f'native_producers/hailo10h/{model}/benchmark_set']
    roots += [Path(p) for p in artifact_roots]
    native = read_json(run / f'native_producers/hailo10h/{model}/benchmark_set/native_pipeline/{case}/hailo10h_to_trt/uint8_dequant_fp16/hailo10_native_fifo_e2e_results.json')
    image_declared = Path(row['run_cfg']['image'])
    image_relative = str(image_declared).split('/suite/', 1)[-1]
    image_candidates = [image_declared] + [r / image_relative for r in roots]
    for root in artifact_roots:
        image_candidates += list(Path(root).rglob(image_declared.name))
    image = exact_file(image_candidates, model + ':original_generic_image')
    hef = exact_file([Path(io['artifact'])] + [r / case / 'hailo/hailo10/part1/compiled.hef' for r in roots],
                     model + ':part1_hef', native['replay_artifact_verification']['hef_sha256'])
    p1 = exact_file([r / case / row['part1'] for r in roots], model + ':reference_part1')
    p2 = exact_file([r / case / row['part2'] for r in roots], model + ':reference_part2')
    remote_artifacts = {
        'engine': {'path': session['engine'], 'sha256': session['engine_sha256']},
        'bridge': {'path': session['source_model'], 'sha256': session['source_model_sha256']},
        'receipt': {'path': session['engine_build_receipt_path'], 'sha256': session['engine_build_receipt_file_sha256']}}
    for role, item in remote_artifacts.items():
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']) or '/_onnx_splitpoint_cache/' not in item['path']:
            raise ValueError('diagnostic_exact_persistent_cache_identity_missing:' + role)
    # HEF provenance identifies this exact split, not any same-shaped model.
    request = {**FLAGS, 'model': model, 'case': case, 'setup_id': row['setup_id'],
               'original_result_sha256': sha256(result_path), 'original_result': row,
               'original_image_path': str(image_declared), 'image_identity_source': 'original_run_cfg_path_and_unique_local_bytes',
               'image_sha256': sha256(image), 'hef_sha256': sha256(hef), 'remote_artifacts': remote_artifacts,
               'runtime_io': io, 'engine_io': session, 'selected_image_count': 1,
               'reference_part1_sha256': sha256(p1), 'reference_part2_sha256': sha256(p2),
               'runtime_limit_s': 90, 'term_grace_s': 5, 'compiler_invoked': False}
    if diagnostic_image is not None:
        selected = Path(diagnostic_image['path']).expanduser().resolve(strict=True)
        if not selected.is_file() or sha256(selected) != diagnostic_image['sha256']:
            raise ValueError('diagnostic_fixed_image_changed')
        # Original result/run_cfg provenance remains intact. Explicitly report
        # the separately frozen development input actually used for this pass.
        request.update(diagnostic_image_id=diagnostic_image['id'],
            diagnostic_image_path=str(selected), image_sha256=diagnostic_image['sha256'],
            image_identity_source='v27934_explicit_fixed4_development_manifest',
            images_usage='opened_development_diagnostic_not_holdout')
        image = selected
    return request, {'hef': hef, 'image': image, 'part1': p1, 'part2': p2}


def prepare_stage(destination, source, cases):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for root_name in ('scripts', 'onnx_splitpoint_tool'):
        for path in (Path(source) / root_name).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            target = destination / 'tool' / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for request, files in cases:
        folder = destination / request['model']
        folder.mkdir()
        shutil.copyfile(files['hef'], folder / 'part1.hef')
        shutil.copyfile(files['image'], folder / 'image.jpg')
        write_json(folder / 'request.json', request)
    return destination


def offline_reference(request, files, directory):
    """F and calculated D outside remote runtime; source graph is unmodified."""
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import onnx
    import onnxruntime as ort
    ort.disable_telemetry_events()
    from native_hailo10_trt_e2e_from_benchmarkset import _image_to_shape
    directory = Path(directory)
    report, arrays = load_packet(directory)
    if report.get('status') != 'captured':
        raise ValueError('diagnostic_capture_not_complete')
    bridge_file = directory / 'bound_bridge.onnx'
    if sha256(bridge_file) != request['remote_artifacts']['bridge']['sha256']:
        raise ValueError('diagnostic_bound_bridge_changed')
    for role in ('part1', 'part2'):
        if sha256(files[role]) != request['reference_' + role + '_sha256']:
            raise ValueError('diagnostic_reference_source_changed')
    for role, closure in request.get('reference_external_data', {}).items():
        for item in closure:
            member = Path(files[role]).parent / item['location']
            if not member.is_file() or sha256(member) != item['sha256']:
                raise ValueError('diagnostic_reference_external_data_changed:' + role)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    p1 = ort.InferenceSession(str(files['part1']), sess_options=opts, providers=['CPUExecutionProvider'])
    p2 = ort.InferenceSession(str(files['part2']), sess_options=opts, providers=['CPUExecutionProvider'])
    input_info = p1.get_inputs()[0]
    if request.get('capture_path') == 'current_generated_generic':
        if 'F_input_bound' not in arrays:
            raise ValueError('diagnostic_exact_generic_float_input_not_available')
        feed = arrays['F_input_bound']
        if tuple(feed.shape) != tuple(input_info.shape) or feed.dtype != np.float32:
            raise ValueError('diagnostic_generic_float_reference_input_contract_mismatch')
    else:
        feed = _image_to_shape(str(files['image']), tuple(input_info.shape), False, preprocess_mode='letterbox', task='detection')
    ref_p1 = dict(zip((v.name for v in p1.get_outputs()), p1.run(None, {input_info.name: feed})))
    if set(ref_p1) != {v.name for v in p2.get_inputs()}:
        raise ValueError('diagnostic_reference_cut_name_mismatch')
    ref_p2 = dict(zip((v.name for v in p2.get_outputs()), p2.run(None, ref_p1)))
    bridge = onnx.load(str(bridge_file), load_external_data=False)
    from onnx.external_data_helper import _get_all_tensors
    if any(t.external_data for t in _get_all_tensors(bridge)):
        declared_external = report.get('bridge_external_data')
        if not declared_external:
            raise ValueError('diagnostic_external_bridge_payload_unsupported')
        from hailo10_yolo26_boundary_probe_v282 import external_closure
        if external_closure(bridge_file) != declared_external:
            raise ValueError('diagnostic_bound_bridge_external_data_changed')
        bridge = onnx.load(str(bridge_file), load_external_data=True)
        onnx.external_data_helper.convert_model_from_external_data(bridge)
    input_name = request['engine_io']['inputs'][0]['name']
    produced = {out for node in bridge.graph.node for out in node.output}
    candidates = [input_name + suffix for suffix in ('__boundary_to_ncw', '__boundary_to_nchw', '__dequant_float')]
    # The bound graph may contain a dequant node followed by a layout node;
    # only the latter is the declared canonical boundary. Two different layout
    # roles remain ambiguous, even if their arrays happen to have equal sizes.
    matches = [name for name in candidates[:2] if name in produced]
    if not matches:
        matches = [name for name in candidates[2:] if name in produced]
    if len(matches) != 1:
        raise ValueError('diagnostic_bound_dequant_boundary_not_identified')
    dname = matches[0]
    # Run the original bound Part2 bridge on exactly the pinned TensorRT C
    # input. The original F comparison alone confounded P1 quantization with
    # a possible Part2 runtime difference. No decoder correction is applied.
    original_bridge_outputs = [value.name for value in bridge.graph.output]
    bridge_cpu = ort.InferenceSession(bridge.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])
    if [value.name for value in bridge_cpu.get_inputs()] != [input_name]:
        raise ValueError('diagnostic_bound_bridge_input_roles_ambiguous')
    bridge_p2 = dict(zip(original_bridge_outputs, bridge_cpu.run(original_bridge_outputs, {input_name: arrays['C_000']})))
    # This CPU execution is a calculated reference graph, never a TensorRT
    # intermediate observation. ONNX source bytes in the cache are preserved.
    bridge.graph.output.append(onnx.helper.make_tensor_value_info(dname, onnx.TensorProto.FLOAT, None))
    ds = ort.InferenceSession(bridge.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])
    logical = ds.run([dname], {input_name: arrays['C_000']})[0]
    quantization = report['runtime_io']['runtime_output_quantization']
    raw_name = report['stages']['B']['names'][0]
    dequantized, quant_details = dequant_reference(arrays['B_000'], quantization[raw_name],
                                                  outputs_dequantized=report['runtime_io'].get('outputs_dequantized', False))
    constants = {item.name: onnx.numpy_helper.to_array(item) for item in bridge.graph.initializer}
    scale = constants.get(input_name + '__dequant_scale_const')
    zero = constants.get(input_name + '__dequant_zp_const')
    observed = quantization[raw_name]
    parameter_match = bool(scale is not None and zero is not None
                           and np.array_equal(scale, np.asarray(observed['scale'], dtype=np.float32))
                           and np.array_equal(zero, np.asarray(observed['zero_point'], dtype=np.float32)))
    report['quantization_comparison'] = {**quant_details, 'hef_vs_bound_bridge_parameters_equal': parameter_match,
        'bound_bridge_scale': scale.tolist() if scale is not None else None,
        'bound_bridge_zero_point': zero.tolist() if zero is not None else None,
        'source': 'actual_loaded_HEF_vs_hash_bound_bridge_initializers', 'repair_applied': False}
    # Capture the exact reference input and compare its unscaled HWC bytes
    # with A. Input preparation differences remain visible independently of P1.
    actual_input = arrays['A_000']
    if actual_input.shape == (1, *actual_input.shape[1:]) and actual_input.ndim == 4:
        actual_input = actual_input[0]
    reference_hwc = np.transpose(feed[0], (1, 2, 0))
    if tuple(actual_input.shape) != tuple(reference_hwc.shape):
        raise ValueError('diagnostic_actual_input_layout_does_not_match_reference')
    report['input_reference_comparison'] = compare_named({'images': actual_input.astype(np.float32) / 255.0},
        {'images': reference_hwc}, {'images': 'images'}, actual_layout='HWC_RGB_norm', reference_layout='HWC_RGB_norm')
    comparisons = {
        'D_vs_reference_P1': compare_named({input_name: logical}, ref_p1, {input_name: input_name}, actual_layout='canonical_onnx', reference_layout='canonical_onnx'),
        'E_vs_reference_P2': compare_named({name: arrays[f'E_{i:03d}'] for i, name in enumerate(report['stages']['E']['names'])}, ref_p2,
                                         {name: name for name in ref_p2}, actual_layout='BN6', reference_layout='BN6')}
    observed_p2 = {name: arrays[f'E_{i:03d}'] for i, name in enumerate(report['stages']['E']['names'])}
    comparisons['E_vs_bound_bridge_P2_same_C'] = compare_named(
        observed_p2, bridge_p2, {name: name for name in bridge_p2},
        actual_layout='bound_engine_output', reference_layout='bound_engine_output')
    report['part2_same_feed_comparison'] = {
        'status': 'evaluated', 'feed_source': 'C_direct_pinned_trt_input_after_existing_prepare_inputs',
        'reference': 'actual_hash_bound_bridge_ONNX_CPU',
        'input_names': [input_name], 'output_names': original_bridge_outputs,
        'raw_outputs_corrected': False, 'first_divergence': 'requires_numeric_transition_review'}
    for i, (_, value) in enumerate(bridge_p2.items()):
        arrays[f'P2_same_C_{i:03d}'] = value
    # Strict original endpoint validator is applied to the unchanged raw output.
    from onnx_splitpoint_tool.native_output_endpoint import runtime_output_contract
    raw = {name: arrays[f'E_{i:03d}'] for i, name in enumerate(report['stages']['E']['names'])}
    endpoint = runtime_output_contract('detection', raw, raw_fallback=False,
                                       declared_contract={'stage': 'decoded_nms', 'output_format': 'bn6_detections'})
    arrays['D_000'] = logical
    report['logical_boundary_channels'] = {
        'channel_axis': 1, 'declaration_source': 'original_YOLO26_canonical_P1_contract',
        'box_channels_0_3': tensor_stats(logical[:, :4]),
        'score_channels_4_onwards': tensor_stats(logical[:, 4:]),
        'shape_alone_used_as_semantic_pass': False}

    arrays['F_input'] = feed
    for i, (name, value) in enumerate(ref_p1.items()):
        arrays[f'F_P1_{i:03d}'] = value
    for i, (name, value) in enumerate(ref_p2.items()):
        arrays[f'F_P2_{i:03d}'] = value
    report['stages']['D'] = {'observation': 'calculated_reference_from_bound_bridge_graph', 'direct_engine_observation': False, 'name': dname}
    report['stages']['F'] = {'observation': 'cpu_onnx_reference_same_declared_image_preparation', 'part1_names': list(ref_p1), 'part2_names': list(ref_p2)}
    report.update(capture_pass=True, endpoint_pass=endpoint.get('endpoint_contract_complete') is True,
                  endpoint_status='passed' if endpoint.get('endpoint_contract_complete') is True else 'failed',
                  endpoint_reason=str((endpoint.get('output_endpoint_attestation') or {}).get('reason') or 'endpoint_contract_incomplete'))
    report.update(comparisons=comparisons, original_endpoint_validation=endpoint, status='diagnostic_complete',
                  root_cause='not_established_requires_review_of_raw_transitions', regular_path_released=False,
                  runtime_fix_verified=False, fresh_central_quality_required=True)
    # Replace packet only after complete calculations; no stale mixed NPZ/JSON.
    (directory / 'raw_tensors.npz').unlink()
    return dump_packet(directory, arrays, report, max_bytes=30 * 1024 * 1024)


def remote_temp_valid(path):
    return re.fullmatch(r'/tmp/onnx-v27931-hailo26-[A-Za-z0-9]{10}', path) is not None


def collector_exit_code(summary):
    return 0 if summary.get('status') == 'diagnostic_complete' and summary.get('collection_complete') is True else 2


def fixed_four_images(path):
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError('diagnostic_requires_exactly_four_fixed_images')
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError('diagnostic_fixed_image_id_required')
        path = Path(row['path']).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError('diagnostic_fixed_image_missing')
        digest = sha256(path)
        if row.get('sha256') and row['sha256'] != digest:
            raise ValueError('diagnostic_fixed_image_changed')
        result.append({'id': row['id'], 'path': str(path), 'sha256': digest})
    if len({row['id'] for row in result}) != 4 or len({row['path'] for row in result}) != 4:
        raise ValueError('diagnostic_four_image_ids_and_paths_must_be_unique')
    return result


def _main_unlocked(argv=None, diagnostic_image=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path.home() / 'Models/EvaluationRuns' / DEFAULT_RUN)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--artifact-root', type=Path, action='append', default=[])
    parser.add_argument('--output-dir', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--ssh', default='nx@192.168.0.145')
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--remote-python', default='/home/nx/venvs/hailo10/bin/python')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--images-json', type=Path, help='four fixed development IDs/paths; run the existing probe sequentially')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+', args.ssh) or not args.remote_python.startswith('/'):
        raise ValueError('diagnostic_remote_configuration_invalid')
    if args.images_json:
        rows = fixed_four_images(args.images_json)
        command = ['--run-dir', str(args.run_dir), '--source-root', str(args.source_root),
            '--output-dir', str(args.output_dir), '--ssh', args.ssh, '--port', str(args.port),
            '--remote-python', args.remote_python]
        for root in args.artifact_root:
            command.extend(['--artifact-root', str(root)])
        if args.plan_only:
            command.append('--plan-only')
        attempts = []
        for row in rows:
            rc = _main_unlocked(command, diagnostic_image=row)
            attempts.append({'id': row['id'], 'sha256': row['sha256'], 'returncode': rc})
            if rc:
                break  # Retain first failing transition, no blind extra run.
        print(json.dumps({**FLAGS, 'fixed_image_count': 4, 'attempts': attempts,
            'collection_complete': len(attempts) == 4 and all(r['returncode'] == 0 for r in attempts),
            'capture_pass': len(attempts) == 4 and all(r['returncode'] == 0 for r in attempts),
            'endpoint_pass': False, 'endpoint_status': 'inspect_bound_per_case_reports',
            'regular_path_released': False, 'quality_status': 'not_evaluated'}, indent=2))
        return 0 if len(attempts) == 4 and all(row['returncode'] == 0 for row in attempts) else 2
    if args.plan_only:
        cases = [resolve_case(args.run_dir, model, args.artifact_root, diagnostic_image) for model in CASES]
        print(json.dumps({**FLAGS, 'cases': [{k: r[k] for k in ('model', 'case', 'setup_id', 'original_image_path', 'image_sha256', 'hef_sha256', 'remote_artifacts', 'runtime_limit_s')} for r, _ in cases], 'model_upload_count': 0, 'dataset_upload_count': 0}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='hailo10_yolo26_v27931_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_', dir=args.output_dir))
    summary = {**FLAGS, 'status': 'started', 'collection_complete': False, 'compiler_invoked': False, 'model_upload_count': 0, 'dataset_upload_count': 0}
    user, host = args.ssh.split('@', 1)
    remote = {'user': user, 'host': host, 'port': args.port}
    remote_dir = ''
    cleanup_complete = False
    try:
        cases = [resolve_case(args.run_dir, model, args.artifact_root, diagnostic_image) for model in CASES]
        for request, files in cases:
            write_json(output / 'requests' / (request['model'] + '.json'), request)
        with tempfile.TemporaryDirectory(prefix='onnx-v27931-hailo26-local-') as tmp:
            stage = prepare_stage(Path(tmp) / 'stage', args.source_root, cases)
            made = run_transport(transport.ssh_base(remote) + ['mktemp -d /tmp/onnx-v27931-hailo26-XXXXXXXXXX'], capture_output=True, text=True, timeout=30, check=True)
            remote_dir = made.stdout.strip()
            if not remote_temp_valid(remote_dir):
                raise ValueError('diagnostic_remote_directory_invalid')
            target = args.ssh + ':' + remote_dir + '/'
            began = time.monotonic()
            with (output / 'transfer.log').open('w') as log:
                run_transport(transport.scp_base(remote) + ['-r', *map(str, stage.iterdir()), target], timeout=300, check=True, stdout=log, stderr=subprocess.STDOUT)
            summary['transfer_s'] = time.monotonic() - began
            command = shlex.join(['timeout', '--signal=TERM', '--kill-after=5s', '220s', 'env', 'ORT_DISABLE_TELEMETRY=1', 'PYTHONNOUSERSITE=1', args.remote_python, '-B',
                                  remote_dir + '/tool/scripts/hailo10_yolo26_boundary_worker_v27931.py', '--stage', remote_dir])
            with (output / 'remote.log').open('w') as log:
                result = run_transport(transport.ssh_base(remote) + [command], timeout=240, stdout=log, stderr=subprocess.STDOUT)
            summary['remote_returncode'] = result.returncode
            with (output / 'collection.log').open('w') as log:
                run_transport(transport.scp_base(remote) + ['-r', target + 'results', str(output / 'results')], timeout=300, check=True, stdout=log, stderr=subprocess.STDOUT)
            remote_summary = read_json(output / 'results/supervision.json')
            cleanup_complete = remote_summary.get('cleanup_complete') is True
            if any(remote_summary.get(k) is not v for k, v in FLAGS.items()) or not cleanup_complete or result.returncode:
                raise ValueError('diagnostic_remote_capture_failed')
            began = time.monotonic()
            for request, files in cases:
                directory = output / 'results' / request['model']
                write_json(directory / 'request.json', request)
                try:
                    offline_reference(request, files, directory)
                except Exception as exc:
                    captured_report, _ = load_packet(directory)
                    captured_report.update(status='diagnostic_incomplete', capture_pass=True,
                        endpoint_pass=False, endpoint_status='not_available',
                        reference_error=type(exc).__name__ + ': ' + str(exc))
                    for stage_name in ('D', 'F'):
                        captured_report['stages'][stage_name].update(status='not_available',
                            reason=type(exc).__name__ + ': ' + str(exc))
                    write_json(directory / 'diagnostic.json', captured_report)
                    raise
            case_reports = {request['model']: read_json(output / 'results' / request['model'] / 'diagnostic.json') for request, _ in cases}
            summary.update(capture_pass=True,
                endpoint_pass=all(report.get('endpoint_pass') is True for report in case_reports.values()),
                cases={model: {'capture_pass': report.get('capture_pass') is True,
                              'endpoint_pass': report.get('endpoint_pass') is True,
                              'endpoint_status': report.get('endpoint_status', 'not_available')}
                       for model, report in case_reports.items()})
            summary.update(status='diagnostic_complete', collection_complete=True, offline_compare_s=time.monotonic() - began,
                           root_cause='open_pending_raw_transition_review', regular_path_released=False)
    except Exception as exc:
        summary.update(status='incomplete', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        if remote_temp_valid(remote_dir) and cleanup_complete:
            try:
                removed = run_transport(transport.ssh_base(remote) + [shlex.join(['rm', '-rf', '--', remote_dir])], timeout=30, capture_output=True)
                summary['temporary_remote_directory_removed'] = removed.returncode == 0
            except Exception as exc:
                summary['cleanup_error'] = str(exc)
        else:
            summary['preserved_remote_directory'] = remote_dir
        summary['exit_code'] = collector_exit_code(summary)
        write_json(output / 'collection_summary.json', summary)
        print('DIAGNOSTIC_ZIP=' + str(transport.make_archive(output)), flush=True)
    return summary['exit_code']


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--plan-only' in arguments or '--help' in arguments or '-h' in arguments:
        return _main_unlocked(arguments)
    # Same exclusion as the normal workflow and compiler diagnostics; all
    # four image passes remain sequential under one held platform lock.
    from hailo_model_build_probe_v27934 import interlock
    with interlock():
        return _main_unlocked(arguments)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
