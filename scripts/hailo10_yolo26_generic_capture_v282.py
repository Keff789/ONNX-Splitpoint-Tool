"""Unmeasured observation of the actual generated Generic session/feed path."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from hailo_boundary_diagnostics_v27931 import FLAGS, sha256, dump_packet, bn6_stats, tensor_stats


def verify_source_closure(request, stage):
    stage = Path(stage).resolve()
    for role, item in request['source_closure'].items():
        path = (stage / item['path']).resolve()
        if not path.is_relative_to(stage) or not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError('diagnostic_remote_source_mismatch:' + role)


def load_generated_runner(path):
    spec = importlib.util.spec_from_file_location('current_generic_diagnostic_runner', path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


def capture(request, folder, output):
    import numpy as np
    from hailo10_yolo26_boundary_probe_v282 import external_closure
    folder = Path(folder).resolve(); output = Path(output)
    stage = folder.parent
    output.mkdir(parents=True, exist_ok=True)
    verify_source_closure(request, stage)
    for name, expected in [('part1.hef', request['hef_sha256']), ('image.jpg', request['image_sha256']),
                           ('part1_interface.onnx', request['interface_sha256'])]:
        if sha256(folder / name) != expected:
            raise ValueError('diagnostic_staged_identity_mismatch:' + name)
    files = {}
    for role, item in request['remote_artifacts'].items():
        source = Path(item['path'])
        if not source.is_file() or sha256(source) != item['sha256']:
            raise ValueError('diagnostic_exact_remote_artifact_missing_or_changed:' + role)
        files[role] = source
    bridge_external = external_closure(files['bridge'])
    if sum(item['size_bytes'] for item in bridge_external) + files['bridge'].stat().st_size > 64 * 1024 * 1024:
        raise ValueError('diagnostic_exact_bridge_external_data_exceeds_transfer_budget_not_available')
    sys.path.insert(0, str(stage / 'tool'))
    runner_path = folder / 'run_split_onnxruntime.py'
    runner = load_generated_runner(runner_path)
    source_proof = runner.generic_runtime_source_binding()
    for role in ('hailo_session', 'tensor_adapter', 'generic_feed'):
        if source_proof[role]['source_sha256'] != request['source_closure']['generated_runner']['sha256']:
            raise ValueError('diagnostic_imported_generic_source_mismatch:' + role)
    if source_proof['quality_feed']['source_sha256'] != request['source_closure']['quality_feed']['sha256']:
        raise ValueError('diagnostic_imported_quality_feed_source_mismatch')
    os.environ['ONNX_SPLITPOINT_ARTIFACT_POLICY'] = 'cache_verify_only'
    runtime, trt = None, None
    began = time.monotonic()
    try:
        io = request['runtime_io']; session = request['engine_io']
        if io.get('runtime_api') != 'infer_model' or io.get('quantized_inputs') is not True:
            raise ValueError('diagnostic_generic_infer_model_native_input_contract_required')
        runtime = runner.HailoInferModelSession(folder / 'part1.hef', quantized_inputs=True,
            quantized_outputs=io['quantized_outputs'], hotloop=True, copy_outputs=True,
            onnx_model_path=folder / 'part1_interface.onnx',
            canonical_input_slot_names=io['canonical_inputs'], canonical_output_slot_names=io['canonical_outputs'])
        actual_io = runtime.describe_io()
        for key in ('hef_inputs', 'hef_outputs', 'canonical_inputs', 'canonical_outputs', 'output_aliases',
                    'runtime_input_shapes', 'runtime_output_shapes', 'onnx_input_shapes', 'onnx_output_shapes',
                    'runtime_input_format', 'runtime_output_format', 'runtime_output_quantization', 'runtime_input_quantization'):
            if actual_io.get(key) != io.get(key):
                raise ValueError('diagnostic_runtime_contract_mismatch:' + key)
        cfg = request['original_result']['run_cfg']
        precision = session.get('precision') or cfg['native_trt_precision']
        workspace = int(cfg.get('native_trt_workspace_mb', 4096))
        verified = runner._verify_explicit_native_trt_engine_receipt(source_onnx=files['bridge'],
            engine_path=files['engine'], receipt_path=files['receipt'], expected_precision=precision,
            expected_workspace_mb=workspace)
        trt = runner.NativeTRTSession('part2', files['bridge'], precision=precision,
            workspace_mb=workspace, allow_build=False, explicit_engine_path=files['engine'],
            explicit_build_receipt_path=files['receipt'], verified_build_receipt=verified)
        for actual, expected, role in [(trt.get_inputs(), session['inputs'], 'input'), (trt.get_outputs(), session['outputs'], 'output')]:
            if [(v.name, list(v.shape), v.type) for v in actual] != [(v['name'], v['shape'], v['type']) for v in expected]:
                raise ValueError('diagnostic_engine_' + role + '_contract_mismatch')
        name = io['canonical_inputs'][0]
        input_shape = tuple(io['onnx_input_shapes'][name])
        if len(input_shape) != 4 or input_shape[:2] != (1, 3):
            raise ValueError('diagnostic_declared_image_input_not_supported')
        # Use the recorded effective scale, never the best output from an auto
        # trial. The same generated loader/quantizer prepares A and float F.
        scale = request['original_result']['benchmark_input_policy']['image_scale']
        if scale != 'norm':
            raise ValueError('diagnostic_yolo26_effective_preprocessing_not_available')
        contract = runner._canonical_image_preprocessing_contract('detection', input_shape[-2:])
        float_input = runner._load_image_as_nchw(folder / 'image.jpg', target_hw=input_shape[-2:],
            dtype=np.dtype('float32'), scale=scale, letterbox=True, preprocessing_contract=contract)
        if float_input is None:
            raise ValueError('diagnostic_input_image_unavailable')
        prepared_input = runner._prepare_hailo_image_input(runtime, float_input, name)
        mapped = runtime.infer({name: prepared_input})
        raw = {key: runtime._binding_output(runtime._hot_binding, key).get_buffer() for key in runtime._hef_output_names}
        arrays = {'F_input_bound': np.array(float_input, copy=True)}
        stages = {}
        for label, tensors, observation in [
            ('A', runtime._hot_input_buffers, 'actual_generated_session_prepared_input'),
            ('B', raw, 'physical_HEF_output_before_mapping_unchanged_by_adapter')]:
            stages[label] = {'names': list(tensors), 'observation': observation}
            for index, (key, value) in enumerate(tensors.items()):
                arrays[f'{label}_{index:03d}'] = np.array(value, copy=True)
        binding = request.get('generic_quality_binding')
        args = SimpleNamespace()
        if binding:
            bridge = binding['native_trt_meta'].get('uint8_cast_bridge', {})
            pre = binding['preselection']
            args._native_split_quality_binding = binding
            args._native_split_boundary_layout = bridge.get('boundary_layout')
            if bridge.get('input_name') != pre['boundary_tensor_name'] or bridge.get('boundary_layout', {}).get('effective') != pre.get('boundary_layout'):
                raise ValueError('diagnostic_sealed_boundary_layout_or_input_mismatch')
            args._native_split_boundary_input_name = pre['boundary_tensor_name']
            args._native_split_boundary_transform = pre.get('boundary_transform', '')
            args._native_split_boundary_dequant_scale = pre.get('dequant_scale')
            args._native_split_boundary_dequant_zero_point = pre.get('dequant_zero_point')
        events = []
        feed = {}
        for item in session['inputs']:
            key = item['name']
            if key not in mapped:
                raise ValueError('diagnostic_canonical_boundary_name_missing:' + key)
            target_dtype = trt._tensor_dtypes[key]
            feed[key] = runner.prepare_generic_native_boundary_input(key, mapped[key], item['shape'], target_dtype,
                args=args, _boundary_mode=cfg['boundary_mode'], _record_boundary_event=events.append,
                _cached_boundary_buffer=lambda name, shape, dtype: np.empty(shape, dtype=dtype))
        # NativeTRTSession receives this exact contiguous host feed; all shape /
        # dtype checks already match, so its run method makes no adaptation.
        stages['C'] = {'names': list(feed), 'observation': 'actual_generated_native_trt_host_feed'}
        for index, value in enumerate(feed.values()):
            arrays[f'C_{index:03d}'] = np.array(value, copy=True)
        values = trt.run(None, feed)
        outputs = dict(zip(trt.output_names, values))
        stages['E'] = {'names': list(outputs), 'observation': 'actual_generated_trt_output_before_postprocessing'}
        for index, value in enumerate(outputs.values()):
            arrays[f'E_{index:03d}'] = np.array(value, copy=True)
        stages['D'] = {'observation': 'pending_bound_bridge_ORT_reference', 'direct_engine_observation': False}
        stages['F'] = {'observation': 'pending_true_float_P1_P2_reference'}
        shutil.copyfile(files['bridge'], output / 'bound_bridge.onnx')
        shutil.copyfile(files['receipt'], output / 'bound_engine_receipt.json')
        for item in bridge_external:
            target = output / item['location']; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files['bridge'].parent / item['location'], target)
            if sha256(target) != item['sha256']:
                raise ValueError('diagnostic_bridge_external_data_changed')
        # Device deserialization/execution cannot retroactively change a receipt.
        for role, item in request['remote_artifacts'].items():
            if sha256(files[role]) != item['sha256']:
                raise ValueError('diagnostic_artifact_changed_during_capture:' + role)
        return dump_packet(output, arrays, {**FLAGS, 'status': 'captured', 'capture_pass': True,
            'endpoint_pass': False, 'endpoint_status': 'not_evaluated', 'model': request['model'], 'case': request['case'],
            'runtime_io': actual_io, 'generic_runtime_source_binding': source_proof,
            'source_closure_verified': True, 'source_path': 'current_generated_generic',
            'bridge_external_data': bridge_external, 'preprocessing_contract': contract,
            'physical_output_storage_before_copy': {k: tensor_stats(v) for k, v in raw.items()},
            'boundary_feed_events': events, 'quant_info': actual_io['runtime_output_quantization'],
            'output_aliases': actual_io['output_aliases'], 'raw_output_checks': {k: bn6_stats(v) for k, v in outputs.items()},
            'stages': stages, 'runtime_s': time.monotonic() - began, 'regular_path_released': False,
            'quality_binding_created': False, 'compiler_invoked': False, 'energy_invoked': False}, max_bytes=30 * 1024 * 1024)
    finally:
        if trt is not None:
            trt.close()
        if runtime is not None:
            runtime.close()
