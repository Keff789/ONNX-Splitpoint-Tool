#!/usr/bin/env python3
"""Owned process supervision and unmeasured Hailo10 A/B/C/E observations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ['ORT_DISABLE_TELEMETRY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hailo_boundary_diagnostics_v27931 import FLAGS, sha256, write_json, dump_packet, bn6_stats
from deepx_full_workflow_smoke_worker_v27930 import supervised_run


def active_measurements(proc_root=Path('/proc')):
    """Read-only guard; a probe never stops somebody else's measurement."""
    active = []
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        basenames = {Path(item.decode(errors='replace')).name for item in command if item}
        if (b'--energy-workload-only' in command or basenames.intersection({
                'native_hailo10_trt_e2e_from_benchmarkset.py',
                'native_hailo10_trt_fifo_from_benchmarkset.py',
                'native_full_baseline_eval_runner.py', 'benchmark_suite.py'})):
            active.append(path.parent.name)
    return active


def capture(request, folder, output):
    if request.get('capture_path') == 'current_generated_generic':
        from hailo10_yolo26_generic_capture_v282 import capture as capture_generic
        return capture_generic(request, folder, output)
    import numpy as np
    import native_hailo10_trt_e2e_from_benchmarkset as native
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, item in request['remote_artifacts'].items():
        source = Path(item['path'])
        if not source.is_file() or sha256(source) != item['sha256']:
            raise ValueError('diagnostic_exact_remote_artifact_missing_or_changed:' + role)
        target = folder / ('bound_' + role + source.suffix)
        shutil.copyfile(source, target)
        if sha256(target) != item['sha256']:
            raise ValueError('diagnostic_staged_artifact_changed:' + role)
        files[role] = target
    for name, expected in [('part1.hef', request['hef_sha256']), ('image.jpg', request['image_sha256'])]:
        if sha256(folder / name) != expected:
            raise ValueError('diagnostic_staged_identity_mismatch:' + name)
    io = request['runtime_io']
    if io.get('quantized_inputs') is not True:
        raise ValueError('diagnostic_original_uint8_input_contract_required')
    opts = {'hef_path': str(folder / 'part1.hef'), 'hw_arch': 'hailo10h', 'runtime_api': 'infer_model',
            'quantized_inputs': True, 'quantized_outputs': io['quantized_outputs'], 'hotloop': True, 'copy_outputs': True,
            'canonical_input_slot_names': io['canonical_inputs'], 'canonical_output_slot_names': io['canonical_outputs']}
    backend = native.HailoBackend(strict=True, **opts)
    prepared, trt = None, None
    old = native._STRICT_SPLIT_BOUNDARY
    try:
        prepared = backend.prepare(native.RunCfg(model_path=folder / 'part1.hef', options=opts), folder / 'hailo_artifacts')
        runtime = prepared.handle.session
        actual_io = runtime.describe_io()
        # Equal sizes do not allow output-name or axis guessing.
        for key in ('hef_outputs', 'canonical_outputs', 'output_aliases', 'runtime_output_shapes', 'runtime_output_format'):
            if actual_io.get(key) != io.get(key):
                raise ValueError('diagnostic_runtime_contract_mismatch:' + key)
        canonical_name = io['canonical_outputs'][0]
        shape = io['runtime_output_shapes'][canonical_name]
        native._STRICT_SPLIT_BOUNDARY = {'name': canonical_name, 'runtime_name': io['hef_outputs'][0],
                                        'shape': shape, 'dtype': io['runtime_output_format']}
        quant = native._exact_hailo10_output_quantization(prepared) if io['quantized_outputs'] else {}
        trt = native.NativeTRT(files['engine'])
        expected_inputs = request['engine_io']['inputs']
        expected_outputs = request['engine_io']['outputs']
        if trt.inputs != [row['name'] for row in expected_inputs] or trt.outputs != [row['name'] for row in expected_outputs]:
            raise ValueError('diagnostic_engine_input_or_output_order_mismatch')
        for row in expected_inputs + expected_outputs:
            if tuple(trt.shapes[row['name']]) != tuple(row['shape']):
                raise ValueError('diagnostic_engine_shape_mismatch:' + row['name'])
            declared_dtype = {'tensor(uint8)': 'uint8', 'tensor(float)': 'float32', 'tensor(float16)': 'float16'}.get(row.get('type'))
            if declared_dtype and str(trt.dtypes[row['name']]) != declared_dtype:
                raise ValueError('diagnostic_engine_dtype_mismatch:' + row['name'])
        inputs = native._make_input(prepared, True, image=str(folder / 'image.jpg'), preprocess_mode='letterbox', task='detection')
        captured = {}
        begin = time.monotonic()
        mapped = native._capture_raw_hailo10_sample(runtime, inputs, diagnostic_capture=captured)
        name, boundary = native._pick_hailo_output(mapped, trt)
        trt.prepare_inputs({name: boundary})
        arrays = {}
        stages = {}
        for stage, tensors, observation in [('A', captured['prepared_inputs'], 'direct_runtime_prepared_input'),
                                             ('B', captured['raw_outputs'], 'direct_physical_hef_named_output'),
                                             ('C', trt.host_in, 'direct_pinned_trt_input_after_existing_prepare_inputs')]:
            stages[stage] = {'names': list(tensors), 'observation': observation}
            for i, (key, value) in enumerate(tensors.items()):
                arrays[f'{stage}_{i:03d}'] = np.array(value, copy=True)
        outputs = trt.run_prepared()
        stages['E'] = {'names': list(outputs), 'observation': 'direct_trt_raw_output_before_host_correction'}
        for i, (key, value) in enumerate(outputs.items()):
            arrays[f'E_{i:03d}'] = np.array(value, copy=True)
        stages['D'] = {'observation': 'pending_management_reference_graph', 'direct_engine_observation': False}
        stages['F'] = {'observation': 'pending_management_cpu_reference'}
        # Copy only the small bound graph/receipt for independent reconstruction.
        if files['bridge'].stat().st_size + files['receipt'].stat().st_size > 2 * 1024 * 1024:
            raise ValueError('diagnostic_bridge_contract_budget_exceeded')
        shutil.copyfile(files['bridge'], output / 'bound_bridge.onnx')
        shutil.copyfile(files['receipt'], output / 'bound_engine_receipt.json')
        return dump_packet(output, arrays, {**FLAGS, 'status': 'captured', 'model': request['model'], 'case': request['case'],
                            'runtime_io': actual_io, 'quant_info': quant, 'output_aliases': io['output_aliases'],
                            'physical_output_storage_before_copy': captured['raw_output_storage'],
                            'raw_output_checks': {key: bn6_stats(value) for key, value in outputs.items()},
                            'stages': stages, 'runtime_s': time.monotonic() - begin,
                            'regular_path_released': False, 'quality_binding_created': False}, max_bytes=20 * 1024 * 1024)
    finally:
        native._STRICT_SPLIT_BOUNDARY = old
        if trt is not None:
            trt.close()
        if prepared is not None:
            backend.cleanup(prepared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--capture-model', choices=['yolo26m', 'yolo26s'])
    args = parser.parse_args(argv)
    stage = args.stage.resolve()
    results = stage / 'results'
    results.mkdir(exist_ok=True)
    if args.capture_model:
        model = args.capture_model
        output = results / model
        output.mkdir(exist_ok=True)
        # Bound a runaway native/library console or dump file independently of
        # the 90-second whole-process-tree deadline enforced by the supervisor.
        resource.setrlimit(resource.RLIMIT_FSIZE, (70 * 1024 * 1024, 70 * 1024 * 1024))
        try:
            request = json.loads((stage / model / 'request.json').read_text())
            capture(request, stage / model, output)
            return 0
        except Exception as exc:
            write_json(output / 'capture_error.json', {**FLAGS, 'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
            return 2
    processes = {}
    for model in ('yolo26m', 'yolo26s'):
        (results / model).mkdir(exist_ok=True)
        measurements = active_measurements()
        if measurements:
            write_json(results / 'supervision.json', {**FLAGS, 'status': 'blocked', 'cleanup_complete': True,
                       'error': 'diagnostic_active_measurement_detected', 'active_process_ids': measurements})
            return 2
        # Sequential: never add device parallelism or overlap an energy run.
        process = supervised_run([sys.executable, '-B', str(Path(__file__).resolve()), '--stage', str(stage), '--capture-model', model],
                                 cwd=stage, log_path=results / model / 'runtime.log', timeout=90, grace=5)
        processes[model] = process
    complete = all(p.get('cleanup_complete') is True for p in processes.values())
    ok = complete and all(p.get('returncode') == 0 and not p.get('timed_out') and not p.get('cancelled') for p in processes.values())
    write_json(results / 'supervision.json', {**FLAGS, 'status': 'captured' if ok else 'failed', 'processes': processes, 'cleanup_complete': complete})
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
