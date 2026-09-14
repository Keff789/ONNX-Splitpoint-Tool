"""Private, receipt-bound Hailo model diagnosis; no scientific PASS synthesis.

This module imports neither SDK nor TensorFlow. The executable delegates the
actual compilation to hailo_backend and keeps comparison inputs local.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Mapping


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def file_identity(path):
    from .hailo_backend import _bare_file_sha256
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('diagnostic_input_not_nonempty_file:' + str(path))
    return {'path': str(path), 'sha256': _bare_file_sha256(path), 'size_bytes': path.stat().st_size}


def check_identity(identity):
    actual = file_identity(identity['path'])
    if actual['sha256'] != identity['sha256'] or actual['size_bytes'] != identity['size_bytes']:
        raise ValueError('diagnostic_frozen_input_changed:' + identity['path'])
    return Path(actual['path'])


def fixed_images(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError('diagnostic_requires_exactly_16_fixed_development_images')
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError('diagnostic_image_id_required')
        label = row.get('label')
        if type(label) is not int or label < 0:
            raise ValueError('diagnostic_nonnegative_integer_label_required')
        identity = file_identity(row['path'])
        if row.get('sha256') and row['sha256'] != identity['sha256']:
            raise ValueError('diagnostic_image_hash_mismatch:' + row['id'])
        result.append({**identity, 'id': row['id'], 'label': label})
    if len({r['id'] for r in result}) != 16 or len({r['path'] for r in result}) != 16:
        raise ValueError('diagnostic_image_ids_and_paths_must_be_distinct')
    return result


def calibration_records(directory, manifest=None):
    """Freeze the files actually selected by the existing calibration scanner.

    Manifest hash alone does not bind an arbitrary directory. This first
    image-based B500 diagnosis requires 500 individually identifiable images;
    opaque NPY/NPZ batches need their original producer evidence first.
    """
    from .hailo_backend import _scan_calib_dir
    directory = Path(directory).resolve(strict=True)
    selected = list(_scan_calib_dir(directory, recursive=True, limit=500).get('items') or [])
    if len(selected) != 500 or any(Path(p).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} for p in selected):
        raise ValueError('diagnostic_requires_500_selected_individual_calibration_images')
    records = [{**file_identity(p), 'relative_path': str(Path(p).relative_to(directory)).replace('\\', '/')}
               for p in selected]
    if manifest:
        payload = json.loads(check_identity(manifest).read_text(encoding='utf-8'))
        items = payload.get('items') if isinstance(payload, Mapping) else None
        if not isinstance(items, list) or len(items) != 500 or payload.get('item_count', 500) != 500:
            raise ValueError('diagnostic_baseline_manifest_requires_500_explicit_items')
        by_relative = {}
        for row in items:
            if not isinstance(row, Mapping) or not row.get('relative_path') or row['relative_path'] in by_relative:
                raise ValueError('diagnostic_baseline_manifest_items_ambiguous')
            by_relative[row['relative_path']] = row
        if set(by_relative) != {r['relative_path'] for r in records}:
            raise ValueError('diagnostic_manifest_does_not_bind_selected_calibration_files')
        for row in records:
            declared = by_relative[row['relative_path']]
            digest = str(declared.get('sha256') or '').removeprefix('sha256:')
            if digest and digest != row['sha256']:
                raise ValueError('diagnostic_manifest_calibration_image_changed:' + row['relative_path'])
            if declared.get('size_bytes') is not None and declared['size_bytes'] != row['size_bytes']:
                raise ValueError('diagnostic_manifest_calibration_image_size_changed:' + row['relative_path'])
    return records


def prepare_request(*, cpu_hef, source_onnx, compiler_onnx, calibration_dir,
                    images_json, venv, timeout_s, calibration_manifest=None):
    """Resolve one existing CPU receipt, never choose an artifact by filename."""
    from .hailo_backend import _load_valid_hailo_receipt, _hailo_receipt_path, _calibration_identity
    cpu = file_identity(cpu_hef)
    source, compiler = file_identity(source_onnx), file_identity(compiler_onnx)
    receipt = _load_valid_hailo_receipt(Path(cpu['path']), source_onnx_sha256=source['sha256'])
    if not receipt:
        raise ValueError('diagnostic_valid_cpu_build_receipt_required')
    if receipt['compiler_onnx_sha256'] != compiler['sha256']:
        raise ValueError('diagnostic_compiler_graph_does_not_match_cpu_receipt')
    recipe = receipt['cache_payload']
    if receipt['hw_arch'] not in {'hailo10', 'hailo10h'}:
        raise ValueError('diagnostic_first_model_requires_hailo10h_cpu_baseline')
    if 'mobilenet_v3_large' not in receipt['net_name'].lower():
        raise ValueError('diagnostic_first_model_requires_mobilenet_v3_large')
    required = {'optimization_level': 1, 'calibration_count': 500,
                'requested_calibration_count': 500, 'calibration_batch_size': 8,
                'integrity': 'relaxed'}
    for key, value in required.items():
        if recipe.get(key) != value or (type(value) is int and type(recipe.get(key)) is not int):
            raise ValueError('diagnostic_cpu_recipe_not_B500_balanced_opt1_batch8_relaxed:' + key)
    if recipe.get('start_nodes') or recipe.get('end_nodes') or recipe.get('activation_part1_sha256'):
        raise ValueError('diagnostic_cpu_baseline_must_be_full_graph')
    if receipt['preprocessing_contract'].get('task') != 'classification':
        raise ValueError('diagnostic_classification_input_contract_required')
    if type(timeout_s) is not int or not 600 <= timeout_s <= 86400:
        raise ValueError('diagnostic_explicit_model_budget_required_600_to_86400_seconds')
    directory = Path(calibration_dir).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('diagnostic_calibration_directory_required')
    manifest = file_identity(calibration_manifest) if calibration_manifest else None
    if manifest:
        identity = 'manifest:' + manifest['sha256']
    else:
        # Ignore an unrelated inherited manifest without changing parent env.
        if os.environ.get('ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST'):
            raise ValueError('diagnostic_calibration_manifest_must_be_explicit')
        identity = _calibration_identity(directory, strict=False)
    if identity != receipt['calibration_identity']:
        raise ValueError('diagnostic_calibration_identity_differs_from_cpu_baseline')
    interpreter = Path(venv).expanduser().absolute()
    if interpreter.is_dir():
        interpreter = interpreter / 'bin' / 'python'
    # Keep lexical venv path; resolving bin/python erases the selected venv.
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError('diagnostic_vendor_interpreter_missing_or_not_executable')
    return {
        'schema': 'hailo_model_diagnostic_request_v27934', 'model': 'mobilenet_v3_large',
        'role': 'full', 'family': 'hailo10h', 'preset': 'balanced',
        'cpu_hef': cpu, 'cpu_receipt': file_identity(_hailo_receipt_path(Path(cpu['path']))),
        'source_onnx': source, 'compiler_onnx': compiler,
        'calibration_dir': str(directory), 'calibration_manifest': manifest,
        'calibration_records': calibration_records(directory, manifest),
        'calibration_identity': identity, 'cpu_build_receipt': receipt,
        'images': fixed_images(images_json), 'images_usage': 'opened_development_diagnostic_not_holdout',
        'venv_python': str(interpreter), 'timeout_s': timeout_s,
        'publication': 'private_only', 'central_force_modified': False,
        'compute_device': 'gpu', 'gpu_selector': '0', 'prepared_unix_s': time.time(),
    }


def validate_request(request):
    if request.get('schema') != 'hailo_model_diagnostic_request_v27934':
        raise ValueError('diagnostic_request_schema_invalid')
    for role in ('cpu_hef', 'cpu_receipt', 'source_onnx', 'compiler_onnx'):
        check_identity(request[role])
    for row in request['images']:
        check_identity(row)
    if len(request['images']) != 16:
        raise ValueError('diagnostic_requires_16_fixed_images')
    from .hailo_backend import _load_valid_hailo_receipt, _calibration_identity
    receipt = _load_valid_hailo_receipt(Path(request['cpu_hef']['path']))
    if receipt != request['cpu_build_receipt']:
        raise ValueError('diagnostic_cpu_receipt_changed_or_invalid')
    recipe = receipt['cache_payload']
    if (request.get('model') != 'mobilenet_v3_large' or request.get('role') != 'full'
            or request.get('family') != 'hailo10h' or request.get('preset') != 'balanced'
            or receipt['hw_arch'] not in {'hailo10', 'hailo10h'}
            or 'mobilenet_v3_large' not in receipt['net_name'].lower()
            or recipe.get('optimization_level') != 1
            or recipe.get('calibration_count') != 500
            or recipe.get('requested_calibration_count') != 500
            or recipe.get('calibration_batch_size') != 8
            or recipe.get('integrity') != 'relaxed'
            or recipe.get('start_nodes') or recipe.get('end_nodes')
            or recipe.get('activation_part1_sha256')
            or request['source_onnx']['sha256'] != receipt['source_onnx_sha256']
            or request['compiler_onnx']['sha256'] != receipt['compiler_onnx_sha256']):
        raise ValueError('diagnostic_frozen_baseline_recipe_or_graph_mismatch')
    if len({row['id'] for row in request['images']}) != 16:
        raise ValueError('diagnostic_fixed_image_ids_not_unique')
    manifest = request.get('calibration_manifest')
    if manifest:
        check_identity(manifest)
        identity = 'manifest:' + manifest['sha256']
    else:
        identity = _calibration_identity(Path(request['calibration_dir']), strict=False)
    if identity != request['calibration_identity'] or identity != receipt['calibration_identity']:
        raise ValueError('diagnostic_calibration_identity_changed')
    if calibration_records(request['calibration_dir'], manifest) != request.get('calibration_records'):
        raise ValueError('diagnostic_selected_calibration_files_changed')
    if request.get('publication') != 'private_only' or request.get('compute_device') != 'gpu':
        raise ValueError('diagnostic_private_gpu_request_required')
    if type(request.get('timeout_s')) is not int or not 600 <= request['timeout_s'] <= 86400:
        raise ValueError('diagnostic_model_budget_invalid')
    return receipt


def builder_arguments(request, work):
    receipt = validate_request(request)
    payload = receipt['cache_payload']
    return {
        # Compile the exact already adjusted CPU build graph, never rerun a
        # potentially different preparation recipe over the original graph.
        'onnx_path': request['compiler_onnx']['path'], 'backend': 'venv',
        'hw_arch': receipt['hw_arch'], 'net_name': receipt['net_name'],
        'outdir': str(Path(work) / 'build'), 'net_input_shapes': payload.get('net_input_shapes'),
        'fixup': False, 'add_conv_defaults': False,
        'disable_rt_metadata_extraction': payload.get('disable_rt_metadata_extraction', True),
        'opt_level': payload['optimization_level'], 'calib_dir': request['calibration_dir'],
        'calib_count': payload['requested_calibration_count'],
        'calib_batch_size': payload['calibration_batch_size'],
        'extra_model_script': payload['extra_model_script'],
        'task': 'classification', 'preprocessing_contract': receipt['preprocessing_contract'],
        'force': True, 'keep_artifacts': True, 'publish_artifacts': False,
        'compute_device': 'gpu', 'gpu_selector': request['gpu_selector'],
        'compiler_context': request.get('compiler_context'),
        'wsl_venv_activate': str(Path(request['venv_python']).parent / 'activate'),
        'wsl_timeout_s': request['timeout_s'],
    }


def private_environment(request, work, parent=None):
    env = dict(os.environ if parent is None else parent)
    work = Path(work).resolve()
    for name in ('tmp', 'xdg_cache', 'cuda_cache', 'triton_cache', 'torch_cache', 'tfhub_cache', 'private_hef_cache', 'private_store', 'private_calibration'):
        (work / name).mkdir(parents=True, exist_ok=True)
    env.update({
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'TMPDIR': str(work / 'tmp'), 'XDG_CACHE_HOME': str(work / 'xdg_cache'),
        'CUDA_CACHE_PATH': str(work / 'cuda_cache'), 'TRITON_CACHE_DIR': str(work / 'triton_cache'),
        'TORCH_HOME': str(work / 'torch_cache'), 'TFHUB_CACHE_DIR': str(work / 'tfhub_cache'),
        'ONNX_SPLITPOINT_HAILO_CACHE_ROOT': str(work / 'private_hef_cache'),
        'ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT': str(work / 'private_store'),
        'ONNX_SPLITPOINT_HAILO_CACHE_ENABLED': '0', 'ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED': '0',
        'ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY': 'relaxed',
        'ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE': request['cpu_build_receipt']['calibration_storage'],
        'ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB': str(request['cpu_build_receipt']['calibration_memory_cap_bytes'] // (1024 * 1024)),
        'TF_NUM_INTEROP_THREADS': env.get('TF_NUM_INTEROP_THREADS', '2'),
        'TF_NUM_INTRAOP_THREADS': env.get('TF_NUM_INTRAOP_THREADS', '2'),
        'TF_FORCE_GPU_ALLOW_GROWTH': env.get('TF_FORCE_GPU_ALLOW_GROWTH', 'true'),
    })
    if request.get('calibration_manifest'):
        env['ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST'] = request['calibration_manifest']['path']
    else:
        env.pop('ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST', None)
    return env


def attributable_gpu_evidence(samples, phase_events, *, gpu_index):
    """Positive per-process SM activity, with its entire interval in optimize.

    VRAM/process presence and other processes never establish GPU execution.
    Unsupported pmon output or missed intervals remain unproven.
    """
    intervals = []
    started = None
    for row in phase_events:
        if row.get('phase') != 'optimize':
            continue
        stamp = row.get('monotonic_s')
        if row.get('event') == 'started' and isinstance(stamp, (int, float)):
            started = stamp
        elif row.get('event') in {'completed', 'failed'} and started is not None and isinstance(stamp, (int, float)):
            intervals.append((started, stamp))
            started = None
    evidence = []
    for row in samples:
        if row.get('gpu_index') != gpu_index or row.get('owned_compiler_descendant') is not True:
            continue
        if type(row.get('sm_utilization_percent')) not in (int, float) or not 0 < row['sm_utilization_percent'] <= 100:
            continue
        left, right = row.get('interval_start_monotonic_s'), row.get('observed_monotonic_s')
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (left, right)) or right <= left:
            continue
        if any(start <= left < right <= end for start, end in intervals):
            evidence.append(row)
    return {'status': 'pass' if evidence else 'gpu_execution_unproven',
            'source': 'nvidia_smi_pmon_per_process_SM_during_owned_DFC_optimize',
            'attributed_samples': evidence, 'optimization_intervals': intervals,
            'vram_alone_accepted': False}


def compare_classification_arrays(stages, *, labels, ids):
    """Recompute actual finite logits/top-k. No threshold or decoder correction."""
    import numpy as np
    if len(ids) != 16 or len(set(ids)) != 16 or len(labels) != 16:
        raise ValueError('diagnostic_comparison_requires_same_fixed_16_ids')
    report, arrays = {}, {}
    common_feed = None
    for name in ('source_float', 'build_float', 'cpu_hef', 'gpu_hef', 'full_precision', 'quantized_emulation'):
        stage = stages.get(name)
        if stage is None:
            report[name] = {'status': 'not_available'}
            continue
        if list(stage['ids']) != list(ids):
            raise ValueError('diagnostic_stage_image_order_mismatch:' + name)
        logits, feed = np.asarray(stage['logits']), np.asarray(stage['feed'])
        if logits.ndim != 2 or logits.shape[0] != 16 or logits.shape[1] < 5:
            raise ValueError('diagnostic_explicit_NC_classification_logits_required:' + name)
        if feed.ndim < 1 or feed.shape[0] != 16 or feed.dtype.hasobject:
            raise ValueError('diagnostic_actual_fixed16_feed_required:' + name)
        if common_feed is None:
            common_feed = feed
        elif feed.dtype != common_feed.dtype or feed.shape != common_feed.shape or not np.array_equal(feed, common_feed):
            raise ValueError('diagnostic_stage_feed_mismatch:' + name)
        finite = bool(np.isfinite(logits).all() and np.isfinite(feed).all())
        row = {'status': 'evaluated' if finite else 'invalid_numeric', 'shape': list(logits.shape),
               'dtype': str(logits.dtype), 'finite': finite, 'origin': stage.get('origin', 'not_available')}
        if finite:
            labels_array = np.asarray(labels)
            if np.any(labels_array >= logits.shape[1]) or labels_array.dtype.kind not in 'iu' or np.any(labels_array < 0):
                raise ValueError('diagnostic_label_class_axis_mismatch')
            top = np.argsort(-logits, axis=1, kind='stable')[:, :5]
            row.update(top1=top[:, 0].tolist(), top5=top.tolist(),
                top1_correct=int(np.sum(top[:, 0] == labels_array)),
                top5_correct=int(np.sum(np.any(top == labels_array[:, None], axis=1))),
                item_count=16)
            arrays[name] = logits
        report[name] = row
    comparisons = {}
    for first, second in (('source_float', 'build_float'), ('build_float', 'cpu_hef'), ('cpu_hef', 'gpu_hef'), ('quantized_emulation', 'gpu_hef')):
        if first in arrays and second in arrays:
            a, b = arrays[first], arrays[second]
            key = first + '_vs_' + second
            if a.shape != b.shape:
                comparisons[key] = {'status': 'shape_mismatch', 'left_shape': list(a.shape), 'right_shape': list(b.shape)}
            else:
                delta = b.astype(np.float64) - a.astype(np.float64)
                comparisons[key] = {'status': 'compared', 'max_abs_difference': float(np.max(np.abs(delta))),
                    'rms_difference': float(np.sqrt(np.mean(delta * delta))),
                    'top1_changed_count': int(np.sum(np.argmax(a, axis=1) != np.argmax(b, axis=1))),
                    'correct_count_delta': report[second]['top1_correct'] - report[first]['top1_correct']}
    return {'status': 'diagnostic_evaluated', 'stages': report, 'comparisons': comparisons,
            'quality_status': 'not_evaluated_against_campaign_gate', 'claim_eligible': False,
            'holdout': False, 'thresholds_changed': False, 'original_quality_decision_preserved': True}
