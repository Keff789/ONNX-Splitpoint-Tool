"""Strict, bounded numeric diagnostics shared by Hailo boundary probes.

No inference selection, repairs, quality binding, or benchmark admission lives
here. Arrays and their semantic declarations must agree; shape alone is not
an output-name, layout, or quantization proof.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

FLAGS = {'diagnostic_only': True, 'counts_as_benchmark': False, 'claim_eligible': False}
MAX_BYTES = 64 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def tensor_stats(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'biuf':
        raise ValueError('diagnostic_non_numeric_or_object_array')
    finite = np.isfinite(a)
    v = a[finite]
    return {'shape': list(a.shape), 'strides': list(a.strides), 'dtype': str(a.dtype),
            'c_contiguous': bool(a.flags.c_contiguous), 'nbytes': a.nbytes,
            'finite_count': int(finite.sum()), 'nonfinite_count': int((~finite).sum()),
            'nonfinite_example_indices': np.argwhere(~finite)[:8].tolist(),
            'min': float(v.min()) if v.size else None, 'max': float(v.max()) if v.size else None}


def compare_named(actual, reference, mapping, *, actual_layout, reference_layout):
    """Compare only an explicit bijective name map in the same declared layout."""
    reasons = []
    if set(mapping) != set(actual) or set(mapping.values()) != set(reference) or len(set(mapping.values())) != len(mapping):
        reasons.append('output_name_mapping_not_bijective')
    if not actual_layout or actual_layout != reference_layout:
        reasons.append('layout_declaration_mismatch')
    rows = {}
    if not reasons:
        for name, target in mapping.items():
            a, b = np.asarray(actual[name]), np.asarray(reference[target])
            row = {'actual': tensor_stats(a), 'reference': tensor_stats(b)}
            if a.shape != b.shape:
                row['reason'] = 'tensor_shape_mismatch'
            elif not np.isfinite(a).all() or not np.isfinite(b).all():
                row['reason'] = 'nonfinite_tensor'
            else:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                row.update(max_abs=float(d.max()) if d.size else 0., mean_abs=float(d.mean()) if d.size else 0.,
                           example_indices=np.argwhere(d > 1e-2)[:8].tolist(),
                           numerically_close=bool(np.allclose(a, b, rtol=1e-3, atol=1e-2)))
            rows[name] = row
    return {**FLAGS, 'status': 'rejected' if reasons else 'compared', 'reason_codes': reasons,
            'semantic_pass': False, 'note': 'Numeric proximity does not qualify a pipeline.', 'tensors': rows}


def dequant_reference(value, quantization, *, outputs_dequantized=False):
    a = np.asarray(value)
    if np.issubdtype(a.dtype, np.floating):
        if outputs_dequantized is not True:
            raise ValueError('float_output_quantization_semantics_missing')
        return a.copy(), {'dequantization_count': 0, 'source': 'runtime_float_already_dequantized'}
    if a.dtype not in (np.dtype('uint8'), np.dtype('int8')) or outputs_dequantized:
        raise ValueError('quantization_dtype_or_double_dequantization_mismatch')
    scale = np.asarray(quantization.get('scale'), dtype=np.float32)
    zp = np.asarray(quantization.get('zero_point'), dtype=np.float32)
    if not np.isfinite(scale).all() or not np.isfinite(zp).all() or np.any(scale <= 0):
        raise ValueError('invalid_quantization_parameters')
    if scale.size > 1 or zp.size > 1:
        axis = quantization.get('axis')
        if not isinstance(axis, int) or not -a.ndim <= axis < a.ndim:
            raise ValueError('quantization_channel_axis_missing_or_invalid')
        axis %= a.ndim
        if scale.size not in (1, a.shape[axis]) or zp.size not in (1, a.shape[axis]):
            raise ValueError('quantization_channel_axis_size_mismatch')
        shape = [1] * a.ndim
        shape[axis] = a.shape[axis]
        if scale.size > 1:
            scale = scale.reshape(shape)
        if zp.size > 1:
            zp = zp.reshape(shape)
    return (a.astype(np.float32) - zp) * scale, {
        'dequantization_count': 1, 'source': 'calculated_from_bound_quant_info', 'axis': quantization.get('axis')}


def bn6_stats(value):
    a = np.asarray(value)
    out = {'tensor': tensor_stats(a), **FLAGS}
    if a.ndim != 3 or a.shape[-1] != 6:
        return {**out, 'status': 'invalid_bn6_shape'}
    boxes, scores, classes = a[..., :4], a[..., 4], a[..., 5]
    score_bad = (~np.isfinite(scores)) | (scores < 0) | (scores > 1)
    box_bad = (~np.isfinite(boxes).all(axis=-1)) | (boxes[..., 2] < boxes[..., 0]) | (boxes[..., 3] < boxes[..., 1])
    class_bad = (~np.isfinite(classes)) | (classes < 0) | (classes >= 80) | (classes != np.floor(classes))
    return {**out, 'status': 'invalid' if score_bad.any() or box_bad.any() or class_bad.any() else 'range_valid_only',
            'box_channels': tensor_stats(boxes), 'score_channels': tensor_stats(scores), 'class_channel': tensor_stats(classes),
            'score_violation_count': int(score_bad.sum()), 'unordered_box_count': int(box_bad.sum()),
            'class_violation_count': int(class_bad.sum()), 'score_violation_example_indices': np.argwhere(score_bad)[:8].tolist()}


def dump_packet(directory, arrays, metadata, *, max_bytes=MAX_BYTES):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if max_bytes <= 0 or max_bytes > MAX_BYTES:
        raise ValueError('diagnostic_byte_budget_invalid')
    normalized = {str(k): np.asarray(v) for k, v in arrays.items()}
    stats = {key: tensor_stats(value) for key, value in normalized.items()}
    # Uncompressed payload budget is enforced before allocation/writing; the
    # archive's actual bytes are checked too. Never truncate an array.
    existing = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if existing + sum(a.nbytes for a in normalized.values()) + 1024 * len(normalized) > max_bytes:
        raise ValueError('diagnostic_byte_budget_exceeded')
    path = directory / 'raw_tensors.npz'
    np.savez(path, **normalized)
    report = {**metadata, **FLAGS, 'arrays': stats, 'npz_file': path.name, 'npz_sha256': sha256(path),
              'raw_data_truncated': False}
    write_json(directory / 'diagnostic.json', report)
    if sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) > max_bytes:
        path.unlink()
        write_json(directory / 'diagnostic.json', {**FLAGS, 'status': 'incomplete', 'error': 'diagnostic_byte_budget_exceeded'})
        raise ValueError('diagnostic_byte_budget_exceeded')
    return report


def load_packet(directory):
    directory = Path(directory)
    report = json.loads((directory / 'diagnostic.json').read_text())
    if any(report.get(k) is not v for k, v in FLAGS.items()):
        raise ValueError('diagnostic_eligibility_flags_invalid')
    if report.get('npz_file') != 'raw_tensors.npz':
        raise ValueError('diagnostic_npz_path_invalid')
    path = directory / 'raw_tensors.npz'
    if sha256(path) != report.get('npz_sha256'):
        raise ValueError('diagnostic_npz_hash_mismatch')
    import zipfile
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist()) > MAX_BYTES or len(z.infolist()) > 64:
            raise ValueError('diagnostic_npz_expanded_budget_exceeded')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if set(arrays) != set(report['arrays']):
        raise ValueError('diagnostic_npz_keys_mismatch')
    for key, value in arrays.items():
        summary = tensor_stats(value)
        if any(summary[k] != report['arrays'][key][k] for k in ('dtype', 'shape', 'nbytes')):
            raise ValueError('diagnostic_npz_tensor_identity_mismatch')
    return report, arrays
