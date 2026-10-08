"""Offline contract and exact-decoder regression tests; no model execution."""
import copy
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as ndp
from onnx_splitpoint_tool.runners.harness import yolo


def heads(mode="logits", *, crowded=False):
    outputs = {}
    rng = np.random.default_rng(20261005)
    for stride in (32, 8, 16):  # Deliberately reverse vendor output order.
        size = 640 // stride
        array = np.full((1, 3, size, size, 85), -20 if mode == "logits" else 0, np.float32)
        flat = array.reshape(-1, 85)
        if mode == "objcls_activated":
            flat[:, :4] = -20
        count = 1100 if crowded and stride == 8 else 23
        for i, row_index in enumerate(np.linspace(0, len(flat)-1, count, dtype=int)):
            row = flat[row_index]
            row[:4] = rng.uniform(-2, 2, 4) if mode != "activated" else rng.uniform(0, 1, 4)
            row[4] = 4 if mode == "logits" else 0.9
            row[5+(i % 80)] = 4 if mode == "logits" else 0.9
            if i % 7 == 0:  # Argmax must retain the first tied class.
                row[5] = row[5+(i % 80)]
        # Exactly-threshold combined score, including a negative-adjacent ULP.
        flat[1, :4] = 0 if mode != "activated" else .5
        flat[1, 4:] = 0 if mode == "logits" else .5
        flat[2, 4:] = np.nextafter(np.float32(0 if mode == "logits" else .5), np.float32(-1))
        outputs[f"vendor_stride_{stride}"] = array
    return outputs


def contract(outputs, mode="logits", schema_version=2):
    value = ndp.build_frozen_postprocess_contract(
        model_id=yolo.YOLOV7_PAPER_MODEL_ID, model_sha256=yolo.YOLOV7_PAPER_ONNX_SHA256,
        outputs=outputs, input_hw=[640, 640], original_wh=[853, 479],
    )
    model_bound = yolo.build_yolov7_decoder_contract(
        model_id=yolo.YOLOV7_PAPER_MODEL_ID, model_sha256=yolo.YOLOV7_PAPER_ONNX_SHA256,
        activation_mode=mode, schema_version=schema_version,
    )
    value.update(model_bound_decoder_contract=model_bound,
                 model_bound_decoder_contract_sha256=model_bound['decoder_contract_sha256'],
                 multiscale_activation_mode=mode)
    return reseal(value)


def reseal(value):
    value.pop('contract_sha256', None)
    value['invariant_identity'] = ndp.frozen_postprocess_invariant_identity(value)
    value['invariant_contract_sha256'] = ndp.canonical_json_sha256(value['invariant_identity'])
    value['contract_sha256'] = ndp.canonical_json_sha256(value)
    return value


@pytest.mark.parametrize('mode', ['logits', 'activated', 'objcls_activated'])
@pytest.mark.parametrize('schema_version', [1, 2])
@pytest.mark.parametrize('layout', ['canonical', 'channels_last'])
def test_sparse_canonical_detections_equal_dense_exactly(mode, schema_version, layout):
    outputs = heads(mode)
    if layout == 'channels_last':
        outputs = {name: arr[0].transpose(1, 2, 0, 3).reshape(arr.shape[2], arr.shape[3], 255)
                   for name, arr in outputs.items()}
    frozen = contract(outputs, mode, schema_version)
    dense = ndp.FrozenDetectionPostprocessor(frozen).process(outputs)
    sparse = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse').process(outputs)
    assert sparse == dense
    assert sparse['detection_count'] > 0


def test_empty_and_detection_limit_and_ties_are_exact():
    for crowded in (False, True):
        outputs = heads(crowded=crowded)
        if not crowded:
            for a in outputs.values(): a.fill(-20)
        frozen = contract(outputs)
        dense = ndp.FrozenDetectionPostprocessor(frozen).process(outputs)
        sparse = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse').process(outputs)
        assert sparse == dense
        assert sparse['detection_count'] == (300 if crowded else 0)


@pytest.mark.parametrize('mode', ['logits', 'activated', 'objcls_activated'])
@pytest.mark.parametrize('channel', [0, 5, 84])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_low_objectness_record_cannot_be_hidden(mode, channel, value):
    outputs = heads(mode)
    frozen = contract(outputs, mode)
    dense = ndp.FrozenDetectionPostprocessor(frozen)
    sparse = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse')
    outputs['vendor_stride_8'][0, 2, 40, 40, channel] = value
    for decoder in (dense, sparse):
        with pytest.raises(ValueError, match='head_nonfinite'):
            decoder.process(outputs)
        assert decoder.completed_count == 0


@pytest.mark.parametrize('mode', ['bound_dense', 'bound_exact_sparse'])
def test_bound_process_has_no_source_file_io(mode, monkeypatch):
    outputs = heads()
    decoder = ndp.FrozenDetectionPostprocessor(contract(outputs), runtime_mode=mode)
    before = decoder.assert_implementation_unchanged()
    def forbidden(*args, **kwargs):raise AssertionError('Source file I/O inside frame processing')
    with monkeypatch.context() as m:
        m.setattr(ndp, '_implementation_artifacts', forbidden)
        m.setattr(ndp, 'file_sha256', forbidden)
        m.setattr(Path, 'open', forbidden)
        for _ in range(2):decoder.process(outputs)
    assert decoder.assert_implementation_unchanged() == before
    assert decoder.completed_count == 2


@pytest.mark.parametrize('mutation', ['nested_anchor', 'geometry_resealed', 'mode'])
def test_bound_contract_mutations_rejected_even_if_resealed(mutation):
    outputs = heads()
    decoder = ndp.FrozenDetectionPostprocessor(contract(outputs), runtime_mode='bound_exact_sparse')
    if mutation == 'nested_anchor':
        decoder.contract['model_bound_decoder_contract']['anchors_by_stride'][0]['anchors_wh'][0][0] += 1
    elif mutation == 'geometry_resealed':
        decoder.contract['original_wh'] = [640, 640]
        reseal(decoder.contract)
    else:
        decoder.runtime_mode = 'bound_dense'
    with pytest.raises(ndp.FrozenPostprocessError, match='bound_contract_mutated'):
        decoder.process(outputs)
    assert decoder.completed_count == 0


def test_bound_state_does_not_alias_callers_contract():
    outputs = heads()
    frozen = contract(outputs)
    decoder = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse')
    expected = decoder.process(outputs)
    frozen['original_wh'][0] += 1
    frozen['model_bound_decoder_contract']['anchors_by_stride'][0]['anchors_wh'][0][0] += 1
    assert decoder.process(outputs) == expected


def test_changed_source_bytes_same_path_size_and_mtime_fail_then_rebind(tmp_path, monkeypatch):
    source = tmp_path/'source.py';source.write_bytes(b'first bytes')
    initial_stat = source.stat()
    real_artifacts = ndp._implementation_artifacts()
    def artifacts():
        value = copy.deepcopy(real_artifacts)
        value['native_detection_postprocess']['sha256'] = ndp.file_sha256(source)
        return value
    monkeypatch.setattr(ndp, '_implementation_artifacts', artifacts)
    outputs = heads();frozen = contract(outputs)
    decoder = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse')
    old = decoder.assert_implementation_unchanged()
    source.write_bytes(b'other bytes')
    os.utime(source, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
    assert source.stat().st_mtime_ns == initial_stat.st_mtime_ns
    assert source.stat().st_size == initial_stat.st_size
    with pytest.raises(ndp.FrozenPostprocessError, match='implementation_changed_during_run'):
        decoder.assert_implementation_unchanged()
    with pytest.raises(ndp.FrozenPostprocessError, match='implementation_mismatch'):
        ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse')
    rebound = ndp.FrozenDetectionPostprocessor(contract(outputs), runtime_mode='bound_exact_sparse')
    assert rebound.assert_implementation_unchanged() != old


def test_mode_selection_and_bad_inputs_never_fall_back():
    outputs = heads();frozen = contract(outputs)
    assert ndp.native_full_postprocess_runtime_mode(frozen) == 'bound_exact_sparse'
    for changed in ({'model_family': 'yolo26'}, {'source_contract_family': 'decoded_pre_nms'},
                    {'model_bound_decoder_contract': None}):
        assert ndp.native_full_postprocess_runtime_mode(dict(frozen, **changed)) == 'bound_dense'
    with pytest.raises(ndp.FrozenPostprocessError, match='runtime_mode_invalid'):
        ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='auto')
    decoder = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse')
    with pytest.raises(ndp.FrozenPostprocessError, match='tensor_signature_mismatch'):
        decoder.process({k: v.astype(np.float16) for k,v in outputs.items()})
    with pytest.raises(ndp.FrozenPostprocessError, match='original_wh_mismatch'):
        decoder.process(outputs, original_wh=[640,640])
    assert decoder.completed_count == 0


@pytest.mark.parametrize('mode', ['logits', 'activated', 'objcls_activated'])
@pytest.mark.parametrize('schema_version', [1, 2])
def test_threshold_adjacent_scores_candidate_order_and_nms_ties(mode, schema_version):
    outputs = heads(mode)
    for array in outputs.values():
        array.fill(-20 if mode == 'logits' else 0)
    rows = outputs['vendor_stride_8'].reshape(-1, 85)
    rows[:4, :4] = 0 if mode != 'activated' else .5
    rows[:4, 4:7] = 0 if mode == 'logits' else .5
    if mode == 'logits':
        # Adjacent raw logits around zero round to the same sigmoid. These
        # values actually straddle the rounded 0.5 activation in both rules.
        rows[1, 4] = -np.float32(1e-6)
        rows[2, 4] = np.float32(1e-6)
    else:
        rows[1, 4] = np.nextafter(np.float32(.5), np.float32(0))
        rows[2, 4] = np.nextafter(np.float32(.5), np.float32(1))
    # Equal scores, first-class argmax ties, and overlapping same-class boxes
    # force the shared NMS tie rule to matter to the completed output.
    rows[3, :4] = rows[0, :4]
    rows[3, 2:4] = 6 if mode != 'activated' else 1
    rows[0, 2:4] = rows[3, 2:4]
    rows[3, 0] = -80 if mode != 'activated' else 0
    frozen = contract(outputs, mode, schema_version)
    model_bound = frozen['model_bound_decoder_contract']
    _, normalized, _ = yolo._normalize_multiscale_outputs(list(outputs.values()))
    dense_candidates = yolo._decode_multiscale_head(
        normalized, (640, 640), .25,
        activation_mode=mode, decoder_contract=model_bound,
    )
    sparse_candidates = yolo.ExactSparseYoloV7Decoder(model_bound).decode(normalized)
    assert dense_candidates.scores.size == 3
    assert dense_candidates.scores[0] == .25
    assert dense_candidates.scores[1] > .25
    assert dense_candidates.scores[2] == .25
    assert dense_candidates.class_ids.tolist() == [0, 0, 0]
    for field in ('boxes_xyxy', 'scores', 'class_ids'):
        np.testing.assert_array_equal(
            getattr(sparse_candidates, field), getattr(dense_candidates, field),
        )
    dense = ndp.FrozenDetectionPostprocessor(frozen).process(outputs)
    sparse = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse').process(outputs)
    assert sparse == dense
    assert 0 < sparse['detection_count'] < 3


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('layout', ['canonical', 'nchw', 'nhwc', 'hwac'])
def test_supported_input_dtypes_and_strided_layouts_preserve_results(dtype, layout):
    outputs = {name: array.astype(dtype) for name, array in heads().items()}
    if layout == 'nchw':
        outputs = {name: array.transpose(0, 1, 4, 2, 3).reshape(1, 255, array.shape[2], array.shape[3])
                   for name, array in outputs.items()}
    elif layout == 'nhwc':
        outputs = {name: array.transpose(0, 2, 3, 1, 4).reshape(1, array.shape[2], array.shape[3], 255)
                   for name, array in outputs.items()}
    elif layout == 'hwac':
        outputs = {name: array[0].transpose(1, 2, 0, 3) for name, array in outputs.items()}
    frozen = contract(outputs)
    dense = ndp.FrozenDetectionPostprocessor(frozen).process(outputs)
    sparse = ndp.FrozenDetectionPostprocessor(frozen, runtime_mode='bound_exact_sparse').process(outputs)
    assert sparse == dense


def test_runtime_source_closure_detects_same_mtime_changes_and_missing_files(tmp_path, monkeypatch):
    source = tmp_path / 'loaded_product.py'
    source.write_bytes(b'first bytes')
    initial_stat = source.stat()
    name = 'onnx_splitpoint_tool._runtime_source_test'
    module = types.ModuleType(name)
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, name, module)
    runner = tmp_path / 'runner.py'
    runner.write_text('runner')
    before = ndp.native_full_runtime_source_hashes(runner)
    assert before[str(source)] == ndp.file_sha256(source)
    assert before[str(runner)] == ndp.file_sha256(runner)
    assert str(Path(ndp.__file__).absolute()) in before
    later = types.ModuleType('onnx_splitpoint_tool._unrelated_later_import')
    later.__file__ = str(tmp_path / 'unrelated.py')
    monkeypatch.setitem(sys.modules, later.__name__, later)
    assert ndp.native_full_runtime_source_hashes(bound_paths=before) == before
    source.write_bytes(b'other bytes')
    os.utime(source, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
    after = ndp.native_full_runtime_source_hashes(bound_paths=before)
    assert after.keys() == before.keys()
    assert after[str(source)] != before[str(source)]
    source.unlink()
    with pytest.raises(ndp.FrozenPostprocessError, match='runtime_source_unreadable'):
        ndp.native_full_runtime_source_hashes(bound_paths=before)
