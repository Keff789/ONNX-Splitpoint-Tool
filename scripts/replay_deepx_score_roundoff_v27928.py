#!/usr/bin/env python3
"""Replay the captured Float32 tensor through the real frozen decoder and NMS.

No inference, SSH, compiler, cache write, campaign or quality acceptance. The
independent comparison only changes permitted score edges on an in-memory copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--original-wh', nargs=2, type=int, default=[500, 335])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from onnx_splitpoint_tool import native_detection_postprocess as pp
    from onnx_splitpoint_tool.runners.harness import yolo
    from onnx_splitpoint_tool.release_identity import VERSION
    file = args.npz.expanduser().resolve(strict=True)
    before_file = hashlib.sha256(file.read_bytes()).hexdigest()
    with np.load(file, allow_pickle=False) as archive:
        if archive.files != ['tensor_000']:
            raise ValueError('expected_one_recorded_tensor')
        original = archive['tensor_000']
    if original.shape != (1, 84, 8400) or original.dtype != np.float32:
        raise ValueError('expected_recorded_float32_1_84_8400')
    if (not np.isfinite(original).all() or np.count_nonzero(original[0, 4:] < 0) != 21
            or np.min(original[0, 4:]) != -(2.0**-24)):
        raise ValueError('not_the_recorded_21_score_roundoff_fixture')
    # A direct unmodified old-rule check must reproduce the reported failure.
    try:
        pp._verify_decoded_pre_nms_values({'model_outputs': original})
    except pp.FrozenPostprocessError as exc:
        old_error = str(exc)
    else:
        raise AssertionError('original_strict_failure_not_reproduced')
    results = []
    for transpose in (False, True):
        values = original.transpose(0, 2, 1) if transpose else original
        before_values = values.tobytes()
        outputs = {'model_outputs': values}
        fixed, observation = pp.prepare_decoded_pre_nms_outputs(outputs, score_policy=pp.decoded_pre_nms_score_policy())
        contract = pp.build_frozen_postprocess_contract(
            model_id='yolo11l', outputs=outputs, input_hw=[640, 640],
            original_wh=args.original_wh, source_contract_family='decoded_pre_nms',
        )
        frozen = pp.FrozenDetectionPostprocessor(contract)
        result = frozen.process(outputs)
        # Independent direct harness call with only a manually corrected copy.
        control = values.copy()
        canonical = control[0] if control.shape[1] == 84 else control[0].T
        scores = canonical[4:]
        scores[scores < 0] = 0
        scores[scores > 1] = 1
        harness = yolo.YoloHarness(conf_thresh=.25, iou_thresh=.45, max_det=300, model_id='yolo11l')
        independent = pp._result_json(harness.postprocess({'model_outputs': control}, {
            'input_hw': [640,640], 'original_wh': args.original_wh, 'variant': 'independent_replay',
        }))
        detections = pp._canonical_detection_records(independent['detections'], max_detections=300)
        # Explicit assertion: none of the real declared probabilities triggers
        # the generic harness's logit fallback during either build or runtime.
        old_sigmoid = yolo._sigmoid
        def forbidden(*_args, **_kwargs):
            raise AssertionError('unexpected_sigmoid_on_declared_probabilities')
        try:
            yolo._sigmoid = forbidden
            assert pp.FrozenDetectionPostprocessor(contract).process(outputs)['detections'] == result['detections']
        finally:
            yolo._sigmoid = old_sigmoid
        assert detections == result['detections'] and result['detection_count'] > 0
        assert values.tobytes() == before_values
        source_channels = values[0] if values.shape[1] == 84 else values[0].T
        assert np.array_equal(source_channels[4:].argmax(0), canonical[4:].argmax(0))
        assert np.array_equal(source_channels[4:].max(0), canonical[4:].max(0))
        results.append({
            'shape': list(values.shape), 'status': 'PASS', 'normalization': observation,
            'completed_detection_count': result['detection_count'],
            'detections': result['detections'],
            'detection_records_exactly_equal_to_independent_harness': True,
            'original_tensor_bytes_unchanged': True,
            'all_anchor_max_scores_and_argmax_classes_unchanged': True,
            'no_sigmoid_verified': True,
        })
    assert hashlib.sha256(file.read_bytes()).hexdigest() == before_file
    report = {
        'status': 'PASS', 'version': VERSION, 'original_strict_error_reproduced': old_error,
        'input_npz_sha256': before_file, 'original_wh': args.original_wh,
        'layouts': results, 'hardware_execution': 'NOT_RUN',
        'full_model_quality_acceptance': 'NOT_EVALUATED_BY_SINGLE_IMAGE_REPLAY',
    }
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.expanduser().write_text(text, encoding='utf-8')
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
