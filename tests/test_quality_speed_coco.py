"""Exact COCO parity on synthetic populations; no inference or hardware paths."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY, assess_accuracy
from onnx_splitpoint_tool.quality_metrics import _CanonicalCOCOEvaluator
from onnx_splitpoint_tool.quality_service import (
    QualityEvaluationRequest,
    _combine_evaluation_shards,
    _evaluate_payload_shard,
    _prediction_identity_is_bound,
    deterministic_resample_plan,
    prepare_evaluation,
)


def _box(*, category=0, x=0.0, score=0.5, **extra):
    return {"class_id": category, "x1": x, "y1": 0.0,
            "x2": x + 10.0, "y2": 10.0, "score": score, **extra}


def _population(name):
    good = _box()
    if name == "score_ties_maxdets":
        # The useful detection crosses exactly the 100-detection boundary.
        # Equal scores make both per-image and global stable ordering relevant.
        misses = [_box(x=50.0 + i) for i in range(100)]
        gt = [[good], [good], [], [good]]
        reference = [misses[:99] + [good, good], [good, _box(x=100)], [], [good]]
        candidate = [misses + [good], [_box(x=100), good], [good], []]
    elif name == "crowd_ignore_area":
        crowd = _box(iscrowd=1, area=1.0)
        ignored = _box(category=1, ignore=1, area=100_000.0)
        gt = [[crowd, _box(x=20, area=0.5)], [ignored], [], [_box(category=2)]]
        reference = [[good, _box(x=20)], [_box(category=1)], [], [_box(category=2)]]
        candidate = [[good, good, _box(x=25)], [_box(category=1, x=5)], [good], []]
    elif name == "empty_detections":
        gt = [[good], [], [_box(category=1)], [good]]
        reference = [[good], [], [_box(category=1)], []]
        candidate = [[], [], [], []]
    elif name == "empty_ground_truth":
        gt = [[], [], [], []]
        reference = [[good], [], [_box(category=1)], []]
        candidate = [[], [good], [], [_box(category=2)]]
    elif name == "category_population":
        gt = [[good], [_box(category=1)], [_box(category=2)], []]
        reference = [[good], [], [_box(category=2)], [_box(category=9)]]
        candidate = [[], [_box(category=1)], [_box(category=2, x=4)], []]
    else:
        raise AssertionError(name)
    return (
        [{"image_id": i, "reference": deepcopy(row)} for i, row in enumerate(reference)],
        [{"image_id": i, "candidate": deepcopy(row)} for i, row in enumerate(candidate)],
        [{"image_id": i, "ground_truth": deepcopy(row)} for i, row in enumerate(gt)],
    )


def _config(engine):
    return {
        "statistics_engine": engine,
        "metric_gate_config": {
            "task": "detection",
            "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
            "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
        },
    }


def _components(value):
    return {"primary": value["primary"], **{
        "guardrails." + name: component for name, component in value["guardrails"].items()
    }}


@pytest.mark.parametrize("population", [
    "score_ties_maxdets", "crowd_ignore_area", "empty_detections",
    "empty_ground_truth", "category_population",
])
def test_compact_coco_matches_every_component_for_all_fixed_multiplicities(population):
    records = _population(population)
    legacy = _CanonicalCOCOEvaluator(*records, _config("legacy"))
    compact = _CanonicalCOCOEvaluator(*records, _config("optimized_coco_v1"))
    # All 35 populations of four images, including all mass on one image.
    draws = [counts for counts in product(range(5), repeat=4) if sum(counts) == 4]
    assert len(draws) == 35
    for draw, counts in enumerate(draws):
        weights = np.asarray(counts, dtype=np.float64)
        expected = legacy.evaluate(weights)
        observed = compact.evaluate(weights)
        assert observed == expected, (population, draw, counts, observed, expected)
    # Re-evaluating a point after the last draw must reset mutable COCO state.
    assert compact.evaluate(np.ones(4)) == legacy.evaluate(np.ones(4))


def test_compact_matching_and_accumulation_use_only_all_area_maxdets100(monkeypatch):
    from pycocotools.cocoeval import COCOeval

    seen = []
    original_evaluate = COCOeval.evaluate
    original_accumulate = COCOeval.accumulate

    def check(evaluator, phase):
        params = evaluator.params
        seen.append(phase)
        assert params.areaRngLbl == ["all"]
        assert params.areaRng == [[0, 100000 ** 2]]
        assert params.maxDets == [100]
        assert len(params.iouThrs) == 10
        assert len(params.recThrs) == 101

    def evaluate(evaluator):
        check(evaluator, "matching")
        return original_evaluate(evaluator)

    def accumulate(evaluator, *args, **kwargs):
        check(evaluator, "accumulation")
        assert evaluator._paramsEval.areaRng == evaluator.params.areaRng
        assert evaluator._paramsEval.maxDets == evaluator.params.maxDets
        return original_accumulate(evaluator, *args, **kwargs)

    def forbidden_summary(*args, **kwargs):
        raise AssertionError("Reduced COCO dimensions must not produce the full official summary")

    monkeypatch.setattr(COCOeval, "evaluate", evaluate)
    monkeypatch.setattr(COCOeval, "accumulate", accumulate)
    monkeypatch.setattr(COCOeval, "summarize", forbidden_summary)
    evaluator = _CanonicalCOCOEvaluator(*_population("category_population"), _config("optimized_coco_v1"))
    evaluator.evaluate(np.asarray([0, 2, 1, 1]))
    assert seen == ["matching", "matching", "accumulation", "accumulation"]


@pytest.mark.parametrize("engine", ["legacy", "optimized_coco_v1"])
@pytest.mark.parametrize("field,value,message", [
    ("x2", -1.0, "invalid final XYXY"),
    ("y1", float("nan"), "invalid final XYXY"),
    ("score", float("inf"), "invalid detection score"),
    ("score", -0.1, "invalid detection score"),
    ("score", 1.1, "invalid detection score"),
])
def test_compact_preserves_invalid_detection_errors(engine, field, value, message):
    reference, candidate, annotations = _population("category_population")
    candidate[1]["candidate"][0][field] = value
    with pytest.raises(ValueError, match=message):
        _CanonicalCOCOEvaluator(reference, candidate, annotations, _config(engine))


@pytest.mark.parametrize("engine", ["legacy", "optimized_coco_v1"])
@pytest.mark.parametrize("weights", [[0, 0, 0, 0], [1, -1, 2, 2], [0.5, 0.5, 1, 2], [1, 1, 2], [float("nan"), 1, 1, 1]])
def test_compact_preserves_invalid_population_errors(engine, weights):
    evaluator = _CanonicalCOCOEvaluator(*_population("category_population"), _config(engine))
    with pytest.raises(ValueError, match="COCO (image multiplicities|population)"):
        evaluator.evaluate(np.asarray(weights))


def _request(reference_hits, candidate_hits, repetitions=61):
    good = _box()
    reference = [{"image_id": i, "reference": [deepcopy(good)] if hit else []}
                 for i, hit in enumerate(reference_hits)]
    candidate = [{"image_id": i, "candidate": [deepcopy(good)] if hit else []}
                 for i, hit in enumerate(candidate_hits)]
    annotations = [{"image_id": i, "ground_truth": [deepcopy(good)]} for i in range(len(reference))]
    return QualityEvaluationRequest(
        reference, candidate, annotations, _config("legacy")["metric_gate_config"],
        repetitions, 20260710, 0.95, 0.01,
        evaluator_factory="onnx_splitpoint_tool.quality_metrics:detection_quality_evaluator",
        reference_prediction_field="reference", candidate_prediction_field="candidate",
        request_id="synthetic-coco-parity", reference_identity="synthetic-bound-cpu",
    )


def _shard(payload, plan, engine):
    payload = deepcopy(payload)
    payload["_statistics"] = {"engine": engine, "capture_draws": True}
    return _evaluate_payload_shard(payload, plan, shard_index=0, repetition_offset=0)


@pytest.mark.parametrize("reference,candidate", [
    ([True, False], [True, False]),
    ([True, False], [False, True]),
    ([True, True, True, True], [True, False, False, False]),
])
def test_service_preserves_absolute_draws_deltas_ratios_masks_and_reporting(reference, candidate):
    request = _request(reference, candidate)
    _, payload = prepare_evaluation(request)
    plan = deterministic_resample_plan(image_count=len(reference), repetitions=request.repetitions, seed=request.seed)
    legacy = _shard(payload, plan, "legacy")
    compact = _shard(payload, plan, "optimized_coco_v1")
    assert compact["component_points"] == legacy["component_points"]
    assert compact["bootstrap"] == legacy["bootstrap"]
    assert compact["relative_loss_draws"] == legacy["relative_loss_draws"]
    assert compact["absolute_bootstrap"] == legacy["absolute_bootstrap"]
    assert compact["repetitions"] == request.repetitions
    assert compact["skipped_reason"] == ""

    evaluator = _CanonicalCOCOEvaluator(
        payload["reference_records"], payload["candidate_records"], payload["annotations"], _config("legacy"))
    for offset, indices in enumerate(plan):
        expected = _components(evaluator.evaluate(np.bincount(indices, minlength=len(reference))))
        for name, component in expected.items():
            assert compact["absolute_bootstrap"][name]["reference"][offset] == component["reference"]
            assert compact["absolute_bootstrap"][name]["candidate"][offset] == component["candidate"]
            assert compact["bootstrap"][name][offset] == component["delta"]
        ratio = assess_accuracy(expected["primary"]["reference"], expected["primary"]["candidate"],
                                policy=DEFAULT_REPORTING_POLICY)["relative_loss"]
        assert compact["relative_loss_draws"][offset] == ratio

    final_legacy = _combine_evaluation_shards(payload, [legacy], elapsed_s=0.0, workers_requested=1)
    final_compact = _combine_evaluation_shards(payload, [compact], elapsed_s=0.0, workers_requested=1)
    for key in ("primary", "guardrails", "accuracy_assessment", "secondary_accuracy_assessments",
                "accuracy_warnings", "decision", "relative_loss_undefined_draws", "reporting_policy"):
        assert final_compact[key] == final_legacy[key], key
    if any(not value for value in reference):
        mask = [value is None for value in compact["relative_loss_draws"]]
        assert any(mask)
        assert not all(mask)
        assert final_compact["relative_loss_undefined_draws"] == sum(mask)
        assert final_compact["accuracy_assessment"]["relative_loss_ci"] is None
    else:
        assert final_compact["decision"] == "accuracy_loss"
        assert final_compact["accuracy_assessment"]["relative_loss_ci"] is not None
        assert len(set(compact["relative_loss_draws"])) > 1


def test_prediction_identity_requires_payload_identity_not_equal_coco_points():
    _, identical = prepare_evaluation(_request([True, False], [True, False]))
    _, equal_point = prepare_evaluation(_request([True, False], [False, True]))
    assert _prediction_identity_is_bound(identical)
    assert not _prediction_identity_is_bound(equal_point)
    evaluator = _CanonicalCOCOEvaluator(
        equal_point["reference_records"], equal_point["candidate_records"],
        equal_point["annotations"], _config("optimized_coco_v1"))
    point = evaluator.evaluate(np.ones(2))
    assert point["primary"]["reference"] == point["primary"]["candidate"]
    assert evaluator.evaluate(np.asarray([2, 0]))["primary"]["delta"] != 0.0


@pytest.mark.parametrize("engine", ["legacy", "optimized_coco_v1"])
def test_extra_area_guardrail_cannot_silently_disappear(engine):
    request = _request([True, True], [True, False], repetitions=3)
    request.metric_gate_config["guardrails"]["ap_small_margin"] = 0.01
    _, payload = prepare_evaluation(request)
    plan = deterministic_resample_plan(image_count=2, repetitions=3, seed=request.seed)
    with pytest.raises(ValueError, match="omitted configured guardrail.*ap_small"):
        _shard(payload, plan, engine)


def test_separate_official_report_retains_all_twelve_metrics(tmp_path, monkeypatch):
    import json
    from pycocotools.cocoeval import COCOeval
    from onnx_splitpoint_tool.validation.official_coco import evaluate_coco_bbox, METRIC_NAMES

    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({
        "info": {}, "images": [{"id": 1}], "categories": [{"id": 1, "name": "one"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
    }))
    original = COCOeval.summarize
    observed = []

    def summarize(evaluator):
        observed.append((evaluator.params.areaRngLbl, evaluator.params.maxDets))
        return original(evaluator)

    monkeypatch.setattr(COCOeval, "summarize", summarize)
    report = evaluate_coco_bbox(
        annotations=annotation_path, output_dir=tmp_path / "report", required=True,
        predictions=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}],
    )
    assert observed == [(["all", "small", "medium", "large"], [1, 10, 100])]
    assert set(report["metrics"]) == set(METRIC_NAMES)
    assert len(report["metrics"]) == 12
    assert report["metrics"]["AP_small"] > 0.99
    tensors = np.load(tmp_path / "report/coco_eval_tensors.npz", allow_pickle=False)
    assert tensors["precision"].shape == (10, 101, 1, 4, 3)
