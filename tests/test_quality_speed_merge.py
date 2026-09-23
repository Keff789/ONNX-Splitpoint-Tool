"""AP04 T27/T28: reject invalid draw partitions before a final result exists.

Four synthetic images and seven fixed draws only. No service, process pool,
hardware, filesystem cache or original-run input is needed by these tests.
"""
from copy import deepcopy

import numpy as np
import pytest

from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
from onnx_splitpoint_tool.quality_service import (
    QualityEvaluationRequest,
    _combine_evaluation_shards,
    _evaluate_payload_shard,
    prepare_evaluation,
)


COMPONENTS = ("primary", "guardrails.top5_accuracy")
RANGES = ((0, 2), (2, 4), (4, 7))


def _payload(*, reporting=True, identity=False, point_fail=False):
    reference_hits = ((True, True), (False, True), (True, True), (False, False))
    candidate_hits = ((True, True), (True, True), (False, True), (False, True))
    if identity:
        candidate_hits = reference_hits
    elif point_fail:
        candidate_hits = tuple((False, top5) for _, top5 in reference_hits)

    def records(side, hits):
        return [{"image_id": i, side: {"top1_hit": top1, "top5_hit": top5}}
                for i, (top1, top5) in enumerate(hits)]

    gate = {"task": "classification", "primary_metric": "top1_accuracy",
            "non_inferiority_margin": 0.01,
            "guardrails": {"top5_accuracy_margin": 0.01}}
    if reporting:
        gate["reporting_policy"] = deepcopy(DEFAULT_REPORTING_POLICY)
    request = QualityEvaluationRequest(
        reference_records=records("reference", reference_hits),
        candidate_records=records("candidate", candidate_hits),
        annotations=[], metric_gate_config=gate,
        repetitions=7, seed=731, confidence_level=0.95, non_inferiority_margin=0.01,
        evaluator_factory="onnx_splitpoint_tool.quality_metrics:classification_quality_evaluator",
        reference_prediction_field="reference", candidate_prediction_field="candidate",
        request_id="synthetic-merge-partition",
    )
    _, payload = prepare_evaluation(request)
    if reporting:
        payload["_statistics"] = {"engine": "optimized_coco_v1", "capture_draws": True}
    return payload


def _shards(payload, *, zero_reference=True):
    # Explicit multiplicity cases exercise both defined and undefined ratios.
    # Merge sees completed draw ranges; it must not invent another random plan.
    plan = np.asarray([
        [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
        [0, 1, 2, 3], [0, 0, 1, 2], [0, 2, 2, 3],
    ], dtype=np.int64)
    if not zero_reference:
        plan[1, 0] = 0
        plan[3, 0] = 0
    return [
        _evaluate_payload_shard(payload, plan[start:stop],
                                shard_index=index, repetition_offset=start)
        for index, (start, stop) in enumerate(RANGES)
    ]


@pytest.fixture
def completed():
    payload = _payload()
    return payload, _shards(payload)


def _merge(payload, shards):
    return _combine_evaluation_shards(payload, shards, elapsed_s=0.0, workers_requested=3)


def _reject(payload, shards):
    with pytest.raises(ValueError) as caught:
        _merge(payload, shards)
    assert str(caught.value), "malformed draw blocks need an actionable validation error"


@pytest.mark.parametrize("zero_reference", [False, True])
def test_complete_reporting_merge_is_exact_under_reversed_completion(zero_reference):
    payload = _payload()
    shards = _shards(payload, zero_reference=zero_reference)
    expected = _merge(payload, shards)
    assert _merge(payload, list(reversed(shards))) == expected
    assert expected["primary"]["bootstrap_repetitions"] == 7
    assert expected["primary"]["ci_computed"] is True
    assert expected["relative_loss_undefined_draws"] == (2 if zero_reference else 0)
    ratio_ci = expected["accuracy_assessment"]["relative_loss_ci"]
    if zero_reference:
        assert ratio_ci is None
    else:
        ratios = [value for shard in shards for value in shard["relative_loss_draws"]]
        assert ratio_ci == [float(value) for value in np.quantile(ratios, [0.025, 0.975])]


@pytest.mark.parametrize("index,offset", [
    pytest.param(1, 0, id="duplicate-offset"),
    pytest.param(1, 1, id="overlap"),
    pytest.param(1, 3, id="interior-gap"),
    pytest.param(0, 1, id="missing-start"),
    pytest.param(0, -1, id="negative-start"),
    pytest.param(2, 7, id="start-at-B"),
    pytest.param(2, 5, id="stop-beyond-B"),
    pytest.param(0, False, id="bool-zero"),
    pytest.param(1, True, id="bool-one"),
    pytest.param(1, 2.0, id="integral-float"),
    pytest.param(1, 2.5, id="fractional-float"),
    pytest.param(1, "2", id="string"),
])
def test_offsets_must_form_exact_typed_half_open_partition(completed, index, offset):
    payload, shards = completed
    shards[index]["repetition_offset"] = offset
    assert sum(len(row["bootstrap"]["primary"]) for row in shards) == 7
    _reject(payload, shards)


def test_duplicate_complete_block_cannot_replace_missing_block(completed):
    payload, shards = completed
    shards[1] = deepcopy(shards[0])
    assert sum(len(row["bootstrap"]["primary"]) for row in shards) == 7
    _reject(payload, shards)


@pytest.mark.parametrize("declared", [0, 1, 3, -1, True, 2.0, "2"])
def test_declared_repetitions_must_match_each_complete_range(completed, declared):
    payload, shards = completed
    shards[1]["repetitions"] = declared
    _reject(payload, shards)


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("field,value", [
    ("reference", 0.625), ("candidate", 0.375), ("delta", 0.125),
    ("margin", 0.02), ("metric", "other_metric"), ("sample_count", 5),
])
def test_all_blocks_must_agree_on_complete_point_components(completed, component, field, value):
    payload, shards = completed
    shards[1]["component_points"][component][field] = value
    _reject(payload, shards)


@pytest.mark.parametrize("field", ["component_points", "bootstrap", "absolute_bootstrap"])
def test_every_block_carries_same_component_set(completed, field):
    payload, shards = completed
    shards[1][field].pop("guardrails.top5_accuracy")
    _reject(payload, shards)


def test_configured_guardrail_contract_must_agree(completed):
    payload, shards = completed
    shards[1]["configured_guardrails"] = []
    _reject(payload, shards)


VECTOR_PATHS = (
    ("bootstrap", "primary"),
    ("bootstrap", "guardrails.top5_accuracy"),
    ("relative_loss_draws",),
    ("absolute_bootstrap", "primary", "reference"),
    ("absolute_bootstrap", "primary", "candidate"),
    ("absolute_bootstrap", "guardrails.top5_accuracy", "reference"),
    ("absolute_bootstrap", "guardrails.top5_accuracy", "candidate"),
)


def _vector(shard, path):
    value = shard
    for field in path:
        value = value[field]
    return value


@pytest.mark.parametrize("path", VECTOR_PATHS, ids=lambda path: ".".join(path))
@pytest.mark.parametrize("change", ["short", "long"])
def test_all_absolute_delta_and_ratio_vectors_match_their_range(completed, path, change):
    payload, shards = completed
    vector = _vector(shards[1], path)
    if change == "short":
        vector.pop()
    else:
        vector.append(vector[-1])
    _reject(payload, shards)


@pytest.mark.parametrize("path", VECTOR_PATHS, ids=lambda path: ".".join(path))
def test_global_vector_length_cannot_hide_wrong_per_block_lengths(completed, path):
    payload, shards = completed
    # All values and their concatenated order survive, but the range binding
    # is invalid: one vector belongs to three draws, the next to only one.
    _vector(shards[0], path).append(_vector(shards[1], path).pop(0))
    assert sum(len(_vector(shard, path)) for shard in shards) == 7
    _reject(payload, shards)


def test_legacy_computed_bootstrap_does_not_require_optional_absolute_capture():
    payload = _payload(reporting=False)
    shards = _shards(payload)
    assert all("absolute_bootstrap" not in shard for shard in shards)
    assert all(shard["relative_loss_draws"] == [] for shard in shards)
    result = _merge(payload, shards)
    assert result["primary"]["bootstrap_repetitions"] == 7
    assert result["primary"]["ci_computed"] is True


@pytest.mark.parametrize("identity", [False, True], ids=["legacy-point-fail", "legacy-identity"])
def test_valid_legacy_skips_keep_zero_draws_without_false_coverage_error(identity):
    payload = _payload(reporting=False, identity=identity, point_fail=not identity)
    if identity:
        shards = [_evaluate_payload_shard(payload, np.empty((0, 4), dtype=np.int64),
                                          shard_index=0, repetition_offset=0)]
    else:
        # Historical point-fail shards have nonzero assigned offsets but zero
        # effective repetitions; they do not claim a computed full interval.
        shards = _shards(payload)
    assert all(shard["repetitions"] == 0 for shard in shards)
    result = _merge(payload, shards)
    assert result["decision"] == ("pass" if identity else "fail")
    for component in (result["primary"], *result["guardrails"].values()):
        assert component["bootstrap_repetitions_requested"] == 7
        assert component["bootstrap_repetitions"] == 0
        assert component["ci_computed"] is False
        assert component["ci_low"] is None and component["ci_high"] is None
    expected_reason = "candidate_reference_identical" if identity else "point_estimate_below_non_inferiority_margin"
    assert result["primary"]["bootstrap_skipped_reason"] == expected_reason
