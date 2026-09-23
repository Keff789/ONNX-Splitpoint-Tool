"""Independent tiny AP04 detection/reuse and legacy/custom compatibility checks."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from onnx_splitpoint_tool.quality_service import (
    ManagementQualityService, QualityEvaluationRequest, _combine_evaluation_shards,
    _evaluate_payload_shard, deterministic_resample_plan, prepare_evaluation,
)

from test_quality_speed_blocks import (
    _EXECUTION_FIELDS, _ordered_vectors, _request as classification_request,
    _science_projection,
)
from test_quality_speed_coco import _request as detection_request


def _legacy(request, *, capture=True):
    _, payload = prepare_evaluation(request)
    if capture:
        payload["_statistics"] = {"engine": "legacy", "capture_draws": True}
    plan = deterministic_resample_plan(image_count=len(request.reference_records),
                                      repetitions=request.repetitions, seed=request.seed)
    shard = _evaluate_payload_shard(payload, plan, shard_index=0, repetition_offset=0)
    result = _combine_evaluation_shards(payload, [shard], elapsed_s=0, workers_requested=1)
    return shard, result


def _same_science(observed, expected):
    keys = set(expected) - _EXECUTION_FIELDS
    assert _science_projection(observed, keys) == _science_projection(expected, keys)


def _phase_preparation_once(observations, phase):
    assert observations
    prepared = Counter()
    for row in observations:
        assert row["component_phase"] == phase
        for name in ("loading_s", "prepare_s", "matching_s", "data_preparation_s",
                     "point_s", "accumulation_s", "ipc_wait_s", "reference_load_s",
                     "cache_accounting_s"):
            assert type(row[name]) is float and row[name] >= 0
        if row["prepared_cache_hit"]:
            assert row["prepare_s"] == row["point_s"] == row["matching_s"] == 0
        prepared[row["worker_pid"]] += row["prepare_count"]
        assert row["prepare_count"] == int(not row["prepared_cache_hit"])
    assert all(count == 1 for count in prepared.values())
    assert any(row["prepared_cache_hit"] for row in observations)


def test_detection_cold_reference_warm_identity_and_pair_warm_are_exact(tmp_path):
    # Same point != same payload (second candidate); third is actual identity.
    requests = [detection_request((True, False), candidate, repetitions=17)
                for candidate in ((True, True), (False, True), (True, False))]
    expected = [_legacy(request) for request in requests]
    cache = tmp_path / "cache"
    with ManagementQualityService(cache, workers=2, statistics={
        "engine": "optimized_coco_v1", "block_repetitions": 4,
        "checkpoint_blocks": True, "capture_draws": True,
    }) as service:
        for index, (request, (baseline, result)) in enumerate(zip(requests, expected)):
            actual = service.evaluate(request, timeout=20)
            _same_science(actual, result)
            observation = actual["statistics_observation"]
            assert observation["reference_cache_hit"] is (index > 0)
            assert observation["draws_recomputed"] == 17
            assert observation["checkpoint_reused_draws"] == 0
            _phase_preparation_once(observation["shards"], "candidate")
            if index == 0:
                _phase_preparation_once(observation["reference_shards"], "reference")
            else:
                assert observation["reference_shards"] == []
            captured = json.loads((cache / "draws" / (actual["evaluation_fingerprint"] + ".json")).read_text())["shards"]
            assert _ordered_vectors(captured, 17) == _ordered_vectors([baseline], 17)
            pair_warm = service.evaluate(request, timeout=10)
            assert pair_warm["cache_hit"] is True
            assert pair_warm["statistics_observation"]["draws_recomputed"] == 0
            _same_science(pair_warm, result)
    assert service.shutdown_state()["finished"] is True


@pytest.mark.parametrize("task", ["classification", "detection"])
@pytest.mark.parametrize("kind", ["identity", "point_fail", "computed"])
def test_opted_engine_preserves_existing_nonreporting_semantics(tmp_path, task, kind):
    request = classification_request(17) if task == "classification" else detection_request(
        (True, False), (False, True), repetitions=17)
    request = deepcopy(request)
    gate = dict(request.metric_gate_config)
    gate.pop("reporting_policy")
    request = replace(request, metric_gate_config=gate)
    if kind == "identity":
        for reference, candidate in zip(request.reference_records, request.candidate_records):
            candidate["candidate"] = deepcopy(reference["reference"])
    elif kind == "point_fail":
        for candidate in request.candidate_records:
            candidate["candidate"] = [] if task == "detection" else {"top1_hit": False, "top5_hit": False}
    _, expected = _legacy(request, capture=False)
    with ManagementQualityService(tmp_path / "cache", workers=1, statistics={
        "engine": "optimized_coco_v1", "block_repetitions": 4,
        "checkpoint_blocks": True,
    }) as service:
        actual = service.evaluate(request, timeout=20)
        _same_science(actual, expected)
        assert actual["primary"]["bootstrap_repetitions"] == (17 if kind == "computed" else 0)
        assert actual["primary"]["ci_computed"] is (kind == "computed")
    assert service.shutdown_state()["finished"] is True


class ScalarEvaluator:
    """Existing custom-factory API permits a scalar delta without absolute sides."""
    def __init__(self, reference, candidate, annotations, config):
        self.delta = np.asarray([c["value"] - r["value"] for r, c in zip(reference, candidate)])

    def evaluate(self, weights):
        return float(np.dot(weights, self.delta) / np.sum(weights))


def test_opted_engine_preserves_scalar_custom_evaluator_without_absolute_values(tmp_path):
    request = QualityEvaluationRequest(
        reference_records=[{"image_id": 0, "value": 0.5}, {"image_id": 1, "value": 0.5}],
        candidate_records=[{"image_id": 0, "value": 0.75}, {"image_id": 1, "value": 0.25}],
        annotations=[], metric_gate_config={}, repetitions=17, seed=1107,
        confidence_level=0.95, non_inferiority_margin=0.5,
        evaluator_factory=f"{__name__}:ScalarEvaluator",
    )
    _, expected = _legacy(request, capture=False)
    assert expected["primary"]["reference"] is expected["primary"]["candidate"] is None
    with ManagementQualityService(tmp_path / "cache", workers=1, statistics={
        "engine": "optimized_coco_v1", "block_repetitions": 4,
        "checkpoint_blocks": True,
    }) as service:
        actual = service.evaluate(request, timeout=20)
    _same_science(actual, expected)
    assert actual["primary"]["bootstrap_repetitions"] == 17


@pytest.mark.parametrize("mutation", ["component_point", "block_plan_identity"])
def test_completed_block_merge_rejects_disagreeing_point_or_plan(mutation):
    request = classification_request(7)
    _, payload = prepare_evaluation(request)
    payload["_statistics"] = {"engine": "optimized_coco_v1", "capture_draws": True}
    plan = deterministic_resample_plan(image_count=4, repetitions=7, seed=731)
    rows = [_evaluate_payload_shard(payload, plan[start:stop], shard_index=index, repetition_offset=start)
            for index, (start, stop) in enumerate(((0, 3), (3, 7)))]
    contract = {"schema": "paired-statistics-block-v1", "evaluation": payload["evaluation_fingerprint"],
                "plan": "a" * 64, "phase": "candidate", "reference": "b" * 64, "B": 7, "n": 4}
    for row in rows:
        row["block_identity"] = deepcopy(contract)
    if mutation == "component_point":
        rows[1]["component_points"]["primary"]["reference"] = 0.25
    else:
        rows[1]["block_identity"]["plan"] = "c" * 64
    with pytest.raises(ValueError):
        _combine_evaluation_shards(payload, rows, elapsed_s=0, workers_requested=2)
