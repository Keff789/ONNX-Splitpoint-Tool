"""AP04 development tests: real bounded workers on four synthetic images.

No original dataset or hardware is used. B=1053 exercises repeated blocks and
rest blocks; it is a development budget, not a reduced original-run budget.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os

import numpy as np
import pytest

from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
from onnx_splitpoint_tool.quality_service import (
    ManagementQualityService,
    QualityEvaluationRequest,
    _combine_evaluation_shards,
    _evaluate_payload_shard,
    deterministic_resample_plan,
    prepare_evaluation,
)
from onnx_splitpoint_tool.quality_statistics_config import resource_budget


def _request(repetitions):
    # Top-1 implies Top-5 for every record. Distinct predictions have equal
    # Top-1 points, while draws differ and may have zero reference accuracy.
    reference = ((True, True), (False, True), (True, True), (False, False))
    candidate = ((True, True), (True, True), (False, True), (False, True))

    def records(role, hits):
        return [{"image_id": i, role: {"top1_hit": top1, "top5_hit": top5}}
                for i, (top1, top5) in enumerate(hits)]

    return QualityEvaluationRequest(
        reference_records=records("reference", reference),
        candidate_records=records("candidate", candidate), annotations=[],
        metric_gate_config={"task": "classification", "primary_metric": "top1_accuracy",
                            "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
                            "guardrails": {"top5_accuracy_margin": 0.01}},
        repetitions=repetitions, seed=731, confidence_level=0.95,
        non_inferiority_margin=0.01,
        evaluator_factory="onnx_splitpoint_tool.quality_metrics:classification_quality_evaluator",
        reference_prediction_field="reference", candidate_prediction_field="candidate",
        request_id=f"synthetic-bounded-classification-B{repetitions}",
    )


@pytest.fixture(scope="module")
def legacy_baseline():
    """Calculate each tiny complete legacy distribution once per module."""
    completed = {}

    def load(repetitions):
        if repetitions not in completed:
            request = _request(repetitions)
            _, payload = prepare_evaluation(request)
            payload["_statistics"] = {"engine": "legacy", "capture_draws": True}
            plan = deterministic_resample_plan(image_count=4, repetitions=repetitions, seed=731)
            assert plan.dtype == np.dtype("int64")
            shard = _evaluate_payload_shard(payload, plan, shard_index=0, repetition_offset=0)
            result = _combine_evaluation_shards(payload, [shard], elapsed_s=0.0, workers_requested=1)
            completed[repetitions] = (request, shard, result)
        return deepcopy(completed[repetitions])

    return load


_EXECUTION_FIELDS = {
    "statistics_observation", "evaluation_fingerprint", "cache_hit",
    "bootstrap_workers_requested", "bootstrap_workers_effective",
    "bootstrap_sharding", "worker_model",
}


def _science_projection(result, keys):
    # Compare every legacy result field except execution observations and the
    # intentionally stronger optimized cache key. No rounded comparisons.
    value = {key: deepcopy(result[key]) for key in keys}
    for component in (value["primary"], *value["guardrails"].values()):
        component.pop("bootstrap_elapsed_s", None)
        component.pop("bootstrap_engine", None)
    return value


def _ordered_vectors(shards, repetitions):
    ordered = sorted(shards, key=lambda row: row["repetition_offset"])
    assert ordered
    names = set(ordered[0]["component_points"])
    absolute = {name: {"reference": [], "candidate": []} for name in names}
    delta = {name: [] for name in names}
    ratios = []
    cursor = 0
    for shard in ordered:
        assert type(shard["repetition_offset"]) is int
        assert type(shard["repetitions"]) is int
        assert shard["repetition_offset"] == cursor
        count = shard["repetitions"]
        assert count > 0
        assert shard["component_points"] == ordered[0]["component_points"]
        assert set(shard["bootstrap"]) == set(shard["absolute_bootstrap"]) == names
        for name in names:
            assert len(shard["bootstrap"][name]) == count
            delta[name].extend(shard["bootstrap"][name])
            assert set(shard["absolute_bootstrap"][name]) == {"reference", "candidate"}
            for side in ("reference", "candidate"):
                values = shard["absolute_bootstrap"][name][side]
                assert len(values) == count
                absolute[name][side].extend(values)
        assert len(shard["relative_loss_draws"]) == count
        ratios.extend(shard["relative_loss_draws"])
        cursor += count
    assert cursor == repetitions
    return {"absolute": absolute, "delta": delta, "relative_loss": ratios,
            "relative_loss_undefined_mask": [value is None for value in ratios]}


def _assert_prepared_once_per_worker(result, *, workers, block_count):
    observations = result["statistics_observation"]["shards"]
    assert len(observations) == block_count
    prepared = Counter()
    visits = defaultdict(list)
    for row in observations:
        pid = row["worker_pid"]
        assert type(pid) is int and pid > 0 and pid != os.getpid()
        assert type(row["prepare_count"]) is int
        assert type(row["prepared_cache_hit"]) is bool
        assert row["prepare_count"] == (0 if row["prepared_cache_hit"] else 1)
        prepared[pid] += row["prepare_count"]
        visits[pid].append(row)
    assert 1 <= len(visits) <= min(workers, block_count)
    assert all(count == 1 for count in prepared.values())
    for rows in visits.values():
        assert sum(row["prepared_cache_hit"] for row in rows) == len(rows) - 1
    if block_count > workers:
        assert any(row["prepared_cache_hit"] for row in observations)


def _assert_service_parity(tmp_path, legacy_baseline, *, workers, block, repetitions):
    request, baseline_shard, expected = legacy_baseline(repetitions)
    cache = tmp_path / "statistics_cache"
    with ManagementQualityService(cache, workers=workers, statistics={
        "engine": "optimized_coco_v1", "block_repetitions": block,
        "checkpoint_blocks": True, "capture_draws": True,
    }) as service:
        actual = service.evaluate(request, timeout=60)
    assert service.shutdown_state()["finished"] is True
    assert actual["cache_hit"] is False
    keys = set(expected) - _EXECUTION_FIELDS
    assert _science_projection(actual, keys) == _science_projection(expected, keys)
    assert actual["primary"]["bootstrap_repetitions"] == repetitions
    assert actual["primary"]["ci_computed"] is True

    capture_path = cache / "draws" / (actual["evaluation_fingerprint"] + ".json")
    captured = json.loads(capture_path.read_text())["shards"]
    ordered = sorted(captured, key=lambda row: row["repetition_offset"])
    expected_ranges = [(start, min(block, repetitions - start))
                       for start in range(0, repetitions, block)]
    assert [(row["repetition_offset"], row["repetitions"]) for row in ordered] == expected_ranges
    assert all(row["component_points"] == baseline_shard["component_points"] for row in captured)
    assert _ordered_vectors(captured, repetitions) == _ordered_vectors([baseline_shard], repetitions)
    _assert_prepared_once_per_worker(actual, workers=workers, block_count=len(expected_ranges))
    if repetitions == 1053:
        mask = _ordered_vectors(captured, repetitions)["relative_loss_undefined_mask"]
        assert any(mask) and not all(mask), "the fixture must exercise positional undefined ratios"


@pytest.mark.parametrize("workers,block", [
    (1, 128), (1, 256), (1, 512),
    (2, 128), (2, 256), (2, 512),
    (4, 128), (4, 256), (4, 512),
    pytest.param(6, 128, id="six-workers-if-quota-allows"),
])
def test_worker_and_block_partition_preserve_all_draws_and_prepare_once(
    tmp_path, legacy_baseline, workers, block,
):
    if workers > 4 and resource_budget()["statistics_cpu_slots"] < workers:
        pytest.skip("current affinity/quota plus controller reserve does not admit six statistics workers")
    _assert_service_parity(tmp_path, legacy_baseline, workers=workers, block=block, repetitions=1053)


@pytest.mark.parametrize("repetitions,block", [(1, 128), (3, 2)])
def test_fewer_draws_than_workers_preserves_single_and_partial_blocks(
    tmp_path, legacy_baseline, repetitions, block,
):
    _assert_service_parity(tmp_path, legacy_baseline, workers=4, block=block, repetitions=repetitions)
