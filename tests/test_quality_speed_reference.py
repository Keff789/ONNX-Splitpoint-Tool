"""Run-local reference reuse, ownership and integrity on tiny fixed inputs."""
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pytest

from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
from onnx_splitpoint_tool.quality_cache import json_fingerprint, prediction_fingerprint
from onnx_splitpoint_tool.quality_service import (
    ManagementQualityService, QualityEvaluationRequest, _evaluate_payload_shard, deterministic_resample_plan,
    prepare_evaluation,
)
from onnx_splitpoint_tool.quality_statistics import ReferenceReuseEvaluator, reference_component_key


def _payload(directory, reference=(True, False), candidate=(True, True)):
    box = {"class_id": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "score": 0.8}
    references = [{"image_id": i, "reference": [deepcopy(box)] if hit else []}
                  for i, hit in enumerate(reference)]
    candidates = [{"image_id": i, "candidate": [deepcopy(box)] if hit else []}
                  for i, hit in enumerate(candidate)]
    annotations = [{"image_id": i, "ground_truth": [deepcopy(box)]} for i in range(len(reference))]
    gate = {"task": "detection", "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
            "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01}}
    request = QualityEvaluationRequest(
        references, candidates, annotations, gate, 17, 1107, 0.95, 0.01,
        evaluator_factory="onnx_splitpoint_tool.quality_metrics:detection_quality_evaluator",
        reference_prediction_field="reference", candidate_prediction_field="candidate")
    _, payload = prepare_evaluation(request)
    payload["_statistics"] = {"engine": "optimized_coco_v1", "capture_draws": True,
                              "reference_dir": str(directory)}
    plan = deterministic_resample_plan(image_count=len(reference), repetitions=17, seed=1107)
    return payload, plan


def _evaluate(payload, plan):
    return _evaluate_payload_shard(payload, plan, shard_index=0, repetition_offset=0)


def _config(payload):
    return {"statistics_engine": "optimized_coco_v1", "metric_gate_config": payload["metric_gate_config"]}


def _wait_for(path, timeout=15):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"test rendezvous missing: {path}")
        time.sleep(0.01)


def _occupied_worker(payload, plan, rendezvous, token):
    root = Path(rendezvous)
    (root / str(token)).write_text(str(os.getpid()))
    _wait_for(root / "0")
    _wait_for(root / "1")
    return _evaluate(payload, plan)


def _crashing_reference_owner(payload, plan, rendezvous):
    # Only this deliberately created child exits; kernel flock cleanup is tested.
    owner = ReferenceReuseEvaluator(payload, plan, 0, _config(payload))
    owner.evaluate(np.ones(len(payload["image_ids"])))
    root = Path(rendezvous)
    (root / "ready").write_text(str(os.getpid()))
    _wait_for(root / "exit")
    os._exit(17)


def test_two_candidates_reuse_one_exact_reference_and_preserve_absolute_draws(tmp_path, monkeypatch):
    from pycocotools.cocoeval import COCOeval
    matching = []
    original = COCOeval.evaluate

    def counted(evaluator):
        matching.append(1)
        return original(evaluator)

    monkeypatch.setattr(COCOeval, "evaluate", counted)
    first, plan = _payload(tmp_path, candidate=(True, True))
    second, _ = _payload(tmp_path, candidate=(False, True))
    # Execution provenance differs, while the absolute reference remains equal.
    second["metric_gate_config"].update(variant="composed", producer_identity_sha256="1" * 64,
                                      artifact_provenance_binding_status="verified")
    second["request_id"] = "second-candidate"
    assert reference_component_key(first, plan, 0) == reference_component_key(second, plan, 0)
    a = _evaluate(first, plan)
    b = _evaluate(second, plan)
    assert len(matching) == 3  # reference + first candidate, then second candidate only
    assert not a["statistics_observation"]["reference_cache_hit"]
    assert b["statistics_observation"]["reference_cache_hit"]
    for metric in a["absolute_bootstrap"]:
        assert a["absolute_bootstrap"][metric]["reference"] == b["absolute_bootstrap"][metric]["reference"]
    assert a["absolute_bootstrap"]["primary"]["candidate"] != b["absolute_bootstrap"]["primary"]["candidate"]
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text())
    assert len(stored["values"]) == len(plan) + 1
    assert stored["sha256"] == json_fingerprint(stored["values"])


@pytest.mark.parametrize("mutation", ["prediction", "image_order", "area", "crowd", "ignore",
                                       "category", "seed", "repetitions", "algorithm", "plan", "offset", "dtype"])
def test_reference_key_binds_every_scientific_component(tmp_path, mutation):
    payload, plan = _payload(tmp_path)
    old = reference_component_key(payload, plan, 0)
    changed = deepcopy(payload)
    alternative_plan = plan.copy()
    offset = 0
    if mutation == "prediction":
        changed["reference_records"][0]["reference"][0]["score"] = 0.7
        changed["reference_predictions_sha256"] = prediction_fingerprint(
            changed["reference_records"], payload_field="reference")
    elif mutation == "image_order":
        changed["image_ids"].reverse()
        changed["reference_records"].reverse()
        changed["candidate_records"].reverse()
        changed["annotations"].reverse()
    elif mutation in {"area", "crowd", "ignore", "category"}:
        field = {"crowd": "iscrowd", "category": "class_id"}.get(mutation, mutation)
        changed["annotations"][0]["ground_truth"][0][field] = 3 if mutation == "area" else 1
    elif mutation in {"seed", "repetitions"}:
        changed["seed_schema"][mutation] += 1
    elif mutation == "algorithm":
        changed["algorithm_version"] += ":different"
    elif mutation == "plan":
        alternative_plan[0, 0] = 1 - alternative_plan[0, 0]
    elif mutation == "offset":
        offset = 1
    elif mutation == "dtype":
        alternative_plan = plan.astype(np.int32)
    assert reference_component_key(changed, alternative_plan, offset) != old


@pytest.mark.parametrize("sequence", ["reference_records", "candidate_records", "annotations"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_post_prepare_record_or_gt_reordering_cannot_change_paired_positions(tmp_path, sequence, warm):
    payload, plan = _payload(tmp_path)
    if warm:
        _evaluate(payload, plan)
    changed = deepcopy(payload)
    changed[sequence].reverse()
    if sequence != "annotations":
        side = sequence.removesuffix("_records")
        # The existing scientific payload hash ignores harmless input order;
        # positional validation is therefore independently necessary here.
        assert prediction_fingerprint(changed[sequence], payload_field=side) == payload[side + "_predictions_sha256"]
    with pytest.raises(ValueError, match="order does not match the paired image population"):
        _evaluate(changed, plan)


@pytest.mark.parametrize("sequence", ["reference_records", "candidate_records", "annotations"])
@pytest.mark.parametrize("image_id", [False, "0"], ids=["boolean", "string"])
def test_positional_binding_preserves_image_id_type(tmp_path, sequence, image_id):
    payload, plan = _payload(tmp_path)
    changed = deepcopy(payload)
    changed[sequence][0]["image_id"] = image_id
    with pytest.raises(ValueError, match="order does not match the paired image population"):
        _evaluate(changed, plan)


@pytest.mark.parametrize("field", ["numpy", "pycocotools", "python", "kernel", "dtype", "plan_dtype"])
def test_reference_key_binds_numerical_environment(tmp_path, monkeypatch, field):
    import onnx_splitpoint_tool.quality_statistics as statistics
    payload, plan = _payload(tmp_path)
    old = reference_component_key(payload, plan, 0)
    environment = statistics.numerical_environment()
    environment[field] = "changed-test-contract"
    monkeypatch.setattr(statistics, "numerical_environment", lambda: environment)
    assert reference_component_key(payload, plan, 0) != old


@pytest.mark.parametrize("mutation", ["partial", "wrong_hash", "shape", "nonfinite", "schema", "key"])
def test_corrupt_or_unknown_reference_is_miss_not_partial_hit(tmp_path, mutation):
    payload, plan = _payload(tmp_path)
    expected = _evaluate(payload, plan)
    path = next(tmp_path.glob("*.json"))
    stored = json.loads(path.read_text())
    if mutation == "partial":
        path.write_text('{"values":[')
    else:
        if mutation == "wrong_hash":
            stored["sha256"] = "0" * 64
        elif mutation == "shape":
            stored["values"].pop()
            stored["sha256"] = json_fingerprint(stored["values"])
        elif mutation == "nonfinite":
            stored["values"][0][0] = float("nan")
        elif mutation == "schema":
            stored["schema"] = "unknown-future-format"
        elif mutation == "key":
            stored["key"] = "0" * 64
        path.write_text(json.dumps(stored))
    observed = _evaluate(payload, plan)
    assert not observed["statistics_observation"]["reference_cache_hit"]
    assert observed["absolute_bootstrap"] == expected["absolute_bootstrap"]
    assert observed["relative_loss_draws"] == expected["relative_loss_draws"]


def test_true_identity_reuses_draws_without_erasing_undefined_positions(tmp_path):
    payload, plan = _payload(tmp_path, candidate=(True, False))
    cold = _evaluate(payload, plan)
    warm = _evaluate(payload, plan)
    assert warm["statistics_observation"]["reference_cache_hit"]
    assert cold["relative_loss_draws"] == warm["relative_loss_draws"]
    ratios = cold["relative_loss_draws"]
    assert None in ratios and 0.0 in ratios
    assert len(ratios) == len(plan)
    for component in warm["absolute_bootstrap"].values():
        assert component["reference"] == component["candidate"]


def test_equal_points_are_not_reference_identity(tmp_path):
    payload, plan = _payload(tmp_path, candidate=(False, True))
    output = _evaluate(payload, plan)
    point = output["component_points"]["primary"]
    assert point["reference"] == point["candidate"]
    assert any(value != 0 for value in output["bootstrap"]["primary"])


@pytest.mark.parametrize("mutated_side", ["reference", "candidate"])
def test_stale_payload_fingerprints_cannot_authorize_identity_or_warm_reuse(tmp_path, mutated_side):
    payload, plan = _payload(tmp_path, reference=(True, True), candidate=(True, True))
    _evaluate(payload, plan)
    changed = deepcopy(payload)
    changed[mutated_side + "_records"][0][mutated_side] = []
    # Either reject the stale binding or recompute valid current payloads; a
    # stale reference/identity shortcut must never hide this real difference.
    try:
        result = _evaluate(changed, plan)
    except ValueError:
        return
    assert result["component_points"]["primary"]["delta"] != 0.0


def _standard_alias_request(task):
    if task == "detection":
        box = {"class_id": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "score": 0.8}
        reference = [[deepcopy(box)], []]
        candidate = [[deepcopy(box)], [deepcopy(box)]]
        annotations = [{"image_id": i, "ground_truth": [deepcopy(box)]} for i in range(2)]
        guardrails = {"ap50_margin": 0.01, "ap75_margin": 0.01}
    else:
        reference = [{"top1_hit": True, "top5_hit": True},
                     {"top1_hit": False, "top5_hit": True}]
        candidate = [{"top1_hit": True, "top5_hit": True},
                     {"top1_hit": True, "top5_hit": True}]
        annotations = []
        guardrails = {"top5_accuracy_margin": 0.01}

    def records(role, values):
        return [{"image_id": i, role: deepcopy(value), "other": deepcopy(value)}
                for i, value in enumerate(values)]

    return QualityEvaluationRequest(
        reference_records=records("reference", reference),
        candidate_records=records("candidate", candidate),
        annotations=annotations,
        metric_gate_config={"task": task, "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
                            "guardrails": guardrails},
        repetitions=7, seed=1107, confidence_level=0.95, non_inferiority_margin=0.01,
        evaluator_factory=f"onnx_splitpoint_tool.quality_metrics:{task}_quality_evaluator",
        reference_prediction_field="other", candidate_prediction_field="other",
    )


def _change_consumed_field(request, task, role):
    changed = deepcopy(request)
    row = getattr(changed, role + "_records")[0]
    if task == "detection":
        row[role] = []
    else:
        row[role]["top1_hit"] = False
    return changed


class _InlineAliasExecutor:
    """Run the tiny real evaluator without another process pool."""
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


@pytest.fixture
def alias_service(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import quality_service
    monkeypatch.setattr(quality_service, "ProcessPoolExecutor", lambda **kwargs: _InlineAliasExecutor())
    with ManagementQualityService(tmp_path / "pair_cache", workers=1,
                                  statistics={"engine": "optimized_coco_v1"}) as service:
        yield service


@pytest.mark.parametrize("task", ["detection", "classification"])
@pytest.mark.parametrize("selection", ["alias", "canonical", "whole_record"])
def test_equivalent_alias_or_fully_bound_records_remain_valid(alias_service, task, selection):
    request = _standard_alias_request(task)
    if selection == "canonical":
        request = replace(request, reference_prediction_field="reference", candidate_prediction_field="candidate")
    elif selection == "whole_record":
        request = replace(request, reference_prediction_field=None, candidate_prediction_field=None)
    result = alias_service.evaluate(request, timeout=5)
    reused = alias_service.evaluate(request, timeout=5)
    assert result["cache_hit"] is False
    assert reused["cache_hit"] is True
    assert result["primary"] == reused["primary"]
    assert result["primary"]["reference"] > 0
    assert result["primary"]["candidate"] >= result["primary"]["reference"]
    assert result["primary"]["bootstrap_repetitions"] == 7


@pytest.mark.parametrize("task", ["detection", "classification"])
@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "pair_cache_warm"])
def test_standard_consumed_field_mismatch_is_rejected_before_pair_cache_lookup(
    alias_service, monkeypatch, task, role, warm,
):
    request = _standard_alias_request(task)
    if warm:
        alias_service.evaluate(request, timeout=5)
        assert alias_service.evaluate(request, timeout=5)["cache_hit"] is True
    changed = _change_consumed_field(request, task, role)
    # The declared alias alone cannot see the changed evaluator input.
    assert prepare_evaluation(changed)[0] == prepare_evaluation(request)[0]

    def forbidden_lookup(*args, **kwargs):
        raise AssertionError("actual standard fields must be validated before any pair-cache lookup")

    monkeypatch.setattr(alias_service.cache, "get", forbidden_lookup)
    with pytest.raises(ValueError) as caught:
        alias_service.evaluate(changed, timeout=5)
    assert str(caught.value)


def test_equivalent_alias_reuses_canonical_reference_components(tmp_path):
    request = _standard_alias_request("detection")
    canonical = replace(request, reference_prediction_field="reference", candidate_prediction_field="candidate")
    _, canonical_payload = prepare_evaluation(canonical)
    _, alias_payload = prepare_evaluation(request)
    for payload in (canonical_payload, alias_payload):
        payload["_statistics"] = {"engine": "optimized_coco_v1", "capture_draws": True,
                                  "reference_dir": str(tmp_path)}
    plan = deterministic_resample_plan(image_count=2, repetitions=7, seed=1107)
    assert reference_component_key(canonical_payload, plan, 0) == reference_component_key(alias_payload, plan, 0)
    expected = _evaluate(canonical_payload, plan)
    observed = _evaluate(alias_payload, plan)
    assert observed["statistics_observation"]["reference_cache_hit"] is True
    for field in ("component_points", "absolute_bootstrap", "bootstrap", "relative_loss_draws"):
        assert observed[field] == expected[field]


@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "reference_cache_warm"])
def test_reference_reuse_rejects_actual_field_drift_behind_stable_alias(tmp_path, role, warm):
    request = _standard_alias_request("detection")
    _, payload = prepare_evaluation(request)
    payload["_statistics"] = {"engine": "optimized_coco_v1", "capture_draws": True,
                              "reference_dir": str(tmp_path)}
    plan = deterministic_resample_plan(image_count=2, repetitions=7, seed=1107)
    if warm:
        _evaluate(payload, plan)
    changed = deepcopy(payload)
    changed[role + "_records"][0][role] = []
    assert prediction_fingerprint(changed[role + "_records"], payload_field="other") == payload[role + "_predictions_sha256"]
    if role == "reference":
        with pytest.raises(ValueError):
            reference_component_key(changed, plan, 0)
    else:
        # A candidate validation failure must not alter the independent
        # reference component's scientific identity.
        assert reference_component_key(changed, plan, 0) == reference_component_key(payload, plan, 0)
    with pytest.raises(ValueError):
        _evaluate(changed, plan)


class _AliasOnlyEvaluator:
    """Custom contracts may consume fields different from standard factories."""
    def __init__(self, reference, candidate, annotations, config):
        self.reference = np.asarray([row["other"] for row in reference], dtype=np.float64)
        self.candidate = np.asarray([row["other"] for row in candidate], dtype=np.float64)

    def evaluate(self, weights):
        total = float(np.sum(weights))
        reference = float(np.dot(weights, self.reference) / total)
        candidate = float(np.dot(weights, self.candidate) / total)
        return {"reference": reference, "candidate": candidate, "delta": candidate - reference}


def test_custom_factory_retains_its_own_selected_field_contract(alias_service):
    request = QualityEvaluationRequest(
        reference_records=[{"image_id": i, "other": 1.0, "reference": "unused standard field"} for i in range(2)],
        candidate_records=[{"image_id": i, "other": 0.75, "candidate": "different unused field"} for i in range(2)],
        annotations=[], metric_gate_config={}, repetitions=7, seed=1107,
        confidence_level=0.95, non_inferiority_margin=0.5,
        evaluator_factory=f"{__name__}:_AliasOnlyEvaluator",
        reference_prediction_field="other", candidate_prediction_field="other",
    )
    expected = alias_service.evaluate(request, timeout=5)
    changed = deepcopy(request)
    changed.reference_records[0]["reference"] = {"unrelated": "still unused"}
    reused = alias_service.evaluate(changed, timeout=5)
    assert reused["cache_hit"] is True
    assert expected["primary"] == reused["primary"]
    assert expected["primary"]["reference"] == 1.0
    assert expected["primary"]["candidate"] == 0.75
    assert expected["primary"]["bootstrap_repetitions"] == 7


def test_incomplete_owner_does_not_publish_and_releases_waiter(tmp_path):
    payload, plan = _payload(tmp_path)
    owner = ReferenceReuseEvaluator(payload, plan, 0, _config(payload))
    owner.evaluate(np.ones(2))
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(_evaluate, payload, plan)
        try:
            with pytest.raises(ValueError, match="incomplete reference"):
                owner.finish()
            assert not list(tmp_path.glob("*.json"))
        finally:
            owner.close()
        result = future.result(timeout=15)
    assert result["repetitions"] == len(plan)
    assert not result["statistics_observation"]["reference_cache_hit"]


def test_two_occupied_process_slots_finish_shared_reference_without_queued_owner(tmp_path):
    directory = tmp_path / "reference"
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    first, plan = _payload(directory, candidate=(True, True))
    second, _ = _payload(directory, candidate=(False, True))
    # Both admitted tasks fill this entire pool before either requests the same
    # reference. No spare slot exists for a separately queued reference owner.
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_occupied_worker, payload, plan, str(rendezvous), index)
                   for index, payload in enumerate((first, second))]
        results = [future.result(timeout=20) for future in futures]
    assert len({int(path.read_text()) for path in rendezvous.iterdir()}) == 2
    assert sorted(result["statistics_observation"]["reference_cache_hit"] for result in results) == [False, True]
    assert all(result["repetitions"] == len(plan) for result in results)


def test_crashed_owner_kernel_lock_releases_waiter_without_partial_hit(tmp_path):
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    directory = tmp_path / "reference"
    payload, plan = _payload(directory)
    process = mp.get_context("spawn").Process(target=_crashing_reference_owner,
        args=(payload, plan, str(rendezvous)))
    process.start()
    try:
        _wait_for(rendezvous / "ready")
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(_evaluate, payload, plan)
            (rendezvous / "exit").write_text("exit this test child")
            process.join(timeout=15)
            assert process.exitcode == 17
            result = future.result(timeout=15)
        assert result["repetitions"] == len(plan)
        assert not result["statistics_observation"]["reference_cache_hit"]
    finally:
        (rendezvous / "exit").touch()
        process.join(timeout=15)
