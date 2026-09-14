"""REV4 T08.1–T08.6: real evaluator, cache, legacy and report consumers.

Recorded Complete-Set rows are read-only projections. All newly evaluated
requests in this module are explicitly synthetic offline fixtures.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.quality_cache import (
    CACHE_SCHEMA, PersistentQualityCache, QualityFingerprintError, json_fingerprint,
)
from onnx_splitpoint_tool.quality_replay import _write_csv
from onnx_splitpoint_tool.quality_result_contract import project_quality_result
from onnx_splitpoint_tool.quality_service import (
    ManagementQualityService, QualityEvaluationRequest, _combine_evaluation_shards,
    _evaluate_payload, _evaluate_payload_shard, _prediction_identity_is_bound,
    deterministic_resample_plan, prepare_evaluation,
)
from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
from onnx_splitpoint_tool.workflow import scientific_reporting as reporting

FIXTURES = Path(__file__).parent / "fixtures/v27931_quality"


def _golden():
    return json.loads((FIXTURES / "synthetic_v30_bootstrap_golden.json").read_text())


def _legacy():
    return json.loads((FIXTURES / "legacy_complete_set_quality.json").read_text())


def _request(*, fast_fail=False, identical=False):
    data = _golden()["request"]
    if fast_fail:
        for row in data["candidate_records"]:
            row["candidate"]["top1_hit"] = False
    if identical:
        for reference, candidate in zip(data["reference_records"], data["candidate_records"]):
            candidate["candidate"] = dict(reference["reference"])
    return QualityEvaluationRequest(**data)


def _evaluate(request):
    _, payload = prepare_evaluation(request)
    return _evaluate_payload(payload)


def _scoped(result):
    return {**result, "model_id": "synthetic_classifier", "task": "classification",
            "case_id": "b001", "variant": "composed", "backend": "deepx_to_trt",
            "setup_id": "orin_nx_deepx_01", "technical_status": "completed"}


def test_t08_1_computed_bootstrap_matches_v30_plan_shards_and_components():
    golden = _golden()
    _, payload = prepare_evaluation(QualityEvaluationRequest(**golden["request"]))
    plan = deterministic_resample_plan(image_count=8, repetitions=37, seed=731)
    assert plan.tolist() == golden["plan"]
    shards = [_evaluate_payload_shard(payload, plan[:19], shard_index=0, repetition_offset=0),
              _evaluate_payload_shard(payload, plan[19:], shard_index=1, repetition_offset=19)]
    for shard, expected in zip(shards, golden["shards"]):
        assert {key: value for key, value in shard.items() if key != "worker_elapsed_s"} == expected
    result = _combine_evaluation_shards(payload, shards, elapsed_s=0.0, workers_requested=2)
    assert result["seed_schema"] == golden["result"]["seed_schema"]
    assert result["algorithm_version"] == golden["result"]["algorithm_version"]
    assert result["decision"] == golden["result"]["decision"]
    for name, component in (("primary", result["primary"]), ("top5_accuracy", result["guardrails"]["top5_accuracy"])):
        previous = golden["result"]["primary"] if name == "primary" else golden["result"]["guardrails"][name]
        assert {key: component[key] for key in previous} == previous
        assert component["ci_computed"] is True
        assert component["decision_basis"] == "paired_bootstrap_lower_bound"
        assert component["gate_bound_value"] == component["ci_low"]


def test_t08_2_real_writer_cache_loader_retains_point_fail_without_ci(tmp_path):
    with ManagementQualityService(tmp_path / "quality_cache", workers=2) as service:
        result = service.evaluate(_request(fast_fail=True))
        hit = service.evaluate(_request(fast_fail=True))
    assert result["quality_result_contract_version"] == 3
    assert hit["cache_hit"] is True
    assert result["decision"] == hit["decision"] == "fail"
    primary = result["primary"]
    assert primary["delta"] == -0.75 and primary["margin"] == 0.01
    assert primary["ci_low"] is None and primary["ci_high"] is None
    assert primary["bootstrap_repetitions_requested"] == 37
    assert primary["bootstrap_repetitions"] == 0
    assert primary["ci_computed"] is False
    assert primary["decision_basis"] == "point_estimate_below_non_inferiority_margin"
    assert primary["gate_bound_value"] == primary["delta"]
    assert result["guardrails"]["top5_accuracy"]["decision"] == "inconclusive"


def test_t08_3_real_ap75_guardrail_fail_does_not_pass_uncomputed_siblings():
    def box(x2):
        return {"class_id": 0, "score": 0.9, "x1": 0.0, "y1": 0.0, "x2": x2, "y2": 10.0}
    truth = {key: value for key, value in box(10.0).items() if key != "score"}
    request = QualityEvaluationRequest(
        reference_records=[{"image_id": "synthetic-image", "reference": [box(10.0)], "ground_truth": [truth]}],
        candidate_records=[{"image_id": "synthetic-image", "candidate": [box(7.0)]}],
        annotations=[{"image_id": "synthetic-image", "ground_truth": [truth]}],
        metric_gate_config={"task": "detection", "primary_metric": "coco_ap_50_95", "non_inferiority_margin": 1.0,
                            "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01}},
        repetitions=19, seed=731, confidence_level=0.95, non_inferiority_margin=1.0,
        evaluator_factory="onnx_splitpoint_tool.quality_metrics:detection_quality_evaluator",
        reference_prediction_field="reference", candidate_prediction_field="candidate")
    result = _evaluate(request)
    assert result["decision"] == "fail"
    assert result["guardrails"]["ap75"]["decision"] == "fail"
    for component in (result["primary"], result["guardrails"]["ap50"]):
        assert component["decision"] == "inconclusive"
        assert component["decision_basis"] == "bootstrap_not_computed_other_component_point_fail"
        assert component["gate_bound_value"] is None
    assert all(component["ci_low"] is None and component["ci_high"] is None and not component["ci_computed"]
               for component in (result["primary"], *result["guardrails"].values()))
    recorded = _legacy()["yolo26s_guardrail_fail"]
    projected = project_quality_result(recorded)
    assert recorded["primary"]["decision"] == "pass"
    assert projected["decision"] == "fail"
    assert projected["primary"]["decision"] == "inconclusive"
    assert projected["primary"]["source_component_decision"] == "pass"


def test_t08_4_identity_requires_bound_payload_not_equal_delta_or_missing_hash():
    result = _evaluate(_request(identical=True))
    assert result["decision"] == "pass"
    for component in (result["primary"], *result["guardrails"].values()):
        assert component["delta"] == 0.0 and component["gate_bound_value"] == 0.0
        assert component["prediction_identity_verified"] is True
        assert component["uncertainty_status"] == "deterministic_identity"
        assert component["ci_computed"] is False and component["ci_low"] is None and component["ci_high"] is None
    distinct = _evaluate(_request())
    assert distinct["primary"]["delta"] == 0.0
    assert distinct["primary"]["ci_computed"] is True
    _, payload = prepare_evaluation(_request())
    payload["candidate_predictions_sha256"] = payload["reference_predictions_sha256"]
    assert _prediction_identity_is_bound(payload) is False
    payload.pop("candidate_predictions_sha256")
    payload.pop("reference_predictions_sha256")
    assert _prediction_identity_is_bound(payload) is False


def test_t08_4_worker_rejects_unbound_identity_shortcut():
    _, payload = prepare_evaluation(_request())
    plan = deterministic_resample_plan(image_count=8, repetitions=37, seed=731)
    shard = _evaluate_payload_shard(payload, plan, shard_index=0, repetition_offset=0)
    shard["skipped_reason"] = "candidate_reference_identical"
    with pytest.raises(ValueError, match="lacks verified prediction identity"):
        _combine_evaluation_shards(payload, [shard], elapsed_s=0.0, workers_requested=1)


def test_t08_5_legacy_result_cache_bytes_and_model_artifacts_stay_unchanged(tmp_path):
    path = FIXTURES / "legacy_complete_set_quality.json"
    original = path.read_bytes()
    legacy = _legacy()
    for result in legacy.values():
        projected = project_quality_result(result)
        assert projected["quality_result_contract_version"] == 2
        assert projected["legacy_quality_result"] is True
        assert projected["quality_result_projection_contract_version"] == 3
        assert projected["decision"] == result["decision"]
    assert path.read_bytes() == original
    golden = _golden()
    old_key = golden["result"]["evaluation_fingerprint"]
    new_key, _ = prepare_evaluation(_request())
    assert new_key != old_key
    cache = PersistentQualityCache(tmp_path / "cache")
    old_path = cache.path_for(old_key)
    old_path.parent.mkdir(parents=True)
    old_payload = {"schema": CACHE_SCHEMA, "schema_version": 2, "result_contract_version": 2,
                   "evaluation_fingerprint": old_key, "result_sha256": json_fingerprint(golden["result"]),
                   "result": golden["result"]}
    old_path.write_text(json.dumps(old_payload))
    old_bytes = old_path.read_bytes()
    model = tmp_path / "existing.dxnn"
    model.write_bytes(b"synthetic preservation sentinel, not a model")
    assert cache.get(old_key) is None
    fresh = _evaluate(_request())
    cache.put(new_key, fresh)
    assert cache.get(new_key)["quality_result_contract_version"] == 3
    assert old_path.read_bytes() == old_bytes
    assert model.read_bytes() == b"synthetic preservation sentinel, not a model"


def test_t08_5_v3_cache_rejects_pseudo_bounds_and_false_identity(tmp_path):
    result = _evaluate(_request(fast_fail=True))
    cache = PersistentQualityCache(tmp_path)
    result["primary"]["ci_low"] = result["primary"]["delta"]
    with pytest.raises(QualityFingerprintError, match="pseudo confidence"):
        cache.put(result["evaluation_fingerprint"], result)
    result = _evaluate(_request(identical=True))
    result["candidate_predictions_sha256"] = "f" * 64
    with pytest.raises(QualityFingerprintError, match="bound deterministic identity"):
        cache.put(result["evaluation_fingerprint"], result)


@pytest.mark.parametrize("legacy", [False, True], ids=["v31_writer_loader", "recorded_v29_complete_set"])
def test_t08_6_json_csv_markdown_latex_accuracy_and_plots_share_null_uncertainty(tmp_path, monkeypatch, legacy):
    import matplotlib
    matplotlib.use("Agg", force=True)
    source = _legacy()["mobilenet_full_point_fail"] if legacy else _scoped(_evaluate(_request(fast_fail=True)))
    original = json.dumps(source, sort_keys=True)
    source_path = tmp_path / "source_result.json"
    source_path.write_text(original)
    loaded = json.loads(source_path.read_text())
    projection = reporting.project_central_quality_status({"status": "ok", "request_count": 1, "results": [loaded]})
    central = projection["results"][0]
    assert central["task_quality_ci_low"] is None and central["task_quality_ci_high"] is None
    row = {"model_id": source["model_id"], "task": "classification", "variant": "composed",
           "backend": "deepx_to_trt", "case_id": "b001", "task_quality_gate": loaded}
    apply_accuracy_gate_to_row(row)
    assert row["accuracy_gate_ci_low"] is None and row["accuracy_gate_ci_high"] is None
    assert row["accuracy_gate_decision"] == "fail"
    assert row["accuracy_gate_ci_computed"] is False
    assert row["accuracy_gate_gate_bound_value"] == source["primary"]["delta"]
    scientific = reporting._scientific_row(row)
    assert scientific["task_quality_ci_low"] is None
    assert scientific["task_quality_uncertainty_status"] == "not_computed_fast_fail"
    _write_csv(tmp_path / "replay.csv", [loaded])
    with (tmp_path / "replay.csv").open() as stream:
        replay_row = next(csv.DictReader(stream))
    assert replay_row["primary_ci_low"] == replay_row["primary_ci_high"] == ""
    assert replay_row["primary_ci_computed"] == "False"
    captured = {}
    real_outputs = reporting._figure_outputs
    def capture(figure, base):
        if base.name == "task_quality_noninferiority":
            captured["interval_count"] = len(figure.axes[0].collections)
            captured["labels"] = figure.axes[0].get_legend_handles_labels()[1]
        return real_outputs(figure, base)
    monkeypatch.setattr(reporting, "_figure_outputs", capture)
    paths = reporting._write_reports(tmp_path / "reports", {"rows": [scientific], "central_quality_results": [central],
                                                            "central_quality_reporting": projection, "summary": {}})
    task_json = json.loads((tmp_path / "reports/task_quality.json").read_text())[0]
    assert task_json["task_quality_ci_low"] is None and task_json["task_quality_ci_high"] is None
    assert task_json["task_quality_ci_computed"] is False
    with (tmp_path / "reports/task_quality.csv").open() as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["task_quality_ci_low"] == csv_row["task_quality_ci_high"] == ""
    assert "Unsicherheit nicht berechnet" in (tmp_path / "reports/scientific_report.md").read_text()
    assert "Unsicherheit nicht berechnet" in (tmp_path / "reports/thesis_tables/task_quality_gates.tex").read_text()
    assert captured["interval_count"] == 0
    assert captured["labels"] == ["Unsicherheit nicht berechnet"]
    assert (tmp_path / "reports/figures/task_quality_noninferiority.png").is_file()
    assert source_path.read_text() == original


def test_t08_6_computed_plot_uses_actual_interval_without_delta_substitution(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg", force=True)
    row = {"row_role": "performance_observation", "task_quality_delta": -0.005,
           "task_quality_ci_low": -0.011, "task_quality_ci_high": 0.001,
           "task_quality_bootstrap_repetitions": 37, "task_quality_bootstrap_skipped_reason": "",
           "task_quality_ci_computed": True, "task_quality_margin": 0.01}
    seen = []
    def capture(figure, base):
        if base.name == "task_quality_noninferiority":
            seen.extend(figure.axes[0].collections[0].get_segments())
        return []
    monkeypatch.setattr(reporting, "_figure_outputs", capture)
    reporting._make_figures([row], [], [], tmp_path)
    assert len(seen) == 1
    assert np.asarray(seen[0])[:, 0].tolist() == [-0.011, 0.001]


def test_t08_6_gui_displays_the_exported_uncertainty_text_without_numeric_coercion(tmp_path):
    # The existing application selects TkAgg on import. Isolate that real GUI
    # import from this module's already-used headless plotting backend, just as
    # the release gate isolates its existing GUI check. No product monkeypatch.
    import subprocess
    import sys
    code = r"""
from types import SimpleNamespace
from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
from onnx_splitpoint_tool.workflow import scientific_reporting as reporting
row = reporting._task_quality_bound_display({"task_quality_delta": -0.066, "task_quality_ci_low": None,
      "task_quality_ci_high": None, "task_quality_ci_computed": False,
      "task_quality_bootstrap_repetitions": 0,
      "task_quality_bootstrap_skipped_reason": "point_estimate_below_non_inferiority_margin"})
text = reporting._md_table([row], [("task_quality_delta", "Delta"), ("task_quality_bound_value", "CI"),
                                  ("task_quality_uncertainty_label", "Uncertainty")])
class Widget:
    value = ""
    def configure(self, **kwargs): pass
    def delete(self, *args): self.value = ""
    def insert(self, _, text): self.value += text
    def see(self, *args): pass
widget = Widget()
SplitPointAnalyserGUI._eval_workflow_text_set(SimpleNamespace(eval_workflow_summary_text=widget), text)
assert widget.value == text
assert "Unsicherheit nicht berechnet" in widget.value
assert "None" not in widget.value
print("GUI_UNCERTAINTY_PASS")
"""
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "GUI_UNCERTAINTY_PASS" in child.stdout


def test_t00_2_original_complete_set_deepx_full_contract_remains_valid():
    from onnx_splitpoint_tool.quality_service import _validate_deepx_candidate_execution_contract
    path = FIXTURES / "original_complete_set_deepx_full_request.json"
    before = path.read_bytes()
    provenance = json.loads((FIXTURES / "PROVENANCE.json").read_text())
    assert hashlib.sha256(before).hexdigest() == provenance["original_complete_set_deepx_full_request"]["sha256"]
    request = json.loads(before)
    validated, digest = _validate_deepx_candidate_execution_contract(request["producer_identity"], role="request", task="detection")
    assert validated == request["producer_identity"]
    assert digest == "93b801e18576edbae1a62d6580b8cf05d65dd0e809b880022c746a5825619d18"
    assert validated["endpoint"]["identity"]["stage"] == "decoded_pre_nms"
    assert path.read_bytes() == before
    # The fixture validates contracts; it cannot stand in for an absent B500.
    assert not (FIXTURES / request["candidate"]["path"]).exists()


@pytest.mark.parametrize("mutation", ["outer_hash", "inner_preprocessing", "inner_attestation"])
def test_t00_2_original_complete_set_contract_mutations_fail_closed(mutation):
    from onnx_splitpoint_tool.quality_service import QualityArtifactIntegrityError, _validate_deepx_candidate_execution_contract
    path = FIXTURES / "original_complete_set_deepx_full_request.json"
    before = path.read_bytes()
    producer = json.loads(before)["producer_identity"]
    # This is a synthetic negative copy; source bytes are never resealed.
    if mutation == "outer_hash":
        producer["model"]["runtime_artifact_sha256"] = "e" * 64
    elif mutation == "inner_preprocessing":
        producer["preprocessing"]["identity"]["synthetic_negative_mutation"] = True
    else:
        producer["completed_task_endpoint_attestation"]["completed_frames"] += 1
    if mutation != "outer_hash":
        producer.pop("producer_identity_sha256")
        producer["producer_identity_sha256"] = json_fingerprint(producer)
    with pytest.raises(QualityArtifactIntegrityError):
        _validate_deepx_candidate_execution_contract(producer, role="request", task="detection")
    assert path.read_bytes() == before


@pytest.mark.parametrize("case", ["yolo11_computed", "yolo26s_guardrail_fail"])
def test_t08_6_existing_evidence_verifier_projects_original_detection_rows_read_only(case):
    from onnx_splitpoint_tool.existing_evidence_verifier import _quality_projection
    row = _legacy()[case]
    before = copy.deepcopy(row)
    projected = _quality_projection(row)
    assert projected["decision"] == row["decision"]
    if case == "yolo26s_guardrail_fail":
        assert projected["primary"]["ci_low"] is None
        assert projected["primary"]["decision"] == "inconclusive"
        assert projected["primary"]["source_component_decision"] == "pass"
    else:
        assert projected["primary"]["ci_low"] == row["primary"]["ci_low"]
    assert row == before


@pytest.mark.parametrize("mode", ["computed", "point_fail", "identical"])
def test_t08_6_existing_evidence_verifier_accepts_v3_writer_and_rejects_mutation(mode):
    from onnx_splitpoint_tool.existing_evidence_verifier import ExistingEvidenceError, _quality_component_projection
    result = _evaluate(_request(fast_fail=mode == "point_fail", identical=mode == "identical"))
    component = result["primary"]
    kwargs = dict(label="synthetic.production.primary", expected_metric="top1_accuracy", expected_margin=0.01,
                  expected_n=8, contract_version=3, prediction_identity_verified=mode == "identical")
    projected = _quality_component_projection(component, **kwargs)
    assert projected["ci_low"] == component["ci_low"]
    assert projected["decision"] == component["decision"]
    invalid = copy.deepcopy(component)
    invalid["gate_bound_value"] = 0.987
    with pytest.raises(ExistingEvidenceError):
        _quality_component_projection(invalid, **kwargs)
