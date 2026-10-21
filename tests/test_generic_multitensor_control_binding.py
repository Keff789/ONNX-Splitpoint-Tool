"""New proof cannot be inferred from old controls or dropped on import."""
from copy import deepcopy
import json

import pytest

from onnx_splitpoint_tool.runners import generic_mixed_control_binding as binding
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1
from tests.test_generic_mixed_control_binding import bound, rebind
from tests.test_generic_multitensor_control import DeepXSession, callbacks, contract


def execute(bound, tmp_path):
    config, payload, _ = bound
    config["boundary_observation"] = contract()
    payload["boundary_observation"] = deepcopy(config["boundary_observation"])
    rebind(config, payload)
    first, transfer, second, runtime = callbacks(config["boundary_observation"])
    session = DeepXSession()
    original = lambda: pytest.fail("throughput prepared input consumed by observation")
    adapter = DeepXAsyncStage1(session, original, inflight=4)
    proof = binding.run_bound_mixed_control(config, tmp_path, 1,
        serial_stage1=first, transfer=transfer, stage2=second, stage1_async=adapter,
        prepare_async_feed=lambda feed: feed, clone_payload=None, runner_source=__file__,
        boundary_runtime=runtime)
    assert not adapter.pending and not session._engine.jobs and adapter.prepare_inputs is original
    repetition = {"stage1_execution": config["stage1_execution"], "handoff_payload_copy": False,
                  "mixed_input_control": proof}
    return config, proof, repetition


def test_persisted_actual_maps_normal_import_and_explicit_summary(bound, tmp_path):
    config, proof, repetition = execute(bound, tmp_path)
    assert binding.validate_bound_mixed_result(repetition, config, result_root=tmp_path) == proof
    assert binding.validate_bound_mixed_result(repetition, config) == proof
    report = json.loads((tmp_path / proof["report"]["path"]).read_text())
    assert report["boundary_observation"] == proof["boundary_observation"]
    assert report["boundary_observation"]["observed_boundary_tensor_count"] == 2
    assert len(report["observations"]) == 32
    for item in report["observations"]:
        assert item["serial"] == item["mixed"]
        for phase in ("serial", "mixed"):
            value = item["boundary_observation"][phase]
            assert value["part1_at_collection"]["cut_b"]["tensor_sha256"]
            assert value["part2_before_inference"]["trt_b"]["tensor_sha256"]


@pytest.mark.parametrize("fault", ["strip_summary", "strip_report", "strip_task", "one_tensor", "second_content",
    "queued_content", "p2_content", "shape", "dtype", "layout", "mapping", "slot_ownership", "slot_overlap", "unbound_contract"])
def test_importer_rejects_downgrade_and_boundary_tampering(bound, tmp_path, fault):
    config, proof, repetition = execute(bound, tmp_path)
    path = tmp_path / proof["report"]["path"]
    report = json.loads(path.read_text())
    if fault == "strip_summary":
        proof.pop("boundary_observation")
    elif fault == "strip_report":
        report.pop("boundary_observation")
    elif fault == "strip_task":
        report["observations"][13].pop("boundary_observation")
    elif fault == "one_tensor":
        report["observations"][13]["boundary_observation"]["mixed"]["part2_before_inference"].pop("trt_b")
    elif fault == "second_content":
        task = report["observations"][13]["boundary_observation"]["mixed"]
        for key in ("part1_at_collection", "part1_before_transfer", "part1_after_transfer"):
            task[key]["cut_b"]["tensor_sha256"] = "f" * 64
    elif fault == "queued_content":
        report["observations"][13]["boundary_observation"]["mixed"]["part1_before_transfer"]["cut_b"]["tensor_sha256"] = "f" * 64
    elif fault == "p2_content":
        report["observations"][13]["boundary_observation"]["mixed"]["part2_before_inference"]["trt_b"]["tensor_sha256"] = "f" * 64
    elif fault in ("shape", "dtype", "layout"):
        tensor = report["observations"][13]["boundary_observation"]["mixed"]["part2_before_inference"]["trt_b"]
        tensor[{"shape": "shape", "dtype": "dtype", "layout": "strides"}[fault]] = {
            "shape": [1, 2, 3, 4], "dtype": "float32", "layout": [1]}[fault]
    elif fault == "mapping":
        report["observations"][13]["boundary_observation"]["mixed"]["runtime"]["part2_to_part1"]["trt_b"] = "cut_a"
    elif fault == "slot_ownership":
        proof["boundary_observation"]["slot_buffer_ownership"]["slots_disjoint"] = False
    elif fault == "slot_overlap":
        ownership = proof["boundary_observation"]["slot_buffer_ownership"]
        ownership["slots"][1]["buffers"] = deepcopy(ownership["slots"][0]["buffers"])
        report["boundary_observation"] = deepcopy(proof["boundary_observation"])
    else:
        config["boundary_observation"]["transfer_contract"] = {"unbound": True}
    path.write_text(json.dumps(report))
    proof["report"]["sha256"] = binding._sha(path)
    with pytest.raises(ValueError):
        binding.validate_bound_mixed_result(repetition, config, result_root=tmp_path)


def test_historical_sidecar_is_not_new_boundary_evidence(bound):
    config, payload, _ = bound
    assert binding.load_bound_mixed_control(config) is not None
    config["boundary_observation"] = contract()
    with pytest.raises(ValueError, match="sidecar differs"):
        binding.load_bound_mixed_control(config)
    payload["boundary_observation"] = deepcopy(config["boundary_observation"])
    rebind(config, payload)
    assert binding.load_bound_mixed_control(config) is not None
    config.pop("boundary_observation")
    with pytest.raises(ValueError):
        binding.load_bound_mixed_control(config)


def test_new_boundary_report_cannot_be_attached_to_old_config(bound, tmp_path):
    config, _, repetition = execute(bound, tmp_path)
    config.pop("boundary_observation")
    with pytest.raises(ValueError, match="not requested"):
        binding.validate_bound_mixed_result(repetition, config, result_root=tmp_path)
