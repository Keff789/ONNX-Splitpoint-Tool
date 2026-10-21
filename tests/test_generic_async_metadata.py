"""Scheduling experiments must not reuse synchronous performance or energy."""
import copy
import csv
import json
from pathlib import Path
import time

import pytest

from onnx_splitpoint_tool.runners import generic_energy as energy
from onnx_splitpoint_tool.runners.task_completion import valid_split_completion_repetitions
from onnx_splitpoint_tool.workflow import generic_completed_task as workflow
from tests.test_generic_completion_dispatch import prepared_case
from tests.test_generic_completion_reporting import case, result


ASYNC = {"policy": "hailo_async_slots_v1", "inflight": 8}
DEEPX_ASYNC = {"policy": "deepx_async_slots_v1", "inflight": 4}
BACKEND_POLICIES = [("hailo10h", "hailo_async_inflight", ASYNC),
                    ("deepx_m1", "deepx_async_inflight", DEEPX_ASYNC)]


def energy_config(provider="hailo10h"):
    config = case()
    config["original_run_cfg"] = {"stage1_provider": provider, "streaming_impl": "generic"}
    return config


def contract(config):
    return energy.build_contract(config, source_files=[{"path": "/fixture/runner.py", "sha256": "a" * 64}],
        runner_path="/fixture/runner.py", completion_runtime_mode="classification_top1_top5",
        execution_identity={"fixture": "metadata only, no hardware"})


@pytest.mark.parametrize("provider,flag,policy", BACKEND_POLICIES)
def test_legacy_energy_binding_is_unchanged_and_explicit_async_is_sealed(provider, flag, policy):
    config = energy_config(provider)
    expected = {key: config[key] for key in energy.OPERATION_FIELDS if key in config}
    expected.update(queue_depth=2, handoff_payload_copy=True)
    assert energy.operation_binding(config) == expected
    historical = contract(config)
    config["energy_capture"] = {"contract": historical}
    assert energy.validate_config(config) == historical

    config["stage1_execution"] = copy.deepcopy(policy)
    with pytest.raises(ValueError, match="contract mismatch"):
        energy.validate_config(config)
    revised = contract(config)
    assert revised["operation"]["stage1_execution"] == policy
    assert revised["command_contract_sha256"] != historical["command_contract_sha256"]
    config["energy_capture"] = {"contract": revised}
    assert energy.validate_config(config) == revised
    config["stage1_execution"]["inflight"] = 1
    with pytest.raises(ValueError, match="contract mismatch"):
        energy.validate_config(config)


@pytest.mark.parametrize("policy,provider,label", [
    *[(ASYNC, provider, "Hailo10H") for provider in ("hailo8", "deepx_m1", "tensorrt", "")],
    *[(DEEPX_ASYNC, provider, "DeepX M1") for provider in ("hailo8", "hailo10h", "tensorrt", "")],
])
def test_async_wrong_backend_fails_before_artifact_access(policy, provider, label):
    original = {"stage1_provider": provider}
    with pytest.raises(ValueError, match=label):
        workflow.runtime_configuration({"original_run_cfg": original}, "/unused", "unused", stage1_execution=policy)
    with pytest.raises(ValueError, match=label):
        energy.operation_binding({"original_run_cfg": original, "stage1_execution": policy})


@pytest.mark.parametrize("policy", [ASYNC, DEEPX_ASYNC])
def test_legacy_results_cannot_resume_async_or_mix_repetition_policies(policy):
    config = case()
    payload = result(config)
    assert len(valid_split_completion_repetitions(payload, config)) == 3
    config["stage1_execution"] = copy.deepcopy(policy)
    with pytest.raises(ValueError, match="stage1 execution"):
        valid_split_completion_repetitions(payload, config)
    payload["stage1_execution"] = copy.deepcopy(policy)
    with pytest.raises(ValueError, match="stage1 execution"):
        valid_split_completion_repetitions(payload, config)
    for repetition in payload["repetitions"]:
        repetition["stage1_execution"] = copy.deepcopy(policy)
    assert len(valid_split_completion_repetitions(payload, config)) == 3
    payload["repetitions"][1]["stage1_execution"]["inflight"] = 1
    with pytest.raises(ValueError, match="stage1 execution"):
        valid_split_completion_repetitions(payload, config)


@pytest.mark.parametrize("provider,flag,policy", BACKEND_POLICIES)
def test_prepared_plan_rejects_policy_drift(tmp_path, provider, flag, policy):
    row = prepared_case(tmp_path)
    config = row["runtime_config"]
    config["original_run_cfg"]["stage1_provider"] = provider
    config["stage1_execution"] = copy.deepcopy(policy)
    scope = tmp_path / "scope.csv"
    with scope.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row["identity"]))
        writer.writeheader()
        writer.writerow(row["identity"])
    source = tmp_path / "historical-plan.json"
    workflow.write_json(source, {"rows": [row]})
    plan = {"scope_mode": "audited_selective_remeasurement", "authorized_row_count": 1,
            "scope_source_sha256": workflow.file_sha256(scope), "rows": [row],
            "execution_id": "new-session", flag: policy["inflight"],
            "source_execution_plans": [{"path": str(source), "sha256": workflow.file_sha256(source)}]}
    assert workflow.validate_plan(plan, scope, tmp_path / "primary") == [row]
    config.pop("stage1_execution")
    with pytest.raises(ValueError, match="prepared policy"):
        workflow.validate_plan(plan, scope, tmp_path / "primary")


@pytest.mark.parametrize("hailo_inflight,deepx_inflight", [(None, None), (8, None), (None, 4), (8, 4)])
def test_prepare_opts_in_requested_backends_independently(tmp_path, monkeypatch, hailo_inflight, deepx_inflight):
    import onnx_splitpoint_tool.generic_completion_reporting as reporting
    rows = [{"setup_id": backend, "original_run_cfg": {"stage1_provider": backend},
             "case_output_relative": backend} for backend in ("hailo10h", "hailo8", "deepx_m1")]
    plan = {"original_run": str(tmp_path / "closed-original"), "scope_source": "unused", "rows": rows}
    source = tmp_path / "scope-plan.json"
    workflow.write_json(source, plan)
    monkeypatch.setattr(workflow, "validate_plan", lambda p, *a: p["rows"])
    monkeypatch.setattr(workflow, "bound_targets", lambda *a: {
        row["setup_id"]: {"remote": {"user": "fixture"}} for row in rows})
    monkeypatch.setattr(workflow, "original_generic_environment", lambda row: {
        "python": "/fixture/python", "environment": {}, "source": "fixture", "original_line": "fixture"})
    calls = []
    def configure(row, remote_root, execution_id, **kwargs):
        calls.append((row["setup_id"], kwargs))
        return dict(kwargs), []
    monkeypatch.setattr(workflow, "runtime_configuration", configure)
    monkeypatch.setattr(reporting, "write_completion_reports", lambda *a: None)
    output = workflow.prepare_execution(source, tmp_path / "new-output",
                                        hailo_async_inflight=hailo_inflight, deepx_async_inflight=deepx_inflight)
    prepared = json.loads(output.read_text())
    assert calls[0][1] == ({"stage1_execution": ASYNC} if hailo_inflight else {})
    assert calls[1][1] == {}
    assert calls[2][1] == ({"stage1_execution": DEEPX_ASYNC} if deepx_inflight else {})
    assert prepared.get("hailo_async_inflight") == hailo_inflight
    assert prepared.get("deepx_async_inflight") == deepx_inflight


@pytest.mark.parametrize("flags,expected", [
    (["--hailo-async-inflight", "8"], {"hailo_async_inflight": 8}),
    (["--deepx-async-inflight", "4"], {"deepx_async_inflight": 4}),
    (["--hailo-async-inflight", "8", "--deepx-async-inflight", "4"],
     {"hailo_async_inflight": 8, "deepx_async_inflight": 4}),
])
def test_cli_policy_is_prepare_only(tmp_path, monkeypatch, flags, expected):
    calls = []
    monkeypatch.setattr(workflow, "prepare_execution", lambda *a, **kw: calls.append(kw) or tmp_path / "plan.json")
    assert workflow.main(["--prepare", "scope.json", "--output-root", str(tmp_path), *flags]) == 0
    assert calls == [expected]
    monkeypatch.setattr(workflow, "execute_plan", lambda *a: pytest.fail("policy cannot change during execute"))
    with pytest.raises(SystemExit):
        workflow.main(["--execute", "plan.json", *flags])


@pytest.mark.parametrize("provider,flag,policy", BACKEND_POLICIES)
def test_actual_async_energy_result_binds_header_repetition_and_slots(tmp_path, monkeypatch, capsys,
                                                                    provider, flag, policy):
    import numpy as np
    from types import SimpleNamespace
    from onnx_splitpoint_tool.runners import generic_async, task_completion
    from tests.test_generic_async_stage1 import SlotSession
    from tests.test_generic_energy_contract import configured

    module, config, logits, _, preflight_path = configured(tmp_path, monkeypatch)
    config["original_run_cfg"]["stage1_provider"] = provider
    config["stage1_execution"] = copy.deepcopy(policy)
    config["runtime_source_files"].append({"path": str(Path(generic_async.__file__)),
                                           "sha256": energy.file_hash(generic_async.__file__)})
    config["runtime_source_files"].sort(key=lambda item: item["path"])
    bound = energy.build_contract(config, source_files=config["runtime_source_files"],
        runner_path=str(Path(module.__file__)), completion_runtime_mode="classification_top1_top5")
    config["energy_capture"] = {"contract": bound}
    runtime = config["_runtime_energy"]
    runtime["energy_command_contract_sha256"] = bound["command_contract_sha256"]
    preflight = energy.build_preflight(config, nonce=runtime["energy_preflight_nonce"],
                                      expected_sha256=bound["command_contract_sha256"])
    preflight_path.write_text(json.dumps(preflight))

    class LogitSession(SlotSession):
        def _materialize_binding_outputs(self, binding, *, copy_outputs):
            assert copy_outputs is True
            return {"cut": logits.copy()}

    if provider == "hailo10h":
        session = LogitSession()
        adapter = generic_async.HailoAsyncStage1(session,
            lambda: {"input": len(session.submitted)}, inflight=policy["inflight"])
    else:
        prepared = np.load(config["input_binding"]["prepared_input"]["path"])

        class DeepXEngine:
            def __init__(self):
                self.submitted, self.pending = [], {}

            def get_input_tensors_info(self):
                return [{"name": "images", "shape": list(prepared.shape), "dtype": str(prepared.dtype)}]

            def get_output_tensors_info(self):
                return [{"name": "cut", "shape": list(logits.shape), "dtype": str(logits.dtype)}]

            def run_async(self, inputs, *, user_arg, output_buffer):
                np.testing.assert_array_equal(inputs[0], prepared)
                value = np.ndarray(logits.shape, dtype=logits.dtype, buffer=output_buffer)
                value[:] = logits
                job = len(self.submitted)
                self.submitted.append(job)
                self.pending[job] = [value]
                return job

            def wait(self, job):
                return self.pending.pop(job)

        session = DeepXEngine()
        generic_session = SimpleNamespace(_engine=session,
            _prepare_async_inputs=lambda inputs: [inputs["images"]],
            _materialize_outputs=lambda outputs: {"cut": outputs[0]})
        adapter = generic_async.DeepXAsyncStage1(generic_session,
            lambda: {"images": prepared}, inflight=policy["inflight"])
    real_timer = task_completion.measure_split_completion
    # CPU scheduling fixture only: synthetic time is never scientific evidence.
    monkeypatch.setattr(task_completion, "measure_split_completion",
        lambda *a, **kw: real_timer(*a, **kw, clock=lambda: time.perf_counter_ns() * 1000))
    output = tmp_path / "async-energy"
    output.mkdir()
    artifacts = {key: Path(value["path"]) for key, value in config["expected_runtime_artifacts"].items()}
    assert module._run_generic_split_completion(config, output,
        lambda: pytest.fail("async energy must not use the synchronous producer"), lambda x: x,
        lambda x: {"logits": x["cut"]}, artifacts, clone_payload=None,
        repetition_index=1, stage1_async=adapter) == 0
    payload = json.loads((output / "generic_energy_results.json").read_text())
    row = energy.validate_energy_result(payload, config, result_root=output)
    assert payload["stage1_execution"] == row["stage1_execution"] == policy
    assert row["completed_work_units"] == len(session.submitted) > 0
    assert not adapter.pending and not session.pending
    assert energy.COMPLETION_MARKER in capsys.readouterr().out
    for target in ("header", "repetition"):
        for replacement in (None, {"policy": policy["policy"], "inflight": 1}):
            changed = copy.deepcopy(payload)
            owner = changed if target == "header" else changed["repetitions"][0]
            if replacement is None:
                owner.pop("stage1_execution")
            else:
                owner["stage1_execution"] = replacement
            with pytest.raises(ValueError, match="mismatch"):
                energy.validate_energy_result(changed, config, result_root=output)
