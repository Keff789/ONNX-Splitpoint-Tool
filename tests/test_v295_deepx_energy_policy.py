from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import native_producer_energy_plan as planner
from scripts import run_native_producer_energy_from_summary as energy_runner
from test_energy_final_contract_regressions import (
    _deepx_full_energy_identity,
    _local_full_hotloop_contract,
    _planner_runtime_options,
    _seal_full_contract,
)
from test_v2712_energy_resume import _plan, _plan_row


def _contract(backend: str, mode: str | None) -> dict:
    policy = {} if mode is None else {
        "deepx_runtime_mode": mode,
        "deepx_max_inflight": 4 if mode == "A4" else 1,
    }
    return {
        "backend": backend,
        "runtime_options": {"queue_depth": 3, **policy},
        "energy_workload": dict(policy),
    }


@pytest.mark.parametrize("backend", ["deepx_to_trt", "native_full_deepx"])
@pytest.mark.parametrize("mode", [None, "S1", "A4"])
def test_energy_policy_projects_only_explicit_bound_deepx_mode(backend, mode):
    contract = _contract(backend, mode)
    expected = {} if mode is None else {
        "deepx_runtime_mode": mode,
        "deepx_max_inflight": 4 if mode == "A4" else 1,
    }
    assert planner._deepx_energy_runtime_identity(contract) == expected
    assert planner._deepx_energy_runtime_identity(contract, row=expected) == expected


@pytest.mark.parametrize("backend", ["hailo8_to_trt", "hailo10h_to_trt", "generic"])
def test_energy_policy_does_not_interpret_other_backend_options(backend):
    assert planner._deepx_energy_runtime_identity({
        "backend": backend,
        "runtime_options": {"deepx_runtime_mode": "invalid"},
    }) == {}


@pytest.mark.parametrize("backend", ["deepx_to_trt", "native_full_deepx"])
@pytest.mark.parametrize("field,value", [
    ("deepx_runtime_mode", "S1"), ("deepx_max_inflight", 1),
    ("deepx_max_inflight", True),
])
def test_energy_policy_rejects_performance_row_from_different_mode(backend, field, value):
    with pytest.raises(ValueError, match="deepx_energy_performance_.*_mismatch"):
        planner._deepx_energy_runtime_identity(
            _contract(backend, "A4"), row={field: value},
        )


@pytest.mark.parametrize("field", ["deepx_runtime_mode", "deepx_max_inflight"])
@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_full_energy_policy_requires_matching_explicit_a4_workload(field, mutation):
    contract = _contract("native_full_deepx", "A4")
    if mutation == "missing":
        contract["energy_workload"].pop(field)
    else:
        contract["energy_workload"][field] = "S1" if field.endswith("mode") else 1
    with pytest.raises(ValueError, match="deepx_energy_workload_" + field):
        planner._deepx_energy_runtime_identity(contract)


def test_real_full_energy_contract_verifier_rejects_crossmode_workload(tmp_path):
    contract = _local_full_hotloop_contract(tmp_path, "deepx")
    contract["runtime_options"] = {
        **_planner_runtime_options(),
        "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
    }
    contract["energy_workload"].update(
        deepx_runtime_mode="A4", deepx_max_inflight=4,
    )
    _seal_full_contract(contract)
    identity = _deepx_full_energy_identity(contract, remote_root=tmp_path)
    verified, reason = planner._verify_full_command_contract(
        contract, expected_identity=identity,
    )
    assert verified is not None, reason
    contract["energy_workload"]["deepx_runtime_mode"] = "S1"
    _seal_full_contract(contract)
    verified, reason = planner._verify_full_command_contract(
        contract, expected_identity=identity,
    )
    assert verified is None
    assert reason == "full_command_contract_deepx_energy_workload_deepx_runtime_mode_mismatch"


def _resume_row(p03: bool) -> dict:
    row = {
        "backend": "deepx_to_trt", "model": "m", "case": "b001",
        "setup_id": "deepx", "successful_command_contract_sha256": "a" * 64,
    }
    if p03:
        row.update(
            runtime_success=True, energy_command_preflight_ok=True,
            full_baseline=False, split_has_valid_part2_input=True,
            native_energy_planner_admission={
                "schema": "onnx-splitpoint/native-energy-planner-admission",
                "schema_version": 1, "selected": True,
                "runtime_success": True, "energy_command_preflight_ok": True,
                "full_baseline": False, "split_has_valid_part2_input": True,
            },
        )
    return row


@pytest.mark.parametrize("p03", [False, True])
def test_resume_keeps_historical_digest_and_binds_new_deepx_mode(p03):
    row = _resume_row(p03)
    plan = {"schema": "energy-plan", "schema_version": 1}
    row_fields = (energy_runner._P03_RESUME_ROW_CONTRACT_FIELDS if p03
                  else energy_runner._RESUME_ROW_CONTRACT_FIELDS)
    plan_fields = (energy_runner._P03_RESUME_PLAN_CONTRACT_FIELDS if p03
                   else energy_runner._RESUME_PLAN_CONTRACT_FIELDS)
    historical = {
        "row": {field: row.get(field) for field in row_fields},
        "plan": {field: plan.get(field) for field in plan_fields},
    }
    if p03:
        historical["technical_admission"] = energy_runner._p03_resume_admission(row)
    for section in ("row", "plan"):
        historical[section].pop("campaign_cohort", None)
    digest = energy_runner._resume_contract_sha256(row, plan)
    assert digest == energy_runner._canonical_json_sha256(historical)
    a4 = {**row, "deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
    s1 = {**row, "deepx_runtime_mode": "S1", "deepx_max_inflight": 1}
    assert len({digest, energy_runner._resume_contract_sha256(a4, plan),
                energy_runner._resume_contract_sha256(s1, plan)}) == 3


def test_real_energy_resume_rejects_reusing_s1_capture_for_a4(tmp_path: Path):
    old = _plan_row(
        tmp_path / "measurements", backend="deepx_to_trt", model="m",
        case="b001", setup_id="deepx", attempt="original",
    )
    plan = _plan([old], attempt="original")
    previous = {"plan_payload": plan, "rows": [{"row": old, "ok": False}]}
    fresh = copy.deepcopy(plan)
    fresh["rows"][0].update(deepx_runtime_mode="A4", deepx_max_inflight=4)
    with pytest.raises(ValueError, match="resume contract drift"):
        energy_runner._validate_resume_selection(previous, fresh, [])


def test_energy_script_mirrors_match():
    root = Path(__file__).resolve().parents[1]
    for name in ("native_producer_energy_plan.py", "run_native_producer_energy_from_summary.py"):
        assert (root / "scripts" / name).read_bytes() == (
            root / "onnx_splitpoint_tool/resources/remote_scripts" / name
        ).read_bytes()


@pytest.mark.parametrize("backend", ["deepx_to_trt", "native_full_deepx"])
def test_prepared_energy_rejects_historical_s1_performance_for_a4(backend):
    from onnx_splitpoint_tool.workflow.native_energy_parent import _verify_deepx_performance_policy
    field = "full_command_contract" if backend.startswith("native_full_") else "native_command_contract"
    row = {"backend": backend, field: _contract(backend, "A4")}
    with pytest.raises(ValueError, match="runtime cohort differs"):
        _verify_deepx_performance_policy(row, {})
    _verify_deepx_performance_policy(row, {
        "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
    })
    _verify_deepx_performance_policy(row, {
        "deepx_prepared_feed_benchmark": {
            "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
        },
    })
    row[field] = _contract(backend, None)
    _verify_deepx_performance_policy(row, {})


@pytest.mark.parametrize("backend", ["hailo8_to_trt", "native_full_hailo10h"])
def test_prepared_energy_keeps_hailo_policy_unchanged(backend):
    from onnx_splitpoint_tool.workflow.native_energy_parent import _verify_deepx_performance_policy
    _verify_deepx_performance_policy({"backend": backend, "deepx_runtime_mode": "invalid"}, {})


@pytest.mark.parametrize("observation", [
    {},
    {"deepx_runtime_mode": "S1", "deepx_max_inflight": 1},
    {"full_command_contract": {"runtime_options": {}}, "deepx_runtime_mode": "A4", "deepx_max_inflight": 4},
    {"deepx_prepared_feed_benchmark": {"deepx_runtime_mode": "S1"}, "deepx_runtime_mode": "A4", "deepx_max_inflight": 4},
])
def test_caller_managed_energy_rejects_s1_or_conflicting_performance(observation):
    from onnx_splitpoint_tool.energy.caller_managed_reporting import _verify_deepx_execution_policy
    with pytest.raises(ValueError, match="Full DeepX runtime cohort differs"):
        _verify_deepx_execution_policy(
            _contract("native_full_deepx", "A4"), observation, label="test source",
        )


@pytest.mark.parametrize("mode", [None, "S1", "A4"])
def test_caller_managed_energy_preserves_matching_historical_and_a4_modes(mode):
    from onnx_splitpoint_tool.energy.caller_managed_reporting import _verify_deepx_execution_policy
    contract = _contract("native_full_deepx", mode)
    _verify_deepx_execution_policy(contract, contract["runtime_options"], label="plan")
    _verify_deepx_execution_policy(contract, {"full_command_contract": contract}, label="request")


@pytest.mark.parametrize("backend", ["hailo8_to_trt", "native_full_hailo10h", "native_full_tensorrt"])
def test_caller_managed_energy_preserves_other_backends(backend):
    from onnx_splitpoint_tool.energy.caller_managed_reporting import _verify_deepx_execution_policy
    _verify_deepx_execution_policy({"backend": backend}, {"deepx_runtime_mode": "invalid"}, label="unchanged")


def test_native_energy_import_receipt_projects_only_optional_deepx_policy():
    from onnx_splitpoint_tool.workflow.energy_parent_import import _normal_native_result_projection
    fields = ["backend", "model"]
    original = {"backend": "native_full_deepx", "model": "m"}
    assert _normal_native_result_projection(original, fields) == original
    policy = {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
    assert _normal_native_result_projection({**original, **policy}, fields) == {**original, **policy}
    hailo = {**original, "backend": "native_full_hailo10h"}
    assert _normal_native_result_projection({**hailo, **policy}, fields) == hailo


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_caller_managed_expectation_rejects_s1_request_before_source_reuse(tmp_path):
    from onnx_splitpoint_tool.energy.caller_managed_reporting import _full_expectation
    request = tmp_path / "request.json"
    request_sha = _write_json(request, {"row": {}})
    contract = {**_contract("native_full_deepx", "A4"), "contract_sha256": "a" * 64}
    contract_path = tmp_path / "command.json"
    contract_sha = _write_json(contract_path, contract)
    row = {"backend": "native_full_deepx", "model": "m", "setup_id": "deepx",
           "command_contract_file": str(contract_path), "command_contract_file_sha256": contract_sha,
           "successful_command_contract_sha256": "a" * 64,
           "deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
    sidecar = tmp_path / "expectation.json"
    sidecar_sha = _write_json(sidecar, {
        "request": str(request), "request_sha256": request_sha,
        "expected_actual_performance_mode": "classification_topk",
        "expected_auto_resolved_energy_mode": "classification_topk",
        **{field: row[field] for field in ("backend", "model", "setup_id")},
    })
    with pytest.raises(ValueError, match="runtime cohort differs: performance request"):
        _full_expectation({"path": str(sidecar), "sha256": sidecar_sha}, row)


def test_real_caller_managed_completion_rejects_s1_report_for_a4(tmp_path):
    from onnx_splitpoint_tool.energy.caller_managed_reporting import _full_runtime_proof
    contract_path = tmp_path / "command.json"
    _write_json(contract_path, _contract("native_full_deepx", "A4"))
    stdout = tmp_path / "workload_stdout.log"
    stdout_sha = _write_json(stdout, {"fresh_report_verified": True, "deepx_runtime_mode": "S1"})
    record = {
        "energy_work_units_used": 8,
        "collector_and_workload_log_diagnostics": {"workload_stdout": {
            "available": True, "bytes": stdout.stat().st_size, "sha256": stdout_sha,
        }},
    }
    importer = SimpleNamespace(_selected_energy_repeat_run_dir=lambda *_args, **_kwargs: (tmp_path, "verified"))
    row = {"backend": "native_full_deepx", "command_contract_file": str(contract_path)}
    with pytest.raises(ValueError, match="runtime cohort differs: actual energy completion"):
        _full_runtime_proof(importer, row, "command", tmp_path, record, {})
