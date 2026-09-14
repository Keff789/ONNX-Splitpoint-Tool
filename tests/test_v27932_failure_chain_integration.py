"""F1: real Stage planning/dispatch, disk persistence, report and energy chain.

Only physical SSH/rsync and irrelevant host telemetry are replaced.  The local
BenchmarkSet admission, quality loaders, job planner, both persistence sites,
final-report subprocess, matrix, energy planner and GUI projections run normally.
Upstream bindings are existing contract fixtures, not v32 hardware evidence.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.native_job_identity import native_identity_key
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy, materialize_native_split_preselection,
    seal_native_split_quality_binding,
)
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.workflow import runner as workflow
from onnx_splitpoint_tool.workflow.evidence_status import (
    derive_native_evidence_status, project_native_evidence_status,
)
from scripts import native_producer_energy_plan as energy
from tests.test_v269d_trt_quality_chain import _result, _strict_producer
from tests.test_v269d_trt_central_quality_producer import _seal_producer
from tests.test_v269f_variant_native_split_quality_first import _binding_and_summary
from tests.test_v269f_native_split_receipt_validation import _fixture_payload, _artifact

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SUMMARY = ROOT / "tests/fixtures/v270b/central_quality_summary_2.70a_smoke.json"
RUN_ID = "resnet_yolo26s_yolo7_20260722_152042"
SETUPS = {"deepx": "orin_nx_deepx_m1_01", "hailo8": "hailo8_setup", "hailo10h": "orin_nx_hailo10_01"}
MODEL, CASE = "yolo26s", "b038"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hailo10_summary(tmp_path):
    """Extend the existing HEF fixture using today's H10 policy, without hardware."""
    payload, paths = _fixture_payload(tmp_path)
    backend, setup = "hailo10h_to_trt", SETUPS["hailo10h"]
    policy = known_native_split_policy(model_id=MODEL, case_id=CASE, setup_id=setup, backend=backend)
    metadata = _read(paths["boundary_metadata"])
    metadata.update(backend=backend, source_run_id=backend, setup_id=setup,
                    boundary_layout=policy["boundary_layout"], boundary_transform=policy["boundary_transform"])
    metadata["boundary_tensor"].update(shape=[2, 3, 4], canonical_part2_shape=[1, 2, 3, 4])
    metadata.pop("metadata_sha256")
    metadata["metadata_sha256"] = canonical_json_sha256(metadata)
    _write(paths["boundary_metadata"], metadata)
    selection = materialize_native_split_preselection(policy=policy,
        part1_artifact=_artifact(paths["part1_runtime"]), boundary_metadata=metadata,
        boundary_metadata_artifact=_artifact(paths["boundary_metadata"]))
    meta = _read(paths["native_trt_meta"])
    meta["inputs"][0]["shape"] = [1, 2, 3, 4]
    meta["uint8_cast_bridge"]["input_shape"] = [1, 2, 3, 4]
    meta["uint8_cast_bridge"]["boundary_layout"] = {
        "requested": policy["boundary_layout"], "effective": policy["boundary_layout"], "applied": False,
    }
    _write(paths["native_trt_meta"], meta)
    boundary = {key: copy.deepcopy(selection[key]) for key in payload["boundary_contract"]}
    payload.update(eval_run_id=RUN_ID, source_run_id=backend, preselection=selection,
        preselection_sha256=selection["selection_sha256"], boundary_contract=boundary,
        boundary_contract_sha256=canonical_json_sha256(boundary), native_trt_meta=meta,
        native_trt_meta_sha256=canonical_json_sha256(meta), artifacts={key: _artifact(path) for key, path in paths.items()})
    binding = seal_native_split_quality_binding(payload)
    identity = {"identity_valid": True, "eval_run_id": RUN_ID, "model_id": MODEL, "task": "detection",
        "case_id": CASE, "source_run_id": backend, "setup_id": setup, "variant": "composed",
        "runtime_precision_identity": selection["precision"], "native_split_quality_binding_required": True,
        "native_split_quality_binding": copy.deepcopy(binding), "native_split_quality_binding_sha256": binding["binding_sha256"],
        "source_request_sha256": "4" * 64}
    return {**copy.deepcopy(identity), "status": "completed", "technical_status": "completed",
            "source_setup_id": setup, "request_identity": identity}


def _upstream_summary(tmp_path, vendors):
    # Copy only eligible Split bindings from immutable historical bytes.  This
    # deliberately does not relabel archived Full results as current evidence.
    before = HISTORICAL_SUMMARY.read_bytes()
    historical = json.loads(before)
    results = [copy.deepcopy(row) for row in historical["results"]
               if row.get("model_id") == MODEL and row.get("case_id") == CASE
               and row.get("source_run_id") == "deepx_to_trt"]
    if "hailo8" in vendors:
        binding, summary = _binding_and_summary(tmp_path / "synthetic_hailo8_upstream")
        binding["eval_run_id"] = RUN_ID
        binding = seal_native_split_quality_binding(binding)
        row = summary["results"][0]
        for target in (row, row["request_identity"]):
            target.update(eval_run_id=RUN_ID, native_split_quality_binding=copy.deepcopy(binding),
                          native_split_quality_binding_sha256=binding["binding_sha256"])
        results.append(row)
    if "hailo10h" in vendors:
        results.append(_hailo10_summary(tmp_path / "synthetic_hailo10_upstream"))
    for vendor in vendors:
        producer = _strict_producer(MODEL)
        producer.update(eval_run_id=RUN_ID, setup_id=SETUPS[vendor])
        results.append(_result(_seal_producer(producer)))
    assert HISTORICAL_SUMMARY.read_bytes() == before
    return {"schema": "onnx-splitpoint/central-quality-summary", "schema_version": 1,
            "merge": {"summary_only_native_full_quality_conflict_count": 0}, "results": results}


def _stage(tmp_path, monkeypatch, *, vendors=("deepx",), transport_error="authentication failed", prior_blocker=False, recover_after_full=False, runner_observer=None):
    runner = workflow.EvaluationWorkflowRunner(workflow.WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = RUN_ID
    runner.run_dir = tmp_path / "current_run"
    runner.run_dir.mkdir(parents=True)
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "reports/remote_lease_journal",
    )
    runner.manifest = {"models": {MODEL: {}}}
    runner.profile_start_snapshot = {}
    cfg = {
        "enabled": True, "models": [MODEL], "backends": list(vendors),
        "precision": "uint8_dequant_fp16" if len(vendors) == 1 and vendors[0].startswith("hailo") else "float32_layout_fp16",
        "case_policy": "case_map_only", "case_map": {MODEL: [CASE]},
        "frames": 3, "warmup": 0, "repetitions": 3,
        "copy_benchmarksets": False, "build_missing_engines": False,
        "cleanup_remote_native_root": False, "disable_known_contract_overrides": True,
        "validation": {"enabled": False}, "energy": {"enabled": False},
        "full_baselines": {"enabled": True, "backends_by_producer": {vendor: [vendor, "tensorrt"] for vendor in vendors}},
        "remotes": {vendor: {"ssh": "synthetic-offline@" + vendor, "setup_id": SETUPS[vendor]} for vendor in vendors},
    }
    runner.profile_payload = {
        "campaign": {"mode": "development"}, "execution_preset": {"id": "smoke"},
        "native_producers": cfg,
    }
    if recover_after_full:
        recovered_backends = ("deepx",) if recover_after_full == "preparation" else ("hailo8", "tensorrt")
        if recover_after_full == "preparation":
            cfg["full_baselines"]["backends_by_producer"] = {"deepx": ["deepx"]}
        runner.profile_payload["run_profiles"] = [
            {"id": backend, "type": "same_backend_reference", "full": backend,
             "stage1": backend, "stage2": backend}
            for backend in recovered_backends
        ]
    suite = runner.run_dir / "models" / MODEL / "benchmark_set"
    _write(suite / CASE / "split_manifest.json", {"part2_external_inputs": ["boundary_tensor"]})
    _write(suite / "benchmark_set.json", {"cases": [{"id": CASE}], "task": "detection"})
    _write(suite / "benchmark_plan.json", {"runs": [{"id": "split"}]})
    (suite / "benchmark_suite.py").write_text("# synthetic offline Stage fixture\n", encoding="utf-8")
    _write(runner.run_dir / "quality_management/central_quality_summary.json", _upstream_summary(tmp_path, vendors))
    if prior_blocker:
        _write(suite / "deepx/deepx_m1/part1/deepx_part1_artifact_status.json", {
            "cases": [{"case_id": CASE, "ok": False, "error": "synthetic_prior_deepx_compile_reject"}],
        })
    assert workflow.benchmark_set_postcondition_v60v(suite)["valid"] is True
    calls = []
    actual_streaming = workflow.run_streaming
    transferred_sources = []
    runtime_attempts = []
    remote_results = tmp_path / "synthetic_remote_results"

    def run_full_at_hardware_boundary():
        from scripts import native_full_baseline_eval_runner as full
        _write(remote_results / MODEL / "benchmark_set/benchmark_set.json", {"task": "detection"})
        def unavailable_accelerator(_benchmark, _model, backend, ns):
            runtime_attempts.append((backend, ns.full_repetition_index))
            raise OSError("synthetic post-start accelerator failure")
        out, err = io.StringIO(), io.StringIO()
        with monkeypatch.context() as physical, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            physical.setattr(full, "_row_for_backend", unavailable_accelerator)
            physical.setattr(sys, "argv", ["native_full_baseline_eval_runner.py", "--root", str(remote_results),
                "--models", MODEL, "--backends", ",".join(recovered_backends), "--setup-id", SETUPS[vendors[0]],
                "--comparison-backend", vendors[0], "--comparison-precision", cfg["precision"],
                "--repetitions", "3", "--frames", "3", "--warmup", "0", "--engine-build-python", sys.executable])
            rc = full.main()
        return SimpleNamespace(returncode=rc, stdout=out.getvalue(), stderr=err.getvalue())

    def transport(command, **kwargs):
        label = str(kwargs.get("label") or "")
        calls.append((list(command), label))
        if Path(command[0]).name in {"ssh", "rsync"}:
            if recover_after_full:
                if label.startswith("full:"):
                    return run_full_at_hardware_boundary()
                if label.startswith("collect:"):
                    return SimpleNamespace(returncode=23, stdout="", stderr="synthetic interrupted result transfer")
                if label.startswith("collect-failure-results:"):
                    shutil.copytree(remote_results, Path(command[-1]), dirs_exist_ok=True)
                elif Path(command[0]).name == "rsync" and Path(command[-2]).is_file():
                    transferred_sources.append(Path(command[-2]))
                if label.startswith(("sync-script-verify:", "sync-asset-verify:")):
                    source = transferred_sources[-1]
                    return SimpleNamespace(returncode=0, stdout=json.dumps({"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "missing_tokens": []}), stderr="")
                if label.startswith("verify-remote-module:"):
                    return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True}), stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=255, stdout="", stderr=transport_error)
        if kwargs.get("label") == "host-telemetry-summary":
            return SimpleNamespace(returncode=0, stdout="telemetry intentionally omitted", stderr="")
        # In particular final_report is a real fresh Python process.
        return actual_streaming(command, **kwargs)

    # A failed synthetic SSH launched no remote process.  Its lease cancellation
    # request is the second SSH boundary; replace the physical request only.
    import onnx_splitpoint_tool.remote.process_lease as lease
    actual_control = lease.subprocess.Popen
    def control(command, *args, **kwargs):
        if Path(command[0]).name in {"ssh", "rsync"}:
            return SimpleNamespace(returncode=0, communicate=lambda **_kw: (
                "__SPLITPOINT_REMOTE_LEASE_CLEANUP__=cancelled_before_lease\n", None))
        return actual_control(command, *args, **kwargs)
    monkeypatch.setattr(lease.subprocess, "Popen", control)
    monkeypatch.setattr(workflow, "run_streaming", transport)
    if callable(runner_observer):
        runner_observer(runner)
    paths, details, message, status = runner._stage_run_native_producers()
    stage = _read(paths["native_producer_stage_json"])
    reports = runner.run_dir / "reports"
    return SimpleNamespace(runner=runner, paths=paths, details=details, message=message, status=status,
                           stage=stage, reports=reports, calls=calls, runtime_attempts=runtime_attempts,
                           remote_results=remote_results)


def _assert_chain(case, monkeypatch, expected_count, *, reasons):
    stage = case.stage
    expected = stage["expected_native_rows"]
    summary = _read(case.reports / "native_producer_combined_summary.json")
    rows = summary["rows"]
    matrix = _read(case.reports / "native_expected_matrix.json")
    assert len(expected) == expected_count
    assert len(rows) == expected_count, [(row.get("backend"), row.get("case"), row.get("failure_reason")) for row in rows]
    assert {native_identity_key(row) for row in rows} == {native_identity_key(row) for row in expected}
    assert [matrix[key] for key in ("expected_row_count", "present_expected_row_count", "successful_expected_row_count", "failed_expected_row_count", "missing_expected_row_count")] == [expected_count, expected_count, 0, expected_count, 0]
    assert matrix["row_presence_complete"] is True
    assert matrix["execution_success_complete"] is False and matrix["matrix_complete"] is False
    assert stage["started_remote_count"] == stage["started_performance_count"] == 0
    assert any(label == "final_report" for _, label in case.calls)
    assert not any(label.startswith(("split:", "full:")) for _, label in case.calls)
    for row in rows:
        key = native_identity_key(row)
        reason = reasons.get(key, reasons.get("default"))
        assert row["ok"] is False and row.get("result_ok") is not True
        assert row["primary_failure_reason"] == reason
        assert row["fps_makespan"] is None and row["runtime_success"] is False
        assert row["repetition_count_requested"] == 3
        assert row["repetition_count_attempted"] == row["repetition_count_valid"] == 0
        assert row["repetition_records"] == []
        assert not row.get("full_command_contract") and not row.get("native_command_contract")
        assert row.get("returncode") is None and row.get("performance_claim_eligible") is False
        if row["backend"].startswith("native_full_"):
            assert row["case"] == "full" and row["execution_mode"] == "native_full_baseline"
            assert row["execution_precision"] == row["full_runtime_precision"] == ""
            assert row["runtime_precision_source"] == "unavailable"
            assert row["failure_stage"] == "native_transfer_or_dispatch"
            assert row["upstream_evidence_path"] == "reports/native_producer_stage.json"
    full_path = case.runner.run_dir / "native_producers/_blocked/analysis_tables/native_full_baseline_eval.json"
    full = _read(full_path)
    assert full["schema"] == "onnx-splitpoint/native-full-baseline-eval" and full["schema_version"] == 5
    assert len(full["rows"]) == expected_count * 2 // 3
    assert full_path in case.paths.values()
    report_roots = [stage["final_report_cmd"][i + 1] for i, token in enumerate(stage["final_report_cmd"]) if token == "--root"]
    assert report_roots.count(str(full_path.parent.parent)) == 1

    out = case.reports / "energy_exclusions"
    monkeypatch.setattr(sys, "argv", ["native_producer_energy_plan.py", "--summary", str(case.reports / "native_producer_combined_summary.json"),
        "--out-dir", str(out), "--remote-root", str(case.runner.run_dir), "--remote-tool-dir", str(ROOT),
        "--deepx-ssh", "synthetic-offline", "--duration-s", "1", "--allow-unpaired", "--measure-all-runtime-successful"])
    assert energy.main() == 0
    plan = _read(out / "native_producer_energy_plan.json")
    assert plan["rows"] == []
    assert len(plan["excluded_rows"]) == expected_count
    assert {native_identity_key(row) for row in plan["excluded_rows"]} == {native_identity_key(row) for row in expected}
    for row in plan["excluded_rows"]:
        assert row["source_failure_reason"] == reasons.get(native_identity_key(row), reasons.get("default"))
        assert row["comparison_backend"] and row["failure_stage"] and row["upstream_evidence_path"]
    evidence = derive_native_evidence_status(run_mode="standard", expected_matrix=matrix,
        validation_payload=None, validation_requested=False, energy_requested=True,
        energy_plan_payload=plan, energy_results_payload=None)
    assert evidence["energy"]["accounting"]["counts"] == {"measured": 0, "excluded": expected_count, "missing": 0, "unexpected": 0}
    assert evidence["runtime"]["present_count"] == expected_count
    assert evidence["runtime"]["failed_count"] == expected_count and evidence["scientific_ready"] is False
    projection = project_native_evidence_status(evidence)
    assert projection["scientific_ready"] is False
    _, concise = workflow._native_concise_summary_v60w(case.reports)
    assert len(concise) == expected_count
    for row in concise:
        assert row["primary_failure_reason"] == reasons.get(native_identity_key(row), reasons.get("default"))
        assert row["runtime_success"] is False and row.get("claim_ok") is not True
    return rows


@pytest.mark.parametrize("vendor", ["deepx", "hailo8", "hailo10h"])
def test_t32_f1_real_stage_auth_failure_to_energy_and_gui(tmp_path, monkeypatch, vendor):
    case = _stage(tmp_path, monkeypatch, vendors=(vendor,))
    assert case.stage["native_prerequisite_ready_count"] == 1, case.stage["native_split_quality_first"]
    assert len(case.stage["backend_results"]) == 1
    assert case.stage["global_infrastructure_stop"]["reason"] == "remote_authentication_failed"
    _assert_chain(case, monkeypatch, 3, reasons={"default": "remote_authentication_failed"})


def test_t32_f1_prior_compile_blocker_survives_later_full_dispatch_failure(tmp_path, monkeypatch):
    case = _stage(tmp_path, monkeypatch, prior_blocker=True)
    expected_split = next(row for row in case.stage["expected_native_rows"] if row["execution_mode"] == "native_split")
    assert case.stage["native_prerequisite_ready_count"] == 0
    _assert_chain(case, monkeypatch, 3, reasons={"default": "remote_authentication_failed",
        native_identity_key(expected_split): "synthetic_prior_deepx_compile_reject"})


def test_t32_f1_global_auth_stop_never_dispatches_remaining_setups(tmp_path, monkeypatch):
    case = _stage(tmp_path, monkeypatch, vendors=("deepx", "hailo8", "hailo10h"))
    assert len(case.stage["expected_native_rows"]) == 9
    assert [row["backend"] for row in case.stage["backend_results"]] == ["deepx"]
    stop = case.stage["global_infrastructure_stop"]
    assert stop["remaining_backends_not_started"] == ["hailo8", "hailo10h"]
    assert not any("synthetic-offline@hailo" in " ".join(command) for command, _ in case.calls)
    assert case.stage["started_performance_count"] == 0


def test_t32_f1_failed_collection_recovers_started_full_results_without_zero_start_blockers(tmp_path, monkeypatch):
    case = _stage(tmp_path, monkeypatch, vendors=("hailo8",), recover_after_full=True)
    assert case.runtime_attempts == [(backend, index) for backend in ("hailo8", "tensorrt") for index in (1, 2, 3)], case.stage
    assert case.stage["started_performance_count"] == case.stage["started_remote_count"] == 1
    assert case.stage["blocked_native_rows"] == []
    assert not (case.runner.run_dir / "native_producers/_blocked").exists()
    remote_json = case.remote_results / "analysis_tables/native_full_baseline_eval.json"
    recovered_json = case.runner.run_dir / "native_producers/hailo8/analysis_tables/native_full_baseline_eval.json"
    assert recovered_json.read_bytes() == remote_json.read_bytes()
    summary = _read(case.reports / "native_producer_combined_summary.json")
    assert len(summary["rows"]) == 2
    for row in summary["rows"]:
        assert row["primary_failure_reason"] == "native_full_exception"
        assert row["repetition_count_requested"] == row["repetition_count_attempted"] == 3
        assert row["repetition_count_valid"] == 0 and row["fps_makespan"] is None
    matrix = _read(case.reports / "native_expected_matrix.json")
    assert matrix["expected_row_count"] == matrix["present_expected_row_count"] == matrix["failed_expected_row_count"] == 2
    assert matrix["missing_expected_row_count"] == 0


def test_t32_f1_recovered_preparation_blocker_keeps_original_three_requested(tmp_path, monkeypatch):
    case = _stage(tmp_path, monkeypatch, vendors=("deepx",), recover_after_full="preparation")
    assert case.runtime_attempts == []
    assert case.stage["started_performance_count"] == 0
    remote_json = case.remote_results / "analysis_tables/native_full_baseline_eval.json"
    original = _read(remote_json)["rows"][0]
    assert original["semantic_dump_failure_reason"] == "semantic_runner_missing"
    assert original["repetition_count_requested"] == 3 and original["repetition_count_attempted"] == 0
    recovered_json = case.runner.run_dir / "native_producers/deepx/analysis_tables/native_full_baseline_eval.json"
    assert recovered_json.read_bytes() == remote_json.read_bytes()
    summary = _read(case.reports / "native_producer_combined_summary.json")
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["repetition_count_requested"] == 3, row
    assert row["repetition_count_attempted"] == row["repetition_count_valid"] == 0
    assert row["semantic_dump_failure_reason"] == "semantic_runner_missing"
    assert row["primary_failure_reason"] == "semantic_runner_missing"
    assert row["preparation_count_attempted"] == 1
    _, concise = workflow._native_concise_summary_v60w(case.reports)
    assert len(concise) == 1 and concise[0]["primary_failure_reason"] == "semantic_runner_missing"
    assert case.stage["blocked_native_rows"] == []
    matrix = _read(case.reports / "native_expected_matrix.json")
    assert matrix["expected_row_count"] == matrix["present_expected_row_count"] == matrix["failed_expected_row_count"] == 1
    assert matrix["missing_expected_row_count"] == 0
