"""Exercise normal private Full staging and the real generated suite selector."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import sys
import types
import zlib

import pytest

from onnx_splitpoint_tool.native_output_endpoint import load_authoritative_output_contract
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.workflow.hardware_acceptance_numerical import execution_lifecycle_safety
from scripts import native_full_baseline_eval_runner as full_runner
from test_v2952_full_staging_metadata import LocalTransport, put, selective_full


def exercise_suite_selector(root, model, frames, warmup, numerical_only, monkeypatch):
    """Keep plan parsing, selection, identity verification and result writing real."""
    script = root / "benchmark_suite.py"
    module = types.ModuleType("selected_private_full_suite")
    module.__file__ = str(script)
    exec(compile(script.read_text(), str(script), "exec"), module.__dict__)
    selected = []

    def native_runtime(root_arg, run, args, *, expected_endpoint_identity, **kwargs):
        assert root_arg == root
        assert expected_endpoint_identity == {"model_id": model, "backend": "deepx_m1", "variant": "full"}
        assert run["id"] == "deepx_m1_full"
        declared = next(row for row in json.loads((root / "benchmark_set.json").read_text())["plan"]["runs"]
                        if row["id"] == "deepx_m1_full")
        assert {key: run[key] for key in declared} == declared
        assert args.deepx_runtime_mode == "A4"
        assert args.throughput_frames == frames and args.throughput_warmup_frames == warmup
        assert args.warmup == warmup and args.deepx_numerical_only is numerical_only
        selected.append(run)
        return [{"run_id": run["id"], "provider": "deepx_m1", "backend": "deepx_m1",
                 "case_id": "full", "variant": "full", "runtime_ok": True}]

    def wrong_route(*args, **kwargs):
        pytest.fail("Full request entered generic provider/per-case execution")

    monkeypatch.setattr(module, "_run_deepx_full_run", native_runtime)
    monkeypatch.setattr(module, "_provider_unavailable_reason", wrong_route)
    argv = [str(script), "--run-id", "deepx_m1_full", "--warmup", str(warmup), "--runs", "3",
            "--throughput-frames", str(frames), "--throughput-warmup-frames", str(warmup),
            "--energy-measurement-only", "--deepx-runtime-mode", "A4"]
    if numerical_only:
        argv += ["--deepx-numerical-images", str(root / "resources/numerical/images.json"), "--deepx-numerical-only"]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    assert len(selected) == 1
    results = list(root.glob("benchmark_results_*.json"))
    assert [path.name for path in results] == ["benchmark_results_deepx_m1_full_auto.json"]
    assert json.loads(results[0].read_text())[0]["provider"] == "deepx_m1"
    return selected[0]


@pytest.mark.parametrize("frames,warmup,numerical_only", [(64, 0, True), (1000, 100, False)])
def test_actual_private_staging_selects_only_deepx_full_before_runtime(selective_full, monkeypatch, frames, warmup, numerical_only):
    point = selective_full
    benchmark = json.loads((point.local / "benchmark_set.json").read_text())
    benchmark["artifact_manifest"] = {"schema": "onnx-splitpoint/benchmark-set", "schema_version": 2,
        "files": {"models": [benchmark["model"]]}, "counts": {"models": 1}}
    put(point.local / "benchmark_set.json", benchmark)
    put(point.remote / "benchmark_set.json", (point.local / "benchmark_set.json").read_bytes())
    point.contract["artifacts"]["benchmark_set_manifest"]["sha256"] = native._sha(point.local / "benchmark_set.json")
    point.item["full_source_case_metadata"] = native._full_case_metadata(point.primary, point.contract, deepx_runtime_mode="A4")
    before = {p: p.read_bytes() for root in (point.local, point.remote) for p in root.rglob("*") if p.is_file()}
    native._stage_full_benchmark_set(LocalTransport(point.tmp), point.item, point.tmp / "receipts")
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert json.loads((point.fresh / "benchmark_plan.json").read_text()) == benchmark["plan"]
    assert (point.fresh / "output_contracts.json").read_bytes() == (point.local / "output_contracts.json").read_bytes()
    authority = load_authoritative_output_contract(point.fresh, backend="deepx_m1", model_id="mobilenet_v3_large",
                                                   variant="full", task="classification")
    assert authority["contract_resolution_status"] == "attested"
    assert authority["declaration_source"] == str(point.fresh / "output_contracts.json")
    exercise_suite_selector(point.fresh, "mobilenet_v3_large", frames, warmup, numerical_only, monkeypatch)


@pytest.mark.parametrize("change", ["missing", "duplicate", "provider", "type", "variant", "case", "artifact", "contract"])
def test_invalid_full_plan_is_rejected_by_normal_prepare(selective_full, change):
    point = selective_full
    benchmark = json.loads((point.local / "benchmark_set.json").read_text())
    run = benchmark["plan"]["runs"][0]
    if change == "missing": benchmark["plan"]["runs"] = []
    elif change == "duplicate": benchmark["plan"]["runs"].append(copy.deepcopy(run))
    elif change == "provider": run["provider"] = "cpu"
    elif change == "type": run["type"] = "onnxruntime"
    elif change == "variant": run["variants"] = ["composed"]
    elif change == "case": run["case_ids"] = ["b099"]
    elif change == "artifact": run["dxnn_path"] = "deepx/deepx_m1/b001/model.dxnn"
    else: run["contract_path"] = "other/output_contract.json"
    put(point.local / "benchmark_set.json", benchmark)
    point.contract["artifacts"]["benchmark_set_manifest"]["sha256"] = native._sha(point.local / "benchmark_set.json")
    with pytest.raises(ValueError, match="plan"):
        native._full_case_metadata(point.primary, point.contract, deepx_runtime_mode="A4")
    assert not point.fresh.exists()


@pytest.mark.parametrize("change", ["missing", "container_drift", "row_hash", "remote_plan", "transported_bytes"])
def test_exact_output_authority_and_plan_cannot_fall_back(selective_full, change):
    point = selective_full
    if change in {"missing", "container_drift", "row_hash"}:
        if change == "missing": (point.local / "output_contracts.json").unlink()
        elif change == "container_drift": (point.local / "output_contracts.json").write_bytes(b"{}")
        else: point.contract["energy_workload"]["input_contract"]["endpoint_semantic_attestation"]["authoritative_contract_sha256"] = "f" * 64
        with pytest.raises((ValueError, FileNotFoundError)):
            native._full_case_metadata(point.primary, point.contract, deepx_runtime_mode="A4")
    else:
        metadata = copy.deepcopy(point.metadata)
        metadata["case_manifest_zlib_base64"] = base64.b64encode(zlib.compress((point.local / point.case / "split_manifest.json").read_bytes())).decode()
        raw = (point.local / "output_contracts.json").read_bytes()
        if change == "transported_bytes": raw += b"\n"
        metadata["output_contracts_zlib_base64"] = base64.b64encode(zlib.compress(raw)).decode()
        if change == "remote_plan": put(point.remote / "benchmark_plan.json", {"runs": []})
        with pytest.raises(ValueError, match="plan|output container"):
            native._copy_full_benchmark(point.remote, point.fresh, point.contract, None, metadata, "A4")
    assert not point.fresh.exists()


@pytest.mark.parametrize("change", ["missing_plan", "missing_run", "duplicate_run", "wrong_provider", "wrong_artifact", "wrong_case", "missing_authority"])
def test_direct_a4_full_cli_stops_before_semantic_or_measured_runtime(selective_full, monkeypatch, change):
    from types import SimpleNamespace
    point = selective_full
    native._stage_full_benchmark_set(LocalTransport(point.tmp), point.item, point.tmp / "receipts")
    plan_path = point.fresh / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    if change == "missing_plan": plan_path.unlink()
    elif change == "missing_run": plan["runs"] = []
    elif change == "duplicate_run": plan["runs"].append(copy.deepcopy(plan["runs"][0]))
    elif change == "wrong_provider": plan["runs"][0]["provider"] = "cpu"
    elif change == "wrong_artifact": plan["runs"][0]["dxnn_path"] = "split/model.dxnn"
    elif change == "wrong_case": plan["runs"][0]["case_ids"] = ["b099"]
    else: (point.fresh / "output_contracts.json").unlink()
    if change not in {"missing_plan", "missing_authority"}: put(plan_path, plan)
    observed = []
    monkeypatch.setattr(full_runner, "_run", lambda *args, **kwargs: observed.append(args))
    ns = SimpleNamespace(deepx_runtime_mode="A4", dump_outputs=True, repetitions=1)
    semantic, blocked = full_runner._deepx_full_series_preflight(point.fresh, "mobilenet_v3_large", ns)
    assert blocked["repetition_count_attempted"] == 0
    assert semantic["failure_reason"] == "deepx_a4_full_suite_route_invalid"
    assert execution_lifecycle_safety(blocked)["pre_runtime_proven"] is True
    measured = full_runner._generic_full_via_suite(point.fresh, "mobilenet_v3_large", "native_full_deepx",
                                                 "deepx_m1_full", ns)
    assert measured["failure_reason"] == "deepx_a4_full_suite_route_invalid"
    assert observed == []


def test_direct_a4_full_worker_uses_explicit_validated_plan(selective_full, monkeypatch):
    from types import SimpleNamespace
    point = selective_full
    native._stage_full_benchmark_set(LocalTransport(point.tmp), point.item, point.tmp / "receipts")
    assert full_runner._deepx_a4_full_route_guard(point.fresh, "mobilenet_v3_large", "deepx_m1_full") is None
    # The manifest rejection remains before child execution after route proof.
    ns = SimpleNamespace(deepx_runtime_mode="A4", frames=64)
    row = full_runner._generic_full_via_suite(point.fresh, "mobilenet_v3_large", "native_full_deepx", "deepx_m1_full", ns)
    assert row["failure_reason"] == "deepx_prepared_input_manifest_missing"
    # Assert against the command built by the real successful process-boundary
    # fixture, including the exact normal prepared-input contract.
    from test_v27930_native_full_semantic_merge import prepare_runner_case
    case = prepare_runner_case(point.tmp / "worker", monkeypatch)
    case.ns.deepx_runtime_mode = "A4"
    commands = []
    original = full_runner._run
    def observe(command, **kwargs):
        commands.append(command)
        return original(command, **kwargs)
    monkeypatch.setattr(full_runner, "_run", observe)
    semantic, blocked = full_runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None
    full_runner._generic_full_via_suite(case.root, case.model, "native_full_deepx", "deepx_m1_full", case.ns,
                                       prepared_input_manifest=Path(semantic["input_manifest"]))
    command = next(command for command in commands if "run_benchmark_suite_from_set.py" in command[1])
    assert command[command.index("--plan") + 1] == str(case.root / "benchmark_plan.json")


def test_fresh_s1_metadata_and_staging_keep_existing_requirements(selective_full):
    point = selective_full
    benchmark = json.loads((point.local / "benchmark_set.json").read_text())
    benchmark.pop("plan")
    put(point.local / "benchmark_set.json", benchmark)
    put(point.remote / "benchmark_set.json", (point.local / "benchmark_set.json").read_bytes())
    point.contract["artifacts"]["benchmark_set_manifest"]["sha256"] = native._sha(point.local / "benchmark_set.json")
    (point.local / "output_contracts.json").unlink()
    point.contract["energy_workload"]["input_contract"].pop("endpoint_semantic_attestation")
    default = native._full_case_metadata(point.primary, point.contract)
    explicit = native._full_case_metadata(point.primary, point.contract, deepx_runtime_mode="S1")
    assert default == explicit and "output_contracts_manifest" not in explicit
    point.item["full_source_case_metadata"] = explicit
    point.item["argv"] = [sys.executable, "--deepx-runtime-mode", "S1"]
    native._stage_full_benchmark_set(LocalTransport(point.tmp), point.item, point.tmp / "receipts")
    assert not (point.fresh / "benchmark_plan.json").exists()
    assert not (point.fresh / "output_contracts.json").exists()


@pytest.mark.parametrize("backend,mode", [("native_full_deepx", "S1"), ("native_full_tensorrt", "A4")])
def test_full_route_guard_does_not_change_other_runtime_paths(tmp_path, monkeypatch, backend, mode):
    from types import SimpleNamespace
    monkeypatch.setattr(full_runner, "_deepx_a4_full_route_guard", lambda *args: pytest.fail("A4-only guard changed another runtime path"))
    row = full_runner._semantic_full_dump(tmp_path, "mobilenet_v3_large", backend, "deepx_m1_full",
        SimpleNamespace(deepx_runtime_mode=mode, dump_outputs=True), force=True)
    assert row["failure_reason"] == "semantic_runner_missing"
