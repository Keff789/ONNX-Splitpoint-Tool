"""Independent scheduler/immutable-reader checks without reference inference."""
from concurrent.futures import Future
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.generic_quality_sidecars import bind_existing_generic_cpu_reference
from tests.test_generic_existing_cpu_reference import _write, sample


def project(sample):
    runner, suite, plan, policy, _, _ = sample
    bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
        model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)
    _write(suite / "benchmark_plan.json", plan)
    return runner, suite, plan, policy


def test_reuse_requires_exact_prepared_plan_before_scheduler_admission(sample):
    runner, suite, plan, _ = project(sample)
    plan["management_cpu_reference"]["reference_path"] += ".other"
    _write(suite / "benchmark_plan.json", plan)
    with pytest.raises(ValueError, match="plan_not_projected"):
        runner._schedule_management_cpu_reference("yolo11l", suite)
    assert runner._management_reference_executor is None
    assert runner._management_reference_futures == {}


@pytest.mark.parametrize("malformed", [None, {}, ""])
def test_explicit_malformed_reference_never_falls_through_to_ordinary_generator(sample, malformed):
    runner, suite, plan, policy, _, _ = sample
    policy["management_reference"]["existing_reference"] = malformed
    plan["runs"] = [{"id": "ordinary", "variants": ["full"]}]
    _write(suite / "benchmark_plan.json", plan)
    with pytest.raises(ValueError, match="explicit_binding_required"):
        runner._schedule_management_cpu_reference("yolo11l", suite)
    assert runner._management_reference_executor is None
    assert runner._management_reference_futures == {}


@pytest.mark.parametrize("binding", ["source_status", "policy_source"])
def test_hash_bound_original_authority_bytes_must_still_exist_and_match(sample, binding):
    runner, suite, plan, policy, _, _ = sample
    descriptor = policy["management_reference"]["existing_reference"][binding]
    Path(descriptor["path"]).write_text("{}")
    with pytest.raises(ValueError):
        bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
            model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)
    assert not list(runner.run_dir.glob("quality_management/**/canonical_cpu_reference.json"))


def test_changed_active_input_dimensions_rejected_despite_rebound_config_hash(sample):
    runner, suite, plan, policy, _, _ = sample
    path = suite / "quality_runtime/b039.json"
    config = json.loads(path.read_text())
    config["completion_contract"]["processor_contract"]["input_hw"] = [320, 320]
    descriptor = _write(path, config)
    plan["runs"][0]["generic_quality_completion_config_sha256"]["b039"] = descriptor["sha256"]
    with pytest.raises(ValueError, match="preprocessing_changed"):
        bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
            model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)


def test_mixed_ordinary_scope_cannot_borrow_explicit_generic_reference(sample):
    runner, suite, plan, policy, _, _ = sample
    plan["runs"].append({"id": "ordinary_native", "variants": ["composed"]})
    with pytest.raises(ValueError, match="source_only_scope"):
        bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
            model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)


def test_original_status_contract_version_must_remain_normal_schema(sample):
    runner, suite, plan, policy, _, _ = sample
    descriptor = policy["management_reference"]["existing_reference"]
    path = Path(descriptor["source_status"]["path"])
    status = json.loads(path.read_text())
    status["schema_version"] = 2
    descriptor["source_status"] = _write(path, status)
    with pytest.raises(ValueError, match="source_status_invalid"):
        bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
            model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)


@pytest.mark.parametrize("damage", ["bytes", "hardlink"])
def test_normal_collector_rechecks_published_reuse_artifact(sample, damage):
    runner, suite, _, _ = project(sample)
    runner._schedule_management_cpu_reference("yolo11l", suite)
    status = runner._management_reference_futures["yolo11l"].result()
    target = Path(status["reference_path"])
    if damage == "bytes":
        target.write_text("{}")
    else:
        target.with_suffix(".other").hardlink_to(target)
    with pytest.raises(RuntimeError):
        runner._management_reference_records("yolo11l")
    assert runner._management_reference_executor is None


def test_ordinary_profile_retains_existing_scheduler_path_without_running_it(sample, monkeypatch):
    runner, suite, _, policy, _, _ = sample
    policy.pop("management_reference")
    _write(suite / "benchmark_plan.json", {"runs": []})
    _write(suite / "benchmark_set.json", {})
    submitted = []
    class RecordingExecutor:
        def __init__(self, **options):
            assert options["max_workers"] == 1
        def submit(self, function, **options):
            submitted.append((function, options))
            return Future()  # Inspect the unchanged submission; do not execute it.
    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", RecordingExecutor)
    runner._schedule_management_cpu_reference("yolo11l", suite)
    assert len(submitted) == 1
    assert submitted[0][1]["reference_function"].__name__ == "forbidden"
    assert "yolo11l" in runner._management_reference_futures
    assert runner._management_reference_source_contracts["yolo11l"]
