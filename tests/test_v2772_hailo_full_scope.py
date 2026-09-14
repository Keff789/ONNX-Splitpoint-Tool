from __future__ import annotations

import json
import inspect
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    materialize_backend_artifact_decisions,
)
from onnx_splitpoint_tool.workflow.hailo_remote_binding import (
    materialize_hailo_artifact_service_binding,
)
from onnx_splitpoint_tool.workflow.generator_binding import (
    _prepare_hailo_full_reuse,
)
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _hailo_full_hef_policy_v2772,
    materialize_legacy_benchmark_set,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.hailo_full_contract_promotion import (
    promote_verified_hailo_full_contracts,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request_accounting_fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    run_dir = tmp_path / "run"
    model_dir = run_dir / "models" / "resnet50"
    suite_dir = model_dir / "benchmark_set" / "legacy_suite"
    source_dir = tmp_path / "prepared_full"
    baselines = []
    contracts = []
    for backend in ("hailo8", "hailo10"):
        artifact = source_dir / backend / "compiled.hef"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"unattested-{backend}".encode("ascii"))
        baselines.append({
            "model_id": "resnet50",
            "backend": backend,
            "variant": "full",
            "artifact_path": str(artifact),
            "task": "classification",
            "endpoint_mode": "decoded",
        })
        contracts.append({
            "model_id": "resnet50",
            "backend": backend,
            "variant": "full",
            "task": "classification",
            "endpoint_mode": "decoded",
        })
        part1 = suite_dir / "b052" / "hailo" / backend / "part1" / "compiled.hef"
        part1.parent.mkdir(parents=True, exist_ok=True)
        part1.write_bytes(f"part1-{backend}".encode("ascii"))

    suite_payload = {
        "cases": [{
            "case_id": "b052",
            "split_index": 52,
            "hailo_case_variant_availability": {
                "hailo8": {"full": False, "part1": True, "part2": False},
                "hailo10": {"full": False, "part1": True, "part2": False},
            },
        }],
    }
    full_plan = {
        "model_id": "resnet50",
        "task": "classification",
        "baselines": baselines,
    }
    output_contracts = {
        "model_id": "resnet50",
        "task": "classification",
        "contracts": contracts,
    }
    benchmark_contract = {
        "materialized": True,
        "legacy_suite_dir": str(suite_dir),
    }
    _write_json(suite_dir / "benchmark_set.json", suite_payload)
    _write_json(
        model_dir / "full_baselines" / "full_baseline_plan.json", full_plan,
    )
    _write_json(
        model_dir / "full_baselines" / "output_contracts.json",
        output_contracts,
    )
    return run_dir, full_plan, output_contracts, benchmark_contract


def _binding_options(*, build_full: bool) -> Namespace:
    return Namespace(
        execution_mode="generate_benchmarksets",
        no_remote=True,
        hailo_build_mode="reuse_and_build_missing",
        hailo_build_full=build_full,
        hailo_build_part1=True,
        hailo_build_part2=False,
        hailo_build_targets=["hailo8", "hailo10"],
        hailo_hw_arch="hailo8",
    )


def _resnet_hailo_to_trt_plan(*, full_hef_policy: str):
    return BenchmarkGenerationService().build_run_plan(
        acc_cpu=False,
        acc_cuda=False,
        acc_trt=True,
        acc_h8=True,
        acc_h10=True,
        acc_deepx=False,
        hailo8_hw="hailo8",
        hailo10_hw="hailo10",
        image_scale="auto",
        validation_images=None,
        validation_max_images=0,
        validation_reference_mode="auto",
        mini_coco_ap50=False,
        benchmark_task="classification",
        mini_classification_eval=False,
        hailo_preset="Custom",
        # Deliberately request Full here: an explicit ``skip`` policy must be
        # authoritative even when an upstream caller carries a stale request.
        hailo_custom_full=True,
        hailo_custom_composed=False,
        hailo_custom_part1=True,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=True,
        matrix_deepx_to_trt=False,
        matrix_trt_to_deepx=False,
        full_hef_policy=full_hef_policy,
    )


def test_explicit_build_full_false_projects_skip_policy() -> None:
    assert _hailo_full_hef_policy_v2772(False) == "skip"
    assert _hailo_full_hef_policy_v2772(True) == "end"


def test_skip_policy_keeps_hailo_to_trt_part1_only(monkeypatch) -> None:
    monkeypatch.setenv("ONNX_SPLITPOINT_ENABLE_HAILO_SAME_BACKEND_SPLIT", "1")
    plan = _resnet_hailo_to_trt_plan(full_hef_policy="skip")

    assert plan.hef_full is False
    assert plan.hef_part1 is True
    assert plan.hef_part2 is False
    assert plan.matrix_variants == ["part1", "part2", "composed"]
    hailo_runs = [
        row
        for row in plan.bench_plan_runs
        if row.get("type") == "hailo"
        or any(
            isinstance(row.get(stage), dict)
            and row[stage].get("type") == "hailo"
            for stage in ("stage1", "stage2")
        )
    ]
    assert [row["id"] for row in hailo_runs] == [
        "hailo8",
        "hailo10",
        "hailo8_to_trt",
        "hailo10_to_tensorrt",
    ]
    same_backend = [row for row in hailo_runs if row.get("type") == "hailo"]
    assert [row["variants"] for row in same_backend] == [["part1"], ["part1"]]
    assert all("full" not in row.get("variants", []) for row in hailo_runs)


def test_end_policy_preserves_existing_full_reference_plan(monkeypatch) -> None:
    monkeypatch.setenv("ONNX_SPLITPOINT_ENABLE_HAILO_SAME_BACKEND_SPLIT", "1")
    plan = _resnet_hailo_to_trt_plan(full_hef_policy="end")

    assert plan.hef_full is True
    assert plan.hef_part1 is True
    assert plan.hef_part2 is False
    assert "full" in plan.matrix_variants
    hailo_runs = [
        row
        for row in plan.bench_plan_runs
        if row.get("type") == "hailo"
        or any(
            isinstance(row.get(stage), dict)
            and row[stage].get("type") == "hailo"
            for stage in ("stage1", "stage2")
        )
    ]
    assert {row["id"] for row in hailo_runs} == {
        "hailo8",
        "hailo10",
        "hailo8_to_trt",
        "hailo10_to_tensorrt",
    }
    assert any("full" in row.get("variants", []) for row in hailo_runs)


def test_binding_forwards_one_effective_full_policy_to_all_three_layers() -> None:
    source = inspect.getsource(materialize_legacy_benchmark_set)

    assert source.count("full_hef_policy=full_hef_policy") == 1
    assert source.count("hef_full_policy=full_hef_policy") == 1
    assert (
        source.count(
            "full_hef_policy=normalize_full_hef_policy(full_hef_policy)"
        )
        == 1
    )
    assert 'full_hef_policy="end"' not in source
    assert 'hef_full_policy="end"' not in source


def test_build_full_false_is_not_reintroduced_as_pending_service_work(
    tmp_path: Path,
) -> None:
    run_dir, full_plan, output_contracts, benchmark_contract = (
        _request_accounting_fixture(tmp_path)
    )
    accidental_full = (
        Path(benchmark_contract["legacy_suite_dir"])
        / "hailo" / "hailo8" / "full" / "compiled.hef"
    )
    accidental_full.parent.mkdir(parents=True, exist_ok=True)
    accidental_full.write_bytes(b"old-unrequested-full")
    for row in full_plan["baselines"]:
        row.update({
            "requested": False,
            "request_status": "not_requested_by_profile",
        })
    for row in output_contracts["contracts"]:
        row.update({
            "requested": False,
            "request_status": "not_requested_by_profile",
        })
    model_dir = run_dir / "models" / "resnet50"
    _write_json(
        model_dir / "full_baselines" / "full_baseline_plan.json", full_plan,
    )
    _write_json(
        model_dir / "full_baselines" / "output_contracts.json",
        output_contracts,
    )
    decisions = materialize_backend_artifact_decisions(
        run_dir=run_dir,
        model_id="resnet50",
        targets=["hailo8", "hailo10"],
        full_baseline_plan=full_plan,
        output_contracts=output_contracts,
        benchmark_set_contract=benchmark_contract,
        # The explicit plan marker is authoritative even if a compatibility
        # caller still carries the old default.
        hailo_full_requested=True,
    )

    assert decisions["status"] == "ok"
    assert decisions["metrics"]["copied_artifacts"] == 0
    assert decisions["metrics"]["hailo_full_not_requested"] == 2
    decision_payload = json.loads(
        decisions["artifacts"]["backend_artifact_decisions_json"].read_text(
            encoding="utf-8",
        )
    )
    assert decision_payload["hailo_full_requested"] is False
    assert {
        (row["backend"], row["requested"], row["decision"])
        for row in decision_payload["baseline_decisions"]
    } == {
        ("hailo8", False, "not_requested_by_profile"),
        ("hailo10", False, "not_requested_by_profile"),
    }

    result = materialize_hailo_artifact_service_binding(
        run_dir=run_dir,
        model_id="resnet50",
        options=_binding_options(build_full=False),
        targets=["hailo8", "hailo10"],
        benchmark_set_contract=benchmark_contract,
        benchmark_plan={},
    )

    assert result["status"] == "ok"
    assert result["metrics"]["full_ready"] == 0
    assert result["metrics"]["detected_hefs"] == 3
    assert result["metrics"]["full_pending"] == 0
    assert result["metrics"]["full_not_requested"] == 2
    assert result["metrics"]["case_hefs_ready"] == 2
    assert result["metrics"]["case_hefs_pending"] == 0
    assert result["metrics"]["build_queue_count"] == 0
    assert result["metrics"]["actionable_build_queue_count"] == 0
    assert result["metrics"]["dispatch_status"] == "not_required"
    assert result["metrics"]["build_service_status"] == "not_required"

    plan = json.loads(
        result["artifacts"]["hailo_artifact_service_plan_json"].read_text(
            encoding="utf-8",
        )
    )
    assert plan["requested_full_baseline_count"] == 0
    assert plan["not_requested_full_baseline_count"] == 2
    assert all(
        row["requested"] is False
        and row["status"] == "not_requested_by_profile"
        for row in plan["full_baseline_requests"]
    )
    queue = json.loads(
        result["artifacts"]["hailo_build_queue_json"].read_text(
            encoding="utf-8",
        )
    )
    assert queue["queue"] == []


def test_build_full_true_remains_fail_closed_when_full_hefs_are_missing(
    tmp_path: Path,
) -> None:
    run_dir, full_plan, output_contracts, benchmark_contract = (
        _request_accounting_fixture(tmp_path)
    )
    materialize_backend_artifact_decisions(
        run_dir=run_dir,
        model_id="resnet50",
        targets=["hailo8", "hailo10"],
        full_baseline_plan=full_plan,
        output_contracts=output_contracts,
        benchmark_set_contract=benchmark_contract,
        hailo_full_requested=True,
    )

    result = materialize_hailo_artifact_service_binding(
        run_dir=run_dir,
        model_id="resnet50",
        options=_binding_options(build_full=True),
        targets=["hailo8", "hailo10"],
        benchmark_set_contract=benchmark_contract,
        benchmark_plan={},
    )

    assert result["status"] == "partial"
    assert result["metrics"]["full_not_requested"] == 0
    assert result["metrics"]["full_pending"] == 2
    assert result["metrics"]["case_hefs_ready"] == 2
    assert result["metrics"]["case_hefs_pending"] == 0
    assert result["metrics"]["build_queue_count"] == 2
    assert result["metrics"]["actionable_build_queue_count"] == 2


def test_prepare_full_baselines_marks_only_hailo_full_as_not_requested(
    tmp_path: Path,
) -> None:
    prepared_hef = tmp_path / "prepared" / "compiled.hef"
    prepared_hef.parent.mkdir(parents=True, exist_ok=True)
    prepared_hef.write_bytes(b"prepared-full")

    class _FakeRunner:
        run_dir = tmp_path / "run"
        options = SimpleNamespace(hailo_build_full=False)
        manifest: dict = {}

        @staticmethod
        def _targets():
            return ["hailo8", "hailo10", "tensorrt"]

        @staticmethod
        def _task_for(_row):
            return "classification"

        @staticmethod
        def _baseline_backends(_targets):
            return ["hailo8", "hailo10", "tensorrt"]

        @staticmethod
        def _prepared_full_hailo_info(_model_path):
            return {
                "ok": True,
                "artifact_path": str(prepared_hef),
                "source": "test_prepared_full",
            }

        @staticmethod
        def _raw_head_end_nodes(_row):
            return []

    artifacts, metrics, _message, status = (
        EvaluationWorkflowRunner._stage_prepare_full_baselines(
            _FakeRunner(), "resnet50", {"family": "resnet50"},
        )
    )

    assert status == "ok"
    assert metrics["reused_baselines"] == 0
    plan = json.loads(
        artifacts["full_baseline_plan_json"].read_text(encoding="utf-8")
    )
    hailo = [
        row for row in plan["baselines"]
        if row["backend"].startswith("hailo")
    ]
    trt = next(
        row for row in plan["baselines"]
        if row["backend"] == "tensorrt"
    )
    assert len(hailo) == 2
    assert all(row["requested"] is False for row in hailo)
    assert all(
        row["request_status"] == "not_requested_by_profile"
        for row in hailo
    )
    assert all(
        row["reuse_status"] == "not_requested_by_profile"
        and row["status"] == "not_requested_by_profile"
        for row in hailo
    )
    assert hailo[0]["artifact_path"] == str(prepared_hef)
    assert "requested" not in trt
    assert "request_status" not in trt


def test_debug_reuse_path_does_not_copy_unrequested_full_hef(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared" / "compiled.hef"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"prepared-full")
    suite = tmp_path / "suite"
    decisions, artifacts = _prepare_hailo_full_reuse(
        suite_dir=suite,
        model_id="resnet50",
        full_baseline_plan={
            "baselines": [{
                "backend": "hailo8",
                "variant": "full",
                "artifact_path": str(source),
                "requested": False,
            }],
        },
        output_contracts={
            "contracts": [{
                "model_id": "resnet50",
                "backend": "hailo8",
                "variant": "full",
                "requested": False,
            }],
        },
        targets=["hailo8"],
    )

    assert artifacts == []
    assert decisions[0]["requested"] is False
    assert decisions[0]["decision"] == "not_requested_by_profile"
    assert not (suite / "hailo" / "hailo8" / "full" / "compiled.hef").exists()


def test_contract_promotion_skips_explicitly_unrequested_full(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _unexpected_evidence(*_args, **_kwargs):
        raise AssertionError("unrequested contract reached artifact promotion")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.hailo_full_contract_promotion."
        "_hailo_full_artifact_evidence",
        _unexpected_evidence,
    )
    contracts = [{
        "model_id": "resnet50",
        "task": "classification",
        "backend": "hailo8",
        "variant": "full",
        "requested": False,
    }]
    promotions = promote_verified_hailo_full_contracts(
        suite_dir=tmp_path,
        model_id="resnet50",
        task="classification",
        suite_bench={},
        contracts=contracts,
        copied_verified={},
    )

    assert promotions == []
