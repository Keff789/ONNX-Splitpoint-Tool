from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _benchmark_runs_from_profile,
)
from onnx_splitpoint_tool.workflow import execution_binding
from onnx_splitpoint_tool.workflow.execution_binding import (
    ExecutionBindingResult,
    _remote_execution_if_requested,
    _run_remote_dispatch_once,
)
from onnx_splitpoint_tool.workflow.full_only_quality_canary import (
    project_full_only_quality_plan_rows,
    resolve_full_only_quality_canary,
)
from onnx_splitpoint_tool.workflow.setup_local_trt_dispatch import (
    build_setup_local_tensorrt_quality_dispatch,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


ROOT = Path(__file__).resolve().parents[1]
H8 = "orin_nx_hailo8_01"
H10 = "orin_nx_hailo10_01"


def _run_profiles() -> list[dict]:
    return [
        {
            "id": "ort_tensorrt",
            "type": "same_backend_reference",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
            "required": True,
        },
        {
            "id": "hailo8",
            "type": "same_backend_reference",
            "full": "hailo8",
            "stage1": "hailo8",
            "stage2": "hailo8",
            "required": True,
        },
        {
            "id": "hailo10",
            "type": "same_backend_reference",
            "full": "hailo10h",
            "stage1": "hailo10h",
            "stage2": "hailo10h",
            "required": True,
        },
    ]


def _profile() -> dict:
    return {
        "model_suite": {
            "primary": [{
                "id": "yolov7_paper", "task": "detection",
                "enabled": True,
            }],
        },
        "run_profiles": _run_profiles(),
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 1,
            "require_single_part2_input": False,
        },
        "workflow": {
            "execution_mode": "generate_and_run",
            "skip_runtime_benchmarks": False,
        },
        "quality_gate": {
            "statistics": {
                "execution_location": "central_management",
                "bootstrap_repetitions": 500,
            },
        },
        "quality_canary": {
            "enabled": True,
            "execution_scope": "full_only",
            "full_run_ids": [
                {
                    "id": "hailo8_full",
                    "run_id": "hailo8",
                    "setup_id": H8,
                    "backend": "hailo8",
                    "execution_role": "full_quality_only",
                },
                {
                    "id": "hailo10h_full",
                    "run_id": "hailo10",
                    "setup_id": H10,
                    "backend": "hailo10h",
                    "execution_role": "full_quality_only",
                },
            ],
            "setup_local_tensorrt_companions": [
                {
                    "id": "tensorrt_at_hailo8_full",
                    "run_id": "ort_tensorrt",
                    "setup_id": H8,
                    "backend": "tensorrt",
                    "execution_role": "full_quality_only",
                },
                {
                    "id": "tensorrt_at_hailo10h_full",
                    "run_id": "ort_tensorrt",
                    "setup_id": H10,
                    "backend": "tensorrt",
                    "execution_role": "full_quality_only",
                },
            ],
        },
        "execution_preset": {
            "id": "standard",
            "overrides": {
                "native_enabled": False,
                "energy_enabled": False,
            },
            "snapshot": {
                "defaults": {
                    "native_enabled": False,
                    "energy_enabled": False,
                },
            },
        },
    }


def _hardware_targets() -> list[dict]:
    return [
        {"id": H8, "accelerator": "hailo8", "enabled": True},
        {"id": H10, "accelerator": "hailo10h", "enabled": True},
    ]


def test_full_only_canary_exposes_exact_four_stable_identities_and_zero_generic_rows() -> None:
    profile = _profile()
    contract = resolve_full_only_quality_canary(
        profile, plan_rows=profile["run_profiles"],
    )

    assert contract["ok"] is True
    assert contract["performance_claims_emitted"] is False
    assert contract["expected_full_quality_identities"] == [
        {
            "id": "hailo8_full", "source_run_id": "hailo8",
            "run_id": "hailo8", "dispatch_run_id": "hailo8",
            "setup_id": H8,
            "backend": "hailo8", "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_hailo8_full",
            "source_run_id": "native_full_tensorrt",
            "run_id": "native_full_tensorrt",
            "dispatch_run_id": "ort_tensorrt", "setup_id": H8,
            "backend": "tensorrt", "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "hailo10h_full", "source_run_id": "hailo10",
            "run_id": "hailo10", "dispatch_run_id": "hailo10",
            "setup_id": H10,
            "backend": "hailo10h", "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_hailo10h_full",
            "source_run_id": "native_full_tensorrt",
            "run_id": "native_full_tensorrt",
            "dispatch_run_id": "ort_tensorrt",
            "setup_id": H10, "backend": "tensorrt", "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]

    plan = build_effective_execution_plan(profile)
    assert plan["quality_canary_enabled"] is True
    assert plan["generic_rows_total"] == 0
    assert plan["expected_generic_result_rows_total"] == 0
    assert plan["effective_generic_run_ids"] == []
    assert plan["expected_full_quality_results_per_model"] == 4
    assert plan["expected_full_quality_results_total"] == 4
    assert plan["expected_full_quality_identities"] == contract[
        "expected_full_quality_identities"
    ]
    assert plan["setup_groups"] == {
        H8: ["hailo8", "ort_tensorrt"],
        H10: ["hailo10", "ort_tensorrt"],
    }
    assert plan["remote_run_invocations_total"] == 2
    assert plan["tensorrt_performance_owner_group"] == ""
    assert plan["performance_claims_emitted"] is False


def test_full_only_projection_and_dispatch_are_quality_only_on_both_setups() -> None:
    profile = _profile()
    rows = _benchmark_runs_from_profile(profile, ["hailo8", "hailo10h"])
    executable = [
        row for row in rows if not row.get("semantic_reference_only")
    ]
    assert {row["id"] for row in executable} == {
        "hailo8", "hailo10", "ort_tensorrt",
    }
    assert all(row["variant"] == "full" for row in executable)
    assert all(row["variants"] == ["full"] for row in executable)
    assert all(row["quality_evidence_only"] is True for row in executable)
    assert all(row["performance_claims_emitted"] is False for row in executable)

    dispatch = build_setup_local_tensorrt_quality_dispatch(
        profile,
        hardware_targets=_hardware_targets(),
        plan_rows=rows,
    )
    assert dispatch["ok"] is True, dispatch["errors"]
    assert dispatch["performance_owner_setup_id"] == ""
    assert dispatch["performance_claims_emitted"] is False
    assert dispatch["expected_full_quality_identities"] == (
        resolve_full_only_quality_canary(
            profile, plan_rows=profile["run_profiles"],
        )["expected_full_quality_identities"]
    )
    by_setup = {
        row["setup_id"]: row for row in dispatch["setup_dispatches"]
    }
    assert by_setup[H8]["run_ids"] == ["hailo8", "ort_tensorrt"]
    assert by_setup[H10]["run_ids"] == ["hailo10", "ort_tensorrt"]
    assert by_setup[H8]["quality_only_run_ids"] == by_setup[H8]["run_ids"]
    assert by_setup[H10]["quality_only_run_ids"] == by_setup[H10]["run_ids"]
    assert all(
        row["performance_claims_emitted"] is False
        for row in dispatch["setup_dispatches"]
    )


def test_real_generator_plan_is_projected_to_only_the_three_full_recipes() -> None:
    generated = BenchmarkGenerationService().build_run_plan(
        acc_cpu=False,
        acc_cuda=False,
        acc_trt=True,
        acc_h8=True,
        acc_h10=True,
        acc_deepx=False,
        hailo8_hw="hailo8",
        hailo10_hw="hailo10",
        hailo_preset="Custom",
        hailo_custom_full=True,
        hailo_custom_composed=False,
        hailo_custom_part1=False,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=False,
        matrix_deepx_to_trt=False,
    ).bench_plan_runs

    rows = project_full_only_quality_plan_rows(_profile(), generated)
    assert [row["id"] for row in rows] == [
        "ort_tensorrt", "hailo8", "hailo10",
    ]
    assert all(row["variants"] == ["full"] for row in rows)
    assert all(row["quality_evidence_only"] is True for row in rows)
    assert all(row["performance_claims_emitted"] is False for row in rows)


@pytest.mark.parametrize(
    "run_id, backend, setup_id",
    [("hailo8", "hailo8", H8), ("hailo10", "hailo10h", H10)],
)
def test_generated_suite_executes_same_backend_vendor_as_full_quality_only(
    tmp_path: Path, run_id: str, backend: str, setup_id: str,
) -> None:
    suite = runpy.run_path(
        str(
            ROOT / "onnx_splitpoint_tool/resources/templates"
            / "benchmark_suite.py.txt"
        ),
        run_name="v27522_generated_suite_test",
    )
    calls: list[dict] = []

    def fake_run_case(_case_dir, **kwargs):
        calls.append(dict(kwargs))
        return {
            "task_quality_input_requests_by_variant": {
                "full": {
                    "schema": (
                        "onnx-splitpoint/central-quality-evaluation-request"
                    ),
                    "variant": "full",
                    "record_count": 12,
                    "status": "pending_central_evaluation",
                },
            },
        }

    suite["_run_declared_full_quality_only_endpoint"].__globals__[
        "_run_case"
    ] = fake_run_case
    run = {
        "id": run_id,
        "type": "same_backend_reference",
        "full": backend,
        "stage1": backend,
        "stage2": backend,
        "variant": "full",
        "variants": ["full"],
        "execution_scope": "full_only",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "quality_canary_setup_ids": [setup_id],
        "quality_canary_endpoint_ids": [f"{run_id}_full"],
        "validation_images": "frozen_validation",
        "validation_max_images": 12,
        "validation_budget_authoritative": True,
        "benchmark_task": "detection",
        "task_quality_gate": {"statistics": {"execution_location": "central_management"}},
    }
    report = suite["_run_declared_full_quality_only_endpoint"](
        root=tmp_path,
        bench={"model_id": "yolov7_paper"},
        plan={},
        run=run,
        run_cases=[{"case_id": "b001", "case_dir": "b001", "boundary": 1}],
        args=SimpleNamespace(
            quality_evidence_setup_id=setup_id,
            quality_evidence_eval_id="eval-v27522",
            quality_evidence_model_id="yolov7_paper",
            validation_images="",
            validation_max_images=0,
        ),
    )

    assert report is not None
    assert report["source_run_id"] == run_id
    assert report["variant"] == "full"
    assert report["performance_claims_emitted"] is False
    assert len(calls) == 1
    assert calls[0]["provider"] == backend
    assert calls[0]["variants"] == ["full"]
    assert calls[0]["warmup"] == 0
    assert calls[0]["runs"] == 1
    assert calls[0]["throughput_frames"] == 0
    assert calls[0]["full_only_quality_identity"] == {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": f"{run_id}_full",
        "eval_run_id": "eval-v27522",
        "model_id": "yolov7_paper",
        "setup_id": setup_id,
        "source_run_id": run_id,
        "backend": backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }


def test_real_vendor_export_seals_full_only_identity_before_candidate_hash(
    tmp_path: Path,
) -> None:
    runner = runpy.run_path(
        str(
            ROOT / "onnx_splitpoint_tool/resources/templates"
            / "run_split_onnxruntime.py.txt"
        ),
        run_name="v27522_real_vendor_export_test",
    )
    identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": "hailo8_full",
        "eval_run_id": "eval-v27522",
        "model_id": "yolov7_paper",
        "setup_id": H8,
        "source_run_id": "hailo8",
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    kwargs = {
        "task": "classification",
        "variant": "full",
        "policy": {
            "statistics": {"execution_location": "central_management"},
            "classification": {"non_inferiority_margin": 0.01},
        },
        "classification_rows": [{
            "image": "image-1.jpg",
            "label_id": 1,
            "gt": {"top1_hit": True, "top5_hit": True},
            "gt_reference": {"top1_hit": True, "top5_hit": True},
        }],
        "endpoint_contract": {
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "a" * 64,
            "stage": "classification_logits",
        },
        "runtime_precision_identity": "float32",
        "full_only_quality_identity": identity,
    }
    exported = runner["_export_central_quality_inputs"](
        out_dir=tmp_path / "export", **kwargs,
    )
    request_path = Path(exported["request"]["path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_path = request_path.parent / request["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    identity_fields = (
        "model_id", "source_run_id", "setup_id", "backend", "variant",
        "execution_role", "performance_claims_emitted",
    )
    for payload in (request, candidate):
        assert payload["full_only_plan_identity_required"] is True
        assert payload["full_only_plan_identity"] == identity
        assert {
            field_name: payload[field_name]
            for field_name in identity_fields
        } == {
            field_name: identity[field_name]
            for field_name in identity_fields
        }
        encoded_identity = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        assert payload["full_only_plan_identity_sha256"] == (
            hashlib.sha256(encoded_identity).hexdigest()
        )
    assert request["candidate"]["sha256"] == hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    assert request["candidate"]["size_bytes"] == candidate_path.stat().st_size

    for invalid_version in (True, "1", 2):
        invalid_identity = {**identity, "schema_version": invalid_version}
        with pytest.raises(RuntimeError, match="exact sealed"):
            runner["_export_central_quality_inputs"](
                out_dir=tmp_path / f"invalid-{invalid_version!s}",
                **{
                    **kwargs,
                    "full_only_quality_identity": invalid_identity,
                },
            )


def test_full_only_central_binder_requires_all_materialized_claim_fields(
    tmp_path: Path,
) -> None:
    plan = build_effective_execution_plan(_profile())
    _write_json(tmp_path / "effective_execution_plan.json", plan)
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.run_id = tmp_path.name
    expected = next(
        row for row in plan["expected_full_quality_identities"]
        if row["source_run_id"] == "hailo8" and row["setup_id"] == H8
    )
    full_only_identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": expected["id"],
        "eval_run_id": tmp_path.name,
        "model_id": "yolov7_paper",
        "setup_id": H8,
        "source_run_id": "hailo8",
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    request_identity = {
        "model_id": "yolov7_paper",
        "source_run_id": "hailo8",
        "setup_id": H8,
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "source_request_sha256": "a" * 64,
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": full_only_identity,
        "full_only_plan_identity_sha256": hashlib.sha256(json.dumps(
            full_only_identity, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "quality_canary_id": expected["id"],
        "eval_run_id": tmp_path.name,
    }
    bound, errors = (
        workflow._bind_full_only_quality_request_to_effective_plan(
            model_id="yolov7_paper", request_identity=request_identity,
        )
    )
    assert errors == []
    assert bound["backend"] == "hailo8"

    missing_errors = {
        "backend": "full_only_request_backend_mismatch",
        "execution_role": "full_only_request_execution_role_mismatch",
        "performance_claims_emitted": (
            "full_only_request_performance_claims_emitted_mismatch"
        ),
    }
    for field_name, expected_error in missing_errors.items():
        incomplete = deepcopy(request_identity)
        incomplete.pop(field_name)
        rebound, observed_errors = (
            workflow._bind_full_only_quality_request_to_effective_plan(
                model_id="yolov7_paper", request_identity=incomplete,
            )
        )
        assert rebound == {}
        assert expected_error in observed_errors

    sealed_drift_cases = {
        "missing_eval_run_id": (
            {**request_identity, "eval_run_id": ""},
            "full_only_request_eval_run_id_mismatch",
        ),
        "wrong_eval_run_id": (
            {**request_identity, "eval_run_id": "older-run"},
            "full_only_request_eval_run_id_mismatch",
        ),
        "wrong_quality_canary_id": (
            {**request_identity, "quality_canary_id": "other-canary"},
            "full_only_request_quality_canary_id_mismatch",
        ),
        "missing_plan_identity": (
            {**request_identity, "full_only_plan_identity": {}},
            "full_only_request_plan_identity_mismatch",
        ),
        "wrong_plan_identity_sha256": (
            {
                **request_identity,
                "full_only_plan_identity_sha256": "b" * 64,
            },
            "full_only_request_plan_identity_sha256_mismatch",
        ),
    }
    for label, (drifted, expected_error) in sealed_drift_cases.items():
        rebound, observed_errors = (
            workflow._bind_full_only_quality_request_to_effective_plan(
                model_id="yolov7_paper", request_identity=drifted,
            )
        )
        assert rebound == {}, label
        assert expected_error in observed_errors, label


def test_generated_suite_marks_same_backend_tensorrt_as_native_companion() -> None:
    suite = runpy.run_path(
        str(
            ROOT / "onnx_splitpoint_tool/resources/templates"
            / "benchmark_suite.py.txt"
        ),
        run_name="v27522_generated_suite_trt_marker_test",
    )
    prepared = suite["_assign_full_baseline_owners"]([{
        "id": "ort_tensorrt",
        "type": "same_backend_reference",
        "full": "tensorrt",
        "stage1": "tensorrt",
        "stage2": "tensorrt",
        "variants": ["full"],
        "execution_scope": "full_only",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }])
    assert prepared[0]["_native_full_trt_quality_companion"] is True


def test_generated_suite_main_dispatches_same_backend_trt_to_native_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_root = tmp_path / "generated_suite"
    suite_root.mkdir()
    template = (
        ROOT / "onnx_splitpoint_tool/resources/templates"
        / "benchmark_suite.py.txt"
    )
    suite_path = suite_root / "benchmark_suite.py"
    suite_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(suite_root / "__BENCH_JSON__", {
        "model_id": "yolov7_paper",
        "cases": [{"case_id": "b001", "case_dir": "b001", "boundary": 1}],
    })
    _write_json(suite_root / "benchmark_plan.json", {"runs": [{
        "id": "ort_tensorrt",
        "type": "same_backend_reference",
        "full": "tensorrt",
        "stage1": "tensorrt",
        "stage2": "tensorrt",
        "variants": ["full"],
        "execution_scope": "full_only",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }]})
    suite = runpy.run_path(
        str(suite_path), run_name="v27522_generated_suite_main_test",
    )
    calls: list[dict] = []

    def fake_native_companion(**kwargs):
        assert kwargs["run"]["_native_full_trt_quality_companion"] is True
        calls.append(dict(kwargs))
        return {
            "schema": "onnx-splitpoint/full-only-quality-evidence-report",
            "record_count": 1,
            "source_run_id": "native_full_tensorrt",
            "setup_id": H8,
            "variant": "full",
        }

    globals_ = suite["main"].__globals__
    globals_["_run_native_full_trt_quality_companion"] = fake_native_companion
    globals_["_run_declared_full_quality_only_endpoint"] = (
        lambda **_kwargs: pytest.fail(
            "TensorRT must not enter the vendor Full-only helper"
        )
    )
    monkeypatch.setattr(sys, "argv", [
        str(suite_path),
        "--plan", "benchmark_plan.json",
        "--run-ids", "ort_tensorrt",
        "--quality-only-run-ids", "ort_tensorrt",
        "--quality-evidence-eval-id", "eval-v27522",
        "--quality-evidence-setup-id", H8,
        "--quality-evidence-model-id", "yolov7_paper",
        "--quality-evidence-endpoint-id", "tensorrt_at_hailo8_full",
        "--no-plot", "--no-csv",
    ])

    assert suite["main"]() == 0
    assert len(calls) == 1
    status = json.loads(
        (suite_root / "benchmark_suite_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["any_quality_evidence"] is True
    assert status["any_rows"] is False
    assert status["performance_claims_emitted"] is False


def test_full_only_canary_passes_the_real_strict_profile_schema() -> None:
    profile = _profile()
    profile.update({
        "name": "v27522_full_only_quality_canary",
        "validation": {},
        "reporting": {},
    })
    assert validate_evaluation_profile_payload(profile)["quality_canary"] == (
        profile["quality_canary"]
    )

    invalid = deepcopy(profile)
    invalid["quality_canary"]["full_endpoints"] = invalid[
        "quality_canary"
    ].pop("full_run_ids")
    with pytest.raises(ValueError, match="full_endpoints"):
        validate_evaluation_profile_payload(invalid)


@pytest.mark.parametrize(
    "mutate, expected_error",
    [
        (
            lambda p: p["quality_canary"].update(
                full_endpoints=p["quality_canary"].pop("full_run_ids")
            ),
            "quality_canary_unknown_field:full_endpoints",
        ),
        (
            lambda p: p["quality_canary"]["full_run_ids"][1].update(
                id="hailo8_full"
            ),
            "quality_canary_duplicate_id:hailo8_full",
        ),
        (
            lambda p: p["run_profiles"][1].update(
                variant="composed", variants=["full", "composed"]
            ),
            "quality_canary_run_recipe_invalid:hailo8:variant_not_full",
        ),
        (
            lambda p: p["quality_canary"]["full_run_ids"][0].update(
                execution_role="full_performance_owner"
            ),
            "implicit_or_explicit_performance_owner_forbidden",
        ),
        (
            lambda p: p["run_profiles"][1].update(
                performance_eligible=True
            ),
            "implicit_or_explicit_performance_owner_forbidden",
        ),
        (
            lambda p: p["run_profiles"].append({
                "id": "hailo8_to_trt", "type": "mixed_backend",
                "stage1": "hailo8", "stage2": "tensorrt",
            }),
            "quality_canary_unrequested_plan_run:hailo8_to_trt",
        ),
    ],
)
def test_full_only_canary_rejects_ambiguous_or_performance_shaped_profiles(
    mutate, expected_error: str,
) -> None:
    profile = _profile()
    mutate(profile)
    contract = resolve_full_only_quality_canary(
        profile, plan_rows=profile["run_profiles"],
    )
    assert contract["ok"] is False
    assert any(expected_error in error for error in contract["errors"])
    with pytest.raises(ValueError, match="full_only_quality_canary_invalid"):
        project_full_only_quality_plan_rows(
            profile, profile["run_profiles"],
        )


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _identified_run(path: Path) -> Path:
    path.mkdir(parents=True)
    _write_json(path / "run_manifest.json", {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": path.name,
        "status": "ok",
    })
    return path


def _write_remote_quality_pair(
    remote_run: Path, *, source_run_id: str, setup_id: str,
    model_id: str = "yolov7_paper", eval_run_id: str = "eval-v27522",
) -> None:
    backend = (
        "native_tensorrt"
        if source_run_id == "native_full_tensorrt"
        else "hailo10h" if source_run_id == "hailo10" else source_run_id
    )
    quality = (
        remote_run / "results/b001"
        / f"results_{source_run_id}/task_quality_inputs"
    )
    quality_canary_id = (
        "tensorrt_at_hailo8_full"
        if source_run_id == "native_full_tensorrt"
        else f"{source_run_id}_full"
    )
    full_only_identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": quality_canary_id,
        "eval_run_id": eval_run_id,
        "model_id": model_id,
        "setup_id": setup_id,
        "source_run_id": source_run_id,
        "backend": "tensorrt" if source_run_id == "native_full_tensorrt" else backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    full_only_sha = hashlib.sha256(json.dumps(
        full_only_identity, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    duplicates = {
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": full_only_identity,
        "full_only_plan_identity_sha256": full_only_sha,
        "quality_canary_id": quality_canary_id,
        "eval_run_id": eval_run_id,
    }
    candidate = quality / "full_candidate.json"
    _write_json(candidate, {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "model_id": model_id,
        "setup_id": setup_id,
        "source_run_id": source_run_id,
        "backend": backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "record_count": 1,
        "records": [{"image_id": "image-1", "prediction": [1, 2, 3]}],
        **duplicates,
    })
    _write_json(quality / "full_request.json", {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "model_id": model_id,
        "setup_id": setup_id,
        "source_run_id": source_run_id,
        "backend": backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "record_count": 1,
        **duplicates,
        "producer_identity": {
            "model_id": model_id,
            "setup_id": setup_id,
            "source_run_id": source_run_id,
            "backend": backend,
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        "candidate": {
            "path": candidate.name,
            "sha256": _sha(candidate),
            "size_bytes": candidate.stat().st_size,
        },
    })


def test_remote_dispatch_accepts_exact_full_quality_evidence_with_zero_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_run = tmp_path / "remote_run"
    live_runner = runpy.run_path(
        str(
            ROOT / "onnx_splitpoint_tool/resources/templates"
            / "run_split_onnxruntime.py.txt"
        ),
        run_name="v27522_remote_live_vendor_export_test",
    )
    live_runner["_export_central_quality_inputs"](
        out_dir=remote_run / "results/b001/results_hailo8",
        task="classification",
        variant="full",
        policy={
            "statistics": {"execution_location": "central_management"},
            "classification": {"non_inferiority_margin": 0.01},
        },
        classification_rows=[{
            "image": "image-1.jpg", "label_id": 1,
            "gt": {"top1_hit": True, "top5_hit": True},
            "gt_reference": {"top1_hit": True, "top5_hit": True},
        }],
        endpoint_contract={
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "a" * 64,
            "stage": "classification_logits",
        },
        runtime_precision_identity="float32",
        full_only_quality_identity={
            "schema": "onnx-splitpoint/full-only-quality-request-identity",
            "schema_version": 1,
            "quality_canary_id": "hailo8_full",
            "eval_run_id": tmp_path.name,
            "model_id": "yolov7_paper",
            "setup_id": H8,
            "source_run_id": "hailo8",
            "backend": "hailo8",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    )
    _write_remote_quality_pair(
        remote_run, source_run_id="native_full_tensorrt", setup_id=H8,
        eval_run_id=tmp_path.name,
    )

    class FakeRemoteBenchmarkService:
        def run(self, **_kwargs):
            return {
                "ok": True,
                "status": "ok",
                "local_run_dir": str(remote_run),
            }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.benchmark.services.RemoteBenchmarkService",
        FakeRemoteBenchmarkService,
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    _write_json(benchmark_set, {})
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    expected = [
        row for row in resolve_full_only_quality_canary(
            _profile(), plan_rows=_run_profiles(),
        )["expected_full_quality_identities"]
        if row["setup_id"] == H8
    ]

    def dispatch(output_dir: Path) -> ExecutionBindingResult:
        output_dir.mkdir(exist_ok=True)
        return _run_remote_dispatch_once(
            run_root=tmp_path,
            model_id="yolov7_paper",
            options=SimpleNamespace(
                remote_working_dir=str(tmp_path / "downloads"),
            ),
            profile_payload=_profile(),
            suite_dir=suite,
            benchmark_set_json=benchmark_set,
            result_dir=output_dir,
            gates={
                "execution_scope": "full_only",
                "expected_full_quality_identities": expected,
            },
            log=None,
            runtime_override={
                "enabled": True, "host": "h8", "user": "nx",
                "setup_id": H8,
            },
            target_id=H8,
            model_task="detection",
        )

    result = dispatch(result_dir)

    assert result.status == "ok", result.message
    assert result.metrics["canonical_result_rows_copied"] == 0
    assert result.metrics["quality_evidence_count"] == 2
    assert result.metrics["quality_evidence_status"] == "verified_exact"
    status = json.loads(
        (result_dir / f"remote_benchmark_status_{H8}.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "ok"
    assert status["canonical_nonempty_row_count"] == 0
    assert status["quality_evidence_count"] == 2

    vendor_request = next(
        remote_run.glob(
            "results/*/results_hailo8/task_quality_inputs/full_request.json"
        )
    )
    original_request = json.loads(vendor_request.read_text(encoding="utf-8"))
    vendor_candidate = vendor_request.parent / original_request["candidate"][
        "path"
    ]
    original_candidate = json.loads(
        vendor_candidate.read_text(encoding="utf-8")
    )
    negative_cases = {
        "wrong_backend": "deepx_m1",
        "missing_role": "execution_role_missing",
        "wrong_model": "model_mismatch",
        "performance_true": "performance_claims_emitted",
        "missing_size": "hash_or_size_mismatch",
        "wrong_schema": "schema_invalid",
        "empty_records": "record_count_invalid",
    }
    for mutation, expected_error in negative_cases.items():
        request_payload = deepcopy(original_request)
        candidate_payload = deepcopy(original_candidate)
        if mutation == "wrong_backend":
            request_payload["backend"] = "deepx_m1"
        elif mutation == "missing_role":
            request_payload.pop("execution_role")
            if isinstance(request_payload.get("producer_identity"), dict):
                request_payload["producer_identity"].pop(
                    "execution_role", None,
                )
        elif mutation == "wrong_model":
            request_payload["model_id"] = "unexpected_model"
        elif mutation == "performance_true":
            request_payload["performance_claims_emitted"] = True
        elif mutation == "wrong_schema":
            candidate_payload["schema"] = "unexpected/candidate-schema"
        elif mutation == "empty_records":
            candidate_payload["records"] = []
            candidate_payload["record_count"] = 0

        _write_json(vendor_candidate, candidate_payload)
        request_payload["candidate"]["sha256"] = _sha(vendor_candidate)
        request_payload["candidate"]["size_bytes"] = (
            vendor_candidate.stat().st_size
        )
        if mutation == "missing_size":
            request_payload["candidate"].pop("size_bytes")
        _write_json(vendor_request, request_payload)

        blocked = dispatch(tmp_path / f"results_blocked_{mutation}")
        assert blocked.status == "partial"
        assert blocked.metrics["quality_evidence_status"] == "blocked"
        assert any(
            expected_error in error
            for error in blocked.metrics["quality_evidence_errors"]
        ), blocked.metrics["quality_evidence_errors"]


def test_full_only_hardware_matrix_exports_quality_evidence_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    targets = []
    for target in _hardware_targets():
        target = dict(target)
        target["runtime"] = {
            "enabled": True,
            "host": "h8" if target["id"] == H8 else "h10",
            "user": "nx",
        }
        targets.append(target)
    profile["hardware_targets"] = targets
    monkeypatch.setattr(
        execution_binding, "matrix_for_runtime", lambda _profile: targets,
    )

    def fake_dispatch(**kwargs):
        identities = list(
            kwargs["gates"].get("expected_full_quality_identities") or []
        )
        assert len(identities) == 2
        return ExecutionBindingResult(
            artifacts={},
            metrics={
                "remote_result_files_copied": 2,
                "quality_evidence_count": len(identities),
            },
            status="ok",
            message="quality evidence verified",
        )

    monkeypatch.setattr(
        execution_binding, "_run_remote_dispatch_once", fake_dispatch,
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    rows = project_full_only_quality_plan_rows(_profile(), _run_profiles())
    _write_json(suite / "benchmark_plan.json", {"runs": rows})
    _write_json(suite / "benchmark_set.json", {})
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    result = _remote_execution_if_requested(
        run_root=tmp_path,
        model_id="yolov7_paper",
        options=SimpleNamespace(
            no_remote=False,
            benchmark_execution_backend="remote",
            parallel_remote_setups=False,
        ),
        profile_payload=profile,
        suite_dir=suite,
        benchmark_set_json=suite / "benchmark_set.json",
        result_dir=result_dir,
        contains_hailo=True,
        gates={},
        log=None,
    )

    assert result is not None and result.status == "ok"
    assert result.metrics["quality_evidence_count"] == 4
    matrix = json.loads(
        (result_dir / "remote_hardware_matrix_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert matrix["quality_evidence_count"] == 4
    assert matrix["expected_full_quality_count"] == 4


def _run_debug_pack(run: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_evaluation_debug_pack.py"),
            "--eval-run-dir", str(run), "--out", str(out),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_debug_pack_requires_and_hashes_all_present_execution_provenance(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "run")
    evidence = [
        run / "models/yolov7_paper/benchmark_set/generated_suite/cache/hailo_hef_build_receipt.json",
        run / "stages/compiler/hailo_hef_build_receipt.json",
        run / "models/yolov7_paper/benchmark_results/setup_local_tensorrt_dispatch_preflight.json",
        run / "models/yolov7_paper/benchmark_results/remote_hardware_matrix_status.json",
    ]
    for index, path in enumerate(evidence):
        _write_json(path, {"ok": True, "index": index})
    out = tmp_path / "pack.zip"
    proc = _run_debug_pack(run, out)
    assert proc.returncode == 0, proc.stdout

    expected = sorted(path.relative_to(run).as_posix() for path in evidence)
    with zipfile.ZipFile(out) as archive:
        assert set(expected) <= set(archive.namelist())
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
        section = manifest["execution_provenance_evidence"]
        assert section["required_members"] == expected
        assert section["present_members"] == expected
        assert section["missing_members"] == []
        assert section["all_present_members_sha256_recorded"] is True
    sidecar = json.loads(
        out.with_name(out.name + ".manifest.json").read_text(encoding="utf-8")
    )
    assert set(expected) <= set(sidecar["required_members"])


def test_debug_pack_includes_descriptors_and_declared_decoded_bodies(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "run")
    quality = (
        run / "models/yolov7_paper/benchmark_results/quality_inputs"
        / H8 / "results/full/task_quality_inputs"
    )
    candidate = quality / "full_candidate.json"
    reference = (
        run / "quality_management/references/yolov7_paper"
        / "canonical_cpu_reference.json"
    )
    annotations = run / "quality_management/references/yolov7_paper/annotations.json"
    ignored = quality / "unreferenced_candidate.json"
    ignored_reference = (
        run / "quality_management/references/unrequested_model"
        / "canonical_cpu_reference.json"
    )
    for path, payload in (
        (candidate, {
            "records": [{"image_id": "1"}],
            "large_detection_payload": "x" * (2 * 1024 * 1024 + 4096),
        }),
        (reference, {"records": [{"image_id": "1"}]}),
        (annotations, {"images": [{"id": 1}]}),
        (ignored, {"must_not": "be globbed"}),
        (ignored_reference, {"must_not": "be_walked"}),
    ):
        _write_json(path, payload)
    request = quality / "full_request.json"
    _write_json(request, {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "variant": "full",
        "candidate": {
            "path": candidate.name,
            "sha256": _sha(candidate),
            "size_bytes": candidate.stat().st_size,
        },
        "reference": {
            "path": str(reference.relative_to(run)),
            "sha256": _sha(reference),
            "size_bytes": reference.stat().st_size,
        },
        "annotations": {
            "path": str(annotations.relative_to(run)),
            "sha256": _sha(annotations),
            "size_bytes": annotations.stat().st_size,
        },
    })
    summary = run / "quality_management/central_quality_summary.json"
    _write_json(summary, {
        "schema": "onnx-splitpoint/central-quality-summary",
        "results": [{
            "model_id": "yolov7_paper",
            "source_request": str(request.relative_to(run)),
            "source_request_sha256": _sha(request),
            "candidate_predictions_sha256": _sha(candidate),
            "reference_predictions_sha256": _sha(reference),
            "annotations_sha256": _sha(annotations),
            "management_cpu_reference": {
                "reference_path": (
                    "/historical/EvaluationRuns/old/"
                    "quality_management/references/yolov7_paper/"
                    "canonical_cpu_reference.json"
                ),
                "reference_sha256": _sha(reference),
            },
        }],
    })
    out = tmp_path / "pack.zip"
    proc = _run_debug_pack(run, out)
    assert proc.returncode == 0, proc.stdout
    expected = {request.relative_to(run).as_posix()}
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert expected <= names
        assert candidate.relative_to(run).as_posix() in names
        assert reference.relative_to(run).as_posix() in names
        assert annotations.relative_to(run).as_posix() in names
        assert ignored.relative_to(run).as_posix() not in names
        assert ignored_reference.relative_to(run).as_posix() not in names
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
        section = manifest["central_quality_replay_inputs"]
        assert set(section["expected_members"]) == expected
        assert set(section["archived_members"]) == expected
        assert section["missing_source_members"] == []
        assert section["replay_payloads_included"] is True
        assert section["mode"] == "request_descriptors_and_declared_decoded_predictions"
        assert section["failures"] == []
        assert {row["kind"] for row in section["referenced_replay_payloads"]} == {
            "candidate", "reference", "annotations",
        }
    sidecar = json.loads(
        out.with_name(out.name + ".manifest.json").read_text(encoding="utf-8")
    )
    assert expected <= set(sidecar["required_members"])


def test_debug_pack_replay_bodies_do_not_consume_descriptor_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _identified_run(tmp_path / "run")
    quality = (
        run / "models/yolov7_paper/benchmark_results/quality_inputs"
        / H8 / "results/full/task_quality_inputs"
    )
    candidate = quality / "full_candidate.json"
    reference = (
        run / "quality_management/references/yolov7_paper"
        / "canonical_cpu_reference.json"
    )
    _write_json(candidate, {"records": [{"image_id": "1"}]})
    _write_json(reference, {"records": [{"image_id": "1"}]})
    request = quality / "full_request.json"
    _write_json(request, {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "variant": "full",
        "candidate": {
            "path": candidate.name,
            "sha256": _sha(candidate),
            "size_bytes": candidate.stat().st_size,
        },
    })
    _write_json(run / "quality_management/central_quality_summary.json", {
        "schema": "onnx-splitpoint/central-quality-summary",
        "results": [{
            "model_id": "yolov7_paper",
            "source_request": str(request.relative_to(run)),
            "source_request_sha256": _sha(request),
            "candidate_predictions_sha256": _sha(candidate),
            "management_cpu_reference": {
                "reference_path": str(reference.relative_to(run)),
                "reference_sha256": _sha(reference),
            },
        }],
    })
    out = tmp_path / "pack.zip"
    proc = _run_debug_pack(run, out)
    assert proc.returncode == 0, proc.stdout
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert request.relative_to(run).as_posix() in names
        assert candidate.relative_to(run).as_posix() in names
        assert reference.relative_to(run).as_posix() in names


def test_debug_pack_marks_missing_source_request_incomplete_but_publishes(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "run")
    _write_json(run / "quality_management/central_quality_summary.json", {
        "schema": "onnx-splitpoint/central-quality-summary",
        "results": [{
            "model_id": "yolov7_paper",
            "source_request": "",
            "status": "completed",
        }],
    })
    out = tmp_path / "pack.zip"

    proc = _run_debug_pack(run, out)

    assert proc.returncode == 0, proc.stdout
    with zipfile.ZipFile(out) as archive:
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
    central = manifest["central_quality_replay_inputs"]
    assert central["complete"] is False
    assert central["failures"][0]["reason"] == "path_missing"


def test_debug_pack_hash_binds_archived_request_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _identified_run(tmp_path / "run")
    quality = (
        run / "models/yolov7_paper/benchmark_results/quality_inputs"
        / H8 / "results/full/task_quality_inputs"
    )
    candidate = quality / "full_candidate.json"
    reference = (
        run / "quality_management/references/yolov7_paper"
        / "canonical_cpu_reference.json"
    )
    _write_json(candidate, {"records": [{"image_id": "before"}]})
    _write_json(reference, {"records": [{"image_id": "before"}]})
    request = quality / "full_request.json"
    _write_json(request, {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "variant": "full",
        "candidate": {
            "path": candidate.name,
            "sha256": _sha(candidate),
            "size_bytes": candidate.stat().st_size,
        },
    })
    _write_json(run / "quality_management/central_quality_summary.json", {
        "schema": "onnx-splitpoint/central-quality-summary",
        "results": [{
            "model_id": "yolov7_paper",
            "source_request": str(request.relative_to(run)),
            "source_request_sha256": _sha(request),
            "candidate_predictions_sha256": _sha(candidate),
            "management_cpu_reference": {
                "reference_path": str(reference.relative_to(run)),
                "reference_sha256": _sha(reference),
            },
        }],
    })
    out = tmp_path / "pack.zip"
    proc = _run_debug_pack(run, out)
    assert proc.returncode == 0, proc.stdout
    request_member = request.relative_to(run).as_posix()
    with zipfile.ZipFile(out) as archive:
        payload = archive.read(request_member)
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
        assert candidate.relative_to(run).as_posix() in archive.namelist()
    record = next(row for row in manifest["files"] if row["path"] == request_member)
    assert record["size_bytes"] == len(payload)
    assert record["sha256"] == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
