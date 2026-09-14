from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import execution_binding
from onnx_splitpoint_tool.workflow import runner as runner_module
from onnx_splitpoint_tool.workflow.execution_binding import (
    ExecutionBindingResult,
    _filter_setup_local_dispatch_for_targeted_full_quality,
    _remote_transport_run_id,
    _remote_execution_if_requested,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.run_control import stable_resume_options


SETUP = "orin_nx_hailo8_01"


def _identities() -> list[dict[str, object]]:
    return [
        {
            "id": "hailo8_full",
            "setup_id": SETUP,
            "source_run_id": "hailo8",
            "run_id": "hailo8",
            "dispatch_run_id": "hailo8",
            "backend": "hailo8",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_hailo8_full",
            "setup_id": SETUP,
            "source_run_id": "native_full_tensorrt",
            "run_id": "native_full_tensorrt",
            "dispatch_run_id": "ort_tensorrt",
            "backend": "tensorrt",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]


def _dispatch_contract() -> dict[str, object]:
    identities = _identities()
    return {
        "ok": True,
        "execution_scope": "full_only",
        "setup_dispatches": [{
            "setup_id": SETUP,
            "producer": "hailo8",
            "run_ids": ["hailo8", "ort_tensorrt"],
            "quality_only_run_ids": ["hailo8", "ort_tensorrt"],
            "expected_full_quality_identities": identities,
            "quality_companion_required": True,
            "quality_companion_endpoint_id": "tensorrt_at_hailo8_full",
            "quality_companion_identity": {
                "id": "tensorrt_at_hailo8_full",
            },
            "tensorrt_execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }],
        "expected_full_quality_identities": identities,
    }


def _target(row: dict[str, object], model_id: str) -> dict[str, object]:
    return {**row, "model_id": model_id}


def _synthetic_admission_runner(
    tmp_path: Path, preserved_count: int,
) -> tuple[EvaluationWorkflowRunner, list[dict[str, object]]]:
    run_dir = tmp_path / "phase5_run"
    run_dir.mkdir()
    model_ids = [
        "mobilenet_v3_large", "regnet_x_1_6gf", "yolo26m", "yolo11l",
    ]
    identities = _identities()
    (run_dir / "effective_execution_plan.json").write_text(json.dumps({
        "quality_canary_enabled": True,
        "quality_canary_execution_scope": "full_only",
        "models": model_ids,
        "expected_full_quality_identities": identities,
    }), encoding="utf-8")

    ordered = [
        ("mobilenet_v3_large", identities[0]),
        ("mobilenet_v3_large", identities[1]),
        ("regnet_x_1_6gf", identities[0]),
        ("regnet_x_1_6gf", identities[1]),
        ("yolo26m", identities[1]),
        ("yolo26m", identities[0]),
        ("yolo11l", identities[0]),
        ("yolo11l", identities[1]),
    ]
    all_results: list[dict[str, object]] = []
    for index, (model_id, plan_row) in enumerate(ordered):
        request_dir = (
            run_dir / "models" / model_id / "benchmark_results"
            / "quality_inputs" / SETUP / f"request_{index}"
        )
        request_dir.mkdir(parents=True)
        candidate = request_dir / "full_candidate.json"
        candidate.write_bytes(f"candidate-{index}".encode())
        expected_plan_identity = {
            "schema": "onnx-splitpoint/full-only-quality-request-identity",
            "schema_version": 1,
            "quality_canary_id": str(plan_row["id"]),
            "eval_run_id": run_dir.name,
            "model_id": model_id,
            "setup_id": SETUP,
            "source_run_id": str(plan_row["source_run_id"]),
            "backend": str(plan_row["backend"]),
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }
        identity = {
            "model_id": model_id,
            "source_run_id": str(plan_row["source_run_id"]),
            "setup_id": SETUP,
            "backend": str(plan_row["backend"]),
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
            "endpoint_contract_hash": hashlib.sha256(
                f"endpoint-{index}".encode()
            ).hexdigest(),
            "runtime_precision_identity": f"runtime-{index}",
            "full_only_plan_identity_sha256": hashlib.sha256(
                f"plan-{index}".encode()
            ).hexdigest(),
            "full_only_plan_identity": expected_plan_identity,
        }
        request = request_dir / "full_request.json"
        request.write_text(json.dumps({
            "candidate": {
                "path": candidate.name,
                "size_bytes": candidate.stat().st_size,
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            },
            "synthetic_identity": identity,
        }), encoding="utf-8")
        identity = {
            **identity,
            "source_request_sha256": hashlib.sha256(
                request.read_bytes()
            ).hexdigest(),
        }
        all_results.append({
            "status": "completed",
            "technical_status": "completed",
            # Real Hailo-8 B500 prefix: both vendor Full classification
            # results are scientific FAILs; both TRT companions pass.
            "decision": "fail" if index in {0, 2} else "pass",
            "source_request": str(request.relative_to(run_dir)),
            "_identity": identity,
        })

    (run_dir / "quality_management").mkdir()
    (run_dir / "quality_management" / "central_quality_summary.json").write_text(
        json.dumps({
            "quality_acceptance_identity_contract": {
                "model_ids": model_ids,
                "expected_identities": identities,
                "postcondition": {
                    "expected_count": 8,
                    "failed_count": 0,
                    "duplicate_count": 0,
                    "contract_error_count": 0,
                },
            },
            "results": all_results[:preserved_count],
        }),
        encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.options = SimpleNamespace(
        resume=True,
        run_id=run_dir.name,
        resume_missing_full_quality_only=True,
        only_model=None,
        max_models=None,
        force_stage=[],
        hailo_force_build=False,
        energy_enabled=False,
        native_producer_enabled=False,
    )
    runner._missing_full_quality_by_model = {}
    runner._missing_full_quality_keys = set()
    runner._missing_full_quality_queued_keys = set()
    runner._preserved_central_quality_results = []
    runner._preserved_central_quality_requests = {}
    runner._central_quality_result_identity = lambda result: (
        dict(result["_identity"]), []
    )

    def request_identity(path, *, model_id, manifest):
        del model_id
        identity = dict(manifest["synthetic_identity"])
        identity.update({
            "source_request_sha256": hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest(),
            "identity_valid": True,
            "identity_errors": [],
        })
        return identity

    runner._quality_request_identity = request_identity
    return runner, all_results


def test_target_filter_selects_hailo_only_or_exact_hailo_trt_pair() -> None:
    identities = _identities()
    hailo_only = _filter_setup_local_dispatch_for_targeted_full_quality(
        _dispatch_contract(), [_target(identities[0], "yolo26m")],
    )
    selected = hailo_only["setup_dispatches"][0]
    assert selected["run_ids"] == ["hailo8"]
    assert selected["quality_only_run_ids"] == ["hailo8"]
    assert selected["quality_companion_required"] is False
    assert selected["quality_companion_endpoint_id"] == ""

    pair = _filter_setup_local_dispatch_for_targeted_full_quality(
        _dispatch_contract(),
        [_target(row, "yolo11l") for row in identities],
    )
    selected = pair["setup_dispatches"][0]
    assert selected["run_ids"] == ["hailo8", "ort_tensorrt"]
    assert selected["quality_only_run_ids"] == ["hailo8", "ort_tensorrt"]
    assert selected["quality_companion_required"] is True
    assert selected["quality_companion_endpoint_id"] == (
        "tensorrt_at_hailo8_full"
    )


def test_exact_five_of_eight_admission_preserves_fail_results_and_merges_three(
    tmp_path: Path,
) -> None:
    runner, all_results = _synthetic_admission_runner(tmp_path, 5)
    original = copy.deepcopy(all_results[:5])

    runner._prepare_missing_full_quality_resume()

    assert runner._preserved_central_quality_results == original
    assert [
        result["decision"] for result in runner._preserved_central_quality_results
    ].count("fail") == 2
    assert {
        (
            result["_identity"]["model_id"],
            result["_identity"]["source_run_id"],
        ): result["decision"]
        for result in runner._preserved_central_quality_results
    } == {
        ("mobilenet_v3_large", "hailo8"): "fail",
        ("mobilenet_v3_large", "native_full_tensorrt"): "pass",
        ("regnet_x_1_6gf", "hailo8"): "fail",
        ("regnet_x_1_6gf", "native_full_tensorrt"): "pass",
        ("yolo26m", "native_full_tensorrt"): "pass",
    }
    assert {(key[0], key[1]) for key in runner._missing_full_quality_keys} == {
        ("yolo26m", "hailo8"),
        ("yolo11l", "hailo8"),
        ("yolo11l", "native_full_tensorrt"),
    }
    merged = runner._finalize_missing_full_quality_results(
        [*runner._preserved_central_quality_results, *all_results[5:]],
    )
    assert len(merged) == 8
    assert merged[:5] == original


def test_four_of_eight_admission_is_blocked(
    tmp_path: Path,
) -> None:
    runner, _all_results = _synthetic_admission_runner(tmp_path, 4)
    with pytest.raises(ValueError, match="scope_not_exact_5_to_7_of_8"):
        runner._prepare_missing_full_quality_resume()


@pytest.mark.parametrize("preserved_count", [6, 7])
def test_six_or_seven_of_eight_admission_preserves_partial_repairs(
    tmp_path: Path, preserved_count: int,
) -> None:
    runner, all_results = _synthetic_admission_runner(
        tmp_path, preserved_count,
    )
    runner._prepare_missing_full_quality_resume()
    assert len(runner._preserved_central_quality_results) == preserved_count
    assert len(runner._missing_full_quality_keys) == 8 - preserved_count
    merged = runner._finalize_missing_full_quality_results(
        [*runner._preserved_central_quality_results, *all_results[preserved_count:]],
    )
    assert len(merged) == 8


def test_target_filter_rejects_unsealed_or_duplicate_identity() -> None:
    identity = _target(_identities()[0], "yolo26m")
    with pytest.raises(RuntimeError, match="identity_set_invalid"):
        _filter_setup_local_dispatch_for_targeted_full_quality(
            _dispatch_contract(), [identity, identity],
        )
    forged = dict(identity)
    forged["source_run_id"] = "other"
    with pytest.raises(RuntimeError, match="not_in_source_contract"):
        _filter_setup_local_dispatch_for_targeted_full_quality(
            _dispatch_contract(), [forged],
        )
    forged_role = dict(identity)
    forged_role["execution_role"] = "performance"
    with pytest.raises(RuntimeError, match="identity_set_invalid"):
        _filter_setup_local_dispatch_for_targeted_full_quality(
            _dispatch_contract(), [forged_role],
        )


def test_targeted_resume_uses_an_isolated_remote_transport_workspace(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "phase5-run"
    original = _remote_transport_run_id(
        run_root=run_root,
        model_id="yolo26m",
        target_id=SETUP,
        gates={},
    )
    (tmp_path / original).mkdir()

    targeted = _remote_transport_run_id(
        run_root=run_root,
        model_id="yolo26m",
        target_id=SETUP,
        gates={"targeted_missing_full_quality_only": True},
        workflow_session_id="a" * 32,
    )

    assert original == f"phase5-run_yolo26m_{SETUP}"
    assert targeted == f"{original}_missing_full_quality_resume_{'a' * 32}"
    assert targeted != original
    assert not (tmp_path / targeted).exists()


def test_target_dispatch_injects_no_build_and_only_one_hailo_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = {
        "id": SETUP,
        "accelerator": "hailo8",
        "enabled": True,
        "runtime": {
            "enabled": True,
            "host": "hailo-host",
            "user": "nx",
            "add_args": "--native-trt-build --timeout 17",
        },
    }
    monkeypatch.setattr(
        execution_binding, "matrix_for_runtime", lambda _profile: [target],
    )
    monkeypatch.setattr(
        execution_binding,
        "build_setup_local_tensorrt_quality_dispatch",
        lambda *_args, **_kwargs: _dispatch_contract(),
    )
    calls: list[dict[str, object]] = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        return ExecutionBindingResult(
            artifacts={},
            metrics={
                "remote_result_files_copied": 1,
                "quality_evidence_count": 1,
            },
            status="ok",
            message="ok",
        )

    monkeypatch.setattr(
        execution_binding, "_run_remote_dispatch_once", fake_dispatch,
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    plan_rows = [
        {
            "id": "hailo8",
            "execution_scope": "full_only",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "ort_tensorrt",
            "execution_scope": "full_only",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]
    (suite / "benchmark_plan.json").write_text(
        json.dumps({"runs": plan_rows}), encoding="utf-8",
    )
    (suite / "benchmark_set.json").write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    result = _remote_execution_if_requested(
        run_root=tmp_path,
        model_id="yolo26m",
        options=SimpleNamespace(
            no_remote=False,
            benchmark_execution_backend="remote",
            parallel_remote_setups=False,
        ),
        profile_payload={
            "quality_gate": {
                "statistics": {"execution_location": "central_management"},
            },
        },
        suite_dir=suite,
        benchmark_set_json=suite / "benchmark_set.json",
        result_dir=result_dir,
        contains_hailo=True,
        gates={},
        log=None,
        targeted_full_quality_identities=[
            _target(_identities()[0], "yolo26m")
        ],
    )

    assert result is not None and result.status == "ok"
    assert len(calls) == 1
    call = calls[0]
    assert call["gates"]["hardware_run_ids"] == ["hailo8"]
    assert call["gates"]["expected_full_quality_identities"] == [
        _identities()[0]
    ]
    tokens = shlex.split(str(call["runtime_override"]["add_args"]))
    assert tokens[-2:] == ["--quality-only-run-ids", "hailo8"]
    assert tokens.count("--no-native-trt-build") == 1
    assert "--native-trt-build" not in tokens
    assert "ort_tensorrt" not in tokens


def test_both_quality_only_templates_preserve_explicit_no_build() -> None:
    root = Path(__file__).resolve().parents[1]
    suite = (
        root / "onnx_splitpoint_tool" / "resources" / "templates"
        / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    runner = (
        root / "onnx_splitpoint_tool" / "resources" / "templates"
        / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    assert 'bool(getattr(args, "native_trt_build", True))' in suite
    assert 'bool(getattr(args, "native_trt_build", True))' in runner
    assert "native_trt_build=not cache_verify_only" not in suite
    assert "args.native_trt_build = not _cache_verify_only()" not in runner


def test_targeted_runtime_refresh_replaces_root_and_rejected_case_runners(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "legacy_suite"
    for case_dir in (suite / "b602", suite / "_rejected_cases" / "b416"):
        case_dir.mkdir(parents=True)
        (case_dir / "split_manifest.json").write_text("{}", encoding="utf-8")
        (case_dir / "run_split_onnxruntime.py").write_text(
            "STALE_V2777_RUNNER\n", encoding="utf-8",
        )
    (suite / "benchmark_set.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/benchmark-set",
            "tool": {"gui": "workflow"},
            "cases": [],
        }),
        encoding="utf-8",
    )
    (suite / "benchmark_suite.py").write_text(
        "STALE_V2777_SUITE\n", encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.log = lambda _message: None

    artifacts = runner._refresh_missing_full_quality_suite_runtime(suite)

    suite_text = (suite / "benchmark_suite.py").read_text(encoding="utf-8")
    root_runner = (suite / "b602" / "run_split_onnxruntime.py").read_bytes()
    rejected_runner = (
        suite / "_rejected_cases" / "b416" / "run_split_onnxruntime.py"
    ).read_bytes()
    assert "STALE_V2777_SUITE" not in suite_text
    assert 'bool(getattr(args, "native_trt_build", True))' in suite_text
    assert b"STALE_V2777_RUNNER" not in root_runner
    assert root_runner == rejected_runner
    assert b'bool(getattr(args, "native_trt_build", True))' in root_runner
    assert set(artifacts) == {
        "targeted_suite_benchmark_runner_py",
        "targeted_suite_endpoint_attestor_py",
        "targeted_suite_case_runner_000_py",
        "targeted_suite_case_runner_001_py",
    }


def test_preserved_candidate_is_same_dir_size_and_hash_attested(
    tmp_path: Path,
) -> None:
    request = tmp_path / "full_request.json"
    candidate = tmp_path / "full_candidate.json"
    candidate.write_bytes(b"candidate-v1")
    import hashlib

    payload = {
        "candidate": {
            "path": candidate.name,
            "size_bytes": candidate.stat().st_size,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        },
    }
    request.write_text(json.dumps(payload), encoding="utf-8")
    attested = EvaluationWorkflowRunner._attest_preserved_quality_candidate(
        request, payload,
    )
    assert attested["path"] == str(candidate.resolve())

    candidate.write_bytes(b"candidate-v2")
    with pytest.raises(ValueError, match="candidate_attestation_failed"):
        EvaluationWorkflowRunner._attest_preserved_quality_candidate(
            request, payload,
        )


@pytest.mark.parametrize(
    ("model_id", "stage"),
    [
        ("yolo11l", "run_benchmarks"),
        ("yolo11l", "validate_outputs"),
        ("yolo11l", "hardware_smoke"),
        (None, "evaluate_quality"),
        (None, "aggregate_results"),
        (None, "generate_report"),
    ],
)
def test_targeted_rebuild_reason_precedes_failed_previous_state(
    tmp_path: Path,
    model_id: str | None,
    stage: str,
) -> None:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.run_id = tmp_path.name
    runner.profile_id = "profile"
    runner.options = SimpleNamespace(
        resume=True,
        resume_missing_full_quality_only=True,
    )
    runner._missing_full_quality_by_model = {"yolo11l": [{}]}

    decision = runner._resume_reuse_decision(
        model_id=model_id,
        stage=stage,
        previous={
            "status": "failed",
            "state": "failed",
            "complete": True,
        },
        expected_hash="a" * 64,
        forced=False,
        stage_job_id="",
        result_path=tmp_path / "stage_result.json",
    )

    assert decision["reusable"] is False
    assert decision["reason"] == "missing_full_quality_targeted_rebuild"
    assert decision["missing_full_quality_targeted_rebuild"] is True


@pytest.mark.parametrize(
    ("model_id", "stage"),
    [
        ("yolo26m", "run_benchmarks"),
        ("yolo11l", "build_backend_artifacts"),
        (None, "run_native_producers"),
    ],
)
def test_targeted_rebuild_reason_does_not_expand_beyond_repair_scope(
    tmp_path: Path,
    model_id: str | None,
    stage: str,
) -> None:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.run_id = tmp_path.name
    runner.profile_id = "profile"
    runner.options = SimpleNamespace(
        resume=True,
        resume_missing_full_quality_only=True,
    )
    runner._missing_full_quality_by_model = {"yolo11l": [{}]}

    decision = runner._resume_reuse_decision(
        model_id=model_id,
        stage=stage,
        previous={
            "status": "failed",
            "state": "failed",
            "complete": True,
        },
        expected_hash="a" * 64,
        forced=False,
        stage_job_id="",
        result_path=tmp_path / "stage_result.json",
    )

    assert decision["reusable"] is False
    assert decision["reason"] == "previous_stage_state_not_reusable:failed"
    assert "missing_full_quality_targeted_rebuild" not in decision


def test_yolo11_pre_nms_attestation_accepts_bare_sidecar_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    model_dir = run_dir / "models" / "yolo11l"
    suite_dir = model_dir / "benchmark_set" / "legacy_suite"
    suite_dir.mkdir(parents=True)

    hef = suite_dir / "hailo" / "full" / "compiled.hef"
    receipt = suite_dir / "hailo" / "full" / "build_receipt.json"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"sealed-hef")
    receipt.write_bytes(b"sealed-receipt")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    contracts = {
        "contracts": [
            {
                "model_id": "yolo11l",
                "backend": "hailo8",
                "variant": "full",
                "recorded_artifact_path": "hailo/full/compiled.hef",
                "recorded_artifact_sha256": digest(hef),
                "hailo_build_receipt_path": (
                    "hailo/full/build_receipt.json"
                ),
                "hailo_build_receipt_file_sha256": digest(receipt),
            },
            {
                "model_id": "yolo11l",
                "backend": "tensorrt",
                "variant": "full",
            },
        ],
    }
    suite_contract = suite_dir / "output_contracts.json"
    formal_contract = model_dir / "full_baselines" / "output_contracts.json"
    formal_contract.parent.mkdir(parents=True)
    for path in (suite_contract, formal_contract):
        path.write_text(json.dumps(contracts), encoding="utf-8")

    model_path = tmp_path / "yolo11l.onnx"
    model_path.write_bytes(b"attested-yolo11")
    model_path.with_suffix(".export.json").write_text(
        json.dumps({
            "verification": {
                "onnx_sha256": digest(model_path),
                "outputs": [{"shape": [1, 84, 8400]}],
            },
            "export": {
                "nms": False,
                "end2end_effective": False,
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner_module,
        "benchmark_set_postcondition_v60v",
        lambda _base: {
            "valid": True,
            "selected_suite_dir": str(suite_dir),
        },
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.options = SimpleNamespace(
        resume_missing_full_quality_only=True,
    )
    runner._missing_full_quality_by_model = {"yolo11l": [{}]}
    runner.manifest = {}
    runner._refresh_missing_full_quality_suite_runtime = lambda _suite: {}
    runner._register_artifacts = lambda *_args, **_kwargs: None

    artifacts = runner._synchronize_missing_full_quality_contracts(
        "yolo11l",
        {"resolved_path": str(model_path)},
    )

    assert set(artifacts) == {
        "targeted_suite_output_contracts_json",
        "targeted_formal_output_contracts_json",
    }
    repaired = json.loads(suite_contract.read_text(encoding="utf-8"))
    tensorrt = next(
        row for row in repaired["contracts"]
        if row["backend"] == "tensorrt"
    )
    assert tensorrt["endpoint_mode"] == "decoded_pre_nms"

    model_path.write_bytes(b"different-yolo11-bytes")
    with pytest.raises(
        ValueError,
        match="missing_full_quality_yolo11_pre_nms_attestation_failed",
    ):
        runner._synchronize_missing_full_quality_contracts(
            "yolo11l",
            {"resolved_path": str(model_path)},
        )


def test_hailo10_detection_full_is_declared_raw_head(tmp_path: Path) -> None:
    class FakeRunner:
        run_dir = tmp_path
        options = SimpleNamespace(hailo_build_full=True)
        manifest: dict[str, object] = {}

        @staticmethod
        def _targets():
            return ["hailo10"]

        @staticmethod
        def _task_for(_row):
            return "detection"

        @staticmethod
        def _baseline_backends(_targets):
            return ["hailo10"]

        @staticmethod
        def _prepared_full_hailo_info(_model_path):
            return {}

        @staticmethod
        def _raw_head_end_nodes(_row):
            return ["raw_head_0"]

    artifacts, metrics, _message, status = (
        EvaluationWorkflowRunner._stage_prepare_full_baselines(
            FakeRunner(), "yolo26m", {"family": "yolo26"},
        )
    )
    contracts = json.loads(
        artifacts["output_contracts_json"].read_text(encoding="utf-8")
    )["contracts"]
    assert status == "ok"
    assert metrics["raw_head_contract_count"] == 1
    assert contracts[0]["backend"] == "hailo10"
    assert contracts[0]["endpoint_mode"] == "raw_detection_head"
    assert contracts[0]["postprocessing_required"] is True


def test_one_command_script_restores_only_invocation_controls(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("id: phase5\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    archived_options = WorkflowOptions(
        profile=str(profile),
        out=str(tmp_path),
        required_run_mode="standard",
        require_fresh_run=True,
    ).to_dict()
    # Exact pre-v2.77.8 manifest shape: all 140 archived fields are present,
    # while the new invocation-only switch naturally is not.
    archived_options.pop("resume_missing_full_quality_only")
    assert len(archived_options) == 140
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "options": archived_options,
    }), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "resume_missing_full_quality.py"
    )
    spec = importlib.util.spec_from_file_location("targeted_resume_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = module._options_from_manifest(run_dir)
    assert options.resume is True
    assert options.require_fresh_run is False
    assert options.required_run_mode == "standard"
    assert options.resume_missing_full_quality_only is True
    assert options.run_id == "run"
    assert options.force_stage == []
    assert options.remote_resume is False
    assert options.remote_no_resume is True
    assert options.remote_reuse_bundle is False
    assert options.remote_no_reuse_bundle is True
    assert options.energy_enabled is False
    assert options.native_producer_enabled is False

    runner = object.__new__(EvaluationWorkflowRunner)
    runner.options = options
    runner.profile_payload = {"execution_preset": {"id": "standard"}}
    runner._validate_requested_start_contract()


def test_fresh_creation_guard_is_not_part_of_resume_contract() -> None:
    archived = WorkflowOptions(
        profile="phase5.yaml",
        out="runs",
        required_run_mode="standard",
        require_fresh_run=True,
    )
    resumed = WorkflowOptions(**archived.to_dict())
    resumed.resume = True
    resumed.run_id = "phase5-run"
    resumed.require_fresh_run = False

    assert stable_resume_options(resumed) == stable_resume_options(archived)
