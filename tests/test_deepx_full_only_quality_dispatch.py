from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.execution_binding import (
    _full_only_quality_evidence_summary,
    _run_remote_dispatch_once,
)
from tests.test_v269a_deepx_full_central_quality import (
    _base_tree,
    _bind_prepared_source,
    _performance_input_evidence,
    _prepared_input_audit_fields,
    _run,
    _suite_module,
)


SETUP_ID = "orin_nx_deepx_m1_01"


def _full_only_identity(
    *, canary_id: str, eval_run_id: str, source_run_id: str, backend: str,
) -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": canary_id,
        "eval_run_id": eval_run_id,
        "model_id": "resnet50",
        "setup_id": SETUP_ID,
        "source_run_id": source_run_id,
        "backend": backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }


def _identity_sha(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _write_full_only_pair(
    directory: Path, *, eval_run_id: str, canary_id: str,
    source_run_id: str, backend: str,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    identity = _full_only_identity(
        canary_id=canary_id,
        eval_run_id=eval_run_id,
        source_run_id=source_run_id,
        backend=backend,
    )
    duplicates = {
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": identity,
        "full_only_plan_identity_sha256": _identity_sha(identity),
        "quality_canary_id": identity["quality_canary_id"],
        "eval_run_id": identity["eval_run_id"],
        "model_id": identity["model_id"],
        "source_run_id": identity["source_run_id"],
        "setup_id": identity["setup_id"],
        "backend": identity["backend"],
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    candidate = directory / "full_candidate.json"
    candidate.write_text(json.dumps({
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "record_count": 1,
        "records": [{"image_id": "a.jpg", "candidate": {"top1": 3}}],
        **duplicates,
    }), encoding="utf-8")
    request = directory / "full_request.json"
    request.write_text(json.dumps({
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "record_count": 1,
        "candidate": {
            "path": candidate.name,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "size_bytes": candidate.stat().st_size,
        },
        **duplicates,
    }), encoding="utf-8")
    return request, candidate


def _write_trt_full_only_pair(
    directory: Path, *, eval_run_id: str,
) -> tuple[Path, Path]:
    return _write_full_only_pair(
        directory,
        eval_run_id=eval_run_id,
        canary_id="tensorrt_at_deepx_m1_full",
        source_run_id="native_full_tensorrt",
        backend="tensorrt",
    )


def _classification_semantic(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    image = root / "validation/a.jpg"
    audit = {
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "layout": "HWC",
        "color_space": "RGB",
        "preprocess_mode": "resize",
        "normalization": "embedded_dxcom_preprocessing",
        "scale": 0.56,
        "pad_x": 0,
        "pad_y": 0,
        "letterbox_pad_value": 0,
        "source_shape_hw": [300, 400],
        "contract_source": "test",
        **_prepared_input_audit_fields(
            task="classification",
            shape=[224, 224, 3],
            layout="HWC",
            normalization="embedded_dxcom_preprocessing",
        ),
    }
    _bind_prepared_source(audit, image)
    records = root / "results/deepx_m1_full/classification_topk.json"
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(json.dumps({"images": [{
        "image": str(image),
        "label_id": 3,
        "label_name": "external-vendor-three",
        "top1": 3,
        "top5": [3, 1, 2, 4, 5],
        "scores": [1.0, 0.5, 0.4, 0.3, 0.2],
        "top1_correct": True,
        "top5_correct": True,
        "preprocessing_audit": audit,
    }]}), encoding="utf-8")
    invariant = {
        key: value for key, value in audit.items()
        if key not in {
            "source_shape_hw", "scale", "pad_x", "pad_y",
            "prepared_tensor_binding",
        }
    }
    semantic = {
        "enabled": True,
        "status": "ok",
        "task": "classification",
        "image_count": 1,
        "validated_image_count": 1,
        "error_count": 0,
        "classification_labeled_samples": 1,
        "classification_top1_accuracy": 1.0,
        "classification_top5_accuracy": 1.0,
        "classification_topk_json": str(records.relative_to(root)),
        "preprocessing_contract": {
            "pass": True,
            "all_samples_same_contract": True,
            "invariant_sample": invariant,
        },
        "runtime_output_contract": {
            "pass": True,
            "all_samples_same_contract": True,
            "validated_observation_count": 1,
            "sample": {
                "outputs": [{
                    "index": 0,
                    "shape": [1, 1000],
                    "dtype": "float32",
                }],
            },
        },
        "classification_source_endpoint_contract": {
            "status": "ok",
            "pass": True,
            "stage": "classification_logits",
            "validated_observation_count": 1,
        },
    }
    prepared = {
        **_performance_input_evidence(image, audit),
        "mean_ms": 1.25,
        "fps_makespan": 800.0,
        "preprocessing_contract_audit": {"pass": True},
    }
    return semantic, prepared


def _bench() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 2,
        "model_name": "resnet50",
        "model_id": "resnet50",
        "model": "models/resnet50.onnx",
        "artifact_manifest": {
            "schema": "onnx-splitpoint/benchmark-set",
            "schema_version": 2,
            "files": {"models": ["models/resnet50.onnx"]},
            "counts": {"models": 1},
        },
    }


def _full_only_run() -> dict[str, Any]:
    return {
        **_run("classification"),
        "type": "same_backend_reference",
        "full": "deepx_m1",
        "stage1": "deepx_m1",
        "stage2": "deepx_m1",
        "execution_scope": "full_only",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "quality_canary_setup_ids": [SETUP_ID],
        "quality_canary_endpoint_ids": ["deepx_m1_full"],
        "validation_max_images": 1,
        "validation_budget_authoritative": True,
    }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        quality_evidence_setup_id=SETUP_ID,
        quality_evidence_eval_id="eval-deepx-full-only",
        quality_evidence_model_id="resnet50",
        validation_images="",
        validation_max_images=0,
        benchmark_task="classification",
        energy_measurement_only=False,
        runs=99,
        warmup=99,
        timeout=30,
        prepared_input_manifest="",
    )


def _prepare_root(tmp_path: Path) -> tuple[Path, Path]:
    root, source, dxnn = _base_tree(
        tmp_path,
        task="classification",
        samples=[{"image": "a.jpg", "label_id": 3, "label_name": "three"}],
    )
    source.rename(root / "models/resnet50.onnx")
    (root / "b001").mkdir()
    return root, dxnn


def test_generated_suite_deepx_full_only_uses_dxnn_export_and_never_case_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _suite_module()
    root, dxnn = _prepare_root(tmp_path)
    semantic, prepared = _classification_semantic(root)

    def forbidden_case_runner(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("DeepX Full-only must never enter _run_case/ORT")

    def forbidden_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("DeepX Full-only must not run the --use-ort diagnostic")

    monkeypatch.setitem(
        module._run_declared_full_quality_only_endpoint.__globals__,
        "_run_case", forbidden_case_runner,
    )
    monkeypatch.setitem(
        module._run_deepx_full_run.__globals__,
        "_run_deepx_prepared_feed_benchmark",
        lambda *_args, **_kwargs: dict(prepared),
    )
    monkeypatch.setitem(
        module._run_deepx_full_run.__globals__,
        "_run_deepx_semantic_validation",
        lambda *_args, **_kwargs: dict(semantic),
    )
    monkeypatch.setattr(module.subprocess, "run", forbidden_subprocess)

    stale_root = root / "benchmark_results_deepx_m1_full_balanced.json"
    stale_root.write_text("[]", encoding="utf-8")
    stale_case = root / "b001/results_deepx_m1_full/validation_report.json"
    stale_case.parent.mkdir(parents=True)
    stale_case.write_text("{}", encoding="utf-8")

    report = module._run_declared_full_quality_only_endpoint(
        root=root,
        bench=_bench(),
        plan={},
        run=_full_only_run(),
        run_cases=[{"case_id": "b001", "case_dir": "b001", "boundary": 1}],
        args=_args(),
    )

    assert report is not None
    dxnn_sha = hashlib.sha256(dxnn.read_bytes()).hexdigest()
    precision = f"deepx_dxnn_sha256:{dxnn_sha}"
    assert report["backend"] == "deepx_m1"
    assert report["performance_claims_emitted"] is False
    assert report["runtime_precision_identity"] == precision
    request = report["request"]
    identity = _full_only_identity(
        canary_id="deepx_m1_full",
        eval_run_id="eval-deepx-full-only",
        source_run_id="deepx_m1_full",
        backend="deepx_m1",
    )
    identity_sha = _identity_sha(identity)
    assert request["full_only_plan_identity"] == identity
    assert request["full_only_plan_identity_sha256"] == identity_sha
    assert request["runtime_precision_identity"] == precision
    assert request["producer_identity"]["backend"] == "deepx_m1"
    assert request["producer_identity"]["runtime_precision_identity"] == precision

    candidate_path = (
        root / "results/deepx_m1_full/task_quality_inputs"
        / request["candidate"]["path"]
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["full_only_plan_identity"] == identity
    assert candidate["full_only_plan_identity_sha256"] == identity_sha
    assert candidate["runtime_precision_identity"] == precision
    assert request["candidate"]["sha256"] == hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    request_path = root / (
        "results/deepx_m1_full/task_quality_inputs/full_request.json"
    )
    collected_identity = EvaluationWorkflowRunner._quality_request_identity(
        request_path, model_id="resnet50",
    )
    assert collected_identity["identity_valid"] is True
    assert collected_identity["setup_id"] == SETUP_ID
    assert collected_identity["backend"] == "deepx_m1"
    assert collected_identity["runtime_precision_identity"] == precision
    assert not stale_root.exists()
    assert not stale_case.exists()
    assert not (
        root / "results/deepx_m1_full/deepx_prepared_feed_benchmark.json"
    ).exists()
    assert not (
        root / "results/deepx_m1_full/run_model_stdout.txt"
    ).exists()


def test_real_full_only_collector_accepts_exact_deepx_and_trt_two_of_two(
    tmp_path: Path,
) -> None:
    module = _suite_module()
    root, dxnn = _prepare_root(tmp_path)
    semantic, prepared = _classification_semantic(root)
    identity = _full_only_identity(
        canary_id="deepx_m1_full",
        eval_run_id="eval-deepx-full-only",
        source_run_id="deepx_m1_full",
        backend="deepx_m1",
    )
    deepx_export = module._deepx_export_central_quality_request(
        root,
        dxnn,
        {
            **_run("classification"),
            "setup_id": SETUP_ID,
        },
        semantic,
        root / "results/deepx_m1_full",
        completed_task_evidence=prepared,
        expected_model_id="resnet50",
        expected_backend="deepx_m1",
        expected_variant="full",
        full_only_quality_identity=identity,
    )
    deepx_request = Path(deepx_export["request"]["path"])
    deepx_candidate = (
        deepx_request.parent / deepx_export["candidate"]["path"]
    )
    deepx_candidate_payload = json.loads(
        deepx_candidate.read_text(encoding="utf-8")
    )
    assert deepx_candidate_payload["record_count"] == 1
    assert deepx_candidate_payload["record_count"] == len(
        deepx_candidate_payload["records"]
    )
    assert deepx_export["record_count"] == (
        deepx_candidate_payload["record_count"]
    )

    trt_dir = (
        root / "quality_inputs" / SETUP_ID
        / "results_native_full_tensorrt" / "task_quality_inputs"
    )
    trt_request, trt_candidate = _write_trt_full_only_pair(
        trt_dir, eval_run_id="eval-deepx-full-only",
    )

    copied = [
        {"kind": "central_quality_input", "destination": str(path)}
        for path in (
            deepx_request, deepx_candidate, trt_request, trt_candidate,
        )
    ]
    expected = [
        {
            "id": "deepx_m1_full",
            "source_run_id": "deepx_m1_full",
            "setup_id": SETUP_ID,
            "backend": "deepx_m1",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_deepx_m1_full",
            "source_run_id": "native_full_tensorrt",
            "setup_id": SETUP_ID,
            "backend": "tensorrt",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]
    summary = _full_only_quality_evidence_summary(
        copied,
        expected_identities=expected,
        setup_id=SETUP_ID,
        model_id="resnet50",
        eval_run_id="eval-deepx-full-only",
    )
    assert summary["status"] == "verified_exact", summary["errors"]
    assert summary["expected_count"] == 2
    assert summary["quality_evidence_count"] == 2
    assert summary["errors"] == []
    assert {
        row["source_run_id"] for row in summary["evidence"]
    } == {"deepx_m1_full", "native_full_tensorrt"}


@pytest.mark.parametrize(
    ("target", "mutation", "expected_error"),
    [
        ("request", "missing_eval", "quality_request_eval_run_id_mismatch"),
        ("request", "wrong_eval", "quality_request_eval_run_id_mismatch"),
        (
            "request", "wrong_identity",
            "quality_request_full_only_plan_identity_mismatch",
        ),
        (
            "request", "wrong_identity_sha",
            "quality_request_full_only_plan_identity_sha256_mismatch",
        ),
        (
            "candidate", "missing_eval",
            "quality_candidate_eval_run_id_mismatch",
        ),
        (
            "candidate", "wrong_identity_sha",
            "quality_candidate_full_only_plan_identity_sha256_mismatch",
        ),
    ],
)
def test_full_only_collector_blocks_replayed_or_unsealed_identity(
    tmp_path: Path, target: str, mutation: str, expected_error: str,
) -> None:
    eval_run_id = "current-evaluation-run"
    deepx_dir = tmp_path / "quality_inputs" / SETUP_ID / (
        "results_deepx_m1_full/task_quality_inputs"
    )
    deepx_request, deepx_candidate = _write_full_only_pair(
        deepx_dir,
        eval_run_id=eval_run_id,
        canary_id="deepx_m1_full",
        source_run_id="deepx_m1_full",
        backend="deepx_m1",
    )
    trt_request, trt_candidate = _write_trt_full_only_pair(
        tmp_path / "quality_inputs" / SETUP_ID
        / "results_native_full_tensorrt/task_quality_inputs",
        eval_run_id=eval_run_id,
    )
    target_path = deepx_request if target == "request" else deepx_candidate
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if mutation == "missing_eval":
        payload.pop("eval_run_id", None)
    elif mutation == "wrong_eval":
        payload["eval_run_id"] = "replayed-evaluation-run"
    elif mutation == "wrong_identity":
        payload["full_only_plan_identity"]["quality_canary_id"] = (
            "different-canary"
        )
        payload["full_only_plan_identity_sha256"] = _identity_sha(
            payload["full_only_plan_identity"]
        )
    elif mutation == "wrong_identity_sha":
        payload["full_only_plan_identity_sha256"] = "b" * 64
    else:  # pragma: no cover - parameter table is exhaustive
        raise AssertionError(mutation)
    target_path.write_text(json.dumps(payload), encoding="utf-8")
    if target == "candidate":
        request_payload = json.loads(
            deepx_request.read_text(encoding="utf-8")
        )
        request_payload["candidate"]["sha256"] = hashlib.sha256(
            deepx_candidate.read_bytes()
        ).hexdigest()
        request_payload["candidate"]["size_bytes"] = (
            deepx_candidate.stat().st_size
        )
        deepx_request.write_text(
            json.dumps(request_payload), encoding="utf-8",
        )

    expected = [
        {
            "id": "deepx_m1_full",
            "source_run_id": "deepx_m1_full",
            "setup_id": SETUP_ID,
            "backend": "deepx_m1",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_deepx_m1_full",
            "source_run_id": "native_full_tensorrt",
            "setup_id": SETUP_ID,
            "backend": "tensorrt",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]
    copied = [
        {"kind": "central_quality_input", "destination": str(path)}
        for path in (
            deepx_request, deepx_candidate, trt_request, trt_candidate,
        )
    ]
    summary = _full_only_quality_evidence_summary(
        copied,
        expected_identities=expected,
        setup_id=SETUP_ID,
        model_id="resnet50",
        eval_run_id=eval_run_id,
    )
    assert summary["status"] == "blocked"
    assert summary["quality_evidence_count"] == 1
    assert any(
        expected_error in error for error in summary["errors"]
    ), summary["errors"]


def test_remote_dispatch_deepx_full_only_skips_performance_materialization_and_collects_two_of_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _suite_module()
    remote_run, dxnn = _prepare_root(tmp_path)
    semantic, prepared = _classification_semantic(remote_run)
    deepx_identity = _full_only_identity(
        canary_id="deepx_m1_full",
        eval_run_id=tmp_path.name,
        source_run_id="deepx_m1_full",
        backend="deepx_m1",
    )
    module._deepx_export_central_quality_request(
        remote_run,
        dxnn,
        {**_run("classification"), "setup_id": SETUP_ID},
        semantic,
        remote_run / "results/deepx_m1_full",
        completed_task_evidence=prepared,
        expected_model_id="resnet50",
        expected_backend="deepx_m1",
        expected_variant="full",
        full_only_quality_identity=deepx_identity,
    )

    trt_dir = (
        remote_run / "results/results_native_full_tensorrt"
        / "task_quality_inputs"
    )
    _write_trt_full_only_pair(trt_dir, eval_run_id=tmp_path.name)

    # The real DeepX Full-only producer leaves the portable prepared input in
    # place, but deliberately emits no benchmark_results performance row.
    # Invalid file bodies make this test prove that materialization is not
    # merely attempted and tolerated: entering its strict validator would fail.
    prepared_input = remote_run / "results/deepx_m1_full/prepared_input"
    prepared_input.mkdir(parents=True)
    (prepared_input / "native_full_input_manifest.json").write_text(
        "{}", encoding="utf-8",
    )
    (prepared_input / "runtime_input.bin").write_bytes(b"quality-only")
    (prepared_input / "input_rgb_uint8.bin").write_bytes(b"quality-only")
    assert not list((remote_run / "results").glob("benchmark_results_*.json"))

    class FakeRemoteBenchmarkService:
        def run(self, **_kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "status": "ok", "local_run_dir": str(remote_run)}

    monkeypatch.setattr(
        "onnx_splitpoint_tool.benchmark.services.RemoteBenchmarkService",
        FakeRemoteBenchmarkService,
    )
    suite = tmp_path / "authoritative_suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "evaluation_results"
    result_dir.mkdir()
    expected = [
        {
            "id": "deepx_m1_full",
            "source_run_id": "deepx_m1_full",
            "run_id": "deepx_m1_full",
            "dispatch_run_id": "deepx_m1_full",
            "setup_id": SETUP_ID,
            "backend": "deepx_m1",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_deepx_m1_full",
            "source_run_id": "native_full_tensorrt",
            "run_id": "native_full_tensorrt",
            "dispatch_run_id": "ort_tensorrt",
            "setup_id": SETUP_ID,
            "backend": "tensorrt",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]
    result = _run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(remote_working_dir=str(tmp_path / "downloads")),
        profile_payload={},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        gates={
            "execution_scope": "full_only",
            "hardware_run_id": "deepx_m1_full,ort_tensorrt",
            "hardware_run_ids": ["deepx_m1_full", "ort_tensorrt"],
            "quality_only_run_ids": ["deepx_m1_full", "ort_tensorrt"],
            "expected_full_quality_identities": expected,
        },
        log=None,
        runtime_override={
            "enabled": True,
            "host": "deepx",
            "user": "nx",
            "setup_id": SETUP_ID,
        },
        target_id=SETUP_ID,
        model_task="classification",
    )

    assert result.status == "ok", result.message
    assert result.metrics["canonical_result_rows_copied"] == 0
    assert result.metrics["quality_evidence_count"] == 2
    assert result.metrics["quality_evidence_status"] == "verified_exact"
    status = json.loads(
        (result_dir / f"remote_benchmark_status_{SETUP_ID}.json").read_text(
            encoding="utf-8",
        )
    )
    materialization = status["deepx_prepared_input_materialization"]
    assert materialization["status"] == "not_applicable"
    assert materialization["reason"] == (
        "full_only_quality_dispatch_has_no_performance_rows"
    )
    assert materialization["performance_claims_emitted"] is False
    assert status["full_only_quality_evidence"]["status"] == "verified_exact"
    assert status["full_only_quality_evidence"]["expected_count"] == 2
    stderr = result_dir / f"remote_benchmark_stderr_{SETUP_ID}.txt"
    assert stderr.is_file()
    assert stderr.read_bytes() == b""
    assert stderr in result.artifacts.values()
    assert not (suite / "results/deepx_m1_full/prepared_input").exists()


def test_deepx_full_only_rejects_cpu_producer_and_cleans_stale_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _suite_module()
    root, dxnn = _prepare_root(tmp_path)
    results = root / "results/deepx_m1_full"
    results.mkdir(parents=True, exist_ok=True)
    for name in (
        "deepx_prepared_feed_benchmark.json",
        "run_model_stdout.txt",
        "run_model_stderr.txt",
    ):
        (results / name).write_text("stale", encoding="utf-8")
    root_result = root / "benchmark_results_deepx_m1_full_balanced.json"
    root_result.write_text("[]", encoding="utf-8")

    candidate = results / "task_quality_inputs/full_candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}", encoding="utf-8")
    stale_request = candidate.with_name("full_request.json")
    stale_request.write_text('{"stale": true}', encoding="utf-8")
    stale_report = results / "full_quality_evidence_report.json"
    stale_report.write_text('{"stale": true}', encoding="utf-8")
    forged_request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "variant": "full",
        "record_count": 1,
        "performance_claims_emitted": False,
        "runtime_precision_identity": "float32",
        "producer_identity": {
            "backend": "cpu",
            "runtime_precision_identity": "float32",
        },
        "candidate": {
            "path": candidate.name,
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "size_bytes": candidate.stat().st_size,
        },
    }
    monkeypatch.setitem(
        module._run_declared_full_quality_only_endpoint.__globals__,
        "_run_case",
        lambda *_args, **_kwargs: pytest.fail(
            "CPU/generic _run_case fallback was attempted"
        ),
    )
    monkeypatch.setitem(
        module._run_declared_full_quality_only_endpoint.__globals__,
        "_run_deepx_full_run",
        lambda *_args, **_kwargs: [{
            "dxnn_path": str(dxnn),
            "task_quality_input_requests_by_variant": {
                "full": forged_request,
            },
        }],
    )

    report = module._run_declared_full_quality_only_endpoint(
        root=root,
        bench=_bench(),
        plan={},
        run=_full_only_run(),
        run_cases=[{"case_id": "b001", "case_dir": "b001", "boundary": 1}],
        args=_args(),
    )

    assert report is None
    assert not root_result.exists()
    for name in (
        "deepx_prepared_feed_benchmark.json",
        "run_model_stdout.txt",
        "run_model_stderr.txt",
    ):
        assert not (results / name).exists()
    assert not (results / "full_quality_evidence_report.json").exists()
    assert not stale_request.exists()
    assert not candidate.exists()


def test_generated_suite_main_finishes_deepx_quality_only_with_zero_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _suite_module()
    root, _dxnn = _prepare_root(tmp_path)
    semantic, prepared = _classification_semantic(root)
    suite_path = root / "benchmark_suite.py"
    suite_path.write_text("# generated suite identity\n", encoding="utf-8")
    module.__file__ = str(suite_path)
    (root / "__BENCH_JSON__").write_text(
        json.dumps({
            **_bench(),
            "cases": [{
                "case_id": "b001", "case_dir": "b001", "boundary": 1,
            }],
        }),
        encoding="utf-8",
    )
    (root / "benchmark_plan.json").write_text(json.dumps({
        "model_id": "resnet50",
        "quality_gate": _full_only_run()["task_quality_gate"],
        "runs": [_full_only_run()],
    }), encoding="utf-8")

    monkeypatch.setitem(
        module._run_declared_full_quality_only_endpoint.__globals__,
        "_run_case",
        lambda *_args, **_kwargs: pytest.fail(
            "DeepX main dispatch attempted the generic case runner"
        ),
    )
    monkeypatch.setitem(
        module._run_deepx_full_run.__globals__,
        "_run_deepx_prepared_feed_benchmark",
        lambda *_args, **_kwargs: dict(prepared),
    )
    monkeypatch.setitem(
        module._run_deepx_full_run.__globals__,
        "_run_deepx_semantic_validation",
        lambda *_args, **_kwargs: dict(semantic),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "DeepX quality-only main invoked the --use-ort diagnostic"
        ),
    )
    monkeypatch.setattr(module.sys, "argv", [
        str(suite_path),
        "--plan", "benchmark_plan.json",
        "--run-id", "deepx_m1_full",
        "--quality-only-run-ids", "deepx_m1_full",
        "--quality-evidence-eval-id", "eval-deepx-full-only",
        "--quality-evidence-setup-id", SETUP_ID,
        "--quality-evidence-model-id", "resnet50",
        "--no-csv",
        "--no-plot",
    ])

    assert module.main() == 0
    status = json.loads(
        (root / "benchmark_suite_status.json").read_text(encoding="utf-8")
    )
    assert status["any_rows"] is False
    assert status["any_quality_evidence"] is True
    assert status["quality_evidence_report_count"] == 1
    assert status["performance_claims_emitted"] is False
    assert not list(root.glob("benchmark_results_deepx_m1_full_*"))
