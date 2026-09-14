from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import onnx_splitpoint_tool.workflow.runner as runner_module
from onnx_splitpoint_tool.workflow.runner import (
    FINAL_STAGES,
    MODEL_STAGES,
    ROOT_STAGES,
    EvaluationWorkflowRunner,
)


MODELS = (
    "mobilenet_v3_large",
    "regnet_x_1_6gf",
    "yolo26m",
    "yolo11l",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record(
    run_dir: Path,
    path: Path,
    *,
    stage: str,
    model_id: str = "",
    current: bool = True,
) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "kind": "test",
        "producer_stage": stage,
        "model_id": model_id,
        "size_bytes": len(raw) if current else len(raw) + 1,
        "sha256": "sha256:" + (
            hashlib.sha256(raw).hexdigest() if current else "0" * 64
        ),
    }


def _descriptor(run_dir: Path, path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cohort_runner(
    tmp_path: Path,
    *,
    historical_stderr_omissions: bool = False,
) -> tuple[EvaluationWorkflowRunner, dict[str, list[Path]]]:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    index_rows: list[dict[str, object]] = []
    authority = {
        "effective_execution_plan.json": "resolve_profile",
        "quality_management/central_quality_summary.json": "evaluate_quality",
    }
    for logical, stage in authority.items():
        path = run_dir / logical
        _write_json(path, {"authority": logical})
        index_rows.append(_record(run_dir, path, stage=stage))

    superseded: dict[str, list[Path]] = {}
    for model_id in MODELS:
        task = (
            "classification" if model_id in MODELS[:2] else "detection"
        )
        formal_contract = (
            run_dir / "models" / model_id / "full_baselines"
            / "output_contracts.json"
        )
        suite_contract = (
            run_dir / "models" / model_id / "benchmark_set" / "legacy_suite"
            / "output_contracts.json"
        )
        contract = {
            "schema": "onnx-splitpoint/output-contracts",
            "schema_version": 1,
            "model_id": model_id,
            "task": task,
            "contracts": [{
                "model_id": model_id,
                "backend": "hailo8",
                "variant": "full",
            }],
        }
        _write_json(formal_contract, contract)
        _write_json(suite_contract, contract)
        index_rows.extend((
            _record(
                run_dir, formal_contract,
                stage="prepare_full_baselines", model_id=model_id,
                current=False,
            ),
            _record(
                run_dir, suite_contract,
                stage="build_backend_artifacts", model_id=model_id,
            ),
        ))
        full_plan = formal_contract.with_name("full_baseline_plan.json")
        _write_json(full_plan, {
            "schema": "onnx-splitpoint/full-baseline-plan",
            "schema_version": 1,
            "model_id": model_id,
            "task": task,
            "baselines": [{"backend": "hailo8", "variant": "full"}],
        })
        index_rows.append(_record(
            run_dir, full_plan,
            stage="prepare_full_baselines", model_id=model_id,
        ))

        base = run_dir / "models" / model_id / "benchmark_set"
        final_aliases = [
            base / "benchmark_set.json",
            base / "benchmark_plan.json",
            base / "legacy_suite" / "benchmark_set.json",
            base / "legacy_suite" / "benchmark_plan.json",
        ]
        for alias in final_aliases:
            _write_json(alias, {"model_id": model_id, "alias": alias.name})
            index_rows.append(_record(
                run_dir, alias,
                stage="generate_benchmark_set", model_id=model_id,
                current=False,
            ))
        superseded[model_id] = [formal_contract, *final_aliases]
        generation_log = base / "legacy_benchmark_generation.log"
        generation_log.write_bytes(b"")
        index_rows.append(_record(
            run_dir, generation_log,
            stage="generate_benchmark_set", model_id=model_id,
        ))

        for stage in MODEL_STAGES:
            stage_dir = (
                run_dir / "models" / model_id / "stages" / stage
            )
            stage_result = stage_dir / "stage_result.json"
            if stage == "prepare_full_baselines":
                artifacts = [full_plan, formal_contract]
            elif stage == "generate_benchmark_set":
                artifacts = [*final_aliases, generation_log]
            elif (
                stage == "run_benchmarks"
                and historical_stderr_omissions
                and model_id in MODELS[:2]
            ):
                results = run_dir / "models" / model_id / "benchmark_results"
                setup = "orin_nx_hailo8_01"
                status = results / f"remote_benchmark_status_{setup}.json"
                dispatch = results / f"remote_benchmark_dispatch_{setup}.json"
                stdout = results / f"remote_benchmark_stdout_{setup}.txt"
                stderr = results / f"remote_benchmark_stderr_{setup}.txt"
                matrix = results / "remote_hardware_matrix_status.json"
                normalized = results / "normalized_results.json"
                stdout.parent.mkdir(parents=True, exist_ok=True)
                stdout.write_text("successful remote dispatch\n", encoding="utf-8")
                evidence = []
                for source_run_id, backend, canary_id in (
                    ("hailo8", "hailo8", "hailo8_full"),
                    (
                        "native_full_tensorrt", "tensorrt",
                        "tensorrt_at_hailo8_full",
                    ),
                ):
                    quality_dir = (
                        results / "quality_inputs" / source_run_id
                    )
                    local_request = quality_dir / "request.json"
                    local_candidate = quality_dir / "candidate.json"
                    quality_dir.mkdir(parents=True, exist_ok=True)
                    local_candidate.write_bytes(
                        f"{model_id}:{source_run_id}:candidate".encode()
                    )
                    candidate_sha = hashlib.sha256(
                        local_candidate.read_bytes()
                    ).hexdigest()
                    _write_json(local_request, {
                        "candidate": {
                            "path": local_candidate.name,
                            "size_bytes": local_candidate.stat().st_size,
                            "sha256": candidate_sha,
                        },
                    })
                    evidence.append({
                        "model_id": model_id,
                        "source_run_id": source_run_id,
                        "setup_id": setup,
                        "backend": backend,
                        "variant": "full",
                        "execution_role": "full_quality_only",
                        "performance_claims_emitted": False,
                        "quality_canary_id": canary_id,
                        "eval_run_id": run_dir.name,
                        "request_path": f"/remote/{model_id}/{source_run_id}_request.json",
                        "request_sha256": hashlib.sha256(
                            local_request.read_bytes()
                        ).hexdigest(),
                        "candidate_path": f"/remote/{model_id}/{source_run_id}_candidate.json",
                        "candidate_sha256": candidate_sha,
                    })
                stdout_rel = stdout.relative_to(run_dir).as_posix()
                stderr_rel = stderr.relative_to(run_dir).as_posix()
                _write_json(status, {
                    "schema": "onnx-splitpoint/remote-benchmark-status",
                    "schema_version": 1,
                    "model_id": model_id,
                    "hardware_target_id": setup,
                    "status": "ok",
                    "reason": "remote_full_only_quality_evidence_verified",
                    "stdout_path": stdout_rel,
                    "stderr_path": stderr_rel,
                    "copied_result_count": 29,
                    "canonical_nonempty_row_count": 0,
                    "quality_evidence_count": 2,
                    "remote_output": {"ok": True, "status": "ok"},
                    "full_only_quality_evidence": {
                        "requested": True,
                        "status": "verified_exact",
                        "expected_count": 2,
                        "quality_evidence_count": 2,
                        "errors": [],
                        "evidence": evidence,
                    },
                })
                _write_json(dispatch, {
                    "schema": "onnx-splitpoint/remote-benchmark-dispatch",
                    "schema_version": 1,
                    "model_id": model_id,
                    "hardware_target_id": setup,
                    "status": "dispatching",
                    "run_id": f"{run_dir.name}_{model_id}_{setup}",
                    "host": {"host": "192.0.2.10"},
                })
                _write_json(matrix, {
                    "schema": "onnx-splitpoint/remote-hardware-matrix-status",
                    "schema_version": 2,
                    "model_id": model_id,
                    "status": "ok",
                    "remote_dispatched": True,
                    "remote_dispatch_failed": False,
                    "quality_evidence_count": 2,
                    "expected_full_quality_count": 2,
                })
                _write_json(normalized, {
                    "schema": "onnx-splitpoint/normalized-benchmark-results",
                    "schema_version": 2,
                    "model_id": model_id,
                    "status": "quality_evidence_only_complete",
                    "quality_evidence_only_complete": True,
                    "performance_matrix_applicable": False,
                    "matrix_complete": True,
                    "quality_evidence_count": 2,
                    "expected_full_quality_count": 2,
                    "result_count": 0,
                    "results": [],
                })
                for artifact in (status, dispatch, stdout, matrix, normalized):
                    index_rows.append(_record(
                        run_dir, artifact,
                        stage=stage, model_id=model_id,
                    ))
                artifacts = [
                    status, dispatch, stdout, stderr, matrix, normalized,
                ]
            else:
                artifact = stage_dir / "evidence.json"
                _write_json(artifact, {"model_id": model_id, "stage": stage})
                index_rows.append(_record(
                    run_dir, artifact, stage=stage, model_id=model_id,
                ))
                artifacts = [artifact]
            _write_json(stage_result, {
                "stage": stage,
                "model_id": model_id,
                "status": "ok",
                "state": "completed",
                "complete": True,
                "details": {
                    "resume_key_hash": "archived",
                    **({
                        "quality_evidence_only_complete": True,
                        "quality_evidence_count": 2,
                        "expected_full_quality_count": 2,
                    } if (
                        stage == "run_benchmarks"
                        and historical_stderr_omissions
                        and model_id in MODELS[:2]
                    ) else {}),
                },
                "artifacts": [
                    path.relative_to(run_dir).as_posix()
                    for path in [*artifacts, stage_result]
                ],
            })
            index_rows.append(_record(
                run_dir, stage_result, stage=stage, model_id=model_id,
            ))

    for stage in ROOT_STAGES + FINAL_STAGES:
        stage_dir = run_dir / "stages" / stage
        stage_result = stage_dir / "stage_result.json"
        artifact = stage_dir / "evidence.json"
        _write_json(artifact, {"stage": stage})
        index_rows.append(_record(run_dir, artifact, stage=stage))
        _write_json(stage_result, {
            "stage": stage,
            "model_id": None,
            "status": "ok",
            "state": "completed",
            "complete": True,
            "details": {"resume_key_hash": "archived"},
            "artifacts": [
                artifact.relative_to(run_dir).as_posix(),
                stage_result.relative_to(run_dir).as_posix(),
            ],
        })
        index_rows.append(_record(run_dir, stage_result, stage=stage))

    artifact_index_path = run_dir / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": run_dir.name,
        "profile_id": "phase5",
        "artifacts": index_rows,
    })
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = run_dir.name
    runner.profile_id = "phase5"
    runner.session_id = "1" * 32
    runner.artifact_index_path = artifact_index_path
    runner.artifact_index = {}
    runner.options = SimpleNamespace(
        resume=True,
        resume_missing_full_quality_only=True,
    )
    runner.profile_payload = {"quality_gate": {}}
    runner._missing_full_quality_by_model = {
        "yolo26m": [{"source_run_id": "hailo8"}],
        "yolo11l": [
            {"source_run_id": "hailo8"},
            {"source_run_id": "native_full_tensorrt"},
        ],
    }
    runner._missing_full_quality_keys = {("a",), ("b",), ("c",)}
    runner._preserved_central_quality_requests = {}
    preserved_rows = (
        ("regnet_x_1_6gf", "hailo8", "hailo8", "fail"),
        (
            "mobilenet_v3_large", "native_full_tensorrt",
            "tensorrt", "pass",
        ),
        (
            "regnet_x_1_6gf", "native_full_tensorrt",
            "tensorrt", "pass",
        ),
        (
            "yolo26m", "native_full_tensorrt",
            "tensorrt", "pass",
        ),
        ("mobilenet_v3_large", "hailo8", "hailo8", "fail"),
    )
    preserved_results = []
    for model_id, source_run_id, backend, decision in preserved_rows:
        request = (
            run_dir / "models" / model_id / "benchmark_results"
            / "quality_inputs" / source_run_id / "request.json"
        )
        candidate = request.with_name("candidate.json")
        request.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(
            f"{model_id}:{source_run_id}:candidate".encode()
        )
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        _write_json(request, {
            "candidate": {
                "path": candidate.name,
                "size_bytes": candidate.stat().st_size,
                "sha256": candidate_sha,
            },
        })
        request_sha = hashlib.sha256(request.read_bytes()).hexdigest()
        preserved_result = {
            "status": "completed",
            "technical_status": "completed",
            "decision": decision,
            "model_id": model_id,
            "case_id": "full",
            "source_run_id": source_run_id,
            "setup_id": "orin_nx_hailo8_01",
            "backend": backend,
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
            "source_request": request.relative_to(run_dir).as_posix(),
            "source_request_sha256": request_sha,
        }
        preserved_results.append(preserved_result)
        runner._preserved_central_quality_requests[str(request.resolve())] = {
            "sha256": request_sha,
            "result_sha256": runner_module.sha256_payload(preserved_result),
            "candidate": {
                "path": str(candidate.resolve()),
                "sha256": candidate_sha,
                "size_bytes": candidate.stat().st_size,
            },
        }
    runner._preserved_central_quality_results = preserved_results
    runner._missing_full_quality_reuse_attestation = {}
    runner._missing_full_quality_committed_omissions = {}
    runner.outputs = {}
    return runner, superseded


def test_cohort_attestation_is_read_only_then_commits_only_current_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(tmp_path)

    def fake_prepare(**kwargs):
        model_id = str(kwargs["model_id"])
        formal = superseded[model_id][0]
        return {
            "ok": True,
            "task": runner._missing_full_quality_model_task(model_id),
            "formal_contract": _descriptor(runner.run_dir, formal),
        }

    def fake_generate(**kwargs):
        model_id = str(kwargs["model_id"])
        return {
            "ok": True,
            "finalized_aliases": {
                f"alias_{index}": _descriptor(runner.run_dir, path)
                for index, path in enumerate(superseded[model_id][1:])
            },
        }

    monkeypatch.setattr(
        runner_module, "attest_prepare_full_baselines_supersession",
        fake_prepare,
    )
    monkeypatch.setattr(
        runner_module, "attest_generate_benchmark_set_supersession",
        fake_generate,
    )
    before_index = runner.artifact_index_path.read_bytes()
    before_scientific = {
        path: path.read_bytes()
        for paths in superseded.values() for path in paths
    }

    runner._prepare_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index
    assert {
        path: path.read_bytes()
        for path in before_scientific
    } == before_scientific
    assert len(
        runner._missing_full_quality_reuse_attestation["rebound_paths"]
    ) == 20

    runner._commit_missing_full_quality_reuse_attestation()

    assert (
        runner.run_dir / "jobs"
        / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    ).is_file()
    for model_id, paths in superseded.items():
        for path in paths:
            logical = path.relative_to(runner.run_dir).as_posix()
            assert any(
                row.get("path") == logical
                and row.get("producer_stage")
                == "missing_full_quality_reuse_attestation"
                and row.get("model_id") == model_id
                for row in runner.artifact_index["artifacts"]
            )
    assert {
        path: path.read_bytes()
        for path in before_scientific
    } == before_scientific
    committed_index = runner.artifact_index_path.read_bytes()
    runner._commit_missing_full_quality_reuse_attestation()
    assert runner.artifact_index_path.read_bytes() == committed_index


def test_cohort_preflight_aggregates_unrelated_tamper_without_committing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(tmp_path)
    monkeypatch.setattr(
        runner_module, "attest_prepare_full_baselines_supersession",
        lambda **kwargs: {
            "ok": True,
            "task": runner._missing_full_quality_model_task(kwargs["model_id"]),
            "formal_contract": _descriptor(
                runner.run_dir, superseded[kwargs["model_id"]][0],
            ),
        },
    )
    monkeypatch.setattr(
        runner_module, "attest_generate_benchmark_set_supersession",
        lambda **kwargs: {
            "ok": True,
            "finalized_aliases": {
                str(index): _descriptor(runner.run_dir, path)
                for index, path in enumerate(
                    superseded[kwargs["model_id"]][1:]
                )
            },
        },
    )
    tampered = (
        runner.run_dir / "models" / "regnet_x_1_6gf" / "stages"
        / "analyze_model" / "evidence.json"
    )
    tampered.write_text("tampered\n", encoding="utf-8")
    before_index = runner.artifact_index_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="stage_artifacts:regnet_x_1_6gf:analyze_model",
    ):
        runner._prepare_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index
    assert not (
        runner.run_dir / "jobs"
        / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    ).exists()


def test_second_resume_keeps_first_attestation_and_upserts_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(tmp_path)

    def fake_prepare(**kwargs):
        model_id = str(kwargs["model_id"])
        formal = superseded[model_id][0]
        return {
            "ok": True,
            "task": runner._missing_full_quality_model_task(model_id),
            "formal_contract": _descriptor(runner.run_dir, formal),
        }

    def fake_generate(**kwargs):
        model_id = str(kwargs["model_id"])
        return {
            "ok": True,
            "finalized_aliases": {
                f"alias_{index}": _descriptor(runner.run_dir, path)
                for index, path in enumerate(superseded[model_id][1:])
            },
        }

    monkeypatch.setattr(
        runner_module, "attest_prepare_full_baselines_supersession",
        fake_prepare,
    )
    monkeypatch.setattr(
        runner_module, "attest_generate_benchmark_set_supersession",
        fake_generate,
    )

    first_session = runner.session_id
    runner._prepare_missing_full_quality_reuse_attestation()
    runner._commit_missing_full_quality_reuse_attestation()
    first_index = json.loads(
        runner.artifact_index_path.read_text(encoding="utf-8")
    )
    first_attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{first_session}.json"
    )
    first_bytes = first_attestation.read_bytes()
    first_session_rows = [
        dict(row) for row in first_index["artifacts"]
        if isinstance(row, dict)
        and row.get("binding_session_id") == first_session
    ]
    assert len(first_session_rows) == 21

    # On the retry, the first session's rebound rows make every must-reuse
    # artifact current.  The semantic supersession helpers must not be needed
    # again, but this invocation still commits its own durable attestation.
    runner.session_id = "2" * 32
    runner._missing_full_quality_reuse_attestation = {}
    monkeypatch.setattr(
        runner_module,
        "attest_prepare_full_baselines_supersession",
        lambda **_kwargs: pytest.fail("unexpected prepare supersession"),
    )
    monkeypatch.setattr(
        runner_module,
        "attest_generate_benchmark_set_supersession",
        lambda **_kwargs: pytest.fail("unexpected generate supersession"),
    )

    runner._prepare_missing_full_quality_reuse_attestation()
    assert runner._missing_full_quality_reuse_attestation["rebound_paths"] == []
    runner._commit_missing_full_quality_reuse_attestation()

    second_attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    )
    final_index = json.loads(
        runner.artifact_index_path.read_text(encoding="utf-8")
    )
    assert first_attestation.read_bytes() == first_bytes
    assert second_attestation.is_file()
    assert all(row in final_index["artifacts"] for row in first_session_rows)
    second_rows = [
        row for row in final_index["artifacts"]
        if isinstance(row, dict)
        and row.get("binding_session_id") == runner.session_id
    ]
    assert len(second_rows) == 1
    assert second_rows[0]["path"] == second_attestation.relative_to(
        runner.run_dir,
    ).as_posix()


def test_commit_rechecks_rebound_bytes_before_any_durable_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "attest_prepare_full_baselines_supersession",
        lambda **kwargs: {
            "ok": True,
            "task": runner._missing_full_quality_model_task(kwargs["model_id"]),
            "formal_contract": _descriptor(
                runner.run_dir, superseded[kwargs["model_id"]][0],
            ),
        },
    )
    monkeypatch.setattr(
        runner_module,
        "attest_generate_benchmark_set_supersession",
        lambda **kwargs: {
            "ok": True,
            "finalized_aliases": {
                str(index): _descriptor(runner.run_dir, path)
                for index, path in enumerate(
                    superseded[kwargs["model_id"]][1:]
                )
            },
        },
    )

    runner._prepare_missing_full_quality_reuse_attestation()
    before_index = runner.artifact_index_path.read_bytes()
    superseded["mobilenet_v3_large"][0].write_text(
        "changed after preflight\n", encoding="utf-8",
    )
    attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    )

    with pytest.raises(
        RuntimeError,
        match="missing_full_quality_rebound_changed_after_attestation",
    ):
        runner._commit_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index
    assert not attestation.exists()


def test_commit_rejects_index_change_after_snapshot_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "attest_prepare_full_baselines_supersession",
        lambda **kwargs: {
            "ok": True,
            "task": runner._missing_full_quality_model_task(kwargs["model_id"]),
            "formal_contract": _descriptor(
                runner.run_dir, superseded[kwargs["model_id"]][0],
            ),
        },
    )
    monkeypatch.setattr(
        runner_module,
        "attest_generate_benchmark_set_supersession",
        lambda **kwargs: {
            "ok": True,
            "finalized_aliases": {
                str(index): _descriptor(runner.run_dir, path)
                for index, path in enumerate(superseded[kwargs["model_id"]][1:])
            },
        },
    )

    runner._prepare_missing_full_quality_reuse_attestation()
    changed = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    changed["artifacts"].append(dict(changed["artifacts"][0]))
    _write_json(runner.artifact_index_path, changed)
    changed_bytes = runner.artifact_index_path.read_bytes()
    attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    )

    with pytest.raises(
        RuntimeError,
        match="missing_full_quality_artifact_index_changed_after_attestation",
    ):
        runner._commit_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == changed_bytes
    assert not attestation.exists()


def test_must_reuse_stage_result_requires_exact_self_artifact_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "attest_prepare_full_baselines_supersession",
        lambda **kwargs: {
            "ok": True,
            "task": runner._missing_full_quality_model_task(kwargs["model_id"]),
            "formal_contract": _descriptor(
                runner.run_dir, superseded[kwargs["model_id"]][0],
            ),
        },
    )
    monkeypatch.setattr(
        runner_module,
        "attest_generate_benchmark_set_supersession",
        lambda **kwargs: {
            "ok": True,
            "finalized_aliases": {
                str(index): _descriptor(runner.run_dir, path)
                for index, path in enumerate(superseded[kwargs["model_id"]][1:])
            },
        },
    )
    stage_result = (
        runner.run_dir / "models" / "regnet_x_1_6gf" / "stages"
        / "analyze_model" / "stage_result.json"
    )
    payload = json.loads(stage_result.read_text(encoding="utf-8"))
    self_path = stage_result.relative_to(runner.run_dir).as_posix()
    payload["artifacts"] = [
        value for value in payload["artifacts"] if value != self_path
    ]
    _write_json(stage_result, payload)
    index = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    replacement = _record(
        runner.run_dir,
        stage_result,
        stage="analyze_model",
        model_id="regnet_x_1_6gf",
    )
    for position, row in enumerate(index["artifacts"]):
        if row.get("path") == self_path:
            index["artifacts"][position] = replacement
            break
    _write_json(runner.artifact_index_path, index)

    with pytest.raises(
        ValueError,
        match="stage_lifecycle:regnet_x_1_6gf:analyze_model",
    ):
        runner._prepare_missing_full_quality_reuse_attestation()


def _install_supersession_stubs(
    runner: EvaluationWorkflowRunner,
    superseded: dict[str, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "attest_prepare_full_baselines_supersession",
        lambda **kwargs: {
            "ok": True,
            "task": runner._missing_full_quality_model_task(kwargs["model_id"]),
            "formal_contract": _descriptor(
                runner.run_dir, superseded[kwargs["model_id"]][0],
            ),
        },
    )
    monkeypatch.setattr(
        runner_module,
        "attest_generate_benchmark_set_supersession",
        lambda **kwargs: {
            "ok": True,
            "finalized_aliases": {
                str(index): _descriptor(runner.run_dir, path)
                for index, path in enumerate(
                    superseded[kwargs["model_id"]][1:]
                )
            },
        },
    )


def test_historical_success_stderr_omissions_are_attested_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    before_index = runner.artifact_index_path.read_bytes()

    runner._prepare_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index
    omissions = runner._missing_full_quality_reuse_attestation[
        "historical_omissions"
    ]
    assert {
        (row["model_id"], row["path"]) for row in omissions
    } == {
        (
            model_id,
            f"models/{model_id}/benchmark_results/"
            "remote_benchmark_stderr_orin_nx_hailo8_01.txt",
        )
        for model_id in MODELS[:2]
    }

    runner._commit_missing_full_quality_reuse_attestation()

    assert set(runner._missing_full_quality_committed_omissions) == {
        row["path"] for row in omissions
    }
    for row in omissions:
        logical = row["path"]
        assert not (runner.run_dir / logical).exists()
        assert not any(
            item.get("path") == logical
            for item in runner.artifact_index["artifacts"]
        )

    model_id = "mobilenet_v3_large"
    result_path = (
        runner.run_dir / "models" / model_id / "stages"
        / "run_benchmarks" / "stage_result.json"
    )
    previous = json.loads(result_path.read_text(encoding="utf-8"))
    decision = runner._resume_reuse_decision(
        model_id=model_id,
        stage="run_benchmarks",
        previous=previous,
        expected_hash="current-build-hash",
        forced=False,
        stage_job_id="",
        result_path=result_path,
    )
    assert decision["reusable"] is True
    assert decision["artifacts_complete"] is True
    assert decision["missing_artifacts"] == []
    assert decision["reason"] == (
        "missing_full_quality_reuse_existing_stage_ignore_hash"
    )


def test_historical_stderr_created_after_preflight_blocks_commit_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    runner._prepare_missing_full_quality_reuse_attestation()
    before_index = runner.artifact_index_path.read_bytes()
    stderr = (
        runner.run_dir / "models" / "mobilenet_v3_large"
        / "benchmark_results"
        / "remote_benchmark_stderr_orin_nx_hailo8_01.txt"
    )
    stderr.write_text("appeared after preflight\n", encoding="utf-8")
    attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    )

    with pytest.raises(
        RuntimeError,
        match="historical_omission_changed_after_attestation",
    ):
        runner._commit_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index
    assert not attestation.exists()


def test_historical_stderr_created_inside_commit_window_blocks_all_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    runner._prepare_missing_full_quality_reuse_attestation()
    before_index = runner.artifact_index_path.read_bytes()
    stderr = (
        runner.run_dir / "models" / "mobilenet_v3_large"
        / "benchmark_results"
        / "remote_benchmark_stderr_orin_nx_hailo8_01.txt"
    )
    original = runner_module.sha256_file
    injected = False
    index_hash_calls = 0

    def hash_and_inject(path):
        nonlocal injected, index_hash_calls
        result = original(path)
        if Path(path) == runner.artifact_index_path:
            index_hash_calls += 1
        if not injected and index_hash_calls == 3:
            injected = True
            stderr.write_text("appeared inside commit window\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        runner_module, "sha256_file", hash_and_inject,
    )
    attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    )

    with pytest.raises(
        RuntimeError,
        match="historical_omission_changed_after_attestation",
    ):
        runner._commit_missing_full_quality_reuse_attestation()

    assert injected is True
    assert runner.artifact_index_path.read_bytes() == before_index
    assert not attestation.exists()


def test_preserved_quality_input_changed_after_preflight_blocks_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    runner._prepare_missing_full_quality_reuse_attestation()
    before_index = runner.artifact_index_path.read_bytes()
    source_request = (
        runner.run_dir
        / runner._preserved_central_quality_results[0]["source_request"]
    )
    payload = json.loads(source_request.read_text(encoding="utf-8"))
    payload["changed_after_preflight"] = True
    _write_json(source_request, payload)
    attestation = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{runner.session_id}.json"
    )

    with pytest.raises(
        ValueError,
        match="missing_full_quality_resume_preserved_input_drift",
    ):
        runner._commit_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index
    assert not attestation.exists()


def test_second_resume_reattests_historical_absence_without_fake_index_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    first_session = runner.session_id
    runner._prepare_missing_full_quality_reuse_attestation()
    runner._commit_missing_full_quality_reuse_attestation()

    runner.session_id = "2" * 32
    runner._missing_full_quality_reuse_attestation = {}
    runner._prepare_missing_full_quality_reuse_attestation()
    omissions = runner._missing_full_quality_reuse_attestation[
        "historical_omissions"
    ]
    assert len(omissions) == 2
    runner._commit_missing_full_quality_reuse_attestation()

    first_receipt = (
        runner.run_dir / "jobs" / "missing_full_quality_reuse_attestations"
        / f"{first_session}.json"
    )
    second_receipt = first_receipt.with_name(f"{runner.session_id}.json")
    assert first_receipt.is_file()
    assert second_receipt.is_file()
    omission_paths = {row["path"] for row in omissions}
    assert not any(
        row.get("path") in omission_paths
        for row in runner.artifact_index["artifacts"]
    )


def test_historical_stderr_with_any_index_role_is_not_an_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    index = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    logical = (
        "models/mobilenet_v3_large/benchmark_results/"
        "remote_benchmark_stderr_orin_nx_hailo8_01.txt"
    )
    index["artifacts"].append({
        "path": logical,
        "kind": "stage_artifact",
        "producer_stage": "run_benchmarks",
        "model_id": "mobilenet_v3_large",
        "size_bytes": 0,
        "sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
    })
    _write_json(runner.artifact_index_path, index)
    before_index = runner.artifact_index_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="historical_stderr_has_index_record",
    ):
        runner._prepare_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index


def test_historical_stderr_dangling_symlink_is_not_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    stderr = (
        runner.run_dir / "models" / "mobilenet_v3_large"
        / "benchmark_results"
        / "remote_benchmark_stderr_orin_nx_hailo8_01.txt"
    )
    stderr.symlink_to(stderr.with_name("missing-target.txt"))
    before_index = runner.artifact_index_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="historical_omission_path_not_absent",
    ):
        runner._prepare_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index


def test_historical_stderr_requires_semantic_remote_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, superseded = _cohort_runner(
        tmp_path, historical_stderr_omissions=True,
    )
    _install_supersession_stubs(runner, superseded, monkeypatch)
    model_id = "mobilenet_v3_large"
    status = (
        runner.run_dir / "models" / model_id / "benchmark_results"
        / "remote_benchmark_status_orin_nx_hailo8_01.json"
    )
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["reason"] = "remote_service_partial_or_failed"
    _write_json(status, payload)
    index = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    logical = status.relative_to(runner.run_dir).as_posix()
    replacement = _record(
        runner.run_dir, status,
        stage="run_benchmarks", model_id=model_id,
    )
    for position, row in enumerate(index["artifacts"]):
        if row.get("path") == logical:
            index["artifacts"][position] = replacement
            break
    _write_json(runner.artifact_index_path, index)
    before_index = runner.artifact_index_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="historical_stderr_remote_success_invalid",
    ):
        runner._prepare_missing_full_quality_reuse_attestation()

    assert runner.artifact_index_path.read_bytes() == before_index


def test_stage_completeness_accepts_exactly_bound_empty_regular_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "empty.log"
    artifact.parent.mkdir()
    artifact.write_bytes(b"")
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.artifact_index = {
        "artifacts": [{
            "path": "empty.log",
            "size_bytes": 0,
            "sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
        }],
    }

    complete, missing = runner._stage_result_artifacts_complete({
        "artifacts": ["empty.log"],
    })

    assert complete is True
    assert missing == []


@pytest.mark.parametrize("invalid_size", [None, False, 0.0, "0", -1, 1])
def test_empty_artifact_rejects_noncanonical_or_wrong_index_size(
    tmp_path: Path, invalid_size: object,
) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "empty.log"
    artifact.parent.mkdir()
    artifact.write_bytes(b"")
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.artifact_index = {
        "artifacts": [{
            "path": "empty.log",
            "size_bytes": invalid_size,
            "sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
        }],
    }

    complete, missing = runner._stage_result_artifacts_complete({
        "artifacts": ["empty.log"],
    })

    assert complete is False
    assert missing == ["unbound_or_tampered:empty.log"]


def test_empty_artifact_still_requires_exact_sha256(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "empty.log"
    artifact.parent.mkdir()
    artifact.write_bytes(b"")
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.artifact_index = {
        "artifacts": [{
            "path": "empty.log",
            "size_bytes": 0,
            "sha256": "sha256:" + "0" * 64,
        }],
    }

    complete, missing = runner._stage_result_artifacts_complete({
        "artifacts": ["empty.log"],
    })

    assert complete is False
    assert missing == ["unbound_or_tampered:empty.log"]
