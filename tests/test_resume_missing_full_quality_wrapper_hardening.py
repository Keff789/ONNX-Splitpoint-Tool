from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_wrapper():
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "resume_missing_full_quality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "resume_missing_full_quality_wrapper_hardening", script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPER = _load_wrapper()


RESULT_IDENTITIES = [
    ("mobilenet_v3_large", "hailo8"),
    ("mobilenet_v3_large", "native_full_tensorrt"),
    ("regnet_x_1_6gf", "hailo8"),
    ("regnet_x_1_6gf", "native_full_tensorrt"),
    ("yolo26m", "native_full_tensorrt"),
    ("yolo26m", "hailo8"),
    ("yolo11l", "hailo8"),
    ("yolo11l", "native_full_tensorrt"),
]


def _write_summary(run_dir: Path, completed: int, *, exact: bool) -> None:
    quality_dir = run_dir / "quality_management"
    quality_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "quality_decision": "fail",
        "technical_status": "completed" if exact else "partial",
        "quality_acceptance_identity_contract": {
            "postcondition": {
                "status": "verified_exact" if exact else "incomplete",
                "expected_count": 8,
                "completed_count": completed,
                "failed_count": 0,
                "missing_count": 8 - completed,
                "duplicate_count": 0,
                "contract_error_count": 0,
            },
        },
        "results": [
            {
                "id": f"result-{index}",
                "status": "completed",
                "technical_status": "completed",
                # Match the archived 5/8 run: MobileNet/Hailo and
                # RegNet/Hailo are the two genuine Quality FAILs.
                "decision": "fail" if index in {0, 2} else "pass",
                "model_id": RESULT_IDENTITIES[index][0],
                "source_run_id": RESULT_IDENTITIES[index][1],
            }
            for index in range(completed)
        ],
    }
    (quality_dir / "central_quality_summary.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _reused(stage: str, model_id: str = "") -> dict[str, object]:
    return {
        "stage": stage,
        "model_id": model_id,
        "status": "ok",
        "skip_reason": "resume_reused_existing_stage_result",
        "details": {
            "resume_decision": {
                "reusable": True,
                "reason": "missing_full_quality_reuse_existing_stage_ignore_hash",
            },
        },
    }


def _targeted_rebuild(stage: str, model_id: str = "") -> dict[str, object]:
    return {
        "stage": stage,
        "model_id": model_id,
        "status": "ok",
        "skip_reason": "",
        "details": {
            "resume_decision": {
                "reusable": False,
                "reason": "missing_full_quality_targeted_rebuild",
                "missing_full_quality_targeted_rebuild": True,
            },
        },
    }


def _valid_stage_results(
    target_models: set[str] | None = None,
) -> list[dict[str, object]]:
    targets = target_models or {"yolo26m", "yolo11l"}
    results = [
        _reused("resolve_profile"),
        _reused("prepare_full_baselines", "mobilenet_v3_large"),
        _reused("build_backend_artifacts", "yolo26m"),
        _targeted_rebuild("evaluate_quality"),
        _targeted_rebuild("aggregate_results"),
        _reused("run_native_producers"),
        _targeted_rebuild("generate_report"),
    ]
    for model_id in sorted(targets):
        results.extend([
            _targeted_rebuild("run_benchmarks", model_id),
            _targeted_rebuild("validate_outputs", model_id),
            _targeted_rebuild("hardware_smoke", model_id),
        ])
    return results


def _write_json(path: Path, payload: object) -> bytes:
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _artifact_row(
    logical: str,
    raw: bytes,
    *,
    kind: str,
    producer_stage: str,
    session_id: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "path": logical,
        "kind": kind,
        "producer_stage": producer_stage,
        "model_id": None,
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "created_at": "2026-08-26T15:32:00+02:00",
    }
    if session_id:
        row["binding_session_id"] = session_id
    return row


def _decision_counts(decisions: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        key = (
            "reused" if decision.get("reusable") is True
            else str(decision.get("reason") or "not_reused")
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_completed_resume_archive(
    run_dir: Path,
    *,
    legacy_v27711: bool,
) -> None:
    run_id = run_dir.name
    profile_id = "phase5-profile"
    session_id = "a" * 32
    workflow_version = (
        WRAPPER._LEGACY_V27711_WORKFLOW_VERSION
        if legacy_v27711 else WRAPPER.WORKFLOW_VERSION
    )
    tool_version = "2.77.11" if legacy_v27711 else WRAPPER.TOOL_VERSION
    # Match the live 5/8 repair cohort exactly: one YOLO26 endpoint and two
    # YOLO11 endpoints were missing when the sealed retry session started.
    target_models = ["yolo11l", "yolo26m"]
    target_keys = {
        ("", stage) for stage in WRAPPER._ALLOWED_ROOT_REBUILD_STAGES
    } | {
        (model_id, stage)
        for model_id in target_models
        for stage in WRAPPER._ALLOWED_MODEL_REBUILD_STAGES
    }
    legacy_keys = (
        WRAPPER._LEGACY_V27711_FAILED_TARGET_KEYS
        if legacy_v27711 else frozenset()
    )
    decisions: list[dict[str, object]] = []
    keys = [
        ("", stage) for stage in WRAPPER._ROOT_STAGE_SURFACE
    ] + [
        (model_id, stage)
        for model_id in WRAPPER._PHASE5_MODELS
        for stage in WRAPPER._MODEL_STAGE_SURFACE
    ]
    for model_id, stage in keys:
        key = (model_id, stage)
        targeted = key in target_keys
        legacy = key in legacy_keys
        decision: dict[str, object] = {
            "schema": "onnx-splitpoint/evaluation-stage-resume-decision",
            "schema_version": 1,
            "created_at": "2026-08-26T15:30:00+02:00",
            "run_id": run_id,
            "profile_id": profile_id,
            "model_id": model_id,
            "stage": stage,
            "job_id": f"stage:{model_id or 'workflow'}:{stage}",
            "stage_result_path": (
                f"models/{model_id}/stages/{stage}/stage_result.json"
                if model_id else f"stages/{stage}/stage_result.json"
            ),
            "resume_requested": True,
            "force_stage": False,
            "previous_status": "failed" if legacy else "ok",
            "artifacts_complete": not legacy,
            "missing_artifacts": [],
            "reusable": not targeted,
        }
        if legacy:
            decision["reason"] = "previous_stage_state_not_reusable:failed"
        elif targeted:
            decision.update({
                "reason": "missing_full_quality_targeted_rebuild",
                "missing_full_quality_targeted_rebuild": True,
            })
        else:
            decision.update({
                "reason": "missing_full_quality_reuse_existing_stage_ignore_hash",
                "missing_full_quality_reuse": True,
            })
        decisions.append(decision)

    _write_summary(run_dir, 8, exact=True)
    quality_path = (
        run_dir / "quality_management" / "central_quality_summary.json"
    )
    quality_raw = quality_path.read_bytes()
    attestation_logical = (
        "jobs/missing_full_quality_reuse_attestations/" f"{session_id}.json"
    )
    attestation = {
        "schema": (
            "onnx-splitpoint/missing-full-quality-reuse-cohort-attestation"
        ),
        "schema_version": 1,
        "status": "verified",
        "created_at": "2026-08-26T15:28:00+02:00",
        "run_id": run_id,
        "session_id": session_id,
        "workflow_version": workflow_version,
        "archived_artifact_index_sha256": "sha256:" + "1" * 64,
        "archived_artifact_index_payload_sha256": "sha256:" + "2" * 64,
        "must_reuse_stage_count": 41,
        "targeted_rebuild_models": target_models,
        "preserved_quality_result_count": 5,
        "remaining_quality_result_count": 3,
        "supersessions": [],
        "historical_omissions": [],
        "rebound_paths": [],
    }
    attestation_raw = _write_json(run_dir / attestation_logical, attestation)
    resume_logical = "jobs/resume_summary.json"
    resume_summary = {
        "schema": "onnx-splitpoint/evaluation-resume-summary",
        "schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "tool_version": tool_version,
        "workflow_version": workflow_version,
        "created_at": "2026-08-26T15:32:00+02:00",
        "decision_count": len(decisions),
        "counts": _decision_counts(decisions),
        "decisions": decisions,
    }
    resume_raw = _write_json(run_dir / resume_logical, resume_summary)
    artifact_index = {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "artifacts": [
            _artifact_row(
                "quality_management/central_quality_summary.json",
                quality_raw,
                kind="stage_artifact",
                producer_stage="evaluate_quality",
            ),
            _artifact_row(
                attestation_logical,
                attestation_raw,
                kind="resume_attestation",
                producer_stage="missing_full_quality_reuse_attestation",
                session_id=session_id,
            ),
            _artifact_row(
                resume_logical,
                resume_raw,
                kind="job_artifact",
                producer_stage="workflow_resume",
            ),
        ],
    }
    _write_json(run_dir / "artifact_index.json", artifact_index)
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "status": "partial",
        "current_session_id": session_id,
        "current_workflow_version": workflow_version,
        "current_tool_version": tool_version,
        "resume_summary": resume_logical,
        "execution_sessions": [{
            "schema": "onnx-splitpoint/evaluation-execution-session",
            "schema_version": 1,
            "session_index": 1,
            "session_id": session_id,
            "started_at": "2026-08-26T15:29:00+02:00",
            "last_updated_at": "2026-08-26T15:33:00+02:00",
            "finished_at": "2026-08-26T15:33:00+02:00",
            "status": "partial",
            "resume_requested": True,
            "resumed_existing_manifest": True,
            "workflow_version": workflow_version,
            "tool_version": tool_version,
        }],
    }
    _write_json(run_dir / "run_manifest.json", manifest)


def _rewrite_resume_summary_with_binding(
    run_dir: Path,
    payload: dict[str, object],
) -> None:
    raw = _write_json(run_dir / "jobs" / "resume_summary.json", payload)
    index_path = run_dir / "artifact_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    row = next(
        item for item in index["artifacts"]
        if item.get("path") == "jobs/resume_summary.json"
        and item.get("producer_stage") == "workflow_resume"
    )
    row["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    row["size_bytes"] = len(raw)
    _write_json(index_path, index)


@pytest.mark.parametrize(
    ("preserved", "repaired"), [(5, 3), (6, 2), (7, 1)],
)
def test_counts_are_derived_from_the_initial_five_to_seven_state(
    tmp_path: Path, preserved: int, repaired: int,
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, preserved, exact=False)

    initial_scope = WRAPPER._initial_resume_scope(run_dir)
    _write_summary(run_dir, 8, exact=True)
    targets = set(initial_scope["allowed_model_rebuilds"])
    payload = WRAPPER._verify_resume_outcome(
        run_dir,
        SimpleNamespace(
            status="partial",
            completed=True,
            stage_results=_valid_stage_results(targets),
        ),
        preserved_results=initial_scope["preserved_results"],
        allowed_model_rebuilds=sorted(targets),
    )

    assert payload["status"] == "PASS"
    assert payload["preserved_results"] == preserved
    assert payload["repaired_results"] == repaired
    assert payload["workflow_status"] == "partial"
    assert payload["must_reuse_stage_rebuilds"] == 0


@pytest.mark.parametrize("count", [4, 8])
def test_initial_state_outside_five_to_seven_is_blocked(
    tmp_path: Path, count: int,
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, count, exact=(count == 8))

    with pytest.raises(RuntimeError, match="exactly 5, 6, or 7"):
        WRAPPER._initial_preserved_result_count(run_dir)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_or_cancelled_workflow_cannot_be_masked_by_exact_summary(
    tmp_path: Path, status: str,
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, 8, exact=True)

    with pytest.raises(RuntimeError, match="did not complete successfully"):
        WRAPPER._verify_resume_outcome(
            run_dir,
            SimpleNamespace(
                status=status,
                completed=False,
                stage_results=_valid_stage_results(),
            ),
            preserved_results=5,
            allowed_model_rebuilds=["yolo26m", "yolo11l"],
        )


@pytest.mark.parametrize(
    "rebuilt",
    [
        _targeted_rebuild("prepare_full_baselines", "mobilenet_v3_large"),
        _targeted_rebuild("generate_benchmark_set", "yolo26m"),
        _targeted_rebuild("build_backend_artifacts", "yolo11l"),
        _targeted_rebuild("run_native_producers"),
    ],
)
def test_must_reuse_stage_rebuild_is_blocked_even_with_targeted_flag(
    tmp_path: Path, rebuilt: dict[str, object],
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, 8, exact=True)

    with pytest.raises(RuntimeError, match="must-reuse stage rebuild detected"):
        WRAPPER._verify_resume_outcome(
            run_dir,
            SimpleNamespace(
                status="partial",
                completed=True,
                stage_results=[_reused("resolve_profile"), rebuilt],
            ),
            preserved_results=5,
            allowed_model_rebuilds=["yolo26m", "yolo11l"],
        )


def test_already_repaired_model_is_must_reuse_on_later_resume(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, 6, exact=False)
    initial_scope = WRAPPER._initial_resume_scope(run_dir)
    assert initial_scope["allowed_model_rebuilds"] == ["yolo11l"]
    _write_summary(run_dir, 8, exact=True)

    with pytest.raises(RuntimeError, match="yolo26m/run_benchmarks"):
        WRAPPER._verify_resume_outcome(
            run_dir,
            SimpleNamespace(
                status="partial",
                completed=True,
                stage_results=[
                    _reused("resolve_profile"),
                    _targeted_rebuild("run_benchmarks", "yolo26m"),
                ],
            ),
            preserved_results=6,
            allowed_model_rebuilds=["yolo11l"],
        )


def test_main_prints_fail_when_failed_workflow_leaves_exact_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, 5, exact=False)

    class FakeRunner:
        def __init__(self, _options, *, log):
            del log

        def run(self):
            _write_summary(run_dir, 8, exact=True)
            return SimpleNamespace(
                status="failed",
                completed=False,
                stage_results=_valid_stage_results(),
            )

    monkeypatch.setattr(WRAPPER, "_options_from_manifest", lambda _path: object())
    monkeypatch.setattr(WRAPPER, "EvaluationWorkflowRunner", FakeRunner)
    monkeypatch.setattr(
        "sys.argv", ["resume_missing_full_quality.py", "--run-dir", str(run_dir)],
    )

    assert WRAPPER.main() == 1
    output = capsys.readouterr().out
    assert "MISSING_FULL_QUALITY_RESUME=FAIL" in output
    assert "MISSING_FULL_QUALITY_RESUME=PASS" not in output


@pytest.mark.parametrize("legacy_v27711", [True, False])
def test_exact_eight_uses_sealed_read_only_completion_without_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    legacy_v27711: bool,
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_completed_resume_archive(
        run_dir, legacy_v27711=legacy_v27711,
    )

    def forbidden_options(_path: Path) -> object:
        pytest.fail("exact 8/8 must not read archived execution options")

    class ForbiddenRunner:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("exact 8/8 must not construct the workflow runner")

    monkeypatch.setattr(WRAPPER, "_options_from_manifest", forbidden_options)
    monkeypatch.setattr(WRAPPER, "EvaluationWorkflowRunner", ForbiddenRunner)
    monkeypatch.setattr(
        "sys.argv", ["resume_missing_full_quality.py", "--run-dir", str(run_dir)],
    )

    assert WRAPPER.main() == 0
    output = capsys.readouterr().out
    assert '"verification_mode": "read_only_exact_8_of_8"' in output
    assert '"runner_invoked": false' in output
    assert "MISSING_FULL_QUALITY_RESUME=PASS" in output


def test_exact_eight_with_unbound_attestation_fails_without_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_completed_resume_archive(run_dir, legacy_v27711=True)
    attestation = next(
        (run_dir / "jobs" / "missing_full_quality_reuse_attestations").iterdir()
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["must_reuse_stage_count"] = 43
    _write_json(attestation, payload)

    monkeypatch.setattr(
        WRAPPER,
        "_options_from_manifest",
        lambda _path: pytest.fail("exact 8/8 must not enter active resume"),
    )
    monkeypatch.setattr(
        WRAPPER,
        "EvaluationWorkflowRunner",
        lambda *_args, **_kwargs: pytest.fail("runner must remain unused"),
    )
    monkeypatch.setattr(
        "sys.argv", ["resume_missing_full_quality.py", "--run-dir", str(run_dir)],
    )

    assert WRAPPER.main() == 1
    output = capsys.readouterr().out
    assert "MISSING_FULL_QUALITY_RESUME=FAIL" in output
    assert "MISSING_FULL_QUALITY_RESUME=PASS" not in output


def test_legacy_failed_state_reason_is_rejected_outside_exact_target_surface(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_completed_resume_archive(run_dir, legacy_v27711=True)
    resume_path = run_dir / "jobs" / "resume_summary.json"
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    illegal = next(
        row for row in resume["decisions"]
        if row["model_id"] == "mobilenet_v3_large"
        and row["stage"] == "prepare_model"
    )
    illegal.update({
        "reusable": False,
        "reason": "previous_stage_state_not_reusable:failed",
        "previous_status": "failed",
        "artifacts_complete": False,
    })
    illegal.pop("missing_full_quality_reuse", None)
    resume["counts"] = _decision_counts(resume["decisions"])
    _rewrite_resume_summary_with_binding(run_dir, resume)

    with pytest.raises(
        RuntimeError, match="persisted resume decision audit failed",
    ):
        WRAPPER._verify_completed_resume_read_only(run_dir)


def test_four_of_eight_still_fails_before_options_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    _write_summary(run_dir, 4, exact=False)
    monkeypatch.setattr(
        WRAPPER,
        "_options_from_manifest",
        lambda _path: pytest.fail("4/8 must fail before option reconstruction"),
    )
    monkeypatch.setattr(
        WRAPPER,
        "EvaluationWorkflowRunner",
        lambda *_args, **_kwargs: pytest.fail("4/8 must not invoke runner"),
    )
    monkeypatch.setattr(
        "sys.argv", ["resume_missing_full_quality.py", "--run-dir", str(run_dir)],
    )

    assert WRAPPER.main() == 1
    output = capsys.readouterr().out
    assert "observed 4" in output
    assert "MISSING_FULL_QUALITY_RESUME=FAIL" in output
