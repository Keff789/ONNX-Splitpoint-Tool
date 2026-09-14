from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

import onnx_splitpoint_tool.resume_preparation as preparation
from onnx_splitpoint_tool.resume_artifact_rehydration import (
    ArtifactRequirement,
)
from onnx_splitpoint_tool.resume_remote_rehydration import (
    RemoteArtifactRehydrationError,
)


REMOTE_ROOT = "/home/nx/native_fifo_evalsets/frozen-run"
SSH_TARGET = "nx@192.168.0.145"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    attempt = tmp_path / "resume_attempt"
    plan = attempt / "plan"
    plan.mkdir(parents=True)
    reports = tmp_path / "EvaluationRun" / "reports"
    reports.mkdir(parents=True)
    summary = reports / "native_producer_combined_summary.json"
    summary.write_text('{"schema":"test-summary"}\n', encoding="utf-8")
    return attempt, plan, summary


def _row(*, backend: str = "native_full_hailo10h") -> dict[str, Any]:
    return {
        "backend": backend,
        "model": "yolov7_paper",
        "case": "full",
        "setup_id": "orin_nx_hailo10_01",
    }


def _context() -> dict[str, str]:
    return {
        "remote_root": REMOTE_ROOT,
        "hailo10_ssh": SSH_TARGET,
        "hailo8_ssh": "nx@192.168.0.104",
        "deepx_ssh": "nx@192.168.0.102",
    }


def _requirement(
    role: str,
    filename: str,
    *,
    size_bytes: int | None,
) -> ArtifactRequirement:
    return ArtifactRequirement(
        role=role,
        remote_path=f"{REMOTE_ROOT}/{filename}",
        sha256=hashlib.sha256(role.encode("utf-8")).hexdigest(),
        size_bytes=size_bytes,
    )


def _contract(requirements: list[ArtifactRequirement]) -> dict[str, Any]:
    return {
        "schema": "test-contract-selection",
        "contract_file": "/attempt/plan/row.command_contract.json",
        "requirements": requirements,
    }


def _probe(
    requirements: list[ArtifactRequirement],
    *,
    exact_paths: set[str],
) -> dict[str, Any]:
    entries = [
        {
            "remote_path": requirement.remote_path,
            "roles": [requirement.role],
            "expected_sha256": requirement.sha256,
            "expected_size_bytes": requirement.size_bytes,
            "remote_status": (
                "exact"
                if requirement.remote_path in exact_paths
                else "missing"
            ),
            "remote_sha256": (
                requirement.sha256
                if requirement.remote_path in exact_paths
                else None
            ),
            "remote_size_bytes": (
                requirement.size_bytes
                if requirement.remote_path in exact_paths
                else None
            ),
            "exact": requirement.remote_path in exact_paths,
            "safe_to_rehydrate": (
                requirement.remote_path not in exact_paths
            ),
        }
        for requirement in requirements
    ]
    exact_count = sum(bool(entry["exact"]) for entry in entries)
    return {
        "schema": "onnx-splitpoint/resume-remote-artifact-probe",
        "schema_version": 1,
        "ok": True,
        "read_only": True,
        "ssh_target": SSH_TARGET,
        "remote_run_root": REMOTE_ROOT,
        "requirement_count": len(entries),
        "exact_count": exact_count,
        "rehydration_required_count": len(entries) - exact_count,
        "all_exact": exact_count == len(entries),
        "entries": entries,
    }


def _cohort(*, ok: bool) -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/resume-cohort-preflight",
        "schema_version": 1,
        "ok": ok,
        "status": "pass" if ok else "failed",
        "measurement_wrapper_allowed": ok,
        "measurement_wrapper_started_count": 0,
        "collector_started_repeat_count": 0,
        "workload_started_repeat_count": 0,
    }


def _forbid(label: str) -> Callable[..., Any]:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{label} must not be called")

    return forbidden


def _assert_zero_execution_counters(report: dict[str, Any]) -> None:
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0


def _assert_persisted(report: dict[str, Any]) -> None:
    path = Path(report["report_path"])
    assert path.is_file()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == report
    body = dict(report)
    declared = body.pop("report_sha256")
    assert declared == _canonical_sha256(body)


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _forbid("real subprocess/network/hardware"),
    )


def test_all_remote_artifacts_exact_skips_local_resolution_and_allows_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    requirements = [
        _requirement(
            "hef", "yolov7/full/compiled.hef", size_bytes=None,
        ),
        _requirement(
            "input_manifest",
            "yolov7/full/input_manifest.json",
            size_bytes=123,
        ),
    ]
    events: list[str] = []

    def contract_loader(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("contract")
        return _contract(requirements)

    def probe(
        observed: list[ArtifactRequirement],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        events.append("probe")
        assert observed == requirements
        return _probe(
            requirements,
            exact_paths={
                requirement.remote_path for requirement in requirements
            },
        )

    def cohort(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("cohort")
        return _cohort(ok=True)

    monkeypatch.setattr(
        preparation, "resume_artifact_requirements", contract_loader,
    )
    monkeypatch.setattr(
        preparation, "probe_remote_requirements", probe,
    )
    monkeypatch.setattr(
        preparation,
        "build_resume_artifact_stage_map",
        _forbid("local stage-map resolver"),
    )
    monkeypatch.setattr(
        preparation,
        "rehydrate_remote_stage_map",
        _forbid("remote rehydration"),
    )
    monkeypatch.setattr(
        preparation, "run_resume_cohort_preflight", cohort,
    )

    report = preparation.prepare_resume_measurement_cohort(
        [_row()],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-exact",
    )

    assert report["ok"] is True
    assert report["status"] == "ready_for_measurement"
    assert report["measurement_wrapper_allowed"] is True
    _assert_zero_execution_counters(report)
    assert events == ["contract", "probe", "cohort"]
    group = report["remote_groups"][0]
    assert group["stage_map_path"] == ""
    assert (
        group["rehydration"]["status"]
        == "all_remote_artifacts_already_exact"
    )
    assert group["rehydration"]["rehydrated_count"] == 0
    _assert_persisted(report)


def test_only_missing_manifest_is_resolved_and_rehydrated_before_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    hef = _requirement(
        "hef", "yolov7/full/compiled.hef", size_bytes=None,
    )
    manifest = _requirement(
        "semantic_output_manifest",
        "yolo26/b038/native_outputs/native_outputs_manifest.json",
        size_bytes=6378,
    )
    requirements = [hef, manifest]
    events: list[str] = []
    stage_map = {
        "schema": "onnx-splitpoint/resume-artifact-stage-map",
        "schema_version": 1,
        "status": "ready",
        "local_only": True,
        "transport_performed": False,
        "allowed_remote_roots": [REMOTE_ROOT],
        "artifact_count": 1,
        "total_bytes": manifest.size_bytes,
        "entries": [{
            "roles": [manifest.role],
            "remote_path": manifest.remote_path,
            "source_path": str(tmp_path / "manifest.json"),
            "sha256": manifest.sha256,
            "size_bytes": manifest.size_bytes,
        }],
        "stage_map_sha256": "a" * 64,
    }

    monkeypatch.setattr(
        preparation,
        "resume_artifact_requirements",
        lambda *_args, **_kwargs: _contract(requirements),
    )

    def probe(
        observed: list[ArtifactRequirement],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        events.append("probe")
        return _probe(observed, exact_paths={hef.remote_path})

    def resolve(
        observed: list[ArtifactRequirement],
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("resolve")
        assert observed == [manifest]
        assert kwargs["allowed_remote_roots"] == [REMOTE_ROOT]
        return stage_map

    def rehydrate(
        observed: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("rehydrate")
        assert observed is stage_map
        assert kwargs["ssh_target"] == SSH_TARGET
        assert kwargs["remote_run_root"] == REMOTE_ROOT
        return {
            "ok": True,
            "status": "complete",
            "entry_count": 1,
            "no_op_count": 0,
            "rehydrated_count": 1,
            "backup_count": 0,
            "transferred_bytes": manifest.size_bytes,
        }

    def cohort(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("cohort")
        return _cohort(ok=True)

    monkeypatch.setattr(
        preparation, "probe_remote_requirements", probe,
    )
    monkeypatch.setattr(
        preparation, "build_resume_artifact_stage_map", resolve,
    )
    monkeypatch.setattr(
        preparation, "rehydrate_remote_stage_map", rehydrate,
    )
    monkeypatch.setattr(
        preparation, "run_resume_cohort_preflight", cohort,
    )

    report = preparation.prepare_resume_measurement_cohort(
        [_row()],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-missing-manifest",
        artifact_store_roots=[tmp_path / "artifact-store"],
    )

    assert report["ok"] is True
    assert report["measurement_wrapper_allowed"] is True
    _assert_zero_execution_counters(report)
    assert events == ["probe", "resolve", "rehydrate", "cohort"]
    group = report["remote_groups"][0]
    assert group["rehydration"]["rehydrated_count"] == 1
    assert Path(group["stage_map_path"]).is_file()
    assert Path(group["rehydration_path"]).is_file()
    _assert_persisted(report)


def test_local_resolver_failure_is_structured_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    manifest = _requirement(
        "semantic_output_manifest",
        "yolo26/b038/native_outputs/native_outputs_manifest.json",
        size_bytes=6378,
    )
    monkeypatch.setattr(
        preparation,
        "resume_artifact_requirements",
        lambda *_args, **_kwargs: _contract([manifest]),
    )
    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        lambda *_args, **_kwargs: _probe([manifest], exact_paths=set()),
    )

    def fail_resolver(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("exact_source_not_found")

    monkeypatch.setattr(
        preparation, "build_resume_artifact_stage_map", fail_resolver,
    )
    monkeypatch.setattr(
        preparation,
        "rehydrate_remote_stage_map",
        _forbid("rehydration after resolver failure"),
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        _forbid("cohort after resolver failure"),
    )

    report = preparation.prepare_resume_measurement_cohort(
        [_row()],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-resolver-failure",
    )

    assert report["ok"] is False
    assert report["status"] == "resume_preparation_failed"
    assert report["failure_type"] == "ValueError"
    assert report["failure"] == "exact_source_not_found"
    assert report["measurement_wrapper_allowed"] is False
    _assert_zero_execution_counters(report)
    _assert_persisted(report)


def test_remote_rehydration_failure_is_structured_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    manifest = _requirement(
        "semantic_output_manifest",
        "yolo26/b038/native_outputs/native_outputs_manifest.json",
        size_bytes=6378,
    )
    stage_map = {"schema": "test-stage-map"}
    partial = {
        "schema": "onnx-splitpoint/resume-remote-artifact-rehydration",
        "schema_version": 1,
        "ok": False,
        "status": "failed",
        "failure_code": "remote_probe_symlink",
        "measurement_wrapper_started_count": 0,
        "collector_started_repeat_count": 0,
        "workload_started_repeat_count": 0,
    }
    monkeypatch.setattr(
        preparation,
        "resume_artifact_requirements",
        lambda *_args, **_kwargs: _contract([manifest]),
    )
    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        lambda *_args, **_kwargs: _probe([manifest], exact_paths=set()),
    )
    monkeypatch.setattr(
        preparation,
        "build_resume_artifact_stage_map",
        lambda *_args, **_kwargs: stage_map,
    )

    def fail_remote(*_args: Any, **_kwargs: Any) -> Any:
        raise RemoteArtifactRehydrationError(
            "remote_probe_symlink",
            detail=manifest.remote_path,
            report=partial,
        )

    monkeypatch.setattr(
        preparation, "rehydrate_remote_stage_map", fail_remote,
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        _forbid("cohort after remote failure"),
    )

    report = preparation.prepare_resume_measurement_cohort(
        [_row()],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-remote-failure",
    )

    assert report["ok"] is False
    assert report["status"] == "resume_preparation_failed"
    assert report["failure_code"] == "remote_probe_symlink"
    assert report["failure_detail"] == manifest.remote_path
    assert report["remote_rehydration_partial"] == partial
    assert report["measurement_wrapper_allowed"] is False
    _assert_zero_execution_counters(report)
    partial_path = (
        attempt
        / "resume_preparation"
        / "group_00_rehydration_partial.json"
    )
    assert json.loads(
        partial_path.read_text(encoding="utf-8")
    ) == partial
    _assert_persisted(report)


def test_cohort_failure_keeps_measurement_gate_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    hef = _requirement(
        "hef", "yolov7/full/compiled.hef", size_bytes=None,
    )
    monkeypatch.setattr(
        preparation,
        "resume_artifact_requirements",
        lambda *_args, **_kwargs: _contract([hef]),
    )
    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        lambda *_args, **_kwargs: _probe(
            [hef], exact_paths={hef.remote_path},
        ),
    )
    monkeypatch.setattr(
        preparation,
        "build_resume_artifact_stage_map",
        _forbid("resolver for exact remote artifact"),
    )
    monkeypatch.setattr(
        preparation,
        "rehydrate_remote_stage_map",
        _forbid("rehydration for exact remote artifact"),
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        lambda *_args, **_kwargs: _cohort(ok=False),
    )

    report = preparation.prepare_resume_measurement_cohort(
        [_row()],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-cohort-failure",
    )

    assert report["ok"] is False
    assert report["status"] == "resume_cohort_preflight_failed"
    assert report["measurement_wrapper_allowed"] is False
    assert report["cohort_preflight"]["ok"] is False
    _assert_zero_execution_counters(report)
    _assert_persisted(report)


def test_unknown_ssh_family_is_blocked_before_contract_or_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    monkeypatch.setattr(
        preparation,
        "resume_artifact_requirements",
        _forbid("contract selection for unknown SSH family"),
    )
    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        _forbid("remote probe for unknown SSH family"),
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        _forbid("cohort for unknown SSH family"),
    )

    unknown_row = _row(backend="native_full_tensorrt")
    unknown_row["setup_id"] = "orin_nx_tensorrt_01"
    report = preparation.prepare_resume_measurement_cohort(
        [unknown_row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-unknown-family",
    )

    assert report["ok"] is False
    assert report["status"] == "resume_preparation_failed"
    assert report["failure_type"] == "ValueError"
    assert "resume_row_ssh_family_unknown" in report["failure"]
    assert report["measurement_wrapper_allowed"] is False
    _assert_zero_execution_counters(report)
    _assert_persisted(report)
