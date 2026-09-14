from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import onnx_splitpoint_tool.resume_preparation as preparation
import onnx_splitpoint_tool.resume_hailo8_source_recovery as source_recovery
from onnx_splitpoint_tool.resume_artifact_contract import (
    SPLIT_COMMAND_CONTRACT_SCHEMA,
    resume_artifact_requirements,
)
from onnx_splitpoint_tool.resume_artifact_rehydration import (
    ArtifactRequirement,
)
from onnx_splitpoint_tool.resume_hailo8_source_recovery import (
    Hailo8ResumeSourceRecoveryError,
    materialize_hailo8_literal_sources,
)


REMOTE_ROOT = "/home/nx/native_fifo_evalsets/campaign_gate"
REAL_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "v2725_hailo8_resume"
    / (
        "hailo8_to_trt__resnet50__b052__orin_nx_hailo8_01__"
        "8eb40a232207.command_contract.json"
    )
)
REAL_FIXTURE_FILE_SHA256 = (
    "4e5bac737c9a071eee2bf218c3f76e996f69dd0183c7796bdca088c8c91aaba6"
)
REAL_FIXTURE_LOGICAL_SHA256 = (
    "8eb40a2322078272add157c50b41309f2a5b43be5dff16eae773eb339f106a8b"
)
REAL_REMOTE_ROOT = (
    "/home/nx/native_fifo_evalsets/"
    "resnet_yolo26s_yolo7_20260729_214045"
)
CMAKE_SHA256 = (
    "a6734f5ccd8f08d2fabd2bdef17bbaa2a567e16a5e8e264ef0d0c128c582cae3"
)
CMAKE_SIZE_BYTES = 1_665
GENERATED_CPP_SHA256 = (
    "99dfa223ac4aa4ec2500ad4b5e032396e26abec197b7dd20e647eff314aa22ae"
)
GENERATED_CPP_SIZE_BYTES = 41_485
SOURCE_RECOVERY_TOTAL_BYTES = (
    CMAKE_SIZE_BYTES + GENERATED_CPP_SIZE_BYTES
)
SSH_TARGET = "nx@192.168.0.104"
CMAKE_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "v2726_hailo8_resume"
    / "CMakeLists.txt"
)
REAL_YOLO26S_PROJECTION = (
    Path(__file__).parent
    / "fixtures"
    / "v2726_hailo8_resume"
    / "yolo26s_b038_python_detection_contract_projection.json"
)
REAL_YOLO26S_SOURCE_CONTRACT_SHA256 = (
    "dc73b798d80e13bda48df8f4625bf2d1a3838f96eac9c58912bc4f8c2e3f10dc"
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("contract_sha256", None)
    return hashlib.sha256(json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _artifact(
    role: str,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    payload = role.encode("utf-8")
    return {
        "path": path or f"{REMOTE_ROOT}/native/{role}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _synthetic_contract() -> dict[str, Any]:
    python_path = "/usr/bin/python3.10"
    python_sha = hashlib.sha256(b"python3.10").hexdigest()
    hailo_sites = ["/home/nx/hailo_py/lib/python3.10/site-packages"]
    source_base = (
        f"{REMOTE_ROOT}/yolo26s/benchmark_set/native_pipeline/b038/"
        "hailo_to_trt/uint8_dequant_fp16"
    )
    contract: dict[str, Any] = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "schema_version": 1,
        "backend": "hailo8_to_trt",
        "model": "yolo26s",
        "case": "b038",
        "setup_id": "orin_nx_hailo8_01",
        "runner": "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "runner_sha256": "1" * 64,
        "python_executable": python_path,
        "native_executable": python_path,
        "native_executable_sha256": python_sha,
        "interpreter_identity": {
            "executable": python_path,
            "resolved_executable": python_path,
            "executable_sha256": python_sha,
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "process_local_extra_sites": hailo_sites,
        },
        "runtime_options": {
            "producer_impl": "hailo8_python_vstreams_fifo",
            "task": "detection",
            "prepared_input_bound": True,
            "energy_prepared_feed_capable": True,
            "process_local_extra_sites": hailo_sites,
            "mixed_runtime_site_policy": (
                "site.addsitedir_after_system_defaults"
            ),
        },
        "mixed_runtime_contract": {
            "status": "ready",
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
            ),
            "site_policy": "site.addsitedir_after_system_defaults",
            "python_executable": python_path,
            "resolved_python_executable": python_path,
            "process_local_extra_sites": hailo_sites,
            "source_closure_ok": True,
        },
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
            "cmake": {
                "path": f"{source_base}/CMakeLists.txt",
                "sha256": CMAKE_SHA256,
                "size_bytes": CMAKE_SIZE_BYTES,
            },
            "generated_cpp": {
                "path": f"{source_base}/main.cpp",
                "sha256": GENERATED_CPP_SHA256,
                "size_bytes": GENERATED_CPP_SIZE_BYTES,
            },
            "native_executable": {
                "path": python_path,
                "sha256": python_sha,
            },
            "future_preflight_artifact": _artifact(
                "future_preflight_artifact"
            ),
            "hef": _artifact(
                "hef",
                path="/home/nx/splitpoint_runs/cache/part1.hef",
            ),
            "python_executable": _artifact(
                "python_executable",
                path=python_path,
            ),
        },
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
        "complete": True,
    }
    contract["artifacts"]["python_executable"]["sha256"] = python_sha
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _bound_requirements(
    tmp_path: Path,
    contract: dict[str, Any],
    *,
    remote_root: str = REMOTE_ROOT,
) -> list[Any]:
    plan = tmp_path / "plan"
    plan.mkdir()
    raw = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    contract_path = plan / "command_contract.json"
    contract_path.write_bytes(raw)
    row = {
        "backend": contract["backend"],
        "model": contract["model"],
        "case": contract["case"],
        "setup_id": contract["setup_id"],
        "command_contract_file": str(contract_path),
        "command_contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "successful_command_contract_sha256": contract["contract_sha256"],
    }
    return list(resume_artifact_requirements(
        row,
        plan_root=plan,
        frozen_remote_root=remote_root,
    )["requirements"])


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    attempt = tmp_path / "resume_attempt"
    plan = attempt / "plan"
    plan.mkdir(parents=True)
    reports = tmp_path / "EvaluationRun" / "reports"
    reports.mkdir(parents=True)
    summary = reports / "native_producer_summary.json"
    summary.write_text('{"schema":"test-summary"}\n', encoding="utf-8")
    return attempt, plan, summary


def _bound_row(
    plan: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    path = plan / "hailo8.command_contract.json"
    raw = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)
    return {
        "backend": contract["backend"],
        "model": contract["model"],
        "case": contract["case"],
        "setup_id": contract["setup_id"],
        "command_contract_file": str(path),
        "command_contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "successful_command_contract_sha256": contract["contract_sha256"],
    }


def _python_detection_contract_with_bound_sources() -> dict[str, Any]:
    contract = _synthetic_contract()
    base = (
        f"{REMOTE_ROOT}/yolo26s/benchmark_set/native_pipeline/b038/"
        "hailo_to_trt/uint8_dequant_fp16"
    )
    contract["artifacts"]["cmake"] = {
        "path": f"{base}/CMakeLists.txt",
        "sha256": CMAKE_SHA256,
        "size_bytes": CMAKE_SIZE_BYTES,
    }
    contract["artifacts"]["generated_cpp"] = {
        "path": f"{base}/main.cpp",
        "sha256": GENERATED_CPP_SHA256,
        "size_bytes": GENERATED_CPP_SIZE_BYTES,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _context() -> dict[str, str]:
    return {
        "remote_root": REMOTE_ROOT,
        "hailo8_ssh": SSH_TARGET,
        "hailo10_ssh": "",
        "deepx_ssh": "",
    }


def _probe_report(
    requirements: list[ArtifactRequirement],
    *,
    exact_roles: set[str],
) -> dict[str, Any]:
    entries = []
    for requirement in requirements:
        exact = requirement.role in exact_roles
        entries.append({
            "remote_path": requirement.remote_path,
            "roles": [requirement.role],
            "expected_sha256": requirement.sha256,
            "expected_size_bytes": requirement.size_bytes,
            "remote_status": "exact" if exact else "missing",
            "remote_sha256": requirement.sha256 if exact else None,
            "remote_size_bytes": (
                requirement.size_bytes if exact else None
            ),
            "exact": exact,
            "safe_to_rehydrate": not exact,
        })
    exact_count = sum(entry["exact"] is True for entry in entries)
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


def _passing_cohort() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/resume-cohort-preflight",
        "schema_version": 1,
        "ok": True,
        "status": "verified",
        "measurement_wrapper_allowed": True,
        "measurement_wrapper_started_count": 0,
        "collector_started_repeat_count": 0,
        "workload_started_repeat_count": 0,
    }


def _manifest_bytes(
    rows: list[dict[str, Any]],
) -> bytes:
    payload = {
        "schema": "test-semantic-manifest",
        "payload_artifacts": rows,
        "payload_artifacts_sha256": _json_sha256(rows),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_source(
    root: Path,
    relative: str,
    payload: bytes,
) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_hailo8_python_detection_resume_restores_preflight_build_sources(
    tmp_path: Path,
) -> None:
    requirements = _bound_requirements(
        tmp_path,
        _synthetic_contract(),
    )
    roles = {requirement.role for requirement in requirements}

    assert roles == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "cmake",
        "generated_cpp",
        "input_image",
    }
    assert "future_preflight_artifact" not in roles
    assert "native_executable" not in roles
    assert "hef" not in roles
    assert "python_executable" not in roles


def test_hailo8_python_detection_requires_bound_interpreter_identity(
    tmp_path: Path,
) -> None:
    contract = _synthetic_contract()
    contract["native_executable"] = "/opt/unbound/python"
    contract["contract_sha256"] = _canonical_sha256(contract)

    roles = {
        requirement.role
        for requirement in _bound_requirements(tmp_path, contract)
    }

    assert "cmake" not in roles
    assert "generated_cpp" not in roles
    assert "native_executable" not in roles


@pytest.mark.parametrize(
    "drift",
    ("runner", "generated_cpp_sha256", "source_closure"),
)
def test_hailo8_source_recovery_profile_rejects_generator_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    contract = _synthetic_contract()
    if drift == "runner":
        contract["runner"] = "scripts/other_generator.py"
    elif drift == "generated_cpp_sha256":
        contract["artifacts"]["generated_cpp"]["sha256"] = "0" * 64
    else:
        contract["mixed_runtime_contract"]["source_closure_ok"] = False
    contract["contract_sha256"] = _canonical_sha256(contract)

    roles = {
        requirement.role
        for requirement in _bound_requirements(tmp_path, contract)
    }

    assert "cmake" not in roles
    assert "generated_cpp" not in roles


def test_archived_hailo8_cpp_contract_is_not_broadened_without_executable(
    tmp_path: Path,
) -> None:
    raw = REAL_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REAL_FIXTURE_FILE_SHA256
    contract = json.loads(raw)
    assert contract["contract_sha256"] == REAL_FIXTURE_LOGICAL_SHA256

    plan = tmp_path / "plan"
    plan.mkdir()
    contract_path = plan / "real_hailo8.command_contract.json"
    contract_path.write_bytes(raw)
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "command_contract_file": str(contract_path),
        "command_contract_file_sha256": REAL_FIXTURE_FILE_SHA256,
        "successful_command_contract_sha256": (
            REAL_FIXTURE_LOGICAL_SHA256
        ),
    }

    requirements = resume_artifact_requirements(
        row,
        plan_root=plan,
        frozen_remote_root=REAL_REMOTE_ROOT,
    )["requirements"]
    roles = {requirement.role for requirement in requirements}

    assert roles == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "input_image",
    }


def test_archived_yolo26s_python_contract_projection_selects_six_roles(
    tmp_path: Path,
) -> None:
    contract = json.loads(
        REAL_YOLO26S_PROJECTION.read_text(encoding="utf-8")
    )
    provenance = contract.pop("fixture_provenance")
    assert provenance["source_contract_sha256"] == (
        REAL_YOLO26S_SOURCE_CONTRACT_SHA256
    )
    assert provenance["source_summary_sha256"] == (
        "a66d9fb108b17c305d4f8169e44150ba"
        "122819bf9035b9994f8803419bce22db"
    )
    contract["contract_sha256"] = _canonical_sha256(contract)

    requirements = _bound_requirements(
        tmp_path,
        contract,
        remote_root=REAL_REMOTE_ROOT,
    )
    by_role = {
        requirement.role: requirement
        for requirement in requirements
    }

    assert set(by_role) == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "cmake",
        "generated_cpp",
        "input_image",
    }
    assert by_role["cmake"].sha256 == CMAKE_SHA256
    assert by_role["cmake"].size_bytes is None
    assert by_role["generated_cpp"].sha256 == GENERATED_CPP_SHA256
    assert by_role["generated_cpp"].size_bytes is None


def test_literal_source_recovery_matches_real_sealed_hashes(
    tmp_path: Path,
) -> None:
    base = f"{REMOTE_ROOT}/native_pipeline"
    requirements = [
        ArtifactRequirement(
            role="cmake",
            remote_path=f"{base}/CMakeLists.txt",
            sha256=CMAKE_SHA256,
            size_bytes=CMAKE_SIZE_BYTES,
        ),
        ArtifactRequirement(
            role="generated_cpp",
            remote_path=f"{base}/main.cpp",
            sha256=GENERATED_CPP_SHA256,
            size_bytes=GENERATED_CPP_SIZE_BYTES,
        ),
    ]
    parent = tmp_path / "derived"
    parent.mkdir()

    result = materialize_hailo8_literal_sources(
        requirements,
        tool_root=Path(__file__).resolve().parents[1],
        destination=parent / "group_00",
    )

    assert result["ok"] is True
    assert result["status"] == "exact_sources_materialized"
    assert result["artifact_count"] == 2
    assert result["total_bytes"] == SOURCE_RECOVERY_TOTAL_BYTES
    assert result["local_only"] is True
    assert result["code_executed"] is False
    assert result["build_started"] is False
    assert result["remote_mutation_performed"] is False
    observed = {
        row["role"]: row for row in result["entries"]
    }
    assert set(observed) == {"cmake", "generated_cpp"}
    assert observed["cmake"]["sha256"] == CMAKE_SHA256
    assert observed["generated_cpp"]["sha256"] == GENERATED_CPP_SHA256
    for row in observed.values():
        path = Path(row["source_path"])
        assert path.is_file()
        assert path.is_symlink() is False
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert path.stat().st_mode & 0o222 == 0


def test_literal_source_recovery_rejects_contract_hash_mismatch(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "derived"
    parent.mkdir()
    requirement = ArtifactRequirement(
        role="generated_cpp",
        remote_path=f"{REMOTE_ROOT}/native_pipeline/main.cpp",
        sha256="0" * 64,
        size_bytes=GENERATED_CPP_SIZE_BYTES,
    )

    with pytest.raises(
        Hailo8ResumeSourceRecoveryError,
        match="derived_source_sha256_mismatch:role=generated_cpp",
    ):
        materialize_hailo8_literal_sources(
            [requirement],
            tool_root=Path(__file__).resolve().parents[1],
            destination=parent / "group_00",
        )
    assert (parent / "group_00").exists() is False


def test_literal_source_recovery_rejects_runner_symlink(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tool"
    scripts = tool / "scripts"
    scripts.mkdir(parents=True)
    scripts.joinpath(
        "native_hailo_trt_fifo_from_benchmarkset.py"
    ).symlink_to(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "native_hailo_trt_fifo_from_benchmarkset.py"
    )
    parent = tmp_path / "derived"
    parent.mkdir()
    requirement = ArtifactRequirement(
        role="cmake",
        remote_path=f"{REMOTE_ROOT}/native_pipeline/CMakeLists.txt",
        sha256=CMAKE_SHA256,
        size_bytes=CMAKE_SIZE_BYTES,
    )

    with pytest.raises(
        Hailo8ResumeSourceRecoveryError,
        match="runner_path_contains_symlink",
    ):
        materialize_hailo8_literal_sources(
            [requirement],
            tool_root=tool,
            destination=parent / "group_00",
        )
    assert (parent / "group_00").exists() is False


def test_literal_source_recovery_rejects_dynamic_ast_assignment(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tool"
    runner = (
        tool
        / "scripts"
        / "native_hailo_trt_fifo_from_benchmarkset.py"
    )
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "CMAKE_TXT = make_source()\nCPP_SOURCE = 'static'\n",
        encoding="utf-8",
    )
    parent = tmp_path / "derived"
    parent.mkdir()
    requirement = ArtifactRequirement(
        role="cmake",
        remote_path=f"{REMOTE_ROOT}/native_pipeline/CMakeLists.txt",
        sha256=CMAKE_SHA256,
        size_bytes=CMAKE_SIZE_BYTES,
    )

    with pytest.raises(
        Hailo8ResumeSourceRecoveryError,
        match="runner_literal_not_static:CMAKE_TXT",
    ):
        materialize_hailo8_literal_sources(
            [requirement],
            tool_root=tool,
            destination=parent / "group_00",
        )
    assert (parent / "group_00").exists() is False


def test_literal_source_recovery_wraps_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "derived"
    parent.mkdir()
    requirement = ArtifactRequirement(
        role="cmake",
        remote_path=f"{REMOTE_ROOT}/native_pipeline/CMakeLists.txt",
        sha256=CMAKE_SHA256,
        size_bytes=CMAKE_SIZE_BYTES,
    )

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("offline injected fsync failure")

    monkeypatch.setattr(source_recovery.os, "fsync", fail_fsync)
    with pytest.raises(
        Hailo8ResumeSourceRecoveryError,
        match="derived_source_write_failed",
    ):
        materialize_hailo8_literal_sources(
            [requirement],
            tool_root=Path(__file__).resolve().parents[1],
            destination=parent / "group_00",
        )


def test_observed_partial_remote_root_restores_only_two_preflight_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the real v2.72.5 CMakeLists.txt preflight failure."""

    attempt, plan, summary = _layout(tmp_path)
    contract = _python_detection_contract_with_bound_sources()
    row = _bound_row(plan, contract)
    events: list[str] = []
    cmake_bytes = CMAKE_FIXTURE.read_bytes()
    assert len(cmake_bytes) == CMAKE_SIZE_BYTES
    assert hashlib.sha256(cmake_bytes).hexdigest() == CMAKE_SHA256
    _write_source(
        summary.parent.parent,
        "archived/native_pipeline/CMakeLists.txt",
        cmake_bytes,
    )

    def probe(
        requirements: list[ArtifactRequirement],
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("probe")
        assert kwargs["ssh_target"] == SSH_TARGET
        assert kwargs["remote_run_root"] == REMOTE_ROOT
        assert {requirement.role for requirement in requirements} == {
            "prepared_input",
            "semantic_boundary_manifest",
            "semantic_output_manifest",
            "cmake",
            "generated_cpp",
            "input_image",
        }
        return _probe_report(
            requirements,
            exact_roles={
                "prepared_input",
                "semantic_boundary_manifest",
                "semantic_output_manifest",
                "input_image",
            },
        )

    def rehydrate(
        stage_map: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("rehydrate")
        assert kwargs["ssh_target"] == SSH_TARGET
        assert kwargs["remote_run_root"] == REMOTE_ROOT
        assert stage_map["artifact_count"] == 2
        assert stage_map["total_bytes"] == SOURCE_RECOVERY_TOTAL_BYTES
        entries = {
            tuple(entry["roles"]): entry
            for entry in stage_map["entries"]
        }
        assert set(entries) == {("cmake",), ("generated_cpp",)}
        assert entries[("cmake",)]["source_kind"] == "run_mirror"
        assert entries[("cmake",)]["source_path"].endswith(
            "archived/native_pipeline/CMakeLists.txt"
        )
        assert "derived_sources/group_00" in (
            entries[("generated_cpp",)]["source_path"]
        )
        for entry in entries.values():
            source = Path(entry["source_path"])
            assert source.is_file()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == (
                entry["sha256"]
            )
        return {
            "ok": True,
            "status": "complete",
            "entry_count": 2,
            "completed_entry_count": 2,
            "no_op_count": 0,
            "rehydrated_count": 2,
            "backup_count": 0,
            "transferred_bytes": SOURCE_RECOVERY_TOTAL_BYTES,
            "remote_run_root_created": False,
            "remote_directory_creation_count": 0,
        }

    def cohort(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("cohort")
        return _passing_cohort()

    monkeypatch.setattr(
        preparation, "probe_remote_requirements", probe,
    )
    monkeypatch.setattr(
        preparation, "rehydrate_remote_stage_map", rehydrate,
    )
    monkeypatch.setattr(
        preparation, "run_resume_cohort_preflight", cohort,
    )

    report = preparation.prepare_resume_measurement_cohort(
        [row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-observed-partial-root",
        artifact_store_roots=[],
    )

    assert report["ok"] is True
    assert report["status"] == "ready_for_measurement"
    assert report["measurement_wrapper_allowed"] is True
    assert events == ["probe", "rehydrate", "cohort"]
    group = report["remote_groups"][0]
    recovery = group["source_recovery"]
    assert recovery["ok"] is True
    assert recovery["artifact_count"] == 2
    assert recovery["total_bytes"] == SOURCE_RECOVERY_TOTAL_BYTES
    assert Path(group["source_recovery_path"]).is_file()
    assert Path(group["stage_map_path"]).is_file()
    assert group["rehydration"]["rehydrated_count"] == 2
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0


def test_source_recovery_failure_stops_before_remote_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, plan, summary = _layout(tmp_path)
    contract = _python_detection_contract_with_bound_sources()
    row = _bound_row(plan, contract)
    cmake_bytes = CMAKE_FIXTURE.read_bytes()
    _write_source(
        summary.parent.parent,
        "archived/native_pipeline/CMakeLists.txt",
        cmake_bytes,
    )

    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        lambda requirements, **_kwargs: _probe_report(
            requirements,
            exact_roles={
                "prepared_input",
                "semantic_boundary_manifest",
                "semantic_output_manifest",
                "input_image",
            },
        ),
    )

    def fail_recovery(*_args: Any, **_kwargs: Any) -> Any:
        raise Hailo8ResumeSourceRecoveryError(
            "derived_source_sha256_mismatch",
            role="generated_cpp",
            detail="offline-injected",
        )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("remote or cohort boundary must remain closed")

    monkeypatch.setattr(
        preparation,
        "materialize_hailo8_literal_sources",
        fail_recovery,
    )
    monkeypatch.setattr(
        preparation, "rehydrate_remote_stage_map", forbidden,
    )
    monkeypatch.setattr(
        preparation, "run_resume_cohort_preflight", forbidden,
    )

    report = preparation.prepare_resume_measurement_cohort(
        [row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-recovery-failure",
        artifact_store_roots=[],
    )

    assert report["ok"] is False
    assert report["status"] == "resume_preparation_failed"
    assert report["failure_type"] == "Hailo8ResumeSourceRecoveryError"
    assert report["failure_code"] == "derived_source_sha256_mismatch"
    assert report["failure_role"] == "generated_cpp"
    assert report["remote_groups"] == []
    assert report["cohort_preflight"] is None
    recovery_path = (
        attempt
        / "resume_preparation"
        / "group_00_source_recovery.json"
    )
    recovery = json.loads(
        recovery_path.read_text(encoding="utf-8")
    )
    assert recovery["ok"] is False
    assert recovery["remote_mutation_performed"] is False
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0


def test_fresh_remote_root_stages_full_nine_file_preflight_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted root restores six direct roles plus three payload files."""

    attempt, plan, summary = _layout(tmp_path)
    mirror = summary.parent.parent / "mirror"
    contract = _python_detection_contract_with_bound_sources()
    base = (
        f"{REMOTE_ROOT}/yolo26s/benchmark_set/native_pipeline/b038/"
        "hailo_to_trt/uint8_dequant_fp16"
    )

    prepared = b"prepared-input"
    image = b"image"
    boundary_payload = b"boundary-payload"
    output_payload_0 = b"output-payload-0"
    output_payload_1 = b"output-payload-1"
    boundary_rows = [{
        "role": "boundary_tensor",
        "path": "boundary_payload.bin",
        "sha256": hashlib.sha256(boundary_payload).hexdigest(),
        "size_bytes": len(boundary_payload),
    }]
    output_rows = [
        {
            "role": "output_0",
            "path": "output_payload_0.bin",
            "sha256": hashlib.sha256(output_payload_0).hexdigest(),
            "size_bytes": len(output_payload_0),
        },
        {
            "role": "output_1",
            "path": "output_payload_1.bin",
            "sha256": hashlib.sha256(output_payload_1).hexdigest(),
            "size_bytes": len(output_payload_1),
        },
    ]
    boundary_manifest = _manifest_bytes(boundary_rows)
    output_manifest = _manifest_bytes(output_rows)

    contract["artifacts"]["prepared_input"] = {
        "path": f"{base}/contract_artifacts/prepared_input.bin",
        "sha256": hashlib.sha256(prepared).hexdigest(),
        "size_bytes": len(prepared),
    }
    contract["artifacts"]["semantic_boundary_manifest"] = {
        "path": f"{base}/native_fifo_boundary/manifest.json",
        "sha256": hashlib.sha256(boundary_manifest).hexdigest(),
        "size_bytes": len(boundary_manifest),
    }
    contract["artifacts"]["semantic_output_manifest"] = {
        "path": f"{base}/native_fifo_outputs/manifest.json",
        "sha256": hashlib.sha256(output_manifest).hexdigest(),
        "size_bytes": len(output_manifest),
    }
    contract["input_image"] = (
        f"{REMOTE_ROOT}/yolo26s/benchmark_set/resources/image.jpg"
    )
    contract["input_image_sha256"] = hashlib.sha256(image).hexdigest()
    contract["contract_sha256"] = _canonical_sha256(contract)
    row = _bound_row(plan, contract)

    _write_source(mirror, "prepared/prepared_input.bin", prepared)
    _write_source(
        mirror, "boundary/manifest.json", boundary_manifest,
    )
    _write_source(
        mirror, "boundary/boundary_payload.bin", boundary_payload,
    )
    _write_source(mirror, "outputs/manifest.json", output_manifest)
    _write_source(
        mirror, "outputs/output_payload_0.bin", output_payload_0,
    )
    _write_source(
        mirror, "outputs/output_payload_1.bin", output_payload_1,
    )
    _write_source(mirror, "validation/image.jpg", image)

    def probe(
        requirements: list[ArtifactRequirement],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return _probe_report(requirements, exact_roles=set())

    def rehydrate(
        stage_map: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert stage_map["artifact_count"] == 9
        roles = {
            role
            for entry in stage_map["entries"]
            for role in entry["roles"]
        }
        assert {"cmake", "generated_cpp", "prepared_input"} <= roles
        assert sum(
            role.startswith("semantic_boundary_manifest.payload[")
            for role in roles
        ) == 1
        assert sum(
            role.startswith("semantic_output_manifest.payload[")
            for role in roles
        ) == 2
        return {
            "ok": True,
            "status": "complete",
            "entry_count": 9,
            "completed_entry_count": 9,
            "no_op_count": 0,
            "rehydrated_count": 9,
            "backup_count": 0,
            "transferred_bytes": stage_map["total_bytes"],
            "remote_run_root_created": True,
            "remote_directory_creation_count": 14,
        }

    monkeypatch.setattr(
        preparation, "probe_remote_requirements", probe,
    )
    monkeypatch.setattr(
        preparation, "rehydrate_remote_stage_map", rehydrate,
    )
    monkeypatch.setattr(
        preparation,
        "run_resume_cohort_preflight",
        lambda *_args, **_kwargs: _passing_cohort(),
    )

    report = preparation.prepare_resume_measurement_cohort(
        [row],
        attempt_dir=attempt,
        plan_root=plan,
        summary_path=summary,
        canonical_execution_context=_context(),
        resume_attempt_id="resume-fresh-remote-root",
        artifact_store_roots=[],
    )

    assert report["ok"] is True
    group = report["remote_groups"][0]
    assert group["probe"]["requirement_count"] == 6
    assert group["probe"]["exact_count"] == 0
    assert group["source_recovery"]["artifact_count"] == 2
    stage_map = json.loads(
        Path(group["stage_map_path"]).read_text(encoding="utf-8")
    )
    assert stage_map["artifact_count"] == 9
    assert group["rehydration"]["rehydrated_count"] == 9
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0
