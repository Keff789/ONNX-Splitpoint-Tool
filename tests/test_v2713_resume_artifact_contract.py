from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.resume_artifact_contract import (
    FULL_COMMAND_CONTRACT_SCHEMA,
    ResumeArtifactContractError,
    SPLIT_COMMAND_CONTRACT_SCHEMA,
    resume_artifact_requirements,
)
from onnx_splitpoint_tool.resume_artifact_rehydration import (
    build_resume_artifact_stage_map,
)


REMOTE_ROOT = "/home/nx/native_fifo_evalsets/frozen_run"


def _canonical_sha(value: dict) -> str:
    body = dict(value)
    body.pop("contract_sha256", None)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _row(plan: Path, contract: dict) -> dict:
    contract = dict(contract)
    contract.setdefault("schema_version", 1)
    contract.setdefault("model", "test_model")
    contract.setdefault("case", "b001")
    contract.setdefault("setup_id", "test_setup")
    contract["contract_sha256"] = _canonical_sha(contract)
    path = plan / "row.command_contract.json"
    raw = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(raw)
    return {
        field: contract[field]
        for field in ("backend", "model", "case", "setup_id")
    } | {
        "command_contract_file": str(path),
        "command_contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "successful_command_contract_sha256": contract["contract_sha256"],
    }


def _artifact(role: str, *, below_root: bool = True) -> dict:
    prefix = REMOTE_ROOT if below_root else "/home/nx/ONNX-Splitpoint-Tool"
    return {
        "path": f"{prefix}/{role}.bin",
        "sha256": hashlib.sha256(role.encode()).hexdigest(),
        "size_bytes": len(role),
    }


def _payload_manifest(
    *,
    remote_path: str,
    payload: bytes,
) -> bytes:
    rows = [{
        "role": "payload",
        "path": remote_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }]
    return json.dumps(
        {
            "schema": "onnx-splitpoint/runner-output-dump",
            "schema_version": 4,
            "payload_artifacts": rows,
            "payload_artifacts_sha256": hashlib.sha256(
                json.dumps(
                    rows,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_hailo10_split_contract_selects_only_bound_frozen_run_data_roles(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "hailo10h_to_trt",
        "runtime_options": {"part1_onnx_used": True},
        "prepared_input_contract": {
            "entries": [{
                "name": "images",
                "artifact_name": "prepared_input_00",
            }],
        },
        "artifacts": {
            "part1_onnx": _artifact("part1_onnx"),
            "prepared_input_00": _artifact("prepared_input_00"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
            "runner": _artifact("runner", below_root=False),
        },
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
    }
    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    roles = {requirement.role for requirement in result["requirements"]}
    assert roles == {
        "part1_onnx",
        "prepared_input_00",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "input_image",
    }
    assert "runner" not in roles
    assert all(
        requirement.remote_path.startswith(f"{REMOTE_ROOT}/")
        for requirement in result["requirements"]
    )


def test_hailo8_split_uses_prepared_input_without_part1_onnx(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "hailo8_to_trt",
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
            # A foreign backend's roles must never become requirements merely
            # because an old or extended contract happens to contain them.
            "part1_onnx": _artifact("part1_onnx"),
            "prepared_input_00": _artifact("prepared_input_00"),
        },
    }

    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    assert {
        requirement.role for requirement in result["requirements"]
    } == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
    }


def test_hailo10_resume_roles_preserve_v2724_contract(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "hailo10h_to_trt",
        "runtime_options": {"part1_onnx_used": False},
        "prepared_input_contract": {
            "entries": [
                {"name": "first", "artifact_name": "prepared_input_00"},
                {"name": "second", "artifact_name": "prepared_input_01"},
            ],
        },
        "artifacts": {
            "part1_onnx": _artifact("part1_onnx"),
            "prepared_input_00": _artifact("prepared_input_00"),
            "prepared_input_01": _artifact("prepared_input_01"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
        },
    }

    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    assert {
        requirement.role for requirement in result["requirements"]
    } == {
        "part1_onnx",
        "prepared_input_00",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
    }


def test_unverified_deepx_split_resume_stays_fail_closed(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "deepx_to_trt",
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "part1_input_contract": _artifact("part1_input_contract"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
        },
    }

    with pytest.raises(
        ResumeArtifactContractError,
        match="unsupported_resume_deepx_contract_profile",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_verified_deepx_split_selects_only_bound_frozen_run_data_roles(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "deepx_to_trt",
        "runtime_options": {
            "prepared_input_bound": True,
        },
        "prepared_input_contract": {
            "format": "numpy_npy_v1",
            "shape": [640, 640, 3],
            "dtype": "uint8",
            "c_contiguous": True,
        },
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
            # These are performance/runtime dependencies outside the frozen
            # transport root.  Resume preparation must never stage them merely
            # because the archived contract contains their identities.
            "part1_input_contract": _artifact(
                "part1_input_contract", below_root=False,
            ),
            "dxnn": _artifact("dxnn", below_root=False),
            "engine": _artifact("engine", below_root=False),
            "python_executable": _artifact(
                "python_executable", below_root=False,
            ),
        },
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
    }

    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    assert {
        requirement.role for requirement in result["requirements"]
    } == {
        "prepared_input",
        "semantic_boundary_manifest",
        "semantic_output_manifest",
        "input_image",
    }
    assert {
        "part1_input_contract",
        "dxnn",
        "engine",
        "python_executable",
    }.isdisjoint(
        requirement.role for requirement in result["requirements"]
    )
    assert all(
        requirement.remote_path.startswith(f"{REMOTE_ROOT}/")
        for requirement in result["requirements"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "float32"),
        ("c_contiguous", False),
        ("shape", [1, 3, 640, 640]),
        ("shape", [640, 0, 3]),
    ],
)
def test_deepx_resume_rejects_unverified_prepared_input_profiles(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    prepared_input_contract = {
        "format": "numpy_npy_v1",
        "shape": [640, 640, 3],
        "dtype": "uint8",
        "c_contiguous": True,
    }
    prepared_input_contract[field] = value
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "deepx_to_trt",
        "runtime_options": {"prepared_input_bound": True},
        "prepared_input_contract": prepared_input_contract,
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
        },
    }

    with pytest.raises(
        ResumeArtifactContractError,
        match="unsupported_resume_deepx_contract_profile",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_deepx_resume_expands_exact_semantic_payload_closure(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    mirror = tmp_path / "run-mirror"
    plan.mkdir()
    prepared = b"exact deepx numpy feed"
    image = b"exact source image"
    boundary_payload = b"exact boundary payload"
    output_payload = b"exact completed output payload"
    remote_base = f"{REMOTE_ROOT}/yolo26s/native/b026/deepx_to_trt"
    boundary_payload_remote = (
        f"{remote_base}/native_fifo_boundary/boundary_float32.bin"
    )
    output_payload_remote = (
        f"{remote_base}/native_outputs/output_00_output0.bin"
    )
    boundary_manifest = _payload_manifest(
        remote_path=boundary_payload_remote,
        payload=boundary_payload,
    )
    output_manifest = _payload_manifest(
        remote_path=output_payload_remote,
        payload=output_payload,
    )
    sources = {
        f"{remote_base}/contract_artifacts/deepx_selected_input.npy":
            prepared,
        f"{remote_base}/native_fifo_boundary/"
        "native_fifo_boundary_manifest.json": boundary_manifest,
        f"{remote_base}/native_outputs/native_outputs_manifest.json":
            output_manifest,
        f"{REMOTE_ROOT}/validation/image.jpg": image,
        boundary_payload_remote: boundary_payload,
        output_payload_remote: output_payload,
    }
    artifacts = {
        "prepared_input": sources[
            f"{remote_base}/contract_artifacts/deepx_selected_input.npy"
        ],
        "semantic_boundary_manifest": boundary_manifest,
        "semantic_output_manifest": output_manifest,
    }
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "deepx_to_trt",
        "runtime_options": {"prepared_input_bound": True},
        "prepared_input_contract": {
            "format": "numpy_npy_v1",
            "shape": [640, 640, 3],
            "dtype": "uint8",
            "c_contiguous": True,
        },
        "artifacts": {
            role: {
                "path": remote_path,
                "sha256": hashlib.sha256(artifacts[role]).hexdigest(),
                "size_bytes": len(artifacts[role]),
            }
            for role, remote_path in {
                "prepared_input": (
                    f"{remote_base}/contract_artifacts/"
                    "deepx_selected_input.npy"
                ),
                "semantic_boundary_manifest": (
                    f"{remote_base}/native_fifo_boundary/"
                    "native_fifo_boundary_manifest.json"
                ),
                "semantic_output_manifest": (
                    f"{remote_base}/native_outputs/"
                    "native_outputs_manifest.json"
                ),
            }.items()
        },
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(image).hexdigest(),
    }
    requirements = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )["requirements"]
    for remote_path, payload in sources.items():
        local = mirror / remote_path.removeprefix(f"{REMOTE_ROOT}/")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)

    stage_map = build_resume_artifact_stage_map(
        requirements,
        run_mirror_roots=[mirror],
        artifact_store_roots=[],
        allowed_remote_roots=[REMOTE_ROOT],
        expand_payload_manifests=True,
    )

    assert stage_map["status"] == "ready"
    assert stage_map["artifact_count"] == len(sources)
    assert {
        row["remote_path"] for row in stage_map["entries"]
    } == set(sources)
    assert all(
        row["source_kind"] == "run_mirror"
        for row in stage_map["entries"]
    )


def test_contract_identity_mismatch_is_rejected_before_role_selection(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "hailo8_to_trt",
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
        },
    }
    row = _row(plan, contract)
    row["backend"] = "hailo10h_to_trt"

    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_row_identity_mismatch:backend",
    ):
        resume_artifact_requirements(
            row,
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


@pytest.mark.parametrize(
    ("schema", "schema_version", "error"),
    [
        (
            "onnx-splitpoint/unknown-command-contract",
            1,
            "unsupported_resume_command_contract_schema:"
            "onnx-splitpoint/unknown-command-contract",
        ),
        (
            SPLIT_COMMAND_CONTRACT_SCHEMA,
            2,
            "unsupported_resume_command_contract_schema_version:2",
        ),
    ],
)
def test_schema_and_version_fail_before_backend_role_selection(
    tmp_path: Path,
    schema: str,
    schema_version: int,
    error: str,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": schema,
        "schema_version": schema_version,
        "backend": "hailo8_to_trt",
        "artifacts": {},
    }

    with pytest.raises(ResumeArtifactContractError, match=error):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_hailo10_does_not_fall_back_to_hailo8_prepared_input_name(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": SPLIT_COMMAND_CONTRACT_SCHEMA,
        "backend": "hailo10h_to_trt",
        "runtime_options": {"part1_onnx_used": False},
        "prepared_input_contract": {
            "entries": [{
                "name": "images",
                "artifact_name": "prepared_input_00",
            }],
        },
        "artifacts": {
            "prepared_input": _artifact("prepared_input"),
            "semantic_boundary_manifest": _artifact(
                "semantic_boundary_manifest"
            ),
            "semantic_output_manifest": _artifact(
                "semantic_output_manifest"
            ),
        },
    }

    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_artifact_missing:part1_onnx",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_full_contract_includes_hef_and_runtime_inputs_but_not_tool_code(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    artifacts = {
        role: _artifact(role)
        for role in (
            "hef",
            "input_manifest",
            "performance_report",
            "runtime_input_tensor",
        )
    }
    artifacts["hotloop_runner"] = _artifact(
        "hotloop_runner", below_root=False,
    )
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_hailo10h",
        "artifacts": artifacts,
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
    }
    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    roles = {requirement.role for requirement in result["requirements"]}
    assert roles == {
        "hef",
        "input_manifest",
        "performance_report",
        "runtime_input_tensor",
        "input_image",
    }
    assert "hotloop_runner" not in roles


def test_deepx_full_contract_requires_dxnn_without_hef(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    artifacts = {
        role: _artifact(role)
        for role in (
            "dxnn",
            "input_manifest",
            "performance_report",
            "runtime_input_tensor",
        )
    }
    artifacts["hef"] = _artifact("hef", below_root=False)
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_deepx",
        "artifacts": artifacts,
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
    }

    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    assert {
        requirement.role for requirement in result["requirements"]
    } == {
        "dxnn",
        "input_manifest",
        "performance_report",
        "runtime_input_tensor",
        "input_image",
    }
    assert all(
        requirement.remote_path.startswith(f"{REMOTE_ROOT}/")
        for requirement in result["requirements"]
    )


@pytest.mark.parametrize(
    ("backend", "present_accelerator_role", "missing_role"),
    [
        ("native_full_deepx", "hef", "dxnn"),
        ("native_full_hailo8", "dxnn", "hef"),
        ("native_full_hailo10h", "dxnn", "hef"),
    ],
)
def test_full_contract_wrong_or_missing_accelerator_role_fails_closed(
    tmp_path: Path,
    backend: str,
    present_accelerator_role: str,
    missing_role: str,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": backend,
        "artifacts": {
            role: _artifact(role)
            for role in (
                present_accelerator_role,
                "input_manifest",
                "performance_report",
                "runtime_input_tensor",
            )
        },
    }

    with pytest.raises(
        ResumeArtifactContractError,
        match=rf"command_contract_artifact_missing:{missing_role}",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_unknown_full_backend_fails_closed_before_role_selection(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_unknown",
        "artifacts": {},
    }

    with pytest.raises(
        ResumeArtifactContractError,
        match="unsupported_resume_full_backend:native_full_unknown",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_deepx_full_dxnn_outside_frozen_root_is_rejected(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_deepx",
        "artifacts": {
            role: _artifact(role)
            for role in (
                "dxnn",
                "input_manifest",
                "performance_report",
                "runtime_input_tensor",
            )
        },
    }
    contract["artifacts"]["dxnn"] = _artifact(
        "dxnn", below_root=False,
    )

    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_artifact_outside_frozen_root:dxnn",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_deepx_full_missing_common_role_fails_closed(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_deepx",
        "artifacts": {
            role: _artifact(role)
            for role in (
                "dxnn",
                "input_manifest",
                "performance_report",
            )
        },
    }

    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_artifact_missing:runtime_input_tensor",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_tensorrt_full_preserves_legacy_restage_roles(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_tensorrt",
        "artifacts": {
            role: _artifact(role)
            for role in (
                "hef",
                "input_manifest",
                "performance_report",
                "runtime_input_tensor",
            )
        },
    }

    result = resume_artifact_requirements(
        _row(plan, contract),
        plan_root=plan,
        frozen_remote_root=REMOTE_ROOT,
    )

    assert {
        requirement.role for requirement in result["requirements"]
    } == {
        "hef",
        "input_manifest",
        "performance_report",
        "runtime_input_tensor",
    }


def test_contract_file_or_logical_hash_drift_is_rejected(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_hailo10h",
        "artifacts": {
            role: _artifact(role)
            for role in (
                "hef",
                "input_manifest",
                "performance_report",
                "runtime_input_tensor",
            )
        },
    }
    row = _row(plan, contract)
    row["command_contract_file_sha256"] = "0" * 64
    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_file_sha256_mismatch",
    ):
        resume_artifact_requirements(
            row,
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )

    row = _row(plan, contract)
    row["successful_command_contract_sha256"] = "1" * 64
    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_logical_sha256_mismatch",
    ):
        resume_artifact_requirements(
            row,
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


def test_contract_path_must_be_inside_plan_root(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    outside = tmp_path / "outside.json"
    raw = b'{"contract_sha256":"' + b"a" * 64 + b'"}'
    outside.write_bytes(raw)
    row = {
        "command_contract_file": str(outside),
        "command_contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "successful_command_contract_sha256": "a" * 64,
    }
    with pytest.raises(
        ResumeArtifactContractError,
        match="command_contract_outside_plan_root",
    ):
        resume_artifact_requirements(
            row,
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )


@pytest.mark.parametrize("field", ["hef", "input_image"])
def test_required_data_path_outside_frozen_run_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    contract = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "backend": "native_full_hailo10h",
        "artifacts": {
            role: _artifact(role)
            for role in (
                "hef",
                "input_manifest",
                "performance_report",
                "runtime_input_tensor",
            )
        },
        "input_image": f"{REMOTE_ROOT}/validation/image.jpg",
        "input_image_sha256": hashlib.sha256(b"image").hexdigest(),
    }
    if field == "hef":
        contract["artifacts"]["hef"] = _artifact(
            "hef", below_root=False,
        )
    else:
        contract["input_image"] = (
            "/home/nx/ONNX-Splitpoint-Tool/image.jpg"
        )

    with pytest.raises(
        ResumeArtifactContractError,
        match="outside_frozen_root",
    ):
        resume_artifact_requirements(
            _row(plan, contract),
            plan_root=plan,
            frozen_remote_root=REMOTE_ROOT,
        )
