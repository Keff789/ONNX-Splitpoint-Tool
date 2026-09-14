from __future__ import annotations

"""Load the narrow, contract-bound artifact set eligible for Energy resume.

The old remote runners and their command contracts remain authoritative.  This
module only selects data artifacts below the frozen EvaluationRun root that may
be restored before a later preflight.  Tool code, interpreters, system
executables, and cache paths outside that root are deliberately excluded.
"""

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .resume_artifact_rehydration import ArtifactRequirement


SPLIT_COMMAND_CONTRACT_SCHEMA = "onnx-splitpoint/native-command-contract"
FULL_COMMAND_CONTRACT_SCHEMA = "onnx-splitpoint/native-full-command-contract"
HAILO8_PYTHON_DETECTION_SOURCE_RECOVERY_PROFILE = (
    "hailo8_python_detection_literal_sources_v1"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_CONTRACT_BYTES = 64 * 1024 * 1024
_COMMAND_CONTRACT_SCHEMA_VERSIONS = {
    SPLIT_COMMAND_CONTRACT_SCHEMA: 1,
    FULL_COMMAND_CONTRACT_SCHEMA: 1,
}
_ROW_IDENTITY_FIELDS = ("backend", "model", "case", "setup_id")
_SPLIT_QUALITY_RESTAGE_ROLES = (
    "semantic_boundary_manifest",
    "semantic_output_manifest",
)
_HAILO8_PYTHON_DETECTION_BUILD_SOURCE_ROLES = (
    "cmake",
    "generated_cpp",
)
_HAILO8_PYTHON_RUNTIME_MODE = (
    "system_tensorrt_with_process_local_hailo_sites"
)
_HAILO8_PYTHON_RUNNER = (
    "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
)
_HAILO8_PYTHON_SITE_POLICY = (
    "site.addsitedir_after_system_defaults"
)
_HAILO8_PYTHON_DETECTION_SOURCE_SHA256 = {
    "cmake": (
        "a6734f5ccd8f08d2fabd2bdef17bbaa2a567e16a5e8e264ef0d0c128c582cae3"
    ),
    "generated_cpp": (
        "99dfa223ac4aa4ec2500ad4b5e032396e26abec197b7dd20e647eff314aa22ae"
    ),
}
_HAILO8_PYTHON_DETECTION_SOURCE_BASENAMES = {
    "cmake": "CMakeLists.txt",
    "generated_cpp": "main.cpp",
}
_SPLIT_RESTAGE_ROLES_BY_BACKEND = {
    # The real Hailo-8 contract binds a single prepared feed under this role.
    "hailo8_to_trt": (
        "prepared_input",
        *_SPLIT_QUALITY_RESTAGE_ROLES,
    ),
    # Preserve the already released v2.72.4 Hailo-10 Resume contract. Dynamic
    # Hailo-10 slot handling is intentionally outside this Hailo-8 hotfix.
    "hailo10h_to_trt": (
        "part1_onnx",
        "prepared_input_00",
        *_SPLIT_QUALITY_RESTAGE_ROLES,
    ),
    # DeepX energy replay consumes the exact prepared uint8 NumPy feed that
    # was sealed by the successful Native command contract.  DXNN/TRT cache
    # artifacts, interpreters and tool code remain outside this narrow
    # run-local restaging allow-list.
    "deepx_to_trt": (
        "prepared_input",
        *_SPLIT_QUALITY_RESTAGE_ROLES,
    ),
}
_FULL_COMMON_RESTAGE_ROLES = (
    "input_manifest",
    "performance_report",
    "runtime_input_tensor",
)
_FULL_RESTAGE_ROLES = (
    "hef",
    *_FULL_COMMON_RESTAGE_ROLES,
)
_FULL_RESTAGE_ROLES_BY_BACKEND = {
    "native_full_deepx": (
        "dxnn",
        *_FULL_COMMON_RESTAGE_ROLES,
    ),
    "native_full_hailo8": _FULL_RESTAGE_ROLES,
    "native_full_hailo10h": _FULL_RESTAGE_ROLES,
    # Preserve the released TensorRT Resume contract for legacy callers.
    "native_full_tensorrt": _FULL_RESTAGE_ROLES,
}


class ResumeArtifactContractError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        message = self.code
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


def _strict_sha256(value: Any, *, code: str) -> str:
    digest = str(value or "").strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if _SHA256_RE.fullmatch(digest) is None:
        raise ResumeArtifactContractError(code)
    return digest


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    duplicate = False

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeArtifactContractError(
            "command_contract_json_invalid",
            type(exc).__name__,
        ) from exc
    if duplicate or not isinstance(value, Mapping):
        raise ResumeArtifactContractError(
            "command_contract_json_invalid",
            "duplicate_keys_or_non_object",
        )
    return dict(value)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_bound_contract(
    row: Mapping[str, Any],
    *,
    plan_root: Path,
) -> tuple[Path, dict[str, Any]]:
    value = str(row.get("command_contract_file") or "").strip()
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ResumeArtifactContractError("command_contract_path_invalid")
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = plan_root / lexical
    if lexical.is_symlink():
        raise ResumeArtifactContractError(
            "command_contract_not_regular",
            str(lexical),
        )
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResumeArtifactContractError(
            "command_contract_not_regular",
            str(lexical),
        ) from exc
    if not _is_within(resolved, plan_root):
        raise ResumeArtifactContractError(
            "command_contract_outside_plan_root",
            str(resolved),
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ResumeArtifactContractError(
            "command_contract_not_regular",
            str(resolved),
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CONTRACT_BYTES
        ):
            raise ResumeArtifactContractError(
                "command_contract_not_regular",
                str(resolved),
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    ):
        raise ResumeArtifactContractError(
            "command_contract_changed_while_reading",
            str(resolved),
        )
    expected_file_sha = _strict_sha256(
        row.get("command_contract_file_sha256"),
        code="command_contract_file_sha256_invalid",
    )
    actual_file_sha = hashlib.sha256(raw).hexdigest()
    if actual_file_sha != expected_file_sha:
        raise ResumeArtifactContractError(
            "command_contract_file_sha256_mismatch",
            str(resolved),
        )
    contract = _strict_json_object(raw)
    expected_contract_sha = _strict_sha256(
        row.get("successful_command_contract_sha256"),
        code="successful_command_contract_sha256_invalid",
    )
    if _strict_sha256(
        contract.get("contract_sha256"),
        code="command_contract_logical_sha256_invalid",
    ) != expected_contract_sha:
        raise ResumeArtifactContractError(
            "command_contract_logical_sha256_mismatch",
            str(resolved),
        )
    return resolved, contract


def _normalise_remote_root(value: Any) -> str:
    raw = str(value or "").strip()
    if (
        not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or not raw.startswith("/")
        or raw.startswith("//")
    ):
        raise ResumeArtifactContractError("frozen_remote_root_invalid")
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or "." in path.parts
        or ".." in path.parts
        or str(path) != raw
        or len(path.parts) < 3
    ):
        raise ResumeArtifactContractError("frozen_remote_root_invalid")
    return str(path)


def _strict_child_remote_path(value: Any, *, remote_root: str) -> str | None:
    raw = str(value or "").strip()
    if (
        not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or not raw.startswith("/")
        or raw.startswith("//")
    ):
        return None
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or str(path) != raw
    ):
        return None
    try:
        relative = path.relative_to(PurePosixPath(remote_root))
    except ValueError:
        return None
    return str(path) if relative.parts else None


def _artifact_requirement(
    role: str,
    value: Mapping[str, Any],
    *,
    remote_root: str,
) -> ArtifactRequirement:
    remote_path = _strict_child_remote_path(
        value.get("path"),
        remote_root=remote_root,
    )
    if remote_path is None:
        raise ResumeArtifactContractError(
            "command_contract_artifact_outside_frozen_root",
            role,
        )
    raw_size = value.get("size_bytes")
    if raw_size in (None, ""):
        raw_size = value.get("bytes")
    if raw_size in (None, ""):
        size_bytes = None
    elif (
        isinstance(raw_size, bool)
        or not isinstance(raw_size, int)
        or raw_size <= 0
    ):
        raise ResumeArtifactContractError(
            "command_contract_artifact_size_invalid",
            role,
        )
    else:
        size_bytes = int(raw_size)
    return ArtifactRequirement(
        role=role,
        remote_path=remote_path,
        sha256=_strict_sha256(
            value.get("sha256"),
            code=f"command_contract_artifact_sha256_invalid:{role}",
        ),
        size_bytes=size_bytes,
    )


def _hailo8_python_detection_source_closure_bound(
    contract: Mapping[str, Any],
) -> bool:
    """Recognise the exact Python-VStreams detection contract shape.

    This is deliberately narrower than the generic Hailo-8 backend.  C++ split
    contracts can bind a native executable below the frozen run root and must
    not become restageable merely because the Python detection workload can
    reconstruct its generated source files.
    """
    options = contract.get("runtime_options")
    options = options if isinstance(options, Mapping) else {}
    interpreter = contract.get("interpreter_identity")
    interpreter = interpreter if isinstance(interpreter, Mapping) else {}
    mixed = contract.get("mixed_runtime_contract")
    mixed = mixed if isinstance(mixed, Mapping) else {}
    artifacts = contract.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    native = artifacts.get("native_executable")
    native = native if isinstance(native, Mapping) else {}
    python = artifacts.get("python_executable")
    python = python if isinstance(python, Mapping) else {}
    source_artifacts = {
        role: (
            artifacts.get(role)
            if isinstance(artifacts.get(role), Mapping) else {}
        )
        for role in _HAILO8_PYTHON_DETECTION_BUILD_SOURCE_ROLES
    }

    native_path = str(contract.get("native_executable") or "").strip()
    python_path = str(contract.get("python_executable") or "").strip()
    native_artifact_path = str(native.get("path") or "").strip()
    python_artifact_path = str(python.get("path") or "").strip()
    native_sha = str(native.get("sha256") or "").strip().lower()
    python_sha = str(python.get("sha256") or "").strip().lower()
    top_native_sha = str(
        contract.get("native_executable_sha256") or ""
    ).strip().lower()
    interpreter_path = str(
        interpreter.get("executable") or ""
    ).strip()
    interpreter_resolved = str(
        interpreter.get("resolved_executable") or ""
    ).strip()
    interpreter_sha = str(
        interpreter.get("executable_sha256") or ""
    ).strip().lower()
    mixed_python = str(
        mixed.get("python_executable") or ""
    ).strip()
    mixed_resolved = str(
        mixed.get("resolved_python_executable") or ""
    ).strip()
    option_sites = options.get("process_local_extra_sites")
    interpreter_sites = interpreter.get("process_local_extra_sites")
    mixed_sites = mixed.get("process_local_extra_sites")
    source_hashes_match = all(
        str(source_artifacts[role].get("sha256") or "")
        .strip()
        .lower()
        == expected_sha
        and PurePosixPath(
            str(source_artifacts[role].get("path") or "")
        ).name
        == _HAILO8_PYTHON_DETECTION_SOURCE_BASENAMES[role]
        for role, expected_sha
        in _HAILO8_PYTHON_DETECTION_SOURCE_SHA256.items()
    )

    return bool(
        str(contract.get("runner") or "").strip()
        == _HAILO8_PYTHON_RUNNER
        and _SHA256_RE.fullmatch(
            str(contract.get("runner_sha256") or "").strip().lower()
        ) is not None
        and str(options.get("producer_impl") or "").strip()
        == "hailo8_python_vstreams_fifo"
        and str(options.get("task") or "").strip().lower() == "detection"
        and options.get("prepared_input_bound") is True
        and options.get("energy_prepared_feed_capable") is True
        and str(interpreter.get("runtime_mode") or "").strip()
        == _HAILO8_PYTHON_RUNTIME_MODE
        and str(mixed.get("status") or "").strip() == "ready"
        and str(mixed.get("runtime_mode") or "").strip()
        == _HAILO8_PYTHON_RUNTIME_MODE
        and str(options.get("mixed_runtime_site_policy") or "").strip()
        == _HAILO8_PYTHON_SITE_POLICY
        and str(mixed.get("site_policy") or "").strip()
        == _HAILO8_PYTHON_SITE_POLICY
        and mixed.get("source_closure_ok") is True
        and isinstance(option_sites, list)
        and bool(option_sites)
        and option_sites == interpreter_sites
        and option_sites == mixed_sites
        and native_path
        and native_path == python_path
        and native_path == native_artifact_path
        and native_path == python_artifact_path
        and native_path == interpreter_path
        and native_path == interpreter_resolved
        and native_path == mixed_python
        and native_path == mixed_resolved
        and _SHA256_RE.fullmatch(native_sha) is not None
        and native_sha == python_sha
        and native_sha == top_native_sha
        and native_sha == interpreter_sha
        and source_hashes_match
    )


def _split_restage_roles(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Select immutable data roles from the sealed backend identity."""
    backend = str(contract.get("backend") or "").strip().lower()
    roles = _SPLIT_RESTAGE_ROLES_BY_BACKEND.get(backend)
    if roles is None:
        raise ResumeArtifactContractError(
            "unsupported_resume_split_backend",
            backend or "<missing>",
        )
    if backend == "deepx_to_trt":
        options = contract.get("runtime_options")
        options = options if isinstance(options, Mapping) else {}
        prepared = contract.get("prepared_input_contract")
        prepared = prepared if isinstance(prepared, Mapping) else {}
        shape = prepared.get("shape")
        artifacts = contract.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, Mapping) else {}
        if (
            options.get("prepared_input_bound") is not True
            or str(prepared.get("format") or "") != "numpy_npy_v1"
            or not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
            or str(prepared.get("dtype") or "") != "uint8"
            or prepared.get("c_contiguous") is not True
            or not isinstance(artifacts.get("prepared_input"), Mapping)
        ):
            raise ResumeArtifactContractError(
                "unsupported_resume_deepx_contract_profile"
            )
    if (
        backend == "hailo8_to_trt"
        and _hailo8_python_detection_source_closure_bound(contract)
    ):
        return (*roles, *_HAILO8_PYTHON_DETECTION_BUILD_SOURCE_ROLES)
    return roles


def _p03_technical_energy_admission(row: Mapping[str, Any]) -> bool:
    """Recognise a frozen P0.3 Split row before omitting annotations."""

    admission = row.get("native_energy_planner_admission")
    if not isinstance(admission, Mapping):
        return False
    return bool(
        admission.get("schema")
        == "onnx-splitpoint/native-energy-planner-admission"
        and admission.get("schema_version") == 1
        and admission.get("selected") is True
        and admission.get("runtime_success") is True
        and admission.get("energy_command_preflight_ok") is True
        and admission.get("full_baseline") is False
        and admission.get("split_has_valid_part2_input") is True
        and row.get("runtime_success") is True
        and row.get("energy_command_preflight_ok") is True
        and row.get("full_baseline") is False
        and row.get("split_has_valid_part2_input") is True
    )


def _restage_roles(
    schema: str,
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    if schema == SPLIT_COMMAND_CONTRACT_SCHEMA:
        return _split_restage_roles(contract)
    if schema == FULL_COMMAND_CONTRACT_SCHEMA:
        backend = str(contract.get("backend") or "").strip().lower()
        roles = _FULL_RESTAGE_ROLES_BY_BACKEND.get(backend)
        if roles is None:
            raise ResumeArtifactContractError(
                "unsupported_resume_full_backend",
                backend or "<missing>",
            )
        return roles
    raise ResumeArtifactContractError(
        "unsupported_resume_command_contract_schema",
        schema,
    )


def _validate_contract_row_binding(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    schema: str,
) -> None:
    expected_version = _COMMAND_CONTRACT_SCHEMA_VERSIONS.get(schema)
    if expected_version is None:
        raise ResumeArtifactContractError(
            "unsupported_resume_command_contract_schema",
            schema or "<missing>",
        )
    if (
        isinstance(contract.get("schema_version"), bool)
        or contract.get("schema_version") != expected_version
    ):
        raise ResumeArtifactContractError(
            "unsupported_resume_command_contract_schema_version",
            str(contract.get("schema_version")),
        )
    for field in _ROW_IDENTITY_FIELDS:
        selected = str(row.get(field) or "").strip()
        archived = str(contract.get(field) or "").strip()
        if not selected or archived != selected:
            raise ResumeArtifactContractError(
                "command_contract_row_identity_mismatch",
                field,
            )


def resume_artifact_requirements(
    row: Mapping[str, Any],
    *,
    plan_root: str | Path,
    frozen_remote_root: str,
) -> dict[str, Any]:
    """Return the narrow restage requirements for one selected plan row."""
    plan = Path(plan_root).expanduser().resolve(strict=True)
    if plan.is_symlink() or not plan.is_dir():
        raise ResumeArtifactContractError("plan_root_not_directory")
    remote_root = _normalise_remote_root(frozen_remote_root)
    contract_path, contract = _read_bound_contract(row, plan_root=plan)
    schema = str(contract.get("schema") or "").strip()
    _validate_contract_row_binding(row, contract, schema=schema)
    hailo8_python_detection_recovery = bool(
        schema == SPLIT_COMMAND_CONTRACT_SCHEMA
        and str(contract.get("backend") or "").strip().lower()
        == "hailo8_to_trt"
        and _hailo8_python_detection_source_closure_bound(contract)
    )
    roles = _restage_roles(schema, contract)
    omitted_downstream_annotation_roles: tuple[str, ...] = ()
    if (
        schema == SPLIT_COMMAND_CONTRACT_SCHEMA
        and _p03_technical_energy_admission(row)
    ):
        omitted_downstream_annotation_roles = tuple(
            role for role in roles
            if role in _SPLIT_QUALITY_RESTAGE_ROLES
        )
        roles = tuple(
            role for role in roles
            if role not in _SPLIT_QUALITY_RESTAGE_ROLES
        )
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ResumeArtifactContractError(
            "command_contract_artifacts_missing"
        )
    requirements: list[ArtifactRequirement] = []
    for role in roles:
        value = artifacts.get(role)
        if not isinstance(value, Mapping):
            raise ResumeArtifactContractError(
                "command_contract_artifact_missing",
                role,
            )
        requirement = _artifact_requirement(
            role,
            value,
            remote_root=remote_root,
        )
        requirements.append(requirement)

    raw_input_path = str(contract.get("input_image") or "").strip()
    if raw_input_path:
        input_path = _strict_child_remote_path(
            raw_input_path,
            remote_root=remote_root,
        )
        if input_path is None:
            raise ResumeArtifactContractError(
                "command_contract_input_image_outside_frozen_root"
            )
        requirements.append(
            ArtifactRequirement(
                role="input_image",
                remote_path=input_path,
                sha256=_strict_sha256(
                    contract.get("input_image_sha256"),
                    code="command_contract_input_image_sha256_invalid",
                ),
                size_bytes=None,
            )
        )
    if not requirements:
        raise ResumeArtifactContractError(
            "resume_artifact_requirements_empty"
        )
    return {
        "schema": "onnx-splitpoint/resume-artifact-requirements",
        "schema_version": 1,
        "command_contract_schema": schema,
        "command_contract_file": str(contract_path),
        "command_contract_file_sha256": str(
            row.get("command_contract_file_sha256") or ""
        ).strip().lower(),
        "successful_command_contract_sha256": str(
            row.get("successful_command_contract_sha256") or ""
        ).strip().lower(),
        "frozen_remote_root": remote_root,
        "source_recovery_profile": (
            HAILO8_PYTHON_DETECTION_SOURCE_RECOVERY_PROFILE
            if hailo8_python_detection_recovery else ""
        ),
        "source_recovery_roles": (
            list(_HAILO8_PYTHON_DETECTION_BUILD_SOURCE_ROLES)
            if hailo8_python_detection_recovery else []
        ),
        "downstream_annotation_roles_omitted": list(
            omitted_downstream_annotation_roles
        ),
        "requirements": requirements,
    }


__all__ = [
    "FULL_COMMAND_CONTRACT_SCHEMA",
    "HAILO8_PYTHON_DETECTION_SOURCE_RECOVERY_PROFILE",
    "ResumeArtifactContractError",
    "SPLIT_COMMAND_CONTRACT_SCHEMA",
    "resume_artifact_requirements",
]
