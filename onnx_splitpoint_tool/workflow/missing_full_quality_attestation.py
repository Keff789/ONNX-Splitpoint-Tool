from __future__ import annotations

"""Read-only attestations for the narrow missing-Full-quality resume path.

Some workflow artifacts are intentionally finalized by a later stage.  Their
creation-stage hash therefore no longer describes the current bytes even
though the later bytes are legitimate.  This module recognizes only the two
known alias families required by the targeted Full-quality repair:

* the formal Full-baseline output contract mirrored by the suite contract;
* the four formal/executable BenchmarkSet plan/set aliases finalized before
  runtime dispatch and anchored by the immutable legacy copies.

The helpers never create, replace, chmod, or register files.  Every path is
resolved below one run root, symlinks are rejected, current bytes are read from
one file descriptor, and Artifact Index bindings always cover the current
size and SHA-256 of the exact logical path.

The sole historical missing-file exception is also attested here.  Early
Full-only remote dispatches listed a management-side ``stderr`` artifact on
their successful path without ever creating it.  That omission is accepted
only for the two already completed Phase-5 classification dispatches and only
when their exact success evidence is still current and index-bound and the
missing path has never had an Artifact Index record.
"""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..validation.accuracy_gates import AccuracyGatePolicy


ARTIFACT_INDEX_SCHEMA = "onnx-splitpoint/artifact-index"
OUTPUT_CONTRACTS_SCHEMA = "onnx-splitpoint/output-contracts"
BENCHMARK_PLAN_SCHEMA = "onnx-splitpoint/benchmark-plan"
BENCHMARK_SET_SCHEMA = "onnx-splitpoint/benchmark-set"
FORMAL_BENCHMARK_SET_SCHEMA = "onnx-splitpoint/benchmark-set-contract"

_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/missing-full-quality-downstream-supersession-attestation"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_INDEX_ROLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_MAX_ATTESTED_FILE_BYTES = 128 * 1024 * 1024
_TASKS = {"classification", "detection"}
_FINAL_TASK_FIELDS = ("task", "model_task", "benchmark_task")
_HISTORICAL_SUCCESS_STDERR_MODELS = {
    "mobilenet_v3_large",
    "regnet_x_1_6gf",
}
_HISTORICAL_SUCCESS_STDERR_SETUP = "orin_nx_hailo8_01"
_STABLE_RUN_FIELDS = (
    "type",
    "kind",
    "backend",
    "provider",
    "variant",
    "variants",
    "case_id",
    "case",
    "setup_id",
    "execution_role",
    "performance_claims_emitted",
)


class MissingFullQualityAttestationError(ValueError):
    """A read-only supersession attestation failed closed."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        message = self.code
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class _CurrentFile:
    logical_path: str
    resolved_path: Path
    size_bytes: int
    sha256: str
    raw: bytes

    def public(self) -> Dict[str, Any]:
        return {
            "path": self.logical_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _fail(code: str, detail: Any = "") -> None:
    raise MissingFullQualityAttestationError(code, str(detail or ""))


def _model_id(value: Any) -> str:
    token = str(value or "").strip().lower()
    if (
        not token
        or token in {".", ".."}
        or _MODEL_ID_RE.fullmatch(token) is None
    ):
        _fail("model_id_invalid", token)
    return token


def _task(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token not in _TASKS:
        _fail("task_invalid", token)
    return token


def _sha256_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    if _SHA256_RE.fullmatch(token) is None:
        _fail("sha256_invalid", token)
    return token


def _logical_path(value: str | Path) -> PurePosixPath:
    raw = str(value or "").strip()
    if (
        not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or "\\" in raw
    ):
        _fail("logical_path_invalid", raw)
    logical = PurePosixPath(raw)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        _fail("logical_path_invalid", raw)
    return logical


def _read_current_file(
    run_dir: str | Path,
    logical_path: str | Path,
    *,
    max_bytes: int = _MAX_ATTESTED_FILE_BYTES,
) -> _CurrentFile:
    root_lexical = Path(run_dir).expanduser()
    try:
        root = root_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MissingFullQualityAttestationError(
            "run_root_invalid", str(root_lexical),
        ) from exc
    if not root.is_dir():
        _fail("run_root_invalid", root)

    logical = _logical_path(logical_path)
    lexical = root
    for part in logical.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            _fail("attested_path_symlink_forbidden", logical.as_posix())
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MissingFullQualityAttestationError(
            "attested_path_missing_or_outside_run", logical.as_posix(),
        ) from exc

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(resolved), flags)
    except OSError as exc:
        raise MissingFullQualityAttestationError(
            "attested_path_not_regular", logical.as_posix(),
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > int(max_bytes)
        ):
            _fail("attested_path_not_regular", logical.as_posix())
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if len(raw) != before.st_size or before_identity != after_identity:
        _fail("attested_path_changed_while_reading", logical.as_posix())
    return _CurrentFile(
        logical_path=logical.as_posix(),
        resolved_path=resolved,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        raw=raw,
    )


def _strict_json_object(current: _CurrentFile) -> Dict[str, Any]:
    duplicate = False

    def _pairs(items: List[Tuple[str, Any]]) -> Dict[str, Any]:
        nonlocal duplicate
        value: Dict[str, Any] = {}
        for key, item in items:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    try:
        payload = json.loads(
            current.raw.decode("utf-8"), object_pairs_hook=_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissingFullQualityAttestationError(
            "attested_json_invalid", current.logical_path,
        ) from exc
    if duplicate or not isinstance(payload, Mapping):
        _fail("attested_json_invalid", current.logical_path)
    return dict(payload)


def _artifact_rows(artifact_index: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if not isinstance(artifact_index, Mapping):
        _fail("artifact_index_invalid")
    schema = str(artifact_index.get("schema") or "").strip()
    if schema != ARTIFACT_INDEX_SCHEMA:
        _fail("artifact_index_schema_invalid", schema)
    rows = artifact_index.get("artifacts")
    if not isinstance(rows, list):
        _fail("artifact_index_invalid", "artifacts_not_list")
    validated: List[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        detail = f"record={index}"
        if not isinstance(row, Mapping):
            _fail("artifact_index_record_invalid", detail + ":not_mapping")

        raw_path = row.get("path")
        if not isinstance(raw_path, str) or raw_path != raw_path.strip():
            _fail("artifact_index_record_invalid", detail + ":path")
        try:
            canonical_path = _logical_path(raw_path).as_posix()
        except MissingFullQualityAttestationError as exc:
            raise MissingFullQualityAttestationError(
                "artifact_index_record_invalid", detail + ":path",
            ) from exc
        if canonical_path != raw_path:
            _fail("artifact_index_record_invalid", detail + ":path")

        for key in ("kind", "producer_stage"):
            raw_role = row.get(key)
            if (
                not isinstance(raw_role, str)
                or raw_role != raw_role.strip()
                or _INDEX_ROLE_RE.fullmatch(raw_role) is None
            ):
                _fail("artifact_index_record_invalid", detail + f":{key}")

        raw_model = row.get("model_id")
        if raw_model is not None and raw_model != "":
            if not isinstance(raw_model, str):
                _fail("artifact_index_record_invalid", detail + ":model_id")
            try:
                canonical_model = _model_id(raw_model)
            except MissingFullQualityAttestationError as exc:
                raise MissingFullQualityAttestationError(
                    "artifact_index_record_invalid", detail + ":model_id",
                ) from exc
            if canonical_model != raw_model:
                _fail("artifact_index_record_invalid", detail + ":model_id")

        raw_size = row.get("size_bytes")
        if type(raw_size) is not int or raw_size < 0:
            _fail("artifact_index_record_invalid", detail + ":size_bytes")
        raw_sha = row.get("sha256")
        if (
            not isinstance(raw_sha, str)
            or raw_sha != "sha256:" + raw_sha[7:].lower()
            or len(raw_sha) != 71
            or _SHA256_RE.fullmatch(raw_sha[7:]) is None
        ):
            _fail("artifact_index_record_invalid", detail + ":sha256")
        validated.append(row)
    return validated


def _matching_index_rows(
    current: _CurrentFile,
    artifact_index: Mapping[str, Any],
    *,
    expected_model_id: Optional[str] = None,
    allowed_producer_stages: Sequence[str] = (),
    current_bytes_required: bool = True,
) -> List[Mapping[str, Any]]:
    model = (
        _model_id(expected_model_id)
        if expected_model_id is not None and expected_model_id != ""
        else ""
    )
    require_model_identity = expected_model_id is not None
    stages = {str(value or "").strip() for value in allowed_producer_stages}
    matches: List[Mapping[str, Any]] = []
    for row in _artifact_rows(artifact_index):
        row_path = row["path"]
        if row_path != current.logical_path:
            continue
        if (
            require_model_identity
            and str(row.get("model_id") or "") != model
        ):
            continue
        if stages and row["producer_stage"] not in stages:
            continue
        if current_bytes_required:
            row_size = row["size_bytes"]
            row_sha = row["sha256"][7:]
            if row_size != current.size_bytes or row_sha != current.sha256:
                continue
        matches.append(row)
    return matches


def attest_current_path(
    *,
    run_dir: str | Path,
    path: str | Path,
    expected_size_bytes: Optional[int] = None,
    expected_sha256: str = "",
) -> Dict[str, Any]:
    """Attest one current regular file without consulting the Artifact Index."""

    current = _read_current_file(run_dir, path)
    if expected_size_bytes is not None and current.size_bytes != int(expected_size_bytes):
        _fail("current_path_size_mismatch", current.logical_path)
    if expected_sha256 and current.sha256 != _sha256_token(expected_sha256):
        _fail("current_path_sha256_mismatch", current.logical_path)
    return {
        "schema": _ATTESTATION_SCHEMA,
        "schema_version": 1,
        "scope": "current_path",
        "ok": True,
        **current.public(),
    }


def attest_current_index_binding(
    *,
    run_dir: str | Path,
    path: str | Path,
    artifact_index: Mapping[str, Any],
    expected_model_id: Optional[str] = None,
    allowed_producer_stages: Sequence[str] = (),
) -> Dict[str, Any]:
    """Attest current size/SHA against an exact-path Artifact Index row."""

    current = _read_current_file(run_dir, path)
    matches = _matching_index_rows(
        current,
        artifact_index,
        expected_model_id=expected_model_id,
        allowed_producer_stages=allowed_producer_stages,
    )
    if not matches:
        _fail("current_path_not_index_bound", current.logical_path)
    return {
        "schema": _ATTESTATION_SCHEMA,
        "schema_version": 1,
        "scope": "current_index_binding",
        "ok": True,
        **current.public(),
        "matching_index_record_count": len(matches),
    }


def _attest_absent_logical_path(
    run_dir: str | Path,
    logical_path: str | Path,
) -> str:
    """Prove that one contained logical path does not exist in any form."""

    root_lexical = Path(run_dir).expanduser()
    try:
        root = root_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MissingFullQualityAttestationError(
            "run_root_invalid", str(root_lexical),
        ) from exc
    if not root.is_dir():
        _fail("run_root_invalid", root)

    logical = _logical_path(logical_path)
    cursor = root
    for part in logical.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(
                "historical_omission_parent_symlink_forbidden",
                logical.as_posix(),
            )
        try:
            resolved = cursor.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise MissingFullQualityAttestationError(
                "historical_omission_parent_invalid", logical.as_posix(),
            ) from exc
        if not resolved.is_dir():
            _fail("historical_omission_parent_invalid", logical.as_posix())

    lexical = cursor / logical.parts[-1]
    if lexical.is_symlink() or os.path.lexists(lexical):
        _fail("historical_omission_path_not_absent", logical.as_posix())
    return logical.as_posix()


def attest_historical_success_stderr_omission(
    *,
    run_dir: str | Path,
    model_id: str,
    artifact_index: Mapping[str, Any],
    setup_id: str = _HISTORICAL_SUCCESS_STDERR_SETUP,
) -> Dict[str, Any]:
    """Attest the exact successful-dispatch stderr producer omission.

    A missing path is never inferred to be harmless merely from an ``ok``
    stage.  The path must be the known setup-local stderr for one of the two
    completed classification models, must never have had an Artifact Index
    row, and its stage, dispatch, status and stdout authorities must all still
    match their exact current ``run_benchmarks`` bindings.
    """

    model = _model_id(model_id)
    setup = str(setup_id or "").strip().lower()
    if model not in _HISTORICAL_SUCCESS_STDERR_MODELS:
        _fail("historical_stderr_model_not_allowed", model)
    if setup != _HISTORICAL_SUCCESS_STDERR_SETUP:
        _fail("historical_stderr_setup_not_allowed", setup)

    try:
        expected_run_id = Path(run_dir).expanduser().resolve(strict=True).name
    except (OSError, RuntimeError) as exc:
        raise MissingFullQualityAttestationError(
            "run_root_invalid", str(run_dir),
        ) from exc
    if str(artifact_index.get("run_id") or "").strip() != expected_run_id:
        _fail("artifact_index_run_id_mismatch", expected_run_id)

    base = PurePosixPath("models") / model
    results = base / "benchmark_results"
    stage_logical = (
        base / "stages" / "run_benchmarks" / "stage_result.json"
    ).as_posix()
    status_logical = (
        results / f"remote_benchmark_status_{setup}.json"
    ).as_posix()
    dispatch_logical = (
        results / f"remote_benchmark_dispatch_{setup}.json"
    ).as_posix()
    stdout_logical = (
        results / f"remote_benchmark_stdout_{setup}.txt"
    ).as_posix()
    stderr_logical = (
        results / f"remote_benchmark_stderr_{setup}.txt"
    ).as_posix()
    matrix_logical = (
        results / "remote_hardware_matrix_status.json"
    ).as_posix()
    normalized_logical = (results / "normalized_results.json").as_posix()

    omitted = _attest_absent_logical_path(run_dir, stderr_logical)
    index_rows = _artifact_rows(artifact_index)
    if any(str(row.get("path") or "") == omitted for row in index_rows):
        _fail("historical_stderr_has_index_record", omitted)

    current = {
        "stage_result": _read_current_file(run_dir, stage_logical),
        "status": _read_current_file(run_dir, status_logical),
        "dispatch": _read_current_file(run_dir, dispatch_logical),
        "stdout": _read_current_file(run_dir, stdout_logical),
        "matrix": _read_current_file(run_dir, matrix_logical),
        "normalized": _read_current_file(run_dir, normalized_logical),
    }
    for value in current.values():
        if not _matching_index_rows(
            value,
            artifact_index,
            expected_model_id=model,
            allowed_producer_stages=("run_benchmarks",),
        ):
            _fail(
                "historical_stderr_authority_not_index_bound",
                value.logical_path,
            )

    stage = _strict_json_object(current["stage_result"])
    artifacts = stage.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(value, str) for value in artifacts
    ):
        _fail("historical_stderr_stage_artifacts_invalid", stage_logical)
    required_once = (
        stage_logical,
        status_logical,
        dispatch_logical,
        stdout_logical,
        stderr_logical,
        matrix_logical,
        normalized_logical,
    )
    if any(artifacts.count(value) != 1 for value in required_once):
        _fail("historical_stderr_stage_artifact_set_invalid", stage_logical)
    details = stage.get("details")
    if not isinstance(details, Mapping):
        _fail("historical_stderr_stage_details_invalid", stage_logical)
    if (
        str(stage.get("stage") or "") != "run_benchmarks"
        or str(stage.get("model_id") or "").strip().lower() != model
        or str(stage.get("status") or "").strip().lower() != "ok"
        or str(stage.get("state") or "").strip().lower() != "completed"
        or stage.get("complete") is not True
        or details.get("quality_evidence_only_complete") is not True
        or type(details.get("quality_evidence_count")) is not int
        or details.get("quality_evidence_count") != 2
        or type(details.get("expected_full_quality_count")) is not int
        or details.get("expected_full_quality_count") != 2
    ):
        _fail("historical_stderr_stage_success_invalid", stage_logical)

    status_payload = _strict_json_object(current["status"])
    evidence = status_payload.get("full_only_quality_evidence")
    remote_output = status_payload.get("remote_output")
    if not isinstance(evidence, Mapping) or not isinstance(remote_output, Mapping):
        _fail("historical_stderr_remote_status_invalid", status_logical)
    if (
        str(status_payload.get("schema") or "")
        != "onnx-splitpoint/remote-benchmark-status"
        or str(status_payload.get("model_id") or "").strip().lower() != model
        or str(status_payload.get("hardware_target_id") or "").strip().lower()
        != setup
        or str(status_payload.get("status") or "").strip().lower() != "ok"
        or str(status_payload.get("reason") or "")
        != "remote_full_only_quality_evidence_verified"
        or str(status_payload.get("stdout_path") or "") != stdout_logical
        or str(status_payload.get("stderr_path") or "") != stderr_logical
        or type(status_payload.get("copied_result_count")) is not int
        or status_payload.get("copied_result_count") <= 0
        or type(status_payload.get("canonical_nonempty_row_count")) is not int
        or status_payload.get("canonical_nonempty_row_count") != 0
        or type(status_payload.get("quality_evidence_count")) is not int
        or status_payload.get("quality_evidence_count") != 2
        or evidence.get("requested") is not True
        or str(evidence.get("status") or "") != "verified_exact"
        or type(evidence.get("expected_count")) is not int
        or evidence.get("expected_count") != 2
        or type(evidence.get("quality_evidence_count")) is not int
        or evidence.get("quality_evidence_count") != 2
        or evidence.get("errors") != []
        or remote_output.get("ok") is not True
        or str(remote_output.get("status") or "").strip().lower() != "ok"
    ):
        _fail("historical_stderr_remote_success_invalid", status_logical)

    evidence_rows = evidence.get("evidence")
    if not isinstance(evidence_rows, list) or len(evidence_rows) != 2:
        _fail("historical_stderr_quality_evidence_invalid", status_logical)
    expected_evidence = {
        (
            model,
            "hailo8",
            setup,
            "hailo8",
            "full",
            "full_quality_only",
            False,
            "hailo8_full",
        ),
        (
            model,
            "native_full_tensorrt",
            setup,
            "tensorrt",
            "full",
            "full_quality_only",
            False,
            "tensorrt_at_hailo8_full",
        ),
    }
    observed_evidence: set[tuple[Any, ...]] = set()
    public_evidence: List[Dict[str, Any]] = []
    expected_eval_run_id = Path(run_dir).expanduser().resolve(strict=True).name
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            _fail("historical_stderr_quality_evidence_invalid", status_logical)
        identity = (
            str(row.get("model_id") or "").strip().lower(),
            str(row.get("source_run_id") or "").strip().lower(),
            str(row.get("setup_id") or "").strip().lower(),
            str(row.get("backend") or "").strip().lower(),
            str(row.get("variant") or "").strip().lower(),
            str(row.get("execution_role") or "").strip().lower(),
            row.get("performance_claims_emitted"),
            str(row.get("quality_canary_id") or "").strip(),
        )
        request_sha = _sha256_token(row.get("request_sha256"))
        candidate_sha = _sha256_token(row.get("candidate_sha256"))
        request_path = str(row.get("request_path") or "").strip()
        candidate_path = str(row.get("candidate_path") or "").strip()
        if (
            identity in observed_evidence
            or str(row.get("eval_run_id") or "").strip()
            != expected_eval_run_id
            or not request_path
            or not candidate_path
        ):
            _fail("historical_stderr_quality_evidence_invalid", status_logical)
        observed_evidence.add(identity)
        public_evidence.append({
            "model_id": identity[0],
            "source_run_id": identity[1],
            "setup_id": identity[2],
            "backend": identity[3],
            "quality_canary_id": identity[7],
            "request_sha256": request_sha,
            "candidate_sha256": candidate_sha,
        })
    if observed_evidence != expected_evidence:
        _fail("historical_stderr_quality_evidence_set_mismatch", status_logical)

    dispatch_payload = _strict_json_object(current["dispatch"])
    host = dispatch_payload.get("host")
    if (
        str(dispatch_payload.get("schema") or "")
        != "onnx-splitpoint/remote-benchmark-dispatch"
        or str(dispatch_payload.get("model_id") or "").strip().lower() != model
        or str(dispatch_payload.get("hardware_target_id") or "").strip().lower()
        != setup
        or str(dispatch_payload.get("status") or "") != "dispatching"
        or not str(dispatch_payload.get("run_id") or "").strip()
        or not isinstance(host, Mapping)
        or not str(host.get("host") or "").strip()
    ):
        _fail("historical_stderr_dispatch_invalid", dispatch_logical)

    matrix_payload = _strict_json_object(current["matrix"])
    if (
        str(matrix_payload.get("schema") or "")
        != "onnx-splitpoint/remote-hardware-matrix-status"
        or str(matrix_payload.get("model_id") or "").strip().lower() != model
        or str(matrix_payload.get("status") or "").strip().lower() != "ok"
        or matrix_payload.get("remote_dispatched") is not True
        or matrix_payload.get("remote_dispatch_failed") is not False
        or type(matrix_payload.get("quality_evidence_count")) is not int
        or matrix_payload.get("quality_evidence_count") != 2
        or type(matrix_payload.get("expected_full_quality_count")) is not int
        or matrix_payload.get("expected_full_quality_count") != 2
    ):
        _fail("historical_stderr_matrix_invalid", matrix_logical)

    normalized_payload = _strict_json_object(current["normalized"])
    if (
        str(normalized_payload.get("schema") or "")
        != "onnx-splitpoint/normalized-benchmark-results"
        or str(normalized_payload.get("model_id") or "").strip().lower()
        != model
        or str(normalized_payload.get("status") or "")
        != "quality_evidence_only_complete"
        or normalized_payload.get("quality_evidence_only_complete") is not True
        or normalized_payload.get("performance_matrix_applicable") is not False
        or normalized_payload.get("matrix_complete") is not True
        or type(normalized_payload.get("quality_evidence_count")) is not int
        or normalized_payload.get("quality_evidence_count") != 2
        or type(normalized_payload.get("expected_full_quality_count")) is not int
        or normalized_payload.get("expected_full_quality_count") != 2
        or type(normalized_payload.get("result_count")) is not int
        or normalized_payload.get("result_count") != 0
    ):
        _fail("historical_stderr_normalized_invalid", normalized_logical)

    return {
        "schema": _ATTESTATION_SCHEMA,
        "schema_version": 1,
        "scope": "historical_success_stderr_producer_omission",
        "ok": True,
        "model_id": model,
        "setup_id": setup,
        "source_stage": "run_benchmarks",
        "path": omitted,
        "absence_kind": "never_indexed_success_stderr",
        "authorities": {
            key: value.public() for key, value in current.items()
        },
        "evidence": sorted(
            public_evidence,
            key=lambda row: str(row.get("source_run_id") or ""),
        ),
        "quality_evidence_count": 2,
        "expected_full_quality_count": 2,
    }


def _require_historical_index_role(
    *,
    current: _CurrentFile,
    artifact_index: Mapping[str, Any],
    model_id: str,
    producer_stage: str,
) -> None:
    matches = _matching_index_rows(
        current,
        artifact_index,
        expected_model_id=model_id,
        allowed_producer_stages=(producer_stage,),
        current_bytes_required=False,
    )
    if not matches:
        _fail("artifact_historical_role_unbound", current.logical_path)


def _require_schema(
    payload: Mapping[str, Any], expected: str, logical_path: str,
) -> None:
    if str(payload.get("schema") or "").strip() != expected:
        _fail("artifact_schema_mismatch", logical_path)


def _require_optional_identity(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected: str,
    logical_path: str,
) -> None:
    if key in payload and str(payload.get(key) or "").strip().lower() != expected:
        _fail(f"artifact_{key}_mismatch", logical_path)


def attest_prepare_full_baselines_supersession(
    *,
    run_dir: str | Path,
    model_id: str,
    artifact_index: Mapping[str, Any],
    suite_contract_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Attest the sole legal ``prepare_full_baselines`` supersession.

    Only ``models/<model>/full_baselines/output_contracts.json`` is eligible.
    Its current bytes must be identical to a currently index-bound suite
    ``output_contracts.json`` for the same model.
    """

    model = _model_id(model_id)
    base = PurePosixPath("models") / model
    formal_logical = base / "full_baselines" / "output_contracts.json"
    suite_logical = (
        _logical_path(suite_contract_path)
        if suite_contract_path is not None
        else base / "benchmark_set" / "legacy_suite" / "output_contracts.json"
    )
    suite_parent = base / "benchmark_set"
    try:
        suite_logical.relative_to(suite_parent)
    except ValueError as exc:
        raise MissingFullQualityAttestationError(
            "suite_output_contract_path_invalid", suite_logical.as_posix(),
        ) from exc
    if suite_logical.name != "output_contracts.json":
        _fail("suite_output_contract_path_invalid", suite_logical.as_posix())

    formal = _read_current_file(run_dir, formal_logical.as_posix())
    suite = _read_current_file(run_dir, suite_logical.as_posix())
    _require_historical_index_role(
        current=formal,
        artifact_index=artifact_index,
        model_id=model,
        producer_stage="prepare_full_baselines",
    )
    suite_matches = _matching_index_rows(
        suite,
        artifact_index,
        expected_model_id=model,
        allowed_producer_stages=(
            "build_backend_artifacts", "run_benchmarks",
        ),
    )
    if not suite_matches:
        _fail("suite_output_contract_not_currently_index_bound", suite.logical_path)
    if (
        formal.size_bytes != suite.size_bytes
        or formal.sha256 != suite.sha256
        or formal.raw != suite.raw
    ):
        _fail("output_contract_alias_bytes_mismatch", formal.logical_path)

    payload = _strict_json_object(formal)
    _require_schema(payload, OUTPUT_CONTRACTS_SCHEMA, formal.logical_path)
    if str(payload.get("model_id") or "").strip().lower() != model:
        _fail("artifact_model_id_mismatch", formal.logical_path)
    task = _task(payload.get("task"))
    return {
        "schema": _ATTESTATION_SCHEMA,
        "schema_version": 1,
        "scope": "prepare_full_baselines_output_contract_supersession",
        "ok": True,
        "model_id": model,
        "task": task,
        "formal_contract": formal.public(),
        "suite_contract": suite.public(),
        "suite_matching_index_record_count": len(suite_matches),
        "byte_identical": True,
        "historical_producer_stage": "prepare_full_baselines",
        "current_authority_producer_stages": sorted({
            str(row.get("producer_stage") or "") for row in suite_matches
        }),
    }


def _payload_runs(
    payload: Mapping[str, Any],
    *,
    logical_path: str,
    require_alias_pair: bool,
) -> List[Dict[str, Any]]:
    raw_runs = payload.get("runs")
    raw_planned = payload.get("planned_runs")
    if not isinstance(raw_runs, list):
        _fail("benchmark_runs_invalid", logical_path)
    runs = [dict(row) for row in raw_runs if isinstance(row, Mapping)]
    if len(runs) != len(raw_runs):
        _fail("benchmark_runs_invalid", logical_path)
    if require_alias_pair:
        if not isinstance(raw_planned, list):
            _fail("benchmark_planned_runs_invalid", logical_path)
        planned = [dict(row) for row in raw_planned if isinstance(row, Mapping)]
        if len(planned) != len(raw_planned) or planned != runs:
            _fail("benchmark_run_alias_mismatch", logical_path)
    elif isinstance(raw_planned, list):
        planned = [dict(row) for row in raw_planned if isinstance(row, Mapping)]
        if len(planned) != len(raw_planned) or planned != runs:
            _fail("immutable_benchmark_run_alias_mismatch", logical_path)
    return runs


def _run_token(row: Mapping[str, Any]) -> str:
    token = str(row.get("id") or row.get("run_id") or "").strip().lower()
    token = token.replace("-", "_")
    provider = str(row.get("provider") or "").strip().lower().replace("-", "_")
    if token in {"cpu_ort", "ort_cpu"} or provider in {
        "cpu", "cpu_ort", "ort_cpu",
    }:
        return "ort_cpu"
    if not token:
        _fail("benchmark_run_id_missing")
    return token


def _run_tokens(rows: Sequence[Mapping[str, Any]], logical_path: str) -> List[str]:
    tokens = [_run_token(row) for row in rows]
    if len(tokens) != len(set(tokens)):
        _fail("benchmark_run_id_duplicate", logical_path)
    return tokens


def _require_final_task(
    payload: Mapping[str, Any], task: str, logical_path: str,
) -> None:
    for key in _FINAL_TASK_FIELDS:
        if str(payload.get(key) or "").strip().lower() != task:
            _fail(f"benchmark_{key}_mismatch", logical_path)


def _require_quality_policy_semantics(
    value: Any,
    *,
    expected_sha256: str,
    logical_path: str,
    field: str,
) -> None:
    if not isinstance(value, Mapping):
        _fail("quality_gate_policy_missing", f"{logical_path}:{field}")
    try:
        observed = AccuracyGatePolicy.from_mapping(value).sha256()
    except Exception as exc:
        raise MissingFullQualityAttestationError(
            "quality_gate_policy_invalid", f"{logical_path}:{field}",
        ) from exc
    if observed != expected_sha256:
        _fail(
            "quality_gate_policy_semantic_mismatch",
            f"{logical_path}:{field}",
        )


def _require_final_run_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    immutable_rows: Sequence[Mapping[str, Any]],
    task: str,
    expected_quality_gate_policy_sha256: str,
    logical_path: str,
) -> None:
    final_tokens = _run_tokens(rows, logical_path)
    immutable_tokens = _run_tokens(immutable_rows, logical_path + ":immutable")
    if final_tokens != immutable_tokens:
        _fail("benchmark_run_identity_set_mismatch", logical_path)
    for immutable, final, token in zip(immutable_rows, rows, final_tokens):
        if (
            str(final.get("task") or "").strip().lower() != task
            or str(final.get("benchmark_task") or "").strip().lower() != task
        ):
            _fail("benchmark_run_task_mismatch", f"{logical_path}:{token}")
        run_detail = f"{logical_path}:{token}"
        if (
            str(final.get("quality_gate_policy_sha256") or "").strip().lower()
            != expected_quality_gate_policy_sha256
        ):
            _fail("quality_gate_policy_sha256_mismatch", run_detail)
        _require_quality_policy_semantics(
            final.get("task_quality_gate"),
            expected_sha256=expected_quality_gate_policy_sha256,
            logical_path=run_detail,
            field="task_quality_gate",
        )
        if token == "ort_cpu":
            required = {
                "id": "ort_cpu",
                "type": "onnxruntime",
                "provider": "cpu",
                "backend": "ort_cpu",
                "variant": "full",
                "semantic_reference_only": True,
                "canonical_cpu_reference": True,
                "performance_eligible": False,
                "energy_eligible": False,
                "ranking_eligible": False,
                "pareto_eligible": False,
                "execution_location": "central_management",
            }
            if any(final.get(key) != value for key, value in required.items()):
                _fail("benchmark_cpu_reference_invariant_invalid", logical_path)
            continue
        for key in _STABLE_RUN_FIELDS:
            if key in immutable and immutable.get(key) != final.get(key):
                _fail(
                    "benchmark_run_stable_identity_mismatch",
                    f"{logical_path}:{token}:{key}",
                )


def _stable_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require_set_run_aliases(
    payload: Mapping[str, Any],
    *,
    runs: Sequence[Mapping[str, Any]],
    logical_path: str,
) -> None:
    planned = payload.get("planned_runs")
    if not isinstance(planned, list) or planned != list(runs):
        _fail("benchmark_set_planned_runs_mismatch", logical_path)
    if "runs" in payload and payload.get("runs") != list(runs):
        _fail("benchmark_set_runs_mismatch", logical_path)
    embedded = payload.get("plan")
    if isinstance(embedded, Mapping):
        if embedded.get("runs") != list(runs) or embedded.get("planned_runs") != list(runs):
            _fail("benchmark_set_embedded_plan_mismatch", logical_path)


def _require_management_invariant(
    payloads: Sequence[Tuple[str, Mapping[str, Any]]],
    runs: Sequence[Mapping[str, Any]],
) -> None:
    present = [
        (path, payload.get("management_cpu_reference_invariant"))
        for path, payload in payloads
        if "management_cpu_reference_invariant" in payload
    ]
    if not present:
        _fail("management_cpu_reference_invariant_alias_missing")
    if len(present) != len(payloads):
        _fail("management_cpu_reference_invariant_alias_missing")
    first = present[0][1]
    if not isinstance(first, Mapping):
        _fail("management_cpu_reference_invariant_invalid", present[0][0])
    expected_hash = _stable_json_sha256(list(runs))
    required = {
        "status": "verified",
        "recipe_count": 1,
        "execution_location": "central_management",
        "performance_dispatch_allowed": False,
        "run_plan_sha256": expected_hash,
    }
    if any(first.get(key) != value for key, value in required.items()):
        _fail("management_cpu_reference_invariant_invalid", present[0][0])
    for path, value in present[1:]:
        if value != first:
            _fail("management_cpu_reference_invariant_alias_mismatch", path)


def attest_generate_benchmark_set_supersession(
    *,
    run_dir: str | Path,
    model_id: str,
    task: str,
    run_id: str,
    expected_quality_gate_policy_sha256: str,
    artifact_index: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attest the four legal finalized BenchmarkSet aliases.

    The two ``legacy_benchmark_*`` copies must still match their immutable
    ``generate_benchmark_set`` Artifact Index records.  The current formal and
    executable plan/set aliases may have later bytes, but only when all four
    agree on schema, model/task/run identity, cases, ordered run identities,
    finalized run rows, and the management-reference seal.
    """

    model = _model_id(model_id)
    expected_task = _task(task)
    expected_quality_hash = _sha256_token(
        expected_quality_gate_policy_sha256,
    )
    expected_run = str(run_id or "").strip()
    if (
        not expected_run
        or "\x00" in expected_run
        or "\n" in expected_run
        or "\r" in expected_run
        or "/" in expected_run
        or "\\" in expected_run
    ):
        _fail("run_id_invalid", expected_run)
    if str(artifact_index.get("run_id") or "").strip() != expected_run:
        _fail("artifact_index_run_id_mismatch", expected_run)

    base = PurePosixPath("models") / model / "benchmark_set"
    paths = {
        "immutable_set": base / "legacy_benchmark_set.json",
        "immutable_plan": base / "legacy_benchmark_plan.json",
        "formal_set": base / "benchmark_set.json",
        "formal_plan": base / "benchmark_plan.json",
        "suite_set": base / "legacy_suite" / "benchmark_set.json",
        "suite_plan": base / "legacy_suite" / "benchmark_plan.json",
    }
    current = {
        key: _read_current_file(run_dir, value.as_posix())
        for key, value in paths.items()
    }
    for key in ("immutable_set", "immutable_plan"):
        matches = _matching_index_rows(
            current[key],
            artifact_index,
            expected_model_id=model,
            allowed_producer_stages=("generate_benchmark_set",),
        )
        if not matches:
            _fail("immutable_legacy_alias_not_index_bound", current[key].logical_path)
    for key in ("formal_set", "formal_plan", "suite_set", "suite_plan"):
        _require_historical_index_role(
            current=current[key],
            artifact_index=artifact_index,
            model_id=model,
            producer_stage="generate_benchmark_set",
        )

    payload = {key: _strict_json_object(value) for key, value in current.items()}
    _require_schema(
        payload["immutable_set"], BENCHMARK_SET_SCHEMA,
        current["immutable_set"].logical_path,
    )
    _require_schema(
        payload["immutable_plan"], BENCHMARK_PLAN_SCHEMA,
        current["immutable_plan"].logical_path,
    )
    _require_schema(
        payload["formal_set"], FORMAL_BENCHMARK_SET_SCHEMA,
        current["formal_set"].logical_path,
    )
    _require_schema(
        payload["formal_plan"], BENCHMARK_PLAN_SCHEMA,
        current["formal_plan"].logical_path,
    )
    _require_schema(
        payload["suite_set"], BENCHMARK_SET_SCHEMA,
        current["suite_set"].logical_path,
    )
    _require_schema(
        payload["suite_plan"], BENCHMARK_PLAN_SCHEMA,
        current["suite_plan"].logical_path,
    )

    if str(payload["formal_set"].get("model_id") or "").strip().lower() != model:
        _fail("artifact_model_id_mismatch", current["formal_set"].logical_path)
    if str(payload["formal_set"].get("run_id") or "").strip() != expected_run:
        _fail("artifact_run_id_mismatch", current["formal_set"].logical_path)
    for key, value in payload.items():
        _require_optional_identity(
            value,
            key="model_id",
            expected=model,
            logical_path=current[key].logical_path,
        )
        if "run_id" in value and str(value.get("run_id") or "").strip() != expected_run:
            _fail("artifact_run_id_mismatch", current[key].logical_path)

    suite_rel = (base / "legacy_suite").as_posix()
    expected_formal_paths = {
        "suite_dir": suite_rel,
        "legacy_suite_dir": suite_rel,
        "legacy_suite_benchmark_set": paths["suite_set"].as_posix(),
        "benchmark_plan": paths["suite_plan"].as_posix(),
    }
    if any(
        str(payload["formal_set"].get(key) or "") != value
        for key, value in expected_formal_paths.items()
    ):
        _fail("formal_benchmark_set_alias_path_mismatch")

    for key in ("formal_set", "formal_plan", "suite_set", "suite_plan"):
        _require_final_task(
            payload[key], expected_task, current[key].logical_path,
        )
        actual_quality_hash = str(
            payload[key].get("quality_gate_policy_sha256") or ""
        ).strip().lower()
        if actual_quality_hash != expected_quality_hash:
            _fail(
                "quality_gate_policy_sha256_mismatch",
                current[key].logical_path,
            )
        _require_quality_policy_semantics(
            payload[key].get("quality_gate"),
            expected_sha256=expected_quality_hash,
            logical_path=current[key].logical_path,
            field="quality_gate",
        )

    immutable_runs = _payload_runs(
        payload["immutable_plan"],
        logical_path=current["immutable_plan"].logical_path,
        require_alias_pair=False,
    )
    formal_runs = _payload_runs(
        payload["formal_plan"],
        logical_path=current["formal_plan"].logical_path,
        require_alias_pair=True,
    )
    suite_runs = _payload_runs(
        payload["suite_plan"],
        logical_path=current["suite_plan"].logical_path,
        require_alias_pair=True,
    )
    if formal_runs != suite_runs:
        _fail("final_benchmark_plan_alias_mismatch")
    _require_final_run_rows(
        formal_runs,
        immutable_rows=immutable_runs,
        task=expected_task,
        expected_quality_gate_policy_sha256=expected_quality_hash,
        logical_path=current["formal_plan"].logical_path,
    )

    immutable_cases = payload["immutable_set"].get("cases")
    if not isinstance(immutable_cases, list):
        _fail("immutable_benchmark_cases_invalid")
    for key in ("formal_set", "suite_set"):
        if payload[key].get("cases") != immutable_cases:
            _fail("benchmark_case_alias_mismatch", current[key].logical_path)
        _require_set_run_aliases(
            payload[key], runs=formal_runs, logical_path=current[key].logical_path,
        )

    _require_management_invariant(
        [
            (current[key].logical_path, payload[key])
            for key in ("formal_set", "formal_plan", "suite_set", "suite_plan")
        ],
        formal_runs,
    )
    return {
        "schema": _ATTESTATION_SCHEMA,
        "schema_version": 1,
        "scope": "generate_benchmark_set_finalized_alias_supersession",
        "ok": True,
        "model_id": model,
        "task": expected_task,
        "run_id": expected_run,
        "quality_gate_policy_sha256": expected_quality_hash,
        "immutable_aliases": {
            key: current[key].public()
            for key in ("immutable_set", "immutable_plan")
        },
        "finalized_aliases": {
            key: current[key].public()
            for key in ("formal_set", "formal_plan", "suite_set", "suite_plan")
        },
        "ordered_run_ids": _run_tokens(formal_runs, "finalized_aliases"),
        "case_count": len(immutable_cases),
        "run_count": len(formal_runs),
        "aliases_consistent": True,
    }


__all__ = [
    "MissingFullQualityAttestationError",
    "attest_current_path",
    "attest_current_index_binding",
    "attest_prepare_full_baselines_supersession",
    "attest_generate_benchmark_set_supersession",
    "attest_historical_success_stderr_omission",
]
