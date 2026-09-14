from __future__ import annotations

"""Exact, read-only compiler build evidence.

The artifact store owns artifact bytes.  This module owns a much smaller
question: has *this exact compiler request* already produced a deterministic
outcome?  The key deliberately includes every compiler-relevant identity used
by the Hailo v3 cache plus the full-model and split endpoint identities that
the backend-local cache does not know about.

Recovery is read-only with respect to the source EvaluationRun.  An index is
written outside that run and contains only relative evidence origins.  A
positive lookup always re-verifies the HEF, its v2 build receipt and the fixed
compiler ONNX before it can be reused.
"""

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence


BUILD_KEY_SCHEMA = "onnx-splitpoint/exact-build-key/v1"
BUILD_RECORD_SCHEMA = "onnx-splitpoint/build-evidence-record/v1"
BUILD_INDEX_SCHEMA = "onnx-splitpoint/build-evidence-index/v1"
BUILD_CONTEXT_SCHEMA = "onnx-splitpoint/build-evidence-context/v1"
HAILO_ATTEMPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-attempt/v1"
HAILO_RECEIPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-receipt/v2"
HAILO_CACHE_SCHEMA_V3 = "onnx-splitpoint/hailo-hef-cache-key-v3"

ARTIFACT_PASS = "ARTIFACT_PASS"
PARSER_UNSUPPORTED = "PARSER_UNSUPPORTED"
COMPILE_INFEASIBLE = "COMPILE_INFEASIBLE"
TRANSIENT_INFRASTRUCTURE = "TRANSIENT_INFRASTRUCTURE"
ABORTED_UNKNOWN = "ABORTED_UNKNOWN"

BUILD_STATES = frozenset({
    ARTIFACT_PASS,
    PARSER_UNSUPPORTED,
    COMPILE_INFEASIBLE,
    TRANSIENT_INFRASTRUCTURE,
    ABORTED_UNKNOWN,
})
DETERMINISTIC_REUSABLE_STATES = frozenset({
    ARTIFACT_PASS,
    PARSER_UNSUPPORTED,
    COMPILE_INFEASIBLE,
})

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CACHE_PREFIX = re.compile(r"^[0-9a-f]{12,64}$")
_JSON_LIMIT = 64 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_EVIDENCE_FILENAMES = frozenset({
    "compiled.hef",
    "hailo_hef_build_receipt.json",
    "hailo_hef_build_result.json",
    "hailo_hef_build_attempt.json",
    "build_evidence_context.json",
    "terminal_attempt.json",
})
_HAILO_BUNDLE_NAMES = ("compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json")
_HAILO_GENERATION = re.compile(r"(?:previous-)?[0-9a-f]{32}\Z")


class BuildEvidenceError(RuntimeError):
    """Fail-closed evidence validation error with a stable reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}{':' + self.detail if self.detail else ''}")


@dataclass(frozen=True)
class FileObservation:
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    data: bytes | None = None

    def stable_identity(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.mode,
            self.size_bytes,
            self.mtime_ns,
            self.ctime_ns,
        )


@dataclass(frozen=True)
class VerifiedHailoArtifact:
    hef_path: Path
    receipt_path: Path
    compiler_onnx_path: Path
    hef_sha256: str
    hef_size_bytes: int
    receipt_sha256: str
    compiler_onnx_sha256: str
    cache_key: str
    receipt: dict[str, Any]
    cache_payload: dict[str, Any]
    cache_meta_path: Path | None = None
    cache_meta_sha256: str = ""


@dataclass(frozen=True)
class BuildEvidenceDecision:
    """Stable lookup result consumed by an optional build controller."""

    status: str
    key_sha256: str
    reusable: bool
    state: str | None
    reason: str
    record: dict[str, Any] | None = None
    evidence_origin: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "key_sha256": self.key_sha256,
            "reusable": self.reusable,
            "state": self.state,
            "reason": self.reason,
            "record": self.record,
            "evidence_origin": self.evidence_origin,
        }


def _need(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise BuildEvidenceError(code, detail)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BuildEvidenceError(
            "noncanonical_payload", type(exc).__name__
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_clone(value: Any, *, field: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except BuildEvidenceError:
        raise
    except Exception as exc:  # pragma: no cover - canonical_json already seals this
        raise BuildEvidenceError("noncanonical_payload", field) from exc


def _digest(value: Any, *, field: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    if not _HEX64.fullmatch(token):
        raise BuildEvidenceError("invalid_sha256", field)
    return token


def _optional_digest(value: Any, *, field: str) -> str:
    if value in (None, ""):
        return ""
    return _digest(value, field=field)


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BuildEvidenceError("invalid_integer", field)
    return int(value)


def _token(value: Any, *, field: str, lower: bool = False) -> str:
    result = str(value or "").strip()
    if not result or any(ord(character) < 32 for character in result):
        raise BuildEvidenceError("invalid_identity", field)
    if result.lower() in {"unknown", "none", "unavailable", "n/a"}:
        raise BuildEvidenceError("incomplete_identity", field)
    return result.lower() if lower else result


def _normalize_hw_arch(value: Any) -> str:
    token = _token(value, field="hw_arch", lower=True).replace("-", "")
    aliases = {
        "hailo10": "hailo10h",
        "hailo10h": "hailo10h",
        "hailo8": "hailo8",
        "hailo8l": "hailo8l",
    }
    return aliases.get(token, token)


def _normalize_backend(value: Any) -> str:
    token = _token(value, field="backend", lower=True).replace("-", "_")
    if token in {"hailo", "hailo_hef", "hailo_sdk", "hailo_dfc"}:
        return "hailo_dfc"
    return token


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise BuildEvidenceError("invalid_string_list", field)
    result: list[str] = []
    for index, item in enumerate(value):
        token = str(item or "").strip()
        if (
            not token
            or token != item
            or any(ord(character) < 32 for character in token)
            or token in result
        ):
            raise BuildEvidenceError("invalid_string_list", f"{field}.{index}")
        result.append(token)
    return result


def _canonical_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(recipe, Mapping), "invalid_recipe")
    body = _json_clone(dict(recipe), field="recipe")
    _need(isinstance(body, dict), "invalid_recipe")
    body["optimization_level"] = _strict_int(
        body.get("optimization_level"),
        field="recipe.optimization_level",
    )
    body["model_script_sha256"] = _digest(
        body.get("model_script_sha256"),
        field="recipe.model_script_sha256",
    )
    if "start_nodes" in body:
        body["start_nodes"] = _string_list(
            body.get("start_nodes"), field="recipe.start_nodes"
        )
    if "end_nodes" in body:
        body["end_nodes"] = _string_list(
            body.get("end_nodes"), field="recipe.end_nodes"
        )
    if "activation_part1_sha256" in body:
        body["activation_part1_sha256"] = _optional_digest(
            body.get("activation_part1_sha256"),
            field="recipe.activation_part1_sha256",
        )
    if "disable_rt_metadata_extraction" in body:
        _need(
            type(body.get("disable_rt_metadata_extraction")) is bool,
            "invalid_boolean",
            "recipe.disable_rt_metadata_extraction",
        )
    return body


def _canonical_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(calibration, Mapping), "invalid_calibration")
    body = _json_clone(dict(calibration), field="calibration")
    _need(isinstance(body, dict), "invalid_calibration")
    body["identity"] = _token(
        body.get("identity"), field="calibration.identity"
    )
    body["effective_count"] = _strict_int(
        body.get("effective_count"),
        field="calibration.effective_count",
        minimum=1,
    )
    body["requested_count"] = _strict_int(
        body.get("requested_count"),
        field="calibration.requested_count",
        minimum=1,
    )
    body["batch_size"] = _strict_int(
        body.get("batch_size"),
        field="calibration.batch_size",
        minimum=1,
    )
    _need(
        body["effective_count"] <= body["requested_count"],
        "invalid_calibration_count_order",
    )
    if "prepared_identity_sha256" in body:
        body["prepared_identity_sha256"] = _digest(
            body.get("prepared_identity_sha256"),
            field="calibration.prepared_identity_sha256",
        )
    return body


def canonical_build_key(
    *,
    full_source_onnx_sha256: str,
    builder_source_onnx_sha256: str,
    compiler_onnx_sha256: str,
    boundary_endpoint_contract_sha256: str,
    backend: str,
    hw_arch: str,
    compiler_version: str,
    recipe: Mapping[str, Any],
    calibration: Mapping[str, Any],
    preprocessing_contract_sha256: str,
    backend_cache_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the canonical exact build key.

    ``recipe`` must explicitly contain ``optimization_level`` and the SHA-256
    of the model script. ``calibration`` must explicitly contain its identity,
    requested/effective counts and batch size. Extra canonical fields are
    retained so a backend may seal additional compile-affecting axes.
    """

    recipe_body = _canonical_recipe(recipe)
    calibration_body = _canonical_calibration(calibration)
    backend_eff = _normalize_backend(backend)
    hw_arch_eff = _normalize_hw_arch(hw_arch)
    compiler_version_eff = _token(
        compiler_version, field="compiler_version"
    )
    compiler_sha = _digest(
        compiler_onnx_sha256, field="compiler_onnx_sha256"
    )
    preprocessing_sha = _digest(
        preprocessing_contract_sha256,
        field="preprocessing_contract_sha256",
    )
    backend_contract = backend_cache_contract_sha256
    if backend_contract in (None, ""):
        backend_contract = canonical_sha256({
            "backend": backend_eff,
            "hw_arch": hw_arch_eff,
            "compiler_version": compiler_version_eff,
            "compiler_onnx_sha256": compiler_sha,
            "recipe": recipe_body,
            "calibration": calibration_body,
            "preprocessing_contract_sha256": preprocessing_sha,
        })
    return {
        "schema": BUILD_KEY_SCHEMA,
        "schema_version": 1,
        "full_source_onnx_sha256": _digest(
            full_source_onnx_sha256,
            field="full_source_onnx_sha256",
        ),
        "builder_source_onnx_sha256": _digest(
            builder_source_onnx_sha256,
            field="builder_source_onnx_sha256",
        ),
        "compiler_onnx_sha256": compiler_sha,
        "boundary_endpoint_contract_sha256": _digest(
            boundary_endpoint_contract_sha256,
            field="boundary_endpoint_contract_sha256",
        ),
        "backend": backend_eff,
        "hw_arch": hw_arch_eff,
        "compiler_version": compiler_version_eff,
        "recipe": recipe_body,
        "recipe_contract_sha256": canonical_sha256(recipe_body),
        "calibration": calibration_body,
        "calibration_contract_sha256": canonical_sha256(calibration_body),
        "preprocessing_contract_sha256": preprocessing_sha,
        "backend_cache_contract_sha256": _digest(
            backend_contract,
            field="backend_cache_contract_sha256",
        ),
    }


def validate_build_key(value: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_build_key")
    _need(value.get("schema") == BUILD_KEY_SCHEMA, "build_key_schema_mismatch")
    _need(value.get("schema_version") == 1, "build_key_schema_version_mismatch")
    canonical = canonical_build_key(
        full_source_onnx_sha256=value.get("full_source_onnx_sha256"),
        builder_source_onnx_sha256=value.get("builder_source_onnx_sha256"),
        compiler_onnx_sha256=value.get("compiler_onnx_sha256"),
        boundary_endpoint_contract_sha256=value.get(
            "boundary_endpoint_contract_sha256"
        ),
        backend=value.get("backend"),
        hw_arch=value.get("hw_arch"),
        compiler_version=value.get("compiler_version"),
        recipe=value.get("recipe") or {},
        calibration=value.get("calibration") or {},
        preprocessing_contract_sha256=value.get(
            "preprocessing_contract_sha256"
        ),
        backend_cache_contract_sha256=value.get(
            "backend_cache_contract_sha256"
        ),
    )
    _need(
        dict(value) == canonical,
        "noncanonical_build_key",
    )
    return canonical


def build_key_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(validate_build_key(value))


def _hailo_v3_cache_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_hailo_cache_payload")
    payload = _json_clone(dict(value), field="cache_payload")
    _need(
        payload.get("schema") == HAILO_CACHE_SCHEMA_V3,
        "hailo_cache_schema_mismatch",
    )
    _digest(payload.get("model_sha256"), field="cache_payload.model_sha256")
    _normalize_hw_arch(payload.get("hw_arch"))
    _token(
        payload.get("hailo_sdk_version"),
        field="cache_payload.hailo_sdk_version",
    )
    _strict_int(
        payload.get("optimization_level"),
        field="cache_payload.optimization_level",
    )
    _token(
        payload.get("calibration_identity"),
        field="cache_payload.calibration_identity",
    )
    effective = _strict_int(
        payload.get("calibration_count"),
        field="cache_payload.calibration_count",
        minimum=1,
    )
    requested = _strict_int(
        payload.get("requested_calibration_count"),
        field="cache_payload.requested_calibration_count",
        minimum=1,
    )
    _need(effective <= requested, "invalid_calibration_count_order")
    _strict_int(
        payload.get("calibration_batch_size"),
        field="cache_payload.calibration_batch_size",
        minimum=1,
    )
    _digest(
        payload.get("preprocessing_contract_sha256"),
        field="cache_payload.preprocessing_contract_sha256",
    )
    _digest(
        payload.get("prepared_calibration_identity_sha256"),
        field="cache_payload.prepared_calibration_identity_sha256",
    )
    _string_list(payload.get("start_nodes"), field="cache_payload.start_nodes")
    _string_list(payload.get("end_nodes"), field="cache_payload.end_nodes")
    _need(
        type(payload.get("disable_rt_metadata_extraction")) is bool,
        "invalid_boolean",
        "cache_payload.disable_rt_metadata_extraction",
    )
    _token(payload.get("net_name"), field="cache_payload.net_name")
    _need("net_input_shapes" in payload, "missing_identity", "net_input_shapes")
    _need(
        isinstance(payload.get("preprocessing_contract"), Mapping),
        "invalid_preprocessing_contract",
    )
    _need(
        canonical_sha256(payload.get("preprocessing_contract"))
        == _digest(
            payload.get("preprocessing_contract_sha256"),
            field="cache_payload.preprocessing_contract_sha256",
        ),
        "preprocessing_contract_sha256_mismatch",
    )
    return payload


def canonical_build_key_from_hailo_v3_payload(
    payload: Mapping[str, Any],
    *,
    builder_source_onnx_sha256: str,
    full_source_onnx_sha256: str,
    boundary_endpoint_contract_sha256: str,
    expected_cache_key: str | None = None,
) -> dict[str, Any]:
    """Project a pre-build Hailo v3 cache payload into an exact build key."""

    payload = _hailo_v3_cache_payload(payload)
    cache_key = canonical_sha256(payload)
    if expected_cache_key is not None:
        _need(
            _digest(expected_cache_key, field="expected_cache_key") == cache_key,
            "hailo_cache_key_mismatch",
        )
    compiler_sha = _digest(
        payload.get("model_sha256"), field="cache_payload.model_sha256"
    )
    builder_sha = _digest(
        builder_source_onnx_sha256,
        field="builder_source_onnx_sha256",
    )
    hw_arch = _normalize_hw_arch(payload.get("hw_arch"))
    compiler_version = _token(
        payload.get("hailo_sdk_version"),
        field="cache_payload.hailo_sdk_version",
    )
    model_script = str(payload.get("extra_model_script") or "")
    recipe = {
        "optimization_level": _strict_int(
            payload.get("optimization_level"),
            field="cache_payload.optimization_level",
        ),
        "model_script_sha256": hashlib.sha256(
            model_script.encode("utf-8")
        ).hexdigest(),
        "activation_part1_sha256": _optional_digest(
            payload.get("activation_part1_sha256"),
            field="cache_payload.activation_part1_sha256",
        ),
        "start_nodes": _string_list(
            payload.get("start_nodes"), field="cache_payload.start_nodes"
        ),
        "end_nodes": _string_list(
            payload.get("end_nodes"), field="cache_payload.end_nodes"
        ),
        "integrity": _token(
            payload.get("integrity"), field="cache_payload.integrity", lower=True
        ),
        "net_name": _token(
            payload.get("net_name"), field="cache_payload.net_name"
        ),
        "net_input_shapes": _json_clone(
            payload.get("net_input_shapes"), field="cache_payload.net_input_shapes"
        ),
        "disable_rt_metadata_extraction": payload.get(
            "disable_rt_metadata_extraction"
        ),
    }
    calibration = {
        "identity": _token(
            payload.get("calibration_identity"),
            field="cache_payload.calibration_identity",
        ),
        "effective_count": _strict_int(
            payload.get("calibration_count"),
            field="cache_payload.calibration_count",
            minimum=1,
        ),
        "requested_count": _strict_int(
            payload.get("requested_calibration_count"),
            field="cache_payload.requested_calibration_count",
            minimum=1,
        ),
        "batch_size": _strict_int(
            payload.get("calibration_batch_size"),
            field="cache_payload.calibration_batch_size",
            minimum=1,
        ),
        "prepared_identity_sha256": _digest(
            payload.get("prepared_calibration_identity_sha256"),
            field="cache_payload.prepared_calibration_identity_sha256",
        ),
        "storage": _token(
            payload.get("calibration_storage"),
            field="cache_payload.calibration_storage",
            lower=True,
        ),
        "memory_cap_bytes": _strict_int(
            payload.get("calibration_memory_cap_bytes"),
            field="cache_payload.calibration_memory_cap_bytes",
            minimum=1,
        ),
    }
    preprocessing_sha = _digest(
        payload.get("preprocessing_contract_sha256"),
        field="cache_payload.preprocessing_contract_sha256",
    )
    expected_prepared = canonical_sha256({
        "calibration_identity": calibration["identity"],
        "preprocessing_contract_sha256": preprocessing_sha,
    })
    _need(
        calibration["prepared_identity_sha256"] == expected_prepared,
        "prepared_calibration_identity_mismatch",
    )
    return canonical_build_key(
        full_source_onnx_sha256=full_source_onnx_sha256,
        builder_source_onnx_sha256=builder_sha,
        compiler_onnx_sha256=compiler_sha,
        boundary_endpoint_contract_sha256=boundary_endpoint_contract_sha256,
        backend="hailo_dfc",
        hw_arch=hw_arch,
        compiler_version=compiler_version,
        recipe=recipe,
        calibration=calibration,
        preprocessing_contract_sha256=preprocessing_sha,
        backend_cache_contract_sha256=cache_key,
    )


def canonical_build_key_from_hailo_v3(
    receipt: Mapping[str, Any],
    *,
    full_source_onnx_sha256: str,
    boundary_endpoint_contract_sha256: str,
    expected_cache_key: str | None = None,
) -> dict[str, Any]:
    """Project a v2 Hailo receipt/v3 cache payload into an exact build key."""

    _need(isinstance(receipt, Mapping), "invalid_hailo_receipt")
    _need(
        receipt.get("schema") == HAILO_RECEIPT_SCHEMA,
        "hailo_receipt_schema_mismatch",
    )
    payload = _hailo_v3_cache_payload(receipt.get("cache_payload") or {})
    cache_key = canonical_sha256(payload)
    _need(
        _digest(receipt.get("cache_key"), field="receipt.cache_key") == cache_key,
        "hailo_cache_key_mismatch",
    )
    if expected_cache_key is not None:
        _need(
            _digest(expected_cache_key, field="expected_cache_key") == cache_key,
            "hailo_cache_key_mismatch",
        )
    compiler_sha = _digest(
        receipt.get("compiler_onnx_sha256"),
        field="receipt.compiler_onnx_sha256",
    )
    _need(
        compiler_sha
        == _digest(payload.get("model_sha256"), field="cache_payload.model_sha256"),
        "compiler_onnx_sha256_mismatch",
    )
    _need(
        _normalize_hw_arch(receipt.get("hw_arch"))
        == _normalize_hw_arch(payload.get("hw_arch")),
        "hw_arch_mismatch",
    )
    _need(
        _token(
            receipt.get("hailo_sdk_version"),
            field="receipt.hailo_sdk_version",
        )
        == _token(
            payload.get("hailo_sdk_version"),
            field="cache_payload.hailo_sdk_version",
        ),
        "compiler_version_mismatch",
    )
    return canonical_build_key_from_hailo_v3_payload(
        payload,
        builder_source_onnx_sha256=_digest(
            receipt.get("source_onnx_sha256"),
            field="receipt.source_onnx_sha256",
        ),
        full_source_onnx_sha256=full_source_onnx_sha256,
        boundary_endpoint_contract_sha256=boundary_endpoint_contract_sha256,
        expected_cache_key=cache_key,
    )


def boundary_endpoint_contract_sha256(
    *,
    stage: str,
    cache_payload: Mapping[str, Any],
    split_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Hash the portable boundary and physical endpoint semantics.

    Compiler graph bytes are independently sealed by the build key.  This
    projection therefore contains tensor/endpoint semantics only and excludes
    timestamps, absolute paths and predicted metrics from split manifests.
    """

    stage_eff = _token(stage, field="stage", lower=True)
    _need(stage_eff in {"full", "part1", "part2"}, "invalid_stage", stage_eff)
    payload = _hailo_v3_cache_payload(cache_payload)
    manifest = dict(split_manifest or {})
    cut_tensors = manifest.get("cut_tensors")
    if cut_tensors is None:
        cut = manifest.get("cut")
        cut_tensors = (
            cut.get("tensors") if isinstance(cut, Mapping) else []
        ) or []
    if not isinstance(cut_tensors, (list, tuple)):
        raise BuildEvidenceError("invalid_string_list", "cut_tensors")
    cut_values = [str(item).strip() for item in cut_tensors]
    _need(
        all(item and item == original for item, original in zip(cut_values, cut_tensors)),
        "invalid_string_list",
        "cut_tensors",
    )
    boundary: int | str = "full"
    if stage_eff != "full":
        boundary = _strict_int(
            manifest.get("boundary", manifest.get("boundary_index")),
            field="split_manifest.boundary",
        )
    hailo = manifest.get("hailo") if isinstance(manifest.get("hailo"), Mapping) else {}
    endpoint: dict[str, Any] = {
        "start_nodes": _string_list(
            payload.get("start_nodes"), field="cache_payload.start_nodes"
        ),
        "end_nodes": _string_list(
            payload.get("end_nodes"), field="cache_payload.end_nodes"
        ),
        "net_name": _token(payload.get("net_name"), field="cache_payload.net_name"),
        "net_input_shapes": _json_clone(
            payload.get("net_input_shapes"), field="cache_payload.net_input_shapes"
        ),
    }
    if stage_eff == "part1" and isinstance(hailo, Mapping):
        endpoint["part1_feature_splitter_identity_fix"] = _json_clone(
            hailo.get("part1_feature_splitter_identity_fix") or {},
            field="hailo.part1_feature_splitter_identity_fix",
        )
    if stage_eff == "part2" and isinstance(hailo, Mapping):
        endpoint.update({
            "part2_output_strategy": str(
                hailo.get("part2_output_strategy") or "original"
            ),
            "part2_effective_outputs": _json_clone(
                hailo.get("part2_effective_outputs") or [],
                field="hailo.part2_effective_outputs",
            ),
            "part2_output_contract": _json_clone(
                hailo.get("part2_output_contract") or {},
                field="hailo.part2_output_contract",
            ),
        })
    contract = {
        "schema": "onnx-splitpoint/boundary-endpoint-contract/v1",
        "stage": stage_eff,
        "boundary": boundary,
        "strict_boundary": bool(manifest.get("strict_boundary", False)),
        "cut_tensors": cut_values,
        "cut": _json_clone(manifest.get("cut") or {}, field="split_manifest.cut"),
        "io": _json_clone(manifest.get("io") or {}, field="split_manifest.io"),
        "pipeline": _json_clone(
            manifest.get("pipeline") or {}, field="split_manifest.pipeline"
        ),
        "endpoint": endpoint,
    }
    return canonical_sha256(contract)


def classify_build_outcome(
    result: Mapping[str, Any] | None,
    *,
    log_text: str = "",
    terminal: bool = True,
) -> str:
    """Classify a build without turning infrastructure failures into negatives."""

    body = dict(result or {})
    combined_parts = [
        str(body.get(name) or "")
        for name in (
            "error",
            "failure_kind",
            "unsupported_reason",
            "timeout_kind",
            "last_stage",
        )
    ]
    details = body.get("details")
    if details:
        with contextlib.suppress(Exception):
            combined_parts.append(
                json.dumps(details, sort_keys=True, ensure_ascii=False)[:100000]
            )
    combined_parts.append(str(log_text or "")[-250000:])
    text = "\n".join(combined_parts).lower()
    returncode = body.get("returncode")
    with contextlib.suppress(TypeError, ValueError, OverflowError):
        returncode = int(returncode)
    semantic = str(body.get("semantic_status") or body.get("status") or "").lower()
    if (
        not terminal
        or returncode in {130, 143, -2, -15}
        or semantic in {"aborted", "cancelled", "canceled", "interrupted"}
        or body.get("aborted") is True or body.get("cancelled") is True
        or any(token in text for token in (
            "keyboardinterrupt",
            "cancelled by user",
            "canceled by user",
            "sigterm",
            "sigint",
            "aborted_unknown",
        ))
    ):
        return ABORTED_UNKNOWN
    if body.get("timed_out") is True:
        return TRANSIENT_INFRASTRUCTURE
    if body.get("ok") is True:
        return ARTIFACT_PASS
    # Host/device resource exhaustion is not a statement about graph mapping.
    # Check these unambiguous failures before parser/mapping phrases that may
    # also occur in wrapper exceptions or earlier log lines. Startup CUDA
    # registration warnings below deliberately do not override a real mapping
    # error (they are common even in successful compiler environments).
    if returncode in {137, -9} or any(token in text for token in (
        "out of memory", "outofmemory", "resource_exhausted", "resourceexhausted",
        "oom when allocating", "oom error",
        "resource exhausted", "cuda_error_out_of_memory", "std::bad_alloc",
        "memoryerror", "cannot allocate memory", "memory allocation failed",
        "cuda allocation failed", "cudamalloc failed",
        "no space left on device", "disk quota exceeded", "input/output error",
        "connection reset", "broken pipe", "remote transport",
        "process launch failed", "killed by oom", "oom-kill",
        "timed out", "timeout",
    )):
        return TRANSIENT_INFRASTRUCTURE
    if any(token in text for token in (
        "unsupportedshufflelayererror",
        "unsupportedmodelerror",
        "unsupported layer",
        "unsupported operation",
        "parser unsupported",
        "parse_unsupported",
        "parsing failed on node",
        "translation failed on node",
    )):
        return PARSER_UNSUPPORTED
    if any(token in text for token in (
        "mapping failed",
        "no successful assignments",
        "agent infeasible",
        "allocator_agent_infeasible",
        "allocator_mapping_failed",
    )):
        return COMPILE_INFEASIBLE
    if any(token in text for token in (
        "timed out",
        "timeout",
        "hailo sdk not available",
        "no module named 'hailo_sdk_client'",
        "cuda_dnn.cc",
        "cudnn_status",
        "unable to register cudnn",
        "unable to register cublas",
        "connection reset",
        "broken pipe",
        "remote transport",
        "no space left on device",
        "input/output error",
        "process launch failed",
    )):
        return TRANSIENT_INFRASTRUCTURE
    return TRANSIENT_INFRASTRUCTURE if terminal else ABORTED_UNKNOWN


def _strict_json(data: bytes, *, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BuildEvidenceError("duplicate_json_key", f"{label}:{key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise BuildEvidenceError("nonfinite_json_number", f"{label}:{value}")

    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except BuildEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildEvidenceError("invalid_json", label) from exc


def _lexical_absolute(value: str | Path, *, label: str) -> Path:
    raw = os.fspath(Path(value).expanduser())
    if "\x00" in raw or any(part == ".." for part in Path(raw).parts):
        raise BuildEvidenceError("unsafe_path", label)
    if not os.path.isabs(raw):
        raw = os.path.abspath(raw)
    return Path(os.path.normpath(raw))


def _path_inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(path), os.fspath(root))) == os.fspath(root)
    except ValueError:
        return False


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise BuildEvidenceError("nofollow_platform_unsupported", label)
    absolute = _lexical_absolute(path, label=label)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError as exc:
                raise BuildEvidenceError("required_path_missing", label) from exc
            except OSError as exc:
                code = (
                    "unsafe_symlink_component"
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else "unsafe_input_directory"
                )
                raise BuildEvidenceError(code, label) from exc
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise BuildEvidenceError("unsafe_input_directory", label)
        return fd
    except Exception:
        os.close(fd)
        raise


def _ensure_directory_nofollow(
    path: Path,
    *,
    label: str,
    mode: int = 0o700,
) -> int:
    """Create missing directory components without ever following symlinks.

    The returned descriptor refers to the final directory and must be closed by
    the caller. Every lookup and mkdir is relative to an already admitted
    directory descriptor, so a symlink ancestor cannot redirect mutations.
    """

    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise BuildEvidenceError("nofollow_platform_unsupported", label)
    absolute = _lexical_absolute(path, label=label)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode, dir_fd=fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    code = (
                        "unsafe_symlink_component"
                        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                        else "unsafe_output_directory"
                    )
                    raise BuildEvidenceError(code, label) from exc
                try:
                    child = os.open(component, flags, dir_fd=fd)
                except OSError as exc:
                    code = (
                        "unsafe_symlink_component"
                        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                        else "unsafe_output_directory"
                    )
                    raise BuildEvidenceError(code, label) from exc
            except OSError as exc:
                code = (
                    "unsafe_symlink_component"
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else "unsafe_output_directory"
                )
                raise BuildEvidenceError(code, label) from exc
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise BuildEvidenceError("unsafe_output_directory", label)
        return fd
    except Exception:
        os.close(fd)
        raise


def validate_external_output_path(
    *,
    source_run: str | Path,
    output: str | Path,
    label: str = "output",
    distinct_from: str | Path | None = None,
) -> Path:
    """Preflight an exclusive output without creating any path component.

    This is an early, side-effect-free admission check. The atomic writers
    still repeat component-wise nofollow checks to close the TOCTOU window.
    """

    root = _lexical_absolute(source_run, label="source_run")
    root_fd = _open_directory_nofollow(root, label="source_run")
    os.close(root_fd)
    target = _lexical_absolute(output, label=label)
    _need(
        not _path_inside(target, root),
        f"{label}_inside_source_run",
        os.fspath(target),
    )
    if distinct_from is not None:
        other = _lexical_absolute(distinct_from, label="distinct_output")
        _need(
            target != other,
            "output_paths_not_distinct",
            os.fspath(target),
        )
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open("/", flags)
    try:
        for component in target.parent.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                break
            except OSError as exc:
                code = (
                    "unsafe_symlink_component"
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else "unsafe_output_directory"
                )
                raise BuildEvidenceError(code, label) from exc
            os.close(fd)
            fd = child
    finally:
        os.close(fd)
    if os.path.lexists(target):
        info = os.lstat(target)
        if stat.S_ISLNK(info.st_mode):
            raise BuildEvidenceError(f"{label}_symlink", os.fspath(target))
        if not stat.S_ISREG(info.st_mode):
            raise BuildEvidenceError(f"{label}_not_regular", os.fspath(target))
        raise BuildEvidenceError(f"{label}_already_exists", os.fspath(target))
    return target


def _read_regular_nofollow(
    value: str | Path,
    *,
    label: str,
    collect: bool,
    size_limit: int | None = None,
) -> FileObservation:
    path = _lexical_absolute(value, label=label)
    parent_fd = _open_directory_nofollow(path.parent, label=f"{label}.parent")
    file_fd = -1
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            before_path = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(before_path.st_mode):
                raise BuildEvidenceError("input_not_regular_file", label)
            file_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise BuildEvidenceError("required_file_missing", label) from exc
        except OSError as exc:
            code = (
                "unsafe_symlink_component"
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                else "unsafe_input_file"
            )
            raise BuildEvidenceError(code, label) from exc
        before_fd = os.fstat(file_fd)
        if not stat.S_ISREG(before_fd.st_mode):
            raise BuildEvidenceError("input_not_regular_file", label)
        if (before_path.st_dev, before_path.st_ino) != (
            before_fd.st_dev,
            before_fd.st_ino,
        ):
            raise BuildEvidenceError("input_path_replaced_before_read", label)
        if size_limit is not None and before_fd.st_size > size_limit:
            raise BuildEvidenceError("input_too_large", label)
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if collect else None
        total = 0
        while True:
            block = os.read(file_fd, _READ_CHUNK)
            if not block:
                break
            total += len(block)
            if size_limit is not None and total > size_limit:
                raise BuildEvidenceError("input_too_large", label)
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after_fd = os.fstat(file_fd)
        after_path = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        identity_before = (
            before_fd.st_dev,
            before_fd.st_ino,
            before_fd.st_mode,
            before_fd.st_size,
            before_fd.st_mtime_ns,
            before_fd.st_ctime_ns,
        )
        identity_after = (
            after_fd.st_dev,
            after_fd.st_ino,
            after_fd.st_mode,
            after_fd.st_size,
            after_fd.st_mtime_ns,
            after_fd.st_ctime_ns,
        )
        if identity_before != identity_after or total != after_fd.st_size:
            raise BuildEvidenceError("input_modified_during_read", label)
        if (after_path.st_dev, after_path.st_ino) != (
            after_fd.st_dev,
            after_fd.st_ino,
        ):
            raise BuildEvidenceError("input_path_replaced_after_read", label)
        return FileObservation(
            path=path,
            sha256=digest.hexdigest(),
            size_bytes=total,
            device=after_fd.st_dev,
            inode=after_fd.st_ino,
            mode=after_fd.st_mode,
            mtime_ns=after_fd.st_mtime_ns,
            ctime_ns=after_fd.st_ctime_ns,
            data=b"".join(chunks) if chunks is not None else None,
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


class _ReadOnlyAttestor:
    def __init__(self, root: Path) -> None:
        self.root = _lexical_absolute(root, label="source_run")
        descriptor = _open_directory_nofollow(self.root, label="source_run")
        os.close(descriptor)
        self._observations: dict[str, FileObservation] = {}
        self._logical: dict[str, set[str]] = {}
        self.bundle_snapshots: dict[Path, Path] = {}

    def _physical_path(self, path: Path) -> Path:
        path = _lexical_absolute(path, label="evidence_path")
        generation = self.bundle_snapshots.get(path.parent)
        return generation / path.name if generation is not None and path.name in _HAILO_BUNDLE_NAMES else path

    def _logical_path(self, path: Path) -> str:
        absolute = _lexical_absolute(path, label="evidence_path")
        if not _path_inside(absolute, self.root):
            raise BuildEvidenceError("evidence_outside_source_run", os.fspath(absolute))
        relative = absolute.relative_to(self.root).as_posix()
        logical = PurePosixPath(relative)
        if (
            not relative
            or logical.is_absolute()
            or any(part in {"", ".", ".."} for part in logical.parts)
        ):
            raise BuildEvidenceError("unsafe_relative_path", relative)
        return relative

    def _remember(self, observed: FileObservation, logical: str) -> None:
        key = os.fspath(observed.path)
        previous = self._observations.get(key)
        if previous is not None and (
            previous.sha256 != observed.sha256
            or previous.stable_identity() != observed.stable_identity()
        ):
            raise BuildEvidenceError("input_changed_between_reads", logical)
        self._observations[key] = observed
        self._logical.setdefault(key, set()).add(logical)

    def file(self, path: Path, *, logical: str | None = None) -> FileObservation:
        relative = self._logical_path(path)
        label = logical or relative
        observed = _read_regular_nofollow(self._physical_path(path), label=label, collect=False)
        self._remember(observed, relative)
        return observed

    def json(self, path: Path, *, logical: str | None = None) -> tuple[dict[str, Any], FileObservation]:
        relative = self._logical_path(path)
        label = logical or relative
        observed = _read_regular_nofollow(
            self._physical_path(path),
            label=label,
            collect=True,
            size_limit=_JSON_LIMIT,
        )
        self._remember(observed, relative)
        payload = _strict_json(observed.data or b"", label=label)
        if not isinstance(payload, Mapping):
            raise BuildEvidenceError("json_object_required", label)
        return dict(payload), observed

    def stable_snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw_path, first in sorted(self._observations.items()):
            second = _read_regular_nofollow(
                first.path,
                label="source_snapshot_recheck",
                collect=False,
            )
            if (
                first.sha256 != second.sha256
                or first.stable_identity() != second.stable_identity()
            ):
                raise BuildEvidenceError(
                    "source_modified_during_harvest",
                    ",".join(sorted(self._logical[raw_path])),
                )
            for logical in sorted(self._logical[raw_path]):
                rows.append({
                    "logical_path": logical,
                    "sha256": first.sha256,
                    "size_bytes": first.size_bytes,
                })
        rows.sort(key=lambda row: (row["logical_path"], row["sha256"]))
        return rows


def _safe_relative_path(value: Any, *, field: str) -> str:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or "\\" in raw
        or any(ord(character) < 32 for character in raw)
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
    ):
        raise BuildEvidenceError("unsafe_relative_path", field)
    return raw


def _resolve_bounded_manifest_path(
    *,
    base: Path,
    raw_value: Any,
    root: Path,
    field: str,
) -> Path:
    raw = str(raw_value or "").strip()
    _need(raw and "\x00" not in raw and "\\" not in raw, "unsafe_manifest_path", field)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    absolute = Path(os.path.abspath(os.path.normpath(os.fspath(candidate))))
    _need(_path_inside(absolute, root), "manifest_path_outside_source_run", field)
    return absolute


def _snapshot_published_hailo_bundle(directory: Path) -> Path:
    """Resolve only the publisher's confined, complete compatibility triplet.

    Read the single pointer once, then access regular files by that immutable
    generation path. Never resolve arbitrary links or follow symlink ancestors.
    A concurrent legitimate pointer replacement cannot mix sibling generations.
    """
    descriptor = _open_directory_nofollow(directory, label="hailo_bundle_directory")
    try:
        for name in _HAILO_BUNDLE_NAMES:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            _need(stat.S_ISLNK(info.st_mode), "unsafe_evidence_symlink", name)
            _need(os.readlink(name, dir_fd=descriptor) == f".hailo-current/{name}",
                  "unsafe_evidence_symlink", name)
        pointer_info = os.stat(".hailo-current", dir_fd=descriptor, follow_symlinks=False)
        _need(stat.S_ISLNK(pointer_info.st_mode), "unsafe_evidence_symlink", ".hailo-current")
        raw = os.readlink(".hailo-current", dir_fd=descriptor)
    except OSError as exc:
        raise BuildEvidenceError("unsafe_evidence_symlink", os.fspath(directory)) from exc
    finally:
        os.close(descriptor)
    parts = PurePosixPath(raw).parts
    _need(len(parts) == 2 and parts[0] == ".hailo-generations"
          and _HAILO_GENERATION.fullmatch(parts[1]) is not None
          and PurePosixPath(raw).as_posix() == raw,
          "unsafe_evidence_symlink", raw)
    generation = directory / raw
    descriptor = _open_directory_nofollow(generation, label="hailo_bundle_generation")
    try:
        for name in _HAILO_BUNDLE_NAMES:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            _need(stat.S_ISREG(info.st_mode), "unsafe_evidence_symlink", name)
    except OSError as exc:
        raise BuildEvidenceError("incomplete_hailo_bundle", os.fspath(generation)) from exc
    finally:
        os.close(descriptor)
    return generation


def _hailo_public_directory(path: Path) -> Path:
    """Return a physical generation's public parent without following links."""
    directory = path.parent
    if directory.parent.name == ".hailo-generations":
        _need(_HAILO_GENERATION.fullmatch(directory.name) is not None,
              "invalid_hailo_generation", os.fspath(directory))
        return directory.parent.parent
    return directory


def _discover_evidence_files(
    root: Path, *, bundle_snapshots: dict[Path, Path] | None = None,
) -> list[Path]:
    snapshots = bundle_snapshots if bundle_snapshots is not None else {}
    result: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for name in sorted(directory_names):
            if name == ".hailo-generations":
                # Retained backups are not additional current artifacts.
                continue
            child = current_path / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode):
                continue
            if stat.S_ISDIR(info.st_mode):
                safe_dirs.append(name)
        directory_names[:] = safe_dirs
        selected_names = {
            name for name in file_names
            if name in _EVIDENCE_FILENAMES
            or name.endswith("_hailo_fixed.onnx")
            or (current_path.name == "hailo_attempt_receipts"
                and name.startswith("attempt_") and name.endswith(".json")
                and not name.endswith(".heartbeat.json"))
        }
        for name in sorted(selected_names):
            candidate = current_path / name
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode):
                if name not in _HAILO_BUNDLE_NAMES:
                    raise BuildEvidenceError("unsafe_evidence_symlink", os.fspath(candidate))
                if current_path not in snapshots:
                    snapshots[current_path] = _snapshot_published_hailo_bundle(current_path)
            elif not stat.S_ISREG(info.st_mode):
                raise BuildEvidenceError("evidence_not_regular", os.fspath(candidate))
            result.append(candidate)
    return sorted(result)


def verify_hailo_artifact(
    hef_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    _bundle_snapshot: Path | None = None,
    _compiler_directory: Path | None = None,
) -> VerifiedHailoArtifact:
    """Strictly verify a Hailo v2 receipt backed by a v3 cache payload."""

    hef = _lexical_absolute(hef_path, label="hef")
    receipt_p = _lexical_absolute(
        receipt_path or hef.parent / "hailo_hef_build_receipt.json",
        label="hailo_receipt",
    )
    public_directory = _hailo_public_directory(hef)
    meta_path: Path | None = None
    if _bundle_snapshot is not None or hef.is_symlink() or receipt_p.is_symlink():
        _need(hef.parent == receipt_p.parent and hef.name == "compiled.hef"
              and receipt_p.name == "hailo_hef_build_receipt.json",
              "unsafe_evidence_symlink", "hailo_bundle_siblings")
        generation = _bundle_snapshot or _snapshot_published_hailo_bundle(hef.parent)
        _need(_hailo_public_directory(generation / hef.name) == hef.parent,
              "unsafe_evidence_symlink", "hailo_bundle_snapshot")
        hef, receipt_p = generation / hef.name, generation / receipt_p.name
        meta_path = generation / "cache_meta.json"
    elif hef.parent.parent.name == ".hailo-generations":
        _need(receipt_p.parent == hef.parent, "hailo_bundle_sibling_mismatch")
        meta_path = hef.parent / "cache_meta.json"
    hef_observed = _read_regular_nofollow(hef, label="hef", collect=False)
    _need(hef_observed.size_bytes > 0, "empty_hef")
    receipt_observed = _read_regular_nofollow(
        receipt_p,
        label="hailo_receipt",
        collect=True,
        size_limit=_JSON_LIMIT,
    )
    receipt_raw = _strict_json(
        receipt_observed.data or b"", label="hailo_receipt"
    )
    _need(isinstance(receipt_raw, Mapping), "json_object_required", "hailo_receipt")
    receipt = dict(receipt_raw)
    _need(receipt.get("schema") == HAILO_RECEIPT_SCHEMA, "hailo_receipt_schema_mismatch")
    _need(
        _digest(receipt.get("hef_sha256"), field="receipt.hef_sha256")
        == hef_observed.sha256,
        "hef_sha256_mismatch",
    )
    _need(
        _strict_int(
            receipt.get("hef_size_bytes"),
            field="receipt.hef_size_bytes",
            minimum=1,
        )
        == hef_observed.size_bytes,
        "hef_size_mismatch",
    )
    compiler_filename = str(receipt.get("compiler_onnx_filename") or "")
    _need(
        compiler_filename
        and Path(compiler_filename).name == compiler_filename
        and compiler_filename.lower().endswith(".onnx"),
        "unsafe_compiler_onnx_filename",
    )
    compiler_path = (_compiler_directory or public_directory) / compiler_filename
    compiler_observed = _read_regular_nofollow(
        compiler_path,
        label="compiler_onnx",
        collect=False,
    )
    compiler_sha = _digest(
        receipt.get("compiler_onnx_sha256"),
        field="receipt.compiler_onnx_sha256",
    )
    _need(
        compiler_observed.sha256 == compiler_sha,
        "compiler_onnx_sha256_mismatch",
    )
    payload = _hailo_v3_cache_payload(receipt.get("cache_payload") or {})
    cache_key = canonical_sha256(payload)
    _need(
        _digest(receipt.get("cache_key"), field="receipt.cache_key") == cache_key,
        "hailo_cache_key_mismatch",
    )
    _need(
        _digest(payload.get("model_sha256"), field="cache_payload.model_sha256")
        == compiler_sha,
        "compiler_onnx_sha256_mismatch",
    )
    _need(
        _normalize_hw_arch(receipt.get("hw_arch"))
        == _normalize_hw_arch(payload.get("hw_arch")),
        "hw_arch_mismatch",
    )
    _need(
        _token(receipt.get("hailo_sdk_version"), field="receipt.hailo_sdk_version")
        == _token(payload.get("hailo_sdk_version"), field="cache_payload.hailo_sdk_version"),
        "compiler_version_mismatch",
    )
    _need(
        _digest(
            receipt.get("preprocessing_contract_sha256"),
            field="receipt.preprocessing_contract_sha256",
        )
        == _digest(
            payload.get("preprocessing_contract_sha256"),
            field="cache_payload.preprocessing_contract_sha256",
        ),
        "preprocessing_contract_sha256_mismatch",
    )
    _need(
        receipt.get("preprocessing_contract") == payload.get("preprocessing_contract"),
        "preprocessing_contract_mismatch",
    )
    _need(
        str(receipt.get("calibration_identity") or "")
        == str(payload.get("calibration_identity") or ""),
        "calibration_identity_mismatch",
    )
    _need(
        receipt.get("calibration_count") == payload.get("calibration_count")
        and receipt.get("requested_calibration_count")
        == payload.get("requested_calibration_count"),
        "calibration_count_mismatch",
    )
    _need(
        _digest(
            receipt.get("prepared_calibration_identity_sha256"),
            field="receipt.prepared_calibration_identity_sha256",
        )
        == _digest(
            payload.get("prepared_calibration_identity_sha256"),
            field="cache_payload.prepared_calibration_identity_sha256",
        ),
        "prepared_calibration_identity_mismatch",
    )
    meta_sha = ""
    if meta_path is not None:
        meta_observed = _read_regular_nofollow(
            meta_path, label="hailo_cache_meta", collect=True, size_limit=_JSON_LIMIT,
        )
        metadata = _strict_json(meta_observed.data or b"", label="hailo_cache_meta")
        expected_meta = {
            "schema": "onnx-splitpoint/hailo-hef-cache-meta-v2",
            "cache_key": cache_key, "payload": payload,
            "hef_size": hef_observed.size_bytes, "hef_sha256": hef_observed.sha256,
            "preprocessing_contract_sha256": receipt.get("preprocessing_contract_sha256"),
            "net_name": str(receipt.get("net_name") or ""),
            "hw_arch": str(receipt.get("hw_arch") or ""),
        }
        _need(isinstance(metadata, Mapping) and all(metadata.get(k) == v for k, v in expected_meta.items()),
              "hailo_cache_meta_mismatch")
        meta_sha = meta_observed.sha256
    return VerifiedHailoArtifact(
        hef_path=hef,
        receipt_path=receipt_p,
        compiler_onnx_path=compiler_path,
        hef_sha256=hef_observed.sha256,
        hef_size_bytes=hef_observed.size_bytes,
        receipt_sha256=receipt_observed.sha256,
        compiler_onnx_sha256=compiler_observed.sha256,
        cache_key=cache_key,
        receipt=receipt,
        cache_payload=payload,
        cache_meta_path=meta_path,
        cache_meta_sha256=meta_sha,
    )


def make_build_evidence_record(
    key: Mapping[str, Any],
    state: str,
    *,
    evidence_origin: Mapping[str, Any],
    artifact: Mapping[str, Any] | None = None,
    reason_code: str = "",
) -> dict[str, Any]:
    key_body = validate_build_key(key)
    state_eff = str(state or "").strip().upper()
    _need(state_eff in BUILD_STATES, "invalid_build_state", state_eff)
    origin = _json_clone(dict(evidence_origin), field="evidence_origin")
    _need(isinstance(origin, dict) and origin, "missing_evidence_origin")
    artifact_body: dict[str, Any] | None = None
    if artifact is not None:
        artifact_body = _json_clone(dict(artifact), field="artifact")
        _need(isinstance(artifact_body, dict), "invalid_artifact")
        artifact_body["relative_path"] = _safe_relative_path(
            artifact_body.get("relative_path"), field="artifact.relative_path"
        )
        artifact_body["receipt_relative_path"] = _safe_relative_path(
            artifact_body.get("receipt_relative_path"),
            field="artifact.receipt_relative_path",
        )
        artifact_body["sha256"] = _digest(
            artifact_body.get("sha256"), field="artifact.sha256"
        )
        artifact_body["size_bytes"] = _strict_int(
            artifact_body.get("size_bytes"),
            field="artifact.size_bytes",
            minimum=1,
        )
        artifact_body["receipt_sha256"] = _digest(
            artifact_body.get("receipt_sha256"),
            field="artifact.receipt_sha256",
        )
    if state_eff == ARTIFACT_PASS:
        _need(artifact_body is not None, "artifact_required_for_pass")
    else:
        _need(artifact_body is None, "artifact_for_nonpass_state")
    body: dict[str, Any] = {
        "schema": BUILD_RECORD_SCHEMA,
        "schema_version": 1,
        "key": key_body,
        "key_sha256": canonical_sha256(key_body),
        "state": state_eff,
        "deterministic": state_eff in DETERMINISTIC_REUSABLE_STATES,
        "reusable": state_eff in DETERMINISTIC_REUSABLE_STATES,
        "reason_code": str(reason_code or "").strip(),
        "evidence_origin": origin,
        "artifact": artifact_body,
    }
    body["record_sha256"] = canonical_sha256(body)
    return body


def validate_build_evidence_record(value: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_build_record")
    body = dict(value)
    expected_hash = _digest(body.pop("record_sha256", None), field="record_sha256")
    _need(
        body.get("schema") == BUILD_RECORD_SCHEMA,
        "build_record_schema_mismatch",
    )
    _need(
        body.get("schema_version") == 1,
        "build_record_schema_version_mismatch",
    )
    key = validate_build_key(body.get("key") or {})
    _need(
        _digest(body.get("key_sha256"), field="key_sha256")
        == canonical_sha256(key),
        "key_sha256_mismatch",
    )
    state = str(body.get("state") or "").strip().upper()
    _need(state in BUILD_STATES, "invalid_build_state", state)
    expected_reusable = state in DETERMINISTIC_REUSABLE_STATES
    _need(
        body.get("deterministic") is expected_reusable
        and body.get("reusable") is expected_reusable,
        "record_reuse_flag_mismatch",
    )
    canonical = make_build_evidence_record(
        key,
        state,
        evidence_origin=body.get("evidence_origin") or {},
        artifact=body.get("artifact"),
        reason_code=str(body.get("reason_code") or ""),
    )
    _need(canonical["record_sha256"] == expected_hash, "record_sha256_mismatch")
    _need(canonical == dict(value), "noncanonical_build_record")
    return canonical


def _index_payload(
    *,
    source_run_name: str,
    records: Sequence[Mapping[str, Any]],
    unresolved_observations: Sequence[Mapping[str, Any]],
    source_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated = [validate_build_evidence_record(record) for record in records]
    validated.sort(
        key=lambda record: (
            record["key_sha256"],
            record["state"],
            record["record_sha256"],
        )
    )
    unresolved = [
        _json_clone(dict(row), field="unresolved_observation")
        for row in unresolved_observations
    ]
    unresolved.sort(
        key=lambda row: (
            str(row.get("state") or ""),
            str(row.get("evidence_origin") or ""),
        )
    )
    for row in unresolved:
        _need(
            row.get("state") in BUILD_STATES,
            "invalid_build_state",
            str(row.get("state") or ""),
        )
        _need(row.get("reusable") is False, "unresolved_must_not_be_reusable")
    observations = [
        _json_clone(dict(row), field="source_observation")
        for row in source_observations
    ]
    observations.sort(
        key=lambda row: (str(row.get("logical_path") or ""), str(row.get("sha256") or ""))
    )
    state_counts = {
        state: sum(1 for record in validated if record["state"] == state)
        + sum(1 for row in unresolved if row.get("state") == state)
        for state in sorted(BUILD_STATES)
    }
    body: dict[str, Any] = {
        "schema": BUILD_INDEX_SCHEMA,
        "schema_version": 1,
        "claim_scope": "retrospective_exact_build_outcomes",
        "source_run": {"basename": _token(source_run_name, field="source_run_name")},
        "records": validated,
        "unresolved_observations": unresolved,
        "record_count": len(validated),
        "reusable_record_count": sum(
            1 for record in validated if record["reusable"] is True
        ),
        "unresolved_observation_count": len(unresolved),
        "state_counts": state_counts,
        "source_observations": observations,
        "source_observations_sha256": canonical_sha256(observations),
        "source_run_mutated": False,
        "runtime_evidence_included": False,
    }
    body["index_payload_sha256"] = canonical_sha256(body)
    return body


def build_evidence_index(
    records: Sequence[Mapping[str, Any]] = (),
    *,
    source_run_name: str = "build-evidence",
    unresolved_observations: Sequence[Mapping[str, Any]] = (),
    source_observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create a canonical in-memory index from already attested records."""

    return _index_payload(
        source_run_name=source_run_name,
        records=records,
        unresolved_observations=unresolved_observations,
        source_observations=source_observations,
    )


def validate_build_evidence_index(value: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_build_index")
    supplied = dict(value)
    expected_hash = _digest(
        supplied.pop("index_payload_sha256", None),
        field="index_payload_sha256",
    )
    _need(supplied.get("schema") == BUILD_INDEX_SCHEMA, "build_index_schema_mismatch")
    _need(supplied.get("schema_version") == 1, "build_index_schema_version_mismatch")
    _need(supplied.get("source_run_mutated") is False, "source_run_mutation_claim_invalid")
    _need(supplied.get("runtime_evidence_included") is False, "runtime_ledger_not_separate")
    records = supplied.get("records")
    unresolved = supplied.get("unresolved_observations")
    observations = supplied.get("source_observations")
    _need(isinstance(records, list), "invalid_build_records")
    _need(isinstance(unresolved, list), "invalid_unresolved_observations")
    _need(isinstance(observations, list), "invalid_source_observations")
    canonical = _index_payload(
        source_run_name=(supplied.get("source_run") or {}).get("basename"),
        records=records,
        unresolved_observations=unresolved,
        source_observations=observations,
    )
    _need(canonical["index_payload_sha256"] == expected_hash, "index_payload_sha256_mismatch")
    _need(canonical == dict(value), "noncanonical_build_index")
    return canonical


def _artifact_record_from_verified(
    verified: VerifiedHailoArtifact,
    *,
    source_root: Path,
) -> dict[str, Any]:
    artifact = {
        "kind": "hailo_hef",
        "relative_path": verified.hef_path.relative_to(source_root).as_posix(),
        "receipt_relative_path": verified.receipt_path.relative_to(source_root).as_posix(),
        "sha256": verified.hef_sha256,
        "size_bytes": verified.hef_size_bytes,
        "receipt_sha256": verified.receipt_sha256,
        "compiler_onnx_sha256": verified.compiler_onnx_sha256,
        "cache_key": verified.cache_key,
    }
    if verified.cache_meta_path is not None:
        artifact["cache_meta_relative_path"] = verified.cache_meta_path.relative_to(source_root).as_posix()
        artifact["cache_meta_sha256"] = verified.cache_meta_sha256
    return artifact


def _verify_positive_record_artifact(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> VerifiedHailoArtifact:
    artifact = record.get("artifact")
    _need(isinstance(artifact, Mapping), "artifact_required_for_pass")
    relative = _safe_relative_path(
        artifact.get("relative_path"), field="artifact.relative_path"
    )
    receipt_relative = _safe_relative_path(
        artifact.get("receipt_relative_path"),
        field="artifact.receipt_relative_path",
    )
    root = _lexical_absolute(artifact_root, label="artifact_root")
    hef = root / Path(relative)
    receipt = root / Path(receipt_relative)
    verified = verify_hailo_artifact(hef, receipt_path=receipt)
    if artifact.get("cache_meta_relative_path") is not None:
        relative_meta = _safe_relative_path(artifact["cache_meta_relative_path"],
                                            field="artifact.cache_meta_relative_path")
        _need(verified.cache_meta_path == root / relative_meta
              and verified.cache_meta_sha256 == _digest(artifact.get("cache_meta_sha256"),
                                                         field="artifact.cache_meta_sha256"),
              "artifact_cache_meta_mismatch")
    _need(
        verified.hef_sha256
        == _digest(artifact.get("sha256"), field="artifact.sha256"),
        "artifact_sha256_mismatch",
    )
    _need(
        verified.hef_size_bytes
        == _strict_int(
            artifact.get("size_bytes"), field="artifact.size_bytes", minimum=1
        ),
        "artifact_size_mismatch",
    )
    _need(
        verified.receipt_sha256
        == _digest(
            artifact.get("receipt_sha256"), field="artifact.receipt_sha256"
        ),
        "artifact_receipt_sha256_mismatch",
    )
    key = validate_build_key(record.get("key") or {})
    _need(
        verified.compiler_onnx_sha256 == key["compiler_onnx_sha256"]
        and verified.cache_key == key["backend_cache_contract_sha256"],
        "artifact_build_key_mismatch",
    )
    return verified


def lookup_build_evidence(
    index_or_path: Mapping[str, Any] | str | Path,
    exact_key: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> BuildEvidenceDecision:
    """Look up one exact request, returning HIT, MISS or CONFLICT.

    ``ARTIFACT_PASS`` requires ``artifact_root`` so the bytes can be rechecked.
    Deterministic negative outcomes need no old artifact bytes because their
    record and enclosing index are self-hashed.
    """

    if isinstance(index_or_path, Mapping):
        index = validate_build_evidence_index(index_or_path)
    else:
        index = load_build_evidence_index(index_or_path)
    key = validate_build_key(exact_key)
    key_hash = canonical_sha256(key)
    candidates = [
        record for record in index["records"]
        if record.get("key_sha256") == key_hash
    ]
    if not candidates:
        return BuildEvidenceDecision(
            status="MISS",
            key_sha256=key_hash,
            reusable=False,
            state=None,
            reason="exact_key_not_found",
        )
    reusable: list[dict[str, Any]] = []
    positive_errors: list[str] = []
    for record in candidates:
        if record.get("reusable") is not True:
            continue
        if record.get("state") == ARTIFACT_PASS:
            if artifact_root is None:
                positive_errors.append("artifact_root_required")
                continue
            try:
                _verify_positive_record_artifact(
                    record,
                    artifact_root=_lexical_absolute(
                        artifact_root, label="artifact_root"
                    ),
                )
            except BuildEvidenceError as exc:
                positive_errors.append(exc.code)
                continue
        reusable.append(record)
    if not reusable:
        return BuildEvidenceDecision(
            status="MISS",
            key_sha256=key_hash,
            reusable=False,
            state=None,
            reason=(
                "positive_artifact_unverified:"
                + ",".join(sorted(set(positive_errors)))
                if positive_errors
                else "matching_records_not_reusable"
            ),
        )
    positives = [record for record in reusable if record["state"] == ARTIFACT_PASS]
    if positives:
        artifact_hashes = {
            str((record.get("artifact") or {}).get("sha256") or "")
            for record in positives
        }
        if len(artifact_hashes) != 1:
            return BuildEvidenceDecision(
                status="CONFLICT",
                key_sha256=key_hash,
                reusable=False,
                state=None,
                reason="multiple_verified_positive_artifacts",
            )
        selected = sorted(positives, key=lambda row: row["record_sha256"])[0]
        return BuildEvidenceDecision(
            status="HIT",
            key_sha256=key_hash,
            reusable=True,
            state=ARTIFACT_PASS,
            reason="verified_positive_supersedes_prior_negative",
            record=selected,
            evidence_origin=dict(selected.get("evidence_origin") or {}),
        )
    states = {str(record["state"]) for record in reusable}
    if len(states) != 1:
        return BuildEvidenceDecision(
            status="CONFLICT",
            key_sha256=key_hash,
            reusable=False,
            state=None,
            reason="conflicting_deterministic_negative_outcomes",
        )
    selected = sorted(reusable, key=lambda row: row["record_sha256"])[0]
    return BuildEvidenceDecision(
        status="HIT",
        key_sha256=key_hash,
        reusable=True,
        state=selected["state"],
        reason="exact_deterministic_outcome",
        record=selected,
        evidence_origin=dict(selected.get("evidence_origin") or {}),
    )


def _atomic_copy_regular_exclusive(source: Path, destination: Path) -> None:
    observed = _read_regular_nofollow(source, label="materialize.source", collect=False)
    destination = _lexical_absolute(destination, label="materialize.destination")
    parent_fd = _ensure_directory_nofollow(
        destination.parent, label="materialize.destination.parent"
    )
    source_fd = -1
    temp_fd = -1
    temp_name = f".{destination.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    created = False
    try:
        source_parent_fd = _open_directory_nofollow(
            source.parent, label="materialize.source.parent"
        )
        try:
            source_fd = os.open(
                source.name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=source_parent_fd,
            )
        finally:
            os.close(source_parent_fd)
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(source_fd, _READ_CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
            offset = 0
            while offset < len(block):
                written = os.write(temp_fd, block[offset:])
                _need(written > 0, "materialize_short_write")
                offset += written
        _need(
            total == observed.size_bytes and digest.hexdigest() == observed.sha256,
            "materialize_source_changed",
        )
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        try:
            os.link(
                temp_name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise BuildEvidenceError(
                "materialize_destination_exists", os.fspath(destination)
            ) from exc
        created = True
        os.unlink(temp_name, dir_fd=parent_fd)
        with contextlib.suppress(OSError):
            os.fsync(parent_fd)
        copied = _read_regular_nofollow(
            destination, label="materialize.destination.verify", collect=False
        )
        _need(
            copied.sha256 == observed.sha256
            and copied.size_bytes == observed.size_bytes,
            "materialize_destination_mismatch",
        )
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name, dir_fd=parent_fd)
        if created:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(destination.name, dir_fd=parent_fd)
        raise
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(parent_fd)


def _admit_or_copy_identical_regular(
    source: Path,
    destination: Path,
    *,
    kind: str,
) -> tuple[bool, str]:
    """Idempotently admit one materialized file only when byte-identical.

    Returns ``(created, mode)``. Existing paths are never overwritten and are
    accepted only as regular non-symlink files with exact size and SHA-256.
    """

    kind_eff = str(kind or "artifact").strip().lower()
    _need(
        kind_eff in {"compiler", "hef", "receipt"},
        "materialize_invalid_file_kind",
        kind_eff,
    )
    source_observed = _read_regular_nofollow(
        source, label=f"materialize.{kind_eff}_source", collect=False
    )
    destination = _lexical_absolute(
        destination, label=f"materialize.{kind_eff}_destination"
    )
    if not os.path.lexists(destination):
        _atomic_copy_regular_exclusive(source, destination)
        return True, "exclusive_copy"
    try:
        info = os.lstat(destination)
    except OSError as exc:
        raise BuildEvidenceError(
            f"materialize_{kind_eff}_destination_unsafe",
            os.fspath(destination),
        ) from exc
    if stat.S_ISLNK(info.st_mode):
        raise BuildEvidenceError(
            f"materialize_{kind_eff}_destination_symlink",
            os.fspath(destination),
        )
    if not stat.S_ISREG(info.st_mode):
        raise BuildEvidenceError(
            f"materialize_{kind_eff}_destination_not_regular",
            os.fspath(destination),
        )
    destination_observed = _read_regular_nofollow(
        destination,
        label=f"materialize.{kind_eff}_destination",
        collect=False,
    )
    _need(
        destination_observed.size_bytes == source_observed.size_bytes
        and destination_observed.sha256 == source_observed.sha256,
        f"materialize_{kind_eff}_identity_conflict",
        os.fspath(destination),
    )
    return False, "preexisting_identical"


def _admit_or_copy_identical_compiler_onnx(
    source: Path,
    destination: Path,
) -> tuple[bool, str]:
    created, mode = _admit_or_copy_identical_regular(
        source, destination, kind="compiler"
    )
    return (
        created,
        "preexisting_identical_probe_output"
        if mode == "preexisting_identical"
        else mode,
    )


def materialize_verified_artifact(
    decision: BuildEvidenceDecision,
    destination_dir: str | Path,
    expected_key: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Materialize a verified positive as a Hailo build-result mapping.

    All three bound files are copied: HEF, receipt and fixed compiler ONNX.
    Existing destinations are never overwritten.  The copied set is verified
    again before the successful ``materialized_build_result`` is returned.
    The benchmark controller remains responsible for wrapping this mapping in
    its outer ``target_outcome`` (including relative case paths).
    """

    key = validate_build_key(expected_key)
    key_hash = canonical_sha256(key)
    _need(
        decision.status == "HIT"
        and decision.reusable is True
        and decision.state == ARTIFACT_PASS
        and decision.key_sha256 == key_hash
        and isinstance(decision.record, Mapping),
        "materialize_decision_not_verified_positive",
    )
    record = validate_build_evidence_record(decision.record)
    _need(record["key"] == key, "materialize_exact_key_mismatch")
    root = _lexical_absolute(artifact_root, label="artifact_root")
    verified = _verify_positive_record_artifact(record, artifact_root=root)
    destination = _lexical_absolute(destination_dir, label="destination_dir")
    descriptor = _ensure_directory_nofollow(destination, label="destination_dir")
    os.close(descriptor)
    target_hef = destination / "compiled.hef"
    target_receipt = destination / "hailo_hef_build_receipt.json"
    target_compiler = destination / verified.compiler_onnx_path.name
    created: list[Path] = []
    compiler_materialization = ""
    file_materialization: dict[str, str] = {}
    try:
        compiler_created, compiler_materialization = (
            _admit_or_copy_identical_compiler_onnx(
                verified.compiler_onnx_path, target_compiler
            )
        )
        if compiler_created:
            created.append(target_compiler)
        file_materialization["compiler_onnx"] = compiler_materialization
        copied_members = (
            ("hef", verified.hef_path, target_hef),
            ("receipt", verified.receipt_path, target_receipt),
        )
        if verified.cache_meta_path is not None:
            # A recovered published generation must remain a complete atomic
            # triplet. The compiler ONNX is an independently verified input.
            from .hailo_cache_bundle import publish_bundle
            copied_members += (("cache_meta", verified.cache_meta_path,
                                destination / "cache_meta.json"),)
            existing_snapshot = None
            if any(target.is_symlink() for _, _, target in copied_members):
                existing_snapshot = _snapshot_published_hailo_bundle(destination)
            for kind, source, target in copied_members:
                if os.path.lexists(target):
                    prior = _read_regular_nofollow(
                        existing_snapshot / target.name if existing_snapshot else target,
                        label=f"materialize_{kind}_destination", collect=False,
                    )
                    expected = _read_regular_nofollow(source, label=f"materialize_{kind}_source", collect=False)
                    _need(prior.sha256 == expected.sha256 and prior.size_bytes == expected.size_bytes,
                          f"materialize_{kind}_identity_conflict")
            meta_observed = _read_regular_nofollow(verified.cache_meta_path,
                                                  label="materialize_cache_meta", collect=True)
            meta = _strict_json(meta_observed.data or b"", label="materialize_cache_meta")

            def validate_staged(staged: Path) -> bool:
                hef_observed = _read_regular_nofollow(staged, label="staged_hef", collect=False)
                staged_receipt = _strict_json(_read_regular_nofollow(
                    staged.parent / "hailo_hef_build_receipt.json", label="staged_receipt", collect=True,
                ).data or b"", label="staged_receipt")
                staged_meta = _strict_json(_read_regular_nofollow(
                    staged.parent / "cache_meta.json", label="staged_meta", collect=True,
                ).data or b"", label="staged_meta")
                return (hef_observed.sha256 == verified.hef_sha256
                        and hef_observed.size_bytes == verified.hef_size_bytes
                        and staged_receipt == verified.receipt and staged_meta == meta)

            publish_bundle(source_hef=verified.hef_path, destination=target_hef,
                           receipt=verified.receipt, cache_meta=meta, validator=validate_staged)
            for kind, _, _ in copied_members:
                file_materialization[kind] = "atomic_bundle_publication"
            copied_members = ()
        for kind, source, target in copied_members:
            was_created, mode = _admit_or_copy_identical_regular(
                source, target, kind=kind
            )
            if was_created:
                created.append(target)
            file_materialization[kind] = mode
        target_verified = verify_hailo_artifact(
            target_hef, receipt_path=target_receipt
        )
        _need(
            target_verified.hef_sha256 == verified.hef_sha256
            and target_verified.receipt_sha256 == verified.receipt_sha256
            and target_verified.compiler_onnx_sha256
            == verified.compiler_onnx_sha256,
            "materialized_artifact_binding_mismatch",
        )
    except Exception:
        for path in reversed(created):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        raise
    return {
        "ok": True,
        "skipped": False,
        "hw_arch": key["hw_arch"],
        "backend": "exact_build_evidence",
        "error": None,
        "hef_path": os.fspath(target_hef),
        "fixed_onnx_path": os.fspath(target_compiler),
        "build_receipt_path": os.fspath(target_receipt),
        "build_receipt": target_verified.receipt,
        "cache_hit": True,
        "cache_key": target_verified.cache_key,
        "failure_kind": None,
        "details": {
            "exact_build_evidence": {
                "state": ARTIFACT_PASS,
                "key_sha256": key_hash,
                "record_sha256": record["record_sha256"],
                "evidence_origin": dict(record.get("evidence_origin") or {}),
                "artifact_sha256": target_verified.hef_sha256,
                "artifact_size_bytes": target_verified.hef_size_bytes,
                "compiler_onnx_materialization": compiler_materialization,
                "file_materialization": file_materialization,
                "verified_after_materialization": True,
            }
        },
    }


def _artifact_coordinates(path: Path, root: Path) -> dict[str, Any]:
    parts = path.relative_to(root).parts
    model = ""
    if "models" in parts:
        index = parts.index("models")
        if index + 1 < len(parts):
            model = parts[index + 1]
    boundary: int | None = None
    for part in parts:
        match = re.fullmatch(r"b(\d+)", part)
        if match:
            boundary = int(match.group(1))
    stage = path.parent.name
    arch = path.parent.parent.name if path.parent.parent.name else ""
    return {
        "model": model,
        "boundary": boundary,
        "stage": stage,
        "hw_arch": _normalize_hw_arch(arch) if arch else "",
    }


def _nearest_split_manifest(
    artifact_dir: Path,
    *,
    root: Path,
    attestor: _ReadOnlyAttestor,
) -> tuple[dict[str, Any] | None, Path | None]:
    current = artifact_dir
    while _path_inside(current, root):
        candidate = current / "split_manifest.json"
        if os.path.lexists(candidate):
            payload, _ = attestor.json(candidate)
            return payload, candidate
        if current == root:
            break
        current = current.parent
    return None, None


def _full_source_identity(
    *,
    stage: str,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    manifest_path: Path | None,
    root: Path,
    attestor: _ReadOnlyAttestor,
) -> tuple[str, str]:
    if manifest is not None and manifest_path is not None:
        for field in ("full_model", "source_full_model"):
            raw = manifest.get(field)
            if not raw:
                continue
            try:
                candidate = _resolve_bounded_manifest_path(
                    base=manifest_path.parent,
                    raw_value=raw,
                    root=root,
                    field=f"split_manifest.{field}",
                )
                if os.path.lexists(candidate):
                    observed = attestor.file(candidate)
                    return observed.sha256, candidate.relative_to(root).as_posix()
            except BuildEvidenceError:
                continue
    if str(stage).lower() == "full":
        return (
            _digest(
                receipt.get("source_onnx_sha256"),
                field="receipt.source_onnx_sha256",
            ),
            "receipt:source_onnx_sha256",
        )
    raise BuildEvidenceError("full_source_onnx_unavailable")


def _key_from_context(
    context: Mapping[str, Any],
    *,
    expected_compiler_sha256: str | None = None,
    expected_cache_key: str | None = None,
) -> dict[str, Any]:
    _need(context.get("schema") == BUILD_CONTEXT_SCHEMA, "build_context_schema_mismatch")
    _need(context.get("schema_version") == 1, "build_context_schema_version_mismatch")
    body = dict(context)
    expected_hash = _digest(body.pop("context_sha256", None), field="context_sha256")
    _need(canonical_sha256(body) == expected_hash, "context_sha256_mismatch")
    key = validate_build_key(context.get("key") or {})
    if expected_compiler_sha256 is not None:
        _need(
            key["compiler_onnx_sha256"]
            == _digest(expected_compiler_sha256, field="expected_compiler_sha256"),
            "compiler_onnx_sha256_mismatch",
        )
    if expected_cache_key is not None:
        _need(
            key["backend_cache_contract_sha256"]
            == _digest(expected_cache_key, field="expected_cache_key"),
            "hailo_cache_key_mismatch",
        )
    return key


def _key_from_attempt(
    attempt: Mapping[str, Any],
    *,
    expected_compiler_sha256: str | None = None,
) -> dict[str, Any]:
    _need(attempt.get("schema") == HAILO_ATTEMPT_SCHEMA, "hailo_attempt_schema_mismatch")
    _need(attempt.get("schema_version") == 1, "hailo_attempt_schema_version_mismatch")
    body = dict(attempt)
    expected_hash = _digest(body.pop("attempt_sha256", None), field="attempt_sha256")
    _need(canonical_sha256(body) == expected_hash, "attempt_sha256_mismatch")
    key = validate_build_key(attempt.get("key") or {})
    cache_payload = attempt.get("cache_payload")
    if cache_payload is not None:
        payload = _hailo_v3_cache_payload(cache_payload)
        cache_key = canonical_sha256(payload)
        _need(
            cache_key
            == _digest(attempt.get("cache_key"), field="attempt.cache_key")
            == key["backend_cache_contract_sha256"],
            "hailo_cache_key_mismatch",
        )
    if expected_compiler_sha256 is not None:
        _need(
            key["compiler_onnx_sha256"]
            == _digest(expected_compiler_sha256, field="expected_compiler_sha256"),
            "compiler_onnx_sha256_mismatch",
        )
    return key


def _workflow_log_contexts(
    workflow_log: str | Path | None,
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], FileObservation | None]:
    if workflow_log is None:
        return {}, None
    observed = _read_regular_nofollow(
        workflow_log,
        label="workflow_log",
        collect=True,
        size_limit=512 * 1024 * 1024,
    )
    text = (observed.data or b"").decode("utf-8", errors="replace")
    contexts: dict[tuple[str, int, str], dict[str, Any]] = {}
    pattern = re.compile(
        r"\[benchmarkset:([^\]]+)\]\s+\(b(\d+)\s+(hailo(?:8l?|10h?))\)\s*(.*)$",
        re.IGNORECASE,
    )
    cache_pattern = re.compile(
        r"\[hailo\]\[cache\]\s+miss\s+key=([0-9a-f]{12,64})\s+net=([^\s]+)",
        re.IGNORECASE,
    )
    cache_pattern_v27920 = re.compile(
        r"\[hailo-cache\]\s+MISS\b.*?\bmodel=([^\s]+)\s+"
        r"identity=([0-9a-f]{12,64})\b",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        key = (
            match.group(1).strip(),
            int(match.group(2)),
            _normalize_hw_arch(match.group(3)),
        )
        row = contexts.setdefault(
            key, {"lines": [], "cache_prefixes": [], "net_names": []}
        )
        remainder = match.group(4)
        row["lines"].append(remainder)
        cache_match = cache_pattern.search(remainder)
        if cache_match:
            prefix = cache_match.group(1).lower()
            if prefix not in row["cache_prefixes"]:
                row["cache_prefixes"].append(prefix)
            net_name = cache_match.group(2).strip()
            if net_name not in row["net_names"]:
                row["net_names"].append(net_name)
        else:
            cache_match_v27920 = cache_pattern_v27920.search(remainder)
            if cache_match_v27920:
                net_name = cache_match_v27920.group(1).strip()
                prefix = cache_match_v27920.group(2).lower()
                if prefix not in row["cache_prefixes"]:
                    row["cache_prefixes"].append(prefix)
                if net_name not in row["net_names"]:
                    row["net_names"].append(net_name)
    for row in contexts.values():
        row["text"] = "\n".join(row.pop("lines"))
    return contexts, observed


def _compiler_file_for_result(directory: Path, result: Mapping[str, Any]) -> Path:
    raw = str(result.get("fixed_onnx_path") or "").strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(directory / Path(raw).name)
    try:
        for child in sorted(directory.iterdir()):
            if child.name.endswith("_hailo_fixed.onnx") and child not in candidates:
                candidates.append(child)
    except OSError as exc:
        raise BuildEvidenceError(
            "unsafe_artifact_directory", os.fspath(directory)
        ) from exc
    regular: list[Path] = []
    for candidate in candidates:
        if not os.path.lexists(candidate):
            continue
        info = os.lstat(candidate)
        if stat.S_ISLNK(info.st_mode):
            raise BuildEvidenceError(
                "unsafe_compiler_onnx_symlink", os.fspath(candidate)
            )
        if stat.S_ISREG(info.st_mode):
            regular.append(candidate)
    unique = list(dict.fromkeys(regular))
    _need(
        len(unique) == 1,
        "compiler_onnx_not_unique",
        os.fspath(directory),
    )
    return unique[0]


def _sibling_verified_artifact(
    result_path: Path,
    verified_by_receipt: Mapping[str, VerifiedHailoArtifact],
) -> VerifiedHailoArtifact | None:
    target_directory = result_path.parent
    stage = target_directory.name
    hailo_directory = target_directory.parent.parent
    candidates: list[VerifiedHailoArtifact] = []
    for raw_path, verified in verified_by_receipt.items():
        receipt = Path(raw_path)
        if (
            receipt.parent.name == stage
            and receipt.parent.parent.parent == hailo_directory
            and receipt.parent != target_directory
        ):
            candidates.append(verified)
    return candidates[0] if len(candidates) == 1 else None


def _legacy_negative_key(
    *,
    result_path: Path,
    result: Mapping[str, Any],
    compiler_observed: FileObservation,
    sibling: VerifiedHailoArtifact,
    compiler_versions: Mapping[str, str],
    log_context: Mapping[str, Any],
    root: Path,
    attestor: _ReadOnlyAttestor,
) -> dict[str, Any]:
    coordinates = _artifact_coordinates(result_path, root)
    hw_arch = coordinates["hw_arch"]
    compiler_version = compiler_versions.get(hw_arch, "")
    _token(compiler_version, field=f"compiler_version.{hw_arch}")
    payload = _json_clone(sibling.cache_payload, field="sibling.cache_payload")
    payload["model_sha256"] = compiler_observed.sha256
    payload["hw_arch"] = hw_arch
    payload["hailo_sdk_version"] = compiler_version
    net_names = list(log_context.get("net_names") or [])
    result_net = str(result.get("net_name") or "").strip()
    if result_net:
        net_names.append(result_net)
    net_names = sorted(set(name for name in net_names if name))
    _need(len(net_names) == 1, "legacy_net_name_not_unique")
    payload["net_name"] = net_names[0]
    payload = _hailo_v3_cache_payload(payload)
    calculated_cache_key = canonical_sha256(payload)
    prefixes = sorted(
        set(str(value).lower() for value in log_context.get("cache_prefixes") or [])
    )
    _need(
        len(prefixes) == 1
        and _CACHE_PREFIX.fullmatch(prefixes[0]) is not None
        and calculated_cache_key.startswith(prefixes[0]),
        "legacy_cache_prefix_mismatch",
    )
    manifest, manifest_path = _nearest_split_manifest(
        result_path.parent,
        root=root,
        attestor=attestor,
    )
    full_sha, _ = _full_source_identity(
        stage=coordinates["stage"],
        receipt=sibling.receipt,
        manifest=manifest,
        manifest_path=manifest_path,
        root=root,
        attestor=attestor,
    )
    endpoint_sha = boundary_endpoint_contract_sha256(
        stage=coordinates["stage"],
        cache_payload=payload,
        split_manifest=manifest,
    )
    projected_receipt = dict(sibling.receipt)
    projected_receipt.update({
        "compiler_onnx_sha256": compiler_observed.sha256,
        "hw_arch": hw_arch,
        "hailo_sdk_version": compiler_version,
        "cache_key": calculated_cache_key,
        "cache_payload": payload,
    })
    return canonical_build_key_from_hailo_v3(
        projected_receipt,
        full_source_onnx_sha256=full_sha,
        boundary_endpoint_contract_sha256=endpoint_sha,
        expected_cache_key=calculated_cache_key,
    )


def _unresolved(
    *,
    state: str,
    origin: Mapping[str, Any],
    reason_code: str,
    partial_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    state_eff = str(state or "").strip().upper()
    _need(state_eff in BUILD_STATES, "invalid_build_state", state_eff)
    return {
        "state": state_eff,
        "reusable": False,
        "reason_code": str(reason_code or "incomplete_exact_identity"),
        "evidence_origin": _json_clone(dict(origin), field="evidence_origin"),
        "partial_artifacts": [
            _json_clone(dict(row), field="partial_artifact")
            for row in partial_artifacts
        ],
    }


def _modern_attempt_key(
    attempt: Mapping[str, Any], *, directory: Path, attestor: _ReadOnlyAttestor,
) -> tuple[dict[str, Any], str]:
    """Admit only recorded exact identities; never infer one from a boundary."""
    containers = [attempt]
    for field in ("metadata", "details", "calib_info", "result_summary"):
        value = attempt.get(field)
        if isinstance(value, Mapping):
            containers.append(value)
            for child in ("details", "calib_info"):
                if isinstance(value.get(child), Mapping):
                    containers.append(value[child])
    candidates: list[dict[str, Any]] = []
    for container in containers:
        evidence = container.get("build_evidence")
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("key"), Mapping):
            continue
        key = validate_build_key(evidence["key"])
        payload = _hailo_v3_cache_payload(evidence.get("cache_payload_v3") or {})
        _need(canonical_sha256(payload)
              == _digest(evidence.get("cache_key_v3"), field="attempt.cache_key_v3")
              == key["backend_cache_contract_sha256"], "hailo_cache_key_mismatch")
        _need(payload["model_sha256"] == key["compiler_onnx_sha256"],
              "compiler_onnx_sha256_mismatch")
        _need(canonical_build_key_from_hailo_v3_payload(
            payload, builder_source_onnx_sha256=key["builder_source_onnx_sha256"],
            full_source_onnx_sha256=key["full_source_onnx_sha256"],
            boundary_endpoint_contract_sha256=key["boundary_endpoint_contract_sha256"],
            expected_cache_key=key["backend_cache_contract_sha256"],
        ) == key, "terminal_attempt_key_payload_mismatch")
        candidates.append(key)
    compiler_sha = _optional_digest(attempt.get("compiler_onnx_sha256"),
                                    field="attempt.compiler_onnx_sha256")
    if candidates:
        _need(all(key == candidates[0] for key in candidates), "conflicting_attempt_exact_keys")
        key = candidates[0]
        key_origin = "terminal_receipt_recorded_exact_key"
    else:
        # A neighboring context is usable only when this attempt independently
        # records its compiler digest. Missing/deleted inputs cannot be guessed.
        _need(bool(compiler_sha), "terminal_attempt_exact_identity_missing")
        old_attempt = directory / "hailo_hef_build_attempt.json"
        context_path = directory / "build_evidence_context.json"
        if os.path.lexists(old_attempt):
            value, _ = attestor.json(old_attempt)
            key = _key_from_attempt(value, expected_compiler_sha256=compiler_sha)
            key_origin = old_attempt.relative_to(attestor.root).as_posix()
        elif os.path.lexists(context_path):
            value, _ = attestor.json(context_path)
            key = _key_from_context(value, expected_compiler_sha256=compiler_sha)
            key_origin = context_path.relative_to(attestor.root).as_posix()
        else:
            raise BuildEvidenceError("terminal_attempt_exact_identity_missing")
    if compiler_sha:
        _need(compiler_sha == key["compiler_onnx_sha256"], "compiler_onnx_sha256_mismatch")
    if attempt.get("hw_arch"):
        _need(_normalize_hw_arch(attempt["hw_arch"]) == key["hw_arch"], "hw_arch_mismatch")
    source_sha = _optional_digest(attempt.get("source_onnx_sha256"),
                                  field="attempt.source_onnx_sha256")
    if source_sha:
        _need(source_sha == key["builder_source_onnx_sha256"], "builder_source_onnx_sha256_mismatch")
    return key, key_origin


def _harvest_modern_attempts(
    paths: Sequence[Path], *, attestor: _ReadOnlyAttestor,
    verified_by_receipt: Mapping[str, VerifiedHailoArtifact],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    snapshots: list[tuple[Path, dict[str, Any], FileObservation]] = []
    for path in paths:
        try:
            body, observed = attestor.json(path)
            _need(body.get("schema") == "onnx-splitpoint/hailo-build-attempt-receipt"
                  and body.get("schema_version") == 2, "terminal_attempt_schema_mismatch")
            _need(body.get("result_summary") is None or isinstance(body.get("result_summary"), Mapping),
                  "terminal_attempt_result_summary_invalid")
            snapshots.append((path, body, observed))
        except BuildEvidenceError as exc:
            unresolved.append(_unresolved(state=ABORTED_UNKNOWN,
                origin={"attempt": path.relative_to(attestor.root).as_posix()},
                reason_code=f"invalid_terminal_attempt:{exc.code}"))
    final_ids = {(path.parent, str(body.get("attempt_id")))
                 for path, body, _ in snapshots if body.get("terminal") is True}
    seen: set[tuple[Path, str]] = set()
    for path, body, observed in snapshots:
        terminal = body.get("terminal") is True
        if not terminal and (path.parent, str(body.get("attempt_id"))) in final_ids:
            continue
        normalized = {key: value for key, value in body.items()
                      if key not in {"immutable_receipt", "immutable_receipt_sha256"}}
        fingerprint = (path.parent, canonical_sha256(normalized))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        directory = path.parent.parent if path.parent.name == "hailo_attempt_receipts" else path.parent
        result = dict(body.get("result_summary") or {})
        result.update({key: body[key] for key in (
            "error", "failure_kind", "unsupported_reason", "timed_out", "returncode",
            "timeout_kind", "details", "semantic_status",
        ) if key in body})
        state = classify_build_outcome(result, terminal=terminal,
            log_text="\n".join(str(body.get(field) or "")
                               for field in ("stdout_tail", "stderr_tail", "error_class")))
        origin = {
            "kind": "hailo_terminal_attempt_receipt" if terminal else "hailo_started_attempt_receipt",
            "attempt": path.relative_to(attestor.root).as_posix(),
            "attempt_sha256": observed.sha256,
            "coordinates": _artifact_coordinates(directory / "terminal_attempt.json", attestor.root),
        }
        if result.get("skipped") is True and result.get("ok") is not True:
            unresolved.append(_unresolved(state=ABORTED_UNKNOWN, origin=origin,
                reason_code="lookup_or_probe_receipt_not_compiler_evidence"))
            continue
        if state == ARTIFACT_PASS:
            if os.fspath(directory / "hailo_hef_build_receipt.json") in verified_by_receipt:
                continue
            unresolved.append(_unresolved(state=ABORTED_UNKNOWN, origin=origin,
                reason_code="positive_result_without_verified_artifact"))
            continue
        try:
            key, key_origin = _modern_attempt_key(body, directory=directory, attestor=attestor)
            origin["key_origin"] = key_origin
            records.append(make_build_evidence_record(key, state, evidence_origin=origin,
                reason_code="terminal_attempt_exact_identity" if terminal else "attempt_without_terminal_result"))
        except BuildEvidenceError as exc:
            unresolved.append(_unresolved(state=state, origin=origin,
                reason_code=f"exact_key_unavailable:{exc.code}"))
    return records, unresolved


def harvest_b5_run(
    run_dir: str | Path,
    *,
    workflow_log: str | Path | None = None,
) -> dict[str, Any]:
    """Harvest exact build outcomes without mutating ``run_dir``.

    Legacy deterministic negatives are admitted only when a sibling v3 cache
    payload, a unique target DFC version, the fixed ONNX bytes and the logged
    cache-key prefix jointly reconstruct the request. Everything else remains
    a visible, non-reusable unresolved observation.
    """

    root = _lexical_absolute(run_dir, label="source_run")
    attestor = _ReadOnlyAttestor(root)
    evidence_files = _discover_evidence_files(root, bundle_snapshots=attestor.bundle_snapshots)
    contexts, workflow_observed = _workflow_log_contexts(workflow_log)
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    resolved_log_coordinates: set[tuple[str, int, str]] = set()
    verified_by_receipt: dict[str, VerifiedHailoArtifact] = {}
    compiler_versions_by_arch: dict[str, set[str]] = {}

    receipt_paths = [
        path for path in evidence_files
        if path.name == "hailo_hef_build_receipt.json"
    ]
    for receipt_path in receipt_paths:
        hef_path = receipt_path.parent / "compiled.hef"
        try:
            verified = verify_hailo_artifact(
                hef_path, receipt_path=receipt_path,
                _bundle_snapshot=attestor.bundle_snapshots.get(receipt_path.parent),
            )
            _need(attestor.file(verified.hef_path).sha256 == verified.hef_sha256,
                  "input_changed_between_reads", "hef")
            _need(attestor.file(verified.compiler_onnx_path).sha256 == verified.compiler_onnx_sha256,
                  "input_changed_between_reads", "compiler_onnx")
            _need(attestor.json(verified.receipt_path)[1].sha256 == verified.receipt_sha256,
                  "input_changed_between_reads", "receipt")
            if verified.cache_meta_path is not None:
                _need(attestor.json(verified.cache_meta_path)[1].sha256 == verified.cache_meta_sha256,
                      "input_changed_between_reads", "cache_meta")
            verified_by_receipt[os.fspath(receipt_path)] = verified
            arch = _normalize_hw_arch(verified.receipt.get("hw_arch"))
            version = _token(
                verified.receipt.get("hailo_sdk_version"),
                field="receipt.hailo_sdk_version",
            )
            compiler_versions_by_arch.setdefault(arch, set()).add(version)
        except BuildEvidenceError as exc:
            partial: list[dict[str, Any]] = []
            for candidate, kind in ((receipt_path, "receipt"), (hef_path, "hailo_hef")):
                if os.path.lexists(candidate):
                    try:
                        observed = attestor.file(candidate)
                        partial.append({
                            "kind": kind,
                            "relative_path": candidate.relative_to(root).as_posix(),
                            "sha256": observed.sha256,
                            "size_bytes": observed.size_bytes,
                        })
                    except BuildEvidenceError:
                        pass
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={"receipt": receipt_path.relative_to(root).as_posix()},
                reason_code=f"invalid_positive_artifact:{exc.code}",
                partial_artifacts=partial,
            ))

    compiler_versions = {
        arch: next(iter(versions))
        for arch, versions in compiler_versions_by_arch.items()
        if len(versions) == 1
    }

    # First create positive records. Their complete receipts also seed exact
    # legacy-negative reconstruction for a sibling architecture.
    for raw_receipt, verified in sorted(verified_by_receipt.items()):
        receipt_path = Path(raw_receipt)
        coordinates = _artifact_coordinates(receipt_path, root)
        try:
            context_path = receipt_path.parent / "build_evidence_context.json"
            if os.path.lexists(context_path):
                context, context_observed = attestor.json(context_path)
                key = _key_from_context(
                    context,
                    expected_compiler_sha256=verified.compiler_onnx_sha256,
                    expected_cache_key=verified.cache_key,
                )
                context_origin: dict[str, Any] = {
                    "context": context_path.relative_to(root).as_posix(),
                    "context_sha256": context_observed.sha256,
                }
            else:
                manifest, manifest_path = _nearest_split_manifest(
                    receipt_path.parent,
                    root=root,
                    attestor=attestor,
                )
                full_sha, full_origin = _full_source_identity(
                    stage=coordinates["stage"],
                    receipt=verified.receipt,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    root=root,
                    attestor=attestor,
                )
                endpoint_sha = boundary_endpoint_contract_sha256(
                    stage=coordinates["stage"],
                    cache_payload=verified.cache_payload,
                    split_manifest=manifest,
                )
                key = canonical_build_key_from_hailo_v3(
                    verified.receipt,
                    full_source_onnx_sha256=full_sha,
                    boundary_endpoint_contract_sha256=endpoint_sha,
                    expected_cache_key=verified.cache_key,
                )
                context_origin = {
                    "full_source": full_origin,
                    "boundary_endpoint_contract_sha256": endpoint_sha,
                }
            origin = {
                "kind": "hailo_v2_receipt_v3_cache",
                "receipt": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": verified.receipt_sha256,
                "cache_key": verified.cache_key,
                "coordinates": coordinates,
                **context_origin,
            }
            records.append(make_build_evidence_record(
                key,
                ARTIFACT_PASS,
                evidence_origin=origin,
                artifact=_artifact_record_from_verified(verified, source_root=root),
                reason_code="verified_hailo_v2_receipt_v3_cache",
            ))
            if coordinates["model"] and coordinates["boundary"] is not None:
                resolved_log_coordinates.add((
                    coordinates["model"],
                    coordinates["boundary"],
                    coordinates["hw_arch"],
                ))
        except BuildEvidenceError as exc:
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={
                    "receipt": receipt_path.relative_to(root).as_posix(),
                    "receipt_sha256": verified.receipt_sha256,
                    "coordinates": coordinates,
                },
                reason_code=f"positive_exact_key_incomplete:{exc.code}",
                partial_artifacts=[
                    _artifact_record_from_verified(verified, source_root=root)
                ],
            ))

    result_paths = [
        path for path in evidence_files
        if path.name == "hailo_hef_build_result.json"
    ]
    consumed_attempts: set[Path] = set()
    result_directories: set[Path] = set()
    for result_path in result_paths:
        result_directories.add(result_path.parent)
        try:
            result, result_observed = attestor.json(result_path)
        except BuildEvidenceError as exc:
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={"result": result_path.relative_to(root).as_posix()},
                reason_code=f"invalid_terminal_result:{exc.code}",
            ))
            continue
        coordinates = _artifact_coordinates(result_path, root)
        if result.get("skipped") is True and result.get("ok") is not True:
            unresolved.append(_unresolved(state=ABORTED_UNKNOWN,
                origin={"result": result_path.relative_to(root).as_posix(),
                        "result_sha256": result_observed.sha256, "coordinates": coordinates},
                reason_code="lookup_or_probe_receipt_not_compiler_evidence"))
            continue
        if coordinates["model"] and coordinates["boundary"] is not None:
            resolved_log_coordinates.add((
                coordinates["model"],
                coordinates["boundary"],
                coordinates["hw_arch"],
            ))
        log_context = contexts.get((
            coordinates["model"],
            coordinates["boundary"],
            coordinates["hw_arch"],
        ), {}) if coordinates["boundary"] is not None else {}
        state = classify_build_outcome(
            result,
            log_text=str(log_context.get("text") or ""),
            terminal=True,
        )
        valid_positive = verified_by_receipt.get(
            os.fspath(result_path.parent / "hailo_hef_build_receipt.json")
        )
        if valid_positive is not None and state == ARTIFACT_PASS:
            continue
        partial: list[dict[str, Any]] = []
        compiler_observed: FileObservation | None = None
        try:
            compiler_path = _compiler_file_for_result(result_path.parent, result)
            compiler_observed = attestor.file(compiler_path)
            partial.append({
                "kind": "compiler_onnx",
                "relative_path": compiler_path.relative_to(root).as_posix(),
                "sha256": compiler_observed.sha256,
                "size_bytes": compiler_observed.size_bytes,
            })
        except BuildEvidenceError:
            compiler_observed = None
        if state == ARTIFACT_PASS:
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={
                    "result": result_path.relative_to(root).as_posix(),
                    "result_sha256": result_observed.sha256,
                    "coordinates": coordinates,
                },
                reason_code="positive_result_without_verified_artifact",
                partial_artifacts=partial,
            ))
            continue
        key: dict[str, Any] | None = None
        key_origin = ""
        failure_reason = ""
        attempt_path = result_path.parent / "hailo_hef_build_attempt.json"
        context_path = result_path.parent / "build_evidence_context.json"
        try:
            if os.path.lexists(attempt_path):
                attempt, attempt_observed = attestor.json(attempt_path)
                key = _key_from_attempt(
                    attempt,
                    expected_compiler_sha256=(
                        compiler_observed.sha256 if compiler_observed else None
                    ),
                )
                consumed_attempts.add(attempt_path)
                key_origin = (
                    f"attempt:{attempt_path.relative_to(root).as_posix()}"
                    f":{attempt_observed.sha256}"
                )
            elif os.path.lexists(context_path):
                context, context_observed = attestor.json(context_path)
                key = _key_from_context(
                    context,
                    expected_compiler_sha256=(
                        compiler_observed.sha256 if compiler_observed else None
                    ),
                )
                key_origin = (
                    f"context:{context_path.relative_to(root).as_posix()}"
                    f":{context_observed.sha256}"
                )
            elif compiler_observed is not None:
                sibling = _sibling_verified_artifact(
                    result_path, verified_by_receipt
                )
                _need(sibling is not None, "legacy_sibling_v3_receipt_missing")
                key = _legacy_negative_key(
                    result_path=result_path,
                    result=result,
                    compiler_observed=compiler_observed,
                    sibling=sibling,
                    compiler_versions=compiler_versions,
                    log_context=log_context,
                    root=root,
                    attestor=attestor,
                )
                key_origin = "legacy_sibling_v3_plus_logged_cache_prefix"
            else:
                raise BuildEvidenceError("compiler_onnx_unavailable")
        except BuildEvidenceError as exc:
            failure_reason = exc.code
        origin = {
            "kind": "hailo_terminal_result",
            "result": result_path.relative_to(root).as_posix(),
            "result_sha256": result_observed.sha256,
            "coordinates": coordinates,
            "key_origin": key_origin,
        }
        if workflow_observed is not None and log_context:
            origin["workflow_log_sha256"] = workflow_observed.sha256
        if key is None:
            unresolved.append(_unresolved(
                state=state,
                origin=origin,
                reason_code=f"exact_key_unavailable:{failure_reason or 'unknown'}",
                partial_artifacts=partial,
            ))
        else:
            records.append(make_build_evidence_record(
                key,
                state,
                evidence_origin=origin,
                reason_code="terminal_result_exact_identity",
            ))

    attempt_paths = [
        path for path in evidence_files
        if path.name == "hailo_hef_build_attempt.json"
    ]
    for attempt_path in attempt_paths:
        if attempt_path in consumed_attempts:
            continue
        try:
            attempt, attempt_observed = attestor.json(attempt_path)
            key = _key_from_attempt(attempt)
            records.append(make_build_evidence_record(
                key,
                ABORTED_UNKNOWN,
                evidence_origin={
                    "kind": "hailo_attempt_without_terminal_result",
                    "attempt": attempt_path.relative_to(root).as_posix(),
                    "attempt_sha256": attempt_observed.sha256,
                    "coordinates": _artifact_coordinates(attempt_path, root),
                },
                reason_code="attempt_without_terminal_result",
            ))
        except BuildEvidenceError as exc:
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={"attempt": attempt_path.relative_to(root).as_posix()},
                reason_code=f"invalid_or_incomplete_attempt:{exc.code}",
            ))

    modern_paths = [path for path in evidence_files
                    if path.name == "terminal_attempt.json"
                    or (path.parent.name == "hailo_attempt_receipts"
                        and path.name.startswith("attempt_") and path.name.endswith(".json"))]
    modern_records, modern_unresolved = _harvest_modern_attempts(
        modern_paths, attestor=attestor, verified_by_receipt=verified_by_receipt,
    )
    records.extend(modern_records)
    unresolved.extend(modern_unresolved)

    # A stopped old B5 run may have no terminal JSON at all. Preserve each
    # unmatched logged cache-miss as ABORTED_UNKNOWN and bind any fixed ONNX
    # that survived, without promoting a 12-character cache prefix to an exact
    # reusable key.
    fixed_candidates = [
        path for path in evidence_files
        if path.name.endswith("_hailo_fixed.onnx")
    ]
    for log_key, log_context in sorted(contexts.items()):
        if log_key in resolved_log_coordinates:
            continue
        prefixes = sorted(set(log_context.get("cache_prefixes") or []))
        if not prefixes:
            continue
        model, boundary, hw_arch = log_key
        partial: list[dict[str, Any]] = []
        for candidate in fixed_candidates:
            coordinates = _artifact_coordinates(candidate, root)
            if (
                coordinates["model"] == model
                and coordinates["boundary"] == boundary
                and coordinates["hw_arch"] == hw_arch
            ):
                observed = attestor.file(candidate)
                partial.append({
                    "kind": "compiler_onnx",
                    "relative_path": candidate.relative_to(root).as_posix(),
                    "sha256": observed.sha256,
                    "size_bytes": observed.size_bytes,
                })
        origin: dict[str, Any] = {
            "kind": "workflow_cache_miss_without_terminal_result",
            "coordinates": {
                "model": model,
                "boundary": boundary,
                "hw_arch": hw_arch,
                "stage": "part1",
            },
            "cache_key_prefixes": prefixes,
            "net_names": sorted(set(log_context.get("net_names") or [])),
        }
        if workflow_observed is not None:
            origin["workflow_log_sha256"] = workflow_observed.sha256
        unresolved.append(_unresolved(
            state=ABORTED_UNKNOWN,
            origin=origin,
            reason_code="logged_attempt_without_terminal_exact_identity",
            partial_artifacts=partial,
        ))

    verified_hef_paths = {
        verified.hef_path for verified in verified_by_receipt.values()
    }
    for hef_path in (
        path for path in evidence_files if path.name == "compiled.hef"
    ):
        if (hef_path in verified_hef_paths or hef_path.parent in result_directories
                or os.fspath(hef_path.parent / "hailo_hef_build_receipt.json") in verified_by_receipt):
            continue
        try:
            observed = attestor.file(hef_path)
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={
                    "kind": "unbound_partial_hef",
                    "coordinates": _artifact_coordinates(hef_path, root),
                },
                reason_code="compiled_hef_without_terminal_identity",
                partial_artifacts=[{
                    "kind": "hailo_hef",
                    "relative_path": hef_path.relative_to(root).as_posix(),
                    "sha256": observed.sha256,
                    "size_bytes": observed.size_bytes,
                }],
            ))
        except BuildEvidenceError as exc:
            unresolved.append(_unresolved(
                state=ABORTED_UNKNOWN,
                origin={"hef": hef_path.relative_to(root).as_posix()},
                reason_code=f"unsafe_partial_hef:{exc.code}",
            ))

    # Remove byte-identical duplicate records without collapsing different
    # observations or conflicting outcomes.
    records_by_hash = {
        record["record_sha256"]: record for record in records
    }
    source_observations = attestor.stable_snapshot()
    if workflow_observed is not None:
        workflow_recheck = _read_regular_nofollow(
            workflow_observed.path,
            label="workflow_log_recheck",
            collect=False,
        )
        _need(
            workflow_recheck.sha256 == workflow_observed.sha256
            and workflow_recheck.stable_identity()
            == workflow_observed.stable_identity(),
            "workflow_log_modified_during_harvest",
        )
    return _index_payload(
        source_run_name=root.name,
        records=list(records_by_hash.values()),
        unresolved_observations=unresolved,
        source_observations=source_observations,
    )


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    output = _lexical_absolute(path, label="output")
    parent_fd = _ensure_directory_nofollow(output.parent, label="output.parent")
    temp_fd = -1
    temp_name = f".{output.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    data = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(data):
            written = os.write(temp_fd, data[offset:])
            _need(written > 0, "output_short_write")
            offset += written
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        try:
            os.link(
                temp_name,
                output.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise BuildEvidenceError("output_already_exists", os.fspath(output)) from exc
        os.unlink(temp_name, dir_fd=parent_fd)
        with contextlib.suppress(OSError):
            os.fsync(parent_fd)
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name, dir_fd=parent_fd)
        os.close(parent_fd)


def write_build_evidence_index(
    output: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    validated = validate_build_evidence_index(payload)
    output_path = _lexical_absolute(output, label="output")
    _atomic_write_json_exclusive(output_path, validated)
    observed = _read_regular_nofollow(
        output_path,
        label="output.verify",
        collect=True,
        size_limit=_JSON_LIMIT,
    )
    written = _strict_json(observed.data or b"", label="output.verify")
    _need(written == validated, "output_verification_mismatch")
    return output_path


def create_build_evidence_index(
    *,
    run_dir: str | Path,
    output: str | Path,
    workflow_log: str | Path | None = None,
) -> dict[str, Any]:
    root = _lexical_absolute(run_dir, label="source_run")
    output_path = validate_external_output_path(
        source_run=root,
        output=output,
        label="output",
    )
    payload = harvest_b5_run(root, workflow_log=workflow_log)
    write_build_evidence_index(output_path, payload)
    return payload


def load_build_evidence_index(path: str | Path) -> dict[str, Any]:
    observed = _read_regular_nofollow(
        path,
        label="build_evidence_index",
        collect=True,
        size_limit=_JSON_LIMIT,
    )
    payload = _strict_json(
        observed.data or b"", label="build_evidence_index"
    )
    _need(isinstance(payload, Mapping), "json_object_required", "build_evidence_index")
    return validate_build_evidence_index(payload)


def verify_build_evidence_index(
    index_or_path: Mapping[str, Any] | str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    index = (
        validate_build_evidence_index(index_or_path)
        if isinstance(index_or_path, Mapping)
        else load_build_evidence_index(index_or_path)
    )
    positive_count = 0
    positive_verified = 0
    positive_failures: list[dict[str, str]] = []
    for record in index["records"]:
        if record["state"] != ARTIFACT_PASS:
            continue
        positive_count += 1
        if artifact_root is None:
            positive_failures.append({
                "key_sha256": record["key_sha256"],
                "reason": "artifact_root_required",
            })
            continue
        try:
            _verify_positive_record_artifact(
                record,
                artifact_root=_lexical_absolute(
                    artifact_root, label="artifact_root"
                ),
            )
            positive_verified += 1
        except BuildEvidenceError as exc:
            positive_failures.append({
                "key_sha256": record["key_sha256"],
                "reason": exc.code,
            })
    return {
        "schema": "onnx-splitpoint/build-evidence-index-verification/v1",
        "schema_version": 1,
        "ok": not positive_failures,
        "record_count": index["record_count"],
        "reusable_record_count": index["reusable_record_count"],
        "positive_record_count": positive_count,
        "positive_verified_count": positive_verified,
        "positive_failures": positive_failures,
        "runtime_evidence_included": False,
    }


__all__ = [
    "ABORTED_UNKNOWN",
    "ARTIFACT_PASS",
    "BUILD_INDEX_SCHEMA",
    "BUILD_KEY_SCHEMA",
    "BUILD_STATES",
    "COMPILE_INFEASIBLE",
    "PARSER_UNSUPPORTED",
    "TRANSIENT_INFRASTRUCTURE",
    "BuildEvidenceDecision",
    "BuildEvidenceError",
    "VerifiedHailoArtifact",
    "boundary_endpoint_contract_sha256",
    "build_evidence_index",
    "build_key_sha256",
    "canonical_build_key",
    "canonical_build_key_from_hailo_v3",
    "canonical_build_key_from_hailo_v3_payload",
    "canonical_sha256",
    "classify_build_outcome",
    "create_build_evidence_index",
    "harvest_b5_run",
    "load_build_evidence_index",
    "lookup_build_evidence",
    "make_build_evidence_record",
    "materialize_verified_artifact",
    "validate_build_evidence_index",
    "validate_build_evidence_record",
    "validate_build_key",
    "validate_external_output_path",
    "verify_build_evidence_index",
    "verify_hailo_artifact",
    "write_build_evidence_index",
]
