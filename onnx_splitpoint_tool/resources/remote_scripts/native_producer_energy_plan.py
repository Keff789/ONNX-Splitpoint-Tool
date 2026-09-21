#!/usr/bin/env python3
"""Create a technically admitted u.RECS energy plan for Native rows.

Plan membership is invariant under Quality, Semantics, pairing and claim
annotations.  A successful runtime row is measured exactly when its Energy
command is constructible and a Split row proves one valid Part-2 input; Native
Full baselines do not have a Part-2 requirement.  Post-hoc annotations may only
downgrade claim eligibility.  Actual work units emitted by the runtime are
preferred by the energy collector; ``FPS * duration`` remains an explicit
fallback only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

def _resolve_tool_root() -> Path:
    source = Path(__file__).resolve()
    candidates = [source.parents[1]]
    if len(source.parents) > 3:
        candidates.append(source.parents[3])
    for candidate in candidates:
        if (
            candidate
            / "onnx_splitpoint_tool"
            / "resources"
            / "validation"
            / "hailo10_yolo26_claim_exclusions_v272.json"
        ).is_file():
            return candidate
    return source.parents[1]


ROOT = _resolve_tool_root()
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_energy_quality_admission import (  # noqa: E402
    canonical_json_sha256 as _quality_admission_sha256,
    validate_energy_quality_admission_axes,
    energy_quality_reason_projection,
)
from onnx_splitpoint_tool.native_performance_identity import (  # noqa: E402
    native_performance_identity,
)
from onnx_splitpoint_tool.preprocessing_contract import (  # noqa: E402
    PREPROCESSING_CONTRACT_SCHEMA,
    PREPROCESSING_CONTRACT_SCHEMA_VERSION,
    RUNTIME_NUMERIC_INPUT_SCHEMA,
    RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION,
    preprocessing_contract_sha256,
    runtime_numeric_input_identity_errors,
)

_DEFAULT_DETECTION_EXCLUSIONS = (
    ROOT / "onnx_splitpoint_tool" / "resources" / "validation"
    / "hailo10_yolo26_claim_exclusions_v272.json"
)
_DEFAULT_DETECTION_EXCLUSIONS_SHA256 = (
    "5303970c1d84150202b93abc4cda2d0196a9613678a150a711f944dbcd4c1fda"
)

def _script_path(name: str) -> Path:
    source = ROOT / "scripts" / name
    return source if source.is_file() else SCRIPT_DIR / name

try:
    from onnx_splitpoint_tool.energy.config import load_energy_defaults, resolve_energy_ab_config
except Exception:  # pragma: no cover - standalone remote use
    load_energy_defaults = None
    resolve_energy_ab_config = None

try:
    from onnx_splitpoint_tool.native_command_contract import (
        split_energy_runtime_argv,
        verify_native_energy_command_contract,
        verify_native_split_part2_input_contract,
    )
except Exception:  # pragma: no cover - standalone copy must fail closed
    split_energy_runtime_argv = None
    verify_native_energy_command_contract = None
    verify_native_split_part2_input_contract = None

try:
    from onnx_splitpoint_tool.native_split_quality import (
        bind_quality_to_native_split,
        validate_native_split_quality_binding,
    )
except Exception:  # pragma: no cover - standalone copy must fail closed
    bind_quality_to_native_split = None
    validate_native_split_quality_binding = None

try:
    from onnx_splitpoint_tool.native_split_quality_authority import (
        apply_native_split_quality_authority,
        canonical_native_split_backend,
        is_native_split_backend,
        native_split_quality_required_for_row,
        resolve_native_split_quality_authority,
    )
except Exception:  # pragma: no cover - standalone copy must fail closed
    apply_native_split_quality_authority = None
    canonical_native_split_backend = None
    is_native_split_backend = None
    native_split_quality_required_for_row = None
    resolve_native_split_quality_authority = None

try:
    from onnx_splitpoint_tool.native_detection_diagnostics import (
        verify_prospective_detection_exclusion_set,
    )
except Exception:  # pragma: no cover - standalone copy must fail closed
    verify_prospective_detection_exclusion_set = None

try:
    from onnx_splitpoint_tool.native_detection_postprocess import (
        FrozenPostprocessError,
        build_completed_detection_endpoint_attestation,
        build_normalized_detection_endpoint_attestation,
        verify_completed_detection_comparison_endpoint_contract,
        verify_detection_completion_execution_attestation,
        verify_detection_completion_execution_contract,
        verify_frozen_decoded_nms_normalization_contract,
        verify_frozen_postprocess_contract,
    )
except Exception:  # pragma: no cover - standalone copy must fail closed
    class FrozenPostprocessError(RuntimeError):
        pass

    def verify_frozen_postprocess_contract(
        _value: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "frozen_postprocess_runtime_closure_unavailable"
        )

    def build_completed_detection_endpoint_attestation(
        _contract: Any, _result: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "completed_endpoint_runtime_closure_unavailable"
        )

    def verify_frozen_decoded_nms_normalization_contract(
        _value: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "direct_normalization_runtime_closure_unavailable"
        )

    def verify_completed_detection_comparison_endpoint_contract(
        _value: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "completed_comparison_runtime_closure_unavailable"
        )

    def verify_detection_completion_execution_contract(
        _value: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "completion_execution_runtime_closure_unavailable"
        )

    def verify_detection_completion_execution_attestation(
        _value: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "completion_execution_runtime_closure_unavailable"
        )

    def build_normalized_detection_endpoint_attestation(
        _contract: Any, _result: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "direct_normalization_runtime_closure_unavailable"
        )


FULL_COMMAND_CONTRACT_SCHEMA = "onnx-splitpoint/native-full-command-contract"
FULL_COMMAND_CONTRACT_VERSION = 1
TRT_ENGINE_BUILD_RECEIPT_SCHEMA = "onnx-splitpoint/tensorrt-engine-build-receipt"
TRT_ENGINE_BUILD_RECEIPT_VERSION = 1
_FULL_ENERGY_WORKLOAD_KINDS = frozenset({
    "hailo_full_hotloop",
    "tensorrt_full_hotloop",
    "tensorrt_full_completed_task_hotloop",
    "deepx_full_prepared_feed_hotloop",
})
_FULL_RUNTIME_PYTHON_WORKLOAD_KINDS = frozenset({
    "hailo_full_hotloop",
    "tensorrt_full_completed_task_hotloop",
    "deepx_full_prepared_feed_hotloop",
})
_TRT_FULL_ENERGY_WORKLOAD_KINDS = frozenset({
    "tensorrt_full_hotloop",
    "tensorrt_full_completed_task_hotloop",
})
_FRESH_OUTPUT_TOKEN = "__ONNX_SPLITPOINT_FRESH_OUTPUT_ROOT__"
_PREFLIGHT_NONCE_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
_PREFLIGHT_ATTESTATION_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__"
_REMOTE_CONTRACT_TOKEN = "__ONNX_SPLITPOINT_REMOTE_COMMAND_CONTRACT__"
_REMOTE_LEASE_ENV = (
    "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID",
    "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID",
    "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR",
)
_REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S = 17.0


def _lease_aware_ssh_script_body(
    *,
    ssh_target: str,
    remote_command: str,
    operation_label: str,
    stdin_path: str | Path | None = None,
    timeout_s: float | None = None,
) -> str:
    """Render one invocation-time lease-aware SSH command.

    Native Energy command files are executed later, and once per preflight or
    physical repeat.  Consequently the lease operation must be minted when the
    generated file is *invoked*, not while its plan is built.  A complete
    workflow lease environment routes the SSH argv through the exact-lease
    broker; a standalone invocation with no lease environment retains the
    historical direct-SSH behaviour.  A partial environment fails closed.

    The broker receives argv after ``--`` without reparsing the remote payload,
    and stdin redirection is applied to the outer command in either branch.
    """

    ssh_argv = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        str(ssh_target),
        str(remote_command),
    ]
    direct = " ".join(shlex.quote(value) for value in ssh_argv)
    broker_argv = [
        sys.executable,
        "-m", "onnx_splitpoint_tool.remote.process_lease_cli",
        "exec", "--label", str(operation_label),
    ]
    if timeout_s is not None:
        broker_argv.extend(["--timeout-s", str(max(0.0, float(timeout_s)))])
    broker_argv.extend(["--", *ssh_argv])
    broker = " ".join(
        shlex.quote(value)
        for value in broker_argv
    )
    redirect = (
        f" < {shlex.quote(str(stdin_path))}"
        if stdin_path is not None else ""
    )
    active = " || ".join(f'[[ -n "${{{name}:-}}" ]]' for name in _REMOTE_LEASE_ENV)
    complete = " && ".join(f'[[ -n "${{{name}:-}}" ]]' for name in _REMOTE_LEASE_ENV)
    required = ", ".join(_REMOTE_LEASE_ENV)
    return (
        f"if {active}; then\n"
        f"    if ! ( {complete} ); then\n"
        f"        printf '%s\\n' {shlex.quote('incomplete Native Energy remote lease environment: ' + required)} >&2\n"
        "        exit 64\n"
        "    fi\n"
        f"    exec {broker}{redirect}\n"
        "fi\n"
        f"exec {direct}{redirect}\n"
    )


def _strict_json_text(text: str, *, label: str) -> Any:
    """Decode JSON while rejecting every duplicate object key.

    Duplicate keys are ambiguous even when both values happen to be identical:
    accepting either form would leave the scientific identity dependent on the
    parser's first/last-key policy.  ``object_pairs_hook`` also covers objects
    nested inside summary rows and attestations.
    """
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"duplicate JSON object key in {label}: {key!r}"
                )
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=_object)


def _strict_json_value(path: Path) -> Any:
    return _strict_json_text(path.read_text(encoding="utf-8"), label=str(path))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = _strict_json_value(path)
        return list(data.get("rows", []) or []) if isinstance(data, dict) else []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_matrix_contract(
    summary: Path,
) -> tuple[dict[str, Any], str]:
    """Load the run-local Native matrix denominator without hard-coding it."""
    candidates = (
        summary.parent / "native_expected_matrix.json",
        summary.parent.parent / "native_expected_matrix.json",
        summary.parent.parent / "reports" / "native_expected_matrix.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _strict_json_value(path)
        except Exception:
            return {}, "native_expected_matrix_invalid"
        if not isinstance(payload, Mapping):
            return {}, "native_expected_matrix_not_mapping"
        expected = payload.get("expected_row_count")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected <= 0
        ):
            return {}, "native_expected_matrix_count_invalid"
        return dict(payload), "verified"
    return {}, "native_expected_matrix_missing"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "ok", "pass", "claim_ok"}


def _runtime_successful_for_energy(row: Mapping[str, Any]) -> bool:
    """Separate physical execution success from a post-run Quality veto."""

    if "runtime_success" in row:
        return _truth(row.get("runtime_success"))
    if "runtime_successful" in row:
        return _truth(row.get("runtime_successful"))
    # Compatibility for old summaries whose ``ok`` field represented both
    # runtime and Quality success.
    return _truth(row.get("ok"))


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    from onnx_splitpoint_tool.native_job_identity import planned_native_identity
    identity = planned_native_identity(row)
    return (identity['backend'], identity['model'], identity['case'] or 'full', identity['precision'])


def _discover_validation_summary(summary: Path, explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    candidates = [
        summary.parent / "native_validation" / "native_producer_validation_summary.json",
        summary.parent / "native_producer_validation_summary.json",
        summary.parent.parent / "native_validation" / "native_producer_validation_summary.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _validation_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    from onnx_splitpoint_tool.native_job_identity import planned_native_identity, native_comparison
    identity = planned_native_identity(row)
    # The validation join retains runtime precision even though Full ledger
    # position is comparison-based; no actual precision is rewritten.
    return (*_key(row), identity['setup_id'], native_comparison(row.get('comparison_backend')))


def _validation_map(path: Path | None) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        document = _strict_json_value(path)
        if not isinstance(document, Mapping):
            raise ValueError("native validation document is not an object")
        raw_rows = document.get("rows", [])
        if not isinstance(raw_rows, list):
            raise ValueError("native validation rows is not a list")
        rows = list(raw_rows)
    else:
        rows = _load_rows(path)
    out: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("native validation row is not an object")
        row = dict(raw_row)
        identity = _validation_identity(row)
        if identity in out:
            previous = out[identity]
            count = int(previous.get("_ambiguous_candidate_count") or 1) + 1
            out[identity] = {
                **{key: value for key, value in zip(
                    ("backend", "model", "case", "precision", "setup_id", "comparison_backend"),
                    identity,
                )},
                "_ambiguous_validation_identity": True,
                "_ambiguous_candidate_count": count,
                "status": "native_validation_identity_ambiguous",
                "claim_ok": False,
                "semantic_ok": False,
                "contract_consistent": False,
            }
        else:
            out[identity] = row
    return out


def _validation_map_with_status(
    path: Path | None,
) -> tuple[
    dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    str,
]:
    """Load downstream annotations without making them plan authorities."""

    if path is None:
        return {}, "not_available"
    try:
        return _validation_map(path), "verified"
    except (OSError, UnicodeError):
        return {}, "unavailable_io_error"
    except Exception:
        # JSON/schema/duplicate-identity transport failures are archived as
        # unavailable annotations.  Runtime/command/Part-2 technical
        # membership is intentionally unaffected.
        return {}, "unavailable_invalid_annotation_document"


def _validation_for_row(
    row: dict[str, Any],
    validations: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    *,
    exact_identity_required: bool = False,
) -> dict[str, Any] | None:
    identity = _validation_identity(row)
    if exact_identity_required and any(not value for value in identity):
        return None
    exact = validations.get(identity)
    if exact is not None:
        return exact
    if exact_identity_required:
        return None
    base = _key(row)
    setup = str(row.get("setup_id") or "").strip()
    comparison = str(row.get("comparison_backend") or "").strip().lower()
    matches = [
        value for key, value in validations.items()
        if key[:4] == base
        and (not setup or not key[4] or key[4] == setup)
        and (not comparison or not key[5] or key[5] == comparison)
    ]
    return matches[0] if len(matches) == 1 else None


def _semantic_decision(
    row: dict[str, Any],
    validations: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    *,
    require_claim: bool = False,
    exact_identity_required: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    backend = str(row.get("backend") or "").lower()
    validation = _validation_for_row(
        row, validations, exact_identity_required=exact_identity_required,
    )
    if validation is not None and validation.get("_ambiguous_validation_identity") is True:
        return False, "native_validation_identity_ambiguous", validation
    if backend.startswith("native_full_"):
        if not _truth(row.get("ok")):
            return False, "native_full_runtime_failed", validation
        if validation is None:
            return False, "native_full_validation_missing", None
        if (
            str(validation.get("task") or "").strip().lower() == "classification"
            and validation.get("top1_match") is not True
        ):
            return False, "classification_top1_mismatch", validation
        semantic_valid = bool(
            validation.get("semantic_ok") is True
            and _truth(validation.get("contract_consistent"))
        )
        if not semantic_valid:
            return (
                False,
                str(
                    validation.get("numerical_similarity_reason")
                    or validation.get("task_quality_reason")
                    or validation.get("failure_reason")
                    or validation.get("error")
                    or validation.get("semantic_validation_status")
                    or validation.get("status")
                    or "native_full_semantic_gate_failed"
                ),
                validation,
            )
        if require_claim and not _truth(validation.get("claim_ok")):
            return (
                False,
                str(validation.get("status") or "native_full_claim_gate_failed"),
                validation,
            )
        return True, "native_full_semantic_gate_pass", validation
    if validation is None:
        return False, (
            "native_validation_exact_identity_missing"
            if exact_identity_required else "native_validation_missing"
        ), None
    task = str(validation.get("task") or "").lower()
    if task == "classification" and validation.get("top1_match") is not True:
        return False, "classification_top1_mismatch", validation
    contract = validation.get("contract_consistent")
    semantic_valid = (
        validation.get("semantic_ok") is True
        and (contract is True or str(contract).lower() in {"true", "pass", "ok"})
    )
    if not semantic_valid:
        return False, str(
            validation.get("numerical_similarity_reason")
            or validation.get("task_quality_reason")
            or validation.get("failure_reason")
            or validation.get("error")
            or validation.get("semantic_validation_status")
            or validation.get("status")
            or "native_semantic_gate_failed"
        ), validation
    if require_claim and not _truth(validation.get("claim_ok")):
        return False, str(validation.get("status") or "native_claim_gate_failed"), validation
    return True, "native_semantic_gate_pass", validation


def _smoke_diagnostic_technical_decision(
    row: dict[str, Any],
    validations: dict[
        tuple[str, str, str, str, str, str], dict[str, Any]
    ],
) -> tuple[bool, str, dict[str, Any] | None]:
    """Admit a bound negative metric observation without weakening claims."""
    validation = _validation_for_row(
        row, validations, exact_identity_required=True,
    )
    if validation is None:
        return False, (
            "native_full_validation_missing"
            if str(row.get("backend") or "").strip().lower().startswith(
                "native_full_"
            )
            else "native_validation_exact_identity_missing"
        ), None
    if validation.get("_ambiguous_validation_identity") is True:
        return False, "native_validation_identity_ambiguous", validation
    if not _truth(row.get("ok")):
        return False, "native_full_runtime_failed", validation
    if validation.get("ok") is not True or validation.get("error"):
        return False, str(
            validation.get("error")
            or validation.get("failure_reason")
            or validation.get("status")
            or "smoke_diagnostic_technical_validation_failed"
        ), validation
    return True, "smoke_diagnostic_technical_validation_pass", validation


def _backend_defaults(backend: str, row: dict[str, Any] | None = None) -> dict[str, str]:
    row = row or {}
    explicit_setup = str(row.get("setup_id") or "").strip()
    if explicit_setup:
        if "hailo8" in explicit_setup:
            return {"setup": explicit_setup, "ssh_arg": "hailo8_ssh"}
        if "hailo10" in explicit_setup:
            return {"setup": explicit_setup, "ssh_arg": "hailo10_ssh"}
        if "deepx" in explicit_setup:
            return {"setup": explicit_setup, "ssh_arg": "deepx_ssh"}
        # Preserve custom setup IDs. Infer only the SSH channel from the row's
        # backend/comparison context so pairing remains strictly setup-local.
        if backend in ("hailo8_to_trt", "native_full_hailo8"):
            return {"setup": explicit_setup, "ssh_arg": "hailo8_ssh"}
        if backend in ("hailo10h_to_trt", "native_full_hailo10h", "native_full_hailo10"):
            return {"setup": explicit_setup, "ssh_arg": "hailo10_ssh"}
        if backend in ("deepx_to_trt", "native_full_deepx"):
            return {"setup": explicit_setup, "ssh_arg": "deepx_ssh"}
        if backend == "native_full_tensorrt":
            context = str(row.get("comparison_backend") or "").lower()
            if "hailo8" in context:
                return {"setup": explicit_setup, "ssh_arg": "hailo8_ssh"}
            if "hailo10" in context:
                return {"setup": explicit_setup, "ssh_arg": "hailo10_ssh"}
            if "deepx" in context:
                return {"setup": explicit_setup, "ssh_arg": "deepx_ssh"}
    if backend in ("hailo8_to_trt", "native_full_hailo8"):
        return {"setup": "orin_nx_hailo8_01", "ssh_arg": "hailo8_ssh"}
    if backend in ("hailo10h_to_trt", "native_full_hailo10h", "native_full_hailo10"):
        return {"setup": "orin_nx_hailo10_01", "ssh_arg": "hailo10_ssh"}
    if backend in ("deepx_to_trt", "native_full_deepx"):
        return {"setup": "orin_nx_deepx_m1_01", "ssh_arg": "deepx_ssh"}
    if backend == "native_full_tensorrt":
        context = str(row.get("comparison_backend") or "").lower()
        if "hailo8" in context:
            return {"setup": explicit_setup or "orin_nx_hailo8_01", "ssh_arg": "hailo8_ssh"}
        if "hailo10" in context:
            return {"setup": explicit_setup or "orin_nx_hailo10_01", "ssh_arg": "hailo10_ssh"}
        if "deepx" in context:
            return {"setup": explicit_setup or "orin_nx_deepx_m1_01", "ssh_arg": "deepx_ssh"}
        return {"setup": explicit_setup, "ssh_arg": ""}
    return {"setup": explicit_setup or backend, "ssh_arg": ""}


def _comparison_context(value: Any) -> str:
    """Return the setup-local accelerator context used for Full matching.

    A Full runtime's own input/compute precision is deliberately *not* part of
    this identity.  Split precision describes the internal boundary contract,
    whereas a Full baseline has no such boundary.  Conflating the two made a
    valid Hailo-10H/ResNet50 triplet disappear in 2.62.
    """
    text = str(value or "").strip().lower().replace("-", "_")
    if (
        canonical_native_split_backend is not None
        and is_native_split_backend is not None
        and is_native_split_backend(text)
    ):
        managed = canonical_native_split_backend(text)
        return {
            "hailo8_to_trt": "hailo8",
            "hailo10h_to_trt": "hailo10h",
            "deepx_to_trt": "deepx",
        }[managed]
    aliases = {
        "h8": "hailo8",
        "hailo_8": "hailo8",
        "hailo8_to_trt": "hailo8",
        "hailo8_to_tensorrt": "hailo8",
        "h10": "hailo10h",
        "hailo10": "hailo10h",
        "hailo_10": "hailo10h",
        "hailo_10h": "hailo10h",
        "hailo10h_to_trt": "hailo10h",
        "hailo10h_to_tensorrt": "hailo10h",
        "deepx_to_trt": "deepx",
        "deepx_to_tensorrt": "deepx",
    }
    return aliases.get(text, text)


def _canonical_native_split_source_run_id(value: Any) -> str:
    """Canonicalize only allowlisted Native Split source-run aliases."""
    if (
        canonical_native_split_backend is None
        or is_native_split_backend is None
        or not is_native_split_backend(value)
    ):
        return ""
    return canonical_native_split_backend(value)


def _full_comparison_precision(
    row: dict[str, Any], validation: dict[str, Any] | None = None,
) -> str:
    """Return the legacy Split stratum attached to a Full observation."""
    for source in (row, validation or {}):
        for key in ("comparison_precision", "legacy_comparison_precision", "precision"):
            value = str(source.get(key) or "").strip().lower()
            if value:
                return value
    return ""


def _full_runtime_precision(
    row: dict[str, Any], validation: dict[str, Any] | None = None,
) -> str:
    """Return only precision explicitly scoped to Full execution.

    Never fall back to ``precision``: Native Full historically stored the Split
    comparison stratum there.
    """
    # Prefer the exact, validation-bound identity (for example a DXNN/HEF
    # artifact digest) over a generic fp16/int8 display label from the summary.
    for key in (
        "runtime_precision_identity", "full_runtime_precision",
        "execution_precision", "runtime_precision",
    ):
        for source in (validation or {}, row):
            value = str(source.get(key) or "").strip().lower()
            if value:
                return value
    return ""


def _dedupe_key(row: dict[str, Any], setup: str) -> tuple[str, str, str, str, str, str]:
    backend, model, case, precision = _key(row)
    comparison = (
        str(row.get("comparison_backend") or "").strip().lower()
        if backend.startswith("native_full_")
        else ""
    )
    return backend, model, case, setup, comparison, precision


def _row_score(row: dict[str, Any], validation: dict[str, Any] | None) -> tuple[int, int, int, float]:
    """Technical-only diagnostic score; never use Quality to select a row."""

    command_exists = int(bool(
        row.get("native_command_contract")
        or row.get("full_command_contract")
    ))
    report_exists = int(bool(str(row.get("report") or "").strip()))
    frames = int(float(row.get("frames") or 0)) if str(row.get("frames") or "").strip() else 0
    try:
        fps = float(row.get("fps_makespan") or row.get("FPS") or row.get("pipeline_fps_selected") or 0)
    except Exception:
        fps = 0.0
    return command_exists, report_exists, frames, fps


def _split_part2_input_admission(
    row: Mapping[str, Any],
    command_contract: Mapping[str, Any] | None,
) -> tuple[bool, str, int | None]:
    """Prove that a Split command consumes exactly one Part-2 input.

    The proof authority is the command contract's sealed, locally re-hashed
    TensorRT metadata closure.  Unsealed summary counts and direct metadata
    copies are annotations only and can neither authorize nor veto a row.
    """
    if not isinstance(command_contract, Mapping):
        return False, "verified_part2_input_contract_missing", None
    if verify_native_split_part2_input_contract is None:
        return False, "part2_technical_verifier_unavailable", None
    metadata, status = verify_native_split_part2_input_contract(
        command_contract
    )
    if metadata is None:
        return False, status, None
    return True, status, 1


def _wrap_runtime(remote_tool_dir: str, child_command: str, *, follow_reports: bool = True) -> str:
    """Legacy/diagnostic helper; scientific 2.62.1 rows execute directly."""
    helper = f"{remote_tool_dir.rstrip('/')}/scripts/run_and_report_work_units.py"
    report_flag = "" if follow_reports else " --no-follow-reports"
    return f"python {shlex.quote(helper)}{report_flag} -- {child_command}"


def _duration_controlled_minimum_work_units(explicit_frames: int) -> tuple[int, str]:
    """Return a minimum work budget without consulting historical FPS.

    Compatibility capability marker: ``runtime_marker_preferred``. Current
    final-energy execution goes further and requires the exact runtime marker.
    """
    value = int(explicit_frames or 0)
    if value > 0:
        return value, "explicit_legacy_minimum_override"
    return 1, "duration_driven_minimum_one"


def _evidence_value(row: dict[str, Any], validation: dict[str, Any] | None, *keys: str) -> Any:
    for source in (validation or {}, row):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return ""


def _current_quality_first_validation_join(
    row: Mapping[str, Any], validation: Mapping[str, Any] | None,
    *, require_performance_claim: bool = True,
) -> tuple[bool, str]:
    """Require the exact Final↔Validation identity and Central selection."""
    if not isinstance(validation, Mapping):
        return False, "native_split_quality_validation_missing"
    row_identity = _validation_identity(dict(row))
    validation_identity = _validation_identity(dict(validation))
    if any(not value for value in row_identity):
        return False, "native_split_quality_final_identity_incomplete"
    if row_identity != validation_identity:
        return False, "native_split_quality_validation_identity_mismatch"
    if validation.get("_ambiguous_validation_identity") is True:
        return False, "native_split_quality_validation_identity_ambiguous"
    if require_performance_claim:
        if row.get("quality_evidence_verified") is not True:
            return False, "native_split_quality_final_quality_evidence_not_verified"
        if row.get("performance_claim_eligible") is not True:
            return False, "native_split_quality_final_performance_claim_not_eligible"
    else:
        if validation.get("central_quality_evidence_verified") is not True:
            return False, "native_split_quality_diagnostic_central_evidence_not_verified"
        if validation.get("ok") is not True or validation.get("error"):
            return False, "native_split_quality_diagnostic_technical_validation_failed"

    request_sha = _strict_sha256_token(
        row.get("native_split_quality_source_request_sha256")
    )
    central_result_sha = _strict_sha256_token(
        row.get("native_split_quality_central_result_sha256")
    )
    selection_sha = _strict_sha256_token(
        row.get("native_split_quality_selection_sha256")
    )
    if not request_sha or not central_result_sha or not selection_sha:
        return False, "native_split_quality_final_central_selection_missing"
    for field, expected in (
        ("native_split_quality_source_request_sha256", request_sha),
        ("native_split_quality_central_result_sha256", central_result_sha),
        ("native_split_quality_selection_sha256", selection_sha),
    ):
        if _strict_sha256_token(validation.get(field)) != expected:
            return False, f"{field}_validation_drift"
    if (
        _strict_sha256_token(row.get("source_request_sha256")) != request_sha
        or _strict_sha256_token(validation.get("source_request_sha256"))
        != request_sha
    ):
        return False, "native_split_quality_generic_source_request_sha256_drift"
    return True, "exact_final_validation_and_central_selection_join_verified"


def _native_quality_bridge_verified(
    validation: dict[str, Any] | None, task: str,
    row: Mapping[str, Any] | None = None,
    *,
    exact_identity_required: bool = False,
    positive_result_required: bool = False,
) -> bool:
    """Verify the exact quality-evidence binding independently of its result.

    v2.72 deliberately permits a bound negative accuracy observation in an
    explicit all-split measurement plan.  Positive accuracy remains a
    separate requirement for claim comparability.
    """
    if not isinstance(validation, dict):
        return False
    if exact_identity_required:
        if not isinstance(row, Mapping):
            return False
        exact_join, _exact_join_status = _current_quality_first_validation_join(
            row,
            validation,
            require_performance_claim=positive_result_required,
        )
        if not exact_join:
            return False
    precision = str(
        validation.get("runtime_precision_identity")
        or validation.get("execution_precision")
        or validation.get("full_runtime_precision")
        or ""
    ).strip()
    endpoint_hash = _normalize_sha256(validation.get("endpoint_contract_hash"))
    required_hashes = [
        _normalize_sha256(
            validation.get("quality_contract_sha256")
            or validation.get("contract_hash")
        ),
        _normalize_sha256(
            validation.get("preprocessing_contract_sha256")
            or validation.get("preprocessing_hash")
        ),
    ]
    if str(task or "").strip().lower() == "detection":
        required_hashes.extend([
            _normalize_sha256(
                validation.get("decoder_contract_sha256")
                or validation.get("decoder_hash")
            ),
            _normalize_sha256(
                validation.get("nms_contract_sha256")
                or validation.get("nms_hash")
            ),
        ])
    provenance, provenance_conflicts = _validated_quality_provenance(validation)
    required_provenance = [
        provenance["source_request_sha256"],
        provenance["model_sha256"],
        provenance["validation_dataset_sha256"],
        provenance["validation_dataset_image_ids_sha256"],
        provenance["validation_dataset_ground_truth_sha256"],
        provenance["accuracy_gate_policy_sha256"],
        provenance["task_quality_policy_sha256"],
        provenance["runtime_quality_gate_policy_sha256"],
    ]
    binding_axis = (
        validation.get("precision_quality_binding_verified")
        if "precision_quality_binding_verified" in validation
        else validation.get("precision_quality_verified")
    )
    task_observation_axis = validation.get(
        "task_quality_observation_valid"
    )
    accuracy_axis = validation.get("accuracy_gate_pass")
    binding_verified = bool(
        validation.get("central_quality_evidence_verified") is True
        and binding_axis is True
        and isinstance(accuracy_axis, bool)
        and task_observation_axis is True
        and validation.get("endpoint_contract_complete") is True
        and len(endpoint_hash) == 64
        and precision
        and all(len(value) == 64 for value in required_hashes)
        and not provenance_conflicts
        and all(len(value) == 64 for value in required_provenance)
        and len({
            provenance["accuracy_gate_policy_sha256"],
            provenance["task_quality_policy_sha256"],
            provenance["runtime_quality_gate_policy_sha256"],
        }) == 1
    )
    if not binding_verified:
        return False
    if not positive_result_required:
        return True
    return bool(
        accuracy_axis is True
        and validation.get("semantic_ok") is True
        and _truth(validation.get("claim_ok"))
        and _truth(validation.get("contract_consistent"))
    )


def _native_output_endpoint_id(
    validation: dict[str, Any] | None, task: str,
) -> str:
    if not isinstance(validation, dict):
        return ""
    explicit = str(validation.get("output_endpoint_id") or "").strip()
    if explicit:
        return explicit
    endpoint_hash = _normalize_sha256(validation.get("endpoint_contract_hash"))
    stage = str(
        validation.get("stage")
        or validation.get("contract_family")
        or validation.get("output_format")
        or ""
    ).strip().lower()
    task = str(task or validation.get("task") or "").strip().lower()
    if (
        validation.get("endpoint_contract_complete") is True
        and task and stage and len(endpoint_hash) == 64
    ):
        return f"{task}:{stage}:{endpoint_hash}"
    return ""


def _endpoint_mapping(
    row: Mapping[str, Any], validation: Mapping[str, Any] | None,
    field: str,
) -> dict[str, Any]:
    for source in (validation or {}, row):
        value = source.get(field)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _verified_fast_energy_completion(
    row: Mapping[str, Any], validation: Mapping[str, Any] | None,
    command: Mapping[str, Any], *, outputs: Mapping[str, Any] | None = None,
    verify_command: bool = True, energy_observation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the original fast-oracle schema without rewriting its relation.

    This is offline evidence work. Counts describe measured task work only;
    the one untimed oracle sentinel never contributes a work unit.
    """
    from onnx_splitpoint_tool.native_three_stage import verify_fast_completion_attestation
    from onnx_splitpoint_tool.native_command_contract import verify_native_command_contract
    sources = [row, validation or {}]
    mode = "native_three_stage_fast_oracle_outside_timing"
    for field in ("model", "case", "setup_id", "comparison_backend", "precision"):
        values = {str(item[field]) for item in [*sources, command]
                  if item.get(field) not in (None, "")}
        if len(values) > 1:
            raise ValueError("fast_energy_job_identity_conflict:" + field)
    if verify_command:
        verified, reason = verify_native_command_contract(command)
        if verified is None:
            raise ValueError("fast_energy_command_invalid:" + reason)
    options = command.get("runtime_options") or {}
    execution = options.get("completion_execution_contract") or row.get("completion_execution_contract")
    verified_execution = verify_detection_completion_execution_contract(execution)
    completion = _endpoint_mapping(row, validation, "completed_task_endpoint_attestation")
    if outputs is None and row.get("native_split_semantic_binding_valid") is True:
        manifest_raw = row.get("native_split_semantic_output_manifest") or row.get("native_output_manifest")
        if manifest_raw:
            import numpy as np
            manifest_path = Path(str(manifest_raw))
            payload_ok, payload_reason = _manifest_payload_bytes_verified(manifest_path)
            if not payload_ok:
                raise ValueError("fast_energy_sentinel_payload_invalid:" + payload_reason)
            manifest = _strict_json_object(manifest_path) or {}
            tensor_records = manifest.get("outputs") or []
            if not tensor_records:
                raise ValueError("fast_energy_dumped_sentinel_outputs_missing")
            payloads = {Path(str(item["path"])).name: item for item in manifest["payload_artifacts"]}
            outputs = {}
            for item in tensor_records:
                name = str(item.get("file") or "")
                if Path(name).name != name or name not in payloads:
                    raise ValueError("fast_energy_dumped_sentinel_path_invalid")
                path = manifest_path.parent / name
                if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
                    raise ValueError("fast_energy_dumped_sentinel_path_invalid")
                shape = item.get("shape")
                dtype = np.dtype(item.get("dtype"))
                if not isinstance(shape, list) or any(type(value) is not int or value <= 0 for value in shape):
                    raise ValueError("fast_energy_dumped_sentinel_shape_invalid")
                if int(np.prod(shape)) * dtype.itemsize != path.stat().st_size:
                    raise ValueError("fast_energy_dumped_sentinel_size_invalid")
                outputs[str(item.get("name"))] = np.frombuffer(path.read_bytes(), dtype=dtype).reshape(shape)
    verified = verify_fast_completion_attestation(completion, execution_contract=verified_execution, outputs=outputs)
    source = verified_execution["source_endpoint"]
    comparison = verify_completed_detection_comparison_endpoint_contract(verified_execution["comparison_endpoint_contract"])
    expected = {
        "completed_task_completion_mode": mode,
        "completion_execution_contract": verified_execution,
        "completion_execution_contract_sha256": verified_execution["contract_sha256"],
        "completion_execution_attestation": verified,
        "completed_task_endpoint_attestation": verified,
        "completion_observation_relation": "postflight_oracle_sentinel",
        "completed_task_endpoint_contract": verified_execution["completed_endpoint_contract"],
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": comparison["endpoint_contract_hash"],
        "completed_task_comparison_output_endpoint_id": comparison["output_endpoint_id"],
        "endpoint_contract_hash": source["endpoint_contract_hash"],
        "stage": source["stage"],
        "contract_family": source["contract_family"],
        "physical_output_endpoint_id": source["output_endpoint_id"],
        "output_endpoint_id": source["output_endpoint_id"],
        "postprocess_included": True,
        "postprocess_completion_verified": True,
    }
    for item in sources:
        for field, value in expected.items():
            if item.get(field) not in (None, "", {}) and item[field] != value:
                raise ValueError("fast_energy_projection_conflict:" + field)
    if str(row.get("completed_task_completion_mode") or "") != mode:
        raise ValueError("fast_energy_completion_mode_missing")
    roles = {str(item.get("measurement_endpoint")) for item in [*sources, command, options]
             if item.get("measurement_endpoint") not in (None, "")}
    if roles != {"completed_task"}:
        raise ValueError("fast_energy_measurement_endpoint_not_completed_task")
    count = verified["completed_work_units"]
    measured_counts = [item.get(field) for item in sources
        for field in ("completed_work_units", "completed_frames", "postprocess_completed_frames")
        if item.get(field) is not None]
    if not measured_counts or any(type(value) is not int or value <= 0 or value != count for value in measured_counts):
        raise ValueError("fast_energy_measured_work_unit_count_mismatch")
    if not energy_observation and options.get("duration_s") in (None, 0, 0.0):
        requested = options.get("frames")
        if requested is not None and (type(requested) is not int or requested != count):
            raise ValueError("fast_energy_warmup_or_sentinel_counted")
    for item in sources:
        if item.get("measurement_boundary") not in (None, "", "workers_ready_to_last_completed_task_frame"):
            raise ValueError("fast_energy_measurement_boundary_conflict")
    if str(row.get("model") or command.get("model") or "") != str(verified_execution.get("model_id") or ""):
        raise ValueError("fast_energy_completion_model_mismatch")
    # Fully supplied projections are assertions: missing aliases may be read,
    # but two different complete assertions may never be silently coalesced.
    for field in ("input_image_sha256", "hef_sha256", "engine_sha256", "native_command_contract_sha256"):
        declared = command.get("contract_sha256" if field == "native_command_contract_sha256" else field)
        if not declared and field in ("hef_sha256", "engine_sha256"):
            declared = ((command.get("artifacts") or {}).get(field.removesuffix("_sha256")) or {}).get("sha256")
        values = [item[field] for item in sources if item.get(field)]
        if declared and any(value != declared for value in values):
            raise ValueError("fast_energy_artifact_or_input_identity_conflict:" + field)
    return verified, comparison


def _energy_endpoint_identity(
    row: Mapping[str, Any], validation: Mapping[str, Any] | None,
    command_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize physical plus strictly revalidated comparison identity.

    Classification retains its historical physical endpoint.  Detection uses
    the canonical completed-task endpoint only after the measured completion
    attestation is rebuilt from its frozen source contract.  A Split without
    such evidence remains physical and is explicitly non-pairable.
    """
    task = str(_evidence_value(
        dict(row), dict(validation) if isinstance(validation, Mapping) else None,
        "task", "benchmark_task",
    ) or "").strip().lower()
    physical_hash = _strict_sha256_token(_evidence_value(
        dict(row), dict(validation) if isinstance(validation, Mapping) else None,
        "endpoint_contract_hash",
    ))
    physical_stage = str(_evidence_value(
        dict(row), dict(validation) if isinstance(validation, Mapping) else None,
        "stage", "contract_family", "output_format",
    ) or "").strip().lower()
    physical_id = _native_output_endpoint_id(
        dict(validation) if isinstance(validation, Mapping) else None,
        task,
    )
    if not physical_id and physical_hash and task and physical_stage:
        physical_id = f"{task}:{physical_stage}:{physical_hash}"
    physical_complete = bool(
        _evidence_value(
            dict(row),
            dict(validation) if isinstance(validation, Mapping) else None,
            "endpoint_contract_complete",
        )
        is True
        and physical_hash and physical_id and physical_stage
    )
    identity: dict[str, Any] = {
        "physical_output_endpoint_id": physical_id,
        "physical_endpoint_contract_hash": physical_hash,
        "physical_endpoint_stage": physical_stage,
        "physical_endpoint_contract_complete": physical_complete,
        "output_endpoint_id": physical_id,
        "endpoint_contract_hash": physical_hash,
        "endpoint_stage": physical_stage,
        "comparison_output_endpoint_id": "",
        "comparison_endpoint_contract_hash": "",
        "comparison_endpoint_stage": "",
        "completion_pairing_eligible": False,
        "completion_pairing_status": "task_or_physical_endpoint_unavailable",
    }
    if task == "classification":
        identity.update({
            "comparison_output_endpoint_id": physical_id,
            "comparison_endpoint_contract_hash": physical_hash,
            "comparison_endpoint_stage": physical_stage,
            "completion_pairing_eligible": physical_complete,
            "completion_pairing_status": (
                "classification_physical_endpoint_preserved"
                if physical_complete else
                "classification_physical_endpoint_unavailable"
            ),
        })
        return identity
    if task != "detection":
        return identity

    completion = _endpoint_mapping(
        row, validation, "completed_task_endpoint_attestation",
    )
    if not completion:
        identity["completion_pairing_status"] = (
            "detection_completed_endpoint_attestation_missing"
        )
        return identity
    completion_mode = str(_evidence_value(
        dict(row), dict(validation) if isinstance(validation, Mapping) else None,
        "completed_task_completion_mode",
    ) or completion.get("completed_task_completion_mode") or "").strip()
    nested_mode = str(
        completion.get("completed_task_completion_mode") or ""
    ).strip()
    comparison_contract = _endpoint_mapping(
        row, validation, "completed_task_comparison_endpoint_contract",
    ) or dict(
        completion.get("completed_task_comparison_endpoint_contract") or {}
    )
    declared_hash = _strict_sha256_token(_evidence_value(
        dict(row), dict(validation) if isinstance(validation, Mapping) else None,
        "completed_task_comparison_endpoint_contract_hash",
    ) or completion.get(
        "completed_task_comparison_endpoint_contract_hash"
    ))
    declared_id = str(_evidence_value(
        dict(row), dict(validation) if isinstance(validation, Mapping) else None,
        "completed_task_comparison_output_endpoint_id",
    ) or completion.get(
        "completed_task_comparison_output_endpoint_id"
    ) or "").strip()
    workload = (
        command_contract.get("energy_workload")
        if isinstance(command_contract.get("energy_workload"), Mapping)
        else {}
    )
    if completion_mode == "native_three_stage_fast_oracle_outside_timing":
        try:
            _fast, comparison = _verified_fast_energy_completion(row, validation, command_contract)
            if not physical_complete:
                raise ValueError("fast_energy_physical_endpoint_incomplete")
        except Exception as exc:
            identity["completion_pairing_status"] = str(exc)
            return identity
        identity.update({
            "output_endpoint_id": comparison["output_endpoint_id"],
            "endpoint_contract_hash": comparison["endpoint_contract_hash"],
            "endpoint_stage": "decoded_nms",
            "comparison_output_endpoint_id": comparison["output_endpoint_id"],
            "comparison_endpoint_contract_hash": comparison["endpoint_contract_hash"],
            "comparison_endpoint_stage": "decoded_nms",
            "completion_pairing_eligible": True,
            "completion_pairing_status": "strict_fast_postflight_completion_verified",
            "completion_observation_relation": "postflight_oracle_sentinel",
            "energy_completion_requires_fresh_observation": True,
        })
        return identity
    try:
        if completion_mode == "detection_completion_execution_v1":
            runtime_options = (
                command_contract.get("runtime_options")
                if isinstance(
                    command_contract.get("runtime_options"), Mapping,
                )
                else {}
            )
            raw_execution = (
                runtime_options.get("completion_execution_contract")
                if isinstance(runtime_options, Mapping) else None
            )
            if not isinstance(raw_execution, Mapping):
                raw_execution = _endpoint_mapping(
                    row, validation, "completion_execution_contract",
                )
            verified_execution = (
                verify_detection_completion_execution_contract(
                    raw_execution
                )
            )
            expected_completion = (
                verify_detection_completion_execution_attestation(
                    completion,
                    execution_contract=verified_execution,
                    expected_observation_relation=(
                        "same_hotloop_sentinel"
                    ),
                )
            )
            verified_comparison = (
                verify_completed_detection_comparison_endpoint_contract(
                    comparison_contract
                )
            )
            source_endpoint = dict(
                verified_execution.get("source_endpoint") or {}
            )
            completed_endpoint = dict(
                verified_execution.get(
                    "completed_endpoint_contract"
                ) or {}
            )
            if (
                source_endpoint.get("endpoint_contract_hash")
                != physical_hash
                or source_endpoint.get("output_endpoint_id")
                != physical_id
                or expected_completion.get(
                    "completed_endpoint_contract"
                ) != completed_endpoint
                or int(
                    expected_completion.get("completed_work_units")
                    or 0
                ) <= 0
                or int(
                    expected_completion.get(
                        "postprocess_completed_frames"
                    ) or 0
                )
                != int(
                    expected_completion.get("completed_work_units")
                    or 0
                )
                or expected_completion.get(
                    "exact_result_claim_bound"
                ) is not True
            ):
                raise FrozenPostprocessError(
                    "completion_execution_endpoint_binding_mismatch"
                )
        elif completion_mode == "frozen_host_tail":
            frozen = (
                workload.get("frozen_postprocess_contract")
                if isinstance(
                    workload.get("frozen_postprocess_contract"), Mapping,
                )
                else _endpoint_mapping(
                    row, validation, "frozen_host_postprocess_contract",
                )
            )
            verified_source = verify_frozen_postprocess_contract(frozen)
            expected_completion = (
                build_completed_detection_endpoint_attestation(
                    verified_source,
                    completion.get("frozen_postprocess_result") or {},
                    completed_frames=completion.get("completed_frames"),
                    postprocess_completed_frames=completion.get(
                        "postprocess_completed_frames"
                    ),
                    source_endpoint_contract_hash=physical_hash,
                )
            )
            verified_comparison = (
                verify_completed_detection_comparison_endpoint_contract(
                    comparison_contract,
                    frozen_contract=verified_source,
                )
            )
            workload_source_sha = _strict_sha256_token(
                workload.get("frozen_postprocess_contract_sha256")
            )
            if (
                workload_source_sha
                and workload_source_sha
                != _strict_sha256_token(
                    verified_source.get("contract_sha256")
                )
            ):
                raise FrozenPostprocessError(
                    "frozen_completion_workload_contract_sha256_mismatch"
                )
        elif (
            completion_mode
            == "integrated_accelerator_plus_frozen_normalization"
        ):
            frozen = workload.get(
                "frozen_decoded_nms_normalization_contract"
            )
            verified_source = (
                verify_frozen_decoded_nms_normalization_contract(frozen)
            )
            expected_completion = (
                build_normalized_detection_endpoint_attestation(
                    verified_source,
                    completion.get(
                        "frozen_decoded_nms_normalization_result"
                    ) or {},
                    completed_frames=completion.get("completed_frames"),
                    postprocess_completed_frames=completion.get(
                        "postprocess_completed_frames"
                    ),
                )
            )
            verified_comparison = (
                verify_completed_detection_comparison_endpoint_contract(
                    comparison_contract,
                    direct_normalization_contract=verified_source,
                )
            )
            source_attestation = _endpoint_mapping(
                row, validation, "output_endpoint_attestation",
            )
            declared_source = source_attestation.get("declared_contract")
            if (
                _strict_sha256_token(
                    workload.get(
                        "frozen_decoded_nms_normalization_contract_sha256"
                    )
                )
                != _strict_sha256_token(
                    verified_source.get("contract_sha256")
                )
                or _strict_sha256_token(
                    verified_source.get("source_endpoint_contract_hash")
                ) != physical_hash
                or str(
                    verified_source.get("source_output_endpoint_id") or ""
                ).strip() != physical_id
                or not isinstance(source_attestation, Mapping)
                or source_attestation.get("schema")
                != "onnx-splitpoint/runtime-output-endpoint-attestation"
                or source_attestation.get("schema_version") != 3
                or isinstance(
                    source_attestation.get("schema_version"), bool,
                )
                or source_attestation.get("attested") is not True
                or str(
                    source_attestation.get("status") or ""
                ).strip().lower() != "passed"
                or str(
                    source_attestation.get("endpoint") or ""
                ).strip().lower() != "decoded_nms"
                or str(
                    source_attestation.get("stage") or ""
                ).strip().lower() != "decoded_nms"
                or source_attestation.get(
                    "values_decoded_xyxy_score_class"
                ) is not True
                or source_attestation.get("declaration_attested") is not True
                or _strict_sha256_token(
                    source_attestation.get("endpoint_contract_hash")
                ) != physical_hash
                or dict(source_attestation.get("tensor_signature") or {})
                != dict(
                    verified_source.get(
                        "source_output_tensor_signature"
                    ) or {}
                )
                or not isinstance(declared_source, Mapping)
                or str(declared_source.get("model_id") or "").strip()
                != str(verified_source.get("model_id") or "").strip()
                or declared_source.get("source_coordinate_space")
                != "model_input_letterbox_xyxy_pixels"
                or _canonical_json_sha256(source_attestation)
                != _strict_sha256_token(
                    verified_source.get(
                        "source_output_endpoint_attestation_sha256"
                    )
                )
            ):
                raise FrozenPostprocessError(
                    "direct_completion_source_endpoint_mismatch"
                )
        else:
            identity["completion_pairing_status"] = (
                "detection_completed_endpoint_mode_unsupported"
            )
            return identity
    except (FrozenPostprocessError, TypeError, ValueError, KeyError):
        identity["completion_pairing_status"] = (
            "detection_completed_endpoint_strict_revalidation_failed"
        )
        return identity

    workload_completion = workload.get(
        "completed_task_endpoint_attestation"
    )
    if (
        completion.get("attested") is not True
        or str(completion.get("status") or "").strip().lower() != "passed"
        or str(completion.get("stage") or "").strip().lower()
        != "decoded_nms"
        or nested_mode != completion_mode
        or dict(completion) != dict(expected_completion)
        or (
            isinstance(workload_completion, Mapping)
            and dict(workload_completion) != dict(completion)
        )
        or dict(
            completion.get(
                "completed_task_comparison_endpoint_contract"
            ) or {}
        ) != dict(verified_comparison)
        or dict(comparison_contract) != dict(verified_comparison)
        or declared_hash
        != _strict_sha256_token(
            verified_comparison.get("endpoint_contract_hash")
        )
        or declared_id
        != str(verified_comparison.get("output_endpoint_id") or "")
    ):
        identity["completion_pairing_status"] = (
            "detection_completed_endpoint_attestation_mismatch"
        )
        return identity
    comparison_hash = _strict_sha256_token(
        verified_comparison.get("endpoint_contract_hash")
    )
    comparison_id = str(
        verified_comparison.get("output_endpoint_id") or ""
    ).strip()
    identity.update({
        "output_endpoint_id": comparison_id,
        "endpoint_contract_hash": comparison_hash,
        "endpoint_stage": "decoded_nms",
        "comparison_output_endpoint_id": comparison_id,
        "comparison_endpoint_contract_hash": comparison_hash,
        "comparison_endpoint_stage": "decoded_nms",
        "completion_pairing_eligible": True,
        "completion_pairing_status":
            "strict_completed_detection_endpoint_verified",
    })
    return identity


def _unavailable_energy_endpoint_identity(reason: str) -> dict[str, Any]:
    """Fail closed for pairing/claims while preserving measurement rows."""

    return {
        "physical_output_endpoint_id": "",
        "physical_endpoint_contract_hash": "",
        "physical_endpoint_stage": "",
        "physical_endpoint_contract_complete": False,
        "output_endpoint_id": "",
        "endpoint_contract_hash": "",
        "endpoint_stage": "",
        "comparison_output_endpoint_id": "",
        "comparison_endpoint_contract_hash": "",
        "comparison_endpoint_stage": "",
        "completion_pairing_eligible": False,
        "completion_pairing_status": str(
            reason or "downstream_endpoint_annotation_unavailable"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path) -> dict[str, Any] | None:
    try:
        parsed = _strict_json_value(path)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _manifest_payload_bytes_verified(path: Path) -> tuple[bool, str]:
    """Use the collector's identical bounded payload resolver on every replay."""
    try:
        from scripts.native_producer_final_report import _verify_manifest_payload_files
    except ImportError:
        from native_producer_final_report import _verify_manifest_payload_files
    ok, status = _verify_manifest_payload_files(path)
    return ok, "manifest_and_payload_bytes_rehashed" if ok else status


def _first_existing_path(values: list[Any]) -> Path | None:
    for value in values:
        if value in (None, ""):
            continue
        path = Path(str(value)).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _split_quality_energy_evidence(
    row: Mapping[str, Any], command: Mapping[str, Any],
    authority: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Re-prove a Quality-first split before producing any energy command.

    The final report's booleans are only a prior observation.  Energy opens the
    collected semantic manifests and every referenced tensor again, and also
    repeats the portable binding/command/consumer-attestation join.
    """
    marker_fields = (
        "native_split_quality_binding", "native_split_quality_binding_sha256",
        "native_split_quality_eval_run_id", "native_split_quality_source_run_id",
        "native_split_quality_source_request_sha256",
        "native_split_quality_central_result_sha256",
        "native_split_quality_selection_sha256",
        "native_split_quality_consumer_attestation",
        "quality_preselection", "quality_preselection_sha256",
    )
    authority_context = (
        authority if isinstance(authority, Mapping)
        else row.get("native_split_quality_authority")
        if isinstance(row.get("native_split_quality_authority"), Mapping)
        else None
    )
    standalone_diagnostic = bool(
        isinstance(authority_context, Mapping)
        and authority_context.get("standalone_unmanaged_diagnostic_only") is True
    )
    authority_required = (
        bool(native_split_quality_required_for_row(row, authority_context))
        if native_split_quality_required_for_row is not None else True
    )
    if standalone_diagnostic:
        authority_required = False
    required = bool(
        authority_required
        or row.get("native_split_quality_required") is True
        or any(row.get(field) not in (None, "", {}) for field in marker_fields)
        or any(command.get(field) not in (None, "", {}) for field in marker_fields)
    )
    if not required:
        return {
            "native_split_quality_required": False,
            "native_split_energy_binding_valid": True,
            "native_split_energy_binding_status": "legacy_split_quality_first_not_required",
            "historical_diagnostic_only": True,
            "scientific_claim_exclusion_reason": (
                "standalone_unmanaged_native_energy_diagnostic_only"
                if standalone_diagnostic else
                "historical_native_split_without_quality_first_authority"
            ),
            "native_split_quality_authority_workflow_version": str(
                (authority_context or {}).get("workflow_version") or ""
            ),
            "native_split_quality_authority_run_id": str(
                (authority_context or {}).get("run_id") or ""
            ),
        }, "legacy_split_quality_first_not_required"
    if (
        authority_required
        and (
            not isinstance(authority_context, Mapping)
            or authority_context.get("valid") is not True
            or str(authority_context.get("mode") or "") != "required"
        )
    ):
        return None, "native_split_quality_authority_invalid"
    if row.get("native_split_quality_required") is not True:
        return None, "native_split_quality_required_marker_missing"
    if row.get("native_split_quality_provenance_conflict") is True:
        return None, "native_split_quality_provenance_conflict"
    if row.get("native_split_final_portable_binding_valid") is not True:
        return None, "final_portable_split_quality_binding_not_verified"
    if row.get("native_split_semantic_binding_valid") is not True:
        return None, "final_split_semantic_manifest_or_payload_binding_not_verified"
    if str(row.get("native_split_quality_consumer_status") or "") != (
        "exact_quality_native_engine_command_and_boundary_match"
    ):
        return None, "native_split_quality_consumer_status_invalid"

    binding = row.get("native_split_quality_binding")
    command_binding = command.get("native_split_quality_binding")
    if (
        not isinstance(binding, Mapping)
        or not isinstance(command_binding, Mapping)
        or dict(binding) != dict(command_binding)
    ):
        return None, "native_split_quality_binding_missing_or_drifted"
    binding = dict(binding)
    binding_sha = _strict_sha256_token(binding.get("binding_sha256"))
    row_binding_sha = _strict_sha256_token(row.get("native_split_quality_binding_sha256"))
    if (
        not binding_sha or row_binding_sha != binding_sha
        or _strict_sha256_token(command.get("native_split_quality_binding_sha256")) != binding_sha
    ):
        return None, "native_split_quality_binding_sha256_drift"
    selection = binding.get("preselection")
    command_selection = command.get("quality_preselection")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(command_selection, Mapping)
        or dict(selection) != dict(command_selection)
    ):
        return None, "native_split_quality_preselection_missing_or_drifted"
    selection_sha = _strict_sha256_token(selection.get("selection_sha256"))
    if (
        not selection_sha
        or _strict_sha256_token(binding.get("preselection_sha256")) != selection_sha
        or _strict_sha256_token(command.get("quality_preselection_sha256")) != selection_sha
    ):
        return None, "native_split_quality_preselection_sha256_drift"
    central_join = (
        (
            "native_split_quality_source_request_sha256",
            _strict_sha256_token(binding.get("source_request_sha256")),
        ),
        (
            "native_split_quality_central_result_sha256",
            _strict_sha256_token(binding.get("central_result_sha256")),
        ),
        (
            "native_split_quality_selection_sha256",
            _strict_sha256_token(binding.get("central_quality_selection_sha256")),
        ),
    )
    if any(not expected for _field, expected in central_join):
        return None, "native_split_quality_central_selection_binding_missing"
    eval_run_id = str(binding.get("eval_run_id") or "").strip()
    source_run_id = str(binding.get("source_run_id") or "").strip()
    if not eval_run_id or not source_run_id:
        return None, "native_split_quality_eval_or_source_run_id_missing"
    if (
        str(row.get("native_split_quality_eval_run_id") or "").strip()
        != eval_run_id
        or str(
            command.get("native_split_quality_eval_run_id") or ""
        ).strip() != eval_run_id
    ):
        return None, "native_split_quality_eval_run_id_drift"
    row_source_run_id = str(
        row.get("native_split_quality_source_run_id") or ""
    ).strip()
    command_source_run_id = str(
        command.get("native_split_quality_source_run_id") or ""
    ).strip()
    canonical_source_run_ids = tuple(
        _canonical_native_split_source_run_id(value)
        for value in (
            source_run_id, row_source_run_id, command_source_run_id,
        )
    )
    if any(not value for value in canonical_source_run_ids):
        return None, (
            "native_split_quality_source_run_id_unknown_or_unsupported"
        )
    if len(set(canonical_source_run_ids)) != 1:
        return None, "native_split_quality_source_run_id_drift"
    canonical_source_run_id = canonical_source_run_ids[0]
    attestation = row.get("native_split_quality_consumer_attestation")
    if not isinstance(attestation, Mapping):
        return None, "native_split_quality_consumer_attestation_missing"
    for field, expected in central_join:
        for container_name, container in (
            ("native_result", row),
            ("native_command", command),
            ("consumer_attestation", attestation),
        ):
            if _strict_sha256_token(container.get(field)) != expected:
                return None, f"{field}_{container_name}_drift"
    if _strict_sha256_token(row.get("source_request_sha256")) != central_join[0][1]:
        return None, "native_split_quality_generic_source_request_sha256_drift"
    command_sha = _strict_sha256_token(command.get("contract_sha256"))
    if (
        not command_sha
        or _strict_sha256_token(row.get("native_command_contract_sha256")) != command_sha
    ):
        return None, "native_split_command_contract_sha256_duplicate_drift"
    portable_row = dict(row)
    portable_row.update({
        "native_command_contract": dict(command),
        "native_command_contract_sha256": command_sha,
    })
    if bind_quality_to_native_split is None:
        return None, "native_split_quality_portable_verifier_unavailable"
    joined, portable_status = bind_quality_to_native_split(
        native_row=portable_row, quality_binding=binding,
        verification_mode="portable",
    )
    if joined is None:
        return None, f"native_split_quality_portable_join_failed:{portable_status}"

    artifacts = command.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    attestation = dict(attestation)
    manifest_specs = (
        (
            "output", "semantic_output_manifest",
            "semantic_output_manifest_sha256",
            [
                row.get("native_split_semantic_output_manifest"),
                row.get("native_output_manifest"), row.get("output_dump_manifest"),
                row.get("native_fifo_output_manifest"),
                (artifacts.get("semantic_output_manifest") or {}).get("path")
                if isinstance(artifacts.get("semantic_output_manifest"), Mapping) else "",
            ],
        ),
        (
            "boundary", "semantic_boundary_manifest",
            "semantic_boundary_manifest_sha256",
            [
                row.get("native_split_semantic_boundary_manifest"),
                row.get("native_fifo_boundary_manifest"), row.get("boundary_manifest"),
                (artifacts.get("semantic_boundary_manifest") or {}).get("path")
                if isinstance(artifacts.get("semantic_boundary_manifest"), Mapping) else "",
            ],
        ),
    )
    manifest_paths: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for role, artifact_name, attestation_field, candidates in manifest_specs:
        artifact = artifacts.get(artifact_name)
        if not isinstance(artifact, Mapping):
            return None, f"native_split_semantic_{role}_manifest_contract_missing"
        expected_sha = _strict_sha256_token(artifact.get("sha256"))
        try:
            from scripts.native_producer_final_report import _resolve_split_semantic_manifest
        except ImportError:
            from native_producer_final_report import _resolve_split_semantic_manifest
        local_output = str(row.get("native_split_semantic_output_manifest") or row.get("native_output_manifest") or row.get("native_fifo_output_manifest") or "")
        anchor = Path(local_output) if local_output else None
        if anchor is not None:
            anchor_dir = anchor.parent.parent if anchor.parent.name in ("native_fifo_outputs", "native_outputs") else anchor.parent
            local_result = anchor_dir / "native_fifo_results.json"
        else:
            local_result = Path(str(row.get("report"))) if row.get("report") else None
        manifest_path, resolution = _resolve_split_semantic_manifest(
            local_result, (row, {"native_command_contract": command}), kind=role,
            fallback_relpaths=(
                "native_fifo_outputs/native_fifo_output_manifest.json",
                "native_outputs/native_outputs_manifest.json",
            ) if role == "output" else (
                "native_fifo_boundary/native_fifo_boundary_manifest.json",
                "native_boundary/native_fifo_boundary_manifest.json",
            ),
        )
        if resolution["manifest_resolution_status"] not in ("manifest_verified", "legacy_manifest_found", "missing"):
            return None, "native_split_semantic_" + role + "_" + resolution["manifest_resolution_status"]
        if not expected_sha or manifest_path is None:
            return None, f"native_split_semantic_{role}_manifest_missing"
        actual_sha = _sha256_file(manifest_path)
        row_sha = _strict_sha256_token(row.get(
            f"native_split_semantic_{role}_manifest_sha256"
        ))
        if (
            actual_sha != expected_sha
            or _strict_sha256_token(attestation.get(attestation_field)) != expected_sha
            or row_sha != expected_sha
        ):
            return None, f"native_split_semantic_{role}_manifest_sha256_drift"
        payload_ok, payload_status = _manifest_payload_bytes_verified(manifest_path)
        if not payload_ok:
            return None, f"native_split_semantic_{role}_{payload_status}"
        manifest_paths[role] = str(manifest_path)
        manifest_hashes[role] = actual_sha

    evidence: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-split-energy-quality-binding",
        "schema_version": 1,
        "native_split_quality_required": True,
        "native_split_energy_binding_valid": True,
        "native_split_energy_binding_status": "portable_join_and_semantic_payload_bytes_rehashed",
        "native_command_contract_sha256": command_sha,
        "native_split_quality_binding_sha256": binding_sha,
        "native_split_quality_preselection_sha256": selection_sha,
        "native_split_quality_source_request_sha256": central_join[0][1],
        "native_split_quality_central_result_sha256": central_join[1][1],
        "native_split_quality_selection_sha256": central_join[2][1],
        "native_split_quality_eval_run_id": eval_run_id,
        "native_split_quality_source_run_id": source_run_id,
        "native_split_quality_source_run_id_canonical":
            canonical_source_run_id,
        "native_split_quality_source_run_id_raw": {
            "binding": source_run_id,
            "native_result": row_source_run_id,
            "native_command": command_source_run_id,
        },
        "native_split_quality_consumer_attestation_sha256": _strict_sha256_token(
            attestation.get("attestation_sha256")
        ),
        "native_split_quality_portable_binding_status": portable_status,
        "native_split_quality_authority_workflow_version": str(
            (authority_context or {}).get("workflow_version") or ""
        ),
        "native_split_quality_authority_run_id": str(
            (authority_context or {}).get("run_id") or ""
        ),
        "native_split_semantic_output_manifest": manifest_paths["output"],
        "native_split_semantic_output_manifest_sha256": manifest_hashes["output"],
        "native_split_semantic_boundary_manifest": manifest_paths["boundary"],
        "native_split_semantic_boundary_manifest_sha256": manifest_hashes["boundary"],
    }
    evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
    return evidence, "portable_join_and_semantic_payload_bytes_rehashed"


def _runtime_observation_split_evidence(
    command_contract_sha256: str,
    *,
    reason: str,
) -> dict[str, Any]:
    """Seal the command identity for a non-claimable Split measurement."""

    evidence: dict[str, Any] = {
        "schema": (
            "onnx-splitpoint/native-split-energy-runtime-observation"
        ),
        "schema_version": 1,
        "native_split_quality_required": True,
        "native_split_energy_binding_valid": False,
        "native_split_energy_binding_status": str(
            reason or "native_split_quality_binding_unavailable"
        ),
        "native_command_contract_sha256": str(
            command_contract_sha256 or ""
        ),
    }
    evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
    return evidence


def _energy_split_quality_authority(summary_path: Path) -> dict[str, Any]:
    """Resolve policy from the immutable EvaluationRun, never row markers."""
    if resolve_native_split_quality_authority is None:
        return {
            "schema": "onnx-splitpoint/native-split-quality-authority",
            "schema_version": 1, "mode": "invalid", "valid": False,
            "native_split_quality_required": True,
            "workflow_version": "", "run_id": "",
            "errors": ["native_split_quality_authority_verifier_unavailable"],
        }
    candidates = [summary_path.parent.parent, summary_path.parent]
    candidates.extend(list(summary_path.parents[2:5]))
    eval_root = next((
        path for path in candidates
        if (path / "run_manifest.json").is_file()
        or (path / "reports/native_producer_stage.json").is_file()
    ), summary_path.parent.parent)
    authority = resolve_native_split_quality_authority(
        run_manifest_path=eval_root / "run_manifest.json",
        stage_path=eval_root / "reports/native_producer_stage.json",
    )
    if (
        authority.get("valid") is not True
        and summary_path.parent.name != "reports"
        and not (eval_root / "run_manifest.json").exists()
        and not (eval_root / "reports/native_producer_stage.json").exists()
    ):
        authority = dict(authority)
        authority["standalone_unmanaged_diagnostic_only"] = True
        authority["errors"] = list(authority.get("errors") or []) + [
            "standalone_summary_without_evaluation_run_authority"
        ]
    return authority


def _historical_energy_diagnostic_only(
    authority: Mapping[str, Any] | None,
) -> bool:
    return bool(
        isinstance(authority, Mapping) and (
            authority.get("standalone_unmanaged_diagnostic_only") is True
            or (
                authority.get("valid") is True
                and str(authority.get("mode") or "") == "legacy"
                and authority.get("native_split_quality_required") is False
            )
        )
    )


def _normalize_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _strict_sha256_token(value: Any) -> str:
    """Return canonical SHA-256 hex for one bare or singly-prefixed token."""
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text) else ""


_QUALITY_PROVENANCE_SHA_ALIASES: dict[str, tuple[str, ...]] = {
    "source_request_sha256": (
        "source_request_sha256", "quality_source_sha256",
    ),
    "model_sha256": (
        "model_sha256", "source_model_sha256", "source_onnx_sha256",
    ),
    "validation_dataset_sha256": (
        "validation_dataset_sha256", "validation_dataset_manifest_sha256",
        "dataset_manifest_sha256", "dataset_sha256",
    ),
    "validation_dataset_image_ids_sha256": (
        "validation_dataset_image_ids_sha256", "validation_image_ids_sha256",
        "dataset_image_ids_sha256", "image_ids_sha256",
    ),
    "validation_dataset_ground_truth_sha256": (
        "validation_dataset_ground_truth_sha256", "validation_ground_truth_sha256",
        "dataset_ground_truth_sha256", "ground_truth_sha256",
    ),
    "accuracy_gate_policy_sha256": (
        "accuracy_gate_policy_sha256",
    ),
    "task_quality_policy_sha256": (
        "task_quality_policy_sha256", "policy_sha256",
    ),
    "runtime_quality_gate_policy_sha256": (
        "runtime_quality_gate_policy_sha256",
    ),
}


def _validated_quality_provenance(
    validation: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
    """Extract only conflict-free, canonical central-quality provenance."""
    source = validation if isinstance(validation, dict) else {}
    identity: dict[str, str] = {}
    conflicts: list[str] = []
    for field, aliases in _QUALITY_PROVENANCE_SHA_ALIASES.items():
        observed: set[str] = set()
        invalid = False
        for alias in aliases:
            value = source.get(alias)
            if value in (None, "") or (isinstance(value, str) and not value.strip()):
                continue
            normalized = _strict_sha256_token(value)
            if not normalized:
                invalid = True
                break
            observed.add(normalized)
        if invalid or len(observed) > 1:
            identity[field] = ""
            conflicts.append(field)
        else:
            identity[field] = next(iter(observed), "")
    policy_fields = (
        "accuracy_gate_policy_sha256", "task_quality_policy_sha256",
        "runtime_quality_gate_policy_sha256",
    )
    policy_values = {identity.get(field) for field in policy_fields if identity.get(field)}
    if len(policy_values) > 1:
        for field in policy_fields:
            identity[field] = ""
            if field not in conflicts:
                conflicts.append(field)
    return identity, conflicts


def _prepair_energy_quality_admission(
    row: Mapping[str, Any],
    validation: dict[str, Any] | None,
    *,
    setup: str,
    command_contract_sha256: str,
    screening_only: bool,
    smoke_diagnostic: bool,
    historical_diagnostic_only: bool,
    force_diagnostic_only: bool = False,
    window_method_validation_probe: bool = False,
    runtime_observation_allowed: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Apply the executor's quality predicate before pair construction."""

    backend = str(row.get("backend") or "")
    model = str(row.get("model") or row.get("model_id") or "")
    case = str(
        row.get("case")
        or row.get("case_id")
        or ("full" if backend.startswith("native_full_") else "")
    )
    is_full = backend.startswith("native_full_")
    precision = (
        _full_comparison_precision(dict(row), validation)
        if is_full
        else str(
            _evidence_value(row, validation, "precision", "dtype") or ""
        ).strip().lower()
    )
    if runtime_observation_allowed and not precision:
        # Display/comparison precision belongs to the downstream Quality
        # annotation.  A technically verified Full command has no Part-2
        # boundary precision and must not disappear when this label is absent.
        precision = "posthoc_annotation_unavailable"
    comparison_backend = str(
        row.get("comparison_backend") or ""
    ).strip()
    if (
        not backend
        or not model
        or not case
        or not setup
        or not precision
        or not comparison_backend
    ):
        return None, "energy_quality_admission_identity_incomplete"
    if not _strict_sha256_token(command_contract_sha256):
        return None, (
            "energy_quality_admission_command_contract_sha256_invalid"
        )
    quality_provenance, provenance_conflicts = (
        _validated_quality_provenance(validation)
    )
    quality_provenance_complete = bool(
        quality_provenance
        and not provenance_conflicts
        and all(quality_provenance.values())
    )
    # Historical positive validation rows predate the explicit v2.72 quality
    # axes.  Project them only when none of the new axes is declared.  A
    # partially migrated row, or any explicit false value, remains fail-closed.
    explicit_quality_axes = (
        "central_quality_evidence_verified",
        "precision_quality_binding_verified",
        "task_quality_observation_valid",
        "accuracy_gate_pass",
        "quality_provenance_complete",
        "quality_claim_result_verified",
    )
    legacy_positive_evidence = bool(
        validation is not None
        and not any(field in validation for field in explicit_quality_axes)
        and validation.get("semantic_ok") is True
        and _truth(validation.get("contract_consistent"))
        and validation.get("claim_ok") is True
        and (
            str(validation.get("task") or row.get("task") or "")
            .strip().lower() != "classification"
            or validation.get("top1_match") is True
        )
    )
    if legacy_positive_evidence:
        quality_provenance_complete = True
    accuracy_value = (
        validation.get("accuracy_gate_pass")
        if validation is not None
        and isinstance(validation.get("accuracy_gate_pass"), bool)
        else bool(legacy_positive_evidence)
    )
    observation_valid = bool(
        validation is not None
        and (
            (
                validation.get("task_quality_observation_valid") is True
                and isinstance(validation.get("accuracy_gate_pass"), bool)
            )
            or legacy_positive_evidence
        )
    )
    semantic_validation_ok = bool(
        validation is not None
        and validation.get("semantic_ok") is True
        and _truth(validation.get("contract_consistent"))
        and (
            str(
                validation.get("task")
                or row.get("task")
                or ""
            ).strip().lower()
            != "classification"
            or validation.get("top1_match") is True
        )
    )
    quality_claim_result_verified = bool(
        validation is not None
        and (
            validation.get("quality_claim_result_verified") is True
            or legacy_positive_evidence
        )
    )
    claim_comparable = bool(
        semantic_validation_ok
        and accuracy_value
        and quality_claim_result_verified
    )
    diagnostic_only = bool(
        screening_only
        or smoke_diagnostic
        or row.get("diagnostic_only") is True
        or bool(row.get("scientific_claim_exclusion_reason"))
        or (historical_diagnostic_only and not legacy_positive_evidence)
        or force_diagnostic_only
        or not claim_comparable
    )
    admission = {
        "schema": "onnx-splitpoint/native-energy-quality-admission",
        "schema_version": 1,
        "admission_scope": (
            "window_method_validation_probe"
            if window_method_validation_probe
            else "native_energy"
        ),
        "backend": backend,
        "model": model,
        "case": case,
        "setup_id": setup,
        "precision": precision,
        "comparison_backend": comparison_backend,
        "successful_command_contract_sha256": str(
            command_contract_sha256 or ""
        ),
        "central_quality_evidence_verified": bool(
            not window_method_validation_probe
            and validation is not None
            and (
                validation.get("central_quality_evidence_verified") is True
                or legacy_positive_evidence
            )
        ),
        "precision_quality_binding_verified": bool(
            not window_method_validation_probe
            and validation is not None
            and (
                validation.get(
                    "precision_quality_binding_verified"
                ) is True
                or legacy_positive_evidence
            )
        ),
        "task_quality_observation_valid": bool(
            observation_valid and not window_method_validation_probe
        ),
        "accuracy_gate_pass": bool(
            accuracy_value and not window_method_validation_probe
        ),
        "quality_provenance_complete": bool(
            quality_provenance_complete
            and not window_method_validation_probe
        ),
        "quality_claim_result_verified": (
            quality_claim_result_verified
            and not window_method_validation_probe
        ),
        "diagnostic_only": bool(
            diagnostic_only or window_method_validation_probe
        ),
        "screening_comparable": bool(
            semantic_validation_ok
            and quality_provenance_complete
            and not window_method_validation_probe
        ),
        "claim_comparable": bool(
            claim_comparable
            and not diagnostic_only
            and not window_method_validation_probe
        ),
        "energy_claim_eligible": bool(
            claim_comparable
            and not diagnostic_only
            and not window_method_validation_probe
        ),
    }
    # Preserve the observed local decision before any cohort-level downgrade.
    local_validation = validation if isinstance(validation, Mapping) else {}
    local_decision = next((str(local_validation.get(key)).strip().lower()
        for key in ("task_quality_gate_decision", "accuracy_gate_decision", "task_quality_status", "task_quality_decision")
        if str(local_validation.get(key) or "").strip().lower() in {"pass", "fail", "inconclusive", "reference", "reference_close", "accuracy_loss", "not_estimable"}), "")
    if local_decision:
        admission["local_task_quality_decision"] = local_decision
    if local_validation.get("accuracy_assessment"):
        admission["accuracy_assessment"] = local_validation["accuracy_assessment"]
        admission["accuracy_gate_semantics"] = local_validation.get("accuracy_gate_semantics")
    if row.get("scientific_claim_exclusion_reason"):
        admission["scientific_claim_exclusion_reason"] = str(row["scientific_claim_exclusion_reason"])
    check_row = {
        **admission,
        "claim_ok": False if diagnostic_only else claim_comparable,
        "semantic_claim_ok": (
            False if diagnostic_only else claim_comparable
        ),
        "claim_eligible": (
            False if diagnostic_only else claim_comparable
        ),
        "eligible_for_energy_results_import": (
            False if diagnostic_only else claim_comparable
        ),
        "eligible_for_scientific_claim": (
            False if diagnostic_only else claim_comparable
        ),
    }
    if runtime_observation_allowed and not all(
        admission.get(field) is True
        for field in (
            "central_quality_evidence_verified",
            "precision_quality_binding_verified",
            "task_quality_observation_valid",
            "quality_provenance_complete",
        )
    ):
        admission = _native_runtime_observation_admission(
            admission,
            reason="quality_evidence_incomplete",
        )
        return (
            admission,
            "shared_native_runtime_observation_admission_verified",
        )
    try:
        validate_energy_quality_admission_axes(
            admission,
            row=check_row,
        )
    except ValueError as exc:
        if runtime_observation_allowed:
            admission = _native_runtime_observation_admission(
                admission,
                reason=str(exc),
            )
            return (
                admission,
                "shared_native_runtime_observation_admission_verified",
            )
        return None, str(exc)
    admission["admission_sha256"] = _quality_admission_sha256(admission)
    return admission, "shared_energy_quality_admission_verified"


def _native_runtime_observation_admission(
    admission: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Seal a measurement-only admission that can never authorize a claim."""

    diagnostic = dict(admission)
    diagnostic.pop("admission_sha256", None)
    if reason == "incomplete_expected_native_matrix":
        reasons = list(diagnostic.get("campaign_exclusion_reasons") or [])
        if reason not in reasons:
            reasons.append(reason)
        diagnostic["campaign_exclusion_reasons"] = reasons
    diagnostic.update({
        "admission_scope": "native_runtime_observation",
        "diagnostic_only": True,
        "screening_comparable": False,
        "claim_comparable": False,
        "energy_claim_eligible": False,
        "runtime_observation_reason": str(
            reason or "quality_or_pairing_not_claim_eligible"
        ),
    })
    check_row = {
        **diagnostic,
        "claim_ok": False,
        "semantic_claim_ok": False,
        "claim_eligible": False,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_claim": False,
    }
    validate_energy_quality_admission_axes(
        diagnostic,
        row=check_row,
    )
    diagnostic["admission_sha256"] = _quality_admission_sha256(
        diagnostic
    )
    return diagnostic


def _fallback_native_runtime_observation_admission(
    row: Mapping[str, Any], *, setup: str, reason: str,
) -> dict[str, Any]:
    """Build a sealed post-hoc annotation without changing plan membership."""

    backend = str(row.get("backend") or "").strip()
    command_contract = row.get("_verified_energy_command_contract")
    command_contract = (
        command_contract if isinstance(command_contract, Mapping) else {}
    )
    admission = {
        "schema": "onnx-splitpoint/native-energy-quality-admission",
        "schema_version": 1,
        "admission_scope": "native_runtime_observation",
        "backend": backend,
        "model": str(
            row.get("model") or row.get("model_id") or ""
        ).strip(),
        "case": str(
            row.get("case")
            or row.get("case_id")
            or ("full" if backend.startswith("native_full_") else "")
        ).strip(),
        "setup_id": str(setup or "").strip(),
        "precision": str(
            row.get("precision") or "posthoc_annotation_unavailable"
        ).strip().lower(),
        "comparison_backend": str(
            row.get("comparison_backend")
            or "posthoc_annotation_unavailable"
        ).strip(),
        "successful_command_contract_sha256": str(
            command_contract.get("contract_sha256") or ""
        ).strip().lower(),
        "central_quality_evidence_verified": False,
        "precision_quality_binding_verified": False,
        "task_quality_observation_valid": False,
        "accuracy_gate_pass": False,
        "quality_provenance_complete": False,
        "quality_claim_result_verified": False,
        "diagnostic_only": True,
        "screening_comparable": False,
        "claim_comparable": False,
        "energy_claim_eligible": False,
    }
    return _native_runtime_observation_admission(
        admission, reason=reason,
    )


def _energy_quality_result_fields(
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose whether measured Energy has complete row-local Quality.

    This is an annotation only.  It neither changes technical measurement
    admission nor weakens any existing claim gate.  A runtime-successful row
    with missing/invalid Quality therefore remains measurable, while its raw
    Energy result is unmistakably non-qualified.
    """

    qualified = bool(
        str(admission.get("admission_scope") or "") == "native_energy"
        and all(
            admission.get(field) is True
            for field in (
                "central_quality_evidence_verified",
                "precision_quality_binding_verified",
                "task_quality_observation_valid",
                "accuracy_gate_pass",
                "quality_provenance_complete",
                "quality_claim_result_verified",
            )
        )
        and admission.get("diagnostic_only") is False
        and admission.get("claim_comparable") is True
        and admission.get("energy_claim_eligible") is True
    )
    return {
        **energy_quality_reason_projection(admission),
        "energy_quality_qualified": qualified,
        "energy_quality_status": (
            "quality_qualified"
            if qualified else "raw_energy_quality_not_qualified"
        ),
        "native_energy_after_technical_error": (
            "not_applicable_quality_qualified"
            if qualified else "collect_raw_quality_unqualified"
        ),
    }


def _energy_preprocess_identity(contract: Mapping[str, Any]) -> dict[str, str]:
    """Return the hardware-independent image-preparation identity.

    Runtime tensor dtype/layout may legitimately differ across accelerators.
    Spatial preparation may not: a paired energy claim must use the same task,
    source image, resize/letterbox policy and effective pad value.
    """
    backend = str(contract.get("backend") or "").strip().lower()
    options = dict(contract.get("runtime_options") or {})
    prepared = dict(contract.get("prepared_input_contract") or {})
    workload = dict(contract.get("energy_workload") or {})
    input_contract = dict(workload.get("input_contract") or {})
    input_row = dict(input_contract.get("input") or {})
    runtime_contract = dict(workload.get("runtime_input_contract") or {})
    preprocess = dict(
        workload.get("preprocess")
        or runtime_contract.get("preprocess")
        or input_row
        or {}
    )

    task = str(
        options.get("task")
        or prepared.get("task")
        or workload.get("task")
        or ""
    ).strip().lower()
    raw_mode = str(
        options.get("preprocess_mode_effective")
        or prepared.get("preprocess_mode_effective")
        or preprocess.get("preprocess_mode_effective")
        or preprocess.get("preprocess_mode")
        or preprocess.get("mode")
        or workload.get("preprocess_mode")
        or ""
    ).strip().lower()
    if raw_mode == "auto":
        raw_mode = "letterbox" if task == "detection" else "resize" if task == "classification" else ""
    mode = "letterbox" if "letterbox" in raw_mode else "resize" if "resize" in raw_mode else ""
    raw_pad = (
        options.get("letterbox_pad_value")
        if options.get("letterbox_pad_value") is not None
        else prepared.get("letterbox_pad_value")
        if prepared.get("letterbox_pad_value") is not None
        else preprocess.get("pad_value")
        if preprocess.get("pad_value") is not None
        else preprocess.get("letterbox_pad_value")
        if preprocess.get("letterbox_pad_value") is not None
        else workload.get("letterbox_pad_value")
    )
    try:
        pad = str(int(raw_pad)) if mode == "letterbox" else "0" if mode == "resize" else ""
    except Exception:
        pad = ""
    source_sha = _normalize_sha256(str(
        prepared.get("source_image_sha256")
        or preprocess.get("source_image_sha256")
        or workload.get("input_image_sha256")
        or contract.get("input_image_sha256")
        or ""
    ))
    return {
        "prepared_feed_task": task,
        "prepared_feed_preprocess_mode": mode,
        "prepared_feed_letterbox_pad_value": pad,
        "prepared_feed_source_image_sha256": source_sha,
        "prepared_feed_identity_source": (
            "successful_split_prepared_input_contract"
            if not backend.startswith("native_full_")
            else "successful_full_runtime_input_contract"
        ),
    }


def _stable_json_sha256(value: Any) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_component(value: Any) -> str:
    text = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")
    ).strip("._")
    return text or "unknown"


def _canonical_remote_absolute_path(value: Any) -> Path | None:
    """Return a canonical lexical remote path without touching local files."""
    raw = str(value or "").strip()
    path = Path(raw)
    if not raw or not path.is_absolute() or ".." in path.parts:
        return None
    normalized = Path(os.path.normpath(raw))
    if str(normalized) != raw:
        return None
    return normalized


def _sealed_deepx_source_model_binding(
    contract: Mapping[str, Any], workload: Mapping[str, Any],
    artifacts: Mapping[str, Any], *, expected_identity: Mapping[str, Any],
) -> bool:
    """Validate the lexical BenchmarkSet Source-ONNX -> DXNN binding."""
    def _positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    model = str(expected_identity.get("model") or contract.get("model") or "").strip()
    benchmark_set = _canonical_remote_absolute_path(
        contract.get("benchmark_set")
    )
    source_key = str(workload.get("source_model_artifact") or "")
    manifest_key = str(
        workload.get("benchmark_set_manifest_artifact") or ""
    )
    dxnn_key = str(workload.get("dxnn_artifact") or "")
    source = artifacts.get(source_key)
    manifest = artifacts.get(manifest_key)
    dxnn = artifacts.get(dxnn_key)
    binding = contract.get("model_binding")
    if (
        not model
        or model != _safe_component(model)
        or Path(model).name != model
        or benchmark_set is None
        or source_key != "source_onnx"
        or manifest_key != "benchmark_set_manifest"
        or dxnn_key != "dxnn"
        or not isinstance(source, Mapping)
        or not isinstance(manifest, Mapping)
        or not isinstance(dxnn, Mapping)
        or not isinstance(binding, Mapping)
    ):
        return False
    source_sha = _strict_sha256_token(source.get("sha256"))
    manifest_sha = _strict_sha256_token(manifest.get("sha256"))
    dxnn_sha = _strict_sha256_token(dxnn.get("sha256"))
    expected_model_sha = _strict_sha256_token(
        expected_identity.get("model_sha256")
        or expected_identity.get("source_model_sha256")
    )
    return bool(
        source_sha and manifest_sha and dxnn_sha
        and (
            not expected_model_sha or expected_model_sha == source_sha
        )
        and _canonical_remote_absolute_path(source.get("path"))
        == benchmark_set / "models" / f"{model}.onnx"
        and _canonical_remote_absolute_path(manifest.get("path"))
        == benchmark_set / "benchmark_set.json"
        and _positive_int(source.get("size_bytes")) is not None
        and _positive_int(manifest.get("size_bytes")) is not None
        and str(source.get("role") or "")
        == "sealed_benchmark_set_source_onnx"
        and str(source.get("relative_path") or "")
        == f"models/{model}.onnx"
        and _strict_sha256_token(
            source.get("benchmark_set_manifest_sha256")
        ) == manifest_sha
        and str(manifest.get("role") or "")
        == "sealed_benchmark_set_manifest"
        and str(workload.get("source_model_binding_status") or "")
        == "benchmark_set_source_onnx_verified_exact"
        and _strict_sha256_token(contract.get("source_model_sha256"))
        == source_sha
        and str(binding.get("source_artifact") or "") == source_key
        and _strict_sha256_token(binding.get("source_onnx_sha256"))
        == source_sha
        and str(binding.get("compiled_artifact") or "") == dxnn_key
        and _strict_sha256_token(
            binding.get("compiled_artifact_sha256")
        ) == dxnn_sha
        and str(binding.get("status") or "")
        == "source_and_compiled_artifact_hash_bound"
    )


def _deepx_energy_artifact_role_paths_valid(
    contract: Mapping[str, Any], workload: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *, expected_identity: Mapping[str, Any],
) -> bool:
    root = _canonical_remote_absolute_path(contract.get("root"))
    benchmark_set = _canonical_remote_absolute_path(
        contract.get("benchmark_set")
    )
    authoritative_root = _canonical_remote_absolute_path(
        expected_identity.get("remote_root")
    )
    authoritative_tool_dir = _canonical_remote_absolute_path(
        expected_identity.get("remote_tool_dir")
    )
    model = str(expected_identity.get("model") or "").strip()
    if (
        root is None
        or benchmark_set is None
        or authoritative_root is None
        or authoritative_tool_dir is None
        or not model
        or model != _safe_component(model)
        or Path(model).name != model
        or root != authoritative_root
        or benchmark_set != authoritative_root / model / "benchmark_set"
    ):
        return False
    semantic_root = (
        benchmark_set / "native_full_outputs"
        / f"model={_safe_component(contract.get('model'))}"
        / "backend=native_full_deepx"
        / f"setup={_safe_component(contract.get('setup_id') or 'unspecified')}"
        / f"comparison={_safe_component(contract.get('comparison_backend') or 'unspecified')}"
    )
    runtime_artifact = artifacts.get(
        str(workload.get("runtime_input_artifact") or "")
    )
    manifest_artifact = artifacts.get(
        str(workload.get("input_manifest_artifact") or "")
    )
    runner_artifact = artifacts.get(
        str(workload.get("runner_artifact") or "")
    )
    dxnn_artifact = artifacts.get(
        str(workload.get("dxnn_artifact") or "")
    )
    source_artifact = artifacts.get(
        str(workload.get("source_model_artifact") or "")
    )
    benchmark_manifest_artifact = artifacts.get(
        str(workload.get("benchmark_set_manifest_artifact") or "")
    )
    if not all(isinstance(value, Mapping) for value in (
        runtime_artifact, manifest_artifact, runner_artifact, dxnn_artifact,
        source_artifact, benchmark_manifest_artifact,
    )):
        return False
    runtime_path = _canonical_remote_absolute_path(
        runtime_artifact.get("path")
    )
    manifest_path = _canonical_remote_absolute_path(
        manifest_artifact.get("path")
    )
    runner_path = _canonical_remote_absolute_path(
        runner_artifact.get("path")
    )
    dxnn_path = _canonical_remote_absolute_path(dxnn_artifact.get("path"))
    source_path = _canonical_remote_absolute_path(
        source_artifact.get("path")
    )
    benchmark_manifest_path = _canonical_remote_absolute_path(
        benchmark_manifest_artifact.get("path")
    )
    if runtime_path != semantic_root / "runtime_input.bin":
        return False
    if manifest_path != semantic_root / "native_full_input_manifest.json":
        return False
    if source_path != benchmark_set / "models" / f"{model}.onnx":
        return False
    if benchmark_manifest_path != benchmark_set / "benchmark_set.json":
        return False
    expected_runner = (
        authoritative_tool_dir / "scripts"
        / "native_deepx_full_energy_hotloop.py"
    )
    if runner_path != expected_runner:
        return False
    if dxnn_path is None:
        return False
    try:
        relative = dxnn_path.relative_to(benchmark_set)
    except ValueError:
        return False
    return bool(
        relative.parts
        and relative.parts[0] == "deepx"
        and "full" in relative.parts[:-1]
        and relative.suffix.lower() == ".dxnn"
    )


def _sealed_trt_build_receipt_valid(
    contract: Mapping[str, Any], workload: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> bool:
    source = artifacts.get(str(workload.get("source_model_artifact") or ""))
    engine = artifacts.get(str(workload.get("engine_artifact") or ""))
    trtexec = artifacts.get(str(workload.get("trtexec_artifact") or ""))
    receipt_artifact = artifacts.get(
        str(workload.get("engine_build_receipt_artifact") or "")
    )
    raw_receipt = contract.get("trt_engine_build_receipt")
    if not all(isinstance(value, Mapping) for value in (
        source, engine, trtexec, receipt_artifact, raw_receipt,
    )):
        return False
    receipt = dict(raw_receipt)
    receipt_sha = _strict_sha256_token(receipt.pop("receipt_sha256", ""))
    if (
        not receipt_sha or _canonical_json_sha256(receipt) != receipt_sha
        or receipt.get("schema") != TRT_ENGINE_BUILD_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TRT_ENGINE_BUILD_RECEIPT_VERSION
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
    ):
        return False
    source_path = str(source.get("path") or "")
    engine_path = str(engine.get("path") or "")
    trtexec_path = str(trtexec.get("path") or "")
    source_sha = _strict_sha256_token(source.get("sha256"))
    engine_sha = _strict_sha256_token(engine.get("sha256"))
    trtexec_sha = _strict_sha256_token(trtexec.get("sha256"))
    command = receipt.get("command")
    argv = [str(value) for value in command] if isinstance(command, list) else []
    return bool(
        source_path and engine_path and trtexec_path
        and source_sha and engine_sha and trtexec_sha
        and _strict_sha256_token(receipt_artifact.get("sha256"))
        and str(receipt.get("source_onnx") or "") == source_path
        and _strict_sha256_token(receipt.get("source_onnx_sha256")) == source_sha
        and str(receipt.get("engine") or "") == engine_path
        and _strict_sha256_token(receipt.get("engine_sha256")) == engine_sha
        and str(receipt.get("trtexec") or "") == trtexec_path
        and _strict_sha256_token(receipt.get("trtexec_sha256")) == trtexec_sha
        and argv and argv[0] == trtexec_path
        and [value for value in argv[1:] if value.startswith("--onnx=")]
        == [f"--onnx={source_path}"]
        and [value for value in argv[1:] if value.startswith("--saveEngine=")]
        == [f"--saveEngine={engine_path}"]
        and str(contract.get("trt_engine_build_receipt_status") or "")
        == "engine_build_receipt_verified"
        and str(workload.get("engine_build_receipt_status") or "")
        == "engine_build_receipt_verified"
    )


def _frozen_implementation_artifacts_bound(
    frozen: Mapping[str, Any], artifacts: Mapping[str, Any],
) -> bool:
    implementation = frozen.get("implementation_artifacts")
    if not isinstance(implementation, Mapping) or not implementation:
        return False
    for name, raw_artifact in implementation.items():
        artifact = artifacts.get(f"frozen_postprocess_{name}")
        if (
            not isinstance(raw_artifact, Mapping)
            or not isinstance(artifact, Mapping)
            or not _strict_sha256_token(raw_artifact.get("sha256"))
            or _strict_sha256_token(artifact.get("sha256"))
            != _strict_sha256_token(raw_artifact.get("sha256"))
        ):
            return False
    return True


def _raw_head_completed_tensorrt_contract_valid(
    workload: Mapping[str, Any], artifacts: Mapping[str, Any],
    *, completed_frames: int, postprocess_frames: int,
) -> bool:
    try:
        frozen = verify_frozen_postprocess_contract(
            workload.get("frozen_postprocess_contract")
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        return False
    original_wh = workload.get("original_image_wh")
    return bool(
        str(workload.get("completed_task_completion_mode") or "")
        in {"", "frozen_host_tail"}
        and workload.get("host_postprocess_frozen") is True
        and workload.get("frozen_postprocess_implementation_bound") is True
        and not isinstance(
            workload.get("frozen_decoded_nms_normalization_contract"),
            Mapping,
        )
        and workload.get("normalization_frozen") is not True
        and isinstance(original_wh, list)
        and len(original_wh) == 2
        and not any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in original_wh
        )
        and list(original_wh) == list(frozen.get("original_wh") or [])
        and _strict_sha256_token(
            workload.get("frozen_postprocess_contract_sha256")
        ) == _strict_sha256_token(frozen.get("contract_sha256"))
        and completed_frames > 0
        and postprocess_frames == completed_frames
        and _frozen_implementation_artifacts_bound(frozen, artifacts)
    )


def _direct_completed_tensorrt_contract_valid(
    workload: Mapping[str, Any], artifacts: Mapping[str, Any],
    *, completed_frames: int, postprocess_frames: int,
) -> bool:
    try:
        frozen = verify_frozen_decoded_nms_normalization_contract(
            workload.get("frozen_decoded_nms_normalization_contract")
        )
        expected_attestation = (
            build_normalized_detection_endpoint_attestation(
                frozen,
                workload.get("frozen_decoded_nms_normalization_result"),
                completed_frames=completed_frames,
                postprocess_completed_frames=postprocess_frames,
            )
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        return False
    original_wh = workload.get("original_image_wh")
    source_signature = workload.get("source_output_tensor_signature")
    result = workload.get("frozen_decoded_nms_normalization_result")
    return bool(
        str(workload.get("completed_task_completion_mode") or "")
        == "integrated_accelerator_plus_frozen_normalization"
        and workload.get("normalization_frozen") is True
        and workload.get("host_postprocess_frozen") is not True
        and not isinstance(workload.get("frozen_postprocess_contract"), Mapping)
        and workload.get("postprocess_completion_verified") is True
        and workload.get("postprocess_completed_frames") == postprocess_frames
        and isinstance(result, Mapping)
        and result.get("task") == "detection"
        and result.get("contract_family") == "decoded_nms"
        and result.get("decoder_format") == "bn6_detections"
        and result.get("coordinate_space")
        == "original_image_xyxy_pixels"
        and isinstance(result.get("detection_count"), int)
        and not isinstance(result.get("detection_count"), bool)
        and int(result.get("detection_count") or 0) >= 0
        and bool(_strict_sha256_token(result.get("detections_sha256")))
        and _strict_sha256_token(
            result.get("normalization_contract_sha256")
        ) == _strict_sha256_token(frozen.get("contract_sha256"))
        and isinstance(original_wh, list)
        and list(original_wh) == list(frozen.get("original_wh") or [])
        and _strict_sha256_token(
            workload.get(
                "frozen_decoded_nms_normalization_contract_sha256"
            )
        ) == _strict_sha256_token(frozen.get("contract_sha256"))
        and _strict_sha256_token(
            workload.get("source_endpoint_contract_hash")
        ) == _strict_sha256_token(
            frozen.get("source_endpoint_contract_hash")
        )
        and str(workload.get("source_output_endpoint_id") or "")
        == str(frozen.get("source_output_endpoint_id") or "")
        and isinstance(source_signature, Mapping)
        and dict(source_signature)
        == dict(frozen.get("source_output_tensor_signature") or {})
        and _strict_sha256_token(
            workload.get("source_output_endpoint_attestation_sha256")
        ) == _strict_sha256_token(
            frozen.get("source_output_endpoint_attestation_sha256")
        )
        and _strict_sha256_token(
            workload.get("letterbox_geometry_contract_sha256")
        ) == _strict_sha256_token(
            frozen.get("letterbox_geometry_contract_sha256")
        )
        and dict(workload.get("completed_task_endpoint_attestation") or {})
        == expected_attestation
        and _frozen_implementation_artifacts_bound(frozen, artifacts)
    )


def _completed_tensorrt_task_contract_valid(
    workload: Mapping[str, Any], artifacts: Mapping[str, Any],
) -> bool:
    """Verify exactly one sealed completed-detection implementation."""
    completed_frames = workload.get("successful_run_completed_frames")
    successful_postprocess_frames = workload.get(
        "successful_run_postprocess_completed_frames"
    )
    reported_postprocess_frames = workload.get(
        "postprocess_completed_frames"
    )
    postprocess_frames = (
        successful_postprocess_frames
        if successful_postprocess_frames is not None
        else reported_postprocess_frames
    )
    stage = "classification_top1_top5" if workload.get("task") == "classification" else "decoded_nms"
    measurement_concurrency = workload.get("measurement_concurrency")
    if (
        str(workload.get("e2e_scope") or "") != "full_task_pipeline"
        or str(workload.get("completed_task_stage") or "") != stage
        or str(workload.get("completed_task_contract_family") or "")
        != stage
        or workload.get("postprocess_required") is not True
        or workload.get("postprocess_included") is not True
        or isinstance(measurement_concurrency, bool)
        or not isinstance(measurement_concurrency, int)
        or measurement_concurrency != 1
        or isinstance(completed_frames, bool)
        or not isinstance(completed_frames, int)
        or completed_frames <= 0
        or isinstance(postprocess_frames, bool)
        or not isinstance(postprocess_frames, int)
        or postprocess_frames != completed_frames
        or (
            successful_postprocess_frames is not None
            and reported_postprocess_frames is not None
            and successful_postprocess_frames
            != reported_postprocess_frames
        )
    ):
        return False
    if workload.get("task") == "classification":
        return bool(
            workload.get("postprocess_completion_verified") is True
            and not workload.get("host_postprocess_frozen")
            and not workload.get("normalization_frozen")
        )
    raw_mode = isinstance(workload.get("frozen_postprocess_contract"), Mapping)
    direct_mode = isinstance(
        workload.get("frozen_decoded_nms_normalization_contract"), Mapping,
    )
    if raw_mode == direct_mode:
        return False
    if direct_mode:
        return _direct_completed_tensorrt_contract_valid(
            workload, artifacts,
            completed_frames=completed_frames,
            postprocess_frames=postprocess_frames,
        )
    return _raw_head_completed_tensorrt_contract_valid(
        workload, artifacts,
        completed_frames=completed_frames,
        postprocess_frames=postprocess_frames,
    )


def _verify_full_command_contract(
    raw: Any, *, expected_identity: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "full_command_contract_missing"
    contract = dict(raw)
    declared = str(contract.pop("contract_sha256", "") or "").strip().lower()
    if len(declared) != 64 or _canonical_json_sha256(contract) != declared:
        return None, "full_command_contract_sha256_mismatch"
    contract["contract_sha256"] = declared
    if (
        contract.get("schema") != FULL_COMMAND_CONTRACT_SCHEMA
        or int(contract.get("schema_version") or 0) != FULL_COMMAND_CONTRACT_VERSION
    ):
        return None, "full_command_contract_incomplete_or_schema_invalid"
    workload = contract.get("energy_workload")
    if (
        contract.get("complete") is not True
        or not isinstance(workload, dict)
        or workload.get("available") is not True
    ):
        return None, "full_energy_hotloop_unavailable"
    for field in ("backend", "model", "case", "setup_id", "comparison_backend"):
        expected = str(expected_identity.get(field) or "").strip().lower()
        actual = str(contract.get(field) or "").strip().lower()
        if expected and expected != actual:
            return None, f"full_command_contract_{field}_mismatch"
    if len(str(contract.get("runner_sha256") or "").strip()) != 64:
        return None, "full_command_contract_runner_sha256_missing"
    for field in ("python_executable", "runner", "root", "benchmark_set", "backend_arg"):
        if not str(contract.get(field) or "").strip():
            return None, f"full_command_contract_{field}_missing"
    options = contract.get("runtime_options")
    if not isinstance(options, dict):
        return None, "full_command_contract_runtime_options_missing"
    for field in (
        "frames", "warmup", "inflight", "trt_precision", "workspace_mb",
        "engine_build_python", "no_shapes", "dump_outputs",
        "diagnostic_deepx_input_probes", "image_map",
    ):
        if field not in options:
            return None, f"full_command_contract_{field}_missing"
    if str(contract.get("backend") or "").startswith("native_full_hailo"):
        if len(str(contract.get("input_image_sha256") or "").strip()) != 64:
            return None, "full_command_contract_input_image_sha256_missing"
        hef = (contract.get("artifacts") or {}).get("hef") if isinstance(contract.get("artifacts"), dict) else None
        if not isinstance(hef, dict) or len(str(hef.get("sha256") or "").strip()) != 64:
            return None, "full_command_contract_hef_sha256_missing"
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        return None, "full_command_contract_artifacts_missing"
    command_python = artifacts.get("command_python_executable")
    if (
        not isinstance(command_python, dict)
        or str(command_python.get("invocation_path") or "")
        != str(contract.get("python_executable") or "")
        or len(str(command_python.get("sha256") or "").strip()) != 64
        or not isinstance(command_python.get("interpreter_identity"), dict)
    ):
        return None, "full_command_contract_command_interpreter_binding_missing"
    workload_kind = str(workload.get("kind") or "")
    if workload_kind not in _FULL_ENERGY_WORKLOAD_KINDS:
        return None, "full_command_contract_workload_kind_unsupported"
    if workload_kind in _FULL_RUNTIME_PYTHON_WORKLOAD_KINDS:
        runtime_key = str(workload.get("runtime_python_artifact") or "")
        runtime_python = artifacts.get(runtime_key)
        if (
            not runtime_key
            or not isinstance(runtime_python, dict)
            or len(str(runtime_python.get("sha256") or "").strip()) != 64
            or not str(runtime_python.get("invocation_path") or "").strip()
            or not isinstance(runtime_python.get("interpreter_identity"), dict)
        ):
            return None, "full_command_contract_runtime_interpreter_binding_missing"
        if len(str(workload.get("input_image_sha256") or "").strip()) != 64:
            return None, "full_command_contract_workload_input_image_sha256_missing"
        if (
            _normalize_sha256(workload.get("input_image_sha256"))
            != _normalize_sha256(contract.get("input_image_sha256"))
        ):
            return None, "full_command_contract_workload_input_image_sha256_mismatch"
    if workload_kind == "hailo_full_hotloop":
        for artifact_name in (
            "runner_artifact", "hef_artifact",
            "input_manifest_artifact", "runtime_input_artifact",
        ):
            key = str(workload.get(artifact_name) or "")
            artifact = artifacts.get(key)
            if not key or not isinstance(artifact, dict) or len(str(artifact.get("sha256") or "").strip()) != 64:
                return None, f"full_command_contract_hailo_{artifact_name}_missing"
        runtime_name = str(workload.get("runtime_input_name") or "").strip()
        runtime_shape = workload.get("runtime_input_shape")
        runtime_dtype = str(workload.get("runtime_input_dtype") or "").strip().lower()
        runtime_bytes = workload.get("runtime_input_bytes")
        dtype_bytes = {
            "uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
            "float16": 2, "uint32": 4, "int32": 4, "float32": 4,
            "float64": 8,
        }
        if (
            not runtime_name
            or not isinstance(runtime_shape, list)
            or not runtime_shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in runtime_shape)
            or runtime_dtype not in dtype_bytes
            or isinstance(runtime_bytes, bool)
            or not isinstance(runtime_bytes, int)
            or runtime_bytes != math.prod(runtime_shape) * dtype_bytes[runtime_dtype]
            or workload.get("runtime_input_mode") != "exact_semantic_dump_runtime_tensor"
        ):
            return None, "full_command_contract_hailo_runtime_input_contract_invalid"
        canonical_inputs = workload.get("canonical_input_slot_names")
        canonical_outputs = workload.get("canonical_output_slot_names")
        if (
            canonical_inputs != [runtime_name]
            or not isinstance(canonical_outputs, list)
            or not canonical_outputs
            or any(not isinstance(name, str) or not name.strip() for name in canonical_outputs)
            or len(set(canonical_outputs)) != len(canonical_outputs)
        ):
            return None, "full_command_contract_hailo_canonical_io_binding_invalid"
        runtime_contract = workload.get("runtime_input_contract")
        runtime_artifact = artifacts.get(str(workload.get("runtime_input_artifact") or ""))
        if (
            not isinstance(runtime_contract, dict)
            or runtime_contract.get("schema") != "onnx-splitpoint/preverified-runtime-input-tensor"
            or int(runtime_contract.get("schema_version") or 0) != 1
            or runtime_contract.get("runtime_input_name") != runtime_name
            or runtime_contract.get("runtime_input_shape") != runtime_shape
            or runtime_contract.get("runtime_input_dtype") != runtime_dtype
            or runtime_contract.get("runtime_input_bytes") != runtime_bytes
            or not isinstance(runtime_contract.get("preprocess"), dict)
            or not isinstance(runtime_artifact, dict)
            or runtime_artifact.get("sha256") != runtime_contract.get("runtime_input_sha256")
            or runtime_artifact.get("bytes") != runtime_bytes
        ):
            return None, "full_command_contract_hailo_preverified_tensor_binding_invalid"
    if workload_kind in _TRT_FULL_ENERGY_WORKLOAD_KINDS:
        required_artifacts = [
            "trtexec_artifact", "engine_artifact",
            "input_manifest_artifact", "runtime_input_artifact",
            "engine_build_receipt_artifact",
        ]
        if workload_kind == "tensorrt_full_completed_task_hotloop":
            required_artifacts.append("runner_artifact")
        for artifact_name in required_artifacts:
            key = str(workload.get(artifact_name) or "")
            artifact = artifacts.get(key)
            if not key or not isinstance(artifact, dict) or len(str(artifact.get("sha256") or "").strip()) != 64:
                return None, f"full_command_contract_tensorrt_{artifact_name}_missing"
        if not str(workload.get("runtime_input_name") or "").strip():
            return None, "full_command_contract_tensorrt_runtime_input_name_missing"
        if str(workload.get("input_mode") or "") != "exact_semantic_dump_runtime_tensor":
            return None, "full_command_contract_tensorrt_random_input_not_allowed"
        source_key = str(workload.get("source_model_artifact") or "")
        engine_key = str(workload.get("engine_artifact") or "")
        source_artifact = artifacts.get(source_key)
        engine_artifact = artifacts.get(engine_key)
        binding = contract.get("model_binding")
        if (
            not source_key
            or not isinstance(source_artifact, dict)
            or not isinstance(engine_artifact, dict)
            or not isinstance(binding, dict)
        ):
            return None, "full_command_contract_tensorrt_source_model_binding_missing"
        source_sha = _strict_sha256_token(source_artifact.get("sha256"))
        engine_sha = _strict_sha256_token(engine_artifact.get("sha256"))
        expected_model_sha = _strict_sha256_token(
            expected_identity.get("model_sha256")
            or expected_identity.get("source_model_sha256")
        )
        if not (
            source_sha and engine_sha
            and _strict_sha256_token(contract.get("source_model_sha256")) == source_sha
            and _strict_sha256_token(
                engine_artifact.get("compiled_from_source_onnx_sha256")
            ) == source_sha
            and str(binding.get("source_artifact") or "") == source_key
            and _strict_sha256_token(binding.get("source_onnx_sha256")) == source_sha
            and str(binding.get("compiled_artifact") or "") == engine_key
            and _strict_sha256_token(binding.get("compiled_artifact_sha256")) == engine_sha
            and str(binding.get("status") or "")
            == "verified_engine_build_receipt_bound"
            and _sealed_trt_build_receipt_valid(contract, workload, artifacts)
        ):
            return None, "full_command_contract_tensorrt_source_model_binding_invalid"
        if expected_model_sha and expected_model_sha != source_sha:
            return None, "full_command_contract_tensorrt_source_model_sha256_mismatch"
        if (
            workload_kind == "tensorrt_full_completed_task_hotloop"
            and not _completed_tensorrt_task_contract_valid(
                workload, artifacts
            )
        ):
            return None, (
                "full_command_contract_tensorrt_completed_task_contract_invalid"
            )
    if workload_kind == "deepx_full_prepared_feed_hotloop":
        if str(workload.get("prepared_feed_contract_version") or "") != "deepx-sealed-runtime-input-v3":
            return None, "full_command_contract_deepx_prepared_feed_contract_mismatch"
        runtime_key = str(workload.get("runtime_input_artifact") or "")
        runtime_artifact = artifacts.get(runtime_key)
        preprocessing = workload.get("runtime_preprocessing_identity")
        numeric = workload.get("runtime_numeric_input_identity")
        preprocessing_sha = str(
            workload.get("runtime_preprocessing_sha256") or ""
        ).strip().lower()
        numeric_sha = str(
            workload.get("runtime_numeric_input_sha256") or ""
        ).strip().lower()
        numeric_errors = (
            runtime_numeric_input_identity_errors(numeric, preprocessing)
            if isinstance(preprocessing, Mapping)
            and isinstance(numeric, Mapping)
            else ["runtime_numeric_input_identity_invalid"]
        )
        if not _deepx_energy_artifact_role_paths_valid(
            contract, workload, artifacts,
            expected_identity=expected_identity,
        ):
            return None, "full_command_contract_deepx_artifact_role_path_mismatch"
        if not _sealed_deepx_source_model_binding(
            contract, workload, artifacts,
            expected_identity=expected_identity,
        ):
            return None, "full_command_contract_deepx_source_model_binding_invalid"
        if (
            workload.get("runtime_input_mode")
            != "exact_semantic_dump_runtime_tensor"
            or workload.get("runtime_input_binding_verified") is not True
            or not runtime_key
            or not isinstance(runtime_artifact, dict)
            or str(runtime_artifact.get("sha256") or "").strip().lower()
            != str(workload.get("runtime_input_sha256") or "").strip().lower()
            or int(runtime_artifact.get("bytes") or 0)
            != int(workload.get("runtime_input_bytes") or 0)
            or not str(workload.get("runtime_input_name") or "").strip()
            or not isinstance(workload.get("runtime_input_shape"), list)
            or not workload.get("runtime_input_shape")
            or not str(workload.get("runtime_input_dtype") or "").strip()
            or not str(workload.get("runtime_input_layout") or "").strip()
            or not isinstance(preprocessing, dict)
            or not isinstance(numeric, dict)
            or preprocessing.get("schema")
            != PREPROCESSING_CONTRACT_SCHEMA
            or int(preprocessing.get("schema_version") or 0)
            != PREPROCESSING_CONTRACT_SCHEMA_VERSION
            or numeric.get("schema") != RUNTIME_NUMERIC_INPUT_SCHEMA
            or int(numeric.get("schema_version") or 0)
            != RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION
            or _canonical_json_sha256(preprocessing) != preprocessing_sha
            or preprocessing_contract_sha256(preprocessing)
            != preprocessing_sha
            or _canonical_json_sha256(numeric) != numeric_sha
            or bool(numeric_errors)
            or str(
                numeric.get("preprocessing_contract_sha256") or ""
            ).strip().lower() != preprocessing_sha
            or str(numeric.get("backend") or "").strip().lower()
            != "native_full_deepx"
            or str(numeric.get("runtime_input_name") or "")
            != str(workload.get("runtime_input_name") or "")
            or list(numeric.get("runtime_input_shape") or [])
            != list(workload.get("runtime_input_shape") or [])
            or str(numeric.get("runtime_input_dtype") or "").strip().lower()
            != str(workload.get("runtime_input_dtype") or "").strip().lower()
            or str(numeric.get("runtime_input_layout") or "").strip().upper()
            != str(workload.get("runtime_input_layout") or "").strip().upper()
            or (
                workload.get("postprocess_required") is True
                and str(
                    workload.get(
                        "performance_completed_result_artifact_status"
                    ) or ""
                ) != "verified_exact"
            )
        ):
            return None, "full_command_contract_deepx_runtime_input_binding_invalid"
        for artifact_name in (
            "runner_artifact", "dxnn_artifact", "runtime_input_artifact",
            "source_model_artifact", "benchmark_set_manifest_artifact",
        ):
            key = str(workload.get(artifact_name) or "")
            artifact = artifacts.get(key)
            if not key or not isinstance(artifact, dict) or len(str(artifact.get("sha256") or "").strip()) != 64:
                return None, f"full_command_contract_deepx_{artifact_name}_missing"
    return contract, "hash_schema_identity_and_configuration_verified"


def _full_runtime_argv(
    contract: dict[str, Any], *, duration_s: float, frames: int,
    remote_tool_dir: str, authoritative_root: str, preflight_nonce: str,
    preflight_attestation: str, preflight_max_age_s: float,
    remote_contract_file: str = _REMOTE_CONTRACT_TOKEN,
) -> list[str]:
    runner = f"{remote_tool_dir.rstrip('/')}/{str(contract['runner']).lstrip('/')}"
    workload = contract.get("energy_workload")
    if not isinstance(workload, dict) or workload.get("available") is not True:
        raise ValueError("full_energy_hotloop_unavailable")
    if str(workload.get("kind") or "") not in _FULL_ENERGY_WORKLOAD_KINDS:
        raise ValueError("full_energy_hotloop_unavailable")
    return [
        str(contract["python_executable"]), "-u", runner,
        "--root", str(authoritative_root),
        "--energy-workload-only",
        "--energy-command-contract-file", str(remote_contract_file),
        "--frames", str(max(1, int(frames))),
        "--duration-s", str(float(duration_s)),
        "--expected-runner-sha256", str(contract["runner_sha256"]),
        "--expected-runner-path", runner,
        "--expected-runner-root", str(
            Path(remote_tool_dir) / "scripts"
        ),
        "--preflight-nonce", str(preflight_nonce),
        "--preflight-attestation", str(preflight_attestation),
        "--preflight-attestation-max-age-s", str(float(preflight_max_age_s)),
        "--out-dir", _FRESH_OUTPUT_TOKEN,
    ]


def _full_preflight_argv(
    contract: dict[str, Any], *, remote_tool_dir: str,
    authoritative_root: str,
    preflight_nonce: str, preflight_attestation: str,
    max_age_s: float,
    remote_contract_file: str = _REMOTE_CONTRACT_TOKEN,
) -> list[str]:
    runner = f"{remote_tool_dir.rstrip('/')}/{str(contract['runner']).lstrip('/')}"
    return [
        str(contract["python_executable"]), "-u", runner,
        "--root", str(authoritative_root),
        "--energy-preflight-only",
        "--energy-command-contract-file", str(remote_contract_file),
        "--expected-runner-sha256", str(contract["runner_sha256"]),
        "--expected-runner-path", runner,
        "--expected-runner-root", str(
            Path(remote_tool_dir) / "scripts"
        ),
        "--preflight-nonce", str(preflight_nonce),
        "--preflight-attestation-out", str(preflight_attestation),
        "--preflight-attestation-max-age-s", str(float(max_age_s)),
    ]


def _split_preflight_argv(
    contract: dict[str, Any], *, remote_tool_dir: str,
    preflight_nonce: str, preflight_attestation: str,
    max_age_s: float,
    remote_contract_file: str = _REMOTE_CONTRACT_TOKEN,
) -> list[str]:
    """Build the hash-heavy split preflight executed before acquisition."""
    helper_name = "native_split_energy_preflight.py"
    local_helper = _script_path(helper_name)
    if not local_helper.is_file():
        raise ValueError("split_energy_preflight_helper_missing")
    remote_helper = f"{remote_tool_dir.rstrip('/')}/scripts/{helper_name}"
    return [
        str(contract["python_executable"]), "-u", remote_helper,
        "--contract-json", str(remote_contract_file),
        "--expected-command-contract-sha256",
        str(contract["contract_sha256"]),
        "--expected-preflight-script-sha256", _sha256_file(local_helper),
        "--nonce", str(preflight_nonce),
        "--attestation", str(preflight_attestation),
        "--tool-root", str(remote_tool_dir),
        "--valid-for-s", str(float(max_age_s)),
    ]


def _shell_join_with_fresh_output(argv: list[str]) -> str:
    """Quote argv while retaining exactly one controlled shell variable."""
    rendered: list[str] = []
    for raw in argv:
        value = str(raw)
        if value == _FRESH_OUTPUT_TOKEN:
            rendered.append('"$fresh_output_root"')
        elif value.startswith(_FRESH_OUTPUT_TOKEN + "/"):
            suffix = value[len(_FRESH_OUTPUT_TOKEN) + 1:]
            rendered.append(f'"${{fresh_output_root}}/{suffix}"')
        else:
            rendered.append(shlex.quote(value))
    return " ".join(rendered)


def _process_local_runtime_environment(
    contract: Mapping[str, Any],
) -> dict[str, str]:
    """Return a sealed per-process environment for mixed Hailo/TRT replay.

    Hailo-8 detection deliberately executes with the archived system Python
    (which owns TensorRT/CUDA) and adds only the already-installed Hailo site
    directory through ``site.addsitedir`` in the child.  Never translate this
    binding into PYTHONPATH: prepending the Hailo environment would allow its
    packages to shadow the system runtime.
    """
    backend = str(contract.get("backend") or "").strip().lower()
    options = contract.get("runtime_options")
    options = dict(options) if isinstance(options, Mapping) else {}
    producer_impl = str(
        options.get("producer_impl") or ""
    ).strip().lower()
    if (
        backend != "hailo8_to_trt"
        or producer_impl != "hailo8_python_vstreams_fifo"
    ):
        return {}

    mixed = contract.get("mixed_runtime_contract")
    if not isinstance(mixed, Mapping):
        raise ValueError("hailo8_mixed_runtime_contract_missing")
    mixed = dict(mixed)
    expected_mode = (
        "system_tensorrt_with_process_local_hailo_sites"
    )
    if (
        mixed.get("status") != "ready"
        or mixed.get("source_closure_ok") is not True
        or str(mixed.get("runtime_mode") or "") != expected_mode
    ):
        raise ValueError("hailo8_mixed_runtime_contract_not_ready")
    if (
        str(options.get("mixed_runtime_site_policy") or "")
        != "site.addsitedir_after_system_defaults"
        or str(mixed.get("site_policy") or "")
        != "site.addsitedir_after_system_defaults"
    ):
        raise ValueError("hailo8_mixed_runtime_site_policy_invalid")

    raw_sites = options.get("process_local_extra_sites")
    mixed_sites = mixed.get("process_local_extra_sites")
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("hailo8_process_local_extra_sites_missing")
    sites: list[str] = []
    for raw in raw_sites:
        value = str(raw or "").strip()
        if (
            not value
            or not value.startswith("/")
            or any(char in value for char in ("\x00", "\n", "\r"))
        ):
            raise ValueError("hailo8_process_local_extra_site_invalid")
        if value in sites:
            raise ValueError("hailo8_process_local_extra_site_duplicate")
        sites.append(value)
    if not isinstance(mixed_sites, list) or [
        str(value or "").strip() for value in mixed_sites
    ] != sites:
        raise ValueError("hailo8_process_local_extra_sites_mismatch")

    identity = contract.get("interpreter_identity")
    if (
        not isinstance(identity, Mapping)
        or str(identity.get("runtime_mode") or "") != expected_mode
        or list(identity.get("process_local_extra_sites") or [])
        != sites
    ):
        raise ValueError("hailo8_mixed_runtime_interpreter_identity_mismatch")
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "SPLITPOINT_EXTRA_SITES": os.pathsep.join(sites),
    }


def _remote_environment_prefix(
    remote_env: str,
    process_environment: Mapping[str, str],
) -> str:
    commands: list[str] = []
    configured = str(remote_env or "").strip()
    if configured:
        commands.append(configured)
    for name in sorted(process_environment):
        if name not in {
            "PYTHONDONTWRITEBYTECODE",
            "SPLITPOINT_EXTRA_SITES",
        }:
            raise ValueError(
                f"unsupported_process_local_environment_variable:{name}"
            )
        commands.append(
            f"export {name}="
            f"{shlex.quote(str(process_environment[name]))}"
        )
    return (" && ".join(commands) + " && ") if commands else ""


def _fresh_report_relative_path(backend: str) -> str:
    return {
        "hailo8_to_trt": "native_fifo_results.json",
        "hailo10h_to_trt": "hailo10_native_fifo_e2e_results.json",
        "deepx_to_trt": "deepx_native_fifo_e2e_results.json",
    }.get(str(backend or "").strip().lower(), "native_full_energy_hotloop.json")


def _load_verified_json(path_value: str, declared_sha256: str) -> tuple[dict[str, Any], str]:
    path_text = str(path_value or "").strip()
    declared = _normalize_sha256(declared_sha256)
    if not path_text or len(declared) != 64:
        return {}, "path_or_sha256_missing"
    path = Path(path_text).expanduser()
    if not path.is_file():
        return {}, "file_missing"
    if _sha256_file(path) != declared:
        return {}, "sha256_mismatch"
    try:
        payload = _strict_json_value(path)
    except Exception:
        return {}, "json_invalid"
    return (dict(payload), "verified") if isinstance(payload, dict) else ({}, "json_not_mapping")


def _verified_contract_manifest(path_value: str, declared_sha256: str) -> tuple[dict[str, Any], str]:
    payload, status = _load_verified_json(path_value, declared_sha256)
    if status != "verified":
        return {}, status
    if payload.get("schema") != "onnx-splitpoint/pipeline-contract-manifest":
        return {}, "contract_manifest_schema_mismatch"
    rows = [dict(row) for row in list(payload.get("contracts") or []) if isinstance(row, dict)]
    if not rows or _normalize_sha256(payload.get("contract_set_sha256")) != _stable_json_sha256(rows):
        return {}, "contract_set_hash_mismatch"
    manifest_path = Path(str(path_value)).expanduser().resolve()
    kinds = {str(row.get("kind") or "").strip().lower() for row in rows}
    if not {"preprocessing", "decoder", "nms"}.issubset(kinds):
        return {}, "required_contract_kind_coverage_missing"
    for row in rows:
        source = Path(str(row.get("path") or "")).expanduser()
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        expected = _normalize_sha256(row.get("sha256"))
        if not source.is_file() or len(expected) != 64 or _sha256_file(source) != expected or row.get("locked") is not True:
            return {}, "contract_source_unverified_or_unlocked"
    return payload, "verified"


def _contract_fields_for_task(manifest: dict[str, Any], task: str) -> tuple[dict[str, str], str]:
    task = str(task or "").strip().lower()
    if task not in {"classification", "detection"}:
        return {}, "task_missing"
    rows = [
        dict(row) for row in list(manifest.get("contracts") or [])
        if isinstance(row, dict) and str(row.get("task") or "all").strip().lower() in {"all", task}
    ]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind") or "").strip().lower(), []).append(row)
    required = ["preprocessing"] + (["decoder", "nms"] if task == "detection" else [])
    if any(len(by_kind.get(kind) or []) != 1 for kind in required):
        return {}, "task_contract_missing_or_ambiguous"
    fields = {
        "contract_hash": str(manifest.get("contract_set_sha256") or ""),
        "preprocessing_hash": str(by_kind["preprocessing"][0].get("sha256") or ""),
        "decoder_hash": str((by_kind.get("decoder") or [{}])[0].get("sha256") or "") if len(by_kind.get("decoder") or []) == 1 else "",
        "nms_hash": str((by_kind.get("nms") or [{}])[0].get("sha256") or "") if len(by_kind.get("nms") or []) == 1 else "",
    }
    return fields, "verified"


def _verified_model_hashes(path_value: str, declared_sha256: str) -> tuple[dict[str, str], str]:
    payload, status = _load_verified_json(path_value, declared_sha256)
    if status != "verified":
        return {}, status
    if payload.get("schema") != "onnx-splitpoint/native-energy-model-hash-map":
        return {}, "model_hash_map_schema_mismatch"
    map_path = Path(str(path_value)).expanduser().resolve()
    out: dict[str, str] = {}
    for row in list(payload.get("rows") or []):
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or row.get("model_id") or "").strip()
        digest = _normalize_sha256(row.get("model_sha256"))
        model_path = Path(str(row.get("model_path") or "")).expanduser()
        manifest_path = Path(str(row.get("model_manifest") or "")).expanduser()
        if not model_path.is_absolute():
            model_path = (map_path.parent / model_path).resolve()
        if not manifest_path.is_absolute():
            manifest_path = (map_path.parent / manifest_path).resolve()
        manifest_digest = _normalize_sha256(row.get("model_manifest_sha256"))
        if not (
            model
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            and model_path.is_file()
            and _sha256_file(model_path) == digest
            and len(manifest_digest) == 64
            and manifest_path.is_file()
            and _sha256_file(manifest_path) == manifest_digest
        ):
            return {}, "model_hash_source_unverified"
        try:
            model_manifest = _strict_json_value(manifest_path)
        except Exception:
            return {}, "model_manifest_invalid"
        if not isinstance(model_manifest, dict) or str(model_manifest.get("model_id") or "").strip() != model:
            return {}, "model_manifest_identity_mismatch"
        manifest_model_path = Path(str(model_manifest.get("resolved_path") or "")).expanduser()
        if not manifest_model_path.is_absolute():
            manifest_model_path = manifest_model_path.resolve()
        if manifest_model_path != model_path.resolve():
            return {}, "model_manifest_path_mismatch"
        old = out.get(model)
        if old and old != digest:
            return {}, "model_hash_identity_ambiguous"
        out[model] = digest
    return (out, "verified") if out else ({}, "no_verified_model_hashes")


def _verified_detection_exclusions(
    path: Any,
    expected_sha256: Any,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], str, str]:
    token = str(path or "").strip()
    if not token:
        return {}, "not_provided", ""
    source = Path(token).expanduser().resolve()
    wanted = _normalize_sha256(expected_sha256)
    if (
        not source.is_file()
        or not wanted
        or _sha256_file(source) != wanted
        or verify_prospective_detection_exclusion_set is None
    ):
        return {}, "file_or_sha256_invalid", ""
    try:
        raw = _strict_json_value(source)
        verified = verify_prospective_detection_exclusion_set(raw)
    except Exception as exc:
        return {}, f"contract_invalid:{type(exc).__name__}", ""
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_entry in list(verified.get("entries") or []):
        entry = dict(raw_entry)
        key = (
            str(entry.get("setup_id") or ""),
            str(entry.get("backend") or ""),
            str(entry.get("model_id") or ""),
        )
        if not all(key) or key in entries:
            return {}, "identity_ambiguous", ""
        entries[key] = entry
    return (
        entries,
        "verified",
        str(verified.get("contract_sha256") or ""),
    )


def _prospective_detection_exclusion(
    row: Mapping[str, Any],
    exclusions: Mapping[
        tuple[str, str, str], Mapping[str, Any]
    ],
) -> dict[str, Any] | None:
    backend = str(row.get("backend") or "").strip().lower()
    model = str(
        row.get("model") or row.get("model_id") or ""
    ).strip().lower()
    setup = str(row.get("setup_id") or "").strip()
    aliases = {
        "hailo10_to_trt": "hailo10h_to_trt",
        "hailo10h_to_tensorrt": "hailo10h_to_trt",
    }
    entry = exclusions.get(
        (setup, aliases.get(backend, backend), model)
    )
    return dict(entry) if isinstance(entry, Mapping) else None


from onnx_splitpoint_tool.energy.task_budget import add_campaign_budget_arguments, campaign_budget_forward_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_campaign_budget_arguments(parser)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--validation-summary", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--hardware-setups-file",
        default="",
        help="Exact hardware registry propagated to every measurement command.",
    )
    parser.add_argument("--remote-tool-dir", default="/home/nx/ONNX-Splitpoint-Tool")
    parser.add_argument("--remote-root", default="/home/nx/native_fifo_evalsets/complete_set_20260626_194014")
    parser.add_argument("--hailo8-ssh", default="")
    parser.add_argument("--hailo10-ssh", default="")
    parser.add_argument("--deepx-ssh", default="")
    parser.add_argument("--hailo8-env", default="")
    parser.add_argument("--hailo10-env", default="")
    parser.add_argument("--deepx-env", default="")
    parser.add_argument("--engine-build-python", default="auto")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=0, help="Explicit fallback only; normally FPS*duration")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--runs", type=int, default=1, help="Independent u.RECS repetitions per energy row.")
    parser.add_argument("--require-runtime-work-units", action="store_true")
    parser.add_argument("--require-command-window-alignment", action="store_true")
    parser.add_argument("--physical-scope", default="")
    parser.add_argument("--window-label", default="command")
    parser.add_argument(
        "--window-method-ab-json",
        default="",
        help="Frozen top-level energy.window_method_ab object for every generated measurement command.",
    )
    parser.add_argument("--calibration-manifest", default="")
    parser.add_argument("--calibration-sha256", default="")
    parser.add_argument("--pipeline-contract-manifest", default="")
    parser.add_argument("--pipeline-contract-sha256", default="")
    parser.add_argument("--model-hash-map", default="")
    parser.add_argument("--model-hash-map-sha256", default="")
    parser.add_argument(
        "--detection-claim-exclusions",
        default=(
            str(_DEFAULT_DETECTION_EXCLUSIONS)
            if _DEFAULT_DETECTION_EXCLUSIONS.is_file() else ""
        ),
        help=(
            "Setup-local sealed prospective detection claim exclusions."
        ),
    )
    parser.add_argument(
        "--detection-claim-exclusions-sha256",
        default=(
            _DEFAULT_DETECTION_EXCLUSIONS_SHA256
            if _DEFAULT_DETECTION_EXCLUSIONS.is_file() else ""
        ),
    )
    parser.add_argument("--allow-unpaired", action="store_true", help="Diagnostic mode: retain rows without both setup-local Full baselines.")
    parser.add_argument(
        "--measure-all-runtime-successful",
        action="store_true",
        help=(
            "Compatibility alias for the invariant Native Energy admission: "
            "every runtime-successful row with a technically replayable "
            "command, preflight and (for Split) one static Part-2 input."
        ),
    )
    parser.add_argument(
        "--smoke-diagnostic", action="store_true",
        help=(
            "Measure technically admitted Native rows for wiring diagnostics "
            "while forcing all energy/claim eligibility false."
        ),
    )
    parser.add_argument(
        "--screening-window-probe",
        action="store_true",
        help=(
            "Build exactly one pairing-independent, non-claimable native target "
            "for the marker-v2 versus historical-window screening probe."
        ),
    )
    parser.add_argument(
        "--screening-energy",
        action="store_true",
        help=(
            "Measure technically admitted Development/Screening rows. Quality, "
            "Semantics and pairing remain post-hoc annotations, and every "
            "generated artifact is explicitly non-claimable."
        ),
    )
    parser.add_argument(
        "--final-all-split-energy",
        action="store_true",
        help=(
            "Build a technically verified Final observation plan for every "
            "Split row, including exactly bound negative accuracy results. "
            "Negative observations remain non-claimable."
        ),
    )
    ns = parser.parse_args()
    hardware_registry_binding: dict[str, str] = {}
    if str(ns.hardware_setups_file or "").strip():
        registry_input = Path(ns.hardware_setups_file).expanduser()
        if registry_input.is_symlink():
            parser.error("--hardware-setups-file must not be a symlink")
        try:
            registry_path = registry_input.resolve(strict=True)
        except OSError as exc:
            parser.error(f"--hardware-setups-file is not readable: {exc}")
        if not registry_path.is_file():
            parser.error("--hardware-setups-file must be a regular file")
        ns.hardware_setups_file = str(registry_path)
        hardware_registry_binding = {
            "hardware_setups_file": str(registry_path),
            "hardware_setups_file_sha256": _sha256_file(registry_path),
        }
    measure_all_runtime_successful_requested = bool(
        ns.measure_all_runtime_successful
    )
    if (
        measure_all_runtime_successful_requested
        and ns.screening_window_probe
    ):
        parser.error(
            "--measure-all-runtime-successful cannot be combined with "
            "--screening-window-probe"
        )
    if sum(bool(value) for value in (
        ns.screening_energy, ns.screening_window_probe, ns.smoke_diagnostic,
    )) > 1:
        parser.error(
            "--screening-energy, --screening-window-probe and "
            "--smoke-diagnostic are mutually exclusive"
        )
    if ns.final_all_split_energy and (
        ns.screening_energy
        or ns.screening_window_probe
        or ns.smoke_diagnostic
    ):
        parser.error(
            "--final-all-split-energy cannot be combined with a diagnostic "
            "or screening tier"
        )
    if ns.screening_energy and (
        ns.require_runtime_work_units
        or ns.require_command_window_alignment
    ):
        parser.error(
            "--screening-energy cannot be combined with Final-only runtime "
            "work-unit or command-window requirements"
        )
    # P0.3: this is the Native Energy planner's admission invariant, not an
    # opt-in campaign relaxation.  Keep the historical CLI switch as a
    # compatibility alias.  The dedicated one-row window-method probe remains
    # a separate diagnostic planner schema.
    ns.measure_all_runtime_successful = bool(
        not ns.screening_window_probe
    )
    plan_attempt_id = uuid.uuid4().hex
    window_ab: dict[str, Any] = {}
    window_ab_json = str(ns.window_method_ab_json or "").strip()
    if window_ab_json:
        try:
            parsed_window_ab = _strict_json_text(
                window_ab_json, label="--window-method-ab-json",
            )
        except Exception as exc:
            parser.error(f"--window-method-ab-json is not valid JSON: {exc}")
        if not isinstance(parsed_window_ab, dict):
            parser.error("--window-method-ab-json must encode an object")
        window_ab = (
            resolve_energy_ab_config(parsed_window_ab)
            if resolve_energy_ab_config is not None
            else dict(parsed_window_ab)
        )
        if window_ab.get("enabled") and window_ab.get("valid") is False:
            parser.error(
                "invalid frozen energy A/B contract: "
                + ", ".join(str(item) for item in window_ab.get("validation_errors") or [])
            )
        window_ab_json = json.dumps(window_ab, sort_keys=True, separators=(",", ":"))
    energy_profile_requested_runs_per_row = max(
        1, int(ns.runs or 1)
    )
    energy_effective_runs_per_row = (
        energy_profile_requested_runs_per_row
    )
    if window_ab.get("enabled"):
        energy_effective_runs_per_row = max(
            energy_effective_runs_per_row,
            max(1, int(window_ab.get("smoke_repeats") or 3)),
        )
    # Compatibility alias: historically this field was the number passed to
    # the collector.  Keep that meaning and archive the profile request
    # separately so a local A/B minimum cannot remain invisible.
    energy_runs_per_row = energy_effective_runs_per_row
    energy_repeat_expansion_applied = bool(
        energy_effective_runs_per_row
        != energy_profile_requested_runs_per_row
    )
    energy_repeat_expansion_reason = (
        "frozen_window_method_ab_minimum"
        if energy_repeat_expansion_applied else "none"
    )
    contract_manifest, contract_manifest_status = _verified_contract_manifest(
        ns.pipeline_contract_manifest, ns.pipeline_contract_sha256
    )
    model_hashes, model_hash_map_status = _verified_model_hashes(
        ns.model_hash_map, ns.model_hash_map_sha256
    )
    (
        detection_exclusions,
        detection_exclusions_status,
        detection_exclusions_contract_sha256,
    ) = _verified_detection_exclusions(
        ns.detection_claim_exclusions,
        ns.detection_claim_exclusions_sha256,
    )
    detection_claim_annotations_invalid = bool(
        ns.detection_claim_exclusions
        and detection_exclusions_status != "verified"
    )
    if detection_claim_annotations_invalid:
        # This contract can only remove scientific claim eligibility.  A bad
        # copy must therefore clamp detection claims fail-closed, never abort
        # or alter the technical measurement matrix.
        detection_exclusions = {}
        detection_exclusions_contract_sha256 = ""
    strict_energy_claim = bool(
        not ns.screening_energy
        and not ns.screening_window_probe
        and (
            ns.require_runtime_work_units
            or ns.require_command_window_alignment
        )
    )
    # Claim eligibility is an annotation only.  It is still calculated for the
    # result consumer, but it is never consulted for technical plan membership.
    require_claim_admission = bool(
        False
    )
    screening_only = bool(ns.screening_energy or ns.screening_window_probe)

    duration_s = float(ns.duration_s or 0.0)
    if duration_s <= 0:
        duration_s = 60.0
        try:
            if load_energy_defaults is not None:
                defaults = load_energy_defaults()
                duration_s = float(getattr(defaults, "native_measurement_duration_s", getattr(defaults, "measurement_duration_s", 60.0)) or 60.0)
        except Exception:
            duration_s = 60.0
    duration_s = max(1.0, duration_s)

    summary_path = Path(ns.summary).expanduser().resolve()
    expected_matrix, expected_matrix_status = _expected_matrix_contract(
        summary_path
    )
    energy_matrix_expected_count = int(
        expected_matrix.get("expected_row_count") or 0
    )
    if ns.final_all_split_energy and expected_matrix_status != "verified":
        parser.error(
            "--final-all-split-energy requires a verified run-local "
            "native_expected_matrix.json"
        )
    split_quality_authority = _energy_split_quality_authority(summary_path)
    historical_diagnostic_only = _historical_energy_diagnostic_only(
        split_quality_authority
    )
    diagnostic_claim_exclusion_reason = (
        "standalone_unmanaged_native_energy_diagnostic_only"
        if split_quality_authority.get(
            "standalone_unmanaged_diagnostic_only"
        ) is True else
        "historical_native_split_without_quality_first_authority"
    )
    validation_path = _discover_validation_summary(summary_path, ns.validation_summary)
    validations, validation_summary_status = _validation_map_with_status(
        validation_path
    )
    all_source_rows = _load_rows(summary_path)
    source_rows = [
        row for row in all_source_rows
        if _runtime_successful_for_energy(row)
    ]
    admitted: list[tuple[dict[str, Any], dict[str, Any] | None, str, str]] = []
    excluded: list[dict[str, Any]] = [
        {
            "backend": str(row.get("backend") or ""),
            "model": row.get("model") or row.get("model_id"),
            "case": row.get("case") or row.get("case_id") or "",
            "setup_id": row.get("setup_id"),
            "comparison_backend": row.get("comparison_backend"),
            **{field: row[field] for field in (
                "execution_mode", "planned_native_identity", "failure_stage",
                "primary_failure_reason", "upstream_evidence_path", "identity_conflicts",
                "child_observation", "source_root", "report", "error", "status_detail",
                "repetition_count_requested", "repetition_count_attempted", "repetition_count_valid",
                "status", "prerequisite_status", "repetition_status", "aggregation_applied",
            ) if field in row},
            "precision": row.get("precision"),
            "reason": "native_performance_row_not_ok",
            "source_failure_reason": str(
                row.get("primary_repetition_failure_reason")
                or row.get("primary_failure_reason")
                or row.get("semantic_dump_failure_reason")
                or row.get("failure_reason")
                or row.get("reason")
                or row.get("error")
                or "native_performance_row_not_ok"
            ),
        }
        for row in all_source_rows
        if not _runtime_successful_for_energy(row)
    ]
    invalid_ledger_identity_count = 0

    def _ledger_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
        nonlocal invalid_ledger_identity_count
        backend = str(row.get("backend") or "")
        defaults = _backend_defaults(backend, dict(row))
        normalized = dict(row)
        if not str(normalized.get("setup_id") or "").strip():
            normalized["setup_id"] = defaults["setup"]
        comparison = str(
            normalized.get("comparison_backend") or ""
        ).strip()
        if not comparison:
            if not backend.startswith("native_full_"):
                comparison = _comparison_context(backend)
            elif backend in {"native_full_hailo8"}:
                comparison = "hailo8"
            elif backend in {
                "native_full_hailo10", "native_full_hailo10h",
            }:
                comparison = "hailo10h"
            elif backend == "native_full_deepx":
                comparison = "deepx"
            elif backend == "native_full_tensorrt":
                setup = defaults["setup"].lower()
                if "hailo8" in setup:
                    comparison = "hailo8"
                elif "hailo10" in setup:
                    comparison = "hailo10h"
                elif "deepx" in setup:
                    comparison = "deepx"
        normalized["comparison_backend"] = comparison
        identity = native_performance_identity(normalized)
        if identity is not None:
            return identity
        invalid_ledger_identity_count += 1
        # Invalid identities remain visible and can never accidentally satisfy
        # expected-matrix coverage.  Precision is diagnostic here only; valid
        # membership deliberately excludes it so planned and imported schemas
        # share one denominator.
        return (
            "__invalid_native_performance_identity__",
            backend.strip().lower(),
            str(row.get("model") or row.get("model_id") or "").strip().lower(),
            str(row.get("case") or row.get("case_id") or "").strip().lower(),
            defaults["setup"].strip().lower(),
            str(row.get("comparison_backend") or "").strip().lower(),
            str(row.get("precision") or "").strip().lower(),
        )

    represented_identities = {
        _ledger_identity(row) for row in all_source_rows
    }
    for expected_failure in [
        *list(expected_matrix.get("failed_expected_rows") or []),
        *list(expected_matrix.get("missing_expected_rows") or []),
    ]:
        if not isinstance(expected_failure, Mapping):
            continue
        identity = _ledger_identity(expected_failure)
        if identity in represented_identities:
            continue
        represented_identities.add(identity)
        canonical_membership = (
            identity
            if len(identity) == 6
            and identity[0] in {
                "native_split", "native_full_baseline",
            }
            else None
        )
        excluded.append({
            "execution_mode": (
                canonical_membership[0]
                if canonical_membership is not None else
                expected_failure.get("execution_mode")
            ),
            "backend": (
                canonical_membership[1]
                if canonical_membership is not None else
                str(
                    expected_failure.get("backend")
                    or "missing_expected_matrix_row"
                )
            ),
            "model": (
                canonical_membership[2]
                if canonical_membership is not None else
                expected_failure.get("model")
                or expected_failure.get("model_id")
            ),
            "case": (
                canonical_membership[3]
                if canonical_membership is not None else
                expected_failure.get("case")
                or expected_failure.get("case_id")
                or ""
            ),
            "setup_id": (
                canonical_membership[4]
                if canonical_membership is not None else
                expected_failure.get("setup_id")
            ),
            "comparison_backend": (
                canonical_membership[5]
                if canonical_membership is not None else
                expected_failure.get("comparison_backend")
            ),
            "precision": expected_failure.get("precision"),
            "reason": str(
                expected_failure.get("failure_reason")
                or expected_failure.get("status_detail")
                or "expected_matrix_row_missing_from_native_summary"
            ),
            "source_failure_reason": str(
                expected_failure.get("status_detail")
                or expected_failure.get("failure_reason")
                or ""
            ),
            "expected_matrix_failure": True,
        })
    for row in source_rows:
        backend = str(row.get("backend") or "")
        defaults = _backend_defaults(backend, row)
        identity = {
            "backend": backend,
            "model": row.get("model") or row.get("model_id"),
            "case": (
                row.get("case")
                or row.get("case_id")
                or ("full" if backend.lower().startswith("native_full_") else "")
            ),
            "precision": row.get("precision"),
            "setup_id": row.get("setup_id"),
            "comparison_backend": row.get("comparison_backend"),
            "model_sha256": (
                row.get("model_sha256") or row.get("source_model_sha256")
                or row.get("source_onnx_sha256")
            ),
            "remote_root": ns.remote_root,
            "remote_tool_dir": ns.remote_tool_dir,
        }
        prospective_exclusion = _prospective_detection_exclusion(
            row, detection_exclusions,
        )
        if backend.lower().startswith("native_full_"):
            command_contract, command_contract_status = (
                _verify_full_command_contract(
                    row.get("full_command_contract"),
                    expected_identity=identity,
                )
            )
        elif verify_native_energy_command_contract is None:
            command_contract = None
            command_contract_status = (
                "native_command_contract_helper_unavailable"
            )
        else:
            command_contract, command_contract_status = (
                verify_native_energy_command_contract(
                    row.get("native_command_contract"),
                    expected_identity=identity,
                )
            )
        # Command constructibility is a technical admission axis and therefore
        # precedes every Quality/Semantics annotation for both Split and Full.
        if command_contract is None:
            excluded.append({
                "backend": backend,
                "model": row.get("model") or row.get("model_id"),
                "case": row.get("case") or row.get("case_id") or "",
                "setup_id": defaults["setup"],
                "comparison_backend": row.get(
                    "comparison_backend"
                ),
                "precision": row.get("precision"),
                "reason": "successful_command_contract_missing_or_invalid",
                "command_contract_status": command_contract_status,
            })
            continue
        if (
            row.get("completed_task_completion_mode") == "native_three_stage_fast_oracle_outside_timing"
            and (command_contract.get("runtime_options") or {}).get("completion_runtime_mode")
            != "fast_oracle_outside_timing"
        ):
            # An old performance attestation cannot select a new energy
            # algorithm through an unbound default. Re-measure with the current
            # command contract instead of spending three doomed repeats.
            excluded.append({
                **{field: identity.get(field) for field in ("backend", "model", "case", "precision", "setup_id", "comparison_backend")},
                "reason": "fast_energy_command_runtime_mode_unbound",
                "failure_stage": "energy_command_preflight", "repetition_count_attempted": 0,
            })
            continue
        split_backend = (
            bool(is_native_split_backend(backend))
            if is_native_split_backend is not None else
            backend.strip().lower() in {
                "hailo8_to_trt", "hailo10h_to_trt", "deepx_to_trt",
            }
        )
        split_has_valid_part2_input = False
        split_part2_input_status = "native_full_not_applicable"
        effective_part2_input_count: int | None = None
        if split_backend:
            (
                split_has_valid_part2_input,
                split_part2_input_status,
                effective_part2_input_count,
            ) = _split_part2_input_admission(row, command_contract)
            if not split_has_valid_part2_input:
                excluded.append({
                    "backend": backend,
                    "model": row.get("model") or row.get("model_id"),
                    "case": row.get("case") or row.get("case_id") or "",
                    "setup_id": defaults["setup"],
                    "comparison_backend": row.get(
                        "comparison_backend"
                    ),
                    "precision": row.get("precision"),
                    "part2_input_count": row.get("part2_input_count"),
                    "effective_part2_input_count": (
                        effective_part2_input_count
                    ),
                    "reason": split_part2_input_status,
                })
                continue
        current_quality_first_split = bool(
            split_backend
            and split_quality_authority.get("valid") is True
            and str(split_quality_authority.get("mode") or "") == "required"
        )
        if ns.measure_all_runtime_successful:
            validation = _validation_for_row(
                row,
                validations,
                exact_identity_required=current_quality_first_split,
            )
            accepted = True
            reason = "native_runtime_success_measurement_admitted"
        elif ns.screening_window_probe:
            # This path validates only window extraction.  A successful native
            # runtime command is sufficient and its result must never become an
            # energy/performance claim.  Native semantic/full-pair gates remain
            # unchanged for the scientific Native Energy plan below.
            accepted, reason = True, "screening_probe_runtime_success"
            validation = _validation_for_row(row, validations)
        elif ns.smoke_diagnostic:
            validation = _validation_for_row(
                row, validations,
                exact_identity_required=True,
            )
            if split_backend:
                if not current_quality_first_split:
                    accepted, reason = (
                        False,
                        "smoke_native_split_without_valid_quality_first_authority",
                    )
                else:
                    accepted, reason = _current_quality_first_validation_join(
                        row, validation, require_performance_claim=False,
                    )
            else:
                accepted, reason, validation = (
                    _smoke_diagnostic_technical_decision(
                        row, validations,
                    )
                )
                if accepted:
                    accepted = _native_quality_bridge_verified(
                        validation,
                        str(
                            (validation or {}).get("task")
                            or row.get("task")
                            or ""
                        ),
                        row,
                        exact_identity_required=False,
                        positive_result_required=False,
                    )
                    reason = (
                        "smoke_diagnostic_exact_quality_binding_verified"
                        if accepted else
                        "smoke_diagnostic_precision_quality_binding_not_verified"
                    )
        else:
            accepted, reason, validation = _semantic_decision(
                row, validations,
                require_claim=require_claim_admission,
                exact_identity_required=current_quality_first_split,
            )
            if accepted and current_quality_first_split:
                accepted, reason = _current_quality_first_validation_join(
                    row, validation,
                    require_performance_claim=require_claim_admission,
                )
            if accepted and ns.final_all_split_energy:
                bound = _native_quality_bridge_verified(
                    validation,
                    str(
                        (validation or {}).get("task")
                        or row.get("task")
                        or ""
                    ),
                    row,
                    exact_identity_required=current_quality_first_split,
                    positive_result_required=False,
                )
                if not bound:
                    accepted = False
                    reason = (
                        "final_all_split_precision_quality_binding_not_verified"
                    )
        if (
            prospective_exclusion is not None
            and not ns.final_all_split_energy
            and not ns.measure_all_runtime_successful
        ):
            accepted = False
            reason = str(prospective_exclusion["reason"])
        if not accepted:
            excluded_row = {
                "backend": backend,
                "model": row.get("model"),
                "case": row.get("case"),
                "setup_id": row.get("setup_id"),
                "precision": row.get("precision"),
                "reason": reason,
            }
            if prospective_exclusion is not None:
                excluded_row.update({
                    "prospective_detection_claim_exclusion": True,
                    "detection_diagnostic_sha256": str(
                        prospective_exclusion[
                            "diagnostic_sha256"
                        ]
                    ),
                    "detection_claim_exclusion_entry_sha256": str(
                        prospective_exclusion["entry_sha256"]
                    ),
                })
            excluded.append(excluded_row)
            continue
        annotation_task = str(
            (validation or {}).get("task")
            or row.get("task")
            or row.get("benchmark_task")
            or ""
        ).strip().lower()
        detection_claim_annotation_invalid_for_row = bool(
            detection_claim_annotations_invalid
            and annotation_task == "detection"
        )
        if (
            not backend.lower().startswith("native_full_")
            and not str(row.get("precision") or "").strip()
            and not ns.measure_all_runtime_successful
        ):
            excluded.append({
                "backend": backend,
                "model": row.get("model"),
                "case": row.get("case"),
                "setup_id": defaults["setup"],
                "comparison_backend": row.get(
                    "comparison_backend"
                ),
                "precision": "",
                "reason": "precision_missing",
            })
            continue
        admitted_row = dict(row)
        if (
            split_backend
            and ns.measure_all_runtime_successful
            and not str(admitted_row.get("precision") or "").strip()
        ):
            # Summary precision is a reporting annotation.  The already
            # verified executable contract is the technical authority for the
            # Split boundary precision used to render the measurement row.
            contract_precision = str(
                command_contract.get("precision") or ""
            ).strip().lower()
            if contract_precision:
                admitted_row["precision"] = contract_precision
                admitted_row["precision_annotation_status"] = (
                    "derived_from_verified_energy_command_contract"
                )
            else:
                admitted_row["precision_annotation_status"] = (
                    "unavailable_downstream_annotation"
                )
        admitted_row["_split_has_valid_part2_input"] = bool(
            split_has_valid_part2_input
        )
        admitted_row["_split_part2_input_status"] = str(
            split_part2_input_status
        )
        admitted_row["_effective_part2_input_count"] = (
            effective_part2_input_count
        )
        if detection_claim_annotation_invalid_for_row:
            admitted_row.update({
                "detection_claim_annotation_invalid": True,
                "detection_claim_annotation_status": (
                    detection_exclusions_status
                ),
            })
        if prospective_exclusion is not None:
            admitted_row.update({
                "prospective_detection_claim_exclusion": True,
                "prospective_detection_claim_exclusion_reason": str(
                    prospective_exclusion["reason"]
                ),
                "detection_diagnostic_sha256": str(
                    prospective_exclusion["diagnostic_sha256"]
                ),
                "detection_claim_exclusion_entry_sha256": str(
                    prospective_exclusion["entry_sha256"]
                ),
                "detection_claim_exclusions_contract_sha256": (
                    detection_exclusions_contract_sha256
                ),
            })
        if split_backend:
            if split_quality_authority.get(
                "standalone_unmanaged_diagnostic_only"
            ) is True:
                admitted_row["native_split_quality_authority"] = dict(
                    split_quality_authority
                )
                admitted_row["native_split_quality_authority_valid"] = False
                admitted_row["native_split_quality_authority_mode"] = "invalid"
                admitted_row["standalone_unmanaged_diagnostic_only"] = True
            elif apply_native_split_quality_authority is None:
                admitted_row["native_split_quality_required"] = True
                admitted_row["native_split_quality_binding_required"] = True
                admitted_row["native_split_quality_authority"] = dict(
                    split_quality_authority
                )
                admitted_row["native_split_quality_authority_valid"] = False
            else:
                try:
                    apply_native_split_quality_authority(
                        admitted_row, split_quality_authority,
                    )
                except Exception as exc:
                    if not ns.measure_all_runtime_successful:
                        raise
                    admitted_row.update({
                        "native_split_quality_required": False,
                        "native_split_quality_binding_required": False,
                        "native_split_quality_authority_valid": False,
                        "native_split_quality_authority_mode": "invalid",
                        "native_split_quality_annotation_status": (
                            "unavailable:"
                            f"{type(exc).__name__}"
                        ),
                    })
        runtime_split_binding_downgraded = False
        if not backend.lower().startswith("native_full_"):
            try:
                split_quality_evidence, split_quality_status = (
                    _split_quality_energy_evidence(
                        admitted_row,
                        command_contract,
                        split_quality_authority,
                    )
                )
            except Exception as exc:
                if not ns.measure_all_runtime_successful:
                    raise
                split_quality_evidence = None
                split_quality_status = (
                    "quality_annotation_unavailable:"
                    f"{type(exc).__name__}"
                )
        else:
            split_quality_evidence, split_quality_status = ({
                "native_split_quality_required": False,
                "native_split_energy_binding_valid": True,
                "native_split_energy_binding_status": "native_full_not_applicable",
            }, "native_full_not_applicable")
        if (
            ns.measure_all_runtime_successful
            and split_backend
            and (
                split_quality_evidence is None
                or split_quality_evidence.get(
                    "native_split_quality_required"
                ) is not True
            )
        ):
            split_quality_evidence = (
                _runtime_observation_split_evidence(
                    str(
                        command_contract.get(
                            "contract_sha256"
                        ) or ""
                    ),
                    reason=split_quality_status,
                )
            )
            split_quality_status = (
                "runtime_observation_split_quality_not_claimable"
            )
            runtime_split_binding_downgraded = True
        elif split_quality_evidence is None:
            excluded.append({
                "backend": backend,
                "model": row.get("model") or row.get("model_id"),
                "case": row.get("case") or row.get("case_id") or "",
                "setup_id": defaults["setup"],
                "comparison_backend": row.get(
                    "comparison_backend"
                ),
                "precision": row.get("precision"),
                "reason": "native_split_quality_energy_binding_invalid",
                "native_split_quality_energy_binding_status": split_quality_status,
                "command_contract_sha256": str(command_contract.get("contract_sha256") or ""),
            })
            continue
        admitted_row["_verified_energy_command_contract"] = command_contract
        admitted_row["_energy_command_contract_status"] = command_contract_status
        admitted_row["_split_quality_energy_evidence"] = split_quality_evidence
        try:
            endpoint_identity = _energy_endpoint_identity(
                admitted_row, validation, command_contract,
            )
        except Exception as exc:
            if not ns.measure_all_runtime_successful:
                raise
            endpoint_identity = _unavailable_energy_endpoint_identity(
                "downstream_endpoint_annotation_unavailable:"
                f"{type(exc).__name__}"
            )
        admitted_row.update(endpoint_identity)
        try:
            (
                preparing_quality_admission,
                preparing_quality_status,
            ) = _prepair_energy_quality_admission(
                admitted_row,
                validation,
                setup=defaults["setup"],
                command_contract_sha256=str(
                    command_contract.get("contract_sha256") or ""
                ),
                screening_only=screening_only,
                smoke_diagnostic=bool(ns.smoke_diagnostic),
                historical_diagnostic_only=historical_diagnostic_only,
                force_diagnostic_only=bool(
                    prospective_exclusion is not None
                    or detection_claim_annotation_invalid_for_row
                ),
                window_method_validation_probe=bool(
                    ns.screening_window_probe
                ),
                runtime_observation_allowed=bool(
                    ns.measure_all_runtime_successful
                ),
            )
        except Exception as exc:
            if not ns.measure_all_runtime_successful:
                raise
            preparing_quality_admission = None
            preparing_quality_status = (
                "quality_annotation_unavailable:"
                f"{type(exc).__name__}"
            )
        if preparing_quality_admission is None:
            if ns.measure_all_runtime_successful:
                preparing_quality_admission = (
                    _fallback_native_runtime_observation_admission(
                        admitted_row,
                        setup=defaults["setup"],
                        reason=preparing_quality_status,
                    )
                )
                preparing_quality_status = (
                    "fallback_native_runtime_observation_annotation_verified"
                )
            else:
                excluded.append({
                    "backend": backend,
                    "model": row.get("model") or row.get("model_id"),
                    "case": row.get("case") or row.get("case_id") or "",
                    "setup_id": defaults["setup"],
                    "comparison_backend": row.get(
                        "comparison_backend"
                    ),
                    "precision": row.get("precision"),
                    "reason": preparing_quality_status,
                    "energy_quality_admission_stage": "before_pairing",
                })
                continue
        if runtime_split_binding_downgraded:
            preparing_quality_admission = (
                _native_runtime_observation_admission(
                    preparing_quality_admission,
                    reason=split_quality_status,
                )
            )
            preparing_quality_status = (
                "shared_native_runtime_observation_"
                "split_binding_downgrade_verified"
            )
        admitted_row["_prepair_energy_quality_admission"] = (
            preparing_quality_admission
        )
        admitted_row["_prepair_energy_quality_admission_status"] = (
            preparing_quality_status
        )
        admitted.append((admitted_row, validation, defaults["setup"], defaults["ssh_arg"]))

    selected: dict[tuple[str, str, str, str, str, str], tuple[dict[str, Any], dict[str, Any] | None, str, str]] = {}
    duplicate_execution_keys: set[tuple[str, str, str, str, str, str]] = set()
    deduplicated: list[dict[str, Any]] = []
    for item in admitted:
        row, validation, setup, ssh_arg = item
        key = _dedupe_key(row, setup)
        if key in duplicate_execution_keys:
            deduplicated.append({
                "identity": key,
                "reason": "duplicate_runtime_execution_identity",
            })
            continue
        old = selected.pop(key, None)
        if old is None:
            selected[key] = item
            continue
        duplicate_execution_keys.add(key)
        deduplicated.append({
            "identity": key,
            "reason": "duplicate_runtime_execution_identity",
        })
        excluded.append({
            "backend": row.get("backend"),
            "model": row.get("model") or row.get("model_id"),
            "case": row.get("case") or row.get("case_id"),
            "setup_id": setup,
            "comparison_backend": row.get("comparison_backend"),
            "precision": row.get("precision"),
            "reason": "duplicate_runtime_execution_identity",
        })

    # Build setup-local comparison groups only as post-hoc annotations:
    # split + vendor Full + TensorRT Full.  A
    # Split's precision is its *boundary* contract.  Full baselines have no
    # split boundary and are therefore matched without requiring that their
    # runtime precision label equals the Split boundary precision.  Every Full
    # baseline must instead be the sole unambiguous annotation candidate for
    # (setup, model, backend, accelerator comparison context).
    def _vendor_full_backend(split_backend: str) -> str:
        return {
            "hailo8_to_trt": "native_full_hailo8",
            "hailo10h_to_trt": "native_full_hailo10h",
            "deepx_to_trt": "native_full_deepx",
        }.get(split_backend, "")

    def _pair_claim_identity(
        item: tuple[dict[str, Any], dict[str, Any] | None, str, str],
    ) -> dict[str, str]:
        claim_row, claim_validation, claim_setup, _claim_ssh = item
        claim_model = str(
            claim_row.get("model") or claim_row.get("model_id") or ""
        ).strip()
        task = str(_evidence_value(
            claim_row, claim_validation, "task", "benchmark_task",
        ) or "").strip().lower()
        frozen, frozen_status = _contract_fields_for_task(contract_manifest, task)

        explicit_model = str(_evidence_value(
            claim_row, claim_validation,
            "model_sha256", "model_hash", "full_model_sha256",
        ) or "").strip()
        command_contract = claim_row.get("_verified_energy_command_contract")
        command_contract = command_contract if isinstance(command_contract, dict) else {}
        input_sha = _normalize_sha256(str(_evidence_value(
            claim_row, claim_validation,
            "validation_input_sha256", "validation_image_sha256",
            "input_image_sha256",
        ) or command_contract.get("input_image_sha256") or ""))
        direction = _comparison_context(
            _evidence_value(claim_row, claim_validation, "direction")
            or claim_row.get("comparison_backend")
            or claim_row.get("backend")
        )
        identity = {
            "setup_id": str(claim_setup or "").strip().lower(),
            "model": claim_model,
            "task": task,
            "direction": direction,
            "model_sha256": _normalize_sha256(
                explicit_model or str(model_hashes.get(claim_model) or "")
            ),
            "validation_input_or_image_sha256": input_sha,
            "pipeline_contract_sha256": _normalize_sha256(
                frozen.get("contract_hash") if frozen_status == "verified" else ""
            ),
            "pipeline_preprocessing_sha256": _normalize_sha256(
                frozen.get("preprocessing_hash") if frozen_status == "verified" else ""
            ),
            "pipeline_decoder_sha256": _normalize_sha256(
                frozen.get("decoder_hash") if task == "detection" and frozen_status == "verified" else ""
            ),
            "pipeline_nms_sha256": _normalize_sha256(
                frozen.get("nms_hash") if task == "detection" and frozen_status == "verified" else ""
            ),
            "quality_contract_sha256": _normalize_sha256(_evidence_value(
                claim_row, claim_validation, "quality_contract_sha256",
            )),
            "preprocessing_contract_sha256": _normalize_sha256(_evidence_value(
                claim_row, claim_validation, "preprocessing_contract_sha256",
            )),
            "decoder_contract_sha256": _normalize_sha256(_evidence_value(
                claim_row, claim_validation, "decoder_contract_sha256",
            )) if task == "detection" else "not_applicable",
            "nms_contract_sha256": _normalize_sha256(_evidence_value(
                claim_row, claim_validation, "nms_contract_sha256",
            )) if task == "detection" else "not_applicable",
            "task_contract_status": frozen_status,
            "comparison_output_endpoint_id": str(
                claim_row.get("comparison_output_endpoint_id") or ""
            ).strip(),
            "comparison_endpoint_contract_hash": _strict_sha256_token(
                claim_row.get("comparison_endpoint_contract_hash")
            ),
            "comparison_endpoint_stage": str(
                claim_row.get("comparison_endpoint_stage") or ""
            ).strip().lower(),
        }
        identity.update(_energy_preprocess_identity(command_contract))
        return identity

    selected_items = list(selected.values())
    runtime_measurement_items = list(selected_items)
    pairing_exclusion_start = len(excluded)
    quality_pairing_posthoc_excluded_rows: list[
        dict[str, Any]
    ] = []
    claim_pair_eligible_keys: set[
        tuple[str, str, str, str, str, str]
    ] = set()

    def _positive_pair_claim_admission(
        item: tuple[
            dict[str, Any], dict[str, Any] | None, str, str,
        ],
    ) -> bool:
        admission = item[0].get(
            "_prepair_energy_quality_admission"
        )
        return bool(
            isinstance(admission, Mapping)
            and admission.get("admission_scope") == "native_energy"
            and admission.get("diagnostic_only") is False
            and admission.get("claim_comparable") is True
            and admission.get("energy_claim_eligible") is True
        )

    paired_missing: list[dict[str, Any]] = []
    paired_groups: list[dict[str, Any]] = []
    if (
        (not ns.allow_unpaired or ns.measure_all_runtime_successful)
        and not ns.screening_window_probe
    ):
        full_index: dict[
            tuple[str, str, str, str],
            list[tuple[dict[str, Any], dict[str, Any] | None, str, str]],
        ] = {}
        split_items: list[tuple[dict[str, Any], dict[str, Any] | None, str, str]] = []
        for item in selected_items:
            row, validation, setup, ssh_arg = item
            backend = str(row.get("backend") or "").strip().lower()
            model = str(row.get("model") or row.get("model_id") or "").strip()
            if backend.startswith("native_full_"):
                # Pair construction is a scientific gate even in a development
                # run: runtime-only or semantically failed Full rows must not be
                # selected just because no other baseline exists.
                full_valid, full_reason, _ = _semantic_decision(
                    row, validations, require_claim=require_claim_admission
                )
                if not full_valid:
                    excluded.append({
                        "backend": backend,
                        "model": model,
                        "case": str(row.get("case") or row.get("case_id") or "full"),
                        "setup_id": setup,
                        "precision": str(row.get("precision") or ""),
                        "comparison_backend": str(row.get("comparison_backend") or ""),
                        "reason": "pair_baseline_runtime_or_semantic_invalid",
                        "semantic_status": full_reason,
                    })
                    continue
                comparison = _comparison_context(row.get("comparison_backend"))
                full_index.setdefault((setup, model, backend, comparison), []).append(item)
            else:
                split_task = str(_evidence_value(
                    row, validation, "task", "benchmark_task",
                ) or "").strip().lower()
                if (
                    split_task == "detection"
                    and row.get("completion_pairing_eligible") is not True
                ):
                    paired_missing.append({
                        "backend": backend,
                        "model": model,
                        "case": str(
                            row.get("case") or row.get("case_id") or ""
                        ),
                        "setup_id": setup,
                        "precision": str(row.get("precision") or ""),
                        "comparison_backend": _comparison_context(backend),
                        "reason":
                            "detection_split_completed_tail_not_measured",
                        "completion_pairing_status": str(
                            row.get("completion_pairing_status") or ""
                        ),
                        "physical_output_endpoint_id": str(
                            row.get("physical_output_endpoint_id") or ""
                        ),
                    })
                    continue
                split_items.append(item)
        paired: dict[tuple[str, str, str, str, str, str], tuple[dict[str, Any], dict[str, Any] | None, str, str]] = {}
        used_full_items: set[tuple[str, str, str, str, str, str]] = set()
        for item in split_items:
            row, validation, setup, ssh_arg = item
            backend = str(row.get("backend") or "").strip().lower()
            model = str(row.get("model") or row.get("model_id") or "").strip()
            split_precision = str(row.get("precision") or "").strip().lower()
            producer = _comparison_context(backend)
            vendor = _vendor_full_backend(backend)
            vendor_key = (setup, model, vendor, producer)
            trt_key = (setup, model, "native_full_tensorrt", producer)
            vendor_candidates = list(full_index.get(vendor_key) or []) if vendor else []
            trt_candidates = list(full_index.get(trt_key) or [])
            vendor_item = vendor_candidates[0] if len(vendor_candidates) == 1 else None
            trt_item = trt_candidates[0] if len(trt_candidates) == 1 else None
            candidates_by_kind = {
                "vendor_full": vendor_candidates,
                "tensorrt_full": trt_candidates,
            }
            missing_kinds = [kind for kind, candidates in candidates_by_kind.items() if not candidates]
            ambiguous_kinds = [kind for kind, candidates in candidates_by_kind.items() if len(candidates) > 1]
            if missing_kinds or ambiguous_kinds:
                reason = (
                    "pair_baseline_missing_or_ambiguous"
                    if missing_kinds and ambiguous_kinds
                    else "pair_baseline_ambiguous"
                    if ambiguous_kinds
                    else "pair_baseline_missing"
                )
                paired_missing.append({
                    "backend": backend, "model": model,
                    "case": row.get("case") or row.get("case_id") or "",
                    "setup_id": setup,
                    "precision": split_precision,
                    "split_boundary_precision": split_precision,
                    "comparison_backend": producer,
                    "reason": reason,
                    "missing_baselines": missing_kinds,
                    "ambiguous_baselines": ambiguous_kinds,
                    "baseline_candidate_counts": {
                        kind: len(candidates)
                        for kind, candidates in candidates_by_kind.items()
                    },
                    "baseline_candidate_comparison_precisions": {
                        kind: sorted({
                            _full_comparison_precision(candidate[0], candidate[1])
                            for candidate in candidates if _full_comparison_precision(candidate[0], candidate[1])
                        })
                        for kind, candidates in candidates_by_kind.items()
                    },
                    "baseline_candidate_runtime_precisions": {
                        kind: sorted({
                            _full_runtime_precision(candidate[0], candidate[1])
                            for candidate in candidates if _full_runtime_precision(candidate[0], candidate[1])
                        })
                        for kind, candidates in candidates_by_kind.items()
                    },
                })
                continue
            assert vendor_item is not None and trt_item is not None
            pair_claim_identities = {
                "split": _pair_claim_identity(item),
                "vendor_full": _pair_claim_identity(vendor_item),
                "tensorrt_full": _pair_claim_identity(trt_item),
            }
            required_pair_fields = [
                "setup_id", "model", "task", "direction", "model_sha256",
                "validation_input_or_image_sha256",
                "prepared_feed_task", "prepared_feed_preprocess_mode",
                "prepared_feed_letterbox_pad_value",
                "prepared_feed_source_image_sha256",
            ]
            pipeline_pair_fields = [
                "pipeline_contract_sha256", "pipeline_preprocessing_sha256",
            ] + (
                ["pipeline_decoder_sha256", "pipeline_nms_sha256"]
                if str(pair_claim_identities["split"].get("task") or "") == "detection"
                else []
            )
            completed_endpoint_pair_fields = (
                [
                    "comparison_output_endpoint_id",
                    "comparison_endpoint_contract_hash",
                    "comparison_endpoint_stage",
                ]
                if str(
                    pair_claim_identities["split"].get("task") or ""
                ) == "detection"
                else []
            )
            if any(
                str(identity.get(field) or "").strip()
                for identity in pair_claim_identities.values()
                for field in pipeline_pair_fields
            ):
                required_pair_fields.extend(pipeline_pair_fields)
            required_pair_fields.extend(completed_endpoint_pair_fields)
            pair_missing_fields = {
                role: [field for field in required_pair_fields if not str(identity.get(field) or "").strip()]
                for role, identity in pair_claim_identities.items()
            }
            pair_missing_fields = {
                role: fields for role, fields in pair_missing_fields.items() if fields
            }
            pair_mismatch_fields = {
                field: sorted({
                    str(identity.get(field) or "")
                    for identity in pair_claim_identities.values()
                })
                for field in required_pair_fields
                if len({
                    str(identity.get(field) or "")
                    for identity in pair_claim_identities.values()
                }) > 1
            }
            if strict_energy_claim and (pair_missing_fields or pair_mismatch_fields):
                paired_missing.append({
                    "backend": backend,
                    "model": model,
                    "case": row.get("case") or row.get("case_id") or "",
                    "setup_id": setup,
                    "precision": split_precision,
                    "split_boundary_precision": split_precision,
                    "comparison_backend": producer,
                    "reason": "pair_contract_identity_missing_or_mismatch",
                    "pair_contract_identities": pair_claim_identities,
                    "missing_claim_identity_fields": pair_missing_fields,
                    "mismatched_claim_identity_fields": pair_mismatch_fields,
                })
                continue
            used_full_items.add(_dedupe_key(vendor_item[0], vendor_item[2]))
            used_full_items.add(_dedupe_key(trt_item[0], trt_item[2]))
            pair_quality_claim_eligible = all(
                _positive_pair_claim_admission(candidate)
                for candidate in (item, vendor_item, trt_item)
            )
            vendor_comparison_precision = _full_comparison_precision(vendor_item[0], vendor_item[1])
            trt_comparison_precision = _full_comparison_precision(trt_item[0], trt_item[1])
            vendor_runtime_precision = _full_runtime_precision(vendor_item[0], vendor_item[1])
            trt_runtime_precision = _full_runtime_precision(trt_item[0], trt_item[1])
            paired_groups.append({
                "model": model, "setup_id": setup, "split_backend": backend,
                "split_case": str(row.get("case") or row.get("case_id") or ""),
                "comparison_backend": producer,
                # ``precision`` remains a compatibility alias for the Split
                # boundary precision.  The explicit fields prevent it from
                # being mistaken for either Full runtime precision.
                "precision": split_precision,
                "split_boundary_precision": split_precision,
                "vendor_full_comparison_precision": vendor_comparison_precision,
                "tensorrt_full_comparison_precision": trt_comparison_precision,
                "vendor_full_runtime_precision": vendor_runtime_precision,
                "tensorrt_full_runtime_precision": trt_runtime_precision,
                "comparison_precision_match_required": False,
                "full_runtime_precision_match_required": False,
                "pair_contract_identity": pair_claim_identities["split"],
                "comparison_output_endpoint_id":
                    pair_claim_identities["split"].get(
                        "comparison_output_endpoint_id", ""
                    ),
                "comparison_endpoint_contract_hash":
                    pair_claim_identities["split"].get(
                        "comparison_endpoint_contract_hash", ""
                    ),
                "comparison_endpoint_stage":
                    pair_claim_identities["split"].get(
                        "comparison_endpoint_stage", ""
                    ),
                "pair_contract_identity_verified": not pair_missing_fields and not pair_mismatch_fields,
                "quality_pair_claim_eligible": (
                    pair_quality_claim_eligible
                ),
                "vendor_full_backend": vendor, "tensorrt_full_backend": "native_full_tensorrt",
            })
            for keep in (item, vendor_item, trt_item):
                krow, kval, ksetup, kssh = keep
                keep_key = _dedupe_key(krow, ksetup)
                paired[keep_key] = keep
                if (
                    ns.measure_all_runtime_successful
                    and pair_quality_claim_eligible
                ):
                    claim_pair_eligible_keys.add(keep_key)
        for candidates in full_index.values():
            for fitem in candidates:
                frow, _fvalidation, fsetup, _fssh = fitem
                if _dedupe_key(frow, fsetup) in used_full_items:
                    continue
                excluded.append({
                    "backend": str(frow.get("backend") or ""),
                    "model": str(frow.get("model") or frow.get("model_id") or ""),
                    "case": str(frow.get("case") or frow.get("case_id") or "full"),
                    "setup_id": fsetup,
                    "precision": str(frow.get("precision") or ""),
                    "comparison_backend": str(frow.get("comparison_backend") or ""),
                    "reason": "unpaired_full_baseline",
                })
        selected_items = list(paired.values())
        excluded.extend(paired_missing)
        if ns.measure_all_runtime_successful:
            quality_pairing_posthoc_excluded_rows = [
                dict(item)
                for item in excluded[pairing_exclusion_start:]
            ]
            del excluded[pairing_exclusion_start:]
            selected_items = runtime_measurement_items
            for (
                measurement_row,
                _measurement_validation,
                measurement_setup,
                _measurement_ssh,
            ) in selected_items:
                if (
                    _dedupe_key(
                        measurement_row, measurement_setup
                    )
                    in claim_pair_eligible_keys
                ):
                    continue
                prepared = measurement_row.get(
                    "_prepair_energy_quality_admission"
                )
                if isinstance(prepared, Mapping):
                    measurement_row[
                        "_prepair_energy_quality_admission"
                    ] = _native_runtime_observation_admission(
                        prepared,
                        reason=(
                            "quality_or_pairing_not_claim_eligible"
                        ),
                    )

    if ns.screening_window_probe:
        supported = {
            "hailo8_to_trt", "hailo10h_to_trt", "deepx_to_trt",
            "native_full_hailo8", "native_full_hailo10h",
            "native_full_hailo10", "native_full_deepx",
            "native_full_tensorrt",
        }

        def _probe_score(
            item: tuple[dict[str, Any], dict[str, Any] | None, str, str]
        ) -> tuple[int, int, float, str, str, str]:
            row, _validation, _setup, ssh_arg = item
            backend = str(row.get("backend") or "").strip().lower()
            try:
                fps = float(
                    row.get("fps_makespan") or row.get("FPS")
                    or row.get("pipeline_fps_selected") or 0
                )
            except Exception:
                fps = 0.0
            # Prefer a real split pipeline, then a vendor Full, then TensorRT
            # Full.  Remaining fields make the choice stable and auditable.
            kind_rank = (
                0 if "_to_trt" in backend
                else 1 if backend.startswith("native_full_") and backend != "native_full_tensorrt"
                else 2
            )
            return (
                kind_rank,
                0 if bool(getattr(ns, ssh_arg, "") if ssh_arg else "") else 1,
                -fps,
                backend,
                str(row.get("model") or row.get("model_id") or ""),
                str(row.get("case") or row.get("case_id") or "full"),
            )

        candidates = []
        for item in selected_items:
            row, _validation, _setup, ssh_arg = item
            backend = str(row.get("backend") or "").strip().lower()
            try:
                fps = float(
                    row.get("fps_makespan") or row.get("FPS")
                    or row.get("pipeline_fps_selected") or 0
                )
            except Exception:
                fps = 0.0
            if backend not in supported:
                excluded.append({
                    "backend": backend, "model": row.get("model"),
                    "case": row.get("case"), "reason": "probe_backend_unsupported",
                })
                continue
            if not ssh_arg or not str(getattr(ns, ssh_arg, "") or "").strip():
                excluded.append({
                    "backend": backend, "model": row.get("model"),
                    "case": row.get("case"), "reason": "probe_ssh_target_missing",
                })
                continue
            if fps <= 0:
                excluded.append({
                    "backend": backend, "model": row.get("model"),
                    "case": row.get("case"), "reason": "probe_fps_missing_or_zero",
                })
                continue
            candidates.append(item)
        candidates.sort(key=_probe_score)
        selected_items = candidates[:1]

    out = Path(ns.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# Native producer energy plan",
        "",
        f"Workload duration: `{duration_s:g}` s.",
        (
            "Runtime-success mode measures every technically constructible "
            "Native row; Quality and pairing remain post-hoc claim gates."
            if ns.measure_all_runtime_successful
            else "Split rows are included only after the native semantic gate."
        )
        + " Energy workloads are duration-driven and emit exact runtime-completed work units; historical FPS is context only.",
        "Split boundary precision and Full runtime precision are reported separately and are not used as an equality key for setup-local Full matching.",
        "",
    ]
    plan: list[dict[str, Any]] = []
    for row, validation, setup, ssh_arg in selected_items:
        backend = str(row.get("backend") or "")
        model = str(row.get("model") or "")
        case = str(row.get("case") or "full")
        is_full = backend.startswith("native_full_")
        split_boundary_precision = (
            "" if is_full
            else str(_evidence_value(row, validation, "precision", "dtype") or "").strip().lower()
        )
        legacy_comparison_precision = (
            _full_comparison_precision(row, validation) if is_full else ""
        )
        full_runtime_precision = (
            _full_runtime_precision(row, validation) if is_full else ""
        )
        execution_precision = (
            full_runtime_precision if is_full
            else str(_evidence_value(
                row, validation, "execution_precision", "runtime_precision_identity",
            ) or "").strip().lower()
        )
        # Compatibility field: Split boundary precision for Split rows, legacy
        # Split comparison stratum for Full rows.
        precision = legacy_comparison_precision if is_full else split_boundary_precision
        technical_identity = {
            "backend": backend,
            "model": model,
            "case": case,
            "setup_id": setup,
            "comparison_backend": row.get(
                "comparison_backend", ""
            ),
            "precision": precision,
        }
        ssh = getattr(ns, ssh_arg, "") if ssh_arg else ""
        env_arg = {"hailo8_ssh": "hailo8_env", "hailo10_ssh": "hailo10_env", "deepx_ssh": "deepx_env"}.get(ssh_arg, "")
        remote_env = str(getattr(ns, env_arg, "") or "") if env_arg else ""
        if not ssh:
            excluded.append({
                **technical_identity,
                "reason": "energy_ssh_target_missing",
            })
            continue
        try:
            fps = float(row.get("fps_makespan") or row.get("FPS") or row.get("pipeline_fps_selected") or 0)
        except Exception:
            fps = 0.0
        # Do not turn a historical performance FPS into the measured energy
        # workload.  Run32 demonstrated that such a stale estimate can shorten
        # a nominal 60 s replay to roughly 42 s.  Full replay runners now use
        # duration as the control variable and report the exact observed work
        # count.  ``--frames`` remains an explicit legacy minimum only.
        fallback_frames, fallback_frame_source = (
            _duration_controlled_minimum_work_units(int(ns.frames or 0))
        )
        command_contract = row.get("_verified_energy_command_contract")
        command_contract_status = str(row.get("_energy_command_contract_status") or "")
        if not isinstance(command_contract, dict):
            excluded.append({
                **technical_identity,
                "reason": "verified_energy_command_contract_unavailable",
            })
            continue
        try:
            if backend.startswith("native_full_"):
                child_argv = _full_runtime_argv(
                    command_contract, duration_s=duration_s, frames=fallback_frames,
                    remote_tool_dir=ns.remote_tool_dir,
                    authoritative_root=ns.remote_root,
                    preflight_nonce=_PREFLIGHT_NONCE_TOKEN,
                    preflight_attestation=_PREFLIGHT_ATTESTATION_TOKEN,
                    preflight_max_age_s=300.0,
                )
            else:
                if split_energy_runtime_argv is None:
                    raise RuntimeError("split_energy_command_contract_helper_unavailable")
                child_argv = split_energy_runtime_argv(
                    command_contract,
                    duration_s=duration_s,
                    fresh_output_root=_FRESH_OUTPUT_TOKEN,
                    remote_tool_dir=ns.remote_tool_dir,
                    preflight_attestation_path=_PREFLIGHT_ATTESTATION_TOKEN,
                    preflight_nonce=_PREFLIGHT_NONCE_TOKEN,
                )
        except Exception as exc:
            excluded.append({
                **technical_identity,
                "reason": "verified_energy_command_contract_not_replayable",
                "detail": f"{type(exc).__name__}: {exc}",
                "command_contract_sha256": str(command_contract.get("contract_sha256") or ""),
            })
            continue
        child = _shell_join_with_fresh_output([str(value) for value in child_argv])
        try:
            process_local_environment = (
                _process_local_runtime_environment(command_contract)
            )
            env_prefix = _remote_environment_prefix(
                remote_env, process_local_environment,
            )
        except Exception as exc:
            excluded.append({
                **technical_identity,
                "reason": "mixed_runtime_environment_contract_invalid",
                "detail": f"{type(exc).__name__}: {exc}",
                "command_contract_sha256": str(
                    command_contract.get("contract_sha256") or ""
                ),
            })
            continue
        safe_setup = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in setup) or "setup"
        safe_identity = "__".join(
            "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value) or "unknown"
            for value in (backend, model, case, safe_setup)
        )
        contract_sha = str(command_contract.get("contract_sha256") or "")
        safe_identity = f"{safe_identity}__{contract_sha[:12]}"
        fresh_base = f"{ns.remote_root.rstrip('/')}/.energy_replays/{safe_identity}"
        contract_payload = (
            json.dumps(
                command_contract, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )
        local_contract_file = out / f"{safe_identity}.command_contract.json"
        local_contract_file.write_text(contract_payload, encoding="utf-8")
        contract_payload_sha = _sha256_file(local_contract_file)
        remote_contract_file = (
            f"{fresh_base}/command_contract_{contract_payload_sha}.json"
        )
        child = child.replace(
            _REMOTE_CONTRACT_TOKEN, remote_contract_file,
        )
        preflight_shell: Path | None = None
        preflight_runtime_attestation = (
            f"{fresh_base}/preflight_{_PREFLIGHT_NONCE_TOKEN}.json"
        )
        try:
            if backend.startswith("native_full_"):
                preflight_argv = _full_preflight_argv(
                    command_contract,
                    remote_tool_dir=ns.remote_tool_dir,
                    authoritative_root=ns.remote_root,
                    preflight_nonce=_PREFLIGHT_NONCE_TOKEN,
                    preflight_attestation=_PREFLIGHT_ATTESTATION_TOKEN,
                    max_age_s=300.0,
                )
            else:
                preflight_argv = _split_preflight_argv(
                    command_contract,
                    remote_tool_dir=ns.remote_tool_dir,
                    preflight_nonce=_PREFLIGHT_NONCE_TOKEN,
                    preflight_attestation=_PREFLIGHT_ATTESTATION_TOKEN,
                    max_age_s=300.0,
                )
        except Exception as exc:
            excluded.append({
                **technical_identity,
                "reason": "energy_preflight_contract_not_replayable",
                "detail": f"{type(exc).__name__}: {exc}",
                "command_contract_sha256": contract_sha,
            })
            continue
        else:
            preflight_child = " ".join(shlex.quote(str(value)) for value in preflight_argv)
            preflight_child = preflight_child.replace(
                _REMOTE_CONTRACT_TOKEN, remote_contract_file,
            )
            preflight_remote_command = (
                f"umask 077 && mkdir -p {shlex.quote(fresh_base)} && "
                f"tmp_contract=$(mktemp {shlex.quote(fresh_base + '/command_contract.XXXXXX')}) && "
                'cat > "$tmp_contract" && '
                'actual_contract_sha=$(sha256sum "$tmp_contract") && '
                'actual_contract_sha=${actual_contract_sha%% *} && '
                f"test \"$actual_contract_sha\" = {shlex.quote(contract_payload_sha)} && "
                f"chmod 600 \"$tmp_contract\" && mv \"$tmp_contract\" {shlex.quote(remote_contract_file)} && "
                f"{env_prefix}"
                f"cd {shlex.quote(ns.remote_tool_dir)} && {preflight_child}"
            )
            preflight_shell = out / f"{safe_identity}.preflight.sh"
            preflight_shell.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                + _lease_aware_ssh_script_body(
                    ssh_target=str(ssh),
                    remote_command=preflight_remote_command,
                    operation_label=f"native-energy-preflight-{safe_identity}",
                    stdin_path=local_contract_file,
                    timeout_s=max(
                        1.0,
                        float(min(int(ns.timeout), 900))
                        - _REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
                    ),
                ),
                encoding="utf-8",
            )
            preflight_shell.chmod(0o755)
        report_relative = _fresh_report_relative_path(backend)
        remote_command = (
            f"{env_prefix}mkdir -p {shlex.quote(fresh_base)} && "
            f"fresh_output_root=$(mktemp -d {shlex.quote(fresh_base)}/replay.XXXXXX) && "
            'touch "$fresh_output_root/.started" && '
            f"cd {shlex.quote(ns.remote_tool_dir)} && {child} && "
            f'test -s "$fresh_output_root/{report_relative}" && '
            f'test "$fresh_output_root/{report_relative}" -nt "$fresh_output_root/.started" && '
            f'printf "[native-energy] FRESH_REPORT path=%s sha_contract={contract_sha}\\n" '
            f'"$fresh_output_root/{report_relative}"'
        )
        shell = out / f"{safe_identity}.sh"
        shell.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + _lease_aware_ssh_script_body(
                ssh_target=str(ssh),
                remote_command=remote_command,
                operation_label=f"native-energy-workload-{safe_identity}",
                timeout_s=max(
                    1.0,
                    float(ns.timeout) - _REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
                ),
            ),
            encoding="utf-8",
        )
        shell.chmod(0o755)
        energy_cli = _script_path("energy_measurement_cli.py")
        measure_parts = [
            shlex.quote(sys.executable), "-u", shlex.quote(str(energy_cli)), "measure",
            "--setup-id", shlex.quote(setup),
            "--run-id", shlex.quote(f"native_{safe_identity}"),
            "--command-file", shlex.quote(str(shell)),
            "--duration", f"{duration_s:g}",
            "--inference-count", str(fallback_frames),
            "--pipeline-fps", f"{fps:.6f}",
            "--timeout", str(ns.timeout),
            "--runs", str(energy_runs_per_row),
            "--physical-scope", shlex.quote(str(ns.physical_scope or "")),
            "--window-label", shlex.quote(str(ns.window_label or "command")),
        ]
        if ns.campaign_budget_file:
            measure_parts += [shlex.quote(value) for value in campaign_budget_forward_args(ns)]
            measure_parts += ["--campaign-row-id", shlex.quote(json.dumps([backend, model, case, setup], separators=(",", ":"))),
                              "--campaign-repeats", str(energy_runs_per_row),
                              "--invalid-repeat-max-retries", str(ns.campaign_max_retries)]
        if str(ns.hardware_setups_file or "").strip():
            measure_parts += [
                "--hardware-setups-file",
                shlex.quote(str(ns.hardware_setups_file)),
            ]
        if str(backend).strip().lower() == "native_full_tensorrt":
            measure_parts += [
                "--host-normalization-source-run-id", "native_full_tensorrt",
                "--host-normalization-target-variant", "full",
            ]
        if ns.smoke_diagnostic:
            measure_parts += [
                "--diagnostic-only",
                "--claim-exclusion-reason", "smoke_diagnostic_only",
            ]
        elif ns.screening_energy:
            measure_parts += [
                "--diagnostic-only",
                "--claim-exclusion-reason",
                "development_screening_energy_only",
            ]
        else:
            prepared_for_measurement = row.get(
                "_prepair_energy_quality_admission"
            )
            if (
                isinstance(prepared_for_measurement, Mapping)
                and prepared_for_measurement.get(
                    "diagnostic_only"
                ) is True
            ):
                measure_parts += [
                    "--diagnostic-only",
                    "--claim-exclusion-reason",
                    shlex.quote(str(
                        prepared_for_measurement.get(
                            "runtime_observation_reason"
                        )
                        or "quality_or_pairing_not_claim_eligible"
                    )),
                ]
        # Every regular Native Energy row receives a plan-local parent below
        # the EvaluationRun.  The execution helper must create a fresh
        # ``attempt_<uuid>`` child immediately before launching the collector
        # and replace this planning-time ``--out`` value with that child.  The
        # parent itself is never a collector output directory.
        measurement_output_base_dir = (
            (out.parent / "measurement").resolve()
            if ns.screening_window_probe
            else (
                out.parent / "measurements" / safe_identity
                / f"plan_{plan_attempt_id}"
            ).resolve()
        )
        measure_parts += [
            "--out", shlex.quote(str(measurement_output_base_dir)),
        ]
        if preflight_shell is not None:
            measure_parts += [
                "--preflight-command-file", shlex.quote(str(preflight_shell)),
                "--preflight-runtime-attestation-path", shlex.quote(preflight_runtime_attestation),
                "--preflight-timeout-s", str(min(int(ns.timeout), 900)),
                "--preflight-attestation-max-age-s", "300",
                "--preflight-expected-command-contract-sha256", shlex.quote(contract_sha),
            ]
        if window_ab_json:
            measure_parts += [
                "--window-method-ab-json", shlex.quote(window_ab_json),
            ]
        if ns.screening_window_probe:
            # The probe is self-contained under its dedicated report tree.  It
            # always requests the paired postprocessor comparison on each trace.
            measure_parts += [
                "--compare-legacy-window",
            ]
        if ns.require_runtime_work_units:
            measure_parts.append("--require-runtime-work-units")
        if ns.require_command_window_alignment:
            measure_parts.append("--require-command-window-alignment")
        if str(ns.calibration_manifest or "").strip():
            measure_parts += ["--calibration-manifest", shlex.quote(str(ns.calibration_manifest))]
        if str(ns.calibration_sha256 or "").strip():
            measure_parts += ["--calibration-sha256", shlex.quote(str(ns.calibration_sha256))]
        measure = " ".join(measure_parts)
        task = str(_evidence_value(row, validation, "task", "benchmark_task") or "").strip().lower()
        task_contract_fields, task_contract_status = _contract_fields_for_task(contract_manifest, task)

        contract_conflicts: list[str] = []
        quality_provenance, quality_provenance_conflicts = (
            _validated_quality_provenance(validation)
        )
        contract_conflicts.extend(
            f"central_quality_provenance:{field}"
            for field in quality_provenance_conflicts
        )
        def _pipeline_value(field: str, frozen_field: str) -> str:
            explicit = str(_evidence_value(row, validation, field) or "").strip()
            frozen = str(task_contract_fields.get(frozen_field) or "").strip()
            if explicit and frozen and _normalize_sha256(explicit) != _normalize_sha256(frozen):
                contract_conflicts.append(field)
                return ""
            return _normalize_sha256(explicit or frozen)

        explicit_model_hash = str(_evidence_value(row, validation, "model_sha256", "model_hash", "full_model_sha256") or "").strip()
        frozen_model_hash = str(model_hashes.get(model) or "")
        if explicit_model_hash and frozen_model_hash and _normalize_sha256(explicit_model_hash) != _normalize_sha256(frozen_model_hash):
            contract_conflicts.append("model_sha256")
            model_sha256 = ""
        else:
            model_sha256 = _normalize_sha256(explicit_model_hash or frozen_model_hash)
        if backend == "native_full_tensorrt":
            contract_source_model_sha = _strict_sha256_token(
                command_contract.get("source_model_sha256")
            )
            if not contract_source_model_sha or contract_source_model_sha != _strict_sha256_token(model_sha256):
                contract_conflicts.append("tensorrt_source_model_sha256")
                model_sha256 = ""
        pipeline_contract_hash = _pipeline_value("pipeline_contract_sha256", "contract_hash")
        pipeline_preprocessing_hash = _pipeline_value(
            "pipeline_preprocessing_sha256", "preprocessing_hash",
        )
        pipeline_decoder_hash = _pipeline_value(
            "pipeline_decoder_sha256", "decoder_hash",
        ) if task == "detection" else ""
        pipeline_nms_hash = _pipeline_value(
            "pipeline_nms_sha256", "nms_hash",
        ) if task == "detection" else ""
        quality_contract_hash = _normalize_sha256(_evidence_value(
            row, validation, "quality_contract_sha256",
        ))
        quality_preprocessing_hash = _normalize_sha256(_evidence_value(
            row, validation, "preprocessing_contract_sha256",
        ))
        quality_decoder_hash = _normalize_sha256(_evidence_value(
            row, validation, "decoder_contract_sha256",
        )) if task == "detection" else ""
        quality_nms_hash = _normalize_sha256(_evidence_value(
            row, validation, "nms_contract_sha256",
        )) if task == "detection" else ""
        pipeline_required = [
            pipeline_contract_hash, pipeline_preprocessing_hash,
        ] + ([pipeline_decoder_hash, pipeline_nms_hash] if task == "detection" else [])
        native_quality_bridge_verified = _native_quality_bridge_verified(
            validation, task, row,
            exact_identity_required=bool(
                row.get("native_split_quality_required") is True
            ),
        )
        pipeline_contract_verified = bool(
            contract_manifest_status == "verified"
            and task_contract_status == "verified"
            and all(pipeline_required)
        )
        completion_pairing_eligible = bool(
            row.get("completion_pairing_eligible") is True
        )
        current_quality_first_claim = bool(
            row.get("native_split_quality_required") is True
        )
        contract_evidence_ok = bool(
            (
                native_quality_bridge_verified
                if current_quality_first_claim
                else (pipeline_contract_verified or native_quality_bridge_verified)
            )
            and model_hash_map_status == "verified"
            and not contract_conflicts
            and model_sha256
            and completion_pairing_eligible
        )
        def _compat_pipeline_value(
            normalized: str, field: str, frozen_field: str,
        ) -> str:
            """Preserve the legacy display value while namespaced SHA fields stay canonical."""
            if not normalized:
                return ""
            return str(
                _evidence_value(row, validation, field)
                or task_contract_fields.get(frozen_field)
                or ""
            ).strip()

        contract_hash = _compat_pipeline_value(
            pipeline_contract_hash, "pipeline_contract_sha256", "contract_hash",
        ) or quality_contract_hash
        preprocessing_hash = _compat_pipeline_value(
            pipeline_preprocessing_hash,
            "pipeline_preprocessing_sha256", "preprocessing_hash",
        ) or quality_preprocessing_hash
        decoder_hash = _compat_pipeline_value(
            pipeline_decoder_hash, "pipeline_decoder_sha256", "decoder_hash",
        ) or quality_decoder_hash
        nms_hash = _compat_pipeline_value(
            pipeline_nms_hash, "pipeline_nms_sha256", "nms_hash",
        ) or quality_nms_hash
        semantic_validation_ok = bool(
            validation is not None
            and validation.get("semantic_ok") is True
            and _truth(validation.get("contract_consistent"))
            and (task != "classification" or validation.get("top1_match") is True)
        )
        semantic_claim_ok = bool(
            semantic_validation_ok
            and validation is not None
            and _truth(validation.get("claim_ok"))
        )
        accuracy_gate_pass = bool(
            validation is not None
            and validation.get("accuracy_gate_pass") is True
        )
        screening_comparable = bool(
            semantic_validation_ok and contract_evidence_ok
        )
        claim_comparable = bool(
            screening_comparable
            and semantic_claim_ok
            and accuracy_gate_pass
        )
        split_energy_evidence = row.get("_split_quality_energy_evidence")
        split_energy_evidence = (
            dict(split_energy_evidence)
            if isinstance(split_energy_evidence, Mapping) else {}
        )
        split_consumer_attestation = row.get(
            "native_split_quality_consumer_attestation"
        )
        split_consumer_attestation = (
            dict(split_consumer_attestation)
            if isinstance(split_consumer_attestation, Mapping) else {}
        )
        prospective_exclusion_reason = str(
            row.get("prospective_detection_claim_exclusion_reason")
            or ""
        )
        prospectively_excluded = bool(
            row.get("prospective_detection_claim_exclusion") is True
            and prospective_exclusion_reason
        )
        detection_claim_annotation_invalid = bool(
            row.get("detection_claim_annotation_invalid") is True
        )
        precision_quality_binding_verified = bool(
            validation is not None
            and validation.get(
                "precision_quality_binding_verified"
            ) is True
        )
        task_quality_observation_valid = bool(
            validation is not None
            and validation.get(
                "task_quality_observation_valid"
            ) is True
            and isinstance(
                validation.get("accuracy_gate_pass"), bool,
            )
        )
        quality_provenance_complete = bool(
            quality_provenance
            and not quality_provenance_conflicts
            and all(quality_provenance.values())
        )
        quality_claim_result_verified = bool(
            validation is not None
            and validation.get(
                "quality_claim_result_verified"
            ) is True
        )
        prepared_admission = row.get(
            "_prepair_energy_quality_admission"
        )
        if not isinstance(prepared_admission, Mapping):
            if ns.measure_all_runtime_successful:
                prepared_admission = (
                    _fallback_native_runtime_observation_admission(
                        row,
                        setup=setup,
                        reason=(
                            "energy_quality_annotation_missing_after_pairing"
                        ),
                    )
                )
            else:
                excluded.append({
                    "backend": backend,
                    "model": model,
                    "case": case,
                    "setup_id": setup,
                    "comparison_backend": row.get(
                        "comparison_backend", ""
                    ),
                    "precision": precision,
                    "reason": (
                        "energy_quality_admission_missing_after_pairing"
                    ),
                    "energy_quality_admission_stage": (
                        "final_plan_row_before_measurement"
                    ),
                })
                continue
        energy_quality_admission = dict(prepared_admission)
        quality_diagnostic_only = bool(
            energy_quality_admission.get("diagnostic_only") is True
        )
        screening_comparable = bool(
            energy_quality_admission.get("screening_comparable") is True
        )
        claim_comparable = bool(
            energy_quality_admission.get("claim_comparable") is True
        )
        technical_admission = {
            "schema": (
                "onnx-splitpoint/native-energy-planner-admission"
            ),
            "schema_version": 1,
            "runtime_success": True,
            "energy_command_preflight_ok": True,
            "energy_command_preflight_scope": (
                "planner_time_verified_contract_ssh_runtime_and_"
                "preflight_command_constructibility"
            ),
            "full_baseline": bool(is_full),
            "split_has_valid_part2_input": bool(
                row.get("_split_has_valid_part2_input") is True
            ),
            "split_part2_input_status": str(
                row.get("_split_part2_input_status") or ""
            ),
            "effective_part2_input_count": row.get(
                "_effective_part2_input_count"
            ),
            "selected": True,
            "predicate": (
                "runtime_success AND energy_command_preflight_ok AND "
                "(full_baseline OR split_has_valid_part2_input)"
            ),
        }
        plan.append({
            "backend": backend,
            "model": model,
            "case": case,
            # Compatibility field with the role made explicit below.
            "precision": precision,
            "precision_role": "legacy_comparison_precision" if is_full else "split_boundary_precision",
            "execution_precision": execution_precision,
            "split_boundary_precision": split_boundary_precision,
            "full_runtime_precision": full_runtime_precision,
            "comparison_precision": legacy_comparison_precision,
            "legacy_comparison_precision": legacy_comparison_precision,
            "runtime_precision_status": "verified_explicit" if execution_precision else "unavailable",
            "setup_id": setup,
            "comparison_backend": row.get("comparison_backend", ""),
            "runtime_success": True,
            "energy_command_preflight_ok": True,
            "full_baseline": bool(is_full),
            "split_has_valid_part2_input": bool(
                technical_admission[
                    "split_has_valid_part2_input"
                ]
            ),
            "native_energy_planner_admission": technical_admission,
            "semantic_gate": (
                "screening_only_not_claimable" if screening_only
                else "smoke_diagnostic_not_claimable" if ns.smoke_diagnostic
                else "historical_diagnostic_only_not_claimable"
                if historical_diagnostic_only
                else "pass" if semantic_claim_ok else "diagnostic_only_not_claimable"
            ),
            "semantic_claim_ok": (
                False if quality_diagnostic_only else semantic_claim_ok
            ),
            "semantic_validation_ok": semantic_validation_ok,
            "claim_ok": bool(
                claim_comparable and not quality_diagnostic_only
            ),
            "screening_comparable": screening_comparable,
            "claim_comparable": bool(
                claim_comparable and not quality_diagnostic_only
            ),
            "final_all_split_energy": bool(ns.final_all_split_energy),
            "screening_only": screening_only,
            "screening_energy": bool(ns.screening_energy),
            **({
                "measure_all_runtime_successful": True,
                "measurement_admission_policy": (
                    "all_runtime_successful_constructible_native_rows"
                ),
            } if ns.measure_all_runtime_successful else {}),
            "energy_evidence_tier": (
                "screening" if ns.screening_energy else
                "window_probe" if ns.screening_window_probe else
                "smoke_diagnostic" if ns.smoke_diagnostic else "final_claim"
            ),
            "energy_tier": (
                "screening" if ns.screening_energy else
                "window_probe" if ns.screening_window_probe else
                "smoke_diagnostic" if ns.smoke_diagnostic else "final_claim"
            ),
            "smoke_diagnostic": bool(ns.smoke_diagnostic),
            "diagnostic_only": quality_diagnostic_only,
            "claim_eligible": (
                False if quality_diagnostic_only else True
            ),
            "prospective_detection_claim_exclusion": (
                prospectively_excluded
            ),
            "prospective_detection_claim_exclusion_reason": (
                prospective_exclusion_reason
            ),
            "detection_diagnostic_sha256": str(
                row.get("detection_diagnostic_sha256") or ""
            ),
            "detection_claim_exclusion_entry_sha256": str(
                row.get(
                    "detection_claim_exclusion_entry_sha256"
                ) or ""
            ),
            "detection_claim_exclusions_contract_sha256": str(
                row.get(
                    "detection_claim_exclusions_contract_sha256"
                ) or ""
            ),
            "detection_claim_annotation_invalid": (
                detection_claim_annotation_invalid
            ),
            "detection_claim_annotation_status": str(
                row.get("detection_claim_annotation_status") or ""
            ),
            "historical_diagnostic_only": historical_diagnostic_only,
            "scientific_claim_exclusion_reasons": (
                [diagnostic_claim_exclusion_reason]
                if historical_diagnostic_only else []
            ) + (
                ["development_screening_energy_only"]
                if ns.screening_energy else []
            ) + (
                [str(
                    row.get("completion_pairing_status")
                    or "detection_completed_endpoint_not_verified"
                )]
                if task == "detection"
                and not completion_pairing_eligible else []
            ) + (
                ["accuracy_gate_failed"]
                if (
                    validation is not None
                    and validation.get("accuracy_gate_pass") is False
                ) else []
            ) + (
                [prospective_exclusion_reason]
                if prospectively_excluded else []
            ) + (
                ["detection_claim_exclusions_contract_invalid"]
                if detection_claim_annotation_invalid else []
            ),
            "eligible_for_energy_results_import": (
                False
                if (
                    quality_diagnostic_only
                    or not completion_pairing_eligible
                )
                else None
            ),
            "eligible_for_scientific_claim": (
                False if quality_diagnostic_only else claim_comparable
            ),
            "energy_claim_eligible": (
                bool(claim_comparable and not quality_diagnostic_only)
            ),
            "contract_consistent": _truth(_evidence_value(row, validation, "contract_consistent")),
            "central_quality_evidence_verified": bool(
                validation is not None
                and validation.get("central_quality_evidence_verified") is True
            ),
            "precision_quality_verified": bool(
                validation is not None
                and validation.get("precision_quality_verified") is True
            ),
            "precision_quality_binding_verified": bool(
                precision_quality_binding_verified
            ),
            "task_quality_observation_valid": bool(
                task_quality_observation_valid
            ),
            "accuracy_gate_pass": accuracy_gate_pass,
            "quality_provenance_complete": (
                quality_provenance_complete
            ),
            "quality_claim_result_verified": (
                quality_claim_result_verified
            ),
            "energy_quality_admission": energy_quality_admission,
            "energy_quality_admission_sha256": (
                energy_quality_admission["admission_sha256"]
            ),
            **_energy_quality_result_fields(
                energy_quality_admission
            ),
            "quality_gate_status": str(
                _evidence_value(
                    row, validation, "quality_gate_status", "gate_status",
                ) or ""
            ),
            "task": task,
            **_energy_preprocess_identity(command_contract),
            "evaluation_role": str(_evidence_value(row, validation, "evaluation_role", "model_role") or ""),
            "runner_regime": "native_full" if backend.startswith("native_full_") else "native_fifo",
            "direction": str(_evidence_value(row, validation, "direction") or (row.get("comparison_backend") if backend.startswith("native_full_") else backend) or ""),
            "contract_hash": contract_hash,
            "preprocessing_hash": preprocessing_hash,
            "decoder_hash": decoder_hash,
            "nms_hash": nms_hash,
            "pipeline_contract_sha256": pipeline_contract_hash,
            "pipeline_preprocessing_sha256": pipeline_preprocessing_hash,
            "pipeline_decoder_sha256": pipeline_decoder_hash,
            "pipeline_nms_sha256": pipeline_nms_hash,
            "quality_contract_sha256": quality_contract_hash,
            "preprocessing_contract_sha256": quality_preprocessing_hash,
            "decoder_contract_sha256": quality_decoder_hash,
            "nms_contract_sha256": quality_nms_hash,
            "physical_output_endpoint_id": str(
                row.get("physical_output_endpoint_id") or ""
            ),
            "physical_endpoint_contract_hash": _strict_sha256_token(
                row.get("physical_endpoint_contract_hash")
            ),
            "physical_endpoint_stage": str(
                row.get("physical_endpoint_stage") or ""
            ),
            "physical_endpoint_contract_complete": bool(
                row.get("physical_endpoint_contract_complete") is True
            ),
            "comparison_output_endpoint_id": str(
                row.get("comparison_output_endpoint_id") or ""
            ),
            "comparison_endpoint_contract_hash": _strict_sha256_token(
                row.get("comparison_endpoint_contract_hash")
            ),
            "comparison_endpoint_stage": str(
                row.get("comparison_endpoint_stage") or ""
            ),
            "output_endpoint_id": str(
                row.get("output_endpoint_id") or ""
            ),
            "endpoint_contract_hash": _strict_sha256_token(
                row.get("endpoint_contract_hash")
            ),
            "endpoint_stage": str(row.get("endpoint_stage") or ""),
            "endpoint_contract_complete": bool(
                row.get("physical_endpoint_contract_complete") is True
                if task != "detection"
                else row.get("completion_pairing_eligible") is True
            ),
            "output_endpoint_match": completion_pairing_eligible,
            "completion_pairing_eligible": completion_pairing_eligible,
            "completion_pairing_status": str(
                row.get("completion_pairing_status") or ""
            ),
            "model_sha256": model_sha256,
            "source_request_sha256": quality_provenance["source_request_sha256"],
            "validation_dataset_sha256": quality_provenance["validation_dataset_sha256"],
            "validation_dataset_image_ids_sha256": quality_provenance[
                "validation_dataset_image_ids_sha256"
            ],
            "validation_dataset_ground_truth_sha256": quality_provenance[
                "validation_dataset_ground_truth_sha256"
            ],
            "accuracy_gate_policy_sha256": quality_provenance[
                "accuracy_gate_policy_sha256"
            ],
            "task_quality_policy_sha256": quality_provenance[
                "task_quality_policy_sha256"
            ],
            "runtime_quality_gate_policy_sha256": quality_provenance[
                "runtime_quality_gate_policy_sha256"
            ],
            "validation_input_or_image_sha256": _normalize_sha256(str(
                _evidence_value(
                    row, validation,
                    "validation_input_sha256", "validation_image_sha256",
                    "input_image_sha256",
                )
                or command_contract.get("input_image_sha256")
                or ""
            )),
            "successful_command_contract_status": command_contract_status,
            "successful_command_contract_sha256": contract_sha,
            **({
                "completed_task_completion_mode": "native_three_stage_fast_oracle_outside_timing",
                "native_command_contract": dict(command_contract),
                "completed_task_comparison_output_endpoint_id": str(row.get("completed_task_comparison_output_endpoint_id") or row.get("comparison_output_endpoint_id") or ""),
                "energy_completion_requires_fresh_observation": True,
            } if row.get("completed_task_completion_mode") == "native_three_stage_fast_oracle_outside_timing" else {}),
            "native_split_energy_quality_binding": split_energy_evidence,
            "native_split_energy_quality_binding_sha256": str(
                split_energy_evidence.get("evidence_sha256") or ""
            ),
            "native_split_quality_required": bool(
                split_energy_evidence.get("native_split_quality_required")
            ),
            "native_split_energy_binding_valid": bool(
                split_energy_evidence.get("native_split_energy_binding_valid")
            ),
            "native_split_energy_binding_status": str(
                split_energy_evidence.get("native_split_energy_binding_status") or ""
            ),
            "native_split_quality_binding_sha256": str(
                split_energy_evidence.get("native_split_quality_binding_sha256") or ""
            ),
            "native_split_quality_preselection_sha256": str(
                split_energy_evidence.get("native_split_quality_preselection_sha256") or ""
            ),
            "native_split_quality_source_request_sha256": str(
                split_energy_evidence.get(
                    "native_split_quality_source_request_sha256"
                ) or ""
            ),
            "native_split_quality_central_result_sha256": str(
                split_energy_evidence.get(
                    "native_split_quality_central_result_sha256"
                ) or ""
            ),
            "native_split_quality_selection_sha256": str(
                split_energy_evidence.get(
                    "native_split_quality_selection_sha256"
                ) or ""
            ),
            "native_split_quality_eval_run_id": str(
                split_energy_evidence.get("native_split_quality_eval_run_id") or ""
            ),
            "native_split_quality_source_run_id": str(
                split_energy_evidence.get("native_split_quality_source_run_id") or ""
            ),
            "native_split_quality_consumer_attestation": split_consumer_attestation,
            "native_split_quality_consumer_attestation_sha256": str(
                split_energy_evidence.get(
                    "native_split_quality_consumer_attestation_sha256"
                ) or ""
            ),
            "native_split_quality_portable_binding_status": str(
                split_energy_evidence.get(
                    "native_split_quality_portable_binding_status"
                ) or ""
            ),
            "native_split_quality_authority_workflow_version": str(
                split_energy_evidence.get(
                    "native_split_quality_authority_workflow_version"
                ) or ""
            ),
            "native_split_quality_authority_run_id": str(
                split_energy_evidence.get(
                    "native_split_quality_authority_run_id"
                ) or ""
            ),
            "native_split_semantic_output_manifest": str(
                split_energy_evidence.get("native_split_semantic_output_manifest") or ""
            ),
            "native_split_semantic_output_manifest_sha256": str(
                split_energy_evidence.get(
                    "native_split_semantic_output_manifest_sha256"
                ) or ""
            ),
            "native_split_semantic_boundary_manifest": str(
                split_energy_evidence.get("native_split_semantic_boundary_manifest") or ""
            ),
            "native_split_semantic_boundary_manifest_sha256": str(
                split_energy_evidence.get(
                    "native_split_semantic_boundary_manifest_sha256"
                ) or ""
            ),
            "fresh_output_policy": "remote_mktemp_unique_per_measurement_invocation",
            "fresh_report_relative_path": report_relative,
            "contract_evidence_ok": contract_evidence_ok,
            "contract_evidence_status": (
                "verified" if contract_evidence_ok else "missing_conflicting_or_unverified"
            ),
            "contract_evidence_source": (
                "exact_native_central_quality_bridge"
                if native_quality_bridge_verified
                else "frozen_pipeline_contract_manifest"
                if contract_manifest_status == "verified" and task_contract_status == "verified"
                else "unavailable"
            ),
            "contract_evidence_conflicts": contract_conflicts,
            "pipeline_contract_manifest_status": contract_manifest_status,
            "task_contract_status": task_contract_status,
            "model_hash_map_status": model_hash_map_status,
            "command_file": str(shell),
            "command_contract_file": str(local_contract_file),
            "command_contract_file_sha256": contract_payload_sha,
            "remote_command_contract_file": remote_contract_file,
            "command_contract_transport": (
                "ssh_stdin_to_hash_verified_remote_file"
            ),
            "process_local_runtime_environment": dict(
                process_local_environment
            ),
            "process_local_runtime_site_policy": (
                "site.addsitedir_after_system_defaults"
                if process_local_environment else "not_required"
            ),
            "preflight_command_file": str(preflight_shell) if preflight_shell is not None else "",
            "preflight_required": bool(preflight_shell is not None),
            "preflight_runtime_attestation_path_template": preflight_runtime_attestation,
            "preflight_nonce_binding": (
                "collector_fresh_nonce_per_repeat" if preflight_shell is not None else "not_applicable"
            ),
            "fps": fps,
            "duration_s": duration_s,
            "configured_fallback_work_units": fallback_frames,
            "configured_fallback_work_units_source": fallback_frame_source,
            "work_units_source": "exact_runtime_marker_required_duration_driven",
            "duration_controlled_workload": True,
            "historical_fps_used_for_work_count": False,
            "historical_fps_context_available": bool(fps > 0.0),
            "require_runtime_work_units": bool(ns.require_runtime_work_units),
            "require_command_window_alignment": bool(ns.require_command_window_alignment),
            "energy_scope": str(ns.physical_scope or ""),
            "energy_window": str(ns.window_label or "command"),
            "energy_calibration_manifest": str(ns.calibration_manifest or ""),
            "energy_calibration_sha256": str(ns.calibration_sha256 or ""),
            # ``measurement_output_dir`` remains the command-binding field at
            # planning time.  The runner preserves it as
            # ``measurement_planned_output_dir`` and replaces it in the result
            # row with the materialized execution attempt directory.
            "measurement_output_dir": str(measurement_output_base_dir),
            "measurement_output_base_dir": str(measurement_output_base_dir),
            "measurement_output_policy": "runner_materialized_unique_attempt_child",
            "measurement_plan_attempt_id": plan_attempt_id,
            "measurement_setup_id": setup,
            "measurement_run_id": f"native_{safe_identity}",
            "measurement_requested_repeats": energy_runs_per_row,
            "measurement_profile_requested_repeats": (
                energy_profile_requested_runs_per_row
            ),
            "measurement_effective_repeats": (
                energy_effective_runs_per_row
            ),
            "measurement_repeat_expansion_applied": (
                energy_repeat_expansion_applied
            ),
            "measurement_repeat_expansion_reason": (
                energy_repeat_expansion_reason
            ),
            "measure_command": measure,
            "window_method_ab": dict(window_ab),
        })
        try:
            validate_energy_quality_admission_axes(
                energy_quality_admission,
                row=plan[-1],
            )
        except ValueError as exc:
            # Annotation drift must never change the technical denominator.
            # Seal the row as a measurement-only observation and clamp every
            # claim axis instead of deleting the already constructible row.
            downgraded = _native_runtime_observation_admission(
                energy_quality_admission,
                reason=str(exc),
            )
            plan[-1].update({
                field: downgraded[field]
                for field in (
                    "central_quality_evidence_verified",
                    "precision_quality_binding_verified",
                    "task_quality_observation_valid",
                    "accuracy_gate_pass",
                    "quality_provenance_complete",
                    "quality_claim_result_verified",
                    "diagnostic_only",
                    "screening_comparable",
                    "claim_comparable",
                    "energy_claim_eligible",
                )
            })
            plan[-1].update({
                "semantic_claim_ok": False,
                "claim_ok": False,
                "claim_eligible": False,
                "eligible_for_energy_results_import": False,
                "eligible_for_scientific_claim": False,
                "energy_quality_admission": downgraded,
                "energy_quality_admission_sha256": downgraded[
                    "admission_sha256"
                ],
                "quality_annotation_downgrade_reason": str(exc),
                **_energy_quality_result_fields(downgraded),
            })
            validate_energy_quality_admission_axes(
                downgraded,
                row=plan[-1],
            )
        markdown += [f"## {backend} / {model} / {case}", "", f"Precision: `{precision}`", f"Command file: `{shell}`", "", "```bash", measure, "```", ""]

    expected_ledger_rows = [
        row
        for key in (
            "present_expected_rows",
            "failed_expected_rows",
            "missing_expected_rows",
        )
        for row in list(expected_matrix.get(key) or [])
        if isinstance(row, Mapping)
    ]
    expected_by_identity: dict[tuple[str, ...], dict[str, Any]] = {}
    for expected_row in expected_ledger_rows:
        expected_by_identity.setdefault(
            _ledger_identity(expected_row),
            dict(expected_row),
        )

    # One expected Native identity must occur exactly once in the Energy
    # ledger.  Preserve additional diagnostics on the canonical exclusion
    # instead of letting repeated pairing reasons inflate the denominator.
    deduplicated_exclusions: dict[
        tuple[str, ...], dict[str, Any]
    ] = {}
    for excluded_row in excluded:
        identity = _ledger_identity(excluded_row)
        existing = deduplicated_exclusions.get(identity)
        if existing is None:
            deduplicated_exclusions[identity] = dict(excluded_row)
            continue
        reasons = [
            str(existing.get("reason") or ""),
            *list(existing.get("additional_reasons") or []),
            str(excluded_row.get("reason") or ""),
        ]
        existing["additional_reasons"] = list(dict.fromkeys(
            reason for reason in reasons if reason
            and reason != str(existing.get("reason") or "")
        ))
    excluded = list(deduplicated_exclusions.values())

    plan_identities = [_ledger_identity(row) for row in plan]
    plan_identity_set = set(plan_identities)
    excluded_identity_set = {
        _ledger_identity(row) for row in excluded
    }
    ledger_overlap_identities = sorted(
        plan_identity_set & excluded_identity_set
    )
    for identity, expected_row in expected_by_identity.items():
        if identity in plan_identity_set or identity in excluded_identity_set:
            continue
        excluded_row = dict(expected_row)
        excluded_row.update({
            "reason": (
                str(expected_row.get("failure_reason") or "")
                or str(expected_row.get("status_detail") or "")
                or "expected_matrix_row_not_selected_by_energy_plan"
            ),
            "expected_matrix_unrepresented": True,
        })
        excluded.append(excluded_row)
        excluded_identity_set.add(identity)

    ledger_identity_set = plan_identity_set | excluded_identity_set
    expected_identity_set = set(expected_by_identity)
    ledger_unexpected_identities = sorted(
        ledger_identity_set - expected_identity_set
    ) if expected_identity_set else []
    ledger_missing_identities = sorted(
        expected_identity_set - ledger_identity_set
    ) if expected_identity_set else []
    expected_matrix_presence_complete = bool(
        expected_matrix.get("row_presence_complete") is True
        or (
            int(expected_matrix.get("present_expected_row_count") or 0)
            == energy_matrix_expected_count
            and int(expected_matrix.get("missing_expected_row_count") or 0)
            == 0
        )
    )
    coverage_contract_valid = bool(
        energy_matrix_expected_count <= 0
        or (
            expected_matrix_presence_complete
            and
            len(plan) + len(excluded)
            == energy_matrix_expected_count
            and len(plan_identities) == len(plan_identity_set)
            and len(ledger_identity_set) == energy_matrix_expected_count
            and invalid_ledger_identity_count == 0
            and not duplicate_execution_keys
            and not ledger_overlap_identities
            and not ledger_unexpected_identities
            and not ledger_missing_identities
            and (
                not expected_identity_set
                or ledger_identity_set == expected_identity_set
            )
        )
    )
    technical_measurement_contract_valid = bool(
        plan
        and len(plan_identities) == len(plan_identity_set)
        and not duplicate_execution_keys
        and all(
            identity
            and identity[0]
            != "__invalid_native_performance_identity__"
            for identity in plan_identities
        )
    )
    if technical_measurement_contract_valid and not coverage_contract_valid:
        # Coverage controls cohort/scientific claims, never whether an exact,
        # replayable successful row may be observed.  Re-seal every included
        # row as diagnostic-only before the collector can see it.
        for planned_row in plan:
            admission = planned_row.get("energy_quality_admission")
            if not isinstance(admission, Mapping):
                admission = _fallback_native_runtime_observation_admission(
                    planned_row,
                    setup=str(planned_row.get("setup_id") or ""),
                    reason="incomplete_expected_native_matrix",
                )
            else:
                admission = _native_runtime_observation_admission(
                    admission,
                    reason="incomplete_expected_native_matrix",
                )
            planned_row.update({
                "diagnostic_only": True,
                "semantic_claim_ok": False,
                "claim_ok": False,
                "screening_comparable": False,
                "claim_comparable": False,
                "claim_eligible": False,
                "energy_claim_eligible": False,
                "eligible_for_energy_results_import": False,
                "eligible_for_scientific_claim": False,
                "scientific_coverage_complete": False,
                "quality_annotation_downgrade_reason": (
                    "incomplete_expected_native_matrix"
                ),
                "energy_quality_admission": admission,
                "energy_quality_admission_sha256": admission[
                    "admission_sha256"
                ],
                **_energy_quality_result_fields(admission),
            })
            validate_energy_quality_admission_axes(
                admission, row=planned_row,
            )
    preflight_status = (
        "passed"
        if technical_measurement_contract_valid
        else "blocked_technical_measurement_contract_invalid"
        if plan
        else "blocked_no_runtime_constructible_rows"
        if ns.measure_all_runtime_successful
        else "blocked_no_complete_pairs"
    )
    diagnostic_measurement_row_count = sum(
        1 for row in plan if row.get("diagnostic_only") is True
    )
    quality_admitted_row_count = sum(
        1
        for row in plan
        if str(
            (row.get("energy_quality_admission") or {}).get(
                "admission_scope"
            )
        ) == "native_energy"
    )
    preflight = {
        "schema": "onnx-splitpoint/native-energy-plan-preflight",
        "schema_version": 1,
        "status": preflight_status,
        "ok": technical_measurement_contract_valid,
        "measurement_start_allowed": bool(
            technical_measurement_contract_valid
        ),
        "technical_measurement_contract_valid": (
            technical_measurement_contract_valid
        ),
        "scientific_coverage_contract_valid": coverage_contract_valid,
        "energy_evidence_tier": "screening" if ns.screening_energy else (
            "window_probe" if ns.screening_window_probe else
            "smoke_diagnostic" if ns.smoke_diagnostic else "final_claim"
        ),
        "energy_tier": "screening" if ns.screening_energy else (
            "window_probe" if ns.screening_window_probe else
            "smoke_diagnostic" if ns.smoke_diagnostic else "final_claim"
        ),
        "screening_only": screening_only,
        "diagnostic_only": bool(
            ns.screening_energy
            or ns.screening_window_probe
            or ns.smoke_diagnostic
        ),
        "final_all_split_energy_required": bool(
            ns.final_all_split_energy
        ),
        **({
            "measure_all_runtime_successful": True,
            "measurement_admission_policy": (
                "all_runtime_successful_constructible_native_rows"
            ),
        } if ns.measure_all_runtime_successful else {}),
        "energy_matrix_expected_count": energy_matrix_expected_count,
        "energy_plan_included_count": len(plan),
        "energy_plan_excluded_count": len(excluded),
        "energy_plan_excluded_rows_count": len(excluded),
        "energy_plan_coverage_contract_valid": (
            coverage_contract_valid
        ),
        "native_expected_matrix_presence_complete": (
            expected_matrix_presence_complete
        ),
        "energy_plan_ledger_overlap_identities": [
            list(identity) for identity in ledger_overlap_identities
        ],
        "energy_plan_ledger_unexpected_identities": [
            list(identity) for identity in ledger_unexpected_identities
        ],
        "energy_plan_ledger_missing_identities": [
            list(identity) for identity in ledger_missing_identities
        ],
        "energy_plan_invalid_membership_identity_count": (
            invalid_ledger_identity_count
        ),
        "energy_quality_admission_required": True,
        "energy_profile_requested_runs_per_row": (
            energy_profile_requested_runs_per_row
        ),
        "energy_effective_runs_per_row": (
            energy_effective_runs_per_row
        ),
        "energy_repeat_expansion_applied": (
            energy_repeat_expansion_applied
        ),
        "energy_repeat_expansion_reason": (
            energy_repeat_expansion_reason
        ),
        "energy_planned_matrix_coverage_fraction": (
            float(len(plan)) / float(energy_matrix_expected_count)
            if energy_matrix_expected_count > 0 else 0.0
        ),
        "energy_matrix_coverage_fraction": (
            float(len(plan)) / float(energy_matrix_expected_count)
            if energy_matrix_expected_count > 0 else 0.0
        ),
        "energy_matrix_coverage_fraction_deprecated_alias_for": (
            "energy_planned_matrix_coverage_fraction"
        ),
        "source_ok_rows": len(source_rows),
        "semantically_admitted_rows": (
            quality_admitted_row_count
            if ns.measure_all_runtime_successful
            else len(admitted)
        ),
        **({
            "runtime_measurement_admitted_rows": len(admitted),
            "quality_admitted_rows": quality_admitted_row_count,
            "diagnostic_measurement_rows": (
                diagnostic_measurement_row_count
            ),
        } if ns.measure_all_runtime_successful else {}),
        "planned_rows": len(plan),
        "pair_count": len(paired_groups),
        "blocked_reason": (
            ""
            if technical_measurement_contract_valid
            else "technical_measurement_contract_invalid"
            if plan
            else "no_runtime_constructible_native_rows"
            if ns.measure_all_runtime_successful
            else "no_complete_setup_local_native_energy_pairs"
        ),
    }
    payload = {
        "schema": (
            "onnx-splitpoint/window-method-validation-probe-plan"
            if ns.screening_window_probe
            else "onnx_splitpoint_native_energy_plan_v60i"
        ),
        "schema_version": 1,
        "screening_only": screening_only,
        "screening_energy": bool(ns.screening_energy),
        **({
            "measure_all_runtime_successful": True,
            "measurement_admission_policy": (
                "all_runtime_successful_constructible_native_rows"
            ),
        } if ns.measure_all_runtime_successful else {}),
        "final_all_split_energy_required": bool(
            ns.final_all_split_energy
        ),
        "final_all_split_energy_complete": bool(
            ns.final_all_split_energy
            and coverage_contract_valid
            and energy_matrix_expected_count > 0
            and len(plan) == energy_matrix_expected_count
            and len(excluded) == 0
        ),
        "native_expected_matrix_status": expected_matrix_status,
        "native_expected_matrix": expected_matrix,
        "detection_claim_exclusions_status": (
            detection_exclusions_status
        ),
        "detection_claim_exclusions_contract_sha256": (
            detection_exclusions_contract_sha256
        ),
        "detection_claim_exclusions": list(
            detection_exclusions.values()
        ),
        "energy_matrix_expected_count": energy_matrix_expected_count,
        "energy_plan_included_count": len(plan),
        "energy_plan_excluded_count": len(excluded),
        "energy_plan_excluded_rows_count": len(excluded),
        "energy_plan_coverage_contract_valid": (
            coverage_contract_valid
        ),
        "technical_measurement_contract_valid": (
            technical_measurement_contract_valid
        ),
        "scientific_coverage_contract_valid": coverage_contract_valid,
        "native_expected_matrix_presence_complete": (
            expected_matrix_presence_complete
        ),
        "energy_plan_ledger_overlap_identities": [
            list(identity) for identity in ledger_overlap_identities
        ],
        "energy_plan_ledger_unexpected_identities": [
            list(identity) for identity in ledger_unexpected_identities
        ],
        "energy_plan_ledger_missing_identities": [
            list(identity) for identity in ledger_missing_identities
        ],
        "energy_plan_invalid_membership_identity_count": (
            invalid_ledger_identity_count
        ),
        "energy_quality_admission_required": True,
        "energy_planned_matrix_coverage_fraction": (
            float(len(plan)) / float(energy_matrix_expected_count)
            if energy_matrix_expected_count > 0 else 0.0
        ),
        "energy_matrix_coverage_fraction": (
            float(len(plan)) / float(energy_matrix_expected_count)
            if energy_matrix_expected_count > 0 else 0.0
        ),
        "energy_matrix_coverage_fraction_deprecated_alias_for": (
            "energy_planned_matrix_coverage_fraction"
        ),
        "energy_evidence_tier": (
            "screening" if ns.screening_energy else
            "window_probe" if ns.screening_window_probe else
            "smoke_diagnostic" if ns.smoke_diagnostic else "final_claim"
        ),
        "energy_tier": (
            "screening" if ns.screening_energy else
            "window_probe" if ns.screening_window_probe else
            "smoke_diagnostic" if ns.smoke_diagnostic else "final_claim"
        ),
        "smoke_diagnostic": bool(ns.smoke_diagnostic),
        "diagnostic_only": bool(
            ns.screening_energy
            or ns.screening_window_probe
            or ns.smoke_diagnostic
        ),
        "eligible_for_energy_results_import": (
            False
            if screening_only or ns.smoke_diagnostic or historical_diagnostic_only
            else None
        ),
        "claim_eligible": (
            False if screening_only or ns.smoke_diagnostic else None
        ),
        "energy_claim_eligible": (
            False if screening_only or ns.smoke_diagnostic else None
        ),
        "eligible_for_scientific_claim": (
            False if screening_only or ns.smoke_diagnostic else None
        ),
        "summary": str(summary_path),
        **hardware_registry_binding,
        "validation_summary": str(validation_path) if validation_path else "",
        "validation_summary_status": validation_summary_status,
        "native_split_quality_authority": split_quality_authority,
        "historical_diagnostic_only": historical_diagnostic_only,
        "eligible_for_scientific_claim": (
            False if historical_diagnostic_only or screening_only or ns.smoke_diagnostic
            else None
        ),
        "scientific_claim_exclusion_reasons": (
            ([diagnostic_claim_exclusion_reason] if historical_diagnostic_only else [])
            + (["smoke_diagnostic_only"] if ns.smoke_diagnostic else [])
            + (["development_screening_energy_only"] if ns.screening_energy else [])
        ),
        "preflight_status": preflight_status,
        "preflight": preflight,
        "duration_s": duration_s,
        "source_ok_rows": len(source_rows),
        "semantically_admitted_rows": (
            quality_admitted_row_count
            if ns.measure_all_runtime_successful
            else len(admitted)
        ),
        **({
            "runtime_measurement_admitted_rows": len(admitted),
            "quality_admitted_rows": quality_admitted_row_count,
            "diagnostic_measurement_rows": (
                diagnostic_measurement_row_count
            ),
        } if ns.measure_all_runtime_successful else {}),
        "deduplicated_count": len(deduplicated),
        "excluded_rows": excluded,
        "deduplicated_rows": deduplicated,
        "pairing_policy": (
            "screening_probe_one_successful_native_target_no_pairing"
            if ns.screening_window_probe
            else "measurement_independent_quality_pairing_posthoc"
            if ns.measure_all_runtime_successful
            else "allow_unpaired" if ns.allow_unpaired
            else "require_unique_semantic_split_vendor_full_and_tensorrt_full_same_setup_model_context"
        ),
        "pairing_identity_fields": [
            "setup_id", "model", "baseline_backend", "comparison_backend",
            "task", "direction", "model_sha256",
            "validation_input_or_image_sha256", "prepared_feed_task",
            "prepared_feed_preprocess_mode",
            "prepared_feed_letterbox_pad_value",
            "prepared_feed_source_image_sha256",
            "pipeline_contract_sha256_when_frozen",
            "pipeline_preprocessing_sha256_when_frozen",
            "pipeline_decoder_sha256_when_frozen",
            "pipeline_nms_sha256_when_frozen",
            "comparison_output_endpoint_id_for_detection",
            "comparison_endpoint_contract_hash_for_detection",
            "comparison_endpoint_stage_for_detection",
        ],
        "comparison_precision_match_required": False,
        "full_runtime_precision_match_required": False,
        "paired_only": (
            False
            if (
                ns.screening_window_probe
                or ns.measure_all_runtime_successful
            )
            else not bool(ns.allow_unpaired)
        ),
        "pair_count": len(paired_groups),
        "paired_groups": paired_groups,
        "paired_missing_rows": paired_missing,
        **({
            "quality_pairing_posthoc_excluded_rows": (
                quality_pairing_posthoc_excluded_rows
            ),
        } if ns.measure_all_runtime_successful else {}),
        "energy_profile_requested_runs_per_row": (
            energy_profile_requested_runs_per_row
        ),
        "energy_effective_runs_per_row": (
            energy_effective_runs_per_row
        ),
        "energy_repeat_expansion_applied": (
            energy_repeat_expansion_applied
        ),
        "energy_repeat_expansion_reason": (
            energy_repeat_expansion_reason
        ),
        "energy_runs_per_row": energy_runs_per_row,
        "measurement_plan_attempt_id": plan_attempt_id,
        "window_method_ab": dict(window_ab),
        "require_runtime_work_units": bool(ns.require_runtime_work_units),
        "require_command_window_alignment": bool(ns.require_command_window_alignment),
        "physical_scope": str(ns.physical_scope or ""),
        "window_label": str(ns.window_label or "command"),
        "calibration_manifest": str(ns.calibration_manifest or ""),
        "calibration_sha256": str(ns.calibration_sha256 or ""),
        "pipeline_contract_manifest": str(ns.pipeline_contract_manifest or ""),
        "pipeline_contract_sha256": str(ns.pipeline_contract_sha256 or ""),
        "pipeline_contract_manifest_status": contract_manifest_status,
        "model_hash_map": str(ns.model_hash_map or ""),
        "model_hash_map_sha256": str(ns.model_hash_map_sha256 or ""),
        "model_hash_map_status": model_hash_map_status,
        "rows": plan,
    }
    (out / "native_producer_energy_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "native_producer_energy_plan.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"ok": True, "rows": len(plan), "excluded": len(excluded), "deduplicated": len(deduplicated), "json": str(out / "native_producer_energy_plan.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
