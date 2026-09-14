from __future__ import annotations

"""Plan and validate setup-local TensorRT quality companions.

The management-side quality join is keyed by the physical setup id.  A single
model-wide TensorRT result produced on one Jetson therefore cannot stand in for
the TensorRT producer on the two other Jetsons.  This module is deliberately
free of SSH/process code: it builds the complete dispatch contract before a
remote worker, upload, or connection is started.
"""

from typing import Any, Dict, List, Mapping, Sequence

from ..native_full_quality import enabled_run_profiles
from .full_only_quality_canary import resolve_full_only_quality_canary
from .hardware_matrix import canon_accelerator, matrix_for_runtime


SCHEMA = "onnx-splitpoint/setup-local-tensorrt-quality-dispatch"
SCHEMA_VERSION = 1
_TRT_RUN_ID = "ort_tensorrt"
_TRT_QUALITY_ID_BY_PRODUCER = {
    "hailo8": "tensorrt_at_hailo8_full",
    "hailo10h": "tensorrt_at_hailo10h_full",
    "deepx": "tensorrt_at_deepx_m1_full",
}
_CPU_ALIASES = {
    "cpu", "cpu_ort", "ort_cpu", "onnxruntime_cpu", "cpu_onnxruntime",
}


def _token(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in (
            "provider", "backend", "target", "accelerator", "hw_arch",
            "runtime", "device",
        ):
            candidate = str(value.get(key) or "").strip()
            if candidate:
                return _token(candidate)
        return ""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "trt": "tensorrt", "tensor_rt": "tensorrt",
        "ort_tensorrt": "tensorrt", "tensorrt_executionprovider": "tensorrt",
        "cuda": "cuda_ort", "ort_cuda": "cuda_ort", "gpu": "cuda_ort",
        "cpu": "cpu_ort", "ort_cpu": "cpu_ort", "cpu_onnxruntime": "cpu_ort",
        "hailo10": "hailo10h", "hailo10n": "hailo10h",
        "deepx": "deepx_m1", "dx_m1": "deepx_m1", "dxm1": "deepx_m1",
    }
    return aliases.get(text, text)


def _run_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("id") or row.get("name") or row.get("run_id")
        or row.get("backend") or ""
    ).strip()


def _enabled_rows(value: Any) -> List[Dict[str, Any]]:
    return [dict(row) for row in enabled_run_profiles(value)]


def _row_backend_tokens(row: Mapping[str, Any]) -> List[str]:
    tokens: List[str] = []
    for key in (
        "full", "full_backend", "full_provider", "provider", "backend",
        "stage1", "stage2",
    ):
        if key not in row:
            continue
        token = _token(row.get(key))
        if token:
            tokens.append(token)
    return tokens


def _is_cpu_reference(row: Mapping[str, Any]) -> bool:
    rid = _run_id(row).strip().lower().replace("-", "_")
    if rid in _CPU_ALIASES:
        return True
    return any(token == "cpu_ort" for token in _row_backend_tokens(row)) or bool(
        row.get("semantic_reference_only") or row.get("canonical_cpu_reference")
    )


def _is_exact_trt_recipe(row: Mapping[str, Any]) -> bool:
    """Accept the real ORT/TensorRT Full recipe, not an id-only disguise."""

    if _run_id(row).strip().lower().replace("-", "_") != _TRT_RUN_ID:
        return False
    run_type = str(row.get("type") or row.get("kind") or "").strip().lower().replace("-", "_")
    if run_type and run_type not in {
        "same_backend_reference", "onnxruntime", "ort", "matrix", "split",
    }:
        return False
    tokens = _row_backend_tokens(row)
    # The logical id is not enough: a row renamed from ORT-CPU must fail closed.
    return bool(tokens) and all(token == "tensorrt" for token in tokens)


def _producer_for_row(row: Mapping[str, Any]) -> str:
    tokens = [_token(row.get(key)) for key in ("full", "stage1", "stage2", "id")]
    for token in tokens:
        acc = canon_accelerator(token)
        if acc == "hailo8":
            return "hailo8"
        if str(acc).startswith("hailo10"):
            return "hailo10h"
        if acc == "deepx_m1":
            return "deepx"
    return ""


def _producer_for_target(target: Mapping[str, Any]) -> str:
    acc = canon_accelerator(
        target.get("accelerator") or target.get("backend")
        or target.get("provider") or target.get("id")
    )
    if acc == "hailo8":
        return "hailo8"
    if str(acc).startswith("hailo10"):
        return "hailo10h"
    if acc == "deepx_m1":
        return "deepx"
    return ""


def _standard_trt_quality_identity(
    *, producer: str, setup_id: str,
) -> Dict[str, Any]:
    """Return the concrete setup-local TRT companion identity.

    Standard/Quality runs do not carry the explicit ``quality_canary`` block
    used by Full-only canaries.  They nevertheless execute the same physical
    TensorRT Full quality producer on every selected Jetson.  Materialise that
    endpoint here, while the scheduler still owns both the producer and the
    physical setup, instead of asking the remote suite to infer an identity
    from a logical run id.
    """

    quality_id = _TRT_QUALITY_ID_BY_PRODUCER.get(str(producer or ""), "")
    setup = str(setup_id or "").strip()
    if not quality_id or not setup:
        return {}
    return {
        "id": quality_id,
        "source_run_id": "native_full_tensorrt",
        "run_id": "native_full_tensorrt",
        "dispatch_run_id": _TRT_RUN_ID,
        "setup_id": setup,
        "backend": "tensorrt",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }


def _canary_trt_quality_identity(
    rows: Sequence[Mapping[str, Any]], *, setup_id: str,
) -> Dict[str, Any]:
    """Select the sole explicit TensorRT companion for one canary setup."""

    setup = str(setup_id or "").strip()
    matches: List[Dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        source_run_id = str(
            row.get("source_run_id") or row.get("run_id") or ""
        ).strip().lower().replace("-", "_")
        dispatch_run_id = str(
            row.get("dispatch_run_id") or row.get("run_id") or ""
        ).strip().lower().replace("-", "_")
        if (
            str(row.get("setup_id") or "").strip() == setup
            and _token(row.get("backend")) == "tensorrt"
            and source_run_id == "native_full_tensorrt"
            and dispatch_run_id == _TRT_RUN_ID
        ):
            matches.append(row)
    return matches[0] if len(matches) == 1 else {}


def _central_management_quality(profile: Mapping[str, Any]) -> bool:
    quality = profile.get("quality_gate") if isinstance(profile.get("quality_gate"), Mapping) else {}
    statistics = quality.get("statistics") if isinstance(quality.get("statistics"), Mapping) else {}
    execution = quality.get("execution") if isinstance(quality.get("execution"), Mapping) else {}
    value = str(
        statistics.get("execution_location") or quality.get("execution_location")
        or execution.get("location") or ""
    ).strip().lower().replace("-", "_")
    return value in {
        "central_management", "management", "management_node", "central",
        "central_cpu",
    }


def _native_setup_override(profile: Mapping[str, Any], producer: str) -> str:
    native = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    remotes = native.get("remotes") if isinstance(native.get("remotes"), Mapping) else {}
    aliases = {
        "hailo8": ("hailo8", "hailo8_to_trt"),
        "hailo10h": ("hailo10h", "hailo10", "hailo10h_to_trt"),
        "deepx": ("deepx", "deepx_m1", "deepx_to_trt"),
    }.get(producer, (producer,))
    for alias in aliases:
        value = remotes.get(alias)
        if isinstance(value, Mapping):
            setup_id = str(value.get("setup_id") or value.get("hardware_setup_id") or "").strip()
            if setup_id:
                return setup_id
    return ""


def build_setup_local_tensorrt_quality_dispatch(
    profile: Mapping[str, Any],
    *,
    hardware_targets: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return the complete setup-local dispatch contract without side effects."""

    profile_payload = dict(profile or {}) if isinstance(profile, Mapping) else {}
    rows = [dict(row) for row in plan_rows if isinstance(row, Mapping)]
    canary = resolve_full_only_quality_canary(
        profile_payload, plan_rows=rows,
    )
    if canary.get("enabled"):
        errors = list(canary.get("errors") or [])
        targets_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for raw in hardware_targets:
            if not isinstance(raw, Mapping) or not bool(raw.get("enabled", True)):
                continue
            target = dict(raw)
            setup_id = str(target.get("id") or "").strip()
            if setup_id:
                targets_by_id.setdefault(setup_id, []).append(target)
        expected = [
            dict(row) for row in list(
                canary.get("expected_full_quality_results") or []
            ) if isinstance(row, Mapping)
        ]
        for row in expected:
            setup_id = str(row.get("setup_id") or "")
            targets = targets_by_id.get(setup_id, [])
            if len(targets) != 1:
                errors.append(
                    f"quality_canary_setup_target_count_not_one:"
                    f"{setup_id}:{len(targets)}"
                )
                continue
            target_backend = _producer_for_target(targets[0])
            expected_backend = str(row.get("backend") or "")
            expected_producer = (
                "hailo8" if expected_backend == "hailo8"
                else "hailo10h" if expected_backend == "hailo10h"
                else "deepx" if expected_backend == "deepx_m1"
                else target_backend if expected_backend == "tensorrt"
                else ""
            )
            if not target_backend or target_backend != expected_producer:
                errors.append(
                    "quality_canary_setup_backend_mismatch:"
                    f"{row.get('id')}:{setup_id}:{expected_backend}:"
                    f"{target_backend or 'unknown'}"
                )
        dispatches: List[Dict[str, Any]] = []
        for setup_id in dict.fromkeys(
            str(row.get("setup_id") or "") for row in expected
            if str(row.get("setup_id") or "")
        ):
            setup_rows = [
                row for row in expected
                if str(row.get("setup_id") or "") == setup_id
            ]
            run_ids = list(dict.fromkeys(
                str(row.get("run_id") or "") for row in setup_rows
                if str(row.get("run_id") or "")
            ))
            target_rows = targets_by_id.get(setup_id, [])
            producer = (
                _producer_for_target(target_rows[0])
                if len(target_rows) == 1 else ""
            )
            trt_identity = _canary_trt_quality_identity(
                setup_rows, setup_id=setup_id,
            )
            if not trt_identity:
                errors.append(
                    "quality_canary_setup_tensorrt_identity_count_not_one:"
                    f"{setup_id}"
                )
            dispatches.append({
                "producer": producer,
                "setup_id": setup_id,
                "run_ids": run_ids,
                "quality_only_run_ids": list(run_ids),
                "quality_ids": [
                    str(row.get("id") or "") for row in setup_rows
                ],
                "expected_full_quality_identities": [
                    identity for identity in list(
                        canary.get("expected_full_quality_identities") or []
                    ) if isinstance(identity, Mapping)
                    and str(identity.get("setup_id") or "") == setup_id
                ],
                "tensorrt_execution_role": "full_quality_only",
                "performance_claims_emitted": False,
                "quality_companion_required": True,
                "quality_companion_endpoint_id": str(
                    trt_identity.get("id") or ""
                ),
                "quality_companion_identity": trt_identity,
            })
        unique_errors = list(dict.fromkeys(str(value) for value in errors))
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "ok": not unique_errors,
            "status": "ready" if not unique_errors else "blocked",
            "errors": unique_errors,
            "execution_scope": "full_only",
            "quality_canary": canary,
            "tensorrt_run_id": _TRT_RUN_ID,
            "performance_owner_producer": "",
            "performance_owner_setup_id": "",
            "performance_claims_emitted": False,
            "cpu_reference_execution_location": "central_management",
            "central_cpu_reference_run_ids": [
                _run_id(row) for row in rows if _is_cpu_reference(row)
            ],
            "remote_cpu_reference_run_ids": [],
            "generic_setup_ids": {
                str(row.get("producer") or ""): str(row.get("setup_id") or "")
                for row in dispatches if row.get("producer")
            },
            "native_setup_ids": {},
            "setup_dispatches": dispatches,
            "expected_full_quality_identities": list(
                canary.get("expected_full_quality_identities") or []
            ),
        }
    errors: List[str] = []

    trt_candidates = [row for row in rows if _run_id(row).strip().lower().replace("-", "_") == _TRT_RUN_ID]
    exact_trt = [row for row in trt_candidates if _is_exact_trt_recipe(row)]
    if len(trt_candidates) != 1:
        errors.append("setup_local_tensorrt_recipe_count_not_one")
    if len(exact_trt) != 1 or len(exact_trt) != len(trt_candidates):
        errors.append("setup_local_tensorrt_recipe_not_executable_trt_full")

    cpu_ids = [_run_id(row) for row in rows if _is_cpu_reference(row)]
    if not _central_management_quality(profile_payload):
        errors.append("cpu_reference_not_bound_to_central_management")

    requested_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        producer = _producer_for_row(row)
        if producer:
            requested_rows.setdefault(producer, []).append(row)

    targets_by_producer: Dict[str, List[Dict[str, Any]]] = {}
    for raw in hardware_targets:
        if not isinstance(raw, Mapping) or not bool(raw.get("enabled", True)):
            continue
        target = dict(raw)
        producer = _producer_for_target(target)
        if producer and producer in requested_rows:
            targets_by_producer.setdefault(producer, []).append(target)

    for producer in sorted(requested_rows):
        if len(targets_by_producer.get(producer, [])) != 1:
            errors.append(f"setup_local_target_count_not_one:{producer}")

    generic_setup_ids: Dict[str, str] = {}
    native_setup_ids: Dict[str, str] = {}
    for producer, targets in targets_by_producer.items():
        if len(targets) != 1:
            continue
        setup_id = str(targets[0].get("id") or "").strip()
        if not setup_id:
            errors.append(f"setup_local_target_id_missing:{producer}")
            continue
        generic_setup_ids[producer] = setup_id
        declared = {
            str(row.get("hardware_setup_id") or row.get("hardware_setup") or "").strip()
            for row in requested_rows.get(producer, [])
            if str(row.get("hardware_setup_id") or row.get("hardware_setup") or "").strip()
        }
        if len(declared) > 1 or (declared and declared != {setup_id}):
            errors.append(f"generic_run_profile_setup_id_mismatch:{producer}")
        native_override = _native_setup_override(profile_payload, producer)
        native_setup_ids[producer] = native_override or setup_id
        if native_override and native_override != setup_id:
            errors.append(f"generic_native_setup_id_mismatch:{producer}")

    owner_producer = "deepx" if "deepx" in generic_setup_ids else (
        next(iter(generic_setup_ids), "")
    )
    owner_setup_id = generic_setup_ids.get(owner_producer, "")
    if "deepx" in requested_rows and owner_producer != "deepx":
        errors.append("deepx_tensorrt_performance_owner_missing")

    dispatches: List[Dict[str, Any]] = []
    for producer in ("hailo8", "hailo10h", "deepx"):
        setup_id = generic_setup_ids.get(producer)
        if not setup_id:
            continue
        vendor_ids: List[str] = []
        for row in requested_rows.get(producer, []):
            rid = _run_id(row)
            if rid and rid not in vendor_ids:
                vendor_ids.append(rid)
        is_owner = setup_id == owner_setup_id
        run_ids = list(vendor_ids)
        if _TRT_RUN_ID not in run_ids:
            run_ids.append(_TRT_RUN_ID)
        trt_identity = _standard_trt_quality_identity(
            producer=producer, setup_id=setup_id,
        )
        if not trt_identity:
            errors.append(
                f"setup_local_tensorrt_quality_identity_missing:{producer}"
            )
        dispatches.append({
            "producer": producer,
            "setup_id": setup_id,
            "run_ids": run_ids,
            "tensorrt_run_id": _TRT_RUN_ID,
            "tensorrt_execution_role": (
                "full_performance_owner" if is_owner else "full_quality_only"
            ),
            "quality_only_run_ids": [] if is_owner else [_TRT_RUN_ID],
            "performance_claims_emitted": bool(is_owner),
            "quality_companion_required": True,
            "quality_companion_endpoint_id": str(
                trt_identity.get("id") or ""
            ),
            "quality_companion_identity": trt_identity,
        })

    if dispatches and sum(
        row["tensorrt_execution_role"] == "full_performance_owner"
        for row in dispatches
    ) != 1:
        errors.append("setup_local_tensorrt_performance_owner_count_not_one")
    if any(
        any(str(run_id).strip().lower().replace("-", "_") in _CPU_ALIASES for run_id in row["run_ids"])
        for row in dispatches
    ):
        errors.append("cpu_reference_leaked_into_remote_dispatch")

    unique_errors = list(dict.fromkeys(errors))
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "ok": not unique_errors,
        "status": "ready" if not unique_errors else "blocked",
        "errors": unique_errors,
        "tensorrt_run_id": _TRT_RUN_ID,
        "performance_owner_producer": owner_producer,
        "performance_owner_setup_id": owner_setup_id,
        "cpu_reference_execution_location": "central_management",
        "central_cpu_reference_run_ids": cpu_ids,
        "remote_cpu_reference_run_ids": [],
        "generic_setup_ids": generic_setup_ids,
        "native_setup_ids": native_setup_ids,
        "setup_dispatches": dispatches,
    }


def validate_setup_local_tensorrt_quality_dispatch(
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a profile before remote staging, upload, threads, or SSH.

    The public wrapper-facing entry point resolves the configured physical
    matrix once and delegates to the side-effect-free contract builder.  It
    returns a structured blocked result instead of silently repairing an
    ambiguous profile.
    """

    profile_payload = dict(profile or {}) if isinstance(profile, Mapping) else {}
    try:
        targets = matrix_for_runtime(profile_payload)
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "status": "blocked",
            "errors": [f"hardware_target_resolution_failed:{type(exc).__name__}:{exc}"],
            "setup_dispatches": [],
        }
    return build_setup_local_tensorrt_quality_dispatch(
        profile_payload,
        hardware_targets=targets,
        plan_rows=_enabled_rows(profile_payload.get("run_profiles")),
    )


__all__ = [
    "build_setup_local_tensorrt_quality_dispatch",
    "validate_setup_local_tensorrt_quality_dispatch",
]
