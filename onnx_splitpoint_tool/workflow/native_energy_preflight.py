from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def _producer(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("native_full_"):
        token = token[len("native_full_"):]
    return {
        "hailo10": "hailo10h",
        "hailo10_to_trt": "hailo10h",
        "hailo10h_to_trt": "hailo10h",
        "hailo10_to_tensorrt": "hailo10h",
        "hailo10h_to_tensorrt": "hailo10h",
        "hailo8_to_trt": "hailo8",
        "hailo8_to_tensorrt": "hailo8",
        "deepx_m1": "deepx",
        "deepx_to_trt": "deepx",
        "deepx_to_tensorrt": "deepx",
        "deepx_m1_to_trt": "deepx",
        "deepx_m1_to_tensorrt": "deepx",
    }.get(token, token)


def _full_backend(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("native_full_"):
        token = token[len("native_full_"):]
    return {
        "hailo10": "hailo10h",
        "deepx_m1": "deepx",
        "trt": "tensorrt",
        "ort_tensorrt": "tensorrt",
    }.get(token, token)


def build_native_energy_preflight(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    energy_requested: bool,
    energy_evidence_tier: str,
    strict_requested: bool,
    validation_requested: bool,
    full_baselines_enabled: bool,
    setup_ids_by_producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the theoretical Native Energy matrix before remote execution.

    This phase intentionally uses only the already-resolved execution plan. It
    does not inspect runtime results and therefore cannot make a scientific
    claim.  P0.3 limits this gate to structural constructibility; validation,
    Full pairing and claim prerequisites are recorded as non-blocking
    annotations for downstream result interpretation.
    """

    setup_ids = {
        _producer(key): str(value or "").strip()
        for key, value in dict(setup_ids_by_producer or {}).items()
    }
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "split_rows": [],
            "vendor_full_rows": [],
            "tensorrt_full_rows": [],
        }
    )
    structural_errors: list[str] = []
    normalized_row_count = 0

    for index, raw in enumerate(expected_rows):
        row = dict(raw)
        mode = str(row.get("execution_mode") or row.get("role") or "").strip().lower()
        backend = str(row.get("backend") or "").strip().lower()
        producer = _producer(
            row.get("backend_key")
            or row.get("producer")
            or row.get("comparison_backend")
            or backend
        )
        model = str(row.get("model") or row.get("model_id") or "").strip()
        setup_id = str(row.get("setup_id") or setup_ids.get(producer) or "").strip()
        if not model:
            structural_errors.append(f"row_{index}:model_missing")
            continue
        if not producer:
            structural_errors.append(f"row_{index}:producer_missing")
            continue
        if not setup_id:
            structural_errors.append(
                f"row_{index}:setup_id_missing:{producer}:{model}"
            )
            continue

        normalized_row_count += 1
        group = groups[(setup_id, producer, model)]
        compact = {
            "backend": backend,
            "case": str(row.get("case") or row.get("case_id") or ""),
            "precision": str(row.get("precision") or ""),
            "variant": str(row.get("variant") or row.get("variant_id") or ""),
        }
        if mode == "native_split" or (
            mode != "native_full_baseline"
            and not backend.startswith("native_full_")
        ):
            group["split_rows"].append(compact)
            continue
        full = _full_backend(backend)
        if full == "tensorrt":
            group["tensorrt_full_rows"].append(compact)
        elif full == producer:
            group["vendor_full_rows"].append(compact)

    setup_model_rows: list[dict[str, Any]] = []
    theoretical_pair_count = 0
    for (setup_id, producer, model), group in sorted(groups.items()):
        split_count = len(group["split_rows"])
        vendor_count = len(group["vendor_full_rows"])
        tensorrt_count = len(group["tensorrt_full_rows"])
        row_errors: list[str] = []
        if split_count == 0:
            row_errors.append("split_path_missing")
        if vendor_count == 0:
            row_errors.append("vendor_full_path_missing")
        if tensorrt_count == 0:
            row_errors.append("tensorrt_full_path_missing")
        if not validation_requested:
            row_errors.append("validation_not_requested")
        pair_count = (
            split_count
            if split_count > 0
            and vendor_count > 0
            and tensorrt_count > 0
            else 0
        )
        theoretical_pair_count += pair_count
        setup_model_rows.append({
            "setup_id": setup_id,
            "producer": producer,
            "model": model,
            "split_path_count": split_count,
            "vendor_full_path_count": vendor_count,
            "tensorrt_full_path_count": tensorrt_count,
            "validation_requested": bool(validation_requested),
            "theoretical_pair_count": pair_count,
            "plan_viable": bool(
                split_count or vendor_count or tensorrt_count
            ),
            "status": "passed" if not row_errors else "annotated",
            "nonblocking_annotations": row_errors,
            "errors": [],
        })

    errors = list(dict.fromkeys(structural_errors))
    nonblocking_annotations: list[str] = []
    if energy_requested:
        if not validation_requested:
            nonblocking_annotations.append("native_validation_not_requested")
        if not full_baselines_enabled:
            nonblocking_annotations.append(
                "native_full_baselines_not_enabled"
            )
        if not groups:
            errors.append("native_energy_expected_matrix_empty")
        if theoretical_pair_count == 0:
            nonblocking_annotations.append(
                "no_theoretical_setup_local_energy_pair"
            )
    errors = list(dict.fromkeys(errors))
    nonblocking_annotations = list(dict.fromkeys(
        nonblocking_annotations
    ))

    plan_viable = bool(
        not energy_requested
        or (
            bool(expected_rows)
            and not structural_errors
            and normalized_row_count == len(expected_rows)
        )
    )
    if not energy_requested:
        status = "not_requested"
    elif plan_viable:
        status = "passed"
    elif structural_errors or not groups:
        status = "blocked_structural_contradiction"
    else:
        status = "blocked_empty_native_energy_matrix"

    tier = str(energy_evidence_tier or "screening").strip() or "screening"
    return {
        "schema": "onnx-splitpoint/native-energy-phase1-preflight",
        "schema_version": 1,
        "phase": "before_remote_or_performance_streaming",
        "energy_requested": bool(energy_requested),
        "energy_evidence_tier": tier,
        "energy_tier": tier,
        "strict_requested": bool(strict_requested),
        "validation_requested": bool(validation_requested),
        "full_baselines_enabled": bool(full_baselines_enabled),
        "expected_row_count": len(expected_rows),
        "normalized_expected_row_count": normalized_row_count,
        "setup_model_rows": setup_model_rows,
        "setup_model_count": len(setup_model_rows),
        "theoretical_pair_count": theoretical_pair_count,
        "plan_viable": plan_viable,
        "status": status,
        "errors": errors,
        "nonblocking_annotations": nonblocking_annotations,
        "admission_policy": (
            "technical_structure_only_quality_pairing_and_claim_posthoc"
        ),
        "started_remote_count": 0,
        "started_performance_count": 0,
    }


def native_energy_preflight_blocks_streaming(
    preflight: Mapping[str, Any],
) -> bool:
    """Return whether an explicitly requested Energy plan must stop phase 1."""
    return bool(
        preflight.get("energy_requested") is True
        and preflight.get("plan_viable") is not True
    )
