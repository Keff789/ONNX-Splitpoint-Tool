from __future__ import annotations

"""Immutable run-scope contracts for evaluation workflows.

The scope is deliberately split into two layers:

* a symbolic run-level receipt written during profile resolution, before any
  compiler can be dispatched; and
* an exact per-model identity receipt written from the generator's final
  accepted-case list before benchmark/runtime dispatch.

The global receipt keeps the requested model/run matrix immutable before build
work starts.  The model receipt is intentionally delayed until deterministic
reject/backfill has produced the executable accepted-case list.  A later
runtime, parser, or quality failure may change only the terminal state of an
accepted identity; it must never remove that identity from the sealed scope.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "onnx-splitpoint/required-run-scope"
SCHEMA_VERSION = 3
STRICT_IDENTITY_MODE = "physical_strict_v3"
LEGACY_IDENTITY_MODE = "legacy_reprojection_v2"
_PHYSICAL_IDENTITY_FIELDS = (
    "model_id", "case_id", "run_id", "backend", "variant",
    "expected_setup_id", "measurement_endpoint", "quality_endpoint",
)
_VOLATILE_FIELDS = {
    "created_at", "created_at_utc", "sealed_at", "verified_at",
    "last_verified_at", "scope_sha256",
}


class RequiredRunScopeError(RuntimeError):
    """Raised when a sealed scope is invalid or would be changed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(payload).items()
        if str(key) not in _VOLATILE_FIELDS
    }


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def canonical_run_id(value: Any) -> str:
    token = _token(value)
    aliases = {
        "ort_cpu": "ort_cpu",
        "cpu": "ort_cpu",
        "cpu_ort": "ort_cpu",
        "tensorrt": "ort_tensorrt",
        "trt": "ort_tensorrt",
        "ort_tensorrt": "ort_tensorrt",
        "hailo8_full": "hailo8",
        "hailo_8": "hailo8",
        "hailo10h": "hailo10",
        "hailo10_full": "hailo10",
        "hailo_10": "hailo10",
        "hailo8_to_tensorrt": "hailo8_to_trt",
        "hailo10_to_trt": "hailo10_to_tensorrt",
        "deepx_full": "deepx_m1_full",
        "deepx_to_tensorrt": "deepx_m1_to_tensorrt",
        "deepx_m1_to_trt": "deepx_m1_to_tensorrt",
    }
    return aliases.get(token, token)


def _canonical_backend(value: Any) -> str:
    token = _token(value)
    aliases = {
        "cpu": "cpu_ort", "ort_cpu": "cpu_ort",
        "trt": "tensorrt", "ort_tensorrt": "tensorrt",
        "hailo10h": "hailo10", "hailo_10": "hailo10",
        "hailo8_to_trt": "hailo8_to_tensorrt",
        "hailo10_to_trt": "hailo10_to_tensorrt",
        "deepx": "deepx_m1", "deepx_full": "deepx_m1",
        "deepx_to_trt": "deepx_m1_to_tensorrt",
        "deepx_to_tensorrt": "deepx_m1_to_tensorrt",
        "deepx_m1_to_trt": "deepx_m1_to_tensorrt",
    }
    return aliases.get(token, token)


def _canonical_execution_recipe(run_id: Any) -> dict[str, Any]:
    """Return the executable recipe implied by one canonical logical run.

    A strict required-run descriptor is more than a reporting identity.  When
    the model-local generator drops a requested row after a build failure, the
    scope projection must restore an executable provider recipe rather than a
    label-only placeholder.  A placeholder without ``type``/``provider`` is
    interpreted by the remote suite as ONNX Runtime ``auto`` and can therefore
    turn a required Hailo Full observation into a TensorRT observation.

    The registry deliberately contains only the frozen canonical run family.
    Returned dictionaries are newly allocated so callers may enrich them with
    model/setup/endpoint evidence without mutating shared state.
    """

    canonical = canonical_run_id(run_id)
    hailo_full = {
        "hailo8": "hailo8",
        "hailo10": "hailo10",
    }
    if canonical in hailo_full:
        hw_arch = hailo_full[canonical]
        return {
            "id": canonical,
            "type": "hailo",
            "provider": hw_arch,
            "backend": hw_arch,
            "full": hw_arch,
            "full_provider": hw_arch,
            "hw_arch": hw_arch,
            "stage1": {"type": "hailo", "hw_arch": hw_arch},
            "stage2": {"type": "hailo", "hw_arch": hw_arch},
            "variants": ["full"],
            "same_backend_split_diagnostics_enabled": False,
        }

    split_stage1 = {
        "hailo8_to_trt": {"type": "hailo", "hw_arch": "hailo8"},
        "hailo10_to_tensorrt": {
            "type": "hailo", "hw_arch": "hailo10",
        },
        "deepx_m1_to_tensorrt": {
            "type": "deepx", "backend": "deepx_m1",
        },
    }
    if canonical in split_stage1:
        return {
            "id": canonical,
            "type": "matrix",
            "provider": "tensorrt",
            "stage1": dict(split_stage1[canonical]),
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "part2", "composed"],
        }

    if canonical == "deepx_m1_full":
        return {
            "id": canonical,
            "type": "deepx",
            "provider": "deepx_m1",
            "backend": "deepx_m1",
            "full": "deepx_m1",
            "full_provider": "deepx_m1",
            "stage1": {"type": "deepx", "backend": "deepx_m1"},
            "stage2": {"type": "deepx", "backend": "deepx_m1"},
            "variants": ["full"],
            "dxnn_path": "deepx/deepx_m1/full/model.dxnn",
            "contract_path": "deepx/deepx_m1/full/output_contract.json",
        }

    ort_provider = {
        "ort_cpu": "cpu",
        "ort_tensorrt": "tensorrt",
    }.get(canonical)
    if ort_provider:
        return {
            "id": canonical,
            "type": "onnxruntime",
            "provider": ort_provider,
            "full_provider": ort_provider,
            "stage1": {"type": "onnxruntime", "provider": ort_provider},
            "stage2": {"type": "onnxruntime", "provider": ort_provider},
        }
    return {"id": canonical} if canonical else {}


def _fill_execution_recipe(
    row: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    """Fill only absent executable fields from the canonical registry."""

    recipe = _canonical_execution_recipe(run_id)
    for field, value in recipe.items():
        if field not in row or row.get(field) in (None, ""):
            if isinstance(value, dict):
                row[field] = dict(value)
            elif isinstance(value, list):
                row[field] = list(value)
            else:
                row[field] = value
    return row


def reference_measurement_variants(run: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Project an explicit reference recipe into complete result variants.

    Missing variants retain the historical Full+Split recipe. An explicit
    Full-only diagnostic is a different requested scope, sealed before any
    build; it must not invent a missing Split after that diagnostic completes.
    Part1/Part2 alone do not represent a complete split measurement.
    """
    raw = (run or {}).get("variants")
    if not isinstance(raw, list):
        return ("full", "split")
    values = {str(value or "").strip().lower() for value in raw}
    return tuple(
        variant for variant, enabled in (
            ("full", "full" in values),
            ("split", "composed" in values),
        ) if enabled
    )


def _run_backend_variants(run_id: Any) -> list[tuple[str, str]]:
    run_id = canonical_run_id(run_id)
    mapping = {
        "ort_cpu": [("cpu_ort", "full"), ("cpu_ort", "split")],
        "ort_tensorrt": [
            ("tensorrt", "full"), ("tensorrt", "split"),
        ],
        "hailo8": [("hailo8", "full")],
        "hailo10": [("hailo10", "full")],
        "deepx_m1_full": [("deepx_m1", "full")],
        "hailo8_to_trt": [("hailo8_to_tensorrt", "split")],
        "hailo10_to_tensorrt": [
            ("hailo10_to_tensorrt", "split"),
        ],
        "deepx_m1_to_tensorrt": [
            ("deepx_m1_to_tensorrt", "split"),
        ],
        "tensorrt_to_hailo8": [
            ("tensorrt_to_hailo8", "split"),
        ],
        "tensorrt_to_hailo10": [
            ("tensorrt_to_hailo10", "split"),
        ],
        "tensorrt_to_deepx_m1": [
            ("tensorrt_to_deepx_m1", "split"),
        ],
    }
    return list(mapping.get(run_id) or [])


def _endpoint_contract(
    run_id: Any, task: Any, variant: Any,
) -> tuple[str, str, str]:
    run_id = canonical_run_id(run_id)
    task = _token(task)
    variant = _token(variant)
    if not _run_backend_variants(run_id):
        return "", "", ""
    contract_id = f"canonical_run_endpoint_v1:{run_id}:{task}:{variant}"
    if task == "classification":
        return (
            "classification_logits", "classification_logits", contract_id,
        )
    if task == "detection":
        if variant == "split":
            return "p2_output", "completed_detection", contract_id
        if variant == "full":
            # The generic CPU/TRT runner times session.run only.  Its later
            # quality postprocessor is outside that interval; dedicated native
            # completed-detection producers retain their separate requirement.
            if run_id in {"ort_cpu", "ort_tensorrt"}:
                return (
                    "raw_model_outputs", "completed_detection",
                    f"canonical_run_endpoint_v2:{run_id}:{task}:{variant}",
                )
            return (
                "completed_detection", "completed_detection", contract_id,
            )
    return "", "", ""


def _target_accelerator(target: Mapping[str, Any]) -> str:
    token = _token(
        target.get("accelerator") or target.get("backend")
        or target.get("provider") or target.get("id")
    )
    if "deepx" in token or "dx_m1" in token:
        return "deepx_m1"
    if "hailo10" in token or "hailo_10" in token:
        return "hailo10"
    if "hailo8" in token or token == "hailo":
        return "hailo8"
    if token in {"trt", "tensor_rt", "tensorrt"}:
        return "tensorrt"
    if token in {"cpu", "ort_cpu", "cpu_ort"}:
        return "cpu_ort"
    return token


def _run_accelerator(
    run_id: Any, effective_plan: Mapping[str, Any],
) -> str:
    run_id = canonical_run_id(run_id)
    if "hailo10" in run_id:
        return "hailo10"
    if "hailo8" in run_id:
        return "hailo8"
    if "deepx" in run_id:
        return "deepx_m1"
    if run_id in {"ort_cpu", "ort_tensorrt"}:
        owner = _token(
            effective_plan.get("tensorrt_performance_owner_group")
        )
        group_accelerators = {
            "hailo8_setup": "hailo8",
            "hailo10h_setup": "hailo10",
            "hailo10_setup": "hailo10",
            "deepx_setup": "deepx_m1",
        }
        if owner in group_accelerators:
            return group_accelerators[owner]
        setup_groups = effective_plan.get("setup_groups")
        if isinstance(setup_groups, Mapping):
            for group, values in setup_groups.items():
                if run_id in {
                    canonical_run_id(value) for value in list(values or [])
                }:
                    accelerator = group_accelerators.get(_token(group), "")
                    if accelerator:
                        return accelerator
    return ""


def _physical_run_descriptors(
    *, model_entries: Sequence[Mapping[str, Any]],
    effective_plan: Mapping[str, Any],
    hardware_targets: Sequence[Mapping[str, Any]],
    run_profiles: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets_by_accelerator: dict[str, list[str]] = {}
    for target in hardware_targets:
        if not isinstance(target, Mapping) or target.get("enabled") is False:
            continue
        accelerator = _target_accelerator(target)
        setup_id = str(
            target.get("id") or target.get("setup_id")
            or target.get("setup_lock_id") or ""
        ).strip()
        if accelerator and setup_id:
            targets_by_accelerator.setdefault(accelerator, []).append(setup_id)

    descriptors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    reference_variants: dict[str, tuple[str, ...]] = {}
    for row in run_profiles:
        if not isinstance(row, Mapping) or row.get("enabled") is False:
            continue
        run_id = canonical_run_id(row.get("id") or row.get("run_id"))
        if run_id != "ort_tensorrt":
            continue
        selected = reference_measurement_variants(row)
        if run_id in reference_variants and reference_variants[run_id] != selected:
            errors.append({"error_class": "reference_variants_ambiguous", "run_id": run_id})
        reference_variants[run_id] = selected
    for model in model_entries:
        if not isinstance(model, Mapping):
            continue
        model_id = str(
            model.get("id") or model.get("model_id") or model.get("name") or ""
        ).strip()
        task = _token(model.get("task"))
        if not model_id:
            continue
        for run_id in authoritative_run_ids(effective_plan):
            variants = _run_backend_variants(run_id)
            if run_id in reference_variants:
                variants = [(backend, variant) for backend, variant in variants
                            if variant in reference_variants[run_id]]
            accelerator = _run_accelerator(run_id, effective_plan)
            setups = sorted(set(targets_by_accelerator.get(accelerator) or []))
            if len(setups) != 1:
                errors.append({
                    "error_class": (
                        "required_scope_setup_missing" if not setups
                        else "required_scope_setup_ambiguous"
                    ),
                    "model_id": model_id,
                    "run_id": run_id,
                    "accelerator": accelerator,
                    "setup_ids": setups,
                })
            if not variants:
                errors.append({
                    "error_class": "required_scope_run_descriptor_unknown",
                    "model_id": model_id,
                    "run_id": run_id,
                })
            for backend, variant in variants:
                (
                    measurement_endpoint, quality_endpoint,
                    endpoint_contract_id,
                ) = _endpoint_contract(
                    run_id, task, variant,
                )
                descriptor = {
                    "model_id": model_id,
                    "task": task,
                    "run_id": canonical_run_id(run_id),
                    "backend": _canonical_backend(backend),
                    "variant": _token(variant),
                    "expected_setup_id": setups[0] if len(setups) == 1 else "",
                    "measurement_endpoint": measurement_endpoint,
                    "quality_endpoint": quality_endpoint,
                    "endpoint_contract_id": endpoint_contract_id,
                    "endpoint_contract_source": (
                        "canonical_run_endpoint_registry_v1"
                    ),
                    "requested": True,
                    "terminal_outcome_required": True,
                    "success_required": False,
                    "scope_source": "effective_plan_physical_descriptor",
                }
                descriptor["global_run_descriptor_sha256"] = stable_sha256(
                    descriptor
                )
                descriptors.append(descriptor)
                for field in (
                    "expected_setup_id", "measurement_endpoint",
                    "quality_endpoint",
                ):
                    if not str(descriptor.get(field) or "").strip():
                        errors.append({
                            "error_class": "required_scope_physical_field_blank",
                            "model_id": model_id,
                            "run_id": run_id,
                            "variant": variant,
                            "field": field,
                        })
    return descriptors, errors


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RequiredRunScopeError(f"required_run_scope_symlink:{path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        try:
            directory_fd = os.open(
                str(path.parent),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def seal_required_run_scope(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create a scope once, or verify semantic identity on resume.

    Timestamps are not part of the immutable identity.  This allows a resume to
    verify an existing receipt without failing merely because ``now_iso()`` was
    called again.
    """

    path = Path(path)
    core = _semantic_payload(payload)
    core.setdefault("schema", SCHEMA)
    core.setdefault("schema_version", SCHEMA_VERSION)
    sealed = dict(payload)
    sealed.update(core)
    sealed["scope_sha256"] = stable_sha256(core)

    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RequiredRunScopeError(f"required_run_scope_unsafe_existing:{path}")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RequiredRunScopeError(
                f"required_run_scope_invalid_existing:{path}:{exc}"
            ) from exc
        if not isinstance(existing, dict):
            raise RequiredRunScopeError(f"required_run_scope_not_object:{path}")
        observed = str(existing.get("scope_sha256") or "")
        calculated = stable_sha256(_semantic_payload(existing))
        if observed != calculated:
            raise RequiredRunScopeError(
                f"required_run_scope_existing_hash_invalid:{path}"
            )
        if observed != sealed["scope_sha256"]:
            raise RequiredRunScopeError(
                "required_run_scope_immutable_mismatch:"
                f"{path}:{observed}!={sealed['scope_sha256']}"
            )
        return existing

    _atomic_json(path, sealed)
    return sealed


def authoritative_run_ids(scope: Mapping[str, Any]) -> list[str]:
    """Return the unique, canonical logical paths sealed at run level."""

    values: list[Any] = []
    for key in (
        "effective_generic_run_ids",
        "logical_run_profiles",
        "requested_run_ids",
    ):
        values.extend(list(scope.get(key) or []))
    for row in list(scope.get("logical_runs") or []):
        if isinstance(row, Mapping):
            values.append(row.get("run_id") or row.get("id"))
    out: list[str] = []
    for value in values:
        token = canonical_run_id(value)
        if token and token not in out:
            out.append(token)
    return out


def merge_authoritative_runs(
    benchmark_plan: Mapping[str, Any], run_ids: Sequence[Any],
) -> dict[str, Any]:
    """Project a sealed global run list into a model-local benchmark plan.

    Existing model-local rows are retained verbatim.  Missing global rows are
    appended as requested placeholders.  This prevents the model-local plan
    from silently shrinking the required matrix after a failed build attempt.
    """

    out = dict(benchmark_plan or {})
    source_rows = list(out.get("runs") or out.get("planned_runs") or [])
    rows = [dict(row) for row in source_rows if isinstance(row, Mapping)]
    existing = {
        canonical_run_id(row.get("id") or row.get("run_id"))
        for row in rows
    }
    injected: list[str] = []
    for raw in run_ids:
        run_id = canonical_run_id(raw)
        if not run_id or run_id in existing:
            continue
        rows.append({
            "id": run_id,
            "required": True,
            "requested": True,
            "terminal_outcome_required": True,
            "success_required": False,
            "scope_source": "sealed_global_required_run_scope",
        })
        existing.add(run_id)
        injected.append(run_id)
    out["runs"] = rows
    out["authoritative_scope_projection"] = {
        "source": "required_run_scope.json",
        "global_run_ids": [canonical_run_id(item) for item in run_ids if canonical_run_id(item)],
        "injected_run_ids": injected,
        "model_local_scope_may_shrink_global_scope": False,
    }
    return out


def authoritative_run_descriptors(
    scope: Mapping[str, Any], *, model_id: str = "",
) -> list[dict[str, Any]]:
    """Return immutable physical run descriptors from a strict global scope."""

    wanted_model = _token(model_id)
    rows: list[dict[str, Any]] = []
    for raw in list(scope.get("physical_run_descriptors") or []):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if wanted_model and _token(row.get("model_id")) != wanted_model:
            continue
        rows.append(row)
    return rows


def _descriptor_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _token(row.get("model_id")),
        canonical_run_id(row.get("run_id") or row.get("id")),
        _canonical_backend(row.get("backend")),
        _token(row.get("variant")),
    )


def merge_authoritative_run_descriptors(
    benchmark_plan: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge a strict physical scope into a model-local plan, fail closed.

    The immutable descriptor is allowed to fill an absent field.  A different
    non-empty setup or endpoint is never overwritten because that would turn a
    physically different run into the requested observation.
    """

    out = dict(benchmark_plan or {})
    source_rows = list(out.get("runs") or out.get("planned_runs") or [])
    rows = [dict(row) for row in source_rows if isinstance(row, Mapping)]
    by_run: dict[str, list[dict[str, Any]]] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            continue
        row = dict(descriptor)
        run_id = canonical_run_id(row.get("run_id") or row.get("id"))
        if run_id:
            by_run.setdefault(run_id, []).append(row)

    existing: dict[str, dict[str, Any]] = {
        canonical_run_id(row.get("id") or row.get("run_id")): row
        for row in rows
        if canonical_run_id(row.get("id") or row.get("run_id"))
    }
    injected: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for run_id, run_descriptors in by_run.items():
        plan_row = existing.get(run_id)
        if plan_row is None:
            plan_row = _canonical_execution_recipe(run_id)
            plan_row.update({
                "required": True,
                "requested": True,
                "terminal_outcome_required": True,
                "success_required": False,
                "scope_source": "sealed_global_physical_required_run_scope",
            })
            rows.append(plan_row)
            existing[run_id] = plan_row
            injected.append(run_id)
        _fill_execution_recipe(plan_row, run_id)

        normalized_descriptors: list[dict[str, Any]] = []
        for descriptor in run_descriptors:
            normalized = dict(descriptor)
            normalized["run_id"] = run_id
            normalized["backend"] = _canonical_backend(
                normalized.get("backend")
            )
            normalized["variant"] = _token(normalized.get("variant"))
            normalized_descriptors.append(normalized)
        plan_row["physical_identity_descriptors"] = normalized_descriptors

        common_fields = (
            "expected_setup_id", "measurement_endpoint", "quality_endpoint",
        )
        for field in common_fields:
            values = sorted({
                str(descriptor.get(field) or "").strip()
                for descriptor in normalized_descriptors
                if str(descriptor.get(field) or "").strip()
            })
            plan_value = str(
                plan_row.get(field)
                or (
                    plan_row.get("setup_id")
                    if field == "expected_setup_id" else ""
                )
                or ""
            ).strip()
            # A full+split ORT run can legitimately expose different endpoint
            # contracts.  Those remain in physical_identity_descriptors and
            # are selected by variant during matrix expansion.
            if len(values) != 1:
                if plan_value:
                    conflicts.append({
                        "error_class": "required_scope_descriptor_conflict",
                        "run_id": run_id,
                        "field": field,
                        "sealed_value": values,
                        "model_plan_value": plan_value,
                    })
                continue
            expected_value = values[0]
            if plan_value and _token(plan_value) != _token(expected_value):
                conflicts.append({
                    "error_class": "required_scope_descriptor_conflict",
                    "run_id": run_id,
                    "field": field,
                    "sealed_value": expected_value,
                    "model_plan_value": plan_value,
                })
                continue
            plan_row[field] = expected_value
            if field == "expected_setup_id":
                plan_row["setup_id"] = expected_value

        plan_backend = _canonical_backend(plan_row.get("backend"))
        descriptor_backends = sorted({
            _canonical_backend(row.get("backend"))
            for row in normalized_descriptors
            if _canonical_backend(row.get("backend"))
        })
        if plan_backend and len(descriptor_backends) == 1 and (
            plan_backend != descriptor_backends[0]
        ):
            conflicts.append({
                "error_class": "required_scope_descriptor_conflict",
                "run_id": run_id,
                "field": "backend",
                "sealed_value": descriptor_backends[0],
                "model_plan_value": plan_backend,
            })
        elif not plan_backend and len(descriptor_backends) == 1:
            plan_row["backend"] = descriptor_backends[0]

    if conflicts:
        raise RequiredRunScopeError(
            "required_scope_physical_descriptor_conflict:"
            + json.dumps(conflicts, sort_keys=True, separators=(",", ":"))
        )
    out["runs"] = rows
    out["authoritative_scope_projection"] = {
        "source": "required_run_scope.json",
        "identity_mode": STRICT_IDENTITY_MODE,
        "descriptor_count": sum(len(values) for values in by_run.values()),
        "global_run_ids": sorted(by_run),
        "injected_run_ids": injected,
        "model_local_scope_may_shrink_global_scope": False,
        "descriptor_conflict_count": 0,
    }
    return out


def _matching_run(
    benchmark_plan: Mapping[str, Any], run_id: str,
) -> Mapping[str, Any]:
    wanted = canonical_run_id(run_id)
    for row in list(
        benchmark_plan.get("runs")
        or benchmark_plan.get("planned_runs")
        or []
    ):
        if not isinstance(row, Mapping):
            continue
        if canonical_run_id(row.get("id") or row.get("run_id")) == wanted:
            return row
    return {}


def quality_applicability(
    measurement: Mapping[str, Any], run: Mapping[str, Any],
) -> str:
    """Resolve pre-dispatch applicability without inventing task completion.

    A split that is intended to execute a completed task remains ``conditional``
    until the runtime records its actual endpoint.  P2-only technical rows are
    resolved to ``not_applicable`` later by the evidence-state projection.
    """

    explicit = _token(
        measurement.get("quality_applicability")
        or run.get("quality_applicability")
    )
    if explicit == "not_applicable_build_excluded":
        # A terminal compiler exclusion has no task-quality measurement. Keep
        # the exact original decision authoritative; a label alone is no veto.
        from ..native_job_identity import known_build_exclusion
        if known_build_exclusion(measurement):
            return "not_applicable"
    if explicit in {"applicable", "not_applicable", "conditional"}:
        return explicit
    if any(
        bool(value)
        for value in (
            measurement.get("quality_not_applicable"),
            run.get("quality_not_applicable"),
            measurement.get("technical_validation_only"),
            run.get("technical_validation_only"),
            run.get("semantic_reference_only"),
        )
    ):
        return "not_applicable"
    quality_endpoint = _token(
        measurement.get("quality_endpoint") or run.get("quality_endpoint")
    )
    if quality_endpoint in {
        "completed_task", "completed_detection", "classification_logits",
        "decoded_nms", "full",
    }:
        return "applicable"
    endpoint = _token(
        measurement.get("measurement_endpoint")
        or measurement.get("endpoint_mode")
        or run.get("measurement_endpoint")
        or run.get("endpoint_mode")
    )
    if endpoint in {"part2", "p2_output", "raw_model_outputs", "technical_only"}:
        return "not_applicable"
    if endpoint in {
        "completed_task", "completed_detection", "classification_logits",
        "decoded_nms", "full",
    }:
        return "applicable"
    variant = _token(measurement.get("variant"))
    if variant == "full":
        return "applicable"
    return "conditional"


def build_global_required_run_scope(
    *,
    profile_id: str,
    model_entries: Sequence[Mapping[str, Any]],
    effective_plan: Mapping[str, Any],
    hardware_targets: Sequence[Mapping[str, Any]],
    created_at: str,
    legacy_reprojection: bool = False,
    run_profiles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for row in model_entries:
        if not isinstance(row, Mapping):
            continue
        model_id = str(
            row.get("id") or row.get("model_id") or row.get("name") or ""
        ).strip()
        if model_id:
            models.append({
                "model_id": model_id,
                "role": str(row.get("role") or row.get("evaluation_role") or ""),
                "requested": True,
                "terminal_outcome_required": True,
                "success_required": False,
            })
    run_ids = authoritative_run_ids(effective_plan)
    physical_descriptors: list[dict[str, Any]] = []
    descriptor_errors: list[dict[str, Any]] = []
    if not legacy_reprojection:
        physical_descriptors, descriptor_errors = _physical_run_descriptors(
            model_entries=model_entries,
            effective_plan=effective_plan,
            hardware_targets=hardware_targets,
            run_profiles=run_profiles,
        )
        if descriptor_errors:
            raise RequiredRunScopeError(
                "incomplete_physical_scope_identity:"
                + json.dumps(
                    descriptor_errors, sort_keys=True, separators=(",", ":"),
                )
            )
    logical_runs = [
        {
            "run_id": run_id,
            "requested": True,
            "terminal_outcome_required": True,
            "success_required": False,
            "quality_applicability": "conditional",
        }
        for run_id in run_ids
    ]
    return {
        "schema": SCHEMA,
        "schema_version": 2 if legacy_reprojection else SCHEMA_VERSION,
        "identity_mode": (
            LEGACY_IDENTITY_MODE if legacy_reprojection
            else STRICT_IDENTITY_MODE
        ),
        "legacy_reprojection": bool(legacy_reprojection),
        "scope_level": "run",
        "profile_id": str(profile_id),
        "created_at": str(created_at),
        "authoritative_scope_source": "effective_execution_plan_before_compiler_dispatch",
        "sealed_before_first_model_compiler_dispatch": True,
        "models": models,
        "logical_runs": logical_runs,
        "physical_run_descriptors": physical_descriptors,
        "physical_run_descriptor_count": len(physical_descriptors),
        "physical_run_descriptors_sha256": (
            stable_sha256(physical_descriptors) if physical_descriptors else ""
        ),
        "requested_run_ids": run_ids,
        "logical_run_profiles": list(effective_plan.get("logical_run_profiles") or []),
        "effective_generic_run_ids": list(effective_plan.get("effective_generic_run_ids") or []),
        "hardware_targets": [
            dict(row) for row in hardware_targets if isinstance(row, Mapping)
        ],
        "policy": {
            "build_failure_may_shrink_scope": False,
            "model_local_plan_may_shrink_scope": False,
            "terminal_failure_is_valid_evidence": True,
            "quality_not_applicable_is_not_missing": True,
            "companions_fill_primary_matrix": False,
            "historical_outcomes_are_never_rewritten": True,
        },
    }


def build_model_required_run_scope(
    *,
    model_id: str,
    measurements: Sequence[Mapping[str, Any]],
    benchmark_plan: Mapping[str, Any],
    benchmark_set_contract: Mapping[str, Any],
    created_at: str,
    global_scope_sha256: str = "",
    legacy_reprojection: bool = False,
    case_scope_source: str = "benchmark_set_contract",
    sealed_before_backend_compiler_dispatch: bool = True,
    sealed_after_case_acceptance_before_runtime_dispatch: bool = False,
) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_identities: list[str] = []
    for ordinal, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            continue
        run = _matching_run(benchmark_plan, str(measurement.get("run_id") or ""))
        applicability = quality_applicability(measurement, run)
        identity = {
            "model_id": str(model_id),
            "case_id": str(measurement.get("case_id") or "full"),
            "run_id": canonical_run_id(measurement.get("run_id")),
            "backend": str(measurement.get("backend") or ""),
            "variant": str(measurement.get("variant") or ""),
            "measurement_endpoint": str(
                measurement.get("measurement_endpoint")
                or run.get("measurement_endpoint")
                or ""
            ),
            "quality_endpoint": str(
                measurement.get("quality_endpoint")
                or run.get("quality_endpoint")
                or ""
            ),
            "expected_setup_id": str(
                measurement.get("expected_setup_id")
                or measurement.get("setup_id")
                or run.get("expected_setup_id")
                or run.get("setup_id")
                or ""
            ),
            "global_run_descriptor_sha256": str(
                measurement.get("global_run_descriptor_sha256")
                or run.get("global_run_descriptor_sha256") or ""
            ),
            "endpoint_contract_id": str(
                measurement.get("endpoint_contract_id")
                or run.get("endpoint_contract_id") or ""
            ),
            "endpoint_contract_source": str(
                measurement.get("endpoint_contract_source")
                or run.get("endpoint_contract_source") or ""
            ),
        }
        if not legacy_reprojection:
            blank = [
                field for field in _PHYSICAL_IDENTITY_FIELDS
                if not str(identity.get(field) or "").strip()
                or _token(identity.get(field)) == "unknown"
            ]
            if blank:
                raise RequiredRunScopeError(
                    "incomplete_physical_scope_identity:"
                    f"model={model_id}:run={identity['run_id']}:"
                    f"variant={identity['variant']}:blank={','.join(blank)}"
                )
        identity_sha = stable_sha256(identity)
        if identity_sha in seen:
            duplicate_identities.append(identity_sha)
        seen.add(identity_sha)
        identities.append({
            "ordinal": ordinal,
            **identity,
            "logical_identity_sha256": identity_sha,
            "requested": True,
            "terminal_outcome_required": bool(
                run.get("terminal_outcome_required", True)
            ),
            "success_required": bool(run.get("success_required", False)),
            "quality_applicability": applicability,
            "scope_source": str(
                run.get("scope_source") or "model_benchmark_plan"
            ),
        })
    return {
        "schema": SCHEMA,
        "schema_version": 2 if legacy_reprojection else SCHEMA_VERSION,
        "identity_mode": (
            LEGACY_IDENTITY_MODE if legacy_reprojection
            else STRICT_IDENTITY_MODE
        ),
        "legacy_reprojection": bool(legacy_reprojection),
        "scope_level": "model",
        "model_id": str(model_id),
        "created_at": str(created_at),
        "global_scope_sha256": str(global_scope_sha256 or ""),
        "case_scope_source": str(case_scope_source),
        "sealed_before_backend_compiler_dispatch": bool(
            sealed_before_backend_compiler_dispatch
        ),
        "sealed_after_case_acceptance_before_runtime_dispatch": bool(
            sealed_after_case_acceptance_before_runtime_dispatch
        ),
        "benchmark_plan_sha256": stable_sha256(benchmark_plan),
        "benchmark_set_contract_sha256": stable_sha256(benchmark_set_contract),
        "identity_count": len(identities),
        "duplicate_logical_identity_count": len(duplicate_identities),
        "duplicate_logical_identities": sorted(set(duplicate_identities)),
        "identities": identities,
        "policy": {
            "build_failure_may_shrink_scope": False,
            "terminal_failure_is_valid_evidence": True,
            "quality_not_applicable_is_not_missing": True,
            "success_required_default": False,
            "accepted_case_list_is_runtime_authority": bool(
                sealed_after_case_acceptance_before_runtime_dispatch
            ),
        },
    }


def validate_required_scope_identities(
    rows: Sequence[Mapping[str, Any]], *, legacy_reprojection: bool = False,
) -> list[dict[str, Any]]:
    """Validate physical descriptor completeness before receipt sealing."""

    if legacy_reprojection:
        return []
    errors: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append({
                "error_class": "required_scope_identity_not_object",
                "ordinal": ordinal,
            })
            continue
        for field in _PHYSICAL_IDENTITY_FIELDS:
            value = row.get(field)
            if field == "expected_setup_id":
                value = value or row.get("setup_id")
            if not str(value or "").strip() or _token(value) == "unknown":
                errors.append({
                    "error_class": "required_scope_physical_field_blank",
                    "ordinal": ordinal,
                    "run_id": canonical_run_id(row.get("run_id")),
                    "field": field,
                })
    return errors


def required_measurements_from_scope(
    scope: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project the immutable per-model identities into matrix expectations."""

    rows: list[dict[str, Any]] = []
    for entry in list(scope.get("identities") or []):
        if not isinstance(entry, Mapping):
            continue
        rows.append({
            "schema": "onnx-splitpoint/required-profile-measurement",
            "schema_version": 3,
            "model_id": str(entry.get("model_id") or scope.get("model_id") or ""),
            "case_id": str(entry.get("case_id") or "full"),
            "run_id": canonical_run_id(entry.get("run_id")),
            "backend": str(entry.get("backend") or ""),
            "variant": str(entry.get("variant") or ""),
            "measurement_endpoint": str(entry.get("measurement_endpoint") or ""),
            "quality_endpoint": str(entry.get("quality_endpoint") or ""),
            "setup_id": str(entry.get("expected_setup_id") or ""),
            "expected_setup_id": str(entry.get("expected_setup_id") or ""),
            "requested": bool(entry.get("requested", True)),
            "terminal_outcome_required": bool(
                entry.get("terminal_outcome_required", True)
            ),
            "success_required": bool(entry.get("success_required", False)),
            "quality_applicability": str(
                entry.get("quality_applicability") or "conditional"
            ),
            "logical_identity_sha256": str(
                entry.get("logical_identity_sha256") or ""
            ),
            "global_run_descriptor_sha256": str(
                entry.get("global_run_descriptor_sha256") or ""
            ),
            "endpoint_contract_id": str(
                entry.get("endpoint_contract_id") or ""
            ),
            "endpoint_contract_source": str(
                entry.get("endpoint_contract_source") or ""
            ),
            "scope_source": "sealed_model_required_run_scope",
        })
    return rows


def audit_materialized_scope(
    *, scope: Mapping[str, Any], materialized_measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare generated plan shape with the immutable pre-dispatch scope.

    Differences are evidence, not permission to rewrite the scope.
    """

    strict = (
        int(scope.get("schema_version") or 0) >= 3
        and str(scope.get("identity_mode") or "") == STRICT_IDENTITY_MODE
    )

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        base = (
            str(row.get("model_id") or scope.get("model_id") or ""),
            str(row.get("case_id") or "full"),
            canonical_run_id(row.get("run_id")),
            _canonical_backend(row.get("backend")),
            _token(row.get("variant")),
        )
        if not strict:
            return base
        return (*base,
            _token(row.get("expected_setup_id") or row.get("setup_id")),
            _token(row.get("measurement_endpoint")),
            _token(row.get("quality_endpoint")),
        )

    sealed_rows = required_measurements_from_scope(scope)
    sealed = {key(row): row for row in sealed_rows}
    materialized = {
        key(row): dict(row)
        for row in materialized_measurements
        if isinstance(row, Mapping)
    }
    missing = sorted(set(sealed) - set(materialized))
    extra = sorted(set(materialized) - set(sealed))
    return {
        "schema": "onnx-splitpoint/required-run-scope-materialization-audit",
        "schema_version": 1,
        "status": "match" if not missing and not extra else "difference_recorded",
        "scope_sha256": str(scope.get("scope_sha256") or ""),
        "sealed_identity_count": len(sealed),
        "materialized_identity_count": len(materialized),
        "missing_from_materialized_count": len(missing),
        "extra_in_materialized_count": len(extra),
        "missing_from_materialized": [
            {
                "model_id": item[0], "case_id": item[1], "run_id": item[2],
                "backend": item[3], "variant": item[4],
                **({
                    "expected_setup_id": item[5],
                    "measurement_endpoint": item[6],
                    "quality_endpoint": item[7],
                } if strict else {}),
            }
            for item in missing
        ],
        "extra_in_materialized": [
            {
                "model_id": item[0], "case_id": item[1], "run_id": item[2],
                "backend": item[3], "variant": item[4],
                **({
                    "expected_setup_id": item[5],
                    "measurement_endpoint": item[6],
                    "quality_endpoint": item[7],
                } if strict else {}),
            }
            for item in extra
        ],
        "scope_rewritten": False,
        "build_failure_may_shrink_scope": False,
    }
