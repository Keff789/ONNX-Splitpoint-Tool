from __future__ import annotations

"""Strict opt-in contract for quality-only Full endpoint canaries.

The ordinary Evaluation Workflow run matrix is performance-oriented.  Even a
same-backend reference can therefore grow incidental split-shaped rows when it
is interpreted by older suite runners.  ``quality_canary`` is deliberately a
separate, fail-closed contract: it names every vendor Full endpoint and every
setup-local TensorRT companion, and none of those executions may emit a
performance claim.

Profiles which do not opt in are returned unchanged.
"""

from typing import Any, Dict, List, Mapping, Sequence


SCHEMA = "onnx-splitpoint/full-only-quality-canary"
SCHEMA_VERSION = 1
EXECUTION_SCOPE = "full_only"
EXECUTION_ROLE = "full_quality_only"
TRT_RUN_ID = "ort_tensorrt"
TRT_SOURCE_RUN_ID = "native_full_tensorrt"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return _truth(value.get("enabled"))
    return str(value or "").strip().lower() in {
        "1", "true", "yes", "on", "enabled",
    }


def _token(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in (
            "target", "backend", "provider", "accelerator", "hw_arch",
            "type", "device",
        ):
            token = _token(value.get(key))
            if token:
                return token
        return ""
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "trt": "tensorrt",
        "tensor_rt": "tensorrt",
        "ort_tensorrt": "tensorrt",
        "tensorrt_executionprovider": "tensorrt",
        "hailo_8": "hailo8",
        "hailo10": "hailo10h",
        "hailo_10": "hailo10h",
        "hailo10n": "hailo10h",
        "deepx": "deepx_m1",
        "dx_m1": "deepx_m1",
        "dxm1": "deepx_m1",
    }
    return aliases.get(token, token)


def _run_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("id") or row.get("run_id") or row.get("name")
        or row.get("backend") or ""
    ).strip()


def _as_entries(value: Any, *, field: str, errors: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        errors.append(f"quality_canary_{field}_must_be_list")
        return []
    entries: List[Dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            errors.append(f"quality_canary_{field}_entry_invalid:{index}")
            continue
        entries.append(dict(raw))
    return entries


def _normalize_entry(
    raw: Mapping[str, Any], *, kind: str, index: int, errors: List[str],
) -> Dict[str, Any]:
    allowed_fields = {
        "id", "quality_id", "run_id", "source_run_id", "setup_id",
        "hardware_setup_id", "backend", "variant", "variants",
        "execution_role", "performance_claims_emitted",
    }
    unknown_fields = sorted(set(raw) - allowed_fields)
    errors.extend(
        f"quality_canary_{kind}:{index}:unknown_field:{field}"
        for field in unknown_fields
    )
    quality_id = str(raw.get("id") or raw.get("quality_id") or "").strip()
    run_id = str(raw.get("run_id") or "").strip()
    source_run_id = str(raw.get("source_run_id") or "").strip()
    setup_id = str(raw.get("setup_id") or raw.get("hardware_setup_id") or "").strip()
    backend = _token(raw.get("backend"))
    role = str(raw.get("execution_role") or EXECUTION_ROLE).strip()
    prefix = f"quality_canary_{kind}:{index}"
    if not quality_id:
        errors.append(prefix + ":id_missing")
    if not run_id:
        errors.append(prefix + ":run_id_missing")
    if not setup_id:
        errors.append(prefix + ":setup_id_missing")
    if not backend:
        errors.append(prefix + ":backend_missing")
    if role != EXECUTION_ROLE:
        errors.append(prefix + ":implicit_or_explicit_performance_owner_forbidden")
    if (
        raw.get("performance_claims_emitted") is not None
        and raw.get("performance_claims_emitted") is not False
    ):
        errors.append(prefix + ":performance_claims_forbidden")
    variant_probe: Dict[str, Any] = {}
    if "variant" in raw:
        variant_probe["variant"] = raw.get("variant")
    if "variants" in raw:
        variant_probe["variants"] = raw.get("variants")
    if variant_probe:
        variant_error = _full_recipe_error(
            # ``_full_recipe_error`` consumes the same Full-provider fields as
            # a run-plan row.  ``backend`` is the normalized canary-entry
            # field, so bridge it explicitly instead of reporting the
            # otherwise valid ``variant: full`` entry as backend-less.
            {**variant_probe, "full": backend}, backend,
        )
        if variant_error:
            errors.append(prefix + ":" + variant_error)
    if kind == "setup_local_tensorrt_companions":
        if run_id.strip().lower().replace("-", "_") != TRT_RUN_ID:
            errors.append(prefix + ":run_id_must_be_ort_tensorrt")
        if backend != "tensorrt":
            errors.append(prefix + ":backend_must_be_tensorrt")
        source_run_id = source_run_id or TRT_SOURCE_RUN_ID
        if source_run_id != TRT_SOURCE_RUN_ID:
            errors.append(prefix + ":source_run_id_must_be_native_full_tensorrt")
    elif backend == "tensorrt":
        errors.append(prefix + ":tensorrt_must_be_declared_as_companion")
    else:
        source_run_id = source_run_id or run_id
        if source_run_id != run_id:
            errors.append(prefix + ":source_run_id_must_equal_run_id")
    return {
        "id": quality_id,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "setup_id": setup_id,
        "backend": backend,
        "execution_role": EXECUTION_ROLE,
        "performance_claims_emitted": False,
        "kind": (
            "setup_local_tensorrt_companion"
            if kind == "setup_local_tensorrt_companions"
            else "full_endpoint"
        ),
    }


def _full_recipe_error(row: Mapping[str, Any], expected_backend: str) -> str:
    """Return an empty string only for an explicit homogeneous Full recipe."""

    variant = str(row.get("variant") or "").strip().lower().replace("-", "_")
    if variant and variant != "full":
        return "variant_not_full"
    raw_variants = row.get("variants")
    if raw_variants is not None:
        if isinstance(raw_variants, str):
            variants = {
                token.strip().lower().replace("-", "_")
                for token in raw_variants.replace(";", ",").split(",")
                if token.strip()
            }
        elif isinstance(raw_variants, Sequence) and not isinstance(
            raw_variants, (str, bytes, bytearray),
        ):
            variants = {
                str(token).strip().lower().replace("-", "_")
                for token in raw_variants if str(token).strip()
            }
        else:
            return "variants_invalid"
        if variants != {"full"}:
            return "variants_not_exactly_full"
    run_type = str(row.get("type") or row.get("kind") or "").strip().lower().replace("-", "_")
    if run_type in {"mixed", "mixed_backend", "split"}:
        return "mixed_or_split_recipe_forbidden"
    execution_role = str(row.get("execution_role") or "").strip()
    if execution_role and execution_role != EXECUTION_ROLE:
        return "implicit_or_explicit_performance_owner_forbidden"
    if (
        row.get("performance_claims_emitted") is not None
        and row.get("performance_claims_emitted") is not False
    ):
        return "performance_claims_forbidden"
    if any(
        _truth(row.get(key))
        for key in (
            "is_performance_owner", "performance_owner",
            "performance_eligible", "ranking_eligible", "energy_eligible",
        )
    ):
        return "implicit_or_explicit_performance_owner_forbidden"
    if any(
        str(row.get(key) or "").strip()
        for key in (
            "performance_owner_setup_id", "performance_owner_group",
            "performance_claim_owner",
        )
    ):
        return "implicit_or_explicit_performance_owner_forbidden"
    stage1 = _token(row.get("stage1"))
    stage2 = _token(row.get("stage2"))
    full = _token(
        row.get("full") or row.get("full_backend")
        or row.get("full_provider") or ""
    )
    provider = _token(row.get("provider"))
    backend = full or provider or stage1 or stage2
    if not backend:
        return "full_backend_missing"
    if backend != expected_backend:
        return "full_backend_mismatch"
    if stage1 and stage1 != expected_backend:
        return "stage1_not_full_backend"
    if stage2 and stage2 != expected_backend:
        return "stage2_not_full_backend"
    if stage1 and stage2 and stage1 != stage2:
        return "heterogeneous_stages_forbidden"
    return ""


def resolve_full_only_quality_canary(
    profile: Mapping[str, Any], *, plan_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Resolve and validate the opt-in contract without mutating the profile."""

    payload = dict(profile or {}) if isinstance(profile, Mapping) else {}
    raw_cfg = payload.get("quality_canary")
    if not isinstance(raw_cfg, Mapping):
        malformed_opt_in = raw_cfg is not None and _truth(raw_cfg)
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "enabled": bool(malformed_opt_in),
            "execution_scope": "",
            "ok": not malformed_opt_in,
            "status": "blocked" if malformed_opt_in else "not_requested",
            "errors": (
                ["quality_canary_must_be_mapping"]
                if malformed_opt_in else []
            ),
            "full_run_ids": [],
            "setup_local_tensorrt_companions": [],
            "expected_full_quality_results": [],
            "expected_full_quality_identities": [],
        }
    if raw_cfg.get("enabled") is not True:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "execution_scope": "",
            "ok": True,
            "status": "not_requested",
            "errors": [],
            "full_run_ids": [],
            "setup_local_tensorrt_companions": [],
            "expected_full_quality_results": [],
            "expected_full_quality_identities": [],
        }
    cfg = dict(raw_cfg)
    errors: List[str] = []
    scope = str(cfg.get("execution_scope") or "").strip()
    if scope != EXECUTION_SCOPE:
        errors.append("quality_canary_execution_scope_must_be_full_only")
    unknown_fields = sorted(
        set(cfg) - {
            "enabled", "execution_scope", "full_run_ids",
            "setup_local_tensorrt_companions", "schema", "schema_version",
        }
    )
    if unknown_fields:
        errors.extend(
            f"quality_canary_unknown_field:{field}" for field in unknown_fields
        )
    quality_gate = (
        payload.get("quality_gate")
        if isinstance(payload.get("quality_gate"), Mapping) else {}
    )
    statistics = (
        quality_gate.get("statistics")
        if isinstance(quality_gate.get("statistics"), Mapping) else {}
    )
    execution = (
        quality_gate.get("execution")
        if isinstance(quality_gate.get("execution"), Mapping) else {}
    )
    quality_location = str(
        statistics.get("execution_location")
        or quality_gate.get("execution_location")
        or execution.get("location") or ""
    ).strip().lower().replace("-", "_")
    if quality_location not in {
        "central_management", "management", "management_node", "central",
        "central_cpu",
    }:
        errors.append("quality_canary_requires_central_management_quality")
    workflow = (
        payload.get("workflow")
        if isinstance(payload.get("workflow"), Mapping) else {}
    )
    if _truth(workflow.get("skip_runtime_benchmarks")):
        errors.append("quality_canary_runtime_execution_must_be_enabled")
    preset = (
        payload.get("execution_preset")
        if isinstance(payload.get("execution_preset"), Mapping) else {}
    )
    snapshot = (
        preset.get("snapshot")
        if isinstance(preset.get("snapshot"), Mapping) else {}
    )
    defaults = (
        snapshot.get("defaults")
        if isinstance(snapshot.get("defaults"), Mapping) else {}
    )
    overrides = (
        preset.get("overrides")
        if isinstance(preset.get("overrides"), Mapping) else {}
    )
    native_enabled = (
        _truth(overrides.get("native_enabled"))
        if "native_enabled" in overrides
        else _truth(defaults.get("native_enabled"))
    )
    if native_enabled:
        errors.append("quality_canary_native_execution_forbidden")
    raw_endpoints = _as_entries(
        cfg.get("full_run_ids"), field="full_run_ids", errors=errors,
    )
    raw_companions = _as_entries(
        cfg.get("setup_local_tensorrt_companions"),
        field="setup_local_tensorrt_companions", errors=errors,
    )
    endpoints = [
        _normalize_entry(
            raw, kind="full_run_ids", index=index, errors=errors,
        )
        for index, raw in enumerate(raw_endpoints)
    ]
    companions = [
        _normalize_entry(
            raw, kind="setup_local_tensorrt_companions", index=index,
            errors=errors,
        )
        for index, raw in enumerate(raw_companions)
    ]
    # Export a stable setup-local order: vendor Full followed by its explicit
    # TensorRT companion.  This is also the four-identity form consumed by the
    # offline Full-only selection contract.
    expected: List[Dict[str, Any]] = []
    consumed_companions: set[int] = set()
    for endpoint in endpoints:
        expected.append(endpoint)
        endpoint_setup = str(endpoint.get("setup_id") or "")
        for index, companion in enumerate(companions):
            if (
                index not in consumed_companions
                and str(companion.get("setup_id") or "") == endpoint_setup
            ):
                expected.append(companion)
                consumed_companions.add(index)
    expected.extend(
        companion for index, companion in enumerate(companions)
        if index not in consumed_companions
    )
    if not endpoints:
        errors.append("quality_canary_full_run_ids_empty")
    if not companions:
        errors.append("quality_canary_setup_local_tensorrt_companions_empty")
    endpoint_setup_ids = [
        str(row.get("setup_id") or "") for row in endpoints
        if str(row.get("setup_id") or "")
    ]
    companion_setup_ids = [
        str(row.get("setup_id") or "") for row in companions
        if str(row.get("setup_id") or "")
    ]
    for setup_id in sorted({
        value for value in endpoint_setup_ids
        if endpoint_setup_ids.count(value) > 1
    }):
        errors.append(f"quality_canary_duplicate_full_endpoint_setup_id:{setup_id}")
    for setup_id in sorted({
        value for value in companion_setup_ids
        if companion_setup_ids.count(value) > 1
    }):
        errors.append(f"quality_canary_duplicate_tensorrt_companion_setup_id:{setup_id}")
    if set(endpoint_setup_ids) != set(companion_setup_ids):
        errors.append(
            "quality_canary_setup_companion_set_mismatch:"
            f"endpoints={','.join(sorted(set(endpoint_setup_ids)))}:"
            f"companions={','.join(sorted(set(companion_setup_ids)))}"
        )
    quality_ids = [row["id"] for row in expected if row.get("id")]
    duplicate_quality_ids = sorted({value for value in quality_ids if quality_ids.count(value) > 1})
    errors.extend(
        f"quality_canary_duplicate_id:{value}" for value in duplicate_quality_ids
    )
    identities = [
        (str(row.get("run_id") or ""), str(row.get("setup_id") or ""))
        for row in expected if row.get("run_id") and row.get("setup_id")
    ]
    duplicate_identities = sorted({value for value in identities if identities.count(value) > 1})
    errors.extend(
        f"quality_canary_duplicate_run_setup:{run_id}@{setup_id}"
        for run_id, setup_id in duplicate_identities
    )
    endpoint_run_ids = [str(row.get("run_id") or "") for row in endpoints]
    duplicate_endpoint_run_ids = sorted({
        value for value in endpoint_run_ids if value and endpoint_run_ids.count(value) > 1
    })
    errors.extend(
        f"quality_canary_duplicate_full_endpoint_run_id:{value}"
        for value in duplicate_endpoint_run_ids
    )

    rows = [dict(row) for row in list(plan_rows or []) if isinstance(row, Mapping)]
    if plan_rows is not None:
        rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            rows_by_id.setdefault(_run_id(row), []).append(row)
        requested_run_ids = list(dict.fromkeys(
            str(row.get("run_id") or "") for row in expected
            if str(row.get("run_id") or "")
        ))
        for run_id in requested_run_ids:
            matches = rows_by_id.get(run_id, [])
            if len(matches) != 1:
                errors.append(
                    f"quality_canary_run_id_count_not_one:{run_id}:{len(matches)}"
                )
                continue
            expected_backends = {
                str(row.get("backend") or "") for row in expected
                if row.get("run_id") == run_id
            }
            if len(expected_backends) != 1:
                errors.append(
                    f"quality_canary_run_id_backend_ambiguous:{run_id}"
                )
                continue
            recipe_error = _full_recipe_error(
                matches[0], next(iter(expected_backends)),
            )
            if recipe_error:
                errors.append(
                    f"quality_canary_run_recipe_invalid:{run_id}:{recipe_error}"
                )
        semantic_rows = [
            row for row in rows
            if bool(
                row.get("semantic_reference_only")
                or row.get("canonical_cpu_reference")
            )
        ]
        allowed_extra = set()
        for row in semantic_rows:
            semantic_id = _run_id(row)
            tokens = {
                _token(row.get(key)) for key in (
                    "backend", "provider", "stage1", "stage2",
                ) if _token(row.get(key))
            }
            if (
                semantic_id in {"ort_cpu", "management_cpu_reference"}
                and tokens
                and tokens <= {"cpu", "cpu_ort", "ort_cpu"}
                and row.get("performance_eligible") is False
                and row.get("energy_eligible") is False
                and row.get("ranking_eligible") is False
            ):
                allowed_extra.add(semantic_id)
            else:
                errors.append(
                    f"quality_canary_invalid_semantic_reference_run:{semantic_id}"
                )
        if len(semantic_rows) > 1:
            errors.append(
                f"quality_canary_semantic_reference_count_not_one:"
                f"{len(semantic_rows)}"
            )
        unrequested = sorted(
            run_id for run_id in rows_by_id
            if run_id not in set(requested_run_ids) | allowed_extra
        )
        errors.extend(
            f"quality_canary_unrequested_plan_run:{run_id}"
            for run_id in unrequested
        )

    unique_errors = list(dict.fromkeys(errors))
    identities = [
        {
            "id": str(row.get("id") or ""),
            "source_run_id": str(
                row.get("source_run_id") or row.get("run_id") or ""
            ),
            # ``run_id`` is the scientific-result identity alias.  The
            # scheduler recipe is separate because setup-local TensorRT is
            # dispatched as ``ort_tensorrt`` but the emitted central-quality
            # authority is ``native_full_tensorrt``.
            "run_id": str(
                row.get("source_run_id") or row.get("run_id") or ""
            ),
            "dispatch_run_id": str(row.get("run_id") or ""),
            "setup_id": str(row.get("setup_id") or ""),
            "backend": str(row.get("backend") or ""),
            "variant": "full",
            "execution_role": EXECUTION_ROLE,
            "performance_claims_emitted": False,
        }
        for row in expected
    ]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "execution_scope": scope,
        "ok": not unique_errors,
        "status": "ready" if not unique_errors else "blocked",
        "errors": unique_errors,
        "full_run_ids": endpoints,
        "setup_local_tensorrt_companions": companions,
        "expected_full_quality_results": expected,
        "expected_full_quality_identities": identities,
        "expected_full_quality_ids": [
            str(row.get("id") or "") for row in expected
        ],
        "expected_full_quality_setup_ids": [
            str(row.get("setup_id") or "") for row in expected
        ],
        "performance_claims_emitted": False,
        "generic_rows_total": 0,
    }


def project_full_only_quality_plan_rows(
    profile: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Project a validated canary to its exact unique Full recipes."""

    source = [dict(row) for row in rows if isinstance(row, Mapping)]
    contract = resolve_full_only_quality_canary(profile, plan_rows=source)
    if not contract.get("enabled"):
        return source
    if contract.get("ok") is not True:
        raise ValueError(
            "full_only_quality_canary_invalid:"
            + ",".join(str(value) for value in contract.get("errors") or [])
        )
    expected = list(contract.get("expected_full_quality_results") or [])
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for entry in expected:
        if isinstance(entry, Mapping):
            by_run.setdefault(str(entry.get("run_id") or ""), []).append(dict(entry))
    projected: List[Dict[str, Any]] = []
    for row in source:
        run_id = _run_id(row)
        entries = by_run.get(run_id, [])
        semantic_reference = bool(
            row.get("semantic_reference_only") or row.get("canonical_cpu_reference")
        )
        if not entries and not semantic_reference:
            continue
        result = dict(row)
        if entries:
            result.update({
                "variant": "full",
                "variants": ["full"],
                "case_id": "full",
                "execution_scope": EXECUTION_SCOPE,
                "execution_role": EXECUTION_ROLE,
                "quality_evidence_only": True,
                "performance_claims_emitted": False,
                "performance_eligible": False,
                "energy_eligible": False,
                "ranking_eligible": False,
                "pareto_eligible": False,
                "quality_canary_endpoint_ids": [
                    str(entry.get("id") or "") for entry in entries
                ],
                "quality_canary_setup_ids": [
                    str(entry.get("setup_id") or "") for entry in entries
                ],
                "quality_canary_source_run_ids": [
                    str(
                        entry.get("source_run_id")
                        or entry.get("run_id") or ""
                    )
                    for entry in entries
                ],
            })
        projected.append(result)
    return projected


__all__ = [
    "EXECUTION_ROLE",
    "EXECUTION_SCOPE",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TRT_SOURCE_RUN_ID",
    "project_full_only_quality_plan_rows",
    "resolve_full_only_quality_canary",
]
