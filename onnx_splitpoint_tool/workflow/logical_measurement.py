from __future__ import annotations

"""Conservative grouping of multiple provenance representations.

The normalizer may retain both a setup-specific row and a setup-less mirror of
one physical observation.  v2.79.2 keeps both source representations, selects a
primary representation for timing, and records a shared logical measurement
identity only when an exact request SHA or exact executable-artifact SHA proves
the relation.  Distinct non-empty setup IDs are never collapsed.
"""

import hashlib
import ast
import json
from typing import Any, Mapping, Sequence


DEEPX_PRECISION_SCHEMA = (
    "onnx-splitpoint/deepx-runtime-precision-contract"
)
_PRECISION_ALIASES = {
    "fp16": "float32_layout_fp16",
    "float16": "float32_layout_fp16",
    "float32_layout_fp16": "float32_layout_fp16",
}


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _strip_sha_prefix(value: Any) -> str:
    token = _token(value)
    return token[7:] if token.startswith("sha256:") else token


def _sha(value: Any) -> str:
    token = _strip_sha_prefix(value)
    return (
        token
        if len(token) == 64
        and all(character in "0123456789abcdef" for character in token)
        else ""
    )


def _stable_id(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _structured_precision(value: Any) -> Any:
    """Recover a structured precision contract without executing input.

    Older normalizers preserved a Mapping with ``str(mapping)`` before the
    DeepX identity decision.  Accept JSON and Python-literal representations so
    those audit rows can be canonicalized, but never use ``eval``.
    """

    if isinstance(value, Mapping):
        return value
    text = str(value or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return value
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return value
    return parsed if isinstance(parsed, Mapping) else value


def canonical_runtime_precision_identity(
    value: Any,
) -> tuple[str, str | None]:
    """Return one stable runtime-precision identity or a validation error.

    DeepX compiled precision is an opaque DXNN-artifact identity.  It is not a
    declaration that the vendor compiler used FP16 or INT8 internally.
    """

    if value in (None, ""):
        return "", None
    value = _structured_precision(value)
    if isinstance(value, Mapping):
        schema = str(value.get("schema") or "").strip().lower()
        if schema == DEEPX_PRECISION_SCHEMA:
            if _token(value.get("artifact_kind")) != "dxnn":
                return "", "deepx_precision_artifact_kind_invalid"
            artifact_sha = _sha(value.get("artifact_sha256"))
            if not artifact_sha:
                return "", "deepx_precision_artifact_sha256_invalid"
            return f"deepx_dxnn_sha256:{artifact_sha}", None
        nested = value.get("identity")
        if nested not in (None, ""):
            return canonical_runtime_precision_identity(nested)
        return "", "unsupported_structured_precision_identity"

    token = str(value or "").strip().lower().replace(" ", "")
    if token.startswith("deepx_dxnn_sha256:"):
        artifact_sha = _sha(token.split(":", 1)[1])
        if not artifact_sha:
            return "", "deepx_precision_string_sha256_invalid"
        return f"deepx_dxnn_sha256:{artifact_sha}", None
    return _PRECISION_ALIASES.get(token, token), None


def _precision_audit_value(value: Any) -> str:
    value = _structured_precision(value)
    if isinstance(value, Mapping):
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        )
    return str(value or "").strip()


def canonical_backend(value: Any) -> str:
    token = _token(value)
    aliases = {
        "cpu": "cpu_ort",
        "ort_cpu": "cpu_ort",
        "ort_tensorrt": "tensorrt",
        "trt": "tensorrt",
        "hailo10h": "hailo10",
        "hailo_10": "hailo10",
        "hailo8_to_trt": "hailo8_to_tensorrt",
        "hailo10_to_trt": "hailo10_to_tensorrt",
        "deepx": "deepx_m1",
        "deepx_full": "deepx_m1",
        "deepx_to_trt": "deepx_m1_to_tensorrt",
        "deepx_to_tensorrt": "deepx_m1_to_tensorrt",
        "deepx_m1_to_trt": "deepx_m1_to_tensorrt",
    }
    return aliases.get(token, token)


def canonical_run_id(value: Any) -> str:
    token = _token(value)
    aliases = {
        "cpu_ort": "ort_cpu",
        "tensorrt": "ort_tensorrt",
        "hailo8_full": "hailo8",
        "hailo10h": "hailo10",
        "hailo10_full": "hailo10",
        "hailo8_to_tensorrt": "hailo8_to_trt",
        "hailo10_to_trt": "hailo10_to_tensorrt",
        "deepx_full": "deepx_m1_full",
        "deepx_to_trt": "deepx_m1_to_tensorrt",
        "deepx_to_tensorrt": "deepx_m1_to_tensorrt",
        "deepx_m1_to_trt": "deepx_m1_to_tensorrt",
    }
    return aliases.get(token, token)


def selected_variant(row: Mapping[str, Any]) -> str:
    variant = _token(
        row.get("quality_source_variant")
        or row.get("primary_variant")
        or row.get("variant")
    )
    return "composed" if variant in {"split", "complete"} else variant


def request_sha(row: Mapping[str, Any], variant: str | None = None) -> str:
    variant = variant or selected_variant(row)
    for value in (
        row.get("source_request_sha256"),
        row.get("central_quality_source_request_sha256"),
        row.get("native_split_quality_source_request_sha256"),
    ):
        token = _sha(value)
        if token:
            return token
    mapping = row.get("quality_request_identities_by_variant")
    if isinstance(mapping, Mapping) and isinstance(mapping.get(variant), Mapping):
        token = _sha(mapping[variant].get("source_request_sha256"))
        if token:
            return token
    gates = row.get("task_quality_gates_by_variant")
    if isinstance(gates, Mapping) and isinstance(gates.get(variant), Mapping):
        token = _sha(gates[variant].get("source_request_sha256"))
        if token:
            return token
    return ""


def direct_setup_ids(row: Mapping[str, Any]) -> list[str]:
    """Return setup identities carried by the representation itself.

    ``mirror_setup_ids`` is reconciliation metadata, not a direct setup
    identity.  Treating that field as direct made an annotated setup-less
    mirror indistinguishable from its setup-bound primary and turned every
    exact request-SHA join into an ambiguity on a second pass.
    """

    values: set[str] = set()
    for key in (
        "setup_id", "source_setup_id", "measurement_setup_id",
        "hardware_setup_id",
    ):
        token = _token(row.get(key))
        if token:
            values.add(token)
    for key in ("setup_ids", "quality_source_setup_ids"):
        raw = row.get(key)
        if isinstance(raw, (str, bytes, bytearray)):
            raw = [raw]
        for value in list(raw or []):
            token = _token(value)
            if token:
                values.add(token)
    return sorted(values)


def setup_ids(row: Mapping[str, Any]) -> list[str]:
    """Return all setup identities, including proven mirror projections."""

    values = set(direct_setup_ids(row))
    raw = row.get("mirror_setup_ids")
    if isinstance(raw, (str, bytes, bytearray)):
        raw = [raw]
    for value in list(raw or []):
        token = _token(value)
        if token:
            values.add(token)
    return sorted(values)


def _mapping_sha(parent: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        token = _sha(parent.get(key))
        if token:
            return token
    return ""


def deepx_executable_sha(row: Mapping[str, Any]) -> str:
    """Return only the exact DXNN producer artefact identity.

    A TensorRT suffix engine is deliberately not accepted here: two DeepX
    producers can share the same suffix while using different DXNN binaries.
    """

    keys = (
        "producer_dxnn_sha256", "full_dxnn_sha256", "dxnn_sha256",
        "part1_dxnn_sha256", "deepx_artifact_sha256",
    )
    token = _mapping_sha(row, keys)
    if token:
        return token
    for parent_key in (
        "producer_identity", "native_split_quality_binding", "artifacts",
    ):
        parent = row.get(parent_key)
        if not isinstance(parent, Mapping):
            continue
        token = _mapping_sha(parent, keys)
        if token:
            return token
        for role in (
            "part1_runtime", "producer_artifact", "deepx_runtime",
            "full_runtime", "dxnn",
        ):
            child = parent.get(role)
            if isinstance(child, Mapping):
                token = _mapping_sha(
                    child, (*keys, "sha256", "content_sha256"),
                )
                if token:
                    return token
    return ""


def hailo_executable_sha(row: Mapping[str, Any]) -> str:
    keys = (
        "producer_hef_sha256", "full_hef_sha256", "part1_hef_sha256",
        "compiled_hef_sha256", "hef_sha256",
    )
    token = _mapping_sha(row, keys)
    if token:
        return token
    for parent_key in (
        "producer_identity", "native_split_quality_binding", "artifacts",
    ):
        parent = row.get(parent_key)
        if not isinstance(parent, Mapping):
            continue
        token = _mapping_sha(parent, keys)
        if token:
            return token
        for role in (
            "part1_runtime", "producer_artifact", "full_runtime", "hef",
        ):
            child = parent.get(role)
            if isinstance(child, Mapping):
                token = _mapping_sha(
                    child, (*keys, "sha256", "content_sha256"),
                )
                if token:
                    return token
    return ""


def producer_artifact_sha(row: Mapping[str, Any]) -> str:
    """Return the producer executable SHA, never the TensorRT suffix SHA."""

    identity = logical_identity(row)
    if "deepx" in identity["backend"] or "deepx" in identity["run_id"]:
        return deepx_executable_sha(row)
    if "hailo" in identity["backend"] or "hailo" in identity["run_id"]:
        return hailo_executable_sha(row)
    return _mapping_sha(
        row, ("producer_artifact_sha256", "producer_executable_sha256"),
    )


def logical_identity(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "model_id": _token(row.get("model_id")),
        "case_id": _token(row.get("case_id") or "full"),
        "run_id": canonical_run_id(
            row.get("quality_source_run_id")
            or row.get("source_run_id")
            or row.get("run_id")
            or row.get("source_tag")
        ),
        "backend": canonical_backend(row.get("backend")),
        "variant": selected_variant(row),
        "task": _token(row.get("task")),
    }


def _source_paths(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    raw = row.get("source_paths")
    if isinstance(raw, (str, bytes, bytearray)):
        raw = [raw]
    for value in list(raw or []):
        if str(value).strip():
            values.append(str(value))
    if str(row.get("source_path") or "").strip():
        values.append(str(row.get("source_path")))
    return sorted(set(values))


def _has_timing(row: Mapping[str, Any]) -> bool:
    fields = (
        "pipeline_fps_selected", "throughput_primary_fps", "fps_makespan",
        "pipeline_cycle_selected_ms", "total_latency_ms", "part1_latency_ms",
        "part2_latency_ms", "split_latency_e2e_ms",
    )
    return any(row.get(field) not in (None, "") for field in fields)


def _primary_priority(row: Mapping[str, Any]) -> tuple[int, int, str]:
    setups = direct_setup_ids(row)
    direct = 0 if len(setups) == 1 else 1
    timing = 0 if _has_timing(row) else 1
    source = (_source_paths(row) or [""])[0]
    return direct, timing, source


def annotate_logical_measurements(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain all representations and add proof-oriented grouping metadata."""

    out = [dict(row) for row in rows]
    base_buckets: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(out):
        ident = logical_identity(row)
        req = request_sha(row, ident["variant"])
        artifact = producer_artifact_sha(row)
        base = tuple(
            ident[key]
            for key in (
                "model_id", "case_id", "run_id", "backend", "variant", "task",
            )
        )
        # Exact request SHA is primary.  Artifact identity is a conservative
        # fallback when no request hash exists.
        proof_kind = "request_sha256" if req else "producer_artifact_sha256" if artifact else "none"
        proof_value = req or artifact
        base_buckets.setdefault((*base, proof_kind, proof_value), []).append(index)
        row["logical_measurement_identity"] = ident
        row["logical_measurement_request_sha256"] = req
        row["logical_measurement_producer_artifact_sha256"] = artifact
        row["logical_measurement_proof_kind"] = proof_kind
        row["logical_measurement_proof_value"] = proof_value

    for key, indices in base_buckets.items():
        proof_kind, proof_value = key[-2], key[-1]
        exact_binding = proof_kind != "none" and bool(proof_value)
        direct_setups = sorted({
            setups[0]
            for idx in indices
            for setups in [direct_setup_ids(out[idx])]
            if len(setups) == 1
        })

        # Distinct real setups define distinct logical measurements.  Setup-less
        # rows are only projected into a setup group when there is exactly one
        # possible setup and an exact proof token.
        setup_less = [idx for idx in indices if not direct_setup_ids(out[idx])]
        ambiguous_setup_rows = [
            idx for idx in indices if len(direct_setup_ids(out[idx])) > 1
        ]
        if exact_binding:
            direct_indices = [
                idx for idx in indices
                if len(direct_setup_ids(out[idx])) == 1
            ]
            # One exact token can prove that setup-less representations mirror
            # one direct row.  It cannot prove that two setup-bound rows are
            # the same physical invocation.  Keep multiple direct rows as
            # separate logical primaries so duplicate admission remains
            # fail-closed.
            groups: list[tuple[str, list[int], bool, bool]] = []
            for direct_idx in direct_indices:
                setup = direct_setup_ids(out[direct_idx])[0]
                members = [direct_idx]
                verified_mirror_group = False
                if len(direct_indices) == 1 and len(direct_setups) == 1:
                    members.extend(setup_less)
                    setup_less = []
                    verified_mirror_group = len(members) > 1
                groups.append(
                    (setup, members, True, verified_mirror_group)
                )
        else:
            # Without an exact request/executable proof, even duplicate-looking
            # rows from the same setup stay separate.  This is intentionally
            # conservative and prevents timing rows from being merged merely
            # because their labels happen to match.
            direct_indices = [
                idx for idx in indices if len(direct_setup_ids(out[idx])) == 1
            ]
            groups = [
                (direct_setup_ids(out[idx])[0], [idx], False, False)
                for idx in direct_indices
            ]
        groups.extend(("", [idx], False, False) for idx in setup_less)
        groups.extend(
            ("", [idx], False, False) for idx in ambiguous_setup_rows
        )

        for setup, members, proven_group, verified_mirror_group in groups:
            all_sources = sorted({
                source
                for idx in members
                for source in _source_paths(out[idx])
            })
            group_payload = {
                "identity": out[members[0]]["logical_measurement_identity"],
                "setup_id": setup,
                "proof_kind": proof_kind,
                "proof_value": proof_value,
            }
            same_setup_direct_count = sum(
                1
                for idx in indices
                if setup and direct_setup_ids(out[idx]) == [setup]
            )
            if same_setup_direct_count > 1:
                direct_idx = next(
                    idx
                    for idx in members
                    if direct_setup_ids(out[idx]) == [setup]
                )
                group_payload["direct_representation_discriminator"] = {
                    "sources": _source_paths(out[direct_idx]),
                    "content_sha256": _stable_id({
                        name: value
                        for name, value in out[direct_idx].items()
                        if not str(name).startswith("logical_measurement_")
                    }),
                }
            if not (proven_group and exact_binding):
                # Rows without an exact request/executable proof are deliberately
                # standalone observations.  Include their source representation
                # and content fingerprint in the ID so two duplicate-looking
                # rows cannot collapse merely because their labels and setup
                # happen to match.
                group_payload["unproven_representation_sources"] = all_sources
                group_payload["unproven_representation_fingerprint"] = _stable_id({
                    key: value
                    for key, value in out[members[0]].items()
                    if not str(key).startswith("logical_measurement_")
                })
            group_id = _stable_id(group_payload)
            primary_idx = min(members, key=lambda idx: _primary_priority(out[idx]))
            for idx in members:
                local_setups = direct_setup_ids(out[idx])
                is_primary = idx == primary_idx
                if len(local_setups) == 1:
                    role = "primary_direct_setup" if is_primary else "direct_setup_representation"
                elif setup and exact_binding:
                    role = "setup_less_mirror"
                elif len(local_setups) > 1:
                    role = "standalone_setup_ambiguous"
                else:
                    role = "standalone_unproven"
                out[idx].update({
                    "logical_measurement_id": group_id,
                    "logical_measurement_setup_id": setup,
                    "logical_measurement_primary": is_primary,
                    "representation_role": role,
                    "mirror_provenance_verified": bool(
                        verified_mirror_group and exact_binding and setup
                    ),
                    "mirror_setup_ids": [setup] if setup else local_setups,
                    "mirror_representation_count": len(members),
                    "representation_sources": all_sources,
                })

    _canonicalize_deepx_precision(out)
    return out


def _precision_values(row: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []

    def add(value: Any) -> None:
        if value in (None, ""):
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                add(item)
            return
        values.append(value)

    for key in (
        "runtime_precision_identity", "execution_precision", "precision",
    ):
        add(row.get(key))
    frozen = row.get("frozen_identity_evidence")
    if isinstance(frozen, Mapping):
        frozen_values = frozen.get("values")
        if isinstance(frozen_values, Mapping):
            add(frozen_values.get("runtime_precision_identity"))
    for parent_key in (
        "producer_identity", "candidate_execution_contract",
        "quality_contract", "quality_input_request",
    ):
        parent = row.get(parent_key)
        if not isinstance(parent, Mapping):
            continue
        add(parent.get("runtime_precision_identity"))
        add(parent.get("precision"))
    return values


def _deepx_resolution(
    values: Sequence[Any], artifact_sha: str,
) -> dict[str, Any]:
    raw = sorted({
        _precision_audit_value(value)
        for value in values
        if _precision_audit_value(value)
    })
    artifact_candidates: set[str] = set()
    numeric_candidates: set[str] = set()
    errors: list[str] = []
    for value in values:
        canonical, error = canonical_runtime_precision_identity(value)
        if error:
            errors.append(error)
            continue
        if not canonical:
            continue
        if canonical.startswith("deepx_dxnn_sha256:"):
            artifact_candidates.add(canonical)
        else:
            numeric_candidates.add(canonical)

    artifact_sha = _sha(artifact_sha)
    expected = (
        f"deepx_dxnn_sha256:{artifact_sha}" if artifact_sha else ""
    )
    if len(numeric_candidates) > 1:
        errors.append("deepx_numeric_precision_declaration_conflict")
    if expected:
        mismatches = sorted(artifact_candidates - {expected})
        if mismatches:
            errors.append("deepx_precision_artifact_identity_conflict")
        canonical = expected
    elif len(artifact_candidates) == 1:
        canonical = next(iter(artifact_candidates))
        artifact_sha = canonical.split(":", 1)[1]
    elif len(artifact_candidates) > 1:
        canonical = ""
        errors.append("deepx_precision_artifact_identity_conflict")
    elif len(numeric_candidates) == 1:
        # Legacy rows without a DXNN SHA retain their declared numeric alias.
        # New physical runs are rejected earlier when the artefact is absent.
        canonical = next(iter(numeric_candidates))
    elif len(numeric_candidates) > 1:
        canonical = ""
    else:
        canonical = ""

    return {
        "canonical": canonical,
        "artifact_sha256": artifact_sha,
        "raw": raw,
        "numeric_declarations": sorted(numeric_candidates),
        "errors": sorted(set(errors)),
        "conflict": bool(errors),
    }


def _canonicalize_deepx_precision(rows: list[dict[str, Any]]) -> None:
    by_base: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        identity = logical_identity(row)
        if "deepx" not in identity["backend"] and "deepx" not in identity["run_id"]:
            continue
        logical_id = str(row.get("logical_measurement_id") or "").strip()
        key = (
            ("logical_measurement_id", logical_id)
            if logical_id else tuple(identity[name] for name in (
                "model_id", "case_id", "run_id", "backend", "variant",
                "task",
            ))
        )
        by_base.setdefault(key, []).append(row)

    for group_number, group in enumerate(by_base.values()):
        by_artifact: dict[str, list[dict[str, Any]]] = {}
        for row_number, row in enumerate(group):
            artifact = deepx_executable_sha(row)
            if not artifact:
                preliminary = _deepx_resolution(_precision_values(row), "")
                artifact = str(preliminary.get("artifact_sha256") or "")
            bucket = artifact or f"__missing__:{group_number}:{row_number}"
            by_artifact.setdefault(bucket, []).append(row)

        physical_artifacts = {
            bucket for bucket in by_artifact
            if not bucket.startswith("__missing__:")
        }
        logical_artifact_conflict = len(physical_artifacts) > 1

        for bucket, artifact_rows in by_artifact.items():
            artifact = "" if bucket.startswith("__missing__:") else bucket
            resolution = _deepx_resolution(
                [
                    value
                    for row in artifact_rows
                    for value in _precision_values(row)
                ],
                artifact,
            )
            if logical_artifact_conflict:
                resolution["canonical"] = ""
                resolution["conflict"] = True
                resolution["errors"] = sorted(set([
                    *list(resolution.get("errors") or []),
                    "deepx_logical_measurement_artifact_conflict",
                ]))
            canonical = str(resolution.get("canonical") or "")
            conflict = bool(resolution.get("conflict"))
            for row in artifact_rows:
                row["runtime_precision_identity_raw_representations"] = list(
                    resolution.get("raw") or []
                )
                row["runtime_precision_identity_representations"] = list(
                    resolution.get("raw") or []
                )
                row["execution_precision_declared_candidates"] = list(
                    resolution.get("numeric_declarations") or []
                )
                row["precision_semantics"] = (
                    "opaque_vendor_compiled_artifact_identity"
                    if resolution.get("artifact_sha256") else "legacy_declared"
                )
                row["deepx_precision_identity_resolution_errors"] = list(
                    resolution.get("errors") or []
                )
                row["deepx_precision_identity_conflict"] = conflict
                row["deepx_precision_identity_canonicalized"] = bool(
                    canonical and len(resolution.get("raw") or []) > 1
                )
                row["deepx_precision_identity_artifact_sha256"] = str(
                    resolution.get("artifact_sha256") or ""
                )
                row["runtime_precision_identity_canonical"] = (
                    "" if conflict else canonical
                )
                row["runtime_precision_identity"] = (
                    "" if conflict else canonical
                )
                if resolution.get("artifact_sha256"):
                    row["runtime_artifact_sha256"] = str(
                        resolution["artifact_sha256"]
                    )

                frozen = row.get("frozen_identity_evidence")
                if isinstance(frozen, Mapping):
                    frozen_copy = dict(frozen)
                    values_copy = dict(frozen_copy.get("values") or {})
                    invalid_copy = dict(frozen_copy.get("invalid") or {})
                    values_copy[
                        "runtime_precision_identity"
                    ] = ([] if conflict or not canonical else [canonical])
                    if conflict:
                        invalid_copy["runtime_precision_identity"] = list(
                            resolution.get("errors") or [
                                "deepx_precision_identity_conflict"
                            ]
                        )
                    else:
                        invalid_copy.pop("runtime_precision_identity", None)
                    frozen_copy["values"] = values_copy
                    frozen_copy["invalid"] = invalid_copy
                    frozen_copy[
                        "runtime_precision_identity_raw_representations"
                    ] = list(resolution.get("raw") or [])
                    frozen_copy[
                        "runtime_precision_identity_resolution_status"
                    ] = "conflict" if conflict else "canonical"
                    row["frozen_identity_evidence"] = frozen_copy


def select_logical_primary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select exactly one row per proof-bound logical observation."""

    already_annotated = all(
        str(row.get("logical_measurement_id") or "")
        and isinstance(row.get("logical_measurement_primary"), bool)
        for row in rows
        if isinstance(row, Mapping)
    )
    annotated = (
        [dict(row) for row in rows]
        if already_annotated else annotate_logical_measurements(rows)
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for row in annotated:
        logical_id = str(row.get("logical_measurement_id") or "")
        if not logical_id:
            errors.append({
                "error_class": "logical_measurement_id_missing",
                "source_path": str(row.get("source_path") or ""),
            })
            continue
        groups.setdefault(logical_id, []).append(row)
    primaries: list[dict[str, Any]] = []
    for logical_id, group in sorted(groups.items()):
        selected = [
            row
            for row in group
            if row.get("logical_measurement_primary") is True
        ]
        if len(selected) != 1:
            errors.append({
                "error_class": "logical_measurement_primary_cardinality",
                "logical_measurement_id": logical_id,
                "primary_count": len(selected),
                "representation_count": len(group),
            })
            continue
        primaries.append(dict(selected[0]))
    return primaries, errors


def summarize_logical_measurements(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    already_annotated = all(
        str(row.get("logical_measurement_id") or "")
        and isinstance(row.get("logical_measurement_primary"), bool)
        for row in rows
        if isinstance(row, Mapping)
    )
    annotated = (
        [dict(row) for row in rows]
        if already_annotated else annotate_logical_measurements(rows)
    )
    primaries, group_errors = select_logical_primary_rows(annotated)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotated:
        groups.setdefault(str(row.get("logical_measurement_id") or ""), []).append(row)
    groups.pop("", None)
    return {
        "raw_representation_count": len(rows),
        "logical_measurement_count": len(groups),
        "mirror_representation_count": sum(
            max(0, len(group) - 1) for group in groups.values()
        ),
        "verified_mirror_representation_count": sum(
            row.get("representation_role") == "setup_less_mirror"
            and row.get("mirror_provenance_verified") is True
            for row in annotated
        ),
        "logical_primary_count": len(primaries),
        "unverified_representation_count": sum(
            row.get("representation_role") in {
                "standalone_unproven", "standalone_setup_ambiguous",
            }
            for row in annotated
        ),
        "logical_group_error_count": len(group_errors),
        "logical_group_errors": group_errors,
        "proven_mirror_group_count": sum(
            any(row.get("representation_role") == "setup_less_mirror" for row in group)
            for group in groups.values()
        ),
        "setup_conflict_count": sum(
            any(row.get("representation_role") == "standalone_setup_ambiguous" for row in group)
            for group in groups.values()
        ),
        "deepx_precision_conflict_count": sum(
            bool(row.get("deepx_precision_identity_conflict")) for row in annotated
        ),
    }
