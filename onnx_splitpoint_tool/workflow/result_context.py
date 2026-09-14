"""Recover omitted Generic result setup identity from the actual copy/dispatch.

The required measurement matrix is deliberately not an identity source.  A
copy receipt must name this exact destination and run, and agree with the
setup-local dispatch record.  These are existing execution records, not a new
attestation or cache identity scheme.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .logical_measurement import canonical_run_id, direct_setup_ids


def _run_id(value: Any) -> str:
    # Existing result-copy manifests name the ``_auto`` file projection.
    token = str(value or "").strip().lower()
    return canonical_run_id(token[:-5] if token.endswith("_auto") else token)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def load_benchmark_source_contexts(
    results_dir: Path, *, model_id: str,
) -> list[dict[str, Any]]:
    """Load only agreeing per-target dispatch and canonical copy records."""
    contexts: list[dict[str, Any]] = []
    for path in sorted((results_dir / "remote_diagnostics").glob(
        "*/result_copy_manifest.json"
    )):
        receipt = _read_mapping(path)
        setup = str(receipt.get("target_id") or "").strip()
        dispatch_path = results_dir / f"remote_benchmark_dispatch_{setup}.json"
        dispatch = _read_mapping(dispatch_path)
        args = dispatch.get("args") or {}
        if not isinstance(args, Mapping):
            continue
        remote_root = Path(str(receipt.get("remote_local_run_dir") or ""))
        if (
            receipt.get("schema") != "onnx-splitpoint/remote-result-copy-manifest"
            or not setup or setup != path.parent.name
            or dispatch.get("hardware_target_id") != setup
            or dispatch.get("model_id") != model_id
            or str(args.get("quality_evidence_setup_id") or setup) != setup
            or not str(dispatch.get("run_id") or "")
            or remote_root.name != dispatch.get("run_id")
        ):
            continue
        copied = [item for item in receipt.get("copied", [])
                  if isinstance(item, Mapping) and item.get("canonical") is True]
        for item in receipt.get("canonical_files", []):
            if not isinstance(item, Mapping):
                continue
            destination = str(item.get("path") or "")
            run_id = _run_id(item.get("run_id"))
            matches = [entry for entry in copied
                       if entry.get("destination") == destination
                       and _run_id(entry.get("canonical_run_id")) == run_id
                       and Path(str(entry.get("source") or "")).parent == remote_root / "results"]
            if not destination or not run_id or len(matches) != 1:
                continue
            contexts.append({
                "source_path": destination,
                "model_id": model_id,
                "run_id": run_id,
                "setup_id": setup,
                "dispatch_path": str(dispatch_path),
                "copy_manifest_path": str(path),
            })
    return contexts


def bind_benchmark_source_context(
    row: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Supply missing setup only; never supply endpoint, timing or success."""
    out = dict(row)
    # Metadata supplied in a benchmark row is not a trusted dispatch record.
    for key in ("source_context_binding_status", "source_context_binding_error",
                "source_context_records"):
        out.pop(key, None)
    matches = [context for context in contexts
               if str(context.get("source_path") or "") == str(row.get("source_path") or "")
               and str(context.get("model_id") or "").lower() == str(row.get("model_id") or "").lower()
               and _run_id(context.get("run_id")) == _run_id(
                   row.get("full_source_run_id") or row.get("run_id") or row.get("source_tag"))]
    if not matches:
        return out
    setups = {setup for context in matches
              for setup in (direct_setup_ids({"setup_id": context.get("setup_id")}) or [""])}
    declared = set(direct_setup_ids(row))
    declared_runs = {_run_id(row.get(key)) for key in (
        "run_id", "full_source_run_id", "quality_source_run_id",
    ) if str(row.get(key) or "").strip()}
    context_runs = {_run_id(context.get("run_id")) for context in matches}
    out["source_context_records"] = [dict(context) for context in matches]
    if declared_runs and declared_runs != context_runs:
        out["source_context_binding_status"] = "conflict"
        out["source_context_binding_error"] = "source_context_run_conflict"
    elif len(setups) != 1 or "" in setups:
        out["source_context_binding_status"] = "ambiguous"
        out["source_context_binding_error"] = "source_context_setup_ambiguous"
    elif declared and declared != setups:
        out["source_context_binding_status"] = "conflict"
        out["source_context_binding_error"] = "source_context_setup_conflict"
    elif not declared and (row.get("quality_identity_valid") is False or row.get("quality_identity_errors")):
        out["source_context_binding_status"] = "conflict"
        out["source_context_binding_error"] = "source_context_embedded_identity_conflict"
    else:
        out["setup_id"] = next(iter(setups))
        out["source_context_binding_status"] = "exact_dispatch_copy_match"
    return out
