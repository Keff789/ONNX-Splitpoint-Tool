"""Exact negative evidence at the common Hailo compiler boundary.

The existing v3 cache payload and v1 build key remain authoritative. Context
is scoped to a call (also across managed compiler helpers), never a new cache
identity. Missing context cannot turn a historical failure into a rejection.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("hailo_negative_evidence", default=None)
CONTEXT_ENV = "ONNX_SPLITPOINT_BUILD_EVIDENCE_CONTEXT_JSON"


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_context(bound: Mapping[str, Any]) -> dict[str, Any]:
    """Use actual nearby manifests only; names alone never attest a split."""
    source = Path(str(bound.get("onnx_path") or "")).expanduser()
    outdir = Path(str(bound.get("outdir") or source.parent)).expanduser()
    for start in (outdir, source.parent):
        for directory in (start, *list(start.parents)[:7]):
            candidate = directory / "split_manifest.json"
            if not candidate.is_file():
                continue
            try:
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
                # Part stage must be explicitly encoded in the actual path.
                stages = {str(p.name).lower() for p in (outdir, source.parent)} & {"part1", "part2"}
                if len(stages) != 1:
                    continue
                for field in ("full_model", "source_full_model"):
                    raw = manifest.get(field)
                    if not isinstance(raw, str) or not raw:
                        continue
                    full = Path(raw).expanduser()
                    if not full.is_absolute():
                        full = directory / full
                    if full.is_file():
                        return {"stage": stages.pop(), "split_manifest": manifest,
                                "full_source_onnx_path": str(full.resolve()),
                                "full_source_onnx_sha256": _file_sha(full)}
            except (OSError, ValueError, TypeError):
                continue
    return {}


@contextmanager
def build_evidence_scope(context: Mapping[str, Any] | None, bound: Mapping[str, Any]):
    inherited = _ACTIVE.get()
    if inherited is not None and context is None:
        yield inherited
        return
    if context is None:
        try:
            loaded = json.loads(os.environ.get(CONTEXT_ENV) or "{}")
            context = loaded if isinstance(loaded, Mapping) else None
        except (ValueError, TypeError):
            context = None
    state = {"context": dict(context or _discover_context(bound)), "info": {}, "recorded": False}
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def child_context() -> dict[str, Any]:
    state = _ACTIVE.get()
    return dict(state.get("context") or {}) if state else {}


def current_evidence_info() -> dict[str, Any]:
    state = _ACTIVE.get()
    return dict(state.get("info") or {}) if state else {}


def _emit(info: Mapping[str, Any], *, net_name: str, hw_arch: str) -> None:
    context = info.get("context") or {}
    manifest = context.get("split_manifest") or {}
    boundary = manifest.get("boundary", manifest.get("boundary_index", "full"))
    print(f"[build-evidence] status={info.get('status')} model={context.get('model_id') or net_name} "
          f"boundary={boundary} stage={context.get('stage') or 'unknown'} backend={hw_arch} "
          f"state={info.get('state') or '-'} reason={info.get('reason') or '-'}", flush=True)


def lookup_before_compile(*, cache_payload: Mapping[str, Any], cache_key: str,
                          source_onnx: Path, net_name: str, hw_arch: str,
                          force: bool = False, authoritative_sdk: bool = True,
                          publish_artifacts: bool = True) -> dict[str, Any]:
    from .build_evidence import boundary_endpoint_contract_sha256, canonical_build_key_from_hailo_v3_payload
    from .build_evidence_store import BuildEvidenceStore

    state = _ACTIVE.get()
    context = dict(state.get("context") or {}) if state else {}
    info: dict[str, Any] = {"status": "UNAVAILABLE", "reusable": False,
                            "reason": "exact_build_context_unavailable", "context": context,
                            "cache_key_v3": cache_key, "cache_payload_v3": dict(cache_payload)}
    try:
        # A private diagnostic intentionally has no productive evidence lookup
        # or publication. That policy is not missing compiler identity.
        if not publish_artifacts:
            raise ValueError("diagnostic_nonpublishing")
        if os.name == "nt":
            raise ValueError("negative_evidence_lookup_deferred_to_wsl")
        if not authoritative_sdk:
            raise ValueError("managed_compiler_identity_unavailable")
        if not context:
            raise ValueError("exact_build_context_unavailable")
        if context.get("identity_error"):
            raise ValueError(str(context["identity_error"]))
        version = str(cache_payload.get("hailo_sdk_version") or "").lower()
        if not version or version in {"unknown", "unavailable", "none"} or version.endswith(":unknown"):
            raise ValueError("compiler_identity_unavailable")
        full_sha = str(context.get("full_source_onnx_sha256") or "")
        full_raw = context.get("full_source_onnx_path")
        if full_raw:
            observed = _file_sha(Path(str(full_raw)).expanduser())
            if full_sha and observed != full_sha:
                raise ValueError("full_source_onnx_identity_changed")
            full_sha = observed
        stage = str(context.get("stage") or "")
        builder_sha = _file_sha(Path(source_onnx))
        if stage == "full" and not full_sha:
            full_sha = builder_sha
        endpoint_sha = boundary_endpoint_contract_sha256(
            stage=stage, cache_payload=cache_payload,
            split_manifest=context.get("split_manifest"),
        )
        key = canonical_build_key_from_hailo_v3_payload(
            cache_payload, builder_source_onnx_sha256=builder_sha,
            full_source_onnx_sha256=full_sha,
            boundary_endpoint_contract_sha256=endpoint_sha,
            expected_cache_key=cache_key,
        )
        info["key"] = key
    except Exception as exc:
        info["reason"] = str(exc) or type(exc).__name__
        if info["reason"] == "full_source_onnx_identity_changed":
            # A retained split and its parent model no longer describe the
            # same request. This is proven drift, not missing history.
            info.update(status="CONFLICT", compiler_dispatch_allowed=False)
    else:
        try:
            decision = BuildEvidenceStore().lookup(key)
            info.update(decision.as_dict())
        except Exception as exc:
            info.update(status="ERROR", reason=f"build_evidence_lookup_failed:{type(exc).__name__}:{exc}")
        info["negative_evidence_hit"] = bool(info.get("status") == "HIT" and info.get("reusable")
                                             and info.get("state") in {"PARSER_UNSUPPORTED", "COMPILE_INFEASIBLE"})
        info["compiler_dispatch_allowed"] = not (info["negative_evidence_hit"] or info.get("status") in {"ERROR", "CONFLICT"})
    if state is not None:
        state["info"] = info
    _emit(info, net_name=net_name, hw_arch=hw_arch)
    return info


def attach_and_record(result: Any, bound: Mapping[str, Any]) -> Any:
    """Persist completed compiler failures, never a cache-only/probe decision."""
    from .build_evidence import classify_build_outcome, validate_build_key
    from .build_evidence_store import BuildEvidenceStore

    state = _ACTIVE.get()
    if not state:
        return result
    details = dict(getattr(result, "details", None) or {})
    calib = dict(getattr(result, "calib_info", None) or {})
    info = dict(details.get("build_evidence") or calib.get("build_evidence") or state.get("info") or {})
    if not info:
        return result
    row = asdict(result) if is_dataclass(result) else dict(vars(result))
    phase = str(row.get("last_stage") or "").lower()
    actual_compiler_failure = bool(row.get("ok") is False and phase in {
        "translate", "parse", "parser", "optimize", "optimization", "compile", "compilation", "mapping",
    })
    if (actual_compiler_failure and os.name != "nt" and not bound.get("cache_only") and not state.get("recorded")
            and not info.get("recorded")
            and not info.get("negative_evidence_hit") and info.get("key")):
        # Keep metadata about older errors out of the new failure classifier.
        classification_row = dict(row)
        classification_row["details"] = {k: v for k, v in details.items() if k != "build_evidence"}
        outcome = classify_build_outcome(classification_row, terminal=True)
        try:
            key = validate_build_key(info["key"])
            origin = {"source": "live_hailo_build", "net_name": str(row.get("net_name") or ""),
                      "outdir": str(bound.get("outdir") or ""), "phase": phase,
                      "error": str(row.get("error") or "")[-12000:],
                      "failure_kind": str(row.get("failure_kind") or ""),
                      "elapsed_s": row.get("elapsed_s")}
            record = BuildEvidenceStore().record(key, outcome, evidence_origin=origin,
                                                reason_code=str(row.get("failure_kind") or outcome))
            info.update(recorded=True, recorded_state=outcome, record_sha256=record.get("record_sha256"))
            state["recorded"] = True
            _emit({**info, "status": "RECORDED", "state": outcome},
                  net_name=str(row.get("net_name") or ""), hw_arch=str(row.get("hw_arch") or ""))
        except Exception as exc:
            info.update(recorded=False, persistence_error=f"{type(exc).__name__}:{exc}")
            _emit({**info, "status": "PERSISTENCE_ERROR", "reason": info["persistence_error"]},
                  net_name=str(row.get("net_name") or ""), hw_arch=str(row.get("hw_arch") or ""))
    details["build_evidence"] = info
    calib["build_evidence"] = info
    if info.get("negative_evidence_hit"):
        details["negative_evidence_hit"] = calib["negative_evidence_hit"] = True
    result.details = details
    result.calib_info = calib
    return result
