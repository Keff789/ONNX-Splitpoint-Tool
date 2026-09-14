"""Bind ordinary Hailo calls to the existing exact build-evidence contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def make_build_evidence_context(
    full_source_onnx_path: str | Path,
    *,
    stage: str,
    split_manifest: Mapping[str, Any] | None = None,
    model_id: str = "",
) -> dict[str, Any]:
    """Capture real source bytes and the existing portable endpoint semantics.

    Incomplete caller input is made visible to the guard; it is never promoted
    to an invented full-model identity.  The backend independently binds the
    actual compiler input, recipe and physical endpoints to this context.
    """
    from .build_evidence import BuildEvidenceError, _read_regular_nofollow

    context: dict[str, Any] = {
        "stage": str(stage),
        "model_id": str(model_id),
    }
    try:
        if stage not in {"full", "part1", "part2"}:
            raise ValueError("invalid_build_stage")
        if not str(full_source_onnx_path or "").strip():
            raise ValueError("full_source_onnx_unavailable")
        source = Path(full_source_onnx_path).expanduser().absolute()
        context["full_source_onnx_path"] = str(source)
        manifest = dict(split_manifest or {})
        if stage != "full":
            boundary = manifest.get("boundary", manifest.get("boundary_index"))
            if type(boundary) is not int or boundary < 0:
                raise ValueError("split_boundary_unavailable")
            context["boundary"] = boundary
        # Freeze the current contract before later target metadata is appended.
        context["split_manifest"] = json.loads(
            json.dumps(manifest, allow_nan=False, ensure_ascii=False)
        )
        try:
            observed = _read_regular_nofollow(
                source, label="build_evidence.full_source_onnx", collect=False,
            )
            context["full_source_onnx_sha256"] = observed.sha256
        except BuildEvidenceError as exc:
            if exc.code != "nofollow_platform_unsupported":
                raise
            # Native Windows cannot perform the POSIX admission. Preserve the
            # real path/contract for translation and hashing in the WSL helper;
            # this marker itself is never an exact identity or a negative HIT.
            context["source_identity_deferred_to_compiler"] = True
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        context["identity_error"] = str(exc)
        context.pop("full_source_onnx_sha256", None)
    return context


def known_negative_build(result: Any) -> bool:
    """Recognize an admitted exact negative, rather than a generic skip."""
    details = (
        result.get("details") if isinstance(result, Mapping)
        else getattr(result, "details", None)
    )
    if not isinstance(details, Mapping):
        return False
    evidence = details.get("build_evidence")
    return bool(
        isinstance(evidence, Mapping)
        and evidence.get("status") == "HIT"
        and evidence.get("reusable") is True
        and evidence.get("state") in {"PARSER_UNSUPPORTED", "COMPILE_INFEASIBLE"}
    )
