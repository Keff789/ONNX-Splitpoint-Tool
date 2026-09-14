#!/usr/bin/env python3
"""Replay one locally validated Native split engine from the remote cache.

This helper is intentionally diagnostic-only.  It validates every candidate
as one complete artifact set on the accelerator host and emits a run-bound
binding set.  It never builds an engine.  Zero compatible hits are rejected.
When several complete sets are semantically compatible, the lexicographically
first cache root is selected deterministically; artifacts from different sets
are never combined.  Producer and artifact hashes remain useful diagnostics,
but are deliberately not a Canary compatibility axis.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.runners.native_split_quality_runtime import (
    prepare_native_split_quality_binding,
)


ARTIFACT_POLICY_ENV = "ONNX_SPLITPOINT_ARTIFACT_POLICY"
CACHE_VERIFY_ONLY = "cache_verify_only"
_CACHE_BINDING_ARTIFACT_ROLES = (
    "part1_runtime", "boundary_metadata", "source_part2_onnx",
    "build_part2_onnx", "engine", "native_trt_meta",
    "engine_build_receipt", "trtexec",
)


@contextmanager
def _cache_verify_environment():
    previous = os.environ.get(ARTIFACT_POLICY_ENV)
    os.environ[ARTIFACT_POLICY_ENV] = CACHE_VERIFY_ONLY
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ARTIFACT_POLICY_ENV, None)
        else:
            os.environ[ARTIFACT_POLICY_ENV] = previous


def _canonical_backend(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "hailo8": "hailo8_to_trt",
        "hailo8_to_tensorrt": "hailo8_to_trt",
        "hailo10": "hailo10h_to_trt",
        "hailo10h": "hailo10h_to_trt",
        "hailo10_to_trt": "hailo10h_to_trt",
        "hailo10_to_tensorrt": "hailo10h_to_trt",
        "hailo10h_to_tensorrt": "hailo10h_to_trt",
        "deepx": "deepx_to_trt",
        "deepx_m1": "deepx_to_trt",
        "deepx_to_tensorrt": "deepx_to_trt",
    }
    return aliases.get(token, token)


def _case_id(value: Any) -> str:
    token = str(value or "").strip().lower()
    digits = token[1:] if token.startswith("b") else token
    return f"b{int(digits):03d}" if digits.isdigit() else token


def _candidate_cache_roots(parent: Path) -> list[Path]:
    expanded = parent.expanduser()
    if expanded.is_symlink():
        return []
    resolved = expanded.resolve()
    if not resolved.is_dir():
        return []
    roots: list[Path] = []
    quality_root = resolved / "native_split_quality"
    if quality_root.is_dir() and not quality_root.is_symlink():
        roots.append(resolved)
    for child in sorted(resolved.iterdir(), key=lambda path: path.name):
        if child.is_symlink() or not child.is_dir():
            continue
        candidate = child.resolve()
        try:
            candidate.relative_to(resolved)
        except ValueError:
            continue
        quality_root = candidate / "native_split_quality"
        if quality_root.is_dir() and not quality_root.is_symlink():
            roots.append(candidate)
    return roots


def _per_root_exact_miss(message: str) -> str:
    """Classify a safe per-root miss while preserving fail-closed errors."""

    text = str(message or "")
    suffixes = {
        "artifact=native_split_quality_binding:reason=missing": "missing",
        "artifact=native_split_quality_binding:reason=exact_hit_count_0": (
            "identity_mismatch"
        ),
    }
    for suffix, reason in suffixes.items():
        if text.endswith(suffix):
            return reason
    return ""


def _per_root_mismatch_axes(message: str) -> list[str]:
    marker = ":mismatch_axes="
    text = str(message or "")
    if marker not in text:
        return []
    raw = text.split(marker, 1)[1].split(":", 1)[0]
    allowed = {
        "part1_runtime_sha256",
        "source_part2_onnx_sha256",
        "policy_sha256",
    }
    values = sorted(set(raw.split(",")))
    return values if values and set(values) <= allowed else []


def _cache_replay_valid(binding: Mapping[str, Any]) -> bool:
    replay = binding.get("cache_verify_replay")
    source_binding_sha256 = str(
        replay.get("source_binding_sha256") if isinstance(replay, Mapping) else ""
    ).strip().lower()
    return bool(
        isinstance(replay, Mapping)
        and replay.get("artifact_policy") == CACHE_VERIFY_ONLY
        and replay.get("compiler_dispatched") is False
        and re.fullmatch(r"[0-9a-f]{64}", source_binding_sha256)
        and str(replay.get("local_validation_status") or "").startswith(
            "local_files_rehashed_"
        )
    )


def _source_binding_sha256(binding: Mapping[str, Any]) -> str:
    replay = binding.get("cache_verify_replay")
    if not isinstance(replay, Mapping):
        return ""
    token = str(replay.get("source_binding_sha256") or "").strip().lower()
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def _artifact_set_sha256(binding: Mapping[str, Any]) -> str:
    artifacts = binding.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(_CACHE_BINDING_ARTIFACT_ROLES)
    ):
        return ""
    identities: dict[str, dict[str, Any]] = {}
    for role in _CACHE_BINDING_ARTIFACT_ROLES:
        raw = artifacts.get(role)
        if not isinstance(raw, Mapping):
            return ""
        sha256 = str(raw.get("sha256") or "").strip().lower()
        try:
            size_bytes = int(raw.get("size_bytes"))
        except (TypeError, ValueError):
            return ""
        if not re.fullmatch(r"[0-9a-f]{64}", sha256) or size_bytes <= 0:
            return ""
        identities[role] = {
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return canonical_json_sha256(identities)


def materialize(
    *, benchmark_set: Path, model_id: str, case_id: str, setup_id: str,
    backend: str, eval_run_id: str, engine_cache_root: Path, output: Path,
) -> dict[str, Any]:
    root = benchmark_set.expanduser().resolve()
    model = str(model_id or "").strip().lower()
    case = _case_id(case_id)
    setup = str(setup_id or "").strip()
    canonical_backend = _canonical_backend(backend)
    eval_id = str(eval_run_id or "").strip()
    if (
        not root.is_dir() or not (root / case).is_dir() or not model
        or not setup or not eval_id
    ):
        raise RuntimeError("cache_verify_native_split_identity_incomplete")
    cache_parent = engine_cache_root.expanduser().resolve()
    candidates = _candidate_cache_roots(cache_parent)
    if not candidates:
        raise RuntimeError(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=trtexec:artifact=native_split_quality_cache_root:"
            "reason=missing"
        )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    hits: list[tuple[Path, dict[str, Any], str, str]] = []
    root_diagnostics: list[dict[str, Any]] = []
    with _cache_verify_environment(), tempfile.TemporaryDirectory(
        prefix="cache_verify_native_split_", dir=str(output.parent),
    ) as temporary:
        temp_root = Path(temporary)
        for index, candidate in enumerate(candidates):
            try:
                replay = prepare_native_split_quality_binding(
                    benchmark_set=root,
                    case_id=case,
                    model_id=model,
                    setup_id=setup,
                    backend=canonical_backend,
                    eval_run_id=eval_id,
                    source_run_id=canonical_backend,
                    cache_root=candidate,
                    output_path=temp_root / f"candidate_{index}.json",
                )
            except RuntimeError as exc:
                message = str(exc)
                miss_reason = _per_root_exact_miss(message)
                if miss_reason:
                    root_diagnostics.append({
                        "cache_root": str(candidate),
                        "status": "no_exact_hit",
                        "reason": miss_reason,
                        "mismatch_axes": _per_root_mismatch_axes(message),
                    })
                    continue
                raise
            binding = replay.get("binding")
            verified, status = validate_native_split_quality_binding(
                binding,
                expected_identity={
                    "model": model,
                    "case": case,
                    "setup_id": setup,
                    "backend": canonical_backend,
                },
                verification_mode="local",
            )
            if verified is None or not _cache_replay_valid(verified):
                raise RuntimeError(
                    "cache_verify_native_split_replay_invalid:" + str(status)
                )
            source_binding_sha256 = _source_binding_sha256(verified)
            artifact_set_sha256 = _artifact_set_sha256(verified)
            if not source_binding_sha256 or not artifact_set_sha256:
                raise RuntimeError(
                    "cache_verify_native_split_equivalence_identity_invalid"
                )
            inner_source_binding_sha256 = str(
                replay.get("cache_verify_source_binding_sha256") or ""
            ).strip().lower()
            inner_artifact_set_sha256 = str(
                replay.get("cache_verify_source_artifact_set_sha256") or ""
            ).strip().lower()
            inner_equivalence_key_sha256 = str(
                replay.get("cache_verify_equivalence_key_sha256") or ""
            ).strip().lower()
            inner_binding_paths = replay.get(
                "cache_verify_equivalent_binding_paths"
            )
            try:
                inner_binding_count = int(
                    replay.get("cache_verify_exact_binding_count")
                )
            except (TypeError, ValueError):
                inner_binding_count = 0
            expected_inner_equivalence_key_sha256 = canonical_json_sha256({
                "binding_sha256": source_binding_sha256,
                "artifact_set_sha256": artifact_set_sha256,
            })
            candidate_root = candidate.resolve()
            inner_paths_within_candidate = bool(
                isinstance(inner_binding_paths, list)
                and inner_binding_paths
                and all(
                    (
                        (resolved_path := Path(path).expanduser().resolve())
                        .is_relative_to(candidate_root)
                        and resolved_path.name
                        == "native_split_quality_binding.json"
                        and "native_split_quality" in resolved_path.relative_to(
                            candidate_root
                        ).parts
                    )
                    for path in inner_binding_paths
                    if isinstance(path, str) and path
                )
                and len(inner_binding_paths) == len([
                    path for path in inner_binding_paths
                    if isinstance(path, str) and path
                ])
            )
            if (
                inner_source_binding_sha256 != source_binding_sha256
                or inner_artifact_set_sha256 != artifact_set_sha256
                or inner_equivalence_key_sha256
                != expected_inner_equivalence_key_sha256
                or inner_binding_count <= 0
                or not isinstance(inner_binding_paths, list)
                or len(inner_binding_paths) != inner_binding_count
                or len(inner_binding_paths) != len(set(inner_binding_paths))
                or inner_binding_paths != sorted(inner_binding_paths)
                or not inner_paths_within_candidate
                or any(
                    not isinstance(path, str) or not path
                    for path in inner_binding_paths
                )
                or str(replay.get("persistent_binding_path") or "")
                != min(inner_binding_paths, default="")
            ):
                raise RuntimeError(
                    "cache_verify_native_split_inner_replay_cross_link_invalid"
                )
            hits.append((
                candidate, dict(verified), source_binding_sha256,
                artifact_set_sha256,
            ))
            root_diagnostics.append({
                "cache_root": str(candidate),
                "status": "exact_hit",
                "reason": "verified",
                "mismatch_axes": [],
                "source_binding_sha256": source_binding_sha256,
                "artifact_set_sha256": artifact_set_sha256,
                "inner_equivalence_key_sha256": (
                    inner_equivalence_key_sha256
                ),
                "inner_exact_binding_count": inner_binding_count,
                "inner_equivalent_binding_paths": inner_binding_paths,
            })

    distinct_source_bindings = sorted({row[2] for row in hits})
    distinct_equivalence_keys = sorted({(row[2], row[3]) for row in hits})
    if not hits:
        raise RuntimeError(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=trtexec:artifact=native_split_quality_binding:"
            "reason=exact_hit_count_0:hit_count=0:"
            f"distinct_source_binding_count={len(distinct_source_bindings)}:"
            f"distinct_equivalence_key_count={len(distinct_equivalence_keys)}:"
            f"searched_root_count={len(candidates)}:"
            "root_diagnostics="
            + json.dumps(
                root_diagnostics,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    selected_root, binding, source_binding_sha256, artifact_set_sha256 = min(
        hits, key=lambda row: str(row[0])
    )
    compatible_cache_roots = sorted(str(row[0]) for row in hits)
    for row in root_diagnostics:
        if row.get("status") == "exact_hit":
            row["selected"] = row.get("cache_root") == str(selected_root)
    replay = dict(binding.get("cache_verify_replay") or {})
    artifacts = binding.get("artifacts")
    artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    artifact_hashes = {
        str(role): str(row.get("sha256") or "")
        for role, row in sorted(artifacts.items())
        if isinstance(row, Mapping)
    }
    cache_attestation: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-split-cache-verify-attestation",
        "schema_version": 1,
        "status": "verified",
        "artifact_policy": CACHE_VERIFY_ONLY,
        "compiler_dispatch_allowed": False,
        "compiler_dispatched": False,
        "eval_run_id": eval_id,
        "model_id": model,
        "case_id": case,
        "setup_id": setup,
        "backend": canonical_backend,
        "engine_cache_root": str(cache_parent),
        "selected_cache_root": str(selected_root),
        "selection_rule": (
            "lexicographic_semantically_compatible_complete_artifact_set"
        ),
        "hashes_are_diagnostic_only": True,
        "exact_hit_count": len(hits),
        "compatible_hit_count": len(hits),
        "distinct_source_binding_sha256_count": len(
            distinct_source_bindings
        ),
        "distinct_equivalence_key_count": len(distinct_equivalence_keys),
        "equivalent_cache_roots": compatible_cache_roots,
        "compatible_cache_roots": compatible_cache_roots,
        "searched_cache_roots": [str(path) for path in candidates],
        "cache_root_diagnostics": root_diagnostics,
        "source_binding_sha256": source_binding_sha256,
        "source_binding_artifact_set_sha256": artifact_set_sha256,
        "replay_binding_sha256": str(binding.get("binding_sha256") or ""),
        "artifact_sha256_by_role": artifact_hashes,
    }
    cache_attestation["attestation_sha256"] = canonical_json_sha256(
        cache_attestation
    )
    key = "|".join((model, case, canonical_backend))
    binding_set: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-split-quality-binding-set",
        "schema_version": 2,
        "mode": CACHE_VERIFY_ONLY,
        "diagnostic_only": True,
        "claim_eligible": False,
        "eval_run_id": eval_id,
        "setup_id": setup,
        "cache_verify_attestation": cache_attestation,
        "cache_verify_attestation_sha256": cache_attestation[
            "attestation_sha256"
        ],
        "bindings_by_model_case_backend": {key: binding},
    }
    binding_set["binding_set_sha256"] = canonical_json_sha256(binding_set)
    temporary_output = output.with_name(output.name + ".tmp")
    temporary_output.write_text(
        json.dumps(binding_set, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output)
    return binding_set


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-set", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--setup-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--engine-cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = materialize(
            benchmark_set=Path(args.benchmark_set),
            model_id=args.model_id,
            case_id=args.case,
            setup_id=args.setup_id,
            backend=args.backend,
            eval_run_id=args.eval_run_id,
            engine_cache_root=Path(args.engine_cache_root),
            output=Path(args.output),
        )
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "status": "cache_miss_blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 4
    print(json.dumps({
        "ok": True,
        "status": "cache_verify_native_split_binding_ready",
        "binding_set_sha256": result["binding_set_sha256"],
        "binding_count": len(result["bindings_by_model_case_backend"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
