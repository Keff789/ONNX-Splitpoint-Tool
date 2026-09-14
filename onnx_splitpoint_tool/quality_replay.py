"""Hardware-free replay of central paired-quality results.

The replay consumes the immutable request, candidate-prediction and canonical
CPU-reference JSON artifacts already present in an EvaluationRun.  It never
invokes ONNX Runtime, TensorRT or an accelerator.  Historical summaries and
their v2.75.21 cache remain read-only; v2.75.22 results are written to a
separate output directory and use the current quality-cache namespace.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

from .quality_result_contract import UNCERTAINTY_FIELDS, project_quality_result
from .quality_cache import CACHE_SCHEMA_VERSION, json_fingerprint
from .quality_service import (
    DETECTION_QUALITY_ALGORITHM_VERSION,
    ManagementQualityService,
    QualityArtifactIntegrityError,
    QualityEvaluationRequest,
    quality_request_from_manifest,
)


REPLAY_SCHEMA = "onnx-splitpoint/central-quality-offline-replay"
REPLAY_SCHEMA_VERSION = 1
CENTRAL_QUALITY_SUMMARY_SCHEMA = "onnx-splitpoint/central-quality-summary"
CENTRAL_QUALITY_SUMMARY_SCHEMA_VERSION = 1
REPLAY_OUTPUT_NAME = "central_quality_replay_v27522.json"
REPLAY_CSV_NAME = "central_quality_replay_v27522.csv"

# ``--full-only`` is deliberately an identity contract, not merely
# ``variant == full``.  The latter admits the generic ort_tensorrt/full
# diagnostic and produced five rows in the v2.75.21 YOLOv7 canary.  These are
# the four canonical vendor-vs-setup-local-TRT authorities agreed for replay.
CANONICAL_FULL_ONLY_IDENTITIES: tuple[tuple[str, str, str], ...] = (
    ("hailo8", "orin_nx_hailo8_01", "full"),
    ("native_full_tensorrt", "orin_nx_hailo8_01", "full"),
    ("hailo10", "orin_nx_hailo10_01", "full"),
    ("native_full_tensorrt", "orin_nx_hailo10_01", "full"),
)


class OfflineQualityReplayError(RuntimeError):
    """Raised when an EvaluationRun cannot be replayed without hardware."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OfflineQualityReplayError(f"invalid JSON: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_snapshot(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.resolve(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _require_unchanged_inputs(
    snapshots: Sequence[Mapping[str, Any]],
) -> None:
    errors: list[str] = []
    unique: dict[Path, dict[str, Any]] = {}
    for raw in snapshots:
        path = Path(str(raw.get("path") or "")).resolve()
        snapshot = dict(raw)
        previous = unique.get(path)
        if previous is not None and (
            previous.get("size_bytes") != snapshot.get("size_bytes")
            or previous.get("sha256") != snapshot.get("sha256")
        ):
            errors.append(f"conflicting input snapshots: {path}")
            continue
        unique[path] = snapshot
    for path, snapshot in sorted(
        unique.items(), key=lambda item: item[0].as_posix()
    ):
        if not path.is_file():
            errors.append(
                f"{snapshot.get('role') or 'input'} disappeared: {path}"
            )
            continue
        observed_size = int(path.stat().st_size)
        observed_sha = _sha256_file(path)
        if (
            observed_size != int(snapshot.get("size_bytes") or 0)
            or observed_sha != str(snapshot.get("sha256") or "")
        ):
            errors.append(
                f"{snapshot.get('role') or 'input'} changed during replay: "
                f"{path}: expected_size={snapshot.get('size_bytes')} "
                f"observed_size={observed_size} "
                f"expected_sha256={snapshot.get('sha256')} "
                f"observed_sha256={observed_sha}"
            )
    if errors:
        raise OfflineQualityReplayError(
            "historical replay inputs changed during offline evaluation; "
            "refusing to publish final JSON/CSV:\n- " + "\n- ".join(errors)
        )


def _normalise_sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    return token


def _valid_sha256(value: Any) -> bool:
    token = _normalise_sha256(value)
    return len(token) == 64 and all(
        character in "0123456789abcdef" for character in token
    )


def _below_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _run_member(run_dir: Path, relative: Any, *, label: str) -> Path:
    raw = str(relative or "").strip()
    if not raw:
        raise OfflineQualityReplayError(f"{label} is empty")
    candidate = (run_dir / raw).resolve()
    if not _below_root(candidate, run_dir):
        raise OfflineQualityReplayError(f"{label} escapes the EvaluationRun: {raw}")
    return candidate


def _descriptor_member(
    descriptor: Any, request_path: Path, run_dir: Path, *, label: str
) -> Optional[Path]:
    if not isinstance(descriptor, Mapping):
        return None
    raw = str(descriptor.get("path") or "").strip()
    if not raw:
        return None
    declared = Path(raw).expanduser()
    if declared.is_absolute():
        declared = declared.resolve()
        if declared.is_file() and _below_root(declared, run_dir):
            return declared
        # Portable runner exports place the artifact beside the request.  This
        # is also the fallback used by the strict artifact loader.
        candidate = (request_path.parent / declared.name).resolve()
    else:
        candidate = (request_path.parent / declared).resolve()
    if not _below_root(candidate, run_dir):
        raise OfflineQualityReplayError(
            f"{label} escapes the EvaluationRun: {candidate}"
        )
    return candidate


def _management_reference_member(
    descriptor: Any,
    run_dir: Path,
    model_id: str,
    *,
    label: str,
) -> Path:
    """Resolve the row-bound management reference inside ``run_dir``.

    Historical rows used one fixed ``canonical_cpu_reference.json`` per model;
    some of them did not record ``reference_path`` at all.  New rows name an
    immutable, source-contract-scoped member.  Absolute producer paths are
    treated as portable logical names and rebased into the EvaluationRun being
    replayed.  No external absolute path, mutable ``latest`` alias, or
    unrecognised reference layout is ever opened.
    """

    reference = dict(descriptor) if isinstance(descriptor, Mapping) else {}
    prefix = ("quality_management", "references", model_id)
    filename = "canonical_cpu_reference.json"
    legacy_parts = (*prefix, filename)
    raw = str(reference.get("reference_path") or "").strip()

    if not raw:
        relative_parts = legacy_parts
        storage_kind = "legacy_fixed"
    else:
        try:
            declared = Path(raw)
        except (TypeError, ValueError) as exc:
            raise OfflineQualityReplayError(
                f"{label} is not a valid path: {raw!r}"
            ) from exc
        declared_parts = tuple(
            part for part in declared.parts if part not in {"", ".", os.sep}
        )
        if any(part == ".." for part in declared_parts):
            raise OfflineQualityReplayError(
                f"{label} escapes the EvaluationRun: {raw}"
            )

        # Absolute status paths survive when an EvaluationRun is copied.  Find
        # the one allowed run-relative suffix and rebase that suffix instead of
        # following the original host path.
        suffixes = [
            declared_parts[index:]
            for index in range(len(declared_parts))
            if declared_parts[index:index + len(prefix)] == prefix
        ]
        if len(suffixes) != 1:
            raise OfflineQualityReplayError(
                f"{label} is not a unique management reference member: {raw}"
            )
        relative_parts = suffixes[0]

        if any(part.lower() == "latest" for part in relative_parts):
            raise OfflineQualityReplayError(
                f"{label} names the mutable/unbound latest reference: {raw}"
            )
        if relative_parts == legacy_parts:
            storage_kind = "legacy_fixed"
        elif (
            len(relative_parts) == len(prefix) + 3
            and relative_parts[:len(prefix)] == prefix
            and relative_parts[len(prefix)] == "by_source_contract"
            and relative_parts[-1] == filename
        ):
            storage_kind = "immutable_source_contract"
        else:
            raise OfflineQualityReplayError(
                f"{label} is not an admitted canonical or by_source_contract "
                f"management reference member: {raw}"
            )

    declared_storage = str(reference.get("reference_storage") or "").strip()
    declared_immutable = reference.get("reference_immutable")
    if storage_kind == "immutable_source_contract":
        contract_in_path = relative_parts[-2]
        declared_contract = _normalise_sha256(
            reference.get("source_contract_sha256")
        )
        if (
            len(contract_in_path) != 64
            or any(
                character not in "0123456789abcdef"
                for character in contract_in_path
            )
            or not _valid_sha256(declared_contract)
            or contract_in_path != declared_contract
        ):
            raise OfflineQualityReplayError(
                f"{label} source-contract path is not bound to "
                "management_cpu_reference.source_contract_sha256"
            )
        if declared_storage != "immutable_source_contract":
            raise OfflineQualityReplayError(
                f"{label} reference_storage does not seal its immutable path"
            )
        if declared_immutable is not True:
            raise OfflineQualityReplayError(
                f"{label} reference_immutable does not seal its immutable path"
            )
    elif (
        declared_storage == "immutable_source_contract"
        or declared_immutable is True
    ):
        raise OfflineQualityReplayError(
            f"{label} claims immutable source-contract storage for a legacy path"
        )

    if "reference_size_bytes" in reference:
        reference_size = reference.get("reference_size_bytes")
        if type(reference_size) is not int or reference_size <= 0:
            raise OfflineQualityReplayError(
                f"{label} reference_size_bytes is not a positive integer"
            )
    elif storage_kind == "immutable_source_contract":
        raise OfflineQualityReplayError(
            f"{label} immutable reference has no reference_size_bytes seal"
        )

    root = run_dir.resolve()
    candidate = root.joinpath(*relative_parts)
    cursor = root
    for part in relative_parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise OfflineQualityReplayError(
                f"{label} traverses a symbolic link: {cursor}"
            )
    resolved = candidate.resolve()
    if not _below_root(resolved, root):
        raise OfflineQualityReplayError(
            f"{label} escapes the EvaluationRun: {raw or candidate}"
        )
    return resolved


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_run_id") or "").strip().lower(),
        str(row.get("source_setup_id") or row.get("setup_id") or "").strip().lower(),
        str(row.get("variant") or "").strip().lower(),
    )


def _canonical_selection(
    rows: Sequence[Mapping[str, Any]], *, strict: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    errors: list[str] = []
    models = sorted({str(row.get("model_id") or "").strip() for row in rows})
    for model_id in models:
        model_rows = [row for row in rows if str(row.get("model_id") or "").strip() == model_id]
        for identity in CANONICAL_FULL_ONLY_IDENTITIES:
            matches = [dict(row) for row in model_rows if _identity(row) == identity]
            if len(matches) != 1:
                errors.append(
                    "canonical full-only identity requires exactly one row: "
                    f"model={model_id or '<missing>'} source_run_id={identity[0]} "
                    f"source_setup_id={identity[1]} variant={identity[2]} observed={len(matches)}"
                )
                continue
            selected.append(matches[0])
    if strict and errors:
        raise OfflineQualityReplayError(
            "canonical full-only selection failed closed:\n- " + "\n- ".join(errors)
        )
    return selected, errors


def _preflight_rows(
    run_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> list[tuple[dict[str, Any], Path, Path, tuple[dict[str, Any], ...]]]:
    prepared: list[
        tuple[dict[str, Any], Path, Path, tuple[dict[str, Any], ...]]
    ] = []
    missing: list[Path] = []
    errors: list[str] = []
    reference_sha_by_path: dict[Path, str] = {}
    for position, raw_row in enumerate(rows):
        row = dict(raw_row)
        candidate_path: Optional[Path] = None
        try:
            request_path = _run_member(
                run_dir, row.get("source_request"),
                label=f"results[{position}].source_request",
            )
        except OfflineQualityReplayError as exc:
            errors.append(str(exc))
            continue
        model_id = str(row.get("model_id") or "").strip()
        if (
            not model_id
            or model_id in {".", ".."}
            or Path(model_id).name != model_id
        ):
            errors.append(
                f"results[{position}].model_id is not a safe EvaluationRun member: "
                f"{model_id or '<missing>'}"
            )
            continue
        management_reference = (
            row.get("management_cpu_reference")
            if isinstance(row.get("management_cpu_reference"), Mapping)
            else {}
        )
        try:
            reference_path = _management_reference_member(
                management_reference,
                run_dir,
                model_id,
                label=(
                    f"results[{position}].management_cpu_reference.reference_path"
                ),
            )
        except OfflineQualityReplayError as exc:
            errors.append(str(exc))
            continue
        if not request_path.is_file():
            missing.append(request_path)
        else:
            expected_request_sha = _normalise_sha256(row.get("source_request_sha256"))
            if not _valid_sha256(expected_request_sha):
                errors.append(
                    "central result does not bind its source request with a "
                    f"valid SHA-256: {request_path}"
                )
            else:
                observed_request_sha = _sha256_file(request_path)
                if expected_request_sha != observed_request_sha:
                    errors.append(
                        f"source request SHA-256 mismatch: {request_path}: "
                        f"expected={expected_request_sha} observed={observed_request_sha}"
                    )
            try:
                manifest = _load_json(request_path)
                candidate_path = _descriptor_member(
                    manifest.get("candidate") if isinstance(manifest, Mapping) else None,
                    request_path,
                    run_dir,
                    label=f"results[{position}].candidate.path",
                )
                if candidate_path is None:
                    errors.append(f"candidate prediction descriptor has no path: {request_path}")
                elif not candidate_path.is_file():
                    missing.append(candidate_path)
            except OfflineQualityReplayError as exc:
                errors.append(str(exc))
        expected_reference_sha = _normalise_sha256(
            management_reference.get("reference_sha256")
        )
        if not _valid_sha256(expected_reference_sha):
            errors.append(
                "central result does not bind the canonical management "
                f"reference with a valid SHA-256: {reference_path}"
            )
        if not reference_path.is_file():
            missing.append(reference_path)
        elif _valid_sha256(expected_reference_sha):
            expected_reference_size = management_reference.get(
                "reference_size_bytes"
            )
            if (
                type(expected_reference_size) is int
                and int(reference_path.stat().st_size)
                != expected_reference_size
            ):
                errors.append(
                    "management reference size mismatch: "
                    f"{reference_path}: expected={expected_reference_size} "
                    f"observed={reference_path.stat().st_size}"
                )
            observed_reference_sha = reference_sha_by_path.get(
                reference_path
            )
            if observed_reference_sha is None:
                observed_reference_sha = _sha256_file(reference_path)
                reference_sha_by_path[reference_path] = observed_reference_sha
            if expected_reference_sha != observed_reference_sha:
                errors.append(
                    f"management reference SHA-256 mismatch: {reference_path}: "
                    f"expected={expected_reference_sha} "
                    f"observed={observed_reference_sha}"
                )
        for field in (
            "reference_predictions_sha256",
            "candidate_predictions_sha256",
            "annotations_sha256",
        ):
            if not _valid_sha256(row.get(field)):
                errors.append(
                    f"results[{position}].{field} is not a valid historical "
                    "SHA-256 identity"
                )
        snapshots = tuple(
            _input_snapshot(path, role=role)
            for role, path in (
                ("source request", request_path),
                ("candidate predictions", candidate_path),
                ("management CPU reference", reference_path),
            )
            if path is not None and path.is_file()
        )
        prepared.append((row, request_path, reference_path, snapshots))
    if missing or errors:
        details = []
        if missing:
            details.append(
                "missing required request/prediction paths:\n- "
                + "\n- ".join(str(path) for path in sorted(set(missing)))
            )
        if errors:
            details.append("integrity/path errors:\n- " + "\n- ".join(errors))
        raise OfflineQualityReplayError(
            "offline quality replay requires the original EvaluationRun with "
            "the request and candidate/reference prediction JSON bytes; compact "
            "debug/report packs are insufficient.\n" + "\n".join(details)
        )
    return prepared


def _stable_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime/cache observations from the reproducible result bytes."""

    stable = copy.deepcopy(dict(result))
    stable.pop("cache_hit", None)
    for component in [stable.get("primary"), *list((stable.get("guardrails") or {}).values())]:
        if isinstance(component, dict):
            component.pop("bootstrap_elapsed_s", None)
    stable["execution_mode"] = "offline_prediction_replay"
    stable["hardware_executed"] = False
    return stable


def _result_row(
    source: Mapping[str, Any], request_path: Path, result: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    stable = _stable_result(result)
    row = {
        "model_id": str(source.get("model_id") or ""),
        "task": str(source.get("task") or ""),
        "case_id": str(source.get("case_id") or ""),
        "variant": str(source.get("variant") or ""),
        "source_run_id": str(source.get("source_run_id") or ""),
        "source_setup_id": str(source.get("source_setup_id") or source.get("setup_id") or ""),
        "setup_id": str(source.get("setup_id") or source.get("source_setup_id") or ""),
        "source_request": request_path.relative_to(run_dir).as_posix(),
        "source_request_sha256": f"sha256:{_sha256_file(request_path)}",
        "historical_algorithm_version": str(source.get("algorithm_version") or ""),
        "historical_evaluation_fingerprint": str(source.get("evaluation_fingerprint") or ""),
        "historical_decision": str(source.get("decision") or ""),
        **stable,
    }
    row["scientific_result_sha256"] = json_fingerprint(stable)
    return row


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "model_id", "source_run_id", "source_setup_id", "variant", "decision", "n",
        "primary_metric", "primary_candidate", "primary_reference", "primary_delta",
        "primary_ci_low", "primary_ci_high", "ap50_candidate", "ap50_reference",
        "ap50_delta", "ap50_ci_low", "ap50_ci_high", "ap50_decision",
        "ap75_candidate", "ap75_reference", "ap75_delta", "ap75_ci_low",
        "ap75_ci_high", "ap75_decision", "algorithm_version", "evaluation_fingerprint",
        "source_request",
    ]
    fields.extend(f"{scope}_{field}" for scope in ("primary", "ap50", "ap75") for field in UNCERTAINTY_FIELDS)
    fields.extend(("quality_result_contract_version", "quality_result_source_contract_version", "legacy_quality_result"))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for raw_row in rows:
                row = project_quality_result(raw_row)
                primary = row.get("primary") if isinstance(row.get("primary"), Mapping) else {}
                guardrails = row.get("guardrails") if isinstance(row.get("guardrails"), Mapping) else {}
                ap50 = guardrails.get("ap50") if isinstance(guardrails.get("ap50"), Mapping) else {}
                ap75 = guardrails.get("ap75") if isinstance(guardrails.get("ap75"), Mapping) else {}
                writer.writerow({
                    **{f"{scope}_{field}": component.get(field) for scope, component in (("primary", primary), ("ap50", ap50), ("ap75", ap75)) for field in UNCERTAINTY_FIELDS},
                    **{field: row.get(field) for field in ("quality_result_contract_version", "quality_result_source_contract_version", "legacy_quality_result")},
                    "model_id": row.get("model_id"),
                    "source_run_id": row.get("source_run_id"),
                    "source_setup_id": row.get("source_setup_id"),
                    "variant": row.get("variant"),
                    "decision": row.get("decision"),
                    "n": row.get("n"),
                    "primary_metric": primary.get("metric"),
                    "primary_candidate": primary.get("candidate"),
                    "primary_reference": primary.get("reference"),
                    "primary_delta": primary.get("delta"),
                    "primary_ci_low": primary.get("ci_low"),
                    "primary_ci_high": primary.get("ci_high"),
                    "ap50_candidate": ap50.get("candidate"),
                    "ap50_reference": ap50.get("reference"),
                    "ap50_delta": ap50.get("delta"),
                    "ap50_ci_low": ap50.get("ci_low"),
                    "ap50_ci_high": ap50.get("ci_high"),
                    "ap50_decision": ap50.get("decision"),
                    "ap75_candidate": ap75.get("candidate"),
                    "ap75_reference": ap75.get("reference"),
                    "ap75_delta": ap75.get("delta"),
                    "ap75_ci_low": ap75.get("ci_low"),
                    "ap75_ci_high": ap75.get("ci_high"),
                    "ap75_decision": ap75.get("decision"),
                    "algorithm_version": row.get("algorithm_version"),
                    "evaluation_fingerprint": row.get("evaluation_fingerprint"),
                    "source_request": row.get("source_request"),
                })
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _decision_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], str]:
    counts: dict[str, int] = {}
    normalized: list[str] = []
    for row in rows:
        raw = str(row.get("decision") or "not_evaluated").strip().lower()
        decision = (
            raw if raw in {"fail", "inconclusive", "pass", "not_evaluated"}
            else "not_evaluated"
        )
        counts[decision] = counts.get(decision, 0) + 1
        normalized.append(decision)
    aggregate = next(
        (
            candidate for candidate in (
                "fail", "inconclusive", "not_evaluated", "pass"
            )
            if candidate in normalized
        ),
        "not_evaluated",
    )
    return counts, aggregate


def replay_evaluation_run(
    eval_run_dir: str | Path,
    *,
    out_dir: Optional[str | Path] = None,
    workers: int = 4,
    full_only: bool = False,
) -> dict[str, Any]:
    """Replay central quality requests from an existing EvaluationRun."""

    run_dir = Path(eval_run_dir).expanduser().resolve()
    summary_path = run_dir / "quality_management" / "central_quality_summary.json"
    if not summary_path.is_file():
        raise OfflineQualityReplayError(f"central quality summary is missing: {summary_path}")
    source_summary_sha256 = _sha256_file(summary_path)
    summary = _load_json(summary_path)
    if not isinstance(summary, Mapping):
        raise OfflineQualityReplayError(f"central quality summary root is not an object: {summary_path}")
    if str(summary.get("schema") or "") != CENTRAL_QUALITY_SUMMARY_SCHEMA:
        raise OfflineQualityReplayError(
            "central quality summary schema is not the sealed replay source: "
            f"{summary_path}"
        )
    summary_schema_version = summary.get("schema_version")
    if (
        type(summary_schema_version) is not int
        or summary_schema_version != CENTRAL_QUALITY_SUMMARY_SCHEMA_VERSION
    ):
        raise OfflineQualityReplayError(
            "central quality summary schema_version is not the supported "
            f"literal integer 1: {summary_path}"
        )
    raw_results = summary.get("results")
    if (
        not isinstance(raw_results, Sequence)
        or isinstance(raw_results, (str, bytes, bytearray))
        or any(not isinstance(row, Mapping) for row in raw_results)
    ):
        raise OfflineQualityReplayError(
            "central quality summary results must be an array of objects; "
            f"refusing a lossy projection: {summary_path}"
        )
    source_rows = [dict(row) for row in raw_results]
    if not source_rows:
        raise OfflineQualityReplayError(f"central quality summary has no result rows: {summary_path}")
    request_count = summary.get("request_count")
    if type(request_count) is not int or request_count != len(source_rows):
        raise OfflineQualityReplayError(
            "central quality summary request_count does not exactly match its "
            f"result array: declared={request_count!r} observed={len(source_rows)}"
        )

    canonical_rows, canonical_errors = _canonical_selection(source_rows, strict=full_only)
    selected_rows = canonical_rows if full_only else source_rows
    prepared = _preflight_rows(run_dir, selected_rows)

    target = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else (run_dir.parent / f"{run_dir.name}_offline_quality_replay_v27522").resolve()
    )
    if _below_root(target, run_dir):
        raise OfflineQualityReplayError(
            "offline replay output must be outside the historical EvaluationRun; "
            f"refusing in-run target: {target}"
        )

    requests: list[QualityEvaluationRequest] = []
    for source, request_path, reference_path, _snapshots in prepared:
        try:
            request = quality_request_from_manifest(
                request_path,
                verify_artifacts=True,
                reference_artifact=reference_path,
            )
        except (QualityArtifactIntegrityError, ValueError, OSError) as exc:
            raise OfflineQualityReplayError(
                f"quality request integrity validation failed: {request_path}: {exc}"
            ) from exc
        if str(source.get("task") or request.metric_gate_config.get("task") or "") == "detection":
            request = replace(request, algorithm_version=DETECTION_QUALITY_ALGORITHM_VERSION)
        request = replace(
            request,
            reference_identity=(
                str(source.get("reference_identity") or "") or request.reference_identity
            ),
            request_id=f"offline-replay:{_sha256_file(request_path)}",
        )
        requests.append(request)

    replay_results: list[dict[str, Any]] = []
    cache_dir = target / "evaluation_cache_v2"
    with ManagementQualityService(cache_dir, workers=int(workers)) as service:
        futures = [service.submit(request) for request in requests]
        for (source, request_path, _, _snapshots), future in zip(
            prepared, futures
        ):
            try:
                evaluated = future.result()
            except Exception as exc:
                raise OfflineQualityReplayError(
                    f"offline paired-quality evaluation failed: {request_path}: {exc}"
                ) from exc
            for field in (
                "reference_predictions_sha256",
                "candidate_predictions_sha256",
                "annotations_sha256",
            ):
                expected = _normalise_sha256(source.get(field))
                observed = _normalise_sha256(evaluated.get(field))
                if expected != observed:
                    raise OfflineQualityReplayError(
                        "offline replay prediction/annotation identity mismatch: "
                        f"{request_path}: field={field} expected={expected} "
                        f"observed={observed}"
                    )
            replay_results.append(_result_row(source, request_path, evaluated, run_dir))

    canonical_result_rows: list[dict[str, Any]] = []
    canonical_projection_errors: list[str] = []
    for canonical_row in canonical_rows:
        source_request = str(canonical_row.get("source_request") or "")
        identity = _identity(canonical_row)
        matches = [
            row for row in replay_results
            if _identity(row) == identity
            and str(row.get("source_request") or "") == source_request
            and str(row.get("model_id") or "") == str(canonical_row.get("model_id") or "")
        ]
        if len(matches) != 1:
            canonical_projection_errors.append(
                "canonical replay result requires exactly one identity/request match: "
                f"model={canonical_row.get('model_id') or '<missing>'} "
                f"source_run_id={identity[0]} source_setup_id={identity[1]} "
                f"variant={identity[2]} source_request={source_request} observed={len(matches)}"
            )
            continue
        canonical_result_rows.append(matches[0])
    canonical_errors = [*canonical_errors, *canonical_projection_errors]
    if full_only and canonical_errors:
        raise OfflineQualityReplayError(
            "canonical full-only result projection failed closed:\n- "
            + "\n- ".join(canonical_errors)
        )

    decision_counts, quality_decision = _decision_summary(replay_results)
    canonical_decision_inputs: list[Mapping[str, Any]] = list(canonical_result_rows)
    canonical_decision_inputs.extend(
        {"decision": "not_evaluated"} for _ in canonical_errors
    )
    canonical_decision_counts, canonical_quality_decision = _decision_summary(
        canonical_decision_inputs
    )
    observed_summary_sha256 = _sha256_file(summary_path)
    if observed_summary_sha256 != source_summary_sha256:
        raise OfflineQualityReplayError(
            "historical central quality summary changed during offline replay; "
            f"expected={source_summary_sha256} observed={observed_summary_sha256}; "
            "refusing to publish results"
        )
    _require_unchanged_inputs([
        snapshot
        for _source, _request, _reference, snapshots in prepared
        for snapshot in snapshots
    ])
    output = {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "completed",
        "technical_status": "ok",
        "scientific_status": quality_decision,
        "scientific_pass": quality_decision == "pass",
        "execution_mode": "offline_prediction_replay",
        "hardware_executed": False,
        "historical_summary_mutated": False,
        "historical_cache_reused": False,
        "eval_run_dir": str(run_dir),
        "source_summary": summary_path.relative_to(run_dir).as_posix(),
        "source_summary_sha256": source_summary_sha256,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "detection_algorithm_version": DETECTION_QUALITY_ALGORITHM_VERSION,
        "full_only": bool(full_only),
        "request_count": len(replay_results),
        "decision_counts": decision_counts,
        "quality_decision": quality_decision,
        "results": replay_results,
        "canonical_full_only_selection_contract": [
            {
                "source_run_id": run_id,
                "source_setup_id": setup_id,
                "variant": variant,
            }
            for run_id, setup_id, variant in CANONICAL_FULL_ONLY_IDENTITIES
        ],
        "canonical_full_only_status": "complete" if not canonical_errors else "unavailable",
        "canonical_full_only_errors": canonical_errors,
        "canonical_full_only_results": canonical_result_rows,
        "canonical_full_only_decision_counts": canonical_decision_counts,
        "canonical_full_only_quality_decision": canonical_quality_decision,
    }
    _write_json_atomic(target / REPLAY_OUTPUT_NAME, output)
    _write_csv(target / REPLAY_CSV_NAME, replay_results)
    return output


__all__ = [
    "CANONICAL_FULL_ONLY_IDENTITIES",
    "OfflineQualityReplayError",
    "REPLAY_CSV_NAME",
    "REPLAY_OUTPUT_NAME",
    "replay_evaluation_run",
]
