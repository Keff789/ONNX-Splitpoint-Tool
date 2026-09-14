"""Management-node CPU reference generation for v2.63 campaigns.

The CPU reference is a semantic input, not a performance baseline.  This
module runs exactly one full-model validation case per model on the management
node, in an isolated hard-linked suite workspace.  The expensive paired
uncertainty calculation is performed separately by :mod:`quality_service`.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from .workflow.artifacts import now_iso, read_json, sha256_file, sha256_json, write_json, write_text
from .process_control import ProcessTreeRegistry, terminate_process_tree
from .native_full_quality import enabled_run_profiles


REFERENCE_JOB_SCHEMA = "onnx-splitpoint/management-cpu-reference-job"


class ManagementCPUReferenceError(RuntimeError):
    """Carry a failed model reference's small diagnostic status to consumers."""

    def __init__(self, model_id: str, status: Mapping[str, Any]) -> None:
        fields = (
            "schema", "schema_version", "status", "model_id", "case_id",
            "return_code", "duration_s", "stdout_path", "source_contract_sha256",
            "execution_location", "provider", "semantic_reference_only",
            "include_in_latency_fps_energy", "include_in_ranking", "include_in_pareto",
            "error", "errors", "failure_stage", "exception_type", "exception_message",
            "cancelled", "timed_out",
        )
        self.reference_status = {key: status[key] for key in fields if key in status}
        cause = status.get("error") or status.get("errors") or status.get("status")
        super().__init__(
            f"management CPU reference failed for {model_id}: {cause}; "
            f"return_code={status.get('return_code')}; stdout_path={status.get('stdout_path') or ''}"
        )


def _reference_stdout_diagnostic(path: Path) -> dict[str, Any]:
    """Read a bounded original log tail; never infer a process RC or timeout."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, size - 65536))
        tail = stream.read(65536).decode("utf-8", errors="replace")
    lines = tail.splitlines()
    important = [line[:2048] for line in lines if any(
        token in line.lower() for token in ("[quality", "[err", "[warn")
    )][-20:]
    # Python's terminal exception line is unindented.  The final traceback
    # takes precedence over a preceding chained exception or SDK log message.
    exception = None
    traceback_start = max((i for i, line in enumerate(lines)
                           if line.startswith("Traceback (most recent call last):")), default=-1)
    candidates = lines[traceback_start + 1:] if traceback_start >= 0 else lines
    for line in candidates:
        match = re.match(r"^([A-Za-z_][\w.]*)(?::\s*(.*))?$", line)
        known_exception_name = bool(match and match.group(1).endswith(
            ("Error", "Exception", "Exit", "Interrupt")
        ))
        if match and (known_exception_name or (
            traceback_start >= 0 and match.group(2) is not None
        )):
            exception = (match.group(1), (match.group(2) or "")[:2048])
            if traceback_start >= 0:
                # The first terminal line after the final traceback is its
                # exception. Ordinary trailing messages must not replace it.
                break
    result: dict[str, Any] = {"important_lines": important}
    if exception is not None:
        result.update({
            "exception_type": exception[0],
            "exception_message": exception[1],
            "exception_summary": f"{exception[0]}: {exception[1]}",
            "stdout_excerpt": "\n".join(candidates[-20:])[-4096:],
            "stdout_excerpt_is_tail": True,
        })
    return result


_IMMUTABLE_REFERENCE_LOCKS_GUARD = threading.Lock()
_IMMUTABLE_REFERENCE_LOCKS: dict[str, threading.RLock] = {}


def management_cpu_reference_required(
    profile: Mapping[str, Any] | None,
) -> bool:
    """Return whether quality needs a central semantic ORT-CPU recipe."""

    payload = dict(profile or {})
    quality = (
        payload.get("quality_gate")
        if isinstance(payload.get("quality_gate"), Mapping)
        else {}
    )
    statistics = (
        quality.get("statistics")
        if isinstance(quality.get("statistics"), Mapping)
        else {}
    )
    execution = (
        quality.get("execution")
        if isinstance(quality.get("execution"), Mapping)
        else {}
    )
    value = str(
        statistics.get("execution_location")
        or quality.get("execution_location")
        or execution.get("location")
        or "local"
    ).strip().lower().replace("-", "_")
    if value in {"management", "management_node", "central", "central_cpu"}:
        value = "central_management"
    return value == "central_management"


def is_cpu_reference_recipe(row: Mapping[str, Any]) -> bool:
    token = _run_id(row).lower().replace("-", "_")
    provider = str(row.get("provider") or "").lower().replace("-", "_")
    return token in {"ort_cpu", "cpu_ort"} or provider in {
        "cpu", "cpu_ort", "ort_cpu",
    }


def profile_has_explicit_cpu_reference(
    profile: Mapping[str, Any] | None,
    targets: Sequence[Any] = (),
) -> bool:
    """Return whether CPU was selected by the profile rather than by quality.

    The automatic management reference is deliberately not a deployment
    target.  Keeping this classification separate from generator switches lets
    producers request the one internal ORT recipe without adding a visible
    ``cpu_ort`` run profile.
    """

    aliases = {"cpu", "cpu_ort", "ort_cpu", "cpu_full", "onnxruntime_cpu"}

    def _token(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_")

    payload = dict(profile or {})
    selected = list(targets or [])
    raw_targets = payload.get("targets")
    if isinstance(raw_targets, (list, tuple)):
        selected.extend(raw_targets)
    if any(_token(value) in aliases for value in selected):
        return True
    for raw in enabled_run_profiles(payload.get("run_profiles")):
        if any(
            _token(raw.get(key)) in aliases
            for key in ("id", "full", "full_reference")
        ):
            return True
    return False


def bind_management_cpu_reference_runs(
    runs: list[dict[str, Any]],
    *,
    automatic: bool,
    require_existing: bool,
) -> list[dict[str, Any]]:
    """Mark exactly one producer-owned CPU recipe as semantic-only.

    Real BenchmarkSet generation uses ``require_existing=True`` so this helper
    cannot conceal a missing generator recipe.  Contract-only producers may
    request the canonical recipe explicitly with ``require_existing=False``.
    """

    rows = [dict(row) for row in runs]
    indexes = [
        index for index, row in enumerate(rows)
        if is_cpu_reference_recipe(row)
    ]
    if len(indexes) > 1:
        raise ValueError("management_cpu_reference_recipe_ambiguous")
    if not indexes:
        if require_existing:
            raise ValueError("management_cpu_reference_recipe_missing")
        rows.append({
            "id": "ort_cpu",
            "type": "onnxruntime",
            "provider": "cpu",
            "backend": "ort_cpu",
            "variant": "full",
            "variants": ["full"],
            "stage1": {"type": "onnxruntime", "provider": "cpu"},
            "stage2": {"type": "onnxruntime", "provider": "cpu"},
            "source": "automatic_management_reference",
        })
        indexes = [len(rows) - 1]
    row = rows[indexes[0]]
    row.update({
        "id": "ort_cpu",
        "type": "onnxruntime",
        "provider": "cpu",
        "backend": "ort_cpu",
        "variant": "full",
        "variants": ["full"],
        "stage1": {"type": "onnxruntime", "provider": "cpu"},
        "stage2": {"type": "onnxruntime", "provider": "cpu"},
        "semantic_reference_only": True,
        "canonical_cpu_reference": True,
        "automatic_reference": bool(automatic),
        "performance_eligible": False,
        "energy_eligible": False,
        "ranking_eligible": False,
        "pareto_eligible": False,
        "execution_location": "central_management",
    })
    rows[indexes[0]] = row
    return rows


def finalize_management_cpu_reference_plan_aliases(
    *,
    executable_plan_path: str | Path,
    formal_plan_path: str | Path,
    profile: Mapping[str, Any] | None,
    cache_verify_enabled: bool,
    automatic: bool,
    require_existing: bool,
    benchmark_set_paths: Sequence[str | Path] = (),
    authoritative_alias: str = "formal",
) -> dict[str, Any]:
    """Write and verify the canonical CPU recipe in both plan aliases.

    ``benchmark_suite.py`` consumes the executable plan, while workflow/report
    code reads the formal mirror.  Treating both files as one post-generation
    transaction prevents a valid-looking formal contract from concealing a
    missing executable recipe.  The caller decides whether the producer must
    already have emitted CPU (the normal legacy generator) or whether this
    finalizer may append the internal recipe (import/direct fallbacks).
    Generation uses the formal logical plan as authority. Runtime finalization
    explicitly selects the executable plan after all row normalization.
    """

    executable_path = Path(executable_plan_path)
    formal_path = Path(formal_plan_path)
    if cache_verify_enabled:
        return {
            "status": "not_applicable_cache_verify_only",
            "recipe_count": 0,
            "performance_dispatch_allowed": False,
        }
    if not management_cpu_reference_required(profile):
        return {
            "status": "not_required",
            "recipe_count": 0,
            "performance_dispatch_allowed": False,
        }

    executable = read_json(executable_path, default={}) or {}
    formal = read_json(formal_path, default={}) or {}
    executable_runs = list(
        executable.get("runs") or executable.get("planned_runs") or []
    ) if isinstance(executable, Mapping) else []
    formal_runs = list(
        formal.get("runs") or formal.get("planned_runs") or []
    ) if isinstance(formal, Mapping) else []
    authority = str(authoritative_alias or "formal").strip().lower()
    if authority not in {"formal", "executable"}:
        raise ValueError(
            f"management_cpu_reference_authoritative_alias_invalid:{authority}"
        )
    # Imported suites are repaired from the formal logical projection. Runtime
    # finalization seals the already-normalized executable rows and mirrors
    # those exact rows back to every formal alias.
    source_runs = (
        executable_runs
        if authority == "executable"
        else formal_runs or executable_runs
    )
    if not source_runs and require_existing:
        raise ValueError("management_cpu_reference_recipe_missing")
    bound_runs = bind_management_cpu_reference_runs(
        [dict(row) for row in source_runs if isinstance(row, Mapping)],
        automatic=automatic,
        require_existing=require_existing,
    )
    run_fingerprint = sha256_json(bound_runs)
    invariant = {
        "status": "verified",
        "recipe_count": 1,
        "execution_location": "central_management",
        "performance_dispatch_allowed": False,
        "run_plan_sha256": run_fingerprint,
    }

    def _payload(base: Any) -> dict[str, Any]:
        value = dict(base) if isinstance(base, Mapping) else {}
        value["runs"] = [dict(row) for row in bound_runs]
        value["planned_runs"] = [dict(row) for row in bound_runs]
        value["management_cpu_reference_invariant"] = dict(invariant)
        return value

    executable_out = _payload(executable or formal)
    formal_out = _payload(formal or executable)
    write_json(executable_path, executable_out)
    if formal_path != executable_path:
        write_json(formal_path, formal_out)
    contract_paths = list(dict.fromkeys(Path(path) for path in benchmark_set_paths))
    for contract_path in contract_paths:
        contract = read_json(contract_path, default={}) or {}
        if not isinstance(contract, Mapping):
            raise ValueError(
                f"management_cpu_reference_contract_invalid:{contract_path}"
            )
        contract_out = dict(contract)
        contract_out["planned_runs"] = [dict(row) for row in bound_runs]
        if "runs" in contract_out:
            contract_out["runs"] = [dict(row) for row in bound_runs]
        embedded_plan = contract_out.get("plan")
        if isinstance(embedded_plan, Mapping):
            embedded_out = dict(embedded_plan)
            embedded_out["runs"] = [dict(row) for row in bound_runs]
            embedded_out["planned_runs"] = [dict(row) for row in bound_runs]
            embedded_out["management_cpu_reference_invariant"] = dict(invariant)
            contract_out["plan"] = embedded_out
        contract_out["management_cpu_reference_invariant"] = dict(invariant)
        write_json(contract_path, contract_out)

    reread_executable = read_json(executable_path, default={}) or {}
    reread_formal = read_json(formal_path, default={}) or {}
    alias_runs: list[list[dict[str, Any]]] = []
    for label, payload in (
        ("executable", reread_executable),
        ("formal", reread_formal),
    ):
        rows = [
            dict(row)
            for row in list(
                payload.get("runs") or payload.get("planned_runs") or []
            )
            if isinstance(row, Mapping)
        ]
        cpu_rows = [row for row in rows if is_cpu_reference_recipe(row)]
        if len(cpu_rows) != 1:
            raise ValueError(
                f"management_cpu_reference_{label}_alias_count={len(cpu_rows)}"
            )
        cpu_row = cpu_rows[0]
        if not (
            cpu_row.get("semantic_reference_only") is True
            and cpu_row.get("canonical_cpu_reference") is True
            and cpu_row.get("performance_eligible") is False
            and cpu_row.get("energy_eligible") is False
            and cpu_row.get("ranking_eligible") is False
            and cpu_row.get("pareto_eligible") is False
            and str(cpu_row.get("execution_location") or "")
            == "central_management"
        ):
            raise ValueError(
                f"management_cpu_reference_{label}_alias_not_semantic_only"
            )
        alias_runs.append(rows)
    if alias_runs[0] != alias_runs[1]:
        raise ValueError("management_cpu_reference_plan_alias_mismatch")
    if sha256_json(alias_runs[0]) != run_fingerprint:
        raise ValueError("management_cpu_reference_plan_serialization_mismatch")
    for contract_path in contract_paths:
        contract = read_json(contract_path, default={}) or {}
        planned = [
            dict(row)
            for row in list(contract.get("planned_runs") or [])
            if isinstance(row, Mapping)
        ]
        if planned != alias_runs[0]:
            raise ValueError(
                f"management_cpu_reference_contract_plan_mismatch:{contract_path}"
            )
    return dict(invariant)


def _run_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("run_id") or "").strip()


def _cpu_reference_run(plan: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    for raw in list(plan.get("runs") or plan.get("planned_runs") or []):
        if not isinstance(raw, Mapping):
            continue
        if is_cpu_reference_recipe(raw):
            row = dict(raw)
            row.update({
                "id": "management_cpu_reference",
                "type": "onnxruntime",
                "provider": "cpu",
                "stage1": {"type": "onnxruntime", "provider": "cpu"},
                "stage2": {"type": "onnxruntime", "provider": "cpu"},
                "variants": ["full"],
                "warmup": 0,
                "runs": 1,
                "phase_runs": 0,
                "throughput_frames": 0,
                "throughput_warmup_frames": 0,
                "semantic_reference_only": True,
                "performance_eligible": False,
                "energy_eligible": False,
                "ranking_eligible": False,
                "pareto_eligible": False,
                "execution_location": "central_management",
            })
            return row
    return None


def _first_case_id(contract: Mapping[str, Any]) -> str:
    for raw in list(contract.get("cases") or []):
        if isinstance(raw, Mapping):
            value = raw.get("case_id") or raw.get("id") or raw.get("case_dir") or raw.get("folder")
            if value not in (None, ""):
                text = str(value).strip()
                if text.isdigit():
                    return f"b{int(text):03d}"
                return text
            try:
                return f"b{int(raw.get('boundary', raw.get('split_index'))):03d}"
            except Exception:
                pass
        elif str(raw or "").strip():
            return str(raw).strip()
    return ""


def _ignore_clone(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lower = name.lower()
        if (
            lower.startswith("results_")
            or lower in {"scientific_report", "__pycache__", ".pytest_cache", ".trt_cache"}
            or lower.endswith(".pyc")
        ):
            ignored.add(name)
    return ignored


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _quality_reference_file(workspace: Path) -> Optional[Path]:
    files = sorted(workspace.rglob("task_quality_inputs/canonical_*_reference.json"))
    return files[0] if files else None


def _validate_generated_reference(path: Path) -> dict[str, Any]:
    """Fail closed before publishing a generated semantic reference."""

    payload = read_json(path, default={}) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("canonical reference root is not an object")
    if payload.get("schema") != "onnx-splitpoint/task-quality-reference-input":
        raise ValueError("canonical reference schema is invalid")
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("canonical reference schema version is invalid")
    if str(payload.get("reference_role") or "").strip().lower() != "canonical_cpu_ort":
        raise ValueError("canonical reference role is not ONNX Runtime CPU")
    if payload.get("semantic_reference_only") is not True:
        raise ValueError("canonical reference is not marked semantic-reference-only")
    task = str(payload.get("task") or "").strip().lower()
    records = payload.get("records")
    if task not in {"classification", "detection"} or not isinstance(records, list) or not records:
        raise ValueError("canonical reference task/records are incomplete")

    image_ids: list[Any] = []
    ground_truth_identity: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or row.get("image_id") in (None, ""):
            raise ValueError(f"canonical reference record {index} has no Image ID")
        image_ids.append(row.get("image_id"))
        if task == "classification":
            metrics = row.get("reference")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"classification reference record {index} lacks metrics")
            if not isinstance(metrics.get("top1_hit"), bool) or not isinstance(metrics.get("top5_hit"), bool):
                raise ValueError(f"classification reference record {index} lacks labeled accuracy")
            continue
        predictions = row.get("reference")
        if not isinstance(predictions, list):
            raise ValueError(f"detection reference record {index} is not a detection list")
        ground_truth = row.get("ground_truth")
        if not isinstance(ground_truth, list):
            raise ValueError(f"detection reference record {index} lacks frozen ground truth")
        ground_truth_identity.append({
            "image_id": row.get("image_id"),
            "ground_truth": ground_truth,
        })
        for detection_index, detection in enumerate(predictions):
            if not isinstance(detection, Mapping):
                raise ValueError(f"detection reference {index}/{detection_index} is not an object")
            numeric: dict[str, float] = {}
            for field in ("x1", "y1", "x2", "y2", "score", "class_id"):
                try:
                    numeric[field] = float(detection.get(field))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"detection reference {index}/{detection_index} lacks numeric {field}"
                    ) from exc
                if not math.isfinite(numeric[field]):
                    raise ValueError(f"detection reference {index}/{detection_index} has non-finite {field}")
            if numeric["x2"] < numeric["x1"] or numeric["y2"] < numeric["y1"]:
                raise ValueError(f"detection reference {index}/{detection_index} has inverted coordinates")
            if numeric["score"] < 0.0 or numeric["score"] > 1.0:
                raise ValueError(f"detection reference {index}/{detection_index} has invalid score")

    from .quality_cache import image_ids_fingerprint, json_fingerprint

    observed_ids_sha = image_ids_fingerprint(image_ids)
    metadata: dict[str, Any] = {
        "task": task,
        "record_count": len(records),
        "image_ids_sha256": observed_ids_sha,
    }
    if task == "detection":
        if payload.get("provenance_required") is not True:
            raise ValueError("detection reference does not require frozen provenance")
        from .quality_service import _validate_detection_quality_contract

        contract, contract_sha = _validate_detection_quality_contract(
            payload.get("quality_contract"), role="generated management reference"
        )
        if str(payload.get("quality_contract_sha256") or "").strip().lower() != contract_sha:
            raise ValueError("detection reference contract binding is inconsistent")
        dataset = contract.get("dataset") if isinstance(contract.get("dataset"), Mapping) else {}
        if str(dataset.get("image_ids_sha256") or "").strip().lower() != observed_ids_sha:
            raise ValueError("detection reference Image IDs violate the frozen dataset contract")
        if int(dataset.get("image_count") or 0) != len(records):
            raise ValueError("detection reference cardinality violates the frozen dataset contract")
        if str(dataset.get("ground_truth_sha256") or "").strip().lower() != json_fingerprint(ground_truth_identity):
            raise ValueError("detection reference violates the frozen ground-truth contract")
        decoder = contract.get("decoder") if isinstance(contract.get("decoder"), Mapping) else {}
        nms = contract.get("nms") if isinstance(contract.get("nms"), Mapping) else {}
        metadata.update({
            "quality_contract": contract,
            "quality_contract_sha256": contract_sha,
            "model_sha256": str((contract.get("model") or {}).get("sha256") or ""),
            "dataset_manifest_sha256": str(dataset.get("manifest_sha256") or ""),
            "preprocessing_sha256": str((contract.get("preprocessing") or {}).get("sha256") or ""),
            "decoder_sha256": str(decoder.get("sha256") or ""),
            "nms_sha256": str(nms.get("sha256") or ""),
            "source_endpoint_is_raw": bool(contract.get("source_endpoint_is_raw")),
            "canonical_record_endpoint": str(contract.get("canonical_record_endpoint") or ""),
        })
    return metadata


def _source_contract(source: Path, plan: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    """Fingerprint prospective reference inputs without hashing every image."""
    validation_manifests: list[dict[str, str]] = []
    for raw in list(plan.get("runs") or plan.get("planned_runs") or []):
        if not isinstance(raw, Mapping):
            continue
        for key in ("validation_images", "validation_manifest", "annotations"):
            value = str(raw.get(key) or "").strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = source / path
            candidates = [path]
            if path.is_dir():
                candidates = [
                    path / "manifest.json",
                    path / "annotations.json",
                    path / "labels.json",
                ]
            for candidate in candidates:
                if candidate.is_file():
                    validation_manifests.append({
                        "path": str(candidate.resolve()),
                        "sha256": str(sha256_file(candidate) or ""),
                    })
    script = source / "benchmark_suite.py"
    runner_artifacts = [
        {
            "path": str(path.relative_to(source)),
            "sha256": str(sha256_file(path) or ""),
        }
        for path in sorted(source.rglob("run_split_onnxruntime.py"))
        if path.is_file()
    ]
    model_artifacts: list[dict[str, str]] = []
    for split_manifest_path in sorted(source.rglob("split_manifest.json")):
        split_manifest = read_json(split_manifest_path, default={}) or {}
        full_model = str(
            split_manifest.get("full_model")
            or split_manifest.get("full")
            or split_manifest.get("model")
            or ""
        ).strip()
        if not full_model:
            continue
        model_path = Path(full_model).expanduser()
        if not model_path.is_absolute():
            model_path = split_manifest_path.parent / model_path
        if model_path.is_file():
            model_artifacts.append({
                "path": str(model_path.relative_to(source)) if source in model_path.parents else model_path.name,
                "sha256": str(sha256_file(model_path) or ""),
            })
    return sha256_json({
        "benchmark_plan": plan,
        "benchmark_set": contract,
        "benchmark_suite_sha256": sha256_file(script) if script.is_file() else "",
        "runner_artifacts": runner_artifacts,
        "full_model_artifacts": model_artifacts,
        "validation_manifests": validation_manifests,
    })


def _sha256_token(value: Any, *, label: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"{label} is not a valid SHA-256")
    return token


def _immutable_reference_path(out: Path, source_contract_sha256: str) -> Path:
    """Return the model-local, source-contract-addressed evidence path."""

    token = _sha256_token(
        source_contract_sha256,
        label="management CPU reference source contract",
    )
    return out / "by_source_contract" / token / "canonical_cpu_reference.json"


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not a real directory: {path}")


def _ensure_real_directory(path: Path, *, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _require_real_directory(path, label=label)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _immutable_reference_thread_lock(lock_path: Path) -> threading.RLock:
    key = str(lock_path.resolve(strict=False))
    with _IMMUTABLE_REFERENCE_LOCKS_GUARD:
        lock = _IMMUTABLE_REFERENCE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _IMMUTABLE_REFERENCE_LOCKS[key] = lock
        return lock


@contextmanager
def _immutable_reference_lock(target: Path):
    """Serialize one source-contract publication across threads/processes."""

    store = target.parent.parent
    _ensure_real_directory(store, label="immutable reference store")
    contract_token = target.parent.name
    _sha256_token(contract_token, label="immutable reference directory")
    lock_path = store / f".{contract_token}.lock"
    local_lock = _immutable_reference_thread_lock(lock_path)
    with local_lock:
        flags = os.O_CREAT | os.O_RDWR | int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(str(lock_path), flags, 0o600)
        file_lock = None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError(
                    f"immutable reference lock is not a regular file: {lock_path}"
                )
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - Linux campaign
                raise RuntimeError(
                    "immutable reference publication requires POSIX file locking"
                ) from exc
            file_lock = fcntl
            file_lock.flock(descriptor, file_lock.LOCK_EX)
            yield
        finally:
            if file_lock is not None:
                try:
                    file_lock.flock(descriptor, file_lock.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)


def _sha256_regular_file(
    path: Path,
    *,
    label: str,
    require_single_link: bool = False,
) -> str:
    """Hash one regular non-symlink file through an ``O_NOFOLLOW`` handle."""

    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or (
        require_single_link and int(before.st_nlink) != 1
    ):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(str(path), flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (require_single_link and int(opened.st_nlink) != 1)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"{label} changed while it was opened: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if require_single_link and int(after.st_nlink) != 1:
            raise RuntimeError(f"{label} gained another hardlink while hashing: {path}")
        return "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def _copy_regular_file_to_fd(source: Path, destination_fd: int) -> str:
    """Copy a producer artifact to a staged fd and return its SHA-256."""

    before = os.lstat(source)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"generated reference is not a regular file: {source}")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    source_fd = os.open(str(source), flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"generated reference changed while it was opened: {source}")
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short write while staging immutable reference")
                view = view[written:]
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
    return "sha256:" + digest.hexdigest()


def _inspect_immutable_reference(
    target: Path,
) -> Optional[tuple[dict[str, Any], str]]:
    """Load a complete immutable artifact or reject a partial identity."""

    with _immutable_reference_lock(target):
        if not os.path.lexists(target.parent):
            return None
        _require_real_directory(
            target.parent,
            label="immutable reference identity directory",
        )
        if not os.path.lexists(target):
            raise RuntimeError(
                "immutable reference identity directory is incomplete; "
                f"refusing repair-in-place: {target.parent}"
            )
        digest = _sha256_regular_file(
            target,
            label="immutable management CPU reference",
            require_single_link=True,
        )
        metadata = _validate_generated_reference(target)
        return metadata, digest


def _publish_immutable_reference(
    source: Path,
    target: Path,
) -> tuple[dict[str, Any], str, bool]:
    """Atomically publish ``source`` without ever replacing ``target``.

    Returns validation metadata, the byte SHA-256, and whether an identical
    artifact already occupied the contract identity.  A different payload for
    the same identity is a hard collision.
    """

    source_metadata = _validate_generated_reference(source)
    source_digest = _sha256_regular_file(
        source,
        label="generated management CPU reference",
    )
    with _immutable_reference_lock(target):
        if os.path.lexists(target.parent):
            _require_real_directory(
                target.parent,
                label="immutable reference identity directory",
            )
            if not os.path.lexists(target):
                raise RuntimeError(
                    "immutable reference identity directory is incomplete; "
                    f"refusing repair-in-place: {target.parent}"
                )
            existing_digest = _sha256_regular_file(
                target,
                label="immutable management CPU reference",
                require_single_link=True,
            )
            if existing_digest != source_digest:
                raise RuntimeError(
                    "management CPU reference source-contract collision: "
                    f"existing={existing_digest} generated={source_digest}"
                )
            existing_metadata = _validate_generated_reference(target)
            return existing_metadata, existing_digest, True

        os.mkdir(target.parent, 0o700)
        _fsync_directory(target.parent.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".canonical_cpu_reference.",
            suffix=".staging",
            dir=str(target.parent),
        )
        temporary = Path(temporary_name)
        try:
            copied_digest = _copy_regular_file_to_fd(source, descriptor)
            os.close(descriptor)
            descriptor = -1
            if copied_digest != source_digest:
                raise RuntimeError(
                    "generated management CPU reference changed during publication"
                )
            try:
                os.link(str(temporary), str(target), follow_symlinks=False)
            except FileExistsError:
                existing_digest = _sha256_regular_file(
                    target,
                    label="immutable management CPU reference",
                    require_single_link=True,
                )
                if existing_digest != source_digest:
                    raise RuntimeError(
                        "management CPU reference source-contract collision: "
                        f"existing={existing_digest} generated={source_digest}"
                    )
                existing_metadata = _validate_generated_reference(target)
                return existing_metadata, existing_digest, True
            _fsync_directory(target.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        published_digest = _sha256_regular_file(
            target,
            label="published immutable management CPU reference",
            require_single_link=True,
        )
        if published_digest != source_digest:
            raise RuntimeError(
                "published management CPU reference failed its byte identity check"
            )
        published_metadata = _validate_generated_reference(target)
        if published_metadata != source_metadata:
            raise RuntimeError(
                "published management CPU reference failed semantic validation"
            )
        return published_metadata, published_digest, False


def _absolute_path_without_symlink_components(
    raw: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve spelling, but reject every existing symlink in the path."""

    expanded = Path(raw).expanduser()
    absolute = Path(os.path.abspath(str(expanded)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            continue
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink: {current}")
        if current != absolute and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(
                f"{label} has a non-directory parent component: {current}"
            )
    return absolute


def _read_diagnostic_status(path: Path) -> dict[str, Any]:
    """Read mutable status without following a pre-positioned symlink."""

    if not os.path.lexists(path):
        return {}
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"management CPU reference status is unavailable: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(
            f"management CPU reference status is not a regular file: {path}"
        )
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(str(path), flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(
                f"management CPU reference status changed while opening: {path}"
            )
        data = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            data.extend(block)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except Exception:
        # Historical status was not written atomically.  A malformed regular
        # diagnostic cannot attest legacy bytes, but must not conceal a valid
        # immutable artifact or prevent a fresh source-contract publication.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_diagnostic_status(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace mutable status without dereferencing special files."""

    _require_real_directory(path.parent, label="management reference status parent")
    if os.path.lexists(path) and not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError(
            f"management CPU reference status target is not regular: {path}"
        )
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".staging",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while staging management reference status")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if os.path.lexists(path) and not stat.S_ISREG(os.lstat(path).st_mode):
            raise RuntimeError(
                f"management CPU reference status target changed type: {path}"
            )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _open_diagnostic_stdout(path: Path):
    """Open mutable runner output without following symlinks/hardlinks."""

    _require_real_directory(path.parent, label="management reference stdout parent")
    before = os.lstat(path) if os.path.lexists(path) else None
    if before is not None and (
        not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1
    ):
        raise RuntimeError(
            f"management CPU reference stdout target is unsafe: {path}"
        )
    flags = os.O_WRONLY | os.O_CREAT
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(str(path), flags, 0o600)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or int(opened.st_nlink) != 1
        or (
            before is not None
            and (opened.st_dev != before.st_dev or opened.st_ino != before.st_ino)
        )
    ):
        os.close(descriptor)
        raise RuntimeError(
            f"management CPU reference stdout changed while opening: {path}"
        )
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        yield stream


def _same_location(raw: Any, expected: Path) -> bool:
    value = str(raw or "").strip()
    if not value or "\x00" in value:
        return False
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        return False
    candidate_normalized = Path(os.path.normpath(str(candidate)))
    expected_normalized = Path(os.path.normpath(str(expected.absolute())))
    return candidate_normalized == expected_normalized


def _successful_reference_status(
    *,
    status: str,
    model_id: str,
    workers: int,
    source_contract_sha256: str,
    reference_path: Path,
    reference_sha256: str,
    reference_metadata: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metadata = os.lstat(reference_path)
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
        raise RuntimeError(
            f"immutable management CPU reference is not a standalone regular file: {reference_path}"
        )
    return {
        "schema": REFERENCE_JOB_SCHEMA,
        "schema_version": 1,
        "created_at": now_iso(),
        "status": str(status),
        "model_id": str(model_id),
        "reference_path": str(reference_path),
        "reference_sha256": str(reference_sha256),
        "reference_size_bytes": int(metadata.st_size),
        "reference_storage": "immutable_source_contract",
        "reference_immutable": True,
        "execution_location": "central_management",
        "provider": "onnxruntime_cpu",
        "cpu_threads": max(1, int(workers)),
        "semantic_reference_only": True,
        "include_in_latency_fps_energy": False,
        "include_in_ranking": False,
        "include_in_pareto": False,
        "source_contract_sha256": source_contract_sha256,
        **dict(extra or {}),
        **dict(reference_metadata),
    }


def _verify_previous_reference_binding(
    previous_status: Mapping[str, Any],
    *,
    source_contract_sha256: str,
    immutable_path: Path,
    legacy_path: Path,
    observed_sha256: str,
) -> None:
    """Reject a mutable status that conflicts with existing immutable bytes."""

    try:
        previous_contract = _sha256_token(
            previous_status.get("source_contract_sha256"),
            label="previous management CPU reference source contract",
        )
    except ValueError:
        return
    current_contract = _sha256_token(
        source_contract_sha256,
        label="management CPU reference source contract",
    )
    if previous_contract != current_contract:
        return
    declared_path = str(previous_status.get("reference_path") or "").strip()
    declared_sha = str(previous_status.get("reference_sha256") or "").strip()
    if declared_path and not (
        _same_location(declared_path, immutable_path)
        or _same_location(declared_path, legacy_path)
    ):
        raise RuntimeError(
            "previous management CPU reference status points outside the "
            "current immutable/legacy contract location"
        )
    if declared_sha:
        declared_token = _sha256_token(
            declared_sha,
            label="previous management CPU reference SHA-256",
        )
        observed_token = _sha256_token(
            observed_sha256,
            label="immutable management CPU reference SHA-256",
        )
        if declared_token != observed_token:
            raise RuntimeError(
                "previous management CPU reference status conflicts with "
                "immutable reference bytes"
            )


def _terminate_reference_process(proc: subprocess.Popen[Any]) -> None:
    terminate_process_tree(proc, grace_s=3.0)


def _verify_reference_reuse_model_binding(
    *, model_id: str, contract: Mapping[str, Any], plan: Mapping[str, Any],
    previous_status: Mapping[str, Any], source_contract_sha256: str,
) -> None:
    """Cache reuse must retain the model guard that fresh Suite dispatch runs.

    BenchmarkSet.model is a model path in the real legacy generator;
    model_name is its logical identity when an explicit model_id is absent.
    Do not derive an ID from a filename or from the requested launcher ID.
    """
    expected = str(model_id or "").strip()
    suite_model_id = str(
        contract.get("model_id") or contract.get("model_name")
        or contract.get("model") or ""
    ).strip()
    previous_contract_matches = str(
        previous_status.get("source_contract_sha256") or ""
    ).removeprefix("sha256:") == str(source_contract_sha256).removeprefix("sha256:")
    # A mutable latest status from another generation cannot override the
    # current source's explicit identity (valid A -> B -> A reuse).
    previous_model_id = str(previous_status.get("model_id") or "").strip() if (
        previous_status.get("status") in {"completed", "cache_hit"}
        and (previous_contract_matches or not suite_model_id)
    ) else ""
    actual = {
        "benchmark_set_model_id": suite_model_id,
        "plan_model_id": str(plan.get("model_id") or "").strip(),
        "previous_status_model_id": previous_model_id,
    }
    # Declared BenchmarkSet contracts need their own source identity. Preserve
    # pre-schema legacy reference migration only with a successful model-bound
    # status; never admit an entirely unbound cached reference.
    missing_identity = not suite_model_id and (
        bool(contract.get("schema")) or not previous_model_id
    )
    if not expected or missing_identity or any(value and value != expected for value in actual.values()):
        raise RuntimeError(
            "management_cpu_reference_context_invalid:model_binding; "
            "model_binding=" + json.dumps({
                "expected_model_id": expected, **actual,
            }, sort_keys=True, ensure_ascii=True)
        )


def generate_management_cpu_reference(
    *,
    suite_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    workers: int = 4,
    cancel_event: Any = None,
    process_registry: ProcessTreeRegistry | None = None,
    log: Optional[Callable[[str], None]] = None,
    timeout_s: int = 10800,
) -> dict[str, Any]:
    """Generate/reuse one canonical ORT-CPU quality reference.

    ``workers`` configures CPU inference threads.  Paired bootstrap workers are
    configured independently in ``ManagementQualityService``.
    """
    source = Path(suite_dir).expanduser().resolve()
    out = _absolute_path_without_symlink_components(
        output_dir,
        label="management CPU reference output",
    )
    out.mkdir(parents=True, exist_ok=True)
    _require_real_directory(out, label="management CPU reference output")
    legacy_reference_out = out / "canonical_cpu_reference.json"
    status_out = out / "management_cpu_reference_status.json"
    stdout_out = out / "management_cpu_reference_stdout.txt"
    plan = read_json(source / "benchmark_plan.json", default={}) or {}
    contract = read_json(source / "benchmark_set.json", default={}) or {}
    source_contract_sha256 = _source_contract(source, plan, contract)
    reference_out = _immutable_reference_path(out, source_contract_sha256)
    try:
        previous_status = _read_diagnostic_status(status_out)
    except Exception as exc:
        return {
            "schema": REFERENCE_JOB_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "status": "failed",
            "model_id": str(model_id),
            "execution_location": "central_management",
            "source_contract_sha256": source_contract_sha256,
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        cached = _inspect_immutable_reference(reference_out)
        if cached is not None:
            _verify_reference_reuse_model_binding(
                model_id=model_id, contract=contract, plan=plan,
                previous_status=previous_status,
                source_contract_sha256=source_contract_sha256,
            )
            cached_reference, cached_sha256 = cached
            _verify_previous_reference_binding(
                previous_status,
                source_contract_sha256=source_contract_sha256,
                immutable_path=reference_out,
                legacy_path=legacy_reference_out,
                observed_sha256=cached_sha256,
            )
            payload = _successful_reference_status(
                status="cache_hit",
                model_id=str(model_id),
                workers=workers,
                source_contract_sha256=source_contract_sha256,
                reference_path=reference_out,
                reference_sha256=cached_sha256,
                reference_metadata=cached_reference,
            )
            _write_diagnostic_status(status_out, payload)
            return payload

        # v2.78.1 and earlier wrote a mutable top-level artifact.  It is only
        # admissible as a one-time migration source when its status binds the
        # exact current source contract and byte hash.  The returned evidence
        # path is always the immutable destination.
        previous_contract_matches = False
        try:
            previous_contract_matches = (
                _sha256_token(
                    previous_status.get("source_contract_sha256"),
                    label="previous management CPU reference source contract",
                )
                == _sha256_token(
                    source_contract_sha256,
                    label="management CPU reference source contract",
                )
            )
        except ValueError:
            previous_contract_matches = False
        if previous_contract_matches and os.path.lexists(legacy_reference_out):
            _verify_reference_reuse_model_binding(
                model_id=model_id, contract=contract, plan=plan,
                previous_status=previous_status,
                source_contract_sha256=source_contract_sha256,
            )
            if not _same_location(
                previous_status.get("reference_path"), legacy_reference_out,
            ):
                raise RuntimeError(
                    "legacy management CPU reference status has no exact path binding"
                )
            legacy_sha256 = _sha256_regular_file(
                legacy_reference_out,
                label="legacy management CPU reference",
            )
            if _sha256_token(
                previous_status.get("reference_sha256"),
                label="legacy management CPU reference status SHA-256",
            ) != _sha256_token(
                legacy_sha256,
                label="legacy management CPU reference SHA-256",
            ):
                raise RuntimeError(
                    "legacy management CPU reference status hash mismatch"
                )
            migrated_reference, migrated_sha256, _ = (
                _publish_immutable_reference(
                    legacy_reference_out,
                    reference_out,
                )
            )
            payload = _successful_reference_status(
                status="cache_hit",
                model_id=str(model_id),
                workers=workers,
                source_contract_sha256=source_contract_sha256,
                reference_path=reference_out,
                reference_sha256=migrated_sha256,
                reference_metadata=migrated_reference,
                extra={"legacy_reference_migrated": True},
            )
            _write_diagnostic_status(status_out, payload)
            return payload
    except Exception as exc:
        payload = {
            "schema": REFERENCE_JOB_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "status": "failed",
            "model_id": str(model_id),
            "execution_location": "central_management",
            "source_contract_sha256": source_contract_sha256,
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_diagnostic_status(status_out, payload)
        return payload

    cpu_run = _cpu_reference_run(plan)
    case_id = _first_case_id(contract)
    script = source / "benchmark_suite.py"
    preflight_errors: list[str] = []
    if cpu_run is None:
        preflight_errors.append("ort_cpu_run_missing")
    if not case_id:
        preflight_errors.append("benchmark_case_missing")
    if not script.is_file():
        preflight_errors.append("benchmark_suite_script_missing")
    if preflight_errors:
        payload = {
            "schema": REFERENCE_JOB_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "status": "failed",
            "model_id": str(model_id),
            "errors": preflight_errors,
            "execution_location": "central_management",
            "source_contract_sha256": source_contract_sha256,
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
        }
        _write_diagnostic_status(status_out, payload)
        return payload

    workspace_parent = out / "workspaces"
    _ensure_real_directory(
        workspace_parent,
        label="management reference workspace parent",
    )
    workspace = Path(tempfile.mkdtemp(
        prefix="management_cpu_reference_",
        dir=str(workspace_parent),
    ))
    suite_copy = workspace / "suite"
    started = time.monotonic()
    lines: list[str] = []
    return_code: Optional[int] = None
    cancelled = False
    timed_out = False
    failure_stage = "reference_preparation"
    try:
        shutil.copytree(
            source,
            suite_copy,
            symlinks=True,
            copy_function=_link_or_copy,
            ignore=_ignore_clone,
        )
        reference_plan = dict(plan)
        reference_plan["runs"] = [cpu_run]
        reference_plan["planned_runs"] = [cpu_run]
        reference_plan["management_cpu_reference"] = {
            "model_id": str(model_id),
            "semantic_reference_only": True,
            "cpu_threads": max(1, int(workers)),
            "excluded_from_performance": True,
        }
        plan_path = write_json(suite_copy / "management_reference_plan.json", reference_plan)
        command = [
            sys.executable,
            str(suite_copy / "benchmark_suite.py"),
            "--plan", str(plan_path),
            "--run-id", "management_cpu_reference",
            # Bind the child's evidence ID to the launcher independently of
            # the generated BenchmarkSet's logical model identity.
            "--quality-evidence-model-id", str(model_id),
            "--case", case_id,
            "--warmup", "0",
            "--runs", "1",
            "--phase-runs", "0",
            "--throughput-frames", "0",
            "--throughput-warmup-frames", "0",
            "--no-plot",
            "--no-csv",
        ]
        environment = dict(os.environ)
        environment.update({
            "OMP_NUM_THREADS": str(max(1, int(workers))),
            "OMP_DYNAMIC": "FALSE",
            "ORT_NUM_THREADS": str(max(1, int(workers))),
            "ONNX_SPLITPOINT_CPU_THREADS": str(max(1, int(workers))),
            "ONNX_SPLITPOINT_QUALITY_EXECUTION_LOCATION": "central_management",
            "ONNX_SPLITPOINT_CPU_REFERENCE_ONLY": "1",
        })
        if callable(log):
            log(
                f"[quality][management] generating CPU reference model={model_id} "
                f"case={case_id} threads={max(1, int(workers))}"
            )
        cancelled = bool(
            (
                cancel_event is not None
                and getattr(cancel_event, "is_set", lambda: False)()
            )
            or (
                process_registry is not None
                and process_registry.cancelled
            )
        )
        with _open_diagnostic_stdout(stdout_out) as output_stream:
            if cancelled:
                return_code = 130
                output_stream.write(
                    "[management-reference] CANCELLED before process start\n"
                )
            else:
                failure_stage = "reference_process"
                proc = subprocess.Popen(
                    command,
                    cwd=str(suite_copy),
                    env=environment,
                    stdout=output_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=(os.name == "posix"),
                )
                if process_registry is not None:
                    process_registry.register(
                        proc, label=f"management-reference:{model_id}"
                    )
                try:
                    deadline = time.monotonic() + max(1, int(timeout_s))
                    while proc.poll() is None:
                        cancelled = bool(
                            (
                                cancel_event is not None
                                and getattr(
                                    cancel_event, "is_set", lambda: False
                                )()
                            )
                            or (
                                process_registry is not None
                                and process_registry.cancelled
                            )
                        )
                        if cancelled:
                            _terminate_reference_process(proc)
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            _terminate_reference_process(proc)
                            lines.append("[management-reference] timeout")
                            break
                        time.sleep(0.1)
                    try:
                        return_code = proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        _terminate_reference_process(proc)
                        return_code = proc.wait(timeout=15)
                except KeyboardInterrupt:
                    cancelled = True
                    _terminate_reference_process(proc)
                    return_code = proc.wait(timeout=15)
                finally:
                    if process_registry is not None:
                        process_registry.unregister(proc)
                # Only a cancellation observed while this process was active
                # is its cause. A later global cancellation must not relabel
                # an already completed process exception.
            if lines:
                output_stream.write("\n" + "\n".join(lines) + "\n")
        diagnostic = _reference_stdout_diagnostic(stdout_out)
        if (
            not cancelled and not timed_out and return_code < 0
            and not diagnostic.get("exception_summary")
            and (
                (cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)())
                or (process_registry is not None and process_registry.cancelled)
            )
        ):
            # The shared registry may have terminated the child between two
            # polls. Preserve a captured process exception, but identify a
            # signalled cancellation even when this loop did not send it.
            cancelled = True
        if callable(log):
            for line in diagnostic.get("important_lines", []):
                log("[quality][management] " + line)
        generated = _quality_reference_file(suite_copy)
        generated_reference: Optional[dict[str, Any]] = None
        published_sha256 = ""
        error = ""
        exception_type = ""
        exception_message = ""
        if not cancelled and not timed_out and return_code == 0 and generated is not None and generated.is_file():
            failure_stage = "reference_publication"
            try:
                (
                    generated_reference,
                    published_sha256,
                    publication_cache_hit,
                ) = _publish_immutable_reference(generated, reference_out)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                exception_type = type(exc).__name__
                exception_message = str(exc)
                if isinstance(exc, ValueError):
                    failure_stage = "reference_validation"
                errors = [
                    f"quality_reference_publication_failed:"
                    f"{type(exc).__name__}:{exc}"
                ]
            else:
                status = "cache_hit" if publication_cache_hit else "completed"
                errors = []
        else:
            status = "cancelled" if cancelled else "failed"
            failure_stage = "reference_process"
            if cancelled:
                error = "reference_cancelled"
            elif timed_out:
                error = f"reference_runner_timeout: budget_s={max(1, int(timeout_s))}"
            elif return_code != 0:
                error = str(diagnostic.get("exception_summary") or f"reference_runner_rc_{return_code}")
                exception_type = str(diagnostic.get("exception_type") or "")
                exception_message = str(diagnostic.get("exception_message") or "")
            else:
                error = "quality_reference_not_emitted"
                failure_stage = "reference_output"
            errors = [error]
            if generated is None and not cancelled and error != "quality_reference_not_emitted":
                errors.append("quality_reference_not_emitted")
        extra = {
            "case_id": case_id,
            "return_code": int(return_code),
            "duration_s": round(max(0.0, time.monotonic() - started), 6),
            "stdout_path": str(stdout_out),
            "errors": errors,
            "cancelled": cancelled,
            "timed_out": timed_out,
        }
        if status in {"completed", "cache_hit"} and generated_reference is not None:
            payload = _successful_reference_status(
                status=status,
                model_id=str(model_id),
                workers=workers,
                source_contract_sha256=source_contract_sha256,
                reference_path=reference_out,
                reference_sha256=published_sha256,
                reference_metadata=generated_reference,
                extra=extra,
            )
        else:
            payload = {
                "schema": REFERENCE_JOB_SCHEMA,
                "schema_version": 1,
                "created_at": now_iso(),
                "status": status,
                "model_id": str(model_id),
                "execution_location": "central_management",
                "source_contract_sha256": source_contract_sha256,
                "reference_storage": "immutable_source_contract",
                "reference_immutable": True,
                "provider": "onnxruntime_cpu",
                "cpu_threads": max(1, int(workers)),
                "semantic_reference_only": True,
                "include_in_latency_fps_energy": False,
                "include_in_ranking": False,
                "include_in_pareto": False,
                **extra,
                "error": error,
                "failure_stage": failure_stage,
                "exception_type": exception_type,
                "exception_message": exception_message,
            }
            if diagnostic.get("stdout_excerpt"):
                payload["stdout_excerpt"] = diagnostic["stdout_excerpt"]
                payload["stdout_excerpt_is_tail"] = True
            if callable(log):
                log(
                    f"[quality][management][err] model={model_id} case={case_id} "
                    f"return_code={return_code} {error[:4096]}; stdout_path={stdout_out}"
                )
        _write_diagnostic_status(status_out, payload)
        return payload
    except Exception as exc:
        payload = {
            "schema": REFERENCE_JOB_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "status": "failed",
            "model_id": str(model_id),
            "duration_s": round(max(0.0, time.monotonic() - started), 6),
            "case_id": case_id,
            "return_code": return_code,
            "stdout_path": str(stdout_out),
            "failure_stage": failure_stage,
            "cancelled": cancelled,
            "timed_out": timed_out,
            "execution_location": "central_management",
            "source_contract_sha256": source_contract_sha256,
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
            "error": f"{type(exc).__name__}: {exc}",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "provider": "onnxruntime_cpu",
            "semantic_reference_only": True,
            "include_in_latency_fps_energy": False,
            "include_in_ranking": False,
            "include_in_pareto": False,
        }
        if callable(log):
            log(
                f"[quality][management][err] model={model_id} case={case_id} "
                f"return_code={return_code} {payload['error'][:4096]}; stdout_path={stdout_out}"
            )
        _write_diagnostic_status(status_out, payload)
        return payload
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


__all__ = [
    "REFERENCE_JOB_SCHEMA",
    "bind_management_cpu_reference_runs",
    "finalize_management_cpu_reference_plan_aliases",
    "generate_management_cpu_reference",
    "is_cpu_reference_recipe",
    "management_cpu_reference_required",
    "profile_has_explicit_cpu_reference",
]
