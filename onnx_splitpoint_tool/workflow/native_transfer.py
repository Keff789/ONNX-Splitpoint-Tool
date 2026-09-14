from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}

# Native runners need models, accelerator artefacts, contracts, small validation
# subsets and optional prebuilt native engines.  They do not need Generic Runner
# result trees, thesis reports, or the several-hundred-MiB suite transport bundle.
_EXCLUDED_DIR_NAMES = {
    "dist",
    "scientific_report",
    "reports",
    "remote_diagnostics",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "analysis_plots",
    "activation_calibration",
    "calibration_cache",
}
_EXCLUDED_DIR_PREFIXES = ("results_",)
_EXCLUDED_FILE_GLOBS = (
    "benchmark_results*.json",
    "benchmark_results*.csv",
    "results_bundle*.tar.gz",
    "suite_bundle*.tar.gz",
    "*.zip",
    "*.lock",
    "*.pyc",
    "*.pyo",
    "*.har",
    "*.log",
    "*.pdf",
    "baseline_full_cpu_outputs.npz",
)


def _is_excluded_relative_path(rel: Path) -> tuple[bool, str]:
    parts = rel.parts
    for part in parts[:-1]:
        low = part.lower()
        if low in _EXCLUDED_DIR_NAMES:
            return True, f"dir:{part}"
        if any(low.startswith(prefix) for prefix in _EXCLUDED_DIR_PREFIXES):
            return True, f"dir_prefix:{part}"
        # Native pipeline outputs are run products.  Never seed a new remote
        # run with stale outputs from a previous attempt.
        if low == "native_pipeline":
            return True, "dir:native_pipeline"
    name = rel.name
    for pat in _EXCLUDED_FILE_GLOBS:
        if fnmatch.fnmatch(name, pat):
            return True, f"file:{pat}"
    return False, ""


@dataclass(frozen=True)
class NativeTransferEntry:
    relative_path: str
    size_bytes: int


def build_native_transfer_inventory(root: str | Path) -> dict[str, Any]:
    """Build a deterministic, lean Native Runner transfer inventory.

    The inventory is intentionally file-based so the exact same set can be
    passed to rsync via ``--files-from``.  This avoids copying large Generic
    Runner bundles such as ``dist/suite_bundle.tar.gz`` to every Native host.
    """
    base = Path(root).expanduser().resolve()
    entries: list[NativeTransferEntry] = []
    excluded_count = 0
    excluded_bytes = 0
    excluded_reasons: dict[str, int] = {}
    largest_excluded: list[dict[str, Any]] = []
    if not base.is_dir():
        raise FileNotFoundError(f"Native benchmark-set root not found: {base}")
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            # Excluded previous run products must not participate in either
            # transfer admission or artifact validation.
            if _is_excluded_relative_path(path.relative_to(base))[0]:
                continue
            # The atomic Hailo publisher deliberately exposes its sealed
            # generation through four local links. rsync -a must receive these
            # links as well as the generation files, otherwise Full readers
            # see a missing HEF after a successful transfer.
            from ..hailo_cache_bundle import (
                POINTER_NAME, RECEIPT_NAME, META_NAME,
                resolve_published_bundle_member,
            )
            if path.name not in {POINTER_NAME, RECEIPT_NAME, META_NAME} and path.suffix != ".hef":
                continue
            if resolve_published_bundle_member(path, root=base) is None:
                raise ValueError(
                    "native_transfer_invalid_hailo_bundle_alias:"
                    + path.relative_to(base).as_posix()
                )
        elif not path.is_file():
            continue
        rel = path.relative_to(base)
        excluded, reason = _is_excluded_relative_path(rel)
        try:
            size = int(path.lstat().st_size)
        except OSError:
            size = 0
        if excluded:
            excluded_count += 1
            excluded_bytes += size
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
            largest_excluded.append({"path": rel.as_posix(), "size_bytes": size, "reason": reason})
            continue
        entries.append(NativeTransferEntry(rel.as_posix(), size))
    largest = sorted(
        ({"path": e.relative_path, "size_bytes": e.size_bytes} for e in entries),
        key=lambda row: int(row["size_bytes"]),
        reverse=True,
    )[:20]
    largest_excluded = sorted(largest_excluded, key=lambda row: int(row["size_bytes"]), reverse=True)[:20]
    total_bytes = sum(e.size_bytes for e in entries)
    payload = {
        "schema": "onnx-splitpoint/native-transfer-inventory",
        "schema_version": 1,
        "root": str(base),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "relative_paths": [e.relative_path for e in entries],
        "files": [{"path": e.relative_path, "size_bytes": e.size_bytes} for e in entries],
        "largest_files": largest,
        "excluded_file_count": excluded_count,
        "excluded_bytes": excluded_bytes,
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "largest_excluded_files": largest_excluded,
    }
    digest_input = json.dumps(
        [(e.relative_path, e.size_bytes) for e in entries],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    payload["inventory_sha256"] = hashlib.sha256(digest_input).hexdigest()
    return payload


def write_rsync_files_from(inventory: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p) for p in list(inventory.get("relative_paths") or []) if str(p).strip()]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def parse_df_available_bytes(text: str) -> int | None:
    """Parse the last numeric available-block value from ``df -Pk`` output."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        cols = line.split()
        # POSIX df: Filesystem 1024-blocks Used Available Capacity Mounted on
        if len(cols) >= 4:
            try:
                return int(cols[3]) * 1024
            except (TypeError, ValueError):
                pass
        nums = re.findall(r"\b\d+\b", line)
        if nums:
            try:
                return int(nums[-1]) * 1024
            except ValueError:
                pass
    return None


def classify_native_transfer_failure(stderr: str, stdout: str = "") -> str:
    text = (str(stderr or "") + "\n" + str(stdout or "")).lower()
    if (
        "central_native_split_quality_binding_set_unavailable" in text
        or "upstream_central_quality_binding_missing" in text
        or "central-quality producer set" in text and "missing" in text
    ):
        return "upstream_central_quality_binding_missing"
    # Runtime/import failures happen after a successful transfer and must not be
    # mislabeled as rsync/native-transfer errors.
    if (
        "modulenotfounderror" in text
        or "importerror" in text
        or "no module named" in text
    ) and (
        "onnx_splitpoint_tool" in text
        or "native_progress" in text
        or "native_full_baseline_eval_runner" in text
    ):
        return "remote_runner_import_failed"
    if "no space left on device" in text or "errno 28" in text or "error in file io (code 11)" in text:
        return "remote_disk_insufficient"
    if "permission denied" in text:
        return "remote_permission_denied"
    if "connection refused" in text or "connection timed out" in text or "no route to host" in text:
        return "remote_connection_failed"
    if "rsync" in text:
        return "remote_rsync_failed"
    if "native_full_baseline_eval_runner" in text:
        return "native_full_runner_failed"
    return "native_transfer_failed"


def required_remote_bytes(payload_bytes: int, *, minimum_margin_bytes: int = 128 * 1024 * 1024, ratio: float = 0.15) -> int:
    size = max(0, int(payload_bytes or 0))
    return size + max(int(minimum_margin_bytes), int(size * max(0.0, float(ratio))))


def safe_native_remote_root(remote_root: str, run_id: str) -> bool:
    """Return whether a remote root is safe for tool-owned cleanup.

    The path must be absolute, contain the current run ID as its final element,
    and live below the dedicated ``native_fifo_evalsets`` hierarchy.
    """
    text = str(remote_root or "").strip()
    rid = str(run_id or "").strip()
    if not text.startswith("/") or not rid:
        return False
    norm = os.path.normpath(text)
    return Path(norm).name == rid and "native_fifo_evalsets" in Path(norm).parts


def _extract_report_image(report: str | Path) -> str:
    p = Path(report)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for path in (("run_cfg", "image"), ("viz", "image"), ("benchmark_input_policy", "image")):
        cur: Any = payload
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur:
            return str(cur)
    return ""


def build_native_validation_image_map(
    run_dir: str | Path,
    models: Sequence[str],
    case_map: Mapping[str, Sequence[str]],
    benchmark_sets: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Resolve the exact Generic-Runner validation image per Native case.

    The preferred source is the ORT-CPU validation report because it records the
    exact image used for the canonical self-reference.  If unavailable, the
    function selects a deterministic image from the run-mode materialised
    validation subset.  Returned values are basenames/relative paths so they can
    be resolved inside the copied remote BenchmarkSet.
    """
    root = Path(run_dir).expanduser().resolve()
    image_map: dict[str, dict[str, str]] = {}
    sources: dict[str, dict[str, str]] = {}
    bs_map = {str(k): Path(v) for k, v in dict(benchmark_sets or {}).items()}
    for model in models:
        for case in list(case_map.get(model) or []):
            candidates = [
                root / "models" / model / "benchmark_results" / "remote_diagnostics" / "case_reports" / "results" / case / "results_ort_cpu" / "validation_report.json",
                root / "models" / model / "benchmark_results" / "remote_diagnostics" / "lean_bundle" / case / "results_ort_cpu" / "validation_report.json",
            ]
            # Some target-specific folders retain the only canonical report.
            candidates.extend(sorted((root / "models" / model / "benchmark_results" / "remote_diagnostics").glob(f"*/case_reports/results/{case}/results_ort_cpu/validation_report.json")))
            candidates.extend(sorted((root / "models" / model / "benchmark_results" / "remote_diagnostics").glob(f"*/lean_bundle/{case}/results_ort_cpu/validation_report.json")))
            chosen = ""
            source = ""
            for report in candidates:
                if not report.is_file():
                    continue
                value = _extract_report_image(report)
                if value:
                    chosen = Path(value).name
                    source = f"generic_ort_validation:{report.relative_to(root).as_posix()}"
                    break
            if not chosen:
                bs = bs_map.get(model) or root / "models" / model / "benchmark_set"
                # Prefer the executable nested suite if present.
                if (bs / "legacy_suite").is_dir():
                    bs = bs / "legacy_suite"
                val_root = bs / "resources" / "validation"
                images = sorted(
                    p for p in val_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ) if val_root.is_dir() else []
                if images:
                    chosen = images[0].relative_to(bs).as_posix()
                    source = "materialised_validation_subset:first_sorted"
            if chosen:
                image_map.setdefault(model, {})[str(case)] = chosen
                sources.setdefault(model, {})[str(case)] = source
    return image_map, sources


__all__ = [
    "build_native_transfer_inventory",
    "write_rsync_files_from",
    "parse_df_available_bytes",
    "classify_native_transfer_failure",
    "required_remote_bytes",
    "safe_native_remote_root",
    "build_native_validation_image_map",
]
