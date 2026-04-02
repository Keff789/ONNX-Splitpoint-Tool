from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


# Patterns are matched against POSIX-style relative paths ("foo/bar.txt").
# IMPORTANT: we only ever add *files* to the archive. This avoids a common pitfall
# where tar.add(directory) recursively pulls in files that the outer traversal will
# also add again.
DEFAULT_EXCLUDES: list[str] = [
    # tool output folders
    "dist",
    "dist/*",
    "Results",
    "Results/*",
    "results",
    "results/*",
    "results_*",
    "results_*/*",
    "*/results_*",
    "*/results_*/*",
    # typical noise
    "benchmark_report_*.json",
    "*.log",
    "__pycache__",
    "__pycache__/*",
    "*/__pycache__",
    "*/__pycache__/*",
    ".pytest_cache",
    ".pytest_cache/*",
    "*/.pytest_cache",
    "*/.pytest_cache/*",
    ".git",
    ".git/*",
    "*/.git",
    "*/.git/*",
    ".idea",
    ".idea/*",
    "*/.idea",
    "*/.idea/*",
    "node_modules",
    "node_modules/*",
    "*/node_modules",
    "*/node_modules/*",
]

# Minimal allowlist for remote benchmark execution. This intentionally excludes
# heavy analysis artefacts (PDFs/SVGs/context graphs/README noise) and only
# keeps files that the remote harness needs to execute cases.
REMOTE_MINIMAL_INCLUDES: list[str] = [
    "benchmark_plan.json",
    "benchmark_set.json",
    "benchmark_suite.py",
    "schemas/**",
    "splitpoint_runners/**",
    "models/**",
    "resources/validation/**",
    "hailo/**",
    "b*/split_manifest.json",
    "b*/run_split_onnxruntime.py",
    "b*/resources/test_images/**",
    "b*/*.onnx",
    "b*/hailo/**",
]

REMOTE_MINIMAL_EXCLUDES: list[str] = [
    "README_BENCHMARK.txt",
    "analysis_plots/**",
    "analysis_tables/**",
    "**/split_context*.pdf",
    "**/split_context*.svg",
    "**/split_context*.dot",
    "**/*.orig",
]


def remote_minimal_bundle_patterns() -> tuple[list[str], list[str]]:
    return list(REMOTE_MINIMAL_INCLUDES), list(REMOTE_MINIMAL_EXCLUDES)


class BundleCancelled(Exception):
    """Raised when suite bundling is cancelled."""


@dataclass
class BundleStats:
    files: int
    total_bytes: int
    bundle_path: Path
    reused: bool = False


def _should_include(rel_posix: str, includes: Optional[Sequence[str]]) -> bool:
    if not includes:
        return True
    for pat in includes:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def _should_exclude(rel_posix: str, excludes: Sequence[str]) -> bool:
    for pat in excludes:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_suite_bundle(
    suite_dir: Path,
    out_path: Path,
    *,
    excludes: Optional[Sequence[str]] = None,
    includes: Optional[Sequence[str]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    reuse_if_unchanged: bool = True,
    manifest_path: Optional[Path] = None,
) -> BundleStats:
    """Create a portable tar.gz bundle for a benchmark suite directory.

    The bundle is extracted on a remote host and must be self-contained.

    Notes:
    - We add files only (no directories) to avoid recursive duplication.
    - We write atomically via a .tmp file and then os.replace().
    - Optional cache: if a manifest exists and matches current suite, reuse the
      existing bundle (skip re-pack).
    """

    suite_dir = Path(suite_dir).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = out_path.with_name(out_path.name + ".manifest.json")

    excludes = list(excludes) if excludes is not None else list(DEFAULT_EXCLUDES)
    includes = list(includes) if includes is not None else None

    out_path_resolved: Optional[Path]
    try:
        out_path_resolved = out_path.resolve()
    except Exception:
        out_path_resolved = None

    # Collect file list for stable progress + caching manifest.
    files: list[tuple[Path, str, int, int, str]] = []  # (path, rel_posix, size, mtime_ns, sha256)
    total_bytes = 0
    for p in suite_dir.rglob("*"):
        if should_cancel and should_cancel():
            raise BundleCancelled("cancelled")

        rel = p.relative_to(suite_dir)
        rel_posix = rel.as_posix()
        if not _should_include(rel_posix, includes):
            continue
        if _should_exclude(rel_posix, excludes):
            continue

        # Add files only.
        try:
            if p.is_dir():
                continue
        except OSError:
            continue

        if out_path_resolved is not None:
            try:
                if p.resolve() == out_path_resolved:
                    continue
            except Exception:
                pass

        try:
            st = p.stat()
            size = int(st.st_size)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        except OSError:
            size = 0
            mtime_ns = 0

        sha256 = _sha256_file(p)
        files.append((p, rel_posix, size, mtime_ns, sha256))
        total_bytes += size

    files.sort(key=lambda t: t[1])

    total_files = len(files)
    if progress_cb:
        mb = total_bytes / (1024 * 1024) if total_bytes else 0.0
        progress_cb(0.0, f"Found {total_files} files ({mb:.1f} MB)")

    manifest = {
        "version": 2,
        "suite_dir": str(suite_dir),
        "files": [
            {"rel": rel_posix, "size": size, "mtime_ns": mtime_ns, "sha256": sha256}
            for (_p, rel_posix, size, mtime_ns, sha256) in files
        ],
        "total_bytes": int(total_bytes),
    }

    # Cache check
    if reuse_if_unchanged and out_path.exists() and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            old = None

        if isinstance(old, dict) and old.get("version") == 2 and old.get("files") == manifest["files"]:
            if progress_cb:
                progress_cb(1.0, "Using cached bundle")
            return BundleStats(files=total_files, total_bytes=total_bytes, bundle_path=out_path, reused=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    with tarfile.open(tmp_path, "w:gz") as tf:
        # Throttle UI updates: aim for ~200 progress updates max.
        step = 1 if total_files <= 200 else max(1, total_files // 200)
        for idx, (p, rel_posix, size, _mtime_ns, _sha256) in enumerate(files, start=1):
            if should_cancel and should_cancel():
                raise BundleCancelled("cancelled")

            # Tell the UI which file we're currently adding. This is important when
            # a *single* large file takes long to compress (otherwise it looks stuck).
            if progress_cb and total_files and (idx == 1 or idx == total_files or (idx % step == 0)):
                try:
                    sz_mb = size / (1024 * 1024)
                    progress_cb((idx - 1) / total_files, f"Adding {idx}/{total_files}: {rel_posix} ({sz_mb:.1f} MB)")
                except Exception:
                    progress_cb((idx - 1) / total_files, f"Adding {idx}/{total_files}: {rel_posix}")

            tf.add(p, arcname=rel_posix, recursive=False)

            if progress_cb and total_files and (idx == 1 or idx == total_files or (idx % step == 0)):
                progress_cb(idx / total_files, f"Bundled {idx}/{total_files}: {rel_posix}")

    os.replace(tmp_path, out_path)
    _write_json_atomic(manifest_path, manifest)

    return BundleStats(files=total_files, total_bytes=total_bytes, bundle_path=out_path, reused=False)
