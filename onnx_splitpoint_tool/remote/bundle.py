from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import tarfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from onnx_splitpoint_tool.hailo_cache_bundle import resolve_published_bundle_member


try:  # POSIX file locks; available on Linux testbeds.
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_bundle_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive_bundle_lock(
    out_path: Path,
    *,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
):
    """Serialize discovery, hashing and archive creation for one suite.

    ``flock`` protects against other processes, while the process-local lock is
    always acquired as well.  The latter is important because Evaluation
    Workflow dispatches setup workers as threads and all of them target the same
    local ``dist/suite_bundle.tar.gz``.
    """
    lock_path = Path(str(out_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_notice = -10.0

    thread_lock = _thread_bundle_lock(lock_path)
    thread_acquired = False
    while not thread_acquired:
        if should_cancel and should_cancel():
            raise BundleCancelled("cancelled while waiting for bundle lock")
        thread_acquired = thread_lock.acquire(timeout=0.5)
        elapsed = time.monotonic() - started
        if not thread_acquired and progress_cb and elapsed - last_notice >= 5.0:
            progress_cb(0.0, f"Waiting for shared bundle builder ({elapsed:.0f}s)")
            last_notice = elapsed

    fh = None
    file_acquired = False
    try:
        if fcntl is not None:
            fh = lock_path.open("a+")
            while not file_acquired:
                if should_cancel and should_cancel():
                    raise BundleCancelled("cancelled while waiting for bundle file lock")
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    file_acquired = True
                except BlockingIOError:
                    elapsed = time.monotonic() - started
                    if progress_cb and elapsed - last_notice >= 5.0:
                        progress_cb(0.0, f"Waiting for shared bundle file lock ({elapsed:.0f}s)")
                        last_notice = elapsed
                    time.sleep(0.25)
        if progress_cb and time.monotonic() - started >= 1.0:
            progress_cb(0.0, "Shared bundle lock acquired")
        yield
    finally:
        try:
            if fh is not None and file_acquired:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            if fh is not None:
                fh.close()
            if thread_acquired:
                thread_lock.release()


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
    "scientific_reporter_v60.py",
    "suite_model_artifacts.json",
    "suite_packaging_preflight.json",
    # Backend-specific endpoint semantics are a runtime input.  DeepX Full and
    # central-quality export bind their observed tensors to this recorded suite
    # contract; omitting it makes detection fail closed after transfer.
    "output_contracts.json",
    # v49j: formal generated suites place the full/reference ONNX at the
    # suite root and split_manifest.json references it as ../<model>.onnx.
    # The previous minimal bundle allowlist only kept b*/*.onnx and therefore
    # remote full_* runs failed with FileNotFoundError for ../<model>.onnx.
    "*.onnx",
    "*.onnx.data",
    "*.data",
    "*.bin",
    "external_data/**",
    "schemas/**",
    "splitpoint_runners/**",
    "models/**",
    "hailo/**",
    "deepx/**",
    "native_trt/**",
    "*.engine",
    "*.plan",
    "*.dxnn",
    "b*/deepx/**",
    "b*/native_trt/**",
    "b*/*.engine",
    "b*/*.plan",
    "b*/*.dxnn",
    "b*/split_manifest.json",
    "b*/run_split_onnxruntime.py",
    "b*/resources/test_images/**",
    "b*/*.onnx",
    "b*/*.onnx.data",
    "b*/*.data",
    "b*/*.bin",
    "b*/external_data/**",
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
    # v50 bundle slimming: remote runtime needs ONNX segments, scripts,
    # manifests and compiled HEFs.  Intermediate Hailo/compiler artefacts are
    # huge and should not be shipped unless a developer explicitly switches to
    # direct transfer/debug mode.
    "**/*.har",
    "**/*.harn",
    "**/*.hn",
    "**/*.hef.meta",
    "**/*quantized*.har",
    "**/*checkpoint*",
    "**/*.npz",
    "**/*.npy",
    "**/*.pdf",
    "**/*.svg",
    "**/*.dot",
    "**/paper_figures/**",
    "**/paper_figures_*",
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
    sha256: str = ""
    manifest_path: Optional[Path] = None


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



def _pattern_has_magic(text: str) -> bool:
    return any(ch in str(text) for ch in ("*", "?", "["))


def _candidate_scan_roots(suite_dir: Path, includes: Optional[Sequence[str]]) -> tuple[list[Path], list[str]]:
    """Return the smallest practical filesystem roots needed by ``includes``.

    ``Path.rglob`` over the suite root still walks an accidentally retained
    50,000-image validation tree even when the allowlist names only a 16-image
    subset.  Deriving literal include prefixes lets Smoke/Standard avoid touching
    unrelated dataset siblings at all.  Root-level wildcard patterns are handled
    separately and never trigger a recursive suite-wide walk.
    """
    if not includes:
        return [suite_dir], []

    roots: set[Path] = set()
    root_file_patterns: list[str] = []
    try:
        top_entries = list(suite_dir.iterdir())
    except OSError:
        top_entries = []

    for raw in includes:
        pat = str(raw or "").strip().replace("\\", "/").lstrip("./")
        if not pat:
            continue
        parts = [part for part in pat.split("/") if part not in {"", "."}]
        if not parts:
            continue
        literal: list[str] = []
        first_magic_index: Optional[int] = None
        for idx, part in enumerate(parts):
            if _pattern_has_magic(part):
                first_magic_index = idx
                break
            literal.append(part)

        if first_magic_index is None:
            candidate = suite_dir.joinpath(*literal)
            if candidate.exists():
                roots.add(candidate)
            continue

        if first_magic_index > 0:
            candidate = suite_dir.joinpath(*literal)
            if candidate.exists():
                roots.add(candidate)
            continue

        # The first path component contains a wildcard.  A pattern without a
        # slash (for example ``*.onnx``) applies only to suite-root files.
        if len(parts) == 1:
            root_file_patterns.append(pat)
            continue

        first_component = parts[0]
        for entry in top_entries:
            if fnmatch.fnmatch(entry.name, first_component):
                roots.add(entry)

    # De-duplicate nested roots: if ``models`` is already scanned, there is no
    # reason to scan ``models/foo`` again.
    ordered = sorted(roots, key=lambda x: (len(x.parts), str(x)))
    compact: list[Path] = []
    for candidate in ordered:
        try:
            if any(parent == candidate or parent in candidate.parents for parent in compact):
                continue
        except Exception:
            pass
        compact.append(candidate)
    return compact, root_file_patterns


def _iter_selected_candidates(suite_dir: Path, includes: Optional[Sequence[str]]):
    roots, root_file_patterns = _candidate_scan_roots(suite_dir, includes)
    yielded: set[str] = set()

    if root_file_patterns:
        try:
            for entry in suite_dir.iterdir():
                if not entry.is_file():
                    continue
                rel = entry.relative_to(suite_dir).as_posix()
                if any(fnmatch.fnmatch(rel, pat) for pat in root_file_patterns):
                    key = str(entry)
                    if key not in yielded:
                        yielded.add(key)
                        yield entry
        except OSError:
            pass

    for root in roots:
        try:
            if root.is_file():
                candidates = (root,)
            elif root == suite_dir:
                candidates = root.rglob("*")
            else:
                candidates = root.rglob("*")
            for candidate in candidates:
                key = str(candidate)
                if key in yielded:
                    continue
                yielded.add(key)
                yield candidate
        except OSError:
            continue

def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    """Compatibility hash helper (may be wrapped by the v60m policy)."""
    return _raw_content_digest(path)


def _raw_content_digest(path: Path) -> str:
    """Hash a file once without the per-file persistent-cache wrapper.

    The old development wrapper rewrote a growing JSON cache after every file.
    For a 50,000-file ImageNet tree that turns one linear pass into effectively
    quadratic metadata I/O.  Bundle reuse is metadata-first in relaxed modes;
    strict mode calls this direct streaming helper exactly once per file.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
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
    """Create a portable suite bundle with setup-safe reuse and progress.

    Relaxed Smoke/Standard modes compare the selected file inventory by
    ``(relative path, size, mtime)`` and do not hash every input file.  The
    resulting archive itself is still SHA-256-addressed for remote transport.
    Strict Final mode hashes every selected input once.  All work is protected by
    one shared lock so parallel setup workers cannot independently hash or build
    the same local archive.
    """
    suite_dir = Path(suite_dir).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    manifest_path = Path(manifest_path).expanduser().resolve()

    excludes = list(excludes) if excludes is not None else list(DEFAULT_EXCLUDES)
    includes = list(includes) if includes is not None else None
    integrity_mode = str(os.environ.get("ONNX_SPLITPOINT_INTEGRITY_MODE") or "fast").strip().lower()
    strict_mode = integrity_mode == "strict"
    try:
        gzip_level = int(os.environ.get("ONNX_SPLITPOINT_BUNDLE_GZIP_LEVEL") or (6 if strict_mode else 1))
    except Exception:
        gzip_level = 6 if strict_mode else 1
    gzip_level = max(0, min(9, gzip_level))

    try:
        out_path_resolved: Optional[Path] = out_path.resolve()
    except Exception:
        out_path_resolved = None

    def _archive_digest(existing_manifest: Optional[dict]) -> str:
        old_digest = str((existing_manifest or {}).get("bundle_sha256") or "").strip().lower()
        old_size = int((existing_manifest or {}).get("bundle_bytes") or 0)
        try:
            current_size = int(out_path.stat().st_size)
        except OSError:
            current_size = 0
        if old_digest and (old_size <= 0 or old_size == current_size):
            return old_digest
        if progress_cb:
            progress_cb(0.99, "Hashing completed suite archive")
        return _raw_content_digest(out_path)

    with _exclusive_bundle_lock(out_path, progress_cb=progress_cb, should_cancel=should_cancel):
        meta_rows: list[tuple[Path, str, int, int]] = []
        hailo_snapshots: dict[Path, Path] = {}
        hailo_generation_links: dict[str, str] = {}
        total_bytes = 0
        last_notice = time.monotonic()
        if progress_cb:
            progress_cb(0.0, "Scanning suite files")
        scan_roots, root_file_patterns = _candidate_scan_roots(suite_dir, includes)
        if progress_cb and includes:
            progress_cb(
                0.0,
                f"Scanning {len(scan_roots)} selected root(s)"
                + (f" and {len(root_file_patterns)} suite-root pattern(s)" if root_file_patterns else ""),
            )
        for p in _iter_selected_candidates(suite_dir, includes):
            if should_cancel and should_cancel():
                raise BundleCancelled("cancelled")
            try:
                rel_posix = p.relative_to(suite_dir).as_posix()
            except Exception:
                continue
            # Historical backups are local retention, not runtime artifacts.
            # Pin each public triplet to one generation before inventory/hash/
            # archive creation so a concurrent publication cannot mix siblings.
            if any(part in {".hailo-generations", ".hailo-current", ".hailo-publish.lock"}
                   for part in Path(rel_posix).parts):
                continue
            if not _should_include(rel_posix, includes) or _should_exclude(rel_posix, excludes):
                continue
            if p.is_symlink() and (
                os.readlink(p) == str(Path(".hailo-current") / p.name)
                or (
                    os.path.lexists(p.parent / ".hailo-current")
                    and (p.suffix == ".hef" or p.name in {
                        "hailo_hef_build_receipt.json", "cache_meta.json",
                    })
                )
            ):
                logical_parent = p.parent
                if os.readlink(p) != str(Path(".hailo-current") / p.name):
                    raise ValueError("hailo_bundle_published_generation_invalid")
                if logical_parent not in hailo_snapshots:
                    selected = resolve_published_bundle_member(p, root=suite_dir)
                    if selected is None:
                        raise ValueError("hailo_bundle_published_generation_invalid")
                    hailo_snapshots[logical_parent] = selected.parent
                p = hailo_snapshots[logical_parent] / p.name
                if not p.is_file() or p.is_symlink():
                    raise ValueError("hailo_bundle_published_file_missing")
                # Full output contracts bind the immutable generation path.
                # Preserve that path as a tar hardlink to the existing public
                # regular file: one payload, identical HEF/receipt/meta bytes,
                # no rewriting contracts and no transfer of older generations.
                hailo_generation_links[p.relative_to(suite_dir).as_posix()] = rel_posix
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
            meta_rows.append((p, rel_posix, size, mtime_ns))
            total_bytes += size
            now = time.monotonic()
            if progress_cb and (len(meta_rows) % 1000 == 0 or now - last_notice >= 5.0):
                progress_cb(0.0, f"Scanning suite: {len(meta_rows)} included files ({total_bytes / (1024*1024):.1f} MB)")
                last_notice = now
        meta_rows.sort(key=lambda row: row[1])
        total_files = len(meta_rows)
        metadata_inventory = [
            {"rel": rel, "size": size, "mtime_ns": mtime_ns}
            for (_p, rel, size, mtime_ns) in meta_rows
        ]
        generation_links = [
            {"path": name, "target": target}
            for name, target in sorted(hailo_generation_links.items())
        ]
        if progress_cb:
            progress_cb(0.03, f"Found {total_files} files ({total_bytes / (1024*1024):.1f} MB)")

        old: Optional[dict] = None
        if reuse_if_unchanged and out_path.exists() and manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                old = loaded if isinstance(loaded, dict) else None
            except Exception:
                old = None
        old_metadata = [
            {
                "rel": str(row.get("rel") or ""),
                "size": int(row.get("size") or 0),
                "mtime_ns": int(row.get("mtime_ns") or 0),
            }
            for row in list((old or {}).get("files") or [])
            if isinstance(row, dict)
        ]
        patterns_match = bool(
            isinstance(old, dict)
            and list(old.get("include_patterns") or []) == list(includes or [])
            and list(old.get("exclude_patterns") or []) == list(excludes or [])
            and list(old.get("hailo_generation_links") or []) == generation_links
        )
        if (
            not strict_mode
            and reuse_if_unchanged
            and out_path.exists()
            and isinstance(old, dict)
            and old_metadata == metadata_inventory
            and patterns_match
        ):
            digest = _archive_digest(old)
            upgraded = dict(old)
            upgraded.update({
                "version": 7,
                "cache_policy": "v60q_locked_metadata_first_deterministic_bundle",
                "integrity_mode": integrity_mode,
                "gzip_level": int(old.get("gzip_level") if old.get("gzip_level") is not None else gzip_level),
                "bundle_sha256": digest,
                "bundle_bytes": int(out_path.stat().st_size),
            })
            _write_json_atomic(manifest_path, upgraded)
            if progress_cb:
                progress_cb(1.0, "Using cached bundle (metadata unchanged)")
            return BundleStats(
                files=total_files,
                total_bytes=total_bytes,
                bundle_path=out_path,
                reused=True,
                sha256=digest,
                manifest_path=manifest_path,
            )

        files: list[tuple[Path, str, int, int, str]] = []
        if strict_mode:
            hash_step = 1 if total_files <= 200 else max(1, total_files // 200)
            for idx, (p, rel_posix, size, mtime_ns) in enumerate(meta_rows, start=1):
                if should_cancel and should_cancel():
                    raise BundleCancelled("cancelled")
                digest = _raw_content_digest(p)
                files.append((p, rel_posix, size, mtime_ns, digest))
                if progress_cb and total_files and (idx == 1 or idx == total_files or idx % hash_step == 0):
                    progress_cb(0.03 + 0.12 * (idx / total_files), f"Hashing {idx}/{total_files}: {rel_posix}")
        else:
            files = [(p, rel, size, mtime_ns, "") for p, rel, size, mtime_ns in meta_rows]
            if progress_cb:
                progress_cb(0.15, "Per-file hashing skipped in relaxed mode; archive remains content-addressed")

        size_by_suffix: dict[str, int] = {}
        largest_files: list[dict[str, int | str]] = []
        validation_files = 0
        validation_bytes = 0
        for (_p, rel_posix, size, _mtime_ns, _digest) in files:
            suffix = Path(rel_posix).suffix.lower() or "<none>"
            size_by_suffix[suffix] = int(size_by_suffix.get(suffix, 0)) + int(size)
            largest_files.append({"rel": rel_posix, "size": int(size)})
            if rel_posix.startswith("resources/validation/"):
                validation_files += 1
                validation_bytes += int(size)
        largest_files = sorted(largest_files, key=lambda x: int(x.get("size", 0)), reverse=True)[:25]

        manifest = {
            "version": 7,
            "cache_policy": "v60q_locked_metadata_first_deterministic_bundle",
            "suite_dir": str(suite_dir),
            "files": [
                {"rel": rel_posix, "size": size, "mtime_ns": mtime_ns, "sha256": digest}
                for (_p, rel_posix, size, mtime_ns, digest) in files
            ],
            "hailo_generation_links": generation_links,
            "total_bytes": int(total_bytes),
            "file_count": int(total_files),
            "include_patterns": list(includes or []),
            "exclude_patterns": list(excludes or []),
            "integrity_mode": integrity_mode,
            "per_file_hashes": bool(strict_mode),
            "gzip_level": int(gzip_level),
            "deterministic_archive": True,
            "archive_metadata_policy": "uid_gid_mtime_zero; modes_preserved; gzip_mtime_zero",
            "slimming_policy": "v60q: run-mode validation subsets only; one locked deterministic build per suite; metadata-first reuse and fast gzip outside strict final campaigns",
            "size_by_suffix": {k: int(v) for k, v in sorted(size_by_suffix.items())},
            "largest_files": largest_files,
            "validation_footprint": {"file_count": int(validation_files), "total_bytes": int(validation_bytes)},
        }

        if (
            strict_mode
            and reuse_if_unchanged
            and out_path.exists()
            and isinstance(old, dict)
            and old.get("files") == manifest["files"]
            and patterns_match
        ):
            digest = _archive_digest(old)
            manifest["bundle_sha256"] = digest
            manifest["bundle_bytes"] = int(out_path.stat().st_size)
            _write_json_atomic(manifest_path, manifest)
            if progress_cb:
                progress_cb(1.0, "Using cached bundle (strict content unchanged)")
            return BundleStats(total_files, total_bytes, out_path, True, digest, manifest_path)

        tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            # v60q: make the transport archive deterministic.  A normal
            # ``tarfile.open(..., 'w:gz')`` writes the current timestamp into
            # the gzip header and preserves volatile uid/gid/mtime metadata.
            # Rebuilding an otherwise identical suite would therefore produce
            # a different SHA-256 and miss the persistent remote cache.
            # Normalising transport-only metadata keeps executable permission
            # bits while making identical suite content map to one archive hash.
            with tmp_path.open("wb") as raw_fh:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=gzip_level,
                    fileobj=raw_fh,
                    mtime=0,
                ) as gzip_fh:
                    with tarfile.open(fileobj=gzip_fh, mode="w", format=tarfile.PAX_FORMAT) as tf:
                        step = 1 if total_files <= 200 else max(1, total_files // 200)
                        for idx, (p, rel_posix, size, _mtime_ns, _digest) in enumerate(files, start=1):
                            if should_cancel and should_cancel():
                                raise BundleCancelled("cancelled")
                            if progress_cb and total_files and (idx == 1 or idx == total_files or idx % step == 0):
                                progress_cb(0.15 + 0.80 * ((idx - 1) / total_files), f"Adding {idx}/{total_files}: {rel_posix} ({size / (1024*1024):.1f} MB)")
                            info = tf.gettarinfo(str(p), arcname=rel_posix)
                            info.uid = 0
                            info.gid = 0
                            info.uname = ""
                            info.gname = ""
                            info.mtime = 0
                            # Avoid PAX fields that can contain host-specific
                            # timestamps while preserving file modes.
                            info.pax_headers = {}
                            with p.open("rb") as source_fh:
                                tf.addfile(info, source_fh)
                            if progress_cb and total_files and (idx == 1 or idx == total_files or idx % step == 0):
                                progress_cb(0.15 + 0.80 * (idx / total_files), f"Bundled {idx}/{total_files}: {rel_posix}")
                        # Targets precede links, including for streaming tar
                        # extraction. Hardlinks consume no second HEF payload.
                        for row in generation_links:
                            info = tarfile.TarInfo(row["path"])
                            info.type = tarfile.LNKTYPE
                            info.linkname = row["target"]
                            info.mode = 0o644
                            tf.addfile(info)
            os.replace(tmp_path, out_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        digest = _raw_content_digest(out_path)
        manifest["bundle_sha256"] = digest
        manifest["bundle_bytes"] = int(out_path.stat().st_size)
        _write_json_atomic(manifest_path, manifest)
        if progress_cb:
            progress_cb(1.0, "Bundle ready")
        return BundleStats(total_files, total_bytes, out_path, False, digest, manifest_path)

# v60m: unchanged files use a stat-keyed SHA-256 cache in development;
# final campaigns still re-hash strictly.
from onnx_splitpoint_tool.v60m_policy import install_hash_wrappers as _v60m_install_hash_wrappers
_v60m_install_hash_wrappers(globals())
