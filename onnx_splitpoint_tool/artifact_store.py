from __future__ import annotations

"""Unified content-addressed compiler artifact library.

Large backend artifacts remain ordinary files. A small SQLite index binds each
artifact to an exact canonical build contract, provenance, verification status,
and optional campaign pin. The store complements the historical backend-local
caches and lets Hailo/DeepX artifacts be reused across EvaluationRuns without
referencing an old run directory.
"""

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .process_control import current_process_registry

SCHEMA_VERSION = 1


def default_artifact_store_root() -> Path:
    raw = os.environ.get("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT") or "~/.onnx_splitpoint_tool/artifact_store"
    return Path(raw).expanduser().resolve()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _fsync_directory(path: Path) -> None:
    """Make a completed directory rename durable on supporting filesystems."""
    if os.name != "posix":  # pragma: no cover - Windows has no directory fsync
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _owned_reflink_copy(src: Path, dst: Path) -> int:
    """Preserve reflink performance while owning a potentially long copy."""

    command = [
        "cp", "--reflink=auto", "--preserve=mode,timestamps", str(src), str(dst),
    ]
    registry = current_process_registry()
    if registry is None:
        return int(subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode)
    if registry.cancelled:
        return 130
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(command, **popen_kwargs)
    try:
        registry.register(proc, label="artifact-store-reflink-copy")
        while True:
            try:
                return int(proc.wait(timeout=0.1))
            except subprocess.TimeoutExpired:
                if registry.cancelled:
                    registry.terminate_registered(proc, grace_s=0.5)
                    return 130
    finally:
        registry.unregister(proc)


def materialize_file(src: Path, dst: Path) -> str:
    """Materialize *src* at *dst*, preferring reflink/hardlink over copy."""
    src = src.expanduser().resolve()
    # Replace a destination alias itself; never overwrite its source snapshot.
    dst = dst.expanduser().absolute()
    if src == dst:
        return "same_path"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    if os.name == "posix" and shutil.which("cp"):
        copy_rc = _owned_reflink_copy(src, tmp)
        if copy_rc == 0 and tmp.is_file():
            os.replace(tmp, dst)
            return "reflink_or_copy"
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        registry = current_process_registry()
        if copy_rc == 130 and registry is not None and registry.cancelled:
            raise RuntimeError("artifact materialization cancelled")
    try:
        os.link(src, tmp)
        os.replace(tmp, dst)
        return "hardlink"
    except OSError:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return "copy"
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


class FileLock:
    """Cross-process advisory lock used for per-contract build leases."""

    def __init__(self, path: Path, timeout_s: float = 7200.0, poll_s: float = 0.25) -> None:
        self.path = path
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._fh = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+b")
        deadline = time.monotonic() + max(0.0, self.timeout_s)
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh.seek(0)
                self._fh.truncate(0)
                self._fh.write(f"pid={os.getpid()} acquired={time.time()}\n".encode("utf-8"))
                self._fh.flush()
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._fh.close()
                    self._fh = None
                    raise TimeoutError(f"timed out waiting for artifact build lock: {self.path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: int
    kind: str
    contract_hash: str
    artifact_hash: str
    filename: str
    size_bytes: int
    object_path: str
    manifest_path: str
    source_run: str
    verification_status: str
    pinned: bool
    pin_label: str
    created_at: float
    last_accessed: float
    contract: dict[str, Any]
    metadata: dict[str, Any]


class ArtifactStore:
    def __init__(self, root: str | Path | None = None, *, read_only: bool = False) -> None:
        self.root = Path(root or default_artifact_store_root()).expanduser().resolve()
        self.objects_dir = self.root / "objects" / "sha256"
        self.manifests_dir = self.root / "manifests"
        self.locks_dir = self.root / "locks"
        self.quarantine_dir = self.root / "quarantine"
        self.db_path = self.root / "registry.sqlite3"
        self.read_only = bool(read_only)
        if self.read_only:
            return
        for path in (self.objects_dir, self.manifests_dir, self.locks_dir, self.quarantine_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            con = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", timeout=60.0, uri=True)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only=ON")
            return con
        con = sqlite3.connect(str(self.db_path), timeout=60.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    contract_hash TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    object_relpath TEXT NOT NULL,
                    manifest_relpath TEXT NOT NULL,
                    contract_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_run TEXT NOT NULL DEFAULT '',
                    verification_status TEXT NOT NULL DEFAULT 'registered',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    pin_label TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    verified_at REAL,
                    UNIQUE(kind, contract_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(artifact_hash);
                CREATE INDEX IF NOT EXISTS idx_artifacts_pinned ON artifacts(pinned);
                CREATE TABLE IF NOT EXISTS artifact_refs (
                    artifact_id INTEGER NOT NULL,
                    ref_type TEXT NOT NULL,
                    ref_value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(artifact_id, ref_type, ref_value),
                    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    artifact_id INTEGER,
                    event TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL
                );
                """
            )
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))

    @staticmethod
    def contract_hash(contract: Mapping[str, Any]) -> str:
        return canonical_hash(dict(contract))

    def build_lock(self, kind: str, contract: Mapping[str, Any] | str, *, timeout_s: float = 7200.0) -> FileLock:
        if self.read_only:
            raise RuntimeError("artifact store is read-only")
        digest = str(contract) if isinstance(contract, str) else self.contract_hash(contract)
        safe_kind = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(kind))
        return FileLock(self.locks_dir / f"{safe_kind}-{digest}.lock", timeout_s=timeout_s)

    def _row_to_record(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=int(row["id"]), kind=str(row["kind"]), contract_hash=str(row["contract_hash"]),
            artifact_hash=str(row["artifact_hash"]), filename=str(row["filename"]), size_bytes=int(row["size_bytes"]),
            object_path=str(self.root / str(row["object_relpath"])), manifest_path=str(self.root / str(row["manifest_relpath"])),
            source_run=str(row["source_run"] or ""), verification_status=str(row["verification_status"] or ""),
            pinned=bool(row["pinned"]), pin_label=str(row["pin_label"] or ""), created_at=float(row["created_at"]),
            last_accessed=float(row["last_accessed"]), contract=json.loads(str(row["contract_json"] or "{}")),
            metadata=json.loads(str(row["metadata_json"] or "{}")),
        )

    @staticmethod
    def candidate_sort_key(record: ArtifactRecord) -> tuple[str, str, str, int]:
        """Stable identity order, independent of access time and scan order."""
        return (record.contract_hash, record.artifact_hash, record.filename, record.artifact_id)

    def candidates_by_metadata(self, *, kind: str, key: str, value: Any) -> list[ArtifactRecord]:
        """Return every alias candidate without mutating access/verification state.

        Backend callers must validate their full semantic contract before choosing;
        the alias itself is not proof that two builds are interchangeable.
        """
        return sorted(
            (record for record in self.list(kind=kind, limit=1_000_000)
             if record.metadata.get(str(key)) == value),
            key=self.candidate_sort_key,
        )

    def bundle_paths(self, record: ArtifactRecord) -> dict[str, Path]:
        """Resolve the three immutable files of a sealed Hailo registration."""
        files = record.metadata.get("bundle_files")
        if record.metadata.get("bundle_status") != "sealed" or not isinstance(files, dict):
            raise ValueError("legacy_unsealed")
        parent = Path(record.object_path).parent.resolve()
        paths: dict[str, Path] = {}
        for role in ("hef", "receipt", "cache_meta"):
            name = files.get(role)
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("invalid_bundle_path")
            path = parent / name
            if path.is_symlink() or path.resolve().parent != parent:
                raise ValueError("invalid_bundle_path")
            paths[role] = path
        if paths["hef"] != Path(record.object_path).resolve():
            raise ValueError("bundle_hef_path_mismatch")
        return paths

    @staticmethod
    def _validate_hailo_files(
        hef: Path, receipt_path: Path, cache_meta_path: Path,
        *, artifact_hash: str, size_bytes: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Check the byte/identity binding shared by all three stored files.

        Hailo's backend additionally validates all compiler/preprocessing axes;
        this store validation ensures an intact backup of that sealed identity.
        """
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or not isinstance(meta, dict):
            raise ValueError("invalid_bundle_json")
        if receipt.get("schema") != "onnx-splitpoint/hailo-hef-build-receipt/v2":
            raise ValueError("invalid_receipt_schema")
        if meta.get("schema") != "onnx-splitpoint/hailo-hef-cache-meta-v2":
            raise ValueError("invalid_cache_meta_schema")
        if (receipt.get("hef_sha256") != artifact_hash
                or receipt.get("hef_size_bytes") != size_bytes
                or meta.get("hef_sha256") != artifact_hash
                or meta.get("hef_size") != size_bytes):
            raise ValueError("bundle_hef_identity_mismatch")
        if (not str(receipt.get("cache_key") or "")
                or not isinstance(receipt.get("cache_payload"), dict)
                or meta.get("cache_key") != receipt.get("cache_key")
                or meta.get("payload") != receipt.get("cache_payload")):
            raise ValueError("bundle_cache_identity_mismatch")
        for key in ("preprocessing_contract_sha256", "net_name", "hw_arch"):
            if not str(receipt.get(key) or "") or meta.get(key) != receipt.get(key):
                raise ValueError(f"bundle_{key}_mismatch")
        if not hef.is_file() or hef.stat().st_size != size_bytes:
            raise ValueError("missing_or_corrupt")
        return receipt, meta

    def validate_record(self, record: ArtifactRecord, *, verify: str = "strict") -> tuple[bool, str]:
        """Read-only validation, including both sidecars for sealed bundles."""
        try:
            obj = Path(record.object_path)
            if not obj.is_file() or obj.stat().st_size != record.size_bytes:
                return False, "missing_or_corrupt"
            if str(verify).lower() in {"strict", "sha256", "full"}:
                if sha256_file(obj) != record.artifact_hash:
                    return False, "artifact_hash_mismatch"
            if record.metadata.get("bundle_status") == "sealed":
                paths = self.bundle_paths(record)
                receipt, _meta = self._validate_hailo_files(
                    paths["hef"], paths["receipt"], paths["cache_meta"],
                    artifact_hash=record.artifact_hash, size_bytes=record.size_bytes,
                )
                if receipt != record.metadata.get("build_receipt"):
                    return False, "bundle_receipt_metadata_mismatch"
                if receipt.get("cache_key") != record.metadata.get("legacy_cache_key"):
                    return False, "bundle_cache_key_metadata_mismatch"
            return True, "verified"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return False, str(exc) or "invalid_bundle"

    def lookup(self, *, kind: str, contract: Mapping[str, Any], verify: str = "metadata") -> Optional[ArtifactRecord]:
        digest = self.contract_hash(contract)
        with self._connect() as con:
            row = con.execute("SELECT * FROM artifacts WHERE kind=? AND contract_hash=?", (str(kind), digest)).fetchone()
            if row is None:
                return None
            record = self._row_to_record(row)
            ok, reason = self.validate_record(record, verify=verify)
            strict = str(verify).lower() in {"strict", "sha256", "full"}
            if self.read_only:
                return record if ok else None
            if not ok:
                con.execute("UPDATE artifacts SET verification_status=? WHERE id=?", ("missing_or_corrupt", record.artifact_id))
                con.execute("INSERT INTO events(artifact_id,event,detail_json,created_at) VALUES(?,?,?,?)",
                            (record.artifact_id, "lookup_failed", canonical_json({"verify": verify, "reason": reason}), time.time()))
                return None
            now = time.time()
            status = "verified_strict" if strict else "verified_metadata"
            con.execute("UPDATE artifacts SET last_accessed=?,verification_status=?,verified_at=? WHERE id=?",
                        (now, status, now, record.artifact_id))
            fresh = con.execute("SELECT * FROM artifacts WHERE id=?", (record.artifact_id,)).fetchone()
            return self._row_to_record(fresh)

    def register(self, *, source_path: str | Path, kind: str, contract: Mapping[str, Any],
                 metadata: Mapping[str, Any] | None = None, source_run: str = "", pin: bool = False,
                 pin_label: str = "") -> ArtifactRecord:
        src = Path(source_path).expanduser().resolve()
        if not src.is_file() or src.stat().st_size <= 0:
            raise FileNotFoundError(f"artifact source missing or empty: {src}")
        contract_obj = json.loads(canonical_json(dict(contract)))
        contract_digest = self.contract_hash(contract_obj)
        artifact_digest = sha256_file(src)
        object_dir = self.objects_dir / artifact_digest[:2] / artifact_digest
        object_path = object_dir / src.name
        manifest_path = self.manifests_dir / str(kind) / f"{contract_digest}.json"
        now = time.time()
        with self.build_lock(str(kind), contract_digest, timeout_s=7200):
            if (not object_path.is_file() or object_path.stat().st_size != src.stat().st_size
                    or sha256_file(object_path) != artifact_digest):
                object_dir.mkdir(parents=True, exist_ok=True)
                materialize_file(src, object_path)
            if sha256_file(object_path) != artifact_digest:
                raise ValueError("artifact changed during registration")
            manifest = {
                "schema": "onnx-splitpoint/artifact-record", "schema_version": SCHEMA_VERSION,
                "kind": str(kind), "contract_hash": contract_digest, "artifact_hash": artifact_digest,
                "filename": src.name, "size_bytes": int(src.stat().st_size),
                "object_path": str(object_path.relative_to(self.root)), "source_path": str(src),
                "source_run": str(source_run or ""), "contract": contract_obj,
                "metadata": dict(metadata or {}), "created_at": now,
            }
            _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            return self._register_published_record(
                src=src, kind=str(kind), contract_obj=contract_obj,
                contract_digest=contract_digest, artifact_digest=artifact_digest,
                filename=src.name, size_bytes=int(src.stat().st_size),
                object_path=object_path, manifest_path=manifest_path,
                metadata=metadata, source_run=source_run, pin=pin,
                pin_label=pin_label, now=now,
            )

    def _register_published_record(
        self, *, src: Path, kind: str, contract_obj: dict[str, Any],
        contract_digest: str, artifact_digest: str, filename: str,
        size_bytes: int, object_path: Path, manifest_path: Path,
        metadata: Mapping[str, Any] | None, source_run: str,
        pin: bool, pin_label: str, now: float,
    ) -> ArtifactRecord:
        """Commit a ready object without deleting existing refs or campaign pins."""
        with self._connect() as con:
            con.execute("PRAGMA synchronous=FULL")
            con.execute(
                """
                INSERT INTO artifacts(kind,contract_hash,artifact_hash,filename,size_bytes,object_relpath,manifest_relpath,
                                      contract_json,metadata_json,source_run,verification_status,pinned,pin_label,created_at,last_accessed,verified_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(kind,contract_hash) DO UPDATE SET
                    artifact_hash=excluded.artifact_hash, filename=excluded.filename,
                    size_bytes=excluded.size_bytes, object_relpath=excluded.object_relpath,
                    manifest_relpath=excluded.manifest_relpath, contract_json=excluded.contract_json,
                    metadata_json=excluded.metadata_json, source_run=excluded.source_run,
                    verification_status='registered', pinned=MAX(artifacts.pinned,excluded.pinned),
                    pin_label=CASE WHEN excluded.pin_label!='' THEN excluded.pin_label ELSE artifacts.pin_label END,
                    last_accessed=excluded.last_accessed, verified_at=excluded.verified_at
                """,
                (str(kind), contract_digest, artifact_digest, filename, size_bytes,
                 str(object_path.relative_to(self.root)), str(manifest_path.relative_to(self.root)),
                 canonical_json(contract_obj), canonical_json(dict(metadata or {})), str(source_run or ""),
                 "registered", int(bool(pin)), str(pin_label or ""), now, now, now),
            )
            row = con.execute("SELECT * FROM artifacts WHERE kind=? AND contract_hash=?", (str(kind), contract_digest)).fetchone()
            con.execute("INSERT INTO events(artifact_id,event,detail_json,created_at) VALUES(?,?,?,?)",
                        (int(row["id"]), "registered", canonical_json({"source": str(src)}), now))
            return self._row_to_record(row)

    def register_hailo_bundle(
        self, *, source_path: str | Path, receipt_path: str | Path,
        cache_meta_path: str | Path, contract: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None, source_run: str = "",
        pin: bool = False, pin_label: str = "",
    ) -> ArtifactRecord:
        """Publish HEF, receipt and cache metadata as one immutable backup.

        The existing HEF SHA and contract identify this artifact. Generation
        directories are merely an atomic publication mechanism, not a new cache
        identity. A failed copy, validation, rename or database commit leaves the
        previous registration and all its bytes available.
        """
        src = Path(source_path).expanduser().resolve()
        if not src.is_file() or src.stat().st_size <= 0:
            raise FileNotFoundError(f"artifact source missing or empty: {src}")
        contract_obj = json.loads(canonical_json(dict(contract)))
        contract_digest = self.contract_hash(contract_obj)
        artifact_digest = sha256_file(src)
        size_bytes = src.stat().st_size
        receipt_source = Path(receipt_path).expanduser().resolve()
        meta_source = Path(cache_meta_path).expanduser().resolve()
        receipt, _cache_meta = self._validate_hailo_files(
            src, receipt_source, meta_source,
            artifact_hash=artifact_digest, size_bytes=size_bytes,
        )
        metadata_obj = dict(metadata or {})
        if metadata_obj.get("build_receipt", receipt) != receipt:
            raise ValueError("bundle_receipt_metadata_mismatch")
        if metadata_obj.get("legacy_cache_key", receipt["cache_key"]) != receipt["cache_key"]:
            raise ValueError("bundle_cache_key_metadata_mismatch")
        files = {"hef": src.name, "receipt": "hailo_hef_build_receipt.json", "cache_meta": "cache_meta.json"}
        if len(set(files.values())) != 3:
            raise ValueError("bundle_filename_collision")
        metadata_obj.update({
            "bundle_status": "sealed", "bundle_files": files,
            "build_receipt": receipt, "legacy_cache_key": receipt["cache_key"],
        })
        parent = self.objects_dir / artifact_digest[:2] / artifact_digest / "sealed" / contract_digest
        with self.build_lock("hailo_hef", contract_digest):
            existing = self.lookup(kind="hailo_hef", contract=contract_obj, verify="strict")
            if (existing is not None and existing.artifact_hash == artifact_digest
                    and existing.metadata.get("bundle_status") == "sealed"
                    and existing.metadata.get("build_receipt") == receipt):
                if pin:
                    self.pin(existing.artifact_id, label=pin_label or existing.pin_label)
                    return self.lookup(kind="hailo_hef", contract=contract_obj, verify="strict") or existing
                return existing
            parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(parent)))
            generation = parent / ("generation-" + staging.name.removeprefix(".staging-"))
            published = False
            try:
                for role, source in (("hef", src), ("receipt", receipt_source), ("cache_meta", meta_source)):
                    target = staging / files[role]
                    # Copies own the sidecars even on filesystems without reflinks.
                    if (os.name != "posix" or not shutil.which("cp")
                            or _owned_reflink_copy(source, target) != 0):
                        registry = current_process_registry()
                        if registry is not None and registry.cancelled:
                            raise RuntimeError("artifact materialization cancelled")
                        shutil.copy2(source, target)
                    _fsync_file(target)
                staged_receipt, _ = self._validate_hailo_files(
                    staging / files["hef"], staging / files["receipt"], staging / files["cache_meta"],
                    artifact_hash=artifact_digest, size_bytes=size_bytes,
                )
                if staged_receipt != receipt or sha256_file(staging / files["hef"]) != artifact_digest:
                    raise ValueError("artifact bundle changed during registration")
                object_path = generation / files["hef"]
                manifest_path = generation / "artifact_manifest.json"
                now = time.time()
                manifest = {
                    "schema": "onnx-splitpoint/artifact-record", "schema_version": SCHEMA_VERSION,
                    "kind": "hailo_hef", "contract_hash": contract_digest, "artifact_hash": artifact_digest,
                    "filename": src.name, "size_bytes": size_bytes,
                    "object_path": str(object_path.relative_to(self.root)), "source_path": str(src),
                    "source_run": source_run, "contract": contract_obj, "metadata": metadata_obj, "created_at": now,
                }
                _atomic_write_text(staging / manifest_path.name, json.dumps(manifest, indent=2) + "\n")
                _fsync_directory(staging)
                os.replace(staging, generation)
                _fsync_directory(parent)
                record = self._register_published_record(
                    src=src, kind="hailo_hef", contract_obj=contract_obj,
                    contract_digest=contract_digest, artifact_digest=artifact_digest,
                    filename=src.name, size_bytes=size_bytes,
                    object_path=object_path, manifest_path=manifest_path,
                    metadata=metadata_obj, source_run=source_run, pin=pin,
                    pin_label=pin_label, now=now,
                )
                published = True
                return record
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
                if not published and generation.exists():
                    shutil.rmtree(generation)

    def find_by_metadata(self, *, kind: str, key: str, value: Any, verify: str = "metadata") -> Optional[ArtifactRecord]:
        """Choose a valid alias candidate in stable identity order.

        This checks physical validity only. Backend reuse must additionally check
        its complete build contract (use candidates_by_metadata for that).
        """
        records = self.candidates_by_metadata(kind=kind, key=key, value=value)
        for record in records:
            ok, _reason = self.validate_record(record, verify=verify)
            strict = str(verify).lower() in {"strict", "sha256", "full"}
            if ok:
                if self.read_only:
                    return record
                with self._connect() as con:
                    con.execute("UPDATE artifacts SET last_accessed=?,verification_status=?,verified_at=? WHERE id=?",
                                (time.time(), "verified_strict" if strict else "verified_metadata", time.time(), record.artifact_id))
                return record
        return None

    def materialize(self, record: ArtifactRecord, destination: str | Path, *, reference: str = "") -> str:
        src = Path(record.object_path)
        dst = Path(destination).expanduser().absolute()
        if not src.is_file():
            raise FileNotFoundError(f"artifact object missing: {src}")
        valid, reason = self.validate_record(record, verify="strict")
        if not valid:
            raise ValueError(f"artifact validation failed: {reason}")
        if record.kind == "hailo_hef" and ".hailo-generations" in dst.parts:
            raise ValueError("cannot overwrite an immutable Hailo generation")
        if record.metadata.get("bundle_status") == "sealed":
            from .hailo_cache_bundle import publish_bundle
            paths = self.bundle_paths(record)
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            cache_meta = json.loads(paths["cache_meta"].read_text(encoding="utf-8"))
            def validate_stage(hef: Path) -> bool:
                staged_receipt, staged_meta = self._validate_hailo_files(
                    hef, hef.parent / "hailo_hef_build_receipt.json", hef.parent / "cache_meta.json",
                    artifact_hash=record.artifact_hash, size_bytes=record.size_bytes,
                )
                return (staged_receipt == receipt and staged_meta == cache_meta
                        and sha256_file(hef) == record.artifact_hash)
            publish_bundle(source_hef=src, destination=dst, receipt=receipt,
                           cache_meta=cache_meta, validator=validate_stage)
            method = "atomic_hailo_bundle"
        else:
            if record.kind == "hailo_hef":
                raise ValueError("legacy_unsealed: cannot materialize a HEF-only artifact")
            method = materialize_file(src, dst)
        with self._connect() as con:
            now = time.time()
            con.execute("UPDATE artifacts SET last_accessed=? WHERE id=?", (now, record.artifact_id))
            if reference:
                con.execute("INSERT OR IGNORE INTO artifact_refs(artifact_id,ref_type,ref_value,created_at) VALUES(?,?,?,?)",
                            (record.artifact_id, "materialized", str(reference), now))
        return method

    def pin(self, artifact_id: int, *, label: str = "final") -> int:
        with self._connect() as con:
            return int(con.execute("UPDATE artifacts SET pinned=1,pin_label=? WHERE id=?", (str(label), int(artifact_id))).rowcount)

    def unpin(self, artifact_id: int) -> int:
        with self._connect() as con:
            return int(con.execute("UPDATE artifacts SET pinned=0,pin_label='' WHERE id=?", (int(artifact_id),)).rowcount)

    def list(self, *, kind: str = "", pinned_only: bool = False, limit: int = 1000) -> list[ArtifactRecord]:
        query = "SELECT * FROM artifacts"
        clauses: list[str] = []
        params: list[Any] = []
        if kind:
            clauses.append("kind=?"); params.append(str(kind))
        if pinned_only:
            clauses.append("pinned=1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY last_accessed DESC,contract_hash,artifact_hash,filename,id LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as con:
            return [self._row_to_record(row) for row in con.execute(query, params).fetchall()]

    def stats(self) -> dict[str, Any]:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) AS n,COALESCE(SUM(size_bytes),0) AS bytes,COALESCE(SUM(pinned),0) AS pinned FROM artifacts").fetchone()
            by_kind = {str(r["kind"]): {"count": int(r["n"]), "bytes": int(r["bytes"])}
                       for r in con.execute("SELECT kind,COUNT(*) AS n,COALESCE(SUM(size_bytes),0) AS bytes FROM artifacts GROUP BY kind")}
            return {"root": str(self.root), "database": str(self.db_path), "artifact_count": int(row["n"]),
                    "logical_bytes": int(row["bytes"]), "pinned_count": int(row["pinned"]), "by_kind": by_kind}

    def verify(self, *, strict: bool = False, quarantine: bool = False) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for record in self.list(limit=1_000_000):
            path = Path(record.object_path)
            ok, reason = self.validate_record(record, verify="strict" if strict else "metadata")
            actual_hash = ""
            if ok and strict:
                actual_hash = sha256_file(path)
            status = "verified_strict" if ok and strict else ("verified_metadata" if ok else "missing_or_corrupt")
            with self._connect() as con:
                con.execute("UPDATE artifacts SET verification_status=?,verified_at=? WHERE id=?", (status, time.time(), record.artifact_id))
            moved = ""
            if not ok and quarantine and path.is_file():
                target = self.quarantine_dir / f"{record.artifact_hash}-{path.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target)); moved = str(target)
            results.append({"artifact_id": record.artifact_id, "kind": record.kind, "ok": ok,
                            "status": status, "reason": reason, "actual_hash": actual_hash, "quarantined_to": moved})
        return {"ok": all(bool(x["ok"]) for x in results), "strict": bool(strict), "count": len(results), "results": results}

    def prune(self, *, older_than_days: float = 30.0, dry_run: bool = True) -> dict[str, Any]:
        cutoff = time.time() - max(0.0, float(older_than_days)) * 86400.0
        with self._connect() as con:
            records = [self._row_to_record(row) for row in con.execute("SELECT * FROM artifacts WHERE pinned=0 AND last_accessed<?", (cutoff,)).fetchall()]
            removed: list[dict[str, Any]] = []
            for record in records:
                item = {"artifact_id": record.artifact_id, "kind": record.kind, "path": record.object_path, "bytes": record.size_bytes}
                if not dry_run:
                    other = con.execute("SELECT COUNT(*) AS n FROM artifacts WHERE object_relpath=(SELECT object_relpath FROM artifacts WHERE id=?) AND id!=?", (record.artifact_id, record.artifact_id)).fetchone()
                    if int(other["n"]) == 0:
                        if record.metadata.get("bundle_status") == "sealed":
                            # Resolve/validate the containment before deleting a generation.
                            paths = self.bundle_paths(record)
                            shutil.rmtree(paths["hef"].parent)
                        else:
                            with contextlib.suppress(FileNotFoundError): Path(record.object_path).unlink()
                    with contextlib.suppress(FileNotFoundError): Path(record.manifest_path).unlink()
                    con.execute("DELETE FROM artifacts WHERE id=?", (record.artifact_id,))
                removed.append(item)
            return {"dry_run": bool(dry_run), "candidate_count": len(removed),
                    "candidate_bytes": sum(int(x["bytes"]) for x in removed), "items": removed}

    def index_existing(self, roots: list[str | Path] | None = None) -> dict[str, Any]:
        """Import existing Hailo/DeepX cache objects into the registry.

        Legacy cache manifests are preserved as the contract when available.
        This operation never deletes or moves the source cache.
        """
        candidates = [Path(x).expanduser() for x in (roots or [
            "~/.cache/onnx_splitpoint/hailo_hef",
            "~/Models/BackendArtifacts/deepx",
        ])]
        indexed: list[dict[str, Any]] = []
        for root in candidates:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".hef", ".dxnn"}:
                    continue
                kind = "hailo_hef" if path.suffix.lower() == ".hef" else "deepx_dxnn"
                manifest = {}
                manifest_path = None
                for name in ("build_manifest.json", "manifest.json", "build_status.json", "cache_manifest.json"):
                    candidate = path.parent / name
                    if candidate.is_file():
                        try:
                            payload = json.loads(candidate.read_text(encoding="utf-8"))
                            if isinstance(payload, dict):
                                manifest = payload; manifest_path = candidate
                        except Exception:
                            pass
                        break
                contract = {
                    "schema": "onnx-splitpoint/indexed-legacy-cache/v1",
                    "kind": kind,
                    "legacy_root": root.name,
                    "relative_path": str(path.relative_to(root)),
                    "manifest": manifest,
                }
                cache_key = str(manifest.get("cache_key") or manifest.get("cacheKey") or path.parent.name)
                try:
                    metadata = {"legacy_cache_key": cache_key, "indexed_from": str(path),
                                "manifest_path": str(manifest_path or "")}
                    receipt_path = path.parent / "hailo_hef_build_receipt.json"
                    cache_meta_path = path.parent / "cache_meta.json"
                    if kind == "hailo_hef" and receipt_path.is_file() and cache_meta_path.is_file():
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        metadata["legacy_cache_key"] = str(receipt.get("cache_key") or cache_key)
                        record = self.register_hailo_bundle(
                            source_path=path, receipt_path=receipt_path, cache_meta_path=cache_meta_path,
                            contract=contract, metadata=metadata, source_run="legacy-cache-index",
                        )
                    else:
                        if kind == "hailo_hef":
                            metadata["bundle_status"] = "legacy_unsealed"
                        record = self.register(source_path=path, kind=kind, contract=contract,
                                               metadata=metadata, source_run="legacy-cache-index")
                    indexed.append({"artifact_id": record.artifact_id, "kind": kind, "source": str(path), "ok": True})
                except Exception as exc:
                    indexed.append({"kind": kind, "source": str(path), "ok": False, "error": str(exc)})
        return {"ok": all(bool(x.get("ok")) for x in indexed), "indexed_count": sum(bool(x.get("ok")) for x in indexed),
                "failed_count": sum(not bool(x.get("ok")) for x in indexed), "items": indexed}

    def export_pack(self, destination: str | Path, *, pinned_only: bool = True) -> Path:
        dst = Path(destination).expanduser().resolve(); dst.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(self.list(pinned_only=pinned_only, limit=1_000_000), key=self.candidate_sort_key)
        packed_records = []
        for record in records:
            valid, reason = self.validate_record(record, verify="strict")
            if not valid:
                raise ValueError(f"cannot export artifact {record.artifact_id}: {reason}")
            packed = asdict(record)
            if record.metadata.get("bundle_status") == "sealed":
                packed["bundle_archive_prefix"] = f"bundles/{record.contract_hash}"
            packed_records.append(packed)
        manifest = {"schema": "onnx-splitpoint/artifact-pack", "schema_version": 1,
                    "created_at": time.time(), "records": packed_records}
        fd, temporary_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                zf.writestr("artifact_pack.json", json.dumps(manifest, indent=2, ensure_ascii=False))
                written = set()
                for record, packed in zip(records, packed_records):
                    obj = Path(record.object_path); man = Path(record.manifest_path)
                    if "bundle_archive_prefix" in packed:
                        for path in self.bundle_paths(record).values():
                            zf.write(path, f"{packed['bundle_archive_prefix']}/{path.name}")
                    else:
                        entry = f"objects/{record.artifact_hash}/{record.filename}"
                        if entry not in written:
                            zf.write(obj, entry)
                            written.add(entry)
                    if man.is_file(): zf.write(man, f"manifests/{record.kind}/{record.contract_hash}.json")
            _fsync_file(Path(temporary_name))
            os.replace(temporary_name, dst)
            _fsync_directory(dst.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
        return dst

    def import_pack(self, source: str | Path) -> dict[str, Any]:
        src = Path(source).expanduser().resolve(); imported: list[dict[str, Any]] = []
        with zipfile.ZipFile(src, "r") as zf:
            manifest = json.loads(zf.read("artifact_pack.json"))
            for rec in manifest.get("records") or []:
                filename = str(rec["filename"])
                if not filename or Path(filename).name != filename:
                    raise ValueError("invalid artifact pack filename")
                sealed = (rec.get("metadata") or {}).get("bundle_status") == "sealed"
                prefix = rec.get("bundle_archive_prefix")
                if sealed and prefix != f"bundles/{rec['contract_hash']}":
                    raise ValueError("sealed artifact pack is missing its complete bundle")
                obj_name = f"{prefix}/{filename}" if sealed else f"objects/{rec['artifact_hash']}/{filename}"
                with tempfile.TemporaryDirectory(prefix="osp-artifact-import-") as td:
                    temp = Path(td) / filename; temp.write_bytes(zf.read(obj_name))
                    if sha256_file(temp) != rec["artifact_hash"]:
                        raise ValueError(f"artifact pack hash mismatch: {obj_name}")
                    kwargs = dict(source_path=temp, contract=rec.get("contract") or {},
                                  metadata=rec.get("metadata") or {}, source_run=rec.get("source_run") or "imported_pack",
                                  pin=bool(rec.get("pinned")), pin_label=rec.get("pin_label") or "")
                    if sealed:
                        receipt = Path(td) / "hailo_hef_build_receipt.json"
                        cache_meta = Path(td) / "cache_meta.json"
                        receipt.write_bytes(zf.read(f"{prefix}/{receipt.name}"))
                        cache_meta.write_bytes(zf.read(f"{prefix}/{cache_meta.name}"))
                        record = self.register_hailo_bundle(**kwargs, receipt_path=receipt, cache_meta_path=cache_meta)
                    else:
                        record = self.register(**kwargs, kind=rec["kind"])
                    imported.append({"artifact_id": record.artifact_id, "kind": record.kind, "contract_hash": record.contract_hash})
        return {"ok": True, "imported_count": len(imported), "items": imported}


def artifact_store_enabled() -> bool:
    return str(os.environ.get("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}


def default_store() -> ArtifactStore:
    return ArtifactStore(default_artifact_store_root())
