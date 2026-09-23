from __future__ import annotations

"""Fail-fast single-writer ownership for EvaluationRun directories."""

from pathlib import Path
from typing import Any, Callable, Mapping
import hashlib
import json
import os
import socket
import threading
import time


_THREAD_GUARD = threading.Lock()
_THREAD_OWNERS: dict[str, dict[str, Any]] = {}


def platform_workflow_interlock_path() -> Path:
    """Return the global workflow/platform-power interlock path.

    Evaluation runs hold a shared POSIX lock for their complete lifetime.  A
    platform-power operation must acquire the same inode exclusively before it
    can change a rail.  Keeping this gate independent of an EvaluationRun's
    output directory closes the cross-run and cross-process gap left by the
    per-run ownership locks below.
    """

    return Path(
        os.path.expanduser(
            "~/.onnx_splitpoint_tool/locks/workflow_platform_interlock.lock"
        )
    )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class WorkflowRunControlError(ValueError):
    status = "blocked"
    error_code = "workflow_run_blocked"

    def __init__(self, message: str, *, owner: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.owner = dict(owner or {})


class WorkflowRunLockedError(WorkflowRunControlError):
    error_code = "run_already_active"


class WorkflowRunCancelledError(WorkflowRunControlError):
    status = "cancelled"
    error_code = "workflow_cancelled_before_start"


class WorkflowRunTargetError(WorkflowRunControlError):
    def __init__(self, message: str, *, error_code: str):
        super().__init__(message)
        self.error_code = str(error_code)


class WorkflowResumeContractError(WorkflowRunControlError):
    error_code = "resume_contract_mismatch"


class WorkflowRunCleanupQuarantineError(WorkflowRunControlError):
    """The previous writer could not prove that all owned work stopped.

    This is deliberately distinct from a normal cooperative cancellation.  A
    caller may safely retry a clean cancellation, while a quarantined run must
    remain closed until its exact ownership evidence has been inspected.
    """

    status = "failed"
    error_code = "run_cleanup_quarantined"



def recoverable_cleanup_evidence(run_dir: Path) -> dict[str, Any]:
    """Read only the existing, run-bound local-cleanup proof.

    A vanished controller PID or released flock never proves that detached
    descendants stopped. Only the existing remote-recovery quarantine records
    local quiescence before a failed remote recovery. Other quarantines remain
    closed, including collector/source fences.
    """

    path = Path(run_dir) / "jobs" / "p01_unresolved_cleanup_quarantine.json"
    try:
        if path.is_symlink() or not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "onnx-splitpoint/p01-cleanup-quarantine"
            or value.get("run_id") != Path(run_dir).name
            or not str(value.get("session_id") or "").strip()
            or value.get("phase") != "resume_prior_remote_recovery"
            or value.get("local_process_quiescence_proven") is not True
            or (value.get("details") or {}).get("reason")
            != "prior_remote_lease_cleanup_unresolved"
        ):
            return {}
        return value
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def stable_resume_options(options: Any) -> dict[str, Any]:
    if isinstance(options, Mapping):
        payload = dict(options)
    elif hasattr(options, "to_dict"):
        payload = dict(options.to_dict())
    else:
        payload = dict(vars(options))
    # These fields choose/control an invocation but do not alter the frozen
    # scientific execution plan.  In particular force_stage and
    # rerun_generated_only must remain available for a valid resume.
    for key in (
        "profile",
        "out",
        "profile_start_snapshot",
        "resume",
        "run_id",
        "force_stage",
        "stop_after",
        "rerun_generated_only",
        "resume_missing_full_quality_only",
        "remote_reuse_bundle",
        "remote_no_reuse_bundle",
        "remote_resume",
        "remote_no_resume",
        "required_run_mode",
        "require_fresh_run",
        "force_build_confirmed_backends",
        "force_build_confirmation_source",
    ):
        payload.pop(key, None)
    return payload


def build_resume_contract(
    *,
    profile_payload: Mapping[str, Any],
    effective_execution_plan: Mapping[str, Any],
    options: Any,
) -> dict[str, Any]:
    profile_sha256 = _canonical_sha256(dict(profile_payload or {}))
    effective_plan_sha256 = _canonical_sha256(
        dict(effective_execution_plan or {})
    )
    stable_options = stable_resume_options(options)
    stable_options_sha256 = _canonical_sha256(stable_options)
    plan_sha256 = _canonical_sha256(
        {
            "effective_execution_plan_sha256": effective_plan_sha256,
            "stable_options_sha256": stable_options_sha256,
        }
    )
    contract_sha256 = _canonical_sha256(
        {
            "profile_sha256": profile_sha256,
            "plan_sha256": plan_sha256,
        }
    )
    return {
        "schema": "onnx-splitpoint/evaluation-resume-contract",
        "schema_version": 1,
        "profile_sha256": profile_sha256,
        "effective_execution_plan_sha256": effective_plan_sha256,
        "stable_options_sha256": stable_options_sha256,
        "plan_sha256": plan_sha256,
        "resume_contract_sha256": contract_sha256,
    }


class EvaluationRunLock:
    """Kernel-backed, nonblocking lease keyed by the resolved run directory."""

    @classmethod
    def for_resource(cls, resource: str, *, owner: Mapping[str, Any]):
        """Use the existing owner/quarantine protocol for a physical resource."""
        root = platform_workflow_interlock_path().parent
        key = hashlib.sha256(str(resource).encode("utf-8")).hexdigest()
        return cls(out_root=root, run_dir=root / ("resource-" + key),
                   owner={**dict(owner), "physical_resource": str(resource)})

    def __init__(
        self,
        *,
        out_root: Path,
        run_dir: Path,
        owner: Mapping[str, Any],
    ) -> None:
        self.out_root = Path(out_root).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        key_hash = hashlib.sha256(str(self.run_dir).encode("utf-8")).hexdigest()
        self.lock_path = (
            self.out_root
            / ".onnx_splitpoint_run_locks"
            / f"{self.run_dir.name}.{key_hash[:16]}.lock"
        )
        self.owner = dict(owner or {})
        self.owner.setdefault("schema", "onnx-splitpoint/evaluation-run-lock")
        self.owner.setdefault("schema_version", 1)
        self.owner.setdefault("run_id", self.run_dir.name)
        self.owner.setdefault("run_dir", str(self.run_dir))
        self.owner.setdefault("pid", os.getpid())
        self.owner.setdefault("ppid", os.getppid())
        self.owner.setdefault("hostname", socket.gethostname())
        self.owner.setdefault("acquired_at", _now_iso())
        self.owner["state"] = "running"
        try:
            self.owner.setdefault("pgid", os.getpgrp())
        except Exception:
            pass
        self._fh: Any = None
        self.previous_owner: dict[str, Any] = {}
        self._platform_interlock_fh: Any = None
        self._thread_key = str(self.lock_path)
        self._thread_acquired = False
        self._quarantined = False
        self.quarantine_path = self.lock_path.with_name(
            self.lock_path.name + ".quarantine.json"
        )

    def _read_owner(self) -> dict[str, Any]:
        try:
            assert self._fh is not None
            self._fh.seek(0)
            raw = self._fh.read().decode("utf-8", errors="replace").strip("\x00\n ")
            parsed = json.loads(raw) if raw else {}
            return dict(parsed) if isinstance(parsed, Mapping) else {}
        except Exception:
            return {}

    @staticmethod
    def _owner_summary(owner: Mapping[str, Any]) -> str:
        parts = []
        for key in ("session_id", "pid", "hostname", "acquired_at"):
            if owner.get(key) not in (None, ""):
                parts.append(f"{key}={owner.get(key)}")
        return ", ".join(parts) or "owner metadata unavailable"

    def acquire(
        self,
        *,
        recover_quarantine: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> "EvaluationRunLock":
        return self._acquire(recover_quarantine=recover_quarantine)

    def acquire_for_recovery(self) -> "EvaluationRunLock":
        """Hold an existing inode without admitting work or changing its fence.

        The explicit physical-resource recovery operator uses this to inspect
        the complete resource group before changing any member. Ordinary
        acquire(), including workflow Resume, keeps its existing behaviour.
        """
        return self._acquire(inspect_only=True)

    def _acquire(
        self,
        *,
        recover_quarantine: Callable[[Mapping[str, Any]], bool] | None = None,
        inspect_only: bool = False,
    ) -> "EvaluationRunLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _THREAD_GUARD:
            local_owner = _THREAD_OWNERS.get(self._thread_key)
            if local_owner is not None:
                raise WorkflowRunLockedError(
                    "Evaluation run is already active in this process: "
                    + self._owner_summary(local_owner),
                    owner=local_owner,
                )
            # Reserve the key before opening the cross-process lock so two GUI
            # worker threads cannot race through flock in one interpreter.
            _THREAD_OWNERS[self._thread_key] = dict(self.owner)
            self._thread_acquired = True

        try:
            if os.name == "posix":
                import fcntl

                interlock_path = platform_workflow_interlock_path()
                interlock_path.parent.mkdir(parents=True, exist_ok=True)
                interlock_fh = interlock_path.open("a+b")
                try:
                    fcntl.flock(
                        interlock_fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB
                    )
                except (BlockingIOError, OSError) as exc:
                    interlock_fh.close()
                    raise WorkflowRunLockedError(
                        "A platform-power operation is active; starting an "
                        "EvaluationRun is blocked"
                    ) from exc
                self._platform_interlock_fh = interlock_fh

            if inspect_only:
                import stat
                descriptor = os.open(
                    self.lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
                )
                self._fh = os.fdopen(descriptor, "r+b")
                if not stat.S_ISREG(os.fstat(self._fh.fileno()).st_mode):
                    raise WorkflowRunCleanupQuarantineError(
                        "Recovery requires an existing regular resource lock"
                    )
            else:
                self._fh = self.lock_path.open("a+b")
            try:
                if os.name == "nt":  # pragma: no cover - Windows installation
                    import msvcrt

                    self._fh.seek(0, os.SEEK_END)
                    if self._fh.tell() == 0:
                        self._fh.write(b"\x00")
                        self._fh.flush()
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(
                        self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
            except (BlockingIOError, OSError) as exc:
                owner = self._read_owner()
                raise WorkflowRunLockedError(
                    f"Evaluation run {self.run_dir.name!r} is already active; "
                    + self._owner_summary(owner),
                    owner=owner,
                ) from exc

            # A durable P0.1 cleanup fence survives kernel-lock release.  It is
            # checked only after exclusive flock acquisition, before this
            # writer can replace owner metadata or touch the EvaluationRun.
            self.previous_owner = self._read_owner()
            if inspect_only:
                # Even an accidental default release must not rewrite the
                # predecessor while the group is still being inspected.
                self._quarantined = True
                return self
            quarantined_owner: dict[str, Any] | None = None
            if os.path.lexists(self.quarantine_path):
                try:
                    raw = self.quarantine_path.read_text(encoding="utf-8")
                    value = json.loads(raw)
                    quarantined_owner = (
                        dict(value) if isinstance(value, Mapping) else {}
                    )
                except Exception:
                    quarantined_owner = {
                        "quarantine_path": str(self.quarantine_path),
                        "state": "quarantined_unreadable",
                    }
            if quarantined_owner is None:
                previous_owner = self._read_owner()
                if str(previous_owner.get("state") or "") == "quarantined":
                    quarantined_owner = previous_owner
                elif str(previous_owner.get("state") or "") == "running":
                    # flock disappearing after SIGKILL/crash proves only that
                    # the writer exited; it cannot prove that separately
                    # sessioned local descendants stopped.  Convert the stale
                    # running owner into a durable fence before any new writer
                    # can replace its metadata or mutate the EvaluationRun.
                    try:
                        self.commit_quarantine_fence({
                            "reason": "previous_writer_terminated_uncleanly",
                            "previous_owner": previous_owner,
                        })
                    except BaseException as exc:
                        raise WorkflowRunCleanupQuarantineError(
                            "Evaluation run has an unclean previous writer and "
                            "a durable quarantine fence could not be committed; "
                            "do not reuse this run id",
                            owner=previous_owner,
                        ) from exc
                    quarantined_owner = previous_owner
            if (
                quarantined_owner is not None
                and recover_quarantine is not None
                and recover_quarantine(quarantined_owner) is True
            ):
                # The callback runs with exclusive flock and must prove the
                # archived local and exact remote ownership before clearing the
                # existing workflow quarantine. Never touch source/collector locks.
                self.quarantine_path.unlink(missing_ok=True)
                self._quarantined = False
                quarantined_owner = None
            if quarantined_owner is not None:
                raise WorkflowRunCleanupQuarantineError(
                    "Evaluation run is fenced by an unresolved P0.1 cleanup "
                    f"quarantine: {self.quarantine_path}",
                    owner=quarantined_owner,
                )

            encoded = (
                json.dumps(self.owner, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            self._fh.seek(0)
            self._fh.truncate(0)
            self._fh.write(encoded)
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            return self
        except BaseException:
            self.release(write_released_state=False)
            raise

    def write_recovery_owner(self, owner: Mapping[str, Any]) -> None:
        """Strict durable owner write while an explicit recovery holds flock.

        Unlike diagnostic release(), failures propagate before any group lock
        is dropped. A surviving running owner fences an interrupted recovery.
        """
        if self._fh is None or not self._thread_acquired:
            raise RuntimeError("resource recovery requires exclusive ownership")
        encoded = (json.dumps(dict(owner), indent=2, ensure_ascii=False,
                              sort_keys=True) + "\n").encode("utf-8")
        self._fh.seek(0)
        self._fh.truncate(0)
        self._fh.write(encoded)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def commit_quarantine_fence(
        self, payload: Mapping[str, Any]
    ) -> Path:
        """Durably fence future writers before releasing an unclean lock.

        The sidecar is the primary cross-session gate.  If creating it fails
        (for example a full directory), the already-open lock inode is
        overwritten with a smaller quarantined owner record and fsynced; the
        next acquirer checks both locations before writing its own metadata.
        """

        if self._fh is None:
            raise RuntimeError("cannot quarantine an unowned EvaluationRun lock")
        # Even when both persistence attempts encounter an I/O failure, release
        # must not overwrite a surviving running/quarantine record with the
        # misleading diagnostic state ``released``.
        self._quarantined = True
        record = {
            "schema": "onnx-splitpoint/evaluation-run-lock-quarantine",
            "schema_version": 1,
            "state": "quarantined",
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "quarantined_at": _now_iso(),
            "owner": dict(payload or {}),
        }
        encoded = (
            json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        temporary = self.quarantine_path.with_name(
            f".{self.quarantine_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        sidecar_error: BaseException | None = None
        try:
            descriptor = os.open(
                str(temporary),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.replace(temporary, self.quarantine_path)
            if os.name == "posix":
                directory_fd = os.open(str(self.quarantine_path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return self.quarantine_path
        except BaseException as exc:
            sidecar_error = exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass

        # The lock inode was already created and written successfully by
        # acquire().  Reusing its allocated blocks is the durable fallback for
        # a directory-create or disk-full failure.
        try:
            self._fh.seek(0)
            self._fh.truncate(0)
            self._fh.write(encoded)
            self._fh.flush()
            os.fsync(self._fh.fileno())
            return self.lock_path
        except BaseException as fallback_exc:
            raise RuntimeError(
                "could not durably commit cleanup quarantine in sidecar or "
                f"lock inode: sidecar={type(sidecar_error).__name__}: "
                f"{sidecar_error}; lock={type(fallback_exc).__name__}: "
                f"{fallback_exc}"
            ) from fallback_exc

    def release(self, *, write_released_state: bool = True) -> None:
        fh = self._fh
        self._fh = None
        try:
            if fh is not None:
                # Release metadata is diagnostic only.  A write failure must
                # never prevent kernel unlock or leak the in-process guard.
                if write_released_state and not self._quarantined:
                    try:
                        released = dict(self.owner)
                        released.update(
                            {"state": "released", "released_at": _now_iso()}
                        )
                        fh.seek(0)
                        fh.truncate(0)
                        fh.write(
                            (
                                json.dumps(
                                    released,
                                    indent=2,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            ).encode("utf-8")
                        )
                        fh.flush()
                    except Exception:
                        pass
                try:
                    if os.name == "nt":  # pragma: no cover
                        import msvcrt

                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    # Closing the descriptor below releases a POSIX flock and
                    # is the final Windows handle cleanup as well.
                    pass
                finally:
                    try:
                        fh.close()
                    except Exception:
                        pass
        finally:
            if self._thread_acquired:
                with _THREAD_GUARD:
                    _THREAD_OWNERS.pop(self._thread_key, None)
                self._thread_acquired = False
            interlock_fh = self._platform_interlock_fh
            self._platform_interlock_fh = None
            if interlock_fh is not None:
                try:
                    if os.name == "posix":
                        import fcntl

                        fcntl.flock(interlock_fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                finally:
                    try:
                        interlock_fh.close()
                    except Exception:
                        pass

    def __enter__(self) -> "EvaluationRunLock":
        return self.acquire()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.release()
