from __future__ import annotations

"""Durable checkpoints for long-running Evaluation Workflow stages.

The checkpoint files in this module are execution state, not scientific result
files.  They provide a small, explicit commit protocol around child processes:

* a stage is reusable only after a durable ``completed`` commit;
* every Native Energy row owns one atomically replaced state file;
* an interrupted ``running`` row remains distinguishable from rows that were
  never started;
* the journal manifest is a derived index.  Row files remain authoritative if
  a process stops between the row commit and the manifest refresh.
"""

import hashlib
import json
import os
import shutil
import uuid
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CHECKPOINT_STATES = {
    "not_started",
    "running",
    "completed",
    "failed",
    "cancelled",
}
TERMINAL_ROW_STATES = {"completed", "failed"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def native_coordinator_input_hash(
    run_root: str | Path,
    config_path: str | Path,
    *, quality_gate_policy_sha256: str = "",
) -> str:
    """Bind the outer Runner and Native coordinator to one exact invocation."""

    root = Path(run_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    if not config.is_file() or config.is_symlink():
        raise ValueError(f"Native coordinator config is invalid: {config}")
    manifest = _strict_json(root / "run_manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("EvaluationRun manifest is invalid")
    resume_contract = manifest.get("resume_contract")
    return canonical_json_sha256({
        "schema": "onnx-splitpoint/native-coordinator-invocation",
        "schema_version": 2,
        "run_id": root.name,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "quality_gate_policy_sha256": str(
            quality_gate_policy_sha256 or ""
        ).strip().lower(),
        "resume_contract_sha256": str(
            (resume_contract or {}).get("resume_contract_sha256")
            if isinstance(resume_contract, Mapping) else ""
        ),
    })


def _strict_json(path: Path) -> Any:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"duplicate JSON object key in {path}: {key!r}"
                )
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Replace *path* atomically and durably where the platform permits it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
    )


def _artifact_record(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"checkpoint artifact is not a regular file: {resolved}")
    try:
        logical = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"checkpoint artifact escapes EvaluationRun: {resolved}"
        ) from exc
    raw = resolved.read_bytes()
    return {
        "path": logical,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_stage_checkpoint(
    path: str | Path,
    *,
    stage: str,
    state: str,
    complete: bool,
    input_hash: str,
    run_root: str | Path,
    artifacts: Iterable[str | Path] = (),
    details: Mapping[str, Any] | None = None,
    error: str = "",
    started_at: str = "",
) -> Path:
    if not str(stage or "").strip():
        raise ValueError("stage checkpoint requires a stage name")
    if not str(input_hash or "").strip():
        raise ValueError("stage checkpoint requires an input hash")
    normalized_state = str(state or "").strip().lower()
    if normalized_state not in CHECKPOINT_STATES:
        raise ValueError(f"invalid checkpoint state: {state!r}")
    if bool(complete) != (normalized_state in TERMINAL_ROW_STATES):
        raise ValueError(
            "stage checkpoint complete=true is allowed only for completed or "
            "failed terminal states"
        )
    root = Path(run_root).expanduser().resolve()
    records = [
        _artifact_record(Path(value), root=root)
        for value in artifacts
    ]
    logical_paths = [str(record.get("path") or "") for record in records]
    if len(logical_paths) != len(set(logical_paths)):
        raise ValueError("stage checkpoint artifact paths are not unique")
    previous_started = str(started_at or "").strip() or _now_iso()
    payload = {
        "schema": "onnx-splitpoint/atomic-stage-checkpoint",
        "schema_version": 1,
        "stage": str(stage or ""),
        "state": normalized_state,
        "complete": bool(complete),
        "input_hash": str(input_hash or ""),
        "started_at": previous_started,
        "updated_at": _now_iso(),
        "finished_at": (
            _now_iso() if normalized_state in TERMINAL_ROW_STATES | {"cancelled"}
            else ""
        ),
        "artifacts": records,
        "details": dict(details or {}),
        "error": str(error or ""),
    }
    payload["checkpoint_sha256"] = canonical_json_sha256(payload)
    return atomic_write_json(path, payload)


def load_stage_checkpoint(
    path: str | Path,
    *,
    stage: str,
    input_hash: str,
    run_root: str | Path,
) -> tuple[dict[str, Any] | None, str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        return None, "checkpoint_missing"
    try:
        raw = _strict_json(checkpoint_path)
    except Exception as exc:
        return None, f"checkpoint_invalid:{type(exc).__name__}"
    if not isinstance(raw, dict):
        return None, "checkpoint_not_object"
    payload = dict(raw)
    recorded_sha = str(payload.pop("checkpoint_sha256", ""))
    if not recorded_sha or canonical_json_sha256(payload) != recorded_sha:
        return None, "checkpoint_digest_mismatch"
    if (
        payload.get("schema") != "onnx-splitpoint/atomic-stage-checkpoint"
        or payload.get("schema_version") != 1
    ):
        return None, "checkpoint_schema_invalid"
    if str(payload.get("stage") or "") != str(stage or ""):
        return None, "checkpoint_stage_mismatch"
    if str(payload.get("input_hash") or "") != str(input_hash or ""):
        return None, "checkpoint_input_hash_mismatch"
    state = str(payload.get("state") or "")
    if state not in CHECKPOINT_STATES:
        return None, "checkpoint_state_invalid"
    expected_complete = state in TERMINAL_ROW_STATES
    if payload.get("complete") is not expected_complete:
        return None, "checkpoint_complete_state_mismatch"
    root = Path(run_root).expanduser().resolve()
    records = payload.get("artifacts")
    if not isinstance(records, list):
        return None, "checkpoint_artifacts_invalid"
    logical_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            return None, "checkpoint_artifact_record_invalid"
        logical = str(record.get("path") or "")
        if not logical or logical in logical_paths:
            return None, "checkpoint_artifact_path_invalid"
        logical_paths.add(logical)
        candidate = (root / logical).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None, "checkpoint_artifact_escapes_run"
        if not candidate.is_file() or candidate.is_symlink():
            return None, f"checkpoint_artifact_missing:{logical}"
        raw_bytes = candidate.read_bytes()
        recorded_size = record.get("size_bytes")
        if (
            type(recorded_size) is not int
            or recorded_size < 0
        ):
            return None, f"checkpoint_artifact_size_invalid:{logical}"
        if len(raw_bytes) != recorded_size:
            return None, f"checkpoint_artifact_size_mismatch:{logical}"
        if hashlib.sha256(raw_bytes).hexdigest() != str(record.get("sha256") or ""):
            return None, f"checkpoint_artifact_digest_mismatch:{logical}"
    payload["checkpoint_sha256"] = recorded_sha
    return payload, "valid"


def load_terminal_stage_checkpoint(
    path: str | Path,
    *,
    stage: str,
    input_hash: str,
    run_root: str | Path,
) -> tuple[dict[str, Any] | None, str]:
    payload, reason = load_stage_checkpoint(
        path,
        stage=stage,
        input_hash=input_hash,
        run_root=run_root,
    )
    if payload is None:
        return None, reason
    if (
        str(payload.get("state") or "") not in TERMINAL_ROW_STATES
        or payload.get("complete") is not True
    ):
        return None, "checkpoint_not_terminal"
    return payload, "terminal"


def load_reusable_stage_checkpoint(
    path: str | Path,
    *,
    stage: str,
    input_hash: str,
    run_root: str | Path,
) -> tuple[dict[str, Any] | None, str]:
    payload, reason = load_terminal_stage_checkpoint(
        path,
        stage=stage,
        input_hash=input_hash,
        run_root=run_root,
    )
    if payload is None:
        return None, reason
    if payload.get("state") != "completed":
        return None, "checkpoint_not_completed"
    return payload, "reusable"


class AtomicRowJournal:
    """Per-row atomic journal with a derived, atomically replaced manifest."""

    SCHEMA = "onnx-splitpoint/atomic-energy-row-journal"
    ROW_SCHEMA = "onnx-splitpoint/atomic-energy-row-checkpoint"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.rows_dir = self.root / "rows"
        self.manifest_path = self.root / "journal.json"
        self._manifest: dict[str, Any] = {}
        self._rows: list[dict[str, Any]] = []
        self._run_state_override = ""

    @staticmethod
    def _row_filename(index: int, identity: str) -> str:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"{int(index):04d}_{digest}.json"

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        plan_hash: str,
        rows: Sequence[Mapping[str, Any]],
        identities: Sequence[str],
        row_contract_hashes: Sequence[str],
    ) -> "AtomicRowJournal":
        if not (len(rows) == len(identities) == len(row_contract_hashes)):
            raise ValueError("journal row metadata has inconsistent cardinality")
        if not rows:
            raise ValueError("energy row journal requires at least one row")
        if not str(plan_hash or "").strip():
            raise ValueError("energy row journal requires a plan hash")
        if len(set(map(str, identities))) != len(identities):
            raise ValueError("energy row journal identities are not unique")
        if any(not str(value or "").strip() for value in identities):
            raise ValueError("energy row journal identity is empty")
        if any(
            not str(value or "").strip() for value in row_contract_hashes
        ):
            raise ValueError("energy row journal row contract is empty")
        destination = Path(root).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                f"energy row journal already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = destination.with_name(
            f".{destination.name}.create-{os.getpid()}-{uuid.uuid4().hex}"
        )
        journal = cls(temporary_root)
        try:
            journal.rows_dir.mkdir(parents=True, exist_ok=False)
            created_at = _now_iso()
            journal._rows = []
            for index, (row, identity, contract_hash) in enumerate(
                zip(rows, identities, row_contract_hashes)
            ):
                row_path = journal.rows_dir / cls._row_filename(
                    index, identity,
                )
                payload = {
                    "schema": cls.ROW_SCHEMA,
                    "schema_version": 1,
                    "index": index,
                    "identity": str(identity),
                    "row_contract_sha256": str(contract_hash),
                    "state": "not_started",
                    "complete": False,
                    "attempt_count": 0,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "started_at": "",
                    "finished_at": "",
                    "row": dict(row),
                    "execution": {},
                    "result": None,
                    "reason": "",
                }
                payload["checkpoint_sha256"] = canonical_json_sha256(
                    payload,
                )
                atomic_write_json(row_path, payload)
                journal._rows.append(payload)
            journal._manifest = {
                "schema": cls.SCHEMA,
                "schema_version": 1,
                "plan_hash": str(plan_hash),
                "state": "not_started",
                "complete": False,
                "created_at": created_at,
                "updated_at": created_at,
                "row_count": len(journal._rows),
                "rows": [],
                "counts": {},
            }
            journal._run_state_override = "not_started"
            journal._write_manifest()
            os.replace(temporary_root, destination)
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY)
            except OSError:
                directory_fd = -1
            if directory_fd >= 0:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
                finally:
                    os.close(directory_fd)
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
        journal.root = destination
        journal.rows_dir = destination / "rows"
        journal.manifest_path = destination / "journal.json"
        return journal

    @classmethod
    def open_for_resume(
        cls,
        root: str | Path,
        *,
        plan_hash: str,
        identities: Sequence[str],
        row_contract_hashes: Sequence[str],
    ) -> "AtomicRowJournal":
        if not identities or len(identities) != len(row_contract_hashes):
            raise ValueError("energy row resume binding is empty or inconsistent")
        if len(set(map(str, identities))) != len(identities):
            raise ValueError("energy row resume identities are not unique")
        if any(not str(value or "").strip() for value in identities):
            raise ValueError("energy row resume identity is empty")
        if any(
            not str(value or "").strip() for value in row_contract_hashes
        ):
            raise ValueError("energy row resume row contract is empty")
        journal = cls(root)
        if not journal.manifest_path.is_file() or journal.manifest_path.is_symlink():
            raise FileNotFoundError(f"energy row journal is missing: {journal.manifest_path}")
        manifest = _strict_json(journal.manifest_path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != cls.SCHEMA
            or manifest.get("schema_version") != 1
            or str(manifest.get("state") or "") not in CHECKPOINT_STATES
        ):
            raise ValueError("energy row journal manifest is invalid")
        if str(manifest.get("plan_hash") or "") != str(plan_hash or ""):
            raise ValueError("energy row journal plan hash drift")
        entries = manifest.get("rows")
        if not isinstance(entries, list) or len(entries) != len(identities):
            raise ValueError("energy row journal cardinality drift")
        journal._manifest = dict(manifest)
        journal._run_state_override = str(manifest.get("state") or "")
        journal._rows = []
        for index, (identity, contract_hash) in enumerate(
            zip(identities, row_contract_hashes)
        ):
            expected_name = cls._row_filename(index, identity)
            row_path = journal.rows_dir / expected_name
            if not row_path.is_file() or row_path.is_symlink():
                raise ValueError(f"energy row checkpoint missing: {expected_name}")
            raw = _strict_json(row_path)
            if not isinstance(raw, dict):
                raise ValueError(f"energy row checkpoint invalid: {expected_name}")
            if (
                raw.get("schema") != cls.ROW_SCHEMA
                or raw.get("schema_version") != 1
            ):
                raise ValueError(
                    f"energy row checkpoint schema invalid: {expected_name}"
                )
            payload = dict(raw)
            recorded_sha = str(payload.pop("checkpoint_sha256", ""))
            if not recorded_sha or canonical_json_sha256(payload) != recorded_sha:
                raise ValueError(f"energy row checkpoint digest mismatch: {expected_name}")
            payload["checkpoint_sha256"] = recorded_sha
            if int(payload.get("index", -1)) != index:
                raise ValueError(f"energy row checkpoint index drift: {expected_name}")
            if str(payload.get("identity") or "") != str(identity):
                raise ValueError(f"energy row checkpoint identity drift: {expected_name}")
            if str(payload.get("row_contract_sha256") or "") != str(contract_hash):
                raise ValueError(f"energy row checkpoint contract drift: {expected_name}")
            if str(payload.get("state") or "") not in CHECKPOINT_STATES:
                raise ValueError(f"energy row checkpoint state invalid: {expected_name}")
            state = str(payload.get("state") or "")
            complete = payload.get("complete") is True
            if complete != (state in TERMINAL_ROW_STATES):
                raise ValueError(
                    f"energy row checkpoint complete/state mismatch: {expected_name}"
                )
            attempt_count = payload.get("attempt_count")
            if type(attempt_count) is not int or attempt_count < 0:
                raise ValueError(
                    f"energy row checkpoint attempt count invalid: {expected_name}"
                )
            if state == "not_started" and attempt_count != 0:
                raise ValueError(
                    f"not-started energy row has an attempt: {expected_name}"
                )
            if state != "not_started" and attempt_count < 1:
                raise ValueError(
                    f"started energy row has no attempt: {expected_name}"
                )
            result = payload.get("result")
            if state in TERMINAL_ROW_STATES and not isinstance(result, Mapping):
                raise ValueError(
                    f"terminal energy row has no result: {expected_name}"
                )
            if state not in TERMINAL_ROW_STATES and result is not None:
                raise ValueError(
                    f"nonterminal energy row has a result: {expected_name}"
                )
            if state == "completed" and not (
                result.get("ok") is True or result.get("dry_run") is True
            ):
                raise ValueError(
                    f"completed energy row result is not successful: {expected_name}"
                )
            if state == "failed" and (
                result.get("ok") is True or result.get("dry_run") is True
            ):
                raise ValueError(
                    f"failed energy row result is successful: {expected_name}"
                )
            journal._rows.append(payload)
        if sum(
            1 for payload in journal._rows
            if str(payload.get("state") or "") == "running"
        ) > 1:
            raise ValueError("energy row journal has more than one running row")
        # Row files are authoritative if the process stopped after committing a
        # row but before refreshing the derived manifest.
        journal._write_manifest()
        return journal

    @property
    def rows(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._rows)

    @property
    def complete(self) -> bool:
        return all(
            str(value.get("state") or "") in TERMINAL_ROW_STATES
            for value in self._rows
        )

    def pending_indexes(self) -> list[int]:
        return [
            index
            for index, value in enumerate(self._rows)
            if str(value.get("state") or "") not in TERMINAL_ROW_STATES
        ]

    def terminal_results(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for payload in self._rows:
            if str(payload.get("state") or "") not in TERMINAL_ROW_STATES:
                continue
            result = payload.get("result")
            if not isinstance(result, Mapping):
                raise ValueError(
                    f"terminal energy row has no result: {payload.get('identity')}"
                )
            values.append(dict(result))
        return values

    def transition(
        self,
        index: int,
        state: str,
        *,
        execution: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        normalized = str(state or "").strip().lower()
        if normalized not in CHECKPOINT_STATES - {"not_started"}:
            raise ValueError(f"invalid row transition state: {state!r}")
        if index < 0 or index >= len(self._rows):
            raise IndexError(index)
        previous = self._rows[index]
        previous_state = str(previous.get("state") or "")
        allowed = {
            "not_started": {"running"},
            "running": {"completed", "failed", "cancelled"},
            "cancelled": {"running", "failed"},
            "completed": set(),
            "failed": set(),
        }
        if normalized not in allowed.get(previous_state, set()):
            raise ValueError(
                f"invalid energy row transition {previous_state!r} -> {normalized!r}"
            )
        if normalized == "running" and any(
            row_index != index
            and str(value.get("state") or "") == "running"
            for row_index, value in enumerate(self._rows)
        ):
            raise ValueError("only one energy row may be running")
        if normalized not in TERMINAL_ROW_STATES and result is not None:
            raise ValueError(
                "nonterminal energy row cannot contain a result"
            )
        if normalized == "running":
            self._run_state_override = "running"
        if normalized == "completed" and not (
            isinstance(result, Mapping)
            and (result.get("ok") is True or result.get("dry_run") is True)
        ):
            raise ValueError("completed energy row requires a successful result")
        if normalized == "failed" and isinstance(result, Mapping) and (
            result.get("ok") is True or result.get("dry_run") is True
        ):
            raise ValueError("failed energy row cannot contain a successful result")
        now = _now_iso()
        payload = dict(previous)
        payload.pop("checkpoint_sha256", None)
        payload.update(
            {
                "state": normalized,
                "complete": normalized in TERMINAL_ROW_STATES,
                "updated_at": now,
                "finished_at": (
                    now if normalized in TERMINAL_ROW_STATES | {"cancelled"}
                    else ""
                ),
                "reason": str(reason or ""),
            }
        )
        if normalized == "running":
            payload["attempt_count"] = int(payload.get("attempt_count") or 0) + 1
            payload["started_at"] = now
            payload["execution"] = dict(execution or {})
            payload["result"] = None
        elif execution is not None:
            payload["execution"] = dict(execution)
        if result is not None:
            payload["result"] = dict(result)
        if normalized in TERMINAL_ROW_STATES and not isinstance(payload.get("result"), Mapping):
            raise ValueError("terminal energy row transition requires a result")
        payload["checkpoint_sha256"] = canonical_json_sha256(payload)
        row_path = self.rows_dir / self._row_filename(
            index, str(payload.get("identity") or "")
        )
        atomic_write_json(row_path, payload)
        self._rows[index] = payload
        self._write_manifest()
        return dict(payload)

    def update_running(
        self, index: int, *, execution: Mapping[str, Any]
    ) -> dict[str, Any]:
        if index < 0 or index >= len(self._rows):
            raise IndexError(index)
        previous = self._rows[index]
        if str(previous.get("state") or "") != "running":
            raise ValueError("only a running energy row can update execution metadata")
        payload = dict(previous)
        payload.pop("checkpoint_sha256", None)
        payload["execution"] = dict(execution)
        payload["updated_at"] = _now_iso()
        payload["checkpoint_sha256"] = canonical_json_sha256(payload)
        row_path = self.rows_dir / self._row_filename(
            index, str(payload.get("identity") or "")
        )
        atomic_write_json(row_path, payload)
        self._rows[index] = payload
        self._write_manifest()
        return dict(payload)

    def recover_terminal(
        self,
        index: int,
        *,
        result: Mapping[str, Any],
        reason: str = "recovered_after_parent_import_gap",
    ) -> dict[str, Any]:
        """Commit an already-finished child without starting another attempt."""

        if index < 0 or index >= len(self._rows):
            raise IndexError(index)
        previous = self._rows[index]
        if str(previous.get("state") or "") not in {"running", "cancelled"}:
            raise ValueError("only an interrupted energy row can be recovered")
        execution = previous.get("execution")
        expected_row = (
            execution.get("runtime_row")
            if isinstance(execution, Mapping) else None
        )
        recovered_row = result.get("row")
        recovered_run = result.get("run")
        if not (
            isinstance(expected_row, Mapping)
            and isinstance(recovered_row, Mapping)
            and canonical_json_sha256(expected_row)
            == canonical_json_sha256(recovered_row)
        ):
            raise ValueError(
                "recovered energy result does not match the running row"
            )
        if not (
            isinstance(recovered_run, Mapping)
            and recovered_run.get("energy_aggregate_verified") is True
        ):
            raise ValueError(
                "recovered energy result has no verified child aggregate"
            )
        expected_attempt = str(
            execution.get("execution_attempt_id") or ""
        )
        if expected_attempt and str(
            recovered_run.get("execution_attempt_id") or ""
        ) != expected_attempt:
            raise ValueError(
                "recovered energy result belongs to another attempt"
            )
        terminal_state = (
            "completed"
            if result.get("ok") is True or result.get("dry_run") is True
            else "failed"
        )
        payload = dict(previous)
        payload.pop("checkpoint_sha256", None)
        now = _now_iso()
        payload.update(
            {
                "state": terminal_state,
                "complete": True,
                "updated_at": now,
                "finished_at": now,
                "result": dict(result),
                "reason": str(reason or "recovered_after_parent_import_gap"),
            }
        )
        payload["checkpoint_sha256"] = canonical_json_sha256(payload)
        row_path = self.rows_dir / self._row_filename(
            index, str(payload.get("identity") or "")
        )
        atomic_write_json(row_path, payload)
        self._rows[index] = payload
        self._write_manifest()
        return dict(payload)

    def mark_run_state(self, state: str) -> None:
        normalized = str(state or "").strip().lower()
        if normalized not in CHECKPOINT_STATES:
            raise ValueError(f"invalid journal state: {state!r}")
        if normalized == "completed" and not self.complete:
            raise ValueError("cannot complete a journal with pending rows")
        self._run_state_override = normalized
        self._write_manifest()

    def _write_manifest(self) -> None:
        counts = {
            state: sum(
                1 for payload in self._rows
                if str(payload.get("state") or "") == state
            )
            for state in sorted(CHECKPOINT_STATES)
        }
        override = str(self._run_state_override or "")
        if override:
            state = override
        elif self.complete:
            state = "completed"
        elif counts["running"]:
            state = "running"
        elif counts["completed"] or counts["failed"] or counts["cancelled"]:
            state = "cancelled" if counts["cancelled"] else "running"
        else:
            state = "not_started"
        self._manifest.update(
            {
                "schema": self.SCHEMA,
                "schema_version": 1,
                "state": state,
                "complete": bool(
                    self.complete and state in {"completed", "failed"}
                ),
                "updated_at": _now_iso(),
                "row_count": len(self._rows),
                "counts": counts,
                "rows": [
                    {
                        "index": int(payload.get("index", -1)),
                        "identity": str(payload.get("identity") or ""),
                        "row_contract_sha256": str(
                            payload.get("row_contract_sha256") or ""
                        ),
                        "state": str(payload.get("state") or ""),
                        "complete": payload.get("complete") is True,
                        "attempt_count": int(payload.get("attempt_count") or 0),
                        "checkpoint": (
                            Path("rows")
                            / self._row_filename(
                                int(payload.get("index", -1)),
                                str(payload.get("identity") or ""),
                            )
                        ).as_posix(),
                    }
                    for payload in self._rows
                ],
            }
        )
        atomic_write_json(self.manifest_path, self._manifest)
