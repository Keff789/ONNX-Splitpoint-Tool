"""Management-node CPU reference and paired quality evaluation service.

This module separates semantic quality evidence from accelerator performance:

* a canonical ONNX Runtime CPU reference is materialised once per
  model/dataset/preprocessing/decoder/Image-ID contract;
* accelerator hosts only need to return their per-image predictions;
* paired uncertainty is evaluated on the management node by a configurable
  process pool (four workers by default);
* a resource admission gate can pause new work during u.RECS acquisition or a
  CPU-heavy local Hailo build;
* completed evaluations are persisted by their minimal scientific fingerprint.

The service has no GPU reference path.  Custom metrics are admitted through an
explicit evaluator-factory import path so backend-specific detection metrics
can prepare their own caches without coupling this orchestration layer to
hardware or decoder implementations.
"""
from __future__ import annotations

from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from .quality_cache import (
    CPUQualityReferenceIdentity,
    PersistentQualityCache,
    QualityFingerprintError,
    canonical_json,
    evaluation_fingerprint,
    image_ids_fingerprint,
    image_id_set_fingerprint,
    indexed_prediction_records,
    json_fingerprint,
    prediction_fingerprint,
)
from .native_split_quality import validate_native_split_quality_binding
from .quality_lifecycle import stamp_exception


GENERATE_CPU_QUALITY_REFERENCE = "generate_cpu_quality_reference"
EVALUATE_PAIRED_QUALITY_UNCERTAINTY = "evaluate_paired_quality_uncertainty"
QUALITY_ALGORITHM_VERSION = "management_paired_quality_v2"
DETECTION_QUALITY_ALGORITHM_VERSION = (
    f"{QUALITY_ALGORITHM_VERSION}:detection-cached-matching-v2-ap75"
)
QUALITY_RESULT_CONTRACT_VERSION = 3
RESAMPLE_SEED_SCHEMA = "numpy-pcg64-index-plan-v1"
URECS_RESOURCE_REASON = "urecs_energy_acquisition"
HAILO_BUILD_RESOURCE_REASON = "local_hailo_compilation"

# Quality metrics live on a compact, normally [-1, 1] scale.  This tolerance
# only makes the configured threshold inclusive despite final binary64
# rounding; it is far too small to absorb a scientifically meaningful loss.
_INCLUSIVE_MARGIN_MAX_ULPS = 8
_INCLUSIVE_MARGIN_ABS_TOL = 1e-15


# ``CPUQualityReferenceStore`` instances are intentionally cheap, so callers
# historically created one for every request.  A per-instance RLock cannot
# serialize those callers, though, and ``flock`` alone is not a reliable
# same-process thread mutex on every supported platform.  Keep one lock per
# resolved store/identity in the process and combine it with a filesystem lock
# for independent coordinator processes.
_REFERENCE_STORE_LOCKS_GUARD = threading.Lock()
_REFERENCE_STORE_LOCKS: dict[str, threading.RLock] = {}


def _reference_store_thread_lock(root: Path, identity_fingerprint: str) -> threading.RLock:
    key = f"{root}:{identity_fingerprint}"
    with _REFERENCE_STORE_LOCKS_GUARD:
        lock = _REFERENCE_STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _REFERENCE_STORE_LOCKS[key] = lock
        return lock


@contextmanager
def _reference_store_file_lock(root: Path, identity_fingerprint: str, *, check_cancelled=None):
    """Serialize one immutable reference identity across Linux processes.

    The benchmark campaign runs on Linux.  On platforms without ``fcntl`` the
    process-wide thread lock still provides correct single-process behaviour;
    the explicit error keeps a multi-process caller from assuming a guarantee
    that is unavailable there.
    """

    shard = root / identity_fingerprint[:2]
    shard.mkdir(parents=True, exist_ok=True)
    lock_path = shard / f".{identity_fingerprint}.lock"
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        if fcntl is not None:
            if check_cancelled is None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            else:
                while True:
                    check_cancelled()
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.05)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for an atomically published directory."""

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


class ImagePairingError(QualityFingerprintError):
    """Raised when candidate and reference predictions cannot be paired exactly."""


class QualityServiceClosedError(RuntimeError):
    """Raised when work is submitted after service shutdown."""


class QualityArtifactIntegrityError(QualityFingerprintError):
    """Raised when a runner-exported central-quality artifact is not intact."""


@dataclass(frozen=True)
class PairedPredictionRecords:
    image_ids: tuple[Any, ...]
    reference: tuple[Mapping[str, Any], ...]
    candidate: tuple[Mapping[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.image_ids)


def _display_tokens(tokens: Iterable[str], limit: int = 10) -> str:
    values = list(sorted(tokens))
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit})"
    return ", ".join(shown) + suffix


def pair_prediction_records(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    image_id_field: str = "image_id",
) -> PairedPredictionRecords:
    """Validate and pair predictions by typed image ID, independent of order.

    Exact set equality is mandatory: silently evaluating only the intersection
    would destroy paired inference and make missing accelerator outputs appear
    scientifically valid.
    """

    try:
        reference = indexed_prediction_records(reference_records, image_id_field=image_id_field)
        candidate = indexed_prediction_records(candidate_records, image_id_field=image_id_field)
    except QualityFingerprintError as exc:
        raise ImagePairingError(str(exc)) from exc
    reference_ids = set(reference)
    candidate_ids = set(candidate)
    missing = reference_ids - candidate_ids
    unexpected = candidate_ids - reference_ids
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing candidate IDs: " + _display_tokens(missing))
        if unexpected:
            details.append("unexpected candidate IDs: " + _display_tokens(unexpected))
        raise ImagePairingError("image_id sets differ; " + "; ".join(details))
    ordered = sorted(reference_ids)
    return PairedPredictionRecords(
        image_ids=tuple(reference[token][image_id_field] for token in ordered),
        reference=tuple(reference[token] for token in ordered),
        candidate=tuple(candidate[token] for token in ordered),
    )


def make_cpu_reference_identity(
    *,
    model: str,
    dataset: str,
    preprocessing: str,
    decoder: str,
    prediction_records: Optional[Sequence[Mapping[str, Any]]] = None,
    image_ids: Optional[Sequence[Any]] = None,
    image_id_field: str = "image_id",
) -> CPUQualityReferenceIdentity:
    """Create the identity before inference from known IDs, or from records.

    Campaign orchestration should normally pass ``image_ids`` from the frozen
    validation manifest so the store can be queried before CPU inference.
    ``prediction_records`` remains a convenient equivalent for imported runs.
    """

    if image_ids is not None and prediction_records is not None:
        raise ValueError("pass either image_ids or prediction_records, not both")
    if image_ids is not None:
        ids_sha = image_ids_fingerprint(image_ids)
    elif prediction_records is not None:
        ids_sha = image_id_set_fingerprint(prediction_records, image_id_field=image_id_field)
    else:
        raise ValueError("image_ids or prediction_records are required")

    return CPUQualityReferenceIdentity(
        model=str(model),
        dataset=str(dataset),
        preprocessing=str(preprocessing),
        decoder=str(decoder),
        image_ids=ids_sha,
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


class CPUQualityReferenceStore:
    """Persistent store that materialises one CPU reference per identity."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _identity_lock(self, identity: CPUQualityReferenceIdentity):
        fingerprint = identity.fingerprint()
        local_lock = _reference_store_thread_lock(self.root, fingerprint)
        with local_lock:
            with _reference_store_file_lock(self.root, fingerprint):
                yield

    def _paths(self, identity: CPUQualityReferenceIdentity) -> tuple[Path, Path]:
        directory = self.root / identity.fingerprint()[:2] / identity.fingerprint()
        return directory / "manifest.json", directory / "predictions.json"

    def get(self, identity: CPUQualityReferenceIdentity) -> Optional[dict[str, Any]]:
        manifest_path, predictions_path = self._paths(identity)
        if not manifest_path.is_file() or not predictions_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(manifest, Mapping) or not isinstance(predictions, list):
            return None
        if manifest.get("reference_identity") != identity.fingerprint():
            return None
        payload_field = manifest.get("prediction_payload_field")
        try:
            observed = prediction_fingerprint(
                predictions,
                image_id_field=str(manifest.get("image_id_field") or "image_id"),
                payload_field=(str(payload_field) if payload_field not in (None, "") else None),
            )
        except QualityFingerprintError:
            return None
        if observed != manifest.get("predictions_sha256"):
            return None
        return {
            "status_name": GENERATE_CPU_QUALITY_REFERENCE,
            "status": "completed",
            "cache_hit": True,
            "identity": identity,
            "manifest": dict(manifest),
            "predictions": predictions,
            "manifest_path": manifest_path,
            "predictions_path": predictions_path,
        }

    def put_if_absent(
        self,
        identity: CPUQualityReferenceIdentity,
        prediction_records: Sequence[Mapping[str, Any]],
        *,
        image_id_field: str = "image_id",
        prediction_payload_field: Optional[str] = None,
    ) -> dict[str, Any]:
        """Persist a generated reference, refusing identity/content collisions."""

        # Validation also guarantees that no duplicate image ID reaches disk.
        observed_ids = image_id_set_fingerprint(prediction_records, image_id_field=image_id_field)
        if observed_ids != identity.image_ids:
            raise ImagePairingError(
                "generated CPU reference Image-ID set does not match its immutable identity"
            )
        predictions_sha = prediction_fingerprint(
            prediction_records,
            image_id_field=image_id_field,
            payload_field=prediction_payload_field,
        )
        with self._identity_lock(identity):
            return self._put_if_absent_locked(
                identity,
                prediction_records,
                image_id_field=image_id_field,
                prediction_payload_field=prediction_payload_field,
                observed_ids=observed_ids,
                predictions_sha=predictions_sha,
            )

    def _put_if_absent_locked(
        self,
        identity: CPUQualityReferenceIdentity,
        prediction_records: Sequence[Mapping[str, Any]],
        *,
        image_id_field: str,
        prediction_payload_field: Optional[str],
        observed_ids: Optional[str] = None,
        predictions_sha: Optional[str] = None,
    ) -> dict[str, Any]:
        """Publish under ``_identity_lock`` as one atomic directory."""

        observed_ids = observed_ids or image_id_set_fingerprint(
            prediction_records, image_id_field=image_id_field
        )
        if observed_ids != identity.image_ids:
            raise ImagePairingError(
                "generated CPU reference Image-ID set does not match its immutable identity"
            )
        predictions_sha = predictions_sha or prediction_fingerprint(
            prediction_records,
            image_id_field=image_id_field,
            payload_field=prediction_payload_field,
        )
        existing = self.get(identity)
        if existing is not None:
            if existing["manifest"].get("predictions_sha256") != predictions_sha:
                raise QualityFingerprintError(
                    "CPU reference identity collision: the same contract produced different predictions"
                )
            return existing

        manifest_path, predictions_path = self._paths(identity)
        target_directory = manifest_path.parent
        if target_directory.exists():
            raise QualityArtifactIntegrityError(
                "existing CPU reference artifact is incomplete or corrupt; refusing to overwrite immutable evidence"
            )
        target_directory.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "onnx-splitpoint/cpu-quality-reference-artifact",
            "schema_version": 1,
            **identity.metadata(),
            "status": "completed",
            "image_id_field": str(image_id_field),
            "prediction_payload_field": prediction_payload_field,
            "image_count": len(prediction_records),
            "image_ids_sha256": observed_ids,
            "predictions_sha256": predictions_sha,
        }
        staging = Path(tempfile.mkdtemp(
            prefix=f".{identity.fingerprint()}.",
            suffix=".staging",
            dir=str(target_directory.parent),
        ))
        try:
            _atomic_write_json(staging / "predictions.json", list(prediction_records))
            _atomic_write_json(staging / "manifest.json", manifest)
            _fsync_directory(staging)
            # The final directory did not exist when the identity lock was
            # acquired.  Publishing the complete directory makes readers see
            # either no artifact or both immutable files, never a half write.
            os.replace(staging, target_directory)
            _fsync_directory(target_directory.parent)
        finally:
            if staging.exists():
                for child in staging.iterdir():
                    try:
                        child.unlink()
                    except OSError:
                        pass
                try:
                    staging.rmdir()
                except OSError:
                    pass
        return {
            "status_name": GENERATE_CPU_QUALITY_REFERENCE,
            "status": "completed",
            "cache_hit": False,
            "identity": identity,
            "manifest": manifest,
            "predictions": list(prediction_records),
            "manifest_path": manifest_path,
            "predictions_path": predictions_path,
        }

    def materialize(
        self,
        identity: CPUQualityReferenceIdentity,
        generator: Callable[[], Sequence[Mapping[str, Any]]],
        *,
        image_id_field: str = "image_id",
        prediction_payload_field: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load the reference or call ``generator`` once in this store instance."""

        # Hold the identity lock across generation.  This is shared by every
        # store instance in this process and backed by a file lock, so multiple
        # coordinator threads/processes cannot publish the same directory in
        # parallel.
        with self._identity_lock(identity):
            existing = self.get(identity)
            if existing is not None:
                return existing
            generated = generator()
            if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
                raise TypeError("CPU quality reference generator must return per-image prediction records")
            return self._put_if_absent_locked(
                identity,
                generated,
                image_id_field=image_id_field,
                prediction_payload_field=prediction_payload_field,
            )


def annotations_fingerprint(annotations: Any) -> str:
    """Fingerprint annotations independent of harmless list ordering."""

    if isinstance(annotations, Sequence) and not isinstance(annotations, (str, bytes, bytearray)):
        ordered = sorted(list(annotations), key=canonical_json)
        return json_fingerprint({"schema": "quality-annotations-v1", "records": ordered})
    return json_fingerprint({"schema": "quality-annotations-v1", "records": annotations})


@dataclass(frozen=True)
class QualityEvaluationRequest:
    reference_records: Sequence[Mapping[str, Any]]
    candidate_records: Sequence[Mapping[str, Any]]
    annotations: Any
    metric_gate_config: Mapping[str, Any]
    repetitions: int
    seed: int
    confidence_level: float
    non_inferiority_margin: float
    evaluator_factory: str = "paired_mean"
    value_field: str = "value"
    image_id_field: str = "image_id"
    reference_prediction_field: Optional[str] = None
    candidate_prediction_field: Optional[str] = None
    prediction_payload_field: Optional[str] = None
    algorithm_version: str = QUALITY_ALGORITHM_VERSION
    reference_identity: Optional[str] = None
    request_id: str = ""
    candidate_execution_completion_contract_sha256: str = ""
    artifact_provenance_binding_status: str = "not_required"
    artifact_provenance_claim_eligible: bool = False

    def __post_init__(self) -> None:
        if int(self.repetitions) < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1.0")
        if float(self.non_inferiority_margin) < 0.0:
            raise ValueError("non_inferiority_margin must be non-negative")
        if not str(self.evaluator_factory).strip():
            raise ValueError("evaluator_factory must not be empty")


def deterministic_resample_plan(*, image_count: int, repetitions: int, seed: int) -> np.ndarray:
    """Generate one worker-count-independent paired image resampling plan."""

    n = int(image_count)
    reps = int(repetitions)
    if n < 1 or reps < 1:
        raise ValueError("image_count and repetitions must be positive")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.integers(0, n, size=(reps, n), dtype=np.int64)


def _seed_schema(request: QualityEvaluationRequest, image_count: int) -> dict[str, Any]:
    return {
        "schema": RESAMPLE_SEED_SCHEMA,
        "seed": int(request.seed),
        "repetitions": int(request.repetitions),
        "image_count": int(image_count),
    }


def _metric_config(request: QualityEvaluationRequest) -> dict[str, Any]:
    return {
        "gate": dict(request.metric_gate_config),
        "evaluator_factory": str(request.evaluator_factory),
        "value_field": str(request.value_field),
        "image_id_field": str(request.image_id_field),
        "reference_prediction_field": request.reference_prediction_field or request.prediction_payload_field,
        "candidate_prediction_field": request.candidate_prediction_field or request.prediction_payload_field,
        "confidence_level": float(request.confidence_level),
        "non_inferiority_margin": float(request.non_inferiority_margin),
        "repetitions": int(request.repetitions),
    }


def prepare_evaluation(request: QualityEvaluationRequest) -> tuple[str, dict[str, Any]]:
    """Validate/pair once and build the process-pool payload plus cache key."""

    paired = pair_prediction_records(
        request.reference_records,
        request.candidate_records,
        image_id_field=request.image_id_field,
    )
    ref_sha = prediction_fingerprint(
        paired.reference,
        image_id_field=request.image_id_field,
        payload_field=request.reference_prediction_field or request.prediction_payload_field,
    )
    candidate_sha = prediction_fingerprint(
        paired.candidate,
        image_id_field=request.image_id_field,
        payload_field=request.candidate_prediction_field or request.prediction_payload_field,
    )
    annotation_sha = annotations_fingerprint(request.annotations)
    seed_schema = _seed_schema(request, paired.count)
    key = evaluation_fingerprint(
        reference_predictions_sha256=ref_sha,
        candidate_predictions_sha256=candidate_sha,
        annotations_sha256=annotation_sha,
        metric_gate_config=_metric_config(request),
        algorithm_version=request.algorithm_version,
        seed_schema=seed_schema,
    )
    payload = {
        "request_id": str(request.request_id),
        "candidate_execution_completion_contract_sha256": str(
            request.candidate_execution_completion_contract_sha256 or ""
        ),
        "artifact_provenance_binding_status": str(
            request.artifact_provenance_binding_status or "not_required"
        ),
        "artifact_provenance_claim_eligible": bool(
            request.artifact_provenance_claim_eligible
        ),
        "reference_identity": request.reference_identity,
        "reference_records": [dict(item) for item in paired.reference],
        "candidate_records": [dict(item) for item in paired.candidate],
        "image_ids": list(paired.image_ids),
        "annotations": request.annotations,
        "metric_gate_config": dict(request.metric_gate_config),
        "repetitions": int(request.repetitions),
        "seed": int(request.seed),
        "confidence_level": float(request.confidence_level),
        "non_inferiority_margin": float(request.non_inferiority_margin),
        "evaluator_factory": str(request.evaluator_factory),
        "value_field": str(request.value_field),
        "image_id_field": str(request.image_id_field),
        "reference_prediction_field": request.reference_prediction_field or request.prediction_payload_field,
        "candidate_prediction_field": request.candidate_prediction_field or request.prediction_payload_field,
        "algorithm_version": str(request.algorithm_version),
        "evaluation_fingerprint": key,
        "reference_predictions_sha256": ref_sha,
        "candidate_predictions_sha256": candidate_sha,
        "annotations_sha256": annotation_sha,
        "seed_schema": seed_schema,
    }
    return key, payload


def _quality_artifact_path(descriptor: Mapping[str, Any], request_path: Path) -> Path:
    raw = str(descriptor.get("path") or "").strip()
    if not raw:
        raise QualityArtifactIntegrityError("quality artifact descriptor has no path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = request_path.parent / candidate
    elif not candidate.is_file():
        # Debug/analysis packs rebase remote result roots.  The v2.63 export
        # places both artifacts beside the request, so basename fallback is
        # unambiguous and remains protected by size/SHA verification below.
        candidate = request_path.parent / candidate.name
    return candidate.resolve()


def _load_quality_artifact(
    descriptor: Mapping[str, Any],
    request_path: Path,
    *,
    verify: bool,
) -> tuple[dict[str, Any], str, Path]:
    path = _quality_artifact_path(descriptor, request_path)
    if not path.is_file():
        raise QualityArtifactIntegrityError(f"quality artifact is missing: {path}")
    encoded = path.read_bytes()
    observed_sha = hashlib.sha256(encoded).hexdigest()
    if verify:
        expected_size = descriptor.get("size_bytes")
        if expected_size is not None and int(expected_size) != len(encoded):
            raise QualityArtifactIntegrityError(
                f"quality artifact size mismatch for {path.name}: expected {expected_size}, got {len(encoded)}"
            )
        expected_sha = str(descriptor.get("sha256") or "").strip().lower()
        if len(expected_sha) != 64 or expected_sha != observed_sha:
            raise QualityArtifactIntegrityError(
                f"quality artifact SHA-256 mismatch for {path.name}"
            )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise QualityArtifactIntegrityError(f"quality artifact is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise QualityArtifactIntegrityError(f"quality artifact root must be an object: {path}")
    return dict(payload), observed_sha, path


def _load_bound_quality_artifact_bytes(
    descriptor: Mapping[str, Any],
    request_path: Path,
    encoded: bytes,
) -> tuple[dict[str, Any], str, Path]:
    """Load already-admitted bytes under their exact descriptor identity.

    This is the active-workflow management-reference boundary.  The runner
    acquired ``encoded`` through one anchored ``O_NOFOLLOW`` descriptor; this
    helper deliberately performs no filesystem query or second path open.
    """

    raw_path = str(descriptor.get("path") or "").strip()
    if not raw_path:
        raise QualityArtifactIntegrityError(
            "management reference descriptor has no path identity"
        )
    logical_path = Path(raw_path).expanduser()
    if not logical_path.is_absolute():
        logical_path = request_path.parent / logical_path
    logical_path = Path(os.path.abspath(os.fspath(logical_path)))
    expected_size = descriptor.get("size_bytes")
    if type(expected_size) is not int or int(expected_size) <= 0:
        raise QualityArtifactIntegrityError(
            "management reference descriptor has no positive byte-size binding"
        )
    if len(encoded) != int(expected_size):
        raise QualityArtifactIntegrityError(
            "management reference descriptor byte-size mismatch"
        )
    expected_sha = _require_contract_sha256(
        descriptor.get("sha256"),
        "management reference descriptor.sha256",
    )
    observed_sha = hashlib.sha256(encoded).hexdigest()
    if observed_sha != expected_sha:
        raise QualityArtifactIntegrityError(
            "management reference descriptor SHA-256 mismatch"
        )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise QualityArtifactIntegrityError(
            "management reference artifact bytes are not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise QualityArtifactIntegrityError(
            "management reference artifact byte root must be an object"
        )
    return dict(payload), observed_sha, logical_path


def _quality_contract_digest(value: Mapping[str, Any]) -> str:
    identity = dict(value)
    identity.pop("quality_contract_sha256", None)
    return json_fingerprint(identity)


def _require_contract_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise QualityArtifactIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_detection_quality_contract(
    value: Any,
    *,
    role: str,
) -> tuple[dict[str, Any], str]:
    """Validate the frozen raw-endpoint -> canonical-record adapter contract."""

    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(f"{role} lacks the required detection quality contract")
    contract = dict(value)
    if contract.get("schema") != "onnx-splitpoint/central-detection-quality-contract":
        raise QualityArtifactIntegrityError(f"{role} has an unsupported detection quality contract schema")
    if int(contract.get("schema_version") or 0) != 1 or str(contract.get("task") or "") != "detection":
        raise QualityArtifactIntegrityError(f"{role} detection quality contract is not schema-v1 detection")
    declared = _require_contract_sha256(
        contract.get("quality_contract_sha256"),
        f"{role}.quality_contract_sha256",
    )
    observed = _quality_contract_digest(contract)
    if declared != observed:
        raise QualityArtifactIntegrityError(f"{role} detection quality contract SHA-256 mismatch")

    model = contract.get("model")
    dataset = contract.get("dataset")
    preprocessing = contract.get("preprocessing")
    decoder = contract.get("decoder")
    nms = contract.get("nms")
    quality_endpoint = contract.get("quality_record_endpoint")
    if not all(isinstance(item, Mapping) for item in (
        model, dataset, preprocessing, decoder, nms, quality_endpoint,
    )):
        raise QualityArtifactIntegrityError(
            f"{role} detection quality contract lacks model/dataset/preprocessing/decoder/NMS/quality-endpoint provenance"
        )
    _require_contract_sha256(model.get("sha256"), f"{role}.model.sha256")
    _require_contract_sha256(dataset.get("manifest_sha256"), f"{role}.dataset.manifest_sha256")
    _require_contract_sha256(dataset.get("image_ids_sha256"), f"{role}.dataset.image_ids_sha256")
    _require_contract_sha256(dataset.get("ground_truth_sha256"), f"{role}.dataset.ground_truth_sha256")
    if int(dataset.get("image_count") or 0) < 1:
        raise QualityArtifactIntegrityError(f"{role}.dataset.image_count must be positive")

    for component_name, component in (
        ("preprocessing", preprocessing), ("decoder", decoder),
        ("nms", nms), ("quality_record_endpoint", quality_endpoint),
    ):
        identity = component.get("identity")
        if not isinstance(identity, Mapping):
            raise QualityArtifactIntegrityError(f"{role}.{component_name}.identity is missing")
        component_sha = _require_contract_sha256(
            component.get("sha256"), f"{role}.{component_name}.sha256"
        )
        if component_sha != json_fingerprint(identity):
            raise QualityArtifactIntegrityError(f"{role}.{component_name} contract SHA-256 mismatch")

    decoder_identity = decoder.get("identity")
    nms_identity = nms.get("identity")
    quality_endpoint_identity = quality_endpoint.get("identity")
    if not all(isinstance(item, Mapping) for item in (
        decoder_identity, nms_identity, quality_endpoint_identity,
    )):
        raise QualityArtifactIntegrityError(f"{role} decoder/NMS identities are missing")
    decoder_runner = _require_contract_sha256(
        decoder_identity.get("implementation_runner_sha256"),
        f"{role}.decoder.implementation_runner_sha256",
    )
    nms_runner = _require_contract_sha256(
        nms_identity.get("implementation_runner_sha256"),
        f"{role}.nms.implementation_runner_sha256",
    )
    if decoder_runner != nms_runner:
        raise QualityArtifactIntegrityError(f"{role} decoder and NMS implementation hashes differ")
    quality_endpoint_sha = _require_contract_sha256(
        quality_endpoint.get("sha256"),
        f"{role}.quality_record_endpoint.sha256",
    )
    _require_contract_sha256(
        quality_endpoint_identity.get(
            "vendored_endpoint_attestor_sha256"
        ),
        f"{role}.quality_record_endpoint.vendored_endpoint_attestor_sha256",
    )
    if (
        str(quality_endpoint_identity.get("schema") or "")
        != (
            "onnx-splitpoint/"
            "detection-quality-record-endpoint-contract"
        )
        or int(quality_endpoint_identity.get("schema_version") or 0) != 1
        or str(quality_endpoint_identity.get(
            "canonical_record_endpoint"
        ) or "") != "decoded_xyxy_score_class_detections"
        or _require_contract_sha256(
            quality_endpoint_identity.get("decoder_contract_sha256"),
            f"{role}.quality_record_endpoint.decoder_contract_sha256",
        ) != _require_contract_sha256(
            decoder.get("sha256"), f"{role}.decoder.sha256",
        )
        or _require_contract_sha256(
            quality_endpoint_identity.get("nms_contract_sha256"),
            f"{role}.quality_record_endpoint.nms_contract_sha256",
        ) != _require_contract_sha256(
            nms.get("sha256"), f"{role}.nms.sha256",
        )
        or _require_contract_sha256(
            quality_endpoint_identity.get("implementation_runner_sha256"),
            f"{role}.quality_record_endpoint.implementation_runner_sha256",
        ) != decoder_runner
        or _require_contract_sha256(
            contract.get("quality_record_endpoint_contract_sha256"),
            f"{role}.quality_record_endpoint_contract_sha256",
        ) != quality_endpoint_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} detection quality-record endpoint binding is inconsistent"
        )
    if str(decoder_identity.get("canonical_record_endpoint") or "") != "decoded_xyxy_score_class_detections":
        raise QualityArtifactIntegrityError(f"{role} decoder does not produce canonical detection records")
    if str(contract.get("canonical_record_endpoint") or "") != "decoded_xyxy_score_class_detections":
        raise QualityArtifactIntegrityError(f"{role} canonical detection-record endpoint is missing")
    if str(contract.get("contract_scope") or "") != "canonical_quality_record_semantics":
        raise QualityArtifactIntegrityError(f"{role} detection quality contract scope is ambiguous")
    if str(contract.get("source_endpoint_role") or "") != "canonical_reference_model_output":
        raise QualityArtifactIntegrityError(f"{role} source endpoint role is ambiguous")
    if bool(contract.get("source_endpoint_is_raw")):
        if str(decoder_identity.get("source_endpoint_semantics") or "") != "raw_multiscale_head":
            raise QualityArtifactIntegrityError(f"{role} raw endpoint semantics are inconsistent")
        if decoder_identity.get("source_endpoint_has_integrated_nms") is not False:
            raise QualityArtifactIntegrityError(
                f"{role} raw multiscale head must not be declared as integrated NMS"
            )
    return contract, declared


def _validate_deepx_pre_nms_completion(
    contract: Mapping[str, Any], *, role: str,
) -> None:
    """Check the existing physical, host-tail and comparison bindings.

    Runtime signatures omit tensor names/dtypes, and quality images need not
    share the geometry of the measured frame. Neither difference changes the
    decoder/NMS invariant. Completion counts belong to their own execution.
    """
    from onnx_splitpoint_tool.native_detection_postprocess import (
        build_completed_detection_endpoint_attestation,
        canonical_json_sha256,
        verify_completed_detection_comparison_endpoint_contract,
        verify_frozen_postprocess_contract,
    )

    try:
        endpoint = contract["endpoint"]["identity"]
        quality = contract["quality_record_endpoint"]["identity"]
        source = quality["source_endpoint"]
        adapter = quality["host_decoder_nms"]
        decoder = quality["decoder_contract"]
        frozen = verify_frozen_postprocess_contract(
            decoder["frozen_postprocess_contract"]
        )
        decoder_identity = contract["quality_contract"]["decoder"]["identity"]
        nms_identity = contract["quality_contract"]["nms"]["identity"]
        if (
            endpoint.get("output_format") != "ultralytics_decoded"
            or source.get("semantics") != "decoded_pre_nms"
            or source.get("has_integrated_nms") is not False
            or frozen.get("source_contract_family") != "decoded_pre_nms"
            or frozen.get("output_contract_family") != "decoded_nms"
            or adapter.get("decoder_applied") is not True
            or adapter.get("nms_applied") is not True
            or decoder.get("host_decoder_applied") is not True
            or decoder.get("host_nms_applied") is not True
            or decoder.get("source_endpoint_semantics") != "decoded_pre_nms"
            or decoder.get("source_endpoint_has_integrated_nms") is not False
            or decoder.get("decoder_format") != frozen["decoder_format"]
            or decoder.get("family") != frozen["model_family"]
            or decoder_identity.get("host_decoder_applied") is not True
            or decoder_identity.get("source_endpoint_semantics") != "decoded_pre_nms"
            or decoder_identity.get("source_endpoint_has_integrated_nms") is not False
            or nms_identity.get("host_nms_applied") is not True
            or nms_identity.get("source_endpoint_has_integrated_nms") is not False
            or decoder.get("frozen_postprocess_contract_sha256")
            != frozen["contract_sha256"]
            or decoder.get("frozen_postprocess_invariant_contract_sha256")
            != frozen["invariant_contract_sha256"]
            or decoder.get("frozen_postprocess_invariant_identity")
            != frozen["invariant_identity"]
            or decoder_identity.get("frozen_postprocess_invariant_contract_sha256")
            != frozen["invariant_contract_sha256"]
        ):
            raise ValueError("pre-NMS source or applied host decoder/NMS differs")
        for adapter_key, frozen_key, decoder_key, identity in (
            ("decoder_id", "decoder_id", "decoder_id", decoder_identity),
            ("confidence_threshold", "confidence_threshold", "confidence_threshold", decoder_identity),
            ("iou_threshold", "iou_threshold", "nms_iou_threshold", nms_identity),
            ("max_detections", "max_detections", "nms_max_detections", nms_identity),
        ):
            if (
                adapter.get(adapter_key) != frozen[frozen_key]
                or decoder.get(decoder_key) != frozen[frozen_key]
                or identity.get(adapter_key) != frozen[frozen_key]
            ):
                raise ValueError("host decoder/NMS settings differ from frozen contract")

        # Compare the shared signature projection; names/dtypes remain checked
        # by the full frozen verifier and its physical completion contract.
        compact = endpoint["tensor_signature"]
        full = frozen["raw_output_tensor_signature"]
        compact_tensors = compact["tensors"]
        full_tensors = full["tensors"]
        if (
            compact.get("tensor_count") != full.get("tensor_count")
            or len(compact_tensors) != len(full_tensors)
            or len(compact_tensors) != compact.get("tensor_count")
        ):
            raise ValueError("runtime/frozen tensor signature counts differ")
        for ordinal, (runtime_tensor, frozen_tensor) in enumerate(
            zip(compact_tensors, full_tensors)
        ):
            if (
                any(type(runtime_tensor.get(key)) is not int for key in ("index", "rank"))
                or runtime_tensor.get("index") != ordinal
                or not isinstance(runtime_tensor.get("shape"), list)
                or any(type(size) is not int or size <= 0 for size in runtime_tensor["shape"])
                or {key: runtime_tensor.get(key) for key in ("index", "rank", "shape")}
                != {key: frozen_tensor.get(key) for key in ("index", "rank", "shape")}
            ):
                raise ValueError("runtime/frozen tensor signature differs")
            for optional in ("name", "dtype"):
                if optional in runtime_tensor and runtime_tensor[optional] != frozen_tensor.get(optional):
                    raise ValueError("runtime/frozen tensor metadata differs")
        observed = source["observed_outputs"]["outputs"]
        if (
            len(observed) != len(full_tensors)
            or any(
                {key: item.get(key) for key in ("index", "shape")}
                != {key: tensor.get(key) for key in ("index", "shape")}
                for item, tensor in zip(observed, full_tensors)
            )
        ):
            raise ValueError("observed/runtime tensor signature differs")
        verify_completed_detection_comparison_endpoint_contract(
            contract["completed_task_endpoint_contract"], frozen_contract=frozen,
        )

        attestation = contract["completed_task_endpoint_attestation"]
        for field in ("completed_frames", "postprocess_completed_frames"):
            if type(attestation.get(field)) is not int or attestation[field] <= 0:
                raise ValueError("completion frame count is invalid")
        # The attestation may describe another image. Reconstruct its expected
        # existing contract from the verified invariant and its own geometry;
        # this is a comparison only, never a mutation of recorded evidence.
        measured_frozen = dict(frozen)
        measured_frozen["original_wh"] = attestation["completed_endpoint_contract"]["original_wh"]
        measured_frozen.pop("contract_sha256", None)
        measured_frozen["contract_sha256"] = canonical_json_sha256(measured_frozen)
        # Its native source contract can carry more metadata than the compact
        # quality endpoint. Verify its own binding; the shared tensor/decoder
        # semantics above, rather than equality of unlike hashes, join them.
        physical_source_hash = _require_contract_sha256(
            attestation["completed_endpoint_contract"].get(
                "source_endpoint_contract_hash"
            ),
            f"{role}.completed_endpoint.source_endpoint_contract_hash",
        )
        expected = build_completed_detection_endpoint_attestation(
            measured_frozen, attestation["frozen_postprocess_result"],
            completed_frames=attestation["completed_frames"],
            postprocess_completed_frames=attestation["postprocess_completed_frames"],
            source_endpoint_contract_hash=physical_source_hash,
        )
        if dict(attestation) != expected:
            raise ValueError("physical completion attestation differs from frozen host tail")
    except Exception as exc:
        raise QualityArtifactIntegrityError(
            f"{role} decoded_pre_nms completion contract is invalid: {exc}"
        ) from exc


def _validate_deepx_candidate_execution_contract(
    value: Any,
    *,
    role: str,
    task: str,
) -> tuple[dict[str, Any], str]:
    """Validate an accelerator candidate contract independently of CPU decode.

    A DeepX host decoder is not the ONNX Runtime reference decoder.  Requiring
    those implementations to have the same hash would be false provenance.
    This validator therefore proves the candidate path internally; the loader
    later joins it to the CPU reference only at model/dataset/GT/Image-ID and
    canonical record semantics.
    """
    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(f"{role} lacks a candidate execution contract")
    contract = dict(value)
    if contract.get("schema") != "onnx-splitpoint/central-quality-producer-identity":
        raise QualityArtifactIntegrityError(f"{role} candidate execution contract schema is unsupported")
    if int(contract.get("schema_version") or 0) != 1:
        raise QualityArtifactIntegrityError(f"{role} candidate execution contract version is unsupported")
    if str(contract.get("task") or "").strip().lower() != str(task or "").strip().lower():
        raise QualityArtifactIntegrityError(f"{role} candidate execution task does not match request")
    declared = _require_contract_sha256(
        contract.get("producer_identity_sha256"), f"{role}.producer_identity_sha256"
    )
    unhashed = dict(contract)
    unhashed.pop("producer_identity_sha256", None)
    if json_fingerprint(unhashed) != declared:
        raise QualityArtifactIntegrityError(f"{role} candidate execution contract SHA-256 mismatch")

    model = contract.get("model")
    dataset = contract.get("dataset")
    preprocessing = contract.get("preprocessing")
    endpoint = contract.get("endpoint")
    quality_record_endpoint = contract.get("quality_record_endpoint")
    precision = contract.get("precision")
    if not all(isinstance(item, Mapping) for item in (
        model, dataset, preprocessing, endpoint, quality_record_endpoint, precision,
    )):
        raise QualityArtifactIntegrityError(
            f"{role} candidate contract lacks model/dataset/preprocessing/runtime-endpoint/quality-endpoint/precision provenance"
        )
    source_onnx = _require_contract_sha256(
        model.get("source_onnx_sha256"), f"{role}.model.source_onnx_sha256"
    )
    runtime_artifact = _require_contract_sha256(
        model.get("runtime_artifact_sha256"), f"{role}.model.runtime_artifact_sha256"
    )
    dataset_hashes = {
        name: _require_contract_sha256(dataset.get(name), f"{role}.dataset.{name}")
        for name in ("manifest_sha256", "image_ids_sha256", "ground_truth_sha256")
    }
    if int(dataset.get("image_count") or 0) < 1:
        raise QualityArtifactIntegrityError(f"{role}.dataset.image_count must be positive")

    for component_name, component in (
        ("preprocessing", preprocessing), ("endpoint", endpoint),
        ("quality_record_endpoint", quality_record_endpoint), ("precision", precision),
    ):
        identity = component.get("identity")
        if not isinstance(identity, Mapping):
            raise QualityArtifactIntegrityError(f"{role}.{component_name}.identity is missing")
        component_sha = _require_contract_sha256(
            component.get("sha256"), f"{role}.{component_name}.sha256"
        )
        if component_sha != json_fingerprint(identity):
            raise QualityArtifactIntegrityError(f"{role}.{component_name} SHA-256 mismatch")

    quality_contract = contract.get("quality_contract")
    if not isinstance(quality_contract, Mapping):
        raise QualityArtifactIntegrityError(f"{role} candidate quality contract is missing")
    quality_contract = dict(quality_contract)
    if quality_contract.get("schema") != "onnx-splitpoint/deepx-central-quality-record-contract":
        raise QualityArtifactIntegrityError(f"{role} candidate quality contract schema is unsupported")
    if int(quality_contract.get("schema_version") or 0) != 1:
        raise QualityArtifactIntegrityError(f"{role} candidate quality contract version is unsupported")
    quality_sha = _require_contract_sha256(
        quality_contract.get("quality_contract_sha256"),
        f"{role}.quality_contract.quality_contract_sha256",
    )
    if quality_sha != _quality_contract_digest(quality_contract):
        raise QualityArtifactIntegrityError(f"{role} candidate quality contract SHA-256 mismatch")
    if _require_contract_sha256(
        contract.get("quality_contract_sha256"), f"{role}.quality_contract_sha256",
    ) != quality_sha:
        raise QualityArtifactIntegrityError(f"{role} candidate quality contract binding is inconsistent")

    quality_model = quality_contract.get("model")
    quality_dataset = quality_contract.get("dataset")
    quality_preprocessing = quality_contract.get("preprocessing")
    quality_endpoint = quality_contract.get("quality_record_endpoint")
    if not all(isinstance(item, Mapping) for item in (
        quality_model, quality_dataset, quality_preprocessing, quality_endpoint,
    )):
        raise QualityArtifactIntegrityError(f"{role} candidate quality components are incomplete")
    if _require_contract_sha256(
        quality_model.get("source_onnx_sha256"),
        f"{role}.quality_contract.model.source_onnx_sha256",
    ) != source_onnx:
        raise QualityArtifactIntegrityError(f"{role} candidate quality model binding is inconsistent")
    for name, expected_hash in dataset_hashes.items():
        if _require_contract_sha256(
            quality_dataset.get(name), f"{role}.quality_contract.dataset.{name}",
        ) != expected_hash:
            raise QualityArtifactIntegrityError(f"{role} candidate quality dataset binding is inconsistent")
    if int(quality_dataset.get("image_count") or 0) != int(dataset.get("image_count") or 0):
        raise QualityArtifactIntegrityError(f"{role} candidate quality dataset count is inconsistent")

    preprocessing_sha = _require_contract_sha256(
        preprocessing.get("sha256"), f"{role}.preprocessing.sha256",
    )
    endpoint_sha = _require_contract_sha256(endpoint.get("sha256"), f"{role}.endpoint.sha256")
    quality_endpoint_sha = _require_contract_sha256(
        quality_record_endpoint.get("sha256"), f"{role}.quality_record_endpoint.sha256",
    )
    if (
        _require_contract_sha256(
            contract.get("preprocessing_contract_sha256"),
            f"{role}.preprocessing_contract_sha256",
        ) != preprocessing_sha
        or _require_contract_sha256(
            quality_preprocessing.get("sha256"),
            f"{role}.quality_contract.preprocessing.sha256",
        ) != preprocessing_sha
    ):
        raise QualityArtifactIntegrityError(f"{role} preprocessing contract binding is inconsistent")
    if (
        _require_contract_sha256(
            quality_endpoint.get("sha256"),
            f"{role}.quality_contract.quality_record_endpoint.sha256",
        ) != quality_endpoint_sha
        or _require_contract_sha256(
            contract.get("quality_record_endpoint_contract_sha256"),
            f"{role}.quality_record_endpoint_contract_sha256",
        ) != quality_endpoint_sha
        or _require_contract_sha256(
            quality_contract.get("quality_record_endpoint_contract_sha256"),
            f"{role}.quality_contract.quality_record_endpoint_contract_sha256",
        ) != quality_endpoint_sha
    ):
        raise QualityArtifactIntegrityError(f"{role} quality-record endpoint binding is inconsistent")
    if _require_contract_sha256(
        contract.get("endpoint_contract_hash"), f"{role}.endpoint_contract_hash"
    ) != endpoint_sha:
        raise QualityArtifactIntegrityError(f"{role} runtime endpoint hash binding is inconsistent")

    completed_fields = (
        "completed_task_endpoint_contract",
        "completed_task_endpoint_contract_hash",
        "completed_task_output_endpoint_id",
        "completed_task_endpoint_attestation",
        "completed_task_endpoint_attestation_sha256",
        "quality_join_endpoint",
    )
    completed_present = [field in contract for field in completed_fields]
    if any(completed_present):
        if task != "detection" or not all(completed_present):
            raise QualityArtifactIntegrityError(
                f"{role} completed-task endpoint binding is incomplete"
            )
        try:
            from onnx_splitpoint_tool.native_detection_postprocess import (
                verify_completed_detection_comparison_endpoint_contract,
            )

            completed_contract = (
                verify_completed_detection_comparison_endpoint_contract(
                    contract.get("completed_task_endpoint_contract")
                )
            )
        except Exception as exc:
            raise QualityArtifactIntegrityError(
                f"{role} completed-task endpoint contract is invalid"
            ) from exc
        completed_hash = _require_contract_sha256(
            contract.get("completed_task_endpoint_contract_hash"),
            f"{role}.completed_task_endpoint_contract_hash",
        )
        completed_id = str(
            contract.get("completed_task_output_endpoint_id") or ""
        )
        attestation = contract.get("completed_task_endpoint_attestation")
        if (
            completed_hash
            != str(completed_contract.get("endpoint_contract_hash") or "")
            or completed_id
            != str(completed_contract.get("output_endpoint_id") or "")
            or str(contract.get("quality_join_endpoint") or "")
            != "completed_task_decoded_nms"
            or not isinstance(attestation, Mapping)
            or attestation.get("attested") is not True
            or str(attestation.get("status") or "").strip().lower()
            != "passed"
            or json_fingerprint(attestation)
            != _require_contract_sha256(
                contract.get(
                    "completed_task_endpoint_attestation_sha256"
                ),
                f"{role}.completed_task_endpoint_attestation_sha256",
            )
        ):
            raise QualityArtifactIntegrityError(
                f"{role} completed-task endpoint binding is inconsistent"
            )

    if task == "detection":
        for component_name in ("decoder", "nms"):
            component = quality_contract.get(component_name)
            if not isinstance(component, Mapping) or not isinstance(component.get("identity"), Mapping):
                raise QualityArtifactIntegrityError(
                    f"{role} candidate {component_name} quality contract is incomplete"
                )
            component_sha = _require_contract_sha256(
                component.get("sha256"), f"{role}.quality_contract.{component_name}.sha256",
            )
            if component_sha != json_fingerprint(component.get("identity")):
                raise QualityArtifactIntegrityError(
                    f"{role} candidate {component_name} quality contract SHA-256 mismatch"
                )
            if _require_contract_sha256(
                contract.get(f"{component_name}_contract_sha256"),
                f"{role}.{component_name}_contract_sha256",
            ) != component_sha:
                raise QualityArtifactIntegrityError(
                    f"{role} candidate {component_name} quality binding is inconsistent"
                )

    endpoint_identity = endpoint.get("identity")
    quality_endpoint_identity = quality_record_endpoint.get("identity")
    precision_identity = precision.get("identity")
    preprocessing_identity = preprocessing.get("identity")
    if not all(isinstance(item, Mapping) for item in (
        endpoint_identity, quality_endpoint_identity, precision_identity, preprocessing_identity,
    )):
        raise QualityArtifactIntegrityError(f"{role} candidate component identity is incomplete")
    if (
        endpoint_identity.get("schema") != "onnx-splitpoint/output-endpoint-contract"
        or int(endpoint_identity.get("schema_version") or 0) != 3
        or str(endpoint_identity.get("task") or "") != task
        or not isinstance(endpoint_identity.get("tensor_signature"), Mapping)
    ):
        raise QualityArtifactIntegrityError(f"{role} runtime endpoint identity is invalid")
    runtime_stage = str(endpoint_identity.get("stage") or "")
    if task == "detection" and runtime_stage not in {"raw_head", "decoded_pre_nms", "decoded_nms"}:
        raise QualityArtifactIntegrityError(f"{role} detection runtime endpoint stage is invalid")
    if task == "detection" and runtime_stage == "decoded_pre_nms":
        if not all(completed_present):
            raise QualityArtifactIntegrityError(
                f"{role} decoded_pre_nms completed-task endpoint binding is incomplete"
            )
        _validate_deepx_pre_nms_completion(contract, role=role)
    if task == "classification" and runtime_stage not in {
        "classification_logits", "classification_probabilities",
    }:
        raise QualityArtifactIntegrityError(f"{role} classification runtime endpoint stage is invalid")

    runner_sha = _require_contract_sha256(
        contract.get("implementation_runner_sha256"), f"{role}.implementation_runner_sha256"
    )
    if _require_contract_sha256(
        quality_endpoint_identity.get("implementation_runner_sha256"),
        f"{role}.quality_record_endpoint.implementation_runner_sha256",
    ) != runner_sha:
        raise QualityArtifactIntegrityError(
            f"{role} quality endpoint implementation is not bound to the suite runner"
        )
    precision_artifact = _require_contract_sha256(
        precision_identity.get("artifact_sha256"), f"{role}.precision.artifact_sha256"
    )
    if precision_artifact != runtime_artifact:
        raise QualityArtifactIntegrityError(f"{role} precision identity targets a different runtime artifact")
    expected_precision_token = f"deepx_dxnn_sha256:{runtime_artifact}"
    if str(contract.get("runtime_precision_identity") or "") != expected_precision_token:
        raise QualityArtifactIntegrityError(f"{role} runtime precision identity is not artifact-bound")
    if str(precision_identity.get("precision_semantics") or "") != "opaque_vendor_compiled_artifact_identity":
        raise QualityArtifactIntegrityError(f"{role} unknown vendor compute precision was misrepresented")
    prepared_input_evidence = contract.get("prepared_input_evidence")
    quality_prepared_input_evidence = quality_contract.get(
        "prepared_input_evidence"
    )
    if isinstance(prepared_input_evidence, Mapping):
        if (
            not isinstance(quality_prepared_input_evidence, Mapping)
            or dict(quality_prepared_input_evidence)
            != dict(prepared_input_evidence)
        ):
            raise QualityArtifactIntegrityError(
                f"{role} prepared-input evidence binding is inconsistent"
            )
        prepared_records_sha = _require_contract_sha256(
            prepared_input_evidence.get("records_sha256"),
            f"{role}.prepared_input_evidence.records_sha256",
        )
        numeric_identity = prepared_input_evidence.get(
            "runtime_numeric_input_identity"
        )
        numeric_sha = _require_contract_sha256(
            prepared_input_evidence.get(
                "runtime_numeric_input_sha256"
            ),
            f"{role}.prepared_input_evidence."
            "runtime_numeric_input_sha256",
        )
        join_binding = prepared_input_evidence.get(
            "performance_quality_input_binding"
        )
        join_sha = _require_contract_sha256(
            prepared_input_evidence.get(
                "performance_quality_input_binding_sha256"
            ),
            f"{role}.prepared_input_evidence."
            "performance_quality_input_binding_sha256",
        )
        join_shape = (
            join_binding.get("prepared_input_shape")
            if isinstance(join_binding, Mapping) else None
        )
        numeric_shape = (
            list(numeric_identity.get("runtime_input_shape") or [])
            if isinstance(numeric_identity, Mapping) else []
        )
        numeric_dtype = str(
            (numeric_identity or {}).get("runtime_input_dtype") or ""
        ).strip().lower() if isinstance(numeric_identity, Mapping) else ""
        numeric_layout = str(
            (numeric_identity or {}).get("runtime_input_layout") or ""
        ).strip().upper() if isinstance(numeric_identity, Mapping) else ""
        numeric_name = str(
            (numeric_identity or {}).get("runtime_input_name") or ""
        ).strip() if isinstance(numeric_identity, Mapping) else ""
        dtype_sizes = {
            "uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
            "float16": 2, "uint32": 4, "int32": 4,
            "float32": 4, "float64": 8,
        }
        numeric_bytes = (
            math.prod(numeric_shape) * dtype_sizes.get(numeric_dtype, 0)
            if numeric_shape else 0
        )
        if (
            prepared_input_evidence.get("schema")
            != "onnx-splitpoint/deepx-quality-prepared-input-set"
            or int(prepared_input_evidence.get("schema_version") or 0)
            != 1
            or int(prepared_input_evidence.get("record_count") or 0)
            != int(dataset.get("image_count") or 0)
            or not isinstance(numeric_identity, Mapping)
            or json_fingerprint(numeric_identity) != numeric_sha
            or str(
                numeric_identity.get(
                    "preprocessing_contract_sha256"
                ) or ""
            ).strip().lower() != preprocessing_sha
            or not isinstance(join_binding, Mapping)
            or join_binding.get("schema")
            != (
                "onnx-splitpoint/"
                "deepx-performance-quality-input-binding"
            )
            or int(join_binding.get("schema_version") or 0) != 1
            or join_binding.get("binding_verified") is not True
            or not str(join_binding.get("source_image_id") or "").strip()
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(join_binding.get("source_image_sha256") or "")
                .strip().lower(),
            ) is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(join_binding.get("prepared_input_sha256") or "")
                .strip().lower(),
            ) is None
            or isinstance(join_binding.get("prepared_input_bytes"), bool)
            or not isinstance(join_binding.get("prepared_input_bytes"), int)
            or int(join_binding.get("prepared_input_bytes") or 0) <= 0
            or not str(join_binding.get("prepared_input_name") or "").strip()
            or not isinstance(join_shape, list)
            or not join_shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (join_shape or [])
            )
            or not str(join_binding.get("prepared_input_dtype") or "").strip()
            or not str(join_binding.get("prepared_input_layout") or "").strip()
            or str(join_binding.get("prepared_input_name") or "").strip()
            != numeric_name
            or list(join_shape or []) != numeric_shape
            or str(
                join_binding.get("prepared_input_dtype") or ""
            ).strip().lower() != numeric_dtype
            or str(
                join_binding.get("prepared_input_layout") or ""
            ).strip().upper() != numeric_layout
            or int(join_binding.get("prepared_input_bytes") or 0)
            != numeric_bytes
            or str(
                join_binding.get("runtime_preprocessing_sha256") or ""
            ).strip().lower() != preprocessing_sha
            or str(
                join_binding.get("runtime_numeric_input_sha256") or ""
            ).strip().lower() != numeric_sha
            or json_fingerprint(join_binding) != join_sha
            or contract.get("prepared_input_join_binding") != join_binding
            or _require_contract_sha256(
                contract.get("prepared_input_join_binding_sha256"),
                f"{role}.prepared_input_join_binding_sha256",
            ) != join_sha
            or _require_contract_sha256(
                contract.get("prepared_input_evidence_sha256"),
                f"{role}.prepared_input_evidence_sha256",
            ) != prepared_records_sha
        ):
            raise QualityArtifactIntegrityError(
                f"{role} prepared-input evidence is invalid"
            )
    else:
        # Backward compatibility for archived v1 DeepX quality artifacts.
        _require_contract_sha256(
            preprocessing_identity.get("per_image_transforms_sha256"),
            f"{role}.preprocessing.per_image_transforms_sha256",
        )
        if int(
            preprocessing_identity.get("per_image_transform_count") or 0
        ) != int(dataset.get("image_count") or 0):
            raise QualityArtifactIntegrityError(
                f"{role} per-image preprocessing transform count differs "
                "from dataset"
            )

    canonical_endpoint = str(quality_endpoint_identity.get("canonical_record_endpoint") or "")
    if task == "detection":
        if canonical_endpoint != "decoded_xyxy_score_class_detections":
            raise QualityArtifactIntegrityError(f"{role} detection candidate endpoint is not canonical")
        source_endpoint = quality_endpoint_identity.get("source_endpoint")
        host_adapter = quality_endpoint_identity.get("host_decoder_nms")
        observed = source_endpoint.get("observed_outputs") if isinstance(source_endpoint, Mapping) else None
        if not isinstance(source_endpoint, Mapping) or not isinstance(host_adapter, Mapping) or not isinstance(observed, Mapping):
            raise QualityArtifactIntegrityError(f"{role} detection source/host endpoint attestation is incomplete")
        if not str(source_endpoint.get("semantics") or ""):
            raise QualityArtifactIntegrityError(f"{role} detection source endpoint semantics are missing")
        outputs = observed.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise QualityArtifactIntegrityError(f"{role} detection runtime output shapes are missing")
        for output in outputs:
            if not isinstance(output, Mapping) or not isinstance(output.get("shape"), list):
                raise QualityArtifactIntegrityError(f"{role} detection runtime output attestation is incomplete")
        if source_endpoint.get("semantics") == "fixed_topk_xyxy_score_class_candidates":
            # This producer has completed TopK, with no NMS in its graph.
            # Admit only the same sealed, graph-bound selection that the
            # actual per-image consumer executed at its unchanged threshold.
            from .native_detection_postprocess import verify_detection_completion_execution_contract
            from .native_output_endpoint import bn6_candidate_selection
            try:
                decoder = quality_endpoint_identity["decoder_contract"]
                selection = bn6_candidate_selection({"source_onnx_detection_endpoint": {
                    "candidate_selection": decoder["candidate_selection"]}})
                execution = verify_detection_completion_execution_contract(decoder["completion_execution_contract"])
                processor = execution["processor_contract"]
                if (not selection or selection["source_onnx_sha256"] != model.get("source_onnx_sha256")
                        or processor.get("candidate_selection") != selection
                        or source_endpoint.get("has_integrated_nms") is not False
                        or host_adapter.get("decoder_applied") is not False
                        or host_adapter.get("nms_applied") is not False
                        or host_adapter.get("confidence_threshold") != processor["confidence_threshold"]
                        or host_adapter.get("iou_threshold") != processor["iou_threshold"]
                        or host_adapter.get("max_detections") != processor["max_detections"]
                        or [r["shape"] for r in outputs] != [selection["output_shape"]]):
                    raise ValueError("candidate selection binding mismatch")
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise QualityArtifactIntegrityError(f"{role} candidate selection contract invalid: {exc}") from exc
        elif source_endpoint.get("has_integrated_nms") is False:
            if host_adapter.get("decoder_applied") is not True or host_adapter.get("nms_applied") is not True:
                raise QualityArtifactIntegrityError(f"{role} pre-NMS source lacks the applied host decoder/NMS")
            for field_name in ("confidence_threshold", "iou_threshold", "max_detections"):
                if host_adapter.get(field_name) is None:
                    raise QualityArtifactIntegrityError(f"{role} host decoder/NMS lacks {field_name}")
    elif task == "classification":
        if canonical_endpoint != "classification_topk_hits":
            raise QualityArtifactIntegrityError(f"{role} classification candidate endpoint is not canonical")
        observed = quality_endpoint_identity.get("runtime_observed_outputs")
        if not isinstance(observed, Mapping) or not isinstance(observed.get("outputs"), list):
            raise QualityArtifactIntegrityError(f"{role} classification runtime output shapes are missing")
    else:
        raise QualityArtifactIntegrityError(f"{role} unsupported candidate task")
    return contract, declared


def _require_contract_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise QualityArtifactIntegrityError(f"{label} must be non-empty")
    return text


def _require_contract_size(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualityArtifactIntegrityError(f"{label} must be a positive integer")
    size = int(value)
    if size < 1:
        raise QualityArtifactIntegrityError(f"{label} must be a positive integer")
    return size


def _validate_trt_artifact(
    value: Any, *, role: str, artifact_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role}.{artifact_name} artifact binding is missing"
        )
    artifact = dict(value)
    _require_contract_text(artifact.get("path"), f"{role}.{artifact_name}.path")
    _require_contract_sha256(
        artifact.get("sha256"), f"{role}.{artifact_name}.sha256"
    )
    _require_contract_size(
        artifact.get("size_bytes"), f"{role}.{artifact_name}.size_bytes"
    )
    return artifact


def _validate_tensorrt_build_receipt(
    value: Any,
    *,
    role: str,
    build_onnx: Mapping[str, Any],
    engine: Mapping[str, Any],
    trtexec: Mapping[str, Any],
    runtime_precision_identity: str,
) -> tuple[dict[str, Any], str]:
    """Verify the canonical build receipt embedded in a TRT producer.

    The management node cannot trust a remote path.  It therefore validates
    the complete receipt by value: canonical digests, artifact identities and
    the exact ``trtexec`` source/output arguments must all agree with the
    separately sealed producer artifacts.
    """

    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role}.engine_build_receipt binding is missing"
        )
    binding = dict(value)
    _require_contract_text(
        binding.get("path"), f"{role}.engine_build_receipt.path"
    )
    declared_binding_sha = _require_contract_sha256(
        binding.get("sha256"), f"{role}.engine_build_receipt.sha256"
    )
    declared_binding_size = _require_contract_size(
        binding.get("size_bytes"), f"{role}.engine_build_receipt.size_bytes"
    )
    raw_receipt = binding.get("receipt")
    if not isinstance(raw_receipt, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role}.engine_build_receipt.receipt is missing"
        )
    receipt = dict(raw_receipt)
    declared_receipt_sha = _require_contract_sha256(
        receipt.get("receipt_sha256"),
        f"{role}.engine_build_receipt.receipt.receipt_sha256",
    )
    receipt_identity = dict(receipt)
    receipt_identity.pop("receipt_sha256", None)
    if json_fingerprint(receipt_identity) != declared_receipt_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build receipt SHA-256 mismatch"
        )
    if declared_binding_sha != json_fingerprint(receipt):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build-receipt binding SHA-256 mismatch"
        )
    canonical_receipt_bytes = canonical_json(receipt).encode("utf-8")
    if declared_binding_size != len(canonical_receipt_bytes):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build-receipt canonical size mismatch"
        )
    if (
        receipt.get("schema")
        != "onnx-splitpoint/tensorrt-engine-build-receipt"
        or int(receipt.get("schema_version") or 0) != 1
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build receipt schema/status is invalid"
        )

    expected_receipt_fields = {
        "source_onnx": str(build_onnx.get("path") or ""),
        "source_onnx_sha256": str(build_onnx.get("sha256") or ""),
        "engine": str(engine.get("path") or ""),
        "engine_sha256": str(engine.get("sha256") or ""),
        "trtexec": str(trtexec.get("path") or ""),
        "trtexec_sha256": str(trtexec.get("sha256") or ""),
    }
    for field_name, expected in expected_receipt_fields.items():
        if str(receipt.get(field_name) or "") != expected:
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT build receipt {field_name} binding mismatch"
            )

    command = receipt.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build receipt command is invalid"
        )
    argv = [str(item) for item in command]
    if argv[0] != expected_receipt_fields["trtexec"]:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build receipt executable mismatch"
        )
    onnx_args = [item for item in argv[1:] if item.startswith("--onnx=")]
    engine_args = [item for item in argv[1:] if item.startswith("--saveEngine=")]
    if (
        onnx_args != [f"--onnx={expected_receipt_fields['source_onnx']}"]
        or engine_args != [f"--saveEngine={expected_receipt_fields['engine']}"]
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build receipt source/engine command mismatch"
        )
    precision_flags = {
        "fp16": "--fp16",
        "int8": "--int8",
    }
    if runtime_precision_identity not in {"fp16", "fp32", "int8"}:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT runtime precision identity is unsupported"
        )
    required_flag = precision_flags.get(runtime_precision_identity)
    if required_flag and required_flag not in argv[1:]:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build command does not attest {runtime_precision_identity}"
        )
    if runtime_precision_identity == "fp32" and any(
        flag in argv[1:] for flag in ("--fp16", "--int8")
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT FP32 identity conflicts with build flags"
        )
    if runtime_precision_identity != "fp16" and "--fp16" in argv[1:]:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT precision identity conflicts with --fp16"
        )
    if runtime_precision_identity != "int8" and "--int8" in argv[1:]:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT precision identity conflicts with --int8"
        )
    return binding, declared_receipt_sha


def _validate_tensorrt_endpoint_authority(
    value: Any,
    *,
    role: str,
    source_onnx_sha256: str,
    endpoint_identity: Mapping[str, Any],
    endpoint_sha256: str,
) -> dict[str, Any]:
    """Bind a TRT endpoint to the recorded suite contract and exact Full graph."""

    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role}.endpoint_authority binding is missing"
        )
    component = dict(value)
    identity = component.get("identity")
    if not isinstance(identity, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role}.endpoint_authority.identity is missing"
        )
    identity = dict(identity)
    authority_sha = _require_contract_sha256(
        component.get("sha256"), f"{role}.endpoint_authority.sha256",
    )
    if authority_sha != json_fingerprint(identity):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint-authority SHA-256 mismatch"
        )
    if (
        identity.get("schema")
        != "onnx-splitpoint/tensorrt-endpoint-authority"
        or int(identity.get("schema_version") or 0) != 1
        or identity.get("graph_binding_source")
        != "authoritative_suite_output_contract_plus_exact_onnx_endpoint:v2"
        or identity.get("endpoint_contract_complete") is not True
        or identity.get("contract_resolution_status") != "attested"
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint authority is incomplete or unattested"
        )

    source_contracts_sha = _require_contract_sha256(
        identity.get("source_contracts_sha256"),
        f"{role}.endpoint_authority.source_contracts_sha256",
    )
    recorded_contract_sha = _require_contract_sha256(
        identity.get("recorded_contract_sha256"),
        f"{role}.endpoint_authority.recorded_contract_sha256",
    )
    if source_contracts_sha == recorded_contract_sha:
        # Container bytes and the selected canonical row are distinct hash
        # domains.  Equality is not impossible cryptographically, but treating
        # it as admissible would erase the provenance distinction this v1
        # authority component exists to preserve.
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint authority collapses container and row identities"
        )
    for field_name in ("full_model_sha256", "terminal_model_sha256"):
        if _require_contract_sha256(
            identity.get(field_name),
            f"{role}.endpoint_authority.{field_name}",
        ) != source_onnx_sha256:
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT endpoint authority targets a different ONNX graph"
            )
    if _require_contract_sha256(
        identity.get("endpoint_contract_hash"),
        f"{role}.endpoint_authority.endpoint_contract_hash",
    ) != endpoint_sha256:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint authority targets a different endpoint"
        )

    endpoint_stage = str(endpoint_identity.get("stage") or "")
    if str(identity.get("stage") or "") != endpoint_stage:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint-authority stage differs from endpoint"
        )
    attestation = identity.get("output_endpoint_attestation")
    if not isinstance(attestation, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint authority lacks runtime attestation"
        )
    if (
        attestation.get("attested") is not True
        or str(attestation.get("status") or "") != "passed"
        or str(attestation.get("stage") or "") != endpoint_stage
        or _require_contract_sha256(
            attestation.get("endpoint_contract_hash"),
            f"{role}.endpoint_authority.output_endpoint_attestation.endpoint_contract_hash",
        ) != endpoint_sha256
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint runtime attestation is inconsistent"
        )
    return component


def _validate_tensorrt_candidate_execution_contract(
    value: Any,
    *,
    role: str,
    task: str,
) -> tuple[dict[str, Any], str]:
    """Validate one setup-local TensorRT Full quality producer contract."""

    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role} lacks a TensorRT candidate execution contract"
        )
    contract = dict(value)
    if (
        contract.get("schema")
        != "onnx-splitpoint/tensorrt-central-quality-producer-identity"
        or int(contract.get("schema_version") or 0) != 1
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT candidate execution contract schema is unsupported"
        )
    declared = _require_contract_sha256(
        contract.get("producer_identity_sha256"),
        f"{role}.producer_identity_sha256",
    )
    unhashed = dict(contract)
    unhashed.pop("producer_identity_sha256", None)
    if json_fingerprint(unhashed) != declared:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT candidate execution contract SHA-256 mismatch"
        )

    raw_task = str(contract.get("task") or "")
    declared_task = raw_task.strip().lower()
    if declared_task not in {"classification", "detection"} or declared_task != str(
        task or ""
    ).strip().lower():
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT candidate execution task does not match request"
        )
    if raw_task != declared_task:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT candidate execution task is not canonical"
        )
    execution_role = str(contract.get("execution_role") or "")
    if execution_role not in {
        "full_quality_only", "full_performance_owner",
    }:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer execution role is invalid"
        )
    performance_claims_emitted = contract.get("performance_claims_emitted")
    if not isinstance(performance_claims_emitted, bool):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT performance-claim status is missing"
        )
    if execution_role == "full_quality_only" and performance_claims_emitted:
        raise QualityArtifactIntegrityError(
            f"{role} quality-only TensorRT producer emitted performance claims"
        )
    if str(contract.get("backend") or "") != "native_tensorrt":
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer backend is not native_tensorrt"
        )
    if str(contract.get("variant") or "") != "full":
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer variant is not full"
        )
    if str(contract.get("case_id") or "") != "full":
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT Full producer case_id is not full"
        )
    for field_name in (
        "eval_run_id", "model_id", "setup_id", "source_run_id",
    ):
        canonical_text = _require_contract_text(
            contract.get(field_name), f"{role}.{field_name}"
        )
        if str(contract.get(field_name)) != canonical_text:
            raise QualityArtifactIntegrityError(
                f"{role}.{field_name} is not canonical"
            )
    _require_contract_sha256(
        contract.get("policy_sha256"), f"{role}.policy_sha256"
    )

    source_onnx = _validate_trt_artifact(
        contract.get("source_onnx"), role=role, artifact_name="source_onnx",
    )
    build_onnx = _validate_trt_artifact(
        contract.get("build_onnx"), role=role, artifact_name="build_onnx",
    )
    engine = _validate_trt_artifact(
        contract.get("engine"), role=role, artifact_name="engine",
    )
    trtexec = _validate_trt_artifact(
        contract.get("trtexec"), role=role, artifact_name="trtexec",
    )
    source_sha = str(source_onnx["sha256"])
    build_sha = str(build_onnx["sha256"])
    engine_sha = str(engine["sha256"])
    if _require_contract_sha256(
        build_onnx.get("source_onnx_sha256"),
        f"{role}.build_onnx.source_onnx_sha256",
    ) != source_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT build ONNX targets a different source model"
        )
    if (
        _require_contract_sha256(
            engine.get("source_onnx_sha256"),
            f"{role}.engine.source_onnx_sha256",
        ) != source_sha
        or _require_contract_sha256(
            engine.get("build_onnx_sha256"),
            f"{role}.engine.build_onnx_sha256",
        ) != build_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT engine model binding is inconsistent"
        )

    runtime_precision = str(
        contract.get("runtime_precision_identity") or ""
    ).strip().lower().replace(" ", "")
    if runtime_precision not in {"fp16", "fp32", "int8"}:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT runtime precision identity is unsupported"
        )
    if str(contract.get("runtime_precision_identity") or "") != runtime_precision:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT runtime precision identity is not canonical"
        )
    _validate_tensorrt_build_receipt(
        contract.get("engine_build_receipt"),
        role=role,
        build_onnx=build_onnx,
        engine=engine,
        trtexec=trtexec,
        runtime_precision_identity=runtime_precision,
    )
    _require_contract_sha256(
        contract.get("engine_build_receipt_file_sha256"),
        f"{role}.engine_build_receipt_file_sha256",
    )

    model = contract.get("model")
    dataset = contract.get("dataset")
    preprocessing = contract.get("preprocessing")
    endpoint = contract.get("endpoint")
    endpoint_authority = contract.get("endpoint_authority")
    precision = contract.get("precision")
    quality_record_endpoint = contract.get("quality_record_endpoint")
    endpoint_attestor = contract.get("endpoint_attestor")
    if not all(isinstance(item, Mapping) for item in (
        model, dataset, preprocessing, endpoint, endpoint_authority, precision,
        quality_record_endpoint, endpoint_attestor,
    )):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer model/dataset/preprocessing/endpoint/precision/attestor provenance is incomplete"
        )
    model = dict(model)
    dataset = dict(dataset)
    preprocessing = dict(preprocessing)
    endpoint = dict(endpoint)
    endpoint_authority = dict(endpoint_authority)
    precision = dict(precision)
    quality_record_endpoint = dict(quality_record_endpoint)
    endpoint_attestor = dict(endpoint_attestor)
    for field_name, observed, expected in (
        ("source_onnx_sha256", model.get("source_onnx_sha256"), source_sha),
        ("build_onnx_sha256", model.get("build_onnx_sha256"), build_sha),
        ("runtime_artifact_sha256", model.get("runtime_artifact_sha256"), engine_sha),
    ):
        if _require_contract_sha256(
            observed, f"{role}.model.{field_name}",
        ) != expected:
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT producer model {field_name} binding mismatch"
            )
    for field_name, observed, expected in (
        (
            "source_onnx_size_bytes", model.get("source_onnx_size_bytes"),
            source_onnx.get("size_bytes"),
        ),
        (
            "build_onnx_size_bytes", model.get("build_onnx_size_bytes"),
            build_onnx.get("size_bytes"),
        ),
        (
            "runtime_artifact_size_bytes",
            model.get("runtime_artifact_size_bytes"), engine.get("size_bytes"),
        ),
    ):
        if _require_contract_size(
            observed, f"{role}.model.{field_name}",
        ) != int(expected):
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT producer model {field_name} binding mismatch"
            )

    quality_contract_raw = contract.get("quality_contract")
    quality_validator = (
        _validate_detection_quality_contract
        if declared_task == "detection"
        else _validate_classification_quality_contract
    )
    quality_contract, quality_sha = quality_validator(
        quality_contract_raw, role=f"{role} TensorRT producer",
    )
    if _require_contract_sha256(
        contract.get("quality_contract_sha256"),
        f"{role}.quality_contract_sha256",
    ) != quality_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer quality contract binding is inconsistent"
        )
    quality_model = quality_contract.get("model")
    quality_dataset = quality_contract.get("dataset")
    quality_preprocessing = quality_contract.get("preprocessing")
    if not all(isinstance(item, Mapping) for item in (
        quality_model, quality_dataset, quality_preprocessing,
    )):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer quality contract is incomplete"
        )
    if _require_contract_sha256(
        quality_model.get("sha256"),
        f"{role}.quality_contract.model.sha256",
    ) != source_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT quality uses a different source ONNX"
        )
    if dataset != dict(quality_dataset):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer dataset differs from quality contract"
        )
    if preprocessing != dict(quality_preprocessing):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT producer preprocessing differs from quality contract"
        )
    quality_contract_endpoint = quality_contract.get(
        "quality_record_endpoint"
    )
    if (
        not isinstance(quality_contract_endpoint, Mapping)
        or quality_record_endpoint != dict(quality_contract_endpoint)
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT quality-record endpoint differs from quality contract"
        )

    for component_name, component in (
        ("endpoint", endpoint),
        ("precision", precision),
        ("quality_record_endpoint", quality_record_endpoint),
        ("endpoint_attestor", endpoint_attestor),
    ):
        identity = component.get("identity")
        if not isinstance(identity, Mapping):
            raise QualityArtifactIntegrityError(
                f"{role}.{component_name}.identity is missing"
            )
        component_sha = _require_contract_sha256(
            component.get("sha256"), f"{role}.{component_name}.sha256",
        )
        if component_sha != json_fingerprint(identity):
            raise QualityArtifactIntegrityError(
                f"{role}.{component_name} SHA-256 mismatch"
            )

    endpoint_identity = endpoint.get("identity")
    endpoint_sha = str(endpoint.get("sha256") or "")
    if (
        not isinstance(endpoint_identity, Mapping)
        or endpoint_identity.get("schema")
        != "onnx-splitpoint/output-endpoint-contract"
        or int(endpoint_identity.get("schema_version") or 0) != 3
        or str(endpoint_identity.get("task") or "") != declared_task
        or not isinstance(endpoint_identity.get("tensor_signature"), Mapping)
        or contract.get("endpoint_contract_complete") is not True
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT runtime endpoint identity is incomplete"
        )
    endpoint_stage = str(endpoint_identity.get("stage") or "")
    valid_stages = (
        {"raw_head", "decoded_pre_nms", "decoded_nms"}
        if declared_task == "detection"
        else {"classification_logits", "classification_probabilities"}
    )
    if endpoint_stage not in valid_stages:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT runtime endpoint stage is invalid"
        )
    if _require_contract_sha256(
        contract.get("endpoint_contract_hash"),
        f"{role}.endpoint_contract_hash",
    ) != endpoint_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT runtime endpoint hash binding is inconsistent"
        )
    _validate_tensorrt_endpoint_authority(
        endpoint_authority,
        role=role,
        source_onnx_sha256=source_sha,
        endpoint_identity=endpoint_identity,
        endpoint_sha256=endpoint_sha,
    )

    precision_identity = precision.get("identity")
    if (
        not isinstance(precision_identity, Mapping)
        or precision_identity.get("schema")
        != "onnx-splitpoint/tensorrt-runtime-precision-contract"
        or int(precision_identity.get("schema_version") or 0) != 1
        or str(precision_identity.get("runtime_precision_identity") or "")
        != runtime_precision
        or _require_contract_sha256(
            precision_identity.get("source_onnx_sha256"),
            f"{role}.precision.source_onnx_sha256",
        ) != source_sha
        or _require_contract_sha256(
            precision_identity.get("build_onnx_sha256"),
            f"{role}.precision.build_onnx_sha256",
        ) != build_sha
        or _require_contract_sha256(
            precision_identity.get("engine_sha256"),
            f"{role}.precision.engine_sha256",
        ) != engine_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT precision contract binding is inconsistent"
        )

    quality_record_identity = quality_record_endpoint.get("identity")
    quality_record_sha = str(quality_record_endpoint.get("sha256") or "")
    if not isinstance(quality_record_identity, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT quality-record endpoint identity is missing"
        )
    canonical_record_endpoint = str(
        quality_record_identity.get("canonical_record_endpoint") or ""
    )
    endpoint_attestor_identity = endpoint_attestor.get("identity")
    if not isinstance(endpoint_attestor_identity, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint-attestor identity is missing"
        )
    vendored_attestor_sha = _require_contract_sha256(
        contract.get("vendored_endpoint_attestor_sha256"),
        f"{role}.vendored_endpoint_attestor_sha256",
    )
    if (
        str(endpoint_attestor_identity.get("schema") or "")
        != "onnx-splitpoint/vendored-endpoint-attestor-identity"
        or int(endpoint_attestor_identity.get("schema_version") or 0) != 1
        or str(endpoint_attestor_identity.get("source") or "")
        != "suite_vendored"
        or _require_contract_sha256(
            endpoint_attestor_identity.get("sha256"),
            f"{role}.endpoint_attestor.identity.sha256",
        ) != vendored_attestor_sha
        or _require_contract_sha256(
            endpoint_attestor_identity.get("expected_sha256"),
            f"{role}.endpoint_attestor.identity.expected_sha256",
        ) != vendored_attestor_sha
        or _require_contract_sha256(
            quality_record_identity.get(
                "vendored_endpoint_attestor_sha256"
            ),
            f"{role}.quality_record_endpoint.vendored_endpoint_attestor_sha256",
        ) != vendored_attestor_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT endpoint-attestor binding is inconsistent"
        )
    if declared_task == "classification":
        postprocessor = quality_contract.get("postprocessor")
        if (
            not isinstance(postprocessor, Mapping)
            or canonical_record_endpoint != "classification_topk_hits"
            or _require_contract_sha256(
                quality_record_identity.get("postprocessor_contract_sha256"),
                f"{role}.quality_record_endpoint.postprocessor_contract_sha256",
            ) != _require_contract_sha256(
                postprocessor.get("sha256"),
                f"{role}.quality_contract.postprocessor.sha256",
            )
        ):
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT classification quality endpoint binding is inconsistent"
            )
        expected_decoder_sha = ""
        expected_nms_sha = ""
    else:
        decoder = quality_contract.get("decoder")
        nms = quality_contract.get("nms")
        if (
            not isinstance(decoder, Mapping)
            or not isinstance(nms, Mapping)
            or canonical_record_endpoint
            != "decoded_xyxy_score_class_detections"
            or _require_contract_sha256(
                quality_record_identity.get("decoder_contract_sha256"),
                f"{role}.quality_record_endpoint.decoder_contract_sha256",
            ) != _require_contract_sha256(
                decoder.get("sha256"),
                f"{role}.quality_contract.decoder.sha256",
            )
            or _require_contract_sha256(
                quality_record_identity.get("nms_contract_sha256"),
                f"{role}.quality_record_endpoint.nms_contract_sha256",
            ) != _require_contract_sha256(
                nms.get("sha256"),
                f"{role}.quality_contract.nms.sha256",
            )
        ):
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT detection quality endpoint binding is inconsistent"
            )
        expected_decoder_sha = str(decoder.get("sha256") or "")
        expected_nms_sha = str(nms.get("sha256") or "")

    if _require_contract_sha256(
        contract.get("preprocessing_contract_sha256"),
        f"{role}.preprocessing_contract_sha256",
    ) != _require_contract_sha256(
        quality_preprocessing.get("sha256"),
        f"{role}.quality_contract.preprocessing.sha256",
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT preprocessing contract binding is inconsistent"
        )
    for field_name, expected in (
        ("decoder_contract_sha256", expected_decoder_sha),
        ("nms_contract_sha256", expected_nms_sha),
    ):
        observed = str(contract.get(field_name) or "").strip().lower()
        if observed != expected:
            raise QualityArtifactIntegrityError(
                f"{role} TensorRT {field_name} binding is inconsistent"
            )
    if _require_contract_sha256(
        contract.get("quality_record_endpoint_contract_sha256"),
        f"{role}.quality_record_endpoint_contract_sha256",
    ) != quality_record_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT quality-record endpoint hash binding is inconsistent"
        )

    runner_sha = _require_contract_sha256(
        contract.get("implementation_runner_sha256"),
        f"{role}.implementation_runner_sha256",
    )
    if _require_contract_sha256(
        quality_record_identity.get("implementation_runner_sha256"),
        f"{role}.quality_record_endpoint.implementation_runner_sha256",
    ) != runner_sha:
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT quality endpoint runner binding is inconsistent"
        )
    if declared_task == "classification":
        postprocessor_identity = quality_contract.get("postprocessor", {}).get("identity")
        implementation_identity = postprocessor_identity
    else:
        implementation_identity = quality_contract.get("decoder", {}).get("identity")
    if (
        not isinstance(implementation_identity, Mapping)
        or _require_contract_sha256(
            implementation_identity.get("implementation_runner_sha256"),
            f"{role}.quality_contract.implementation_runner_sha256",
        ) != runner_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} TensorRT quality contract runner binding is inconsistent"
        )
    return contract, declared


def _validate_generic_composed_candidate_execution_contract(
    value: Any,
    *,
    role: str,
    task: str,
) -> tuple[dict[str, Any], str]:
    """Validate the setup-local producer of one generic composed candidate."""

    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role} lacks a generic composed producer identity"
        )
    contract = dict(value)
    if (
        contract.get("schema")
        != "onnx-splitpoint/generic-composed-quality-producer-identity"
        or type(contract.get("schema_version")) is not int
        or contract.get("schema_version") != 1
    ):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed producer schema is unsupported"
        )
    declared = _require_contract_sha256(
        contract.get("producer_identity_sha256"),
        f"{role}.producer_identity_sha256",
    )
    unhashed = dict(contract)
    unhashed.pop("producer_identity_sha256", None)
    if json_fingerprint(unhashed) != declared:
        raise QualityArtifactIntegrityError(
            f"{role} generic composed producer SHA-256 mismatch"
        )
    task_text = _require_contract_text(contract.get("task"), f"{role}.task")
    if task_text != str(task or "").strip().lower() or task_text not in {
        "classification", "detection",
    }:
        raise QualityArtifactIntegrityError(
            f"{role} generic composed producer task mismatch"
        )
    if (
        str(contract.get("variant") or "") != "composed"
        or str(contract.get("execution_role") or "")
        != "generic_composed_quality_candidate"
        or contract.get("performance_claims_emitted") is not False
    ):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed producer role/variant is invalid"
        )
    for field_name in (
        "eval_run_id", "model_id", "setup_id", "source_run_id", "case_id",
        "backend", "stage1_backend", "stage2_backend",
    ):
        observed = _require_contract_text(
            contract.get(field_name), f"{role}.{field_name}",
        )
        if str(contract.get(field_name)) != observed:
            raise QualityArtifactIntegrityError(
                f"{role}.{field_name} is not canonical"
            )
    if str(contract.get("backend")) != str(contract.get("source_run_id")):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed backend/source-run mismatch"
        )

    model = contract.get("model")
    dataset = contract.get("dataset")
    preprocessing = contract.get("preprocessing")
    quality_endpoint = contract.get("quality_record_endpoint")
    quality_contract = contract.get("quality_contract")
    endpoint_contract = contract.get("endpoint_contract")
    if not all(isinstance(item, Mapping) for item in (
        model, dataset, preprocessing, quality_endpoint, quality_contract,
        endpoint_contract,
    )):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed model/dataset/runtime bindings are incomplete"
        )
    for field_name in (
        "source_onnx_sha256", "part1_onnx_sha256", "part2_onnx_sha256",
        "runtime_artifact_sha256",
    ):
        _require_contract_sha256(model.get(field_name), f"{role}.model.{field_name}")
    for field_name in (
        "source_onnx_size_bytes", "part1_onnx_size_bytes",
        "part2_onnx_size_bytes",
    ):
        if type(model.get(field_name)) is not int or int(model[field_name]) < 1:
            raise QualityArtifactIntegrityError(
                f"{role}.model.{field_name} must be a positive integer"
            )
    runtime_artifacts = contract.get("runtime_artifacts")
    if not isinstance(runtime_artifacts, list) or len(runtime_artifacts) < 2:
        raise QualityArtifactIntegrityError(
            f"{role} generic composed runtime artifact set is incomplete"
        )
    artifact_roles: set[str] = set()
    for index, raw_artifact in enumerate(runtime_artifacts):
        if not isinstance(raw_artifact, Mapping):
            raise QualityArtifactIntegrityError(
                f"{role}.runtime_artifacts[{index}] is invalid"
            )
        artifact_role = _require_contract_text(
            raw_artifact.get("role"),
            f"{role}.runtime_artifacts[{index}].role",
        )
        _require_contract_text(
            raw_artifact.get("backend"),
            f"{role}.runtime_artifacts[{index}].backend",
        )
        _require_contract_text(
            raw_artifact.get("name"),
            f"{role}.runtime_artifacts[{index}].name",
        )
        _require_contract_sha256(
            raw_artifact.get("sha256"),
            f"{role}.runtime_artifacts[{index}].sha256",
        )
        if (
            artifact_role in artifact_roles
            or type(raw_artifact.get("size_bytes")) is not int
            or int(raw_artifact["size_bytes"]) < 1
        ):
            raise QualityArtifactIntegrityError(
                f"{role} generic composed runtime artifact roles/sizes are invalid"
            )
        artifact_roles.add(artifact_role)
    runtime_set_sha = _require_contract_sha256(
        contract.get("runtime_artifact_set_sha256"),
        f"{role}.runtime_artifact_set_sha256",
    )
    if (
        runtime_set_sha
        != json_fingerprint({
            "schema": "onnx-splitpoint/composed-runtime-artifact-set",
            "schema_version": 1,
            "artifacts": runtime_artifacts,
        })
        or str(model.get("runtime_artifact_sha256") or "") != runtime_set_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed runtime artifact-set hash mismatch"
        )
    _require_contract_sha256(
        contract.get("implementation_runner_sha256"),
        f"{role}.implementation_runner_sha256",
    )
    if (
        type(contract.get("implementation_runner_size_bytes")) is not int
        or int(contract["implementation_runner_size_bytes"]) < 1
    ):
        raise QualityArtifactIntegrityError(
            f"{role}.implementation_runner_size_bytes is invalid"
        )

    quality_sha = _require_contract_sha256(
        contract.get("quality_contract_sha256"),
        f"{role}.quality_contract_sha256",
    )
    if (
        quality_sha != _quality_contract_digest(quality_contract)
        or str(quality_contract.get("quality_contract_sha256") or "")
        != quality_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed quality-contract hash mismatch"
        )
    endpoint_hash = _require_contract_sha256(
        contract.get("endpoint_contract_hash"),
        f"{role}.endpoint_contract_hash",
    )
    if str(endpoint_contract.get("endpoint_contract_hash") or "") != endpoint_hash:
        raise QualityArtifactIntegrityError(
            f"{role} generic composed endpoint binding mismatch"
        )
    precision = _require_contract_text(
        contract.get("runtime_precision_identity"),
        f"{role}.runtime_precision_identity",
    )

    completion = contract.get("candidate_execution_completion_contract")
    if not isinstance(completion, Mapping):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed completion contract is missing"
        )
    completion_value = dict(completion)
    embedded_completion_sha = _require_contract_sha256(
        completion_value.pop("contract_sha256", ""),
        f"{role}.candidate_execution_completion_contract.contract_sha256",
    )
    completion_sha = _require_contract_sha256(
        contract.get("candidate_execution_completion_contract_sha256"),
        f"{role}.candidate_execution_completion_contract_sha256",
    )
    if embedded_completion_sha != completion_sha or json_fingerprint(
        completion_value
    ) != completion_sha:
        raise QualityArtifactIntegrityError(
            f"{role} generic composed completion-contract hash mismatch"
        )
    if (
        str(completion.get("schema") or "")
        != "onnx-splitpoint/composed-candidate-completion-contract"
        or int(completion.get("schema_version") or 0) != 1
        or str(completion.get("variant") or "") != "composed"
        or str(completion.get("task") or "") != task_text
        or str(completion.get("runtime_artifact_set_sha256") or "")
        != runtime_set_sha
        or str(completion.get("completed_endpoint_contract_hash") or "")
        != endpoint_hash
        or str(completion.get("runtime_precision_identity") or "")
        != precision
        or any(
            str(completion.get(field_name) or "")
            != str(contract.get(field_name) or "")
            for field_name in (
                "eval_run_id", "model_id", "setup_id", "source_run_id",
                "case_id", "stage1_backend", "stage2_backend",
            )
        )
    ):
        raise QualityArtifactIntegrityError(
            f"{role} generic composed completion identity mismatch"
        )
    return contract, declared


def _validate_candidate_execution_contract(
    value: Any,
    *,
    role: str,
    task: str,
) -> tuple[dict[str, Any], str]:
    """Dispatch producer validation without weakening vendor-specific paths."""

    schema = value.get("schema") if isinstance(value, Mapping) else None
    if schema == "onnx-splitpoint/central-quality-producer-identity":
        return _validate_deepx_candidate_execution_contract(
            value, role=role, task=task,
        )
    if schema == "onnx-splitpoint/tensorrt-central-quality-producer-identity":
        return _validate_tensorrt_candidate_execution_contract(
            value, role=role, task=task,
        )
    if schema == "onnx-splitpoint/generic-composed-quality-producer-identity":
        return _validate_generic_composed_candidate_execution_contract(
            value, role=role, task=task,
        )
    raise QualityArtifactIntegrityError(
        f"{role} candidate execution contract schema is unsupported"
    )


def _validate_classification_quality_contract(
    value: Any, *, role: str,
) -> tuple[dict[str, Any], str]:
    """Validate the frozen classification labels/preprocess/Top-K contract."""
    if not isinstance(value, Mapping):
        raise QualityArtifactIntegrityError(f"{role} lacks the required classification quality contract")
    contract = dict(value)
    if contract.get("schema") != "onnx-splitpoint/central-classification-quality-contract":
        raise QualityArtifactIntegrityError(f"{role} has an unsupported classification quality contract schema")
    if int(contract.get("schema_version") or 0) != 1 or str(contract.get("task") or "") != "classification":
        raise QualityArtifactIntegrityError(f"{role} classification quality contract is invalid")
    declared = _require_contract_sha256(
        contract.get("quality_contract_sha256"), f"{role}.quality_contract_sha256",
    )
    if declared != _quality_contract_digest(contract):
        raise QualityArtifactIntegrityError(f"{role} classification quality contract SHA-256 mismatch")
    model = contract.get("model")
    dataset = contract.get("dataset")
    preprocessing = contract.get("preprocessing")
    postprocessor = contract.get("postprocessor")
    quality_endpoint = contract.get("quality_record_endpoint")
    if not all(isinstance(item, Mapping) for item in (
        model, dataset, preprocessing, postprocessor, quality_endpoint,
    )):
        raise QualityArtifactIntegrityError(
            f"{role} classification contract lacks model/dataset/preprocessing/postprocessor/quality-endpoint provenance"
        )
    _require_contract_sha256(model.get("sha256"), f"{role}.model.sha256")
    for field in ("manifest_sha256", "image_ids_sha256", "ground_truth_sha256"):
        _require_contract_sha256(dataset.get(field), f"{role}.dataset.{field}")
    if int(dataset.get("image_count") or 0) < 1:
        raise QualityArtifactIntegrityError(f"{role}.dataset.image_count must be positive")
    for component_name, component in (
        ("preprocessing", preprocessing), ("postprocessor", postprocessor),
        ("quality_record_endpoint", quality_endpoint),
    ):
        identity = component.get("identity")
        if not isinstance(identity, Mapping):
            raise QualityArtifactIntegrityError(f"{role}.{component_name}.identity is missing")
        component_sha = _require_contract_sha256(
            component.get("sha256"), f"{role}.{component_name}.sha256",
        )
        if component_sha != json_fingerprint(identity):
            raise QualityArtifactIntegrityError(f"{role}.{component_name} SHA-256 mismatch")
    post_identity = postprocessor.get("identity")
    quality_endpoint_identity = quality_endpoint.get("identity")
    if not isinstance(post_identity, Mapping) or not isinstance(
        quality_endpoint_identity, Mapping,
    ):
        raise QualityArtifactIntegrityError(f"{role}.postprocessor identity is missing")
    postprocessor_sha = _require_contract_sha256(
        postprocessor.get("sha256"), f"{role}.postprocessor.sha256",
    )
    postprocessor_runner_sha = _require_contract_sha256(
        post_identity.get("implementation_runner_sha256"),
        f"{role}.postprocessor.implementation_runner_sha256",
    )
    quality_endpoint_sha = _require_contract_sha256(
        quality_endpoint.get("sha256"),
        f"{role}.quality_record_endpoint.sha256",
    )
    _require_contract_sha256(
        quality_endpoint_identity.get(
            "vendored_endpoint_attestor_sha256"
        ),
        f"{role}.quality_record_endpoint.vendored_endpoint_attestor_sha256",
    )
    if (
        str(quality_endpoint_identity.get("schema") or "")
        != (
            "onnx-splitpoint/"
            "classification-quality-record-endpoint-contract"
        )
        or int(quality_endpoint_identity.get("schema_version") or 0) != 1
        or str(quality_endpoint_identity.get(
            "canonical_record_endpoint"
        ) or "") != "classification_topk_hits"
        or _require_contract_sha256(
            quality_endpoint_identity.get(
                "postprocessor_contract_sha256"
            ),
            f"{role}.quality_record_endpoint.postprocessor_contract_sha256",
        ) != postprocessor_sha
        or _require_contract_sha256(
            quality_endpoint_identity.get("implementation_runner_sha256"),
            f"{role}.quality_record_endpoint.implementation_runner_sha256",
        ) != postprocessor_runner_sha
        or _require_contract_sha256(
            contract.get("quality_record_endpoint_contract_sha256"),
            f"{role}.quality_record_endpoint_contract_sha256",
        ) != quality_endpoint_sha
    ):
        raise QualityArtifactIntegrityError(
            f"{role} classification quality-record endpoint binding is inconsistent"
        )
    if str(post_identity.get("canonical_record_endpoint") or "") != "classification_topk_hits":
        raise QualityArtifactIntegrityError(f"{role} postprocessor endpoint is not classification_topk_hits")
    if str(contract.get("canonical_record_endpoint") or "") != "classification_topk_hits":
        raise QualityArtifactIntegrityError(f"{role} canonical classification endpoint is missing")
    if str(contract.get("contract_scope") or "") != "canonical_quality_record_semantics":
        raise QualityArtifactIntegrityError(f"{role} classification quality contract scope is ambiguous")
    return contract, declared


def quality_request_from_manifest(
    request_file: str | Path,
    *,
    verify_artifacts: bool = True,
    reference_artifact: Optional[str | Path | Mapping[str, Any]] = None,
    reference_artifact_bytes: Optional[bytes] = None,
) -> QualityEvaluationRequest:
    """Load the portable runner export into a central evaluation request.

    File SHA-256 values are checked exactly once here, at the artifact boundary.
    ``reference_artifact`` supplies the once-per-model reference generated on
    the management node and deliberately overrides the remote request's local
    reference descriptor.  Active workflow callers pass a descriptor mapping
    containing the status-bound path, SHA-256 and byte size together with the
    exact bytes admitted through the runner's single ``O_NOFOLLOW`` read; that
    identity is rechecked during this actual load without reopening the path.
    A bare path remains accepted for compatibility with direct historical
    analysis callers.  The bootstrap worker only receives already-loaded
    records and never hashes a file or prediction inside its resampling loop.
    """

    request_path = Path(request_file).expanduser().resolve()
    try:
        manifest = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QualityArtifactIntegrityError(f"central quality request is not valid JSON: {request_path}") from exc
    if not isinstance(manifest, Mapping):
        raise QualityArtifactIntegrityError("central quality request root must be an object")
    if manifest.get("schema") != "onnx-splitpoint/central-quality-evaluation-request":
        raise QualityArtifactIntegrityError("unsupported central quality request schema")
    if int(manifest.get("schema_version") or 0) != 1:
        raise QualityArtifactIntegrityError("unsupported central quality request schema version")
    if str(manifest.get("pairing_key") or "") != "image_id":
        raise QualityArtifactIntegrityError("central quality requests must pair exclusively by image_id")
    if str(manifest.get("execution_location") or "") not in {"management_node", "central_management"}:
        raise QualityArtifactIntegrityError("central quality request is not assigned to the management node")
    reference_descriptor = manifest.get("reference")
    candidate_descriptor = manifest.get("candidate")
    if not isinstance(reference_descriptor, Mapping) or not isinstance(candidate_descriptor, Mapping):
        raise QualityArtifactIntegrityError("central quality request lacks reference/candidate descriptors")
    if reference_artifact is None:
        if reference_artifact_bytes is not None:
            raise QualityArtifactIntegrityError(
                "management reference bytes require a bound descriptor mapping"
            )
        reference_payload, reference_file_sha, _ = _load_quality_artifact(
            reference_descriptor, request_path, verify=verify_artifacts
        )
    elif isinstance(reference_artifact, Mapping):
        management_descriptor = dict(reference_artifact)
        if type(management_descriptor.get("size_bytes")) is not int or int(
            management_descriptor.get("size_bytes") or 0
        ) <= 0:
            raise QualityArtifactIntegrityError(
                "management reference descriptor has no positive byte-size binding"
            )
        _require_contract_sha256(
            management_descriptor.get("sha256"),
            "management reference descriptor.sha256",
        )
        if reference_artifact_bytes is None:
            # Compatibility for direct descriptor callers.  The active
            # workflow always supplies the already-admitted byte handoff.
            reference_payload, reference_file_sha, _ = _load_quality_artifact(
                management_descriptor,
                request_path,
                verify=True,
            )
        elif not isinstance(reference_artifact_bytes, bytes):
            raise QualityArtifactIntegrityError(
                "management reference byte handoff must be immutable bytes"
            )
        else:
            reference_payload, reference_file_sha, _ = (
                _load_bound_quality_artifact_bytes(
                    management_descriptor,
                    request_path,
                    reference_artifact_bytes,
                )
            )
    else:
        if reference_artifact_bytes is not None:
            raise QualityArtifactIntegrityError(
                "management reference bytes require a bound descriptor mapping"
            )
        # The canonical reference generated once on the management node must
        # replace the per-runner placeholder/reference descriptor.  Its content
        # is still fingerprinted once and validated against task/Image IDs.
        # New active-workflow code does not use this compatibility branch.
        reference_payload, reference_file_sha, _ = _load_quality_artifact(
            {"path": str(Path(reference_artifact).expanduser().resolve())},
            request_path,
            verify=False,
        )
    candidate_payload, _candidate_file_sha, _ = _load_quality_artifact(
        candidate_descriptor, request_path, verify=verify_artifacts
    )
    task = str(manifest.get("task") or "").strip().lower()
    variant = str(manifest.get("variant") or "").strip().lower()
    if task not in {"classification", "detection"}:
        raise QualityArtifactIntegrityError(f"unsupported central quality task: {task!r}")
    if reference_payload.get("schema") != "onnx-splitpoint/task-quality-reference-input":
        raise QualityArtifactIntegrityError("invalid central quality reference artifact schema")
    if candidate_payload.get("schema") != "onnx-splitpoint/task-quality-candidate-input":
        raise QualityArtifactIntegrityError("invalid central quality candidate artifact schema")
    if int(reference_payload.get("schema_version") or 0) != 1 or int(candidate_payload.get("schema_version") or 0) != 1:
        raise QualityArtifactIntegrityError("unsupported central quality artifact schema version")
    reference_role = str(reference_payload.get("reference_role") or "canonical_cpu_ort").strip().lower()
    if reference_role not in {"canonical_cpu_ort", "onnxruntime_cpu", "canonical_full_onnx"}:
        raise QualityArtifactIntegrityError("canonical quality reference is not an ONNX Runtime CPU reference")
    if reference_payload.get("semantic_reference_only") is False:
        raise QualityArtifactIntegrityError("canonical CPU reference must be marked semantic_reference_only")
    if str(reference_payload.get("pairing_key") or "") != "image_id" or str(candidate_payload.get("pairing_key") or "") != "image_id":
        raise QualityArtifactIntegrityError("quality artifacts must pair exclusively by image_id")
    if str(reference_payload.get("task") or "") != task or str(candidate_payload.get("task") or "") != task:
        raise QualityArtifactIntegrityError("task mismatch between request and quality artifacts")
    if str(candidate_payload.get("variant") or "").strip().lower() != variant:
        raise QualityArtifactIntegrityError("variant mismatch between request and candidate artifact")
    native_split_binding: Optional[dict[str, Any]] = None
    split_binding_present = any(
        payload.get("native_split_quality_binding") is not None
        or payload.get("native_split_quality_binding_sha256") not in (None, "")
        or payload.get("native_split_quality_binding_required") is True
        for payload in (manifest, candidate_payload)
    )
    if split_binding_present:
        request_binding = manifest.get("native_split_quality_binding")
        candidate_binding = candidate_payload.get("native_split_quality_binding")
        if (
            manifest.get("native_split_quality_binding_required") is not True
            or candidate_payload.get("native_split_quality_binding_required") is not True
            or not isinstance(request_binding, Mapping)
            or not isinstance(candidate_binding, Mapping)
            or dict(request_binding) != dict(candidate_binding)
        ):
            raise QualityArtifactIntegrityError(
                "request and candidate use different Native split quality bindings"
            )
        expected_split_identity = {
            "model": manifest.get("model_id") or candidate_payload.get("model_id"),
            "case": manifest.get("case_id") or candidate_payload.get("case_id"),
            "setup_id": (
                manifest.get("setup_id") or manifest.get("source_setup_id")
                or candidate_payload.get("setup_id")
                or candidate_payload.get("source_setup_id")
            ),
            "backend": (
                manifest.get("source_run_id") or manifest.get("backend")
                or candidate_payload.get("source_run_id")
                or candidate_payload.get("backend")
            ),
            "task": task,
            "precision": (
                manifest.get("runtime_precision_identity")
                or candidate_payload.get("runtime_precision_identity")
            ),
        }
        native_split_binding, split_binding_status = (
            validate_native_split_quality_binding(
                request_binding,
                expected_identity=expected_split_identity,
            )
        )
        if native_split_binding is None:
            raise QualityArtifactIntegrityError(
                f"Native split quality binding is invalid: {split_binding_status}"
            )
        binding_sha = str(
            native_split_binding.get("binding_sha256") or ""
        ).strip().lower()
        declared_binding_shas = {
            _require_contract_sha256(
                manifest.get("native_split_quality_binding_sha256"),
                "request.native_split_quality_binding_sha256",
            ),
            _require_contract_sha256(
                candidate_payload.get("native_split_quality_binding_sha256"),
                "candidate.native_split_quality_binding_sha256",
            ),
            binding_sha,
        }
        if len(declared_binding_shas) != 1:
            raise QualityArtifactIntegrityError(
                "Native split quality binding SHA-256 duplicates differ"
            )
    request_completion = manifest.get("candidate_execution_completion_contract")
    candidate_completion = candidate_payload.get(
        "candidate_execution_completion_contract"
    )
    request_completion_sha = str(
        manifest.get("candidate_execution_completion_contract_sha256") or ""
    ).strip().lower()
    candidate_completion_sha = str(
        candidate_payload.get(
            "candidate_execution_completion_contract_sha256"
        ) or ""
    ).strip().lower()
    completion_present = bool(
        request_completion or candidate_completion
        or request_completion_sha or candidate_completion_sha
    )
    if completion_present:
        if (
            not isinstance(request_completion, Mapping)
            or not isinstance(candidate_completion, Mapping)
            or dict(request_completion) != dict(candidate_completion)
        ):
            raise QualityArtifactIntegrityError(
                "request and candidate use different execution completion contracts"
            )
        completion_payload = dict(request_completion)
        embedded_sha = str(
            completion_payload.pop("contract_sha256", "") or ""
        ).strip().lower()
        observed_sha = json_fingerprint(completion_payload)
        if (
            _require_contract_sha256(
                request_completion_sha,
                "request.candidate_execution_completion_contract_sha256",
            ) != observed_sha
            or _require_contract_sha256(
                candidate_completion_sha,
                "candidate.candidate_execution_completion_contract_sha256",
            ) != observed_sha
            or _require_contract_sha256(
                embedded_sha,
                "candidate execution completion contract.contract_sha256",
            ) != observed_sha
        ):
            raise QualityArtifactIntegrityError(
                "candidate execution completion contract hash mismatch"
            )
    completed_task_fields = (
        "completed_task_endpoint_contract",
        "completed_task_endpoint_contract_hash",
        "completed_task_output_endpoint_id",
        "completed_task_endpoint_attestation",
        "completed_task_endpoint_attestation_sha256",
        "quality_join_endpoint",
    )
    completed_task_present = any(
        field in payload
        for payload in (manifest, candidate_payload)
        for field in completed_task_fields
    )
    if completed_task_present:
        if task != "detection":
            raise QualityArtifactIntegrityError(
                "completed-task endpoint binding requires detection"
            )
        for field in completed_task_fields:
            if (
                field not in manifest
                or field not in candidate_payload
                or manifest.get(field) != candidate_payload.get(field)
            ):
                raise QualityArtifactIntegrityError(
                    "request and candidate use different completed-task "
                    f"endpoint bindings: {field}"
                )
        try:
            from onnx_splitpoint_tool.native_detection_postprocess import (
                verify_completed_detection_comparison_endpoint_contract,
            )

            completed_task_contract = (
                verify_completed_detection_comparison_endpoint_contract(
                    manifest.get("completed_task_endpoint_contract")
                )
            )
        except Exception as exc:
            raise QualityArtifactIntegrityError(
                "completed-task endpoint contract is invalid"
            ) from exc
        completed_task_attestation = manifest.get(
            "completed_task_endpoint_attestation"
        )
        if (
            _require_contract_sha256(
                manifest.get(
                    "completed_task_endpoint_contract_hash"
                ),
                "request.completed_task_endpoint_contract_hash",
            )
            != str(
                completed_task_contract.get("endpoint_contract_hash")
                or ""
            )
            or str(
                manifest.get("completed_task_output_endpoint_id") or ""
            )
            != str(completed_task_contract.get("output_endpoint_id") or "")
            or str(manifest.get("quality_join_endpoint") or "")
            != "completed_task_decoded_nms"
            or not isinstance(completed_task_attestation, Mapping)
            or completed_task_attestation.get("attested") is not True
            or str(
                completed_task_attestation.get("status") or ""
            ).strip().lower() != "passed"
            or _require_contract_sha256(
                manifest.get(
                    "completed_task_endpoint_attestation_sha256"
                ),
                "request.completed_task_endpoint_attestation_sha256",
            )
            != json_fingerprint(completed_task_attestation)
        ):
            raise QualityArtifactIntegrityError(
                "completed-task endpoint binding is inconsistent"
            )
    producer_provenance_required = bool(
        manifest.get("producer_provenance_required")
        or candidate_payload.get("producer_provenance_required")
        or (isinstance(manifest.get("producer_identity"), Mapping) and manifest.get("producer_identity"))
        or (isinstance(candidate_payload.get("producer_identity"), Mapping) and candidate_payload.get("producer_identity"))
    )
    provenance_required = bool(
        manifest.get("provenance_required")
        or reference_payload.get("provenance_required")
        or candidate_payload.get("provenance_required")
        or (isinstance(manifest.get("quality_contract"), Mapping) and manifest.get("quality_contract"))
        or (isinstance(reference_payload.get("quality_contract"), Mapping) and reference_payload.get("quality_contract"))
        or (isinstance(candidate_payload.get("quality_contract"), Mapping) and candidate_payload.get("quality_contract"))
    )
    detection_quality_contract: Optional[dict[str, Any]] = None
    classification_quality_contract: Optional[dict[str, Any]] = None
    producer_execution_contract: Optional[dict[str, Any]] = None
    producer_execution_schema = ""
    if producer_provenance_required:
        request_producer, request_producer_sha = _validate_candidate_execution_contract(
            manifest.get("producer_identity"), role="request", task=task,
        )
        candidate_producer, candidate_producer_sha = _validate_candidate_execution_contract(
            candidate_payload.get("producer_identity"), role="candidate", task=task,
        )
        declared_producer_shas = {
            _require_contract_sha256(
                manifest.get("producer_identity_sha256"), "request.producer_identity_sha256"
            ),
            _require_contract_sha256(
                candidate_payload.get("producer_identity_sha256"), "candidate.producer_identity_sha256"
            ),
            request_producer_sha,
            candidate_producer_sha,
        }
        if (
            len(declared_producer_shas) != 1
            or request_producer != candidate_producer
            or manifest.get("producer_provenance_required") is not True
            or candidate_payload.get("producer_provenance_required") is not True
        ):
            raise QualityArtifactIntegrityError(
                "request and candidate use different candidate execution contracts"
            )
        producer_execution_schema = str(request_producer.get("schema") or "")
        # The portable request intentionally repeats the scientific join keys
        # outside the signed producer object so the workflow can index them
        # without understanding a vendor-specific contract.  Those duplicates
        # are not independent evidence: require exact equality here, including
        # explicit empty decoder/NMS hashes for classification.  Otherwise a
        # valid nested producer could silently coexist with stale or tampered
        # top-level values and be indexed under the wrong identity.
        duplicate_hash_fields = (
            "quality_contract_sha256",
            "preprocessing_contract_sha256",
            "decoder_contract_sha256",
            "nms_contract_sha256",
            "quality_record_endpoint_contract_sha256",
        )
        for duplicate_role, payload in (
            ("request", manifest), ("candidate", candidate_payload),
        ):
            if payload.get("quality_contract") != request_producer.get("quality_contract"):
                raise QualityArtifactIntegrityError(
                    f"{duplicate_role} top-level quality contract differs from producer identity"
                )
            for field_name in duplicate_hash_fields:
                if field_name not in payload:
                    raise QualityArtifactIntegrityError(
                        f"{duplicate_role} lacks duplicated producer binding {field_name}"
                    )
                expected = str(request_producer.get(field_name) or "").strip().lower()
                observed = str(payload.get(field_name) or "").strip().lower()
                if observed != expected:
                    raise QualityArtifactIntegrityError(
                        f"{duplicate_role} top-level {field_name} differs from producer identity"
                    )
            if (
                producer_execution_schema
                == "onnx-splitpoint/central-quality-producer-identity"
            ):
                for field_name in (
                    "prepared_input_evidence_sha256",
                    "prepared_input_join_binding",
                    "prepared_input_join_binding_sha256",
                ):
                    if field_name not in payload:
                        raise QualityArtifactIntegrityError(
                            f"{duplicate_role} lacks duplicated producer "
                            f"binding {field_name}"
                        )
                    if payload.get(field_name) != request_producer.get(
                        field_name
                    ):
                        raise QualityArtifactIntegrityError(
                            f"{duplicate_role} top-level {field_name} "
                            "differs from producer identity"
                        )
            completed_join_fields = (
                "completed_task_endpoint_contract",
                "completed_task_endpoint_contract_hash",
                "completed_task_output_endpoint_id",
                "completed_task_endpoint_attestation",
                "completed_task_endpoint_attestation_sha256",
                "quality_join_endpoint",
            )
            completed_in_producer = any(
                field_name in request_producer
                for field_name in completed_join_fields
            )
            if completed_in_producer:
                for field_name in completed_join_fields:
                    if payload.get(field_name) != request_producer.get(
                        field_name
                    ):
                        raise QualityArtifactIntegrityError(
                            f"{duplicate_role} top-level {field_name} "
                            "differs from producer identity"
                        )
        endpoint_precision_payloads = [("request", manifest)]
        if producer_execution_schema in {
            "onnx-splitpoint/tensorrt-central-quality-producer-identity",
            "onnx-splitpoint/generic-composed-quality-producer-identity",
        }:
            endpoint_precision_payloads.append(("candidate", candidate_payload))
        for duplicate_role, payload in endpoint_precision_payloads:
            for field_name in ("endpoint_contract_hash", "runtime_precision_identity"):
                if field_name not in payload:
                    raise QualityArtifactIntegrityError(
                        f"{duplicate_role} lacks duplicated producer binding {field_name}"
                    )
                expected = str(request_producer.get(field_name) or "").strip().lower()
                observed = str(payload.get(field_name) or "").strip().lower()
                if observed != expected:
                    raise QualityArtifactIntegrityError(
                        f"{duplicate_role} top-level {field_name} differs from producer identity"
                    )
        if producer_execution_schema == "onnx-splitpoint/tensorrt-central-quality-producer-identity":
            if _require_contract_sha256(
                manifest.get("policy_sha256"), "request.policy_sha256",
            ) != _require_contract_sha256(
                request_producer.get("policy_sha256"),
                "request producer.policy_sha256",
            ):
                raise QualityArtifactIntegrityError(
                    "request policy SHA-256 differs from TensorRT producer identity"
                )
            source_onnx = request_producer.get("source_onnx")
            build_onnx = request_producer.get("build_onnx")
            engine = request_producer.get("engine")
            trtexec = request_producer.get("trtexec")
            receipt_binding = request_producer.get("engine_build_receipt")
            receipt = (
                receipt_binding.get("receipt")
                if isinstance(receipt_binding, Mapping) else {}
            )
            if not all(isinstance(item, Mapping) for item in (
                source_onnx, build_onnx, engine, trtexec, receipt_binding, receipt,
            )):
                raise QualityArtifactIntegrityError(
                    "TensorRT producer artifact duplicate bindings are incomplete"
                )
            trt_duplicate_bindings = {
                "eval_run_id": request_producer.get("eval_run_id"),
                "model_id": request_producer.get("model_id"),
                "setup_id": request_producer.get("setup_id"),
                "source_run_id": request_producer.get("source_run_id"),
                "execution_role": request_producer.get("execution_role"),
                "backend": request_producer.get("backend"),
                "variant": request_producer.get("variant"),
                "case_id": request_producer.get("case_id"),
                "task": request_producer.get("task"),
                "source_model_sha256": source_onnx.get("sha256"),
                "build_onnx_sha256": build_onnx.get("sha256"),
                "runtime_artifact_sha256": engine.get("sha256"),
                "trtexec_sha256": trtexec.get("sha256"),
                "engine_build_receipt_sha256": receipt_binding.get("sha256"),
                "engine_build_receipt_file_sha256": request_producer.get(
                    "engine_build_receipt_file_sha256"
                ),
                "trt_engine_build_receipt_sha256": receipt.get("receipt_sha256"),
            }
            for duplicate_role, payload in (
                ("request", manifest), ("candidate", candidate_payload),
            ):
                for field_name, expected_value in trt_duplicate_bindings.items():
                    if field_name not in payload:
                        raise QualityArtifactIntegrityError(
                            f"{duplicate_role} lacks duplicated TensorRT producer binding {field_name}"
                        )
                    expected = str(expected_value or "")
                    observed = str(payload.get(field_name) or "")
                    if observed != expected:
                        raise QualityArtifactIntegrityError(
                            f"{duplicate_role} top-level {field_name} differs from TensorRT producer identity"
                        )
        if (
            producer_execution_schema
            == "onnx-splitpoint/generic-composed-quality-producer-identity"
        ):
            producer_model = request_producer.get("model")
            if not isinstance(producer_model, Mapping):
                raise QualityArtifactIntegrityError(
                    "generic composed producer model binding is incomplete"
                )
            generic_duplicate_bindings = {
                "artifact_provenance_contract_version": 1,
                "eval_run_id": request_producer.get("eval_run_id"),
                "model_id": request_producer.get("model_id"),
                "setup_id": request_producer.get("setup_id"),
                "source_run_id": request_producer.get("source_run_id"),
                "case_id": request_producer.get("case_id"),
                "backend": request_producer.get("backend"),
                "execution_role": request_producer.get("execution_role"),
                "performance_claims_emitted": False,
                "source_model_sha256": producer_model.get(
                    "source_onnx_sha256"
                ),
                "part1_model_sha256": producer_model.get(
                    "part1_onnx_sha256"
                ),
                "part2_model_sha256": producer_model.get(
                    "part2_onnx_sha256"
                ),
                "runtime_artifact_set_sha256": request_producer.get(
                    "runtime_artifact_set_sha256"
                ),
                "candidate_execution_completion_contract_sha256": (
                    request_producer.get(
                        "candidate_execution_completion_contract_sha256"
                    )
                ),
            }
            for duplicate_role, payload in (
                ("request", manifest), ("candidate", candidate_payload),
            ):
                if payload.get("endpoint_contract") != request_producer.get(
                    "endpoint_contract"
                ):
                    raise QualityArtifactIntegrityError(
                        f"{duplicate_role} top-level endpoint contract differs "
                        "from generic composed producer identity"
                    )
                if payload.get(
                    "candidate_execution_completion_contract"
                ) != request_producer.get(
                    "candidate_execution_completion_contract"
                ):
                    raise QualityArtifactIntegrityError(
                        f"{duplicate_role} top-level completion contract differs "
                        "from generic composed producer identity"
                    )
                for field_name, expected_value in (
                    generic_duplicate_bindings.items()
                ):
                    if field_name not in payload or payload.get(
                        field_name
                    ) != expected_value:
                        raise QualityArtifactIntegrityError(
                            f"{duplicate_role} top-level {field_name} differs "
                            "from generic composed producer identity"
                        )
        producer_execution_contract = request_producer
        if task == "detection":
            # Validate the CPU adapter contract on its own terms.  Its decoder
            # and NMS implementation are intentionally not equal to DeepX.
            reference_contract, reference_contract_sha = _validate_detection_quality_contract(
                reference_payload.get("quality_contract"), role="management reference"
            )
            if _require_contract_sha256(
                reference_payload.get("quality_contract_sha256"),
                "management reference.quality_contract_sha256",
            ) != reference_contract_sha:
                raise QualityArtifactIntegrityError(
                    "management reference detection contract binding is inconsistent"
                )
            detection_quality_contract = reference_contract
        else:
            reference_contract, reference_contract_sha = _validate_classification_quality_contract(
                reference_payload.get("quality_contract"), role="management reference"
            )
            if _require_contract_sha256(
                reference_payload.get("quality_contract_sha256"),
                "management reference.quality_contract_sha256",
            ) != reference_contract_sha:
                raise QualityArtifactIntegrityError(
                    "management reference classification contract binding is inconsistent"
                )
            classification_quality_contract = reference_contract
        if (
            producer_execution_schema in {
                "onnx-splitpoint/tensorrt-central-quality-producer-identity",
                "onnx-splitpoint/generic-composed-quality-producer-identity",
            }
            and _require_contract_sha256(
                reference_descriptor.get("quality_contract_sha256"),
                "reference descriptor.quality_contract_sha256",
            ) != reference_contract_sha
        ):
            raise QualityArtifactIntegrityError(
                "reference descriptor quality contract differs from TensorRT producer identity"
            )
    elif provenance_required:
        validator = (
            _validate_detection_quality_contract
            if task == "detection" else _validate_classification_quality_contract
        )
        manifest_contract, manifest_contract_sha = validator(
            manifest.get("quality_contract"), role="request",
        )
        reference_contract, reference_contract_sha = validator(
            reference_payload.get("quality_contract"), role="management reference",
        )
        candidate_contract, candidate_contract_sha = validator(
            candidate_payload.get("quality_contract"), role="candidate",
        )
        declared_shas = {
            _require_contract_sha256(manifest.get("quality_contract_sha256"), "request.quality_contract_sha256"),
            _require_contract_sha256(reference_payload.get("quality_contract_sha256"), "management reference.quality_contract_sha256"),
            _require_contract_sha256(candidate_payload.get("quality_contract_sha256"), "candidate.quality_contract_sha256"),
            _require_contract_sha256(reference_descriptor.get("quality_contract_sha256"), "reference descriptor.quality_contract_sha256"),
            manifest_contract_sha,
            reference_contract_sha,
            candidate_contract_sha,
        }
        if len(declared_shas) != 1 or manifest_contract != reference_contract or manifest_contract != candidate_contract:
            raise QualityArtifactIntegrityError(
                "request, candidate and management reference use different quality contracts"
            )
        if task == "detection":
            detection_quality_contract = manifest_contract
        else:
            classification_quality_contract = manifest_contract
    reference_records = reference_payload.get("records")
    candidate_records = candidate_payload.get("records")
    if not isinstance(reference_records, list) or not isinstance(candidate_records, list):
        raise QualityArtifactIntegrityError("quality artifacts must contain a records array")
    if manifest.get("reference_record_count") is not None and int(manifest["reference_record_count"]) != len(reference_records):
        raise QualityArtifactIntegrityError("reference record count does not match request")
    if manifest.get("record_count") is not None and int(manifest["record_count"]) != len(candidate_records):
        raise QualityArtifactIntegrityError("candidate record count does not match request")
    # Fail at load time, not hours into a campaign worker.  Rebind to the
    # canonical Image-ID order so annotations and both prediction streams use
    # exactly the same positional contract in every evaluator.
    paired = pair_prediction_records(reference_records, candidate_records, image_id_field="image_id")
    reference_records = [dict(record) for record in paired.reference]
    candidate_records = [dict(record) for record in paired.candidate]
    if detection_quality_contract is not None:
        dataset_contract = detection_quality_contract.get("dataset")
        if not isinstance(dataset_contract, Mapping):
            raise QualityArtifactIntegrityError("detection quality dataset contract is missing")
        observed_contract_ids_sha = image_ids_fingerprint(list(paired.image_ids))
        declared_contract_ids_sha = _require_contract_sha256(
            dataset_contract.get("image_ids_sha256"),
            "quality_contract.dataset.image_ids_sha256",
        )
        if observed_contract_ids_sha != declared_contract_ids_sha:
            raise QualityArtifactIntegrityError(
                "quality records violate the frozen dataset Image-ID contract"
            )
        if int(dataset_contract.get("image_count") or 0) != len(paired.image_ids):
            raise QualityArtifactIntegrityError(
                "quality records violate the frozen dataset cardinality contract"
            )
        ground_truth_identity: list[dict[str, Any]] = []
        for reference_record, candidate_record in zip(reference_records, candidate_records):
            reference_ground_truth = reference_record.get("ground_truth")
            candidate_ground_truth = candidate_record.get("ground_truth")
            if not isinstance(reference_ground_truth, list) or not isinstance(candidate_ground_truth, list):
                raise QualityArtifactIntegrityError(
                    "detection quality records lack frozen ground truth"
                )
            if canonical_json(reference_ground_truth) != canonical_json(candidate_ground_truth):
                raise QualityArtifactIntegrityError(
                    "candidate and management reference ground truth differ"
                )
            ground_truth_identity.append({
                "image_id": candidate_record.get("image_id"),
                "ground_truth": candidate_ground_truth,
            })
        observed_ground_truth_sha = json_fingerprint(ground_truth_identity)
        declared_ground_truth_sha = _require_contract_sha256(
            dataset_contract.get("ground_truth_sha256"),
            "quality_contract.dataset.ground_truth_sha256",
        )
        if observed_ground_truth_sha != declared_ground_truth_sha:
            raise QualityArtifactIntegrityError(
                "quality records violate the frozen ground-truth contract"
            )
    if classification_quality_contract is not None:
        dataset_contract = classification_quality_contract.get("dataset")
        if not isinstance(dataset_contract, Mapping):
            raise QualityArtifactIntegrityError("classification quality dataset contract is missing")
        observed_ids_sha = image_ids_fingerprint(list(paired.image_ids))
        if observed_ids_sha != _require_contract_sha256(
            dataset_contract.get("image_ids_sha256"),
            "classification_quality_contract.dataset.image_ids_sha256",
        ):
            raise QualityArtifactIntegrityError(
                "classification quality records violate the frozen Image-ID contract"
            )
        if int(dataset_contract.get("image_count") or 0) != len(paired.image_ids):
            raise QualityArtifactIntegrityError(
                "classification quality records violate the frozen dataset cardinality contract"
            )
        label_identity: list[dict[str, Any]] = []
        legacy_label_identity: list[dict[str, Any]] = []
        legacy_reference_label_identity: list[dict[str, Any]] = []
        for reference_record, candidate_record in zip(reference_records, candidate_records):
            try:
                reference_label_id = int(reference_record.get("label_id"))
                candidate_label_id = int(candidate_record.get("label_id"))
            except (TypeError, ValueError) as exc:
                raise QualityArtifactIntegrityError(
                    "classification quality record lacks an integer label_id"
                ) from exc
            if reference_label_id != candidate_label_id:
                raise QualityArtifactIntegrityError(
                    "classification candidate/reference label_id differs"
                )
            label_identity.append({
                "image_id": str(candidate_record.get("image_id") or ""),
                "label_id": candidate_label_id,
            })
            legacy_label_identity.append({
                "image_id": str(candidate_record.get("image_id") or ""),
                "label_id": candidate_label_id,
                "label_name": candidate_record.get("label_name"),
            })
            legacy_reference_label_identity.append({
                "image_id": str(reference_record.get("image_id") or ""),
                "label_id": reference_label_id,
                "label_name": reference_record.get("label_name"),
            })
        declared_label_sha = _require_contract_sha256(
            dataset_contract.get("ground_truth_sha256"),
            "classification_quality_contract.dataset.ground_truth_sha256",
        )
        accepted_label_shas = {
            json_fingerprint(sorted(label_identity, key=lambda item: item["image_id"])),
            # Compatibility with 2.70 and older generated suites.  Names are
            # presentation metadata only, but their old self-hash remains
            # verifiable while cross-producer identity is label_id-only.
            json_fingerprint(sorted(legacy_label_identity, key=lambda item: item["image_id"])),
            json_fingerprint(sorted(legacy_reference_label_identity, key=lambda item: item["image_id"])),
        }
        if declared_label_sha not in accepted_label_shas:
            raise QualityArtifactIntegrityError(
                "classification quality records violate the frozen label contract"
            )
    if producer_execution_contract is not None:
        producer_is_deepx = (
            producer_execution_schema
            == "onnx-splitpoint/central-quality-producer-identity"
        )
        producer_is_tensorrt = (
            producer_execution_schema
            == "onnx-splitpoint/tensorrt-central-quality-producer-identity"
        )
        producer_dataset = producer_execution_contract.get("dataset")
        producer_preprocessing = producer_execution_contract.get("preprocessing")
        producer_model = producer_execution_contract.get("model")
        if not all(isinstance(item, Mapping) for item in (producer_dataset, producer_preprocessing, producer_model)):
            raise QualityArtifactIntegrityError("candidate execution binding is incomplete")
        observed_ids_sha = image_ids_fingerprint(list(paired.image_ids))
        if observed_ids_sha != _require_contract_sha256(
            producer_dataset.get("image_ids_sha256"),
            "producer_identity.dataset.image_ids_sha256",
        ):
            raise QualityArtifactIntegrityError("candidate records violate producer Image-ID binding")
        if int(producer_dataset.get("image_count") or 0) != len(paired.image_ids):
            raise QualityArtifactIntegrityError("candidate records violate producer cardinality binding")
        producer_ground_truth_identity: list[dict[str, Any]] = []
        for candidate_record in candidate_records:
            if task == "detection":
                ground_truth = candidate_record.get("ground_truth")
                predictions = candidate_record.get("candidate")
                if not isinstance(ground_truth, list) or not isinstance(predictions, list):
                    raise QualityArtifactIntegrityError("DeepX detection candidate lacks GT/prediction records")
                for detection_index, detection in enumerate(predictions):
                    if not isinstance(detection, Mapping):
                        raise QualityArtifactIntegrityError("DeepX detection candidate contains a non-object detection")
                    numeric: dict[str, float] = {}
                    for field_name in ("x1", "y1", "x2", "y2", "score", "class_id"):
                        try:
                            numeric[field_name] = float(detection.get(field_name))
                        except (TypeError, ValueError) as exc:
                            raise QualityArtifactIntegrityError(
                                f"DeepX detection {detection_index} lacks numeric {field_name}"
                            ) from exc
                        if not np.isfinite(numeric[field_name]):
                            raise QualityArtifactIntegrityError("DeepX detection candidate contains non-finite values")
                    if numeric["x2"] < numeric["x1"] or numeric["y2"] < numeric["y1"]:
                        raise QualityArtifactIntegrityError("DeepX detection candidate contains inverted coordinates")
                    if not 0.0 <= numeric["score"] <= 1.0 or not str(detection.get("class_name") or ""):
                        raise QualityArtifactIntegrityError("DeepX detection candidate score/class provenance is invalid")
                ground_truth_value: Any = ground_truth
            else:
                if candidate_record.get("label_id") is None:
                    raise QualityArtifactIntegrityError("DeepX classification candidate lacks frozen label")
                producer_ground_truth_identity.append({
                    "image_id": candidate_record.get("image_id"),
                    "label_id": int(candidate_record.get("label_id")),
                })
                continue
            producer_ground_truth_identity.append({
                "image_id": candidate_record.get("image_id"),
                "ground_truth": ground_truth_value,
            })
        declared_producer_gt_sha = _require_contract_sha256(
            producer_dataset.get("ground_truth_sha256"),
            "producer_identity.dataset.ground_truth_sha256",
        )
        accepted_producer_gt_shas = {json_fingerprint(producer_ground_truth_identity)}
        if task == "classification":
            accepted_producer_gt_shas.add(json_fingerprint([
                {
                    "image_id": record.get("image_id"),
                    "label_id": int(record.get("label_id")),
                    "label_name": record.get("label_name"),
                }
                for record in candidate_records
            ]))
        if declared_producer_gt_sha not in accepted_producer_gt_shas:
            raise QualityArtifactIntegrityError("candidate records violate producer GT binding")

        if producer_is_deepx:
            transforms = candidate_payload.get("per_image_transforms")
            preprocessing_identity = producer_preprocessing.get("identity")
            if not isinstance(transforms, list) or not isinstance(preprocessing_identity, Mapping):
                raise QualityArtifactIntegrityError("candidate per-image preprocessing transforms are missing")
            if len(transforms) != len(paired.image_ids):
                raise QualityArtifactIntegrityError("candidate per-image transform cardinality differs from records")
            prepared_input_evidence = producer_execution_contract.get(
                "prepared_input_evidence"
            )
            expected_transforms_sha = (
                prepared_input_evidence.get("records_sha256")
                if isinstance(prepared_input_evidence, Mapping)
                else preprocessing_identity.get(
                    "per_image_transforms_sha256"
                )
            )
            if json_fingerprint(transforms) != _require_contract_sha256(
                expected_transforms_sha,
                "producer_identity.prepared_input_evidence.records_sha256",
            ):
                raise QualityArtifactIntegrityError("candidate per-image preprocessing transform hash mismatch")
            transform_ids = [row.get("image_id") for row in transforms if isinstance(row, Mapping)]
            if image_ids_fingerprint(transform_ids) != observed_ids_sha:
                raise QualityArtifactIntegrityError("candidate per-image transforms violate the Image-ID binding")
            join_binding = producer_execution_contract.get(
                "prepared_input_join_binding"
            )
            if not isinstance(join_binding, Mapping):
                raise QualityArtifactIntegrityError(
                    "DeepX performance/quality input join binding is missing"
                )
            join_projection = {
                field: join_binding.get(field)
                for field in (
                    "source_image_id", "source_image_sha256",
                    "prepared_input_sha256", "prepared_input_bytes",
                    "prepared_input_name", "prepared_input_shape",
                    "prepared_input_dtype", "prepared_input_layout",
                    "runtime_preprocessing_sha256",
                    "runtime_numeric_input_sha256",
                )
            }
            matching_join_transforms = []
            for raw_transform in transforms:
                if not isinstance(raw_transform, Mapping):
                    continue
                tensor_binding = raw_transform.get(
                    "prepared_tensor_binding"
                )
                if not isinstance(tensor_binding, Mapping):
                    continue
                transform_projection = {
                    field: tensor_binding.get(field)
                    for field in join_projection
                }
                if (
                    str(raw_transform.get("image_id") or "")
                    == str(join_binding.get("source_image_id") or "")
                    and transform_projection == join_projection
                ):
                    matching_join_transforms.append(raw_transform)
            if len(matching_join_transforms) != 1:
                raise QualityArtifactIntegrityError(
                    "DeepX performance/quality input join is not an exact "
                    "unique member of per-image transforms"
                )

        # DeepX has a vendor-specific host adapter, so it joins to the CPU
        # reference only through shared source/dataset semantics.  TensorRT is
        # produced by the same canonical quality adapter and must therefore
        # carry the exact complete task-specific quality contract.
        reference_contract = (
            detection_quality_contract if task == "detection"
            else classification_quality_contract
        )
        reference_model = reference_contract.get("model") if reference_contract else None
        reference_dataset = reference_contract.get("dataset") if reference_contract else None
        if not isinstance(reference_model, Mapping) or not isinstance(reference_dataset, Mapping):
            raise QualityArtifactIntegrityError(
                f"management reference {task} provenance is incomplete"
            )
        if str(producer_model.get("source_onnx_sha256") or "") != str(reference_model.get("sha256") or ""):
            producer_name = "DeepX" if producer_is_deepx else "TensorRT"
            raise QualityArtifactIntegrityError(
                f"{producer_name} and CPU reference use different source ONNX models"
            )
        shared_dataset_fields = ["manifest_sha256", "image_ids_sha256"]
        if task == "detection":
            shared_dataset_fields.append("ground_truth_sha256")
        for field_name in shared_dataset_fields:
            if str(producer_dataset.get(field_name) or "") != str(reference_dataset.get(field_name) or ""):
                producer_name = "DeepX" if producer_is_deepx else "TensorRT"
                raise QualityArtifactIntegrityError(
                    f"{producer_name} and CPU reference use different dataset {field_name} bindings"
                )
        if producer_is_tensorrt:
            producer_quality_contract = producer_execution_contract.get("quality_contract")
            if (
                not isinstance(producer_quality_contract, Mapping)
                or dict(producer_quality_contract) != dict(reference_contract)
            ):
                raise QualityArtifactIntegrityError(
                    "TensorRT and CPU reference use different quality contracts"
                )
    expected_ids = reference_descriptor.get("expected_image_ids")
    if expected_ids is None:
        expected_ids = manifest.get("expected_image_ids")
    if expected_ids is not None:
        if not isinstance(expected_ids, list):
            raise QualityArtifactIntegrityError("management reference expected_image_ids must be an array")
        try:
            expected_ids_sha = image_ids_fingerprint(expected_ids)
            observed_ids_sha = image_ids_fingerprint(list(paired.image_ids))
        except QualityFingerprintError as exc:
            raise QualityArtifactIntegrityError(f"invalid management reference Image-ID contract: {exc}") from exc
        declared_ids_sha = str(
            reference_descriptor.get("expected_image_ids_sha256")
            or manifest.get("expected_image_ids_sha256")
            or ""
        ).strip().lower()
        if declared_ids_sha and declared_ids_sha != expected_ids_sha:
            raise QualityArtifactIntegrityError("management reference expected Image-ID digest is inconsistent")
        if observed_ids_sha != expected_ids_sha:
            raise QualityArtifactIntegrityError("candidate/management reference Image-ID set violates the remote contract")

    gate = dict(manifest.get("metric_gate_config") or {})
    gate["policy_sha256"] = str(manifest.get("policy_sha256") or "")
    gate["task"] = task
    gate["variant"] = variant
    if detection_quality_contract is not None:
        gate["quality_contract_sha256"] = str(
            detection_quality_contract.get("quality_contract_sha256") or ""
        )
    if producer_execution_contract is not None:
        gate["producer_identity_sha256"] = str(
            producer_execution_contract.get("producer_identity_sha256") or ""
        )
    if native_split_binding is not None:
        gate["native_split_quality_binding_sha256"] = str(
            native_split_binding.get("binding_sha256") or ""
        )
    if variant == "composed":
        if (
            producer_execution_schema
            == "onnx-splitpoint/generic-composed-quality-producer-identity"
            or native_split_binding is not None
        ):
            artifact_provenance_binding_status = "verified"
            artifact_provenance_claim_eligible = True
        else:
            # v2.79.10 and older envelopes remain loadable for diagnosis, but
            # their content hash cannot prove which physical setup/runtime
            # produced byte-identical candidates.
            artifact_provenance_binding_status = "legacy_unbound"
            artifact_provenance_claim_eligible = False
    else:
        artifact_provenance_binding_status = (
            "verified" if producer_execution_contract is not None else "not_required"
        )
        artifact_provenance_claim_eligible = bool(
            producer_execution_contract is not None
        )
    gate["artifact_provenance_binding_status"] = (
        artifact_provenance_binding_status
    )
    gate["artifact_provenance_claim_eligible"] = bool(
        artifact_provenance_claim_eligible
    )
    statistics = dict(manifest.get("statistics") or {})
    gate["statistics_method"] = str(statistics.get("method") or "paired_bootstrap")
    gate["decision_rule"] = str(statistics.get("decision") or "lower_one_sided_bound")
    try:
        margin = float(gate.get("non_inferiority_margin") or 0.01)
        repetitions = int(statistics.get("bootstrap_repetitions") or 500)
        seed = int(statistics.get("seed") or 20260710)
        confidence = float(statistics.get("confidence_level") or 0.95)
    except (TypeError, ValueError) as exc:
        raise QualityArtifactIntegrityError("invalid statistical values in central quality request") from exc
    annotations = []
    if task == "classification":
        for record_index, (reference_record, candidate_record) in enumerate(zip(reference_records, candidate_records)):
            remote_label = candidate_record.get("label_id")
            reference_label = reference_record.get("label_id")
            if remote_label is not None and reference_label is not None and remote_label != reference_label:
                raise QualityArtifactIntegrityError("classification label mismatch between remote GT and management reference")
            image_id = candidate_record.get("image_id")
            for role, record, payload_field in (
                ("reference", reference_record, "reference"),
                ("candidate", candidate_record, "candidate"),
            ):
                metrics = record.get(payload_field)
                if not isinstance(metrics, Mapping):
                    raise QualityArtifactIntegrityError(
                        f"classification {role} record {record_index} ({image_id!r}) lacks {payload_field} metrics"
                    )
                for metric in ("top1_hit", "top5_hit"):
                    if not isinstance(metrics.get(metric), bool):
                        raise QualityArtifactIntegrityError(
                            f"classification {role} record {record_index} ({image_id!r}) lacks "
                            f"{payload_field}.{metric}; labeled accuracy cannot be evaluated"
                        )
            annotations.append(
                {
                    "image_id": candidate_record.get("image_id"),
                    "label_id": remote_label if remote_label is not None else reference_label,
                    "label_name": candidate_record.get("label_name") or reference_record.get("label_name"),
                }
            )
        evaluator_factory = "onnx_splitpoint_tool.quality_metrics:classification_quality_evaluator"
        algorithm_version = f"{QUALITY_ALGORITHM_VERSION}:classification-accuracy-v2"
    else:
        for reference_record, candidate_record in zip(reference_records, candidate_records):
            ground_truth = candidate_record.get("ground_truth")
            if ground_truth is None:
                ground_truth = reference_record.get("ground_truth")
            if not isinstance(ground_truth, list):
                raise QualityArtifactIntegrityError("detection request lacks remote ground truth")
            annotations.append(
                {"image_id": candidate_record.get("image_id"), "ground_truth": ground_truth}
            )
        evaluator_factory = "onnx_splitpoint_tool.quality_metrics:detection_quality_evaluator"
        algorithm_version = DETECTION_QUALITY_ALGORITHM_VERSION
    return QualityEvaluationRequest(
        reference_records=reference_records,
        candidate_records=candidate_records,
        annotations=annotations,
        metric_gate_config=gate,
        repetitions=repetitions,
        seed=seed,
        confidence_level=confidence,
        non_inferiority_margin=margin,
        evaluator_factory=evaluator_factory,
        image_id_field="image_id",
        reference_prediction_field="reference",
        candidate_prediction_field="candidate",
        algorithm_version=algorithm_version,
        reference_identity=(
            str(reference_payload.get("reference_identity") or "")
            or f"runner_reference_artifact:{reference_file_sha}"
        ),
        request_id=f"{task}:{variant}:{manifest.get('policy_sha256') or ''}",
        candidate_execution_completion_contract_sha256=(
            request_completion_sha if completion_present else ""
        ),
        artifact_provenance_binding_status=(
            artifact_provenance_binding_status
        ),
        artifact_provenance_claim_eligible=bool(
            artifact_provenance_claim_eligible
        ),
    )


def _nested_value(record: Mapping[str, Any], path: str) -> float:
    value: Any = record
    for component in str(path).split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"quality value path {path!r} is missing")
        value = value[component]
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"quality value path {path!r} is not finite")
    return numeric


class _PairedMeanEvaluator:
    def __init__(
        self,
        reference_records: Sequence[Mapping[str, Any]],
        candidate_records: Sequence[Mapping[str, Any]],
        _annotations: Any,
        config: Mapping[str, Any],
    ) -> None:
        path = str(config.get("value_field") or "value")
        self.reference = np.asarray([_nested_value(row, path) for row in reference_records], dtype=np.float64)
        self.candidate = np.asarray([_nested_value(row, path) for row in candidate_records], dtype=np.float64)

    def evaluate(self, multiplicities: np.ndarray) -> Mapping[str, float]:
        weights = np.asarray(multiplicities, dtype=np.float64).reshape(-1)
        denominator = float(np.sum(weights))
        if denominator <= 0.0 or len(weights) != len(self.reference):
            raise ValueError("invalid bootstrap multiplicities")
        reference = float(np.dot(weights, self.reference) / denominator)
        candidate = float(np.dot(weights, self.candidate) / denominator)
        return {"candidate": candidate, "reference": reference, "delta": candidate - reference}


def _load_evaluator_factory(path: str) -> Callable[..., Any]:
    if path == "paired_mean":
        return _PairedMeanEvaluator
    if ":" not in path:
        raise ValueError("custom evaluator_factory must use 'module:function' syntax")
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"quality evaluator factory is not callable: {path}")
    return factory


def _normalise_component(raw: Any, *, metric: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        out = dict(raw)
        if out.get("delta") is None and out.get("candidate") is not None and out.get("reference") is not None:
            out["delta"] = float(out["candidate"]) - float(out["reference"])
    else:
        out = {"delta": float(raw)}
    if out.get("delta") is None or not np.isfinite(float(out["delta"])):
        raise ValueError(f"quality evaluator did not return a finite delta for {metric}")
    out["metric"] = str(out.get("metric") or metric)
    out["delta"] = float(out["delta"])
    for name in ("candidate", "reference", "margin"):
        if out.get(name) is not None:
            out[name] = float(out[name])
    count_fields = ("candidate_hits", "reference_hits", "sample_count")
    present_count_fields = [name for name in count_fields if out.get(name) is not None]
    if present_count_fields:
        if len(present_count_fields) != len(count_fields):
            raise ValueError(
                f"quality evaluator returned incomplete classification hit counts for {metric}"
            )
        counts: dict[str, int] = {}
        for name in count_fields:
            raw_count = out[name]
            if isinstance(raw_count, bool):
                raise ValueError(f"quality evaluator returned a non-integral {name} for {metric}")
            try:
                numeric_count = float(raw_count)
                integer_count = int(raw_count)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"quality evaluator returned a non-integral {name} for {metric}"
                ) from exc
            if not math.isfinite(numeric_count) or numeric_count != float(integer_count):
                raise ValueError(f"quality evaluator returned a non-integral {name} for {metric}")
            counts[name] = integer_count
        sample_count = counts["sample_count"]
        if (
            sample_count <= 0
            or not 0 <= counts["candidate_hits"] <= sample_count
            or not 0 <= counts["reference_hits"] <= sample_count
        ):
            raise ValueError(f"quality evaluator returned invalid classification hit counts for {metric}")
        candidate_value = counts["candidate_hits"] / sample_count
        reference_value = counts["reference_hits"] / sample_count
        out.update(counts)
        out["candidate"] = float(candidate_value)
        out["reference"] = float(reference_value)
        # One division of the exact integer difference avoids the additional
        # rounding error from candidate/sample - reference/sample.
        out["delta"] = float(
            (counts["candidate_hits"] - counts["reference_hits"]) / sample_count
        )
    return out


def _evaluate_metric(evaluator: Any, multiplicities: np.ndarray) -> dict[str, Any]:
    if hasattr(evaluator, "evaluate") and callable(evaluator.evaluate):
        raw = evaluator.evaluate(multiplicities)
    elif callable(evaluator):
        raw = evaluator(multiplicities)
    else:
        raise TypeError("quality evaluator must be callable or expose evaluate(multiplicities)")
    if isinstance(raw, Mapping) and isinstance(raw.get("primary"), Mapping):
        primary_raw = raw.get("primary")
        guard_raw = raw.get("guardrails") if isinstance(raw.get("guardrails"), Mapping) else {}
        return {
            "primary": _normalise_component(primary_raw, metric="primary"),
            "guardrails": {
                str(name): _normalise_component(component, metric=str(name))
                for name, component in guard_raw.items()
            },
        }
    else:
        return {"primary": _normalise_component(raw, metric="primary"), "guardrails": {}}


def configured_guardrail_names(metric_gate_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return task-relevant guardrail result names required by a policy.

    Guardrail configuration keys use the ``*_margin`` spelling while result
    components use the metric spelling.  A policy that configures a guardrail
    is a contract: silently omitting that component would otherwise allow an
    incomplete evaluator or stale cache entry to report a false pass.

    Some legacy test/export helpers placed both classification and detection
    margins into one shared mapping.  The declared task/primary metric selects
    the relevant family; unknown guardrails remain required and therefore fail
    closed rather than disappearing.
    """

    gate = dict(metric_gate_config or {})
    raw = gate.get("guardrails")
    if not isinstance(raw, Mapping):
        return ()
    task = str(gate.get("task") or "").strip().lower()
    if not task:
        primary = str(gate.get("primary_metric") or "").strip().lower()
        if primary.startswith("coco_") or primary.startswith("ap"):
            task = "detection"
        elif primary.startswith("top") or "accuracy" in primary:
            task = "classification"
    required: set[str] = set()
    for configured_name in raw:
        name = str(configured_name).strip()
        if not name:
            continue
        metric = name[:-7] if name.endswith("_margin") else name
        if task == "detection" and metric.startswith("top"):
            continue
        if task == "classification" and metric.startswith("ap"):
            continue
        required.add(metric)
    return tuple(sorted(required))


def require_configured_guardrails(
    metric_gate_config: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[str, ...]:
    """Validate that every configured task guardrail is materialised."""

    required = configured_guardrail_names(metric_gate_config)
    raw = result.get("guardrails")
    observed = set(raw) if isinstance(raw, Mapping) else set()
    missing = sorted(set(required) - observed)
    if missing:
        raise ValueError(
            "quality evaluator omitted configured guardrail(s): "
            + ", ".join(missing)
            + "; evaluation fails closed"
        )
    return required


def _cached_guardrail_contract_matches_request(
    metric_gate_config: Mapping[str, Any], result: Mapping[str, Any],
) -> bool:
    """Bind a structurally valid cache row to the current gate request.

    The cache envelope can prove that every name listed by the result exists,
    but only the live request knows whether that list itself omitted a newly
    configured guardrail.  Compare the exact canonical name set before a hit
    is admitted so a forged/stale AP50-only row cannot satisfy an AP50+AP75
    fingerprint merely by claiming its shorter contract is complete.
    """

    configured = result.get("configured_guardrails")
    if not isinstance(configured, Sequence) or isinstance(
        configured, (str, bytes, bytearray),
    ):
        return False
    observed = tuple(str(name) for name in configured)
    return observed == configured_guardrail_names(metric_gate_config)


def _strictly_below_inclusive_threshold(value: float, threshold: float) -> bool:
    """Return whether *value* is materially below an inclusive threshold."""

    numeric_value = float(value)
    numeric_threshold = float(threshold)
    if not math.isfinite(numeric_value) or not math.isfinite(numeric_threshold):
        raise ValueError("quality threshold comparison requires finite values")
    if numeric_value >= numeric_threshold:
        return False
    tolerance = max(
        _INCLUSIVE_MARGIN_ABS_TOL,
        _INCLUSIVE_MARGIN_MAX_ULPS * math.ulp(numeric_value),
        _INCLUSIVE_MARGIN_MAX_ULPS * math.ulp(numeric_threshold),
    )
    return (numeric_threshold - numeric_value) > tolerance


def _point_estimate_below_margin(component: Mapping[str, Any], default_margin: float) -> bool:
    margin = float(component.get("margin", default_margin))
    if all(component.get(name) is not None for name in ("candidate_hits", "reference_hits", "sample_count")):
        sample_count = int(component["sample_count"])
        delta = (int(component["candidate_hits"]) - int(component["reference_hits"])) / sample_count
    else:
        delta = float(component["delta"])
    return _strictly_below_inclusive_threshold(float(delta), -margin)


def _decision(component: Mapping[str, Any], ci_low: float | None, margin: float) -> str:
    if _point_estimate_below_margin(component, margin):
        return "fail"
    if ci_low is None:
        return "inconclusive"
    if not _strictly_below_inclusive_threshold(float(ci_low), -float(margin)):
        return "pass"
    return "inconclusive"


def _prediction_identity_is_bound(payload: Mapping[str, Any]) -> bool:
    """Verify selected scientific payloads, not equal metrics or absent hashes.

    Called at admission/worker boundaries, never inside a bootstrap repetition.
    The selected field names are the same ones used by prepare_evaluation.
    """
    reference_sha = str(payload.get("reference_predictions_sha256") or "")
    candidate_sha = str(payload.get("candidate_predictions_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", reference_sha) or candidate_sha != reference_sha:
        return False
    try:
        return all(
            prediction_fingerprint(
                list(payload.get(f"{role}_records") or []),
                image_id_field=str(payload.get("image_id_field") or "image_id"),
                payload_field=payload.get(f"{role}_prediction_field"),
            ) == reference_sha
            for role in ("reference", "candidate")
        )
    except (QualityFingerprintError, TypeError, ValueError):
        return False


def _reporting(payload):
    from .accuracy_reporting import active_policy
    return active_policy(payload.get("metric_gate_config") or {})


def _evaluate_payload_shard(payload, plan, *, shard_index, repetition_offset):
    from .quality_statistics import WorkerProgress
    with WorkerProgress(payload, repetition_offset, repetition_offset + len(plan)) as progress:
        return _evaluate_payload_shard_impl(payload, plan, shard_index=shard_index,
            repetition_offset=repetition_offset, progress=progress)


def _evaluate_payload_shard_impl(
    payload: Mapping[str, Any],
    plan: np.ndarray,
    *,
    shard_index: int,
    repetition_offset: int,
    progress: Any,
    prepared: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate one deterministic, contiguous slice of a paired bootstrap.

    The complete PCG64 index plan is generated once by the management service
    and split without reordering.  A worker therefore never invents its own
    random stream, and concatenating shards in ``repetition_offset`` order is
    bit-for-bit equivalent to a one-process run.
    """

    started = time.perf_counter()
    statistics = dict(payload.get("_statistics") or {})
    progress.emit("preparing")
    reference = list(payload["reference_records"])
    candidate = list(payload["candidate_records"])
    n = len(reference)
    config = {
        "metric_gate_config": dict(payload.get("metric_gate_config") or {}),
        "value_field": str(payload.get("value_field") or "value"),
        "image_ids": list(payload.get("image_ids") or []),
        "statistics_engine": statistics.get("engine", "legacy"),
    }
    factory = _load_evaluator_factory(str(payload.get("evaluator_factory") or "paired_mean"))
    reusable = (statistics.get("engine") == "optimized_coco_v1"
                and statistics.get("reference_dir") and _reporting(payload)
                and str(payload.get("evaluator_factory", "")) in {
                    "onnx_splitpoint_tool.quality_metrics:detection_quality_evaluator",
                    "onnx_splitpoint_tool.quality_metrics.detection_quality_evaluator"})
    if prepared is not None:
        reusable = False
        evaluator = prepared["evaluator"]
    elif reusable:
        from .quality_statistics import ReferenceReuseEvaluator
        evaluator = ReferenceReuseEvaluator(payload, plan, repetition_offset, config)
    else:
        evaluator = factory(reference, candidate, payload.get("annotations"), config)
    prepared_at = time.perf_counter()
    progress.emit("point", cache_state="reference_hit" if getattr(evaluator, "hit", False) else "cold")
    try:
        ones = np.ones((n,), dtype=np.float64)
        point = prepared["point"] if prepared is not None else _evaluate_metric(evaluator, ones)
        configured_guardrails = require_configured_guardrails(
            dict(payload.get("metric_gate_config") or {}), point
        )
        primary_point = dict(point["primary"])
        default_margin = float(payload["non_inferiority_margin"])
        component_points: dict[str, dict[str, Any]] = {"primary": primary_point}
        for name, component in dict(point.get("guardrails") or {}).items():
            component_points[f"guardrails.{name}"] = dict(component)
        bootstrap: dict[str, list[float]] = {name: [] for name in component_points}
        absolute = {name: {"reference": [], "candidate": []} for name in component_points}
        ratio_draws = []
        point_at = time.perf_counter()
        progress.emit("accumulating")
        reporting = _reporting(payload)
        skipped_reason = ""
        if not reporting and _prediction_identity_is_bound(payload):
            if any(float(component["delta"]) != 0.0 for component in component_points.values()):
                raise ValueError("identical prediction payloads produced nonzero metric differences")
            skipped_reason = "candidate_reference_identical"
        elif not reporting and any(
            _point_estimate_below_margin(component, default_margin)
            for component in component_points.values()
        ):
            skipped_reason = "point_estimate_below_non_inferiority_margin"
        else:
            plan_array = np.asarray(plan, dtype=np.int64)
            if plan_array.ndim != 2 or plan_array.shape[1] != n:
                raise ValueError("bootstrap shard does not match the paired image count")
            for indices in plan_array:
                multiplicities = np.bincount(indices, minlength=n).astype(np.float64, copy=False)
                sampled = _evaluate_metric(evaluator, multiplicities)
                sampled_components: dict[str, Mapping[str, Any]] = {"primary": sampled["primary"]}
                sampled_components.update(
                    {f"guardrails.{name}": component for name, component in sampled.get("guardrails", {}).items()}
                )
                if set(sampled_components) != set(component_points):
                    raise ValueError("quality evaluator returned a different metric set during resampling")
                for name, component in sampled_components.items():
                    bootstrap[name].append(float(component["delta"]))
                    if statistics.get("capture_draws"):
                        for side in ("reference", "candidate"):
                            absolute[name][side].append(component.get(side))
                progress.emit("accumulating", len(bootstrap["primary"]))
                if reporting:
                    from .accuracy_reporting import assess_accuracy
                    assessment = assess_accuracy(sampled["primary"]["reference"], sampled["primary"]["candidate"], policy=reporting)
                    ratio_draws.append(assessment["relative_loss"])
        if reusable:
            evaluator.finish()
            evaluator.close()
        progress.emit("completed", len(bootstrap["primary"]))
        extra = {}
        if statistics:
            preparation = getattr(getattr(evaluator, "evaluator", evaluator), "preparation_observation", {})
            extra["statistics_observation"] = {"worker_pid": os.getpid(),
                "engine": statistics.get("engine", "legacy"),
                "prepare_s": prepared_at-started, "point_s": point_at-prepared_at,
                "ipc_wait_s": max(0.0, started-float(payload.get("_submitted_monotonic", started))),
                "accumulation_s": time.perf_counter()-point_at,
                "reference_cache_hit": bool(getattr(evaluator, "hit", False)),
                "prepare_count": 1, **preparation}
        if statistics.get("capture_draws"):
            extra["absolute_bootstrap"] = absolute
        return {
            **extra,
            "shard_index": int(shard_index),
            "repetition_offset": int(repetition_offset),
            "repetitions": int(len(next(iter(bootstrap.values()), []))),
            "component_points": component_points,
            "configured_guardrails": list(configured_guardrails),
            "bootstrap": bootstrap,
            "relative_loss_draws": ratio_draws,
            "skipped_reason": skipped_reason,
            "worker_elapsed_s": float(time.perf_counter() - started),
        }
    finally:
        if reusable:
            evaluator.close()


def _combine_evaluation_shards(
    payload: Mapping[str, Any],
    shard_results: Sequence[Mapping[str, Any]],
    *,
    elapsed_s: float,
    workers_requested: int,
) -> dict[str, Any]:
    """Combine ordered shard distributions into the canonical result shape."""

    rows = list(shard_results)
    repetitions_requested = int(payload["repetitions"])
    for row in rows:
        if (not isinstance(row, Mapping)
                or type(row.get("repetition_offset")) is not int
                or type(row.get("repetitions")) is not int
                or not 0 <= row["repetition_offset"] <= repetitions_requested
                or row["repetitions"] < 0):
            raise ValueError("paired bootstrap shard has an invalid typed draw range")
    ordered = sorted(rows, key=lambda row: row["repetition_offset"])
    if not ordered:
        raise ValueError("paired bootstrap produced no worker result")
    if any("block_identity" in row for row in ordered):
        identity = ordered[0].get("block_identity")
        if (not isinstance(identity, Mapping)
                or identity.get("schema") != "paired-statistics-block-v1"
                or identity.get("evaluation") != payload["evaluation_fingerprint"]
                or identity.get("phase") != "candidate"
                or identity.get("B") != repetitions_requested
                or identity.get("n") != len(payload["reference_records"])
                or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("plan", "")))
                or any(row.get("block_identity") != identity for row in ordered)):
            raise ValueError("paired bootstrap blocks have incompatible scientific/plan identities")
    component_points = {
        str(name): dict(component)
        for name, component in dict(ordered[0].get("component_points") or {}).items()
    }
    if not component_points or "primary" not in component_points:
        raise ValueError("paired bootstrap worker did not return a primary metric")
    for shard in ordered[1:]:
        if set(dict(shard.get("component_points") or {})) != set(component_points):
            raise ValueError("paired bootstrap shards returned different metric sets")
        if canonical_json(shard["component_points"]) != canonical_json(component_points):
            raise ValueError("paired bootstrap shards returned different point components")
        if tuple(shard.get("configured_guardrails") or ()) != tuple(
            ordered[0].get("configured_guardrails") or ()
        ):
            raise ValueError("paired bootstrap shards returned different guardrail contracts")
    skip_reasons = {str(shard.get("skipped_reason") or "") for shard in ordered}
    if len(skip_reasons) != 1:
        raise ValueError("paired bootstrap shards disagreed about early termination")
    skipped_reason = next(iter(skip_reasons))
    default_margin = float(payload["non_inferiority_margin"])
    if skipped_reason == "candidate_reference_identical":
        if not _prediction_identity_is_bound(payload) or any(float(component["delta"]) != 0.0 for component in component_points.values()):
            raise ValueError("candidate_reference_identical lacks verified prediction identity")
        intervals = {name: (None, None) for name in component_points}
        repetitions_effective = 0
    elif skipped_reason == "point_estimate_below_non_inferiority_margin":
        intervals = {name: (None, None) for name in component_points}
        repetitions_effective = 0
    elif skipped_reason:
        raise ValueError(f"unsupported paired bootstrap skip reason: {skipped_reason}")
    else:
        bootstrap: dict[str, list[float]] = {name: [] for name in component_points}
        cursor = 0
        require_absolute = bool((payload.get("_statistics") or {}).get("capture_draws")) or any(
            "absolute_bootstrap" in shard for shard in ordered)
        for shard in ordered:
            count = shard["repetitions"]
            if (count <= 0 or shard["repetition_offset"] != cursor
                    or cursor + count > repetitions_requested):
                raise ValueError("paired bootstrap draw ranges overlap, have gaps, or exceed the plan")
            shard_values = dict(shard.get("bootstrap") or {})
            if set(shard_values) != set(component_points):
                raise ValueError("paired bootstrap shard returned a different metric set")
            for name in component_points:
                if len(shard_values[name]) != count:
                    raise ValueError("paired bootstrap delta length does not match its draw range")
                if not all(type(value) in (int, float) and math.isfinite(value)
                           for value in shard_values[name]):
                    raise ValueError("paired bootstrap contains an invalid delta")
                bootstrap[name].extend(float(value) for value in list(shard_values[name]))
            if _reporting(payload):
                ratios = shard.get("relative_loss_draws", [])
                if len(ratios) != count or not all(value is None or (
                    type(value) in (int, float) and math.isfinite(value)) for value in ratios):
                    raise ValueError("paired bootstrap ratio values do not match their draw range")
            if require_absolute:
                absolute = shard.get("absolute_bootstrap")
                if not isinstance(absolute, Mapping) or set(absolute) != set(component_points):
                    raise ValueError("paired bootstrap absolute metric set is incomplete")
                for values in absolute.values():
                    if not isinstance(values, Mapping) or set(values) != {"reference", "candidate"}:
                        raise ValueError("paired bootstrap absolute sides are incomplete")
                    for side in values.values():
                        if len(side) != count or not all(type(value) in (int, float)
                            and math.isfinite(value) for value in side):
                            raise ValueError("paired bootstrap absolute values do not match their draw range")
            cursor += count
        if cursor != repetitions_requested:
            raise ValueError("paired bootstrap draw ranges do not cover the entire plan")
        if any(len(values) != repetitions_requested for values in bootstrap.values()):
            raise ValueError("paired bootstrap shards did not cover the registered repetition plan exactly")
        alpha = max(0.0, min(0.5, 1.0 - float(payload["confidence_level"])))
        if _reporting(payload):
            alpha /= 2.0
        intervals = {}
        for name, values in bootstrap.items():
            distribution = np.asarray(values, dtype=np.float64)
            intervals[name] = (
                float(np.quantile(distribution, alpha)),
                float(np.quantile(distribution, 1.0 - alpha)),
            )
        repetitions_effective = repetitions_requested
    n = len(list(payload["reference_records"]))
    workers_effective = 1 if repetitions_effective == 0 else min(
        int(workers_requested), len(ordered), repetitions_effective
    )
    completed_components: dict[str, dict[str, Any]] = {}
    for name, point_component in component_points.items():
        margin = float(point_component.get("margin", default_margin))
        ci_low, ci_high = intervals[name]
        delta = float(point_component["delta"])
        decision = "pass" if skipped_reason == "candidate_reference_identical" else _decision(point_component, ci_low, margin)
        if skipped_reason == "candidate_reference_identical":
            decision_basis = "candidate_reference_identical"
            uncertainty_status = "deterministic_identity"
            gate_bound_value = 0.0
        elif skipped_reason:
            decision_basis = (
                "point_estimate_below_non_inferiority_margin" if decision == "fail"
                else "bootstrap_not_computed_other_component_point_fail"
            )
            uncertainty_status = "not_computed_fast_fail"
            gate_bound_value = delta if decision == "fail" else None
        else:
            decision_basis = "paired_bootstrap_lower_bound"
            uncertainty_status = "computed_bootstrap"
            gate_bound_value = ci_low
        completed_component = {
            "metric": str(point_component.get("metric") or name),
            "candidate": point_component.get("candidate"),
            "reference": point_component.get("reference"),
            "delta": delta,
            "ci_low": float(ci_low) if ci_low is not None else None,
            "ci_high": float(ci_high) if ci_high is not None else None,
            "ci_computed": not bool(skipped_reason),
            "decision_basis": decision_basis,
            "gate_bound_value": gate_bound_value,
            "uncertainty_status": uncertainty_status,
            "prediction_identity_verified": skipped_reason == "candidate_reference_identical",
            "margin": margin,
            "n": n,
            "decision": decision,
            "status": decision,
            "bootstrap_repetitions_requested": repetitions_requested,
            "bootstrap_repetitions": repetitions_effective,
            "bootstrap_engine": f"management_process_pool:{payload.get('evaluator_factory')}",
            "bootstrap_skipped_reason": skipped_reason,
            "bootstrap_elapsed_s": float(elapsed_s),
        }
        if all(
            point_component.get(field) is not None
            for field in ("candidate_hits", "reference_hits", "sample_count")
        ):
            completed_component.update(
                {
                    "candidate_hits": int(point_component["candidate_hits"]),
                    "reference_hits": int(point_component["reference_hits"]),
                    "sample_count": int(point_component["sample_count"]),
                    "point_estimate_comparison_basis": "integer_hit_counts",
                }
            )
        else:
            completed_component["point_estimate_comparison_basis"] = (
                "float_inclusive_threshold_tolerance"
            )
        completed_components[name] = completed_component
    primary = completed_components.pop("primary")
    guardrails = {
        name.split(".", 1)[1]: component
        for name, component in completed_components.items()
        if name.startswith("guardrails.")
    }
    component_decisions = {str(primary["decision"])} | {
        str(component["decision"]) for component in guardrails.values()
    }
    decision = (
        "fail" if "fail" in component_decisions
        else "pass" if component_decisions == {"pass"}
        else "inconclusive"
    )
    configured_guardrails = tuple(ordered[0].get("configured_guardrails") or ())
    missing_configured_guardrails = sorted(set(configured_guardrails) - set(guardrails))
    if missing_configured_guardrails:
        raise ValueError(
            "completed quality result omitted configured guardrail(s): "
            + ", ".join(missing_configured_guardrails)
            + "; evaluation fails closed"
        )
    reporting_fields = {}
    reporting = _reporting(payload)
    if reporting:
        from .accuracy_reporting import assess_accuracy
        if float(payload["confidence_level"]) != reporting["confidence_level"]:
            raise ValueError("reporting confidence level mismatch")
        expected_metric = reporting["primary_metrics"].get(str((payload.get("metric_gate_config") or {}).get("task") or ""))
        if expected_metric and primary["metric"] != expected_metric:
            raise ValueError("reporting primary metric mismatch")
        draws = [v for shard in ordered for v in shard.get("relative_loss_draws", [])]
        if len(draws) != repetitions_requested:
            raise ValueError("relative-loss draws do not cover the registered paired plan")
        undefined = sum(v is None for v in draws)
        ci = None if undefined else [float(x) for x in np.quantile(draws, [0.025, 0.975])]
        assessment = assess_accuracy(primary["reference"], primary["candidate"], ci, policy=reporting,
            interval_reason="undefined_bootstrap_zero_reference" if undefined else "")
        secondary = {name: assess_accuracy(c["reference"], c["candidate"], policy=reporting) for name, c in guardrails.items()}
        reporting_fields = {
            "reporting_policy": reporting, "accuracy_assessment": assessment,
            "secondary_accuracy_assessments": secondary,
            "accuracy_warnings": [name + "_relative_loss_gt_5pct" for name, a in secondary.items() if a["accuracy_class"] == "accuracy_loss"],
            "legacy_decision": decision, "relative_loss_undefined_draws": undefined,
            "observed_image_ids": list(payload.get("image_ids") or []),
            "evaluated_images": n,
        }
        decision = assessment["accuracy_class"] or "not_estimable"
        for component in [primary, *guardrails.values()]:
            component["legacy_decision"] = component.pop("decision")
            component["status"] = "completed"
    return {
        **reporting_fields,
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "schema_version": 1,
        "quality_result_contract_version": QUALITY_RESULT_CONTRACT_VERSION,
        "status_name": EVALUATE_PAIRED_QUALITY_UNCERTAINTY,
        "status": "completed",
        "decision": decision,
        "primary": primary,
        "guardrails": guardrails,
        "configured_guardrails": list(configured_guardrails),
        "guardrail_contract_complete": True,
        "n": n,
        "request_id": str(payload.get("request_id") or ""),
        "candidate_execution_completion_contract_sha256": str(
            payload.get("candidate_execution_completion_contract_sha256") or ""
        ),
        "reference_identity": payload.get("reference_identity"),
        "evaluation_fingerprint": str(payload["evaluation_fingerprint"]),
        "reference_predictions_sha256": str(payload["reference_predictions_sha256"]),
        "candidate_predictions_sha256": str(payload["candidate_predictions_sha256"]),
        "annotations_sha256": str(payload["annotations_sha256"]),
        "algorithm_version": str(payload["algorithm_version"]),
        "seed_schema": dict(payload["seed_schema"]),
        "execution_location": "management_node",
        "execution_device": "management_cpu",
        "worker_model": "process_pool",
        "bootstrap_workers_requested": int(workers_requested),
        "bootstrap_workers_effective": int(workers_effective),
        "bootstrap_sharding": "contiguous_deterministic_resample_plan",
        "gpu_used": False,
        "cpu_reference_semantic_only": True,
        "cpu_reference_include_in_latency_fps_energy": False,
        "cpu_reference_include_in_ranking": False,
        "cpu_reference_include_in_pareto": False,
        "cache_hit": False,
    }


def _evaluate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Single-process compatibility entry point using the canonical plan."""

    started = time.perf_counter()
    n = len(list(payload["reference_records"]))
    if not _reporting(payload) and _prediction_identity_is_bound(payload):
        plan = np.empty((0, n), dtype=np.int64)
    else:
        plan = deterministic_resample_plan(
            image_count=n,
            repetitions=int(payload["repetitions"]),
            seed=int(payload["seed"]),
        )
    shard = _evaluate_payload_shard(payload, plan, shard_index=0, repetition_offset=0)
    return _combine_evaluation_shards(
        payload,
        [shard],
        elapsed_s=float(time.perf_counter() - started),
        workers_requested=1,
    )


class ResourcePauseGate:
    """Thread-safe admission gate for new quality worker jobs.

    Running process-pool tasks are allowed to finish; newly queued work waits.
    This avoids unsafe process suspension while keeping u.RECS windows and local
    Hailo compilation free from newly admitted management CPU load.
    """

    def __init__(self, *, available_cpu=None, available_memory_bytes=None, transfer_slots=1, postcalc_slots=1) -> None:
        from .quality_statistics_config import resource_budget
        budget = resource_budget()
        self.available_cpu = int(budget["statistics_cpu_slots"] if available_cpu is None else available_cpu)
        self.available_memory_bytes = int(budget["available_memory_bytes"] if available_memory_bytes is None else available_memory_bytes)
        if self.available_cpu <= 0 or self.available_memory_bytes < 0:
            raise ValueError("controller CPU capacity must be positive and RAM capacity nonnegative")
        self.transfer_slots = max(1, int(transfer_slots))
        self.postcalc_slots = max(1, int(postcalc_slots))
        self._condition = threading.Condition()
        self._reasons: set[str] = set()
        self._activities = {}
        self._reservations = {}
        self._tickets = []
        self._serial = 0

    def snapshot(self):
        with self._condition:
            return {
                "cpu_capacity": self.available_cpu,
                "transfer_capacity": self.transfer_slots,
                "postcalc_capacity": self.postcalc_slots,
                "cpu_active": sum(a["cpu"] for a in self._activities.values()),
                "memory_bytes_active": sum(a["memory_bytes"] for a in self._activities.values()),
                "activities": [dict(a) for a in self._activities.values()],
                "reservations": [dict(r) for r in self._reservations.values()],
                "waiting": len(self._tickets),
            }

    def begin_activity(self, reason, *, cpu=1, memory_bytes=0, resources=None, honor_pause=True, ignore_pause_reasons=(), continuation_token=None):
        resources = frozenset(resources if resources is not None else ("controller:cpu", "controller:io"))
        if cpu < 0 or memory_bytes < 0 or cpu > self.available_cpu or memory_bytes > self.available_memory_bytes:
            raise ValueError("controller activity exceeds available CPU/RAM budget")
        with self._condition:
            self._serial += 1
            ticket = {"id": self._serial, "cpu": cpu, "memory_bytes": memory_bytes, "resources": resources,
                      "honor_pause": honor_pause, "ignore_pause_reasons": tuple(ignore_pause_reasons), "continuation_token": continuation_token, "reason": reason, "owner_thread": threading.get_ident()}
            self._tickets.append(ticket)
            return ticket["id"]

    def _activity_conflict(self, ticket):
        resources = ticket["resources"]
        continuation = self._activities.get(ticket["continuation_token"])
        return ((ticket["honor_pause"] and bool(self._reasons.difference(ticket["ignore_pause_reasons"])) and bool(resources))
            or any(resources.intersection(r["resources"]) and (r["state"] != "DRAINING" or r["id"] < ticket["id"])
                   and not (r["state"] == "DRAINING" and continuation
                            and set(continuation["resources"]).intersection(r["resources"]))
                   for r in self._reservations.values())
            or any(tag in resources and sum(tag in a["resources"] for a in self._activities.values()) >= limit
                   for tag, limit in (("controller:transfer", self.transfer_slots), ("controller:postcalc", self.postcalc_slots)))
            or any(any(key.startswith(("dut:", "source:")) for key in resources.intersection(a["resources"]))
                   for a in self._activities.values()))

    def poll_activity(self, token):
        with self._condition:
            if token in self._activities:
                return True
            ticket = next(t for t in self._tickets if t["id"] == token)
            cpu_used = sum(a["cpu"] for a in self._activities.values())
            memory_used = sum(a["memory_bytes"] for a in self._activities.values())
            earlier = self._tickets[:self._tickets.index(ticket)]
            priority = not any(t["memory_bytes"] + memory_used <= self.available_memory_bytes
                and not self._activity_conflict(t)
                and t["cpu"] + cpu_used <= self.available_cpu for t in earlier)
            if (self._activity_conflict(ticket) or not priority
                    or cpu_used + ticket["cpu"] > self.available_cpu
                    or memory_used + ticket["memory_bytes"] > self.available_memory_bytes):
                return False
            self._tickets.remove(ticket)
            self._activities[token] = {**ticket, "resources": sorted(ticket["resources"]), "started_monotonic": time.monotonic()}
            self._condition.notify_all()
            return True

    def acquire_activity(self, reason, *, cpu=1, memory_bytes=0, resources=None,
                         check_cancelled=lambda: None, timeout=None, honor_pause=True, ignore_pause_reasons=(), continuation_token=None):
        token = self.begin_activity(reason, cpu=cpu, memory_bytes=memory_bytes, resources=resources,
                                    honor_pause=honor_pause, ignore_pause_reasons=ignore_pause_reasons, continuation_token=continuation_token)
        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = False
        try:
            with self._condition:
                while True:
                    check_cancelled()
                    if self.poll_activity(token):
                        acquired = True
                        return token
                    if deadline is not None and time.monotonic() >= deadline:
                        return None
                    self._condition.wait(0.05)
        finally:
            if not acquired:
                self.release_activity(token)

    def release_activity(self, token):
        with self._condition:
            if token in self._activities:
                self._activities.pop(token)
            else:
                ticket = next((t for t in self._tickets if t["id"] == token), None)
                if ticket is None:
                    raise RuntimeError("controller activity owner already released or unknown")
                self._tickets.remove(ticket)
            self._condition.notify_all()

    @contextmanager
    def activity(self, reason, **kwargs):
        token = self.acquire_activity(reason, **kwargs)
        if token is None:
            raise TimeoutError("controller activity admission timed out")
        try:
            yield token
        finally:
            self.release_activity(token)

    def begin_quiet(self, reason, *, resources=None, owner=None):
        resources = frozenset(resources if resources is not None else ("controller:cpu", "controller:io"))
        owner = threading.get_ident() if owner is None else owner
        with self._condition:
            if any(a["owner_thread"] == owner and resources.intersection(a["resources"])
                   for a in self._activities.values()):
                raise RuntimeError("quiet reservation conflicts with its own active work")
            self._serial += 1
            token = self._serial
            self._reservations[token] = {"id": token, "reason": reason, "resources": sorted(resources),
                "owner_thread": owner, "state": "DRAINING", "reserved_monotonic": time.monotonic()}
            self._condition.notify_all()
            return token

    def poll_quiet(self, token):
        with self._condition:
            row = self._reservations[token]
            resources = set(row["resources"])
            previous = False
            for other in self._reservations.values():
                conflict = resources.intersection(other["resources"])
                if other["owner_thread"] == row["owner_thread"] or not conflict:
                    continue
                first_owned = min(owned["id"] for owned in self._reservations.values()
                                  if owned["owner_thread"] == row["owner_thread"]
                                  and conflict.intersection(owned["resources"]))
                if other["state"] != "DRAINING" or other["id"] < first_owned:
                    previous = True
                    break
            cpu_used = sum(a["cpu"] for a in self._activities.values())
            memory_used = sum(a["memory_bytes"] for a in self._activities.values())
            older_ready = any(t["id"] < row["id"] and resources.intersection(t["resources"])
                and t["cpu"] + cpu_used <= self.available_cpu
                and t["memory_bytes"] + memory_used <= self.available_memory_bytes
                and not self._activity_conflict(t) for t in self._tickets)
            active = older_ready or any(resources.intersection(activity["resources"]) for activity in self._activities.values())
            if not previous and not active and row["state"] == "DRAINING":
                row["state"] = "QUIET_CONFIRMED"
                row["quiet_monotonic"] = time.monotonic()
            return dict(row)

    def end_quiet(self, token):
        with self._condition:
            row = self._reservations.pop(token)
            row["state"] = "RELEASED"
            self._condition.notify_all()
            return dict(row)

    @contextmanager
    def reserve_quiet(self, reason, *, resources=None, check_cancelled=lambda: None):
        token = self.begin_quiet(reason, resources=resources)
        try:
            with self._condition:
                while self.poll_quiet(token)["state"] != "QUIET_CONFIRMED":
                    check_cancelled()
                    self._condition.wait(0.05)
                check_cancelled()
                row = self._reservations[token]
            yield row
        finally:
            self.end_quiet(token)

    def pause(self, reason: str) -> None:
        value = str(reason).strip()
        if not value:
            raise ValueError("pause reason must not be empty")
        with self._condition:
            self._reasons.add(value)

    def resume(self, reason: str) -> None:
        value = str(reason).strip()
        with self._condition:
            self._reasons.discard(value)
            if not self._reasons:
                self._condition.notify_all()

    def resume_all(self) -> None:
        with self._condition:
            self._reasons.clear()
            self._condition.notify_all()

    @property
    def paused(self) -> bool:
        with self._condition:
            return bool(self._reasons)

    @property
    def reasons(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(sorted(self._reasons))

    def wait_until_resumed(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._reasons:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    @contextmanager
    def hold(self, reason: str):
        # Separate owners keep nested holds from releasing each other.
        with self._condition:
            self._serial += 1
            owned_reason = f"{reason}:{self._serial}"
        self.pause(owned_reason)
        try:
            yield self
        finally:
            self.resume(owned_reason)


@dataclass
class _QueuedEvaluation:
    key: str
    payload: dict[str, Any]


_REQUEST_SCOPED_RESULT_FIELDS = (
    "request_id",
    "reference_identity",
    "candidate_execution_completion_contract_sha256",
    "artifact_provenance_binding_status",
    "artifact_provenance_claim_eligible",
)


def _request_scoped_result_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project metadata that belongs to one caller, not the scientific core."""

    return {name: payload[name] for name in _REQUEST_SCOPED_RESULT_FIELDS}


def _rebind_request_scoped_result(
    result: Mapping[str, Any], request_fields: Mapping[str, Any]
) -> dict[str, Any]:
    """Return one result view with the submitting client's request metadata."""

    rebound = dict(result)
    rebound.update(_request_scoped_result_fields(request_fields))
    return rebound


@dataclass(frozen=True)
class _EvaluationWaiter:
    client: Future
    request_fields: Mapping[str, Any]


@dataclass
class _ShardEvaluation:
    key: str
    payload: dict[str, Any]
    pending: int
    workers_requested: int
    started: float = field(default_factory=time.perf_counter)
    results: list[dict[str, Any]] = field(default_factory=list)
    first_error: Optional[BaseException] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    seen_workers: set[Future] = field(default_factory=set)
    workers: list[Future] = field(default_factory=list)
    cancelled: bool = False


class ManagementQualityService:
    """Persistent, cached management-node paired-quality process pool."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        workers: int = 4,
        pause_gate: Optional[ResourcePauseGate] = None,
        statistics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.statistics = dict(statistics or {})
        if statistics is not None:
            from .quality_statistics_config import DEFAULTS, statistics_options
            unknown = set(self.statistics) - set(DEFAULTS) - {"capture_draws"}
            if unknown:
                raise ValueError("unknown statistics options: " + ", ".join(sorted(unknown)))
            capture = self.statistics.get("capture_draws", False)
            if type(capture) is not bool:
                raise ValueError("capture_draws must be boolean")
            self.statistics = {**statistics_options({"quality_gate": {"statistics": {
                key: value for key, value in self.statistics.items() if key in DEFAULTS}}}),
                "capture_draws": capture}
        self._started_at = time.time()
        if self.statistics.get("engine") == "optimized_coco_v1" and (
            type(workers) is not int or not 1 <= workers <= 64
        ):
            raise ValueError("optimized statistics workers must be an integer in [1, 64]")
        count = int(workers)
        if count < 1:
            raise ValueError("workers must be at least 1")
        self.workers_requested = count
        self._bounded_blocks = self.statistics.get("engine") == "optimized_coco_v1"
        from .quality_statistics_config import resource_budget
        count = min(count, (pause_gate.available_cpu if pause_gate is not None else resource_budget()["statistics_cpu_slots"]))
        self.workers = count
        self.cache = PersistentQualityCache(cache_dir)
        self._submission_lock = threading.Lock()
        self._scratch_cleanup_errors = []
        if self._bounded_blocks:
            import tempfile
            self._payload_scratch = Path(tempfile.mkdtemp(prefix="session-", dir=self.cache.root))
        self.pause_gate = pause_gate or ResourcePauseGate()
        self._worker_memory_token = None
        self._worker_memory_lock = threading.Lock()
        self._private_worker_bytes = count * (self.statistics.get("prepared_cache_limit_mib", 512) + 64) * 1024**2
        # The GUI/workflow is intentionally multi-threaded.  Explicit ``spawn``
        # avoids forking that live process, which can inherit locked runtime or
        # BLAS state and deadlock during long campaigns.
        self._executor = ProcessPoolExecutor(
            max_workers=count,
            mp_context=mp.get_context("spawn"),
        )
        # Do not pre-fill ProcessPoolExecutor's unbounded private work queue.
        # At most one task per worker is admitted, so a later u.RECS/Hailo pause
        # can still hold all remaining campaign evaluations on our own queue.
        self._worker_slots = threading.BoundedSemaphore(count)
        self._queue: queue.Queue[Optional[_QueuedEvaluation]] = queue.Queue()
        self._lock = threading.RLock()
        self._inflight: dict[str, list[_EvaluationWaiter]] = {}
        self._groups: dict[str, _ShardEvaluation] = {}
        self._admission_state = {"phase": "idle"}
        self._admission_contexts = {}
        self._context_local = threading.local()
        self._cancel_context: dict[str, Any] = {}
        self._closed = False
        self._shutdown_lock = threading.RLock()
        self._shutdown_processes = []
        self._shutdown_manager_thread = None
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="management-quality-dispatch",
            daemon=True,
        )
        self._dispatcher.start()

    def __enter__(self) -> "ManagementQualityService":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown(wait=True)

    def progress_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            active = set(self._inflight)
            processes = list((getattr(self._executor, "_processes", None) or {}).values())
        owned = {process.pid for process in processes if process.is_alive()}
        rows = []
        for path in (self.cache.root / "progress").glob("*.json"):
            try:
                row = json.loads(path.read_text())
                if (isinstance(row, dict) and type(row.get("worker_pid")) is int
                        and row["worker_pid"] in owned
                        and path.stem == str(row["worker_pid"])
                        and row.get("evaluation_fingerprint") in active
                        and row.get("phase") in {"loading", "preparing", "matching", "point", "accumulating"}
                        and type(row.get("last_heartbeat_at")) in (int, float)
                        and math.isfinite(row["last_heartbeat_at"])
                        and row["last_heartbeat_at"] >= self._started_at):
                    rows.append(row)
            except (OSError, ValueError, TypeError):
                continue
        return sorted(rows, key=lambda row: row["worker_pid"])

    def pause(self, reason: str) -> None:
        self.pause_gate.pause(reason)

    def admission_snapshot(self):
        """Coordinator state, distinct from actually executing worker ranges."""
        with self._lock:
            return {**self._admission_state, "contexts": {k: dict(v) for k, v in self._admission_contexts.items()},
                    "resources": self.pause_gate.snapshot(), "queued_descriptors": self._queue.qsize(),
                    "pause_reasons": list(self.pause_gate.reasons)}

    def _statistics_phase(self, phase, **fields):
        with self._lock:
            key = fields.get("evaluation_fingerprint") or getattr(self._context_local, "key", None)
            state = {**self._admission_contexts.get(key, {}), "phase": phase, **fields}
            if key:
                self._admission_contexts[key] = state
            self._admission_state = state if len(self._admission_contexts) <= 1 else {"phase": "multiple_contexts"}

    def resume(self, reason: str) -> None:
        self.pause_gate.resume(reason)

    def pause_for_urecs(self) -> None:
        self.pause(URECS_RESOURCE_REASON)

    def resume_after_urecs(self) -> None:
        self.resume(URECS_RESOURCE_REASON)

    def pause_for_hailo_build(self) -> None:
        self.pause(HAILO_BUILD_RESOURCE_REASON)

    def resume_after_hailo_build(self) -> None:
        self.resume(HAILO_BUILD_RESOURCE_REASON)

    @contextmanager
    def resource_hold(self, reason: str):
        with self.pause_gate.hold(reason):
            yield self

    def submit(self, request: QualityEvaluationRequest) -> Future:
        # Serialize preparation before it can materialize another large paired
        # payload. Queued requests retain only disk-bound JSON descriptors.
        if self._bounded_blocks:
            with self._submission_lock:
                from .quality_statistics_blocks import reachable_bytes
                required = 4 * reachable_bytes(request)
                if required + self._private_worker_bytes > self.pause_gate.available_memory_bytes:
                    client = Future()
                    client.set_exception(MemoryError(f"statistics admission needs {required + self._private_worker_bytes} bytes; available {self.pause_gate.available_memory_bytes}"))
                    return client
                with self.pause_gate.activity("quality_input_prepare",
                        memory_bytes=required, honor_pause=False):
                    return self._submit(request)
        from .quality_statistics_blocks import reachable_bytes
        with self.pause_gate.activity("quality_legacy_input_prepare", honor_pause=False,
                memory_bytes=4 * reachable_bytes(request)):
            return self._submit(request)

    def _submit(self, request: QualityEvaluationRequest) -> Future:
        key, payload = prepare_evaluation(request)
        if self.statistics.get("engine") == "optimized_coco_v1":
            from .quality_statistics import (numerical_environment, validate_bound_order,
                                             validate_canonical_prediction_fields)
            validate_bound_order(payload)
            validate_canonical_prediction_fields(payload)
            payload["legacy_evaluation_fingerprint"] = key
            key = json_fingerprint({"schema": "optimized-quality-pair-v1", "legacy": key,
                "image_ids": payload["image_ids"], "annotations": payload["annotations"],
                "environment": numerical_environment()})
            payload["evaluation_fingerprint"] = key
        if self.statistics:
            payload["_statistics"] = {**self.statistics,
                "progress_dir": str(self.cache.root / "progress"),
                "reference_dir": str(self.cache.root / "reference_components")}
        cached = self.cache.get(key)
        if cached is not None and not _cached_guardrail_contract_matches_request(
            request.metric_gate_config, cached,
        ):
            cached = None
        client: Future = Future()
        if cached is not None:
            if self.statistics:
                cached["statistics_observation"] = {
                    **cached.get("statistics_observation", {}),
                    "cache_state": "complete_result_reused", "draws_recomputed": 0,
                    "engine_requested": self.statistics.get("engine", "legacy")}
            client.set_result(_rebind_request_scoped_result(cached, payload))
            return client
        waiter = _EvaluationWaiter(
            client=client,
            request_fields=_request_scoped_result_fields(payload),
        )
        with self._lock:
            if self._closed:
                raise self._closed_error()
            waiters = self._inflight.get(key)
            if waiters is not None:
                waiters.append(waiter)
                return client
        if self._bounded_blocks:
            from .quality_statistics_blocks import spool_payload
            queued_payload = spool_payload(self._payload_scratch, key, payload)
        else:
            queued_payload = payload
        with self._lock:
            if self._closed:
                if self._bounded_blocks:
                    self._remove_spooled_payload(queued_payload)
                raise self._closed_error()
            waiters = self._inflight.get(key)
            if waiters is not None:
                waiters.append(waiter)
                return client
            self._inflight[key] = [waiter]
            try:
                self._queue.put(_QueuedEvaluation(key=key, payload=queued_payload))
            except BaseException:
                self._inflight.pop(key, None)
                if self._bounded_blocks:
                    self._remove_spooled_payload(queued_payload)
                raise
        return client

    def evaluate(self, request: QualityEvaluationRequest, timeout: Optional[float] = None) -> dict[str, Any]:
        future = self.submit(request)
        try:
            return future.result(timeout=timeout)
        except CancelledError as exc:
            raise stamp_exception(exc, getattr(future, "quality_cancel_context", None))

    def _closed_error(self) -> BaseException:
        return stamp_exception(
            QualityServiceClosedError("management quality service is closed"),
            self._cancel_context,
        )

    def evaluate_many(
        self,
        requests: Sequence[QualityEvaluationRequest],
        timeout: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if self._bounded_blocks:
            # Keep caller payload loading bounded as well as the worker pool.
            width = self.statistics["max_active_requests"]
            pending, results = [], []
            for request in requests:
                if len(pending) == width:
                    results.append(pending.pop(0).result(timeout=timeout))
                pending.append(self.submit(request))
            return results + [future.result(timeout=timeout) for future in pending]
        futures = [self.submit(request) for request in requests]
        return [future.result(timeout=timeout) for future in futures]

    def _dispatch_loop(self) -> None:
        if self._bounded_blocks:
            self._dispatch_blocks()
            return
        while True:
            item = self._queue.get()
            if item is None:
                return
            self.pause_gate.wait_until_resumed()
            payload = item.payload
            repetitions = int(payload.get("repetitions") or 1)
            identical = not _reporting(payload) and _prediction_identity_is_bound(payload)
            shard_count = 1 if identical else max(1, min(self.workers, repetitions))
            def check():
                if self._closed:
                    raise self._closed_error()
            from .quality_statistics_blocks import reachable_bytes
            try:
                activity_token = self.pause_gate.acquire_activity("quality_legacy:" + item.key,
                    cpu=shard_count, memory_bytes=4 * reachable_bytes(payload) + 16 * repetitions * len(payload.get("reference_records") or []),
                    check_cancelled=check)
            except BaseException as exc:
                self._finish_exception(item.key, exc)
                continue
            acquired = 0
            for _ in range(shard_count):
                self._worker_slots.acquire()
                acquired += 1
            with self._lock:
                if self._closed:
                    for _ in range(acquired):
                        self._worker_slots.release()
                    self.pause_gate.release_activity(activity_token)
                    self._finish_exception(item.key, self._closed_error())
                    continue
            try:
                n = len(list(payload.get("reference_records") or []))
                if identical:
                    plan = np.empty((0, n), dtype=np.int64)
                else:
                    plan = deterministic_resample_plan(
                        image_count=n,
                        repetitions=repetitions,
                        seed=int(payload.get("seed") or 0),
                    )
                boundaries = np.linspace(0, len(plan), shard_count + 1, dtype=np.int64)
                group = _ShardEvaluation(
                    key=item.key,
                    payload=payload,
                    pending=shard_count,
                    workers_requested=self.workers,
                )
            except BaseException as exc:
                for _ in range(acquired):
                    self._worker_slots.release()
                self.pause_gate.release_activity(activity_token)
                self._finish_exception(item.key, exc)
                continue
            group.activity_token = activity_token
            with self._lock:
                self._groups[item.key] = group
            submitted = 0
            try:
                for shard_index in range(shard_count):
                    start = int(boundaries[shard_index])
                    stop = int(boundaries[shard_index + 1])
                    # No worker may be created after controlled shutdown has
                    # taken its owned-process snapshot.
                    with self._lock:
                        if self._closed:
                            raise self._closed_error()
                        worker = self._executor.submit(
                            _evaluate_payload_shard,
                            {**payload, "_submitted_monotonic": time.perf_counter()},
                            plan[start:stop],
                            shard_index=shard_index,
                            repetition_offset=start,
                        )
                        with group.lock:
                            group.workers.append(worker)
                    submitted += 1
                    worker.add_done_callback(
                        lambda done, current=group: self._shard_worker_done(current, done)
                    )
            except BaseException as exc:
                stamp_exception(exc)
                unsubmitted = shard_count - submitted
                for _ in range(unsubmitted):
                    self._worker_slots.release()
                with group.lock:
                    if group.first_error is None:
                        group.first_error = exc
                    # Submitted callbacks may already have reduced pending.
                    # Remove only work that will never receive a callback.
                    group.pending -= unsubmitted
                    complete = group.pending == 0
                    error = group.first_error
                if complete:
                    self._finish_exception(item.key, error)
                    self._release_legacy_activity(group)

    def _check_block_cancelled(self, group):
        with group.lock:
            if group.first_error is not None:
                raise group.first_error
            if group.cancelled or self._closed:
                raise self._closed_error()

    def _block_worker_done(self, group, worker):
        # Callback only records terminal errors and releases its owned slot.
        # The dispatcher alone admits ranges, validates and publishes results.
        with group.lock:
            if worker in group.seen_workers:
                return
            group.seen_workers.add(worker)
            try:
                worker.result()
            except BaseException as exc:
                if group.first_error is None:
                    group.first_error = stamp_exception(exc)
            group.pending -= 1
        self._worker_slots.release()

    def _dispatch_blocks(self):
        threads = [threading.Thread(target=self._dispatch_block_context,
                    name=f"management-quality-context-{index}", daemon=True)
                   for index in range(self.statistics["max_active_requests"])]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self._queue.get_nowait()  # consume the shared terminal marker

    def _dispatch_block_context(self):
        from .quality_statistics_blocks import load_payload, execute_request
        while True:
            item = self._queue.get()
            if item is None:
                # Each context consumes one terminal marker without changing
                # the existing dispatcher/shutdown ownership contract.
                self._queue.put(None)
                return
            group = None
            memory_token = None
            self._context_local.key = item.key
            try:
                self._statistics_phase("waiting_resource" if self.pause_gate.paused else "loading_payload",
                                       evaluation_fingerprint=item.key, request_id=item.payload.get("request_id"))
                with self._lock:
                    if self._closed:
                        raise self._closed_error()
                def check():
                    if self._closed:
                        raise self._closed_error()
                with self._worker_memory_lock:
                    if self._worker_memory_token is None:
                        self._worker_memory_token = self.pause_gate.acquire_activity(
                            "statistics_worker_preparation", cpu=0, resources=(),
                            memory_bytes=self._private_worker_bytes, check_cancelled=check)
                memory_token = self.pause_gate.acquire_activity("quality_context:" + item.key,
                    cpu=0, resources=(), check_cancelled=check,
                    memory_bytes=item.payload.get("admission_memory_bytes", 12 * item.payload["size_bytes"]))
                with self.pause_gate.activity("quality_payload_load:" + item.key, check_cancelled=check):
                    payload = load_payload(item.payload)
                group = _ShardEvaluation(key=item.key, payload=payload, pending=0,
                                         workers_requested=self.workers_requested)
                with self._lock:
                    if self._closed:
                        raise self._closed_error()
                    self._groups[item.key] = group
                execute_request(self, group, item.payload)
            except BaseException as exc:
                if group is not None:
                    with group.lock:
                        if group.first_error is None:
                            group.first_error = stamp_exception(exc)
                        exc = group.first_error
                    # Drain only this service's admitted tasks before admitting
                    # another request after an I/O or evaluator failure.
                    for future in group.workers:
                        try:
                            future.result()
                        except BaseException:
                            pass
                self._finish_exception(item.key, exc)
            finally:
                cleanup_token = None
                try:
                    def cleanup_check():
                        if self._closed:
                            raise self._closed_error()
                    try:
                        cleanup_token = self.pause_gate.acquire_activity(
                            "quality_payload_cleanup:" + item.key,
                            check_cancelled=cleanup_check)
                    except BaseException:
                        if not self._closed:
                            raise
                        # Closing forbids more calculations, not safe cleanup.
                        # Never wait for or cross an unresolved capture fence.
                        cleanup_token = self.pause_gate.acquire_activity(
                            "quality_payload_cleanup:" + item.key, timeout=0)
                    if cleanup_token is not None:
                        self._remove_spooled_payload(item.payload)
                finally:
                    if cleanup_token is not None:
                        self.pause_gate.release_activity(cleanup_token)
                    if memory_token is not None:
                        self.pause_gate.release_activity(memory_token)
                with self._lock:
                    self._admission_contexts.pop(item.key, None)
                    self._admission_state = next(iter(self._admission_contexts.values()), {"phase": "idle"})

    def _remove_spooled_payload(self, descriptor):
        """Release only the JSON payload created by this service session."""
        path = Path(descriptor["path"])
        expected = self._payload_scratch / "payloads" / (descriptor["key"] + ".json")
        if path != expected:
            raise ValueError("refusing to remove a foreign statistics payload")
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            self._scratch_cleanup_errors.append(str(exc))

    def _complete_block_result(self, group, result):
        # Caller holds self._lock from merge through cache and waiter publish.
        waiters = self._inflight.pop(group.key, [])
        self._groups.pop(group.key, None)
        for waiter in waiters:
            if not waiter.client.done():
                waiter.client.set_result(_rebind_request_scoped_result(result, waiter.request_fields))

    def _release_legacy_activity(self, group):
        with group.lock:
            token = getattr(group, "activity_token", None)
            group.activity_token = None
        if token is not None:
            self.pause_gate.release_activity(token)

    def _shard_worker_done(self, group: _ShardEvaluation, worker: Future) -> None:
        with group.lock:
            if worker in group.seen_workers:
                return
            group.seen_workers.add(worker)
            # Observation and first-error storage share the shutdown lock.
            # A cancellation between them must never erase an earlier error.
            try:
                shard_result = dict(worker.result())
            except BaseException as exc:
                stamp_exception(exc)
                shard_result = {}
                shard_error: Optional[BaseException] = exc
            else:
                shard_error = None
            if shard_error is not None and group.first_error is None:
                group.first_error = shard_error
            if shard_result:
                group.results.append(shard_result)
            group.pending -= 1
            complete = group.pending == 0
            cancelled = group.cancelled
        self._worker_slots.release()
        if not complete:
            return
        try:
            if cancelled:
                with self._lock:
                    self._groups.pop(group.key, None)
                return
            if group.first_error is not None:
                self._finish_exception(group.key, group.first_error)
                return
            # Serialize complete-result publication with cancellation. Once every
            # shard is available, either its validated result publishes atomically
            # or shutdown wins first; an incomplete group never populates cache.
            with self._lock:
                if group.cancelled:
                    self._groups.pop(group.key, None)
                    return
                try:
                    merge_started = time.perf_counter()
                    result = _combine_evaluation_shards(
                        group.payload,
                        group.results,
                        elapsed_s=float(time.perf_counter() - group.started),
                        workers_requested=group.workers_requested,
                    )
                    if self.statistics.get("capture_draws"):
                        _atomic_write_json(self.cache.root / "draws" / (group.key + ".json"),
                            {"payload_identity": group.key, "shards": group.results})
                    if self.statistics:
                        result["statistics_observation"] = {"engine": self.statistics.get("engine", "legacy"),
                            "merge_s": time.perf_counter()-merge_started,
                            "shards": [r.get("statistics_observation", {}) for r in group.results]}
                    self.cache.put(group.key, result)
                except BaseException as exc:
                    self._finish_exception(group.key, exc)
                    return
                waiters = self._inflight.pop(group.key, [])
                self._groups.pop(group.key, None)
                for waiter in waiters:
                    client = waiter.client
                    if not client.done():
                        client.set_result(
                            _rebind_request_scoped_result(result, waiter.request_fields)
                        )
        finally:
            # Only the callback whose own decrement reached zero may release
            # admission, and only after it has published/checkpointed results.
            self._release_legacy_activity(group)

    def _finish_exception(self, key: str, exc: BaseException) -> None:
        stamp_exception(exc)
        with self._lock:
            waiters = self._inflight.pop(key, [])
            self._groups.pop(key, None)
            for waiter in waiters:
                client = waiter.client
                if not client.done():
                    client.set_exception(exc)

    def shutdown_state(self) -> dict[str, Any]:
        processes = list(self._shutdown_processes or
                         (getattr(self._executor, "_processes", {}) or {}).values())
        manager = self._shutdown_manager_thread or getattr(self._executor, "_executor_manager_thread", None)
        workers = [{"pid": p.pid, "exitcode": p.exitcode, "alive": p.is_alive()} for p in processes]
        state = {"admission_closed": self._closed,
                 "dispatcher_alive": self._dispatcher.is_alive(),
                 "manager_alive": bool(manager and manager.is_alive()),
                 "workers": workers, "inflight_keys": list(self._inflight),
                 "group_keys": list(self._groups), "queue_size": self._queue.qsize(),
                 "scratch_cleanup_errors": list(self._scratch_cleanup_errors)}
        state["finished"] = bool(self._closed and not state["dispatcher_alive"]
                                 and not state["manager_alive"] and not state["inflight_keys"]
                                 and not state["group_keys"] and state["queue_size"] == 0
                                 and not any(p["alive"] for p in workers))
        return state

    def shutdown(self, wait: bool = True, cancel_futures: bool = False,
                 terminate_workers: bool = False,
                 cancellation_context: Optional[Mapping[str, Any]] = None) -> None:
        # Preserve ownership before Python clears its executor references on
        # shutdown(wait=False), including a subsequent bounded finalizer.
        with self._shutdown_lock:
            self._shutdown_impl(wait, cancel_futures, terminate_workers, cancellation_context)
            if wait:
                for process in self._shutdown_processes:
                    process.join(timeout=0)
                manager = self._shutdown_manager_thread
                if manager is not None and manager is not threading.current_thread():
                    manager.join()
                state = self.shutdown_state()
                if not state["finished"]:
                    raise RuntimeError("management_quality_shutdown_unresolved: " + str(state))
                if self._worker_memory_token is not None:
                    self.pause_gate.release_activity(self._worker_memory_token)
                    self._worker_memory_token = None
                if self._bounded_blocks:
                    # Empty directories from this session only. Persistent
                    # plans, checkpoints and completed results stay reusable.
                    for directory in (self._payload_scratch / "payloads", self._payload_scratch):
                        try:
                            directory.rmdir()
                        except FileNotFoundError:
                            pass
                        except OSError:
                            pass

    def _shutdown_impl(
        self,
        wait: bool = True,
        cancel_futures: bool = False,
        terminate_workers: bool = False,
        cancellation_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._lock:
            already_closed = self._closed
            if not self._shutdown_processes:
                self._shutdown_processes = list((getattr(self._executor, "_processes", {}) or {}).values())
            if self._shutdown_manager_thread is None:
                self._shutdown_manager_thread = getattr(self._executor, "_executor_manager_thread", None)
            if not already_closed:
                self._closed = True
                if cancel_futures:
                    self._cancel_context = dict(cancellation_context or {})
                    self._cancel_context["shutdown_monotonic"] = time.monotonic()
                    for key, waiters in list(self._inflight.items()):
                        group = self._groups.get(key)
                        with group.lock if group is not None else self._lock:
                            error = group.first_error if group is not None else None
                            if group is not None and error is None:
                                # A process result may be terminal while its
                                # callback waits for this lock. Inspect only
                                # already-done futures; never wait on a worker.
                                for worker in group.workers:
                                    if worker.done() and not worker.cancelled():
                                        observed_error = worker.exception()
                                        if observed_error is not None:
                                            error = stamp_exception(observed_error)
                                            group.first_error = error
                                            break
                            if group is not None:
                                group.cancelled = True
                        # Preserve a genuine failure already observed before
                        # the user stopped other outstanding requests.
                        if error is not None:
                            self._finish_exception(key, error)
                            continue
                        for waiter in waiters:
                            waiter.client.quality_cancel_context = dict(self._cancel_context)
                            waiter.client.cancel()
                    self._inflight.clear()
        if already_closed:
            if terminate_workers:
                for process in self._shutdown_processes:
                    if process.is_alive():
                        process.terminate()
                for process in self._shutdown_processes:
                    process.join(timeout=0.5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=0.5)
            if wait:
                self._dispatcher.join()
                self._executor.shutdown(
                    wait=True,
                    cancel_futures=cancel_futures,
                )
            return
        # Never leave the dispatch thread blocked behind a campaign resource.
        self.pause_gate.resume_all()
        self._queue.put(None)
        if terminate_workers:
            # ProcessPoolExecutor offers no public immediate-stop API before
            # Python 3.14.  Explicit user cancellation must nevertheless not
            # leave a minutes-long bootstrap running.  Limit this private-API
            # fallback to that terminal path and always join/kill explicitly.
            processes = list((getattr(self._executor, "_processes", {}) or {}).values())
            for process in processes:
                try:
                    if process.is_alive():
                        process.terminate()
                except Exception:
                    pass
            deadline = time.monotonic() + 3.0
            for process in processes:
                try:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                    if process.is_alive() and hasattr(process, "kill"):
                        process.kill()
                        process.join(timeout=1.0)
                except Exception:
                    pass
        if wait:
            self._dispatcher.join()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


__all__ = [
    "CPUQualityReferenceIdentity",
    "CPUQualityReferenceStore",
    "DETECTION_QUALITY_ALGORITHM_VERSION",
    "EVALUATE_PAIRED_QUALITY_UNCERTAINTY",
    "GENERATE_CPU_QUALITY_REFERENCE",
    "HAILO_BUILD_RESOURCE_REASON",
    "ImagePairingError",
    "ManagementQualityService",
    "PairedPredictionRecords",
    "QualityEvaluationRequest",
    "QualityArtifactIntegrityError",
    "QualityServiceClosedError",
    "QUALITY_RESULT_CONTRACT_VERSION",
    "ResourcePauseGate",
    "URECS_RESOURCE_REASON",
    "annotations_fingerprint",
    "configured_guardrail_names",
    "deterministic_resample_plan",
    "make_cpu_reference_identity",
    "pair_prediction_records",
    "prepare_evaluation",
    "quality_request_from_manifest",
    "require_configured_guardrails",
    "_cached_guardrail_contract_matches_request",
]
