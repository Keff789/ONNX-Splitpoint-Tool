"""Content-addressed cache primitives for management-side quality evaluation.

The cache deliberately fingerprints only the scientific inputs of one paired
evaluation: reference predictions, candidate predictions, annotations, the
metric/gate configuration and the resampling algorithm.  Image IDs are the
pairing key, but are not sufficient as a cache key because two accelerators can
produce different predictions for the same images.

Hashes are calculated once at artifact/evaluation boundaries.  Nothing in this
module hashes data inside a bootstrap repetition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


CACHE_SCHEMA = "onnx-splitpoint/management-quality-cache"
CACHE_SCHEMA_VERSION = 3
CACHE_RESULT_CONTRACT_VERSION = 3
EVALUATION_FINGERPRINT_SCHEMA = "onnx-splitpoint/minimal-quality-evaluation-v3"
REFERENCE_IDENTITY_SCHEMA = "onnx-splitpoint/cpu-quality-reference-v1"


class QualityFingerprintError(ValueError):
    """Raised when quality evidence cannot be fingerprinted unambiguously."""


def _json_projection(value: Any) -> Any:
    """Return a deterministic JSON projection for common prediction values."""

    if is_dataclass(value):
        return _json_projection(asdict(value))
    if isinstance(value, np.ndarray):
        # Shape and dtype prevent otherwise ambiguous byte/list projections.
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": [int(x) for x in value.shape],
            "values": _json_projection(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _json_projection(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_projection(item) for item in value]
    if isinstance(value, set):
        projected = [_json_projection(item) for item in value]
        return sorted(projected, key=lambda item: canonical_json(item))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QualityFingerprintError("NaN and infinity are not valid quality evidence")
        # Normalise negative zero without rounding scientifically relevant data.
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise QualityFingerprintError(
        f"Unsupported quality evidence type: {type(value).__module__}.{type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Serialise *value* deterministically without locale-dependent formatting."""

    return json.dumps(
        _json_projection(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_image_id(value: Any) -> str:
    """Return a typed token so e.g. integer ``1`` cannot collide with string ``"1"``."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None:
        raise QualityFingerprintError("image_id must be a string or integer, not bool/null")
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, str):
        if value == "":
            raise QualityFingerprintError("image_id must not be empty")
        return f"str:{value}"
    raise QualityFingerprintError(
        f"image_id must be a string or integer, got {type(value).__name__}"
    )


def _record_image_id(record: Mapping[str, Any], image_id_field: str) -> Any:
    if image_id_field not in record:
        raise QualityFingerprintError(f"prediction record is missing {image_id_field!r}")
    return record[image_id_field]


def indexed_prediction_records(
    records: Sequence[Mapping[str, Any]],
    *,
    image_id_field: str = "image_id",
) -> dict[str, Mapping[str, Any]]:
    """Index per-image predictions and reject duplicate identities."""

    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise QualityFingerprintError(f"prediction record {position} is not a mapping")
        token = canonical_image_id(_record_image_id(record, image_id_field))
        if token in indexed:
            raise QualityFingerprintError(
                f"duplicate {image_id_field} in prediction records: {record[image_id_field]!r}"
            )
        indexed[token] = record
    if not indexed:
        raise QualityFingerprintError("prediction records must not be empty")
    return indexed


def prediction_fingerprint(
    records: Sequence[Mapping[str, Any]],
    *,
    image_id_field: str = "image_id",
    payload_field: Optional[str] = None,
) -> str:
    """Fingerprint per-image predictions independent of input record order.

    ``payload_field`` can select the canonical prediction payload when a record
    also contains diagnostic/runtime metadata.  Without it, every field except
    ``image_id`` is part of the scientific prediction artifact.
    """

    indexed = indexed_prediction_records(records, image_id_field=image_id_field)
    projected = []
    for token in sorted(indexed):
        record = indexed[token]
        if payload_field is not None:
            if payload_field not in record:
                raise QualityFingerprintError(
                    f"prediction record {record[image_id_field]!r} is missing payload field {payload_field!r}"
                )
            payload = record[payload_field]
        else:
            payload = {key: value for key, value in record.items() if key != image_id_field}
        projected.append({"image_id": token, "prediction": payload})
    return json_fingerprint({"schema": "per-image-predictions-v1", "records": projected})


@dataclass(frozen=True)
class CPUQualityReferenceIdentity:
    """Immutable identity of the one canonical CPU reference per contract."""

    model: str
    dataset: str
    preprocessing: str
    decoder: str
    image_ids: str
    provider: str = "onnxruntime_cpu"
    execution_device: str = "management_cpu"
    schema: str = REFERENCE_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        required = {
            "model": self.model,
            "dataset": self.dataset,
            "preprocessing": self.preprocessing,
            "decoder": self.decoder,
            "image_ids": self.image_ids,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise QualityFingerprintError(
                "CPU quality reference identity is incomplete: " + ", ".join(missing)
            )
        if str(self.provider).strip().lower() != "onnxruntime_cpu":
            raise QualityFingerprintError(
                "The canonical quality reference must use ONNX Runtime CPU; GPU references are not supported"
            )
        if "gpu" in str(self.execution_device).strip().lower() or "cuda" in str(self.execution_device).strip().lower():
            raise QualityFingerprintError("GPU execution is not valid for the canonical CPU quality reference")

    def fingerprint(self) -> str:
        return json_fingerprint(asdict(self))

    def metadata(self) -> dict[str, Any]:
        return {
            "status_name": "generate_cpu_quality_reference",
            "reference_identity": self.fingerprint(),
            "identity_contract": asdict(self),
            "provider": self.provider,
            "execution_device": self.execution_device,
            "semantic_reference_only": True,
            "include_in_latency_fps_energy": False,
            "include_in_ranking": False,
            "include_in_pareto": False,
            "gpu_reference_allowed": False,
        }


def image_id_set_fingerprint(
    records: Sequence[Mapping[str, Any]], *, image_id_field: str = "image_id"
) -> str:
    indexed = indexed_prediction_records(records, image_id_field=image_id_field)
    return json_fingerprint(sorted(indexed))


def image_ids_fingerprint(image_ids: Sequence[Any]) -> str:
    """Fingerprint a known validation Image-ID set before inference starts."""

    tokens = [canonical_image_id(value) for value in image_ids]
    if not tokens:
        raise QualityFingerprintError("image_ids must not be empty")
    if len(tokens) != len(set(tokens)):
        raise QualityFingerprintError("image_ids contain duplicates")
    return json_fingerprint(sorted(tokens))


def evaluation_fingerprint(
    *,
    reference_predictions_sha256: str,
    candidate_predictions_sha256: str,
    annotations_sha256: str,
    metric_gate_config: Mapping[str, Any],
    algorithm_version: str,
    seed_schema: Mapping[str, Any],
) -> str:
    """Build the minimal cache key for one paired quality evaluation."""

    hashes = {
        "reference_predictions_sha256": reference_predictions_sha256,
        "candidate_predictions_sha256": candidate_predictions_sha256,
        "annotations_sha256": annotations_sha256,
    }
    for label, value in hashes.items():
        raw = str(value).strip().lower()
        if len(raw) != 64 or any(ch not in "0123456789abcdef" for ch in raw):
            raise QualityFingerprintError(f"{label} must be a SHA-256 hex digest")
    if not str(algorithm_version).strip():
        raise QualityFingerprintError("algorithm_version must not be empty")
    return json_fingerprint(
        {
            "schema": EVALUATION_FINGERPRINT_SCHEMA,
            **hashes,
            "metric_gate_config": dict(metric_gate_config),
            "algorithm_version": str(algorithm_version),
            "seed_schema": dict(seed_schema),
        }
    )


def _validate_result_contract(result: Mapping[str, Any]) -> None:
    """Validate the structural contract shared by cache writes and reads."""

    if (
        int(result.get("quality_result_contract_version") or 0)
        != CACHE_RESULT_CONTRACT_VERSION
    ):
        raise QualityFingerprintError(
            "quality cache result has an unsupported result contract version"
        )
    if result.get("guardrail_contract_complete") is not True:
        raise QualityFingerprintError(
            "quality cache refuses an incomplete result contract"
        )

    configured = result.get("configured_guardrails")
    if not isinstance(configured, Sequence) or isinstance(
        configured, (str, bytes, bytearray)
    ):
        raise QualityFingerprintError(
            "quality cache result configured_guardrails must be a sequence"
        )
    configured_names: list[str] = []
    seen: set[str] = set()
    for name in configured:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise QualityFingerprintError(
                "quality cache result configured_guardrails must contain "
                "non-empty, trimmed strings"
            )
        if name in seen:
            raise QualityFingerprintError(
                f"quality cache result configured_guardrails contains duplicate {name!r}"
            )
        seen.add(name)
        configured_names.append(name)

    guardrails = result.get("guardrails")
    if not isinstance(guardrails, Mapping):
        raise QualityFingerprintError(
            "quality cache result guardrails must be a mapping"
        )
    missing = [name for name in configured_names if name not in guardrails]
    if missing:
        raise QualityFingerprintError(
            "quality cache result omitted configured guardrail(s): "
            + ", ".join(missing)
        )
    malformed = [name for name in configured_names if not isinstance(guardrails[name], Mapping)]
    if malformed:
        raise QualityFingerprintError(
            "quality cache result configured guardrail value(s) must be mappings: "
            + ", ".join(malformed)
        )


    primary = result.get("primary")
    if not isinstance(primary, Mapping):
        raise QualityFingerprintError("quality cache result primary must be a mapping")
    for name, component in (("primary", primary), *guardrails.items()):
        if not isinstance(component, Mapping):
            raise QualityFingerprintError(f"quality cache component {name} must be a mapping")
        computed = component.get("ci_computed")
        repetitions = component.get("bootstrap_repetitions")
        if not isinstance(computed, bool) or isinstance(repetitions, bool) or not isinstance(repetitions, int):
            raise QualityFingerprintError(f"quality cache component {name} lacks v3 uncertainty provenance")
        if not component.get("decision_basis") or not component.get("uncertainty_status") or "gate_bound_value" not in component:
            raise QualityFingerprintError(f"quality cache component {name} lacks v3 decision provenance")
        if computed:
            try:
                finite_bounds = all(math.isfinite(float(component[field])) for field in ("ci_low", "ci_high"))
            except (TypeError, ValueError, KeyError):
                finite_bounds = False
            if repetitions <= 0 or component.get("bootstrap_skipped_reason") or not finite_bounds:
                raise QualityFingerprintError(f"quality cache component {name} has uncomputed claimed confidence bounds")
            if (component.get("decision_basis") != "paired_bootstrap_lower_bound"
                    or component.get("uncertainty_status") != "computed_bootstrap"
                    or component.get("gate_bound_value") != component.get("ci_low")
                    or float(component["ci_low"]) > float(component["ci_high"])):
                raise QualityFingerprintError(f"quality cache component {name} has inconsistent computed uncertainty")
        elif repetitions != 0 or component.get("ci_low") is not None or component.get("ci_high") is not None:
            raise QualityFingerprintError(f"quality cache component {name} exposes pseudo confidence bounds")
        elif component.get("decision_basis") not in {
            "point_estimate_below_non_inferiority_margin", "bootstrap_not_computed_other_component_point_fail", "candidate_reference_identical",
        }:
            raise QualityFingerprintError(f"quality cache component {name} has unsupported uncomputed uncertainty")
        elif component.get("decision_basis") == "point_estimate_below_non_inferiority_margin" and (
            component.get("decision") != "fail" or component.get("gate_bound_value") != component.get("delta")
            or component.get("uncertainty_status") != "not_computed_fast_fail"
            or component.get("bootstrap_skipped_reason") != "point_estimate_below_non_inferiority_margin"
        ):
            raise QualityFingerprintError(f"quality cache component {name} has inconsistent point failure")
        elif component.get("decision_basis") == "bootstrap_not_computed_other_component_point_fail" and (
            component.get("decision") != "inconclusive" or component.get("gate_bound_value") is not None
            or component.get("uncertainty_status") != "not_computed_fast_fail"
            or component.get("bootstrap_skipped_reason") != "point_estimate_below_non_inferiority_margin"
        ):
            raise QualityFingerprintError(f"quality cache component {name} claims a decision without computed uncertainty")
        elif component.get("decision_basis") == "candidate_reference_identical":
            reference_sha = str(result.get("reference_predictions_sha256") or "")
            if (len(reference_sha) != 64 or any(c not in "0123456789abcdef" for c in reference_sha)
                    or result.get("candidate_predictions_sha256") != reference_sha
                    or component.get("prediction_identity_verified") is not True
                    or component.get("delta") != 0.0 or component.get("gate_bound_value") != 0.0
                    or component.get("decision") != "pass" or component.get("uncertainty_status") != "deterministic_identity"
                    or component.get("bootstrap_skipped_reason") != "candidate_reference_identical"):
                raise QualityFingerprintError(f"quality cache component {name} lacks bound deterministic identity")


class PersistentQualityCache:
    """Small atomic JSON cache keyed by :func:`evaluation_fingerprint`."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_key(key: str) -> str:
        value = str(key).strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise QualityFingerprintError("cache key must be a SHA-256 hex digest")
        return value

    def path_for(self, key: str) -> Path:
        digest = self._validate_key(key)
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        digest = self._validate_key(key)
        path = self.path_for(digest)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            if not isinstance(payload, Mapping):
                return None
            if payload.get("schema") != CACHE_SCHEMA or int(payload.get("schema_version") or 0) != CACHE_SCHEMA_VERSION:
                return None
            if int(payload.get("result_contract_version") or 0) != CACHE_RESULT_CONTRACT_VERSION:
                return None
            if str(payload.get("evaluation_fingerprint") or "") != digest:
                return None
            result = payload.get("result")
            if not isinstance(result, Mapping):
                return None
            _validate_result_contract(result)
            # Corruption/torn-write detection without re-hashing prediction files.
            if str(payload.get("result_sha256") or "") != json_fingerprint(result):
                return None
        except Exception:
            return None
        out = dict(result)
        out["cache_hit"] = True
        out["evaluation_fingerprint"] = digest
        return out

    def put(self, key: str, result: Mapping[str, Any]) -> Path:
        digest = self._validate_key(key)
        path = self.path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        result_payload = dict(result)
        _validate_result_contract(result_payload)
        payload = {
            "schema": CACHE_SCHEMA,
            "schema_version": CACHE_SCHEMA_VERSION,
            "result_contract_version": CACHE_RESULT_CONTRACT_VERSION,
            "evaluation_fingerprint": digest,
            "result_sha256": json_fingerprint(result_payload),
            "result": result_payload,
        }
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=str(path.parent))
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
        return path
