"""Execution-only helpers for exact offline statistics (no inference imports)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from .quality_cache import canonical_json, json_fingerprint, prediction_fingerprint

ENGINE = "optimized_coco_v1"


def numerical_environment():
    import platform
    try:
        coco_version = importlib.metadata.version("pycocotools")
    except importlib.metadata.PackageNotFoundError:
        coco_version = "unavailable"
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pycocotools": coco_version,
            "dtype": "float64", "plan_dtype": "int64", "kernel": ENGINE}


def validate_bound_order(payload):
    expected = canonical_json(payload["image_ids"])
    field = payload.get("image_id_field", "image_id")
    for role in ("reference", "candidate"):
        observed = [row[field] for row in payload[role + "_records"]]
        if canonical_json(observed) != expected:
            raise ValueError(role + " record order does not match the paired image population")
    annotations = payload.get("annotations")
    if isinstance(annotations, list) and annotations and all(isinstance(row, dict) and field in row for row in annotations):
        if canonical_json([row[field] for row in annotations]) != expected:
            raise ValueError("annotation order does not match the paired image population")


def validate_canonical_prediction_fields(payload, *, roles=("reference", "candidate")):
    """Bind the actual fields read by the two standard metric factories."""
    factory = str(payload.get("evaluator_factory", "")).replace(":", ".")
    if factory not in {
        "onnx_splitpoint_tool.quality_metrics.detection_quality_evaluator",
        "onnx_splitpoint_tool.quality_metrics.classification_quality_evaluator",
    }:
        return
    for role in roles:
        selected = payload.get(role + "_prediction_field")
        records = payload[role + "_records"]
        observed = prediction_fingerprint(records,
            image_id_field=payload.get("image_id_field", "image_id"), payload_field=selected)
        if observed != payload[role + "_predictions_sha256"]:
            raise ValueError(role + " payload changed after preparation")
        # An unselected whole-record fingerprint already covers the standard
        # field. An alias is safe only when it names the same actual payload.
        if selected is not None and selected != role:
            actual = prediction_fingerprint(records,
                image_id_field=payload.get("image_id_field", "image_id"), payload_field=role)
            if actual != observed:
                raise ValueError(role + " fingerprint does not bind the actual canonical metric field")


def reference_component_key(payload, plan, offset):
    # Bind positional annotations as consumed, including crowd/ignore/area. The
    # full gate contains candidate/producer provenance and is deliberately not
    # the identity of this policy-independent absolute reference component.
    validate_bound_order(payload)
    validate_canonical_prediction_fields(payload, roles=("reference",))
    observed = prediction_fingerprint(payload["reference_records"],
        image_id_field=payload.get("image_id_field", "image_id"),
        payload_field=payload.get("reference_prediction_field"))
    if observed != payload["reference_predictions_sha256"]:
        raise ValueError("reference payload changed after preparation")
    return json_fingerprint({"schema": "coco-reference-component-v1",
        "reference_predictions_sha256": payload["reference_predictions_sha256"],
        "image_ids": payload["image_ids"], "annotations": payload["annotations"],
        "algorithm_version": payload["algorithm_version"],
        "seed_schema": payload["seed_schema"], "offset": int(offset),
        "plan_shape": list(plan.shape),
        "plan_sha256": hashlib.sha256(np.ascontiguousarray(plan).view(np.uint8)).hexdigest(),
        "environment": numerical_environment(),
        "metrics": ["coco_ap_50_95", "ap50", "ap75"],
        "coco": {"iou": [0.5, 0.95, 10], "recall": [0, 1, 101],
                 "area": [0, 1e10], "maxDets": 100}})


class WorkerProgress:
    """Atomic per-worker phases and heartbeats; only draw events advance progress."""
    def __init__(self, payload, start, stop):
        import threading
        self.directory = payload.get("_statistics", {}).get("progress_dir")
        self.base = {"request_id": payload.get("request_id"),
            "evaluation_fingerprint": payload.get("evaluation_fingerprint"),
            "worker_pid": os.getpid(), "repetition_start": start,
            "repetition_stop": stop, "draws_requested": stop-start}
        self.started = time.perf_counter()
        self.last = 0.0
        self.phase = ""
        self.completed = 0
        self.last_progress = None
        self.fields = {}
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.directory:
            import threading
            def heartbeat():
                while not self._stop.wait(5):
                    with self._guard:
                        self._write()
            self._thread = threading.Thread(target=heartbeat, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.emit("failed", self.completed, error_type=exc_type.__name__)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _write(self):
        from .quality_service import _atomic_write_json
        self.last = time.perf_counter()
        _atomic_write_json(Path(self.directory) / f"{os.getpid()}.json",
            {**self.base, "phase": self.phase, "draws_completed": self.completed,
             "last_progress_at": self.last_progress,
             "last_heartbeat_at": time.time(), "elapsed_s": self.last-self.started,
             **self.fields})

    def emit(self, phase, completed=0, **fields):
        if not self.directory:
            return
        with self._guard:
            changed = phase != self.phase
            if completed != self.completed:
                self.last_progress = time.time()
            self.phase, self.completed = phase, completed
            self.fields.update(fields)
            if changed or time.perf_counter()-self.last >= 5:
                self._write()


class ReferenceReuseEvaluator:
    """Private mutable COCO accumulators plus immutable completed reference rows.

    A shard owner calculates its own reference: no admitted candidate ever waits
    on a reference task queued behind it. Kernel file locks release on owner death.
    """
    def __init__(self, payload, plan, offset, config):
        from .quality_metrics import _CanonicalCOCOEvaluator
        from .quality_service import _reference_store_file_lock, _atomic_write_json
        validate_bound_order(payload)
        validate_canonical_prediction_fields(payload, roles=("candidate",))
        self._write = _atomic_write_json
        self.key = reference_component_key(payload, plan, offset)
        self.path = Path(payload["_statistics"]["reference_dir"]) / (self.key + ".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = _reference_store_file_lock(self.path.parent, self.key)
        self.lock.__enter__()
        self.values = []
        self.position = 0
        self.expected = len(plan)+1
        self.hit = False
        try:
            if self.path.is_file():
                try:
                    cached = json.loads(self.path.read_text())
                    values = cached["values"]
                    array = np.asarray(values, dtype=np.float64)
                    self.hit = (cached.get("schema") == "coco-reference-component-v1"
                                and cached["key"] == self.key and array.shape == (self.expected, 3)
                                and np.isfinite(array).all()
                                and cached["sha256"] == json_fingerprint(values))
                    if self.hit:
                        self.values = values
                except (OSError, ValueError, KeyError, TypeError):
                    self.hit = False
            observed = prediction_fingerprint(payload["candidate_records"],
                image_id_field=payload.get("image_id_field", "image_id"),
                payload_field=payload.get("candidate_prediction_field"))
            if observed != payload["candidate_predictions_sha256"]:
                raise ValueError("candidate payload changed after preparation")
            from .quality_service import _prediction_identity_is_bound
            self.identity = _prediction_identity_is_bound(payload)
            sides = (() if self.hit else ("reference",)) + (() if self.identity else ("candidate",))
            self.evaluator = _CanonicalCOCOEvaluator(payload["reference_records"],
                payload["candidate_records"], payload["annotations"], {**config, "coco_sides": sides})
        except BaseException:
            self.close()
            raise

    def evaluate(self, multiplicities):
        metrics = self.evaluator.evaluate_sides(multiplicities)
        if self.hit:
            reference = self.values[self.position]
        else:
            reference = metrics.pop(0)
            self.values.append(reference)
        candidate = reference if self.identity else metrics.pop(0)
        self.position += 1
        return self.evaluator.pair_components(reference, candidate)

    def finish(self):
        if self.position != self.expected:
            raise ValueError("incomplete reference component")
        if not self.hit:
            self._write(self.path, {"schema": "coco-reference-component-v1", "key": self.key,
                "values": self.values, "sha256": json_fingerprint(self.values)})

    def close(self):
        if getattr(self, "lock", None) is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None
