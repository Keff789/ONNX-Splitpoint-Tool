"""Accuracy reporting v1: project decision frozen before R9J, separate from validity."""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Mapping

DEFAULT_REPORTING_POLICY = {'policy_id': 'accuracy_reporting_v1', 'technical_fail_only': True, 'primary_metrics': {'classification': 'top1_accuracy', 'detection': 'coco_ap_50_95'}, 'loss_definition': '(reference-candidate)/reference', 'relative_loss_threshold': 0.05, 'uncertainty_band_low': 0.03, 'uncertainty_band_high': 0.07, 'confidence_level': 0.95, 'interval_kind': 'paired_bootstrap_two_sided_relative_loss', 'secondary_metrics': 'report_warning_only', 'quality_affects_technical_pass': False, 'preserve_historical_results': True, 'dataset_policy': 'preserve_images_and_denominators', 'energy_window_policy': 'unchanged_command'}


def reporting_policy(value):
    """Validate the versioned contract; absent means historical semantics."""
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("reporting_policy must be a mapping")
    for key, expected in DEFAULT_REPORTING_POLICY.items():
        if key not in value or value[key] != expected or (isinstance(expected, bool) and type(value[key]) is not bool):
            raise ValueError(f"invalid accuracy reporting contract: {key}")
    return deepcopy(dict(value))


def active_policy(config):
    return reporting_policy((config or {}).get("reporting_policy"))


def _metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("primary/secondary metric must be finite and normalized to [0,1]")
    return float(value)


def assess_accuracy(reference, candidate, interval=None, *, policy=None, interval_reason="interval_not_computed"):
    p = reporting_policy(policy or DEFAULT_REPORTING_POLICY)
    reference, candidate = _metric(reference), _metric(candidate)
    out = dict(policy_id=p["policy_id"], accuracy_class=None, relative_loss=None,
               absolute_loss_pp=100 * (reference - candidate), relative_loss_ci=None,
               confidence_level=p["confidence_level"], uncertainty="not_estimated",
               uncertainty_reason=interval_reason, technical_fail=False)
    if reference == 0:
        out.update(uncertainty="not_estimable", uncertainty_reason="zero_reference")
        return out
    loss = (reference - candidate) / reference
    threshold = p["relative_loss_threshold"]
    close = loss <= threshold or math.isclose(loss, threshold, rel_tol=0, abs_tol=1e-12)
    out.update(relative_loss=loss, accuracy_class="reference_close" if close else "accuracy_loss")
    if interval is None:
        if interval_reason.startswith("undefined_bootstrap"):
            out["uncertainty"] = "not_estimable"
        return out
    if len(interval) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in interval):
        raise ValueError("invalid relative loss interval")
    low, high = map(float, interval)
    if low > high:
        raise ValueError("reversed relative loss interval")
    out.update(relative_loss_ci=[low, high], uncertainty_reason="")
    if high <= threshold or low > threshold:
        agrees = (close and high <= threshold) or (not close and low > threshold)
        out["uncertainty"] = "supported" if agrees else "inconclusive"
        if not agrees:
            out["uncertainty_reason"] = "point_interval_disagreement"
    elif low < p["uncertainty_band_low"] and high > p["uncertainty_band_high"]:
        out["uncertainty"] = "inconclusive"
    else:
        out["uncertainty"] = "borderline"
    return out


def assessment_fields(assessment):
    """Lossless flat CSV/UI projection alongside the canonical JSON block."""
    if not isinstance(assessment, Mapping):
        return {}
    return {"accuracy_assessment": dict(assessment), **{
        "accuracy_" + key if key not in {"accuracy_class"} else key: value
        for key, value in assessment.items()
    }}


def accuracy_label(assessment):
    if not isinstance(assessment, Mapping):
        return ""
    cls = {"reference_close": "Referenznah", "accuracy_loss": "Genauigkeitsverlust"}.get(assessment.get("accuracy_class"), "Nicht einstufbar")
    uncertainty = {"supported": "statistisch gestützt", "borderline": "grenznah", "inconclusive": "unsicher", "not_estimated": "nicht geschätzt", "not_estimable": "nicht schätzbar"}.get(assessment.get("uncertainty"), "")
    return f"{cls} / {uncertainty}"


def observed_coverage(rows, *, run_id, run_root=None):
    """Coverage comes exclusively from completed paired-evaluator inputs."""
    import hashlib
    import json
    from pathlib import Path
    def image_id(value):
        token = Path(str(value)).stem
        return str(int(token)) if token.isdecimal() else token
    def backend(value):
        return {"deepx": "deepx_m1", "deepx_m1_full": "deepx_m1", "native_full_deepx": "deepx_m1"}.get(value, value)
    root = Path(run_root).resolve() if run_root is not None else None
    consumers = []
    rejected = []
    for row in rows:
        ids = row.get("observed_image_ids") or []
        if row.get("task") != "detection" or row.get("technical_status") not in {"completed", "ok"} or not row.get("accuracy_assessment"):
            continue
        normalized_ids = [image_id(v) for v in ids]
        if not ids or len(set(normalized_ids)) != len(ids) or row.get("evaluated_images") != len(ids):
            continue
        binding = {}
        if root is not None:
            try:
                request_path = (root / row["source_request"]).resolve()
                if not request_path.is_relative_to(root):
                    raise ValueError("request_outside_run")
                raw = request_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != row.get("source_request_sha256"):
                    raise ValueError("request_bytes_mismatch")
                request = json.loads(raw)
                producer = request.get("producer_identity") or {}
                if (request.get("eval_run_id", run_id) != run_id
                        or row.get("collection_eval_run_id", run_id) != run_id
                        or request.get("model_id", producer.get("model_id", row.get("model_id"))) != row.get("model_id")
                        or backend(request.get("backend", producer.get("backend"))) != backend(row.get("backend"))
                        or request.get("variant") != row.get("variant")
                        or request.get("record_count") != len(ids)
                        or sorted(image_id(v) for v in request.get("expected_image_ids", [])) != sorted(normalized_ids)):
                    raise ValueError("request_consumer_binding_mismatch")
                candidate = request["candidate"]
                candidate_path = (request_path.parent / candidate["path"]).resolve()
                if not candidate_path.is_relative_to(root):
                    raise ValueError("candidate_outside_run")
                raw = candidate_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != candidate.get("sha256"):
                    raise ValueError("candidate_bytes_mismatch")
                records = json.loads(raw)["records"]
                record_ids = [image_id(r["image_id"]) for r in records]
                if len(records) != len(ids) or sorted(record_ids) != sorted(normalized_ids):
                    raise ValueError("candidate_observations_mismatch")
                metric = row.get("task_quality_metric") or (row.get("primary") or {}).get("metric")
                if not row.get("reference_identity") or metric != "coco_ap_50_95":
                    raise ValueError("canonical_reference_or_metric_missing")
                reference_id = str(row["reference_identity"])
                store_root = root / "quality_management/cpu_reference_store"
                reference_path = (store_root / reference_id[:2] / reference_id / "manifest.json").resolve()
                if not reference_path.is_relative_to(root):
                    raise ValueError("reference_outside_run")
                reference_manifest = json.loads(reference_path.read_bytes())
                identity_fields = reference_manifest["identity_contract"]
                predictions_path = reference_path.with_name("predictions.json").resolve()
                if not predictions_path.is_relative_to(root):
                    raise ValueError("reference_predictions_outside_run")
                reference_records = json.loads(predictions_path.read_bytes())
                from .quality_service import make_cpu_reference_identity, CPUQualityReferenceStore
                identity = make_cpu_reference_identity(
                    **{k: identity_fields[k] for k in ("model", "dataset", "preprocessing", "decoder")},
                    prediction_records=reference_records)
                if identity.fingerprint() != reference_id:
                    raise ValueError("reference_identity_mismatch")
                stored = CPUQualityReferenceStore(store_root).get(identity)
                if not stored or stored["manifest"].get("image_count") != len(ids):
                    raise ValueError("reference_records_invalid")
                if sorted(image_id(r["image_id"]) for r in stored["predictions"]) != sorted(normalized_ids):
                    raise ValueError("reference_observations_mismatch")
                binding = dict(candidate_path=str(candidate_path.relative_to(root)),
                    candidate_sha256=candidate["sha256"], candidate_record_count=len(records),
                    source_request_sha256=row["source_request_sha256"],
                    reference_identity=row["reference_identity"], metric=metric,
                    reference_manifest=str(reference_path.relative_to(root)),
                    consumer_binding_status="verified", run_id=run_id)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(model_id=row.get("model_id"), backend=row.get("backend"),
                    variant=row.get("variant"), source_request=row.get("source_request"), reason=str(exc)))
                continue
        consumers.append(dict(model_id=row.get("model_id"), backend=row.get("backend"),
            variant=row.get("variant"), setup_id=row.get("setup_id") or row.get("source_setup_id"),
            source_request=row.get("source_request"), evaluated_images=len(ids),
            observed_image_ids=normalized_ids, **binding))
    observed = sorted(set(v for c in consumers for v in c["observed_image_ids"]))
    return dict(run_id=run_id, source="observed_completed_quality_inputs", observed_image_ids=observed,
        evaluated_images=len(observed), excluded_images=[], consumers=consumers,
        rejected_consumers=rejected,
        evidence_files=["quality_management/central_quality_summary.json"] if consumers else [])
