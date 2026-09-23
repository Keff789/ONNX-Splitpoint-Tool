"""Prepared task metrics for management-side paired quality evaluation.

Factories in this module implement the payload contract exported by the remote
runner.  They contain no backend or hardware code: inputs are already-decoded,
per-image semantic predictions.  Detection matching is prepared once and each
bootstrap repetition only applies image multiplicities to cached events.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("metric_gate_config")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _margin(config: Mapping[str, Any], key: str, default: float = 0.01) -> float:
    value: Any = _gate_config(config)
    for component in key.split("."):
        if not isinstance(value, Mapping):
            return float(default)
        value = value.get(component)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class _ClassificationAccuracyEvaluator:
    def __init__(
        self,
        reference_records: Sequence[Mapping[str, Any]],
        candidate_records: Sequence[Mapping[str, Any]],
        annotations: Any,
        config: Mapping[str, Any],
    ) -> None:
        def values(records: Sequence[Mapping[str, Any]], outer: str, metric: str) -> np.ndarray:
            out = []
            for record in records:
                nested = record.get(outer)
                if not isinstance(nested, Mapping) or nested.get(metric) is None:
                    raise ValueError(f"classification quality record is missing {outer}.{metric}")
                out.append(1.0 if bool(nested.get(metric)) else 0.0)
            return np.asarray(out, dtype=np.float64)

        self.n = len(reference_records)
        for side, records in (("reference", reference_records), ("candidate", candidate_records)):
            if side in config.get("metric_sides", ("reference", "candidate")):
                setattr(self, side + "_top1", values(records, side, "top1_hit"))
                setattr(self, side + "_top5", values(records, side, "top5_hit"))
        self.top1_margin = _margin(config, "non_inferiority_margin")
        self.top5_margin = _margin(config, "guardrails.top5_accuracy_margin")

    def evaluate_absolute(self, multiplicities, side):
        """One side of the same integral paired classification calculation."""
        weights = np.asarray(multiplicities, dtype=np.float64).reshape(-1)
        denominator = float(np.sum(weights))
        if len(weights) != self.n or denominator <= 0 or not np.array_equal(weights, np.rint(weights)):
            raise ValueError("classification component requires integral image multiplicities")
        result = {}
        for name, suffix in (("primary", "top1"), ("guardrails.top5_accuracy", "top5")):
            total = float(np.dot(weights, getattr(self, side + "_" + suffix)))
            result[name] = {"value": float(total / denominator),
                            "hits": int(round(total)), "sample_count": int(round(denominator))}
        return result

    @staticmethod
    def _component(
        reference: np.ndarray,
        candidate: np.ndarray,
        weights: np.ndarray,
        *,
        metric: str,
        margin: float,
    ) -> dict[str, float | int | str]:
        denominator = float(np.sum(weights))
        reference_total = float(np.dot(weights, reference))
        candidate_total = float(np.dot(weights, candidate))
        reference_value = float(reference_total / denominator)
        candidate_value = float(candidate_total / denominator)
        component: dict[str, float | int | str] = {
            "metric": metric,
            "candidate": candidate_value,
            "reference": reference_value,
            "delta": candidate_value - reference_value,
            "margin": float(margin),
        }
        # Point estimates and paired bootstrap multiplicities are integral for
        # classification accuracy.  Preserve those exact hit counts so the
        # inclusive non-inferiority boundary does not depend on subtracting two
        # independently rounded binary64 ratios (for example 398/500 - 403/500).
        rounded_weights = np.rint(weights)
        if np.array_equal(weights, rounded_weights):
            component.update(
                {
                    "candidate_hits": int(round(candidate_total)),
                    "reference_hits": int(round(reference_total)),
                    "sample_count": int(round(denominator)),
                }
            )
        return component

    def evaluate(self, multiplicities: np.ndarray) -> Mapping[str, Any]:
        weights = np.asarray(multiplicities, dtype=np.float64).reshape(-1)
        if len(weights) != len(self.reference_top1) or float(np.sum(weights)) <= 0.0:
            raise ValueError("classification evaluator received invalid image multiplicities")
        return {
            "primary": self._component(
                self.reference_top1,
                self.candidate_top1,
                weights,
                metric="top1_accuracy",
                margin=self.top1_margin,
            ),
            "guardrails": {
                "top5_accuracy": self._component(
                    self.reference_top5,
                    self.candidate_top5,
                    weights,
                    metric="top5_accuracy",
                    margin=self.top5_margin,
                )
            },
        }


def classification_quality_evaluator(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    annotations: Any,
    config: Mapping[str, Any],
) -> _ClassificationAccuracyEvaluator:
    return _ClassificationAccuracyEvaluator(reference_records, candidate_records, annotations, config)


def _boxes_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = (float(a[index]) for index in range(4))
    bx1, by1, bx2, by2 = (float(b[index]) for index in range(4))
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    width = max(0.0, inter_x2 - inter_x1)
    height = max(0.0, inter_y2 - inter_y1)
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    rec = np.asarray(recall, dtype=np.float32).reshape(-1)
    prec = np.asarray(precision, dtype=np.float32).reshape(-1)
    if rec.size <= 0 or prec.size <= 0:
        return 0.0
    if rec.size != prec.size:
        count = min(int(rec.size), int(prec.size))
        rec, prec = rec[:count], prec[:count]
    value = 0.0
    for threshold in np.linspace(0.0, 1.0, 101, dtype=np.float32):
        mask = rec >= float(threshold)
        value += float(np.max(prec[mask])) if np.any(mask) else 0.0
    return float(value / 101.0)


def _prepare_detection_cache(
    ground_truth: Sequence[Sequence[Mapping[str, Any]]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    image_count = len(ground_truth)
    class_ids = sorted(
        {
            int(detection.get("class_id", -1))
            for image in ground_truth
            for detection in image
            if int(detection.get("class_id", -1)) >= 0
        }
    )
    per_iou = []
    total_events = 0
    for threshold in thresholds:
        classes = []
        for class_id in class_ids:
            ground_truth_counts = np.zeros((image_count,), dtype=np.int32)
            events: list[tuple[float, int, int, float, float]] = []
            for image_index in range(image_count):
                gt_boxes = [
                    np.asarray(
                        [row.get("x1", 0.0), row.get("y1", 0.0), row.get("x2", 0.0), row.get("y2", 0.0)],
                        dtype=np.float32,
                    )
                    for row in ground_truth[image_index]
                    if int(row.get("class_id", -1)) == class_id
                ]
                ground_truth_counts[image_index] = len(gt_boxes)
                matched = [False] * len(gt_boxes)
                predicted = [
                    (
                        float(row.get("score", 0.0)),
                        np.asarray(
                            [row.get("x1", 0.0), row.get("y1", 0.0), row.get("x2", 0.0), row.get("y2", 0.0)],
                            dtype=np.float32,
                        ),
                    )
                    for row in predictions[image_index]
                    if int(row.get("class_id", -1)) == class_id
                ]
                predicted.sort(key=lambda item: float(item[0]), reverse=True)
                for local_order, (score, box) in enumerate(predicted):
                    best_iou = 0.0
                    best_index = -1
                    for ground_truth_index, gt_box in enumerate(gt_boxes):
                        if matched[ground_truth_index]:
                            continue
                        iou = _boxes_iou_xyxy(box, gt_box)
                        if iou > best_iou:
                            best_iou = float(iou)
                            best_index = ground_truth_index
                    true_positive, false_positive = 0.0, 1.0
                    if best_index >= 0 and best_iou >= float(threshold):
                        matched[best_index] = True
                        true_positive, false_positive = 1.0, 0.0
                    events.append(
                        (float(score), image_index, local_order, true_positive, false_positive)
                    )
            if int(np.sum(ground_truth_counts)) <= 0:
                continue
            events.sort(key=lambda item: (-float(item[0]), int(item[1]), int(item[2])))
            event_image = np.asarray([event[1] for event in events], dtype=np.int32)
            total_events += int(event_image.size)
            classes.append(
                {
                    "ground_truth_counts": ground_truth_counts,
                    "event_image": event_image,
                    "event_true_positive": np.asarray([event[3] for event in events], dtype=np.float32),
                    "event_false_positive": np.asarray([event[4] for event in events], dtype=np.float32),
                }
            )
        per_iou.append(classes)
    return {"per_iou": per_iou, "event_count": total_events}


def _detection_metric(
    cache: Mapping[str, Any], multiplicities: np.ndarray
) -> tuple[float, float, float]:
    counts = np.asarray(multiplicities, dtype=np.float32).reshape(-1)
    threshold_values: list[float | None] = []
    for classes in cache.get("per_iou", []):
        ap_values = []
        for class_row in classes:
            gt_counts = np.asarray(class_row["ground_truth_counts"], dtype=np.float32).reshape(-1)
            gt_total = float(np.dot(counts, gt_counts))
            if gt_total <= 0.0:
                continue
            event_image = np.asarray(class_row["event_image"], dtype=np.int32).reshape(-1)
            if event_image.size <= 0:
                ap_values.append(0.0)
                continue
            weights = counts[event_image]
            true_positive = np.asarray(class_row["event_true_positive"], dtype=np.float32) * weights
            false_positive = np.asarray(class_row["event_false_positive"], dtype=np.float32) * weights
            tp_cumulative = np.cumsum(true_positive, dtype=np.float32)
            fp_cumulative = np.cumsum(false_positive, dtype=np.float32)
            recall = tp_cumulative / float(max(gt_total, 1.0))
            precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
            ap_values.append(_ap_101(recall, precision))
        threshold_values.append(float(np.mean(ap_values)) if ap_values else None)
    finite = [float(value) for value in threshold_values if value is not None and np.isfinite(value)]
    ap_50_95 = float(np.mean(finite)) if finite else 0.0

    thresholds = [float(value) for value in list(cache.get("iou_thresholds") or [])]

    def value_at_iou(target: float) -> float:
        if not thresholds:
            return 0.0
        matches = [
            index for index, threshold in enumerate(thresholds)
            if abs(float(threshold) - float(target)) <= 1e-9
        ]
        if len(matches) != 1:
            raise ValueError(
                f"detection metric cache does not contain exactly one IoU={target:.2f} slice"
            )
        value = threshold_values[matches[0]]
        return float(value) if value is not None and np.isfinite(value) else 0.0

    return ap_50_95, value_at_iou(0.50), value_at_iou(0.75)


class _DetectionCOCOProxyEvaluator:
    def __init__(
        self,
        reference_records: Sequence[Mapping[str, Any]],
        candidate_records: Sequence[Mapping[str, Any]],
        annotations: Any,
        config: Mapping[str, Any],
    ) -> None:
        ground_truth = []
        reference = []
        candidate = []
        annotation_rows = list(annotations) if isinstance(annotations, Sequence) and not isinstance(annotations, (str, bytes)) else []
        if annotation_rows and len(annotation_rows) != len(reference_records):
            raise ValueError("detection annotations do not match the paired Image-ID count")
        for index, (reference_record, candidate_record) in enumerate(zip(reference_records, candidate_records)):
            annotation = annotation_rows[index] if annotation_rows else {}
            gt = annotation.get("ground_truth") if isinstance(annotation, Mapping) else None
            if gt is None:
                gt = candidate_record.get("ground_truth")
            if gt is None:
                gt = reference_record.get("ground_truth")
            ref = reference_record.get("reference")
            cand = candidate_record.get("candidate")
            if not isinstance(gt, list) or not isinstance(ref, list) or not isinstance(cand, list):
                raise ValueError("detection quality records require ground_truth/reference/candidate lists")
            ground_truth.append(gt)
            reference.append(ref)
            candidate.append(cand)
        self.thresholds = [float(value) for value in np.arange(0.50, 0.951, 0.05)]
        self.reference_cache = _prepare_detection_cache(ground_truth, reference, self.thresholds)
        self.candidate_cache = _prepare_detection_cache(ground_truth, candidate, self.thresholds)
        self.reference_cache["iou_thresholds"] = list(self.thresholds)
        self.candidate_cache["iou_thresholds"] = list(self.thresholds)
        self.primary_margin = _margin(config, "non_inferiority_margin")
        self.ap50_margin = _margin(config, "guardrails.ap50_margin")
        self.ap75_margin = _margin(config, "guardrails.ap75_margin")
        gate = _gate_config(config)
        configured_guardrails = gate.get("guardrails")
        self.ap75_configured = bool(
            isinstance(configured_guardrails, Mapping)
            and "ap75_margin" in configured_guardrails
        )

    def evaluate(self, multiplicities: np.ndarray) -> Mapping[str, Any]:
        candidate_primary, candidate_ap50, candidate_ap75 = _detection_metric(
            self.candidate_cache, multiplicities
        )
        reference_primary, reference_ap50, reference_ap75 = _detection_metric(
            self.reference_cache, multiplicities
        )
        guardrails = {
            "ap50": {
                "metric": "ap50",
                "candidate": candidate_ap50,
                "reference": reference_ap50,
                "delta": candidate_ap50 - reference_ap50,
                "margin": self.ap50_margin,
            }
        }
        # AP75 is a policy guardrail only when the policy actually configures
        # it.  This keeps old AP50-only policies from acquiring an implicit new
        # threshold while ensuring every AP75-configured request materialises
        # the metric (and its paired bootstrap interval) in the canonical
        # result.
        if self.ap75_configured:
            guardrails["ap75"] = {
                "metric": "ap75",
                "candidate": candidate_ap75,
                "reference": reference_ap75,
                "delta": candidate_ap75 - reference_ap75,
                "margin": self.ap75_margin,
            }
        return {
            "primary": {
                "metric": "coco_ap_50_95",
                "candidate": candidate_primary,
                "reference": reference_primary,
                "delta": candidate_primary - reference_primary,
                "margin": self.primary_margin,
            },
            "guardrails": guardrails,
        }


def detection_quality_evaluator(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    annotations: Any,
    config: Mapping[str, Any],
) -> _DetectionCOCOProxyEvaluator:
    from .accuracy_reporting import active_policy
    if active_policy(_gate_config(config)):
        return _CanonicalCOCOEvaluator(reference_records, candidate_records, annotations, config)
    return _DetectionCOCOProxyEvaluator(reference_records, candidate_records, annotations, config)


__all__ = ["classification_quality_evaluator", "detection_quality_evaluator"]


class _CanonicalCOCOEvaluator:
    """Official COCO matching once, population accumulation for every paired draw.

    Repeated images duplicate complete evalImgs records, including crowd/ignore,
    maxDets, area ranges and score ties. Never average per-image AP.
    """
    def __init__(self, reference_records, candidate_records, annotations, config):
        import contextlib
        import io
        import time
        from copy import deepcopy
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        preparation_started = time.perf_counter()
        self.preparation_observation = {"matching_s": 0.0, "matched_sides": 0}
        self.n = len(reference_records)
        gt_rows = [a["ground_truth"] for a in annotations]
        categories = sorted({int(d["class_id"]) for row in gt_rows for d in row})
        images = [{"id": i} for i in range(self.n)]
        ground = []
        def box(d):
            coords = np.asarray([d[k] for k in ("x1", "y1", "x2", "y2")], dtype=float)
            if not np.isfinite(coords).all() or coords[2] < coords[0] or coords[3] < coords[1]:
                raise ValueError("invalid final XYXY quality record")
            return [float(coords[0]), float(coords[1]), float(coords[2]-coords[0]), float(coords[3]-coords[1])]
        for i, row in enumerate(gt_rows):
            for d in row:
                b = box(d)
                ground.append(dict(id=len(ground)+1, image_id=i, category_id=int(d["class_id"]), bbox=b,
                    area=float(d.get("area", b[2]*b[3])), iscrowd=int(d.get("iscrowd", 0)), ignore=int(d.get("ignore", 0))))
        self.evaluators = []
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = dict(images=images, categories=[{"id": k} for k in categories], annotations=ground)
            gt.createIndex()
            for records, field in [(reference_records, "reference"), (candidate_records, "candidate")]:
                if field not in config.get("coco_sides", ("reference", "candidate")):
                    continue
                detections = []
                for i, record in enumerate(records):
                    for d in record[field]:
                        score = float(d["score"])
                        if not np.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("invalid detection score")
                        detections.append(dict(image_id=i, category_id=int(d["class_id"]), bbox=box(d), score=score))
                if detections:
                    dt = gt.loadRes(detections)
                else:
                    dt = COCO()
                    dt.dataset = dict(images=images, categories=gt.dataset["categories"], annotations=[])
                    dt.createIndex()
                evaluator = COCOeval(gt, dt, "bbox")
                if config.get("statistics_engine") == "optimized_coco_v1":
                    # Preserve the library's exact all-area bounds, grid and maxDets.
                    # The separate official report retains all twelve standard metrics.
                    evaluator.params.areaRng = [evaluator.params.areaRng[0]]
                    evaluator.params.areaRngLbl = [evaluator.params.areaRngLbl[0]]
                    evaluator.params.maxDets = [evaluator.params.maxDets[-1]]
                matching_started = time.perf_counter()
                evaluator.evaluate()
                self.preparation_observation["matching_s"] += time.perf_counter() - matching_started
                self.preparation_observation["matched_sides"] += 1
                self.evaluators.append((evaluator, list(evaluator.evalImgs), deepcopy(evaluator._paramsEval)))
        self.config = config
        self.preparation_observation["data_preparation_s"] = (
            time.perf_counter() - preparation_started - self.preparation_observation["matching_s"])

    def evaluate_sides(self, multiplicities):
        import contextlib
        import io
        from copy import deepcopy
        counts = np.asarray(multiplicities)
        if counts.shape != (self.n,) or not np.isfinite(counts).all() or np.any(counts < 0) or not np.array_equal(counts, np.rint(counts)):
            raise ValueError("invalid COCO image multiplicities")
        indices = np.repeat(np.arange(self.n), counts.astype(int))
        if not len(indices):
            raise ValueError("empty COCO population")
        metrics = []
        with contextlib.redirect_stdout(io.StringIO()):
            for evaluator, original, params in self.evaluators:
                evaluator.params = deepcopy(params)
                evaluator.params.imgIds = list(range(len(indices)))
                evaluator._paramsEval = deepcopy(evaluator.params)
                evaluator.evalImgs = [original[base+i] for base in range(0, len(original), self.n) for i in indices]
                evaluator.accumulate()
                precision = evaluator.eval["precision"][:, :, :, 0, -1]
                def mean_valid(a):
                    a = a[a > -1]
                    return float(np.mean(a)) if a.size else 0.0
                metrics.append([mean_valid(precision), mean_valid(precision[0]), mean_valid(precision[5])])
        return metrics

    def evaluate(self, multiplicities):
        return self.pair_components(*self.evaluate_sides(multiplicities))

    def pair_components(self, reference, candidate):
        names = ["coco_ap_50_95", "ap50", "ap75"]
        parts = [dict(metric=name, reference=r, candidate=c, delta=c-r,
                 margin=_margin(self.config, "non_inferiority_margin" if idx == 0 else f"guardrails.{name}_margin"))
                 for idx, (name, r, c) in enumerate(zip(names, reference, candidate))]
        return {"primary": parts[0], "guardrails": {"ap50": parts[1], "ap75": parts[2]}}
