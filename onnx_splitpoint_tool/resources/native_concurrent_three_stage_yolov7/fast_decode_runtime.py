#!/usr/bin/env python3
"""Contract-bound exact sparse YOLOv7 postprocessing for the three-stage canary.

The performance callback contains only the actual task work:

* objectness-first exact candidate rejection;
* class selection under the frozen clipped-sigmoid semantics;
* selective box decoding;
* frozen class-aware NMS;
* inverse letterbox;
* canonical detection record materialization.

Cryptographic hashing, contract discovery and the slow oracle are intentionally
kept outside the measured callback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np


def exact_sigmoid(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array = np.clip(array, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-array))


@dataclass(frozen=True)
class HeadPlan:
    stride: int
    grid_h: int
    grid_w: int
    channels: int
    grid_x_flat: np.ndarray
    grid_y_flat: np.ndarray
    anchor_index_flat: np.ndarray
    anchors_wh: np.ndarray


class ExactSparseYoloV7Decoder:
    """Prebound sparse decoder that preserves the v2.78.2 oracle semantics."""

    def __init__(self, processor_contract: Mapping[str, Any]) -> None:
        self.contract = dict(processor_contract)
        self.input_hw = tuple(int(v) for v in self.contract["input_hw"])
        self.confidence_threshold = np.float32(
            self.contract["confidence_threshold"]
        )
        model_bound = self.contract.get("model_bound_decoder_contract")
        if not isinstance(model_bound, Mapping):
            raise ValueError("model_bound_decoder_contract_missing")
        if str(model_bound.get("activation_mode") or "") != "logits":
            raise ValueError("three_stage_canary_requires_logits_contract")
        self.plans: list[HeadPlan] = []
        for record in model_bound["anchors_by_stride"]:
            stride = int(record["stride"])
            grid_h, grid_w = (int(v) for v in record["grid_hw"])
            anchors = np.asarray(record["anchors_wh"], dtype=np.float32)
            if anchors.shape != (3, 2):
                raise ValueError("unexpected_anchor_shape")
            ys, xs = np.meshgrid(
                np.arange(grid_h), np.arange(grid_w), indexing="ij"
            )
            anchor_count = int(anchors.shape[0])
            grid_x_flat = np.broadcast_to(
                xs, (anchor_count, grid_h, grid_w)
            ).reshape(-1).astype(np.float32)
            grid_y_flat = np.broadcast_to(
                ys, (anchor_count, grid_h, grid_w)
            ).reshape(-1).astype(np.float32)
            anchor_index_flat = np.broadcast_to(
                np.arange(anchor_count)[:, None, None],
                (anchor_count, grid_h, grid_w),
            ).reshape(-1)
            self.plans.append(
                HeadPlan(
                    stride=stride,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    channels=85,
                    grid_x_flat=grid_x_flat,
                    grid_y_flat=grid_y_flat,
                    anchor_index_flat=anchor_index_flat,
                    anchors_wh=anchors,
                )
            )
        self.plans.sort(key=lambda item: item.stride)

    def ordered_heads(self, outputs: Mapping[str, Any]) -> list[np.ndarray]:
        heads: list[np.ndarray] = []
        for plan in self.plans:
            name = f"yolov7_stride_{plan.stride}"
            if name not in outputs:
                raise ValueError(f"missing_prebound_head:{name}")
            value = np.asarray(outputs[name])
            expected = (1, 3, plan.grid_h, plan.grid_w, plan.channels)
            if tuple(int(v) for v in value.shape) != expected:
                raise ValueError(
                    f"prebound_head_shape_mismatch:{name}:{value.shape}:{expected}"
                )
            if value.dtype != np.float32:
                value = value.astype(np.float32, copy=False)
            heads.append(value)
        return heads

    def decode(
        self,
        outputs: Mapping[str, Any],
        *,
        yolo_module: Any,
        stage_samples: MutableMapping[str, list[float]] | None = None,
    ) -> tuple[Any, list[dict[str, int]]]:
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []
        counts: list[dict[str, int]] = []

        for plan, tensor in zip(self.plans, self.ordered_heads(outputs)):
            flat = tensor[0].reshape(-1, plan.channels)
            total = int(flat.shape[0])

            started = time.perf_counter_ns()
            objectness = exact_sigmoid(flat[:, 4]).astype(
                np.float32, copy=False
            )
            if stage_samples is not None:
                stage_samples.setdefault("objectness_sigmoid", []).append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )

            started = time.perf_counter_ns()
            objectness_indices = np.flatnonzero(
                objectness >= self.confidence_threshold
            )
            if stage_samples is not None:
                stage_samples.setdefault("objectness_exact_prune", []).append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )
            if objectness_indices.size == 0:
                counts.append(
                    {
                        "stride": plan.stride,
                        "total_candidates": total,
                        "objectness_survivors": 0,
                        "score_survivors": 0,
                    }
                )
                continue

            selected = flat[objectness_indices]

            # Preserve the oracle's clipping, sigmoid, float dtype and tie
            # semantics, but evaluate classes only for objectness survivors.
            started = time.perf_counter_ns()
            class_probabilities = exact_sigmoid(selected[:, 5:]).astype(
                np.float32, copy=False
            )
            class_ids = np.argmax(class_probabilities, axis=1)
            class_scores = class_probabilities[
                np.arange(class_probabilities.shape[0]), class_ids
            ]
            scores = (
                objectness[objectness_indices] * class_scores
            ).astype(np.float32, copy=False)
            if stage_samples is not None:
                stage_samples.setdefault(
                    "survivor_class_sigmoid_argmax", []
                ).append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )

            started = time.perf_counter_ns()
            score_keep = scores >= self.confidence_threshold
            final_indices = objectness_indices[score_keep]
            if stage_samples is not None:
                stage_samples.setdefault("combined_score_prune", []).append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )
            counts.append(
                {
                    "stride": plan.stride,
                    "total_candidates": total,
                    "objectness_survivors": int(objectness_indices.size),
                    "score_survivors": int(final_indices.size),
                }
            )
            if final_indices.size == 0:
                continue

            started = time.perf_counter_ns()
            raw = flat[final_indices]
            txy = exact_sigmoid(raw[:, 0:2]).astype(np.float32, copy=False)
            twh = exact_sigmoid(raw[:, 2:4]).astype(np.float32, copy=False)
            grid = np.stack(
                [
                    plan.grid_x_flat[final_indices],
                    plan.grid_y_flat[final_indices],
                ],
                axis=1,
            )
            xy = (txy * 2.0 - 0.5 + grid) * float(plan.stride)
            wh = (
                (twh * 2.0) ** 2
                * plan.anchors_wh[plan.anchor_index_flat[final_indices]]
            )
            boxes = np.stack(
                [
                    xy[:, 0] - wh[:, 0] / 2.0,
                    xy[:, 1] - wh[:, 1] / 2.0,
                    xy[:, 0] + wh[:, 0] / 2.0,
                    xy[:, 1] + wh[:, 1] / 2.0,
                ],
                axis=1,
            )
            if stage_samples is not None:
                stage_samples.setdefault("selected_box_decode", []).append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )

            all_boxes.append(boxes.astype(np.float32, copy=False))
            all_scores.append(scores[score_keep])
            all_classes.append(class_ids[score_keep].astype(np.int64))

        if not all_boxes:
            detections = yolo_module._Detections(
                boxes_xyxy=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                class_ids=np.zeros((0,), np.int64),
            )
        else:
            detections = yolo_module._Detections(
                boxes_xyxy=np.concatenate(all_boxes, axis=0).astype(
                    np.float32
                ),
                scores=np.concatenate(all_scores, axis=0).astype(np.float32),
                class_ids=np.concatenate(all_classes, axis=0).astype(np.int64),
            )
        return detections, counts


def complete_fast_dynamic(
    decoder: ExactSparseYoloV7Decoder,
    outputs: Mapping[str, Any],
    *,
    original_wh: tuple[int, int],
    yolo_module: Any,
    ndp_module: Any,
    processor_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    detections, counts = decoder.decode(outputs, yolo_module=yolo_module)
    boxes = detections.boxes_xyxy
    scores = detections.scores
    class_ids = detections.class_ids

    keep_all: list[int] = []
    for class_id in np.unique(class_ids) if class_ids.size else []:
        indexes = np.where(class_ids == class_id)[0]
        kept = yolo_module._nms_xyxy(
            boxes[indexes],
            scores[indexes],
            iou_thresh=float(processor_contract["iou_threshold"]),
        )
        keep_all.extend(indexes[index] for index in kept)
    keep_all = sorted(
        set(keep_all), key=lambda index: float(scores[index]), reverse=True
    )
    max_detections = int(processor_contract["max_detections"])
    if max_detections > 0:
        keep_all = keep_all[:max_detections]
    boxes = boxes[keep_all] if keep_all else boxes[:0]
    scores = scores[keep_all] if keep_all else scores[:0]
    class_ids = class_ids[keep_all] if keep_all else class_ids[:0]

    original_width, original_height = (int(v) for v in original_wh)
    input_height, input_width = (int(v) for v in processor_contract["input_hw"])
    gain = min(
        input_width / float(original_width),
        input_height / float(original_height),
    )
    resized_width = int(round(original_width * gain))
    resized_height = int(round(original_height * gain))
    pad_left = (input_width - resized_width) // 2
    pad_top = (input_height - resized_height) // 2
    boxes_out = boxes.copy()
    boxes_out[:, 0] = (boxes_out[:, 0] - pad_left) / gain
    boxes_out[:, 2] = (boxes_out[:, 2] - pad_left) / gain
    boxes_out[:, 1] = (boxes_out[:, 1] - pad_top) / gain
    boxes_out[:, 3] = (boxes_out[:, 3] - pad_top) / gain
    boxes_out[:, 0] = np.clip(boxes_out[:, 0], 0, original_width)
    boxes_out[:, 2] = np.clip(boxes_out[:, 2], 0, original_width)
    boxes_out[:, 1] = np.clip(boxes_out[:, 1], 0, original_height)
    boxes_out[:, 3] = np.clip(boxes_out[:, 3], 0, original_height)

    raw_records: list[dict[str, Any]] = []
    for index in range(boxes_out.shape[0]):
        x1, y1, x2, y2 = [float(v) for v in boxes_out[index].tolist()]
        raw_records.append(
            {
                "class_id": int(class_ids[index]),
                "score": float(scores[index]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return (
        ndp_module._canonical_detection_records(
            raw_records, max_detections=max_detections
        ),
        counts,
    )


def canonical_oracle_dynamic(
    harness: Any,
    outputs: Mapping[str, Any],
    *,
    original_wh: tuple[int, int],
    ndp_module: Any,
    processor_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload = ndp_module._result_json(
        harness.postprocess(
            dict(outputs),
            {
                "input_hw": list(processor_contract["input_hw"]),
                "original_wh": [int(original_wh[0]), int(original_wh[1])],
                "variant": "native_split",
            },
        )
    )
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise RuntimeError("oracle_detections_missing")
    return ndp_module._canonical_detection_records(
        detections,
        max_detections=int(processor_contract["max_detections"]),
    )
