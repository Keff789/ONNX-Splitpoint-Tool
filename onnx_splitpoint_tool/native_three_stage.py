"""Contract-bound Native three-stage execution helpers.

Release 2.79 keeps exactly two scientific runner roles (Generic and Native)
while allowing one Native deployment to expose two physical observation
boundaries:

``p2_output``
    The output of the accelerator/TensorRT P2 stage.  It is the primary
    hardware-performance surface and the only endpoint admitted to the
    Generic-to-Native ranking bridge when the physical contract matches.

``completed_detection``
    The contract-bound detection result after the optional postprocessing
    stage.  It is the application-throughput and application-energy surface.

Quality/reference processing is deliberately outside the measured hot loop.
This module provides reusable timing/reporting primitives and a fast,
contract-bound postprocess runtime.  It does not change candidate generation,
ranking, Quality policy or any compiler recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import threading
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


P2_OUTPUT_ENDPOINT = "p2_output"
COMPLETED_DETECTION_ENDPOINT = "completed_detection"
CLASSIFICATION_LOGITS_ENDPOINT = "classification_logits"

THREE_STAGE_RESULT_SCHEMA = "onnx-splitpoint/native-three-stage-result"
THREE_STAGE_RESULT_VERSION = 1
THREE_STAGE_TIMING_SCHEMA = "onnx-splitpoint/native-three-stage-timing"
THREE_STAGE_TIMING_VERSION = 1
QUALITY_ORACLE_SCHEMA = "onnx-splitpoint/native-three-stage-quality-oracle"
QUALITY_ORACLE_VERSION = 1

ADAPTER_CLASSIFICATION_LOGITS = "classification_logits_noop"
ADAPTER_YOLOV7_SPARSE = "yolov7_anchor_multiscale_sparse"
ADAPTER_YOLO26_DECODED = "yolo26_decoded_nms_materialize"
ADAPTER_YOLO11_DFL16 = "yolo11_regcls_dfl16"
ADAPTER_DECODED_PRE_NMS = "ultralytics_decoded_pre_nms"


class NativeThreeStageError(RuntimeError):
    """Fail-closed Native three-stage contract error."""


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(values: Iterable[Any]) -> dict[str, Any]:
    """Return a deterministic, JSON-ready timing summary in milliseconds."""

    cleaned = [
        number for number in (_finite_number(value) for value in values)
        if number is not None and number >= 0.0
    ]
    if not cleaned:
        return {
            "schema": THREE_STAGE_TIMING_SCHEMA,
            "schema_version": THREE_STAGE_TIMING_VERSION,
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "schema": THREE_STAGE_TIMING_SCHEMA,
        "schema_version": THREE_STAGE_TIMING_VERSION,
        "count": len(cleaned),
        "mean_ms": float(statistics.fmean(cleaned)),
        "p50_ms": float(_percentile(cleaned, 0.50)),
        "p95_ms": float(_percentile(cleaned, 0.95)),
        "min_ms": float(min(cleaned)),
        "max_ms": float(max(cleaned)),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def infer_p2_output_contract_family(
    payload: Mapping[str, Any],
    *,
    model_id: str = "",
    task: str = "",
) -> str:
    """Resolve the physical P2-output family without shape guessing.

    Existing endpoint attestations take precedence.  Model identity is used
    only to distinguish already-attested raw-head families.  Unsupported or
    ambiguous contracts fail closed.
    """

    task_value = str(task or payload.get("task") or "").strip().lower()
    if task_value == "classification":
        return "classification_logits"

    attestation = payload.get("output_endpoint_attestation")
    if not isinstance(attestation, Mapping):
        attestation = {}
    stage = str(
        payload.get("accelerator_output_stage")
        or payload.get("stage")
        or attestation.get("stage")
        or attestation.get("endpoint")
        or ""
    ).strip().lower()
    output_format = str(
        payload.get("output_format")
        or attestation.get("output_format")
        or ""
    ).strip().lower()
    family = str(
        payload.get("contract_family")
        or payload.get("accelerator_output_contract_family")
        or attestation.get("contract_family")
        or ""
    ).strip().lower()
    token = str(model_id or payload.get("model_id") or payload.get("model") or "")
    token = token.strip().lower().replace("-", "").replace("_", "")

    if stage in {"classification_logits", "logits"} or output_format == "classification_logits":
        return "classification_logits"
    if stage == "decoded_nms" or output_format == "bn6_detections" or family == "decoded_nms":
        return "yolo26_decoded_nms"
    if stage == "decoded_pre_nms" and family == "decoded_pre_nms":
        if "yolo11" in token or "yolo26" in token:
            return "ultralytics_decoded_pre_nms"
    if stage == "raw_head" or family in {"raw_head", "raw_heads"}:
        if "yolov7" in token or "yolo7" in token:
            return "yolov7_anchor_multiscale_raw"
        if "yolo11" in token:
            return "yolo11_regcls_dfl16_raw"
        if "yolo26" in token:
            return "yolo26_regcls_raw"
    raise NativeThreeStageError(
        "native_three_stage_p2_output_contract_family_unresolved"
    )


def adapter_id_for_contract_family(contract_family: str) -> str:
    value = str(contract_family or "").strip().lower()
    mapping = {
        "classification_logits": ADAPTER_CLASSIFICATION_LOGITS,
        "yolov7_anchor_multiscale_raw": ADAPTER_YOLOV7_SPARSE,
        "yolo26_decoded_nms": ADAPTER_YOLO26_DECODED,
        "yolo11_regcls_dfl16_raw": ADAPTER_YOLO11_DFL16,
        "ultralytics_decoded_pre_nms": ADAPTER_DECODED_PRE_NMS,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise NativeThreeStageError(
            f"native_three_stage_adapter_unsupported:{value}"
        ) from exc


def project_three_stage_endpoints(
    payload: Mapping[str, Any],
    *,
    p2_output_fps: Any,
    completed_detection_fps: Any | None,
    p2_output_contract_family: str,
    postprocess_adapter_id: str,
    postprocess_location: str,
    p1_samples_ms: Iterable[Any] = (),
    p2_samples_ms: Iterable[Any] = (),
    post_samples_ms: Iterable[Any] = (),
    p1_to_p2_queue_wait_ms: Iterable[Any] = (),
    p2_to_post_queue_wait_ms: Iterable[Any] = (),
    directly_measured: bool = True,
    projection_source: str = "",
) -> dict[str, Any]:
    """Return a lossless v2.79 endpoint/stage projection.

    Legacy field names remain as explicit aliases.  A projected historical row
    must set ``directly_measured=False`` and identify its source; the function
    never fabricates a second throughput measurement.
    """

    p2_fps = _finite_number(p2_output_fps)
    completed_fps = _finite_number(completed_detection_fps)
    if p2_fps is None or p2_fps <= 0.0:
        raise NativeThreeStageError("native_three_stage_p2_output_fps_invalid")
    if completed_fps is not None and completed_fps <= 0.0:
        raise NativeThreeStageError(
            "native_three_stage_completed_detection_fps_invalid"
        )
    if not directly_measured and not str(projection_source or "").strip():
        raise NativeThreeStageError(
            "native_three_stage_projection_source_required"
        )

    contract_family = str(p2_output_contract_family or "").strip()
    adapter_id = str(postprocess_adapter_id or "").strip()
    expected_adapter = adapter_id_for_contract_family(contract_family)
    if adapter_id != expected_adapter:
        raise NativeThreeStageError(
            "native_three_stage_adapter_contract_mismatch"
        )

    result = dict(payload)
    result.update({
        "schema": THREE_STAGE_RESULT_SCHEMA,
        "schema_version": THREE_STAGE_RESULT_VERSION,
        "native_runner_role": "native",
        "native_stage_model": "p1_p2_optional_contract_bound_postprocess",
        "performance_endpoint": P2_OUTPUT_ENDPOINT,
        "primary_performance_endpoint": P2_OUTPUT_ENDPOINT,
        "application_performance_endpoint": (
            COMPLETED_DETECTION_ENDPOINT if completed_fps is not None else ""
        ),
        "throughput_primary_metric": "p2_output_makespan_fps",
        "throughput_primary_fps": p2_fps,
        "p2_output_fps": p2_fps,
        "completed_detection_fps": completed_fps,
        "application_throughput_fps": completed_fps,
        "completed_to_p2_ratio": (
            completed_fps / p2_fps if completed_fps is not None else None
        ),
        "p2_output_contract_family": contract_family,
        "postprocess_adapter_id": adapter_id,
        "postprocess_location": str(postprocess_location or "").strip(),
        "directly_measured": bool(directly_measured),
        "projection_source": str(projection_source or ""),
        "stage_timings": {
            "p1": summarize_ms(p1_samples_ms),
            "p2": summarize_ms(p2_samples_ms),
            "postprocess": summarize_ms(post_samples_ms),
            "p1_to_p2_queue_wait": summarize_ms(p1_to_p2_queue_wait_ms),
            "p2_to_post_queue_wait": summarize_ms(p2_to_post_queue_wait_ms),
        },
        # Explicit compatibility aliases for pre-2.79 consumers.
        "legacy_performance_endpoint": "raw_model_outputs",
        "legacy_application_performance_endpoint": (
            "completed_task" if completed_fps is not None else ""
        ),
        "raw_model_outputs_fps_median": p2_fps,
        "completed_task_fps_median": completed_fps,
    })
    return result


@dataclass(frozen=True)
class _YoloV7HeadPlan:
    stride: int
    grid_h: int
    grid_w: int
    channels: int
    grid_x_flat: np.ndarray
    grid_y_flat: np.ndarray
    anchor_index_flat: np.ndarray
    anchors_wh: np.ndarray


def _exact_sigmoid(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array = np.clip(array, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-array))


class _ExactSparseYoloV7Decoder:
    """Prebound sparse YOLOv7 decoder preserving frozen clipped-sigmoid semantics."""

    def __init__(self, processor_contract: Mapping[str, Any]) -> None:
        self.contract = dict(processor_contract)
        self.input_hw = tuple(int(value) for value in self.contract["input_hw"])
        self.confidence_threshold = np.float32(
            self.contract["confidence_threshold"]
        )
        model_bound = self.contract.get("model_bound_decoder_contract")
        if not isinstance(model_bound, Mapping):
            raise NativeThreeStageError("model_bound_decoder_contract_missing")
        if str(model_bound.get("activation_mode") or "") != "logits":
            raise NativeThreeStageError(
                "yolov7_sparse_requires_logits_contract"
            )
        from .runners.harness.yolo import YOLOV7_SIGMOID_ARITHMETIC, _yolov7_sigmoid_float64_to_float32
        self._sigmoid = (
            _yolov7_sigmoid_float64_to_float32
            if model_bound.get("sigmoid_arithmetic") == YOLOV7_SIGMOID_ARITHMETIC
            else _exact_sigmoid
        )
        plans: list[_YoloV7HeadPlan] = []
        for record in model_bound["anchors_by_stride"]:
            stride = int(record["stride"])
            grid_h, grid_w = (int(value) for value in record["grid_hw"])
            anchors = np.asarray(record["anchors_wh"], dtype=np.float32)
            if anchors.shape != (3, 2):
                raise NativeThreeStageError("yolov7_anchor_shape_invalid")
            ys, xs = np.meshgrid(
                np.arange(grid_h), np.arange(grid_w), indexing="ij"
            )
            anchor_count = int(anchors.shape[0])
            plans.append(_YoloV7HeadPlan(
                stride=stride,
                grid_h=grid_h,
                grid_w=grid_w,
                channels=85,
                grid_x_flat=np.broadcast_to(
                    xs, (anchor_count, grid_h, grid_w)
                ).reshape(-1).astype(np.float32),
                grid_y_flat=np.broadcast_to(
                    ys, (anchor_count, grid_h, grid_w)
                ).reshape(-1).astype(np.float32),
                anchor_index_flat=np.broadcast_to(
                    np.arange(anchor_count)[:, None, None],
                    (anchor_count, grid_h, grid_w),
                ).reshape(-1),
                anchors_wh=anchors,
            ))
        self.plans = sorted(plans, key=lambda item: item.stride)

    def ordered_heads(self, outputs: Mapping[str, Any]) -> list[np.ndarray]:
        # The execution contract already canonicalizes YOLOv7 physical names.
        values = list(outputs.values())
        by_grid: dict[int, np.ndarray] = {}
        for value in values:
            array = np.asarray(value)
            if array.ndim != 5 or array.shape[0] != 1 or array.shape[1] != 3:
                raise NativeThreeStageError(
                    "yolov7_sparse_head_shape_invalid"
                )
            if int(array.shape[-1]) != 85 or int(array.shape[2]) != int(array.shape[3]):
                raise NativeThreeStageError(
                    "yolov7_sparse_head_geometry_invalid"
                )
            grid = int(array.shape[2])
            if grid in by_grid:
                raise NativeThreeStageError(
                    "yolov7_sparse_duplicate_head_geometry"
                )
            by_grid[grid] = array.astype(np.float32, copy=False)
        heads: list[np.ndarray] = []
        for plan in self.plans:
            try:
                heads.append(by_grid[plan.grid_h])
            except KeyError as exc:
                raise NativeThreeStageError(
                    f"yolov7_sparse_missing_head:{plan.grid_h}"
                ) from exc
        if len(by_grid) != len(self.plans):
            raise NativeThreeStageError("yolov7_sparse_extra_head")
        return heads

    def decode(self, outputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []
        for plan, tensor in zip(self.plans, self.ordered_heads(outputs)):
            flat = tensor[0].reshape(-1, plan.channels)
            objectness = self._sigmoid(flat[:, 4]).astype(
                np.float32, copy=False
            )
            objectness_indices = np.flatnonzero(
                objectness >= self.confidence_threshold
            )
            if objectness_indices.size == 0:
                continue
            selected = flat[objectness_indices]
            # Preserve clipping, sigmoid dtype and tie behaviour, but only for
            # candidates that can still pass the final score threshold.
            class_probabilities = self._sigmoid(selected[:, 5:]).astype(
                np.float32, copy=False
            )
            class_ids = np.argmax(class_probabilities, axis=1)
            class_scores = class_probabilities[
                np.arange(class_probabilities.shape[0]), class_ids
            ]
            scores = (
                objectness[objectness_indices] * class_scores
            ).astype(np.float32, copy=False)
            score_keep = scores >= self.confidence_threshold
            final_indices = objectness_indices[score_keep]
            if final_indices.size == 0:
                continue
            raw = flat[final_indices]
            txy = self._sigmoid(raw[:, 0:2]).astype(np.float32, copy=False)
            twh = self._sigmoid(raw[:, 2:4]).astype(np.float32, copy=False)
            grid = np.stack([
                plan.grid_x_flat[final_indices],
                plan.grid_y_flat[final_indices],
            ], axis=1)
            xy = (txy * 2.0 - 0.5 + grid) * float(plan.stride)
            wh = (
                (twh * 2.0) ** 2
                * plan.anchors_wh[plan.anchor_index_flat[final_indices]]
            )
            boxes = np.stack([
                xy[:, 0] - wh[:, 0] / 2.0,
                xy[:, 1] - wh[:, 1] / 2.0,
                xy[:, 0] + wh[:, 0] / 2.0,
                xy[:, 1] + wh[:, 1] / 2.0,
            ], axis=1)
            all_boxes.append(boxes.astype(np.float32, copy=False))
            all_scores.append(scores[score_keep])
            all_classes.append(class_ids[score_keep].astype(np.int64))
        if not all_boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.concatenate(all_boxes, axis=0).astype(np.float32, copy=False),
            np.concatenate(all_scores, axis=0).astype(np.float32, copy=False),
            np.concatenate(all_classes, axis=0).astype(np.int64, copy=False),
        )


class FastDetectionCompletionRuntime:
    """Task-only, contract-bound postprocess runtime for measured hot loops.

    This runtime intentionally performs no per-frame SHA-256, canonical JSON
    evidence construction or generic contract discovery.  An independent
    :class:`DetectionCompletionRuntime` remains the postflight Quality oracle.
    """

    def __init__(self, execution_contract: Mapping[str, Any]) -> None:
        from .native_detection_postprocess import (
            FrozenPostprocessError,
            YoloHarness,
            _canonical_detection_records,
            verify_detection_completion_execution_contract,
        )

        self._error_type = FrozenPostprocessError
        self._canonical_records = _canonical_detection_records
        self.execution_contract = verify_detection_completion_execution_contract(
            execution_contract
        )
        self._lock = threading.Lock()
        self.completed_count = 0
        self.last_result: dict[str, Any] = {}
        self.last_detections: list[dict[str, Any]] = []
        self.last_source_outputs: dict[str, Any] = {}
        self._source_snapshot_count = 0
        self._sentinel_context: dict[str, Any] = {}
        mode = str(self.execution_contract["completion_mode"])
        self.mode = mode
        self.model_family = str(self.execution_contract.get("model_family") or "")
        processor_contract = self.execution_contract["processor_contract"]
        self.processor_contract = dict(processor_contract)
        if mode == "raw_head_frozen_decode_nms" and self.model_family == "yolov7":
            self.adapter_id = ADAPTER_YOLOV7_SPARSE
            self._sparse = _ExactSparseYoloV7Decoder(processor_contract)
            self._harness = None
        elif mode in {"raw_head_frozen_decode_nms", "decoded_pre_nms_frozen_nms"}:
            self.adapter_id = (
                ADAPTER_DECODED_PRE_NMS
                if mode == "decoded_pre_nms_frozen_nms"
                else ADAPTER_YOLO11_DFL16
                if self.model_family == "yolo11"
                else "contract_bound_raw_head_host_tail"
            )
            model_bound = processor_contract.get("model_bound_decoder_contract")
            self._harness = YoloHarness(
                conf_thresh=float(processor_contract["confidence_threshold"]),
                iou_thresh=float(processor_contract["iou_threshold"]),
                max_det=int(processor_contract["max_detections"]),
                multiscale_activation_mode=processor_contract.get(
                    "multiscale_activation_mode"
                ),
                model_id=str(processor_contract.get("model_id") or ""),
                multiscale_decoder_contract=(
                    model_bound if isinstance(model_bound, Mapping) else None
                ),
            )
            self._sparse = None
        elif mode == "decoded_nms_attested_materialization_no_second_nms":
            self.adapter_id = ADAPTER_YOLO26_DECODED
            self._sparse = None
            self._harness = None
        else:
            raise NativeThreeStageError(
                f"native_three_stage_completion_mode_unsupported:{mode}"
            )

    def _deletterbox_records(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> list[dict[str, Any]]:
        from .runners.harness import yolo as yolo_module

        keep_all: list[int] = []
        for class_id in np.unique(class_ids) if class_ids.size else []:
            indexes = np.where(class_ids == class_id)[0]
            kept = yolo_module._nms_xyxy(
                boxes[indexes],
                scores[indexes],
                iou_thresh=float(self.processor_contract["iou_threshold"]),
            )
            keep_all.extend(indexes[index] for index in kept)
        keep_all = sorted(
            set(keep_all), key=lambda index: float(scores[index]), reverse=True
        )
        max_detections = int(self.processor_contract["max_detections"])
        if max_detections > 0:
            keep_all = keep_all[:max_detections]
        boxes = boxes[keep_all] if keep_all else boxes[:0]
        scores = scores[keep_all] if keep_all else scores[:0]
        class_ids = class_ids[keep_all] if keep_all else class_ids[:0]

        original_width, original_height = (
            int(value) for value in self.processor_contract["original_wh"]
        )
        input_height, input_width = (
            int(value) for value in self.processor_contract["input_hw"]
        )
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
        records = [{
            "class_id": int(class_ids[index]),
            "score": float(scores[index]),
            "x1": float(boxes_out[index, 0]),
            "y1": float(boxes_out[index, 1]),
            "x2": float(boxes_out[index, 2]),
            "y2": float(boxes_out[index, 3]),
        } for index in range(boxes_out.shape[0])]
        return self._canonical_records(
            records, max_detections=max_detections
        )

    def _decoded_nms_records(self, outputs: Mapping[str, Any]) -> list[dict[str, Any]]:
        array = np.asarray(next(iter(outputs.values())))
        rows = np.asarray(array, dtype=np.float64).reshape(-1, 6)
        if not bool(np.isfinite(rows).all()):
            raise NativeThreeStageError(
                "decoded_nms_materialization_nonfinite_values"
            )
        scores = rows[:, 4]
        classes = rows[:, 5]
        if (
            bool(np.any(scores < 0.0))
            or bool(np.any(scores > 1.0))
            or bool(np.any(classes < 0.0))
            or not bool(np.equal(classes, np.rint(classes)).all())
            or bool(np.any(rows[:, 2] < rows[:, 0]))
            or bool(np.any(rows[:, 3] < rows[:, 1]))
        ):
            raise NativeThreeStageError(
                "decoded_nms_materialization_source_values_invalid"
            )
        kept = rows[
            scores >= float(self.processor_contract["confidence_threshold"])
        ]
        if len(kept) > int(self.processor_contract["max_detections"]):
            raise NativeThreeStageError(
                "decoded_nms_materialization_count_exceeds_source_contract"
            )
        geometry = self.processor_contract["letterbox_geometry_contract"]
        gain = float(geometry["gain"])
        pad_left = float(geometry["pad_left"])
        pad_top = float(geometry["pad_top"])
        original_width, original_height = (
            int(value) for value in geometry["original_wh"]
        )
        records: list[dict[str, Any]] = []
        for row in kept:
            x1 = min(original_width, max(0.0, (float(row[0]) - pad_left) / gain))
            y1 = min(original_height, max(0.0, (float(row[1]) - pad_top) / gain))
            x2 = min(original_width, max(0.0, (float(row[2]) - pad_left) / gain))
            y2 = min(original_height, max(0.0, (float(row[3]) - pad_top) / gain))
            if x2 < x1 or y2 < y1:
                raise NativeThreeStageError(
                    "decoded_nms_materialization_inverse_geometry_invalid"
                )
            records.append({
                "class_id": int(row[5]),
                "score": float(row[4]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })
        return self._canonical_records(
            records,
            max_detections=int(self.processor_contract["max_detections"]),
        )

    def process(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            source_signature = self.execution_contract["source_endpoint"][
                "tensor_signature"
            ]
            from .native_detection_postprocess import tensor_signature
            if tensor_signature(outputs) != source_signature:
                raise NativeThreeStageError(
                    "native_three_stage_source_tensor_signature_mismatch"
                )
            numeric_observation = None
            if self._sparse is not None:
                boxes, scores, class_ids = self._sparse.decode(outputs)
                detections = self._deletterbox_records(boxes, scores, class_ids)
            elif self.mode == "decoded_nms_attested_materialization_no_second_nms":
                detections = self._decoded_nms_records(outputs)
            else:
                processing_outputs = dict(outputs)
                if self.mode == "decoded_pre_nms_frozen_nms":
                    from .native_detection_postprocess import prepare_decoded_pre_nms_outputs
                    processing_outputs, numeric_observation = prepare_decoded_pre_nms_outputs(
                        outputs, score_policy=self.processor_contract.get("decoded_pre_nms_score_policy"),
                    )
                context = {
                    "input_hw": list(self.processor_contract["input_hw"]),
                    "original_wh": list(self.processor_contract["original_wh"]),
                    "variant": "native_three_stage",
                }
                from .native_detection_postprocess import _result_json
                result = self._harness.postprocess(processing_outputs, context)
                payload = _result_json(result)
                if payload.get("format") != self.processor_contract["decoder_format"]:
                    raise NativeThreeStageError("native_three_stage_postprocess_format_mismatch")
                if not isinstance(payload, Mapping):
                    raise NativeThreeStageError(
                        "native_three_stage_postprocess_result_invalid"
                    )
                raw = payload.get("detections")
                if not isinstance(raw, list):
                    raise NativeThreeStageError(
                        "native_three_stage_postprocess_detections_missing"
                    )
                detections = self._canonical_records(
                    raw,
                    max_detections=int(self.processor_contract["max_detections"]),
                )
            self.completed_count += 1
            # These are borrowed synchronized P2 views until postflight. Publish
            # them only after this frame completed successfully. The caller must
            # drain the FIFO before attestation, while the P2 buffers are alive.
            self.last_source_outputs = {
                str(name): value for name, value in outputs.items()
            }
            self._source_snapshot_count = 0
            self.last_detections = detections
            self.last_result = {
                "task": "detection",
                "contract_family": "decoded_nms",
                "postprocess_adapter_id": self.adapter_id,
                "coordinate_space": "original_image_xyxy_pixels",
                "detection_count": len(detections),
                "detections": detections,
                "completed_count": self.completed_count,
                "evidence_mode": "task_only_no_crypto_in_timed_hotloop",
            }
            if numeric_observation is not None:
                self.last_result["decoded_pre_nms_score_normalization"] = numeric_observation
            return dict(self.last_result)

    def bind_sentinel_context(self, context: Mapping[str, Any]) -> None:
        """Bind existing invocation identities before starting measured work."""
        with self._lock:
            if self.completed_count:
                raise NativeThreeStageError("fast_oracle_context_bound_after_start")
            self._sentinel_context = dict(context)

    def snapshot_completed_source(self) -> dict[str, Any]:
        """Own the last synchronized frame once, after the producer has drained.

        NativeTRT returns reusable pinned-memory views. Copying here keeps that
        memory alive independently of the runtime's next call or close, without
        copying tensors or hashing inside the measured per-frame work.
        """
        with self._lock:
            if self.completed_count <= 0 or not self.last_source_outputs:
                raise NativeThreeStageError("native_three_stage_postflight_sentinel_missing")
            if self._source_snapshot_count != self.completed_count:
                snapshot = {
                    name: np.array(value, copy=True, order="C")
                    for name, value in self.last_source_outputs.items()
                }
                for value in snapshot.values():
                    value.flags.writeable = False
                self.last_source_outputs = snapshot
                self._source_snapshot_count = self.completed_count
            return dict(self.last_source_outputs)


    def attestation(
        self,
        *,
        completed_work_units: int | None = None,
    ) -> dict[str, Any]:
        """Attest task completion through one untimed frozen-oracle sentinel.

        Every measured frame still executes the contract-bound task work.  Only
        cryptographic evidence and the slower reference implementation are
        reduced to a postflight sentinel outside the performance interval.
        """
        from .native_detection_postprocess import (
            DetectionCompletionRuntime,
            canonical_json_sha256,
        )

        count = self.completed_count if completed_work_units is None else int(
            completed_work_units
        )
        if count <= 0 or count != self.completed_count:
            raise NativeThreeStageError(
                "native_three_stage_completion_count_mismatch"
            )
        if not self.last_source_outputs or not self.last_result:
            raise NativeThreeStageError(
                "native_three_stage_postflight_sentinel_missing"
            )
        source_snapshot = self.snapshot_completed_source()
        oracle = DetectionCompletionRuntime(
            self.execution_contract,
            observation_relation="same_hotloop_sentinel",
        )
        oracle_result = oracle.process(source_snapshot)
        oracle_attestation = oracle.attestation(completed_work_units=1)
        fast_detections = list(self.last_result.get("detections") or [])
        oracle_detections = list(oracle_result.get("detections") or [])
        exact = fast_detections == oracle_detections
        if not exact:
            raise NativeThreeStageError(
                "native_three_stage_quality_oracle_mismatch"
            )
        payload: dict[str, Any] = {
            "schema": QUALITY_ORACLE_SCHEMA,
            "schema_version": QUALITY_ORACLE_VERSION,
            "attested": True,
            "status": "passed",
            "observation_relation": "postflight_oracle_sentinel",
            "exact_result_claim_bound": True,
            "completed_work_units": count,
            "completion_count": self.completed_count,
            "postprocess_completed_frames": self.completed_count,
            "sentinel_identity": {
                **self._sentinel_context,
                "completed_work_unit_index": count,
                "execution_contract_sha256": self.execution_contract["contract_sha256"],
                "source_endpoint_contract_hash": self.execution_contract["source_endpoint"]["endpoint_contract_hash"],
                "source_snapshot_ownership": "owned_postflight_after_fifo_drain",
            },
            "execution_contract_sha256": str(
                self.execution_contract.get("contract_sha256") or ""
            ),
            "completed_endpoint_contract": dict(
                oracle_attestation.get("completed_endpoint_contract") or {}
            ),
            "comparison_endpoint_contract": dict(
                oracle_attestation.get("comparison_endpoint_contract") or {}
            ),
            # The fast hotloop deliberately produces no per-frame artifact.
            # Once the postflight oracle has proved exact detection equality,
            # carry its sealed sentinel through the existing persistence
            # contract. Keep hashing and file publication outside timing.
            "artifact": dict(oracle_attestation["artifact"]),
            "last_result": {
                **dict(self.last_result),
                "artifact": dict(oracle_attestation["artifact"]),
                "artifact_sha256": oracle_attestation["artifact_sha256"],
            },
            "fast_detection_count": len(fast_detections),
            "oracle_detection_count": len(oracle_detections),
            "fast_content_sha256": canonical_json_sha256(fast_detections),
            "oracle_content_sha256": canonical_json_sha256(oracle_detections),
            "quality_oracle_location": "outside_performance_timing",
            "timed_hotloop_evidence_mode": (
                "task_work_only_no_per_frame_crypto"
            ),
            "artifact_sha256": str(
                oracle_attestation.get("artifact_sha256") or ""
            ),
            "schema_sha256": str(
                oracle_attestation.get("schema_sha256") or ""
            ),
            "content_sha256": str(
                oracle_attestation.get("content_sha256") or ""
            ),
            "invocation_sha256": str(
                oracle_attestation.get("invocation_sha256") or ""
            ),
            "relation_sha256": str(
                oracle_attestation.get("relation_sha256") or ""
            ),
        }
        payload["attestation_sha256"] = canonical_json_sha256(payload)
        return payload


def verify_fast_completion_attestation(
    raw: Any, *, execution_contract: Mapping[str, Any],
    outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the existing postflight oracle without relabelling it hotloop.

    The sealed last fast result must equal the canonical oracle result, and all
    counts, artifact identities and source contracts remain bound. Optional
    dumped tensors additionally replay the oracle against the exact sentinel.
    This runs only while validating evidence, outside measurement timing.
    """
    from .native_detection_postprocess import (
        DetectionCompletionRuntime, canonical_json_sha256,
        verify_detection_completion_execution_contract,
        _canonical_detection_records, _completion_result_artifact,
        CANONICAL_DETECTION_RECORD_SCHEMA, CANONICAL_DETECTION_SORT_POLICY,
    )
    if not isinstance(raw, Mapping):
        raise NativeThreeStageError("fast_oracle_attestation_missing")
    att = dict(raw)
    seal = att.pop("attestation_sha256", "")
    if canonical_json_sha256(att) != seal:
        raise NativeThreeStageError("fast_oracle_attestation_sha256_mismatch")
    contract = verify_detection_completion_execution_contract(execution_contract)
    result = att.get("last_result")
    artifact = att.get("artifact")
    if not isinstance(result, Mapping) or not isinstance(artifact, Mapping):
        raise NativeThreeStageError("fast_oracle_result_missing")
    count = att.get("completed_work_units")
    if type(count) is not int or count <= 0 or any(
        type(value) is not int or value != count for value in (
            att.get("completion_count"), att.get("postprocess_completed_frames"),
            result.get("completed_count"),
        )
    ):
        raise NativeThreeStageError("fast_oracle_count_mismatch")
    detections = _canonical_detection_records(
        result.get("detections"),
        max_detections=int(contract["processor_contract"]["max_detections"]),
    )
    content_sha = canonical_json_sha256({
        "schema": "onnx-splitpoint/completed-detection-content", "schema_version": 1,
        "record_schema": CANONICAL_DETECTION_RECORD_SCHEMA,
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": CANONICAL_DETECTION_SORT_POLICY, "detections": detections,
    })
    source_sha = artifact.get("source_content_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64 or any(
        char not in "0123456789abcdef" for char in source_sha
    ):
        raise NativeThreeStageError("fast_oracle_source_hash_invalid")
    expected_artifact = _completion_result_artifact(
        execution_contract=contract, observation_relation="same_hotloop_sentinel",
        invocation_index=1, source_content_sha256=source_sha,
        content_sha256=content_sha, detections=detections,
    )
    artifact_sha = canonical_json_sha256(expected_artifact)
    invocation_sha = canonical_json_sha256({
        "schema": "onnx-splitpoint/detection-completion-invocation", "schema_version": 1,
        "execution_contract_sha256": contract["contract_sha256"],
        "observation_relation": "same_hotloop_sentinel", "invocation_index": 1,
        "source_content_sha256": source_sha, "content_sha256": content_sha,
        "artifact_sha256": artifact_sha, "implementation_sha256": contract["implementation_sha256"],
        "schema_sha256": contract["schema_sha256"], "relation_sha256": contract["relation_sha256"],
    })
    expected = {
        "schema": QUALITY_ORACLE_SCHEMA, "schema_version": QUALITY_ORACLE_VERSION,
        "attested": True, "status": "passed",
        "observation_relation": "postflight_oracle_sentinel", "exact_result_claim_bound": True,
        "execution_contract_sha256": contract["contract_sha256"],
        "completed_endpoint_contract": contract["completed_endpoint_contract"],
        "comparison_endpoint_contract": contract["comparison_endpoint_contract"],
        "artifact": expected_artifact, "artifact_sha256": artifact_sha,
        "schema_sha256": contract["schema_sha256"], "relation_sha256": contract["relation_sha256"],
        "content_sha256": content_sha, "invocation_sha256": invocation_sha,
        "fast_detection_count": len(detections), "oracle_detection_count": len(detections),
        "fast_content_sha256": canonical_json_sha256(detections),
        "oracle_content_sha256": canonical_json_sha256(detections),
        "quality_oracle_location": "outside_performance_timing",
        "timed_hotloop_evidence_mode": "task_work_only_no_per_frame_crypto",
    }
    if any(att.get(key) != value for key, value in expected.items()) or (
        result.get("detections") != detections
        or result.get("detection_count") != len(detections)
        or result.get("artifact") != expected_artifact
        or result.get("artifact_sha256") != artifact_sha
        or result.get("evidence_mode") != "task_only_no_crypto_in_timed_hotloop"
        or result.get("task") != "detection"
        or result.get("contract_family") != "decoded_nms"
        or result.get("coordinate_space") != "original_image_xyxy_pixels"
    ):
        raise NativeThreeStageError("fast_oracle_contract_or_result_mismatch")
    identity = att.get("sentinel_identity")
    if identity is not None:
        if not isinstance(identity, Mapping) or any(identity.get(key) != value for key, value in {
            "completed_work_unit_index": count,
            "execution_contract_sha256": contract["contract_sha256"],
            "source_endpoint_contract_hash": contract["source_endpoint"]["endpoint_contract_hash"],
            "source_snapshot_ownership": "owned_postflight_after_fifo_drain",
        }.items()):
            raise NativeThreeStageError("fast_oracle_sentinel_identity_mismatch")
    if outputs is not None:
        oracle = DetectionCompletionRuntime(contract, observation_relation="same_hotloop_sentinel")
        replay = oracle.process(outputs)
        if replay["artifact"].get("source_content_sha256") != source_sha:
            raise NativeThreeStageError("fast_oracle_dumped_sentinel_mismatch:source_tensor_content_differs")
        if replay["detections"] != detections:
            raise NativeThreeStageError("fast_oracle_dumped_sentinel_mismatch:same_source_decoded_result_differs")
        if replay["artifact"] != expected_artifact:
            raise NativeThreeStageError("fast_oracle_dumped_sentinel_mismatch:same_source_artifact_contract_differs")
    return {**att, "attestation_sha256": seal}


def verify_fast_completion_dump_binding(
    attestation: Mapping[str, Any], dump_manifest: Mapping[str, Any],
    endpoint_evidence: Mapping[str, Any],
) -> None:
    """Check the existing attestation's frame/repetition against its saved dump.

    Historical attestations without this association still require tensor replay;
    no association is invented for them. A new association may never be dropped.
    """
    identity = attestation.get("sentinel_identity")
    dumped_identity = dump_manifest.get("completion_sentinel_identity")
    if identity is None and dumped_identity is None:
        return
    if (
        not isinstance(identity, Mapping)
        or dict(dumped_identity or {}) != dict(identity)
        or dump_manifest.get("completion_attestation_sha256") != attestation.get("attestation_sha256")
        or dump_manifest.get("dump_inference_scope") != "last_measured_completion_source_outputs"
        or dump_manifest.get("input_image_sha256") != identity.get("input_image_sha256")
    ):
        raise NativeThreeStageError("fast_oracle_dump_sentinel_binding_mismatch")
    for projected, bound in (
        ("semantic_evidence_repetition_index", "process_local_repetition_index"),
        ("semantic_evidence_repetition_id", "repetition_id"),
        ("semantic_evidence_runtime_instance_id", "runtime_instance_id"),
    ):
        if bound in identity and endpoint_evidence.get(projected) != identity[bound]:
            raise NativeThreeStageError("fast_oracle_dump_repetition_selection_mismatch")


def verify_fast_runtime_against_oracle(
    *,
    execution_contract: Mapping[str, Any],
    outputs: Mapping[str, Any],
    fast_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen Quality oracle outside timing and compare results."""

    from .native_detection_postprocess import (
        DetectionCompletionRuntime,
        canonical_json_sha256,
    )

    oracle = DetectionCompletionRuntime(execution_contract)
    oracle_result = oracle.process(outputs)
    fast_detections = list(fast_result.get("detections") or [])
    oracle_detections = list(oracle_result.get("detections") or [])
    fast_hash = canonical_json_sha256(fast_detections)
    oracle_hash = canonical_json_sha256(oracle_detections)
    exact = fast_hash == oracle_hash and fast_detections == oracle_detections
    result = {
        "schema": QUALITY_ORACLE_SCHEMA,
        "schema_version": QUALITY_ORACLE_VERSION,
        "status": "passed" if exact else "failed",
        "exact": exact,
        "fast_detection_count": len(fast_detections),
        "oracle_detection_count": len(oracle_detections),
        "fast_content_sha256": fast_hash,
        "oracle_content_sha256": oracle_hash,
        "oracle_execution_contract_sha256": str(
            oracle.execution_contract.get("contract_sha256") or ""
        ),
        "quality_oracle_location": "outside_performance_timing",
    }
    if not exact:
        raise NativeThreeStageError(
            "native_three_stage_quality_oracle_mismatch"
        )
    return result
