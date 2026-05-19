#!/usr/bin/env python3
"""Export torchvision classification models or Ultralytics YOLO models to ONNX.

This script unifies two common export workflows used in the splitpoint project:

1. Torchvision image-classification models (e.g. ResNet, MobileNet, EfficientNet,
   RegNet) loaded via ``torchvision.models.get_model()``.
2. Ultralytics YOLO models (e.g. YOLO11, YOLO26) loaded via ``ultralytics.YOLO``
   and exported with Ultralytics' official export path.

Highlights
----------
- ``--source auto|torchvision|ultralytics`` to choose the export path.
- Torchvision path keeps the previous behavior: DEFAULT weights, ImageNet
  preprocessing sidecar, optional ORT equivalence verification.
- Ultralytics path can use official ``*.pt`` model names directly (Ultralytics
  will download them automatically if missing), then exports to ONNX and writes
  a task-aware sidecar JSON.
- Optional ORT smoke test for exported Ultralytics ONNX models.
- Writes categories/name metadata JSON when available.

Examples
--------
Torchvision classification:
    python export_vision_models_to_onnx.py \
        --source torchvision \
        --model resnet50 \
        --output exports/resnet50.onnx \
        --verify ort

Ultralytics YOLO:
    python export_vision_models_to_onnx.py \
        --source ultralytics \
        --model yolo11s.pt \
        --output exports/yolo11s.onnx \
        --imgsz 640 \
        --verify ort
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import onnx
except Exception:  # pragma: no cover - optional dependency
    onnx = None  # type: ignore

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore

LOGGER = logging.getLogger("vision_onnx_export")

RECOMMENDED_ULTRALYTICS_MODELS: Tuple[str, ...] = (
    # Detection
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is None:
        return str(value)
    return f"{value.__class__.__name__}.{name}"


def _to_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _interpolation_to_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _source_auto(model_ref: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lower = model_ref.lower()
    if lower.startswith("yolo") or lower.endswith(".pt") or lower.endswith(".yaml"):
        return "ultralytics"
    return "torchvision"


def _import_torchvision_models():
    try:
        import torchvision
        from torchvision.models import get_model, get_model_weights, list_models

        return torchvision, get_model, get_model_weights, list_models
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Torchvision could not be imported cleanly. This usually means torch and "
            "torchvision are version-mismatched or torchvision was installed without "
            f"the expected compiled ops. Original error: {exc}"
        ) from exc


def _import_ultralytics():
    try:
        import ultralytics
        from ultralytics import YOLO

        return ultralytics, YOLO
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Ultralytics could not be imported. Install it with 'pip install ultralytics'. "
            f"Original error: {exc}"
        ) from exc


def _extract_transforms_info(weights: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if weights is None:
        return info

    try:
        transforms = weights.transforms()
    except Exception as exc:
        LOGGER.warning("Could not instantiate weight transforms: %s", exc)
        return info

    for attr in ("resize_size", "crop_size", "mean", "std", "antialias"):
        if hasattr(transforms, attr):
            info[attr] = _to_list(getattr(transforms, attr))
    if hasattr(transforms, "interpolation"):
        info["interpolation"] = _interpolation_to_name(getattr(transforms, "interpolation"))
    return info


def _default_preprocess_from_weights(weights: Any) -> Dict[str, Any]:
    info = _extract_transforms_info(weights)
    if not info:
        info = {
            "resize_size": [256],
            "crop_size": [224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "BILINEAR",
            "antialias": True,
        }
    return info


def _infer_input_size(preprocess: Dict[str, Any]) -> Tuple[int, int]:
    crop = preprocess.get("crop_size")
    if isinstance(crop, list) and len(crop) >= 2:
        return int(crop[0]), int(crop[1])
    if isinstance(crop, list) and len(crop) == 1:
        size = int(crop[0])
        return size, size
    return 224, 224


def _resolve_weights(get_model_weights: Any, model_name: str, weights_name: str) -> Any:
    if weights_name.lower() == "none":
        return None
    weights_enum = get_model_weights(model_name)
    if weights_name.upper() == "DEFAULT":
        return weights_enum.DEFAULT
    try:
        return getattr(weights_enum, weights_name)
    except AttributeError as exc:
        valid = [w.name for w in weights_enum]
        raise ValueError(
            f"Unknown weights '{weights_name}' for model '{model_name}'. Valid: {valid}"
        ) from exc


def _list_classification_models(torchvision_mod: Any, list_models_fn: Any) -> List[str]:
    names = list_models_fn(module=torchvision_mod.models)
    return sorted(names)


def _build_dummy_input(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)


def _try_dynamo_export(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> bool:
    export_kwargs: Dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset,
        "dynamo": True,
        "report": False,
    }
    if dynamic_batch:
        export_kwargs["dynamic_shapes"] = ({0: "batch"},)

    LOGGER.info("Exporting with torch.onnx.export(..., dynamo=True)")
    program = torch.onnx.export(model, (dummy_input,), **export_kwargs)
    if hasattr(program, "save"):
        program.save(str(output_path))
        return True
    return output_path.exists()


def _legacy_export(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    export_kwargs: Dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset,
    }
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }
    LOGGER.info("Exporting with legacy torch.onnx.export(..., dynamo=False)")
    torch.onnx.export(model, (dummy_input,), str(output_path), **export_kwargs)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-x, kth=min(k - 1, x.shape[-1] - 1), axis=-1)[..., :k]
    scores = np.take_along_axis(x, idx, axis=-1)
    order = np.argsort(-scores, axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def _verify_classification_with_onnxruntime(
    output_path: Path,
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
) -> Dict[str, Any]:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed, cannot run ORT verification.")

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(output_path), providers=providers)
    input_name = session.get_inputs()[0].name

    with torch.no_grad():
        torch_out = model(dummy_input).detach().cpu().numpy()
    ort_out = session.run(None, {input_name: dummy_input.detach().cpu().numpy()})[0]

    torch_probs = _softmax(torch_out)
    ort_probs = _softmax(ort_out)

    torch_top1 = _topk_indices(torch_probs, 1)
    ort_top1 = _topk_indices(ort_probs, 1)
    torch_top5 = _topk_indices(torch_probs, min(5, torch_probs.shape[-1]))
    ort_top5 = _topk_indices(ort_probs, min(5, ort_probs.shape[-1]))

    top1_match = bool(np.array_equal(torch_top1, ort_top1))
    top5_match = bool(np.array_equal(torch_top5, ort_top5))

    flat_t = torch_out.reshape(torch_out.shape[0], -1)
    flat_o = ort_out.reshape(ort_out.shape[0], -1)
    denom = np.linalg.norm(flat_t, axis=1) * np.linalg.norm(flat_o, axis=1)
    cosine = np.where(denom > 0, np.sum(flat_t * flat_o, axis=1) / denom, 0.0)

    return {
        "providers": providers,
        "top1_match": top1_match,
        "top5_match": top5_match,
        "max_abs": float(np.max(np.abs(torch_out - ort_out))),
        "mean_abs": float(np.mean(np.abs(torch_out - ort_out))),
        "mean_cosine": float(np.mean(cosine)),
        "torch_top1": torch_top1.tolist(),
        "ort_top1": ort_top1.tolist(),
    }


def _inspect_onnx_io(output_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if onnx is not None:
        try:
            proto = onnx.load(str(output_path))

            def _shape(value_info: Any) -> List[Any]:
                dims: List[Any] = []
                tensor_type = value_info.type.tensor_type
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        dims.append(int(dim.dim_value))
                    elif dim.HasField("dim_param"):
                        dims.append(dim.dim_param)
                    else:
                        dims.append(None)
                return dims

            info["inputs"] = [{"name": x.name, "shape": _shape(x)} for x in proto.graph.input]
            info["outputs"] = [{"name": x.name, "shape": _shape(x)} for x in proto.graph.output]
            return info
        except Exception as exc:
            LOGGER.debug("ONNX graph inspection failed: %s", exc)
    if ort is not None:
        try:
            sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            info["inputs"] = [{"name": x.name, "shape": list(x.shape), "type": x.type} for x in sess.get_inputs()]
            info["outputs"] = [{"name": x.name, "shape": list(x.shape), "type": x.type} for x in sess.get_outputs()]
        except Exception as exc:
            LOGGER.debug("ORT graph inspection failed: %s", exc)
    return info


def _embed_onnx_metadata(output_path: Path, metadata: Dict[str, Any]) -> None:
    if onnx is None:
        LOGGER.warning("onnx is not installed; skipping metadata embedding.")
        return

    model_proto = onnx.load(str(output_path))
    keep = {
        "task_type": metadata.get("task_type"),
        "source": metadata.get("source"),
        "model_name": metadata.get("model_name"),
        "weights": metadata.get("weights"),
        "resize_size": json.dumps(metadata.get("preprocess", {}).get("resize_size")),
        "crop_size": json.dumps(metadata.get("preprocess", {}).get("crop_size")),
        "mean": json.dumps(metadata.get("preprocess", {}).get("mean")),
        "std": json.dumps(metadata.get("preprocess", {}).get("std")),
        "interpolation": metadata.get("preprocess", {}).get("interpolation"),
        "num_classes": str(metadata.get("num_classes")) if metadata.get("num_classes") is not None else None,
        "ultralytics_task": metadata.get("ultralytics_task"),
    }

    existing = {p.key: p for p in model_proto.metadata_props}
    for key in list(existing.keys()):
        if key in keep:
            del existing[key]

    del model_proto.metadata_props[:]
    for key, value in existing.items():
        prop = model_proto.metadata_props.add()
        prop.key = key
        prop.value = value.value
    for key, value in keep.items():
        if value is None:
            continue
        prop = model_proto.metadata_props.add()
        prop.key = key
        prop.value = str(value)

    onnx.save(model_proto, str(output_path))


def _normalize_categories(categories: Any) -> Optional[List[str]]:
    if categories is None:
        return None
    if isinstance(categories, dict):
        items = []
        try:
            for key, value in sorted(categories.items(), key=lambda kv: int(kv[0])):
                items.append(str(value))
            return items
        except Exception:
            return [str(v) for _, v in categories.items()]
    if isinstance(categories, (list, tuple)):
        return [str(x) for x in categories]
    return None


def _ultralytics_device_arg(requested: str) -> Union[str, int]:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return 0 if torch.cuda.is_available() else "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def _normalize_imgsz(values: Sequence[int]) -> Union[int, Tuple[int, int]]:
    if not values:
        return 640
    if len(values) == 1:
        return int(values[0])
    return int(values[0]), int(values[1])


def _bool_or_none_from_str(value: str) -> Optional[bool]:
    value = value.lower()
    if value == "auto":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean/auto value: {value}")


def _ort_smoke_test(
    output_path: Path,
    batch_size: int,
    height: int,
    width: int,
) -> Dict[str, Any]:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed, cannot run ORT smoke test.")

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(output_path), providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    dummy = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    outputs = session.run(None, {input_name: dummy})
    output_metas = session.get_outputs()
    return {
        "providers": providers,
        "input_name": input_name,
        "input_shape_used": list(dummy.shape),
        "outputs": [
            {
                "name": meta.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            for meta, arr in zip(output_metas, outputs)
        ],
    }


def _export_torchvision(args: argparse.Namespace) -> Dict[str, Any]:
    torchvision_mod, get_model, get_model_weights, _ = _import_torchvision_models()

    model_name = args.model
    output_path: Path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = _resolve_weights(get_model_weights, model_name, args.weights)
    preprocess = _default_preprocess_from_weights(weights)
    input_h, input_w = _infer_input_size(preprocess)
    categories: Optional[List[str]] = None
    if weights is not None:
        categories = list(weights.meta.get("categories", []))

    device = _pick_device(args.device)
    model = get_model(model_name, weights=weights)
    model.eval()
    model.to(device)

    dummy_input = _build_dummy_input(
        batch_size=args.batch_size,
        height=input_h,
        width=input_w,
        device=device,
        dtype=torch.float32,
    )

    export_used = "dynamo"
    try:
        ok = _try_dynamo_export(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            opset=args.opset,
            dynamic_batch=args.dynamic_batch,
        )
        if not ok:
            raise RuntimeError("Dynamo export did not produce an ONNX file.")
    except Exception as exc:
        if args.no_fallback_legacy:
            raise
        LOGGER.warning("Dynamo export failed, falling back to legacy exporter: %s", exc)
        export_used = "legacy"
        _legacy_export(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            opset=args.opset,
            dynamic_batch=args.dynamic_batch,
        )

    metadata: Dict[str, Any] = {
        "task_type": "classification",
        "source": "torchvision",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_mod.__version__,
        "model_name": model_name,
        "weights": _enum_name(weights),
        "num_classes": len(categories) if categories is not None else None,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [args.batch_size, 3, input_h, input_w],
        "preprocess": preprocess,
        "export": {
            "path": str(output_path),
            "opset": args.opset,
            "dynamic_batch": args.dynamic_batch,
            "exporter": export_used,
            "device": str(device),
        },
    }

    if categories is not None:
        categories_path = output_path.with_suffix(".categories.json")
        _write_json(categories_path, {"categories": categories})
        metadata["categories_path"] = str(categories_path)

    if args.verify == "ort":
        try:
            verification = _verify_classification_with_onnxruntime(output_path, model, dummy_input)
            metadata["verification"] = verification
            LOGGER.info(
                "ORT verification: top1_match=%s top5_match=%s max_abs=%.6g mean_abs=%.6g cosine=%.6f",
                verification["top1_match"],
                verification["top5_match"],
                verification["max_abs"],
                verification["mean_abs"],
                verification["mean_cosine"],
            )
        except Exception as exc:
            metadata["verification"] = {"error": str(exc)}
            LOGGER.warning("Verification skipped/failed: %s", exc)

    io_info = _inspect_onnx_io(output_path)
    if io_info:
        metadata["onnx_io"] = io_info

    if not args.no_embed_metadata:
        try:
            _embed_onnx_metadata(output_path, metadata)
        except Exception as exc:
            LOGGER.warning("Could not embed ONNX metadata: %s", exc)

    sidecar_path = output_path.with_suffix(".export.json")
    _write_json(sidecar_path, metadata)

    LOGGER.info("Wrote ONNX model: %s", output_path)
    LOGGER.info("Wrote export metadata: %s", sidecar_path)
    return metadata


def _export_ultralytics(args: argparse.Namespace) -> Dict[str, Any]:
    ultralytics_mod, YOLO = _import_ultralytics()

    model_ref = args.model
    output_path: Path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imgsz = _normalize_imgsz(args.imgsz)
    if isinstance(imgsz, int):
        input_h = input_w = imgsz
    else:
        input_h, input_w = int(imgsz[0]), int(imgsz[1])

    device_arg = _ultralytics_device_arg(args.device)
    model = YOLO(model_ref)

    end2end = _bool_or_none_from_str(args.end2end)
    export_kwargs: Dict[str, Any] = {
        "format": "onnx",
        "imgsz": imgsz,
        "half": bool(args.half),
        "dynamic": bool(args.dynamic_batch),
        "simplify": not args.no_simplify,
        "opset": args.opset,
        "nms": bool(args.nms),
        "batch": int(args.batch_size),
        "device": device_arg,
    }
    if end2end is not None:
        export_kwargs["end2end"] = end2end

    LOGGER.info("Exporting Ultralytics model '%s' to ONNX with args=%s", model_ref, export_kwargs)
    exported = model.export(**export_kwargs)
    exported_path = Path(exported).resolve()
    if exported_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported_path), str(output_path))
        LOGGER.info("Moved exported ONNX from %s to %s", exported_path, output_path)

    task = getattr(model, "task", None) or getattr(getattr(model, "model", None), "task", None)
    task = str(task) if task is not None else None
    task_type = "classification" if task == "classify" else "detection"
    categories = _normalize_categories(getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None))

    # Ultralytics models include their own preprocessing inside the project pipeline,
    # but the ONNX model itself expects normalized float images in BCHW.
    preprocess: Dict[str, Any] = {
        "resize_size": [input_h, input_w],
        "crop_size": [input_h, input_w],
        "mean": None,
        "std": None,
        "interpolation": "BILINEAR",
        "note": "Ultralytics export expects BCHW float input; letterbox/resize is handled upstream.",
    }

    metadata: Dict[str, Any] = {
        "task_type": task_type,
        "source": "ultralytics",
        "ultralytics_version": getattr(ultralytics_mod, "__version__", None),
        "torch_version": torch.__version__,
        "model_name": Path(model_ref).name,
        "model_ref": model_ref,
        "weights": None,
        "ultralytics_task": task,
        "num_classes": len(categories) if categories is not None else None,
        "input_shape": [args.batch_size, 3, input_h, input_w],
        "preprocess": preprocess,
        "export": {
            "path": str(output_path),
            "opset": args.opset,
            "dynamic_batch": args.dynamic_batch,
            "exporter": "ultralytics",
            "device": str(device_arg),
            "imgsz": imgsz,
            "half": args.half,
            "simplify": not args.no_simplify,
            "nms": args.nms,
            "batch": args.batch_size,
            "end2end": end2end,
        },
    }

    if categories is not None:
        categories_path = output_path.with_suffix(".categories.json")
        _write_json(categories_path, {"categories": categories})
        metadata["categories_path"] = str(categories_path)

    if args.verify == "ort":
        try:
            verification = _ort_smoke_test(output_path, args.batch_size, input_h, input_w)
            verification["note"] = (
                "ORT smoke test only. For Ultralytics exports this validates ONNX loadability and output shapes, "
                "not semantic equivalence against PyTorch postprocessing."
            )
            metadata["verification"] = verification
        except Exception as exc:
            metadata["verification"] = {"error": str(exc)}
            LOGGER.warning("Ultralytics ORT smoke test skipped/failed: %s", exc)

    io_info = _inspect_onnx_io(output_path)
    if io_info:
        metadata["onnx_io"] = io_info

    if not args.no_embed_metadata:
        try:
            _embed_onnx_metadata(output_path, metadata)
        except Exception as exc:
            LOGGER.warning("Could not embed ONNX metadata: %s", exc)

    sidecar_path = output_path.with_suffix(".export.json")
    _write_json(sidecar_path, metadata)

    LOGGER.info("Wrote ONNX model: %s", output_path)
    LOGGER.info("Wrote export metadata: %s", sidecar_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "torchvision", "ultralytics"], default="auto")
    parser.add_argument("--model", type=str, help="Model name/path. Example: resnet50 or yolo11s.pt")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Torchvision weights enum name, DEFAULT, or none")
    parser.add_argument("--output", type=Path, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Example batch size used for export/verification")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch dimension")
    parser.add_argument("--verify", choices=["none", "ort"], default="ort", help="Post-export verification mode")
    parser.add_argument("--no-embed-metadata", action="store_true", help="Do not write metadata into the ONNX file")
    parser.add_argument("--no-fallback-legacy", action="store_true", help="Do not fall back to legacy exporter for torchvision")
    parser.add_argument("--list-models", action="store_true", help="List available/known models for the selected source and exit")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Ultralytics-specific arguments.
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="*",
        default=[640],
        help="Ultralytics input image size. One value for square, two values for H W.",
    )
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported (mainly Ultralytics export)")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX graph simplification for Ultralytics export")
    parser.add_argument("--nms", action="store_true", help="Embed NMS when supported by Ultralytics export")
    parser.add_argument(
        "--end2end",
        choices=["auto", "true", "false"],
        default="auto",
        help="Ultralytics end2end override for models that support NMS-free export (e.g. YOLO26).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    if args.list_models:
        source = args.source
        if source in {"auto", "torchvision"}:
            torchvision_mod, _, _, list_models_fn = _import_torchvision_models()
            if source == "auto":
                print("# Torchvision classification models")
            for name in _list_classification_models(torchvision_mod, list_models_fn):
                print(name)
            if source == "torchvision":
                return 0
        if source in {"auto", "ultralytics"}:
            if source == "auto":
                print("\n# Recommended Ultralytics model refs")
            for name in RECOMMENDED_ULTRALYTICS_MODELS:
                print(name)
            return 0

    if not args.model:
        raise SystemExit("--model is required unless --list-models is used.")
    if args.output is None:
        raise SystemExit("--output is required unless --list-models is used.")

    chosen_source = _source_auto(args.model, args.source)
    LOGGER.info("Using source=%s for model=%s", chosen_source, args.model)

    if chosen_source == "torchvision":
        _export_torchvision(args)
    else:
        _export_ultralytics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
